function out = lf_safety_filter(u, p)
%LF_SAFETY_FILTER 安全约束层：离散 CBF + 箱式约束 + 速率约束的一次投影。
%
% 2026-08-09 从原 line_follower_safety_governor 拆出（另一半是 lf_line_search）。
% **本文件对指令来源不可知**——不管 (v, omega) 是 PID 产生的、RL 残差叠加的、
% 还是丢线恢复行为产生的，一律同样对待。这正是论文 §III-B-2 里 safety layer /
% OptLayer 的位置：把策略输出再过一层控制滤波器。信号链上它在 lf_line_search
% 之后、差速运动学之前，是下发执行器前的最后一道。
%
% 方法出处：Gu et al., "A Review of Safe Reinforcement Learning: Methods,
% Theories, and Applications", IEEE TPAMI 46(12):11216-11235, 2024。
%   - safety layer 的整体位置：§III-B-2 [67]。好处是**不需要重新训练**——
%     现有 SAC 检查点直接可用，而 CPO/IPO 那类代价值优化必须重训。
%   - CBF 部分：§III-A-2 / §III-B-2 的控制屏障函数一族 [33],[127]-[130]。
%
% 约束类型（该文 §II-A）：这里实现的是**瞬时约束**里的**显式**一类——每步都有
% 闭式可校验的表达式，而不是式(1)-(3) 那种沿轨迹累积的 cumulative constraint。
% 选瞬时约束正是因为累积约束要改训练目标。
%
% ── CBF 推导 ────────────────────────────────────────────────────────
%
% 记 y_L 为前视点在机体系下的横向偏移（m，Y 轴向右为正），psi 为前视点处路径
% 切线角（rad）。视觉模块输出的是归一化量 e = y_L / lateralNorm，
% lateralNorm = 0.55 m（**必须与 originbot_sliding_window_path_generator.m
% 里同名常数一致**，那里是唯一真相源）。
%
% 差速小车前视点横向偏移动力学（忽略路径曲率项）：
%
%     d(y_L)/dt = v*sin(psi) + L*omega
%
% 其中 L = 前视距离。第二项符号：omega>0 表示左转（Diff_Drive_Kinematics 里
% right = v + omega_wheel，右轮更快即左转），机体左转时前方的点在机体系里向右
% 移动，故 y_L 增大。第一项：路径切线朝右(psi>0)时前进使前视点右移。
%
% 用**前视点**而非车体质心横向偏差，是因为前视点偏移对 omega 的 relative
% degree 为 1（omega 直接出现在一阶导里），CBF 条件退化成一条关于 omega 的
% 仿射不等式，闭式可解；对质心偏差建屏障则 relative degree 为 2，要上 HOCBF
% 级联。视觉模块给出的 lateral_error 恰好就是前视点偏移，不需要额外估计。
%
% 屏障函数（安全集 h>=0 即 |e| <= LateralMax）：
%
%     h(e) = LateralMax^2 - e^2
%
% CBF 条件 dh/dt >= -alpha*h，代入 de/dt = (v*sin(psi) + L*omega)/lateralNorm，
% 两边同乘 lateralNorm/2 > 0，整理成 A*omega >= b：
%
%     A = -e*L
%     b = -(alpha*lateralNorm/2)*h + e*v*sin(psi)
%
% e>0（路径在右、车偏左）时 A<0，约束化为 omega <= b/A，即**禁止继续左转**；
% e<0 时对称。e≈0 时 A≈0 且 b<0，约束自动失效（0 >= 负数恒真），故用
% CBFSingularTol 做奇异保护。单条仿射约束 + 标量决策变量，其"投影"就是一次
% clamp——即 OptLayer 那个 QP 在本问题下的闭式解，不需要 QP 求解器。
%
% v 只受箱式/速率约束和边界减速启发式约束，不进 CBF：把决策变量限制成标量
% omega 才有上面的闭式解。边界减速项（SpeedBackoff）是**启发式，不属于 CBF
% 的前向不变性保证**，这条区分请勿在文档里模糊掉。
%
% ── 输入/输出 ───────────────────────────────────────────────────────
% u (5×1) = [v_des; omega_des; lateral_error; heading_error; found]
%   v_des     线速度期望值，单位是 VOmega 信号自身的单位（× p.SpeedScale 才是
%             m/s；两类模型的约定不同，见 lf_safety_defaults.m 的 signalUnits）
%   omega_des 偏航角速度期望值 (rad/s)
%
% out (7×1) = [v_safe; omega_safe; h; omega_lo; omega_hi; cbf_active; infeasible]
%   omega_lo/omega_hi 是最终生效的有限边界（已与箱式/速率约束求交）。
%   与输入的 v_des/omega_des 相减即本层的干预量。
%
% p 见 lf_safety_defaults.m。p.EnableFilter=0 时**边界与诊断照算照输出**
% （这正是"不加安全层会违约多少"的统计口径），但不做投影，直接下发期望值。

persistent vPrev omegaPrev

if isempty(vPrev),     vPrev     = 0; end
if isempty(omegaPrev), omegaPrev = 0; end

u = u(:);
if numel(u) ~= 5
    error("LineFollowerSafety:BadInput", ...
        "expected a 5-element [v; omega; lateral_error; heading_error; found] vector, got %d", ...
        numel(u));
end

vDes     = u(1);
omegaDes = u(2);
e        = u(3);
psi      = u(4);
found    = u(5) > 0.5;

% ── CBF 关于 omega 的仿射约束 ─────────────────────────────────────
vPhys      = vDes * p.SpeedScale;                 % 信号单位 → m/s
h          = p.LateralMax^2 - e^2;
cbfLo      = -inf;
cbfHi      =  inf;
cbfActive  = 0;
if found
    % 丢线时视觉模块把 lateral_error 强制置零，h 恒为正、约束本就失效；
    % 这里再显式按 found 关一次，避免依赖那个隐式行为
    A = -e * p.Lookahead;
    b = -0.5 * p.CBFAlpha * p.LateralNorm * h + e * vPhys * sin(psi);
    if abs(A) > p.CBFSingularTol
        bound = b / A;
        if A > 0
            cbfLo = bound;
        else
            cbfHi = bound;
        end
        cbfActive = 1;
    end
end

% ── 箱式约束 ∩ 速率约束（硬集，恒非空）────────────────────────────
dOmegaMax = p.MaxAccelOmega * p.Ts;
hardLo = max(-p.MaxOmega, omegaPrev - dOmegaMax);
hardHi = min( p.MaxOmega, omegaPrev + dOmegaMax);

omegaLo = max(hardLo, cbfLo);
omegaHi = min(hardHi, cbfHi);

infeasible = 0;
if omegaLo > omegaHi
    % CBF 要求的转向超出本拍执行器/速率能力：退回硬集里最接近 CBF 边界的那
    % 一点（等价于给 CBF 约束加松弛变量并最小化违反量）。物理可行性优先于
    % 屏障条件——否则会下发一个电机根本做不到的指令。
    infeasible = 1;
    if cbfLo > hardHi
        omegaSafe = hardHi;
    else
        omegaSafe = hardLo;
    end
    omegaLo = hardLo;
    omegaHi = hardHi;
else
    omegaSafe = min(max(omegaDes, omegaLo), omegaHi);
end

% ── 线速度约束 ────────────────────────────────────────────────────
% 靠近屏障边界时减速：**启发式**，不参与 CBF 的前向不变性保证，纯粹是给
% omega 的修正留出更多时间。SpeedBackoff=0 即关闭。
vScale = 1;
if found && p.SpeedBackoff > 0
    margin = max(0, h) / max(eps, p.LateralMax^2);   % 中心=1，边界=0
    vScale = 1 - p.SpeedBackoff * (1 - margin);
end
vTarget = vDes * vScale;

dVMax = p.MaxAccelV * p.Ts;
vLo = max(0, vPrev - dVMax);                 % v>=0：残差策略不得倒车
vHi = min(p.MaxV, vPrev + dVMax);
vSafe = min(max(vTarget, vLo), vHi);

if ~p.EnableFilter
    % 消融：h / omegaLo / omegaHi / cbfActive / infeasible 照算照输出，但不做
    % 投影。vPrev/omegaPrev 仍记录**实际下发**值，故下一拍速率窗口依然正确。
    vSafe     = vDes;
    omegaSafe = omegaDes;
end

vPrev     = vSafe;
omegaPrev = omegaSafe;

out = [vSafe; omegaSafe; h; omegaLo; omegaHi; cbfActive; infeasible];
end
