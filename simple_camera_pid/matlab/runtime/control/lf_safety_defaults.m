function p = lf_safety_defaults(model, signalUnits)
%LF_SAFETY_DEFAULTS 构造 lf_line_search 与 lf_safety_filter 的共用参数结构体。
%
% 本文件是安全层/找线状态机所有默认值的**唯一真相源**——两个算法文件
% （lf_line_search.m / lf_safety_filter.m）里没有任何硬编码默认值。它们共用
% 同一份结构体，各取所需字段。
%
% 两类字段来源不同，改的时候别混：
%
%   A) 从模型工作区**已有**变量派生（Ts_Control / BaseSpeedScale /
%      LocalPath_LookaheadDistance / BaseLinearSpeed / MaxAngularSpeed）。
%      这些不在这里给默认值，读不到就报错——它们本来就该存在，静默兜底会
%      掩盖模型工作区被改坏的情况。
%
%   B) 本功能新增的可调量（Safety_*）。沿用 exportLocalPathParameters 的
%      约定：模型工作区已有就读、没有就写入默认值，于是用户可以直接在模型
%      工作区里调参而不必改本文件。
%
% 默认值的选取原则：**对现有纯 PID 基线尽量透明**，只在真出事时才起作用。
%   - MaxV 取 BaseLinearSpeed，而 PID 侧 v 的上限恰为 BaseLinearSpeed×1.0×1.0，
%     所以纯 PID 下箱式约束永不触发；只有 2-D 残差往上顶 v 时才会限幅。
%   - 速率约束给得很松（≈1.2 m/s²、10 rad/s²），本质是抖动/急动保护，
%     不是性能限制。要做更强的舒适性约束请自行调小并重新评估跟踪性能。
%   - LateralMax=0.8（归一化），对应 0.8×0.55 = 0.44 m 前视点横向偏移。
%     视觉模块本身把 lateral_error 截断在 ±1，故 0.8 留了 20% 余量给屏障
%     在真正出界前起作用。
%
% ScanAmplitudes 的选取：本车相机水平 FOV 约 46°，扫描 ±25° 即覆盖约 96°
% 地面朝向，±80° 覆盖约 206°。三级幅度走完的总扫掠角约 540°(9.4 rad)，
% 在 ScanRate=0.9 rad/s 下约 10.5 s，故 ScanTimeout 给到 15 s 留出速率
% 约束爬坡的余量。改 ScanRate/ScanAmplitudes 时记得同步核算 ScanTimeout，
% 否则会在扫完之前就被超时踢进 GIVEUP。

% signalUnits 显式指定线速度信号的单位约定——**两类模型不一样，且无法从
% 数值上区分**（都是几个正实数），所以照搬 originbot_camera_profile 的做法
% 强制显式传参，不做自动推断：
%
%   "scaled"  MuJoCo 模型 (visual_line_follower_sac_residual.slx)。VOmega 的
%             v 分量是 BaseLinearSpeed 刻度，× BaseSpeedScale 才是 m/s
%             （见 Diff_Drive_Kinematics 的 V_To_Wheel 增益）。
%   "mps"     真机模型 (visual_line_follower_sac_residual_real.slx)。
%             PID_Controller 直接输出 /cmd_vel 用的 linear_x，已经是 m/s。
%
% 传错的后果是 CBF 里 v*sin(psi) 那一项差 ~33 倍（1/BaseSpeedScale），
% 屏障不会报错、只会变得过松或过紧——属于静默失效，务必对号入座。
arguments
    model (1, 1) string
    signalUnits (1, 1) string {mustBeMember(signalUnits, ["scaled", "mps"])}
end

workspace = get_param(model, "ModelWorkspace");

% ── A) 从模型工作区已有变量派生 ────────────────────────────────────
p = struct();
p.Ts          = readRequired(workspace, "Ts_Control");
p.Lookahead   = readRequired(workspace, "LocalPath_LookaheadDistance");
p.MaxOmega    = readRequired(workspace, "MaxAngularSpeed");
switch signalUnits
    case "scaled"
        p.SpeedScale = readRequired(workspace, "BaseSpeedScale");
        p.MaxV       = readRequired(workspace, "BaseLinearSpeed");
        % 40 信号单位/s × BaseSpeedScale(0.03) ≈ 1.2 m/s²
        defaultMaxAccelV = 40.0;
    case "mps"
        p.SpeedScale = 1.0;
        % 整车能跑到的最大前进速度 = 最大轮速 × 轮半径
        p.MaxV = readRequired(workspace, "MaxWheelSpeed") * ...
                 readRequired(workspace, "TB3_WheelRadius");
        defaultMaxAccelV = 1.2;   % m/s²，与 "scaled" 分支等价的物理加速度
end

% originbot_sliding_window_path_generator.m 里的同名常数，是 lateral_error
% 归一化的分母。**那里是真相源**；两处必须一致，否则 CBF 的物理量纲会错。
% 该值在视觉函数里是写死的局部常数（不是模型工作区变量），无法程序化读取，
% 因此这里只能复制一份并靠注释锁定——改视觉那边时必须同步改这里。
p.LateralNorm = 0.55;

% ── B) 本功能新增的可调量 ─────────────────────────────────────────
names = [
    "Safety_EnableFilter"      % 1=启用安全层投影；0=只统计不干预（消融用）
    "Safety_EnableSearch"      % 1=启用丢线扫描找线；0=只统计不干预（消融用）
    "Safety_LateralMax"        % 归一化横向偏移安全界，h = LateralMax^2 - e^2
    "Safety_CBFAlpha"          % CBF 的 class-K 增益 (1/s)，越大越激进
    "Safety_CBFSingularTol"    % |A| 小于此值判为奇异，本拍不加 CBF 约束
    "Safety_SpeedBackoff"      % 0~1，靠近屏障边界的减速比例（启发式）
    "Safety_MaxAccelV"         % v 速率约束，单位随 signalUnits（见上）
    "Safety_MaxAccelOmega"     % omega 速率约束 (rad/s²)
    "Safety_HoldTime"          % 丢线后不干预的时长 (s)，对齐视觉的 freezeTimeout
    "Safety_BrakeTime"         % 刹停时长 (s)
    "Safety_ScanRate"          % 原地扫描角速度 (rad/s)
    "Safety_ScanTolerance"     % 扫掠角到位判据 (rad)
    "Safety_ScanTimeout"       % 扫描总时间预算 (s)
    "Safety_DirDeadband"       % 判定扫描方向时的死区
    % ── 几何记忆 / 定向恢复（2026-08-09 加，取代原地扫描成为主恢复策略）──
    "Safety_MemMaxAge"         % 记忆年龄上限 (s)，超过即丢弃退回扫描
    "Safety_MemMinPoints"      % 记忆至少要有几个有效路径点才算可用
    "Safety_MemLookahead"      % 从记忆点里选目标时的前视距离 (m)
    "Safety_RecoverGain"       % 方位角 → omega 的比例增益 (1/s)
    "Safety_RecoverAlignCone"  % |方位角| 超过它就先原地转正 (rad)
    "Safety_RecoverSpeedFrac"  % 恢复时的前进速度，占 MaxV 的比例
    "Safety_RecoverTimeout"    % 定向恢复的时间预算 (s)，超时退回扫描
    ];
defaults = {
    1     % EnableFilter：约束层默认开启，实测是净收益（见 README 的测试小节）
    % EnableSearch 默认**关闭**。2026-08-09 闭环实测：无论原地扫描还是
    % 基于几何记忆的定向恢复，都不如"什么都不做"（沿用模型原有的
    % Minimum_Search_Speed 前爬）——baseline 能找回线并跑完整圈，开了找线的
    % 反而停死。根因在观测而不在恢复策略（ROI 只看到 12 cm 地面带，丢线前
    % 最后一个可靠帧只剩 8 mm 路径、不含任何转弯信息），在 ROI/检测器修好
    % 之前打开它只会变差。改成 1 之前请先重跑 run_safety_governor_sweep。
    0
    0.8
    2.0
    1e-3
    0.3
    defaultMaxAccelV
    10.0
    0.4
    0.3
    0.9
    0.05
    15.0
    0.02
    % 记忆 / 定向恢复。
    % MemMaxAge=8s：baseline 实测从丢线到重新找回约 5.2 s（见
    % runtime/control/README.md 的测试小节），而记忆的年龄是从**最后一个可靠
    % 帧**起算、比丢线时刻还早约 0.5 s，所以预算必须明显大于 5.2 s 才能给定向
    % 恢复公平的机会。8 s 内以约 0.1 m/s 前进只走 0.8 m，死推漂移可接受。
    % RecoverTimeout 同样给 8 s，让**记忆年龄**成为真正的约束条件，而不是让
    % 两个预算互相打架。
    % MemLookahead 取 0.25 m，落在 ROI 实际可见的 0.163~0.283 m 地面带内，
    % 不要设成前视距离 0.40 m——那个值本身就已超出 ROI 可见范围、只能靠多项式
    % 外推得到。
    8.0
    3
    0.25
    1.5
    deg2rad(45)
    0.25
    8.0
    };

values = cell(size(names));
for k = 1:numel(names)
    values{k} = readOrCreate(workspace, names(k), defaults{k});
end

p.EnableFilter   = values{1} > 0.5;
p.EnableSearch   = values{2} > 0.5;
p.LateralMax     = values{3};
p.CBFAlpha       = values{4};
p.CBFSingularTol = values{5};
p.SpeedBackoff   = values{6};
p.MaxAccelV      = values{7};
p.MaxAccelOmega  = values{8};
p.HoldTime       = values{9};
p.BrakeTime      = values{10};
p.ScanRate       = values{11};
p.ScanTolerance  = values{12};
p.ScanTimeout    = values{13};
p.DirDeadband    = values{14};
p.MemMaxAge          = values{15};
p.MemMinPoints       = values{16};
p.MemLookahead       = values{17};
p.RecoverGain        = values{18};
p.RecoverAlignCone   = values{19};
p.RecoverSpeedFrac   = values{20};
p.RecoverTimeout     = values{21};

% 扫描幅度是向量，单独处理（上面那张表只走标量）
p.ScanAmplitudes = readOrCreate(workspace, "Safety_ScanAmplitudes", deg2rad([25, 50, 80]));
p.ScanAmplitudes = sort(abs(p.ScanAmplitudes(:).'));   % 逐级扩大：强制升序、取正
end

function value = readRequired(workspace, name)
try
    value = workspace.evalin(name);
catch
    error("LineFollowerSafety:MissingParameter", ...
        "model workspace variable '%s' is required by the safety governor but does not exist", ...
        name);
end
end

function value = readOrCreate(workspace, name, defaultValue)
try
    value = workspace.evalin(name);
catch
    value = defaultValue;
    workspace.assignin(name, value);
end
end
