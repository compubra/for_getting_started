function out = lf_line_search(u, p)
%LF_LINE_SEARCH 丢线找线状态机 + 几何记忆（**行为生成**，不含任何安全约束）。
%
% 2026-08-09 从原 line_follower_safety_governor 拆出。拆分理由：这一层和
% lf_safety_filter 是两件独立的事——
%   - 本文件**产生**期望指令：线还在就透传 PID+残差的指令，线丢了就接管，
%     按几何记忆定向恢复、必要时原地扫描。它关心的是"该往哪走"。
%   - lf_safety_filter **约束**指令：不管指令是谁产生的（PID、RL 残差、还是
%     本文件的恢复行为），一律投影进安全集。它关心的是"允不允许这么走"。
% 混在一个函数里会让"安全层对指令来源不可知"这条设计原则变得看不出来，也
% 使两者无法各自单独测试/消融。信号链上本文件在前、滤波在后。
%
% ── 状态机 ──────────────────────────────────────────────────────────
%   0 TRACK   found=1，指令透传；同时把本帧路径点存入几何记忆
%   1 HOLD    丢线 <= HoldTime：透传。此时视觉模块自身还在冻结/外推
%             steering_error（见其 freezeTimeout），瞬时遮挡不该被打断
%   5 RECOVER 记忆有效时的**定向恢复**：朝记忆中的路径点前进
%   2 BRAKE   记忆失效/过期时才走：v、omega 线性归零，先停稳再扫
%   3 SCAN    v=0 原地扫描，扫掠角目标序列 +d*A1, -d*A1, +d*A2, ... 逐级扩大
%   4 GIVEUP  扫完或超时：停车等待
% 任何状态下 found=1 立即回 TRACK。
%
% RECOVER 取代原地扫描成为主策略的依据（2026-08-09 闭环实测，详见
% runtime/control/README.md）：丢线 20 s 期间机器人距赛道中心线始终只有
% 3.7~4.5 cm（压在线上），线有 32.2% 的时间就在 ROI 视场内却检测不出来；
% 不干预的 baseline 靠"继续向前爬"约 5 s 就找回了线，原地扫描把车停下反而
% 永远找不回。**保持沿原路径前进**才是有效的恢复行为。
%
% ── 几何记忆 ────────────────────────────────────────────────────────
% 存最后一个**可靠**帧的地面路径点，之后每拍用实测轮速死推
% （lf_memory_propagate），于是丢线期间仍知道线在机体系的哪个方位。
%
% "只存可靠帧"是实测出来的，别改回"每帧都存"：真实丢线不是突然发生的，
% validCount 会先一路衰减再归零——实测记录到丢线前 11 拍是
% 21,19,18,16,15,13,11,9,7,5,2 然后 0。每帧都存的话记忆里留下的必然是丢线
% 前那一帧**最差**的几何（当时只剩 2 个点）。
%
% **已知局限**：记忆的空间跨度受限于 ROI 的地面可视深度。实测最后一个可靠帧
% 的路径点 x 范围只有 [0.163, 0.171] m ——**整条记忆 8 毫米，不含任何转弯
% 信息**。所以定向恢复在急弯丢线场景下同样救不回来，瓶颈在观测不在恢复策略。
% p.EnableSearch 默认因此为 0。
%
% ── 输入/输出 ───────────────────────────────────────────────────────
% u (74×1) = [v_cmd; omega_cmd; lateral_error; heading_error; found;
%             v_meas; omega_meas; path_debug(67)]
%   v_meas / omega_meas  **实测**线速度(m/s)/偏航角速度(rad/s)。用于记忆死推，
%                        以及扫描时扫掠角的积分——用实测而非指令值：指令下游
%                        还要过安全滤波和轮速饱和，用指令积分会高估转过的角度。
%   path_debug           视觉模块 67 路调试向量
%                        [validCount, lookaheadX, lookaheadY, curvature,
%                         a, b, c, x1,y1, ..., x30,y30]，有效点紧凑存放在
%                         前 validCount 个槽位，机体系、X 前 / Y 右正、米。
%
% out (7×1) = [v_des; omega_des; state; phi; mem_valid; mem_age; mem_bearing_deg]
%   v_des/omega_des 是**期望**指令，尚未经过安全约束；与 lf_safety_filter 的
%   输出相减即安全层的干预量。
%
% p 见 lf_safety_defaults.m（两个文件共用同一份参数结构体）。
% p.EnableSearch=0 时状态机照跑（state 仍如实反映丢线阶段，供离线统计），
% 但**不接管指令**——这正是消融对比的口径。

persistent state tLost tFallback phi legIdx dirSign memPoints memCount memAge

STATE_TRACK   = 0;
STATE_HOLD    = 1;
STATE_BRAKE   = 2;
STATE_SCAN    = 3;
STATE_GIVEUP  = 4;
STATE_RECOVER = 5;

MAX_PATH_POINTS = 30;
DEBUG_WIDTH     = 7 + 2 * MAX_PATH_POINTS;   % 67

if isempty(state),     state     = STATE_TRACK; end
if isempty(tLost),     tLost     = 0;           end
if isempty(tFallback), tFallback = 0;           end
if isempty(phi),       phi       = 0;           end
if isempty(legIdx),    legIdx    = 1;           end
if isempty(dirSign),   dirSign   = 1;           end
if isempty(memPoints) || ~isequal(size(memPoints), [MAX_PATH_POINTS, 2])
    memPoints = zeros(MAX_PATH_POINTS, 2);
end
if isempty(memCount),  memCount  = 0;           end
if isempty(memAge),    memAge    = inf;         end

u = u(:);
if numel(u) ~= 7 + DEBUG_WIDTH
    error("LineFollowerSearch:BadInput", ...
        ["expected a %d-element [v; omega; lateral_error; heading_error; found; " ...
         "v_meas; omega_meas; path_debug(%d)] vector, got %d"], ...
        7 + DEBUG_WIDTH, DEBUG_WIDTH, numel(u));
end

vCmd      = u(1);
omegaCmd  = u(2);
e         = u(3);
found     = u(5) > 0.5;
omegaMeas = u(7);
pathDebug = u(8:end);

prevState = state;

% ── 几何记忆的维护 ─────────────────────────────────────────────────
newCount = max(0, min(MAX_PATH_POINTS, round(pathDebug(1))));
if found && newCount >= p.MemMinPoints
    memCount = newCount;
    memPoints(:) = 0;
    for k = 1:memCount
        memPoints(k, 1) = pathDebug(7 + 2 * k - 1);
        memPoints(k, 2) = pathDebug(7 + 2 * k);
    end
    memAge = 0;
else
    memPoints = lf_memory_propagate(memPoints, memCount, u(6), omegaMeas, p.Ts);
    memAge = memAge + p.Ts;
end
memValid = memCount >= p.MemMinPoints && memAge <= p.MemMaxAge;

memBearing = 0;
targetOk = false;
if memValid
    [xTgt, yTgt, targetOk] = lf_memory_target(memPoints, memCount, p.MemLookahead);
    if targetOk
        memBearing = atan2(yTgt, xTgt);   % >0 表示目标在机体右侧
    end
end
memUsable = memValid && targetOk;

% ── 状态机 ─────────────────────────────────────────────────────────
if found
    state     = STATE_TRACK;
    tLost     = 0;
    tFallback = 0;
    legIdx    = 1;
    phi       = 0;
    % 锁存扫描方向：路径偏右(e>0) → 先向右扫(omega<0)
    if abs(e) > p.DirDeadband
        dirSign = -sign(e);
    elseif abs(omegaCmd) > p.DirDeadband
        dirSign = sign(omegaCmd);
    end
    vDes     = vCmd;
    omegaDes = omegaCmd;
else
    tLost = tLost + p.Ts;
    if tLost <= p.HoldTime
        % 瞬时遮挡：不干预，交给视觉模块自己的冻结/外推
        state    = STATE_HOLD;
        vDes     = vCmd;
        omegaDes = omegaCmd;
        tFallback = 0;
    elseif memUsable && tLost <= p.HoldTime + p.RecoverTimeout
        % 定向恢复：朝记忆中的路径点走。**不经过 BRAKE**——实测表明保持沿原
        % 路径前进才是有效的恢复行为，刹停再重新加速反而丢掉了这个机制。
        state = STATE_RECOVER;
        omegaDes = min(p.MaxOmega, max(-p.MaxOmega, -p.RecoverGain * memBearing));
        if abs(memBearing) > p.RecoverAlignCone
            vDes = 0;             % 目标偏得太多：先原地转正，避免朝错误方向冲
        else
            vDes = p.RecoverSpeedFrac * p.MaxV;
        end
        tFallback = 0;
    elseif tFallback < p.BrakeTime
        % 记忆不可用或已过期 → 回退到 刹停→原地扫描→放弃 链。这条链用自己的
        % 时钟 tFallback，不能沿用 tLost：RECOVER 可能已经先占掉了几秒，按
        % tLost 计时会让刹停/扫描的时长被莫名压缩甚至跳过。
        state = STATE_BRAKE;
        tFallback = tFallback + p.Ts;
        ramp  = 1 - tFallback / max(eps, p.BrakeTime);
        vDes     = vCmd * ramp;
        omegaDes = omegaCmd * ramp;
    elseif tFallback <= p.BrakeTime + p.ScanTimeout && legIdx <= 2 * numel(p.ScanAmplitudes)
        state = STATE_SCAN;
        tFallback = tFallback + p.Ts;
        if prevState ~= STATE_SCAN
            % 进入扫描的这一拍把扫掠角清零：phi 是相对"扫描起点航向"的角度，
            % 不是相对丢线瞬间航向（HOLD/RECOVER 期间还在转动）
            phi    = 0;
            legIdx = 1;
        end
        % 目标序列 +d*A1, -d*A1, +d*A2, -d*A2, ... 逐级扩大
        ampIdx = ceil(legIdx / 2);
        if mod(legIdx, 2) == 1
            legSign = 1;
        else
            legSign = -1;
        end
        target = dirSign * legSign * p.ScanAmplitudes(ampIdx);
        phiErr = target - phi;
        if abs(phiErr) <= p.ScanTolerance
            legIdx   = legIdx + 1;   % 本段到位，下一拍换目标
            omegaDes = 0;
        else
            omegaDes = sign(phiErr) * p.ScanRate;
        end
        vDes = 0;   % 原地扫描：不前进，绝不会因找线冲出赛道
    else
        state     = STATE_GIVEUP;
        tFallback = tFallback + p.Ts;
        vDes      = 0;
        omegaDes  = 0;
    end
end

% 扫掠角按**实测**偏航角速度积分。拆分之前这里用的是安全滤波之后的指令值；
% 现在本文件看不到滤波结果，而实测值本来就比任何指令值都准（下游还有轮速
% 饱和），顺势改用它，也避免了"滤波→状态机"的代数环。
if state == STATE_SCAN
    phi = phi + omegaMeas * p.Ts;
end

if ~p.EnableSearch
    % 消融：状态机照跑（state 仍如实反映丢线阶段），但不接管指令
    vDes     = vCmd;
    omegaDes = omegaCmd;
end

out = [vDes; omegaDes; state; phi; ...
       double(memUsable); min(memAge, 1e6); rad2deg(memBearing)];
end
