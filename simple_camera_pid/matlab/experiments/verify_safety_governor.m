function results = verify_safety_governor()
%VERIFY_SAFETY_GOVERNOR lf_line_search + lf_safety_filter 的独立单元验证。
%
% 不需要打开任何 .slx、不需要 MuJoCo——纯函数级验证，跑一次几秒。改动安全层
% 或找线状态机后请重跑：
%
%   cd matlab; verify_safety_governor
%
% 覆盖的东西（见各 case 的名字）：CBF 的符号/边界/奇异保护、箱式与速率约束、
% 对纯 PID 基线的透明性、找线状态机的完整状态序列与扫掠角轨迹。
%
% 不覆盖的东西（诚实声明，别把这个脚本当成"安全层已验证"的全部依据）：
%   - 不验证闭环性能。CBF 只保证"若模型正确且约束可行则安全集前向不变"，
%     真实横向动力学与本文件用的 d(y_L)/dt = v*sin(psi)+L*omega 有建模误差
%     （忽略了路径曲率项、轮子打滑、视觉延迟），闭环效果必须靠仿真/实机看。
%   - 不验证与 SAC 残差策略的交互（残差在求和处已进 v_cmd/omega_cmd，
%     本层看不到它是谁产生的）。
%   - 不验证实机时序（Interpreted MATLAB Function 的执行开销未测）。

addpath(genpath(fullfile(fileparts(mfilename("fullpath")), "..", "runtime")));

results = struct("name", {}, "pass", {}, "detail", {});
p = testParams();

results = runCase(results, "CBF_violated_positive_forces_right_turn", @() cbfViolatedPositive(p));
results = runCase(results, "CBF_violated_negative_forces_left_turn",  @() cbfViolatedNegative(p));
results = runCase(results, "CBF_centre_is_unconstrained",             @() cbfCentre(p));
results = runCase(results, "CBF_inside_set_is_permissive",            @() cbfInsideSet(p));
results = runCase(results, "CBF_drives_state_back_into_safe_set",     @() cbfForwardInvariance(p));
results = runCase(results, "box_forbids_reverse_and_caps_v",          @() boxLimits(p));
results = runCase(results, "rate_limits_bound_per_tick_change",       @() rateLimits(p));
results = runCase(results, "baseline_pid_passes_through_untouched",   @() baselineTransparent(p));
results = runCase(results, "search_state_sequence_is_complete",       @() searchSequence(p));
results = runCase(results, "search_scans_toward_last_seen_side",      @() searchDirection(p));
results = runCase(results, "search_sweep_amplitudes_expand",          @() searchSweep(p));
results = runCase(results, "search_never_drives_forward",             @() searchNoForward(p));
results = runCase(results, "reacquire_returns_to_track_immediately",  @() searchReacquire(p));
results = runCase(results, "ablation_filter_off_still_reports_h",     @() ablationFilterOff(p));
results = runCase(results, "ablation_search_off_leaves_cmd_alone",    @() ablationSearchOff(p));
results = runCase(results, "memory_deadreckons_a_pure_translation",   @() memTranslate());
results = runCase(results, "memory_deadreckons_a_pure_rotation",      @() memRotate());
results = runCase(results, "memory_target_follows_path_order",        @() memTargetOrder());
results = runCase(results, "recover_steers_toward_remembered_line",   @() recoverSteers(p));
results = runCase(results, "recover_expires_then_falls_back_to_scan", @() recoverExpires(p));

nPass = sum([results.pass]);
fprintf("\n==== verify_safety_governor: %d/%d passed ====\n", nPass, numel(results));
for k = 1:numel(results)
    if results(k).pass
        fprintf("  PASS  %s\n", results(k).name);
    else
        fprintf("  FAIL  %s\n        %s\n", results(k).name, results(k).detail);
    end
end
if nPass < numel(results)
    error("VerifySafetyGovernor:Failed", "%d case(s) failed", numel(results) - nPass);
end
end

% ─────────────────────────────────────────────────────────────────────
function p = testParams()
% 与 lf_safety_defaults.m 的默认值一致，但不依赖模型工作区（保持本脚本可
% 脱离 .slx 运行）。改了 lf_safety_defaults 的默认值请同步这里。
p = struct( ...
    "EnableFilter", true, "EnableSearch", true, ...
    "Ts", 0.05, "SpeedScale", 0.03, "Lookahead", 0.40, "LateralNorm", 0.55, ...
    "MaxV", 5.0, "MaxOmega", 1.5, ...
    "LateralMax", 0.8, "CBFAlpha", 2.0, "CBFSingularTol", 1e-3, "SpeedBackoff", 0.3, ...
    "MaxAccelV", 40.0, "MaxAccelOmega", 10.0, ...
    "HoldTime", 0.4, "BrakeTime", 0.3, "ScanRate", 0.9, "ScanTolerance", 0.05, ...
    "ScanTimeout", 15.0, "DirDeadband", 0.02, ...
    "MemMaxAge", 8.0, "MemMinPoints", 3, "MemLookahead", 0.25, ...
    "RecoverGain", 1.5, "RecoverAlignCone", deg2rad(45), ...
    "RecoverSpeedFrac", 0.25, "RecoverTimeout", 8.0, ...
    "ScanAmplitudes", deg2rad([25, 50, 80]));
end

function out = stepOnce(p, v, omega, e, psi, found)
% 单拍调用，先清 persistent，用于只关心稳态约束的 case。为了绕开速率约束
% 的冷启动（vPrev/omegaPrev 从 0 起），先用同样的输入预热若干拍。
clear lf_line_search; clear lf_safety_filter;
for k = 1:40
    out = governor(mkInput(v, omega, e, psi, found), p);
end
end

function [out, idx] = unpack()
% 本文件 governor() shim 拼出的 16 路布局（与两级拆分前保持一致，使既有用例
% 不必重写）。前两路是最终下发值；3/4 是其副本；5/6 是安全滤波**之前**的期望值。
idx = struct("v", 1, "omega", 2, "vDes", 5, "omegaDes", 6, "state", 7, "h", 8, ...
    "omegaLo", 9, "omegaHi", 10, "cbfActive", 11, "infeasible", 12, "phi", 13, ...
    "memValid", 14, "memAge", 15, "memBearing", 16);
out = [];
end

function out = governor(u, p)
%GOVERNOR 把拆分后的两级串起来：状态机(lf_line_search) → 安全滤波(lf_safety_filter)。
%
% 拆分后模型里是两个独立子系统，本 shim 只为让既有 20 个用例继续以"整条链"的
% 口径断言，同时保持与拆分前一致的 16 路输出布局。想单独测某一级就直接调那一级
% （memTranslate/memRotate/memTargetOrder 就是这么做的）。
s = lf_line_search(u, p);                                   % [vDes; omegaDes; state; phi; memValid; memAge; memBearing]
f = lf_safety_filter([s(1); s(2); u(3); u(4); u(5)], p);    % [vSafe; omegaSafe; h; lo; hi; act; infeas]
out = [f(1); f(2); ...            % 1,2  最终下发
       f(1); f(2); ...            % 3,4  副本（与拆分前布局一致）
       s(1); s(2); ...            % 5,6  滤波前的期望值
       s(3); ...                  % 7    state
       f(3); f(4); f(5); f(6); f(7); ...  % 8..12 h/lo/hi/active/infeasible
       s(4); s(5); s(6); s(7)];   % 13..16 phi/memValid/memAge/memBearing
end

function u = mkInput(v, omega, e, psi, found, opts)
%MKINPUT 组装 74 路输入向量。
% opts.points 为 (N×2) 机体系路径点；缺省时给一条正前方的直线，使 found=1
% 的用例自然带上一段可用记忆（与真实运行一致）。
arguments
    v; omega; e; psi; found
    opts.points double = [(0.16:0.02:0.34).', zeros(10, 1)]
    opts.vMeas double = NaN
    opts.omegaMeas double = NaN
end
vMeas = opts.vMeas; if isnan(vMeas), vMeas = v * 0.03; end       % 信号单位→m/s
omegaMeas = opts.omegaMeas; if isnan(omegaMeas), omegaMeas = omega; end

debugWidth = 7 + 2 * 30;
dbg = zeros(debugWidth, 1);
n = size(opts.points, 1);
dbg(1) = n;
for k = 1:n
    dbg(7 + 2 * k - 1) = opts.points(k, 1);
    dbg(7 + 2 * k)     = opts.points(k, 2);
end
u = [v; omega; e; psi; found; vMeas; omegaMeas; dbg];
end

% ── CBF ──────────────────────────────────────────────────────────────
function [ok, detail] = cbfViolatedPositive(p)
[~, I] = unpack();
% e=+0.9 已越界（LateralMax=0.8）。路径在右、车偏左 → 必须右转(omega<0)
out = stepOnce(p, 5, 1.5, 0.9, 0, 1);           % 指令要求打满左转
ok = out(I.omega) < 0 && out(I.h) < 0 && out(I.cbfActive) == 1;
detail = sprintf("omega=%.3f (expect<0), h=%.3f (expect<0), cbfActive=%d", ...
    out(I.omega), out(I.h), out(I.cbfActive));
end

function [ok, detail] = cbfViolatedNegative(p)
[~, I] = unpack();
out = stepOnce(p, 5, -1.5, -0.9, 0, 1);
ok = out(I.omega) > 0 && out(I.h) < 0 && out(I.cbfActive) == 1;
detail = sprintf("omega=%.3f (expect>0), h=%.3f (expect<0), cbfActive=%d", ...
    out(I.omega), out(I.h), out(I.cbfActive));
end

function [ok, detail] = cbfCentre(p)
[~, I] = unpack();
% e=0 → A=-e*L=0 落在奇异保护里，本拍不加 CBF 约束，指令应原样通过
out = stepOnce(p, 5, 1.0, 0.0, 0, 1);
ok = out(I.cbfActive) == 0 && abs(out(I.omega) - 1.0) < 1e-9;
detail = sprintf("cbfActive=%d (expect 0), omega=%.6f (expect 1.0)", ...
    out(I.cbfActive), out(I.omega));
end

function [ok, detail] = cbfInsideSet(p)
[~, I] = unpack();
% e=0.5 仍在安全集内(h>0)，屏障应给出一个宽松的上界而非把指令压死
out = stepOnce(p, 5, 0.5, 0.5, 0, 1);
ok = out(I.h) > 0 && out(I.omegaHi) > 0.5 && abs(out(I.omega) - 0.5) < 1e-9;
detail = sprintf("h=%.3f, omegaHi=%.3f (expect>0.5), omega=%.3f (expect 0.5 untouched)", ...
    out(I.h), out(I.omegaHi), out(I.omega));
end

function [ok, detail] = cbfForwardInvariance(p)
[~, I] = unpack();
% 闭环小实验：用本文件 CBF 所依据的同一套横向动力学积分，从安全集内部
% (e0=0.6) 出发，控制器一直要求打满左转（会把 e 推向 +1 越界）。CBF 应当
% 把 e 卡在 LateralMax 附近而不越界。
% 注意：这**只验证"在建模假设成立时约束逻辑自洽"**，不是对真实小车的
% 安全性证明——真实动力学与该模型有误差，见文件头的免责说明。
clear lf_line_search; clear lf_safety_filter;
e = 0.6; psi = 0; vCmd = 5;
eMax = 0;
for k = 1:200
    out = governor(mkInput(vCmd, 1.5, e, psi, 1), p);
    vPhys = out(I.v) * p.SpeedScale;
    e = e + p.Ts * (vPhys * sin(psi) + p.Lookahead * out(I.omega)) / p.LateralNorm;
    eMax = max(eMax, abs(e));
end
% 允许一点越界余量：离散化 + 速率约束下屏障不可能是逐拍精确的
ok = eMax <= p.LateralMax + 0.05;
detail = sprintf("max|e|=%.4f (expect <= %.2f)", eMax, p.LateralMax + 0.05);
end

% ── 箱式 / 速率约束 ──────────────────────────────────────────────────
function [ok, detail] = boxLimits(p)
[~, I] = unpack();
outNeg = stepOnce(p, -3, 0, 0, 0, 1);       % 残差把 v 拉成负 → 应被夹到 0
outBig = stepOnce(p, 99, 0, 0, 0, 1);       % v 超上限 → 应夹到 MaxV
outOmg = stepOnce(p, 5, 99, 0, 0, 1);       % omega 超上限 → 夹到 MaxOmega
ok = outNeg(I.v) >= 0 && abs(outBig(I.v) - p.MaxV) < 1e-9 && ...
     abs(outOmg(I.omega) - p.MaxOmega) < 1e-9;
detail = sprintf("v(neg)=%.3f (expect>=0), v(big)=%.3f (expect %.1f), omega=%.3f (expect %.1f)", ...
    outNeg(I.v), outBig(I.v), p.MaxV, outOmg(I.omega), p.MaxOmega);
end

function [ok, detail] = rateLimits(p)
[~, I] = unpack();
clear lf_line_search; clear lf_safety_filter;
% 从静止直接要求打满，逐拍变化量不得超过 accel*Ts
prevV = 0; prevW = 0; worstV = 0; worstW = 0;
for k = 1:10
    out = governor(mkInput(5, 1.5, 0, 0, 1), p);
    worstV = max(worstV, abs(out(I.v) - prevV));
    worstW = max(worstW, abs(out(I.omega) - prevW));
    prevV = out(I.v); prevW = out(I.omega);
end
okAll = worstV <= p.MaxAccelV * p.Ts + 1e-9 && worstW <= p.MaxAccelOmega * p.Ts + 1e-9;
ok = okAll;
detail = sprintf("max dv=%.4f (limit %.4f), max domega=%.4f (limit %.4f)", ...
    worstV, p.MaxAccelV * p.Ts, worstW, p.MaxAccelOmega * p.Ts);
end

function [ok, detail] = baselineTransparent(p)
[~, I] = unpack();
% 典型纯 PID 工况：线在视野内、误差不大、指令远离所有边界 → 安全层必须
% 完全不改指令。这条挂了说明默认参数会伤到既有 PID 调参。
v = 4.0; omega = 0.4; e = 0.15;
out = stepOnce(p, v, omega, e, 0.05, 1);
% v 会被 SpeedBackoff 按边界余量缩一点，这是设计内行为；omega 必须原样
expectedScale = 1 - p.SpeedBackoff * (1 - max(0, p.LateralMax^2 - e^2) / p.LateralMax^2);
ok = abs(out(I.omega) - omega) < 1e-9 && abs(out(I.v) - v * expectedScale) < 1e-9;
detail = sprintf("omega=%.6f (expect %.6f), v=%.6f (expect %.6f)", ...
    out(I.omega), omega, out(I.v), v * expectedScale);
end

% ── 找线状态机 ───────────────────────────────────────────────────────
function [states, outs] = runLostSequence(p, nTicks, e0)
clear lf_line_search; clear lf_safety_filter;
outs = zeros(16, nTicks);
% 先跟踪 20 拍建立 dirSign，再一直丢线
for k = 1:20
    governor(mkInput(5, 0.3, e0, 0, 1), p);
end
% omega_meas 必须闭环回灌：lf_line_search 的扫掠角 phi 现在按**实测**角速度
% 积分（模型里来自轮速反馈），若像 mkInput 默认那样固定成输入指令，phi 会
% 单调爬升、扫描腿永远换不过来。这里用"理想执行器"近似——上一拍实际施加的
% omega 即本拍的实测值——这也是真实模型里那条反馈路径的一阶模型。
omegaMeas = 0;
for k = 1:nTicks
    outs(:, k) = governor(mkInput(5, 0.3, 0, 0, 0, omegaMeas=omegaMeas), p);
    omegaMeas = outs(2, k);
end
[~, I] = unpack();
states = outs(I.state, :);
end

function [ok, detail] = searchSequence(p)
% 20 s 足够走完 HOLD→BRAKE→SCAN→GIVEUP
[states, ~] = runLostSequence(p, 400, 0.3);
seen = unique(states, "stable");
% 记忆可用时先走 RECOVER(5)，记忆过期(MemMaxAge)后才回退到 刹停→扫描→放弃
ok = isequal(seen(:).', [1 5 2 3 4]);
detail = sprintf("state sequence = [%s] (expect [1 5 2 3 4] = HOLD,RECOVER,BRAKE,SCAN,GIVEUP)", ...
    num2str(seen(:).'));
end

function [ok, detail] = searchDirection(p)
[~, I] = unpack();
% e>0 表示路径在右侧 → 应先向右扫，即扫描段首个非零 omega 为负
[states, outsR] = runLostSequence(p, 400, 0.5);
firstScan = find(states == 3, 1);
omegaR = outsR(I.omega, firstScan + 2);          % 跳过速率约束爬坡的头两拍
[statesL, outsL] = runLostSequence(p, 400, -0.5);
firstScanL = find(statesL == 3, 1);
omegaL = outsL(I.omega, firstScanL + 2);
ok = omegaR < 0 && omegaL > 0;
detail = sprintf("e>0 -> omega=%.3f (expect<0); e<0 -> omega=%.3f (expect>0)", omegaR, omegaL);
end

function [ok, detail] = searchSweep(p)
[~, I] = unpack();
[states, outs] = runLostSequence(p, 400, 0.5);
scanMask = states == 3;
phi = outs(I.phi, scanMask);
% 逐级扩大：每一级的正/负峰值应当依次增大，且最大扫掠角接近最大幅度
peaks = [max(phi), min(phi)];
ok = abs(min(phi)) > deg2rad(70) && max(phi) > deg2rad(45) && ...
     abs(min(phi)) <= max(p.ScanAmplitudes) + p.ScanTolerance + 0.05;
detail = sprintf("phi range = [%.1f, %.1f] deg (expect to reach about -80 and +50)", ...
    rad2deg(peaks(2)), rad2deg(peaks(1)));
end

function [ok, detail] = searchNoForward(p)
[~, I] = unpack();
[states, outs] = runLostSequence(p, 400, 0.5);
scanning = ismember(states, [3 4]);          % SCAN 与 GIVEUP（RECOVER=5 是有意前进的，不在此列）
ok = all(outs(I.v, scanning) < 1e-9);
detail = sprintf("max v while scanning/giving up = %.6g (expect 0)", ...
    max(outs(I.v, scanning)));
end

function [ok, detail] = searchReacquire(p)
[~, I] = unpack();
clear lf_line_search; clear lf_safety_filter;
for k = 1:20
    governor(mkInput(5, 0.3, 0.5, 0, 1), p);
end
for k = 1:200                                 % 丢线 10 s，确保已进 SCAN
    out = governor(mkInput(5, 0.3, 0, 0, 0), p);
end
wasScanning = out(I.state) == 3;
outBack = governor(mkInput(5, 0.3, 0.2, 0, 1), p);
ok = wasScanning && outBack(I.state) == 0 && outBack(I.v) > 0;
detail = sprintf("state before=%d (expect 3), after reacquire=%d (expect 0), v=%.3f (expect>0)", ...
    out(I.state), outBack(I.state), outBack(I.v));
end

% ── 消融开关 ─────────────────────────────────────────────────────────
function [ok, detail] = ablationFilterOff(p)
[~, I] = unpack();
% 关掉安全层：越界的 e=0.9 + 打满左转指令必须原样通过（这就是"无约束基线"），
% 但 h 仍要如实报告为负，否则统计不出违约率
q = p; q.EnableFilter = false;
out = stepOnce(q, 5, 1.5, 0.9, 0, 1);
ok = abs(out(I.omega) - 1.5) < 1e-9 && out(I.h) < 0 && out(I.cbfActive) == 1;
detail = sprintf("omega=%.6f (expect 1.5 untouched), h=%.3f (expect<0), cbfActive=%d (expect 1)", ...
    out(I.omega), out(I.h), out(I.cbfActive));
end

function [ok, detail] = ablationSearchOff(p)
[~, I] = unpack();
% 关掉找线：长时间丢线时指令不被接管，但 state 仍要走到 GIVEUP 以便统计
q = p; q.EnableSearch = false;
[states, outs] = runLostSequence(q, 400, 0.5);
lastState = states(end);
% 两条断言：(1) 指令全程未被改动——丢线时 e 被视觉置零，CBF 也无约束，
% omega 应始终等于输入的 0.3；(2) 状态机仍如实推进到了恢复链里（不再停在
% TRACK/HOLD），这正是"关掉也能统计"的意义。
% 不断言具体停在哪个状态：RECOVER 引入后，到达 GIVEUP 的时刻取决于
% MemMaxAge + BrakeTime + 扫描进度，而扫描的 omega 在消融下并未真正施加、
% 扫掠角不会按预期推进，钉死某个末态只会变成一条脆弱的断言。
ok = all(abs(outs(I.omega, :) - 0.3) < 1e-9) && ~ismember(lastState, [0 1]);
detail = sprintf("final state=%d (expect not 0/1), max|omega-0.3|=%.2e (expect 0)", ...
    lastState, max(abs(outs(I.omega, :) - 0.3)));
end

% ── 几何记忆 / 定向恢复 ──────────────────────────────────────────────
function [ok, detail] = memTranslate()
% 纯前进：正前方 0.30 m 的点，前进 0.10 m 后应变成 0.20 m，横向不变
pts = [0.30, 0.0; 0.40, 0.0];
out = lf_memory_propagate(pts, 2, 1.0, 0.0, 0.10);   % v*dt = 0.10 m
ok = abs(out(1,1) - 0.20) < 1e-12 && abs(out(2,1) - 0.30) < 1e-12 && ...
     all(abs(out(:,2)) < 1e-12);
detail = sprintf("x: %.4f/%.4f (expect 0.20/0.30), y: %.2e/%.2e (expect 0)", ...
    out(1,1), out(2,1), out(1,2), out(2,2));
end

function [ok, detail] = memRotate()
% 纯左转 90°(omega>0)：正前方的点应转到正右方(+y)。符号写反的话会跑到 -y，
% 而丢线时没有视觉能纠正它 —— 这条就是防那个静默错误的。
pts = [0.30, 0.0];
out = lf_memory_propagate(pts, 1, 0.0, pi/2, 1.0);   % dtheta = +90 deg
ok = abs(out(1,1)) < 1e-12 && abs(out(1,2) - 0.30) < 1e-12;
detail = sprintf("(x,y) = (%.4f, %.4f), expect (0, +0.30) 即正右方", out(1,1), out(1,2));
end

function [ok, detail] = memTargetOrder()
% 目标必须按**路径顺序**选，而不是按距离最近选。构造一条向前伸展的路径，
% 机器人已转身使近端点跑到身后：按距离会挑到身后的点、把车往回引。
pts = [-0.10, 0.0; 0.10, 0.0; 0.30, 0.0; 0.50, 0.0];
[xt, yt, ok1] = lf_memory_target(pts, 4, 0.25);
ok = ok1 && abs(xt - 0.30) < 1e-12 && abs(yt) < 1e-12;
detail = sprintf("target=(%.3f, %.3f) (expect (0.30, 0) = 路径顺序上第一个 >=0.25m 的点)", xt, yt);
end

function [ok, detail] = recoverSteers(p)
[~, I] = unpack();
% 记忆里的线偏在右侧 → 丢线后应进 RECOVER 并右转(omega<0)
right = [0.20, 0.10; 0.26, 0.14; 0.32, 0.18; 0.38, 0.22];
clear lf_line_search; clear lf_safety_filter;
for k = 1:20
    governor(mkInput(5, 0, 0.2, 0, 1, points=right), p);
end
out = [];
for k = 1:20   % 1.0 s：越过 HoldTime(0.4s) 进入 RECOVER
    out = governor(mkInput(5, 0, 0, 0, 0, points=right), p);
end
ok = out(I.state) == 5 && out(I.omega) < 0 && out(I.memValid) == 1 && out(I.memBearing) > 0;
detail = sprintf("state=%d (expect 5=RECOVER), omega=%.3f (expect<0), memValid=%d, bearing=%.1f deg (expect>0)", ...
    out(I.state), out(I.omega), out(I.memValid), out(I.memBearing));
end

function [ok, detail] = recoverExpires(p)
% 记忆过期(MemMaxAge)之后必须退回刹停→扫描，不能一直 RECOVER
[states, ~] = runLostSequence(p, 400, 0.3);
tMem = p.MemMaxAge / p.Ts;                    % 记忆失效的拍数
early = states(round(tMem * 0.5));            % 记忆仍有效
late  = states(end);                          % 早已过期
ok = early == 5 && ismember(late, [3 4]);
detail = sprintf("t=%.1fs state=%d (expect 5=RECOVER); 末态 state=%d (expect 3=SCAN 或 4=GIVEUP)", ...
    0.5 * p.MemMaxAge, early, late);
end

% ─────────────────────────────────────────────────────────────────────
function results = runCase(results, name, fn)
try
    [ok, detail] = fn();
catch err
    ok = false;
    detail = sprintf("threw %s: %s", err.identifier, err.message);
end
results(end + 1) = struct("name", name, "pass", logical(ok), "detail", detail);
end
