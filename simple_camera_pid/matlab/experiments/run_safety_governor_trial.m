function dataFile = run_safety_governor_trial(label, options)
%RUN_SAFETY_GOVERNOR_TRIAL 跑一次带安全层的残差循线试验并归档。
%
%   dataFile = run_safety_governor_trial("both")
%   dataFile = run_safety_governor_trial("baseline", EnableFilter=false, ...
%                                        EnableSearch=false, StopTime=120)
%
% 每次运行保存一个带时间戳的 .mat 到 simulation_data/safety_governor/，内含
% 原始 SimulationOutput、真值中心线、以及下面算好的指标结构体。对比分析用
% compare_safety_governor.m 读这些文件。
%
% **必须在能渲染的 MATLAB 进程里跑。** 本机 MATLAB 直接启动时 MuJoCo 相机
% 渲染是坏的（MATLAB 自带 libstdc++ 比系统 Mesa 需要的旧，详见
% runtime/control/README.md 末节），视觉会全程 found=0，跑出来的只是原地扫描。
% 从命令行这样起：
%
%   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 matlab -batch "..."
%
% 脚本会自己检查 found 率，若疑似渲染失效会显式警告，不会让你拿到一份看起来
% 正常其实全无效的数据。
%
% 指标口径见文件末尾 computeMetrics 的注释。真值横向/航向偏差用仓库已有的
% rl_true_track_errors + get_track_centerline，不另造一套。

arguments
    label (1, 1) string
    options.StopTime (1, 1) double {mustBePositive} = 120
    options.MapKey (1, 1) string = "simple"
    options.EnableFilter (1, 1) logical = true
    options.EnableSearch (1, 1) logical = true
    options.AgentFile (1, 1) string = ""
    options.LateralSafeBound (1, 1) double = 0.8   % 归一化，用于统计违约率
    % 连续丢线多少个控制拍后 Done_Detection 终止仿真。模型默认 100 拍(5 s)，
    % 那是 **RL 训练的回合终止条件**——用它评估部署行为会把找线功能判死：
    % HOLD(0.4s)+BRAKE(0.3s) 之后只剩 4.3 s 扫描，而完整三级扫描要 ~10.5 s，
    % 仿真在扫完之前就被掐断，search_only 和 baseline 必然看起来一模一样。
    % 评估找线时把它调到大于扫描总预算（默认 400 拍 = 20 s）。
    options.DoneSteps (1, 1) double {mustBePositive} = 400
end

model = "visual_line_follower_sac_residual";
matlabDir = fileparts(fileparts(mfilename("fullpath")));
addpath(genpath(fullfile(matlabDir, "runtime")));
addpath(genpath(fullfile(matlabDir, "train")));

if ~bdIsLoaded(model)
    load_system(model);
end

% ── 载入训练好的残差智能体 ─────────────────────────────────────────
% 模型里的 SAC_Agent 块读 base 工作区的 trainedAgent，模型自己不会赋值
agentFile = options.AgentFile;
if strlength(agentFile) == 0
    candidates = dir(fullfile(matlabDir, "simulation_data", "sac_training", "sac_agent_*.mat"));
    if isempty(candidates)
        error("SafetyTrial:NoAgent", ...
            "no trained agent found under simulation_data/sac_training/");
    end
    [~, newest] = max([candidates.datenum]);
    agentFile = string(fullfile(candidates(newest).folder, candidates(newest).name));
end
loaded = load(agentFile);
assignin("base", "trainedAgent", loaded.trainedAgent);

% ── 设置地图与安全层开关 ──────────────────────────────────────────
% 必须用 set_turtlebot3_mujoco_scene 换图，**不能只写 TurtleBot3MuJoCoMap**：
% 该模型的 InitFcn(configurePortablePaths) 是刻意设计成"只把 Plant 当前指向的
% 场景重新锚定路径、不挑选地图"的（这样用户在 Simulink 里手动换图不会被覆盖），
% 所以写模型工作区变量对它毫无作用——2026-08-09 第一轮 sweep 就是这么把
% track_hard 跑成了 simple，八组结果逐位相同才发现。
set_turtlebot3_mujoco_scene(model, options.MapKey);
workspace = get_param(model, "ModelWorkspace");
% 写模型工作区（而不是 base）：InitFcn 里的 lf_safety_defaults 是"有就读、
% 没有才写默认值"，所以这里写进去的值会被读走并转成 LF_Safety
workspace.assignin("Safety_EnableFilter", double(options.EnableFilter));
workspace.assignin("Safety_EnableSearch", double(options.EnableSearch));
workspace.assignin("Safety_LateralMax", options.LateralSafeBound);
workspace.assignin("SAC_Done_Steps", options.DoneSteps);

fprintf("=== trial '%s': filter=%d search=%d map=%s stop=%gs ===\n", ...
    label, options.EnableFilter, options.EnableSearch, options.MapKey, options.StopTime);

out = sim(model, "StopTime", num2str(options.StopTime));

% ── 真值中心线 ───────────────────────────────────────────────────
plant = Simulink.ID.getFullName(model + ":16");
sceneFile = string(strip(string(get_param(plant, "xmlFile"))));
% 断言实际跑的就是请求的那张图。换图是"写模型工作区无效、必须调
% set_turtlebot3_mujoco_scene"的坑（见上），静默跑错图会让整轮对比作废且
% 极难察觉——宁可在这里直接报错。
[~, actualScene] = fileparts(sceneFile);
if ~contains(actualScene, options.MapKey)
    error("SafetyTrial:WrongMap", ...
        "requested map '%s' but the plant actually ran '%s'", ...
        options.MapKey, actualScene);
end
centerline = get_track_centerline(sceneFile);

parameters = struct( ...
    "Label", label, "MapKey", options.MapKey, "SceneFile", sceneFile, ...
    "AgentFile", agentFile, "StopTime", options.StopTime, ...
    "EnableFilter", options.EnableFilter, "EnableSearch", options.EnableSearch, ...
    "LateralSafeBound", options.LateralSafeBound, ...
    "DoneSteps", options.DoneSteps, ...
    "MaxWheelSpeed", workspace.evalin("MaxWheelSpeed"), ...
    "WheelRadius", workspace.evalin("TB3_WheelRadius"), ...
    "WheelSeparation", workspace.evalin("TB3_WheelSeparation"), ...
    "LF_Safety", evalin("base", "LF_Safety"));

metrics = computeMetrics(out, centerline, parameters);

dataDir = fullfile(matlabDir, "simulation_data", "safety_governor");
if ~isfolder(dataDir)
    mkdir(dataDir);
end
timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
% 文件名带地图 key：compare_safety_governor 按 "<map>_<label>_*.mat" 检索
safeLabel = regexprep(label, "[^A-Za-z0-9_-]", "_");
dataFile = fullfile(dataDir, options.MapKey + "_" + safeLabel + "_" + timestamp + ".mat");
save(dataFile, "out", "parameters", "metrics", "centerline", "-v7.3");

printMetrics(label, metrics);
fprintf("saved: %s\n\n", dataFile);
end

% =====================================================================
function m = computeMetrics(out, centerline, parameters)
%COMPUTEMETRICS 指标口径。
%
% 时间基准说明：safety_diag / 视觉调试向量按模型基准步长(MJ_StepTime=0.002)
% 记录，而治理器本身按 Ts_Control=0.05 更新，即每个控制拍在日志里重复约 25
% 次。因此所有"比率"类指标都是**时间占比**而非"控制拍占比"——两者对定长仿真
% 等价，且时间占比正是我们想要的口径。

% ── 位姿与真值误差 ───────────────────────────────────────────────
% 各 ToWorkspace 块的采样率/长度不保证一致（历史上 MaxDataPoints 也不一致），
% 所以一律以里程计时间为基准重采样，不假设等长
pos  = squeeze(out.mujoco_odom_position.Data);
if size(pos, 1) ~= 3, pos = pos.'; end
tOdom = out.mujoco_odom_position.Time(:);

tQuat = out.mujoco_odom_quaternion.Time(:);
quat  = squeeze(out.mujoco_odom_quaternion.Data);
if size(quat, 1) ~= 4, quat = quat.'; end
quat = interp1(tQuat, quat.', tOdom, "previous", "extrap").';

% MuJoCo 在第一个真实位姿之前会吐一个全零初始化采样；四元数模长为 0，
% rl_true_track_errors 对它返回 [0;0]，但用来算行程会把出生点跳变算进去
valid = tOdom > 0 & vecnorm(quat, 2, 1).' > 0.5;
pos = pos(:, valid); quat = quat(:, valid); tOdom = tOdom(valid);

n = numel(tOdom);
trueLateral = zeros(n, 1);
trueHeading = zeros(n, 1);
for k = 1:n
    e = rl_true_track_errors([pos(:, k); quat(:, k)], centerline);
    trueLateral(k) = e(1);
    trueHeading(k) = e(2);
end

xy = pos(1:2, :).';
stepDistance = hypot(diff(xy(:, 1)), diff(xy(:, 2)));
cumulative = [0; cumsum(stepDistance)];
distanceToStart = hypot(xy(:, 1) - xy(1, 1), xy(:, 2) - xy(1, 2));
lapIndex = find(cumulative > 6 & distanceToStart < 0.25, 1);

% ── 视觉 ────────────────────────────────────────────────────────
% 调试向量宽度 = 7 标量 + 30 个 [x,y] 路径点槽位 = 67（不是 72——72 是视觉
% 函数 result 的总长，含前面 5 个标量输出，而这里记录的只是 y6 那段）
debugWidth = 67;
vis = squeeze(out.mujoco_local_path_debug.Data);
if size(vis, 1) ~= debugWidth, vis = vis.'; end
tVis = out.mujoco_local_path_debug.Time(:);
found = vis(1, :).' > 0;   % 槽位 1 = 有效窗点数，丢线时被置 0

% ── 安全层诊断 ──────────────────────────────────────────────────
% safety_diag 由模型根部的 Safety_Diag_Mux 拼成（2026-08-09 状态机/约束层
% 拆成两个子系统后的布局），14 路：
%   1-2   VOmegaSafe   (Safety_Filter 输出，最终下发)
%   3-4   VOmegaDes    (Line_Search 输出，安全滤波**之前**的期望值)
%   5-9   SearchDiag   [state, phi, memValid, memAge, memBearing]
%   10-14 FilterDiag   [h, omegaLo, omegaHi, cbfActive, infeasible]
% 干预量 = 1:2 与 3:4 之差。**不要**从轮速指令反算 omega——那已经过了
% SAC_PID_Sat 轮速饱和，会把安全层干预量和轮速饱和混为一谈。
D = squeeze(out.safety_diag.signals.values);
if size(D, 1) ~= 14, D = D.'; end
tDiag = out.safety_diag.time(:);

% Simulink 在治理器第一次执行之前会记录一个全零采样；它会把 h 拉到 0，污染
% MinBarrier。按"整条诊断向量全零"剔除，与里程计那边剔除全零初始化采样同理。
realSample = any(D ~= 0, 1).';
D = D(:, realSample);
tDiag = tDiag(realSample);

vApplied     = D(1, :).';
omegaApplied = D(2, :).';
omegaDes     = D(4, :).';
state        = D(5, :).';
phi          = D(6, :).';   %#ok<NASGU>  留给离线分析，指标里暂未使用
h            = D(10, :).';
cbfActive    = D(13, :).' > 0.5;
infeasible   = D(14, :).' > 0.5;

% 轮速只用来统计执行器饱和，不再用来反算 omega
wheel = squeeze(out.wheel_cmd_log.signals.values);
if size(wheel, 1) ~= 2, wheel = wheel.'; end
wheelOnDiag = interp1(out.wheel_cmd_log.time(:), wheel.', tDiag, "previous", "extrap").';

intervention = abs(omegaApplied - omegaDes);
intervened = cbfActive & intervention > 1e-6;

dt = median(diff(tDiag));
domega = [0; diff(omegaApplied)] / max(eps, dt);

m = struct();
% 存活/完成
m.CompletedTime      = tVis(end);
m.PathLength         = cumulative(end);
m.LapCompleted       = ~isempty(lapIndex);
m.LapTime            = NaN;
if ~isempty(lapIndex), m.LapTime = tOdom(lapIndex); end
% 真值循迹精度（相对赛道中心线，单位 m / rad）
m.TrueLateralMeanAbs = mean(abs(trueLateral));
m.TrueLateralRMS     = sqrt(mean(trueLateral.^2));
m.TrueLateralMax     = max(abs(trueLateral));
m.TrueHeadingRMS     = sqrt(mean(trueHeading.^2));
% 视觉
m.FoundRatio         = mean(found);
m.LongestLineLoss    = longestRun(~found, tVis);
m.LostLineEpisodes   = numel(runStarts(~found));
% 安全约束
m.ViolationRatio     = mean(h < 0);                 % |lateral_error| 越界的时间占比
m.MinBarrier         = min(h);
m.CBFActiveRatio     = mean(cbfActive);
m.CBFInterveneRatio  = mean(intervened);
m.CBFInterveneMean   = mean(intervention(intervened));
m.CBFInterveneMax    = max([0; intervention(intervened)]);
m.InfeasibleRatio    = mean(infeasible);
% 找线状态机（时间占比）
m.TimeTrack          = mean(state == 0);
m.TimeHold           = mean(state == 1);
m.TimeBrake          = mean(state == 2);
m.TimeScan           = mean(state == 3);
m.TimeGiveUp         = mean(state == 4);
m.Reacquisitions     = nnz(diff(state >= 2) == -1);  % 从 BRAKE/SCAN/GIVEUP 回到跟踪
% 控制平顺性
m.VMean              = mean(vApplied);
m.OmegaRMS           = sqrt(mean(omegaApplied.^2));
m.OmegaRateRMS       = sqrt(mean(domega.^2));
m.WheelSatRatio      = mean(any(abs(wheelOnDiag) >= parameters.MaxWheelSpeed - 1e-6, 1));
% 原始序列，供画图
m.trajectory         = xy;
m.time               = tOdom;
m.trueLateral        = trueLateral;
m.tDiag              = tDiag;
m.barrier            = h;
m.state              = state;
m.omegaApplied       = omegaApplied;
m.omegaDesired       = omegaDes;

if m.FoundRatio < 0.05
    warning("SafetyTrial:NoVision", ...
        ["found ratio is %.1f%% -- 视觉几乎全程没看到线。极可能是 MuJoCo 相机" ...
         "渲染失效（见本文件头的 LD_PRELOAD 说明），而不是控制器的问题。" ...
         "这份数据不能用来做效果对比。"], 100 * m.FoundRatio);
end
end

% =====================================================================
function printMetrics(label, m)
fprintf("--- %s ---\n", label);
fprintf("  存活时间        %7.2f s   路径长度 %6.2f m   完成整圈 %d\n", ...
    m.CompletedTime, m.PathLength, m.LapCompleted);
fprintf("  真值横向误差    平均 %.4f m  RMS %.4f m  最大 %.4f m\n", ...
    m.TrueLateralMeanAbs, m.TrueLateralRMS, m.TrueLateralMax);
fprintf("  视觉 found 率   %6.2f%%     最长丢线 %.2f s   丢线次数 %d\n", ...
    100 * m.FoundRatio, m.LongestLineLoss, m.LostLineEpisodes);
fprintf("  约束违反占比    %6.2f%%     最小屏障 h = %+.4f\n", ...
    100 * m.ViolationRatio, m.MinBarrier);
fprintf("  CBF 介入占比    %6.2f%%     平均介入 %.4f rad/s  最大 %.4f\n", ...
    100 * m.CBFInterveneRatio, m.CBFInterveneMean, m.CBFInterveneMax);
fprintf("  找线状态占比    TRACK %.1f%%  HOLD %.1f%%  BRAKE %.1f%%  SCAN %.1f%%  GIVEUP %.1f%%\n", ...
    100*m.TimeTrack, 100*m.TimeHold, 100*m.TimeBrake, 100*m.TimeScan, 100*m.TimeGiveUp);
fprintf("  重新捕获次数    %d\n", m.Reacquisitions);
fprintf("  omega RMS       %6.3f rad/s   角加速度 RMS %6.3f rad/s^2   轮饱和 %.1f%%\n", ...
    m.OmegaRMS, m.OmegaRateRMS, 100 * m.WheelSatRatio);
end

% =====================================================================
function d = longestRun(mask, t)
starts = runStarts(mask);
d = 0;
for k = 1:numel(starts)
    s = starts(k);
    e = s;
    while e < numel(mask) && mask(e + 1), e = e + 1; end
    d = max(d, t(e) - t(s));
end
end

function s = runStarts(mask)
mask = mask(:);
s = find(mask & [true; ~mask(1:end-1)]);
end
