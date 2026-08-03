function resetFcn = rl_domain_randomization(mdl, dr)
%RL_DOMAIN_RANDOMIZATION Build a per-episode domain-randomization ResetFcn.
%
%   为残差 RL 训练构造一个「域随机化」重置函数，SAC 与 PPO 训练共用
%   （域随机化与算法无关，只作用于模型工作区变量）。每个训练回合开始时，
%   rlSimulinkEnv 会调用返回的 resetFcn，对模型工作区变量施加随机扰动，
%   让学到的残差策略对起始位姿、机器人动力学、底层速度/PID、感知参数的
%   变化更鲁棒（sim-to-real / 泛化）。
%
%   Returns a function handle  resetFcn(in) -> in  that perturbs model
%   workspace variables on a Simulink.SimulationInput before each episode.
%   Nothing in the .slx is modified permanently — each episode gets its own
%   randomized copy via setVariable.
%
%   用法 / Usage
%   ------------
%     dr = rl_dr_defaults();          % 见下方本文件内的默认配置
%     env.ResetFcn = rl_domain_randomization(mdl, dr);
%
%   dr 配置结构 / Config struct
%   --------------------------
%     dr.Enable            (logical)  总开关，false 时返回恒等 reset（不随机化）
%     dr.Pose.Enable       + .LateralRange  起始横向偏移 ±range (同 MJ_TargetLateralPosition 单位)
%     dr.Dynamics.Enable   + .WheelRadiusPct .WheelSepPct .MaxWheelPct  相对基准的 ±百分比
%     dr.Control.Enable    + .BaseSpeedMin .BaseSpeedMax（绝对范围）.KpPct .KiPct .KdPct（相对基准 ±百分比）
%     dr.Perception.Enable + .ErrorScalePct .MinBrightnessPct .ROIStartPct .ROIHeightPct
%
%   百分比 p 表示每回合从 [base*(1-p), base*(1+p)] 均匀采样。
%   Percent p means: sample uniformly from base*(1 +/- p) each episode.
%
%   See also TRAIN_SAC_RESIDUAL_LIVE, TRAIN_PPO_RESIDUAL_LIVE, RL_IO_SPECS.

arguments
    mdl (1,1) string
    dr  (1,1) struct
end

% 总开关关闭 → 恒等重置（等价于原来注释掉的占位符）。
if ~isfield(dr, "Enable") || ~dr.Enable
    resetFcn = @(in) in;
    return
end

% 读取各基准值一次（重置函数闭包捕获，避免每回合重复查询模型工作区）。
ws   = get_param(mdl, "ModelWorkspace");
base = struct();

% ── 地图随机化：预解析每个地图 key 对应的场景 XML 绝对路径（闭包捕获）──
% 只在构造 ResetFcn 时解析一次；每回合从这些路径里随机挑一个，通过
% setBlockParameter 覆写 MuJoCo Plant(:16) 的 xmlFile。
base.PlantBlock = "";
base.MapScenes  = string.empty;
base.GenTrack   = struct("Enable", false);   % filled below if requested
% MuJoCo Plant block (SID :16) — needed by both Map and GenTrack scene switching
if grp(dr, "Map") || grp(dr, "GenTrack")
    try
        base.PlantBlock = string(Simulink.ID.getFullName(mdl + ":16"));
    catch
        base.PlantBlock = "";
    end
end
% Fixed-pool map randomization: pre-resolve each key's scene XML once.
% (RL_TrackCenterline 不在这里设：模型 InitFcn 每次仿真开始时按 Plant 实际
% xmlFile 刷新中心线——块参数在 InitFcn 之前应用，已实验验证——因此换图后
% 真值奖励/出界终止自动跟随，单一权威、无需在 DR 里同步。)
if grp(dr, "Map")
    maps = getf(dr.Map, "Maps", string.empty);
    if ~isempty(maps)
        modelDir = string(fileparts(which(char(mdl))));
        scenes = string.empty;
        for m = string(maps(:)')
            try
                sf = resolve_turtlebot3_mujoco_scene(mdl, modelDir, m);
                scenes(end+1) = string(sf); %#ok<AGROW>
            catch ME
                warning("RLDR:BadMap", ...
                    "Skipping map '%s' (unresolvable): %s", m, ME.message);
            end
        end
        base.MapScenes = scenes;
    end
end
% Per-episode freshly-generated track config (takes priority over Map).
if grp(dr, "GenTrack")
    base.GenTrack = struct( ...
        "Enable",     true, ...
        "Generator",  getf(dr.GenTrack, "Generator", "random"), ...
        "Shape",      getf(dr.GenTrack, "Shape", "random"), ...
        "Difficulty", getf(dr.GenTrack, "Difficulty", 0.4), ...
        "Name",       getf(dr.GenTrack, "Name", "rl_gen_episode"));
end
base.Model         = string(mdl);
base.LateralTarget = getVar(ws, "MJ_TargetLateralPosition", 0);
base.WheelRadius   = getVar(ws, "TB3_WheelRadius",         0.033);
base.WheelSep      = getVar(ws, "TB3_WheelSeparation",     0.160);
base.MaxWheel      = getVar(ws, "MaxWheelSpeed",           25);
base.BaseSpeed     = getVar(ws, "BaseLinearSpeed",         20);
base.Kp            = getVar(ws, "Kp_Line",                 5);
base.Ki            = getVar(ws, "Ki_Line",                 0);
base.Kd            = getVar(ws, "Kd_Line",                 0.35);
base.ErrorScale    = getVar(ws, "OriginBot_ErrorScale",    500);
base.MinBright     = getVar(ws, "OriginBot_MinBrightness", 70);
base.ROIStart      = getVar(ws, "OriginBot_ROIStartFraction", 0.72);
base.ROIHeight     = getVar(ws, "OriginBot_ROIHeight",    45);

resetFcn = @(in) localReset(in, dr, base);
end


% ═══════════════════════════════════════════════════════════════════════════
%  Per-episode reset: perturb variables on the SimulationInput
% ═══════════════════════════════════════════════════════════════════════════
function in = localReset(in, dr, base)

% ── 赛道 / 场景选择 ──
% 优先级：GenTrack（每回合现场新生成一张）> Map（固定池随机选）。两者都通过
% setBlockParameter 覆写 MuJoCo Plant 的 xmlFile，只作用于当前回合。
sceneSet = false;
if base.GenTrack.Enable && strlength(base.PlantBlock) > 0
    try
        generator = base.GenTrack.Generator;
        if generator == "random"
            pool = ["simple", "complex", "wandering"];
            generator = pool(randi(numel(pool)));
        end
        switch generator
            case "simple"
                gi = gen_simple_track_scene( ...
                    "Name",       base.GenTrack.Name, ...
                    "Shape",      base.GenTrack.Shape, ...
                    "Difficulty", base.GenTrack.Difficulty, ...
                    "Overwrite",  true);
            case "complex"
                gi = gen_complex_track_scene( ...
                    "Name",       base.GenTrack.Name, ...
                    "Difficulty", base.GenTrack.Difficulty, ...
                    "Overwrite",  true);
            case "wandering"
                gi = gen_wandering_track_scene( ...
                    "Name",       base.GenTrack.Name, ...
                    "Difficulty", base.GenTrack.Difficulty, ...
                    "Overwrite",  true);
            otherwise
                error("RLDR:BadGenerator", ...
                    "Unknown GenTrack.Generator '%s' (expected simple|complex|wandering|random).", ...
                    generator);
        end
        in = setBlockParameter(in, char(base.PlantBlock), ...
            "xmlFile", char(gi.SceneFile));
        sceneSet = true;
    catch ME
        warning("RLDR:GenTrackFailed", ...
            "Episode track generation failed, keeping current scene: %s", ...
            ME.message);
    end
end
if ~sceneSet && grp(dr, "Map") && ~isempty(base.MapScenes) && ...
        strlength(base.PlantBlock) > 0
    pick = base.MapScenes(randi(numel(base.MapScenes)));
    in = setBlockParameter(in, char(base.PlantBlock), "xmlFile", char(pick));
end

% ⚠ 以下全部变量都定义在「模型工作区」。setVariable 不带 Workspace 参数时
% 写的是全局作用域，模型工作区同名变量优先级更高 → 扰动被静默忽略（已用
% 最小模型实验证实）。必须显式 'Workspace', base.Model 才真正生效。
% ── 起始位姿 / Start pose ──
if grp(dr, "Pose")
    r  = getf(dr.Pose, "LateralRange", 0.05);
    in = setMwsVar(in, base, "MJ_TargetLateralPosition", ...
        base.LateralTarget + uni(-r, r));
end

% ── 机器人动力学 / Robot dynamics ──
if grp(dr, "Dynamics")
    in = setMwsVar(in, base, "TB3_WheelRadius", ...
        pct(base.WheelRadius, getf(dr.Dynamics, "WheelRadiusPct", 0.05)));
    in = setMwsVar(in, base, "TB3_WheelSeparation", ...
        pct(base.WheelSep,    getf(dr.Dynamics, "WheelSepPct",    0.05)));
    in = setMwsVar(in, base, "MaxWheelSpeed", ...
        pct(base.MaxWheel,    getf(dr.Dynamics, "MaxWheelPct",    0.10)));
end

% ── 基础速度 / PID ──
if grp(dr, "Control")
    % 基础线速度用绝对范围（Burger 真实线速度量程），不是相对基准的百分比抖动。
    in = setMwsVar(in, base, "BaseLinearSpeed", ...
        uni(getf(dr.Control, "BaseSpeedMin", 0), ...
            getf(dr.Control, "BaseSpeedMax", base.BaseSpeed)));
    in = setMwsVar(in, base, "Kp_Line", ...
        pct(base.Kp,        getf(dr.Control, "KpPct",        0.15)));
    % Ki 基准常为 0，按百分比扰动仍是 0；用绝对小抖动更有意义。
    in = setMwsVar(in, base, "Ki_Line", ...
        max(0, base.Ki + uni(0, getf(dr.Control, "KiAbs", 0))));
    in = setMwsVar(in, base, "Kd_Line", ...
        pct(base.Kd,        getf(dr.Control, "KdPct",        0.15)));
end

% ── 感知 / 视觉 / Perception ──
if grp(dr, "Perception")
    in = setMwsVar(in, base, "OriginBot_ErrorScale", ...
        pct(base.ErrorScale, getf(dr.Perception, "ErrorScalePct", 0.10)));
    in = setMwsVar(in, base, "OriginBot_MinBrightness", ...
        pct(base.MinBright,  getf(dr.Perception, "MinBrightnessPct", 0.15)));
    in = setMwsVar(in, base, "OriginBot_ROIStartFraction", ...
        clamp01(pct(base.ROIStart, getf(dr.Perception, "ROIStartPct", 0.05))));
    in = setMwsVar(in, base, "OriginBot_ROIHeight", ...
        round(pct(base.ROIHeight, getf(dr.Perception, "ROIHeightPct", 0.10))));
end
end


% ─── small helpers ─────────────────────────────────────────────────────────
function tf = grp(dr, name)
% Is a sub-group present AND enabled?
tf = isfield(dr, name) && isfield(dr.(name), "Enable") && dr.(name).Enable;
end

function v = getf(s, name, dflt)
% Field with default.
if isfield(s, name), v = s.(name); else, v = dflt; end
end

function in = setMwsVar(in, base, name, value)
% Workspace-scoped setVariable：目标是模型工作区变量，必须显式指定作用域。
in = setVariable(in, name, value, "Workspace", base.Model);
end

function v = getVar(ws, name, dflt)
% Read a model-workspace variable with a fallback default.
if ws.hasVariable(name), v = ws.getVariable(name); else, v = dflt; end
end

function y = uni(lo, hi)
% Uniform sample in [lo, hi].
y = lo + (hi - lo) * rand();
end

function y = pct(baseVal, p)
% Sample uniformly from baseVal*(1 +/- p).
y = baseVal * (1 + uni(-p, p));
end

function y = clamp01(x)
y = min(1, max(0, x));
end
