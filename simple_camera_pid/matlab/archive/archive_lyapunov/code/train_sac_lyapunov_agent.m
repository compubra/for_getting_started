function [agentOut, trainInfo] = train_sac_lyapunov_agent(options)
%TRAIN_SAC_LYAPUNOV_AGENT Train (or just load) the SAC residual controller that
%uses the Lyapunov-function reward.
%
% This is the Lyapunov twin of TRAIN_SAC_RESIDUAL_AGENT. It targets the model
% visual_line_follower_sac_lyapunov, whose Reward_Calculation subsystem computes
%
%   V      = Q_e*e_steer^2 + Q_l*e_lat^2 + Q_h*e_head^2 + Q_v*v_lat^2  (Lyapunov)
%     where v_lat = ||v_odom|| * sin(e_head) is the lateral approach speed:
%     the robot's TRUE ground speed (odom) times the sine of its VISION heading
%     error. This adds real car-state (velocity) to V while staying track-
%     agnostic (heading comes from the camera, not a hardcoded centreline), so
%     it generalizes across randomized maps. v_lat=0 when aligned or line-lost.
%   Vdot   = (V[k]-V[k-1]) / Ts_Control                      (discrete rate)
%   reward = -tanh(V + Lambda*Vdot) - R*u^2 - P_Lost*(1-found)
%            - found * K_inv*(1 - exp(-Beta*(|e_steer|+|e_lat|+|e_head|)))
%
%   All four terms are <= 0 (bounded penalties), so the per-step reward sits
%   in roughly [-1.4, 0]: it approaches 0 as the tracking errors shrink and
%   grows more negative as they grow. The Lyapunov energy term is squashed by
%   tanh into (-1, 0], and the line term is the bounded penalty
%   -K_inv*(1 - exp(-Beta*||e||)) in [-K_inv, 0] (found-gated so a lost line
%   is only ever penalised by the P_Lost term, never rewarded).
%
%   NOTE: the inverse-error bonus is gated by `found`. Without the gate the
%   line-lost branch of the vision generator zeros all three errors, which
%   collapses the bonus denominator to Eps and pays a huge +K_inv/Eps reward
%   for LOSING the line — the agent then learns to drive off the track. The
%   Inv_Bonus subsystem multiplies the bonus by found so a lost line scores
%   -P_Lost (correct) instead of +90/step.
%
% Two things set it apart from the residual trainer:
%   1. It writes the Lyapunov reward weights (SAC_Q_e/Q_l/Q_h/Lambda/R/P_Lost/
%      K_inv/Eps) into the model workspace, not the scalar SAC_Q of the
%      residual reward.
%   2. There is NO reward-based early stopping. Training always runs the full
%      MaxEpisodes so you get a complete learning curve; SaveReward is only a
%      checkpoint-save threshold.
%
% DoTrain option
% --------------
%   options.DoTrain = true  (default) : build a fresh agent and train it.
%   options.DoTrain = false           : skip training, load the newest saved
%                                       agent from SaveDir (or LoadAgentFile if
%                                       given), push it into the SAC_Agent block
%                                       and return it ready for sim()/evaluation.
%
% Observation (5-D): [steering_error; lateral_error; heading_error; found; prev_rl_action]
% Action      (1-D):  delta_omega in [-MaxDeltaOmega, MaxDeltaOmega] rad/s
%
% Usage
% -----
%   % Train 800 episodes with random maps (domain-randomization default):
%   [agent, info] = train_sac_lyapunov_agent("MaxEpisodes", 800);
%
%   % Don't train — just load the latest agent and evaluate it:
%   agent = train_sac_lyapunov_agent("DoTrain", false);
%   sim("visual_line_follower_sac_lyapunov");
%
% See also TRAIN_SAC_RESIDUAL_AGENT, RUN_SAC_TRAINING, RL_DR_DEFAULTS.

arguments
    % ── Train-or-load switch ──
    options.DoTrain              (1,1) logical                  = true
    options.LoadAgentFile        (1,1) string                   = ""
    % ── Model ──
    options.ModelName            (1,1) string = "visual_line_follower_sac_lyapunov"
    % Clear a stale/crashed MuJoCo MEX host and reload the model before training
    % (recovers from a prior interrupted/errored run). Set false to skip.
    options.RecoverMujoco        (1,1) logical                  = true
    % ── Training loop (NO early stop: always runs MaxEpisodes) ──
    options.MaxEpisodes          (1,1) double  {mustBePositive} = 800
    options.MaxStepsPerEpisode   (1,1) double  {mustBePositive} = 2000
    options.ScoreWindow          (1,1) double  {mustBePositive} = 20
    options.MapKey               (1,1) string                   = "training"
    options.UseFast              (1,1) logical                  = false
    options.UseParallel          (1,1) logical                  = false
    options.NumWorkers           (1,1) double  {mustBePositive} = 2
    options.ShowPlot             (1,1) logical                  = false
    % SaveReward is ONLY a checkpoint threshold — episodes whose reward clears
    % it get auto-saved. It is not an early-stop criterion.
    options.SaveReward           (1,1) double                   = -0.4
    % ── Lyapunov reward weights (written into the model workspace) ──
    options.RewardQe             (1,1) double  {mustBeNonnegative} = 1.0
    options.RewardQl             (1,1) double  {mustBeNonnegative} = 1.0
    options.RewardQh             (1,1) double  {mustBeNonnegative} = 1.0
    options.RewardQv             (1,1) double  {mustBeNonnegative} = 0.3
    % ── True-state V weights (only the _truestate model defines these; on other
    %    models the assignIfPresent below is a harmless no-op). See RewardQv note.
    options.RewardQlat           (1,1) double  {mustBeNonnegative} = 1.0
    options.RewardQhead          (1,1) double  {mustBeNonnegative} = 1.0
    options.RewardQvlat          (1,1) double  {mustBeNonnegative} = 0.3
    options.RewardLambda         (1,1) double  {mustBeNonnegative} = 0.5
    options.RewardR              (1,1) double  {mustBeNonnegative} = 0.1
    options.RewardPLost          (1,1) double  {mustBeNonnegative} = 1.0
    options.RewardKinv           (1,1) double  {mustBeNonnegative} = 0.4
    options.RewardBeta           (1,1) double  {mustBePositive}    = 3.0
    options.RewardEps            (1,1) double  {mustBePositive}    = 0.01
    options.DoneSteps            (1,1) double  {mustBePositive}    = 100
    % ── SAC agent hyper-parameters ──
    options.MiniBatchSize        (1,1) double  {mustBePositive} = 256
    options.BufferLength         (1,1) double  {mustBePositive} = 2e5
    options.ActorLearnRate       (1,1) double  {mustBePositive} = 3e-4
    options.CriticLearnRate      (1,1) double  {mustBePositive} = 3e-4
    options.DiscountFactor       (1,1) double  {mustBePositive} = 0.99
    options.EntropyWeight        (1,1) double  {mustBePositive} = 0.2
    options.TargetSmooth         (1,1) double  {mustBePositive} = 5e-3
    options.MaxDeltaOmega        (1,1) double  {mustBePositive} = 1.0
    % ── Domain randomization (per-episode ResetFcn) ──
    %    [] (default) = rl_dr_defaults (all groups on, incl. random map).
    options.DomainRand                                          = []
    % ── Output ──
    options.SaveDir              (1,1) string                   = ""
end

experimentDir = string(fileparts(mfilename("fullpath")));
modelDir      = fullfile(experimentDir, "..");
runtimeDir    = fullfile(modelDir, "runtime");
addpath(runtimeDir);
addpath(fullfile(modelDir, "train", "shared"));   % rl_io_specs / rl_dr_defaults / rl_domain_randomization

mdl      = options.ModelName;
mdlFile  = fullfile(modelDir, mdl + ".slx");
agentBlk = mdl + "/SAC_Residual_Controller/SAC_Agent";
outDir   = saveDir(options.SaveDir, modelDir);

assert(isfile(mdlFile), "TrainLyap:MissingModel", ...
    "Model not found: %s", mdlFile);
assert(~isempty(ver("rl")), "TrainLyap:NoRL", ...
    "Reinforcement Learning Toolbox license required.");
assert(~isempty(which("mj_initbus")), "TrainLyap:NoMuJoCo", ...
    "Simulink Blockset for MuJoCo Simulator is not installed.");

% Recover from a stale/crashed MuJoCo MEX host before touching the model. A
% previous run that was Ctrl+C'd or errored can leave mj_initbus_mex in a bad
% state; the next set_turtlebot3_mujoco_scene then crashes with
% "'MATLABMexHost' ... crashed or was destroyed outside MATLAB". Closing the
% model, clearing the MEX host, then reopening fresh clears that state. This is
% idempotent — on a healthy session it just reloads the model.
if options.RecoverMujoco
    recover_mujoco_host(mdl);
end
if ~bdIsLoaded(mdl), open_system(mdlFile); end

% ══════════════════════════════════════════════════════════════════════════
%  BRANCH: load-only (DoTrain=false) — no training, just evaluate a saved agent
% ══════════════════════════════════════════════════════════════════════════
if ~options.DoTrain
    agentFile = resolveLoadFile(options.LoadAgentFile, outDir);
    fprintf("DoTrain=false → loading saved agent:\n  %s\n", agentFile);
    S = load(agentFile);
    assert(isfield(S, "trainedAgent"), "TrainLyap:NoAgentVar", ...
        "File has no 'trainedAgent' variable: %s", agentFile);
    agentOut = S.trainedAgent;
    if isfield(S, "trainInfo"), trainInfo = S.trainInfo; else, trainInfo = []; end
    assignin("base", "trainedAgent", agentOut);
    set_param(agentBlk, "Agent", "trainedAgent");
    set_turtlebot3_mujoco_scene(mdl, options.MapKey);
    fprintf("Loaded agent into %s. Ready to evaluate:\n", agentBlk);
    fprintf("  sim(""%s"")\n", mdl);
    return
end

% ══════════════════════════════════════════════════════════════════════════
%  TRAIN branch
% ══════════════════════════════════════════════════════════════════════════

% ── 1. Base map (per-episode randomization can override it) ────────────────
set_turtlebot3_mujoco_scene(mdl, options.MapKey);
fprintf("Base training map: %s\n", options.MapKey);

% ── 1b. Push Lyapunov reward weights into the model workspace ──────────────
% Only assign vars the model actually defines (assignIfPresent) so we never
% pollute the workspace if a weight is renamed/removed in the model.
ws = get_param(mdl, "ModelWorkspace");
assignIfPresent(ws, "SAC_Q_e",         options.RewardQe);
assignIfPresent(ws, "SAC_Q_l",         options.RewardQl);
assignIfPresent(ws, "SAC_Q_h",         options.RewardQh);
assignIfPresent(ws, "SAC_Q_v",         options.RewardQv);
% True-state V weights — only the _truestate model defines these vars, so on
% the vision-error models these three writes are silently skipped.
assignIfPresent(ws, "SAC_Q_lat",       options.RewardQlat);
assignIfPresent(ws, "SAC_Q_head",      options.RewardQhead);
assignIfPresent(ws, "SAC_Q_vlat",      options.RewardQvlat);
assignIfPresent(ws, "SAC_Lambda",      options.RewardLambda);
assignIfPresent(ws, "SAC_R",           options.RewardR);
assignIfPresent(ws, "SAC_P_Lost",      options.RewardPLost);
assignIfPresent(ws, "SAC_K_inv",       options.RewardKinv);
assignIfPresent(ws, "SAC_Beta",        options.RewardBeta);
assignIfPresent(ws, "SAC_Eps",         options.RewardEps);
assignIfPresent(ws, "SAC_Done_Steps",  options.DoneSteps);
assignIfPresent(ws, "SAC_MaxDeltaOmega", options.MaxDeltaOmega);
fprintf(['Lyapunov reward weights: Q_e=%.3g Q_l=%.3g Q_h=%.3g Q_v=%.3g | ' ...
    'Lambda=%.3g R=%.3g P_lost=%.3g | K_inv=%.3g Beta=%.3g Eps=%.1g | done@%d\n'], ...
    options.RewardQe, options.RewardQl, options.RewardQh, options.RewardQv, ...
    options.RewardLambda, options.RewardR, options.RewardPLost, ...
    options.RewardKinv, options.RewardBeta, options.RewardEps, options.DoneSteps);
if ws.hasVariable("SAC_Q_lat")   % true-state V model (_truestate)
    fprintf(['True-state V weights: Q_lat=%.3g Q_head=%.3g Q_vlat=%.3g ' ...
        '(V uses odom errors; centreline hardcoded to track_hard)\n'], ...
        options.RewardQlat, options.RewardQhead, options.RewardQvlat);
end

% ── 2. Observation/action specs (shared source of truth) ───────────────────
[obsInfo, actInfo] = rl_io_specs(options.MaxDeltaOmega);

% ── 3. Build SAC agent (auto-generated Gaussian actor + twin critics) ──────
agentOpts = rlSACAgentOptions( ...
    "SampleTime",             -1, ...
    "DiscountFactor",          options.DiscountFactor, ...
    "MiniBatchSize",           options.MiniBatchSize, ...
    "ExperienceBufferLength",  options.BufferLength, ...
    "TargetSmoothFactor",      options.TargetSmooth);
agentOpts.EntropyWeightOptions.EntropyWeight  = options.EntropyWeight;
agentOpts.ActorOptimizerOptions.LearnRate     = options.ActorLearnRate;
agentOpts.CriticOptimizerOptions(1).LearnRate = options.CriticLearnRate;
agentOpts.CriticOptimizerOptions(2).LearnRate = options.CriticLearnRate;
agent = rlSACAgent(obsInfo, actInfo, agentOpts);

set_param(agentBlk, "Agent", "agent");
assignin("base", "agent", agent);

% ── 4. Simulink environment + domain randomization ─────────────────────────
env = rlSimulinkEnv(mdl, agentBlk, obsInfo, actInfo);
if isempty(options.DomainRand)
    drCfg = rl_dr_defaults();
else
    drCfg = options.DomainRand;
end
env.ResetFcn = rl_domain_randomization(mdl, drCfg);

% Per-episode scene switching overwrites the MuJoCo Plant's nontunable xmlFile,
% which Fast Restart forbids — so turn Fast Restart off whenever the map is
% randomized (fixed pool) OR a fresh track is generated each episode.
mapRand = isfield(drCfg, "Enable") && drCfg.Enable && ...
    isfield(drCfg, "Map") && isfield(drCfg.Map, "Enable") && drCfg.Map.Enable && ...
    isfield(drCfg.Map, "Maps") && ~isempty(drCfg.Map.Maps);
genTrack = isfield(drCfg, "Enable") && drCfg.Enable && ...
    isfield(drCfg, "GenTrack") && isfield(drCfg.GenTrack, "Enable") && ...
    drCfg.GenTrack.Enable;
if mapRand || genTrack
    env.UseFastRestart = "off";
end
if isfield(drCfg, "Enable") && drCfg.Enable
    on = string.empty;
    for g = ["GenTrack","Map","Pose","Dynamics","Control","Perception"]
        if isfield(drCfg, g) && isfield(drCfg.(g), "Enable") && drCfg.(g).Enable
            on(end+1) = g; %#ok<AGROW>
        end
    end
    fprintf("Domain randomization ON: %s\n", strjoin(on, ", "));
    if genTrack
        fprintf("Per-episode freshly-generated simple track (Fast Restart OFF).\n");
    elseif mapRand
        fprintf("Per-episode random map from %d maps (Fast Restart OFF).\n", ...
            numel(drCfg.Map.Maps));
    end
else
    fprintf("Domain randomization OFF.\n");
end

% ── 5. Training options — NO reward-based early stop ───────────────────────
if options.ShowPlot, plotFlag = "training-progress"; else, plotFlag = "none"; end
trainOpts = rlTrainingOptions( ...
    "MaxEpisodes",               options.MaxEpisodes, ...
    "MaxStepsPerEpisode",        options.MaxStepsPerEpisode, ...
    "ScoreAveragingWindowLength",options.ScoreWindow, ...
    "StopTrainingCriteria",      "none", ...
    "SaveAgentCriteria",         "EpisodeReward", ...
    "SaveAgentValue",            options.SaveReward, ...
    "SaveAgentDirectory",        outDir, ...
    "Verbose",                   true, ...
    "Plots",                     plotFlag, ...
    "UseParallel",               options.UseParallel);

% Number of parallel workers (only meaningful when UseParallel=true).
% RL Toolbox uses however many workers the active parallel pool has — there is
% no NumWorkers on ParallelizationOptions — so we size the pool here.
if options.UseParallel
    ensurePool(options.NumWorkers);
end

if options.UseFast
    set_param(mdl + ":16", "renderingType", "None");
end
if options.UseParallel
    save_system(mdl);
    fprintf("UseParallel=true: saved model for parallel workers.\n");
end

% ── 6. Train (runs full MaxEpisodes) ───────────────────────────────────────
fprintf("Starting Lyapunov SAC training: %d episodes (no early stop).\n", ...
    options.MaxEpisodes);
fprintf("  Obs dim: %s | Act dim: %s | Batch: %d | Buffer: %g\n", ...
    mat2str(obsInfo.Dimension), mat2str(actInfo.Dimension), ...
    options.MiniBatchSize, options.BufferLength);

trainInfo = train(agent, env, trainOpts);
agentOut  = agent;

% ── 7. Save + load into the model block ────────────────────────────────────
ts        = string(datetime("now","Format","yyyyMMdd_HHmmss"));
agentFile = fullfile(outDir, "sac_lyapunov_agent_" + ts + ".mat");
trainedAgent = agentOut;   % saved under this name so the load-branch finds it
save(agentFile, "trainedAgent", "trainInfo", "obsInfo", "actInfo", "-v7.3");
fprintf("Saved trained agent: %s\n", agentFile);

assignin("base", "trainedAgent", agentOut);
set_param(agentBlk, "Agent", "trainedAgent");
save_system(mdl);
fprintf("Loaded trained agent into %s.\n", agentBlk);
fprintf("Run  sim(""%s"")  to evaluate.\n", mdl);
end


% ─── Helpers ───────────────────────────────────────────────────────────────
function d = saveDir(userDir, modelDir)
if strlength(userDir) > 0
    d = char(userDir);
else
    d = fullfile(char(modelDir), "simulation_data", "sac_training");
end
if ~isfolder(d), mkdir(d); end
end

function assignIfPresent(ws, name, value)
%ASSIGNIFPRESENT Assign to a model-workspace var only if it already exists.
if ws.hasVariable(name)
    assignin(ws, name, value);
end
end

function f = resolveLoadFile(userFile, outDir)
%RESOLVELOADFILE Pick the agent .mat to load: explicit LoadAgentFile if given,
% otherwise the newest sac_lyapunov_agent_*.mat (falling back to any
% sac_agent_*.mat) in outDir.
if strlength(userFile) > 0
    assert(isfile(userFile), "TrainLyap:NoLoadFile", ...
        "LoadAgentFile not found: %s", userFile);
    f = char(userFile);
    return
end
patterns = ["sac_lyapunov_agent_*.mat", "sac_agent_*.mat"];
best = ""; bestTime = -inf;
for p = patterns
    d = dir(fullfile(outDir, p));
    for k = 1:numel(d)
        if d(k).datenum > bestTime
            bestTime = d(k).datenum;
            best = fullfile(d(k).folder, d(k).name);
        end
    end
end
assert(strlength(best) > 0, "TrainLyap:NoSavedAgent", ...
    ["No saved agent found in %s.\n" ...
     "Train first (DoTrain=true) or pass LoadAgentFile."], outDir);
f = char(best);
end

function ensurePool(n)
%ENSUREPOOL Make the active parallel pool have exactly n process workers.
% RL parallel training uses whatever pool is open, so this is how the caller's
% NumWorkers takes effect. Requires Parallel Computing Toolbox.
if isempty(ver("parallel"))
    warning("TrainLyap:NoPCT", ...
        "UseParallel=true but Parallel Computing Toolbox is not installed; " + ...
        "training will fall back to serial.");
    return;
end
c = parcluster("Processes");
n = min(n, c.NumWorkers);             % never ask for more than the host allows
p = gcp("nocreate");
if isempty(p)
    parpool("Processes", n);
elseif p.NumWorkers ~= n
    delete(p);
    parpool("Processes", n);
end
fprintf("Parallel workers: %d\n", n);
end
