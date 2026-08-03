%% TRAIN_SAC_RESIDUAL_SCRIPT  去函数化的 SAC 残差强化学习训练/评估脚本
%
%   ⚠️ 备用入口：正式的 SAC 训练入口是 train_sac_residual_live.m。两者现在
%   读同一份参数中心 sac_training_config.m（本脚本只是不依赖 Live Editor 的
%   纯脚本外壳，适合无 Live Editor 环境，如 HPC 批处理），不会再出现两边
%   参数不同步的问题——调参只改 sac_training_config.m 一处。
%
%   Fallback plain-script entry for environments without the Live Editor
%   (e.g. HPC batch). Reads the same sac_training_config.m as the canonical
%   train_sac_residual_live.m entry, so the two never drift out of sync.
%
%   训练 / 仅评估开关 / Train-vs-eval-only switch
%   -------------------------------------------
%     cfg.TrainAgent (sac_training_config.m) 控制本脚本的行为：
%       true  = 训练模式：构建新 agent，调用 train()，训练完存档并载入模型块
%               （原有行为）。
%       false = 仅评估模式：跳过 agent/环境/训练选项的构建，直接把
%               cfg.EvalAgentFile（或 SaveDir 下最新的 sac_agent_*.mat）
%               载入 SAC_Agent 块，然后 sim(model) 跑一次仿真评估——不训练。
%
%   运行前提 / Prerequisites
%   ------------------------
%     - 已安装 Reinforcement Learning Toolbox 和 MuJoCo Blockset
%     - 模型 visual_line_follower_sac_residual.slx 已存在
%     - 仅评估模式还需要一个已训练好的 sac_agent_*.mat（先跑一次训练模式，
%       或用 cfg.EvalAgentFile 指定一个已有存档）
%
%   与函数版的区别 / Differences from the function version
%   -----------------------------------------------------
%     - 训练模式下，训练好的 agent、obsInfo、actInfo、trainInfo 会留在
%       「基础工作区」，方便训练后立刻 sim() 评估，不需要再从 .mat 载入。
%
%   See also TRAIN_SAC_RESIDUAL_LIVE, SAC_TRAINING_CONFIG.

%% ───────────────────────────────────────────────────────────────────────
%  0) 路径准备 / Path setup
%  ───────────────────────────────────────────────────────────────────────
%  本脚本在 .../matlab/train/sac/ 下，模型与 runtime/experiments 在上两层，
%  共享的 rl_* 训练函数在 train/shared/。
scriptDir = fileparts(mfilename("fullpath"));
matlabDir = fullfile(scriptDir, "..", "..");    % .../simple_camera_pid/matlab
addpath(genpath(fullfile(matlabDir, "runtime"))); % runtime/ 分 init/vision/scene/ops 子目录，递归加入
addpath(scriptDir);                             % 参数中心 sac_training_config 所在目录（本目录）
addpath(fullfile(matlabDir, "train", "shared")); % rl_io_specs / rl_dr_defaults 等共享训练函数
cd(matlabDir);

%% ───────────────────────────────────────────────────────────────────────
%  1) 参数区 / PARAMETERS — 与 train_sac_residual_live 共读 sac_training_config.m
%  ───────────────────────────────────────────────────────────────────────
%  调参改 sac_training_config.m，不要在本脚本里改数字——本脚本只负责读取并
%  展开成局部变量，逻辑与 train_sac_residual_live.m 保持一致。
cfg = sac_training_config();
TrainAgent         = cfg.TrainAgent;
EvalAgentFile      = cfg.EvalAgentFile;
MaxEpisodes        = cfg.MaxEpisodes;
MaxStepsPerEpisode = cfg.MaxStepsPerEpisode;
StopReward         = cfg.StopReward;
SaveReward         = cfg.SaveReward;
SaveEveryNEpisodes = cfg.SaveEveryNEpisodes;
ScoreWindow        = cfg.ScoreWindow;
UseFast            = cfg.UseFast;
UseParallel        = cfg.UseParallel;
NumWorkers         = cfg.NumWorkers;
ShowPlot           = cfg.ShowPlot;
MapKey             = cfg.MapKey;
RewardQ            = cfg.Reward.Q;
RewardQLateral     = cfg.Reward.Q_Lateral;
RewardQHeading     = cfg.Reward.Q_Heading;
RewardR            = cfg.Reward.R;
RewardPLost        = cfg.Reward.P_Lost;
DoneSteps          = cfg.Reward.DoneSteps;
ActorLearnRate     = cfg.Agent.ActorLearnRate;
CriticLearnRate    = cfg.Agent.CriticLearnRate;
DiscountFactor     = cfg.Agent.DiscountFactor;
MiniBatchSize      = cfg.Agent.MiniBatchSize;
BufferLength       = cfg.Agent.BufferLength;
EntropyWeight      = cfg.Agent.EntropyWeight;
TargetSmooth       = cfg.Agent.TargetSmooth;
MaxDeltaOmega      = cfg.Agent.MaxDeltaOmega;
MaxDeltaV          = cfg.Agent.MaxDeltaV;
UseDomainRand      = cfg.UseDomainRand;
DomainRand         = cfg.DomainRand;
SaveDir            = cfg.SaveDir;

%% ───────────────────────────────────────────────────────────────────────
%  2) 解析模型路径并做前置检查 / Resolve model & preflight checks
%  ───────────────────────────────────────────────────────────────────────
mdl      = "visual_line_follower_sac_residual";
mdlFile  = fullfile(matlabDir, mdl + ".slx");
agentBlk = mdl + "/SAC_Residual_Controller/SAC_Agent";

assert(isfile(mdlFile), "TrainSAC:MissingModel", ...
    "Model not found: %s\nRun build_sac_residual_controller() first.", mdlFile);
assert(~isempty(ver("rl")), "TrainSAC:NoRL", ...
    "Reinforcement Learning Toolbox license required.");
assert(~isempty(which("mj_initbus")), "TrainSAC:NoMuJoCo", ...
    "Simulink Blockset for MuJoCo Simulator is not installed.");

% 解析输出目录（内联了函数版里的 saveDir 辅助函数）——训练模式下新 agent 存
% 到这里；仅评估模式下也是自动挑最新 sac_agent_*.mat 的默认位置。
if strlength(SaveDir) > 0
    outDir = char(SaveDir);
else
    outDir = fullfile(char(matlabDir), "simulation_data", "sac_training");
end
if ~isfolder(outDir), mkdir(outDir); end

%% ───────────────────────────────────────────────────────────────────────
%  3) 打开模型 + 设置地图 / Open model & set map
%  ───────────────────────────────────────────────────────────────────────
if ~bdIsLoaded(mdl), open_system(mdlFile); end
set_turtlebot3_mujoco_scene(mdl, MapKey);
fprintf("Training map: %s\n", MapKey);

% 把奖励权重写进模型工作区（模型里的 Reward_Calculation / Done_Detection
% 子系统直接引用这些变量）。训练模式下这就是训练时实际生效的奖励；仅评估
% 模式下不影响已训练好的策略，只影响仿真过程中记录的 reward 数值。
ws = get_param(mdl, "ModelWorkspace");
assignin(ws, "SAC_Q",             RewardQ);
assignin(ws, "SAC_Q_Lateral",     RewardQLateral);
assignin(ws, "SAC_Q_Heading",     RewardQHeading);
assignin(ws, "SAC_R",             RewardR);
assignin(ws, "SAC_P_Lost",        RewardPLost);
assignin(ws, "SAC_Done_Steps",    DoneSteps);
assignin(ws, "SAC_MaxDeltaOmega", MaxDeltaOmega);
assignin(ws, "SAC_MaxDeltaV",     MaxDeltaV);
fprintf("Reward weights: Q=%.3g Q_lat=%.3g Q_head=%.3g R=%.3g P_lost=%.3g | done after %d lost steps\n", ...
    RewardQ, RewardQLateral, RewardQHeading, RewardR, RewardPLost, DoneSteps);
if TrainAgent
    fprintf("Mode: TRAIN (cfg.TrainAgent = true)\n");
else
    fprintf("Mode: EVAL ONLY (cfg.TrainAgent = false)\n");
end

%% ───────────────────────────────────────────────────────────────────────
%  4) 训练或仅评估 / Train or evaluate-only (cfg.TrainAgent switch)
%  ───────────────────────────────────────────────────────────────────────
if TrainAgent
    % ---- 4a) 观测/动作规格 + 构建 SAC agent -----------------------------
    % 动作是 2 维 [delta_v; delta_omega]，两者分别跟 PID 的线速度/角速度指令
    % 相加后，在根层共享子系统 Diff_Drive_Kinematics 里统一转成左右轮速。
    [obsInfo, actInfo] = rl_io_specs(MaxDeltaOmega, MaxDeltaV);

    agentOpts = rlSACAgentOptions( ...
        "SampleTime",             -1, ...
        "DiscountFactor",          DiscountFactor, ...
        "MiniBatchSize",           MiniBatchSize, ...
        "ExperienceBufferLength",  BufferLength, ...
        "TargetSmoothFactor",      TargetSmooth);

    agentOpts.EntropyWeightOptions.EntropyWeight  = EntropyWeight;
    agentOpts.ActorOptimizerOptions.LearnRate     = ActorLearnRate;
    agentOpts.CriticOptimizerOptions(1).LearnRate = CriticLearnRate;
    agentOpts.CriticOptimizerOptions(2).LearnRate = CriticLearnRate;

    agent = rlSACAgent(obsInfo, actInfo, agentOpts);

    % 把 agent 载入模型的 RL Agent 块（块引用基础工作区变量 'agent'）。
    set_param(agentBlk, "Agent", "agent");
    assignin("base", "agent", agent);

    % ---- 4b) 训练环境与选项 ---------------------------------------------
    env = rlSimulinkEnv(mdl, agentBlk, obsInfo, actInfo);

    % 域随机化：每回合 ResetFcn 扰动模型工作区变量（起始位姿/动力学/速度PID/感知）。
    DomainRand.Enable = logical(UseDomainRand);
    env.ResetFcn = rl_domain_randomization(mdl, DomainRand);
    if DomainRand.Enable
        fprintf("Domain randomization ON.\n");
    else
        fprintf("Domain randomization OFF.\n");
    end

    % logical → 绘图标志（避免 if/else 在常量参数下触发 "unreachable" 警告）
    plotChoices = ["none", "training-progress"];
    plotFlag    = plotChoices(double(logical(ShowPlot)) + 1);

    % StopReward=Inf 时不做基于奖励的提前停止：跑满 MaxEpisodes 拿完整学习曲线。
    if isfinite(StopReward)
        stopArgs = {"StopTrainingCriteria", "AverageReward", ...
                    "StopTrainingValue", StopReward};
    else
        stopArgs = {"StopTrainingCriteria", "none"};
    end
    % 存档：SaveEveryNEpisodes 非空且 >0 时按固定回合数存档；否则退回 SaveReward
    % 阈值——回合奖励高于它就自动存档。
    if ~isempty(SaveEveryNEpisodes) && SaveEveryNEpisodes > 0
        saveArgs = {"SaveAgentCriteria", "EpisodeFrequency", ...
                    "SaveAgentValue",    SaveEveryNEpisodes};
    else
        saveArgs = {"SaveAgentCriteria", "EpisodeReward", ...
                    "SaveAgentValue",    SaveReward};
    end

    trainOpts = rlTrainingOptions( ...
        "MaxEpisodes",                MaxEpisodes, ...
        "MaxStepsPerEpisode",         MaxStepsPerEpisode, ...
        "ScoreAveragingWindowLength", ScoreWindow, ...
        stopArgs{:}, ...
        saveArgs{:}, ...
        "SaveAgentDirectory",         outDir, ...
        "Verbose",                    true, ...
        "Plots",                      plotFlag, ...
        "UseParallel",                UseParallel);

    % UseFast=true 时 MuJoCo 切到无渲染模式，加快墙钟时间（块路径 :16 = MuJoCo_Plant）。
    if logical(UseFast)
        set_param(mdl + ":16", "renderingType", "None");
    end

    % 并行/批量训练：按 NumWorkers 配置并行池（RL Toolbox 用当前池的 worker 数，
    % 没有单独选项），再存盘——worker 从磁盘加载模型副本，看不到内存中未保存的
    % 改动（设地图、写工作区变量、set_param 等），否则报 "model has unsaved changes"。
    if logical(UseParallel)
        pool = gcp("nocreate");
        if isempty(pool) || pool.NumWorkers ~= NumWorkers
            if ~isempty(pool), delete(pool); end
            parpool(NumWorkers);
        end
        save_system(mdl);
        fprintf("UseParallel=true: saved model for parallel workers.\n");
    end

    % ---- 4c) 训练 ---------------------------------------------------------
    fprintf("Starting SAC training: %d episodes, map=%s\n", MaxEpisodes, MapKey);
    fprintf("  Obs dim: %s  |  Act dim: %s\n", ...
        mat2str(obsInfo.Dimension), mat2str(actInfo.Dimension));
    fprintf("  Mini-batch: %d  |  Buffer: %d\n", MiniBatchSize, BufferLength);

    % train() 返回训练统计；agent 是句柄对象，训练中原地更新。
    trainInfo    = train(agent, env, trainOpts);
    trainedAgent = agent;

    % ---- 4d) 保存 + 载入模型块 ----------------------------------------
    ts        = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
    agentFile = fullfile(outDir, "sac_agent_" + ts + ".mat");
    save(agentFile, "trainedAgent", "trainInfo", "obsInfo", "actInfo", "-v7.3");
    fprintf("Saved trained agent: %s\n", agentFile);

    % 把训练好的 agent 持久化到 SAC_Agent 块，便于直接仿真评估。
    assignin("base", "trainedAgent", trainedAgent);
    set_param(agentBlk, "Agent", "trainedAgent");
    save_system(mdl);
    fprintf("Loaded trained agent into %s.\n", agentBlk);

    % 因为是脚本，trainedAgent / obsInfo / actInfo 都留在基础工作区，
    % 训练完可以直接评估，不必再从 .mat 载入：
    %   sim("visual_line_follower_sac_residual")
    fprintf("Done. Evaluate with:  sim(""%s"")\n", mdl);
else
    % ---- 仅评估模式：不训练，载入已有 agent 后直接 sim() -----------------
    % 解析要评估哪个 agent：显式指定优先，否则挑 outDir 下最新的
    % sac_agent_*.mat（按文件修改时间）。
    if strlength(EvalAgentFile) > 0
        agentFile = char(EvalAgentFile);
        assert(isfile(agentFile), "TrainSAC:NoAgentToEval", ...
            "cfg.EvalAgentFile 指定的文件不存在: %s", agentFile);
    else
        agentFiles = dir(fullfile(outDir, "sac_agent_*.mat"));
        assert(~isempty(agentFiles), "TrainSAC:NoAgentToEval", ...
            "%s 下找不到已训练的 sac_agent_*.mat，先训练一次（cfg.TrainAgent=true）或设置 cfg.EvalAgentFile。", ...
            outDir);
        [~, newestIdx] = max([agentFiles.datenum]);
        agentFile = fullfile(agentFiles(newestIdx).folder, agentFiles(newestIdx).name);
    end

    loaded = load(agentFile, "trainedAgent");
    assignin("base", "trainedAgent", loaded.trainedAgent);
    set_param(agentBlk, "Agent", "trainedAgent");
    fprintf("Loaded agent for evaluation: %s\n", agentFile);

    % 不强制 UseFast/关渲染——评估通常是想看着车跑，保留模型当前的渲染设置。
    simOut = sim(mdl);
    assignin("base", "simOut", simOut);
    fprintf("Evaluation sim() complete. Inspect simOut in the base workspace.\n");
end
