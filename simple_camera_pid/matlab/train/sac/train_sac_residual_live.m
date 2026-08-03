%[text] # SAC 残差强化学习训练（实时脚本版）
%[text] SAC 残差训练的**唯一入口**。所有可调参数集中在同目录的参数中心 `sac_training_config.m`，本脚本读取后执行。新增 `cfg.TrainAgent` 开关：`true` = 从头训练一个新 agent；`false` = 跳过训练，直接把已训练好的 agent 载入模型并 `sim()` 跑一次评估。（旧的函数式入口 `run_sac_training` / `train_sac_residual_agent` 已移除。）
%[text] **运行前提**：已安装 Reinforcement Learning Toolbox 与 MuJoCo Blockset，且模型 `visual_line_follower_sac_residual.slx` 已存在；`cfg.TrainAgent=false` 时还需要 `simulation_data/sac_training/`（或 `cfg.EvalAgentFile` 指定路径）下已有一个训练好的 `sac_agent_*.mat`。
%[text] **请从** `.../simple_camera_pid/matlab/train/sac/` **目录运行本实时脚本**（实时脚本中 `mfilename` 不可用，这里用 `pwd` 定位项目，因此当前目录必须是 train/sac/）。
%[text:tableOfContents]{"heading":"目录"}
%%
%[text] ## 路径准备
%[text] 本脚本位于 `.../matlab/train/sac/`，模型与 `runtime` / `experiments` 在上两层，共享的 RL 函数在 `train/shared/`。优先用 Live Editor 的当前文件名定位脚本目录（比 `pwd` 稳健，不依赖当前目录），失败再回退到 `pwd`。随后校验 `runtime` 子目录确实存在，避免路径算错时静默失败。
try
    scriptDir = fileparts(matlab.desktop.editor.getActiveFilename);
catch
    scriptDir = pwd;   % 回退：请从 train/sac/ 目录运行
end
matlabDir = fullfile(scriptDir, "..", "..");    % .../simple_camera_pid/matlab
assert(isfolder(fullfile(matlabDir, "runtime")), "TrainSAC:BadDir", ...
    "找不到 runtime 目录：%s\n请从 .../simple_camera_pid/matlab/train/sac/ 运行本脚本。", ...
    fullfile(matlabDir, "runtime"));
addpath(genpath(fullfile(matlabDir, "runtime"))); % runtime/ 分 init/vision/scene/ops 子目录，递归加入
addpath(scriptDir);                             % 参数中心 sac_training_config 所在目录（本目录）
addpath(fullfile(matlabDir, "train", "shared")); % rl_* 共享训练函数所在目录
cd(matlabDir);
%%
%[text] ## 参数
%[text] 所有可调参数集中在同目录的参数中心 [`sac_training_config.m`](file:./sac_training_config.m)：模型、训练/评估开关、训练流程、地图、奖励权重、SAC 超参数、域随机化、输出目录。**调参请编辑那个文件**，本脚本只负责读取并执行。下面把 cfg 展开成局部变量，保持后续代码简洁。
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
%%
%[text] ## 解析模型路径并做前置检查
%[text] 定义模型名与 RL Agent 块路径，确认模型文件、RL 工具箱、MuJoCo Blockset 都就位，并解析输出目录（内联了函数版里的 `saveDir` 辅助逻辑）。这个目录也是仅评估模式下自动寻找最新已训练 agent 的默认位置。
mdl      = cfg.ModelName;
mdlFile  = fullfile(matlabDir, mdl + ".slx");
agentBlk = mdl + "/" + cfg.AgentBlockPath;
assert(isfile(mdlFile), "TrainSAC:MissingModel", ...
    "Model not found: %s\nRun build_sac_residual_controller() first.", mdlFile);
assert(~isempty(ver("rl")), "TrainSAC:NoRL", ...
    "Reinforcement Learning Toolbox license required.");
assert(~isempty(which("mj_initbus")), "TrainSAC:NoMuJoCo", ...
    "Simulink Blockset for MuJoCo Simulator is not installed.");
if strlength(SaveDir) > 0
    outDir = char(SaveDir);
else
    outDir = fullfile(char(matlabDir), "simulation_data", "sac_training");
end
if ~isfolder(outDir), mkdir(outDir); end
%%
%[text] ## 打开模型并设置地图
%[text] 打开模型、设置 MuJoCo 场景，并把奖励权重写进模型工作区。模型里的 `Reward_Calculation` / `Done_Detection` 子系统直接引用这些变量：训练模式下这就是训练时实际生效的奖励；仅评估模式下不影响已训练好的策略，只影响仿真过程中记录的 reward 数值。
% 清理上次崩溃/中断可能残留的 MuJoCo MEX host（健康会话上是无害的幂等操作）
recover_mujoco_host(mdl);
if ~bdIsLoaded(mdl), open_system(mdlFile); end
set_turtlebot3_mujoco_scene(mdl, MapKey); %[output:67d0685a]
ws = get_param(mdl, "ModelWorkspace");
assignin(ws, "SAC_Q",             RewardQ);
assignin(ws, "SAC_Q_Lateral",     RewardQLateral);
assignin(ws, "SAC_Q_Heading",     RewardQHeading);
assignin(ws, "SAC_R",             RewardR);
assignin(ws, "SAC_P_Lost",        RewardPLost);
assignin(ws, "SAC_Done_Steps",    DoneSteps);
assignin(ws, "SAC_MaxDeltaOmega", MaxDeltaOmega);
assignin(ws, "SAC_MaxDeltaV",     MaxDeltaV);
disp("Reward weights written to model workspace:") %[output:4c17d287]
disp(table(RewardQ, RewardQLateral, RewardQHeading, RewardR, RewardPLost, DoneSteps)) %[output:9a8407bb]
if TrainAgent %[output:group:37afdc78]
    disp("Mode: TRAIN (cfg.TrainAgent = true) - building a new agent and calling train().")
else
    disp("Mode: EVAL ONLY (cfg.TrainAgent = false) - loading an existing agent and calling sim(), no training.") %[output:59e7aba5]
end %[output:group:37afdc78]
%%
%[text] ## 训练或仅评估（`cfg.TrainAgent` 开关）
%[text] `cfg.TrainAgent = true` 时执行原有训练流程：构建观测/动作规格与新 SAC 智能体（`train/shared/rl_io_specs.m`，动作是 2 维 `[delta_v; delta_omega]`，2026-07-21 起 PID 和 SAC 共用根层 `Diff_Drive_Kinematics` 转轮速），挂上域随机化 `ResetFcn`，配置 `rlTrainingOptions`，调用 `train()`，最后把训练好的 agent 存档并载入 `SAC_Agent` 块。`cfg.TrainAgent = false` 时跳过以上全部步骤，改为从 `cfg.EvalAgentFile`（或 `outDir` 下最新的 `sac_agent_*.mat`）载入一个已训练好的 agent 到同一个块，再直接 `sim(mdl)` 跑一次仿真评估，即"只调用模型，不训练"。
if TrainAgent %[output:group:1f59f53b]
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
    set_param(agentBlk, "Agent", "agent");      % 块引用基础工作区变量 'agent'
    assignin("base", "agent", agent);
    env = rlSimulinkEnv(mdl, agentBlk, obsInfo, actInfo);
    DomainRand.Enable = logical(UseDomainRand);
    env.ResetFcn = rl_domain_randomization(mdl, DomainRand);
    % 换图/现场生成赛道要在 ResetFcn 里改 MuJoCo Plant 的 nontunable 参数 xmlFile，
    % Fast Restart 开启时会报错（第 2 回合 reset 即崩），必须关掉。
    mapRand  = logical(UseDomainRand) && DomainRand.Map.Enable && ...
        ~isempty(DomainRand.Map.Maps);
    genTrack = logical(UseDomainRand) && DomainRand.GenTrack.Enable;
    if mapRand || genTrack
        env.UseFastRestart = "off";
    end
    disp("Domain randomization enabled:")
    disp(DomainRand.Enable)
    plotChoices = ["none", "training-progress"];
    plotFlag    = plotChoices(double(logical(ShowPlot)) + 1);
    % StopReward=Inf 时不做基于奖励的提前停止：跑满 MaxEpisodes 拿完整学习曲线。
    if isfinite(StopReward)
        stopArgs = {"StopTrainingCriteria", "AverageReward", ...
                    "StopTrainingValue", StopReward};
    else
        stopArgs = {"StopTrainingCriteria", "none"};
    end
    % 存档：SaveEveryNEpisodes 非空且 >0 时按固定回合数存档（不管奖励高低，
    % 更可控）；否则退回 SaveReward 阈值——回合奖励高于它就自动存档。
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
    if logical(UseFast)
        set_param(mdl + ":16", "renderingType", "None");
    end
    % 并行/批量训练要求模型先存盘：每个 worker 从磁盘加载模型副本，看不到内存中
    % 未保存的改动（设地图、写工作区变量、set_param 等）。
    if logical(UseParallel)
        pool = gcp("nocreate");
        if isempty(pool) || pool.NumWorkers ~= NumWorkers
            if ~isempty(pool), delete(pool); end
            parpool(NumWorkers);
        end
        save_system(mdl);
        disp("UseParallel=true：已保存模型以供并行 worker 加载。")
    end
    trainInfo    = train(agent, env, trainOpts);
    trainedAgent = agent;
    ts        = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
    agentFile = fullfile(outDir, "sac_agent_" + ts + ".mat");
    save(agentFile, "trainedAgent", "trainInfo", "obsInfo", "actInfo", "-v7.3");
    assignin("base", "trainedAgent", trainedAgent);
    set_param(agentBlk, "Agent", "trainedAgent");
    save_system(mdl);
    disp("Saved trained agent to:")
    disp(agentFile)
else
    if strlength(EvalAgentFile) > 0
        agentFile = char(EvalAgentFile);
        assert(isfile(agentFile), "TrainSAC:NoAgentToEval", ...
            "cfg.EvalAgentFile 指定的文件不存在：%s", agentFile);
    else
        agentFiles = dir(fullfile(outDir, "sac_agent_*.mat"));
        assert(~isempty(agentFiles), "TrainSAC:NoAgentToEval", ...
            "%s 下找不到已训练的 sac_agent_*.mat，先训练一次（cfg.TrainAgent=true）或设置 cfg.EvalAgentFile。", outDir);
        [~, newestIdx] = max([agentFiles.datenum]);
        agentFile = fullfile(agentFiles(newestIdx).folder, agentFiles(newestIdx).name);
    end
    loaded = load(agentFile, "trainedAgent");
    assignin("base", "trainedAgent", loaded.trainedAgent);
    set_param(agentBlk, "Agent", "trainedAgent");
    disp("Loaded agent for evaluation:") %[output:7f17db0c]
    disp(agentFile) %[output:9770fc6f]
    % 不强制关渲染——评估通常是想看着车跑，保留模型当前的渲染设置。
    simOut = sim(mdl); %[output:327e3e03] %[output:17d63306]
    assignin("base", "simOut", simOut);
    disp("Evaluation sim() complete. Inspect simOut in the base workspace.") %[output:742c65ff]
end %[output:group:1f59f53b]

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
%[output:67d0685a]
%   data: {"dataType":"text","outputData":{"text":"Set visual_line_follower_sac_residual MuJoCo scene to \/media\/kevin\/ding\/final_project\/Sheffield\/for_getting_started\/src\/simple_camera_pid\/matlab\/..\/model\/mujoco\/turtlebot3\/simple_camera_track_turtlebot3_burger_visual_scene.xml (in memory; not saved)\n","truncated":false}}
%---
%[output:4c17d287]
%   data: {"dataType":"text","outputData":{"text":"Reward weights written to model workspace:\n","truncated":false}}
%---
%[output:9a8407bb]
%   data: {"dataType":"text","outputData":{"text":"    <strong>RewardQ<\/strong>    <strong>RewardQLateral<\/strong>    <strong>RewardQHeading<\/strong>    <strong>RewardR<\/strong>    <strong>RewardPLost<\/strong>    <strong>DoneSteps<\/strong>\n    <strong>_______<\/strong>    <strong>______________<\/strong>    <strong>______________<\/strong>    <strong>_______<\/strong>    <strong>___________<\/strong>    <strong>_________<\/strong>\n\n       1            0.5               0.5            0.1          10            100   \n\n","truncated":false}}
%---
%[output:59e7aba5]
%   data: {"dataType":"text","outputData":{"text":"Mode: EVAL ONLY (cfg.TrainAgent = false) - loading an existing agent and calling sim(), no training.\n","truncated":false}}
%---
%[output:7f17db0c]
%   data: {"dataType":"text","outputData":{"text":"Loaded agent for evaluation:\n","truncated":false}}
%---
%[output:9770fc6f]
%   data: {"dataType":"text","outputData":{"text":"\/media\/kevin\/ding\/final_project\/Sheffield\/for_getting_started\/src\/simple_camera_pid\/matlab\/simulation_data\/sac_training\/sac_agent_20260722_191331.mat\n","truncated":false}}
%---
%[output:327e3e03]
%   data: {"dataType":"text","outputData":{"text":"Active TurtleBot3 map: Simple camera track (simple)\n","truncated":false}}
%---
%[output:17d63306]
%   data: {"dataType":"warning","outputData":{"text":"Warning: '<a href=\"matlab:Simulink.internal.open_and_hilite_port_hyperlink('hilite', ['visual_line_follower_sac_residual\/OriginBot_Line_Follower'], 'Outport', 5);\">Output Port 5<\/a>' of block '<a href=\"matlab:open_and_hilite_hyperlink ('visual_line_follower_sac_residual\/OriginBot_Line_Follower','error')\">visual_line_follower_sac_residual\/OriginBot_Line_Follower<\/a>' is not connected."}}
%---
%[output:742c65ff]
%   data: {"dataType":"text","outputData":{"text":"Evaluation sim() complete. Inspect simOut in the base workspace.\n","truncated":false}}
%---
