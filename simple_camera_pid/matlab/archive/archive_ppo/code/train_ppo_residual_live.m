%[text] # PPO 残差强化学习训练（实时脚本版）
%[text] 这是 PPO 残差控制器的独立训练入口：没有 function 包装，所有可调参数集中在「参数」一节，改完数字直接运行即可。
%[text] **运行前提**：已安装 Reinforcement Learning Toolbox 与 MuJoCo Blockset，且模型 `visual_line_follower_ppo_residual.slx` 已存在。
%[text] **请从** `.../simple_camera_pid/matlab/train/` **目录运行本实时脚本**（实时脚本中 `mfilename` 不可用，这里用 `pwd` 定位项目，因此当前目录必须是 train/）。
%[text:tableOfContents]{"heading":"目录"}
%%
%[text] ## 路径准备
%[text] 本脚本位于 `.../matlab/train/`，模型与 `runtime` / `experiments` 在上一层。优先用 Live Editor 的当前文件名定位脚本目录（比 `pwd` 稳健，不依赖当前目录），失败再回退到 `pwd`。随后校验 `runtime` 子目录确实存在，避免路径算错时静默失败。
try
    scriptDir = fileparts(matlab.desktop.editor.getActiveFilename);
catch
    scriptDir = pwd;   % 回退：请从 train/ 目录运行
end
matlabDir = fullfile(scriptDir, "..");      % .../simple_camera_pid/matlab
assert(isfolder(fullfile(matlabDir, "runtime")), "TrainPPO:BadDir", ...
    "找不到 runtime 目录：%s\n请从 .../simple_camera_pid/matlab/train/ 运行本脚本。", ...
    fullfile(matlabDir, "runtime"));
addpath(fullfile(matlabDir, "runtime"));
addpath(scriptDir);                             % 参数中心 ppo_training_config 所在目录（本目录）
addpath(fullfile(matlabDir, "train", "shared")); % rl_* 共享训练函数所在目录
cd(matlabDir);
%%
%[text] ## 参数
%[text] 所有可调参数集中在同目录的参数中心 [`ppo_training_config.m`](./ppo_training_config.m)：模型、训练流程、地图、奖励权重、PPO 超参数、域随机化、输出目录。**调参请编辑那个文件**，本脚本只负责读取并执行。下面把 cfg 展开成局部变量，保持后续代码简洁。
cfg = ppo_training_config();
MaxEpisodes        = cfg.MaxEpisodes;
MaxStepsPerEpisode = cfg.MaxStepsPerEpisode;
StopReward         = cfg.StopReward;
SaveReward         = cfg.SaveReward;
ScoreWindow        = cfg.ScoreWindow;
UseFast            = cfg.UseFast;
UseParallel        = cfg.UseParallel;
NumWorkers         = cfg.NumWorkers;
ShowPlot           = cfg.ShowPlot;
MapKey             = cfg.MapKey;
RewardQ            = cfg.Reward.Q;
RewardR            = cfg.Reward.R;
RewardPLost        = cfg.Reward.P_Lost;
DoneSteps          = cfg.Reward.DoneSteps;
ActorLearnRate     = cfg.Agent.ActorLearnRate;
CriticLearnRate    = cfg.Agent.CriticLearnRate;
DiscountFactor     = cfg.Agent.DiscountFactor;
ExperienceHorizon  = cfg.Agent.ExperienceHorizon;
MiniBatchSize      = cfg.Agent.MiniBatchSize;
NumEpoch           = cfg.Agent.NumEpoch;
ClipFactor         = cfg.Agent.ClipFactor;
EntropyLossWeight  = cfg.Agent.EntropyLossWeight;
GAEFactor          = cfg.Agent.GAEFactor;
MaxDeltaOmega      = cfg.Agent.MaxDeltaOmega;
UseDomainRand      = cfg.UseDomainRand;
DomainRand         = cfg.DomainRand;
SaveDir            = cfg.SaveDir;
%%
%[text] ## 解析模型路径并做前置检查
%[text] 定义模型名与 RL Agent 块路径，确认模型文件、RL 工具箱、MuJoCo Blockset 都就位，并解析输出目录。
mdl      = cfg.ModelName;
mdlFile  = fullfile(matlabDir, mdl + ".slx");
agentBlk = mdl + "/" + cfg.AgentBlockPath;
assert(isfile(mdlFile), "TrainPPO:MissingModel", ...
    "Model not found: %s.", mdlFile);
assert(~isempty(ver("rl")), "TrainPPO:NoRL", ...
    "Reinforcement Learning Toolbox license required.");
assert(~isempty(which("mj_initbus")), "TrainPPO:NoMuJoCo", ...
    "Simulink Blockset for MuJoCo Simulator is not installed.");
if strlength(SaveDir) > 0
    outDir = char(SaveDir);
else
    outDir = fullfile(char(matlabDir), "simulation_data", "ppo_training");
end
if ~isfolder(outDir), mkdir(outDir); end
%%
%[text] ## 打开模型并设置地图
%[text] 打开模型、设置 MuJoCo 场景，并把奖励权重写进模型工作区。模型里的 `Reward_Calculation` / `Done_Detection` 子系统直接引用这些变量，所以改这里就改了训练时实际生效的奖励。
% 清理上次崩溃/中断可能残留的 MuJoCo MEX host（健康会话上是无害的幂等操作）
recover_mujoco_host(mdl);
if ~bdIsLoaded(mdl), open_system(mdlFile); end
set_turtlebot3_mujoco_scene(mdl, MapKey);
ws = get_param(mdl, "ModelWorkspace");
assignin(ws, "PPO_Q",             RewardQ);
assignin(ws, "PPO_R",             RewardR);
assignin(ws, "PPO_P_Lost",        RewardPLost);
assignin(ws, "PPO_Done_Steps",    DoneSteps);
assignin(ws, "PPO_MaxDeltaOmega", MaxDeltaOmega);
disp("Reward weights written to model workspace:")
disp(table(RewardQ, RewardR, RewardPLost, DoneSteps))
%%
%[text] ## 观测/动作规格与构建 PPO 智能体
%[text] 观测/动作规格来自共享函数 `train/shared/rl_io_specs.m`（SAC/PPO 残差控制器 I/O 契约相同，与模型 I/O 同源，保证维度与边界不分叉）。然后用 MATLAB 默认网络构建 PPO 智能体。
[obsInfo, actInfo] = rl_io_specs(MaxDeltaOmega);
agentOpts = rlPPOAgentOptions( ...
    "SampleTime",          -1, ...
    "DiscountFactor",       DiscountFactor, ...
    "ExperienceHorizon",    ExperienceHorizon, ...
    "MiniBatchSize",        MiniBatchSize, ...
    "NumEpoch",             NumEpoch, ...
    "ClipFactor",           ClipFactor, ...
    "EntropyLossWeight",    EntropyLossWeight, ...
    "GAEFactor",            GAEFactor);
agentOpts.ActorOptimizerOptions.LearnRate  = ActorLearnRate;
agentOpts.CriticOptimizerOptions.LearnRate = CriticLearnRate;
agent = rlPPOAgent(obsInfo, actInfo, agentOpts);
set_param(agentBlk, "Agent", "agent");      % 块引用基础工作区变量 'agent'
assignin("base", "agent", agent);
%%
%[text] ## 训练环境与域随机化
%[text] 建立 Simulink 强化学习环境，并挂上域随机化重置函数：每回合开始时 `ResetFcn` 会扰动模型工作区变量（起始位姿 / 动力学 / 速度PID / 感知），让残差策略更能泛化。`UseDomainRand=false` 时退化为恒等重置。
env = rlSimulinkEnv(mdl, agentBlk, obsInfo, actInfo);
DomainRand.Enable = logical(UseDomainRand);
env.ResetFcn = rl_domain_randomization(mdl, DomainRand);
% 换图/现场生成赛道要在 ResetFcn 里改 MuJoCo Plant 的 nontunable 参数 xmlFile，
% Fast Restart 开启时会报错（第 2 回合 reset 即崩），必须关掉。Fast Restart 只
% 加速同结构复跑，回合间换场景后本来也没有收益。与 train_sac_residual_live 一致。
mapRand  = logical(UseDomainRand) && DomainRand.Map.Enable && ...
    ~isempty(DomainRand.Map.Maps);
genTrack = logical(UseDomainRand) && DomainRand.GenTrack.Enable;
if mapRand || genTrack
    env.UseFastRestart = "off";
end
disp("Domain randomization enabled:")
disp(DomainRand.Enable)
%%
%[text] ## 训练选项
%[text] 配置训练选项。`UseFast=true` 时把 MuJoCo 切到无渲染模式以加快墙钟时间（块路径 `:16` 即 MuJoCo_Plant）。
plotChoices = ["none", "training-progress"];
plotFlag    = plotChoices(double(logical(ShowPlot)) + 1);
% StopReward=Inf 时不做基于奖励的提前停止：跑满 MaxEpisodes 拿完整学习曲线。
% SaveReward 是检查点阈值——回合奖励高于它就自动存档，崩溃时不至于全丢。
if isfinite(StopReward)
    stopArgs = {"StopTrainingCriteria", "AverageReward", ...
                "StopTrainingValue", StopReward};
else
    stopArgs = {"StopTrainingCriteria", "none"};
end
trainOpts = rlTrainingOptions( ...
    "MaxEpisodes",                MaxEpisodes, ...
    "MaxStepsPerEpisode",         MaxStepsPerEpisode, ...
    "ScoreAveragingWindowLength", ScoreWindow, ...
    stopArgs{:}, ...
    "SaveAgentCriteria",          "EpisodeReward", ...
    "SaveAgentValue",             SaveReward, ...
    "SaveAgentDirectory",         outDir, ...
    "Verbose",                    true, ...
    "Plots",                      plotFlag, ...
    "UseParallel",                UseParallel);
if logical(UseFast)
    set_param(mdl + ":16", "renderingType", "None");
end
% 并行/批量训练要求模型先存盘：每个 worker 从磁盘加载模型副本，看不到内存中
% 未保存的改动（设地图、写工作区变量、set_param 等）。不存盘会报
% "model has unsaved changes"，本项目上曾进一步导致 MATLAB 崩溃。
if logical(UseParallel)
    % 按 NumWorkers 配置并行池（RL Toolbox 用当前池的 worker 数，没有单独选项）
    pool = gcp("nocreate");
    if isempty(pool) || pool.NumWorkers ~= NumWorkers
        if ~isempty(pool), delete(pool); end
        parpool(NumWorkers);
    end
    save_system(mdl);
    disp("UseParallel=true：已保存模型以供并行 worker 加载。")
end
%%
%[text] ## 训练
%[text] `train` 返回训练统计；`agent` 是句柄对象，训练过程中原地更新。这一步耗时较长（不加速时尤其明显）。
trainInfo    = train(agent, env, trainOpts);
trainedAgent = agent;
%%
%[text] ## 保存并载入模型块
%[text] 保存训练好的智能体与统计，并把它持久化到 `PPO_Agent` 块，便于直接仿真评估。因为是脚本运行，`trainedAgent` / `obsInfo` / `actInfo` 会留在基础工作区，训练完可直接 `sim("visual_line_follower_ppo_residual")` 评估，无需再从 `.mat` 载入。
ts        = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
agentFile = fullfile(outDir, "ppo_agent_" + ts + ".mat");
save(agentFile, "trainedAgent", "trainInfo", "obsInfo", "actInfo", "-v7.3");
assignin("base", "trainedAgent", trainedAgent);
set_param(agentBlk, "Agent", "trainedAgent");
save_system(mdl);
disp("Saved trained agent to:")
disp(agentFile)

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
