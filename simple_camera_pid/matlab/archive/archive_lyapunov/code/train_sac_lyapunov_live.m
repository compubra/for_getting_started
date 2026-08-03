%[text] # SAC 残差强化学习训练 · 李雅普诺夫奖励版（实时脚本）
%[text] 这是 **李雅普诺夫奖励** 版训练的实时脚本入口，对应模型 `visual_line_follower_sac_lyapunov.slx`（其 `Reward_Calculation` 子系统用原生块计算李雅普诺夫奖励）。所有可调参数集中在「参数」一节，改完直接运行。
%[text] 奖励： $reward = -(V + \\lambda\\dot V) - R u^2 - P\_{lost}(1-found) + \\dfrac{K\_{inv}}{|e\_s|+|e\_l|+|e\_h|+\\varepsilon}$，其中 $V = Q\_e e\_s^2 + Q\_l e\_l^2 + Q\_h e\_h^2$。
%[text] **两个关键点**
%[text] - **是否训练开关** `DoTrain`：`true` 从零训练；`false` 跳过训练，直接加载最新已存 agent 载入模型块，可立即 `sim` 评估。
%[text] - **无阈值早停**：训练一定跑满 `MaxEpisodes`，得到完整学习曲线（`SaveReward` 只是「达到即存盘」的检查点阈值，不影响停止）。 \
%[text] **运行前提**：已安装 Reinforcement Learning Toolbox 与 MuJoCo Blockset。**请从** `.../simple_camera_pid/matlab/train/` **目录运行本实时脚本**。
%[text:tableOfContents]{"heading":"目录"}
%%
%[text] ## 路径准备
%[text] 优先用 Live Editor 当前文件名定位脚本目录（比 `pwd` 稳健），失败再回退 `pwd`。随后切到 `.../matlab` 并把 `runtime` / `experiments` 加入路径。
try
    scriptDir = fileparts(matlab.desktop.editor.getActiveFilename);
catch
    scriptDir = pwd;   % 回退：请从 train/ 目录运行
end
matlabDir = fullfile(scriptDir, "..");      % .../simple_camera_pid/matlab
assert(isfolder(fullfile(matlabDir, "runtime")), "TrainLyap:BadDir", ...
    "找不到 runtime 目录：%s\n请从 .../simple_camera_pid/matlab/train/ 运行本脚本。", ...
    fullfile(matlabDir, "runtime"));
addpath(fullfile(matlabDir, "runtime"));
addpath(fullfile(matlabDir, "experiments"));
addpath(fullfile(matlabDir, "train", "shared"));  % rl_dr_defaults / plot_rl_training 等共享训练函数
cd(matlabDir);
%%
%[text] ## 参数
%[text] **是否训练**：`DoTrain=false` 时只加载已训练 agent 去评估，不训练。留空 `LoadAgentFile` 则自动挑最新的 `sac_lyapunov_agent_*.mat`。
DoTrain       = true;
LoadAgentFile = "";        % 仅在 DoTrain=false 时用；留空=自动选最新
%[text] **训练流程**（无阈值早停：一定跑满 `MaxEpisodes`）
MaxEpisodes        = 1500;      % 训练回合数上限（跑满，无早停）
MaxStepsPerEpisode = 300;     % 每回合最大步数 (2000*0.05s = 100s)
ScoreWindow        = 20;       % 平均回报滑动窗口
SaveReward         = -0.4;     % 检查点存盘阈值（回合奖励超过即存盘）；非早停
UseFast            = true;    % true = 关闭 MuJoCo 渲染，加速训练
UseParallel        = false;    % 随机切图与 UseParallel 不建议同时用
ShowPlot           = true;    % false = 训练时不弹 Monitor（训练后再画曲线）
%[text] **地图**：基础地图 key。域随机化开启时每回合会从 `rl_dr_defaults` 的 `Map.Maps` 里**随机选图**，此处只是回退地图。可用 key：`simple` / `complex` / `ellipse` / `training` / `track_easy` / `track_medium` / `track_hard`。
MapKey = "track_hard";
%[text] **李雅普诺夫奖励权重**：写入模型工作区 `SAC_Q_e/Q_l/Q_h/Lambda/R/P_Lost/K_inv/Eps/Done_Steps`。
RewardQe     = 1.0;    % 循线误差(steer)二次惩罚权重
RewardQl     = 1.0;    % 横向误差(lateral)二次惩罚权重
RewardQh     = 1.0;    % 航向误差(heading)二次惩罚权重
RewardLambda = 0.5;    % 李雅普诺夫率 Vdot 的权重（调大→更压制误差发散）
RewardR      = 0.1;    % 动作能耗惩罚
RewardPLost  = 10.0;   % 丢线惩罚
RewardKinv   = 0.1;    % 逆误差奖励强度（三误差都很小时给正奖励）
RewardEps    = 1e-3;   % 逆误差分母防零项
DoneSteps    = 100;    % 连续丢线多少步判回合结束 (100*0.05s = 5s)
%[text] **SAC 智能体超参数**
ActorLearnRate  = 3e-4;
CriticLearnRate = 3e-4;
DiscountFactor  = 0.99;
MiniBatchSize   = 256;
BufferLength    = 2e5;
EntropyWeight   = 0.2;
TargetSmooth    = 5e-3;
MaxDeltaOmega   = 1.0;    % 残差动作上下界 |delta_omega| (rad/s)
%[text] **域随机化**：`UseDomainRand=false` 关闭全部随机化。默认含**每回合随机切图**（`rl_dr_defaults` 的 `Map` 组）。如需关掉随机切图但保留其它：`DomainRand.Map.Enable = false;`
UseDomainRand = true;
DomainRand    = rl_dr_defaults();
%[text] **输出目录**：留空 `""` 用默认 `simulation_data/sac_training`。
SaveDir = "";
%%
%[text] ## 调用训练函数
%[text] 把上面的参数一次性传给 `train_sac_lyapunov_agent`。它内部：写李雅普诺夫奖励权重 → 建 SAC agent → 挂域随机化 ResetFcn（含随机切图时自动关闭 Fast Restart）→ 训练（无早停）→ 存盘并载入 `SAC_Agent` 块。`DoTrain=false` 时它直接加载最新 agent 并返回，不训练。
DomainRand.Enable = logical(UseDomainRand);
[trainedAgent, trainInfo] = train_sac_lyapunov_agent( ... %[output:group:2dba887a] %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "DoTrain",            logical(DoTrain), ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "LoadAgentFile",      LoadAgentFile, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "MaxEpisodes",        MaxEpisodes, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "MaxStepsPerEpisode", MaxStepsPerEpisode, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "ScoreWindow",        ScoreWindow, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "SaveReward",         SaveReward, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "MapKey",             MapKey, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "UseFast",            logical(UseFast), ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "UseParallel",        logical(UseParallel), ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "ShowPlot",           logical(ShowPlot), ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "RewardQe",           RewardQe, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "RewardQl",           RewardQl, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "RewardQh",           RewardQh, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "RewardLambda",       RewardLambda, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "RewardR",            RewardR, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "RewardPLost",        RewardPLost, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "RewardKinv",         RewardKinv, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "RewardEps",          RewardEps, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "DoneSteps",          DoneSteps, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "ActorLearnRate",     ActorLearnRate, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "CriticLearnRate",    CriticLearnRate, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "DiscountFactor",     DiscountFactor, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "MiniBatchSize",      MiniBatchSize, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "BufferLength",       BufferLength, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "EntropyWeight",      EntropyWeight, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "TargetSmooth",       TargetSmooth, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "MaxDeltaOmega",      MaxDeltaOmega, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "DomainRand",         DomainRand, ... %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
    "SaveDir",            SaveDir); %[output:group:2dba887a] %[output:1263f506] %[output:12238576] %[output:5255dc20] %[output:49751424] %[output:10d24b3c] %[output:92052139] %[output:8680e2de] %[output:5412be19] %[output:47560a56]
%%
%[text] ## 训练后（可选）
%[text] 训练完 `trainedAgent` / `trainInfo` 已在基础工作区，且已载入 `SAC_Agent` 块。可直接评估或画曲线：
%[text] - 评估： `sim("visual_line_follower_sac_lyapunov")`
%[text] - 画学习曲线： `plot_rl_training(trainInfo)`（`DoTrain=false` 时 `trainInfo` 可能为 `[]`，改用 `plot_rl_training()` 自动读最新 .mat） \
if DoTrain && ~isempty(trainInfo)
    plot_rl_training(trainInfo);
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
%[output:1263f506]
%   data: {"dataType":"warning","outputData":{"text":"Warning: An error has occurred during subscription to Action named navigateBackAction."}}
%---
%[output:12238576]
%   data: {"dataType":"warning","outputData":{"text":"Warning: Attempting to subscribe to an Action named navigateBackAction, which is not in the configuration."}}
%---
%[output:5255dc20]
%   data: {"dataType":"warning","outputData":{"text":"Warning: An error has occurred during subscription to Action named navigateForwardAction."}}
%---
%[output:49751424]
%   data: {"dataType":"warning","outputData":{"text":"Warning: Attempting to subscribe to an Action named navigateForwardAction, which is not in the configuration."}}
%---
%[output:10d24b3c]
%   data: {"dataType":"warning","outputData":{"text":"Warning: An error has occurred during subscription to Action named navigateUpAction."}}
%---
%[output:92052139]
%   data: {"dataType":"warning","outputData":{"text":"Warning: Attempting to subscribe to an Action named navigateUpAction, which is not in the configuration."}}
%---
%[output:8680e2de]
%   data: {"dataType":"warning","outputData":{"text":"Warning: An error has occurred during subscription to Action named simulinkKeyboardShortcutsAction."}}
%---
%[output:5412be19]
%   data: {"dataType":"warning","outputData":{"text":"Warning: Attempting to subscribe to an Action named simulinkKeyboardShortcutsAction, which is not in the configuration."}}
%---
%[output:47560a56]
%   data: {"dataType":"error","outputData":{"errorType":"runtime","text":"Error using <a href=\"matlab:matlab.lang.internal.introspective.errorDocCallback('set_turtlebot3_mujoco_scene', 'D:\\final_project\\Sheffield\\for_getting_started\\src\\simple_camera_pid\\matlab\\runtime\\set_turtlebot3_mujoco_scene.m', 61)\" style=\"font-weight:bold\">set_turtlebot3_mujoco_scene<\/a> (<a href=\"matlab: opentoline('D:\\final_project\\Sheffield\\for_getting_started\\src\\simple_camera_pid\\matlab\\runtime\\set_turtlebot3_mujoco_scene.m',61,0)\">line 61<\/a>)\nError in '<a href=\"matlab:open_and_hilite_hyperlink ('visual_line_follower_sac_lyapunov\/MuJoCo_Plant','error')\">visual_line_follower_sac_lyapunov\/MuJoCo_Plant<\/a>': Failed to evaluate mask initialization commands.\n\nError in <a href=\"matlab:matlab.lang.internal.introspective.errorDocCallback('train_sac_lyapunov_agent', 'D:\\final_project\\Sheffield\\for_getting_started\\src\\simple_camera_pid\\matlab\\experiments\\train_sac_lyapunov_agent.m', 141)\" style=\"font-weight:bold\">train_sac_lyapunov_agent<\/a> (<a href=\"matlab: opentoline('D:\\final_project\\Sheffield\\for_getting_started\\src\\simple_camera_pid\\matlab\\experiments\\train_sac_lyapunov_agent.m',141,0)\">line 141<\/a>)\nset_turtlebot3_mujoco_scene(mdl, options.MapKey);\n\nCaused by:\n    Error using <a href=\"matlab:matlab.lang.internal.introspective.errorDocCallback('set_turtlebot3_mujoco_scene', 'D:\\final_project\\Sheffield\\for_getting_started\\src\\simple_camera_pid\\matlab\\runtime\\set_turtlebot3_mujoco_scene.m', 61)\" style=\"font-weight:bold\">set_turtlebot3_mujoco_scene<\/a> (<a href=\"matlab: opentoline('D:\\final_project\\Sheffield\\for_getting_started\\src\\simple_camera_pid\\matlab\\runtime\\set_turtlebot3_mujoco_scene.m',61,0)\">line 61<\/a>)\n    Error executing C++ MEX function 'mj_initbus_mex'. 'MATLABMexHost' with process ID '8036' crashed or was destroyed outside MATLAB."}}
%---
