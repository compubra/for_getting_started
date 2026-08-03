function cfg = sac_training_config()
%SAC_TRAINING_CONFIG Central tuning interface for SAC residual RL training.
%
%   SAC 残差训练的参数中心：训练入口是同目录的 train_sac_residual_live.m，
%   它在开头调用本函数读取全部可调参数。接手训练时只改这个文件，然后从
%   train/sac/ 目录运行 live 脚本即可。
%   This is the ONE file you edit to tune SAC training. The entry point is
%   train/sac/train_sac_residual_live.m, which loads this config at the top.
%   (The old function-style entry run_sac_training / train_sac_residual_agent
%   has been removed — the live script is now the single entry.)
%
%   注意：本函数引用 train/shared/rl_dr_defaults，live 脚本会先把
%   train/shared/ 加入路径；单独调用本函数前请自行 addpath("../shared")。
%
%   See also TRAIN_SAC_RESIDUAL_LIVE, RL_DR_DEFAULTS.
%   (PPO 线已于 2026-07-19 归档至 evaluation/archive_ppo/。)

% ════════════════════════════════════════════════════════════════════════
%  0) 模型 / Model
% ════════════════════════════════════════════════════════════════════════
cfg.ModelName      = "visual_line_follower_sac_residual";
% RL Agent 块在模型内的相对路径（一般不用改）
cfg.AgentBlockPath = "SAC_Residual_Controller/SAC_Agent";

% ════════════════════════════════════════════════════════════════════════
%  1) 训练 / 仅评估开关 / Train-vs-eval-only switch
% ════════════════════════════════════════════════════════════════════════
%   true  = 训练模式：从头构建一个新 SAC agent 并调用 train()（原有行为）。
%   false = 仅评估模式：跳过 agent/环境/训练选项的构建，直接把一个已经训练
%           好的 agent 载入 SAC_Agent 块，然后 sim(model) 跑一次仿真评估——
%           即"只调用模型，不训练"。
%   两个训练入口 train_sac_residual_live.m / train_sac_residual_script.m
%   都读这个开关，行为完全一致。
cfg.TrainAgent    = false;   % true=训练, false=仅调用已训练 agent 做仿真评估
% 仅在 cfg.TrainAgent=false 时使用：留空 "" 表示自动挑 7) 节 cfg.SaveDir（或
% 其默认值 simulation_data/sac_training）下最新的 sac_agent_*.mat；也可以填
% 一个具体 .mat 的绝对路径，指定评估某一次存档而不是最新的。
cfg.EvalAgentFile = "";

% ════════════════════════════════════════════════════════════════════════
%  2) 训练流程 / Training loop
% ════════════════════════════════════════════════════════════════════════
cfg.MaxEpisodes        = 500;    % 训练回合数上限
cfg.MaxStepsPerEpisode = 200;    % 每回合最大步数 (200*0.05s = 10s/回合)
% 提前停止：平均回报达到 StopReward 即停。设 Inf = 不提前停止，跑满
% MaxEpisodes，得到完整学习曲线。
cfg.StopReward         = Inf;
% 检查点存档：回合奖励高于此值就把 agent 自动存进 SaveDir，训练中途崩溃
% 也不至于全部丢失。回合步数 400→200 后奖励尺度减半（好回合约 -7 ~ -15），
% 阈值随之从 -30 调到 -15，仍表示"较好的回合都会存档"；设很大的值可关闭存档。
% 只在 SaveEveryNEpisodes 为空/0（即不用按固定回合数存档）时生效。
cfg.SaveReward         = -15;
% 按固定回合数存档（优先于 SaveReward）：每隔这么多回合存一次，不管奖励高低。
% 500 回合 / 50 = 最多 10 个检查点，比奖励阈值宽松时可能存出上百个快照
% （之前 7/20 那次训练教训：SaveReward 相对奖励量级偏松，165 个中途快照吃了
% 6.7GB 磁盘）更可控。设 0 或 [] 则退回用 SaveReward 阈值判断。
cfg.SaveEveryNEpisodes = 100;
cfg.ScoreWindow        = 20;     % 计算平均回报的滑动窗口长度
cfg.UseFast            = true;   % true = 关闭 MuJoCo 渲染，加速训练
cfg.UseParallel        = false;  % true = 并行训练 (需 Parallel Computing Toolbox)
cfg.NumWorkers         = 2;      % 并行 worker 数（仅 UseParallel=true 时生效；
                                 % 每个 worker 约需 2 GB 内存，量力而行）
cfg.ShowPlot           = false;  % false = 不弹 Training Monitor（训练后用
                                 % plot_rl_training 画曲线）

% ════════════════════════════════════════════════════════════════════════
%  3) 地图 / Map
% ════════════════════════════════════════════════════════════════════════
%   可用 key：simple / complex / ellipse / training / track_easy /
%   track_medium / track_hard / winding，也可填完整场景 XML 路径。
cfg.MapKey = "simple";

% ════════════════════════════════════════════════════════════════════════
%  4) 奖励函数权重 / Reward weights
%     reward(t) = -(Q*e^2 + Q_Lateral*e_lat^2 + Q_Heading*e_head^2 + R*||u||^2)
%                 - P_Lost*(1-found)
%   e 是视觉 steering_error，e_lat/e_head 是同一视觉子系统输出的
%   lateral_error/heading_error（均为纯车载信号，训练端不依赖任何位姿——
%   真车微调时同一套奖励直接可用，不需要外部定位）。e_lat/e_head 是在
%   steering_error 之外新增的整形项（2026-07-23 起），steering_error 本身
%   已是两者的线性组合，这两项权重刻意设得比 Q 低，避免重复计权压制转向
%   误差信号。u 是 2 维动作 [delta_v; delta_omega]（2026-07-21 起，见 5) 节
%   MaxDeltaV；两个分量分别跟 PID 的线速度/角速度指令相加后，在共享子系统
%   Diff_Drive_Kinematics 里统一转轮速，SAC 内部不再单独转一遍），
%   ||u||^2 是两个分量的平方和，R 惩罚的是两个残差自由度的总能耗，
%   不只是转向。
%   写入模型工作区变量 SAC_Q / SAC_Q_Lateral / SAC_Q_Heading / SAC_R /
%   SAC_P_Lost / SAC_Done_Steps，模型里的 Reward_Calculation /
%   Done_Detection 直接引用它们。
% ════════════════════════════════════════════════════════════════════════
cfg.Reward.Q         = 1.0;    % 循线误差惩罚。调大→更贴线（可能更激进）
cfg.Reward.Q_Lateral = 0.5;    % 横向误差惩罚（整形项）
cfg.Reward.Q_Heading = 0.5;    % 航向误差惩罚（整形项）
cfg.Reward.R         = 0.1;    % 动作能耗惩罚。调大→动作更平滑保守
cfg.Reward.P_Lost    = 10.0;   % 丢线惩罚。调大→更强烈避免丢线
cfg.Reward.DoneSteps = 100;    % 连续丢线多少步判回合结束 (100*0.05s = 5s)

% ════════════════════════════════════════════════════════════════════════
%  5) SAC 智能体超参数 / SAC agent hyper-parameters
% ════════════════════════════════════════════════════════════════════════
cfg.Agent.ActorLearnRate  = 3e-4;   % Actor 学习率
cfg.Agent.CriticLearnRate = 3e-4;   % Critic 学习率 (两个 critic 共用)
cfg.Agent.DiscountFactor  = 0.99;   % 折扣因子 gamma
cfg.Agent.MiniBatchSize   = 256;    % 每次更新采样的经验数
cfg.Agent.BufferLength    = 2e5;    % 经验回放缓冲区大小
cfg.Agent.EntropyWeight   = 0.2;    % 熵权重 (探索强度)。调大→更多探索
cfg.Agent.TargetSmooth    = 5e-3;   % 目标网络软更新系数 tau
cfg.Agent.MaxDeltaOmega   = 1.0;    % 残差动作上下界 |delta_omega| (rad/s)
cfg.Agent.MaxDeltaV       = 1.0;    % 线速度补偿动作上下界 |delta_v| (rad/s,
                                     % 轮速量纲，等量加到左右轮速指令上)

% ════════════════════════════════════════════════════════════════════════
%  6) 域随机化 / Domain randomization (per-episode ResetFcn)
% ════════════════════════════════════════════════════════════════════════
%   每回合重置时随机扰动起始位姿 / 动力学 / 速度PID / 感知参数。
%   范围与分组开关见 train/shared/rl_dr_defaults；整体关闭设 UseDomainRand=false。
%   仅在 cfg.TrainAgent=true 时生效——仅评估模式不重建环境/ResetFcn。
cfg.UseDomainRand = true;
cfg.DomainRand    = rl_dr_defaults();
% 关地图随机化：每回合固定用 cfg.MapKey。开启换图（true）会强制关闭
% Fast Restart（live 脚本自动处理），每回合完整重编译、速度差别不大，
% 但注意 MuJoCo + 长时间训练存在偶发原生崩溃的风险，靠 SaveReward 存档兜底。
cfg.DomainRand.Map.Enable = false;

% ════════════════════════════════════════════════════════════════════════
%  7) 输出 / Output
% ════════════════════════════════════════════════════════════════════════
%   留空 "" 表示用默认目录 simulation_data/sac_training。训练模式下新 agent
%   存到这里；仅评估模式（cfg.TrainAgent=false 且 cfg.EvalAgentFile=""）默认
%   也从这里挑最新的 sac_agent_*.mat。
cfg.SaveDir = "";

end
