function cfg = ppo_training_config()
%PPO_TRAINING_CONFIG Central tuning interface for PPO residual RL training.
%
%   PPO 残差训练的参数中心：训练入口是同目录的 train_ppo_residual_live.m，
%   它在开头调用本函数读取全部可调参数。接手训练时只改这个文件，然后从
%   train/ 目录运行 live 脚本即可。
%   This is the ONE file you edit to tune PPO training. The entry point is
%   train/train_ppo_residual_live.m, which loads this config at the top.
%
%   与 sac_training_config 的结构一致，仅第 4 组（agent 超参数）不同——
%   PPO 是 on-policy，超参数是 horizon/clip/GAE 一族而非回放缓冲一族。
%
%   注意：本函数引用同目录的 rl_dr_defaults，live 脚本会先把 train/ 加入
%   路径；单独调用本函数前请自行 addpath 本目录（train/）。
%
%   See also TRAIN_PPO_RESIDUAL_LIVE, SAC_TRAINING_CONFIG, RL_DR_DEFAULTS.

% ════════════════════════════════════════════════════════════════════════
%  0) 模型 / Model
% ════════════════════════════════════════════════════════════════════════
cfg.ModelName      = "visual_line_follower_ppo_residual";
% RL Agent 块在模型内的相对路径（一般不用改）
cfg.AgentBlockPath = "PPO_Residual_Controller/PPO_Agent";

% ════════════════════════════════════════════════════════════════════════
%  1) 训练流程 / Training loop
% ════════════════════════════════════════════════════════════════════════
cfg.MaxEpisodes        = 800;    % 训练回合数上限
cfg.MaxStepsPerEpisode = 200;    % 每回合最大步数 (200*0.05s = 10s/回合)
% 提前停止：平均回报达到 StopReward 即停。设 Inf = 不提前停止，跑满
% MaxEpisodes，得到完整学习曲线。
cfg.StopReward         = Inf;
% 检查点存档：回合奖励高于此值就把 agent 自动存进 SaveDir，训练中途崩溃
% 也不至于全部丢失。按当前奖励尺度（好回合约 -15 ~ -30）默认 -30 表示
% "较好的回合都会存档"；设很大的值（如 1e5）可关闭存档。
cfg.SaveReward         = -30;
cfg.ScoreWindow        = 20;     % 计算平均回报的滑动窗口长度
cfg.UseFast            = true;   % true = 关闭 MuJoCo 渲染，加速训练
cfg.UseParallel        = false;  % true = 并行训练 (需 Parallel Computing Toolbox)
cfg.NumWorkers         = 2;      % 并行 worker 数（仅 UseParallel=true 时生效；
                                 % 每个 worker 约需 2 GB 内存，量力而行）
cfg.ShowPlot           = false;  % false = 不弹 Training Monitor（训练后用
                                 % plot_rl_training 画曲线）

% ════════════════════════════════════════════════════════════════════════
%  2) 地图 / Map
% ════════════════════════════════════════════════════════════════════════
%   可用 key：simple / complex / ellipse / training / track_easy /
%   track_medium / track_hard / winding，也可填完整场景 XML 路径。
cfg.MapKey = "track_hard";

% ════════════════════════════════════════════════════════════════════════
%  3) 奖励函数权重 / Reward weights
%     reward(t) = -(Q*e^2 + R*u^2) - P_Lost*(1-found)
%   e 是视觉 steering_error（纯车载信号，训练端不依赖任何位姿——真车
%   微调时同一套奖励直接可用，不需要外部定位）。
%   写入模型工作区变量 PPO_Q / PPO_R / PPO_P_Lost / PPO_Done_Steps，
%   模型里的 Reward_Calculation / Done_Detection 直接引用它们。
% ════════════════════════════════════════════════════════════════════════
cfg.Reward.Q         = 1.0;    % 循线误差惩罚。调大→更贴线（可能更激进）
cfg.Reward.R         = 0.1;    % 动作能耗惩罚。调大→动作更平滑保守
cfg.Reward.P_Lost    = 10.0;   % 丢线惩罚。调大→更强烈避免丢线
cfg.Reward.DoneSteps = 100;    % 连续丢线多少步判回合结束 (100*0.05s = 5s)

% ════════════════════════════════════════════════════════════════════════
%  4) PPO 智能体超参数 / PPO agent hyper-parameters
% ════════════════════════════════════════════════════════════════════════
cfg.Agent.ActorLearnRate    = 3e-4;  % Actor 学习率
cfg.Agent.CriticLearnRate   = 3e-4;  % Critic 学习率
cfg.Agent.DiscountFactor    = 0.99;  % 折扣因子 gamma
cfg.Agent.ExperienceHorizon = 512;   % 每次策略更新前收集的 on-policy 步数
cfg.Agent.MiniBatchSize     = 128;   % PPO 每个 epoch 的 mini-batch 大小
cfg.Agent.NumEpoch          = 3;     % 每批经验重复优化次数
cfg.Agent.ClipFactor        = 0.2;   % PPO clipping 系数
cfg.Agent.EntropyLossWeight = 0.01;  % 熵损失权重。调大→更多探索
cfg.Agent.GAEFactor         = 0.95;  % GAE(lambda) 优势估计系数
cfg.Agent.MaxDeltaOmega     = 1.0;   % 残差动作上下界 |delta_omega| (rad/s)

% ════════════════════════════════════════════════════════════════════════
%  5) 域随机化 / Domain randomization (per-episode ResetFcn)
% ════════════════════════════════════════════════════════════════════════
%   每回合重置时随机扰动起始位姿 / 动力学 / 速度PID / 感知参数。
%   范围与分组开关见同目录 rl_dr_defaults；整体关闭设 UseDomainRand=false。
cfg.UseDomainRand = true;
cfg.DomainRand    = rl_dr_defaults();
% GenTrack 生成的场景文件基名与 SAC 训练区分开，避免并发训练互相覆盖。
cfg.DomainRand.GenTrack.Name = "ppo_gen_episode";
% 关地图随机化：每回合固定用 cfg.MapKey。开启换图（true）会强制关闭
% Fast Restart（live 脚本自动处理），每回合完整重编译、速度差别不大，
% 但注意 MuJoCo + 长时间训练存在偶发原生崩溃的风险，靠 SaveReward 存档兜底。
cfg.DomainRand.Map.Enable = false;

% ════════════════════════════════════════════════════════════════════════
%  6) 输出 / Output
% ════════════════════════════════════════════════════════════════════════
%   留空 "" 表示用默认目录 simulation_data/ppo_training。
cfg.SaveDir = "";

end
