function fig = plot_rl_training(src)
%PLOT_RL_TRAINING Plot residual-RL (SAC/PPO) training curves after training.
%
%   训练时若关闭了 Training Monitor（ShowPlot=false / Plots="none"），用本
%   函数在训练结束后画学习曲线。SAC 与 PPO 训练共用：数据来自 train() 返回
%   的统计对象 trainInfo，或训练自动保存的 *_agent_*.mat 文件。
%
%   用法 / Usage
%   ------------
%     % 直接用工作区里的 trainInfo（脚本/函数训练完就有）：
%     plot_rl_training(trainInfo)
%
%     % 或从保存的 .mat 文件读取（含 trainInfo 变量）：
%     plot_rl_training("simulation_data/sac_training/sac_agent_20260630_012421.mat")
%     plot_rl_training("simulation_data/ppo_training/ppo_agent_20260709_120000.mat")
%
%     % 不传参数时，自动在 simulation_data 的 sac_training / ppo_training
%     % 两个目录里找最新的 agent .mat：
%     plot_rl_training()
%
%   画出 / Plots
%   -----------
%     上图：每回合奖励 (EpisodeReward) + 滑动平均 (AverageReward)
%     下图：每回合初始 Q 值估计 (EpisodeQ0) —— 反映 critic 价值估计的走势
%
%   See also TRAIN_SAC_RESIDUAL_LIVE, TRAIN_PPO_RESIDUAL_LIVE.

% ── 1. 解析数据来源 ────────────────────────────────────────────────────────
if nargin == 0
    src = latestAgentMat();
end

if isa(src, "rl.train.result.rlTrainingResult")
    info = src;                                   % 直接传对象
elseif (isstring(src) || ischar(src)) && isfile(src)
    S = load(char(src), "trainInfo");             % 从 .mat 读
    assert(isfield(S, "trainInfo"), ...
        "PlotRL:NoTrainInfo", "File has no 'trainInfo': %s", src);
    info = S.trainInfo;
else
    error("PlotRL:BadInput", ...
        "Pass a trainInfo object, a .mat path, or nothing (auto-find latest).");
end

% ── 2. 取数据 ──────────────────────────────────────────────────────────────
ep     = double(info.EpisodeIndex);
reward = double(info.EpisodeReward);
avg    = double(info.AverageReward);
q0     = double(info.EpisodeQ0);

% ── 3. 画图 ────────────────────────────────────────────────────────────────
fig = figure("Name", "Residual RL training", "Color", "w");
tiledlayout(fig, 2, 1, "TileSpacing", "compact", "Padding", "compact");

nexttile
plot(ep, reward, "-", "Color", [0.7 0.7 0.9], "DisplayName", "Episode reward");
hold on
plot(ep, avg, "-", "LineWidth", 2, "Color", [0.10 0.30 0.75], ...
    "DisplayName", "Average reward");
hold off
grid on
xlabel("Episode"); ylabel("Reward");
title(sprintf("Learning curve  (final avg = %.3f over %d episodes)", ...
    avg(end), ep(end)));
legend("Location", "best");

nexttile
plot(ep, q0, "-", "LineWidth", 1.2, "Color", [0.75 0.35 0.10]);
grid on
xlabel("Episode"); ylabel("Episode Q_0");
title("Critic initial-state value estimate (Q_0)");
end


% ─── helper：在 SAC/PPO 两个保存目录里找最新的 agent .mat ──────────────────
function f = latestAgentMat()
here    = fileparts(mfilename("fullpath"));           % .../matlab/train/shared
dataDir = fullfile(here, "..", "..", "simulation_data");
saveDirs = [
    fullfile(dataDir, "sac_training")
    fullfile(dataDir, "ppo_training")
    ];
listing = [];
for k = 1:numel(saveDirs)
    listing = [listing; dir(fullfile(saveDirs(k), "*agent_*.mat"))]; %#ok<AGROW>
end
assert(~isempty(listing), "PlotRL:NoFiles", ...
    "No agent .mat found under %s (sac_training / ppo_training). " + ...
    "Train first, or pass a file/trainInfo.", dataDir);
[~, idx] = max([listing.datenum]);
f = fullfile(listing(idx).folder, listing(idx).name);
fprintf("plot_rl_training: using latest %s\n", listing(idx).name);
end
