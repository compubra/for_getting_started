%RUN_LYAPUNOV_1500 One-shot launcher for the 1500-episode Lyapunov SAC training.
%
%   在桌面 MATLAB 命令窗口直接运行（不要通过外部工具阻塞驱动，长训练会超时）：
%       >> cd('D:\final_project\Sheffield\for_getting_started\src\simple_camera_pid\matlab')
%       >> run train/run_lyapunov_1500.m
%
%   1500 回合、每回合 300 步、7 图池随机切、UseFast、无早停、弹 Training Monitor。
%   训练结束会存盘并把 trainInfo 留在基础工作区。

matlabDir = 'D:\final_project\Sheffield\for_getting_started\src\simple_camera_pid\matlab';
cd(matlabDir);
addpath(fullfile(matlabDir,'runtime'));
addpath(fullfile(matlabDir,'experiments'));
addpath(fullfile(matlabDir,'train','shared'));   % rl_dr_defaults / plot_rl_training

logFile = fullfile(matlabDir,'simulation_data','sac_training','train_1500_lyapunov.log');
diary(logFile);
fprintf('=== 1500-ep Lyapunov training | 7-map pool | UseFast | no early stop ===\n');
fprintf('start: %s\n', string(datetime("now")));

DomainRand = rl_dr_defaults();      % GenTrack off, Map(7-pool) on

[trainedAgent, trainInfo] = train_sac_lyapunov_agent( ...
    "DoTrain",            true, ...
    "MaxEpisodes",        1500, ...
    "MaxStepsPerEpisode", 300, ...
    "ScoreWindow",        20, ...
    "SaveReward",         -0.4, ...
    "MapKey",             "track_hard", ...
    "UseFast",            true, ...
    "UseParallel",        false, ...
    "ShowPlot",           true, ...
    "DomainRand",         DomainRand);

assignin('base','trainedAgent', trainedAgent);
assignin('base','trainInfo',    trainInfo);
fprintf('=== COMPLETE: %d episodes | %s ===\n', ...
    numel(trainInfo.EpisodeIndex), string(datetime("now")));
diary off;

% 训练后自动画学习曲线
plot_rl_training(trainInfo);
