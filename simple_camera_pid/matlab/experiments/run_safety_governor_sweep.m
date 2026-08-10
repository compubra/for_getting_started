function run_safety_governor_sweep(options)
%RUN_SAFETY_GOVERNOR_SWEEP 跑完整消融矩阵：{地图} × {四种开关组合}。
%
%   run_safety_governor_sweep
%   run_safety_governor_sweep(Maps=["simple","track_hard"], StopTime=150)
%
% 四种组合对应论文的消融表：
%   baseline      两个都关 —— 原始 PID+残差行为，安全层只统计不干预
%   filter_only   只开约束层
%   search_only   只开丢线扫描
%   both          全开
%
% **必须在能渲染的 MATLAB 进程里跑**，否则视觉全程 found=0，四组结果都是
% 原地扫描、毫无意义。从命令行这样起（见 runtime/control/README.md 末节）：
%
%   cd matlab
%   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 \
%       matlab -batch "addpath(genpath('runtime'));addpath(genpath('train'));addpath('experiments');run_safety_governor_sweep"
%
% 跑完用 compare_safety_governor(mapKey) 出图和指标表（那个不需要渲染）。

arguments
    options.Maps (1, :) string = ["simple", "track_hard"]
    options.StopTime (1, 1) double {mustBePositive} = 150
    options.AgentFile (1, 1) string = ""
    % 见 run_safety_governor_trial 的同名选项：模型默认的 100 拍(5 s)是训练
    % 回合终止条件，会在扫描完成前掐断仿真，令找线功能无法体现
    options.DoneSteps (1, 1) double {mustBePositive} = 400
end

configs = struct( ...
    "label",  {"baseline", "filter_only", "search_only", "both"}, ...
    "filter", {false,      true,          false,         true}, ...
    "search", {false,      false,         true,          true});

total = numel(options.Maps) * numel(configs);
done = 0;
sweepStart = tic;
for mapKey = options.Maps
    for c = configs
        done = done + 1;
        fprintf("\n[%d/%d] map=%s config=%s\n", done, total, mapKey, c.label);
        t0 = tic;
        try
            run_safety_governor_trial(c.label, ...
                StopTime=options.StopTime, MapKey=mapKey, ...
                EnableFilter=c.filter, EnableSearch=c.search, ...
                AgentFile=options.AgentFile, DoneSteps=options.DoneSteps);
        catch err
            fprintf("  TRIAL FAILED: %s | %s\n", err.identifier, err.message);
            for k = 1:numel(err.stack)
                fprintf("    at %s line %d\n", err.stack(k).name, err.stack(k).line);
            end
        end
        fprintf("  (%.1f s wall)\n", toc(t0));
    end
end
fprintf("\nsweep finished in %.1f min\n", toc(sweepStart) / 60);
end
