function compare_safety_governor(mapKey, options)
%COMPARE_SAFETY_GOVERNOR 安全层/找线功能的消融对比：轨迹图 + 指标表。
%
%   compare_safety_governor("simple")
%   compare_safety_governor("track_hard", OutputDir="/tmp/figs")
%
% 读 simulation_data/safety_governor/ 下由 run_safety_governor_trial.m 存的
% .mat（每个 label 取最新一次），画三张图并打印指标表：
%
%   1) 轨迹对比    赛道贴图为底 + 真值中心线 + 各配置的小车轨迹
%   2) 横向误差    相对中心线的真值横向偏差随时间
%   3) 安全诊断    屏障函数 h(t) 与找线状态机状态
%
% 本脚本**不跑仿真**，只做离线分析，所以不需要 MuJoCo 渲染，在普通 MATLAB
% 会话里就能跑。

arguments
    mapKey (1, 1) string
    options.Labels (1, :) string = ["baseline", "filter_only", "search_only", "both"]
    options.OutputDir (1, 1) string = ""
end

matlabDir = fileparts(fileparts(mfilename("fullpath")));
addpath(genpath(fullfile(matlabDir, "runtime")));
dataDir = fullfile(matlabDir, "simulation_data", "safety_governor");
outDir = options.OutputDir;
if strlength(outDir) == 0
    outDir = fullfile(dataDir, "figures");
end
if ~isfolder(outDir), mkdir(outDir); end

% ── 载入每个 label 的最新一次运行 ──────────────────────────────────
trials = struct("label", {}, "metrics", {}, "parameters", {}, "centerline", {});
for label = options.Labels
    pattern = fullfile(dataDir, mapKey + "_" + label + "_*.mat");
    files = dir(pattern);
    if isempty(files)
        warning("Compare:Missing", "no trial found for '%s' (%s)", label, pattern);
        continue
    end
    [~, newest] = max([files.datenum]);
    S = load(fullfile(files(newest).folder, files(newest).name), ...
        "metrics", "parameters", "centerline");
    trials(end + 1) = struct("label", label, "metrics", S.metrics, ...
        "parameters", S.parameters, "centerline", S.centerline); %#ok<AGROW>
end
if isempty(trials)
    error("Compare:NoTrials", "no trials found for map '%s'", mapKey);
end

printTable(mapKey, trials);

colors = lines(numel(trials));
centerline = trials(1).centerline;

% ── 图 1：轨迹对比（赛道贴图为底）────────────────────────────────
f1 = figure("Visible", "off", "Position", [100 100 900 820]);
ax = axes(f1);
hold(ax, "on");
drawTrackBackground(ax, trials(1).parameters.SceneFile);
plot(ax, centerline([1:end, 1], 1), centerline([1:end, 1], 2), "--", ...
    "Color", [0 0.85 0.3], "LineWidth", 1.8, "DisplayName", "赛道中心线(真值)");
for k = 1:numel(trials)
    xy = trials(k).metrics.trajectory;
    plot(ax, xy(:, 1), xy(:, 2), "-", "Color", colors(k, :), "LineWidth", 2.0, ...
        "DisplayName", sprintf("%s (%.1fs, %.2fm)", trials(k).label, ...
        trials(k).metrics.CompletedTime, trials(k).metrics.PathLength));
    plot(ax, xy(end, 1), xy(end, 2), "o", "Color", colors(k, :), ...
        "MarkerFaceColor", colors(k, :), "MarkerSize", 8, "HandleVisibility", "off");
end
plot(ax, centerline(1, 1), centerline(1, 2), "p", "MarkerSize", 1, ...
    "HandleVisibility", "off");
axis(ax, "equal"); grid(ax, "on");
xlim(ax, [-2.3 2.3]); ylim(ax, [-2.3 2.3]);
xlabel(ax, "x (m)"); ylabel(ax, "y (m)");
title(ax, sprintf("轨迹对比 — 地图 %s（实心点=终点）", mapKey), "Interpreter", "none");
legend(ax, "Location", "northeastoutside", "Interpreter", "none");
f1File = fullfile(outDir, mapKey + "_trajectories.png");
exportgraphics(f1, f1File, "Resolution", 150);
close(f1);

% ── 图 2：真值横向误差 ───────────────────────────────────────────
f2 = figure("Visible", "off", "Position", [100 100 1000 520]);
ax = axes(f2); hold(ax, "on");
for k = 1:numel(trials)
    m = trials(k).metrics;
    plot(ax, m.time, 100 * m.trueLateral, "-", "Color", colors(k, :), ...
        "LineWidth", 1.3, "DisplayName", trials(k).label);
end
yline(ax, 0, "k-", "HandleVisibility", "off");
grid(ax, "on");
xlabel(ax, "时间 (s)"); ylabel(ax, "相对中心线横向偏差 (cm)");
title(ax, sprintf("真值循迹误差 — 地图 %s", mapKey), "Interpreter", "none");
legend(ax, "Location", "best", "Interpreter", "none");
f2File = fullfile(outDir, mapKey + "_lateral_error.png");
exportgraphics(f2, f2File, "Resolution", 150);
close(f2);

% ── 图 3：屏障函数与找线状态 ─────────────────────────────────────
f3 = figure("Visible", "off", "Position", [100 100 1000 720]);
tl = tiledlayout(f3, 2, 1, "TileSpacing", "compact");
axA = nexttile(tl); hold(axA, "on");
for k = 1:numel(trials)
    m = trials(k).metrics;
    plot(axA, m.tDiag, m.barrier, "-", "Color", colors(k, :), "LineWidth", 1.2, ...
        "DisplayName", trials(k).label);
end
yline(axA, 0, "r--", "LineWidth", 1.5, "DisplayName", "安全边界 h=0");
grid(axA, "on"); ylabel(axA, "屏障函数 h");
title(axA, sprintf("CBF 屏障函数（h<0 即越界）— 地图 %s", mapKey), "Interpreter", "none");
legend(axA, "Location", "best", "Interpreter", "none");

axB = nexttile(tl); hold(axB, "on");
for k = 1:numel(trials)
    m = trials(k).metrics;
    stairs(axB, m.tDiag, m.state, "-", "Color", colors(k, :), "LineWidth", 1.4, ...
        "DisplayName", trials(k).label);
end
grid(axB, "on");
yticks(axB, 0:4);
yticklabels(axB, ["TRACK", "HOLD", "BRAKE", "SCAN", "GIVEUP"]);
ylim(axB, [-0.3 4.3]);
xlabel(axB, "时间 (s)"); ylabel(axB, "找线状态");
legend(axB, "Location", "best", "Interpreter", "none");
f3File = fullfile(outDir, mapKey + "_safety_diag.png");
exportgraphics(f3, f3File, "Resolution", 150);
close(f3);

fprintf("\n图已保存:\n  %s\n  %s\n  %s\n", f1File, f2File, f3File);
end

% =====================================================================
function drawTrackBackground(ax, sceneFile)
% 把赛道贴图按世界坐标铺在底图上（贴图覆盖 4.4x4.4 m 地面，见
% get_track_centerline 的同一约定）。找不到贴图就跳过，不影响其余绘图。
try
    scene = fileread(char(sceneFile));
    tok = regexp(scene, '<texture[^>]*file="([^"]+)"', "tokens", "once");
    dirTok = regexp(scene, 'texturedir="([^"]+)"', "tokens", "once");
    if isempty(tok), return; end
    pngFile = fullfile(fileparts(char(sceneFile)), char(dirTok{1}), char(tok{1}));
    if ~isfile(pngFile), return; end
    img = imread(pngFile);
    % MuJoCo 贴图行序与世界 y 轴相反，翻转后按 [-2.2, 2.2] 铺开
    image(ax, "XData", [-2.2 2.2], "YData", [-2.2 2.2], "CData", flipud(img));
    set(ax, "YDir", "normal");
catch
    % 贴图读不到只是少个背景，不该让整张图失败
end
end

% =====================================================================
function printTable(mapKey, trials)
rows = {
    "存活时间 (s)",           "CompletedTime",       "%8.2f"
    "路径长度 (m)",           "PathLength",          "%8.2f"
    "完成整圈",               "LapCompleted",        "%8d"
    "真值横向误差 平均 (cm)", "TrueLateralMeanAbs",  "%8.2f"
    "真值横向误差 RMS (cm)",  "TrueLateralRMS",      "%8.2f"
    "真值横向误差 最大 (cm)", "TrueLateralMax",      "%8.2f"
    "真值航向误差 RMS (deg)", "TrueHeadingRMS",      "%8.2f"
    "视觉 found 率 (%)",      "FoundRatio",          "%8.2f"
    "最长丢线 (s)",           "LongestLineLoss",     "%8.2f"
    "丢线次数",               "LostLineEpisodes",    "%8d"
    "约束违反占比 (%)",       "ViolationRatio",      "%8.2f"
    "最小屏障 h",             "MinBarrier",          "%8.4f"
    "CBF 介入占比 (%)",       "CBFInterveneRatio",   "%8.2f"
    "CBF 平均介入 (rad/s)",   "CBFInterveneMean",    "%8.4f"
    "CBF 最大介入 (rad/s)",   "CBFInterveneMax",     "%8.4f"
    "约束不可行占比 (%)",     "InfeasibleRatio",     "%8.2f"
    "SCAN 时间占比 (%)",      "TimeScan",            "%8.2f"
    "GIVEUP 时间占比 (%)",    "TimeGiveUp",          "%8.2f"
    "重新捕获次数",           "Reacquisitions",      "%8d"
    "omega RMS (rad/s)",      "OmegaRMS",            "%8.3f"
    "角加速度 RMS (rad/s^2)", "OmegaRateRMS",        "%8.3f"
    "轮速饱和占比 (%)",       "WheelSatRatio",       "%8.2f"
    };
% 需要换算的字段：比例 -> 百分比，米 -> 厘米，弧度 -> 度
pct = ["FoundRatio", "ViolationRatio", "CBFInterveneRatio", "InfeasibleRatio", ...
       "TimeScan", "TimeGiveUp", "WheelSatRatio"];
cm  = ["TrueLateralMeanAbs", "TrueLateralRMS", "TrueLateralMax"];
deg = "TrueHeadingRMS";

% repmat 的第一参必须是 char，用 string 会得到字符串数组、被 fprintf 逐元素
% 换行打印成一列横杠
width = 30 + 13 * numel(trials);
fprintf("\n%s\n 地图 %s：安全层消融对比\n%s\n", repmat('=', 1, width), mapKey, repmat('=', 1, width));
fprintf("%-30s", "指标");
for k = 1:numel(trials), fprintf("%13s", trials(k).label); end
fprintf("\n%s\n", repmat('-', 1, width));
for r = 1:size(rows, 1)
    % 中文标签在 fprintf 里按字节计宽，%-30s 对不齐，用 pad 按字符数补
    fprintf("%s", pad(string(rows{r, 1}), 30, "right"));
    for k = 1:numel(trials)
        v = trials(k).metrics.(rows{r, 2});
        if ismember(rows{r, 2}, pct), v = 100 * v; end
        if ismember(rows{r, 2}, cm),  v = 100 * v; end
        if rows{r, 2} == deg,         v = rad2deg(v); end
        if isnan(v)
            fprintf("%13s", "-");
        else
            fprintf("%13s", sprintf(rows{r, 3}, v));
        end
    end
    fprintf("\n");
end
fprintf("%s\n", repmat('=', 1, width));
end
