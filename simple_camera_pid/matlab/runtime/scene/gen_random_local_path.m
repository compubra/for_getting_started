function paths = gen_random_local_path(options)
%GEN_RANDOM_LOCAL_PATH Generate and (optionally) plot smooth random reference paths.
%
%   为循线/跟踪可视化生成随机的平滑参考路径。路径沿 X(前向)单调推进，横向
%   Y 由一串随机控制点做样条平滑得到，可调弯曲难度、长度、点数和随机种子。
%
%   A path is a forward-progressing smooth curve y = f(x): X marches from 0 to
%   Length, Y wanders laterally. "Difficulty" controls how curvy it is.
%
%   Name-value options
%   ------------------
%     Count       (1,1)  条数，画多条对比           default 5
%     NumPoints   (1,1)  每条路径采样点数            default 30  (matches LocalPath_NumPoints)
%     Length      (1,1)  前向总长度 (m)             default 2.0
%     Difficulty  (1,1)  0..1 弯曲难度              default 0.5
%                        0   = 近乎直线
%                        0.5 = 缓弯
%                        1   = S 弯 / 急弯
%     NumControl  (1,1)  随机控制点数（越多越蜿蜒）   default 5
%     MaxLateral  (1,1)  横向最大幅度 (m)           default 0.6
%     Seed               rng 种子，[] = 不设种子      default []
%     Plot        (1,1)  是否画图                    default true
%
%   Returns
%   -------
%     paths : 1xCount struct array, fields:
%             .X          NumPoints x 1  前向坐标 (m)
%             .Y          NumPoints x 1  横向坐标 (m)
%             .Curvature  NumPoints x 1  每点曲率 (1/m)
%             .Heading    NumPoints x 1  航向角 (rad)
%             .Coeffs     二次拟合 [a b c]（与 fitQuadraticLookahead 同约定）
%
%   Examples
%   --------
%     gen_random_local_path();                                  % 5 条默认路径，画图
%     gen_random_local_path("Count", 8, "Difficulty", 0.9);     % 8 条急弯
%     p = gen_random_local_path("Seed", 42, "Plot", false);     % 可复现，不画图
%
%   See also ORIGINBOT_LOCAL_PATH_GENERATOR.

arguments
    options.Count      (1,1) double {mustBePositive}    = 5
    options.NumPoints  (1,1) double {mustBePositive}    = 30
    options.Length     (1,1) double {mustBePositive}    = 2.0
    options.Difficulty (1,1) double {mustBeGreaterThanOrEqual(options.Difficulty,0), mustBeLessThanOrEqual(options.Difficulty,1)} = 0.5
    options.NumControl (1,1) double {mustBePositive}    = 5
    options.MaxLateral (1,1) double {mustBePositive}    = 0.6
    options.Seed                                        = []
    options.Plot       (1,1) logical                    = true
end

if ~isempty(options.Seed)
    rng(options.Seed);
end

n   = round(options.NumPoints);
nc  = max(2, round(options.NumControl));
% 弯曲幅度：难度 0→很小，难度 1→满幅。用平方让低难度更平缓。
amp = options.MaxLateral * (0.15 + 0.85 * options.Difficulty^2);

paths = struct("X", {}, "Y", {}, "Curvature", {}, "Heading", {}, "Coeffs", {});

for i = 1:round(options.Count)
    X = linspace(0, options.Length, n).';

    % 随机控制点：起点横向 0（机器人当前对准路径），其余为独立随机横向偏移，
    % 幅度由 amp 决定，最后整体缩放使最大偏移恰好命中目标幅度，保证不同难度
    % 视觉上真的分得开（避免随机游走自相抵消导致路径近乎笔直）。
    xc = linspace(0, options.Length, nc).';
    yc = [0; (2*rand(nc-1, 1) - 1) * amp];
    peak = max(abs(yc));
    if peak > eps
        yc = yc * (amp / peak);   % 规范化到目标幅度
    end
    yc = max(-options.MaxLateral, min(options.MaxLateral, yc));

    % 平滑：对控制点做样条插值到 n 个采样点
    Y = spline(xc, yc, X);

    % 一阶/二阶导（对 X）。X 均匀分布，用中心差分。
    yp  = gradient(Y, X);          % dy/dx
    ypp = gradient(yp, X);         % d2y/dx2
    heading   = atan(yp);
    % 曲率 κ = y'' / (1 + y'^2)^{3/2}
    curvature = ypp ./ max(eps, (1 + yp.^2).^1.5);

    % 二次拟合（与 originbot fitQuadraticLookahead 同约定 y=a x^2+b x+c）
    coeffs = polyfit(X, Y, 2);

    paths(i) = struct("X", X, "Y", Y, "Curvature", curvature, ...
        "Heading", heading, "Coeffs", coeffs);
end

if options.Plot
    plotPaths(paths, options);
end
end


% ═══════════════════════════════════════════════════════════════════════════
function plotPaths(paths, options)
cnt = numel(paths);
cols = lines(max(cnt, 1));

fig = figure("Name", "Random local reference paths", "Color", "w");
tiledlayout(fig, 2, 1, "TileSpacing", "compact", "Padding", "compact");

% ── 上图：路径 (X 前向, Y 横向) ──
nexttile
hold on
for i = 1:cnt
    plot(paths(i).X, paths(i).Y, "-", "LineWidth", 1.8, "Color", cols(i,:), ...
        "DisplayName", sprintf("path %d", i));
end
plot([0 options.Length], [0 0], "k--", "LineWidth", 0.5, ...
    "HandleVisibility", "off");   % 参考中线
hold off
axis equal
grid on
xlabel("X forward (m)"); ylabel("Y lateral (m)");
title(sprintf("%d random paths | difficulty=%.2f | length=%.2g m", ...
    cnt, options.Difficulty, options.Length));
legend("Location", "eastoutside");

% ── 下图：曲率沿路径 ──
nexttile
hold on
for i = 1:cnt
    plot(paths(i).X, paths(i).Curvature, "-", "LineWidth", 1.2, ...
        "Color", cols(i,:), "HandleVisibility", "off");
end
hold off
grid on
xlabel("X forward (m)"); ylabel("curvature (1/m)");
title("Path curvature");
end
