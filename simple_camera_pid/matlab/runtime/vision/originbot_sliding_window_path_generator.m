function result = originbot_sliding_window_path_generator(rgbVector, ...
    roiBottomFraction, waypointCount, minBrightness, maxSaturation, ...
    minPixels, errorScale, lookaheadDistance, lateralGain, headingGain, ...
    curvatureGain, platform)
%ORIGINBOT_SLIDING_WINDOW_PATH_GENERATOR 滑窗+多项式拟合+霍夫变换循线算法（方案 B）。
%
% MuJoCo/Gazebo/real 三个平台共用本文件——不再有 "_gazebo" 后缀的姐妹文件。
% platform（'mujoco'|'gazebo'|'real'，2026-08-03 起必填第 12 参）显式选择
% originbot_camera_profile 里的相机/像素尺度常数（该函数是三平台参数的唯一
% 真相源，本文件与调试可视化 originbot_line_follower_debug_frame 都调用它，
% 不再各自维护硬编码副本）；算法主体（霍夫种子、滑动窗口、多项式拟合、
% 丢线分级）完全共享、单一实现。
%
% platform 由 numel(rgbVector) 自动识别改为显式传参：三平台分辨率于
% 2026-08-03 统一为 640x480 后，像素总数已无法区分平台，靠图像尺寸推断不再
% 可靠——现在从模型工作区变量传入字面常量（每个模型文件固定对应一个平台，
% 不会在运行时变化）。
%
% 模型 MATLABFcn 表达式调用方式（各平台模型文件在末尾追加各自的 platform 字面量）：
%   originbot_sliding_window_path_generator(u, LocalPath_ROIFraction, ...
%       LocalPath_NumPoints, OriginBot_MinBrightness, OriginBot_MaxSaturation, ...
%       OriginBot_MinPixels, OriginBot_ErrorScale, LocalPath_LookaheadDistance, ...
%       LocalPath_LateralGain, LocalPath_HeadingGain, LocalPath_CurvatureGain, ...
%       'mujoco')   % 或各自模型对应的 'gazebo' / 'real'
%
% 管线（三项技术的分工）：
%   1) 霍夫变换   ROI 白线掩码上取主导直线段 → 滑窗起始列 + 列漂移斜率预测。
%   2) 滑动窗口   从 ROI 底部向上叠 N 个窗（N=waypointCount），每窗取白线像素
%                 质心并重定中下一窗；空窗按最近漂移速率外推。按行分层，
%                 天然保证每个前向距离一个点（单值），为多项式拟合备好点列。
%   3) 多项式拟合 窗心 IPM 投影到地面坐标后拟合 Y = a·X² + b·X + c，前视点取
%                 多项式上 X=lookahead 处，航向=切线角，曲率=Menger 公式。
%                 系数 a/b/c 写入调试向量槽位 5:7。
%
% 输出布局（72×1，两平台一致）：
%   1 steering_error  2 lateral_error  3 heading_error  4 confidence  5 found
%   6:72 局部路径调试向量 [有效点数, 前视x, 前视y, 曲率, a, b, c, x1,y1,...x30,y30]
%
% 相机安装（2026-07-22 起两平台一致，均为水平安装、pitch=0）：
%   MuJoCo turtlebot3_burger_vehicle_body.xml 的相机已拍平（原 15° 下俯，
%   见 archive/archive_vision_scheme_a），与 Gazebo r200 一直以来的水平安装
%   对齐。两平台仍保留独立的分辨率/FOV（真实传感器差异，未被要求统一），
%   以及独立调参的 LocalPath_ROIFraction/LookaheadDistance（各自模型工作区，
%   pitch=0 后近端可视距离下限升高到约 0.3~0.4m，两平台都需相应变大，
%   不能再用旧的 15°-下俯时代数值——具体见 originbot_camera_profile.m 内注释）。
%
% 适用性说明：滑窗按图像行向上爬，线走成水平/折返（U 型弯掉头）时窗列会提前
% 停止，该场景已归档的方案 A（骨架+纯追踪）更强；常规弯道上本算法点列更
% 干净、拟合输出更平滑抗噪。

% 丢线时保持上一次转向和路径调试向量，避免控制器输出跳变
persistent lastVisibleSteering lastLocalPathDebug lostLineTime

maxPathPoints = 30;
debugWidth = 7 + 2 * maxPathPoints;
prof = originbot_camera_profile(numel(rgbVector), platform);

% ── 与平台无关的行为常数（两平台共用，改这里两边同时生效）───────────────
steerMixPath     = 0.65;  % 路径转向(前视预测)权重
steerMixCentroid = 0.35;  % 质心转向(即时反馈)权重
lateralNorm      = 0.55;  % 侧向偏差满偏(m)：机器人半宽+安全余量
curvatureScale   = 0.20;  % 曲率进转向前的缩放
controlPeriod    = 0.05;  % = 模型 Ts_Control(20Hz)，用于丢线计时
freezeTimeout    = 0.5;   % <0.5s 瞬时遮挡：冻结上次转向
slowdownTimeout  = 1.5;   % 0.5~1.5s：转向线性衰减；>1.5s 判真丢线

if isempty(lastVisibleSteering), lastVisibleSteering = 0; end
if isempty(lastLocalPathDebug) || numel(lastLocalPathDebug) ~= debugWidth
    lastLocalPathDebug = zeros(debugWidth, 1);
end
if isempty(lostLineTime), lostLineTime = 0; end

% 缺参回退值与各自平台 InitFcn 导出表(实际调参值)一致
if nargin < 2  || isempty(roiBottomFraction), roiBottomFraction = prof.DefaultROI; end
if nargin < 3  || isempty(waypointCount),     waypointCount = maxPathPoints; end
if nargin < 4  || isempty(minBrightness),     minBrightness = 70; end
if nargin < 5  || isempty(maxSaturation),     maxSaturation = 0.30; end
if nargin < 6  || isempty(minPixels),         minPixels = 30; end
if nargin < 7  || isempty(errorScale),        errorScale = 500; end
if nargin < 8  || isempty(lookaheadDistance), lookaheadDistance = prof.DefaultLookahead; end
if nargin < 9  || isempty(lateralGain),       lateralGain = 0.6; end
if nargin < 10 || isempty(headingGain),       headingGain = 0.35; end
if nargin < 11 || isempty(curvatureGain),     curvatureGain = 0.04; end

roiBottomFraction = max(0.05, min(0.95, double(roiBottomFraction)));
windowCount = max(2, min(maxPathPoints, round(double(waypointCount))));
minBrightness = double(minBrightness);
maxSaturation = double(maxSaturation);
minPixels = double(minPixels);
errorScale = max(eps, double(errorScale));
lookaheadDistance = double(lookaheadDistance);

% 还原图像；仅 MuJoCo 需要翻正（framebuffer 为 bottom-row-first），
% Gazebo/ROS 帧本就正立
rgb = reshape(rgbVector, prof.ImageHeight, prof.ImageWidth, 3);
if prof.NeedsFlip
    rgb = flip(rgb, 1);
end

pathTop = max(1, min(prof.ImageHeight, ...
    floor((1 - roiBottomFraction) * prof.ImageHeight) + 1));
pathBottom = prof.ImageHeight;
centerX = 0.5 * (prof.ImageWidth + 1);

% ── 白线掩码：HSV 自适应亮度 + 低饱和判据 + 最大连通块 ──────────────────
roi = rgb(pathTop:pathBottom, :, :);
value = max(roi, [], 3);
minimumChannel = min(roi, [], 3);
saturation = (value - minimumChannel) ./ max(value, 1);
roiMaximum = max(value, [], "all");
adaptiveBrightness = min(minBrightness, max(40, 0.55 * roiMaximum));
mask = value >= adaptiveBrightness & saturation <= maxSaturation;
totalPixels = sum(mask, "all");

if totalPixels > 0
    [labels, numComponents] = bwlabel(mask, 8);
    if numComponents > 1
        componentSizes = zeros(numComponents, 1);
        for c = 1:numComponents
            componentSizes(c) = sum(labels(:) == c);
        end
        [~, largestLabel] = max(componentSizes);
        mask = (labels == largestLabel);
    end
end

% ── 1) 霍夫变换：主导直线段 → 滑窗种子列 + 列漂移斜率 ──────────────────
[baseCol, driftPerRow, houghFound] = houghSeed(mask, prof.HoughMaxPeaks, ...
    prof.HoughMinLength, prof.HoughFillGap, prof.MaxDriftPerRow);

% 霍夫失败回退：ROI 下三分之一列直方图峰值
if ~houghFound && totalPixels > 0
    nearBand = mask(max(1, end - floor(size(mask, 1) / 3)):end, :);
    columnHist = sum(nearBand, 1);
    [peakVal, peakCol] = max(columnHist);
    if peakVal > 0
        baseCol = peakCol;
    else
        baseCol = centerX;
    end
    driftPerRow = 0;
end

% ── 2) 滑动窗口：自底向上收集窗内质心（行号返回全图坐标）──────────────
[winCols, winRows, winValid, nearPixelError] = slideWindows(mask, pathTop, ...
    baseCol, driftPerRow, windowCount, prof.WindowHalfWidth, ...
    prof.MinWindowPixels, centerX);

% 窗心 IPM 投影到地面坐标（米，X 前向 / Y 右正）
xPoints = zeros(maxPathPoints, 1);
yPoints = zeros(maxPathPoints, 1);
valid = false(maxPathPoints, 1);
writeIdx = 0;
for k = 1:windowCount
    if ~winValid(k), continue; end
    [xg, yg] = originbot_pixel_to_ground(winCols(k), winRows(k), prof);
    if isnan(xg) || isnan(yg), continue; end
    writeIdx = writeIdx + 1;
    xPoints(writeIdx) = xg;
    yPoints(writeIdx) = yg;
    valid(writeIdx) = true;
end
validCount = sum(valid);

found = double(validCount >= 1 && totalPixels >= minPixels);
confidence = min(1, totalPixels / max(1, windowCount * minPixels));

% ── 3) 多项式拟合 + 前视几何 ───────────────────────────────────────────
[lookaheadX, lookaheadY, headingErrorRaw, curvature, coefficients] = ...
    fitPolyLookahead(xPoints, yPoints, valid, lookaheadDistance);

lateralError = max(-1, min(1, lookaheadY / lateralNorm));
headingError = max(-1, min(1, headingErrorRaw));
curvatureForControl = max(-1, min(1, curvatureScale * curvature));
centroidSteering = nearPixelError / errorScale;
pathSteering = lateralGain * lateralError + headingGain * headingError + ...
    curvatureGain * curvatureForControl;
currentSteeringError = max(-1.5, min(1.5, ...
    steerMixPath * pathSteering + steerMixCentroid * centroidSteering));

localPathDebug = zeros(debugWidth, 1);
localPathDebug(1) = validCount;
localPathDebug(2) = lookaheadX;
localPathDebug(3) = lookaheadY;
localPathDebug(4) = curvature;
localPathDebug(5:7) = coefficients(:);
offset = 8;
for k = 1:maxPathPoints
    localPathDebug(offset) = xPoints(k);
    localPathDebug(offset + 1) = yPoints(k);
    offset = offset + 2;
end

% ── 丢线分级处理 ────────────────────────────────────────────────────────
if found > 0.5
    lostLineTime = 0;
    steeringError = currentSteeringError;
    lastVisibleSteering = steeringError;
    lastLocalPathDebug = localPathDebug;
else
    lostLineTime = lostLineTime + controlPeriod;
    lateralError = 0;
    headingError = 0;
    localPathDebug = lastLocalPathDebug;
    localPathDebug(1) = 0;
    if lostLineTime <= freezeTimeout
        steeringError = lastVisibleSteering;
    elseif lostLineTime <= slowdownTimeout
        decay = 1 - (lostLineTime - freezeTimeout) / ...
            max(eps, slowdownTimeout - freezeTimeout);
        steeringError = lastVisibleSteering * decay;
    else
        steeringError = 0;
        lastVisibleSteering = 0;
    end
end

result = [steeringError; lateralError; headingError; confidence; found; ...
    localPathDebug];
end


function [baseCol, driftPerRow, houghFound] = houghSeed(mask, maxPeaks, ...
    minLength, fillGap, maxDrift)
%HOUGHSEED 霍夫变换找 ROI 掩码中的主导直线段，返回滑窗种子。
%   baseCol      直线延长至掩码底行的列坐标（滑窗起始列）
%   driftPerRow  行号 +1（向下/向近处）时列的变化量（滑窗上爬时取负向外推）
%   houghFound   是否找到有效直线段
baseCol = 0;
driftPerRow = 0;
houghFound = false;
if ~any(mask, "all")
    return
end

[H, theta, rho] = hough(mask);
peaks = houghpeaks(H, maxPeaks, "Threshold", ceil(0.3 * max(H(:))));
if isempty(peaks)
    return
end
segments = houghlines(mask, theta, rho, peaks, ...
    "FillGap", fillGap, "MinLength", minLength);
if isempty(segments)
    return
end

% 取最长线段为主导线（最能代表白线整体走向）
bestLen = -1;
bestSeg = segments(1);
for s = segments
    d = double(s.point2) - double(s.point1);   % [dCol dRow]
    len = hypot(d(1), d(2));
    if len > bestLen
        bestLen = len;
        bestSeg = s;
    end
end

p1 = double(bestSeg.point1);                   % [col row] 掩码局部坐标
d = double(bestSeg.point2) - p1;
if d(2) < 0                                    % 归一化方向：行分量指向下(近处)
    d = -d;                                    % 外推用线上任一点即可
end

bottomRow = size(mask, 1);
if abs(d(2)) < 1e-6
    % 近水平线：无法外推到底行，用线段中点列，斜率视为 0
    baseCol = p1(1) + 0.5 * d(1);
    driftPerRow = 0;
else
    driftPerRow = max(-maxDrift, min(maxDrift, d(1) / d(2)));
    baseCol = p1(1) + (bottomRow - p1(2)) * driftPerRow;
end
baseCol = max(1, min(size(mask, 2), baseCol));
houghFound = true;
end


function [winCols, winRows, winValid, nearPixelError] = slideWindows(mask, ...
    pathTop, baseCol, driftPerRow, windowCount, halfWidth, minWinPixels, ...
    centerX)
%SLIDEWINDOWS 自底向上滑动窗口：逐窗测出白线游程、取其中点并重定中下一窗。
%   winCols/winRows  各窗中点（列 / 全图行号）
%   winValid         该窗是否找到足量白像素的游程
%   nearPixelError   最近端有效窗中点的列偏差(px)，供 PID 即时反馈
%
% 2026-08-08：由「固定 halfWidth 窗内像素质心」改为「逐行带测量游程取中点」，
% 移植自 Python 侧 common/vision.py 的 _find_nearest_run/_slide_windows
% （那边 2026-08-05 就改了，MATLAB 一直停在旧算法上）。原因是实测：640 宽真
% 车帧上胶带线宽 130~161px，而 halfWidth=20 只有 40px 宽的搜索框，永远只能
% 看到线的一小片，质心跟着那一片相对线的位置漂，逐帧在两条边之间跳——这是
% 真车转圈/摆动事故的直接成因。改用实测游程自身的中点后，窗宽自适应每行的
% 真实线宽，不再需要手调像素常数；halfWidth 降级为「防止跳到远处无关亮区」
% 的搜索半径闸门，其取值同步在 originbot_camera_profile.m 里调大。
roiHeight = size(mask, 1);
imageWidth = size(mask, 2);
windowHeight = max(2, floor(roiHeight / windowCount));

winCols = zeros(windowCount, 1);
winRows = zeros(windowCount, 1);
winValid = false(windowCount, 1);

currentCol = baseCol;
% 空窗外推速率：初值来自霍夫斜率（上爬一窗 = 行号减 windowHeight）
lastDelta = -windowHeight * driftPerRow;

for w = 1:windowCount
    rowHigh = roiHeight - (w - 1) * windowHeight;   % 窗底(近)
    rowLow = max(1, rowHigh - windowHeight + 1);    % 窗顶(远)
    rowBand = mask(rowLow:rowHigh, :);              % 整行带，不再按 halfWidth 裁列
    colMass = sum(rowBand, 1);

    [runLeft, runRight, pixelCount] = findNearestRun( ...
        colMass, currentCol, halfWidth, minWinPixels);
    if runLeft > 0
        centroidCol = 0.5 * (runLeft + runRight);   % 游程中点，而非像素质心
        rowMass = sum(rowBand(:, runLeft:runRight), 2);
        centroidRow = rowLow - 1 + ...
            sum(rowMass .* (1:numel(rowMass))') / pixelCount;
        if w > 1
            lastDelta = centroidCol - currentCol;
        end
        currentCol = centroidCol;
        winCols(w) = centroidCol;
        winRows(w) = centroidRow + pathTop - 1;     % 转回全图行号
        winValid(w) = true;
    else
        % 空窗：按最近漂移速率外推列位置，继续向上找
        currentCol = max(1, min(imageWidth, currentCol + lastDelta));
    end
end

nearIdx = find(winValid, 1, "first");
if isempty(nearIdx)
    nearPixelError = 0;
else
    nearPixelError = winCols(nearIdx) - centerX;
end
end


function [left, right, pixelCount] = findNearestRun( ...
    colMass, currentCol, searchRadius, minPixels)
%FINDNEARESTRUN 在 currentCol ± searchRadius 内找像素数达标的连续亮列游程，
% 优先取中点离 currentCol 最近的那条（时空连续性，防止跳到搜索带里另一块
% 不相干的亮区）。返回 1-indexed 的 [left, right] 与该游程像素数；未找到
% 时 left 返回 0。
%
% 2026-08-08 新增，移植自 Python 侧 common/vision.py 的 _find_nearest_run
% （见 slideWindows 顶部注释里的移植缘由）。searchRadius 即原来的
% halfWidth：现在只是「别跳太远」的闸门，不再是质心计算窗本身。
width = numel(colMass);
left = 0;
right = 0;
pixelCount = 0;

lo = max(1, floor(currentCol - searchRadius));
hi = min(width, ceil(currentCol + searchRadius));
if hi < lo
    return;
end

gated = colMass(lo:hi);
lit = find(gated > 0);
if isempty(lit)
    return;
end

gaps = find(diff(lit(:)) > 1);
runStarts = [1; gaps + 1];
runEnds = [gaps; numel(lit)];

bestDist = inf;
for r = 1:numel(runStarts)
    s = lit(runStarts(r));
    e = lit(runEnds(r));
    runPixels = sum(gated(s:e));
    if runPixels < minPixels
        continue;
    end
    absLeft = lo + s - 1;
    absRight = lo + e - 1;
    dist = abs(0.5 * (absLeft + absRight) - currentCol);
    if dist < bestDist
        bestDist = dist;
        left = absLeft;
        right = absRight;
        pixelCount = runPixels;
    end
end
end


function [lookaheadX, lookaheadY, headingError, curvature, coefficients] = ...
    fitPolyLookahead(xPoints, yPoints, valid, lookaheadDistance)
%FITPOLYLOOKAHEAD 地面坐标二次多项式拟合 + 前视几何。
% 滑窗按行分层保证每个前向距离至多一个点，单值假设 Y=f(X) 由构造成立。
%   headingError  前视点处切线角(弧度)
%   curvature     Menger 曲率 k = 2a/(1+slope²)^1.5（右正）
coefficients = [0, 0, 0];
validIndex = find(valid);
if isempty(validIndex)
    lookaheadX = max(0.05, lookaheadDistance);
    lookaheadY = 0;
    headingError = 0;
    curvature = 0;
    return
end

xValid = xPoints(validIndex);
yValid = yPoints(validIndex);

if isscalar(validIndex) || (max(xValid) - min(xValid)) < 0.02
    % 单点或前向跨度过小：无法拟合，把最近点视作前视点
    lookaheadX = xValid(1);
    lookaheadY = yValid(1);
    headingError = atan2(yValid(1), max(eps, xValid(1)));
    curvature = 2 * lookaheadY / max(eps, xValid(1)^2 + yValid(1)^2);
    return
end

% 前视距离钳位在有效点范围内，避免外推误差过大
lookaheadX = min(max(lookaheadDistance, min(xValid)), max(xValid));

if numel(validIndex) >= 3
    coefficients = polyfit(xValid, yValid, 2);
else
    slope2 = (yValid(2) - yValid(1)) / max(eps, xValid(2) - xValid(1));
    coefficients = [0, slope2, yValid(1) - slope2 * xValid(1)];
end

lookaheadY = polyval(coefficients, lookaheadX);
slope = 2 * coefficients(1) * lookaheadX + coefficients(2);
headingError = atan(slope);
curvature = 2 * coefficients(1) / max(eps, (1 + slope^2)^(1.5));
end
