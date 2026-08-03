function debugFrame = originbot_line_follower_debug_frame(rgbInput, ...
    roiBottomFraction, waypointCount, minBrightness, maxSaturation, ...
    minPixels, ~, platform)
%ORIGINBOT_LINE_FOLLOWER_DEBUG_FRAME 方案 B 视觉调试画面（Camera Monitor 用）。
%
% 控制转向由 originbot_sliding_window_path_generator（方案 B）计算；本函数
% 无状态地复刻其检测管线（掩码+最大连通块 → 霍夫种子 → 滑动窗口 → 地面
% 二次拟合），把中间量画在相机图上，仅供可视化、不影响控制：
%   - 蓝色横线    ROI 上下边界（丢线时变红）
%   - 绿色像素    白线掩码（与控制器一致：已做最大连通块筛选）
%   - 青色方框    有效滑窗（窗内白像素达标）
%   - 灰色方框    空滑窗（按漂移速率外推经过的位置）
%   - 品红色方块  各有效窗的白线质心
%   - 黄色曲线    地面系二次拟合 Y=aX²+bX+c 反投影回图像的路径
%   - 白色竖线    图像中心列（转向零点）
%   - 红色竖线    最近端有效窗质心列（PID 即时反馈项的来源）
%
% MuJoCo/Gazebo/real 三个平台共用本文件——不再有 "_gazebo" 后缀的姐妹
% 文件。相机/像素尺度常数经 originbot_camera_profile(numel(rgbInput), platform)
% 按显式 platform 参数取得（2026-08-03 起必填第 8 参，三平台分辨率统一为
% 640x480 后像素总数已无法自动识别平台，见该函数的文档），与控制函数
% originbot_sliding_window_path_generator 共用同一份定义（不再是两份需要
% 手动同步的硬编码副本）；掩码/霍夫/滑窗/拟合的算法逻辑镜像自控制函数，
% 第 7 参 ErrorScale 为保持调用表达式一致而保留，本函数不使用。

maxPathPoints = 30;
prof = originbot_camera_profile(numel(rgbInput), platform);

roiBottomFraction = max(0.05, min(0.95, double(roiBottomFraction)));
windowCount = max(2, min(maxPathPoints, round(double(waypointCount))));
minBrightness = double(minBrightness);
maxSaturation = double(maxSaturation);
minPixels = double(minPixels);

% 统一转换为 uint8 H×W×3，兼容向量和三维数组两种输入格式
rgb = reshapeToRgb(rgbInput, prof.ImageHeight, prof.ImageWidth);
% 仅 MuJoCo 需要翻正（framebuffer 为 bottom-row-first），与控制函数一致；
% Gazebo/ROS 帧本就正立
if prof.NeedsFlip
    rgb = flip(rgb, 1);
end

pathTop = max(1, min(prof.ImageHeight, ...
    floor((1 - roiBottomFraction) * prof.ImageHeight) + 1));
pathBottom = prof.ImageHeight;
centerX = 0.5 * (prof.ImageWidth + 1);

% ── 掩码：HSV 自适应亮度 + 低饱和 + 最大连通块（镜像控制函数）──────────
roi = double(rgb(pathTop:pathBottom, :, :));
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

% ── 霍夫种子 + 直方图回退（镜像控制函数）───────────────────────────────
[baseCol, driftPerRow, houghFound] = houghSeed(mask, prof.HoughMaxPeaks, ...
    prof.HoughMinLength, prof.HoughFillGap, prof.MaxDriftPerRow);
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

% ── 滑动窗口（镜像控制函数，额外返回窗框坐标用于绘制）──────────────────
[winCols, winRows, winValid, winRects] = slideWindows(mask, pathTop, ...
    baseCol, driftPerRow, windowCount, prof.WindowHalfWidth, ...
    prof.MinWindowPixels);

% 窗心 IPM → 地面坐标（与控制函数共用同一份投影实现）
xValid = [];
yValid = [];
for k = 1:windowCount
    if ~winValid(k), continue; end
    [xg, yg] = originbot_pixel_to_ground(winCols(k), winRows(k), prof);
    if isnan(xg) || isnan(yg), continue; end
    xValid(end + 1) = xg; %#ok<AGROW>
    yValid(end + 1) = yg; %#ok<AGROW>
end
validCount = numel(xValid);
found = validCount >= 1 && totalPixels >= minPixels;

% ── 绘制 ───────────────────────────────────────────────────────────────
debugFrame = rgb;
debugFrame = drawRoi(debugFrame, pathTop, pathBottom, found);
debugFrame = overlayMask(debugFrame, mask, pathTop);

% 滑窗框：有效窗青色、空窗暗灰
for k = 1:windowCount
    if winValid(k)
        boxColor = uint8([0 255 255]);
    else
        boxColor = uint8([110 110 110]);
    end
    debugFrame = drawRectOutline(debugFrame, winRects(k, :), boxColor);
end

% 拟合抛物线（地面系采样 → 反投影回像素）：黄色曲线
if validCount >= 2 && (max(xValid) - min(xValid)) >= 0.02
    if validCount >= 3
        coefficients = polyfit(xValid, yValid, 2);
    else
        slope2 = (yValid(2) - yValid(1)) / max(eps, xValid(2) - xValid(1));
        coefficients = [0, slope2, yValid(1) - slope2 * xValid(1)];
    end
    xs = linspace(min(xValid), max(xValid), 40);
    ys = polyval(coefficients, xs);
    [us, vs] = originbot_ground_to_pixel(xs, ys, prof);
    debugFrame = drawPixelCurve(debugFrame, us, vs, uint8([255 220 0]));
end

% 窗心标记：品红
for k = 1:windowCount
    if winValid(k)
        debugFrame = drawMarker(debugFrame, round(winRows(k)), ...
            round(winCols(k)), uint8([255 0 255]));
    end
end

% 白色中心列
debugFrame = drawVerticalLine(debugFrame, round(centerX), pathTop, ...
    pathBottom, uint8([255 255 255]));
% 红色竖线：最近端有效窗质心列（centroidSteering 的来源）
nearIdx = find(winValid, 1, "first");
if ~isempty(nearIdx)
    debugFrame = drawVerticalLine(debugFrame, round(winCols(nearIdx)), ...
        pathTop, pathBottom, uint8([255 0 0]));
end

% Simulink MATLABFcn 仅接受有限 double 向量，返回展平向量由模型侧还原
debugFrame = double(debugFrame(:));
end


function rgb = reshapeToRgb(rgbInput, imageHeight, imageWidth)
if ndims(rgbInput) == 3
    rgb = rgbInput;
else
    rgb = reshape(rgbInput, imageHeight, imageWidth, 3);
end
if ~isa(rgb, "uint8")
    rgb = uint8(min(max(round(rgb), 0), 255));
end
end


function [baseCol, driftPerRow, houghFound] = houghSeed(mask, maxPeaks, ...
    minLength, fillGap, maxDrift)
% 镜像 originbot_sliding_window_path_generator/houghSeed，两处同步修改
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

bestLen = -1;
bestSeg = segments(1);
for s = segments
    d = double(s.point2) - double(s.point1);
    len = hypot(d(1), d(2));
    if len > bestLen
        bestLen = len;
        bestSeg = s;
    end
end

p1 = double(bestSeg.point1);
d = double(bestSeg.point2) - p1;
if d(2) < 0
    d = -d;
end

bottomRow = size(mask, 1);
if abs(d(2)) < 1e-6
    baseCol = p1(1) + 0.5 * d(1);
    driftPerRow = 0;
else
    driftPerRow = max(-maxDrift, min(maxDrift, d(1) / d(2)));
    baseCol = p1(1) + (bottomRow - p1(2)) * driftPerRow;
end
baseCol = max(1, min(size(mask, 2), baseCol));
houghFound = true;
end


function [winCols, winRows, winValid, winRects] = slideWindows(mask, ...
    pathTop, baseCol, driftPerRow, windowCount, halfWidth, minWinPixels)
% 镜像 originbot_sliding_window_path_generator/slideWindows，两处同步修改。
% winRects(k,:) = [rowLow rowHigh colLow colHigh]（全图坐标），供绘制窗框。
roiHeight = size(mask, 1);
imageWidth = size(mask, 2);
windowHeight = max(2, floor(roiHeight / windowCount));

winCols = zeros(windowCount, 1);
winRows = zeros(windowCount, 1);
winValid = false(windowCount, 1);
winRects = zeros(windowCount, 4);

currentCol = baseCol;
lastDelta = -windowHeight * driftPerRow;

for w = 1:windowCount
    rowHigh = roiHeight - (w - 1) * windowHeight;
    rowLow = max(1, rowHigh - windowHeight + 1);
    colLow = max(1, round(currentCol - halfWidth));
    colHigh = min(imageWidth, round(currentCol + halfWidth));
    winRects(w, :) = [rowLow + pathTop - 1, rowHigh + pathTop - 1, ...
        colLow, colHigh];

    window = mask(rowLow:rowHigh, colLow:colHigh);
    pixelCount = sum(window, "all");
    if pixelCount >= minWinPixels
        colMass = sum(window, 1);
        rowMass = sum(window, 2);
        centroidCol = colLow - 1 + ...
            sum(colMass .* (1:numel(colMass))) / pixelCount;
        centroidRow = rowLow - 1 + ...
            sum(rowMass .* (1:numel(rowMass))') / pixelCount;
        if w > 1 || winValid(1)
            lastDelta = centroidCol - currentCol;
        end
        currentCol = centroidCol;
        winCols(w) = centroidCol;
        winRows(w) = centroidRow + pathTop - 1;
        winValid(w) = true;
    else
        currentCol = max(1, min(imageWidth, currentCol + lastDelta));
    end
end
end


function frame = drawPixelCurve(frame, us, vs, color)
% 把像素采样点序列连成折线；NaN/出界点断开
lastValid = false;
prevU = 0; prevV = 0;
for k = 1:numel(us)
    u = round(us(k)); v = round(vs(k));
    ok = isfinite(u) && isfinite(v) && u >= 1 && u <= size(frame, 2) && ...
        v >= 1 && v <= size(frame, 1);
    if ok && lastValid
        frame = drawSegment(frame, prevV, prevU, v, u, color);
    elseif ok
        frame = drawMarker(frame, v, u, color);
    end
    if ok
        prevU = u; prevV = v;
    end
    lastValid = ok;
end
end


function frame = drawRectOutline(frame, rect, color)
% rect = [rowLow rowHigh colLow colHigh]，画 1px 边框
rowLow = max(1, rect(1)); rowHigh = min(size(frame, 1), rect(2));
colLow = max(1, rect(3)); colHigh = min(size(frame, 2), rect(4));
for c = 1:3
    channel = frame(:, :, c);
    channel(rowLow, colLow:colHigh) = color(c);
    channel(rowHigh, colLow:colHigh) = color(c);
    channel(rowLow:rowHigh, colLow) = color(c);
    channel(rowLow:rowHigh, colHigh) = color(c);
    frame(:, :, c) = channel;
end
end


function frame = drawMarker(frame, row, col, color)
% 画 3×3 像素方块作为标记，便于低分辨率下辨识
rows = max(1, row-1):min(size(frame, 1), row+1);
cols = max(1, col-1):min(size(frame, 2), col+1);
for c = 1:3
    channel = frame(:, :, c);
    channel(rows, cols) = color(c);
    frame(:, :, c) = channel;
end
end


function frame = drawSegment(frame, rowA, colA, rowB, colB, color)
% 类 Bresenham 光栅化：步数取行列方向最大差值，保证对角线无空洞
steps = max(abs(rowB - rowA), abs(colB - colA)) + 1;
if steps <= 1
    frame = drawMarker(frame, rowA, colA, color);
    return
end
for n = 0:steps-1
    alpha = double(n) / double(steps - 1);
    row = round((1 - alpha) * rowA + alpha * rowB);
    col = round((1 - alpha) * colA + alpha * colB);
    frame = drawMarker(frame, row, col, color);
end
end


function frame = drawRoi(frame, topRow, bottomRow, found)
% 找到白线时用蓝色标示 ROI 边界，丢线时改为红色提示异常
if found
    color = uint8([0 120 255]);
else
    color = uint8([255 0 0]);
end
topRows = max(1, topRow-1):min(size(frame, 1), topRow+1);
bottomRows = max(1, bottomRow-1):min(size(frame, 1), bottomRow+1);
frame(topRows, :, :) = paintRows(frame(topRows, :, :), color);
frame(bottomRows, :, :) = paintRows(frame(bottomRows, :, :), color);
end


function painted = paintRows(rows, color)
painted = rows;
for c = 1:3
    painted(:, :, c) = color(c);
end
end


function frame = overlayMask(frame, mask, topRow)
[maskRows, maskCols] = find(mask);
if isempty(maskRows)
    return
end
rows = min(max(maskRows + topRow - 1, 1), size(frame, 1));
cols = min(max(maskCols, 1), size(frame, 2));
idx = sub2ind([size(frame, 1), size(frame, 2)], rows, cols);

red = frame(:, :, 1);
green = frame(:, :, 2);
blue = frame(:, :, 3);
red(idx) = uint8(0);
green(idx) = uint8(255);
blue(idx) = uint8(0);
frame(:, :, 1) = red;
frame(:, :, 2) = green;
frame(:, :, 3) = blue;
end


function frame = drawVerticalLine(frame, col, topRow, bottomRow, color)
col = min(max(col, 1), size(frame, 2));
cols = max(1, col-1):min(size(frame, 2), col+1);
rows = max(1, topRow):min(size(frame, 1), bottomRow);
for c = 1:3
    channel = frame(:, :, c);
    channel(rows, cols) = color(c);
    frame(:, :, c) = channel;
end
end
