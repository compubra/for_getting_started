function debugFrame = originbot_line_follower_debug_frame_gazebo(rgbInput, ...
    roiBottomFraction, waypointCount, minBrightness, maxSaturation, ...
    minPixels, ~)
%ORIGINBOT_LINE_FOLLOWER_DEBUG_FRAME Build a live vision debug image.
%
% The controller uses originbot_local_path_generator for the actual
% steering command. This helper mirrors the local-path ROI/mask selection
% and returns an RGB frame for Camera Monitor:
%   - blue horizontal lines: ROI limits
%   - green pixels: low-saturation bright line mask
%   - white line: image center
%   - red line: detected line centroid
%   - yellow line: local path samples
%   - red ROI border: line not found
%
% It deliberately has no persistent state and does not affect control.

imageHeight = 1080;
imageWidth = 1920;
maxPathPoints = 30;
roiBottomFraction = max(0.05, min(0.95, double(roiBottomFraction)));
waypointCount = max(2, min(maxPathPoints, round(double(waypointCount))));
% 统一转换为 uint8 H×W×3，兼容向量和三维数组两种输入格式
rgb = reshapeToRgb(rgbInput, imageHeight, imageWidth);

% ROI 从底部向上延伸 roiBottomFraction 比例，与控制器保持一致
correctedTop = max(1, min(imageHeight, ...
    floor((1 - roiBottomFraction) * imageHeight) + 1));
correctedBottom = imageHeight;

% 正向姿态下检测白线掩码和质心
[uprightMask, uprightCount, uprightCentroid] = detectMask( ...
    rgb, correctedTop, correctedBottom, minBrightness, maxSaturation);

% 180° 翻转后再次检测，用于应对 MuJoCo 渲染帧旋转的情况
rotatedRgb = flip(flip(rgb, 1), 2);
[rotatedMask, rotatedCount, rotatedCentroid] = detectMask( ...
    rotatedRgb, correctedTop, correctedBottom, minBrightness, maxSaturation);

% 选取白色像素更多的方向作为显示帧，与控制器判断逻辑一致
if rotatedCount > uprightCount
    displayRgb = rotatedRgb;
    mask = rotatedMask;
    pixelCount = rotatedCount;
    centroidX = rotatedCentroid;
else
    displayRgb = rgb;
    mask = uprightMask;
    pixelCount = uprightCount;
    centroidX = uprightCentroid;
end

debugFrame = displayRgb;
% ROI 边界线：找到线时显示蓝色，丢线时显示红色
debugFrame = drawRoi(debugFrame, correctedTop, correctedBottom, ...
    pixelCount >= minPixels);
% 将白线检测掩码叠加为绿色像素
debugFrame = overlayMask(debugFrame, mask, correctedTop);
% 按路径航点采样，生成局部路径折线点
[pathRows, pathCols, pathValid] = detectLocalPathPixels(displayRgb, ...
    correctedTop, correctedBottom, minBrightness, maxSaturation, ...
    waypointCount);
% 用黄色连线绘制局部路径采样点
debugFrame = drawLocalPath(debugFrame, pathRows, pathCols, pathValid, ...
    uint8([255 220 0]));
% 白色竖线标示图像中心列
debugFrame = drawVerticalLine(debugFrame, round(0.5 * (imageWidth + 1)), ...
    correctedTop, correctedBottom, uint8([255 255 255]));

% 红色竖线标示检测到的白线质心列（未找到线时不绘制）
if pixelCount >= minPixels
    debugFrame = drawVerticalLine(debugFrame, round(centroidX), ...
        correctedTop, correctedBottom, uint8([255 0 0]));
end

% Simulink MATLABFcn 仅接受有限 double 向量/二维信号，
% 返回展平的 double 向量，由 Simulink 侧再还原为 RGB
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

function [mask, pixelCount, centroidX] = detectMask(rgb, topRow, ...
    bottomRow, minBrightness, maxSaturation)
imageWidth = size(rgb, 2);
roi = double(rgb(topRow:bottomRow, :, :));
% value：最大 RGB 分量，等价于 HSV 的 V 通道
value = max(roi, [], 3);
minimumChannel = min(roi, [], 3);
saturation = (value - minimumChannel) ./ max(value, 1);
roiMaximum = max(value, [], "all");
% 自适应亮度阈值：跟随 ROI 实际最亮值动态降低，避免阴影/光照变化导致漏检
adaptiveBrightness = min(minBrightness, max(40, 0.55 * roiMaximum));
% 白线像素：亮且低饱和度（非彩色）
mask = value >= adaptiveBrightness & saturation <= maxSaturation;

pixelCount = sum(mask, "all");
centroidX = 0.5 * (imageWidth + 1);
if pixelCount > 0
    columnMass = sum(mask, 1);
    centroidX = sum(columnMass .* (1:imageWidth)) / pixelCount;
end
end

function [pathRows, pathCols, pathValid] = detectLocalPathPixels(rgb, ...
    pathTop, pathBottom, minBrightness, maxSaturation, waypointCount)
imageWidth = size(rgb, 2);
pathRows = zeros(waypointCount, 1);
pathCols = zeros(waypointCount, 1);
pathValid = false(waypointCount, 1);
pathSpan = max(1, pathBottom - pathTop);
% 水平带半高与航点数成反比，确保相邻带不重叠且覆盖整个 ROI
bandHalfHeight = max(4, round(pathSpan / (2.4 * waypointCount)));

for k = 1:waypointCount
    if waypointCount == 1
        progress = 0;
    else
        % progress = 0 对应最近（图像底部），progress = 1 对应最远（图像顶部）
        progress = double(k - 1) / double(waypointCount - 1);
    end
    rowCenter = round(pathBottom - progress * pathSpan);
    topRow = max(pathTop, rowCenter - bandHalfHeight);
    bottomRow = min(pathBottom, rowCenter + bandHalfHeight);
    [pixelCount, centroidX] = detectBandCentroid(rgb, topRow, bottomRow, ...
        minBrightness, maxSaturation);
    pathRows(k) = rowCenter;
    pathCols(k) = round(min(max(centroidX, 1), imageWidth));
    pathValid(k) = pixelCount > 0;
end
end

function [pixelCount, centroidX] = detectBandCentroid(rgb, topRow, ...
    bottomRow, minBrightness, maxSaturation)
imageWidth = size(rgb, 2);
roi = double(rgb(topRow:bottomRow, :, :));
value = max(roi, [], 3);
minimumChannel = min(roi, [], 3);
saturation = (value - minimumChannel) ./ max(value, 1);
roiMaximum = max(value, [], "all");
adaptiveBrightness = min(minBrightness, max(40, 0.55 * roiMaximum));
mask = value >= adaptiveBrightness & saturation <= maxSaturation;

pixelCount = sum(mask, "all");
centroidX = 0.5 * (imageWidth + 1);
if pixelCount > 0
    columnMass = sum(mask, 1);
    centroidX = sum(columnMass .* (1:imageWidth)) / pixelCount;
end
end

function frame = drawLocalPath(frame, rows, cols, valid, color)
validIndices = find(valid);
for k = 1:numel(validIndices)
    idx = validIndices(k);
    frame = drawMarker(frame, rows(idx), cols(idx), color);
    if k > 1
        % 用线段连接相邻有效航点，形成连续路径曲线
        previous = validIndices(k - 1);
        frame = drawSegment(frame, rows(previous), cols(previous), ...
            rows(idx), cols(idx), color);
    end
end
end

function frame = drawMarker(frame, row, col, color)
% 画 3×3 像素方块作为航点标记，便于在低分辨率调试图中辨识
rows = max(1, row-1):min(size(frame, 1), row+1);
cols = max(1, col-1):min(size(frame, 2), col+1);
for c = 1:3
    channel = frame(:, :, c);
    channel(rows, cols) = color(c);
    frame(:, :, c) = channel;
end
end

function frame = drawSegment(frame, rowA, colA, rowB, colB, color)
% 步数取行列方向最大差值，保证对角线方向无空洞（类 Bresenham 光栅化）
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

% 将 ROI 内的掩码行号偏移回全图坐标
rows = maskRows + topRow - 1;
cols = maskCols;
rows = min(max(rows, 1), size(frame, 1));
cols = min(max(cols, 1), size(frame, 2));
% 用线性索引批量写入，避免循环逐像素操作
idx = sub2ind([size(frame, 1), size(frame, 2)], rows, cols);

red = frame(:, :, 1);
green = frame(:, :, 2);
blue = frame(:, :, 3);
% 将检测到的白线像素染为绿色覆盖显示
red(idx) = uint8(0);
green(idx) = uint8(255);
blue(idx) = uint8(0);
frame(:, :, 1) = red;
frame(:, :, 2) = green;
frame(:, :, 3) = blue;
end

function frame = drawVerticalLine(frame, col, topRow, bottomRow, color)
col = min(max(col, 1), size(frame, 2));
% 3 像素宽竖线，确保在缩略图中清晰可见
cols = max(1, col-1):min(size(frame, 2), col+1);
rows = max(1, topRow):min(size(frame, 1), bottomRow);
for c = 1:3
    channel = frame(:, :, c);
    channel(rows, cols) = color(c);
    frame(:, :, c) = channel;
end
end
