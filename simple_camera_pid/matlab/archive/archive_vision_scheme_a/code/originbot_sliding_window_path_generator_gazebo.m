function result = originbot_sliding_window_path_generator_gazebo(rgbVector, ...
    roiBottomFraction, waypointCount, minBrightness, maxSaturation, ...
    minPixels, errorScale, lookaheadDistance, lateralGain, headingGain, ...
    curvatureGain)
%ORIGINBOT_SLIDING_WINDOW_PATH_GENERATOR_GAZEBO 方案 B 的 Gazebo 相机适配版。
%
% 与 originbot_sliding_window_path_generator（MuJoCo 版方案 B：霍夫种子+滑窗+
% 地面二次多项式拟合）算法完全相同，仅适配 Gazebo 相机几何：
%   - 分辨率 1920×1080（MuJoCo 为 640×480），像素级常数按 3 倍宽度缩放
%   - 图像正立（ROS image_raw 顶行在上），无需 MuJoCo 的 flipud 翻正；
%     旧 gazebo 函数的"正向 vs 180° 择优"已移除（MuJoCo 侧实测该逻辑在
%     V 弯误切换朝向；若实测发现帧上下颠倒，请先用旧函数确认再调整此处）
%   - 相机水平安装（pitch=0，MuJoCo 下俯 15°），地平线在图像中线附近，
%     因此 ROIFraction 必须保持小值（默认 0.10，只取底部地面带）
%
% 接口与输出布局与 MuJoCo 版一致（72×1），可直接替换模型 MATLABFcn 表达式：
%   originbot_sliding_window_path_generator_gazebo(u, LocalPath_ROIFraction, ...)
%
% ⚠️ cameraParams/pixelToGround 与 originbot_local_path_generator_gazebo 的
% 副本保持一致（跨文件副本），改 Gazebo 相机 SDF 请同步两处。

% 丢线时保持上一次转向和路径调试向量，避免控制器输出跳变
persistent lastVisibleSteering lastLocalPathDebug lostLineTime

imageHeight = 1080;
imageWidth = 1920;
maxPathPoints = 30;
debugWidth = 7 + 2 * maxPathPoints;

% ── 本算法内部参数（像素常数已按 1920/640=3 倍分辨率缩放）────────────────
windowHalfWidth  = 60;    % 滑窗半宽(px)：MuJoCo 版 20px × 3
minWindowPixels  = 15;    % 单窗判有效的最少白像素数（约 5 × 3）
maxDriftPerRow   = 3;     % 霍夫斜率钳位(px列/px行)，比值量纲、不随分辨率缩放
houghMaxPeaks    = 4;     % 霍夫峰值数
houghMinLength   = 120;   % 有效直线段最短长度(px)：40 × 3
houghFillGap     = 60;    % 同一直线允许的断口(px)：20 × 3
% ── 与 MuJoCo 方案 B 保持一致的行为常数（两文件需同步）──────────────────
steerMixPath     = 0.65;  % 路径转向(前视预测)权重
steerMixCentroid = 0.35;  % 质心转向(即时反馈)权重
lateralNorm      = 0.55;  % 侧向偏差满偏(m)
curvatureScale   = 0.20;  % 曲率进转向前的缩放
controlPeriod    = 0.05;  % = 模型 Ts_Control(20Hz)，用于丢线计时
freezeTimeout    = 0.5;   % <0.5s 瞬时遮挡：冻结上次转向
slowdownTimeout  = 1.5;   % 0.5~1.5s：转向线性衰减；>1.5s 判真丢线

if isempty(lastVisibleSteering), lastVisibleSteering = 0; end
if isempty(lastLocalPathDebug) || numel(lastLocalPathDebug) ~= debugWidth
    lastLocalPathDebug = zeros(debugWidth, 1);
end
if isempty(lostLineTime), lostLineTime = 0; end

% 缺参回退值与 gazebo InitFcn 导出表(实际调参值)一致
if nargin < 2  || isempty(roiBottomFraction), roiBottomFraction = 0.10; end
if nargin < 3  || isempty(waypointCount),     waypointCount = maxPathPoints; end
if nargin < 4  || isempty(minBrightness),     minBrightness = 70; end
if nargin < 5  || isempty(maxSaturation),     maxSaturation = 0.30; end
if nargin < 6  || isempty(minPixels),         minPixels = 30; end
if nargin < 7  || isempty(errorScale),        errorScale = 500; end
if nargin < 8  || isempty(lookaheadDistance), lookaheadDistance = 0.20; end
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

% 还原图像（Gazebo/ROS 帧正立，直接使用，无需翻转）
rgb = reshape(rgbVector, imageHeight, imageWidth, 3);

pathTop = max(1, min(imageHeight, ...
    floor((1 - roiBottomFraction) * imageHeight) + 1));
pathBottom = imageHeight;
centerX = 0.5 * (imageWidth + 1);

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
[baseCol, driftPerRow, houghFound] = houghSeed(mask, houghMaxPeaks, ...
    houghMinLength, houghFillGap, maxDriftPerRow);

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
    baseCol, driftPerRow, windowCount, windowHalfWidth, minWindowPixels, ...
    centerX);

% 窗心 IPM 投影到地面坐标（米，X 前向 / Y 右正）
p = cameraParams();
xPoints = zeros(maxPathPoints, 1);
yPoints = zeros(maxPathPoints, 1);
valid = false(maxPathPoints, 1);
writeIdx = 0;
for k = 1:windowCount
    if ~winValid(k), continue; end
    [xg, yg] = pixelToGround(winCols(k), winRows(k), p);
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

% ── 丢线分级处理（与 MuJoCo 方案 B 一致）───────────────────────────────
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

% 取最长线段为主导线
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
    baseCol = 0.5 * (2 * p1(1) + d(1));        % 近水平线：用线段中点列
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
%SLIDEWINDOWS 自底向上滑动窗口：逐窗取白线质心并重定中下一窗。
roiHeight = size(mask, 1);
imageWidth = size(mask, 2);
windowHeight = max(2, floor(roiHeight / windowCount));

winCols = zeros(windowCount, 1);
winRows = zeros(windowCount, 1);
winValid = false(windowCount, 1);

currentCol = baseCol;
lastDelta = -windowHeight * driftPerRow;

for w = 1:windowCount
    rowHigh = roiHeight - (w - 1) * windowHeight;   % 窗底(近)
    rowLow = max(1, rowHigh - windowHeight + 1);    % 窗顶(远)
    colLow = max(1, round(currentCol - halfWidth));
    colHigh = min(imageWidth, round(currentCol + halfWidth));

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
        winRows(w) = centroidRow + pathTop - 1;     % 转回全图行号
        winValid(w) = true;
    else
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


function [lookaheadX, lookaheadY, headingError, curvature, coefficients] = ...
    fitPolyLookahead(xPoints, yPoints, valid, lookaheadDistance)
%FITPOLYLOOKAHEAD 地面坐标二次多项式拟合 + 前视几何。
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
    lookaheadX = xValid(1);
    lookaheadY = yValid(1);
    headingError = atan2(yValid(1), max(eps, xValid(1)));
    curvature = 2 * lookaheadY / max(eps, xValid(1)^2 + yValid(1)^2);
    return
end

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


function p = cameraParams()
% ⚠️ 与 originbot_local_path_generator_gazebo/cameraParams 保持同步。
% TurtleBot3 Burger Gazebo 前置相机（RealSense r200 光学参数，水平安装）。
p.imageWidth = 1920;
p.imageHeight = 1080;
p.fovyDeg = 35.3069;    % 由 SDF horizontal_fov=1.02974 rad 换算的等效竖直 FOV
p.mountHeight = 0.133;  % 相机离地高度(m)
p.pitchDeg = 0;         % 水平安装，无下俯
end


function [Xground, Yground] = pixelToGround(u, v, p)
% ⚠️ 与 originbot_local_path_generator_gazebo/pixelToGround 保持同步。
% 针孔 + 平地假设，把像素 (u,v) 投到地面 (X 前向, Y 右正)，单位 m。
fy = (p.imageHeight / 2) / tand(p.fovyDeg / 2);
fx = fy;
cx = (p.imageWidth + 1) / 2;
cy = (p.imageHeight + 1) / 2;

xc = (u - cx) / fx;
yc = (v - cy) / fy;
zc = ones(size(u));

ph = deg2rad(p.pitchDeg);
dDown = cos(ph) .* yc + sin(ph) .* zc;
dForward = -sin(ph) .* yc + cos(ph) .* zc;
dRight = xc;

t = p.mountHeight ./ dDown;
t(dDown <= 1e-6) = NaN;

Xground = t .* dForward;
Yground = t .* dRight;

maxGroundRange = 5.0;                        % m，循线前视有效距离上限
beyond = ~(Xground <= maxGroundRange);
Xground(beyond) = NaN;
Yground(beyond) = NaN;
end
