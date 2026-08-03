function result = originbot_local_path_generator_gazebo(rgbVector, roiBottomFraction, ...
    waypointCount, minBrightness, maxSaturation, minPixels, errorScale, ...
    lookaheadDistance, lateralGain, headingGain, curvatureGain)
%ORIGINBOT_LOCAL_PATH_GENERATOR Build a camera-local path for line tracking.
%
% Output layout:
%   1 steering_error
%   2 lateral_error
%   3 heading_error
%   4 confidence
%   5 found
%   6:72 local path debug vector
%
% Local path debug layout:
%   1 valid point count
%   2 lookahead x
%   3 lookahead y
%   4 fitted curvature
%   5 quadratic coefficient a, y = a*x^2 + b*x + c
%   6 quadratic coefficient b
%   7 quadratic coefficient c
%   8:67 thirty [x;y] local path point slots

% 丢线时保持上一次转向和路径调试向量，避免控制器输出跳变
persistent lastVisibleSteering lastLocalPathDebug lostLineTime

imageHeight = 1080;
imageWidth = 1920;
maxPathPoints = 30;
% 调试向量长度 = 7 个标量 + 每个路径点的 [x; y]
debugWidth = 7 + 2 * maxPathPoints;

if isempty(lastVisibleSteering)
    lastVisibleSteering = 0;
end
if isempty(lastLocalPathDebug) || numel(lastLocalPathDebug) ~= debugWidth
    lastLocalPathDebug = zeros(debugWidth, 1);
end
if isempty(lostLineTime)
    lostLineTime = 0;
end

if nargin < 2 || isempty(roiBottomFraction)
    roiBottomFraction = 0.10;
end
if nargin < 3 || isempty(waypointCount)
    waypointCount = maxPathPoints;
end
if nargin < 4 || isempty(minBrightness)
    minBrightness = 70;
end
if nargin < 5 || isempty(maxSaturation)
    maxSaturation = 0.30;
end
if nargin < 6 || isempty(minPixels)
    minPixels = 30;
end
if nargin < 7 || isempty(errorScale)
    errorScale = 500;
end
if nargin < 8 || isempty(lookaheadDistance)
    lookaheadDistance = 0.20;
end
if nargin < 9 || isempty(lateralGain)
    lateralGain = 1.0;
end
if nargin < 10 || isempty(headingGain)
    headingGain = 0.35;
end
if nargin < 11 || isempty(curvatureGain)
    curvatureGain = 0.08;
end

roiBottomFraction = max(0.05, min(0.95, double(roiBottomFraction)));
activePointCount = max(2, min(maxPathPoints, round(double(waypointCount))));
minBrightness = double(minBrightness);
maxSaturation = double(maxSaturation);
minPixels = double(minPixels);
errorScale = max(eps, double(errorScale));
lookaheadDistance = double(lookaheadDistance);
lateralGain = double(lateralGain);
headingGain = double(headingGain);
curvatureGain = double(curvatureGain);

% 将摄像头 RGB 向量还原为 H×W×3 图像矩阵
rgb = reshape(rgbVector, imageHeight, imageWidth, 3);
% ROI 顶部行号：从图像底部向上 roiBottomFraction 比例
pathTop = max(1, min(imageHeight, ...
    floor((1 - roiBottomFraction) * imageHeight) + 1));
pathBottom = imageHeight;

% 正向和 180° 翻转各提取一次路径，取白色像素更多的结果
uprightPath = extractLocalPath(rgb, pathTop, pathBottom, minBrightness, ...
    maxSaturation, minPixels, activePointCount, maxPathPoints, imageWidth);
rotatedRgb = flip(flip(rgb, 1), 2);
rotatedPath = extractLocalPath(rotatedRgb, pathTop, pathBottom, ...
    minBrightness, maxSaturation, minPixels, activePointCount, ...
    maxPathPoints, imageWidth);

if rotatedPath.TotalPixels > uprightPath.TotalPixels
    localPath = rotatedPath;
else
    localPath = uprightPath;
end

found = double(localPath.ValidCount >= 1 && localPath.TotalPixels >= minPixels);
% 置信度以"每个航点恰好满足 minPixels"为满分，衡量线的清晰程度
confidence = min(1, localPath.TotalPixels / ...
    max(1, activePointCount * minPixels));

% 在路径曲线上拟合二次多项式，并计算前视点的侧向偏差、航向偏差和曲率
[lookaheadX, lookaheadY, headingError, curvature, coefficients] = ...
    fitQuadraticLookahead(localPath.X, localPath.Y, localPath.Valid, ...
    lookaheadDistance);

% 侧向偏差归一化：0.55 m 对应满偏（机器人半宽 + 安全余量）
lateralError = max(-1, min(1, lookaheadY / 0.55));
headingError = max(-1, min(1, headingError));
% 曲率缩放因子 0.20：避免急弯时曲率项主导转向
curvatureForControl = max(-1, min(1, 0.20 * curvature));
% 近端质心误差保留原始像素单位，供 PID 混合使用
centroidSteering = localPath.NearPixelError / errorScale;
pathSteering = lateralGain * lateralError + headingGain * headingError + ...
    curvatureGain * curvatureForControl;
% 路径转向（前视预测）占 65%，质心转向（即时反馈）占 35%，提升曲线跟踪稳定性
currentSteeringError = max(-1.5, min(1.5, ...
    0.65 * pathSteering + 0.35 * centroidSteering));

localPathDebug = packLocalPathDebug(localPath, lookaheadX, lookaheadY, ...
    curvature, coefficients, maxPathPoints);

% 控制周期：= 模型 Ts_Control (20Hz)。无外部时钟，按固定步长累加丢线时长
controlPeriod = 0.05;
% 丢线分级阈值
freezeTimeout = 0.5;    % < 0.5s：视为瞬时遮挡，冻结上次转向、全速继续
slowdownTimeout = 1.5;  % 0.5~1.5s：转向幅度线性衰减，控制器据此减速
% > 1.5s：判定真丢线，停止转向，交由上层停车/搜索

if found > 0.5
    % 正常找到线：重置丢线计时并更新记忆
    lostLineTime = 0;
    steeringError = currentSteeringError;
    lastVisibleSteering = steeringError;
    lastLocalPathDebug = localPathDebug;
else
    % 丢线：累加时长并分级处理；侧向/航向误差归零防止积分饱和
    lostLineTime = lostLineTime + controlPeriod;
    lateralError = 0;
    headingError = 0;
    localPathDebug = lastLocalPathDebug;
    % 有效点数清零，通知下游路径已丢失
    localPathDebug(1) = 0;

    if lostLineTime <= freezeTimeout
        % 瞬时遮挡：冻结上次转向
        steeringError = lastVisibleSteering;
    elseif lostLineTime <= slowdownTimeout
        % 持续丢线：转向随时间线性衰减，提示控制器减速
        decay = 1 - (lostLineTime - freezeTimeout) / ...
            max(eps, slowdownTimeout - freezeTimeout);
        steeringError = lastVisibleSteering * decay;
    else
        % 长时间丢线：停止转向，confidence/found 已为 0，上层应停车/搜索
        steeringError = 0;
        lastVisibleSteering = 0;
    end
end

result = [steeringError; lateralError; headingError; confidence; found; ...
    localPathDebug];
end

function localPath = extractLocalPath(rgb, pathTop, pathBottom, minBrightness, ...
    maxSaturation, minPixels, activePointCount, maxPathPoints, imageWidth)
centerX = 0.5 * (imageWidth + 1);
xPoints = zeros(maxPathPoints, 1);
yPoints = zeros(maxPathPoints, 1);
valid = false(maxPathPoints, 1);
pixelCounts = zeros(maxPathPoints, 1);
pixelErrors = zeros(maxPathPoints, 1);
pathSpan = max(1, pathBottom - pathTop);
% 水平带半高与航点数成反比，确保相邻带不重叠且覆盖整个 ROI
bandHalfHeight = max(2, round(pathSpan / (2.4 * activePointCount)));
% 单个带的有效像素门限设为 minPixels 的 20%，宽松判断以减少噪声导致的漏点
validPixelThreshold = max(3, 0.20 * minPixels);

for k = 1:activePointCount
    if activePointCount == 1
        progress = 0;
    else
        % progress = 0 → 图像底部（最近），progress = 1 → 图像顶部（最远）
        progress = double(k - 1) / double(activePointCount - 1);
    end
    rowCenter = round(pathBottom - progress * pathSpan);
    topRow = max(pathTop, rowCenter - bandHalfHeight);
    bottomRow = min(pathBottom, rowCenter + bandHalfHeight);
    [pixelError, pixelCount] = detectBand(rgb, topRow, bottomRow, ...
        minBrightness, maxSaturation, imageWidth);

    % IPM 逆透视：把该带质心像素投影到真实地面坐标(米)，取代旧的手调魔数尺度。
    % X(前向)只取决于行号 rowCenter，与是否检测到线无关，故先用图像中心列算出
    % 该带的几何 X，保证全部路径点的 X 处于同一真实尺度（避免回退魔数污染 polyfit）。
    p = cameraParams();
    [xGeom, ~] = pixelToGround(centerX, rowCenter, p);
    if isnan(xGeom)
        % 该带在地平线以上，无地面交点：此带整体无效
        xPoints(k) = 0.18 + 0.78 * progress;   % 占位，valid=false 不参与拟合
        yPoints(k) = 0;
        pixelErrors(k) = pixelError;
        pixelCounts(k) = pixelCount;
        valid(k) = false;
        continue
    end
    xPoints(k) = xGeom;
    % Y(横向)用检测到的质心列做 IPM；用同一行的几何确保 X/Y 一致
    centroidU = centerX + pixelError;     % 质心列像素(pixelError 相对图像中心)
    [~, Yg] = pixelToGround(centroidU, rowCenter, p);
    if isnan(Yg)
        yPoints(k) = 0;
    else
        yPoints(k) = Yg;
    end
    pixelErrors(k) = pixelError;
    pixelCounts(k) = pixelCount;
    valid(k) = pixelCount >= validPixelThreshold;
end

% 对无效带（线未检测到的行）做线性插值，避免路径断裂影响多项式拟合
if any(valid)
    yPoints = fillMissingPathY(xPoints, yPoints, valid, activePointCount);
end

% 近端（最靠近机器人）的质心误差，用于 PID 的即时反馈项
nearIndex = find(valid, 1, "first");
nearPixelError = 0;
if ~isempty(nearIndex)
    nearPixelError = pixelErrors(nearIndex);
end

localPath = struct( ...
    "X", xPoints, ...
    "Y", yPoints, ...
    "Valid", valid, ...
    "ValidCount", double(sum(valid)), ...
    "TotalPixels", double(sum(pixelCounts)), ...
    "NearPixelError", nearPixelError);
end

function yPoints = fillMissingPathY(xPoints, yPoints, valid, activePointCount)
validIndex = find(valid(1:activePointCount));
if isscalar(validIndex)
    % 只有一个有效点时，用该点横向值填充全部（直线延伸假设）
    yPoints(1:activePointCount) = yPoints(validIndex);
    return
end

for k = 1:activePointCount
    if valid(k)
        continue
    end
    lower = validIndex(find(validIndex < k, 1, "last"));
    upper = validIndex(find(validIndex > k, 1, "first"));
    if isempty(lower)
        % 缺口在最近端，用最近有效点外推
        yPoints(k) = yPoints(upper);
    elseif isempty(upper)
        % 缺口在最远端，用最远有效点外推
        yPoints(k) = yPoints(lower);
    else
        % 在两个有效点之间做 x 坐标线性插值
        alpha = (xPoints(k) - xPoints(lower)) / ...
            max(eps, xPoints(upper) - xPoints(lower));
        yPoints(k) = (1 - alpha) * yPoints(lower) + alpha * yPoints(upper);
    end
end
end

function [pixelError, pixelCount] = detectBand(rgb, topRow, bottomRow, ...
    minBrightness, maxSaturation, imageWidth)
% 输入 rgb 已由 Simulink 的 RGB_To_Double 块转为 double，此处无需再转
roi = rgb(topRow:bottomRow, :, :);
% value：HSV 的 V 通道（最大 RGB 分量）
value = max(roi, [], 3);
minimumChannel = min(roi, [], 3);
saturation = (value - minimumChannel) ./ max(value, 1);
roiMaximum = max(value, [], "all");
% 自适应亮度阈值跟随 ROI 实际最亮值，防止阴影区域漏检
adaptiveBrightness = min(minBrightness, max(40, 0.55 * roiMaximum));
mask = value >= adaptiveBrightness & saturation <= maxSaturation;

pixelError = 0;
pixelCount = 0;
if ~any(mask, "all")
    return
end

% 连通域筛选：只保留最大白色连通块，排除岔路/十字路口的第二条线及反光杂点，
% 避免列投影质心落到两条线中间的空白处。水平带很矮，4 连通足够。
[labels, numComponents] = bwlabel(mask, 4);
if numComponents > 1
    componentSizes = zeros(numComponents, 1);
    for c = 1:numComponents
        componentSizes(c) = sum(labels(:) == c);
    end
    [~, largestLabel] = max(componentSizes);
    mask = (labels == largestLabel);
end

pixelCount = sum(mask, "all");
if pixelCount == 0
    return
end

% 用列质量加权求质心，pixelError > 0 表示线偏右
columnMass = sum(mask, 1);
centroidX = sum(columnMass .* (1:imageWidth)) / pixelCount;
centerX = 0.5 * (imageWidth + 1);
pixelError = centroidX - centerX;
end

function [lookaheadX, lookaheadY, headingError, curvature, coefficients] = ...
    fitQuadraticLookahead(xPoints, yPoints, valid, lookaheadDistance)
validIndex = find(valid);
coefficients = [0, 0, 0];
if isempty(validIndex)
    % 无有效点时输出零偏差，防止控制器突变
    lookaheadX = max(0.18, min(0.96, lookaheadDistance));
    lookaheadY = 0;
    headingError = 0;
    curvature = 0;
    return
end

xValid = xPoints(validIndex);
yValid = yPoints(validIndex);
% 前视距离钳位在有效点范围内，避免外推误差过大
lookaheadX = min(max(lookaheadDistance, min(xValid)), max(xValid));

if numel(validIndex) >= 3
    % 三点及以上：二次多项式拟合，捕捉曲线形状
    coefficients = polyfit(xValid, yValid, 2);
elseif numel(validIndex) == 2
    % 两点：退化为线性拟合
    slope = (yValid(2) - yValid(1)) / max(eps, xValid(2) - xValid(1));
    intercept = yValid(1) - slope * xValid(1);
    coefficients = [0, slope, intercept];
else
    % 单点：视为水平线，无法判断航向
    coefficients = [0, 0, yValid(1)];
end

lookaheadY = polyval(coefficients, lookaheadX);
% 前视点处多项式导数（切线斜率）
slope = 2 * coefficients(1) * lookaheadX + coefficients(2);
% Menger 曲率公式：k = 2a / (1 + slope²)^(3/2)
curvature = 2 * coefficients(1) / max(eps, (1 + slope^2)^(1.5));
% 航向偏差为切线角度（弧度）
headingError = atan(slope);
end

function localPathDebug = packLocalPathDebug(localPath, lookaheadX, ...
    lookaheadY, curvature, coefficients, maxPathPoints)
% 打包调试向量：[有效点数, 前视x, 前视y, 曲率, a, b, c, x1, y1, x2, y2, ...]
localPathDebug = zeros(7 + 2 * maxPathPoints, 1);
localPathDebug(1) = localPath.ValidCount;
localPathDebug(2) = lookaheadX;
localPathDebug(3) = lookaheadY;
localPathDebug(4) = curvature;
localPathDebug(5:7) = coefficients(:);

offset = 8;
for k = 1:maxPathPoints
    localPathDebug(offset) = localPath.X(k);
    localPathDebug(offset + 1) = localPath.Y(k);
    offset = offset + 2;
end
end

function p = cameraParams()
% TurtleBot3 Burger Gazebo 前置相机实测参数。
% 官方 turtlebot3_burger 无内置相机，本项目在此局部 Gazebo 模型
% (model/gazebo/turtlebot3/models/turtlebot3_burger_line_follower/model.sdf)
% 上按与 stock turtlebot3_waffle 相同的 Intel RealSense r200 光学参数
% (horizontal_fov=1.02974 rad, 1920x1080) 加装了一个前置相机，camera_joint
% roll/pitch/yaw 全部为 0——水平前视安装，不同于 MuJoCo 场景专门下俯 15° 的相机。
p.imageWidth = 1920;
p.imageHeight = 1080;
% SDF 给的是 horizontal_fov=1.02974 rad，此处换算为等效竖直 FOV(度)，
% 与 pixelToGround() 的 fy = (imageHeight/2)/tand(fovyDeg/2) 配套使用：
%   fx = (imageWidth/2)/tan(hfov/2); fovyDeg = 2*atand((imageHeight/imageWidth)*tan(hfov/2))
p.fovyDeg = 35.3069;
p.mountHeight = 0.133;  % 相机离地高度(m): camera_joint z=0.110 + camera_rgb_joint z=0.013 + base_joint 离地 0.010
p.pitchDeg = 0;         % 下俯角(度)：本模型水平安装，无下俯
end

function [Xground, Yground] = pixelToGround(u, v, p)
% 针孔 + 平地假设，把像素 (u,v) 投到地面 (X 前向, Y 右正)，单位 m。
% MuJoCo fovy 定义垂直视场，故先由 fovy 求 fy，方形像素令 fx = fy。
% Y 取"右为正"以匹配下游控制律约定（线偏右 -> yPoints>0 -> 右转为正），
% 与旧魔数法 yPoints=(pixelError/centerX)*scale 的符号保持一致。
% 返回 NaN 表示该像素在地平线以上、与地面无交点。
fy = (p.imageHeight / 2) / tand(p.fovyDeg / 2);
fx = fy;
cx = (p.imageWidth + 1) / 2;
cy = (p.imageHeight + 1) / 2;

% 相机坐标系归一化射线 (x右, y下, z前)
xc = (u - cx) / fx;
yc = (v - cy) / fy;
zc = ones(size(u));

% 绕相机右轴下俯 pitch，转到车体水平参考系
ph = deg2rad(p.pitchDeg);
dDown = cos(ph) .* yc + sin(ph) .* zc;     % 车体系向下分量
dForward = -sin(ph) .* yc + cos(ph) .* zc; % 车体系前向分量
dRight = xc;

% 与地面求交：沿射线下降量达到安装高度
t = p.mountHeight ./ dDown;
t(dDown <= 1e-6) = NaN;                     % 射线未朝下 → 地平线以上，无交点

Xground = t .* dForward;
Yground = t .* dRight;                      % 右正，匹配下游 yPoints 约定

% 近地平线奇点保护：dDown→0 时 t 爆炸，前向距离被病态放大(可达数十米)，
% 会污染 polyfit。循线前视 <1m 才有意义，超过 maxGroundRange 视为无效。
maxGroundRange = 5.0;                        % m
beyond = ~(Xground <= maxGroundRange);      % 同时把 NaN 也算作 beyond
Xground(beyond) = NaN;
Yground(beyond) = NaN;
end
