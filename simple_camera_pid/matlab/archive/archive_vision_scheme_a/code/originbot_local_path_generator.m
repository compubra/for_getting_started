function result = originbot_local_path_generator(rgbVector, roiBottomFraction, ...
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

imageHeight = 480;
imageWidth = 640;
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
% MuJoCo framebuffer 为 bottom-row-first，raw 帧上下颠倒（左右正常）。运动标定
% 实验证实为纯 flipud（纯前进时线往图像上移出；原地左转 cCol 右移正常）。
% 在此翻正一次：翻正后 row 大=近处，下方 ROI 与 pixelToGround 的行号语义才成立。
rgb = flip(rgb, 1);
% 正图下(已翻正): 图像上半=前方远处, 下半=近处脚下。循线需要前瞻(look-ahead)，
% 故 ROI 取图像“中部到前方”一段: 从最底行往上 roiBottomFraction 比例覆盖近端反馈，
% 但检测/追踪的有效前视来自更靠上的行。extractOrderedLine 从 ROI 最底行(最近)向上
% (向前方远处)追踪有序点，pure-pursuit 再沿弧长取前视点。
% 说明: 旧颠倒图时“底部 ROI”恰好对应物理前方(颠倒把远处翻到底部)，故能循线；翻正
% 后同一份底部 ROI 变成脚下近处、丢了前瞻。这里保持底部锚定但由 pure-pursuit 的
% lookahead 提供前瞻，ROI 需足够大(见 LocalPath_ROIFraction, 建议 >=0.6)。
pathTop = max(1, min(imageHeight, ...
    floor((1 - roiBottomFraction) * imageHeight) + 1));
pathBottom = imageHeight;

% ===================== 方案 A：弧长参数化有序追踪 =====================
% 旧的“分带扫描 + 二次多项式拟合”隐含假设“线是前向距离 x 的单值函数 y(x)”，
% 掉头/U 型弯违反该假设（同一 x 出现两支），必然拟合失败。方案 A 改为：
%   1) 对整个 ROI 求白线骨架，从最近端起做最近邻排序得到“有序像素点列”；
%   2) 逐点 IPM 投影成地面有序点 (X,Y)，折返/横走/U 型都能如实表示；
%   3) 用纯追踪(pure-pursuit)沿弧长取前视点算转向，对任意曲率均成立。
% 注：曾有“正向 vs 180°翻转取白像素多者”的择优逻辑，已删除——raw 是纯 flipud
% （非 180°），上面显式翻正即可；择优会在 V 弯误切换朝向、造成画面倒转与转向抽搐。
localPath = extractOrderedLine(rgb, pathTop, pathBottom, minBrightness, ...
    maxSaturation, maxPathPoints);

found = double(localPath.ValidCount >= 1 && localPath.TotalPixels >= minPixels);
% 置信度以"每个航点恰好满足 minPixels"为满分，衡量线的清晰程度
confidence = min(1, localPath.TotalPixels / ...
    max(1, activePointCount * minPixels));

% 纯追踪：沿有序地面点的弧长取前视点，得到侧向/航向偏差与转向曲率。
% 无需多项式，故 coefficients 恒为 0（仅为兼容调试向量布局而保留）。
[lookaheadX, lookaheadY, headingError, curvature] = ...
    purePursuitControl(localPath.X, localPath.Y, localPath.Valid, ...
    lookaheadDistance);
coefficients = [0, 0, 0];

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

function localPath = extractOrderedLine(rgb, pathTop, pathBottom, ...
    minBrightness, maxSaturation, maxPathPoints)
%EXTRACTORDEREDLINE 方案 A 核心：把 ROI 白线提取为“有序地面点列”。
% 与旧 extractLocalPath 的根本区别：不再按行分带（每个前向距离一个点），
% 而是沿线的骨架从最近端追踪，允许折返/横走/U 型这类多值曲线。
% 返回字段与旧结构体保持一致，以复用下游打包与丢线逻辑：
%   X/Y            maxPathPoints×1 有序地面坐标(米，X 前向 / Y 右正)
%   Valid          对应点是否有效(有序点填 true，其余补零段填 false)
%   ValidCount     有效有序点数
%   TotalPixels    ROI 白线掩码像素总数(供 found/confidence 与翻转比较)
%   NearPixelError 最近端骨架点的列偏差(像素)，供 PID 即时反馈项

imageWidth = size(rgb, 2);
centerX = 0.5 * (imageWidth + 1);
p = cameraParams();

xPoints = zeros(maxPathPoints, 1);
yPoints = zeros(maxPathPoints, 1);
valid = false(maxPathPoints, 1);

% 1) 整块 ROI 求白线掩码（复用与分带一致的 HSV 自适应亮度 + 低饱和判据）
roi = rgb(pathTop:pathBottom, :, :);
value = max(roi, [], 3);
minimumChannel = min(roi, [], 3);
saturation = (value - minimumChannel) ./ max(value, 1);
roiMaximum = max(value, [], "all");
adaptiveBrightness = min(minBrightness, max(40, 0.55 * roiMaximum));
mask = value >= adaptiveBrightness & saturation <= maxSaturation;

totalPixels = sum(mask, "all");
if totalPixels == 0
    localPath = emptyOrderedLine(xPoints, yPoints, valid);
    return
end

% 2) 只保留最大连通块，排除岔路第二条线/反光杂点（U 型自身仍是单连通）
[labels, numComponents] = bwlabel(mask, 8);
if numComponents > 1
    componentSizes = zeros(numComponents, 1);
    for c = 1:numComponents
        componentSizes(c) = sum(labels(:) == c);
    end
    [~, largestLabel] = max(componentSizes);
    mask = (labels == largestLabel);
end

% 3) 骨架化：把有一定宽度的线细化成 1 像素宽中心线，便于沿线排序
skeleton = bwskel(mask);
[skRows, skCols] = find(skeleton);
if numel(skRows) < 2
    % 骨架太短无法排序：退化为整块质心的单点（valid 仅置 1 个）
    columnMass = sum(mask, 1);
    centroidCol = sum(columnMass .* (1:imageWidth)) / max(1, sum(columnMass));
    nearRowFull = pathBottom;
    [xg, yg] = pixelToGround(centroidCol, nearRowFull, p);
    if ~isnan(xg)
        xPoints(1) = xg;
        yPoints(1) = yg;
        valid(1) = true;
    end
    localPath = struct("X", xPoints, "Y", yPoints, "Valid", valid, ...
        "ValidCount", double(sum(valid)), "TotalPixels", double(totalPixels), ...
        "NearPixelError", centroidCol - centerX);
    return
end

% 骨架行号是 ROI 内局部行，转回整图行号（供 IPM 用真实几何）
skRowsFull = skRows + pathTop - 1;

% 4) 从“最近端”起做贪心最近邻排序，得到有序像素点列。
% 最近端 = 图像最底部(行号最大)的骨架点，对应机器人脚下。
[~, startIdx] = max(skRowsFull);
ordered = greedyOrderPoints(skCols, skRowsFull, startIdx);

% 骨架点通常远多于 maxPathPoints，等间隔抽稀到不超过槽位数
orderedCount = size(ordered, 1);
if orderedCount > maxPathPoints
    pick = round(linspace(1, orderedCount, maxPathPoints));
    ordered = ordered(pick, :);
    orderedCount = maxPathPoints;
end

% 5) 逐个有序像素点 IPM 投影成地面坐标(米)。地平线以上的点(NaN)跳过。
writeIdx = 0;
for k = 1:orderedCount
    uCol = ordered(k, 1);
    vRow = ordered(k, 2);
    [xg, yg] = pixelToGround(uCol, vRow, p);
    if isnan(xg) || isnan(yg)
        continue
    end
    writeIdx = writeIdx + 1;
    xPoints(writeIdx) = xg;
    yPoints(writeIdx) = yg;
    valid(writeIdx) = true;
end

% 最近端骨架点的列偏差，供 PID 即时反馈项（与旧 NearPixelError 语义一致）
nearPixelError = ordered(1, 1) - centerX;

localPath = struct( ...
    "X", xPoints, ...
    "Y", yPoints, ...
    "Valid", valid, ...
    "ValidCount", double(sum(valid)), ...
    "TotalPixels", double(totalPixels), ...
    "NearPixelError", nearPixelError);
end

function localPath = emptyOrderedLine(xPoints, yPoints, valid)
% 无白像素时的空结果，字段与正常结构体保持一致
localPath = struct("X", xPoints, "Y", yPoints, "Valid", valid, ...
    "ValidCount", 0, "TotalPixels", 0, "NearPixelError", 0);
end

function ordered = greedyOrderPoints(cols, rows, startIdx)
%GREEDYORDERPOINTS 贪心最近邻把散乱骨架点排成一条有序链。
% 从 startIdx 出发，每步跳到“尚未访问点中欧氏距离最近”的一个。
% 骨架已是 1 像素宽的近似曲线，连续段相邻点间距约 1~1.4 px。
%
% 大跳截断：粗弧带经 bwskel 会在弯曲/变宽处产生分叉骨架，贪心最近邻走完一支
% 后会“跳”到另一支（实测可达 110 px），使有序点出现回折/交叉，污染 pure-pursuit
% 的弧长/前视/航向。故当到最近未访问点的距离 > maxJump 时，判定已走到主段末端
% （再走就是跳向断裂的分支），立即截断——只保留从起点连续可达的主线段。
% 该主段足够 pure-pursuit 取近端前视点；U 型折返的骨架是连续的，不会被误截。
maxJump = 8;                     % px，允许的最大相邻间距（宽容小间隙，截断走岔）
maxJumpSq = maxJump ^ 2;
n = numel(cols);
ordered = zeros(n, 2);
visited = false(n, 1);

current = startIdx;
for step = 1:n
    ordered(step, :) = [cols(current), rows(current)];
    visited(current) = true;
    % 计算当前点到所有未访问点的平方距离，取最近者作为下一个
    dx = cols - cols(current);
    dy = rows - rows(current);
    distSquared = dx .^ 2 + dy .^ 2;
    distSquared(visited) = inf;
    [minDist, nextIdx] = min(distSquared);
    if isinf(minDist) || minDist > maxJumpSq
        % 全部访问完毕，或下一点是跨分支大跳 → 截断，保留连续主段
        ordered = ordered(1:step, :);
        return
    end
    current = nextIdx;
end
end

function [lookaheadX, lookaheadY, headingError, curvature] = ...
    purePursuitControl(xPoints, yPoints, valid, lookaheadDistance)
%PUREPURSUITCONTROL 纯追踪：沿有序地面点弧长取前视点，算转向几何。
% 关键：不依赖 y=f(x) 单值假设，故对掉头/U 型弯同样成立。
%   lookaheadX/Y  前视点地面坐标(米)
%   headingError  最近端切线方向角(弧度)
%   curvature     pure-pursuit 曲率 = 2*y_L / L_d^2（车体系，右正）
validIndex = find(valid);
if isempty(validIndex)
    % 无有效点：输出零偏差，防止控制器突变
    lookaheadX = max(0.18, min(0.96, lookaheadDistance));
    lookaheadY = 0;
    headingError = 0;
    curvature = 0;
    return
end

xValid = xPoints(validIndex);
yValid = yPoints(validIndex);

if isscalar(validIndex)
    % 单点：无法算切线，直接把该点当前视点
    lookaheadX = xValid(1);
    lookaheadY = yValid(1);
    headingError = atan2(yValid(1), max(eps, xValid(1)));
    curvature = 2 * lookaheadY / max(eps, xValid(1) ^ 2 + yValid(1) ^ 2);
    return
end

% 1) 沿有序点累积弧长（相邻点欧氏距离之和），得到每点的弧长坐标 s
segLengths = hypot(diff(xValid), diff(yValid));
arcLength = [0; cumsum(segLengths)];
totalArc = arcLength(end);

% 2) 目标前视弧长钳位在 [首点, 全长] 内，避免外推
targetArc = min(max(lookaheadDistance, arcLength(1)), totalArc);

% 3) 在弧长上线性插值出前视点坐标（沿线走 targetArc 到达的位置）
lookaheadX = interp1(arcLength, xValid, targetArc, "linear");
lookaheadY = interp1(arcLength, yValid, targetArc, "linear");

% 4) 航向偏差 = 最近端到前视点的割线方向（跨前视距离取平均方向）。
% 不用“首段两点”切线：近端相邻骨架点噪声大、X 差可能≈0 使 atan2 饱和到 ±90°
% （正图下直线段实测 heading 频繁饱和到 1.0 即此因）。割线跨越整段前视，稳健得多。
headingError = atan2(lookaheadY - yValid(1), ...
    max(eps, lookaheadX - xValid(1)));

% 5) pure-pursuit 曲率：连接车体原点到前视点的圆弧曲率 κ = 2*y_L / L_d^2，
% 其中 L_d 为前视点直线距离。κ 对任意大转角(含掉头)都有定义，正=右转。
lookaheadDistSquared = lookaheadX ^ 2 + lookaheadY ^ 2;
curvature = 2 * lookaheadY / max(eps, lookaheadDistSquared);
end

function localPath = extractLocalPath(rgb, pathTop, pathBottom, minBrightness, ...
    maxSaturation, minPixels, activePointCount, maxPathPoints, imageWidth) %#ok<DEFNU>
% [LEGACY，方案 A 后不再调用] 旧的分带扫描 + 单值假设路径提取。
% 保留仅供对照/回退，掉头场景会失败，勿在控制路径中使用。
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
% [LEGACY] 仅供旧 extractLocalPath 调用；方案 A 有序追踪不产生行内缺口。
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
    fitQuadraticLookahead(xPoints, yPoints, valid, lookaheadDistance) %#ok<DEFNU>
% [LEGACY，方案 A 后由 purePursuitControl 取代] 二次多项式拟合前视。
% 单值假设 y=a*x^2+b*x+c 无法表示掉头/U 型弯，保留仅供对照。
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
% TurtleBot3 Burger MuJoCo 前置相机实测参数。
% 来源: turtlebot3_burger_vehicle_body.xml + scene <global offwidth/offheight>。
% 相机已改为下俯 15°，见 xyaxes="0 -1 0 0.258819 0 0.965926"。
p.imageWidth = 640;
p.imageHeight = 480;
p.fovyDeg = 45.9857;    % MJCF <camera fovy> 是垂直 FOV（非水平）
p.mountHeight = 0.133;  % 相机离地高度(m): camera_link z=0.110 + camera_rgb_frame z=0.013 + base 离地 0.010
p.pitchDeg = 15;        % 下俯角(度)：与 xyaxes 编码一致
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
