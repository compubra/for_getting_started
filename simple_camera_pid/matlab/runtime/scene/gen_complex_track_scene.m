function info = gen_complex_track_scene(options)
%GEN_COMPLEX_TRACK_SCENE Procedurally generate a complex closed-loop track
%combining straights with independently-radiused corner arcs (MuJoCo scene).
%
%   与 gen_simple_track_scene（纯椭圆 / 两段直线+两个半圆端的胶囊）不同，
%   本函数把闭环拆成 NumCorners 个顶点，顶点间是直线段，每个顶点用**独立
%   随机半径**的圆弧倒角——相邻顶点各转各的，急弯（小半径）、缓弯（大半径）
%   随机混在同一条赛道上，形状远比椭圆/胶囊多样，且默认线宽更细。
%
%   算法（4 步，闭合是几何构造上天然保证的，不需要额外拼接/修正）：
%   1) 顶点生成：NumCorners 个点绕中心大致均匀分布——先按 2π/N 等分角度，
%      再叠加随机抖动，抖动幅度设了上限，排序后再做最小夹角限制，双重保证
%      角度严格递增 → 多边形不自交。每个顶点到中心的半径也独立随机，形状
%      因此不规则（不是正多边形）。
%   2) 顶点依次连线，得到一个不规则闭合多边形骨架。
%   3) 每个顶点独立随机取一个倒角半径，用标准的"角点倒圆角"公式
%      （切线长 = R·tan(|Δ|/2)，Δ 为该顶点的转角）算出圆弧倒角；相邻顶点
%      若倒角切线长度加起来超过它们共享那条边的长度，会按比例收缩两侧半径，
%      避免相邻倒角互相侵入、自交。
%   4) 倒角之间用直线段连接，得到最终"直线+圆弧"交替的完整闭环，光栅化成
%      白线 PNG 贴到 MuJoCo 地面 mesh 上。
%
%   Name-value options
%   ------------------
%     Name            (1,1) string  场景基名                    default 随机
%     NumCorners       (1,1) double  顶点/弯道数（越多越复杂）    default 8，最少 4
%     Seed                          rng 种子，[]=不设            default []
%     ImgSize          (1,1) double  PNG 边长(px)                default 512
%     LineWidth        (1,1) double  白线线宽(px)，比 gen_simple_track_scene
%                                    的默认 26px 细               default 12
%     Difficulty       (1,1) double  0..1，越大顶点角度抖动越大、
%                                    半径越偏向下限（急弯更常见）  default 0.5
%     MinCornerRadius  (1,1) double  单个弯道倒角半径下限(m)      default 0.15
%     MaxCornerRadius  (1,1) double  单个弯道倒角半径上限(m)      default 0.55
%     OutModelDir/OutTexDir/Overwrite  同 gen_simple_track_scene
%
%   Returns  info struct:
%     .Name .MapKey .SceneFile .PngFile
%     .StartPos [x y z]  .StartQuat [w x y z]
%     .Vertices  NumCorners×2  多边形顶点世界坐标(m)
%     .CornerRadii  1×NumCorners  每个顶点实际生效的倒角半径(m)，
%                   与请求半径不同说明被相邻边裁剪过
%
%   Example
%   -------
%     info = gen_complex_track_scene("Seed", 1, "NumCorners", 10, "Difficulty", 0.7);
%
%   See also GEN_SIMPLE_TRACK_SCENE.

arguments
    options.Name             (1,1) string = ""
    options.NumCorners       (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.NumCorners, 4)} = 8
    options.Seed                          = []
    options.ImgSize          (1,1) double {mustBePositive} = 512
    options.LineWidth        (1,1) double {mustBePositive} = 4
    options.Difficulty       (1,1) double {mustBeGreaterThanOrEqual(options.Difficulty,0), mustBeLessThanOrEqual(options.Difficulty,1)} = 0.5
    options.MinCornerRadius  (1,1) double {mustBePositive} = 0.15
    options.MaxCornerRadius  (1,1) double {mustBePositive} = 0.55
    options.OutModelDir      (1,1) string = ""
    options.OutTexDir        (1,1) string = ""
    options.Overwrite        (1,1) logical = true
end

if ~isempty(options.Seed)
    rng(options.Seed);
end
assert(options.MaxCornerRadius > options.MinCornerRadius, "GenComplexTrack:BadRadiusRange", ...
    "MaxCornerRadius must exceed MinCornerRadius.");

% ── 目录：自本文件位置逐级向上搜索 model/mujoco（与 gen_simple_track_scene
% 相同的自适应写法，不写死层数，随目录搬迁自动适配）──────────────────────
runtimeDir = string(fileparts(mfilename("fullpath")));
repoRoot   = findRepoRoot(runtimeDir);
mujocoDir  = fullfile(repoRoot, "model", "mujoco");
modelDir   = options.OutModelDir;
texDir     = options.OutTexDir;
if strlength(modelDir) == 0
    modelDir = fullfile(mujocoDir, "turtlebot3");
end
if strlength(texDir) == 0
    texDir = fullfile(mujocoDir, "shared", "assets", "textures");
end
assert(isfolder(modelDir), "GenComplexTrack:BadDir", "Scene dir not found: %s", modelDir);
assert(isfolder(texDir),   "GenComplexTrack:BadDir", "Texture dir not found: %s", texDir);

% ── 名字与 map key ──────────────────────────────────────────────────────────
name = options.Name;
if strlength(name) == 0
    name = "complex_rand_" + string(dec2hex(randi([0 65535]), 4));
end
mapKey    = lower(name);
pngName   = name + "_track.png";
sceneName = name + "_turtlebot3_burger_visual_scene.xml";
pngFile   = fullfile(texDir, pngName);
sceneFile = fullfile(modelDir, sceneName);
if ~options.Overwrite
    assert(~isfile(sceneFile), "GenComplexTrack:Exists", "Scene exists: %s", sceneFile);
end

% ── 1) 顶点：NumCorners 个点绕中心分布，角度与半径都随机 ────────────────────
n = options.NumCorners;
worldHalf = 2.2;                       % 地面 mesh 半宽(m)，与 gen_simple 一致
margin    = 0.55;                      % 顶点到边界的安全余量
maxR      = worldHalf - margin;        % 顶点半径上限 ~1.65m
baseR     = maxR * (0.55 + 0.25*rand());          % 整体尺度也随机，避免总是顶格
radiusJitterFrac = 0.15 + 0.25 * options.Difficulty; % 难度越高顶点半径越不规则

angleStep       = 2*pi / n;
angleJitterFrac = 0.25 + 0.45 * options.Difficulty;  % 难度越高角度抖动越大→弯道分布越不均匀
baseAngles = (0:n-1) * angleStep;
angles = baseAngles + (2*rand(1,n)-1) * angleStep * angleJitterFrac * 0.5;
angles = sort(mod(angles, 2*pi));
angles = enforceMinAngleGap(angles, deg2rad(10));    % 防止排序后两点角度几乎重合

vertRadii = baseR * (1 + (2*rand(1,n)-1) * radiusJitterFrac);
vx = vertRadii .* cos(angles);
vy = vertRadii .* sin(angles);

% 随机整体旋转 + 小幅中心偏移，同参数多次生成也不会长得一样
theta0 = (2*rand()-1) * pi;
Rmat = [cos(theta0) -sin(theta0); sin(theta0) cos(theta0)];
V = (Rmat * [vx; vy]).';
cShift = 0.10 * baseR * (2*rand(1,2) - 1);
V = V + cShift;

% ── 2) 每个顶点的有符号转角（用于倒角公式，也用于起点朝向判断左右）──────────
turnAngle = zeros(1, n);
edgeLen   = zeros(1, n);   % edgeLen(i) = 顶点 i -> 顶点 i+1 的边长
for i = 1:n
    iPrev = mod(i-2, n) + 1;
    iNext = mod(i, n) + 1;
    dIn  = V(i,:)     - V(iPrev,:);
    dOut = V(iNext,:) - V(i,:);
    edgeLen(i) = norm(dOut);
    dIn  = dIn  / max(eps, norm(dIn));
    dOut = dOut / max(eps, norm(dOut));
    turnAngle(i) = atan2(dIn(1)*dOut(2) - dIn(2)*dOut(1), dot(dIn, dOut));
end

% ── 3) 每个顶点独立随机倒角半径 → 切线长，再做相邻边冲突裁剪 ────────────────
% 指数 (1+Difficulty) 让难度越高越偏向抽到小半径（急弯更常见，大半径的缓弯变少）
targetRadius = options.MinCornerRadius + ...
    (options.MaxCornerRadius - options.MinCornerRadius) * rand(1,n) .^ (1 + options.Difficulty);

tanDist = zeros(1, n);
for i = 1:n
    if abs(turnAngle(i)) > deg2rad(1)
        tanDist(i) = targetRadius(i) * tan(abs(turnAngle(i)) / 2);
    end
end
% relax 迭代：若某条边两端顶点的切线长之和超过边长的 85%，按比例收缩两者，
% 避免相邻倒角圆弧互相侵入（自交）。3 轮足够收敛到安全配置。
for pass = 1:3
    for i = 1:n
        iNext = mod(i, n) + 1;
        avail = 0.85 * edgeLen(i);
        s = tanDist(i) + tanDist(iNext);
        if s > avail && s > eps
            f = avail / s;
            tanDist(i)     = tanDist(i) * f;
            tanDist(iNext) = tanDist(iNext) * f;
        end
    end
end
effRadius = zeros(1, n);
for i = 1:n
    if abs(turnAngle(i)) > deg2rad(1)
        effRadius(i) = tanDist(i) / tan(abs(turnAngle(i)) / 2);
    end
end

% ── 4) 生成"直线 -> 圆弧 -> 直线 -> 圆弧..."的采样点序列 ────────────────────
wx = [];
wy = [];
for i = 1:n
    iPrev = mod(i-2, n) + 1;
    iNext = mod(i, n) + 1;
    dIn  = (V(i,:)     - V(iPrev,:)); dIn  = dIn  / max(eps, norm(dIn));
    dOut = (V(iNext,:) - V(i,:));     dOut = dOut / max(eps, norm(dOut));

    Tin  = V(i,:) - dIn  * tanDist(i);
    Tout = V(i,:) + dOut * tanDist(i);

    % 直线段：上一个顶点的出弧切点 -> 本顶点的入弧切点
    wx(end+1) = Tin(1); %#ok<AGROW>
    wy(end+1) = Tin(2); %#ok<AGROW>

    if effRadius(i) > eps
        perp = [-dIn(2), dIn(1)] * sign(turnAngle(i));   % 转向内侧的法向
        center = Tin + effRadius(i) * perp;
        startAng = atan2(Tin(2)-center(2), Tin(1)-center(1));
        numArcPts = max(4, round(abs(turnAngle(i)) * 24));
        sweep = linspace(0, turnAngle(i), numArcPts);
        arcAng = startAng + sweep;
        wx = [wx, center(1) + effRadius(i)*cos(arcAng)]; %#ok<AGROW>
        wy = [wy, center(2) + effRadius(i)*sin(arcAng)]; %#ok<AGROW>
    else
        % 转角太小不倒角，直接经过顶点
        wx(end+1) = V(i,1); %#ok<AGROW>
        wy(end+1) = V(i,2); %#ok<AGROW>
    end

    wx(end+1) = Tout(1); %#ok<AGROW>
    wy(end+1) = Tout(2); %#ok<AGROW>
end

% ── 光栅化白线到 PNG（与 gen_simple_track_scene 相同的圆盘描边法）──────────
img = rasterizeLine(wx.', wy.', worldHalf, options.ImgSize, options.LineWidth);
imwrite(img, char(pngFile));

% ── 起始位姿：闭环上随机取一点，朝向沿切线方向（逆时针遍历）────────────────
i0  = randi(numel(wx));
sx  = wx(i0);  sy = wy(i0);
inx = mod(i0, numel(wx)) + 1;
tx  = wx(inx) - sx;  ty = wy(inx) - sy;
yaw = atan2(ty, tx);
quat = yawToQuat(yaw);
startPos = [sx, sy, 0.010];

% ── 写场景 XML（复用与 gen_simple_track_scene 相同的模板）──────────────────
xml = buildSceneXml(name, pngName, startPos, quat);
fid = fopen(char(sceneFile), "w");
assert(fid > 0, "GenComplexTrack:Write", "Cannot write scene: %s", sceneFile);
fwrite(fid, xml);
fclose(fid);

% ── 结果 ────────────────────────────────────────────────────────────────────
info = struct( ...
    "Name", name, "MapKey", mapKey, "SceneFile", string(sceneFile), ...
    "PngFile", string(pngFile), "StartPos", startPos, "StartQuat", quat, ...
    "Vertices", V, "CornerRadii", effRadius);

fprintf("Generated complex track '%s' (%d corners):\n", name, n);
fprintf("  scene : %s\n", sceneFile);
fprintf("  png   : %s\n", pngFile);
fprintf("  radii : "); fprintf("%.2f ", effRadius); fprintf("m\n");
fprintf("  start : pos=[%.3f %.3f %.3f] yaw=%.1f deg\n", ...
    startPos(1), startPos(2), startPos(3), rad2deg(yaw));
end


% ═══════════════════════════════════════════════════════════════════════════
function repoRoot = findRepoRoot(runtimeDir)
% 自 runtimeDir 逐级向上寻找包含 model/mujoco 的仓库根目录
current = string(runtimeDir);
for depth = 1:6
    current = fullfile(current, "..");
    if isfolder(fullfile(current, "model", "mujoco"))
        repoRoot = current;
        return
    end
end
repoRoot = fullfile(runtimeDir, "..", "..", "..");
end

% ═══════════════════════════════════════════════════════════════════════════
function angles = enforceMinAngleGap(angles, minGap)
% 已排序的角度序列若相邻(含首尾环绕)夹角小于 minGap，把后一个点顺时针方向
% 推开，避免对应边长趋于 0 导致倒角公式病态。
n = numel(angles);
for i = 1:n
    j = mod(i, n) + 1;
    gap = mod(angles(j) - angles(i), 2*pi);
    if i == n
        gap = 2*pi - mod(angles(i) - angles(1), 2*pi); %#ok<NASGU>
        continue  % 首尾环绕间隙由整体 2π 约束隐式保证，不单独调整避免震荡
    end
    if gap < minGap
        angles(j) = angles(i) + minGap;
    end
end
angles = mod(angles, 2*pi);
end

% ═══════════════════════════════════════════════════════════════════════════
function img = rasterizeLine(wx, wy, worldHalf, sz, lineW)
% 把世界坐标闭环白线画到 sz x sz 的深色 PNG（与 gen_simple_track_scene 同法）。
sz = round(sz);
bg = uint8(cat(3, 43*ones(sz), 43*ones(sz), 46*ones(sz)));  % ~#2b2b2e 深灰
u  = (wx + worldHalf) / (2*worldHalf);
v  = (wy + worldHalf) / (2*worldHalf);
px = u * sz;
py = (1 - v) * sz;
mask = false(sz, sz);
half = max(1, round(lineW/2));
for k = 1:numel(px)-1
    mask = stampSegment(mask, px(k), py(k), px(k+1), py(k+1), half, sz);
end
mask = stampSegment(mask, px(end), py(end), px(1), py(1), half, sz);  % 闭合
img = bg;
white = uint8(235);
R = img(:,:,1); G = img(:,:,2); B = img(:,:,3);
R(mask) = white; G(mask) = white; B(mask) = white;
img = cat(3, R, G, B);
end

function mask = stampSegment(mask, x1, y1, x2, y2, half, sz)
d = hypot(x2-x1, y2-y1);
ns = max(2, ceil(d));
xs = round(linspace(x1, x2, ns));
ys = round(linspace(y1, y2, ns));
for j = 1:ns
    mask = stampDisk(mask, xs(j), ys(j), half, sz);
end
end

function mask = stampDisk(mask, cx, cy, r, sz)
x0 = max(1, cx-r); x1 = min(sz, cx+r);
y0 = max(1, cy-r); y1 = min(sz, cy+r);
if x1 < x0 || y1 < y0, return; end
[xg, yg] = meshgrid(x0:x1, y0:y1);
inside = (xg-cx).^2 + (yg-cy).^2 <= r^2;
sub = mask(y0:y1, x0:x1);
sub(inside) = true;
mask(y0:y1, x0:x1) = sub;
end

% ═══════════════════════════════════════════════════════════════════════════
function q = yawToQuat(yaw)
q = [cos(yaw/2), 0, 0, sin(yaw/2)];
end

% ═══════════════════════════════════════════════════════════════════════════
function xml = buildSceneXml(name, pngName, startPos, quat)
q = sprintf("%.10f %.10f %.10f %.10f", quat(1), quat(2), quat(3), quat(4));
p = sprintf("%.6f %.6f %.6f", startPos(1), startPos(2), startPos(3));
xml = sprintf([ ...
'<mujoco model="%s_turtlebot3_burger_visual_scene">\n' ...
'  <compiler angle="radian" meshdir="../shared/assets/meshes" texturedir="../shared/assets/textures" autolimits="true" />\n' ...
'  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" iterations="50" />\n' ...
'  <visual>\n' ...
'    <global offwidth="640" offheight="480" />\n' ...
'    <quality shadowsize="1024" />\n' ...
'    <headlight ambient="0.40 0.40 0.40" diffuse="0.90 0.90 0.90" specular="0.10 0.10 0.10" />\n' ...
'  </visual>\n' ...
'  <asset>\n' ...
'    <texture name="%s_texture" type="2d" file="%s" />\n' ...
'    <material name="%s_material" texture="%s_texture" texrepeat="1 1" texuniform="false" rgba="1 1 1 1" reflectance="0" />\n' ...
'    <mesh name="simple_camera_track_surface" file="simple_camera_track_surface.obj" />\n' ...
'    <include file="turtlebot3_burger_vehicle_assets.xml" />\n' ...
'  </asset>\n' ...
'  <worldbody>\n' ...
'    <light name="overhead_a" pos="-1.0 0 3.5" dir="0.3 0 -1" diffuse="0.95 0.95 0.95" specular="0.05 0.05 0.05" />\n' ...
'    <light name="overhead_b" pos=" 1.0 0 3.5" dir="-0.3 0 -1" diffuse="0.95 0.95 0.95" specular="0.05 0.05 0.05" />\n' ...
'    <geom name="%s_visual_surface" type="mesh" mesh="simple_camera_track_surface" material="%s_material" contype="0" conaffinity="0" group="1" />\n' ...
'    <geom name="%s_floor" type="box" pos="0 0 -0.005" size="2.2 2.2 0.005" rgba="0.04 0.04 0.05 1" friction="1.0 0.005 0.0001" />\n' ...
'    <body name="turtlebot3" pos="%s" quat="%s">\n' ...
'      <include file="turtlebot3_burger_vehicle_body.xml" />\n' ...
'    </body>\n' ...
'  </worldbody>\n' ...
'  <actuator>\n' ...
'    <include file="turtlebot3_burger_vehicle_actuators.xml" />\n' ...
'  </actuator>\n' ...
'  <sensor>\n' ...
'    <include file="turtlebot3_burger_vehicle_sensors.xml" />\n' ...
'  </sensor>\n' ...
'</mujoco>\n'], ...
name, name, pngName, name, name, name, name, name, p, q);
end
