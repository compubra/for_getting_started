function info = gen_simple_track_scene(options)
%GEN_SIMPLE_TRACK_SCENE Procedurally generate a random simple closed-loop track
%as a MuJoCo scene (PNG line texture + scene XML + start pose on the line).
%
%   为相机循线仿真程序化生成一张随机的「简单赛道」：一个平滑闭合白线环
%   （椭圆或胶囊形，随机尺寸/旋转），画成 PNG 贴在共享地面网格上，并算出
%   机器人在白线上的起始位姿，写出可直接被 LSAC 仿真器加载的场景 XML。
%
%   赛道白线是画在深色地面上的 PNG 纹理（同项目其它赛道），机器人前置相机
%   看这条白线循迹——不是往仿真里喂 [x;y] 路径点。
%
%   Name-value options
%   ------------------
%     Name        (1,1) string  场景基名（决定文件名和 map key）  default 随机 "simple_rand_XXXX"
%     Shape       (1,1) string  "ellipse" | "capsule" | "random"  default "random"
%     Seed                      rng 种子，[]=不设                  default []
%     ImgSize     (1,1) double  PNG 边长(px)                       default 512
%     LineWidth   (1,1) double  白线线宽(px)                       default 26
%     Difficulty  (1,1) double  0..1，越大环越小越弯               default 0.4
%     OutModelDir (1,1) string  场景 XML 输出目录                  default 项目 turtlebot3 目录
%     OutTexDir   (1,1) string  PNG 输出目录                       default 项目 textures 目录
%     Overwrite   (1,1) logical 允许覆盖同名文件                   default true
%
%   Returns  info struct:
%     .Name .MapKey .SceneFile .PngFile .Shape
%     .StartPos [x y z]  .StartQuat [w x y z]  .Semi [a b]  .Center [cx cy]
%
%   Example
%   -------
%     info = gen_simple_track_scene("Seed", 1);
%     % 直接喂给 LSAC 仿真器：
%     set_turtlebot3_mujoco_scene("visual_line_follower_sac_lyapunov", info.SceneFile);
%     sim("visual_line_follower_sac_lyapunov");
%
%   See also SET_TURTLEBOT3_MUJOCO_SCENE, GEN_RANDOM_LOCAL_PATH.

arguments
    options.Name        (1,1) string  = ""
    options.Shape       (1,1) string  = "random"
    options.Seed                      = []
    options.ImgSize     (1,1) double {mustBePositive} = 512
    options.LineWidth   (1,1) double {mustBePositive} = 4
    options.Difficulty  (1,1) double {mustBeGreaterThanOrEqual(options.Difficulty,0), mustBeLessThanOrEqual(options.Difficulty,1)} = 0.4
    options.OutModelDir (1,1) string  = ""
    options.OutTexDir   (1,1) string  = ""
    options.Overwrite   (1,1) logical = true
end

if ~isempty(options.Seed)
    rng(options.Seed);
end

% ── 目录 ────────────────────────────────────────────────────────────────────
runtimeDir = string(fileparts(mfilename("fullpath")));
% matlab 目录在不同检出中的深度不同（src/simple_camera_pid/matlab 上溯 3 层，
% src/matlab 上溯 2 层），故自 runtimeDir 逐级向上搜索 model/mujoco 定位仓库根
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
assert(isfolder(modelDir), "GenTrack:BadDir", "Scene dir not found: %s", modelDir);
assert(isfolder(texDir),   "GenTrack:BadDir", "Texture dir not found: %s", texDir);

% ── 名字与 map key ──────────────────────────────────────────────────────────
name = options.Name;
if strlength(name) == 0
    name = "simple_rand_" + string(dec2hex(randi([0 65535]), 4));
end
mapKey    = lower(name);
pngName   = name + "_track.png";
sceneName = name + "_turtlebot3_burger_visual_scene.xml";
pngFile   = fullfile(texDir, pngName);
sceneFile = fullfile(modelDir, sceneName);
if ~options.Overwrite
    assert(~isfile(sceneFile), "GenTrack:Exists", "Scene exists: %s", sceneFile);
end

% ── 几何：随机闭环参数 ──────────────────────────────────────────────────────
% 地面 mesh 是 4.4x4.4 m（世界 [-2.2,2.2]），UV 0..1 覆盖全域。相机循线只需
% 环落在中心区域即可（半轴 ~0.9..1.5 m，随难度缩小并留边距）。
shape = resolveShape(options.Shape);
worldHalf = 2.2;                      % 地面半宽 (m)
margin    = 0.55;                     % 环到边界的安全余量 (m)
maxSemi   = worldHalf - margin;       % 半轴上限 ~1.65 m
% 难度越高环越小（转弯更急）：a 在 [0.9, maxSemi] 内随难度线性缩小
aHi = maxSemi;  aLo = 0.9;
a   = aHi - (aHi - aLo) * options.Difficulty * (0.6 + 0.4*rand());
b   = a * (0.62 + 0.33*rand());       % 短轴/长轴比 0.62..0.95
theta = (2*rand()-1) * pi;            % 随机整体旋转
% 随机中心偏移，但保证旋转后仍在边界内
cmax = worldHalf - margin - max(a,b);
cmax = max(0, cmax);
cx = (2*rand()-1) * cmax * 0.6;
cy = (2*rand()-1) * cmax * 0.6;

% 参数化闭环（胶囊= 直段+半圆端；椭圆= 纯椭圆）
tt = linspace(0, 2*pi, 720).';
if shape == "capsule"
    [lx, ly] = capsuleLoop(a, b, numel(tt));
else
    lx = a * cos(tt);
    ly = b * sin(tt);
end
% 旋转 + 平移到世界坐标
R  = [cos(theta) -sin(theta); sin(theta) cos(theta)];
P  = (R * [lx.'; ly.']).';
wx = P(:,1) + cx;
wy = P(:,2) + cy;

% ── 光栅化白线到 PNG ────────────────────────────────────────────────────────
img = rasterizeLine(wx, wy, worldHalf, options.ImgSize, options.LineWidth);
imwrite(img, char(pngFile));

% ── 起始位姿：环上随机取一点，朝切线方向（逆时针）──────────────────────────
i0  = randi(numel(wx));
sx  = wx(i0);  sy = wy(i0);
% 切线方向（沿参数增大方向）
inx = mod(i0, numel(wx)) + 1;
tx  = wx(inx) - sx;  ty = wy(inx) - sy;
yaw = atan2(ty, tx);
quat = yawToQuat(yaw);
startPos = [sx, sy, 0.010];

% ── 写场景 XML ──────────────────────────────────────────────────────────────
xml = buildSceneXml(name, pngName, startPos, quat);
fid = fopen(char(sceneFile), "w");
assert(fid > 0, "GenTrack:Write", "Cannot write scene: %s", sceneFile);
fwrite(fid, xml);
fclose(fid);

% ── 结果 ────────────────────────────────────────────────────────────────────
info = struct( ...
    "Name", name, "MapKey", mapKey, "SceneFile", string(sceneFile), ...
    "PngFile", string(pngFile), "Shape", shape, ...
    "StartPos", startPos, "StartQuat", quat, ...
    "Semi", [a b], "Center", [cx cy]);

fprintf("Generated simple track '%s' (%s):\n", name, shape);
fprintf("  scene : %s\n", sceneFile);
fprintf("  png   : %s\n", pngFile);
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
% 未找到时退回旧版固定四层假设，保持原有报错路径可读
repoRoot = fullfile(runtimeDir, "..", "..", "..", "..");
end

% ═══════════════════════════════════════════════════════════════════════════
function shape = resolveShape(s)
s = lower(strip(s));
if s == "random"
    opts = ["ellipse","capsule"];
    shape = opts(randi(2));
elseif any(s == ["ellipse","capsule"])
    shape = s;
else
    error("GenTrack:BadShape", "Shape must be ellipse|capsule|random, got '%s'.", s);
end
end

% ═══════════════════════════════════════════════════════════════════════════
function [lx, ly] = capsuleLoop(a, b, npts)
% 胶囊环：两段直线 + 两个半圆端，参数均匀分布。a=总半长, b=半宽。
r = b;                                 % 端部半圆半径 = 半宽
sx = max(0.05, a - r);                 % 直段半长（由 a 决定，忽略传入 straight）
% 四段：上直段(-sx->sx), 右半圆, 下直段(sx->-sx), 左半圆
seg = round(npts/4);
% 上直段 y=+r
t1x = linspace(-sx,  sx, seg).';  t1y =  r*ones(seg,1);
% 右半圆 中心(sx,0) 从 +pi/2 到 -pi/2
ph  = linspace(pi/2, -pi/2, seg).';   t2x = sx + r*cos(ph);  t2y = r*sin(ph);
% 下直段 y=-r
t3x = linspace( sx, -sx, seg).';  t3y = -r*ones(seg,1);
% 左半圆 中心(-sx,0) 从 -pi/2 到 -3pi/2
ph2 = linspace(-pi/2, -3*pi/2, seg).'; t4x = -sx + r*cos(ph2); t4y = r*sin(ph2);
lx = [t1x; t2x; t3x; t4x];
ly = [t1y; t2y; t3y; t4y];
end

% ═══════════════════════════════════════════════════════════════════════════
function img = rasterizeLine(wx, wy, worldHalf, sz, lineW)
% 把世界坐标闭环白线画到 sz x sz 的深色 PNG。
% 世界->UV: u=(x+H)/(2H), v=(y+H)/(2H)；UV->像素: px=u*sz, py=(1-v)*sz。
sz = round(sz);
bg = uint8(cat(3, 43*ones(sz), 43*ones(sz), 46*ones(sz)));  % ~#2b2b2e 深灰
% 世界 -> 像素
u  = (wx + worldHalf) / (2*worldHalf);
v  = (wy + worldHalf) / (2*worldHalf);
px = u * sz;
py = (1 - v) * sz;
% 用一张 double mask 累积线条，再阈值化，保证闭合无缝
mask = false(sz, sz);
half = max(1, round(lineW/2));
% 沿折线逐段插值打点（稠密），每点画一个圆盘
for k = 1:numel(px)-1
    x1=px(k); y1=py(k); x2=px(k+1); y2=py(k+1);
    d = hypot(x2-x1, y2-y1);
    ns = max(2, ceil(d));
    xs = round(linspace(x1, x2, ns));
    ys = round(linspace(y1, y2, ns));
    for j = 1:ns
        mask = stampDisk(mask, xs(j), ys(j), half, sz);
    end
end
% 闭合最后一段到起点
mask = stampSegment(mask, px(end), py(end), px(1), py(1), half, sz);
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
% 在 (cx,cy) 画半径 r 的实心圆盘（像素坐标，行=y 列=x）。
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
% 绕 Z 轴 yaw 的四元数 [w x y z]。
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
