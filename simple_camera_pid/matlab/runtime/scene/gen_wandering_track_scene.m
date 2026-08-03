function info = gen_wandering_track_scene(options)
%GEN_WANDERING_TRACK_SCENE Procedurally generate a smooth closed-loop track
%by spline-interpolating a random radial profile (MuJoCo scene).
%
%   把原先的孤儿工具 gen_random_local_path.m（随机控制点 + 三次样条平滑，
%   产出一条不闭合的开放曲线，只用于离线测试拟合数学、不接触 MuJoCo）改造
%   成能真正下场跑仿真的**闭环**赛道生成器：同样是"随机控制点 + 样条平滑"
%   的核心技巧，但控制点不再沿前向 X 单调排布，而是绕圆心按角度分布、
%   随机扰动的是每个角度上的半径；样条改成首尾周期延拓（把首尾若干控制点
%   各自搬到 -2π / +2π 外做 padding 再插值），保证闭合处平滑无接缝，不是
%   硬连接。
%
%   与另外两个赛道生成器的形状特征对比：
%     gen_simple_track_scene  —— 椭圆 / 胶囊，规整几何体
%     gen_complex_track_scene —— 折线+独立随机圆弧倒角，带尖角的多边形观感
%     gen_wandering_track_scene（本文件）—— 处处光滑的有机蜿蜒闭环，没有
%       尖角也没有直线段，弯曲程度沿途连续变化，形状最不规则
%
%   算法（4 步）：
%   1) NumControl 个控制角度绕圆心等分（不含随机抖动——抖动会破坏角度到
%      弧长的均匀映射，这里只让半径随机，形状的不规则性完全来自半径扰动）。
%   2) 每个控制角度的半径 = 基准半径 + 随机偏移，偏移幅度 amp 由 Difficulty
%      非线性决定（0.15+0.85*Difficulty^2，与 gen_random_local_path 完全
%      相同的映射），且做与 gen_random_local_path 相同的"归一化到目标幅度"
%      处理：抽完随机偏移后整体缩放，让最大绝对偏移精确命中 amp，避免独立
%      随机抵消导致低难度时和高难度时看起来差不多。
%   3) 首尾各取 3 个控制点搬到 -2π/+2π 外做周期延拓，再用三次样条对
%      "角度 -> 半径" 插值出 720 个采样点——延拓让样条在 0/2π 接缝处看到
%      两侧邻居，闭合处天然平滑，不会有可见的斜率突变。
%   4) 半径>0 的下限保护后转回 (X,Y) 世界坐标，光栅化成白线 PNG，闭合是
%      角度参数化天然保证的（0 到 2π 首尾相接），不需要额外拼接。
%
%   Name-value options
%   ------------------
%     Name         (1,1) string  场景基名                        default 随机
%     NumControl   (1,1) double  控制角度数（越多越蜿蜒曲折）      default 6，最少 4
%     Seed                       rng 种子，[]=不设                default []
%     ImgSize      (1,1) double  PNG 边长(px)                    default 512
%     LineWidth    (1,1) double  白线线宽(px)                    default 12
%     Difficulty   (1,1) double  0..1，越大半径扰动幅度越大        default 0.5
%     BaseRadius   (1,1) double  基准半径(m)，[]=按世界尺寸自动取  default []
%     Overwrite/OutModelDir/OutTexDir  同 gen_simple_track_scene
%
%   Returns info struct 字段与 gen_complex_track_scene 一致（无 CornerRadii，
%   改为 .ControlAngles .ControlRadii 便于检查随机采样结果）。
%
%   Example
%   -------
%     info = gen_wandering_track_scene("Seed", 3, "NumControl", 8, "Difficulty", 0.7);
%
%   See also GEN_SIMPLE_TRACK_SCENE, GEN_COMPLEX_TRACK_SCENE, GEN_RANDOM_LOCAL_PATH.

arguments
    options.Name        (1,1) string = ""
    options.NumControl  (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.NumControl,4)} = 6
    options.Seed                     = []
    options.ImgSize     (1,1) double {mustBePositive} = 512
    options.LineWidth   (1,1) double {mustBePositive} = 4
    options.Difficulty  (1,1) double {mustBeGreaterThanOrEqual(options.Difficulty,0), mustBeLessThanOrEqual(options.Difficulty,1)} = 0.5
    options.BaseRadius              = []
    options.OutModelDir (1,1) string = ""
    options.OutTexDir   (1,1) string = ""
    options.Overwrite   (1,1) logical = true
end

if ~isempty(options.Seed)
    rng(options.Seed);
end

% ── 目录：自适应向上搜索 model/mujoco，与另外两个生成器一致 ────────────────
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
assert(isfolder(modelDir), "GenWanderingTrack:BadDir", "Scene dir not found: %s", modelDir);
assert(isfolder(texDir),   "GenWanderingTrack:BadDir", "Texture dir not found: %s", texDir);

name = options.Name;
if strlength(name) == 0
    name = "wandering_rand_" + string(dec2hex(randi([0 65535]), 4));
end
mapKey    = lower(name);
pngName   = name + "_track.png";
sceneName = name + "_turtlebot3_burger_visual_scene.xml";
pngFile   = fullfile(texDir, pngName);
sceneFile = fullfile(modelDir, sceneName);
if ~options.Overwrite
    assert(~isfile(sceneFile), "GenWanderingTrack:Exists", "Scene exists: %s", sceneFile);
end

% ── 1)+2) 控制角度等分，半径随机扰动（与 gen_random_local_path 相同的
% amp/归一化技巧，只是把"沿 X 的横向偏移"换成"绕圆心的半径偏移"）────────────
worldHalf = 2.2;
margin    = 0.55;
maxR      = worldHalf - margin;
if isempty(options.BaseRadius)
    baseRadius = maxR * (0.55 + 0.20*rand());
else
    baseRadius = double(options.BaseRadius);
end
amp = baseRadius * 0.5 * (0.15 + 0.85 * options.Difficulty^2);   % 幅度随难度非线性放大

nc = options.NumControl;
controlAngles = (0:nc-1) * (2*pi/nc);
radialOffset = (2*rand(1,nc) - 1) * amp;
peak = max(abs(radialOffset));
if peak > eps
    radialOffset = radialOffset * (amp / peak);   % 归一化命中目标幅度，避免随机抵消
end
controlRadii = max(0.3, baseRadius + radialOffset);   % 半径下限保护，防止塌缩自交

% ── 3) 周期延拓 + 三次样条插值，闭合处平滑无接缝 ────────────────────────────
padN = min(3, nc-1);
anglesExt = [controlAngles(end-padN+1:end) - 2*pi, controlAngles, ...
             controlAngles(1:padN) + 2*pi];
radiiExt  = [controlRadii(end-padN+1:end), controlRadii, controlRadii(1:padN)];

numSamples = 720;
theta = linspace(0, 2*pi, numSamples+1);
theta(end) = [];   % 避免 0 和 2π 重复采样同一点
rSample = spline(anglesExt, radiiExt, theta);
rSample = max(0.3, rSample);   % 插值可能轻微越过控制点范围，兜底防负/防塌缩

% ── 4) 转回世界坐标（随机整体旋转，增加多样性），闭合天然成立 ──────────────
theta0 = (2*rand()-1) * pi;
wx = rSample .* cos(theta + theta0);
wy = rSample .* sin(theta + theta0);

% ── 光栅化白线到 PNG（与另外两个生成器相同的圆盘描边法）────────────────────
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

% ── 写场景 XML（与另外两个生成器相同的模板）─────────────────────────────────
xml = buildSceneXml(name, pngName, startPos, quat);
fid = fopen(char(sceneFile), "w");
assert(fid > 0, "GenWanderingTrack:Write", "Cannot write scene: %s", sceneFile);
fwrite(fid, xml);
fclose(fid);

% ── 结果 ────────────────────────────────────────────────────────────────────
info = struct( ...
    "Name", name, "MapKey", mapKey, "SceneFile", string(sceneFile), ...
    "PngFile", string(pngFile), "StartPos", startPos, "StartQuat", quat, ...
    "ControlAngles", controlAngles, "ControlRadii", controlRadii);

fprintf("Generated wandering track '%s' (%d control points):\n", name, nc);
fprintf("  scene : %s\n", sceneFile);
fprintf("  png   : %s\n", pngFile);
fprintf("  radii : "); fprintf("%.2f ", controlRadii); fprintf("m (base=%.2f, amp=%.2f)\n", baseRadius, amp);
fprintf("  start : pos=[%.3f %.3f %.3f] yaw=%.1f deg\n", ...
    startPos(1), startPos(2), startPos(3), rad2deg(yaw));
end


% ═══════════════════════════════════════════════════════════════════════════
function repoRoot = findRepoRoot(runtimeDir)
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
function img = rasterizeLine(wx, wy, worldHalf, sz, lineW)
sz = round(sz);
bg = uint8(cat(3, 43*ones(sz), 43*ones(sz), 46*ones(sz)));
u  = (wx + worldHalf) / (2*worldHalf);
v  = (wy + worldHalf) / (2*worldHalf);
px = u * sz;
py = (1 - v) * sz;
mask = false(sz, sz);
half = max(1, round(lineW/2));
for k = 1:numel(px)-1
    mask = stampSegment(mask, px(k), py(k), px(k+1), py(k+1), half, sz);
end
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
