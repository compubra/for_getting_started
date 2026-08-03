function centerline = get_track_centerline(sceneOrPng, options)
%GET_TRACK_CENTERLINE Extract a track's centreline in world metres.
%
%   centerline = get_track_centerline(sceneXmlOrPng) returns an N x 2
%   [x y] polyline (world metres, N = options.NumPoints) tracing the white
%   line of a camera-track scene. Works for every track in this project —
%   fixed maps (simple/complex/ellipse/training) and per-episode generated
%   ones — because they all share the same convention: a white line drawn
%   on a dark PNG texture mapped onto the 4.4 x 4.4 m ground surface
%   (world x,y in [-2.2, 2.2], UV covers the full plane).
%
%   训练端真值奖励/出界终止用的赛道中心线。从场景 XML 找到贴图 PNG，
%   提取白线骨架、按最近邻排序成有序点列，再按弧长重采样成固定 N 点。
%   固定 N 保证模型工作区变量 RL_TrackCenterline 尺寸不随地图变化，
%   避免逐回合换图时触发 Simulink 重编译尺寸检查。
%
%   Input
%   -----
%     sceneOrPng   scene XML path (texture resolved from <texture file=...>)
%                  or a PNG path directly.
%
%   Name-value options
%   ------------------
%     NumPoints      (1,1) double  output polyline length      default 600
%     WorldHalf      (1,1) double  ground half-extent (m)      default 2.2
%     WhiteThreshold (1,1) double  grayscale white cut (0-255) default 180
%
%   See also RL_TRUE_TRACK_ERRORS, GEN_SIMPLE_TRACK_SCENE,
%   SET_TURTLEBOT3_MUJOCO_SCENE.

arguments
    sceneOrPng (1,1) string
    options.NumPoints      (1,1) double {mustBePositive} = 600
    options.WorldHalf      (1,1) double {mustBePositive} = 2.2
    options.WhiteThreshold (1,1) double = 180
end

pngFile = resolveTexture(sceneOrPng);

% Memoize by PNG path + modification time: the model InitFcn refreshes the
% centreline at every simulation start (every training episode), so fixed
% maps must be free after the first call. Regenerated tracks reuse the same
% filename but get a new mtime -> cache miss -> recompute (correct).
persistent cacheKey cacheValue
if isempty(cacheKey)
    cacheKey = strings(0, 1);
    cacheValue = {};
end
d = dir(char(pngFile));
key = string(pngFile) + "|" + string(d(1).datenum) + "|" + ...
    options.NumPoints + "|" + options.WorldHalf + "|" + ...
    options.WhiteThreshold;
if ~isempty(cacheKey) && any(cacheKey == key)
    centerline = cacheValue{cacheKey == key};
    return
end

img = imread(char(pngFile));
if size(img, 3) > 1
    gray = max(img, [], 3);   % white on dark: max channel is robust to tint
else
    gray = img;
end
mask = double(gray) >= options.WhiteThreshold;
assert(any(mask, "all"), "Centerline:NoLine", ...
    "No white line found in texture: %s", pngFile);

% Largest connected component only (mirrors the vision pipeline: drop
% reflections / secondary marks).
comps = bwconncomp(mask, 8);
if comps.NumObjects > 1
    numPixels = cellfun(@numel, comps.PixelIdxList);
    [~, largest] = max(numPixels);
    mask = false(size(mask));
    mask(comps.PixelIdxList{largest}) = true;
end

% Thin to a 1-px skeleton, then order the pixels into a chain by greedy
% nearest-neighbour (same approach as originbot_local_path_generator, but
% offline and for a closed loop). MinBranchLength prunes the short spurs
% bwskel grows where the drawn line bulges (verified necessary on the
% 1500 px fixed maps; without it the chain dead-ends in a spur).
skeleton = bwskel(mask, "MinBranchLength", 60);
[rows, cols] = find(skeleton);
assert(numel(rows) >= 8, "Centerline:TooShort", ...
    "Track skeleton too short (%d px) in %s", numel(rows), pngFile);
ordered = greedyOrder(cols, rows);
coverage = size(ordered, 1) / numel(rows);
if coverage < 0.7
    warning("Centerline:LowCoverage", ...
        "Ordered chain covers only %.0f%% of the skeleton (%s); " + ...
        "centreline may be truncated at a branch.", 100 * coverage, pngFile);
end

% Pixel -> world. Inverse of gen_simple_track_scene/rasterizeLine:
%   px = (x+H)/(2H) * sz ,  py = (1 - (y+H)/(2H)) * sz
sz = size(mask);
H  = options.WorldHalf;
x  = (ordered(:, 1) ./ sz(2)) * (2 * H) - H;
y  = (1 - ordered(:, 2) ./ sz(1)) * (2 * H) - H;

% Arc-length resample to a fixed point count. The chain is a closed loop,
% so close it explicitly before resampling for even spacing.
x(end + 1) = x(1);
y(end + 1) = y(1);
seg = hypot(diff(x), diff(y));
keep = [true; seg > eps];          % drop duplicate pixels
x = x(keep);  y = y(keep);
s = [0; cumsum(hypot(diff(x), diff(y)))];
targets = linspace(0, s(end), options.NumPoints + 1).';
targets(end) = [];                 % closed loop: omit duplicated end point
centerline = [interp1(s, x, targets), interp1(s, y, targets)];

% Store in the memo cache (bounded: keep the most recent 16 tracks).
cacheKey(end + 1) = key;
cacheValue{end + 1} = centerline;
if numel(cacheKey) > 16
    cacheKey(1) = [];
    cacheValue(1) = [];
end
end


% ═══════════════════════════════════════════════════════════════════════════
function pngFile = resolveTexture(sceneOrPng)
% Scene XML -> texture PNG path (or pass a PNG straight through).
if endsWith(lower(sceneOrPng), ".png")
    pngFile = sceneOrPng;
else
    xmlText = string(fileread(char(sceneOrPng)));
    texMatch = regexp(xmlText, '<texture[^>]*file="([^"]+)"', ...
        "tokens", "once");
    assert(~isempty(texMatch), "Centerline:NoTexture", ...
        "No <texture file=...> in scene: %s", sceneOrPng);
    texName = string(texMatch{1});
    dirMatch = regexp(xmlText, 'texturedir="([^"]+)"', "tokens", "once");
    sceneDir = string(fileparts(char(sceneOrPng)));
    if isempty(dirMatch)
        pngFile = fullfile(sceneDir, texName);
    else
        pngFile = fullfile(sceneDir, string(dirMatch{1}), texName);
    end
end
assert(isfile(pngFile), "Centerline:NoPng", ...
    "Track texture not found: %s", pngFile);
end


% ═══════════════════════════════════════════════════════════════════════════
function ordered = greedyOrder(cols, rows)
% Greedy nearest-neighbour chain over skeleton pixels. Starting from the
% leftmost pixel (an extreme point is always ON the loop, never inside a
% residual branch) lets the chain run the whole loop in one direction.
% maxJump truncates at remaining skeleton branches so only the contiguous
% main loop is kept — same rationale as the online vision detector.
maxJumpSq = 40 ^ 2;
n = numel(cols);
ordered = zeros(n, 2);
visited = false(n, 1);
[~, current] = min(cols);
for step = 1:n
    ordered(step, :) = [cols(current), rows(current)];
    visited(current) = true;
    dx = cols - cols(current);
    dy = rows - rows(current);
    distSq = dx .^ 2 + dy .^ 2;
    distSq(visited) = inf;
    [minDist, nextIdx] = min(distSq);
    if isinf(minDist) || minDist > maxJumpSq
        ordered = ordered(1:step, :);
        return
    end
    current = nextIdx;
end
end
