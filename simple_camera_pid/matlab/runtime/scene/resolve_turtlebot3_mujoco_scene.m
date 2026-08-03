function [sceneFile, mapKey, mapDisplayName] = ...
    resolve_turtlebot3_mujoco_scene(model, modelDir, requestedMap, ...
    requestedScene)
%RESOLVE_TURTLEBOT3_MUJOCO_SCENE Resolve the MJCF scene used by the model.
%
% Selection priority:
%   1. requestedScene argument
%   2. TURTLEBOT3_MUJOCO_SCENE environment variable
%   3. TurtleBot3MuJoCoScene model/base-workspace variable
%   4. requestedMap argument
%   5. TURTLEBOT3_MUJOCO_MAP environment variable
%   6. the MuJoCo Plant's current xmlFile if it maps to a known scene
%   7. TurtleBot3MuJoCoMap model/base-workspace variable
%   8. the simple TurtleBot3 camera track (fallback)
%
% Priority 6 sits ABOVE the workspace map variable so that double-clicking
% the MuJoCo Plant block in Simulink and changing its xmlFile is honoured at
% run time, instead of being overwritten back to the workspace variable's
% map. Explicit arguments / environment variables still win over both.

arguments
    model (1, 1) string
    modelDir (1, 1) string
    requestedMap (1, 1) string = ""
    requestedScene (1, 1) string = ""
end

[mapKeys, sceneNames, displayNames] = knownScenes();
% matlab 目录在不同检出中的深度不同（src/simple_camera_pid/matlab 上溯 3 层，
% src/matlab 上溯 2 层），故自 modelDir 逐级向上搜索 model/mujoco 定位资源根
assetRoot = findAssetRoot(modelDir);
% turtlebot3Dir：新版场景 XML 统一放置目录
turtlebot3Dir = fullfile(assetRoot, "model", "mujoco", "turtlebot3");
% legacyMujocoDir：兼容旧版直接放在 mujoco/ 下的场景文件
legacyMujocoDir = fullfile(assetRoot, "model", "mujoco");

% 优先级 1-3：直接场景路径（参数 → 环境变量 → 模型/Base 工作区变量）
requestedScene = firstNonempty( ...
    requestedScene, ...
    string(getenv("TURTLEBOT3_MUJOCO_SCENE")), ...
    workspaceString(model, "TurtleBot3MuJoCoScene"));
if strlength(requestedScene) > 0
    [sceneFile, mapKey, mapDisplayName] = resolveSceneFile( ...
        requestedScene, turtlebot3Dir, legacyMujocoDir, modelDir, ...
        mapKeys, sceneNames, displayNames);
    return
end

% 优先级 4-5：地图键名（仅 参数 → 环境变量；显式请求优先级最高）
requestedMap = firstNonempty( ...
    requestedMap, ...
    string(getenv("TURTLEBOT3_MUJOCO_MAP")));
if strlength(requestedMap) > 0
    [sceneFile, mapKey, mapDisplayName] = resolveMapKey( ...
        requestedMap, turtlebot3Dir, mapKeys, sceneNames, displayNames);
    return
end

% 优先级 6：MuJoCo Plant 当前 xmlFile（尊重用户在 Simulink 中双击 Plant 手动
% 改的地图——若它指向一个存在的已知场景，则采用它，不被工作区变量覆盖）
currentScene = currentPlantScene(model);
if strlength(currentScene) > 0
    try
        [sceneFile, mapKey, mapDisplayName] = resolveSceneFile( ...
            currentScene, turtlebot3Dir, legacyMujocoDir, modelDir, ...
            mapKeys, sceneNames, displayNames);
        return
    catch
        % Plant 中保存的路径已失效（文件被移动/删除），继续向下回退
    end
end

% 优先级 7：模型/Base 工作区变量 TurtleBot3MuJoCoMap（脚本/持久化的选择）
requestedMap = workspaceString(model, "TurtleBot3MuJoCoMap");
if strlength(requestedMap) > 0
    [sceneFile, mapKey, mapDisplayName] = resolveMapKey( ...
        requestedMap, turtlebot3Dir, mapKeys, sceneNames, displayNames);
    return
end

% 优先级 8（最终兜底）：simple 直线赛道
[sceneFile, mapKey, mapDisplayName] = resolveMapKey( ...
    "simple", turtlebot3Dir, mapKeys, sceneNames, displayNames);
end

function assetRoot = findAssetRoot(modelDir)
% 自 modelDir 逐级向上寻找包含 model/mujoco 的项目根目录
current = string(modelDir);
for depth = 1:6
    current = fullfile(current, "..");
    if isfolder(fullfile(current, "model", "mujoco"))
        assetRoot = current;
        return
    end
end
% 未找到时退回旧版固定三层假设，保持原有报错路径可读
assetRoot = fullfile(modelDir, "..", "..", "..");
end

function value = firstNonempty(varargin)
% 返回参数列表中第一个非空字符串，用于实现多级优先级覆盖
value = "";
for k = 1:nargin
    candidate = strip(string(varargin{k}));
    if strlength(candidate) > 0
        value = candidate;
        return
    end
end
end

function value = workspaceString(model, variableName)
% 先查模型工作区，再查 Base 工作区，均不存在时返回空串
value = "";
try
    workspace = get_param(model, "ModelWorkspace");
    variables = string({workspace.whos.name});
    if any(variables == variableName)
        value = strip(string(workspace.evalin(variableName)));
        if strlength(value) > 0
            return
        end
    end
catch
    value = "";
end

try
    if evalin("base", "exist('" + variableName + "', 'var')")
        value = strip(string(evalin("base", variableName)));
    end
catch
    value = "";
end
end

function scene = currentPlantScene(model)
% SID 16 是模型中 MuJoCo Plant 块的固定 Simulink ID
try
    plant = Simulink.ID.getFullName(model + ":16");
    scene = strip(string(get_param(plant, "xmlFile")));
catch
    scene = "";
end
end

function [mapKeys, sceneNames, displayNames] = knownScenes()
% 内置地图注册表：mapKey → XML 文件名 → 显示名称
mapKeys = ["simple", "complex", "ellipse", "training", "track_easy", "track_medium", "track_hard", "winding"];
sceneNames = [
    "simple_camera_track_turtlebot3_burger_visual_scene.xml"
    "complex_camera_track_turtlebot3_burger_visual_scene.xml"
    "ellipse_camera_track_turtlebot3_burger_visual_scene.xml"
    "training_oval_track_turtlebot3_burger_visual_scene.xml"
    "track_easy_turtlebot3_burger_visual_scene.xml"
    "track_medium_turtlebot3_burger_visual_scene.xml"
    "track_hard_turtlebot3_burger_visual_scene.xml"
    "winding_camera_track_turtlebot3_burger_visual_scene.xml"
    ];
displayNames = [
    "Simple camera track"
    "Complex chicane"
    "Ellipse camera track"
    "Training oval (compact, RL-optimised)"
    "Track Easy (long straights + gentle curves)"
    "Track Medium (straights + curves + 2 sharp turns)"
    "Track Hard (S-bends + hairpin)"
    "Winding 5-lobe serpentine ring"
    ];
end

function [sceneFile, mapKey, displayName] = resolveMapKey( ...
    requestedMap, turtlebot3Dir, mapKeys, sceneNames, displayNames)
requestedMap = lower(strip(string(requestedMap)));
match = find(mapKeys == requestedMap, 1);
if isempty(match)
    error("VisualLineFollower:UnknownMap", ...
        "Unknown TurtleBot3 MuJoCo map '%s'. Valid map keys: %s.", ...
        requestedMap, strjoin(mapKeys, ", "));
end

sceneFile = fullfile(turtlebot3Dir, sceneNames(match));
if ~isfile(sceneFile)
    error("VisualLineFollower:MissingMJCF", ...
        "MuJoCo scene file was not found: %s", sceneFile);
end

mapKey = mapKeys(match);
displayName = displayNames(match);
end

function [sceneFile, mapKey, displayName] = resolveSceneFile( ...
    requestedScene, turtlebot3Dir, legacyMujocoDir, modelDir, ...
    mapKeys, sceneNames, displayNames)
requestedScene = strip(string(requestedScene));
% 按优先级依次尝试：绝对路径 → turtlebot3 目录 → 旧版目录 → 模型所在目录
candidates = [
    requestedScene
    fullfile(turtlebot3Dir, requestedScene)
    fullfile(legacyMujocoDir, requestedScene)
    fullfile(modelDir, requestedScene)
    ];

sceneFile = "";
for candidate = candidates.'
    if isfile(candidate)
        sceneFile = candidate;
        break
    end
end

if strlength(sceneFile) == 0
    error("VisualLineFollower:MissingMJCF", ...
        ["MuJoCo scene file was not found: %s\n" ...
         "Use a full XML path, a known file name, or set " ...
         "TURTLEBOT3_MUJOCO_SCENE."], requestedScene);
end

% 将解析后的文件名反查注册表，以便获取标准 mapKey 和显示名称
[~, baseName, extension] = fileparts(sceneFile);
sceneName = baseName + extension;
match = find(sceneNames == sceneName, 1);
if isempty(match)
    % 自定义场景文件，无对应 mapKey
    mapKey = "custom";
    displayName = sceneName;
else
    mapKey = mapKeys(match);
    displayName = displayNames(match);
end
end
