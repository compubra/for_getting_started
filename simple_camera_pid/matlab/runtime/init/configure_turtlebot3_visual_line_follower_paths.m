% Configure portable file references for the visual line follower model.
% This script is invoked by the model InitFcn callback. It deliberately
% resolves paths at run time so model execution does not depend on the
% platform-specific absolute path last saved in the SLX file.

% 确保整个 runtime/ 树都在 MATLAB 搜索路径中，使 init/vision/scene/ops 各
% 子目录下的函数（含本文件调用的 resolve_turtlebot3_mujoco_scene、模型
% MATLABFcn 块调用的 originbot_sliding_window_path_generator 等）均可被调用。
% 本文件位于 runtime/init/，故取上一级目录再 genpath 递归加入。
runtimeRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(genpath(runtimeRoot));
% 清除 persistent 变量，使检测函数以全新状态启动（防止上次仿真残留数据污染）
% 方案 A（originbot_local_path_generator）已归档，见 archive/archive_vision_scheme_a
clear originbot_sliding_window_path_generator;
clear originbot_line_follower_debug_frame;
configurePortablePaths(bdroot);
exportLocalPathParameters(bdroot);

function configurePortablePaths(model)
arguments
    model (1, 1) string
end

modelFile = string(get_param(model, "FileName"));
if strlength(modelFile) == 0
    error("VisualLineFollower:UnsavedModel", ...
        "Save the model before resolving project-relative files.");
end

modelDir = string(fileparts(modelFile));

% 将 cache/codegen 目录放在模型旁并使用短名称，
% 防止 Windows 260 字符路径限制截断 Stateflow 生成代码路径
cacheFolder = fullfile(modelDir, "lf_cache");
codeGenFolder = fullfile(modelDir, "lf_codegen");
if ~isfolder(cacheFolder)
    mkdir(cacheFolder);
end
if ~isfolder(codeGenFolder)
    mkdir(codeGenFolder);
end
Simulink.fileGenControl("set", ...
    "CacheFolder", cacheFolder, ...
    "CodeGenFolder", codeGenFolder, ...
    "createDir", true);

% 通过固定 SID 16 定位 MuJoCo Plant，避免因块名变更导致路径失效
plant = Simulink.ID.getFullName(model + ":16");

% InitFcn 不再“挑选”地图：只把 Plant 当前指向的那张场景，按文件名重新锚定到
% 本机项目目录（修复 SLX 中存死的机器特定绝对路径），地图本身由 Plant.xmlFile
% 决定。这样用户在 Simulink 里双击 Plant 改地图后，运行时不会被切回别的图。
currentScene = strip(string(get_param(plant, "xmlFile")));
[~, baseName, extension] = fileparts(currentScene);
currentSceneFile = baseName + extension;
try
    if strlength(currentSceneFile) > 0
        % 用文件名作 requestedScene（解析优先级最高），仅重锚路径、不改地图
        [sceneFile, mapKey, mapDisplayName] = ...
            resolve_turtlebot3_mujoco_scene(model, modelDir, "", currentSceneFile);
    else
        % Plant 无有效场景（新建/被清空）：兜底用 simple，保证模型仍可运行
        [sceneFile, mapKey, mapDisplayName] = ...
            resolve_turtlebot3_mujoco_scene(model, modelDir, "simple", "");
    end
catch
    % 文件名无法解析（文件确实缺失）：兜底回 simple
    [sceneFile, mapKey, mapDisplayName] = ...
        resolve_turtlebot3_mujoco_scene(model, modelDir, "simple", "");
end

% 暂存并在函数退出后还原 Dirty 标志：
% xmlFile 的平台绝对路径只在内存中生效，不能写回 SLX 以免破坏跨平台可移植性
originalDirty = string(get_param(model, "Dirty"));
restoreDirty = onCleanup(@() set_param(model, "Dirty", originalDirty));
set_param(plant, ...
    "xmlFile", sceneFile, ...
    "xmlFileRel", sceneFile);
fprintf("Active TurtleBot3 map: %s (%s)\n", mapDisplayName, mapKey);
end

function exportLocalPathParameters(model)
% 将模型工作区的路径规划参数同步到 Base 工作区，
% 供 Simulink 信号观测器和调试脚本直接访问
workspace = get_param(model, "ModelWorkspace");
names = [
    "LocalPath_ROIFraction"
    "LocalPath_NumPoints"
    "LocalPath_LookaheadDistance"
    "LocalPath_LateralGain"
    "LocalPath_HeadingGain"
    "LocalPath_CurvatureGain"
    "OriginBot_MinBrightness"
    "OriginBot_MaxSaturation"
    "OriginBot_MinPixels"
    "OriginBot_ErrorScale"
    ];
% 2026-07-22：MuJoCo 相机从 15° 下俯拍平为水平安装（与 Gazebo 对齐，见
% archive/archive_vision_scheme_a），近端可视地面距离下限升到约 0.31m
% （由相机安装高度/FOV 决定，水平相机 physically 看不到更近的地面）。
% ROIFraction 0.50→0.30、Lookahead 0.20→0.40：与
% runtime/originbot_camera_profile.m 的 MuJoCo DefaultROI/DefaultLookahead
% 保持一致，是几何推算的合理起点，尚未跑过真值调参 sweep（旧的 0.20 前视
% 拍平后已物理不可达，会被 clamp 强制抬高且不易察觉，故不能沿用）。
defaults = [0.30, 30, 0.40, 0.6, 0.35, 0.04, 70, 0.30, 30, 500];

for k = 1:numel(names)
    value = defaults(k);
    try
        % 模型工作区已有值时直接读取，否则写入默认值
        value = workspace.evalin(names(k));
    catch
        workspace.assignin(names(k), value);
    end
    assignin("base", names(k), value);
end
end
