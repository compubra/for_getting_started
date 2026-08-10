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
clear lf_line_search;
clear lf_safety_filter;
configurePortablePaths(bdroot);
exportLocalPathParameters(bdroot);
exportSafetyParameters(bdroot);

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

function exportSafetyParameters(model)
% 把安全层/找线状态机的参数结构体导出到 Base 工作区，供 Line_Search /
% Safety_Filter 两个子系统里的 Interpreted MATLAB Function 块按名字引用（该块表达式在 Base 工作区求值，
% 与视觉块引用 LocalPath_* 是同一机制）。
%
% 本 InitFcn 由**两个**模型共用（visual_line_follower_with_debug 与
% visual_line_follower_sac_residual），而这两个子系统只存在于装了安全层
% 的那些模型里。故先探测块是否存在再导出：没有该块的模型不会平白多出一堆
% Safety_* 模型工作区变量，也不会因缺少某个必需变量而在 InitFcn 阶段就报错。
if isempty(find_system(model, "SearchDepth", 1, "BlockType", "SubSystem", ...
        "Name", "Safety_Filter"))
    return;
end
% "scaled"：本 InitFcn 服务的两个 MuJoCo 模型里，VOmega 的 v 分量是
% BaseLinearSpeed 刻度而非 m/s。真机模型走 mps，见
% configure_visual_line_follower_sac_residual_real_debug.m。
assignin("base", "LF_Safety", lf_safety_defaults(model, "scaled"));
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
% 2026-08-09：Lookahead 0.40→0.20；ROIFraction **保持 0.30**（不要改成
% profile 里的 DefaultROI=0.50，理由见下）。
%
% 改前视：ROI=0.30 时地面可视带是 0.163~0.283m，而 Lookahead=0.40 **超出该
% 带**，前视点一直靠多项式外推得到；0.20 落在观测范围内。
%
% 不改 ROI：profile 的 DefaultROI=0.50 只是缺参兜底，不是这些地图的调参值。
% 2026-08-09 实测把 ROI 静态调到 0.50（150s，simple 图）：视觉 found 率确实
% 从 96.5% 升到 99.6%、最长丢线 5.15s 降到 0.30s，**但真值横向 RMS 从
% 4.8cm 恶化到 12.5cm，并出现一段持续 10.6 秒、最大 51cm 的跑偏**，屏障
% h 最小 -0.36（真的越界了）。原因就是 Python 侧 roi_widen_step 文档早已写
% 明的那条：更深的 ROI 会在急弯处捡到赛道**另一条邻近分支**，污染多项式拟合
% ——simple 是 S 形图，正好会自我折返。
%
% 正确做法是"窄 ROI 打底 + 自适应加宽兜底"，这也正是 roi_widen_step 存在的
% 意义（2026-08-09 已移植，见 originbot_sliding_window_path_generator.m）：
% 平时用窄而精确的 ROI，只在窄 ROI 什么都没找到时才加宽重试一次。
%
% 定稿配置(0.30/0.20 + 全部移植)实测：横向 RMS 5.16cm、路径 15.10m(四种配置
% 里最长)、角加速度 RMS 从 12.59 降到 5.62 rad/s²、越界占比 0%。
defaults = [0.30, 30, 0.20, 0.6, 0.35, 0.04, 70, 0.30, 30, 500];

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
