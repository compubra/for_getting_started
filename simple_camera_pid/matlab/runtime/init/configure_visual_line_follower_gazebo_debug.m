% Configure runtime support for the Gazebo visual line follower models.
% Invoked by each _gazebo model's InitFcn. Unlike
% configure_turtlebot3_visual_line_follower_paths.m, there is no MuJoCo
% Plant block to re-anchor, so this only sets up the local-path/debug-frame
% parameters and clears persistent detector state. Vision runs in-model via
% OriginBot_Line_Follower against /camera/image_raw, matching the MuJoCo
% models — not via an external ROS error topic.
%
% 2026-07-22: originbot_sliding_window_path_generator / originbot_line_
% follower_debug_frame are now unified — one file each, shared with the
% MuJoCo models, auto-detecting platform via input image size
% (originbot_camera_profile.m). The former "_gazebo"-suffixed sibling files
% are archived under archive/archive_vision_scheme_a.

% 本文件位于 runtime/init/，取上一级目录再 genpath 递归加入整个 runtime/
% 树（init/vision/scene/ops），使 originbot_sliding_window_path_generator
% 等其它子目录下的函数也可被模型调用。
runtimeRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(genpath(runtimeRoot));
clear originbot_sliding_window_path_generator;
clear originbot_line_follower_debug_frame;
exportLocalPathParameters(bdroot);

function exportLocalPathParameters(model)
% Mirrors exportLocalPathParameters in
% configure_turtlebot3_visual_line_follower_paths.m, scoped to a local
% function so loop variables do not leak into the base workspace (this
% script runs via run() from InitFcn, in the base workspace context).
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
% Lookahead 0.20 与 MuJoCo 主线同步(2026-07-19 真车速度下实测最优);
% ROIFraction 0.10 保留——gazebo 相机水平安装,ROI 再大会含地平线。
defaults = [0.10, 30, 0.20, 0.6, 0.35, 0.04, 70, 0.30, 30, 500];

for k = 1:numel(names)
    value = defaults(k);
    try
        value = workspace.evalin(names(k));
    catch
        workspace.assignin(names(k), value);
    end
    assignin("base", names(k), value);
end
end
