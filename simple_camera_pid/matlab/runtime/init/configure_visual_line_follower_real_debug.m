% Configure runtime support for the real-TurtleBot3 visual line follower
% debug model (visual_line_follower_with_debug_real.slx). Invoked by that
% model's InitFcn. Created 2026-08-01 by duplicating
% visual_line_follower_with_debug_gazebo.slx wholesale -- the Gazebo model
% already talks to real ROS2 topics via ros2lib Subscribe/Publish blocks
% (no internal Gazebo/MuJoCo Plant block), so it needed no structural
% changes to point at a real robot on the same ROS_DOMAIN_ID: Image_Subscribe
% is /camera/image_raw, Odom_Subscribe is /odom, CmdVel_Publish is /cmd_vel
% -- exactly the topics turtlebot3_bringup + camera_ros publish/expect on
% the real hardware too (see simple_camera_pid/config/real/
% real_line_follower.yaml and simple_camera_pid/real/vision_node.py for the
% ROS2-side equivalents of this same split).
%
% *** NOT VALIDATED -- created but not opened/compiled/simulated. Known
% gaps to resolve before trusting this model against real hardware: ***
%
%   1. RESOLVED 2026-08-03: originbot_camera_profile.m no longer
%      auto-detects platform by raw pixel count (that broke once all three
%      platforms' resolutions were unified to 640x480, since pixel count
%      alone could no longer disambiguate them -- see that function's
%      module doc). It now takes an explicit 'mujoco'|'gazebo'|'real'
%      argument instead, and this model's Camera_Local_Path_Generator /
%      Line_Process_Debug_Frame MATLABFcn expressions were updated to pass
%      'real' explicitly. Verified by direct function calls (not yet by
%      simulating this specific model, see the note above this list).
%   2. CmdVel_Publish (blk_135 as of this model's creation) is configured
%      for MessageType "geometry_msgs/Twist". Direct evidence from this
%      robot's own ~/line_follower_bags/lap_20260731_152709 recording
%      (analyzed 2026-08-01) shows /cmd_vel actually carrying
%      geometry_msgs/msg/TwistStamped on this robot's real ROS2 Jazzy
%      turtlebot3_node -- and the recorded odometry tracked that
%      TwistStamped command closely, i.e. the real base DID respond to it.
%      A plain-Twist publisher will very likely fail to connect to (or be
%      silently ignored by) turtlebot3_node's real /cmd_vel subscription.
%      Not fixed here (Blank_Twist/CmdVel_Assignment/CmdVel_Publish would
%      need re-wiring for the nested Twist-under-TwistStamped bus + a
%      Header, which needs to be verified against a running model, not
%      guessed blind) -- this exact same gap also applies to this
%      project's Python real-hardware nodes
%      (simple_camera_pid.gazebo.gazebo_line_follower_node /
%      simple_camera_pid.real.control_node both publish plain Twist too).
%
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
% configure_visual_line_follower_gazebo_debug.m, scoped to a local
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
% Copied from the Gazebo defaults as a starting point (same horizontal
% mount, same 640x480 resolution as of 2026-08-03) -- NOT re-tuned against
% real camera frames/lighting. See config/real/real_line_follower.yaml's
% own "still needs retuning against real ambient lighting" note for the
% Python-side equivalent of this same caveat.
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
