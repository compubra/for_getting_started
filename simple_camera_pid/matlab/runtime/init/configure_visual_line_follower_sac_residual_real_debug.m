% Configure runtime support for the real-TurtleBot3 SAC-residual visual line
% follower debug model (visual_line_follower_sac_residual_real.slx). Invoked
% by that model's InitFcn. Created 2026-08-01 by duplicating
% visual_line_follower_sac_residual_gazebo.slx wholesale -- like the plain
% debug model's _gazebo/_real pair (see configure_visual_line_follower_real_
% debug.m), the SAC residual Gazebo model already talks to real ROS2 topics
% via ros2lib Subscribe/Publish blocks (Image_Subscribe /camera/image_raw,
% Odom_Subscribe /odom -- also feeding Twist_Speed_Feedback for the SAC
% observation's normalized wheel-speed terms --, CmdVel_Publish /cmd_vel),
% so no structural changes were needed to point it at a real robot on the
% same ROS_DOMAIN_ID.
%
% *** NOT VALIDATED -- created but not opened/compiled/simulated. Known
% gaps to resolve before trusting this model against real hardware: ***
%
%   1. RESOLVED 2026-08-03 (see configure_visual_line_follower_real_debug.m
%      for the full writeup): originbot_camera_profile.m now takes an
%      explicit 'mujoco'|'gazebo'|'real' argument instead of guessing from
%      pixel count (broke once all three platforms' resolutions were
%      unified to 640x480); this model's MATLABFcn expressions pass 'real'.
%   2. STILL OPEN: CmdVel_Publish here is configured for plain
%      "geometry_msgs/Twist" where this robot's real turtlebot3_node (ROS2
%      Jazzy) almost certainly expects TwistStamped (evidence:
%      ~/line_follower_bags/lap_20260731_152709 on the Pi, and confirmed
%      directly 2026-08-03 against the live robot on the Python side --
%      see gazebo_line_follower_node.py's/control_node.py's
%      cmd_vel_stamped parameter, the equivalent fix for this same gap).
%   3. The SAC_Agent block (rllib/RL Agent, inside SAC_Residual_Controller)
%      references a workspace variable named "trainedAgent" -- neither the
%      model workspace nor this InitFcn assigns it; whoever runs this model
%      must load/assign a trained agent into the BASE workspace as
%      "trainedAgent" first (see train/sac/train_sac_residual_live.m, or
%      load a saved agent .mat and rename the loaded variable). Loading
%      nothing there will error at simulation start, not at load time.
%   4. Every trained agent this project has (train/sac/'s own output, and
%      the hpc/0720-0724 sweep checkpoints referenced by
%      launch/gazebo/gazebo_line_follower.launch.py's
%      DEFAULT_RESIDUAL_MODEL_PATH) was trained entirely in MuJoCo/Gazebo
%      simulation. None has been run against, let alone validated on, real
%      hardware -- sim-to-real transfer is unverified. Do not point this
%      model's "trainedAgent" at a real run without the same caution this
%      project's Python real-hardware nodes already apply (pure-PID first,
%      see config/real/real_line_follower.yaml's own note on this).
%
% 本文件位于 runtime/init/，取上一级目录再 genpath 递归加入整个 runtime/
% 树（init/vision/scene/ops），使 originbot_sliding_window_path_generator
% 等其它子目录下的函数也可被模型调用。
runtimeRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(genpath(runtimeRoot));
clear originbot_sliding_window_path_generator;
clear originbot_line_follower_debug_frame;
clear lf_line_search;
clear lf_safety_filter;
exportLocalPathParameters(bdroot);
exportSafetyParameters(bdroot);

function exportSafetyParameters(model)
% 安全层 + 丢线扫描找线（Line_Search / Safety_Filter 两个子系统）的参数，导出到
% Base 工作区供它们的 Interpreted MATLAB Function 块按名字引用。
%
% "mps"：**本模型与 MuJoCo 模型的线速度单位约定不同**。这里 PID_Controller
% 直接输出 /cmd_vel 用的 linear_x（m/s），而 MuJoCo 模型的 VOmega.v 是
% BaseLinearSpeed 刻度。传错只会让 CBF 的 v*sin(psi) 项差约 33 倍且不报错，
% 详见 lf_safety_defaults.m 里 signalUnits 的说明。
if isempty(find_system(model, "SearchDepth", 1, "BlockType", "SubSystem", ...
        "Name", "Safety_Filter"))
    return;
end
assignin("base", "LF_Safety", lf_safety_defaults(model, "mps"));
end

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
% 2026-08-09：与 config/real/real_line_follower.yaml（真车运行时实际读的那份）
% 对齐——ROIFraction 0.10→0.30、MinBrightness 70→170、MaxSaturation 0.30→0.20。
%
% 0.10 是早已删除的 Gazebo 线留下的值。170/0.20 则是真机侧 2026-08-04 从
% lap_20260804_162821 那次转圈事故的 bag 里逐帧反推出来的：误检帧有 50.5%
% 的 ROI 地面像素落在 value=[70,90)，普通沥青亮度漂到了旧阈值 70 正上方，
% 约一半地面被判为线候选。**MATLAB 侧一直停在会触发该事故的 70/0.30。**
%
% 注意：这张硬编码表本身就是待办项（见本文件头的坑 4 与
% load_real_params_from_yaml.m）——真相源应当是那份 yaml，这里只是把数值先
% 同步过来，接线仍未做。
defaults = [0.30, 30, 0.20, 0.6, 0.35, 0.04, 170, 0.20, 30, 500];

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
