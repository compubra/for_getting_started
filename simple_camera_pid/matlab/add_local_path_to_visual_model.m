function add_local_path_to_visual_model()
%ADD_LOCAL_PATH_TO_VISUAL_MODEL Wire camera-local path generation into SLX.

modelFile = fullfile(fileparts(mfilename("fullpath")), ...
    "visual_line_follower_with_debug.slx");
[modelDir, modelName] = fileparts(modelFile);
% runtime/ 现按功能分 init/vision/scene/ops 子目录，genpath 递归加入全部
addpath(genpath(fullfile(modelDir, "runtime")));

open_system(modelFile);
cleanup = onCleanup(@() close_system(modelName, 0));

workspace = get_param(modelName, "ModelWorkspace");
workspace.assignin("LocalPath_ROIFraction", 0.50);  % 正图前视甜区(见 configure_...paths.m)
workspace.assignin("LocalPath_NumPoints", 30);
% 0.20：2026-07-19 真车速度(0.125 m/s)下实测最优(≈1.6s 前视)，见
% configure_turtlebot3_visual_line_follower_paths.m 同名默认值。
workspace.assignin("LocalPath_LookaheadDistance", 0.20);
workspace.assignin("LocalPath_LateralGain", 0.6);
workspace.assignin("LocalPath_HeadingGain", 0.35);
workspace.assignin("LocalPath_CurvatureGain", 0.04);
workspace.assignin("OriginBot_MinBrightness", 70);
workspace.assignin("OriginBot_MaxSaturation", 0.30);
workspace.assignin("OriginBot_MinPixels", 30);
workspace.assignin("OriginBot_ErrorScale", 500);

follower = modelName + "/OriginBot_Line_Follower";
oldDetector = follower + "/OriginBot_ROI_Centroid";
newDetector = follower + "/Camera_Local_Path_Generator";
if getSimulinkBlockHandle(newDetector) == -1
    detector = oldDetector;
else
    detector = newDetector;
end

assert(getSimulinkBlockHandle(detector) ~= -1, ...
    "LocalPath:MissingDetector", ...
    "Could not find the camera detector block in %s.", follower);

set_param(detector, "MATLABFcn", ...
    "originbot_local_path_generator(u, LocalPath_ROIFraction, " + ...
    "LocalPath_NumPoints, OriginBot_MinBrightness, " + ...
    "OriginBot_MaxSaturation, OriginBot_MinPixels, OriginBot_ErrorScale, " + ...
    "LocalPath_LookaheadDistance, LocalPath_LateralGain, " + ...
    "LocalPath_HeadingGain, LocalPath_CurvatureGain)");
set_param(detector, "OutputDimensions", "72", "Output1D", "on");
if strcmp(get_param(detector, "Name"), "OriginBot_ROI_Centroid")
    set_param(detector, "Name", "Camera_Local_Path_Generator");
end

demux = follower + "/Detector_Outputs";
set_param(demux, "Outputs", "[1 1 1 1 1 67]");

debugFrame = modelName + "/Line_Process_Debug_Frame";
if getSimulinkBlockHandle(debugFrame) ~= -1
    set_param(debugFrame, "MATLABFcn", ...
        "originbot_line_follower_debug_frame(u, LocalPath_ROIFraction, " + ...
        "LocalPath_NumPoints, OriginBot_MinBrightness, " + ...
        "OriginBot_MaxSaturation, OriginBot_MinPixels, OriginBot_ErrorScale)");
end

localPathLog = follower + "/Local_Path_Debug_Log";
if getSimulinkBlockHandle(localPathLog) == -1
    add_block("simulink/Sinks/To Workspace", localPathLog, ...
        "Position", [520 345 695 380]);
end
set_param(localPathLog, ...
    "VariableName", "mujoco_local_path_debug", ...
    "SaveFormat", "Timeseries");

connectOutputToInput(follower, demux, 6, localPathLog, 1);

set_param(modelName, "SimulationCommand", "update");
save_system(modelName, modelFile);
fprintf("Added camera-local path generator to %s\n", modelFile);
end

function connectOutputToInput(system, source, sourcePort, destination, ...
    destinationPort)
src = get_param(source, "PortHandles");
dst = get_param(destination, "PortHandles");

existingInputLine = get_param(dst.Inport(destinationPort), "Line");
if existingInputLine ~= -1
    currentSource = get_param(existingInputLine, "SrcBlockHandle");
    currentPort = get_param(existingInputLine, "SrcPortHandle");
    if currentSource == getSimulinkBlockHandle(source) && ...
            currentPort == src.Outport(sourcePort)
        return
    end
    delete_line(existingInputLine);
end

add_line(system, src.Outport(sourcePort), dst.Inport(destinationPort), ...
    "autorouting", "on");
end
