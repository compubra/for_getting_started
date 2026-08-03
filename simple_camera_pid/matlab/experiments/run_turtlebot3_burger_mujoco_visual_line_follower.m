function [out, dataFile, summary] = ...
    run_turtlebot3_burger_mujoco_visual_line_follower( ...
    renderingType, stopTime, options)
%RUN_TURTLEBOT3_BURGER_MUJOCO_VISUAL_LINE_FOLLOWER Run and archive one trial.
% Every successful simulation is saved to simulation_data as a timestamped
% MAT file containing the SimulationOutput, parameters, and summary metrics.

arguments
    renderingType (1, 1) string {mustBeMember(renderingType, ...
        ["Global", "Local", "None"])} = "Global"
    stopTime (1, 1) double {mustBePositive} = 200
    options.BaseLinearSpeed (1, 1) double = NaN
    options.BaseSpeedScale (1, 1) double {mustBePositive} = 0.03
    options.MaxWheelSpeed (1, 1) double {mustBePositive} = 20
    options.KpLine (1, 1) double = NaN
    options.KiLine (1, 1) double = NaN
    options.KdLine (1, 1) double = NaN
    options.NLine (1, 1) double = NaN
    options.ROIFraction (1, 1) double {mustBePositive} = 0.35
    options.LocalPathPoints (1, 1) double {mustBePositive} = 30
    options.LocalPathLookaheadDistance (1, 1) double {mustBePositive} = 0.20
    options.LocalPathLateralGain (1, 1) double = 0.6
    options.LocalPathHeadingGain (1, 1) double = 0.35
    options.LocalPathCurvatureGain (1, 1) double = 0.04
    options.MinBrightness (1, 1) double {mustBeNonnegative} = 70
    options.MaxSaturation (1, 1) double {mustBeNonnegative} = 0.30
    options.MinPixels (1, 1) double {mustBePositive} = 30
    options.ErrorScale (1, 1) double {mustBePositive} = 500
    options.OpenCameraMonitor (1, 1) logical = renderingType ~= "None"
    options.MapKey (1, 1) string = ""
    options.SceneFile (1, 1) string = ""
    options.TrialLabel (1, 1) string = "run"
    % 视觉算法选择：方案 A(骨架+纯追踪) 或方案 B(霍夫+滑窗+多项式拟合)。
    % 两者接口/输出布局一致，可直接互换做 A/B 对比。
    options.DetectorFunction (1, 1) string {mustBeMember( ...
        options.DetectorFunction, ["originbot_local_path_generator", ...
        "originbot_sliding_window_path_generator"])} = ...
        "originbot_local_path_generator"
end

experimentDir = string(fileparts(mfilename("fullpath")));
modelDir = fullfile(experimentDir, "..");
runtimeDir = fullfile(modelDir, "runtime");
% runtime/ 现按功能分 init/vision/scene/ops 子目录，genpath 递归加入全部
addpath(genpath(runtimeDir));
model = "visual_line_follower_with_debug";
modelFile = fullfile(modelDir, model + ".slx");
assert(isfile(modelFile), "VisualLineFollower:MissingModel", ...
    "Visual line-following model not found: %s", modelFile);
assert(~isempty(which("mj_initbus")), ...
    "VisualLineFollower:MissingBlockset", ...
    "Simulink Blockset for MuJoCo Simulator is not installed.");
assert(bdIsLoaded(model), "VisualLineFollower:ModelNotOpen", ...
    "Open %s before calling this run function.", modelFile);

[sceneFile, mapKey, mapDisplayName] = resolve_turtlebot3_mujoco_scene( ...
    model, string(modelDir), options.MapKey, options.SceneFile);

workspace = get_param(model, "ModelWorkspace");
parameters = struct( ...
    "BaseLinearSpeed", valueOrDefault( ...
        options.BaseLinearSpeed, workspace.evalin("BaseLinearSpeed")), ...
    "BaseSpeedScale", options.BaseSpeedScale, ...
    "KpLine", valueOrDefault(options.KpLine, workspace.evalin("Kp_Line")), ...
    "KiLine", valueOrDefault(options.KiLine, workspace.evalin("Ki_Line")), ...
    "KdLine", valueOrDefault(options.KdLine, workspace.evalin("Kd_Line")), ...
    "NLine", valueOrDefault(options.NLine, workspace.evalin("N_Line")), ...
    "ROIFraction", options.ROIFraction, ...
    "LocalPathPoints", options.LocalPathPoints, ...
    "LocalPathLookaheadDistance", options.LocalPathLookaheadDistance, ...
    "LocalPathLateralGain", options.LocalPathLateralGain, ...
    "LocalPathHeadingGain", options.LocalPathHeadingGain, ...
    "LocalPathCurvatureGain", options.LocalPathCurvatureGain, ...
    "MinBrightness", options.MinBrightness, ...
    "MaxSaturation", options.MaxSaturation, ...
    "MinPixels", options.MinPixels, ...
    "ErrorScale", options.ErrorScale, ...
    "MaxWheelSpeed", valueOrDefault( ...
        options.MaxWheelSpeed, workspace.evalin("MaxWheelSpeed")), ...
    "StopTime", stopTime, ...
    "MapKey", mapKey, ...
    "MapDisplayName", mapDisplayName, ...
    "SceneFile", sceneFile, ...
    "RenderingType", renderingType, ...
    "TrialLabel", options.TrialLabel, ...
    "DetectorFunction", options.DetectorFunction);

in = Simulink.SimulationInput(model);
in = in.setModelParameter("StopTime", num2str(stopTime));
in = in.setVariable("BaseLinearSpeed", parameters.BaseLinearSpeed, ...
    "Workspace", model);
in = in.setVariable("MaxWheelSpeed", parameters.MaxWheelSpeed, ...
    "Workspace", model);
in = in.setVariable("Kp_Line", parameters.KpLine, "Workspace", model);
in = in.setVariable("Ki_Line", parameters.KiLine, "Workspace", model);
in = in.setVariable("Kd_Line", parameters.KdLine, "Workspace", model);
in = in.setVariable("N_Line", parameters.NLine, "Workspace", model);
in = in.setVariable("LocalPath_ROIFraction", ...
    parameters.ROIFraction, "Workspace", model);
in = in.setVariable("LocalPath_NumPoints", ...
    parameters.LocalPathPoints, "Workspace", model);
in = in.setVariable("LocalPath_LookaheadDistance", ...
    parameters.LocalPathLookaheadDistance, "Workspace", model);
in = in.setVariable("LocalPath_LateralGain", ...
    parameters.LocalPathLateralGain, "Workspace", model);
in = in.setVariable("LocalPath_HeadingGain", ...
    parameters.LocalPathHeadingGain, "Workspace", model);
in = in.setVariable("LocalPath_CurvatureGain", ...
    parameters.LocalPathCurvatureGain, "Workspace", model);
in = in.setVariable("OriginBot_MinBrightness", ...
    parameters.MinBrightness, "Workspace", model);
in = in.setVariable("OriginBot_MaxSaturation", ...
    parameters.MaxSaturation, "Workspace", model);
in = in.setVariable("OriginBot_MinPixels", ...
    parameters.MinPixels, "Workspace", model);
in = in.setVariable("OriginBot_ErrorScale", ...
    parameters.ErrorScale, "Workspace", model);

plant = Simulink.ID.getFullName(model + ":16");
cameraMonitor = blockFromSidOrName(model, 50, "Camera_Monitor");
rotateCamera180 = blockFromSidOrEmpty(model, 63);
originBotDetector = blockFromSidOrName(model, 73, ...
    "OriginBot_Line_Follower/Camera_Local_Path_Generator");
detectorDemux = blockFromSidOrName(model, 74, ...
    "OriginBot_Line_Follower/Detector_Outputs");
linearToWheel = Simulink.ID.getFullName(model + ":8");
in = in.setBlockParameter(plant, "renderingType", renderingType);
in = in.setBlockParameter(plant, "xmlFile", sceneFile);
in = in.setBlockParameter(plant, "xmlFileRel", sceneFile);
if strlength(cameraMonitor) > 0
    in = in.setBlockParameter(cameraMonitor, "OpenAtMdlStart", ...
        onOff(options.OpenCameraMonitor));
    in = in.setBlockParameter(cameraMonitor, "Commented", ...
        onOff(~options.OpenCameraMonitor));
end
if strlength(rotateCamera180) > 0
    in = in.setBlockParameter(rotateCamera180, "Commented", ...
        onOff(~options.OpenCameraMonitor));
end
detectorExpression = sprintf( ...
    parameters.DetectorFunction + "(u, %.17g, %.17g, " + ...
    "%.17g, %.17g, %.17g, %.17g, %.17g, %.17g, %.17g, %.17g)", ...
    parameters.ROIFraction, parameters.LocalPathPoints, ...
    parameters.MinBrightness, parameters.MaxSaturation, ...
    parameters.MinPixels, parameters.ErrorScale, ...
    parameters.LocalPathLookaheadDistance, parameters.LocalPathLateralGain, ...
    parameters.LocalPathHeadingGain, parameters.LocalPathCurvatureGain);
in = in.setBlockParameter(originBotDetector, "MATLABFcn", ...
    detectorExpression);
in = in.setBlockParameter(originBotDetector, "OutputDimensions", "72");
if strlength(detectorDemux) > 0
    in = in.setBlockParameter(detectorDemux, "Outputs", ...
        "[1 1 1 1 1 67]");
end
in = in.setBlockParameter(linearToWheel, "Gain", ...
    sprintf("%.17g/TB3_WheelRadius", parameters.BaseSpeedScale));

out = sim(in);

dataDir = fullfile(modelDir, "simulation_data");
if ~isfolder(dataDir)
    mkdir(dataDir);
end
timestamp = string(datetime("now", ...
    "Format", "yyyyMMdd_HHmmss_SSS"));
safeLabel = regexprep(options.TrialLabel, "[^A-Za-z0-9_-]", "_");
dataFile = fullfile(dataDir, ...
    model + "_" + safeLabel + "_" + timestamp + ".mat");

% Archive the raw output first so a completed simulation is never lost if
% a later metric calculation needs adjustment.
save(dataFile, "out", "parameters", "-v7.3");

odom = orientSamples(squeeze(out.mujoco_odom_position.Data), 3);
vision = orientSamples(squeeze(out.mujoco_vision_debug.Data), 5);
wheel = orientSamples(squeeze(out.mujoco_wheel_command.Data), 2);

% MuJoCo emits an all-zero initialization sample before the first world
% pose. Exclude it so the jump to the spawn position is not counted as
% travelled distance. Evaluate the path at the vision sample rate to match
% the controller's observations.
odomTime = out.mujoco_odom_position.Time;
validOdom = odomTime > 0;
odom = odom(validOdom, :);
odomTime = odomTime(validOdom);
pathTime = out.mujoco_vision_debug.Time;
pathTime = pathTime(pathTime >= odomTime(1));
xy = interp1(odomTime, odom(:, 1:2), pathTime);
stepDistance = hypot(diff(xy(:, 1)), diff(xy(:, 2)));
cumulativeDistance = [0; cumsum(stepDistance)];
distanceToStart = hypot(xy(:, 1) - xy(1, 1), xy(:, 2) - xy(1, 2));
lapIndex = find(cumulativeDistance > 6 & distanceToStart < 0.25, 1);

found = vision(:, 5) > 0.5;
summary = struct( ...
    "CompletedSimulationTime", out.mujoco_vision_debug.Time(end), ...
    "PathLength", cumulativeDistance(end), ...
    "LapCompleted", ~isempty(lapIndex), ...
    "LapTime", NaN, ...
    "LineFoundRatio", mean(found), ...
    "LongestLineLoss", longestLoss( ...
        found, out.mujoco_vision_debug.Time), ...
    "MaxAbsLateralError", max(abs(vision(:, 2))), ...
    "MaxAbsHeadingError", max(abs(vision(:, 3))), ...
    "WheelSaturationRatio", mean(any( ...
        abs(wheel) >= parameters.MaxWheelSpeed - 1e-6, 2)), ...
    "FinalPosition", odom(end, :));
if ~isempty(lapIndex)
    summary.LapTime = pathTime(lapIndex);
end

save(dataFile, "summary", "-append");

fprintf("Saved simulation data: %s\n", dataFile);
fprintf("Active TurtleBot3 map: %s (%s)\n", ...
    parameters.MapDisplayName, parameters.MapKey);
fprintf(['Path %.3f m, found %.1f%%, longest loss %.2f s, ' ...
    'wheel saturation %.1f%%.\n'], ...
    summary.PathLength, 100 * summary.LineFoundRatio, ...
    summary.LongestLineLoss, 100 * summary.WheelSaturationRatio);
if summary.LapCompleted
    fprintf("Completed a lap in %.3f s.\n", summary.LapTime);
else
    fprintf("No complete lap was detected.\n");
end
end

function blockPath = blockFromSidOrEmpty(model, sid)
try
    blockPath = string(Simulink.ID.getFullName(model + ":" + sid));
catch
    blockPath = "";
end
end

function blockPath = blockFromSidOrName(model, sid, relativePath)
blockPath = blockFromSidOrEmpty(model, sid);
if strlength(blockPath) > 0
    return
end

candidate = model + "/" + relativePath;
if getSimulinkBlockHandle(candidate) ~= -1
    blockPath = candidate;
end
end

function value = valueOrDefault(value, defaultValue)
if isnan(value)
    value = defaultValue;
end
end

function value = onOff(tf)
if tf
    value = "on";
else
    value = "off";
end
end

function data = orientSamples(data, signalCount)
if size(data, 2) ~= signalCount && size(data, 1) == signalCount
    data = data.';
end
assert(size(data, 2) == signalCount, ...
    "VisualLineFollower:UnexpectedLogShape", ...
    "Expected %d logged signals, received size %s.", ...
    signalCount, mat2str(size(data)));
end

function duration = longestLoss(found, time)
duration = 0;
startIndex = 0;
for k = 1:numel(found)
    if ~found(k) && startIndex == 0
        startIndex = k;
    elseif found(k) && startIndex ~= 0
        duration = max(duration, time(k - 1) - time(startIndex));
        startIndex = 0;
    end
end
if startIndex ~= 0
    duration = max(duration, time(end) - time(startIndex));
end
end
