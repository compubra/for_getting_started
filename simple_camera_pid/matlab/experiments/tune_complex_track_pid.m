function results = tune_complex_track_pid()
%TUNE_COMPLEX_TRACK_PID Compare PD gains on the complex MuJoCo track.

experimentDir = string(fileparts(mfilename("fullpath")));
modelDir = fullfile(experimentDir, "..");
workspaceDir = fullfile(modelDir, "..", "..", "..");
model = "visual_line_follower_with_debug";
modelFile = fullfile(modelDir, model + ".slx");
sceneFile = fullfile(workspaceDir, "model", ...
    "mujoco", "turtlebot3", ...
    "complex_camera_track_turtlebot3_burger_visual_scene.xml");

assert(isfile(modelFile), "ComplexTrackPID:MissingModel", ...
    "Model not found: %s", modelFile);
assert(isfile(sceneFile), "ComplexTrackPID:MissingScene", ...
    "Scene not found: %s", sceneFile);

open_system(modelFile);
plant = Simulink.ID.getFullName(model + ":16");
cameraMonitor = Simulink.ID.getFullName(model + ":50");
rotateCamera180 = Simulink.ID.getFullName(model + ":63");

% [Kp Ki Kd N]. The first row is the current baseline.
gainSets = [
    4.0 0 0.20 20
    4.5 0 0.30 20
    5.0 0 0.35 20
    5.5 0 0.40 20
    5.0 0 0.45 15
    ];
stopTime = 22;

in(1, size(gainSets, 1)) = Simulink.SimulationInput(model);
for k = 1:size(gainSets, 1)
    in(k) = Simulink.SimulationInput(model);
    in(k) = in(k).setModelParameter("StopTime", num2str(stopTime));
    in(k) = in(k).setVariable("Kp_Line", gainSets(k, 1), ...
        "Workspace", model);
    in(k) = in(k).setVariable("Ki_Line", gainSets(k, 2), ...
        "Workspace", model);
    in(k) = in(k).setVariable("Kd_Line", gainSets(k, 3), ...
        "Workspace", model);
    in(k) = in(k).setVariable("N_Line", gainSets(k, 4), ...
        "Workspace", model);
    in(k) = in(k).setBlockParameter(plant, "xmlFileRel", sceneFile);
    in(k) = in(k).setBlockParameter(plant, "xmlFile", sceneFile);
    in(k) = in(k).setBlockParameter(plant, "renderingType", "None");
    in(k) = in(k).setBlockParameter(cameraMonitor, "Commented", "on");
    in(k) = in(k).setBlockParameter(rotateCamera180, "Commented", "on");
end

% The MuJoCo C S-function does not declare complete operating-point
% compliance, so each trial must start from a clean simulation state.
out = sim(in, "UseFastRestart", "off", "ShowProgress", "off");

trial = (1:size(gainSets, 1)).';
lineFoundRatio = zeros(size(trial));
meanAbsLateralError = zeros(size(trial));
rmsLateralError = zeros(size(trial));
maxAbsLateralError = zeros(size(trial));
wheelSaturationRatio = zeros(size(trial));
pathLength = zeros(size(trial));
lapCompleted = false(size(trial));
lapTime = nan(size(trial));

for k = 1:numel(out)
    vision = orientSamples(squeeze(out(k).mujoco_vision_debug.Data), 5);
    wheel = orientSamples(squeeze(out(k).mujoco_wheel_command.Data), 2);
    odom = orientSamples(squeeze(out(k).mujoco_odom_position.Data), 3);

    lineFoundRatio(k) = mean(vision(:, 5) > 0.5);
    meanAbsLateralError(k) = mean(abs(vision(:, 2)));
    rmsLateralError(k) = sqrt(mean(vision(:, 2).^2));
    maxAbsLateralError(k) = max(abs(vision(:, 2)));
    wheelSaturationRatio(k) = mean(any(abs(wheel) >= 20 - 1e-6, 2));

    odomTime = out(k).mujoco_odom_position.Time;
    validOdom = odomTime > 0;
    odom = odom(validOdom, :);
    odomTime = odomTime(validOdom);
    pathTime = out(k).mujoco_vision_debug.Time;
    pathTime = pathTime(pathTime >= odomTime(1));
    xy = interp1(odomTime, odom(:, 1:2), pathTime);
    cumulativeDistance = [0; cumsum(hypot( ...
        diff(xy(:, 1)), diff(xy(:, 2))))];
    distanceToStart = hypot( ...
        xy(:, 1) - xy(1, 1), xy(:, 2) - xy(1, 2));
    lapIndex = find(cumulativeDistance > 7.0 & ...
        distanceToStart < 0.25, 1);
    pathLength(k) = cumulativeDistance(end);
    lapCompleted(k) = ~isempty(lapIndex);
    if lapCompleted(k)
        lapTime(k) = pathTime(lapIndex);
    end
end

score = 3*(1 - lineFoundRatio) + meanAbsLateralError + ...
    0.35*maxAbsLateralError + 0.25*wheelSaturationRatio + ...
    0.5*double(~lapCompleted);

results = table(trial, gainSets(:, 1), gainSets(:, 2), gainSets(:, 3), ...
    gainSets(:, 4), lineFoundRatio, meanAbsLateralError, ...
    rmsLateralError, maxAbsLateralError, wheelSaturationRatio, ...
    pathLength, lapCompleted, lapTime, score, ...
    VariableNames=["Trial", "Kp", "Ki", "Kd", "N", ...
    "LineFoundRatio", "MeanAbsLateralError", "RmsLateralError", ...
    "MaxAbsLateralError", "WheelSaturationRatio", "PathLength", ...
    "LapCompleted", "LapTime", "Score"]);
results = sortrows(results, "Score");

timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss_SSS"));
resultFile = fullfile(modelDir, "simulation_data", ...
    "complex_track_pid_sweep_" + timestamp + ".mat");
save(resultFile, "results", "out", "gainSets", "sceneFile", "-v7.3");

disp(results);
fprintf("Saved PID sweep: %s\n", resultFile);
fprintf("Best gains: Kp=%.3f, Ki=%.3f, Kd=%.3f, N=%.3f\n", ...
    results.Kp(1), results.Ki(1), results.Kd(1), results.N(1));
end

function data = orientSamples(data, signalCount)
if size(data, 2) ~= signalCount && size(data, 1) == signalCount
    data = data.';
end
assert(size(data, 2) == signalCount, ...
    "ComplexTrackPID:UnexpectedLogShape", ...
    "Expected %d signals, received size %s.", ...
    signalCount, mat2str(size(data)));
end
