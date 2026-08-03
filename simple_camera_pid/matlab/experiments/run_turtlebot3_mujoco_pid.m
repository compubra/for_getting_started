function out = run_turtlebot3_mujoco_pid( ...
    renderingType, stopTime, targetLateralPosition)
%RUN_TURTLEBOT3_MUJOCO_PID Run the TurtleBot3 PID baseline in MuJoCo.
%
%   out = run_turtlebot3_mujoco_pid("Global", 10) opens the MuJoCo window.
%   out = run_turtlebot3_mujoco_pid("None", 2) runs headless verification.
%   out = run_turtlebot3_mujoco_pid("None", 3, 0.1) tests a nonzero target.

arguments
    renderingType (1,1) string {mustBeMember(renderingType, ...
        ["Global", "Local", "None"])} = "Global"
    stopTime (1,1) double {mustBePositive} = 10
    targetLateralPosition (1,1) double {mustBeFinite} = 0
end

experimentDir = fileparts(mfilename("fullpath"));
modelDir = fileparts(experimentDir);
workspaceDir = fullfile(modelDir, "..", "..", "..");
modelFile = fullfile(modelDir, "turtlebot3_burger_pid_baseline.slx");
xmlFile = fullfile(workspaceDir, "model", "mujoco", "turtlebot3", ...
    "simple_camera_track_turtlebot3_burger_visual_scene.xml");

assert(isfile(modelFile), "MuJoCoPID:MissingModel", ...
    "PID baseline model not found: %s", modelFile);
assert(isfile(xmlFile), "MuJoCoPID:MissingMJCF", ...
    "MuJoCo scene not found: %s", xmlFile);
assert(~isempty(which("mj_initbus")), "MuJoCoPID:MissingBlockset", ...
    "Simulink Blockset for MuJoCo Simulator is not installed.");

[~, model] = fileparts(modelFile);
plant = model + "/MuJoCo_Plant";

in = Simulink.SimulationInput(model);
in = in.setModelParameter("StopTime", num2str(stopTime));
in = in.setVariable("MJ_TargetLateralPosition", ...
    targetLateralPosition, "Workspace", model);
in = in.setBlockParameter(plant, "xmlFileRel", xmlFile);
in = in.setBlockParameter(plant, "xmlFile", xmlFile);
in = in.setBlockParameter(plant, "renderingType", renderingType);

out = sim(in);

odom = out.mujoco_odom_position;
wheel = out.mujoco_wheel_command;
assert(all(isfinite(odom.Data), "all"), ...
    "MuJoCoPID:InvalidOdometry", "MuJoCo odometry contains NaN or Inf.");
assert(all(isfinite(wheel.Data), "all"), ...
    "MuJoCoPID:InvalidWheelCommand", ...
    "Wheel commands contain NaN or Inf.");

odomData = squeeze(odom.Data);
if size(odomData, 1) == 3
    finalPosition = odomData(:, end);
elseif size(odomData, 2) == 3
    finalPosition = odomData(end, :).';
else
    error("MuJoCoPID:InvalidOdometryShape", ...
        "Expected a 3-element position signal, got %s.", ...
        mat2str(size(odom.Data)));
end

fprintf("MuJoCo simulation complete: %.3f s, %d odometry samples.\n", ...
    odom.Time(end), numel(odom.Time));
fprintf("Final position [x y z] = [%.4f %.4f %.4f] m.\n", ...
    finalPosition(1), finalPosition(2), finalPosition(3));
fprintf("Final lateral error = %.4f m.\n", ...
    targetLateralPosition - finalPosition(2));
fprintf("Peak wheel command = %.4f rad/s.\n", ...
    max(abs(wheel.Data), [], "all"));
end
