% Launch one visible 200-second run and record its completion status.
experimentDir = string(fileparts(mfilename("fullpath")));
projectDir = fullfile(experimentDir, "..");
model = "visual_line_follower_with_debug";

addpath(experimentDir);
cd(projectDir);
open_system(fullfile(projectDir, model + ".slx"));
drawnow;

[out, dataFile, summary] = ...
    run_turtlebot3_burger_mujoco_visual_line_follower( ...
    "Global", 200, ...
    "OpenCameraMonitor", true, ...
    "TrialLabel", "manual_visible_200s");

statusFile = fullfile(projectDir, "last_manual_run_status.mat");
save(statusFile, "dataFile", "summary");
disp(dataFile);
disp(summary);
