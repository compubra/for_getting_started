function results = tune_localpath_gains(gainSets, options)
%TUNE_LOCALPATH_GAINS Sweep LocalPath steering gains on a MuJoCo track.
%
%   results = tune_localpath_gains(gainSets) runs one headless trial per row
%   of gainSets through run_turtlebot3_burger_mujoco_visual_line_follower and
%   ranks the rows by the project's composite track-following score (lower is
%   better). Each row is [LateralGain HeadingGain CurvatureGain].
%
%   The LocalPath gains were rescaled when the vision algorithm switched to
%   real IPM (meter) coordinates, so this sweep re-tunes them. PID gains are
%   held at the model-workspace baseline.
%
%   Name-value options:
%     MapKey   (default "complex")  track to tune on
%     StopTime (default 40)         seconds per trial (>=40 lets a lap close)
%     KpLine, KiLine, KdLine, NLine (default NaN -> model workspace baseline)
%
%   Composite score (identical to tune_complex_track_pid):
%     3*(1-LineFoundRatio) + meanAbsLateralError + 0.35*maxAbsLateralError
%       + 0.25*wheelSaturationRatio + 0.5*~lapCompleted
%
%   Example:
%     g = [0.6 0.35 0.08; 0.8 0.35 0.08; 1.0 0.35 0.08; 1.2 0.35 0.08; 1.5 0.35 0.08];
%     results = tune_localpath_gains(g);

arguments
    gainSets (:, 3) double
    options.MapKey (1, 1) string = "complex"
    options.StopTime (1, 1) double {mustBePositive} = 40
    options.KpLine (1, 1) double = NaN
    options.KiLine (1, 1) double = NaN
    options.KdLine (1, 1) double = NaN
    options.NLine (1, 1) double = NaN
end

experimentDir = string(fileparts(mfilename("fullpath")));
modelDir = fullfile(experimentDir, "..");
model = "visual_line_follower_with_debug";
modelFile = fullfile(modelDir, model + ".slx");
assert(isfile(modelFile), "TuneLocalPath:MissingModel", ...
    "Model not found: %s", modelFile);
if ~bdIsLoaded(model)
    open_system(modelFile);
end

nTrials = size(gainSets, 1);
lateralGain = gainSets(:, 1);
headingGain = gainSets(:, 2);
curvatureGain = gainSets(:, 3);

lineFoundRatio = zeros(nTrials, 1);
meanAbsLateralError = zeros(nTrials, 1);
rmsLateralError = zeros(nTrials, 1);
maxAbsLateralError = zeros(nTrials, 1);
maxAbsHeadingError = zeros(nTrials, 1);
wheelSaturationRatio = zeros(nTrials, 1);
pathLength = zeros(nTrials, 1);
lapCompleted = false(nTrials, 1);
lapTime = nan(nTrials, 1);

% Forward optional PID overrides only when provided (NaN -> harness default).
pidArgs = {};
if ~isnan(options.KpLine), pidArgs = [pidArgs, {"KpLine", options.KpLine}]; end
if ~isnan(options.KiLine), pidArgs = [pidArgs, {"KiLine", options.KiLine}]; end
if ~isnan(options.KdLine), pidArgs = [pidArgs, {"KdLine", options.KdLine}]; end
if ~isnan(options.NLine),  pidArgs = [pidArgs, {"NLine",  options.NLine}];  end

for k = 1:nTrials
    label = sprintf("lp_sweep_%02d", k);
    [out, ~, summary] = run_turtlebot3_burger_mujoco_visual_line_follower( ...
        "None", options.StopTime, ...
        "MapKey", options.MapKey, ...
        "LocalPathLateralGain", lateralGain(k), ...
        "LocalPathHeadingGain", headingGain(k), ...
        "LocalPathCurvatureGain", curvatureGain(k), ...
        pidArgs{:}, ...
        "TrialLabel", label);

    % Mean/RMS lateral are not in the harness summary; recompute from the
    % vision log (column 2 = lateral error), mirroring tune_complex_track_pid.
    vision = orientSamples(squeeze(out.mujoco_vision_debug.Data), 5);
    meanAbsLateralError(k) = mean(abs(vision(:, 2)));
    rmsLateralError(k) = sqrt(mean(vision(:, 2).^2));

    lineFoundRatio(k) = summary.LineFoundRatio;
    maxAbsLateralError(k) = summary.MaxAbsLateralError;
    maxAbsHeadingError(k) = summary.MaxAbsHeadingError;
    wheelSaturationRatio(k) = summary.WheelSaturationRatio;
    pathLength(k) = summary.PathLength;
    lapCompleted(k) = logical(summary.LapCompleted);
    lapTime(k) = summary.LapTime;
end

score = 3*(1 - lineFoundRatio) + meanAbsLateralError + ...
    0.35*maxAbsLateralError + 0.25*wheelSaturationRatio + ...
    0.5*double(~lapCompleted);

trial = (1:nTrials).';
results = table(trial, lateralGain, headingGain, curvatureGain, ...
    lineFoundRatio, meanAbsLateralError, rmsLateralError, ...
    maxAbsLateralError, maxAbsHeadingError, wheelSaturationRatio, ...
    pathLength, lapCompleted, lapTime, score, ...
    VariableNames=["Trial", "LateralGain", "HeadingGain", "CurvatureGain", ...
    "LineFoundRatio", "MeanAbsLateralError", "RmsLateralError", ...
    "MaxAbsLateralError", "MaxAbsHeadingError", "WheelSaturationRatio", ...
    "PathLength", "LapCompleted", "LapTime", "Score"]);
results = sortrows(results, "Score");

timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss_SSS"));
dataDir = fullfile(modelDir, "simulation_data");
if ~isfolder(dataDir)
    mkdir(dataDir);
end
resultFile = fullfile(dataDir, ...
    "localpath_gain_sweep_" + options.MapKey + "_" + timestamp + ".mat");
save(resultFile, "results", "gainSets", "-v7.3");

disp(results);
fprintf("Saved LocalPath sweep: %s\n", resultFile);
fprintf("Best: Lateral=%.3f Heading=%.3f Curvature=%.3f  Score=%.4f\n", ...
    results.LateralGain(1), results.HeadingGain(1), ...
    results.CurvatureGain(1), results.Score(1));
end

function data = orientSamples(data, signalCount)
if size(data, 2) ~= signalCount && size(data, 1) == signalCount
    data = data.';
end
assert(size(data, 2) == signalCount, ...
    "TuneLocalPath:UnexpectedLogShape", ...
    "Expected %d signals, received size %s.", ...
    signalCount, mat2str(size(data)));
end
