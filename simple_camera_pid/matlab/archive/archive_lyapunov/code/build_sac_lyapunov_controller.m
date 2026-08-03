function modelFile = build_sac_lyapunov_controller()
%BUILD_SAC_LYAPUNOV_CONTROLLER Lyapunov-reward variant of the SAC residual model.
%
% Creates visual_line_follower_sac_lyapunov.slx from
% visual_line_follower_sac_residual.slx, replacing the quadratic reward with a
% Lyapunov-function reward that uses all three errors plus a V-dot term plus an
% inverse-error bonus.
%
% Reward
% ======
%   V   = SAC_Q_e*e_steer^2 + SAC_Q_l*e_lat^2 + SAC_Q_h*e_head^2   (Lyapunov fn)
%   Vdot ~ (V[k] - V[k-1]) / Ts_Control                            (discrete rate)
%   reward = -(V + SAC_Lambda*Vdot) - SAC_R*u^2 - SAC_P_Lost*(1-found)
%            + SAC_K_inv / (|e_steer| + |e_lat| + |e_head| + SAC_Eps)
%
%   李雅普诺夫稳定要求 V>0 且 Vdot<0（三误差在收缩）。奖励里 -SAC_Lambda*Vdot
%   在 Vdot<0（误差变小）时为正 → 鼓励误差单调下降。倒数项在贴线（三误差都小）
%   时给正奖励，SAC_Eps 防除零。
%
% New model-workspace variables (all tunable, like the existing SAC_* vars)
%   SAC_Q_e   = 1.0    steering-error weight in V
%   SAC_Q_l   = 1.0    lateral-error weight in V
%   SAC_Q_h   = 1.0    heading-error weight in V
%   SAC_Lambda= 0.5    weight on Vdot (Lyapunov decrease term)
%   SAC_K_inv = 0.1    coefficient of the inverse-error bonus
%   SAC_Eps   = 1e-3   small constant to avoid divide-by-zero
%   (SAC_R, SAC_P_Lost are reused from the source model.)
%
% Usage
% -----
%   cd("<workspace>/src/simple_camera_pid/matlab")
%   addpath("runtime")
%   build_sac_lyapunov_controller()

buildDir = string(fileparts(mfilename("fullpath")));
modelDir = string(fileparts(buildDir));
srcFile  = fullfile(modelDir, "visual_line_follower_sac_residual.slx");
dstFile  = fullfile(modelDir, "visual_line_follower_sac_lyapunov.slx");
newMdl   = "visual_line_follower_sac_lyapunov";

assert(isfile(srcFile), "SACLyap:MissingSource", ...
    "Source model not found:\n  %s\nBuild the residual model first.", srcFile);
assert(~isempty(ver("rl")), "SACLyap:MissingToolbox", ...
    "Reinforcement Learning Toolbox not installed.");

% ── 1. Copy & load ────────────────────────────────────────────────────────
if bdIsLoaded(newMdl), close_system(newMdl, 0); end
copyfile(srcFile, dstFile);
addpath(fullfile(modelDir, "runtime"));
load_system(dstFile);
fprintf("Loaded copy: %s\n", dstFile);

% ── 2. New Lyapunov reward workspace variables ─────────────────────────────
ws = get_param(newMdl, "ModelWorkspace");
safeAssign(ws, "SAC_Q_e",    1.0);
safeAssign(ws, "SAC_Q_l",    1.0);
safeAssign(ws, "SAC_Q_h",    1.0);
safeAssign(ws, "SAC_Lambda", 0.5);
safeAssign(ws, "SAC_K_inv",  0.1);
safeAssign(ws, "SAC_Eps",    1e-3);
fprintf("Added Lyapunov reward workspace variables.\n");

% ── 3. Rebuild the Reward_Calculation subsystem ───────────────────────────
rebuildRewardSubsystem(newMdl);

% ── 4. Compile check & save ───────────────────────────────────────────────
set_param(newMdl, "SimulationCommand", "update");
save_system(newMdl, dstFile);
close_system(newMdl, 0);

modelFile = dstFile;
fprintf("==============================================\n");
fprintf("Built: %s\n", dstFile);
fprintf("Reward is now a Lyapunov function of the three errors.\n");
fprintf("==============================================\n");
end


% ═══════════════════════════════════════════════════════════════════════════
%  Replace Reward_Calculation internals with a Lyapunov MATLAB Function block,
%  and wire lateral/heading errors into it.
% ═══════════════════════════════════════════════════════════════════════════
function rebuildRewardSubsystem(mdl)
ctrl   = mdl + "/SAC_Residual_Controller";
reward = ctrl + "/Reward_Calculation";

% ── 3a. Grow the subsystem from 3 to 5 inports ─────────────────────────────
% Existing inports (from build of residual model): e (u1), u (u2), found (u3).
% Rename e -> e_steer for clarity, then add e_lat (u4) and e_head (u5).
renameIfPresent(reward + "/e", reward + "/e_steer");
addInportIfMissing(reward, "e_lat",  4, [30 200 60 214]);
addInportIfMissing(reward, "e_head", 5, [30 250 90 264]);

% ── 3b. Delete old reward internals (keep the 5 inports + reward outport) ──
keep = ["e_steer","u","found","e_lat","e_head","reward"];
% Children only (SearchDepth 1 includes the subsystem itself — skip it).
blks = find_system(reward, "SearchDepth", 1, "Type", "Block");
for k = 1:numel(blks)
    if strcmp(blks{k}, char(reward)), continue; end   % the subsystem itself
    name = string(get_param(blks{k}, "Name"));
    if ~any(name == keep)
        try, delete_block(blks{k}); catch; end
    end
end
% remove any dangling lines
delete_line(find_system(reward, "SearchDepth", 1, "FindAll", "on", ...
    "Type", "line"));

% ── 3c. Add the Lyapunov MATLAB Function block ─────────────────────────────
fcn = reward + "/Lyapunov_Reward";
add_block("simulink/User-Defined Functions/MATLAB Function", fcn, ...
    "Position", [200 120 360 260]);
chart = find(sfroot, "-isa", "Stateflow.EMChart", "Path", char(fcn));
chart(1).Script = lyapunovCode();

% Declare the reward weights as Parameter-scope data so the function resolves
% them from the model workspace (edit the vars -> retune the reward).
markRewardParameters(chart(1));

% ── 3d. Wire inports -> function -> outport ────────────────────────────────
fph = get_param(fcn, "PortHandles");
connect(reward, "e_steer", fph.Inport(1));
connect(reward, "e_lat",   fph.Inport(2));
connect(reward, "e_head",  fph.Inport(3));
connect(reward, "u",       fph.Inport(4));
connect(reward, "found",   fph.Inport(5));
% function output -> reward outport
outBlk = reward + "/reward";
add_line(reward, fph.Outport(1), ...
    get_param(outBlk, "PortHandles").Inport(1), "autorouting", "on");

% ── 3e. Wire lateral/heading from the controller into the new inports ──────
% Inside SAC_Residual_Controller: In3=lateral_error, In4=heading_error are the
% subsystem inports feeding ObsMux. Tap them into Reward_Calculation In4/In5.
wireControllerErrorsIntoReward(ctrl, reward);

fprintf("Rebuilt Reward_Calculation as Lyapunov reward (5 inputs).\n");
end


% ═══════════════════════════════════════════════════════════════════════════
%  Lyapunov reward code (self-contained, codegen compatible)
% ═══════════════════════════════════════════════════════════════════════════
function code = lyapunovCode()
% The weights (SAC_Q_e ... SAC_Eps, SAC_R, SAC_P_Lost) are resolved from the
% model workspace: they are declared as Parameter-scope data on the block
% (see markRewardParameters), so editing the workspace variables retunes the
% reward without touching this code — same pattern as the residual model's
% SAC_Q / SAC_R gains.
lines = [
    "function reward = lyapunov_reward(e_steer, e_lat, e_head, u, found)"
    "%#codegen"
    "% Lyapunov-function reward for the SAC residual line follower."
    "%   V    = SAC_Q_e*e_steer^2 + SAC_Q_l*e_lat^2 + SAC_Q_h*e_head^2"
    "%   Vdot ~ (V[k]-V[k-1])/Ts_Control"
    "%   reward = -(V + SAC_Lambda*Vdot) - SAC_R*u^2 - SAC_P_Lost*(1-found)"
    "%            + SAC_K_inv/(|e_steer|+|e_lat|+|e_head|+SAC_Eps)"
    "% Lyapunov function (positive-definite quadratic form in the 3 errors)"
    "V = SAC_Q_e*e_steer^2 + SAC_Q_l*e_lat^2 + SAC_Q_h*e_head^2;"
    "% Discrete Lyapunov rate: (V[k] - V[k-1]) / Ts_Control"
    "persistent V_prev;"
    "if isempty(V_prev), V_prev = V; end"
    "Vdot = (V - V_prev) / Ts_Control;"
    "V_prev = V;"
    "% Inverse-error bonus (large when all three errors are small)"
    "inv_bonus = SAC_K_inv / (abs(e_steer) + abs(e_lat) + abs(e_head) + SAC_Eps);"
    "reward = -(V + SAC_Lambda*Vdot) - SAC_R*u^2 ..."
    "         - SAC_P_Lost*(1.0 - double(found > 0.5)) + inv_bonus;"
    "end"
    ];
code = strjoin(lines, newline);
end


% ═══════════════════════════════════════════════════════════════════════════
%  Mark the reward-weight names as Parameter-scope data on the function block,
%  so they resolve from the model workspace instead of being undefined vars.
% ═══════════════════════════════════════════════════════════════════════════
function markRewardParameters(chart)
paramNames = ["SAC_Q_e","SAC_Q_l","SAC_Q_h","SAC_Lambda","SAC_R", ...
              "SAC_P_Lost","SAC_K_inv","SAC_Eps","Ts_Control"];
existing = string({chart.find("-isa","Stateflow.Data").Name});
for nm = paramNames
    if any(existing == nm), continue; end
    d = Stateflow.Data(chart);
    d.Name  = char(nm);
    d.Scope = "Parameter";
    % Leave Type at its default — a Parameter-scope datum inherits its type
    % and value from the model-workspace variable of the same name.
end
end


% ─── helpers ───────────────────────────────────────────────────────────────
function renameIfPresent(oldPath, newPath)
if getSimulinkBlockHandle(char(oldPath)) ~= -1
    set_param(oldPath, "Name", extractAfter(newPath, ...
        strlength(get_param(oldPath, "Parent")) + 1));
end
end

function addInportIfMissing(sys, name, portNum, pos)
p = sys + "/" + name;
if getSimulinkBlockHandle(char(p)) == -1
    add_block("simulink/Ports & Subsystems/In1", p, ...
        "Port", num2str(portNum), "Position", pos);
end
end

function connect(sys, inportName, dstPort)
src = get_param(sys + "/" + inportName, "PortHandles");
add_line(sys, src.Outport(1), dstPort, "autorouting", "on");
end

function wireControllerErrorsIntoReward(ctrl, reward)
% Reward_Calculation now has inports 4 (e_lat) and 5 (e_head). Feed them from
% the controller's In3 (lateral) and In4 (heading) subsystem inports.
rewPH = get_param(reward, "PortHandles");
% In3 = lateral, In4 = heading (subsystem inports of SAC_Residual_Controller)
latPH  = get_param(ctrl + "/In3", "PortHandles");
headPH = get_param(ctrl + "/In4", "PortHandles");
add_line(ctrl, latPH.Outport(1),  rewPH.Inport(4), "autorouting", "on");
add_line(ctrl, headPH.Outport(1), rewPH.Inport(5), "autorouting", "on");
end

function safeAssign(ws, name, value)
try
    ws.assignin(name, value);
catch
    ws.evalin(sprintf("%s = %g;", name, value));
end
end
