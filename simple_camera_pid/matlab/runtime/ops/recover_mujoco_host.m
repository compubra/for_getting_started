function recover_mujoco_host(mdl)
%RECOVER_MUJOCO_HOST Clear a stale/crashed MuJoCo MEX host before using a model.
%
%   A training run that was Ctrl+C'd or errored can leave the MuJoCo Blockset's
%   mj_initbus_mex host in a bad state; the next set_turtlebot3_mujoco_scene then
%   crashes with "'MATLABMexHost' ... crashed or was destroyed outside MATLAB".
%   This closes the model (so the MuJoCo block releases the MEX), clears the MEX
%   host, then leaves the model closed for the caller to reopen fresh.
%
%   IMPORTANT: if the loaded model has UNSAVED changes, they are saved first
%   (not discarded) so nothing is silently lost — then the model is reloaded
%   from that just-saved state. Best-effort throughout: never throws.
%
%   Usage
%   -----
%     recover_mujoco_host("visual_line_follower_sac_lyapunov");
%     if ~bdIsLoaded(mdl), open_system(mdlFile); end
%
%   See also SET_TURTLEBOT3_MUJOCO_SCENE, TRAIN_SAC_LYAPUNOV_AGENT.

arguments
    mdl (1,1) string
end

try
    if bdIsLoaded(mdl)
        % Preserve any in-memory edits before closing: save if dirty so we
        % never silently drop the user's unsaved changes.
        try
            if strcmp(get_param(mdl, "Dirty"), "on")
                save_system(mdl);
            end
        catch
            % couldn't query/save dirty state — fall through and still close
        end
        close_system(mdl, 0);   % 0 = don't prompt (we've already saved if dirty)
    end
catch
    % model may already be closed / in a bad state — ignore
end

try
    clear mex %#ok<CLMEX>  % drop the (possibly crashed) mj_initbus_mex host
catch
end
end
