# simple_camera_pid.training

SAC/PPO residual-RL training for the TurtleBot3 line follower. Moved out of
`../mujoco/` because none of it is MuJoCo-specific in spirit (it's the
residual-RL-training half of the project) even though its rollouts run
through `../mujoco/sim/turtlebot3_mujoco_env.py`'s physics+vision+PID loop.

## Layout

| File | Contents |
| --- | --- |
| `residual_env.py` | `ResidualLineFollowerEnv`: wraps `mujoco.sim.turtlebot3_mujoco_env.Turtlebot3LineFollowerEnv`, adding the SAC/PPO residual action, reward, done logic, and domain randomization |
| `domain_randomization.py` | Perturbs pose/dynamics/PID/perception (and optionally map) once per training episode |
| `io_specs.py` | Single source of truth for the 5-D observation / 1-D action space -- both training and `mujoco_line_follower_node.py`'s residual-inference path read from here (conceptually; the node hardcodes the same layout, see its docstring) |
| `plotting.py` | Reads back a `monitor.csv` and plots the training curve |
| `sac_training_config.py`, `ppo_training_config.py` | Hyperparameters -- edit these to tune training |
| `train_sac_residual.py`, `train_ppo_residual.py` | Training entry points (`ros2 run simple_camera_pid mujoco_train_sac`/`mujoco_train_ppo`, or `python3 -m simple_camera_pid.training.train_sac_residual`) |
| `hpc_scripts/` | Standalone HPC-cluster training scripts (`lsac`-based external package, **not vendored in this repo** -- these are meant to be copied to a cluster where `lsac` exists, not run here). Trains a 7-D-observation variant (adds normalized wheel-speed feedback) -- see `hpc_scripts/train_sac_residual.py`'s docstring for the exact spec if you need to load one of these checkpoints for inference (`mujoco_line_follower_node`'s/`line_follower_node`'s `residual_use_wheel_speed_obs:=true`) |

## Quick start

```bash
python3 -m simple_camera_pid.training.train_sac_residual
python3 -m simple_camera_pid.training.train_ppo_residual
```

Or, once built:

```bash
ros2 run simple_camera_pid mujoco_train_sac
ros2 run simple_camera_pid mujoco_train_ppo
```

Tune hyperparameters by editing `sac_training_config.py`/`ppo_training_config.py`
directly (no CLI flags). Output (checkpoints, final agent `.zip`, monitor
CSV) goes to `simulation_data/sac_training`/`ppo_training` by default.

## Design decisions

- **RL training**: MATLAB's Reinforcement Learning Toolbox has no direct
  Python equivalent; this uses `stable-baselines3` (SAC/PPO), the closest
  mainstream match. A few hyperparameters don't map 1:1 — see the
  docstrings in `train_sac_residual.py`/`train_ppo_residual.py` (e.g.
  stable-baselines3 shares one learning rate between actor and critic,
  where MATLAB configures them separately — both were `3e-4` in the
  defaults, so this doesn't change behavior for the shipped config).
- **Live-Script/HPC-batch duplication removed**: the MATLAB project kept
  two near-duplicate training entries per algorithm (a Live Script and a
  plain-script fallback for environments without the Live Editor, with
  parameters hand-kept-in-sync). That split exists only because MATLAB Live
  Scripts aren't always available; Python scripts have no such distinction,
  so each algorithm has one canonical in-repo entry point (`train_*.py`),
  driven by one config module (`*_training_config.py`). The genuinely
  separate `hpc_scripts/` variant exists for a different reason (a 7-D
  observation with wheel-speed feedback, trained on a remote GPU cluster),
  not as a Live-Script/plain-script duplicate.
- **Domain randomization**: `OriginBot_ROIStartFraction`/`OriginBot_ROIHeight`
  (perturbed by `rl_domain_randomization.m`) aren't present in the current
  `.slx` model workspace or referenced by any live block — they appear to be
  vestigial from an older ROI parameterization. `domain_randomization.py`
  instead perturbs `LocalPath_ROIFraction`, the ROI parameter the current
  model (and this port) actually uses.
- **`.mat` files**: the `matlab/archive_lyapunov/` folder and the 500+
  `.mat` files under it (trained MATLAB RL Toolbox agents, workspace
  snapshots) were out of scope (legacy/archived work) and are not
  translated or convertible — a MATLAB RL Toolbox agent object has no
  meaningful representation as a stable-baselines3 model; retraining from
  scratch here is the only path to an equivalent Python agent.

## What was actually verified

- `ResidualLineFollowerEnv` passes `gymnasium.utils.env_checker.check_env`
  and runs correctly under domain randomization (map switching, pose/
  dynamics/control/perception perturbation).
- Short live training runs (tens of timesteps) complete successfully for
  both `train_sac_residual` and `train_ppo_residual`, including checkpoint
  saving and `plotting.py` reading back the resulting `monitor.csv`.
- Re-verified after the `../mujoco/` → `../training/` move: a 40-timestep
  `train_sac_residual` smoke run completes end-to-end (env reset/step,
  checkpoint save) through the new module paths, and
  `ResidualLineFollowerEnv` itself resets/steps correctly, confirming the
  cross-package import chain into `../mujoco/sim/turtlebot3_mujoco_env.py`
  and `../common/` still resolves correctly at runtime, not just at import
  time.

## Before you rely on this for real training runs

The above covers correctness of the wiring and short runs; it does not
constitute a full training run to convergence (that takes MATLAB-scale
wall-clock time neither this session nor a smoke test can absorb) or a
side-by-side comparison of resulting policies against the MATLAB agents.
Treat this package as ready to launch, not as pre-validated to produce a
policy matching the archived MATLAB agents' performance.
