# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace layout: three different "roots"

These are three distinct directories and confusing them breaks builds and path
resolution:

| Root | Path | Notes |
| --- | --- | --- |
| Colcon workspace root | `for_getting_started/` | Where `colcon build` / `ros2 launch` are run. Contains `build/`, `install/`, `log/`, `model/`. **Not under version control.** |
| Git repository root | `for_getting_started/src/` | The tracked repo (`.gitignore`, `pyproject.toml` live here). |
| ROS package root | `for_getting_started/src/simple_camera_pid/` | This package; also the default cwd for Claude Code sessions here. |

`simple_camera_pid/model` is a **symlink to `for_getting_started/model/`** —
outside the git repo, gitignored, and not distributable with the package. All
Gazebo track SDFs and MuJoCo MJCF/textures live behind it.

## Build and run

```bash
cd /media/kevin/ding/final_project/Sheffield/for_getting_started
colcon build --packages-select simple_camera_pid
source install/setup.bash
```

MuJoCo dependencies are not rosdep-resolvable — `pip install -r requirements-mujoco.txt`.

`README.md` (package root) is the authoritative file-by-file map, organized by
deployment line (Gazebo / MuJoCo / real robot / training). Consult it before
guessing which file belongs to which line. Per-directory design notes live in
`simple_camera_pid/{common,mujoco,training}/README.md`.

## Architecture: four deployment lines over one shared core

`simple_camera_pid/common/` holds the entire control-relevant algorithm and is
shared verbatim by all four lines:

- `vision.py` (`LineFollowerVision`) — Hough seed → sliding window → ground
  quadratic fit. **One implementation for every platform**; per-platform
  differences are only the `camera_geometry.py` `CameraParams` preset and the
  `flip_vertical` flag. Do not reintroduce per-platform vision modules — that
  split existed once and was removed to match MATLAB's 2026-07-22 unification.
- `control/` — filtered-derivative PID with back-calculation anti-windup plus
  differential-drive kinematics. Physics-agnostic: vision errors in, wheel
  commands out.
- `config.py` — the shared dataclass defaults, whose numbers were read out of
  the `.slx` model workspace. Treat it as the port's parameter source of truth.
- `residual_policy.py` — inference-side loading/running of trained SAC/PPO
  residual policies, shared by all control nodes.

The lines on top:

- **Gazebo** (`gazebo/gazebo_line_follower_node.py`) — subscribes to real
  camera topics. This same node is also the **single-process real-robot
  deployment**; `camera_profile:=gazebo|real` swaps camera geometry and the
  node is otherwise agnostic to the image source.
- **MuJoCo** (`mujoco/mujoco_line_follower_node.py`) — runs physics, camera
  render, vision, and PID *inside its own timer*. Published `/camera/image_raw`,
  `/odom`, `/cmd_vel` are **diagnostics for rviz/rqt only; the control loop is
  not closed over ROS topics**. Everything else under `mujoco/` is plain
  importable Python with no ROS dependency.
- **Real robot, split deployment** (`real/vision_node.py` on the Raspberry Pi,
  `real/control_node.py` on the PC) — the Pi computes vision locally and sends
  only 5 floats over the network (raw 800x600 RGB will not fit the WiFi link).
  Wire format is hand-packed `Float32MultiArray` in `real/local_path_msg.py`;
  there is deliberately no separate `.msg` interface package. `control_node.py`
  has a watchdog that treats staleness as line-loss rather than replaying old
  commands.
- **Training** (`training/`) — SAC/PPO residual RL. Never imported by the
  online control path; it wraps `mujoco/sim/turtlebot3_mujoco_env.py` for
  rollouts. Hyperparameters are edited in `sac_training_config.py` /
  `ppo_training_config.py` — there are no CLI flags.

### Residual policy observation/action layout

The most error-prone part of the codebase. `common/residual_policy.py` exposes
**two independent boolean switches** (`residual_use_wheel_speed_obs`,
`residual_use_2d_action`) that together select the observation width, so a
checkpoint only loads against the switch combination it was trained with:

- base 5-D obs / 1-D action — `training/io_specs.py`, the in-repo trainer
- +`use_2d_action` → 6-D obs / 2-D action `[delta_v, delta_omega]`
- +`use_wheel_speed_obs` → 7-D — the `training/hpc_scripts/` variant
- both → **8-D obs / 2-D action**, the HPC 2026-07-24 sweep checkpoints, and
  the current launch default

Checkpoints from newer HPC Numpy versions need stable-baselines3
`custom_objects` shims; that workaround already lives in `residual_policy.py`.
When touching this area, read that module's docstrings first — the failure mode
is a silent shape mismatch, not a clear error.

## Non-obvious constraints

- **Hardcoded absolute workspace paths are intentional.** `KNOWN_WORKSPACE_ROOT`
  / `KNOWN_WORKSPACE_TRACKS_PATH` in `mujoco/mujoco_line_follower_node.py`,
  `launch/mujoco/mujoco_line_follower.launch.py`, and the Gazebo launch files
  are layer 1 of a three-layer fallback (hardcoded → search upward from cwd for
  a `model/` marker → walk up from the installed file location). They exist
  because `colcon build --symlink-install` silently degrades to copying on
  modern setuptools, which breaks the file-location walk. Do not "clean them up"
  into a single derived path.
- **`simple_camera_pid/training/pipo_verl/` is unrelated vendored code** (an LLM
  RL framework, ~400 files, has its own `.git`). Excluded in `setup.py`'s
  `find_packages()` and gitignored. Ignore it in searches and never edit it.
- **`src/seldom/` is third-party proprietary code** (Horizon Robotics SDK,
  headers declare trade secrets). Gitignored; must not be committed or copied
  into shareable output.
- `.gitignore` deliberately excludes ~20 GB of regenerable artifacts —
  simulation data, rosbags, `.mat`/`.zip`/`.pth` weights, Simulink caches
  (`slprj/`, `lf_cache/`, `lf_codegen/`). If a needed file appears missing, check
  whether it is an ignored regenerable artifact before assuming it was deleted.
- `matlab/map_building` and `model_building` generators **overwrite live MJCF and
  track textures by design**. Run them only as a deliberate rebuild, never as a
  verification side effect.
- `/cmd_vel` message type: the real TurtleBot3 (`ROS2 Jazzy` `turtlebot3_node`)
  uses `TwistStamped`, not `Twist`. The Python nodes handle both via a
  `cmd_vel_stamped` parameter; the Simulink `_real.slx` variants do not.

## Testing conventions

There is no pytest suite for the package (`hpc/tests/test_translation.py` is
HPC-script scoped and not part of the package). Verification in this project
means, in ascending order of cost:

1. `python3 -m py_compile <touched files>`
2. Headless smoke runs via the MuJoCo experiment entry points, e.g.
   `python3 -m simple_camera_pid.mujoco.experiments.run_turtlebot3_mujoco_pid --stop-time 10`
3. `gymnasium.utils.env_checker.check_env` on `ResidualLineFollowerEnv` after
   any observation/action change
4. A real `colcon build` + `ros2 launch` when node startup or path resolution is
   involved

Each subpackage README carries a **"What was actually verified"** section that
distinguishes executed checks from claims. Preserve this distinction: when
changing behavior, update that section and state explicitly what was *not*
verified (e.g. the vision port is behavioral, not numerically verified against
MATLAB, because `skimage`'s probabilistic Hough is not bit-exact with MATLAB's).

## MATLAB/Simulink side

`matlab/` holds the original `.slx` work package that `mujoco/` was ported from;
it remains the reference for parameter values (read via the MATLAB MCP
`model_read` / `model_query_params` tools rather than by opening models blindly).
Variants ending `_real.slx` were created 2026-08-01 but **never opened or
validated** — `matlab/README.md` lists their known unresolved issues.

## Stale documentation

`for_getting_started/PROJECT_HANDOVER.md` (2026-06-25) describes a much older
tree with `originbot_*` packages, a top-level `main.py`, and a
`config/simple_camera_pid.yaml` that no longer exist. Do not use it for file
locations or parameter values; the package `README.md` supersedes it.
