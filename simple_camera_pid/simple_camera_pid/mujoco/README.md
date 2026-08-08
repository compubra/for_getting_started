# simple_camera_pid.mujoco

Python port of the workspace's `matlab/` TurtleBot3 visual-line-follower
work package: a camera-based line follower for TurtleBot3 in MuJoCo, with
PID and SAC/PPO-residual-RL controllers. There is no `.slx` here — Simulink
block diagrams have no mechanical Python equivalent, so each model's
*behavior* was reimplemented directly as a `Gymnasium` simulation loop
(`sim/`), verified block-by-block against the `.slx` sources (via the
MATLAB MCP `model_read`/`model_query_params` tools) rather than translated
line-by-line.

The ROS2-facing entry point is `mujoco_line_follower_node.py` in this same
directory (see the package's top-level README and
`launch/mujoco_line_follower.launch.py`) — everything else under this
directory is plain importable Python, independent of ROS. Two things that
used to live here have since moved out:

  - The PID controller (`control/`) and vision algorithm (`vision.py`) now
    live under `../common/`, since `line_follower_node.py`/
    `vision_debug_node.py` use the exact same implementations against real
    Gazebo camera frames — see `../common/README.md`.
  - Everything RL-training-specific (`sim/residual_env.py`,
    `runtime/{domain_randomization,io_specs,plotting}.py`, `training/`,
    `hpc_scripts/`) now lives under `../training/`, since none of it is
    MuJoCo-only in spirit (it's the residual-RL-training half of the
    project, which happens to run its rollouts through the MuJoCo `sim/`
    env below) — see `../training/README.md`.

What's left here is the plain PID-baseline simulation (`sim/
turtlebot3_mujoco_env.py`), the ROS node, and MuJoCo-specific tooling
(`experiments/`, `map_building/`, `model_building/`, and the remaining
`runtime/{mujoco_scene,track_generation}.py`).

## Install

```bash
pip install -r ../../requirements-sim.txt
```

## Layout

| Directory | MATLAB counterpart | Contents |
| --- | --- | --- |
| `runtime/` | `matlab/runtime/` | MuJoCo scene resolution (`mujoco_scene.py`), track generation (`track_generation.py`) -- vision/camera geometry moved to `../common/`, RL-training bits moved to `../training/`, see above |
| `sim/` | the `.slx` models themselves | `Turtlebot3LineFollowerEnv`: the PID-baseline physics+vision+PID loop. `ResidualLineFollowerEnv` (the RL-training wrapper) moved to `../training/residual_env.py` |
| `experiments/` | `matlab/experiments/` | Trial runner, PID/LocalPath gain sweeps, manual run |
| `map_building/`, `model_building/` | same names in `matlab/` | Procedural track texture + MJCF scene generators |

`../common/config.py` holds the shared defaults (PID gains, robot dimensions,
vision ROI/threshold parameters), read directly out of the `.slx` model
workspace via MATLAB MCP tools so the port starts from the same numbers as
the original model. `../common/control/` (the PID controller) and
`../common/vision.py` (the vision algorithm) live outside this directory
precisely because they're shared with `line_follower_node.py`.

## Quick start (outside ROS)

```bash
python3 -m simple_camera_pid.mujoco.experiments.run_turtlebot3_mujoco_pid --stop-time 10
python3 -m simple_camera_pid.mujoco.experiments.run_turtlebot3_burger_mujoco_visual_line_follower \
    --map-key complex --stop-time 60
python3 -m simple_camera_pid.mujoco.experiments.tune_complex_track_pid
```

RL training moved to `../training/` (see `../training/README.md`):

```bash
python3 -m simple_camera_pid.training.train_sac_residual
python3 -m simple_camera_pid.training.train_ppo_residual
```

Or, once the package is built (`colcon build --packages-select simple_camera_pid`):

```bash
ros2 run simple_camera_pid mujoco_run_pid_baseline --stop-time 10
ros2 run simple_camera_pid mujoco_train_sac
```

Programmatic use:

```python
from pathlib import Path
from simple_camera_pid.mujoco.sim.turtlebot3_mujoco_env import Turtlebot3LineFollowerEnv

env = Turtlebot3LineFollowerEnv(repo_root=Path("/path/to/workspace"), map_key="ellipse")
obs, info = env.reset()
for _ in range(400):
    obs, reward, terminated, truncated, info = env.step(None)  # None = pure PID, no RL residual
```

The ROS2 node (`mujoco_line_follower_node.py`, this same directory) wraps
exactly this class in a timer loop — see the top-level package README for
`ros2 launch` usage.

## Design decisions / where this deliberately diverges from a literal translation

- **PID controller**: Simulink's `PID Controller` block (`slpidlib`) is a
  closed-source S-function. `../common/control/pid.py` reimplements its
  parallel form with a trapezoidal-discretized filtered derivative and
  back-calculation anti-windup, **numerically verified against MATLAB's
  `pid` object** (`IFormula=BackwardEuler, DFormula=Trapezoidal`, the same
  underlying algorithm) via `tf(c)`/`lsim` step responses — it matches to at
  least 4 significant figures away from saturation. It is not a bit-exact
  port of Simulink's internal numerics right at saturation, where the
  anti-windup gain `Kb` is a documented but not fully specified
  implementation detail.
- **Vision** (`../common/vision.py`, `LineFollowerVision`): ports
  `originbot_sliding_window_path_generator.m` -- Hough-seed + sliding-window
  + ground-quadratic-fit, the single algorithm MATLAB has used for *both*
  platforms since a 2026-07-22 unification (auto-detected there via
  `numel(rgbVector)`; this port just constructs the class with a different
  `camera_geometry` preset per platform instead). Before that unification,
  MATLAB had two separate algorithms and so did this port
  (`vision_mujoco.py`'s skeletonize + ordered-line pure pursuit, and
  `gazebo/vision_gazebo.py`'s band-scan + quadratic-fit) — both are now
  archived on the MATLAB side (`matlab/archive/archive_vision_scheme_a/`)
  and were removed here to match, since `vision_gazebo.py` was unused dead
  code even before that (the skeletonize variant already generalized better
  to Gazebo's real camera frames than Gazebo's own native algorithm did).
  **This is a behavioral, not numerically-verified, port**: the Hough-seed
  step uses `skimage.transform.probabilistic_hough_line`, which is not a
  bit-exact match for MATLAB's `hough`/`houghpeaks`/`houghlines` (different
  peak-finding algorithm) — unlike the PID controller below, no MATLAB-side
  ground-truth run was captured and compared against. The sliding-window and
  quadratic-fit stages *are* direct, faithful ports of the MATLAB loops.
- **Image orientation**: the MATLAB vision function unconditionally
  `flipud`s the camera frame because the Simulink MuJoCo Plant block exposes
  MuJoCo's raw bottom-row-first OpenGL framebuffer. The Python `mujoco`
  package's `Renderer.render()` already returns a conventional top-row-first
  image, so `sim/turtlebot3_mujoco_env.py` constructs the vision module with
  `flip_vertical=False` (as does `line_follower_node.py`, for ROS
  `sensor_msgs/Image`'s own top-row-first convention). The vision module
  keeps the flip as a constructor-configurable default (`True`) for fidelity
  to the MATLAB behavior on a raw framebuffer.
- **Simulink-plumbing scripts with no Python equivalent** — not translated,
  since there is no block diagram to wire or MEX host to recover:
  - `add_local_path_to_visual_model.m` / `build_sac_residual_controller.m`:
    block-wiring code. Their *architectural content* (reward formula, done
    condition, observation/action layout, wheel-mixing point) is preserved
    faithfully in `../training/residual_env.py` and
    `../common/control/line_follower_controller.py`.
  - `configure_turtlebot3_visual_line_follower_paths.m` /
    `configure_visual_line_follower_gazebo_debug.m`: Simulink `InitFcn`
    path/cache plumbing. Their only lasting effect (default parameter
    values) lives in `../common/config.py`.
  - `recover_mujoco_host.m`: recovers a crashed Simulink/MuJoCo MEX host.
    Python has no MEX host to crash.

See `../training/README.md` for the RL-training-specific design decisions
(stable-baselines3 mapping, domain randomization, the HPC/`lsac` scripts,
the archived MATLAB `.mat` agents) — all of that now lives under
`../training/`.

## What was actually verified

Everything below was executed, not just read for plausibility (in a scratch
virtualenv with the full `requirements-sim.txt`, including `mujoco` and
`stable-baselines3`, against the real scene files in `model/mujoco/`):

- `../common/control/pid.py`'s discrete filtered-PID output matches MATLAB's `pid`
  object (`IFormula=BackwardEuler, DFormula=Trapezoidal`, i.e. the same
  algorithm as the Simulink block) to at least 4 significant figures, via
  `tf(c)`/`lsim` step responses computed directly in MATLAB through the MCP
  tools and compared against this port's output.
- `Turtlebot3LineFollowerEnv` runs end-to-end against every real map
  (`simple`, `complex`, `ellipse`, `training`, `track_easy/medium/hard`):
  the vision pipeline finds the line (`found=True`, `confidence=1.0`) and
  produces smoothly evolving steering from a fresh reset.
- An 8-second PID trial via `experiments/trial_runner.py` tracks the line
  100% of the time with plausible lateral/heading error and wheel-saturation
  stats.
- `map_building`/`model_building`'s pure geometry/rasterization functions
  (closed-loop curve generation, line rasterization, disk structuring
  element) were unit-tested directly; the full generator *functions* were
  deliberately **not** run against the real repo, because they overwrite
  live MJCF/texture files by design (`track_easy_*`, `complex_camera_track.png`,
  etc.) — that's a rebuild operation to run intentionally, not something to
  trigger as a side effect.

**One real bug was found and fixed by this testing**: `mujoco.Renderer`
silently renders all-black frames if a new one is constructed for a new
`MjModel` while the previous instance for the old model hasn't been
`.close()`d first (this only shows up when a running episode's MuJoCo scene
is reloaded — e.g. domain randomization switching maps between episodes —
not on first construction). `Turtlebot3LineFollowerEnv.load_scene()` now
closes the previous renderer before creating the new one.

**A second real bug was found and fixed in the ROS2 integration itself**:
`colcon build --symlink-install` for `ament_python` packages relies on the
legacy `setup.py develop --editable` command; on a machine with a modern
`setuptools` (which dropped the `--editable` flag), `colcon-core` silently
falls back to copying files instead of symlinking them — there is no
separate "colcon-symlink-install" extension package to install, that was a
mistaken assumption in an earlier version of this note. The launch file's
first version computed its `repo_root` default by walking up from its own
installed file location (mirroring the since-deleted `track_bringup.launch.py`'s
`DEFAULT_TRACKS_PATH` convention) — under a real symlink-install that walk
lands back in the source tree; under a plain copy it resolves into
`install/simple_camera_pid`, which has no `model/mujoco/`, and the node
crashed on startup. Fixed with three layers, checked in order: (1) a
hardcoded known-workspace-root constant matching
the since-deleted `track_world.launch.py`'s `KNOWN_WORKSPACE_TRACKS_PATH` pattern, checked
first by both the node's own fallback and the launch file's `repo_root`
default; (2) searching upward from the current working directory for a
`model/mujoco` marker (works for the standard
`cd <workspace_root> && ros2 launch ...` pattern); (3) the original
file-location walk, correct only under a genuinely working symlink-install.

`mujoco_line_follower_node` was verified with an actual `colcon build` +
`ros2 launch simple_camera_pid mujoco_line_follower.launch.py`, confirming:
`/odom` publishes a valid pose + orientation, `/camera/image_raw` publishes
640x480 `rgb8` frames (matching the vision pipeline's expected input) at a
sustained rate (observed 8-15 Hz on this machine — MuJoCo rendering + the
vision pipeline cost more than the nominal 20 Hz control period allows in
wall-clock time on this hardware; the simulation itself stays internally
consistent, it just runs slower than real time here),
`/cmd_vel` publishes a plausible linear/angular echo of the wheel command,
and `/mujoco_line_follower/debug` shows sensible steering/lateral/heading/
confidence/found values with `found=1.0` on both the `simple` and `ellipse`
maps. The optional residual-policy path (`residual_model_path`/
`residual_algo`) was also verified end-to-end by launching with a freshly
trained SAC checkpoint loaded — the node starts with `residual=on` and
continues publishing valid diagnostics with the policy's action mixed in.

## 2026-07-23: vision pipeline resynced to MATLAB's Hough/sliding-window unification

`../common/vision.py` was rewritten (replacing the former `vision_mujoco.py`
scheme-A port and removing the unused `gazebo/vision_gazebo.py`) to match
MATLAB's 2026-07-22 unification described above, and
`../common/camera_geometry.py`'s MuJoCo camera preset was corrected from
`pitch_deg=15.0` to `0.0` -- the shared `model/mujoco/turtlebot3/
turtlebot3_burger_vehicle_body.xml` scene had already been flattened to a
level camera mount, so this port's IPM ground projection had been silently
using the wrong camera model since that XML changed (a real, pre-existing
bug, independent of the algorithm swap). Verified: `py_compile` on all
touched files; a synthetic-frame smoke test against both camera presets;
an end-to-end PID trial via the `experiments/` tooling on a real map; and
`gymnasium.utils.env_checker.check_env` on `ResidualLineFollowerEnv` (see
`../training/README.md`). **Not verified**: any numerical comparison against
a MATLAB-side run of the new algorithm (see the Hough-seed caveat above), and
no existing trained SAC/PPO checkpoint was re-evaluated against the new
vision numbers -- their observation layout is unchanged (5-D:
steering/lateral/heading/found/prev_action) so old checkpoints still load
and run, but the same-named quantities are now computed differently, so
driving performance may have shifted. Retraining was out of scope for this
sync.

See `../training/README.md` for training-run caveats (short smoke runs only,
not a full convergence run).
