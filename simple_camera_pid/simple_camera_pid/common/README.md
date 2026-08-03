# simple_camera_pid.common

Code shared between the MuJoCo (`../mujoco/`) and Gazebo (`../gazebo/`)
line-follower pipelines. Everything here started life under `mujoco/` (the
package this repo's port work began with) and moved here once
`gazebo_line_follower_node.py` started reusing it unchanged against real
Gazebo camera frames instead of its own separate implementation.

## Layout

| File | Contents |
| --- | --- |
| `config.py` | Shared dataclasses: `RobotConfig`, `ControllerConfig`, `CurveSpeedGovernorConfig`, `VisionConfig`, `ResidualRewardConfig` |
| `camera_geometry.py` | `CameraParams`, `pixel_to_ground`/`ground_to_pixel`, and both camera presets: `turtlebot3_burger_mujoco_camera()` (640x480, level since the 2026-07-22 flattening) and `turtlebot3_burger_gazebo_camera()` (1920x1080, always level, matches this project's local turtlebot3_burger_line_follower Gazebo model) |
| `vision.py` | `LineFollowerVision`: Hough-seed + sliding-window + ground-quadratic-fit vision, shared by both platforms since MATLAB unified what used to be two separate algorithms on 2026-07-22 (see `vision.py`'s module docstring) |
| `control/pid.py`, `control/line_follower_controller.py` | Filtered-derivative PID + recovery steering + differential-drive kinematics -- physics-agnostic, takes vision output in, wheel commands out |
| `random_path.py` | Reference-curve generator for sanity-checking pure-pursuit steering geometry, independent of any platform |
| `debug_frame.py` | OpenCV-based vision debug overlay (ROI, candidate-line mask, readout) -- see also `../gazebo/vision_debug_node.py` for the ROS-wrapped, live version of this idea |

## Why there's one `vision.py`, not a `vision_mujoco.py` / `vision_gazebo.py` pair

Earlier versions of this port had two separate vision modules, mirroring what
was then two separate MATLAB algorithms: a MuJoCo-only skeletonize +
ordered-line pure-pursuit scheme (`vision_mujoco.py`) and a Gazebo-only
band-scan + quadratic-fit scheme (`gazebo/vision_gazebo.py`) -- except in
practice both this port's MuJoCo and Gazebo nodes already constructed
`vision_mujoco.py`'s class (it generalized better; `vision_gazebo.py` was
unused dead code). MATLAB has since formally unified the same way: since
2026-07-22, `originbot_sliding_window_path_generator.m` is one file, shared
by both platforms and auto-detecting which one it's looking at, and the two
old algorithms are archived under `matlab/archive/archive_vision_scheme_a/`.
This module (`vision.py`, `LineFollowerVision`) mirrors that: both
`mujoco_line_follower_node.py` and
`gazebo_line_follower_node.py`/`vision_debug_node.py` construct the same
class, just with different `camera_geometry` presets and `flip_vertical`
settings, and there is no separate Gazebo-only vision module anymore.
