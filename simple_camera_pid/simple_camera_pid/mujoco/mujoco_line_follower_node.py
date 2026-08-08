#!/usr/bin/env python3
"""ROS2 node: self-contained MuJoCo camera line follower.

Runs :class:`simple_camera_pid.mujoco.sim.turtlebot3_mujoco_env.
Turtlebot3LineFollowerEnv` (MuJoCo physics + camera render + the pure-pursuit
vision pipeline + filtered-derivative PID, translated from the project's
``matlab/`` Simulink models) internally on a wall-clock ROS timer at
``Ts_Control`` (20 Hz, 0.05 s) — independent of Gazebo. This mirrors the
tight, single-solver-step coupling of the original Simulink model: vision,
PID, and physics all advance together every tick inside this process, rather
than being split across a distributed pub/sub round trip.

Publishes ``/camera/image_raw``, ``/odom``, ``/cmd_vel`` (an informational
echo of the wheel command actually applied — not something anything
subscribes to, to close the loop), and a diagnostics topic, purely so
rqt/rviz can watch what the simulation is doing.

Optionally loads a trained SAC/PPO residual policy (see
``simple_camera_pid.common.residual_policy`` for the observation/action
layout, and ``simple_camera_pid.training`` for the in-repo trainer) and adds
its action to the PID output every tick, for inference-only evaluation of a
residual-RL controller outside of MATLAB.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import mujoco.viewer
import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Header

from simple_camera_pid.common.config import ControllerConfig, RobotConfig, VisionConfig
from simple_camera_pid.common.residual_policy import (
    load_residual_model, residual_observation, residual_spaces,
)
from simple_camera_pid.mujoco.sim.turtlebot3_mujoco_env import Turtlebot3LineFollowerEnv


# This workspace's known location on this machine — checked first, same
# pattern as the since-deleted track_world.launch.py's KNOWN_WORKSPACE_TRACKS_PATH. The
# mujoco_line_follower.launch.py default already hardcodes this as the
# repo_root argument, so this constant only matters for `ros2 run
# mujoco_line_follower_node` directly (no launch file) with no repo_root set.
KNOWN_WORKSPACE_ROOT = '/media/kevin/ding/final_project/Sheffield/for_getting_started'


def _default_repo_root() -> str:
    """Best-effort workspace root guess.

    The MuJoCo model assets under ``model/mujoco`` live outside any
    package's installed share directory (matching how the Gazebo track
    models are handled — see the since-deleted ``track_bringup.launch.py``'s
    ``DEFAULT_TRACKS_PATH``), so there is no installed copy to point at.
    Walking up from this file's own location only works when colcon
    actually symlinks the installed package back to source; on this
    machine, ``--symlink-install`` silently falls back to a plain copy
    (modern setuptools dropped the ``--editable`` flag ``colcon-core``
    needs for ``ament_python`` symlink installs — there is no separate
    "colcon-symlink-install" extension to install, that was a mistaken
    assumption), and the walk resolves into ``install/`` instead of the
    workspace root.

    Order of resolution: the known hardcoded path, then a search upward
    from the current working directory for a ``model/mujoco`` marker
    (valid for the standard ``cd <workspace_root> && ros2 launch ...``
    invocation pattern), then the file-location walk (correct for a source
    checkout or a genuinely working symlink-install) as a last resort; the
    final result is not guaranteed to be right, but ``__init__`` validates
    it and raises a clear, actionable error rather than failing silently.
    """
    if Path(KNOWN_WORKSPACE_ROOT, "model", "mujoco").is_dir():
        return KNOWN_WORKSPACE_ROOT
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "model" / "mujoco").is_dir():
            return str(candidate)
    return str(Path(__file__).resolve().parents[4])


class MujocoLineFollowerNode(Node):
    def __init__(self) -> None:
        super().__init__("mujoco_line_follower_node")
        self._declare_parameters()

        # Empty string (the launch file's default) means "auto-detect"; recompute
        # here rather than trusting declare_parameter's default verbatim, since
        # that default was captured at class-definition/declare time, not at the
        # actual `ros2 launch`/`ros2 run` working directory.
        repo_root_param = self.get_parameter("repo_root").value
        repo_root = Path(repo_root_param) if repo_root_param else Path(_default_repo_root())
        if not (repo_root / "model" / "mujoco").is_dir():
            raise FileNotFoundError(
                f"repo_root={repo_root} has no model/mujoco/ directory. "
                "Set the 'repo_root' parameter to your workspace root "
                "(the directory containing src/ and model/)."
            )
        map_key = self.get_parameter("map_key").value

        robot = RobotConfig(
            wheel_radius=self.get_parameter("wheel_radius").value,
            wheel_separation=self.get_parameter("wheel_separation").value,
            max_wheel_speed=self.get_parameter("max_wheel_speed").value,
        )
        controller = ControllerConfig(
            kp=self.get_parameter("kp").value,
            ki=self.get_parameter("ki").value,
            kd=self.get_parameter("kd").value,
            n_filter=self.get_parameter("n_filter").value,
        )
        vision = VisionConfig(
            roi_bottom_fraction=self.get_parameter("roi_bottom_fraction").value,
            lookahead_distance=self.get_parameter("lookahead_distance").value,
            lateral_gain=self.get_parameter("lateral_gain").value,
            heading_gain=self.get_parameter("heading_gain").value,
            curvature_gain=self.get_parameter("curvature_gain").value,
            roi_widen_step=self.get_parameter("roi_widen_step").value,
            roi_widen_max=self.get_parameter("roi_widen_max").value,
        )

        self.env = Turtlebot3LineFollowerEnv(
            repo_root, map_key=map_key, robot=robot, controller=controller, vision=vision,
        )
        self.env.reset()

        self._viewer = None
        if self.get_parameter("use_viewer").value:
            # Passive viewer: an on-screen GLFW window showing the whole MuJoCo
            # scene (track + car) from a free camera, separate from the
            # offscreen onboard-camera render used for /camera/image_raw.
            self._viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)

        self._residual_model = None
        self._residual_max_delta_omega = float(self.get_parameter("residual_max_delta_omega").value)
        self._residual_max_delta_v = float(self.get_parameter("residual_max_delta_v").value)
        self._residual_use_wheel_speed_obs = bool(
            self.get_parameter("residual_use_wheel_speed_obs").value
        )
        self._residual_use_2d_action = bool(self.get_parameter("residual_use_2d_action").value)
        self._prev_delta_v = 0.0
        self._prev_delta_omega = 0.0
        self._prev_wheel_command = (0.0, 0.0)
        residual_path = self.get_parameter("residual_model_path").value
        if residual_path:
            # Resolve against repo_root rather than the process's cwd: a
            # relative residual_model_path (the common case when passed on
            # the `ros2 launch` command line) otherwise only loads when
            # launched from the workspace root, and fails with a confusing
            # FileNotFoundError from deep inside stable-baselines3 zip
            # loading if launched from anywhere else. If residual_path is
            # already absolute, repo_root / residual_path simply evaluates
            # to residual_path (pathlib semantics).
            residual_path = str(repo_root / residual_path)
            obs_space, action_space = residual_spaces(
                self._residual_use_wheel_speed_obs, self._residual_use_2d_action,
                self._residual_max_delta_v, self._residual_max_delta_omega,
            )
            self._residual_model = load_residual_model(
                residual_path, self.get_parameter("residual_algo").value, obs_space, action_space,
            )

        self.image_pub = self.create_publisher(Image, self.get_parameter("image_topic").value, 10)
        self.odom_pub = self.create_publisher(Odometry, self.get_parameter("odom_topic").value, 10)
        self.cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)
        self.diag_pub = self.create_publisher(
            Float32MultiArray, self.get_parameter("diagnostic_topic").value, 10
        )
        self.odom_frame_id = self.get_parameter("odom_frame_id").value
        self.base_frame_id = self.get_parameter("base_frame_id").value
        self.camera_frame_id = self.get_parameter("camera_frame_id").value

        ts_control = self.env.controller_cfg.ts_control
        self.timer = self.create_timer(ts_control, self._on_timer)
        self.get_logger().info(
            f"MuJoCo line follower started: map={self.env.map_display_name} "
            f"({self.env.map_key}), residual={'on' if self._residual_model else 'off'}, "
            f"control rate={1.0 / ts_control:.1f} Hz"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("repo_root", _default_repo_root())
        self.declare_parameter("map_key", "simple")
        self.declare_parameter("use_viewer", False)
        self.declare_parameter("wheel_radius", RobotConfig.wheel_radius)
        self.declare_parameter("wheel_separation", RobotConfig.wheel_separation)
        self.declare_parameter("max_wheel_speed", RobotConfig.max_wheel_speed)
        self.declare_parameter("kp", ControllerConfig.kp)
        self.declare_parameter("ki", ControllerConfig.ki)
        self.declare_parameter("kd", ControllerConfig.kd)
        self.declare_parameter("n_filter", ControllerConfig.n_filter)
        self.declare_parameter("roi_bottom_fraction", VisionConfig.roi_bottom_fraction)
        self.declare_parameter("lookahead_distance", VisionConfig.lookahead_distance)
        self.declare_parameter("lateral_gain", VisionConfig.lateral_gain)
        self.declare_parameter("heading_gain", VisionConfig.heading_gain)
        self.declare_parameter("curvature_gain", VisionConfig.curvature_gain)
        self.declare_parameter("roi_widen_step", VisionConfig.roi_widen_step)
        self.declare_parameter("roi_widen_max", VisionConfig.roi_widen_max)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("diagnostic_topic", "/mujoco_line_follower/debug")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("camera_frame_id", "camera_link")
        # Optional inference-only SAC/PPO residual policy (see
        # simple_camera_pid.common.residual_policy). Empty path = pure PID
        # baseline.
        self.declare_parameter("residual_model_path", "")
        self.declare_parameter("residual_algo", "sac")  # "sac" | "ppo"
        self.declare_parameter("residual_max_delta_omega", 1.0)
        self.declare_parameter("residual_max_delta_v", 1.0)  # only used if residual_use_2d_action
        # False (default): 5-D obs, matches simple_camera_pid.training's
        # in-repo SAC/PPO (io_specs.py / residual_env.py).
        # True: adds normalized wheel-speed feedback (v_norm, omega_norm),
        # matching the HPC-side lsac-based train_sac_residual.py/
        # train_ppo_residual.py at the package root.
        self.declare_parameter("residual_use_wheel_speed_obs", False)
        # False (default): 1-D action, delta_omega only -- matches every
        # checkpoint generation before the 2026-07-21 HPC refactor (the
        # in-repo trainer, and hpc/0720 and earlier). True: 2-D action
        # [delta_v, delta_omega], matching hpc/0721+/hpc/0724's residual
        # model (see simple_camera_pid.common.residual_policy's docstring
        # for the full checkpoint-generation table). Requires
        # residual_use_wheel_speed_obs=true too -- no checkpoint combines a
        # 2-D action with a 5-D (no-wheel-speed) observation.
        self.declare_parameter("residual_use_2d_action", False)

    def _residual_action(self) -> Tuple[float, float]:
        """Returns ``(delta_v, delta_omega)``; ``delta_v`` is 0.0 whenever no
        model is loaded or the loaded model is a 1-D-action checkpoint."""
        if self._residual_model is None:
            return 0.0, 0.0
        vision = self.env.last_vision
        obs = residual_observation(
            vision.steering_error, vision.lateral_error, vision.heading_error, vision.found,
            self._prev_delta_v, self._prev_delta_omega,
            self._residual_use_wheel_speed_obs, self._residual_use_2d_action,
            self._prev_wheel_command, self.env.robot.max_wheel_speed,
        )
        predicted, _ = self._residual_model.predict(obs, deterministic=True)
        if self._residual_use_2d_action:
            delta_v = float(np.clip(predicted[0], -self._residual_max_delta_v, self._residual_max_delta_v))
            delta_omega = float(np.clip(predicted[1], -self._residual_max_delta_omega,
                                         self._residual_max_delta_omega))
        else:
            delta_v = 0.0
            delta_omega = float(np.clip(predicted[0], -self._residual_max_delta_omega,
                                         self._residual_max_delta_omega))
        self._prev_delta_v, self._prev_delta_omega = delta_v, delta_omega
        return delta_v, delta_omega

    def _on_timer(self) -> None:
        if self._viewer is not None and not self._viewer.is_running():
            # User closed the viewer window: shut down rather than keep
            # simulating headlessly with a dead handle.
            self.get_logger().info("Viewer window closed, shutting down.")
            rclpy.shutdown()
            return

        action = self._residual_action()
        _, _, terminated, truncated, info = self.env.step(action)
        log = info["log"]
        self._prev_wheel_command = (float(log.wheel_command[0]), float(log.wheel_command[1]))

        if self._viewer is not None:
            self._viewer.sync()

        stamp = self.get_clock().now().to_msg()
        self._publish_image(log, stamp)
        self._publish_odom(log, stamp)
        self._publish_cmd_vel_echo(log)
        self._publish_diagnostics(log)

        if terminated or truncated:
            # Only reachable if a residual policy's episode-style bookkeeping
            # were wired in; the plain PID baseline never sets these. Reset in
            # place so the node keeps running rather than needing a respawn.
            self.env.reset()
            self._prev_delta_v = 0.0
            self._prev_delta_omega = 0.0
            self._prev_wheel_command = (0.0, 0.0)

    def _publish_image(self, log, stamp) -> None:
        frame = self.env.last_frame
        msg = Image()
        msg.header = Header(stamp=stamp, frame_id=self.camera_frame_id)
        msg.height, msg.width = frame.shape[0], frame.shape[1]
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        self.image_pub.publish(msg)

    def _publish_odom(self, log, stamp) -> None:
        msg = Odometry()
        msg.header = Header(stamp=stamp, frame_id=self.odom_frame_id)
        msg.child_frame_id = self.base_frame_id
        x, y, z = (float(v) for v in log.odom_position)
        qw, qx, qy, qz = (float(v) for v in log.odom_quaternion)  # MuJoCo order: w, x, y, z
        msg.pose.pose = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        )
        self.odom_pub.publish(msg)

    def _publish_cmd_vel_echo(self, log) -> None:
        left, right = (float(v) for v in log.wheel_command)
        wheel_radius = self.env.robot.wheel_radius
        wheel_separation = self.env.robot.wheel_separation
        linear = wheel_radius * (left + right) / 2.0
        angular = wheel_radius * (right - left) / wheel_separation
        self.cmd_pub.publish(Twist(
            linear=Vector3(x=linear, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=angular),
        ))

    def _publish_diagnostics(self, log) -> None:
        vision = log.vision
        msg = Float32MultiArray()
        msg.data = [
            float(vision.steering_error),
            float(vision.lateral_error),
            float(vision.heading_error),
            float(vision.confidence),
            1.0 if vision.found else 0.0,
            float(log.wheel_command[0]),
            float(log.wheel_command[1]),
        ]
        self.diag_pub.publish(msg)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = MujocoLineFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._viewer is not None:
            node._viewer.close()
        node.env.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
