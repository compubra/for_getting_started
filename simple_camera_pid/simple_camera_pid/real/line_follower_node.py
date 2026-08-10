#!/usr/bin/env python3
"""ROS2 node: SAC/PPO-residual (or plain-PID) TurtleBot3 line follower, single-process.

Vision + PID (+ optional residual policy) + ``/cmd_vel`` all inside one node,
driven by whatever publishes ``image_topic``. The split alternative is
``vision_node.py`` (on the Pi) + ``control_node.py`` (on the PC) -- use that
when the camera and the controller need to live on different machines; use
this when one machine can see the camera topic and drive the robot.

Runs the same controller family as mujoco_line_follower_node.py
(``common.control.line_follower_controller.LineFollowerController``, and the
same residual-model observation spec via ``residual_use_wheel_speed_obs``),
and the same vision algorithm too: ``common.vision.LineFollowerVision``
(Hough-seed + sliding-window + ground-quadratic-fit, shared by every platform
since MATLAB's 2026-07-22 vision unification -- see ``common/vision.py``'s
module docstring).

``camera_profile`` selects where the camera geometry comes from: ``"real"``
(default) builds it from this robot's own measured FOV/mount height/pitch/
roll/yaw parameters (see ``turtlebot3_burger_real_camera()`` in
``common/camera_geometry.py``); ``"gazebo"`` is a fixed 640x480 preset kept
only as a fallback -- this package dropped its Gazebo deployment line on
2026-08-08, so that profile no longer corresponds to anything this package
can launch.

This file lived at ``gazebo/line_follower_node.py`` until that
removal: it was always the Gazebo node AND the single-process real-robot
deployment, which is why it now sits under ``real/``. ``LineFollowerController``'s
``FilteredPID`` bakes a fixed ``Ts_Control`` (20 Hz) into its derivative/
integrator terms at construction (it takes no per-call dt) -- exactly like
mujoco_line_follower_node.py's timer-driven loop. So this node does the same:
the image callback only updates a "latest frame" buffer, and a separate fixed
20 Hz timer runs the vision->control step on whatever frame is latest
(sample-and-hold), keeping the control loop's dynamics correctly calibrated
regardless of the camera's actual publish rate.

Usage
-----
    ros2 launch simple_camera_pid real_line_follower.launch.py
    ros2 launch simple_camera_pid real_line_follower.launch.py \\
        residual_model_path:=/path/to/sac_agent.zip residual_algo:=sac \\
        residual_use_wheel_speed_obs:=true residual_use_2d_action:=true
"""
from __future__ import annotations

import signal
from typing import Optional, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped, Vector3
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from simple_camera_pid.common.config import ControllerConfig, CurveSpeedGovernorConfig, RobotConfig
from simple_camera_pid.common.control.line_follower_controller import LineFollowerController
from simple_camera_pid.common.residual_policy import (
    load_residual_model, residual_observation, residual_spaces,
)
from simple_camera_pid.common.camera_geometry import (
    turtlebot3_burger_gazebo_camera, turtlebot3_burger_real_camera,
)
from simple_camera_pid.common.vision import LineFollowerVision


def _image_msg_to_rgb(msg: Image) -> Optional[np.ndarray]:
    """Decode a sensor_msgs/Image into an (H, W, 3) uint8 RGB array."""
    encoding = msg.encoding.lower()
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    if height <= 0 or width <= 0 or step <= 0:
        return None
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    try:
        rows = raw.reshape((height, step))
    except ValueError:
        return None

    if encoding in ('rgb8', 'bgr8'):
        channels = rows[:, :width * 3].reshape((height, width, 3))
        if encoding == 'bgr8':
            channels = channels[:, :, ::-1]
        return np.ascontiguousarray(channels)
    if encoding in ('rgba8', 'bgra8'):
        channels = rows[:, :width * 4].reshape((height, width, 4))[:, :, :3]
        if encoding == 'bgra8':
            channels = channels[:, :, ::-1]
        return np.ascontiguousarray(channels)
    return None


class LineFollowerNode(Node):
    def __init__(self) -> None:
        super().__init__("line_follower_node")
        self._declare_parameters()

        self.robot = RobotConfig(
            wheel_radius=self.get_parameter("wheel_radius").value,
            wheel_separation=self.get_parameter("wheel_separation").value,
            max_wheel_speed=self.get_parameter("max_wheel_speed").value,
        )
        controller_cfg = ControllerConfig(
            kp=self.get_parameter("kp").value,
            ki=self.get_parameter("ki").value,
            kd=self.get_parameter("kd").value,
            n_filter=self.get_parameter("n_filter").value,
            steering_sign=self.get_parameter("steering_sign").value,
            base_linear_speed=self.get_parameter("base_linear_speed").value,
            base_speed_scale=self.get_parameter("base_speed_scale").value,
            max_angular_speed=self.get_parameter("max_angular_speed").value,
        )
        # Curve_Speed_Governor from parameters, not bare defaults -- see the
        # matching comment in control_node.py's _declare_parameters for the
        # measurement that made this necessary.
        governor_cfg = CurveSpeedGovernorConfig(
            heading_weight=float(self.get_parameter("curve_heading_weight").value),
            lateral_weight=float(self.get_parameter("curve_lateral_weight").value),
            slowdown_gain=float(self.get_parameter("curve_slowdown_gain").value),
            slowdown_bias=float(self.get_parameter("curve_slowdown_bias").value),
            min_speed_scale=float(self.get_parameter("curve_min_speed_scale").value),
            max_speed_scale=float(self.get_parameter("curve_max_speed_scale").value),
        )
        self.controller = LineFollowerController(self.robot, controller_cfg, governor_cfg)
        self.controller.reset()

        # Same vision algorithm as the MuJoCo side. camera_profile selects the
        # geometry: "real" (default) builds it from the camera_fovy_deg/
        # camera_mount_height/camera_pitch_deg/camera_roll_deg/camera_yaw_deg/
        # image_width/image_height parameters below, measured off this
        # robot's actual camera (see turtlebot3_burger_real_camera()'s
        # docstring in common/camera_geometry.py); "gazebo" is a fixed
        # 640x480 preset kept only as a fallback, left over from the Gazebo
        # deployment line this package dropped on 2026-08-08.
        # roi_bottom_fraction=0.3 (not MATLAB's Gazebo default of 0.10) plus
        # roi_widen_step/roi_widen_max below are jointly the 2026-07-29 fix
        # for this camera's found-rate problem -- see LineFollowerVision's
        # roi_widen_step docstring for the measured before/after numbers
        # (16.5% -> ~97% found on the ellipse map, track_hard's already-good
        # rate untouched). A single static roi_bottom_fraction could not
        # serve both gentle- and sharp-curve maps at once; the narrow/wide
        # two-attempt retry can.
        # flip_vertical=False: ROS Image messages (and Gazebo's camera
        # plugin) are conventional top-row-first, same reasoning as
        # sim/turtlebot3_mujoco_env.py's use of this class.
        camera_profile = self.get_parameter("camera_profile").value
        if camera_profile == "real":
            # Real camera: geometry has to come from an actual measurement of
            # this robot's camera, not a simulator default -- see
            # turtlebot3_burger_real_camera()'s docstring for how to get
            # these four numbers.
            image_width = int(self.get_parameter("image_width").value)
            image_height = int(self.get_parameter("image_height").value)
            camera_params = turtlebot3_burger_real_camera(
                image_width=image_width,
                image_height=image_height,
                fovy_deg=self.get_parameter("camera_fovy_deg").value,
                mount_height=self.get_parameter("camera_mount_height").value,
                pitch_deg=self.get_parameter("camera_pitch_deg").value,
                roll_deg=self.get_parameter("camera_roll_deg").value,
                yaw_deg=self.get_parameter("camera_yaw_deg").value,
            )
        elif camera_profile == "gazebo":
            camera_params = turtlebot3_burger_gazebo_camera()
            image_width, image_height = camera_params.image_width, camera_params.image_height
        else:
            raise ValueError(f"unknown camera_profile: {camera_profile!r} (expected 'gazebo' or 'real')")

        self.vision = LineFollowerVision(
            roi_bottom_fraction=self.get_parameter("roi_bottom_fraction").value,
            waypoint_count=self.get_parameter("num_points").value,
            lookahead_distance=self.get_parameter("lookahead_distance").value,
            lateral_gain=self.get_parameter("lateral_gain").value,
            heading_gain=self.get_parameter("heading_gain").value,
            curvature_gain=self.get_parameter("curvature_gain").value,
            min_brightness=self.get_parameter("min_brightness").value,
            max_saturation=self.get_parameter("max_saturation").value,
            min_pixels=self.get_parameter("min_pixels").value,
            error_scale=self.get_parameter("error_scale").value,
            otsu_min_contrast=self.get_parameter("otsu_min_contrast").value,
            adaptive_brightness_fraction=self.get_parameter(
                "adaptive_brightness_fraction").value,
            roi_widen_step=self.get_parameter("roi_widen_step").value,
            roi_widen_max=self.get_parameter("roi_widen_max").value,
            image_height=image_height,
            image_width=image_width,
            camera=camera_params,
            flip_vertical=False,
        )

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
            obs_space, action_space = residual_spaces(
                self._residual_use_wheel_speed_obs, self._residual_use_2d_action,
                self._residual_max_delta_v, self._residual_max_delta_omega,
            )
            self._residual_model = load_residual_model(
                residual_path, self.get_parameter("residual_algo").value, obs_space, action_space,
            )

        self._latest_frame = None        # newest RGB frame, written by _image_callback
        self._latest_frame_stamp = None  # rclpy.time.Time of that frame
        self._last_vision = None         # LocalPathResult, updated by _on_timer

        # If the camera stops publishing, treat it as line-loss rather than
        # driving on an arbitrarily old frame. The split deployment's
        # control_node has always had this (see its module docstring); this
        # node did not, so a camera driver that died mid-run left the timer
        # re-using the last frame's found=True result forever and the robot
        # kept executing that command -- turtlebot3_node has no command
        # watchdog of its own. Same 1.0 s default and same meaning.
        self._camera_timeout = float(self.get_parameter("camera_timeout").value)

        # cmd_vel_stamped: plain Twist is the historical default (it is what
        # the since-removed Gazebo ros_gz_bridge wanted), but this project's
        # real TurtleBot3's turtlebot3_node subscribes to TwistStamped (a
        # newer turtlebot3_node/ros2_control convention) -- publishing plain
        # Twist against it is not an error, it just silently never connects
        # (different message type on the same topic name), so the robot
        # never moves. config/real/real_line_follower.yaml sets this true.
        self._cmd_vel_stamped = bool(self.get_parameter("cmd_vel_stamped").value)
        cmd_vel_type = TwistStamped if self._cmd_vel_stamped else Twist
        self.cmd_pub = self.create_publisher(cmd_vel_type, self.get_parameter("cmd_vel_topic").value, 10)
        self.diag_pub = self.create_publisher(
            Float32MultiArray, self.get_parameter("diagnostic_topic").value, 10
        )
        self.create_subscription(
            Image, self.get_parameter("image_topic").value, self._image_callback, 10
        )

        ts_control = controller_cfg.ts_control
        self.timer = self.create_timer(ts_control, self._on_timer)
        self.get_logger().info(
            f"line_follower_node started: residual={'on' if self._residual_model else 'off'}, "
            f"control rate={1.0 / ts_control:.1f} Hz (camera may publish at a different rate)"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("image_topic", "/camera/image_raw")
        # No camera frame within this many seconds -> treat as line-loss
        # (found=False) instead of re-running vision on a stale frame. Mirrors
        # control_node.py's watchdog_timeout, which guards the analogous
        # network dropout in the split deployment.
        self.declare_parameter("camera_timeout", 1.0)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("diagnostic_topic", "/line_follower/debug")
        # False (default): plain Twist. True:
        # TwistStamped, matches this project's real TurtleBot3's
        # turtlebot3_node -- see the cmd_vel_stamped comment in __init__.
        self.declare_parameter("cmd_vel_stamped", False)
        self.declare_parameter("wheel_radius", RobotConfig.wheel_radius)
        self.declare_parameter("wheel_separation", RobotConfig.wheel_separation)
        self.declare_parameter("max_wheel_speed", RobotConfig.max_wheel_speed)
        self.declare_parameter("kp", ControllerConfig.kp)
        self.declare_parameter("ki", ControllerConfig.ki)
        self.declare_parameter("kd", ControllerConfig.kd)
        self.declare_parameter("n_filter", ControllerConfig.n_filter)
        self.declare_parameter("steering_sign", ControllerConfig.steering_sign)
        # These three set the actual commanded speed magnitude -- previously
        # NOT exposed as parameters at all (silently used ControllerConfig's
        # Gazebo-tuned dataclass defaults regardless of any yaml config file,
        # a real gap found 2026-07-31: a real robot commanded 0.343 m/s
        # linear / saturated 1.5 rad/s angular off these untouched defaults,
        # well above TurtleBot3 Burger's rated ~0.22 m/s). Tune these down
        # for real hardware -- see real_line_follower.yaml.
        self.declare_parameter("base_linear_speed", ControllerConfig.base_linear_speed)
        self.declare_parameter("base_speed_scale", ControllerConfig.base_speed_scale)
        self.declare_parameter("max_angular_speed", ControllerConfig.max_angular_speed)
        # Curve_Speed_Governor -- exposed 2026-08-10, same gap and same
        # reasoning as the three above; see control_node.py's copy of this
        # comment for the measurement.
        self.declare_parameter("curve_heading_weight", CurveSpeedGovernorConfig.heading_weight)
        self.declare_parameter("curve_lateral_weight", CurveSpeedGovernorConfig.lateral_weight)
        self.declare_parameter("curve_slowdown_gain", CurveSpeedGovernorConfig.slowdown_gain)
        self.declare_parameter("curve_slowdown_bias", CurveSpeedGovernorConfig.slowdown_bias)
        self.declare_parameter("curve_min_speed_scale", CurveSpeedGovernorConfig.min_speed_scale)
        self.declare_parameter("curve_max_speed_scale", CurveSpeedGovernorConfig.max_speed_scale)
        # Camera geometry: "gazebo" (default, 640x480 sim camera) or "real"
        # (a real robot's own camera -- see turtlebot3_burger_real_camera()'s
        # docstring in common/camera_geometry.py for how to fill in the
        # camera_fovy_deg/camera_mount_height/camera_pitch_deg below from an
        # actual measurement of this robot). image_width/image_height/
        # camera_fovy_deg below are only read (and only matter) when
        # camera_profile is "real" -- their defaults are just a starting
        # placeholder, not tied to the "gazebo" profile's own fixed 640x480
        # (which always comes from turtlebot3_burger_gazebo_camera()
        # regardless of these three parameters).
        # Default flipped "gazebo" -> "real" on 2026-08-08 with the removal of
        # the Gazebo line: this node's only remaining deployment is the real
        # robot, so the previous default pointed at a profile nothing launches.
        self.declare_parameter("camera_profile", "real")
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("camera_fovy_deg", 45.9857)
        self.declare_parameter("camera_mount_height", 0.133)
        self.declare_parameter("camera_pitch_deg", 15.0)
        self.declare_parameter("camera_roll_deg", 0.0)
        self.declare_parameter("camera_yaw_deg", 0.0)
        self.declare_parameter("roi_bottom_fraction", 0.3)
        self.declare_parameter("num_points", 30)
        self.declare_parameter("lookahead_distance", 0.20)
        self.declare_parameter("lateral_gain", 0.6)
        self.declare_parameter("heading_gain", 0.35)
        self.declare_parameter("curvature_gain", 0.04)
        self.declare_parameter("min_brightness", 70.0)
        self.declare_parameter("max_saturation", 0.30)
        self.declare_parameter("min_pixels", 30.0)
        self.declare_parameter("error_scale", 500.0)
        # Adaptive-ROI fallback (see LineFollowerVision.roi_widen_step's
        # docstring): if roi_bottom_fraction finds nothing this frame, retry
        # once with the ROI widened by this much (0.0 = disabled, restores
        # the old single-attempt behavior).
        # 0.0 = min_brightness is a fixed floor (default since 2026-08-10);
        # 0.55 restores MATLAB's adaptive-ceiling rule. See
        # LineFollowerVision._brightness_threshold for the real-bag evidence.
        self.declare_parameter("otsu_min_contrast", 0.0)
        self.declare_parameter("adaptive_brightness_fraction", 0.0)
        self.declare_parameter("roi_widen_step", 0.2)
        self.declare_parameter("roi_widen_max", 0.7)
        # Optional inference-only SAC/PPO residual policy (see
        # simple_camera_pid.common.residual_policy and
        # mujoco_line_follower_node.py, which shares the same spec). Empty
        # path = pure PID baseline.
        self.declare_parameter("residual_model_path", "")
        self.declare_parameter("residual_algo", "sac")  # "sac" | "ppo"
        self.declare_parameter("residual_max_delta_omega", 1.0)
        self.declare_parameter("residual_max_delta_v", 1.0)  # only used if residual_use_2d_action
        self.declare_parameter("residual_use_wheel_speed_obs", False)
        self.declare_parameter("residual_use_2d_action", False)

    def _image_callback(self, msg: Image) -> None:
        """Buffer the newest frame only -- the vision pass itself runs in
        ``_on_timer``.

        This callback used to call ``self.vision.step(rgb)`` directly, which
        contradicted this module's own docstring and had two costs on real
        hardware. (1) ``LineFollowerVision`` advances its lost-line state
        machine by a fixed ``CONTROL_PERIOD`` (0.05 s) per call, so running it
        per frame made FREEZE_TIMEOUT/SLOWDOWN_TIMEOUT mean "10/30 frames"
        rather than "0.5/1.5 s" -- at the camera's measured 15-30 Hz those
        disagreed with the controller's own lost_speed_* timeouts, which are
        driven by this timer and were correct. (2) It ran the (then ~61 ms)
        vision pass at camera rate instead of control rate, starving the
        20 Hz timer: /cmd_vel in the 2026-08-04..07 bags came out at
        1.8-21 Hz (median ~6) while /odom held a steady 20 Hz.
        """
        rgb = _image_msg_to_rgb(msg)
        if rgb is None:
            return
        self._latest_frame = rgb
        self._latest_frame_stamp = self.get_clock().now()

    def _residual_action(self) -> Tuple[float, float]:
        """Returns ``(delta_v, delta_omega)``; ``delta_v`` is 0.0 whenever no
        model is loaded or the loaded model is a 1-D-action checkpoint."""
        if self._residual_model is None or self._last_vision is None:
            return 0.0, 0.0
        vision = self._last_vision
        obs = residual_observation(
            vision.steering_error, vision.lateral_error, vision.heading_error, vision.found,
            self._prev_delta_v, self._prev_delta_omega,
            self._residual_use_wheel_speed_obs, self._residual_use_2d_action,
            self._prev_wheel_command, self.robot.max_wheel_speed,
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
        if self._latest_frame is None:
            return  # no camera frame received yet

        stale = (
            self._latest_frame_stamp is None
            or (self.get_clock().now() - self._latest_frame_stamp).nanoseconds
            > self._camera_timeout * 1e9
        )
        if stale:
            # Do not re-run vision on the old frame: that would keep
            # re-deriving found=True from it. Step the controller's
            # not-found/recovery path instead, exactly as control_node does
            # when the network drops.
            self._last_vision = None
            wheel_cmd = self.controller.step(
                steering_error=0.0, found=False, heading_error=0.0, lateral_error=0.0,
            )
            self._prev_wheel_command = (wheel_cmd.left, wheel_cmd.right)
            linear = self.robot.wheel_radius * (wheel_cmd.left + wheel_cmd.right) / 2.0
            angular = (self.robot.wheel_radius * (wheel_cmd.right - wheel_cmd.left)
                       / self.robot.wheel_separation)
            self.cmd_pub.publish(self._make_cmd_vel(linear, angular))
            return

        # Sample-and-hold: one vision pass per control tick, on whatever
        # frame is newest -- see _image_callback's docstring.
        self._last_vision = self.vision.step(self._latest_frame)
        vision = self._last_vision
        residual_delta_v, residual_delta_omega = self._residual_action()

        wheel_cmd = self.controller.step(
            steering_error=vision.steering_error, found=vision.found,
            heading_error=vision.heading_error, lateral_error=vision.lateral_error,
            residual_delta_omega=residual_delta_omega, residual_delta_v=residual_delta_v,
        )
        self._prev_wheel_command = (wheel_cmd.left, wheel_cmd.right)

        linear = self.robot.wheel_radius * (wheel_cmd.left + wheel_cmd.right) / 2.0
        angular = self.robot.wheel_radius * (wheel_cmd.right - wheel_cmd.left) / self.robot.wheel_separation
        self.cmd_pub.publish(self._make_cmd_vel(linear, angular))
        self._publish_diagnostics(vision, wheel_cmd)

    def _make_cmd_vel(self, linear: float, angular: float):
        twist = Twist(linear=Vector3(x=linear, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=angular))
        if not self._cmd_vel_stamped:
            return twist
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist = twist
        return msg

    def _publish_diagnostics(self, vision, wheel_cmd) -> None:
        msg = Float32MultiArray()
        msg.data = [
            float(vision.steering_error),
            float(vision.lateral_error),
            float(vision.heading_error),
            float(vision.confidence),
            1.0 if vision.found else 0.0,
            float(wheel_cmd.left),
            float(wheel_cmd.right),
        ]
        self.diag_pub.publish(msg)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = LineFollowerNode()

    # rclpy.init()'s own default SIGINT handler can invalidate the rcl
    # context before this function's `except KeyboardInterrupt` block ever
    # runs, so publishing the safety-stop there raced with it and regularly
    # failed (RCLError: "publisher's context is invalid") -- confirmed on
    # real hardware 2026-08-07 (with this node driving a real TurtleBot3 via
    # camera_profile:=real): the node died without ever sending the zero
    # command, leaving the robot executing its last nonzero /cmd_vel
    # indefinitely (turtlebot3_node has no command watchdog of its own).
    # Publishing synchronously from our own handler, before rclpy's default
    # handler (called via previous_handler below) gets a chance to run,
    # closes that race. Harmless under Gazebo, where an abandoned /cmd_vel
    # only spins a simulated robot -- kept unconditional so the two
    # deployments can't drift apart. Same fix in real/control_node.py.
    previous_handler = signal.getsignal(signal.SIGINT)

    def _publish_safety_stop(signum, frame):
        try:
            node.cmd_pub.publish(node._make_cmd_vel(0.0, 0.0))
        except Exception:
            pass
        if callable(previous_handler):
            previous_handler(signum, frame)

    signal.signal(signal.SIGINT, _publish_safety_stop)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
