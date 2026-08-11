#!/usr/bin/env python3
"""ROS2 node: control-only half of the split real-hardware architecture.

Meant to run ON THE PC: subscribes to the small steering/lateral/heading-error
message ``vision_node.py`` publishes (see ``local_path_msg.py``) instead of a
raw camera frame, runs
``common.control.line_follower_controller.LineFollowerController`` (+
optional SAC/PPO residual policy, same as ``line_follower_node.py``)
at a fixed 20 Hz control period, and publishes ``/cmd_vel``. See
``vision_node.py``'s module docstring for why this is split across two
machines/two nodes instead of ``line_follower_node.py``'s
single-node design.

Same sample-and-hold pattern as ``line_follower_node.py``: the
local-path subscription callback only updates a "latest" buffer; a separate
fixed-rate timer runs the control step, so the control loop's PID dynamics
stay correctly calibrated regardless of ``vision_node``'s actual publish rate
(which now also includes real network jitter between the two machines, not
just the camera's own rate).

Network-dropout watchdog: unlike the single-node design, a machine-to-machine
link can go down while each side keeps running fine on its own -- silently
replaying the last real ``local_path`` message forever would drive the robot
on stale, possibly long-outdated steering data. If no ``local_path`` message
has arrived within ``watchdog_timeout`` seconds, this node treats the input
as lost (``found=False``, all errors zeroed) for that tick instead, which
routes through ``LineFollowerController``'s existing not-found/recovery path
the same way a real on-camera line loss would.

Usage (on the PC)
------------------
    ros2 run simple_camera_pid line_follower_control_node --ros-args \\
        --params-file install/simple_camera_pid/share/simple_camera_pid/config/real/real_line_follower.yaml
"""
from __future__ import annotations

import signal
from typing import Optional, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped, Vector3
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from simple_camera_pid.common.config import (
    ControllerConfig, CurveSpeedGovernorConfig, LineSearchConfig, RobotConfig,
    SafetyFilterConfig, VisionConfig,
)
from simple_camera_pid.common.control.line_follower_controller import LineFollowerController
from simple_camera_pid.common.control.line_search import LineSearch
from simple_camera_pid.common.control.safety_filter import SafetyFilter
from simple_camera_pid.common.residual_policy import (
    load_residual_model, residual_observation, residual_spaces,
)
from simple_camera_pid.real.local_path_msg import unpack_local_path


class LineFollowerControlNode(Node):
    def __init__(self) -> None:
        super().__init__("line_follower_control_node")
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
            # One period drives the PID discretization, this node's timer, and
            # the safety layer's rate limits. Keeping them from a single
            # parameter means they cannot silently disagree.
            ts_control=float(self.get_parameter("control_period").value),
        )
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

        # Safety layer + lost-line recovery, between the controller and the
        # wheels. ``speed_scale`` must match the units command() returns v in,
        # i.e. base_speed_scale -- see SafetyFilterConfig's docstring, getting
        # it wrong is silent.
        ts_control = float(self.get_parameter("control_period").value)
        self.safety_filter = SafetyFilter(SafetyFilterConfig(
            enable=bool(self.get_parameter("safety_enable_filter").value),
            ts=ts_control,
            speed_scale=controller_cfg.base_speed_scale,
            lookahead=float(self.get_parameter("lookahead_distance").value),
            lateral_max=float(self.get_parameter("safety_lateral_max").value),
            cbf_alpha=float(self.get_parameter("safety_cbf_alpha").value),
            speed_backoff=float(self.get_parameter("safety_speed_backoff").value),
            max_v=controller_cfg.base_linear_speed,
            max_omega=controller_cfg.max_angular_speed,
            max_accel_v=float(self.get_parameter("safety_max_accel_v").value),
            max_accel_omega=float(self.get_parameter("safety_max_accel_omega").value),
        ))
        self.line_search = LineSearch(LineSearchConfig(
            enable=bool(self.get_parameter("safety_enable_search").value),
            ts=ts_control,
            max_v=controller_cfg.base_linear_speed,
            max_omega=controller_cfg.max_angular_speed,
        ))

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

        self._watchdog_timeout = float(self.get_parameter("watchdog_timeout").value)
        self._last_local_path = None  # (steering_error, lateral_error, heading_error, confidence, found)
        self._last_local_path_stamp = None  # rclpy.time.Time

        # cmd_vel_stamped: this project's real TurtleBot3's turtlebot3_node
        # subscribes to TwistStamped (a newer turtlebot3_node/ros2_control
        # convention), not plain Twist -- publishing the wrong type isn't an
        # error, it just silently never connects (different message type on
        # the same topic name), so the robot never moves.
        # config/real/real_line_follower.yaml sets this true.
        self._cmd_vel_stamped = bool(self.get_parameter("cmd_vel_stamped").value)
        cmd_vel_type = TwistStamped if self._cmd_vel_stamped else Twist
        self.cmd_pub = self.create_publisher(cmd_vel_type, self.get_parameter("cmd_vel_topic").value, 10)
        self.diag_pub = self.create_publisher(
            Float32MultiArray, self.get_parameter("diagnostic_topic").value, 10
        )
        self.create_subscription(
            Float32MultiArray, self.get_parameter("local_path_topic").value, self._local_path_callback, 10
        )

        self.timer = self.create_timer(controller_cfg.ts_control, self._on_timer)
        self.get_logger().info(
            f"Control node started: residual={'on' if self._residual_model else 'off'}, "
            f"control rate={1.0 / ts_control:.1f} Hz, "
            f"watchdog_timeout={self._watchdog_timeout:.1f}s, "
            f"subscribing to {self.get_parameter('local_path_topic').value}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("local_path_topic", "/line_follower/local_path")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("diagnostic_topic", "/line_follower/debug")
        # False (default): plain Twist. True: TwistStamped, matches this
        # project's real TurtleBot3's turtlebot3_node -- see the
        # cmd_vel_stamped comment in __init__.
        self.declare_parameter("cmd_vel_stamped", False)
        self.declare_parameter("wheel_radius", RobotConfig.wheel_radius)
        self.declare_parameter("wheel_separation", RobotConfig.wheel_separation)
        self.declare_parameter("max_wheel_speed", RobotConfig.max_wheel_speed)
        self.declare_parameter("kp", ControllerConfig.kp)
        self.declare_parameter("ki", ControllerConfig.ki)
        self.declare_parameter("kd", ControllerConfig.kd)
        self.declare_parameter("n_filter", ControllerConfig.n_filter)
        self.declare_parameter("steering_sign", ControllerConfig.steering_sign)
        # These three set the actual commanded speed magnitude -- see the
        # matching comment in line_follower_node.py's
        # _declare_parameters for why these needed adding (2026-07-31: a
        # real robot commanded 0.343 m/s linear / saturated 1.5 rad/s
        # angular off the untouched Gazebo-tuned defaults, before this fix
        # these values weren't reachable from any yaml config at all).
        self.declare_parameter("base_linear_speed", ControllerConfig.base_linear_speed)
        self.declare_parameter("base_speed_scale", ControllerConfig.base_speed_scale)
        self.declare_parameter("max_angular_speed", ControllerConfig.max_angular_speed)
        # Curve_Speed_Governor. Exposed 2026-08-10 for the same reason the
        # three above were on 2026-07-31: the node built
        # CurveSpeedGovernorConfig() from its bare dataclass defaults, so
        # nothing written in the yaml reached it. Measured on this robot at
        # kp 0.7 / kd 0 over two 60 s runs: speed is a -1.000-correlated
        # function of |heading_error| (heading_weight 1.0 always beats
        # lateral_weight 0.8 * a well-tracked lateral error), and
        # slowdown_gain -0.9 held the robot at 34-41% speed as a matter of
        # course and at the min_speed_scale 0.1 floor for 8-10% of each run.
        # That is the visible "crawls through curves, and stops and starts"
        # behaviour: 19-23 excursions below 10 mm/s per minute.
        #
        # 2026-08-11: do NOT retune these against the present signal. An
        # earlier version of this comment read heading_error's 0.66-0.73 rad
        # median as a real curvature measurement implying a ~0.31 m track
        # radius; that was wrong (retracted in 45a4467, see
        # roi_bottom_fraction's comment in real_line_follower.yaml). With the
        # robot stationary, heading_error covers its whole -1.0..+1.0 range on
        # frames whose brightness varies by 2 counts, because it is the slope
        # of a quadratic fitted over the ~0.08 m of ground depth
        # roi_bottom_fraction 0.3 leaves visible. So this governor is
        # amplifying a noisy input, not merely mistuned for the track. Fix the
        # ROI depth first; choosing a slowdown_gain against the current signal
        # is fitting to noise.
        self.declare_parameter("curve_heading_weight", CurveSpeedGovernorConfig.heading_weight)
        self.declare_parameter("curve_lateral_weight", CurveSpeedGovernorConfig.lateral_weight)
        self.declare_parameter("curve_slowdown_gain", CurveSpeedGovernorConfig.slowdown_gain)
        self.declare_parameter("curve_slowdown_bias", CurveSpeedGovernorConfig.slowdown_bias)
        self.declare_parameter("curve_min_speed_scale", CurveSpeedGovernorConfig.min_speed_scale)
        self.declare_parameter("curve_max_speed_scale", CurveSpeedGovernorConfig.max_speed_scale)
        # If no local_path message arrives within this many seconds, treat
        # the input as lost (found=False, zeroed errors) rather than replay
        # stale data -- see the module docstring's "Network-dropout
        # watchdog" section. Matches the watchdog_timeout convention already
        # used by this project's other from-scratch real-robot node
        # (~/line_follower_ws on the Pi).
        self.declare_parameter("watchdog_timeout", 1.0)
        self.declare_parameter("residual_model_path", "")
        self.declare_parameter("residual_algo", "sac")  # "sac" | "ppo"
        self.declare_parameter("residual_max_delta_omega", 1.0)
        self.declare_parameter("residual_max_delta_v", 1.0)  # only used if residual_use_2d_action
        self.declare_parameter("residual_use_wheel_speed_obs", False)
        self.declare_parameter("residual_use_2d_action", False)

        # --- Safety layer + lost-line recovery (ported from MATLAB 2026-08-09).
        # control_period must match this node's timer; the safety layer's rate
        # limits and the line memory's dead reckoning both assume a fixed dt.
        # On real hardware the loop is NOT exactly periodic (camera frame rate,
        # network, processing jitter), so this is an approximation -- measure
        # the actual period before trusting the rate limits quantitatively.
        self.declare_parameter("control_period", ControllerConfig.ts_control)
        # lookahead_distance must equal the vision node's, since the CBF's
        # barrier is built on the lookahead point that vision reports.
        self.declare_parameter("lookahead_distance", VisionConfig.lookahead_distance)
        self.declare_parameter("safety_enable_filter", SafetyFilterConfig.enable)
        # Default False: 2026-08-09 MuJoCo ablation found neither in-place
        # scanning nor memory-guided recovery beat leaving the existing
        # forward-crawl behavior alone. See LineSearchConfig's docstring.
        self.declare_parameter("safety_enable_search", LineSearchConfig.enable)
        self.declare_parameter("safety_lateral_max", SafetyFilterConfig.lateral_max)
        self.declare_parameter("safety_cbf_alpha", SafetyFilterConfig.cbf_alpha)
        self.declare_parameter("safety_speed_backoff", SafetyFilterConfig.speed_backoff)
        self.declare_parameter("safety_max_accel_v", SafetyFilterConfig.max_accel_v)
        self.declare_parameter("safety_max_accel_omega", SafetyFilterConfig.max_accel_omega)

    def _local_path_callback(self, msg: Float32MultiArray) -> None:
        try:
            self._last_local_path = unpack_local_path(msg)
        except ValueError as exc:
            self.get_logger().warn(f"malformed local_path message: {exc}")
            return
        self._last_local_path_stamp = self.get_clock().now()

    def _residual_action(self, steering_error: float, lateral_error: float,
                          heading_error: float, found: bool) -> Tuple[float, float]:
        """Returns ``(delta_v, delta_omega)``; ``delta_v`` is 0.0 whenever no
        model is loaded or the loaded model is a 1-D-action checkpoint."""
        if self._residual_model is None:
            return 0.0, 0.0
        obs = residual_observation(
            steering_error, lateral_error, heading_error, found,
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
        if self._last_local_path is None:
            return  # no local_path message received yet

        stale = (
            self._last_local_path_stamp is None
            or (self.get_clock().now() - self._last_local_path_stamp).nanoseconds
            > self._watchdog_timeout * 1e9
        )
        if stale:
            steering_error, lateral_error, heading_error, confidence, found = 0.0, 0.0, 0.0, 0.0, False
        else:
            steering_error, lateral_error, heading_error, confidence, found = self._last_local_path

        residual_delta_v, residual_delta_omega = self._residual_action(
            steering_error, lateral_error, heading_error, found
        )

        # PID (+residual) -> lost-line recovery -> safety layer -> wheels, the
        # same order as the MATLAB models' Sum -> Line_Search -> Safety_Filter
        # -> Diff_Drive_Kinematics chain.
        v_cmd, omega_cmd = self.controller.command(
            steering_error=steering_error, found=found,
            heading_error=heading_error, lateral_error=lateral_error,
            residual_delta_omega=residual_delta_omega, residual_delta_v=residual_delta_v,
        )

        # Body velocity reconstructed from the wheel speeds this node last
        # issued. This is NOT odometry -- control_node has no /odom
        # subscription -- so it cannot see wheel slip or actuator lag. It feeds
        # only the line memory's dead reckoning, which is bounded by
        # mem_max_age and, with path points unavailable here, inert anyway.
        # Subscribe to /odom before relying on it for anything.
        left_prev, right_prev = self._prev_wheel_command
        v_meas = self.robot.wheel_radius * (left_prev + right_prev) / 2.0
        omega_meas = (self.robot.wheel_radius * (right_prev - left_prev)
                      / self.robot.wheel_separation)

        # path_points=None: this node's wire format (real/local_path_msg.py)
        # carries five scalars and no path geometry, so the geometric memory
        # never becomes valid here and recovery degrades to BRAKE -> SCAN ->
        # GIVEUP. Intended, not a defect -- extending the message is a separate
        # decision. The single-process node has the points locally.
        search = self.line_search.step(
            v_cmd, omega_cmd, lateral_error, found, v_meas, omega_meas,
            path_points=None, path_point_count=0,
        )
        safety = self.safety_filter.step(
            search.v, search.omega, lateral_error, heading_error, found,
        )
        wheel_cmd = self.controller.to_wheels(safety.v, safety.omega)
        self._prev_wheel_command = (wheel_cmd.left, wheel_cmd.right)

        linear = self.robot.wheel_radius * (wheel_cmd.left + wheel_cmd.right) / 2.0
        angular = self.robot.wheel_radius * (wheel_cmd.right - wheel_cmd.left) / self.robot.wheel_separation
        self.cmd_pub.publish(self._make_cmd_vel(linear, angular))
        self._publish_diagnostics(steering_error, lateral_error, heading_error, confidence, found, wheel_cmd)

    def _make_cmd_vel(self, linear: float, angular: float):
        twist = Twist(linear=Vector3(x=linear, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=angular))
        if not self._cmd_vel_stamped:
            return twist
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist = twist
        return msg

    def _publish_diagnostics(self, steering_error: float, lateral_error: float, heading_error: float,
                              confidence: float, found: bool, wheel_cmd) -> None:
        msg = Float32MultiArray()
        msg.data = [
            float(steering_error), float(lateral_error), float(heading_error),
            float(confidence), 1.0 if found else 0.0,
            float(wheel_cmd.left), float(wheel_cmd.right),
        ]
        self.diag_pub.publish(msg)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = LineFollowerControlNode()

    # See the identical fix/comment in real/line_follower_node.py's main():
    # rclpy.init()'s own default SIGINT handler can invalidate the rcl
    # context before `except KeyboardInterrupt` below ever runs, so
    # publishing the safety-stop there raced with it and regularly failed
    # (RCLError: "publisher's context is invalid") -- confirmed on real
    # hardware 2026-08-07. Publishing synchronously from our own handler,
    # before rclpy's default handler (called via previous_handler below)
    # gets a chance to run, closes that race.
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
