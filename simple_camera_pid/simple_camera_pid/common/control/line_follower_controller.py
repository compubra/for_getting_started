"""PID line-follower controller: recovery steering, PID, curve slowdown, kinematics.

Translated from the ``Subsystem`` (renamed ``PID_Controller`` in the SAC/PPO
residual models) and ``Curve_Speed_Governor`` blocks inside
``visual_line_follower_with_debug.slx``, read via the MATLAB MCP
``model_read``/``model_query_params`` tools:

  1. Recovery steering: while the line is visible, steer on
     ``steering_error`` directly; once lost, switch to
     ``clamp(10 * steering_error, -0.45, 0.45)`` — the vision module already
     freezes/decays ``steering_error`` itself while lost (see
     ``vision.py``), so this is a secondary safety clamp on the
     *pre-lost-line* value, matching the Simulink ``Switch``/``Gain``/
     ``Saturate`` chain exactly.
  2. ``LinePID``: filtered-derivative PID (see ``pid.py``) on the switched
     steering signal, output saturated to +/-``max_angular_speed``.
  3. ``Curve_Speed_Governor``: slows the robot ahead of curves based on how
     large ``heading_error``/``lateral_error`` currently are.
  4. Wheel mixing: base (forward) speed and yaw-rate command are combined
     into differential wheel speeds and saturated to +/-``max_wheel_speed``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import ControllerConfig, CurveSpeedGovernorConfig, RobotConfig
from .pid import FilteredPID


@dataclass
class WheelCommand:
    left: float
    right: float


def curve_speed_scale(heading_error: float, lateral_error: float, cfg: CurveSpeedGovernorConfig) -> float:
    """Translated from the ``Curve_Speed_Governor`` subsystem."""
    heading_term = cfg.heading_weight * abs(heading_error)
    lateral_term = cfg.lateral_weight * abs(lateral_error)
    severity = min(1.0, max(0.0, max(heading_term, lateral_term)))
    scale = cfg.slowdown_gain * severity + cfg.slowdown_bias
    return min(cfg.max_speed_scale, max(cfg.min_speed_scale, scale))


@dataclass
class LineFollowerController:
    """Stateful PID line-follower (the ``Subsystem``/``PID_Controller`` block)."""

    robot: RobotConfig = field(default_factory=RobotConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    governor: CurveSpeedGovernorConfig = field(default_factory=CurveSpeedGovernorConfig)

    _pid: FilteredPID = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pid = FilteredPID(
            kp=self.controller.kp, ki=self.controller.ki, kd=self.controller.kd,
            n_filter=self.controller.n_filter, ts=self.controller.ts_control,
            out_min=-self.controller.max_angular_speed, out_max=self.controller.max_angular_speed,
        )

    def reset(self) -> None:
        self._pid.kp = self.controller.kp
        self._pid.ki = self.controller.ki
        self._pid.kd = self.controller.kd
        self._pid.n_filter = self.controller.n_filter
        self._pid.reset()

    def step(self, steering_error: float, found: bool, heading_error: float,
              lateral_error: float, residual_delta_omega: float = 0.0,
              residual_delta_v: float = 0.0) -> WheelCommand:
        """``residual_delta_omega``/``residual_delta_v`` are the SAC/PPO residual
        action (rad/s yaw-rate / linear-speed-units), summed with the PID
        (omega, v) commands *before* the wheel-speed conversion and the single
        final saturation below -- translated from the root-level residual sum
        + shared Diff_Drive_Kinematics + SAC_PID_Sat that
        ``build_sac_residual_controller.m`` wires alongside (not inside) the
        PID_Controller subsystem: v_total = v_pid + delta_v, omega_total =
        omega_pid + delta_omega, [left, right] = clamp(diff_drive(v_total,
        omega_total)). Both zero for the plain PID baseline (and
        ``residual_delta_v`` alone is zero for a 1-D-action (delta_omega-only)
        residual policy).
        """
        cfg = self.controller

        if found:
            switched_steering = steering_error
        else:
            switched_steering = min(cfg.recovery_limit,
                                     max(-cfg.recovery_limit, cfg.recovery_gain * steering_error))

        angular_cmd = cfg.steering_sign * self._pid.step(switched_steering)
        differential_speed = angular_cmd * self.robot.wheel_separation / (2 * self.robot.wheel_radius)

        speed_scale = curve_speed_scale(heading_error, lateral_error, self.governor)
        line_found_gain = cfg.line_found_speed_gain * (1.0 if found else 0.0) + cfg.min_search_speed_bias
        vision_enable_speed = cfg.base_linear_speed * line_found_gain * speed_scale
        base_wheel_speed = vision_enable_speed * cfg.base_speed_scale / self.robot.wheel_radius

        residual_v_wheel = residual_delta_v * cfg.base_speed_scale / self.robot.wheel_radius
        residual_diff = residual_delta_omega * self.robot.wheel_separation / (2 * self.robot.wheel_radius)
        left = base_wheel_speed + residual_v_wheel - differential_speed + residual_diff
        right = base_wheel_speed + residual_v_wheel + differential_speed - residual_diff
        limit = self.robot.max_wheel_speed
        return WheelCommand(
            left=min(limit, max(-limit, left)),
            right=min(limit, max(-limit, right)),
        )
