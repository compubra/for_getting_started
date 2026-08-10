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

import sys
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
    _lost_time: float = field(default=0.0, init=False, repr=False)

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
        self._lost_time = 0.0

    def _lost_speed_decay(self, found: bool) -> float:
        """1.0 while the line is visible or only briefly lost (matches prior
        behavior); ramps to 0.0 (full stop) the longer it stays lost -- see
        lost_speed_freeze_timeout/lost_speed_stop_timeout's docstring in
        config.py for why this exists."""
        cfg = self.controller
        if found:
            self._lost_time = 0.0
            return 1.0
        self._lost_time += cfg.ts_control
        if self._lost_time <= cfg.lost_speed_freeze_timeout:
            return 1.0
        if self._lost_time <= cfg.lost_speed_stop_timeout:
            span = max(sys.float_info.epsilon, cfg.lost_speed_stop_timeout - cfg.lost_speed_freeze_timeout)
            return 1.0 - (self._lost_time - cfg.lost_speed_freeze_timeout) / span
        return 0.0

    def command(self, steering_error: float, found: bool, heading_error: float,
                 lateral_error: float, residual_delta_omega: float = 0.0,
                 residual_delta_v: float = 0.0) -> tuple:
        """The ``(v, omega)`` command, *before* the differential-drive mixing.

        Split out of :meth:`step` on 2026-08-09 so the safety layer
        (``safety_filter.py``) and lost-line recovery (``line_search.py``) can
        sit between the controller and the wheels, which is where the MATLAB
        models put them -- ``Sum -> Line_Search -> Safety_Filter ->
        Diff_Drive_Kinematics``. ``step()`` is unchanged for every existing
        caller: it is now just ``to_wheels(command(...))``.

        ``v`` is in ``base_linear_speed`` units, not m/s (multiply by
        ``base_speed_scale``); ``omega`` is rad/s. See ``SafetyFilterConfig``'s
        ``speed_scale`` docstring for why that distinction is kept explicit.

        The lost-line speed decay is applied here rather than to the wheel
        speeds. That is equivalent -- the mixing is linear in both v and omega,
        so scaling the pair scales both wheel speeds identically -- and it
        keeps the decay visible to whatever inspects the command.
        """
        cfg = self.controller

        if found:
            switched_steering = steering_error
        else:
            switched_steering = min(cfg.recovery_limit,
                                     max(-cfg.recovery_limit, cfg.recovery_gain * steering_error))

        angular_cmd = cfg.steering_sign * self._pid.step(switched_steering)

        speed_scale = curve_speed_scale(heading_error, lateral_error, self.governor)
        line_found_gain = cfg.line_found_speed_gain * (1.0 if found else 0.0) + cfg.min_search_speed_bias
        vision_enable_speed = cfg.base_linear_speed * line_found_gain * speed_scale

        # Scale the WHOLE command, not just the forward part, or a long-lost
        # line spins the robot in place forever once the forward term alone
        # reaches zero. Equivalent to scaling both wheel speeds (the mixing is
        # linear in v and omega), just visible one stage earlier.
        lost_speed_decay = self._lost_speed_decay(found)
        v = (vision_enable_speed + residual_delta_v) * lost_speed_decay

        # NOTE the MINUS on the residual term. This reproduces what step() has
        # always computed, and step() is what BOTH deployment and training run
        # through (sim/turtlebot3_mujoco_env.py calls it), so every Python-side
        # checkpoint learned this convention and Python is self-consistent.
        #
        # It nevertheless disagrees with two things and that is NOT resolved
        # here -- flipping it would invert the residual of every checkpoint
        # trained against it:
        #   * step()'s own docstring below, which says "omega_total =
        #     omega_pid + delta_omega";
        #   * the MATLAB models, where the root Sum is a plain
        #     VOmega_pid + VOmegaResidual and Diff_Drive_Kinematics then does
        #     left = v_w - omega_w, i.e. a PLUS.
        # So a checkpoint moved between the MATLAB and Python chains has its
        # residual applied with the wrong sign. residual_policy.py's header
        # records that this exact failure (opposite signs on the omega term
        # between two copies of an equivalent conversion) already bit this
        # project once, on 2026-07-21.
        omega = (angular_cmd - residual_delta_omega) * lost_speed_decay
        return v, omega

    def to_wheels(self, v: float, omega: float) -> WheelCommand:
        """Differential-drive mixing + the single final saturation.

        Translated from the shared ``Diff_Drive_Kinematics`` +  ``SAC_PID_Sat``
        that ``build_sac_residual_controller.m`` wires at the model root.
        """
        base_wheel_speed = v * self.controller.base_speed_scale / self.robot.wheel_radius
        differential_speed = omega * self.robot.wheel_separation / (2 * self.robot.wheel_radius)
        left = base_wheel_speed - differential_speed
        right = base_wheel_speed + differential_speed
        limit = self.robot.max_wheel_speed
        return WheelCommand(
            left=min(limit, max(-limit, left)),
            right=min(limit, max(-limit, right)),
        )

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

        Since 2026-08-09 this is exactly ``to_wheels(*command(...))`` -- see
        those two for why it was split. Numerically identical to the previous
        inlined implementation (verified against it for zero and non-zero
        residuals before the split landed), including the residual-sign
        convention documented in :meth:`command`.
        """
        return self.to_wheels(*self.command(
            steering_error, found, heading_error, lateral_error,
            residual_delta_omega=residual_delta_omega, residual_delta_v=residual_delta_v,
        ))
