"""PID controller + residual mixing — translation of the PID_Controller and
SAC_Residual_Controller subsystems.

Two mixing pipelines live here side by side, matching two DIFFERENT MATLAB
models that share this module:

  * lyapunov model (visual_line_follower_sac_lyapunov.slx, archived,
    untouched since translation) — 1-D action (delta_omega only). PID_Controller
    converts straight to wheel speeds and saturates internally; the residual
    is mixed in WHEEL space via the OLD `pid_wheels()` / `mix_residual()`
    pair below. Left exactly as originally translated.

  * residual model (visual_line_follower_sac_residual.slx, ACTIVE line,
    refactored 2026-07-21) — 2-D action (delta_v, delta_omega). PID_Controller
    no longer converts to wheel speeds or saturates itself: it only outputs
    the raw (v, omega) command via `pid_velocity_command()`. The SAC action is
    now added in VELOCITY space (v_total = v_pid + delta_v, omega_total =
    omega_pid + delta_omega) via `mix_residual_velocity()`, which calls the
    single shared `diff_drive_kinematics()` conversion and saturates ONCE —
    matching the root-level Diff_Drive_Kinematics subsystem + SAC_PID_Sat
    extracted out of both PID_Controller and SAC_Residual_Controller on the
    MATLAB side (see session notes 2026-07-21: the two subsystems used to each
    carry their own copy of this conversion, with SILENTLY OPPOSITE signs for
    the omega term — PID had left=v-omega_term/right=v+omega_term, the old SAC
    residual gains had left=+omega_term/right=-omega_term. Doing the
    conversion once, downstream of where the two commands are summed, makes
    that kind of drift structurally impossible: there is only one formula).

Block wiring (verified with model_read on the .slx):

  steer_in = steering_error            if found >= 0.5
           = clamp(10*steering_error, ±0.45)   otherwise   (recovery steering)
  pid_out  = discrete PID(steer_in)    (parallel, Ts=Ts_Control,
                                        integrator: Backward Euler,
                                        derivative filter: Trapezoidal, N)
  omega_pid = SteeringSign * pid_out                              (angular_cmd)

  speed_gain = 0.8*found + 0.2                       (Line_Found_Speed_Gain+Bias)
  curve_slow = clamp(1 - 0.75*|steer_in|, 0.25, 1)   (Curve_Slowdown chain)
  v_pid      = BaseLinearSpeed * speed_gain * curve_slow    (Vision_Enable_Speed)

Diff_Drive_Kinematics (shared, root-level; v,omega -> left,right):
  left  = v*(BaseSpeedScale/WheelRadius) - omega*(WheelSeparation/(2*WheelRadius))
  right = v*(BaseSpeedScale/WheelRadius) + omega*(WheelSeparation/(2*WheelRadius))

Residual add (root level, velocity space, BEFORE Diff_Drive_Kinematics):
  v_total     = v_pid + delta_v
  omega_total = omega_pid + delta_omega
  [left, right] = clamp(diff_drive_kinematics(v_total, omega_total), ±MaxWheelSpeed)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ControllerParams:
    """Model-workspace values of the CURRENT visual_line_follower_sac_residual
    model (real-car speed sync, 2026-07-20; Burger, not the lyapunov/Waffle
    model this file was originally translated from — see git history / session
    notes for the lyapunov values if you need to reproduce the old model)."""

    base_linear_speed: float = 5.0    # BaseLinearSpeed (was 20.0 pre-sync)
    base_speed_scale: float = 0.03    # BaseSpeedScale
    kp: float = 5.0                   # Kp_Line
    ki: float = 0.0                   # Ki_Line
    kd: float = 0.35                  # Kd_Line
    n_filter: float = 20.0            # N_Line (derivative filter coefficient)
    steering_sign: float = -1.0       # SteeringSign (residual model; +1 was the
                                      # lyapunov model's mirrored-camera twin)
    wheel_radius: float = 0.033       # TB3_WheelRadius
    wheel_separation: float = 0.16    # TB3_WheelSeparation (Burger; Waffle=0.288)
    max_wheel_speed: float = 7.9      # MaxWheelSpeed (was 25.0 pre-sync)
    ts: float = 0.05                  # Ts_Control

    @property
    def yaw_to_wheel(self) -> float:
        """Ko: rad/s (yaw rate) -> rad/s (wheel differential)."""
        return self.wheel_separation / (2.0 * self.wheel_radius)

    @property
    def v_to_wheel(self) -> float:
        """Kv: BaseLinearSpeed units -> rad/s (wheel forward component)."""
        return self.base_speed_scale / self.wheel_radius


class DiscretePID:
    """Simulink discrete PID, parallel form:
    P + I*Ts*z/(z-1) [Backward Euler] + D*N/(1 + N*Ts/2*(z+1)/(z-1)) [Trapezoidal].
    """

    def __init__(self, kp: float, ki: float, kd: float, n: float, ts: float):
        self.kp, self.ki, self.kd, self.n, self.ts = kp, ki, kd, n, ts
        self.reset()

    def reset(self) -> None:
        self._xi = 0.0        # integrator state
        self._xf = 0.0        # derivative filter integrator state
        self._yd_prev = 0.0   # previous derivative-path output

    def step(self, u: float) -> float:
        # Integrator, Backward Euler: x[k] = x[k-1] + Ts*u[k]
        self._xi += self.ts * u
        i_term = self.ki * self._xi

        # Filtered derivative, trapezoidal filter integrator:
        #   y_d = N*(Kd*u - x_f),  x_f[k] = x_f[k-1] + Ts/2*(y_d[k] + y_d[k-1])
        yd = (self.n * (self.kd * u - self._xf - 0.5 * self.ts * self._yd_prev)
              / (1.0 + self.n * self.ts * 0.5))
        self._xf += 0.5 * self.ts * (yd + self._yd_prev)
        self._yd_prev = yd

        return self.kp * u + i_term + yd


class LineFollowController:
    """PID_Controller subsystem + root-level residual sum and saturation."""

    def __init__(self, params: ControllerParams | None = None):
        self.params = params or ControllerParams()
        self.pid = DiscretePID(self.params.kp, self.params.ki, self.params.kd,
                               self.params.n_filter, self.params.ts)

    def reset(self) -> None:
        # PID gains may have been re-randomized between episodes
        p = self.params
        self.pid = DiscretePID(p.kp, p.ki, p.kd, p.n_filter, p.ts)

    def _steer_in(self, steering_error: float, found: float) -> float:
        if found >= 0.5:
            return steering_error
        return float(np.clip(10.0 * steering_error, -0.45, 0.45))

    def _speed_terms(self, steer_in: float, found: float) -> tuple[float, float]:
        """(speed_gain, curve_slow) shared by both pipelines below."""
        speed_gain = 0.8 * found + 0.2
        curve_slow = float(np.clip(1.0 - 0.75 * abs(steer_in), 0.25, 1.0))
        return speed_gain, curve_slow

    # ── lyapunov pipeline (1-D action, wheel-space mixing) — UNCHANGED ──────
    def pid_wheels(self, steering_error: float, found: float) -> np.ndarray:
        """One control tick of the PID_Controller subsystem → [left, right].
        lyapunov model only — converts to wheel speeds and saturates here."""
        p = self.params
        steer_in = self._steer_in(steering_error, found)

        pid_out = self.pid.step(steer_in)
        angular = p.steering_sign * pid_out
        diff = angular * p.yaw_to_wheel

        speed_gain, curve_slow = self._speed_terms(steer_in, found)
        base_wheel = p.base_linear_speed * speed_gain * curve_slow * p.v_to_wheel

        left = float(np.clip(base_wheel - diff, -p.max_wheel_speed, p.max_wheel_speed))
        right = float(np.clip(base_wheel + diff, -p.max_wheel_speed, p.max_wheel_speed))
        return np.array([left, right])

    def mix_residual(self, pid_wheels: np.ndarray, delta_omega: float) -> np.ndarray:
        """Root-level Sum + SAC_PID_Sat: add the SAC residual, clamp.
        lyapunov model only — residual mixed in wheel space."""
        p = self.params
        residual = delta_omega * p.yaw_to_wheel * np.array([1.0, -1.0])
        return np.clip(pid_wheels + residual, -p.max_wheel_speed, p.max_wheel_speed)

    # ── residual-model pipeline (2-D action, velocity-space mixing) ─────────
    #    2026-07-21 refactor: PID_Controller no longer converts to wheel
    #    speeds or saturates by itself; that now happens once, downstream,
    #    after PID's and SAC's (v, omega) commands are summed.
    def pid_velocity_command(self, steering_error: float,
                             found: float) -> tuple[float, float]:
        """One control tick of the (refactored) PID_Controller subsystem →
        (v_pid, omega_pid). No wheel conversion, no saturation."""
        p = self.params
        steer_in = self._steer_in(steering_error, found)

        pid_out = self.pid.step(steer_in)
        omega_pid = p.steering_sign * pid_out

        speed_gain, curve_slow = self._speed_terms(steer_in, found)
        v_pid = p.base_linear_speed * speed_gain * curve_slow

        return v_pid, omega_pid

    def diff_drive_kinematics(self, v: float, omega: float) -> np.ndarray:
        """Diff_Drive_Kinematics subsystem (root level, shared by PID and
        SAC): the ONE place (v, omega) -> [left, right] wheel speed
        conversion happens. Not saturated here — SAC_PID_Sat does that once,
        after this."""
        p = self.params
        left = p.v_to_wheel * v - p.yaw_to_wheel * omega
        right = p.v_to_wheel * v + p.yaw_to_wheel * omega
        return np.array([left, right])

    def mix_residual_velocity(self, v_pid: float, omega_pid: float,
                              delta_v: float, delta_omega: float) -> np.ndarray:
        """Root-level Sum (PID + SAC in velocity space) -> shared
        Diff_Drive_Kinematics -> SAC_PID_Sat (single final saturation)."""
        p = self.params
        v_total = v_pid + delta_v
        omega_total = omega_pid + delta_omega
        wheels = self.diff_drive_kinematics(v_total, omega_total)
        return np.clip(wheels, -p.max_wheel_speed, p.max_wheel_speed)
