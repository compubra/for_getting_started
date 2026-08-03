"""PID controller + residual mixing — translation of the PID_Controller and
SAC_Residual_Controller subsystems of visual_line_follower_sac_lyapunov.slx.

Block wiring (verified with model_read on the .slx):

  steer_in = steering_error            if found >= 0.5
           = clamp(10*steering_error, ±0.45)   otherwise   (recovery steering)
  pid_out  = discrete PID(steer_in)    (parallel, Ts=Ts_Control,
                                        integrator: Backward Euler,
                                        derivative filter: Trapezoidal, N)
  angular  = SteeringSign * pid_out
  diff     = angular * WheelSeparation / (2*WheelRadius)          (YawToWheel)

  speed_gain = 0.8*found + 0.2                       (Line_Found_Speed_Gain+Bias)
  curve_slow = clamp(1 - 0.75*|steer_in|, 0.25, 1)   (Curve_Slowdown chain)
  base_wheel = BaseLinearSpeed * speed_gain * curve_slow
               * BaseSpeedScale / WheelRadius        (LinearToWheel gain)

  pid_left  = clamp(base_wheel - diff, ±MaxWheelSpeed)
  pid_right = clamp(base_wheel + diff, ±MaxWheelSpeed)

Residual add (root level):
  left_total  = clamp(pid_left  + delta_omega*YawToWheel, ±MaxWheelSpeed)
  right_total = clamp(pid_right - delta_omega*YawToWheel, ±MaxWheelSpeed)
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
        return self.wheel_separation / (2.0 * self.wheel_radius)


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

    def pid_wheels(self, steering_error: float, found: float) -> np.ndarray:
        """One control tick of the PID_Controller subsystem → [left, right]."""
        p = self.params

        if found >= 0.5:
            steer_in = steering_error
        else:
            steer_in = float(np.clip(10.0 * steering_error, -0.45, 0.45))

        pid_out = self.pid.step(steer_in)
        angular = p.steering_sign * pid_out
        diff = angular * p.yaw_to_wheel

        speed_gain = 0.8 * found + 0.2
        curve_slow = float(np.clip(1.0 - 0.75 * abs(steer_in), 0.25, 1.0))
        base_wheel = (p.base_linear_speed * speed_gain * curve_slow
                      * p.base_speed_scale / p.wheel_radius)

        left = float(np.clip(base_wheel - diff, -p.max_wheel_speed, p.max_wheel_speed))
        right = float(np.clip(base_wheel + diff, -p.max_wheel_speed, p.max_wheel_speed))
        return np.array([left, right])

    def mix_residual(self, pid_wheels: np.ndarray, delta_omega: float) -> np.ndarray:
        """Root-level Sum + SAC_PID_Sat: add the SAC residual, clamp."""
        p = self.params
        residual = delta_omega * p.yaw_to_wheel * np.array([1.0, -1.0])
        return np.clip(pid_wheels + residual, -p.max_wheel_speed, p.max_wheel_speed)
