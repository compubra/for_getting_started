"""Lyapunov reward + done detection — translation of the Reward_Calculation and
Done_Detection subsystems of the R2026a visual_line_follower_sac_lyapunov.slx
(wiring extracted from the raw .slx XML; matches the formula documented in
archive_lyapunov/code/train_sac_lyapunov_agent.m):

  V      = Q_e*e_steer^2 + Q_l*e_lat^2 + Q_h*e_head^2 + Q_v*v_lat^2
    v_lat = hypot(vx, vy) * sin(e_head)      (true ground speed × vision heading)
  Vdot   = gate * (V[k] - V[k-1]) / Ts       (gate = 0 on the first sample only)
  reward = -tanh(V + Lambda*Vdot)                       (V_pen)
           - R*u^2                                      (Action_Cost)
           - P_Lost*(1 - (found > 0.5))                 (Lost_Penalty)
           - found * K_inv*(1 - exp(-Beta*(|e_s|+|e_l|+|e_h|)))   (Inv_Bonus)
           + Alive*(found > 0.5)                        (Alive_Bonus)

The four archive terms are <= 0, which made every episode-total negative and
early termination (lost-line done) strictly better than surviving the full
episode — the 2000-episode SAC run (job 10797287) plateaued because of this.
Two additions fix the incentive ordering:
  Alive_Bonus  (alive > typical tracking cost) makes a well-tracked step
               net-positive, so longer successful episodes score higher;
  P_Done       one-time penalty applied by the env on lost-line termination,
               so dying early is strictly worse than any full episode.

Done: consecutive-lost-line counter; is_done when count > done_steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RewardParams:
    q_e: float = 1.0        # SAC_Q_e
    q_l: float = 1.0        # SAC_Q_l
    q_h: float = 1.0        # SAC_Q_h
    q_v: float = 0.3        # SAC_Q_v (weight on the lateral approach speed)
    lam: float = 0.5        # SAC_Lambda
    r: float = 0.1          # SAC_R
    p_lost: float = 1.0     # SAC_P_Lost
    k_inv: float = 0.4      # SAC_K_inv
    beta: float = 3.0       # SAC_Beta
    ts: float = 0.05        # Ts_Control
    alive: float = 0.5      # per-step bonus while the line is found
    p_done: float = 50.0    # one-time penalty on lost-line termination


class LyapunovReward:
    def __init__(self, params: RewardParams | None = None):
        self.params = params or RewardParams()
        self.reset()

    def reset(self) -> None:
        self._v_prev = 0.0
        self._gate = 0.0  # UnitDelay of constant 1: 0 on first sample, then 1

    def lyapunov_v(self, e_steer: float, e_lat: float, e_head: float,
                   v_lat: float) -> float:
        p = self.params
        return (p.q_e * e_steer ** 2 + p.q_l * e_lat ** 2
                + p.q_h * e_head ** 2 + p.q_v * v_lat ** 2)

    def step(self, e_steer: float, e_lat: float, e_head: float,
             v_lat: float, u: float, found: float) -> float:
        p = self.params

        v = self.lyapunov_v(e_steer, e_lat, e_head, v_lat)
        vdot = self._gate * (v - self._v_prev) / p.ts
        self._v_prev = v
        self._gate = 1.0

        v_pen = -math.tanh(v + p.lam * vdot)
        a_cost = -p.r * u ** 2
        l_pen = -p.p_lost * (1.0 - float(found > 0.5))
        bonus = -found * p.k_inv * (
            1.0 - math.exp(-p.beta * (abs(e_steer) + abs(e_lat) + abs(e_head))))
        alive = p.alive * float(found > 0.5)
        return v_pen + a_cost + l_pen + bonus + alive


@dataclass
class QuadraticRewardParams:
    """Model-workspace weights of the SAC/PPO residual models
    (sac_training_config.m / ppo_training_config.m, section 3)."""

    q: float = 1.0        # SAC_Q / PPO_Q       — steering-error penalty
    r: float = 0.1        # SAC_R / PPO_R       — action-cost penalty
    p_lost: float = 10.0  # SAC_P_Lost / PPO_P_Lost — per-step lost-line penalty
    p_done: float = 0.0   # not present in the MATLAB models (kept for the env
                          # termination hook; 0 = faithful translation)


class QuadraticReward:
    """Reward_Calculation subsystem of visual_line_follower_{sac,ppo}_residual:

      reward(t) = -(Q*e^2 + R*u^2) - P_Lost*(1 - found)

    e is the vision steering_error only (no pose information), u the residual
    action. Stateless — the reset/step interface just mirrors LyapunovReward
    so the env can use either interchangeably.
    """

    def __init__(self, params: QuadraticRewardParams | None = None):
        self.params = params or QuadraticRewardParams()

    def reset(self) -> None:
        pass

    def step(self, e_steer: float, e_lat: float, e_head: float,
             v_lat: float, u: float, found: float) -> float:
        p = self.params
        return (-(p.q * e_steer ** 2 + p.r * u ** 2)
                - p.p_lost * (1.0 - float(found > 0.5)))


class DoneDetector:
    """Consecutive lost-line counter (Done_Detection subsystem).

    count[k] = 0 if found >= 0.5 else count[k-1] + 1;  done = count > limit.
    With Ts=0.05 and limit=100 that is 5 s of continuous line loss.
    """

    def __init__(self, done_steps: int = 100):
        self.done_steps = done_steps
        self.reset()

    def reset(self) -> None:
        self._count = 0

    def step(self, found: float) -> bool:
        self._count = 0 if found >= 0.5 else self._count + 1
        return self._count > self.done_steps
