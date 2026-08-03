#!/usr/bin/env python
"""Smoke check for the residual-model spec (8-D obs, 2-D action
[delta_v, delta_omega], SteeringSign=-1 + upright camera, ROIFraction=0.5)
before the full SAC/PPO runs.

⚠️ 2026-07-21: action grew from 1-D to 2-D and PID+residual mixing now goes
through the shared Diff_Drive_Kinematics conversion (see lsac/controller.py
and lsac/env.py docstrings) — this smoke check exercises that path with a
zero action, which is exactly equivalent to the old PID-only check (zero
delta_v/delta_omega contribute nothing to either v or omega before mixing).

Asserts the PID-only loop still tracks the line (the sign/ROI values were read
out of the .slx models, which were tuned on a different pipeline snapshot), and
that both SB3 agents train a few episodes end-to-end on the 8-D observation.
Exits nonzero on failure so sbatch --dependency=afterok gates the full jobs.
"""

from __future__ import annotations

import sys

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

from lsac.controller import ControllerParams
from lsac.domain_rand import DRConfig
from lsac.env import LineFollowerEnv
from lsac.reward import QuadraticReward, QuadraticRewardParams
from lsac.vision import VisionParams


def make_env(domain_rand: bool = False, seed: int = 0) -> LineFollowerEnv:
    return LineFollowerEnv(
        map_key="simple",
        controller_params=ControllerParams(steering_sign=-1.0),
        vision_params=VisionParams(roi_bottom_fraction=0.5),
        reward_fn=QuadraticReward(QuadraticRewardParams()),
        domain_rand=DRConfig(map_enable=False) if domain_rand else None,
        mirror_camera=False,
        observe_wheel_speeds=True,
        seed=seed,
    )


def check_pid_tracks() -> None:
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (8,), f"expected 8-D obs, got {obs.shape}"
    found = 0
    n = 400  # 20 s PID-only
    for k in range(n):
        obs, r, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        found += info["found"] > 0.5
        if term:
            raise AssertionError(f"PID-only rollout terminated at step {k}")
    pct = 100.0 * found / n
    print(f"PID-only rollout: {n} steps, line found {pct:.1f}%, "
          f"final obs {np.round(obs, 3)}", flush=True)
    assert pct > 80.0, f"PID lost the line too often ({pct:.1f}% found)"
    env.close()


def check_training(algo: str) -> None:
    env = Monitor(TimeLimit(make_env(domain_rand=True, seed=1),
                            max_episode_steps=50))
    kwargs = dict(policy_kwargs={"net_arch": [256, 256]}, seed=0,
                  device="cpu", verbose=0)
    if algo == "sac":
        model = SAC("MlpPolicy", env, learning_starts=64, batch_size=64,
                    **kwargs)
    else:
        model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=2,
                    **kwargs)
    model.learn(total_timesteps=150)
    print(f"{algo.upper()} smoke training OK (150 steps, DR on)", flush=True)
    env.close()


if __name__ == "__main__":
    check_pid_tracks()
    check_training("sac")
    check_training("ppo")
    print("SMOKE OK")
    sys.exit(0)
