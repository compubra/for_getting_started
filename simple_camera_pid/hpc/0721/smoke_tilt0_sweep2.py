#!/usr/bin/env python
"""One-off validation for submit_tilt0_sweep2.sh's 12 new runs before
submitting them. Round 1 (submit_tilt0_sweep.sh) already validated that
tilt=0 + roi=0.3 + gen-track(shape=random) renders and tracks the line; round
2 only changes SAC/reward hyperparameters (lr, gamma, tau, p-lost) plus two
axes that DO touch the env/scene:
  - gen-track-shape pinned to "ellipse" and "capsule" (round 1 only ever saw
    whichever shape "random" happened to sample) -- must confirm the PID can
    still track both shapes at tilt=0/roi=0.3, not just the random mix.
  - max-delta-v/max-delta-omega bounds of 0.5 and 2.0 -- confirm the action
    space and SAC construction still work at both extremes.

Exits nonzero on failure.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/users/elp25qz/LSAC/train_bundle/python/0721")

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from lsac.controller import ControllerParams
from lsac.domain_rand import DRConfig
from lsac.env import LineFollowerEnv
from lsac.reward import QuadraticReward, QuadraticRewardParams
from lsac.vision import VisionParams


def make_env(shape: str, max_delta: float = 1.0, p_lost: float = 10.0,
            seed: int = 0) -> LineFollowerEnv:
    return LineFollowerEnv(
        map_key="simple",  # ignored, gen_track_enable=True takes priority
        max_delta_v=max_delta,
        max_delta_omega=max_delta,
        controller_params=ControllerParams(steering_sign=-1.0),
        vision_params=VisionParams(roi_bottom_fraction=0.3, cam_pitch_deg=0.0),
        reward_fn=QuadraticReward(QuadraticRewardParams(p_lost=p_lost)),
        domain_rand=DRConfig(map_enable=False, gen_track_enable=True,
                             gen_track_shape=shape, gen_track_difficulty=0.4),
        mirror_camera=False,
        observe_wheel_speeds=True,
        cam_pitch_deg=0.0,
        seed=seed,
    )


def check_shape_tracks(shape: str) -> None:
    env = make_env(shape)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (8,), f"expected 8-D obs, got {obs.shape}"
    found = 0
    n = 300
    for k in range(n):
        obs, r, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        found += info["found"] > 0.5
        if term:
            raise AssertionError(f"{shape}: PID-only rollout terminated at step {k}")
    pct = 100.0 * found / n
    print(f"shape={shape}: {n} steps, line found {pct:.1f}%", flush=True)
    assert pct > 80.0, f"{shape}: PID lost the line too often ({pct:.1f}% found)"
    env.close()


def check_action_bounds(max_delta: float) -> None:
    env = make_env("random", max_delta=max_delta)
    obs, _ = env.reset(seed=0)
    assert env.action_space.low[0] == -max_delta
    assert env.action_space.high[0] == max_delta
    for _ in range(20):
        env.step(env.action_space.sample())
    env.close()
    print(f"max-delta-v/omega={max_delta}: action space + step OK", flush=True)


def check_sac_construction(lr: float, gamma: float, tau: float) -> None:
    env = Monitor(TimeLimit(make_env("random"), max_episode_steps=50))
    model = SAC("MlpPolicy", env, learning_rate=lr, gamma=gamma, tau=tau,
               batch_size=64, learning_starts=64,
               policy_kwargs={"net_arch": [256, 256]}, seed=0, device="cpu",
               verbose=0)
    model.learn(total_timesteps=150)
    env.close()
    print(f"SAC construction lr={lr} gamma={gamma} tau={tau}: 150 steps OK",
          flush=True)


if __name__ == "__main__":
    check_shape_tracks("ellipse")
    check_shape_tracks("capsule")
    check_action_bounds(0.5)
    check_action_bounds(2.0)
    check_sac_construction(1e-4, 0.99, 5e-3)
    check_sac_construction(1e-3, 0.99, 5e-3)
    check_sac_construction(3e-4, 0.95, 5e-3)
    check_sac_construction(3e-4, 0.999, 5e-3)
    check_sac_construction(3e-4, 0.99, 1e-3)
    check_sac_construction(3e-4, 0.99, 2e-2)
    print("SMOKE OK")
    sys.exit(0)
