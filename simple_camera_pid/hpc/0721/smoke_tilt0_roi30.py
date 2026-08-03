#!/usr/bin/env python
"""One-off validation for the genpath_roi30_tilt0 sweep config: gen-track
(random shape, difficulty 0.4), roi_bottom_fraction=0.3, cam_pitch_deg=0.0.

Checks the PID-only loop can still find the line often enough with the
camera un-tilted (view direction along +X, no downward pitch) cropped to the
bottom 30% of the image -- this combination wasn't validated before since
cam_pitch_deg didn't exist until this session. Not wired into the general
smoke_residual.sbatch gate; run standalone before submitting the sweep.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/users/elp25qz/LSAC/train_bundle/python/0721")

from lsac.controller import ControllerParams
from lsac.domain_rand import DRConfig
from lsac.env import LineFollowerEnv
from lsac.reward import QuadraticReward, QuadraticRewardParams
from lsac.vision import VisionParams


def make_env(seed: int = 0) -> LineFollowerEnv:
    return LineFollowerEnv(
        map_key="simple",  # ignored, gen_track_enable=True takes priority
        controller_params=ControllerParams(steering_sign=-1.0),
        vision_params=VisionParams(roi_bottom_fraction=0.3, cam_pitch_deg=0.0),
        reward_fn=QuadraticReward(QuadraticRewardParams()),
        domain_rand=DRConfig(map_enable=False, gen_track_enable=True,
                             gen_track_shape="random", gen_track_difficulty=0.4),
        mirror_camera=False,
        observe_wheel_speeds=True,
        cam_pitch_deg=0.0,
        seed=seed,
    )


def main() -> None:
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (8,), f"expected 8-D obs, got {obs.shape}"
    found = 0
    terminated_at = None
    n = 400  # 20 s PID-only, several gen-track episodes via resets below
    for k in range(n):
        obs, r, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        found += info["found"] > 0.5
        if term:
            terminated_at = k
            obs, _ = env.reset()
    pct = 100.0 * found / n
    print(f"cam_pitch_deg=0, roi=0.3, gen-track random/0.4: {n} steps, "
          f"line found {pct:.1f}%, first termination at step "
          f"{terminated_at}, final obs {np.round(obs, 3)}", flush=True)
    env.close()
    if pct < 50.0:
        print("WARNING: found-rate below 50% -- tilt=0 + roi=0.3 may be too "
              "aggressive a crop for this camera geometry", flush=True)
        sys.exit(1)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
