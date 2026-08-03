"""Single source of truth for the residual-RL observation/action spaces.

Translated from ``rl_io_specs.m``. Both ``sim/residual_env.py`` and
``training/train_sac_residual.py`` / ``training/train_ppo_residual.py`` call
this so the SAC and PPO residual controllers can never drift apart.

Observation (5-D): [steering_error, lateral_error, heading_error, found, prev_rl_action]
Action (1-D): delta_omega in [-max_delta_omega, +max_delta_omega]  rad/s
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

OBS_LOW = np.array([-1.5, -1.0, -1.0, 0.0, -1.0], dtype=np.float32)
OBS_HIGH = np.array([1.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
OBS_NAMES = ("steering_error", "lateral_error", "heading_error", "found", "prev_rl_action")


def observation_space() -> spaces.Box:
    return spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)


def action_space(max_delta_omega: float = 1.0) -> spaces.Box:
    bound = np.array([max_delta_omega], dtype=np.float32)
    return spaces.Box(low=-bound, high=bound, dtype=np.float32)
