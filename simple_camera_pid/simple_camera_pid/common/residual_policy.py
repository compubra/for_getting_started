"""Shared inference-time helpers for the optional SAC/PPO residual policy
loaded by mujoco_line_follower_node.py and line_follower_node.py.

Centralized here rather than duplicated in both nodes: the two node files
consume the SAME HPC-side observation/action layout, and hpc/*/lsac/
controller.py's own session notes (2026-07-21) record a real bug where two
independent copies of an equivalent conversion silently diverged (opposite
signs on the omega term) -- keeping one copy here avoids that class of bug.

Three checkpoint generations exist (see hpc/ sweep history):
  * 5-D obs / 1-D action (delta_omega only) -- simple_camera_pid.training's
    in-repo SAC/PPO (io_specs.py / residual_env.py). No wheel-speed
    feedback, single prev_action slot.
  * 7-D obs / 1-D action -- pre-2026-07-21 HPC lsac checkpoints (hpc/0720
    and the 2026-07-12 runs). Adds normalized wheel-speed feedback
    (v_norm, omega_norm) to the 5-D base.
  * 8-D obs / 2-D action ([delta_v, delta_omega]) -- 2026-07-21+ HPC lsac
    checkpoints (hpc/0721, hpc/0724), after the Diff_Drive_Kinematics
    refactor moved the residual sum into velocity space. The single
    prev_action slot becomes two (prev_delta_v, prev_delta_omega).
use_wheel_speed_obs and use_2d_action are independent switches (four
combinations), matching the two nodes' residual_use_wheel_speed_obs /
residual_use_2d_action parameters -- only the first three combinations
above correspond to a checkpoint anyone has actually trained.
"""
from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import numpy as np


def residual_observation(
    steering_error: float, lateral_error: float, heading_error: float, found: bool,
    prev_delta_v: float, prev_delta_omega: float,
    use_wheel_speed_obs: bool, use_2d_action: bool,
    wheel_command: Tuple[float, float] = (0.0, 0.0), max_wheel_speed: float = 1.0,
) -> np.ndarray:
    """Build the observation vector, matching hpc/*/lsac/env.py's ObsMux
    layout exactly (order matters: it must match residual_spaces() below and
    whatever the loaded checkpoint was trained against)."""
    obs = [steering_error, lateral_error, heading_error, 1.0 if found else 0.0]
    obs += [prev_delta_v, prev_delta_omega] if use_2d_action else [prev_delta_omega]
    if use_wheel_speed_obs:
        left, right = wheel_command
        obs += [(left + right) / (2.0 * max_wheel_speed), (right - left) / (2.0 * max_wheel_speed)]
    return np.array(obs, dtype=np.float32)


def residual_spaces(
    use_wheel_speed_obs: bool, use_2d_action: bool,
    max_delta_v: float, max_delta_omega: float,
) -> Tuple[gym.spaces.Box, gym.spaces.Box]:
    """The observation/action Box a checkpoint of the given generation was
    trained with, matching hpc/*/lsac/env.py's Box definitions exactly.
    Used to override the pickled (and possibly Numpy-version-incompatible)
    spaces at load time -- see load_residual_model()."""
    obs_low = [-1.5, -1.0, -1.0, 0.0]
    obs_high = [1.5, 1.0, 1.0, 1.0]
    if use_2d_action:
        obs_low += [-max_delta_v, -max_delta_omega]
        obs_high += [max_delta_v, max_delta_omega]
        action_low, action_high = [-max_delta_v, -max_delta_omega], [max_delta_v, max_delta_omega]
    else:
        obs_low += [-max_delta_omega]
        obs_high += [max_delta_omega]
        action_low, action_high = [-max_delta_omega], [max_delta_omega]
    if use_wheel_speed_obs:
        obs_low += [-1.0, -1.0]
        obs_high += [1.0, 1.0]
    observation_space = gym.spaces.Box(
        low=np.array(obs_low, dtype=np.float32), high=np.array(obs_high, dtype=np.float32))
    action_space = gym.spaces.Box(
        low=np.array(action_low, dtype=np.float32), high=np.array(action_high, dtype=np.float32))
    return observation_space, action_space


def load_residual_model(path: str, algo: str, observation_space: gym.spaces.Box,
                         action_space: gym.spaces.Box):
    """Load a SAC/PPO residual checkpoint for inference (``predict()``) only.

    Overrides observation_space/action_space -- and every training-only
    bookkeeping field SB3 restores for resuming training (replay buffer,
    train_freq, lr_schedule, last-obs/episode bookkeeping) -- via
    stable_baselines3's ``custom_objects`` mechanism rather than trusting
    whatever gets unpickled. Two independent reasons:

      1. We already know the correct spaces from our own launch parameters
         and residual_observation()'s layout -- no need to round-trip them
         through the checkpoint's pickle at all.
      2. Checkpoints trained on HPC under Numpy 2.x (see their
         system_info.txt) fail to unpickle under Numpy 1.x
         (``ModuleNotFoundError: No module named 'numpy._core.numeric'``) if
         SB3 is allowed to deserialize those fields normally -- routing
         around them via custom_objects avoids the numpy call entirely
         (verified 2026-07-28; do NOT try aliasing numpy._core ->
         numpy.core in sys.modules instead, that segfaults rather than
         raising -- the fields it corrupts are read by C-level tensor/array
         reconstruction, not caught by any Python try/except).

    None of the skipped fields are read by predict()-only inference, so this
    is safe for every checkpoint generation, not just the Numpy-2.x ones --
    a same-Numpy-version checkpoint just gets the identical values it would
    have unpickled anyway.
    """
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit

    custom_objects = {
        "observation_space": observation_space,
        "action_space": action_space,
        "lr_schedule": lambda _: 0.0,
        "_last_obs": None,
        "_last_original_obs": None,
        "_last_episode_starts": None,
        "ep_info_buffer": [],
        "ep_success_buffer": [],
        "replay_buffer_class": ReplayBuffer,
        "train_freq": TrainFreq(1, TrainFrequencyUnit.STEP),
    }
    if algo.lower() == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(path, custom_objects=custom_objects)
    from stable_baselines3 import SAC

    return SAC.load(path, custom_objects=custom_objects)
