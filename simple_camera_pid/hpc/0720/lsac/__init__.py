"""Python translation of the MATLAB/Simulink Lyapunov-SAC visual line follower.

Source of truth for the translation:
  - src/simple_camera_pid/matlab/runtime/originbot_local_path_generator.m
  - archive_lyapunov/models/visual_line_follower_sac_lyapunov.slx (R2026a,
    block wiring extracted from the raw .slx XML)
  - archive_lyapunov/code/train_sac_lyapunov_agent.m
"""

from .env import LineFollowerEnv
from .scenes import resolve_scene, MAP_KEYS

__all__ = ["LineFollowerEnv", "resolve_scene", "MAP_KEYS"]
