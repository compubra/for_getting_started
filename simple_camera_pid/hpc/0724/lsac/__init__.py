"""Python translation of the MATLAB/Simulink SAC residual visual line
follower, updated 2026-07-23 to track the current
visual_line_follower_sac_residual.slx (superseding hpc/0721's snapshot):

  - Reward_Calculation gained SAC_Q_Lateral/SAC_Q_Heading terms (see
    reward.py's QuadraticReward) alongside the pre-existing steering-error/
    action-energy/lost-line terms.
  - rl_dr_defaults.m gained a 3-way GenTrack.Generator choice
    (simple/complex/wandering/random, see track_gen.py + domain_rand.py) on
    top of the pre-existing ellipse/capsule "simple" generator.

Source of truth for the translation:
  - src/simple_camera_pid/matlab/train/sac/sac_training_config.m +
    train_sac_residual_script.m (training loop, reward weights)
  - src/simple_camera_pid/matlab/train/shared/rl_io_specs.m,
    rl_domain_randomization.m, rl_dr_defaults.m
  - src/simple_camera_pid/matlab/runtime/scene/gen_simple_track_scene.m,
    gen_complex_track_scene.m, gen_wandering_track_scene.m
  - src/simple_camera_pid/matlab/visual_line_follower_sac_residual.slx
    (block wiring extracted via model_read)
"""

from .env import LineFollowerEnv
from .scenes import resolve_scene, MAP_KEYS

__all__ = ["LineFollowerEnv", "resolve_scene", "MAP_KEYS"]
