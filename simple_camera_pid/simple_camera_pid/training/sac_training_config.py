"""Central tuning interface for SAC residual RL training.

Translated from ``sac_training_config.m``. Edit this file to tune training;
the entry point is ``training/train_sac_residual.py``. The MATLAB project
kept two duplicate entry points (a Live Script and a plain-script fallback
for environments without the Live Editor, e.g. HPC batch) — that split is a
MATLAB Live Editor concern with no Python equivalent, so this port has one
canonical training script.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..common.config import (
    ControllerConfig, CurveSpeedGovernorConfig, ResidualRewardConfig, VisionConfig,
)
from .domain_randomization import DomainRandomizationConfig, dr_defaults


def real_robot_baseline_controller() -> ControllerConfig:
    """The base controller the residual is trained to correct, matching the real
    robot as tuned on 2026-08-11 (see config/real/real_line_follower.yaml).

    A residual policy learns a correction on top of a FIXED base controller, so
    the base has to be the one the policy will meet on the robot. Until this
    existed, training silently used ``ControllerConfig``'s own defaults --
    kp 5.0 / base_speed_scale 0.03, the pre-2026-08-04 Gazebo values -- which
    drove the simulated robot off the track: found_rate 0.236 over 60 s with
    the wheel command pinned at its 20 rad/s limit. With these values the same
    run holds found_rate 1.000 and a median |lateral_error| of 0.030, against
    0.032 measured on the robot.

    Kept as a function rather than a dataclass default so that
    ``ControllerConfig()`` elsewhere (the MuJoCo ROS node, the experiments)
    keeps its own history and is not silently retuned by an RL-side decision.
    """
    return ControllerConfig(kp=1.2, ki=0.1, kd=0.05,
                            max_angular_speed=1.4, base_speed_scale=0.007)


def real_robot_baseline_governor() -> CurveSpeedGovernorConfig:
    """Curve speed governor as tuned on the robot 2026-08-11: the shipped
    -0.9/0.1 held it at 34-41% speed as a matter of course and stopped it
    outright 16-22 times a minute."""
    return CurveSpeedGovernorConfig(slowdown_gain=-0.5, min_speed_scale=0.35)


def simulation_vision() -> VisionConfig:
    """Vision settings for the SIMULATED track -- deliberately NOT the robot's.

    roi_bottom_fraction and lookahead_distance depend on the track's tightest
    curve and on the line's width, and the two differ by an order of magnitude:
    the MuJoCo track is a 5.9 cm line, the real track a ~40 cm wide path. The
    robot's 0.4 was tried here and lost the line at t=50 s on `simple`'s
    hairpin -- at that depth the ROI spans only 0.163-0.363 m of ground, and
    the line leaves it faster than the robot can turn. 0.50 is the value this
    simulation already used and it does not.

    The brightness/saturation thresholds are likewise left at the dataclass
    defaults: the robot's were measured against real carpet under real
    lighting, and mean nothing against a rendered frame.
    """
    return VisionConfig(roi_bottom_fraction=0.50, lookahead_distance=0.20)


@dataclass
class SACAgentConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    discount_factor: float = 0.99
    batch_size: int = 256
    buffer_size: int = 200_000
    entropy_weight: float = 0.2       # stable-baselines3 tunes entropy automatically by
                                       # default (ent_coef="auto"); see train_sac_residual.py
    target_smooth_tau: float = 5e-3
    max_delta_omega: float = 1.0


@dataclass
class SACTrainingConfig:
    # 0) Model / map
    map_key: str = "track_hard"

    # 1) Training loop
    #
    # 2026-08-11: episodes lengthened 400 -> 1200 steps alongside the drop to
    # the robot's own base_speed_scale (0.03 -> 0.007). At 0.107 m/s a 400-step
    # episode covers about a quarter of the ground the old 0.4 m/s one did, so
    # the agent would see correspondingly less of the track per episode; 1200
    # steps is 60 s, which is also the length of every measured run on the
    # robot. total_timesteps is held at 320k so the wall-clock cost does not
    # triple with it -- that is 267 episodes rather than 800, and is the number
    # to raise first if learning is still improving when it stops.
    total_timesteps: int = 320_000     # 267 episodes * 1200 steps
    max_steps_per_episode: int = 1200  # 1200*0.05s = 60s/episode
    save_checkpoint_every: int = 20 * 1200  # ScoreWindow-episode cadence, in timesteps

    # 2) Reward weights + done timeout
    reward: ResidualRewardConfig = field(default_factory=ResidualRewardConfig)

    # 3) SAC hyperparameters
    agent: SACAgentConfig = field(default_factory=SACAgentConfig)

    # 4) Domain randomization
    use_domain_rand: bool = True
    domain_rand: DomainRandomizationConfig = field(default_factory=dr_defaults)

    # 5) The base plant the residual sits on top of. These are what make the
    # simulated baseline the same controller the policy will meet on the robot
    # -- see each factory's docstring, and note that `vision` is deliberately
    # the SIMULATION's values, not the robot's.
    controller: ControllerConfig = field(default_factory=real_robot_baseline_controller)
    governor: CurveSpeedGovernorConfig = field(default_factory=real_robot_baseline_governor)
    vision: VisionConfig = field(default_factory=simulation_vision)

    # 6) Output
    save_dir: str = ""  # "" -> simulation_data/sac_training

    def __post_init__(self) -> None:
        self.reward.max_delta_omega = self.agent.max_delta_omega
        # Map randomization is off by default so training repeatedly sees `map_key`
        # (matches sac_training_config.m: `cfg.DomainRand.Map.Enable = false;`).
        self.domain_rand.map.enable = False
        if not self.use_domain_rand:
            self.domain_rand.enable = False
