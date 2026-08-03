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

from ..common.config import ResidualRewardConfig
from .domain_randomization import DomainRandomizationConfig, dr_defaults


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
    total_timesteps: int = 800 * 400   # MaxEpisodes * MaxStepsPerEpisode (800 episodes * 20s @ 20Hz)
    max_steps_per_episode: int = 400   # 400*0.05s = 20s/episode
    save_checkpoint_every: int = 20 * 400  # ScoreWindow-episode cadence, in timesteps

    # 2) Reward weights + done timeout
    reward: ResidualRewardConfig = field(default_factory=ResidualRewardConfig)

    # 3) SAC hyperparameters
    agent: SACAgentConfig = field(default_factory=SACAgentConfig)

    # 4) Domain randomization
    use_domain_rand: bool = True
    domain_rand: DomainRandomizationConfig = field(default_factory=dr_defaults)

    # 5) Output
    save_dir: str = ""  # "" -> simulation_data/sac_training

    def __post_init__(self) -> None:
        self.reward.max_delta_omega = self.agent.max_delta_omega
        # Map randomization is off by default so training repeatedly sees `map_key`
        # (matches sac_training_config.m: `cfg.DomainRand.Map.Enable = false;`).
        self.domain_rand.map.enable = False
        if not self.use_domain_rand:
            self.domain_rand.enable = False
