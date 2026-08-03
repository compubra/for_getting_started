"""Central tuning interface for PPO residual RL training.

Translated from ``ppo_training_config.m``. Structure mirrors
``sac_training_config.py`` except for the agent hyperparameters, which are
PPO's on-policy horizon/clip/GAE family instead of SAC's replay-buffer
family.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..common.config import ResidualRewardConfig
from .domain_randomization import DomainRandomizationConfig, dr_defaults


@dataclass
class PPOAgentConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4          # stable-baselines3 PPO shares one optimizer/learning
                                      # rate for the combined actor-critic network; see
                                      # train_ppo_residual.py for how this is reconciled.
    discount_factor: float = 0.99
    n_steps: int = 512               # ExperienceHorizon: on-policy steps collected per update
    batch_size: int = 128
    n_epochs: int = 3
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    max_delta_omega: float = 1.0


@dataclass
class PPOTrainingConfig:
    # 0) Model / map
    map_key: str = "track_hard"

    # 1) Training loop
    total_timesteps: int = 800 * 200   # MaxEpisodes * MaxStepsPerEpisode (800 episodes * 10s @ 20Hz)
    max_steps_per_episode: int = 200   # 200*0.05s = 10s/episode
    save_checkpoint_every: int = 20 * 200

    # 2) Reward weights + done timeout
    reward: ResidualRewardConfig = field(default_factory=ResidualRewardConfig)

    # 3) PPO hyperparameters
    agent: PPOAgentConfig = field(default_factory=PPOAgentConfig)

    # 4) Domain randomization
    use_domain_rand: bool = True
    domain_rand: DomainRandomizationConfig = field(default_factory=dr_defaults)

    # 5) Output
    save_dir: str = ""  # "" -> simulation_data/ppo_training

    def __post_init__(self) -> None:
        self.reward.max_delta_omega = self.agent.max_delta_omega
        # GenTrack scenes generated during PPO training use a distinct basename so a
        # concurrent SAC run never overwrites the same file (matches ppo_training_config.m).
        self.domain_rand.gen_track.name = "ppo_gen_episode"
        self.domain_rand.map.enable = False
        if not self.use_domain_rand:
            self.domain_rand.enable = False
