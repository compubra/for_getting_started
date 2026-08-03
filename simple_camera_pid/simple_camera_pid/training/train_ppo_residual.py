"""Train the PPO residual controller with stable-baselines3.

Translated from ``train_ppo_residual_live.m``, driven entirely by
``ppo_training_config.py`` (see ``train_sac_residual.py`` for why the MATLAB
project's Live-Script-vs-plain-script duplication collapses to one entry
point here).

Mapping notes vs. ``rlPPOAgentOptions``:
  - stable-baselines3's ``PPO`` uses one ``learning_rate`` for the combined
    actor-critic network (MATLAB's RL Toolbox keeps them as separate
    networks with separate learning rates); this uses ``cfg.agent.actor_lr``
    (both were 3e-4 in the MATLAB defaults anyway).
  - ``ExperienceHorizon`` -> ``n_steps``, ``ClipFactor`` -> ``clip_range``,
    ``EntropyLossWeight`` -> ``ent_coef``, ``GAEFactor`` -> ``gae_lambda``,
    ``NumEpoch`` -> ``n_epochs``.

Usage
-----
    python3 -m simple_camera_pid.training.train_ppo_residual
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from .residual_env import ResidualEnvConfig, ResidualLineFollowerEnv
from .ppo_training_config import PPOTrainingConfig


def train_ppo_residual(repo_root: Path, cfg: PPOTrainingConfig | None = None) -> Path:
    cfg = cfg or PPOTrainingConfig()

    out_dir = Path(cfg.save_dir) if cfg.save_dir else Path(__file__).resolve().parents[1] / \
        "simulation_data" / "ppo_training"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = ResidualEnvConfig(
        map_key=cfg.map_key, reward=cfg.reward, domain_randomization=cfg.domain_rand,
        max_steps_per_episode=cfg.max_steps_per_episode,
    )
    env = Monitor(ResidualLineFollowerEnv(repo_root, config=env_cfg), filename=str(out_dir / "train"))

    agent = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.agent.actor_lr,
        n_steps=cfg.agent.n_steps,
        batch_size=cfg.agent.batch_size,
        n_epochs=cfg.agent.n_epochs,
        gamma=cfg.agent.discount_factor,
        gae_lambda=cfg.agent.gae_lambda,
        clip_range=cfg.agent.clip_range,
        ent_coef=cfg.agent.entropy_coef,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.save_checkpoint_every, save_path=str(out_dir), name_prefix="ppo_agent",
    )
    agent.learn(total_timesteps=cfg.total_timesteps, callback=checkpoint_callback)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_file = out_dir / f"ppo_agent_{timestamp}.zip"
    agent.save(agent_file)
    print(f"Saved trained agent to: {agent_file}")
    return agent_file


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    train_ppo_residual(repo_root)


if __name__ == "__main__":
    main()
