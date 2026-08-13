"""Train the SAC residual controller with stable-baselines3.

Translated from ``train_sac_residual_live.m`` / ``train_sac_residual_script.m``
(the MATLAB project kept both a Live Script entry and a plain-script HPC-batch
fallback with duplicated, hand-kept-in-sync parameters; that split exists only
because MATLAB's Live Editor is unavailable in some environments, which has no
Python equivalent, so this is the one canonical entry point, driven entirely
by ``sac_training_config.py``).

Mapping notes vs. ``rlSACAgentOptions``:
  - stable-baselines3's ``SAC`` shares one ``learning_rate`` between actor and
    both critics (no separate actor/critic learning rates); this uses
    ``cfg.agent.actor_lr`` (both were 3e-4 in the MATLAB defaults anyway).
  - ``EntropyWeightOptions`` auto-tunes entropy from the configured initial
    value by default in RL Toolbox; the closest stable-baselines3 equivalent
    is ``ent_coef="auto_<value>"``.
  - MATLAB's ``UseFastRestart``/``UseParallel`` address Simulink-specific
    recompilation costs that do not apply to a plain Python/MuJoCo loop.

Usage
-----
    python3 -m simple_camera_pid.training.train_sac_residual
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from .residual_env import ResidualEnvConfig, ResidualLineFollowerEnv
from .sac_training_config import SACTrainingConfig


def train_sac_residual(repo_root: Path, cfg: SACTrainingConfig | None = None) -> Path:
    cfg = cfg or SACTrainingConfig()

    out_dir = Path(cfg.save_dir) if cfg.save_dir else Path(__file__).resolve().parents[1] / \
        "simulation_data" / "sac_training"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = ResidualEnvConfig(
        map_key=cfg.map_key, reward=cfg.reward, domain_randomization=cfg.domain_rand,
        max_steps_per_episode=cfg.max_steps_per_episode,
    )
    # controller/governor/vision go through ResidualLineFollowerEnv's
    # **base_env_kwargs to Turtlebot3LineFollowerEnv. Without them the base
    # env falls back to ControllerConfig()'s own defaults, which are the
    # pre-2026-08-04 Gazebo gains -- a base controller that drives the
    # simulated robot off the track, so the residual would be learning to
    # correct a controller nobody deploys. See sac_training_config.py.
    env = Monitor(
        ResidualLineFollowerEnv(
            repo_root, config=env_cfg,
            controller=cfg.controller, governor=cfg.governor, vision=cfg.vision,
        ),
        filename=str(out_dir / "train"),
    )

    agent = SAC(
        "MlpPolicy",
        env,
        learning_rate=cfg.agent.actor_lr,
        buffer_size=cfg.agent.buffer_size,
        batch_size=cfg.agent.batch_size,
        gamma=cfg.agent.discount_factor,
        tau=cfg.agent.target_smooth_tau,
        ent_coef=f"auto_{cfg.agent.entropy_weight}",
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.save_checkpoint_every, save_path=str(out_dir), name_prefix="sac_agent",
    )
    agent.learn(total_timesteps=cfg.total_timesteps, callback=checkpoint_callback)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_file = out_dir / f"sac_agent_{timestamp}.zip"
    agent.save(agent_file)
    print(f"Saved trained agent to: {agent_file}")
    return agent_file


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    train_sac_residual(repo_root)


if __name__ == "__main__":
    main()
