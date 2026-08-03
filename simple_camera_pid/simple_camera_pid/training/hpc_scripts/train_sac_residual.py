#!/usr/bin/env python
"""Train the quadratic-reward SAC residual agent — Python translation of
SAC/sac_training_config.m + SAC/train_sac_residual_script.m
(model visual_line_follower_sac_residual.slx).

Reward (Reward_Calculation subsystem, wiring verified via model_read on the
.slx: Math=square blocks, u = the raw RL action delta_omega):
  reward(t) = -(Q*e^2 + R*u^2) - P_Lost*(1-found)
e is the vision steering_error only — no pose information enters the reward.

Observation (7-D, from the model's SAC_ObsMux + Wheel_Speed_Feedback):
  [steering_error, lateral_error, heading_error, found, prev_action,
   v_norm, omega_norm]
with v_norm=(wL+wR)/(2*MaxWheelSpeed), omega_norm=(wR-wL)/(2*MaxWheelSpeed)
from the measured wheel jointvel sensors.

Hyper-parameter mapping (sac_training_config.m → SB3 SAC):
  Agent.ActorLearnRate/CriticLearnRate 3e-4 → learning_rate=3e-4
  Agent.DiscountFactor 0.99                 → gamma=0.99
  Agent.MiniBatchSize 256                   → batch_size=256
  Agent.BufferLength 2e5                    → buffer_size=200_000
  Agent.TargetSmooth 5e-3                   → tau=0.005
  Agent.EntropyWeight 0.2 (auto-tuned)      → ent_coef="auto_0.2"
  Agent.MaxDeltaOmega 1.0                   → action bound
  default MATLAB nets                       → net_arch=[256, 256]
Training loop mapping:
  MaxEpisodes / StopReward=Inf              → stop after N episodes, no early stop
  MaxStepsPerEpisode                        → gym TimeLimit
  periodic checkpoint every 100 episodes    → EpisodeTracker callback
  UseDomainRand=true, DomainRand.Map.Enable=false
                                            → DRConfig(map_enable=False)
  UseFast / ShowPlot=false                  → offscreen EGL render, no monitor GUI

Run on a GPU node with MUJOCO_GL=egl (the camera must render offscreen):
  python train_sac_residual.py --max-episodes 2000 --max-steps 500
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from lsac.controller import ControllerParams
from lsac.domain_rand import DRConfig
from lsac.env import LineFollowerEnv
from lsac.reward import QuadraticReward, QuadraticRewardParams
from lsac.vision import VisionParams


class EpisodeTracker(BaseCallback):
    """Episode bookkeeping: reward log, periodic checkpoint every
    checkpoint_every episodes, stop after max_episodes."""

    def __init__(self, max_episodes: int, checkpoint_every: int, save_dir: Path,
                 score_window: int = 20, verbose: int = 1):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.checkpoint_every = checkpoint_every
        self.save_dir = save_dir
        self.score_window = score_window
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self._t0 = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue
            self.episode_rewards.append(float(ep["r"]))
            self.episode_lengths.append(int(ep["l"]))
            n = len(self.episode_rewards)
            avg = float(np.mean(self.episode_rewards[-self.score_window:]))
            if self.verbose:
                print(f"Episode {n:5d}/{self.max_episodes} | steps {ep['l']:4d} "
                      f"| reward {ep['r']:9.3f} | avg{self.score_window} {avg:9.3f} "
                      f"| {time.time() - self._t0:7.0f}s", flush=True)
            if n % self.checkpoint_every == 0:
                self.model.save(self.save_dir / f"checkpoint_ep{n:05d}")
            if n >= self.max_episodes:
                return False
        return True


def build_env(args) -> Monitor:
    dr = None
    if args.domain_rand:
        # sac_training_config.m §5: rl_dr_defaults() with Map.Enable=false —
        # pose/dynamics/PID/perception randomized, map fixed to --map-key.
        dr = DRConfig(map_enable=False)
    reward = QuadraticReward(QuadraticRewardParams(
        q=args.q, r=args.r_u, p_lost=args.p_lost))
    # Model-workspace values read out of visual_line_follower_sac_residual.slx
    # (they differ from the lyapunov model the lsac defaults came from):
    #   SteeringSign = -1 (paired with the upright camera image — the mirrored
    #   image + sign +1 of the lyapunov pipeline is its exact mirror twin),
    #   LocalPath_ROIFraction = 0.5 (lyapunov used 0.1).
    ctrl = ControllerParams(steering_sign=-1.0)
    vision = VisionParams(roi_bottom_fraction=0.5)
    env = LineFollowerEnv(map_key=args.map_key,
                          max_delta_omega=args.max_delta_omega,
                          done_steps=args.done_steps,
                          controller_params=ctrl,
                          vision_params=vision,
                          reward_fn=reward,
                          domain_rand=dr,
                          mirror_camera=False,
                          observe_wheel_speeds=True,
                          seed=args.seed)
    env = TimeLimit(env, max_episode_steps=args.max_steps)
    return Monitor(env)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # ── training loop (sac_training_config.m §1 defaults) ──
    p.add_argument("--max-episodes", type=int, default=800)
    p.add_argument("--max-steps", type=int, default=400,
                   help="MaxStepsPerEpisode (400*0.05s = 20s/episode)")
    p.add_argument("--score-window", type=int, default=20)
    p.add_argument("--checkpoint-every", type=int, default=100,
                   help="save a checkpoint every N episodes")
    p.add_argument("--map-key", default="track_hard")
    p.add_argument("--no-domain-rand", dest="domain_rand", action="store_false")
    p.add_argument("--seed", type=int, default=None)
    # ── reward weights (sac_training_config.m §3) ──
    p.add_argument("--q", type=float, default=1.0, help="SAC_Q")
    p.add_argument("--r-u", type=float, default=0.1, help="SAC_R")
    p.add_argument("--p-lost", type=float, default=10.0, help="SAC_P_Lost")
    p.add_argument("--done-steps", type=int, default=100, help="SAC_Done_Steps")
    # ── SAC hyper-parameters (sac_training_config.m §4) ──
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer", type=int, default=200_000)
    p.add_argument("--ent-init", type=float, default=0.2)
    p.add_argument("--tau", type=float, default=5e-3)
    p.add_argument("--max-delta-omega", type=float, default=1.0)
    # ── output ──
    p.add_argument("--save-dir", default="")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) if args.save_dir else (
        Path(__file__).parent / "simulation_data" / "sac_residual_training" / ts)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args)
    print(f"Training map: {args.map_key} | "
          f"domain randomization {'ON (map fixed)' if args.domain_rand else 'OFF'}")
    print(f"Reward weights: Q={args.q} R={args.r_u} P_lost={args.p_lost} "
          f"| done after {args.done_steps} lost steps")

    model = SAC(
        "MlpPolicy", env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer,
        tau=args.tau,
        ent_coef=f"auto_{args.ent_init}",
        learning_starts=args.batch_size,   # ~ MATLAB warm-start = MiniBatchSize
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [256, 256]},
        seed=args.seed,
        device=args.device,
        verbose=0,
    )

    cb = EpisodeTracker(args.max_episodes, args.checkpoint_every, save_dir,
                        args.score_window)
    print(f"Starting SAC residual training: {args.max_episodes} episodes x "
          f"{args.max_steps} steps (no early stop). Save dir: {save_dir}")
    model.learn(total_timesteps=args.max_episodes * args.max_steps, callback=cb)

    agent_file = save_dir / f"sac_residual_agent_{ts}"
    model.save(agent_file)
    print(f"Saved trained agent: {agent_file}.zip")

    with open(save_dir / "episode_rewards.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "steps"])
        for i, (r, l) in enumerate(zip(cb.episode_rewards, cb.episode_lengths), 1):
            w.writerow([i, r, l])

    try:  # learning curve (plot_rl_training equivalent)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rewards = np.asarray(cb.episode_rewards)
        window = args.score_window
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(rewards, alpha=0.35, label="episode reward")
        if len(rewards) >= window:
            avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window - 1, len(rewards)), avg,
                    label=f"moving avg ({window})")
        ax.set_xlabel("episode"), ax.set_ylabel("reward")
        ax.set_title("SAC residual training"), ax.legend(), ax.grid(alpha=0.3)
        fig.savefig(save_dir / "training_curve.png", dpi=150, bbox_inches="tight")
        print(f"Saved learning curve: {save_dir / 'training_curve.png'}")
    except Exception as e:  # matplotlib optional
        print(f"(skipped learning-curve plot: {e})")

    env.close()


if __name__ == "__main__":
    main()
