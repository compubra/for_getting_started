#!/usr/bin/env python
"""Train the quadratic-reward SAC residual agent — Python translation of
train/sac/sac_training_config.m + train/sac/train_sac_residual_script.m
(model visual_line_follower_sac_residual.slx).

⚠️ 2026-07-21 refactor (see lsac/controller.py + lsac/env.py docstrings for
the full kinematics-side rationale): the action grew from 1-D (delta_omega)
to 2-D [delta_v, delta_omega], and the PID/residual mixing moved from two
separate wheel-space conversions (with a silently OPPOSITE omega sign between
them — an unnoticed bug fixed the same day) to a single shared
Diff_Drive_Kinematics conversion applied once, downstream, after PID's and
SAC's (v, omega) commands are summed.

Reward (Reward_Calculation subsystem, wiring verified via model_read on the
.slx: Math=square blocks + USq_Demux/Sum reduction over the 2-D action):
  reward(t) = -(Q*e^2 + R*||u||^2) - P_Lost*(1-found)
e is the vision steering_error only — no pose information enters the reward.
u = [delta_v, delta_omega]; ||u||^2 sums the squares of both components.

Observation (8-D, from the model's SAC_ObsMux + Wheel_Speed_Feedback):
  [steering_error, lateral_error, heading_error, found,
   prev_delta_v, prev_delta_omega, v_norm, omega_norm]
with v_norm=(wL+wR)/(2*MaxWheelSpeed), omega_norm=(wR-wL)/(2*MaxWheelSpeed)
from the measured wheel jointvel sensors.

Hyper-parameter mapping (sac_training_config.m → SB3 SAC):
  Agent.ActorLearnRate/CriticLearnRate 3e-4 → learning_rate=3e-4
  Agent.DiscountFactor 0.99                 → gamma=0.99
  Agent.MiniBatchSize 256                   → batch_size=256
  Agent.BufferLength 2e5                    → buffer_size=200_000
  Agent.TargetSmooth 5e-3                   → tau=0.005
  Agent.EntropyWeight 0.2 (auto-tuned)      → ent_coef="auto_0.2"
  Agent.MaxDeltaOmega 1.0, Agent.MaxDeltaV 1.0 → action bounds (2-D)
  default MATLAB nets                       → net_arch=[256, 256]
Training loop mapping:
  MaxEpisodes / StopReward=Inf              → stop after N episodes, no early stop
  MaxStepsPerEpisode                        → gym TimeLimit
  SaveEveryNEpisodes=50 (EpisodeFrequency)   → EpisodeTracker callback
  UseDomainRand=true, DomainRand.Map.Enable=false
                                            → DRConfig(map_enable=False)
  DomainRand.Control.BaseSpeedMin/Max        → DRConfig base_speed_min/max
                                            (absolute Burger 0~0.22 m/s range,
                                            NOT a percentage jitter — see
                                            lsac/domain_rand.py)
  UseFast / ShowPlot=false                  → offscreen EGL render, no monitor GUI

Run on a GPU node with MUJOCO_GL=egl (the camera must render offscreen):
  python train_sac_residual.py --max-episodes 500 --max-steps 200

⚠️ 2026-07-21 same-day training-run finding: with BaseSpeedMax at the full
Burger-rated 7.333 (0.22 m/s), a 500x200 run on --map-key simple never
converged (flat ~-260 average reward for all 500 episodes) — at high sampled
base speeds there's too little steering headroom left under MaxWheelSpeed
before saturating, so a large fraction of episodes are physically uncorrectable
regardless of policy quality. Consider passing a lower --base-speed-max (e.g.
~5.6, leaving ~60-70% of MaxWheelSpeed for the forward component) until this
is re-validated.

Train on procedurally generated random tracks (unlimited variety) instead of
a single fixed map — recommended after an earlier generalization check found
a fixed-map-trained agent got WORSE than the plain PID baseline on an unseen
track:
  python train_sac_residual.py --gen-track --gen-track-shape random \\
      --gen-track-difficulty 0.5
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
                 score_window: int = 20, start_episode: int = 0, verbose: int = 1):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.checkpoint_every = checkpoint_every
        self.save_dir = save_dir
        self.score_window = score_window
        self.start_episode = start_episode
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
            # n counts from start_episode so a --resume-from run continues the
            # episode/checkpoint numbering of the run it resumed instead of
            # restarting at 1 (e.g. resuming at 2400 -> next checkpoint is
            # ep02450, not ep00050).
            n = self.start_episode + len(self.episode_rewards)
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
        # base_speed_min/max (2026-07-21): ABSOLUTE range, not a percentage
        # jitter — see lsac/domain_rand.py and this file's module docstring
        # for the same-day finding that the full Burger range (0~7.333) left
        # too little steering headroom to converge on --map-key simple.
        # --gen-track instead trains on a fresh procedurally generated track
        # every episode (ellipse/capsule loop, unlimited variety) rather than
        # the fixed --map-key — this directly targets the single-map
        # overfitting a same-day generalization check found (a track_hard-
        # only-trained agent got ~2x WORSE than the plain PID baseline on an
        # unseen map). Takes priority over the fixed map when enabled.
        dr = DRConfig(map_enable=args.rand_map, gen_track_enable=args.gen_track,
                     gen_track_shape=args.gen_track_shape,
                     gen_track_difficulty=args.gen_track_difficulty,
                     base_speed_min=args.base_speed_min,
                     base_speed_max=args.base_speed_max)
    reward = QuadraticReward(QuadraticRewardParams(
        q=args.q, r=args.r_u, p_lost=args.p_lost))
    # Model-workspace values read out of visual_line_follower_sac_residual.slx
    # (they differ from the lyapunov model the lsac defaults came from):
    #   SteeringSign = -1 (paired with the upright camera image — the mirrored
    #   image + sign +1 of the lyapunov pipeline is its exact mirror twin),
    #   LocalPath_ROIFraction = 0.5 (lyapunov used 0.1).
    ctrl = ControllerParams(steering_sign=-1.0)
    vision_kwargs = {"roi_bottom_fraction": args.roi_fraction}
    if args.cam_pitch_deg is not None:
        vision_kwargs["cam_pitch_deg"] = args.cam_pitch_deg
    vision = VisionParams(**vision_kwargs)
    env = LineFollowerEnv(map_key=args.map_key,
                          max_delta_omega=args.max_delta_omega,
                          max_delta_v=args.max_delta_v,
                          done_steps=args.done_steps,
                          controller_params=ctrl,
                          vision_params=vision,
                          reward_fn=reward,
                          domain_rand=dr,
                          mirror_camera=False,
                          observe_wheel_speeds=True,
                          cam_pitch_deg=args.cam_pitch_deg,
                          seed=args.seed)
    env = TimeLimit(env, max_episode_steps=args.max_steps)
    return Monitor(env)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # ── training loop (sac_training_config.m §1 defaults; 2026-07-20
    #    real-car-speed run used 500x200 — see session notes) ──
    p.add_argument("--max-episodes", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=200,
                   help="MaxStepsPerEpisode (200*0.05s = 10s/episode)")
    p.add_argument("--score-window", type=int, default=20)
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="save a checkpoint every N episodes "
                        "(SaveEveryNEpisodes)")
    p.add_argument("--map-key", default="simple",
                   help="ignored when --gen-track is set")
    p.add_argument("--no-domain-rand", dest="domain_rand", action="store_false")
    p.add_argument("--base-speed-min", type=float, default=0.0,
                   help="DomainRand.Control.BaseSpeedMin (model units, "
                        "absolute range, not a %% jitter)")
    p.add_argument("--base-speed-max", type=float, default=0.22 / 0.03,
                   help="DomainRand.Control.BaseSpeedMax; default is the "
                        "Burger-rated 0.22 m/s / BaseSpeedScale -- see this "
                        "file's module docstring, this full range did NOT "
                        "converge in the 2026-07-21 --map-key simple run")
    p.add_argument("--gen-track", action="store_true",
                   help="train on a fresh procedurally generated track every "
                        "episode instead of the fixed --map-key (needs "
                        "--domain-rand, the default)")
    p.add_argument("--gen-track-shape", default="random",
                   choices=["ellipse", "capsule", "random"])
    p.add_argument("--gen-track-difficulty", type=float, default=0.4,
                   help="0..1, higher = smaller/tighter generated loop")
    p.add_argument("--roi-fraction", type=float, default=0.5,
                   help="LocalPath_ROIFraction: bottom fraction of the camera "
                        "image kept for the lane-line ROI mask")
    p.add_argument("--cam-pitch-deg", type=float, default=None,
                   help="override the front camera's downward mount pitch "
                        "(deg); default None keeps the scene XML's baked-in "
                        "15 deg untouched. Overrides both the rendered "
                        "MuJoCo camera orientation (env.py) and the "
                        "pixel-to-ground IPM math (vision.py) together, so "
                        "the two stay in sync -- e.g. --cam-pitch-deg 0 for "
                        "an un-tilted, forward-looking camera.")
    p.add_argument("--rand-map", action="store_true",
                   help="DomainRand.Map.Enable=true -- randomize --map-key "
                        "every episode from the 7-map pool (lsac/domain_rand.py "
                        "DRConfig.maps) instead of training on a single fixed "
                        "map. Ignored when --gen-track is set (gen-track takes "
                        "priority, same as MATLAB).")
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
    p.add_argument("--max-delta-omega", type=float, default=1.0,
                   help="SAC_MaxDeltaOmega -- action[1]")
    p.add_argument("--max-delta-v", type=float, default=1.0,
                   help="SAC_MaxDeltaV -- action[0], added to the PID's own "
                        "v command before Diff_Drive_Kinematics")
    # ── resuming an interrupted run ──
    p.add_argument("--resume-from", default="",
                   help="path to a saved SB3 checkpoint .zip (e.g. "
                        "checkpoint_ep02400.zip) to warm-start the policy "
                        "from. NOTE: model.save() checkpoints don't include "
                        "the replay buffer, so training resumes with an "
                        "empty buffer and repeats the learning_starts "
                        "warm-up -- only the policy/optimizer state carries "
                        "over.")
    p.add_argument("--start-episode", type=int, default=0,
                   help="episode count already completed by --resume-from; "
                        "offsets episode numbering, checkpoint filenames, "
                        "and the --max-episodes stop condition so the run "
                        "continues instead of restarting at episode 1")
    # ── output ──
    p.add_argument("--save-dir", default="")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.start_episode >= args.max_episodes:
        raise SystemExit(f"--start-episode ({args.start_episode}) must be < "
                          f"--max-episodes ({args.max_episodes})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.save_dir:
        save_dir = Path(args.save_dir)
    elif args.resume_from:
        # default to the checkpoint's own directory so a resumed run keeps
        # writing into the same folder it came from, rather than silently
        # splitting the run across a new timestamp dir (easy to forget
        # --save-dir when resuming).
        save_dir = Path(args.resume_from).resolve().parent
    else:
        save_dir = Path(__file__).parent / "simulation_data" / "sac_residual_training" / ts
    save_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args)
    if args.gen_track:
        print(f"Training map: procedurally generated ({args.gen_track_shape}, "
              f"difficulty={args.gen_track_difficulty}) | domain randomization "
              f"{'ON' if args.domain_rand else 'OFF'}")
    elif args.rand_map:
        print(f"Training map: randomized every episode from the domain-rand "
              f"map pool | domain randomization {'ON' if args.domain_rand else 'OFF'}")
    else:
        print(f"Training map: {args.map_key} | "
              f"domain randomization {'ON (map fixed)' if args.domain_rand else 'OFF'}")
    print(f"SAC entropy: ent_coef=auto_{args.ent_init}")
    print(f"Reward weights: Q={args.q} R={args.r_u} P_lost={args.p_lost} "
          f"| done after {args.done_steps} lost steps")

    if args.resume_from:
        model = SAC.load(args.resume_from, env=env, device=args.device)
        print(f"Resumed policy from {args.resume_from} "
              f"(starting at episode {args.start_episode}; replay buffer is "
              f"NOT restored -- checkpoints don't include it, so the buffer "
              f"starts empty and re-runs the learning_starts warm-up)")
    else:
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
                        args.score_window, start_episode=args.start_episode)
    remaining_episodes = args.max_episodes - args.start_episode
    print(f"Starting SAC residual training: {remaining_episodes} more episodes "
          f"({args.start_episode} -> {args.max_episodes}) x {args.max_steps} "
          f"steps (no early stop). Save dir: {save_dir}")
    model.learn(total_timesteps=remaining_episodes * args.max_steps, callback=cb)

    agent_file = save_dir / f"sac_residual_agent_{ts}"
    model.save(agent_file)
    print(f"Saved trained agent: {agent_file}.zip")

    with open(save_dir / "episode_rewards.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "steps"])
        for i, (r, l) in enumerate(zip(cb.episode_rewards, cb.episode_lengths),
                                   args.start_episode + 1):
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
