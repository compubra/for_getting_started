"""SAC/PPO residual controller wrapper around the PID line-follower env.

Translated from ``build_sac_residual_controller.m``: an RL agent outputs a
scalar ``delta_omega`` residual (rad/s) that is summed with the PID
controller's differential wheel output (see
``control/line_follower_controller.py``'s ``residual_delta_omega``
parameter, added at the exact same point in the signal chain as the
Simulink model's ``SAC_LeftDiffGain``/``SAC_RightDiffGain``/
``SAC_LeftSum``/``SAC_RightSum`` blocks).

Reward and done logic mirror the ``Reward_Calculation``/``Done_Detection``
subsystems referenced by ``sac_training_config.m`` / ``ppo_training_config.m``
(``SAC_Q``/``SAC_R``/``SAC_P_Lost``/``SAC_Done_Steps`` and their ``PPO_*``
counterparts — both algorithms share the same weights structure, hence one
``ResidualRewardConfig`` here):

    reward(t) = -(Q*e^2 + Q_lat*e_lat^2 + Q_head*e_head^2 + R*u^2) - P_Lost*(1 - found)
    done      = line lost for more than done_steps consecutive control ticks

Domain randomization (translated from ``runtime/domain_randomization.py``,
itself from ``rl_domain_randomization.m``) is applied once per episode in
``reset()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from ..common.config import ResidualRewardConfig
from ..mujoco.sim.turtlebot3_mujoco_env import Turtlebot3LineFollowerEnv
from . import io_specs
from .domain_randomization import DomainRandomizationConfig, dr_defaults, sample_episode


@dataclass
class ResidualEnvConfig:
    map_key: str = "simple"
    reward: ResidualRewardConfig = field(default_factory=ResidualRewardConfig)
    domain_randomization: DomainRandomizationConfig = field(default_factory=dr_defaults)
    max_steps_per_episode: int = 400  # 400 * 0.05s = 20s/episode (matches sac_training_config.m)


class ResidualLineFollowerEnv(gym.Env):
    """Gymnasium env for SAC/PPO residual training, wrapping :class:`Turtlebot3LineFollowerEnv`.

    Observation (5-D): [steering_error, lateral_error, heading_error, found, prev_rl_action]
    Action (1-D): delta_omega in [-max_delta_omega, +max_delta_omega] rad/s
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        repo_root: str | Path,
        config: ResidualEnvConfig | None = None,
        render_mode: str | None = None,
        **base_env_kwargs: Any,
    ) -> None:
        super().__init__()
        self.cfg = config or ResidualEnvConfig()
        self.render_mode = render_mode
        self.base_env = Turtlebot3LineFollowerEnv(
            repo_root, map_key=self.cfg.map_key, render_mode=render_mode, **base_env_kwargs
        )
        self.repo_root = Path(repo_root)

        self.observation_space = io_specs.observation_space()
        self.action_space = io_specs.action_space(self.cfg.reward.max_delta_omega)

        self._prev_action = 0.0
        self._lost_steps = 0
        self._step_count = 0

        # One-time baseline snapshot for domain-randomization sampling (matches the
        # MATLAB closure's one-time read of the model-workspace baseline).
        self._dr_base = dict(
            lateral_target=0.0,
            wheel_radius=self.base_env.robot.wheel_radius,
            wheel_separation=self.base_env.robot.wheel_separation,
            max_wheel_speed=self.base_env.robot.max_wheel_speed,
            base_linear_speed=self.base_env.controller_cfg.base_linear_speed,
            kp=self.base_env.controller_cfg.kp,
            ki=self.base_env.controller_cfg.ki,
            kd=self.base_env.controller_cfg.kd,
            error_scale=self.base_env.vision.error_scale,
            min_brightness=self.base_env.vision.min_brightness,
            roi_bottom_fraction=self.base_env.vision.roi_bottom_fraction,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        rng = self.np_random  # seeded by super().reset()

        episode = sample_episode(self.cfg.domain_randomization, self._dr_base, rng, self.repo_root)
        self._apply_randomization(episode)

        base_options: dict[str, Any] = {}
        if episode.scene_path is not None:
            base_options["scene_path"] = episode.scene_path
            base_options["map_key"] = episode.map_key
        if episode.lateral_target is not None:
            base_options["lateral_target"] = episode.lateral_target

        _, base_info = self.base_env.reset(seed=seed, options=base_options or None)

        self._prev_action = 0.0
        self._lost_steps = 0
        self._step_count = 0
        observation = self._observation(self.base_env.last_vision, self._prev_action)
        return observation, base_info

    def step(self, action: np.ndarray):
        delta_omega = float(np.clip(action, -self.cfg.reward.max_delta_omega,
                                     self.cfg.reward.max_delta_omega).reshape(-1)[0])

        self.base_env.step(delta_omega)
        vision = self.base_env.last_vision
        self._step_count += 1

        reward = self._reward(
            vision.steering_error, vision.lateral_error, vision.heading_error,
            delta_omega, vision.found,
        )

        if vision.found:
            self._lost_steps = 0
        else:
            self._lost_steps += 1
        terminated = self._lost_steps > self.cfg.reward.done_steps
        truncated = self._step_count >= self.cfg.max_steps_per_episode

        observation = self._observation(vision, delta_omega)
        self._prev_action = delta_omega
        # Deliberately excludes the base env's per-step StepLog (raw odometry/wheel
        # arrays meant for experiments/trial_runner.py) — an RL-facing info dict
        # should stay small and, ideally, comparable (gymnasium's check_env asserts
        # info-dict equivalence across identically-seeded replays).
        info = {"map_key": self.base_env.map_key, "lost_steps": self._lost_steps}
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _reward(
        self, steering_error: float, lateral_error: float, heading_error: float,
        action: float, found: bool,
    ) -> float:
        cfg = self.cfg.reward
        penalty = (
            cfg.q * steering_error ** 2
            + cfg.q_lateral * lateral_error ** 2
            + cfg.q_heading * heading_error ** 2
            + cfg.r * action ** 2
        )
        lost_penalty = cfg.p_lost * (0.0 if found else 1.0)
        return -penalty - lost_penalty

    def _observation(self, vision, prev_action: float) -> np.ndarray:
        return np.array([
            vision.steering_error, vision.lateral_error, vision.heading_error,
            1.0 if vision.found else 0.0, prev_action,
        ], dtype=np.float32)

    def _apply_randomization(self, episode) -> None:
        if episode.wheel_radius is not None:
            self.base_env.robot.wheel_radius = episode.wheel_radius
        if episode.wheel_separation is not None:
            self.base_env.robot.wheel_separation = episode.wheel_separation
        if episode.max_wheel_speed is not None:
            self.base_env.robot.max_wheel_speed = episode.max_wheel_speed
        if episode.base_linear_speed is not None:
            self.base_env.controller_cfg.base_linear_speed = episode.base_linear_speed
        if episode.kp is not None:
            self.base_env.controller_cfg.kp = episode.kp
        if episode.ki is not None:
            self.base_env.controller_cfg.ki = episode.ki
        if episode.kd is not None:
            self.base_env.controller_cfg.kd = episode.kd
        if episode.error_scale is not None:
            self.base_env.vision.error_scale = episode.error_scale
        if episode.min_brightness is not None:
            self.base_env.vision.min_brightness = episode.min_brightness
        if episode.roi_bottom_fraction is not None:
            self.base_env.vision.roi_bottom_fraction = episode.roi_bottom_fraction

    def render(self):
        return self.base_env.render()

    def close(self) -> None:
        self.base_env.close()
