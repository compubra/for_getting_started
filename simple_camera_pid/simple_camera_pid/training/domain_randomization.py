"""Per-episode domain randomization for residual RL training.

Translated from ``rl_dr_defaults.m`` and ``rl_domain_randomization.m``. The
MATLAB version returns a ``resetFcn(in) -> in`` closure that perturbs
``Simulink.SimulationInput`` workspace variables; there is no SimulationInput
here, so ``sample_episode()`` instead returns a plain
:class:`EpisodeRandomization` of concrete values that
``sim/turtlebot3_mujoco_env.py`` applies directly to its controller/robot/
vision config and scene choice at ``reset()``.

Percent fields ``p`` mean: sample uniformly from ``base * (1 +/- p)`` each
episode, exactly like the MATLAB ``pct()`` helper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..mujoco.runtime import mujoco_scene


@dataclass
class MapRandomizationConfig:
    enable: bool = True
    maps: list[str] = field(default_factory=lambda: [
        "simple", "complex", "ellipse", "training", "track_easy", "track_medium", "track_hard",
    ])


@dataclass
class GenTrackConfig:
    """Generate a fresh random track every episode (takes priority over Map)."""

    enable: bool = False
    shape: str = "random"        # "ellipse" | "capsule" | "random"
    difficulty: float = 0.4
    name: str = "rl_gen_episode"


@dataclass
class PoseRandomizationConfig:
    enable: bool = True
    lateral_range: float = 0.05


@dataclass
class DynamicsRandomizationConfig:
    enable: bool = True
    wheel_radius_pct: float = 0.05
    wheel_sep_pct: float = 0.05
    max_wheel_pct: float = 0.10


@dataclass
class ControlRandomizationConfig:
    enable: bool = True
    base_speed_pct: float = 0.10
    kp_pct: float = 0.15
    ki_abs: float = 0.0    # Ki baseline is usually 0; perturb with an absolute jitter instead
    kd_pct: float = 0.15


@dataclass
class PerceptionRandomizationConfig:
    enable: bool = True
    error_scale_pct: float = 0.10
    min_brightness_pct: float = 0.15
    roi_fraction_pct: float = 0.05  # perturbs LocalPath_ROIFraction (this port's live ROI knob)


@dataclass
class DomainRandomizationConfig:
    enable: bool = True
    map: MapRandomizationConfig = field(default_factory=MapRandomizationConfig)
    gen_track: GenTrackConfig = field(default_factory=GenTrackConfig)
    pose: PoseRandomizationConfig = field(default_factory=PoseRandomizationConfig)
    dynamics: DynamicsRandomizationConfig = field(default_factory=DynamicsRandomizationConfig)
    control: ControlRandomizationConfig = field(default_factory=ControlRandomizationConfig)
    perception: PerceptionRandomizationConfig = field(default_factory=PerceptionRandomizationConfig)


def dr_defaults() -> DomainRandomizationConfig:
    """Default domain-randomization config, shared by SAC and PPO training."""
    return DomainRandomizationConfig()


@dataclass
class EpisodeRandomization:
    scene_path: Path | None = None
    map_key: str | None = None
    lateral_target: float | None = None
    wheel_radius: float | None = None
    wheel_separation: float | None = None
    max_wheel_speed: float | None = None
    base_linear_speed: float | None = None
    kp: float | None = None
    ki: float | None = None
    kd: float | None = None
    error_scale: float | None = None
    min_brightness: float | None = None
    roi_bottom_fraction: float | None = None


def sample_episode(
    dr: DomainRandomizationConfig,
    base: dict,
    rng: np.random.Generator,
    repo_root: Path,
) -> EpisodeRandomization:
    """Sample one episode's randomized parameters.

    ``base`` supplies the un-randomized baseline values keyed exactly like
    the ``EpisodeRandomization`` fields (e.g. ``base["kp"]``), mirroring the
    MATLAB closure's one-time read of the model-workspace baseline.
    """
    out = EpisodeRandomization()
    if not dr.enable:
        return out

    scene_set = False
    if dr.gen_track.enable:
        from .track_generation import gen_simple_track_scene

        info = gen_simple_track_scene(
            repo_root, name=dr.gen_track.name, shape=dr.gen_track.shape,
            difficulty=dr.gen_track.difficulty, rng=rng, overwrite=True,
        )
        out.scene_path = info.scene_file
        out.map_key = info.map_key
        scene_set = True
    if not scene_set and dr.map.enable and dr.map.maps:
        key = dr.map.maps[int(rng.integers(len(dr.map.maps)))]
        scene_path, map_key, _ = mujoco_scene.resolve_turtlebot3_mujoco_scene(
            repo_root, requested_map=key
        )
        out.scene_path = scene_path
        out.map_key = map_key

    if dr.pose.enable:
        r = dr.pose.lateral_range
        out.lateral_target = base["lateral_target"] + rng.uniform(-r, r)

    if dr.dynamics.enable:
        out.wheel_radius = _pct(base["wheel_radius"], dr.dynamics.wheel_radius_pct, rng)
        out.wheel_separation = _pct(base["wheel_separation"], dr.dynamics.wheel_sep_pct, rng)
        out.max_wheel_speed = _pct(base["max_wheel_speed"], dr.dynamics.max_wheel_pct, rng)

    if dr.control.enable:
        out.base_linear_speed = _pct(base["base_linear_speed"], dr.control.base_speed_pct, rng)
        out.kp = _pct(base["kp"], dr.control.kp_pct, rng)
        out.ki = max(0.0, base["ki"] + rng.uniform(0.0, dr.control.ki_abs))
        out.kd = _pct(base["kd"], dr.control.kd_pct, rng)

    if dr.perception.enable:
        out.error_scale = _pct(base["error_scale"], dr.perception.error_scale_pct, rng)
        out.min_brightness = _pct(base["min_brightness"], dr.perception.min_brightness_pct, rng)
        out.roi_bottom_fraction = float(
            np.clip(_pct(base["roi_bottom_fraction"], dr.perception.roi_fraction_pct, rng), 0.0, 1.0)
        )

    return out


def _pct(base_value: float, p: float, rng: np.random.Generator) -> float:
    return base_value * (1.0 + rng.uniform(-p, p))
