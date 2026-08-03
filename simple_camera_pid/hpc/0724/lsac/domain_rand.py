"""Per-episode domain randomization — translation of rl_dr_defaults.m +
rl_domain_randomization.m.

Percent fields p mean: sample uniformly from base*(1 ± p) each episode.
Exception: base_speed_min/base_speed_max (2026-07-21) are an ABSOLUTE range,
not a percentage around a base value — see the field docstring below.

Notes on fidelity:
  * In the Simulink pipeline Dynamics randomization (wheel radius/separation)
    only perturbs the CONTROLLER-side constants (gains built from
    TB3_WheelRadius / TB3_WheelSeparation), never the MuJoCo model. Same here.
  * The MATLAB Perception group also perturbed OriginBot_ROIStartFraction and
    OriginBot_ROIHeight, but the lyapunov model's detector reads
    LocalPath_ROIFraction instead, so those two writes were no-ops. They are
    therefore omitted here (set roi_fraction_pct > 0 to randomize the ROI that
    is actually used).
  * GenTrack (per-episode freshly generated tracks) is disabled by default in
    MATLAB too, but IS now ported (2026-07-20, track_gen.py) — see gen_track_*
    fields below. It takes priority over Map when enabled, same as MATLAB.
  * gen_track_generator (2026-07-23, mirrors MATLAB rl_dr_defaults.m's
    dr.GenTrack.Generator, added 2026-07-22): selects which of the three
    track_gen algorithms runs each episode. gen_track_shape is only consulted
    when the resolved generator is "simple" (it plays the role MATLAB's
    gen_simple_track_scene Shape option plays, independent of the higher-level
    Generator choice).
  * gen_track_enable defaults to True (2026-07-23; was False here since
    2026-07-20): rl_dr_defaults.m's dr.GenTrack.Enable is actually `true` in
    the current MATLAB source — its own inline comment claims "默认关"
    (default off), but that comment is stale; the live-script training run
    cached in train_sac_residual_live.m generated a fresh track every single
    episode, confirming Enable=true is what the 2026-07-21 500-episode run
    actually trained against, not a fixed --map-key. sac_training_config.m
    only overrides Map.Enable=false, never GenTrack.Enable, so this default
    carries through. Pass gen_track_enable=False to opt back into a fixed map.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .controller import ControllerParams
from .vision import VisionParams


@dataclass
class DRConfig:
    enable: bool = True

    # Map pool (per-episode random scene). Empty tuple = keep base map.
    # NOTE: "winding" is deliberately excluded from the default pool to mirror
    # the MATLAB side's rl_dr_defaults.m; a same-day generalization check found
    # a track_hard-trained agent got ~2x WORSE on winding than the plain PID
    # baseline (vs ~1.6x better on track_hard itself) — training with map
    # randomization off a single map does not generalize. Add "winding" here
    # if training with map_enable=True to actually address that gap.
    map_enable: bool = True
    maps: tuple[str, ...] = ("simple", "complex", "ellipse", "training",
                             "track_easy", "track_medium", "track_hard")

    # Per-episode freshly generated track (track_gen.py). Takes priority over
    # Map when enabled — mirrors MATLAB's GenTrack group. Off by default: it's
    # slower (forces a full model/renderer reload every episode, see
    # env.py's force_reload path) but gives effectively unlimited track
    # variety instead of a fixed 7-map pool, which is one real fix for the
    # single-map overfitting a generalization check found.
    gen_track_enable: bool = True
    gen_track_generator: str = "random"    # simple | complex | wandering | random
    gen_track_shape: str = "random"        # ellipse | capsule | random (simple only)
    gen_track_difficulty: float = 0.4      # 0..1, shared semantics across all 3
    gen_track_name: str = "rl_gen_episode"  # reused every episode (overwritten)

    # Start pose: lateral spawn offset (m), uniform in ±lateral_range.
    pose_enable: bool = True
    lateral_range: float = 0.05

    # Robot dynamics (controller-side constants).
    dynamics_enable: bool = True
    wheel_radius_pct: float = 0.05
    wheel_sep_pct: float = 0.05
    max_wheel_pct: float = 0.10

    # Base speed / PID.
    control_enable: bool = True
    # 2026-07-21: base_linear_speed no longer jitters ±pct around the current
    # base value — it's sampled uniformly over an ABSOLUTE range spanning the
    # real TurtleBot3 Burger linear-speed envelope (0 m/s standstill to the
    # ROBOTIS-rated 0.22 m/s max). base_linear_speed = v_real / base_speed_scale
    # (wheel radius cancels out of that ratio, so this range stays correct
    # regardless of the wheel_radius_pct dynamics jitter below).
    base_speed_min: float = 0.0
    base_speed_max: float = 0.22 / 0.03  # ~= 7.3333
    kp_pct: float = 0.15
    ki_abs: float = 0.0
    kd_pct: float = 0.15

    # Perception / vision.
    perception_enable: bool = True
    error_scale_pct: float = 0.10
    min_brightness_pct: float = 0.15
    roi_fraction_pct: float = 0.0  # MATLAB equivalent was a no-op; see module doc


@dataclass
class EpisodeSample:
    """One episode's randomized values."""

    map_key: str | None
    lateral_offset: float
    controller: ControllerParams
    vision: VisionParams
    is_generated: bool = False  # True when map_key is a gen_track() scene
                                # path, not a fixed registry key — the env
                                # must force-reload rather than trust its
                                # path-keyed model/renderer cache (the file
                                # gets overwritten with new geometry each
                                # episode; same path, different content).


def _pct(rng: np.random.Generator, base: float, p: float) -> float:
    return base * (1.0 + rng.uniform(-p, p))


def _gen_track(rng: np.random.Generator, cfg: DRConfig) -> str:
    """Dispatch to one of the three track_gen generators and return the
    resulting scene path — mirrors MATLAB's rl_domain_randomization.m switch
    on dr.GenTrack.Generator (with "random" picking uniformly among the three
    each episode, same pool as the MATLAB side)."""
    from . import track_gen

    generator = cfg.gen_track_generator
    if generator == "random":
        generator = ("simple", "complex", "wandering")[rng.integers(3)]

    if generator == "simple":
        info = track_gen.gen_simple_track_scene(
            name=cfg.gen_track_name, shape=cfg.gen_track_shape,
            difficulty=cfg.gen_track_difficulty, overwrite=True, rng=rng)
    elif generator == "complex":
        info = track_gen.gen_complex_track_scene(
            name=cfg.gen_track_name, difficulty=cfg.gen_track_difficulty,
            overwrite=True, rng=rng)
    elif generator == "wandering":
        info = track_gen.gen_wandering_track_scene(
            name=cfg.gen_track_name, difficulty=cfg.gen_track_difficulty,
            overwrite=True, rng=rng)
    else:
        raise ValueError(f"gen_track_generator must be simple|complex|"
                         f"wandering|random, got {cfg.gen_track_generator!r}")
    return info.scene_file  # resolve_scene() also accepts a raw path


def sample_episode(rng: np.random.Generator, cfg: DRConfig,
                   base_ctrl: ControllerParams,
                   base_vision: VisionParams) -> EpisodeSample:
    ctrl = ControllerParams(**vars(base_ctrl))
    vision = VisionParams(**vars(base_vision))
    map_key: str | None = None
    is_generated = False
    lateral = 0.0

    if not cfg.enable:
        return EpisodeSample(None, 0.0, ctrl, vision)

    if cfg.gen_track_enable:
        map_key = _gen_track(rng, cfg)
        is_generated = True
    elif cfg.map_enable and cfg.maps:
        map_key = str(cfg.maps[rng.integers(len(cfg.maps))])

    if cfg.pose_enable:
        lateral = float(rng.uniform(-cfg.lateral_range, cfg.lateral_range))

    if cfg.dynamics_enable:
        ctrl.wheel_radius = _pct(rng, base_ctrl.wheel_radius, cfg.wheel_radius_pct)
        ctrl.wheel_separation = _pct(rng, base_ctrl.wheel_separation, cfg.wheel_sep_pct)
        ctrl.max_wheel_speed = _pct(rng, base_ctrl.max_wheel_speed, cfg.max_wheel_pct)

    if cfg.control_enable:
        ctrl.base_linear_speed = float(
            rng.uniform(cfg.base_speed_min, cfg.base_speed_max))
        ctrl.kp = _pct(rng, base_ctrl.kp, cfg.kp_pct)
        ctrl.ki = max(0.0, base_ctrl.ki + rng.uniform(0.0, cfg.ki_abs))
        ctrl.kd = _pct(rng, base_ctrl.kd, cfg.kd_pct)

    if cfg.perception_enable:
        vision.error_scale = _pct(rng, base_vision.error_scale, cfg.error_scale_pct)
        vision.min_brightness = _pct(rng, base_vision.min_brightness, cfg.min_brightness_pct)
        if cfg.roi_fraction_pct > 0:
            vision.roi_bottom_fraction = float(np.clip(
                _pct(rng, base_vision.roi_bottom_fraction, cfg.roi_fraction_pct),
                0.05, 0.95))

    return EpisodeSample(map_key, lateral, ctrl, vision, is_generated)
