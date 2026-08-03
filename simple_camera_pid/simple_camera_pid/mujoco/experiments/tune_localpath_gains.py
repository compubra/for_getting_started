"""Sweep LocalPath steering gains on a MuJoCo track.

Translated from ``tune_localpath_gains.m``: one headless trial per
(lateral_gain, heading_gain, curvature_gain) triple, ranked by
``TrialSummary.score()`` (lower is better). PID gains are held at the
model-workspace baseline unless overridden.

Usage
-----
    python3 -m simple_camera_pid.mujoco.experiments.tune_localpath_gains
"""
from __future__ import annotations

import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ...common.config import ControllerConfig, VisionConfig
from ..sim.turtlebot3_mujoco_env import Turtlebot3LineFollowerEnv
from .trial_runner import run_trial


def tune_localpath_gains(
    repo_root: Path,
    gain_sets: list[tuple[float, float, float]],
    map_key: str = "complex",
    stop_time: float = 40.0,
    kp_line: float | None = None,
    ki_line: float | None = None,
    kd_line: float | None = None,
    n_line: float | None = None,
) -> list[dict]:
    controller_overrides = {}
    if kp_line is not None:
        controller_overrides["kp"] = kp_line
    if ki_line is not None:
        controller_overrides["ki"] = ki_line
    if kd_line is not None:
        controller_overrides["kd"] = kd_line
    if n_line is not None:
        controller_overrides["n_filter"] = n_line

    results = []
    for trial, (lateral_gain, heading_gain, curvature_gain) in enumerate(gain_sets, start=1):
        controller = ControllerConfig(**controller_overrides)
        vision = VisionConfig(lateral_gain=lateral_gain, heading_gain=heading_gain,
                               curvature_gain=curvature_gain)
        env = Turtlebot3LineFollowerEnv(repo_root, map_key=map_key, controller=controller, vision=vision)
        summary, _ = run_trial(env, stop_time)
        results.append({
            "trial": trial, "lateral_gain": lateral_gain, "heading_gain": heading_gain,
            "curvature_gain": curvature_gain, **asdict(summary), "score": summary.score(),
        })
        print(f"[{trial}/{len(gain_sets)}] lateral={lateral_gain:.3f} heading={heading_gain:.3f} "
              f"curvature={curvature_gain:.3f} -> score={summary.score():.4f}")

    results.sort(key=lambda r: r["score"])

    data_dir = Path(__file__).resolve().parents[1] / "simulation_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    result_file = data_dir / f"localpath_gain_sweep_{map_key}_{timestamp}.pkl"
    with result_file.open("wb") as f:
        pickle.dump(results, f)

    best = results[0]
    print(f"\nSaved LocalPath sweep: {result_file}")
    print(f"Best: Lateral={best['lateral_gain']:.3f} Heading={best['heading_gain']:.3f} "
          f"Curvature={best['curvature_gain']:.3f}  Score={best['score']:.4f}")
    return results


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    default_gain_sets = [(0.6, 0.35, 0.08), (0.8, 0.35, 0.08), (1.0, 0.35, 0.08),
                         (1.2, 0.35, 0.08), (1.5, 0.35, 0.08)]
    tune_localpath_gains(repo_root, default_gain_sets)


if __name__ == "__main__":
    main()
