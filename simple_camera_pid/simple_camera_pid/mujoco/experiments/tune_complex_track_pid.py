"""Compare PID gains on the complex MuJoCo track.

Translated from ``tune_complex_track_pid.m``.

Usage
-----
    python3 -m simple_camera_pid.mujoco.experiments.tune_complex_track_pid
"""
from __future__ import annotations

import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ...common.config import ControllerConfig
from ..sim.turtlebot3_mujoco_env import Turtlebot3LineFollowerEnv
from .trial_runner import run_trial

# [Kp, Ki, Kd, N]. The first row is the current baseline.
GAIN_SETS = [
    (4.0, 0.0, 0.20, 20.0),
    (4.5, 0.0, 0.30, 20.0),
    (5.0, 0.0, 0.35, 20.0),
    (5.5, 0.0, 0.40, 20.0),
    (5.0, 0.0, 0.45, 15.0),
]
STOP_TIME = 22.0


def tune_complex_track_pid(repo_root: Path) -> list[dict]:
    results = []
    for kp, ki, kd, n in GAIN_SETS:
        controller = ControllerConfig(kp=kp, ki=ki, kd=kd, n_filter=n)
        env = Turtlebot3LineFollowerEnv(repo_root, map_key="complex", controller=controller)
        summary, _ = run_trial(env, STOP_TIME)
        results.append({"kp": kp, "ki": ki, "kd": kd, "n": n, **asdict(summary), "score": summary.score()})
        print(f"Kp={kp:.2f} Ki={ki:.2f} Kd={kd:.2f} N={n:.1f} -> score={summary.score():.4f}")

    results.sort(key=lambda r: r["score"])

    data_dir = Path(__file__).resolve().parents[1] / "simulation_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    result_file = data_dir / f"complex_track_pid_sweep_{timestamp}.pkl"
    with result_file.open("wb") as f:
        pickle.dump(results, f)

    best = results[0]
    print(f"\nSaved PID sweep: {result_file}")
    print(f"Best gains: Kp={best['kp']:.3f}, Ki={best['ki']:.3f}, Kd={best['kd']:.3f}, N={best['n']:.3f}")
    return results


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    tune_complex_track_pid(repo_root)


if __name__ == "__main__":
    main()
