"""Run and archive one PID line-follower trial.

Translated from ``run_turtlebot3_burger_mujoco_visual_line_follower.m``.
Every trial is saved to ``simulation_data/`` as a timestamped ``.npz`` +
pickled step-log, containing the parameters and summary metrics (the MATLAB
version saves a ``.mat`` with the full ``Simulink.SimulationOutput``; this
saves the equivalent :class:`~experiments.trial_runner.TrialSummary` plus the
per-step logs instead of a MATLAB-specific format).

Usage
-----
    python3 -m simple_camera_pid.mujoco.experiments.run_turtlebot3_burger_mujoco_visual_line_follower \\
        --stop-time 60 --map-key complex --trial-label demo
"""
from __future__ import annotations

import argparse
import pickle
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ...common.config import ControllerConfig, RobotConfig, VisionConfig
from ..sim.turtlebot3_mujoco_env import Turtlebot3LineFollowerEnv
from .trial_runner import TrialSummary, run_trial


@dataclass
class TrialParameters:
    base_linear_speed: float
    base_speed_scale: float
    kp_line: float
    ki_line: float
    kd_line: float
    n_line: float
    roi_fraction: float
    local_path_points: int
    local_path_lookahead_distance: float
    local_path_lateral_gain: float
    local_path_heading_gain: float
    local_path_curvature_gain: float
    min_brightness: float
    max_saturation: float
    min_pixels: float
    error_scale: float
    max_wheel_speed: float
    stop_time: float
    map_key: str
    trial_label: str


def run_turtlebot3_burger_mujoco_visual_line_follower(
    repo_root: Path,
    stop_time: float = 200.0,
    map_key: str = "",
    trial_label: str = "run",
    base_linear_speed: float | None = None,
    kp_line: float | None = None,
    ki_line: float | None = None,
    kd_line: float | None = None,
    n_line: float | None = None,
    roi_fraction: float = 0.35,
    local_path_points: int = 30,
    local_path_lookahead_distance: float = 0.72,
    local_path_lateral_gain: float = 0.6,
    local_path_heading_gain: float = 0.35,
    local_path_curvature_gain: float = 0.04,
    min_brightness: float = 70.0,
    max_saturation: float = 0.30,
    min_pixels: float = 30.0,
    error_scale: float = 500.0,
    max_wheel_speed: float = 20.0,
    base_speed_scale: float = 0.03,
) -> tuple[TrialSummary, Path]:
    robot = RobotConfig(max_wheel_speed=max_wheel_speed)
    controller = ControllerConfig(
        kp=kp_line if kp_line is not None else ControllerConfig.kp,
        ki=ki_line if ki_line is not None else ControllerConfig.ki,
        kd=kd_line if kd_line is not None else ControllerConfig.kd,
        n_filter=n_line if n_line is not None else ControllerConfig.n_filter,
        base_linear_speed=base_linear_speed if base_linear_speed is not None
        else ControllerConfig.base_linear_speed,
        base_speed_scale=base_speed_scale,
    )
    vision = VisionConfig(
        roi_bottom_fraction=roi_fraction, num_points=local_path_points,
        lookahead_distance=local_path_lookahead_distance, lateral_gain=local_path_lateral_gain,
        heading_gain=local_path_heading_gain, curvature_gain=local_path_curvature_gain,
        min_brightness=min_brightness, max_saturation=max_saturation, min_pixels=min_pixels,
        error_scale=error_scale,
    )

    env = Turtlebot3LineFollowerEnv(
        repo_root, map_key=map_key or "simple", robot=robot, controller=controller, vision=vision,
    )
    summary, logs = run_trial(env, stop_time)

    parameters = TrialParameters(
        base_linear_speed=controller.base_linear_speed, base_speed_scale=controller.base_speed_scale,
        kp_line=controller.kp, ki_line=controller.ki, kd_line=controller.kd, n_line=controller.n_filter,
        roi_fraction=roi_fraction, local_path_points=local_path_points,
        local_path_lookahead_distance=local_path_lookahead_distance,
        local_path_lateral_gain=local_path_lateral_gain, local_path_heading_gain=local_path_heading_gain,
        local_path_curvature_gain=local_path_curvature_gain, min_brightness=min_brightness,
        max_saturation=max_saturation, min_pixels=min_pixels, error_scale=error_scale,
        max_wheel_speed=max_wheel_speed, stop_time=stop_time, map_key=env.map_key,
        trial_label=trial_label,
    )
    data_file = _save_trial(repo_root, env, parameters, summary, logs)

    print(f"Saved simulation data: {data_file}")
    print(f"Active TurtleBot3 map: {env.map_display_name} ({env.map_key})")
    print(f"Path {summary.path_length:.3f} m, found {100 * summary.line_found_ratio:.1f}%, "
          f"longest loss {summary.longest_line_loss:.2f} s, "
          f"wheel saturation {100 * summary.wheel_saturation_ratio:.1f}%.")
    if summary.lap_completed:
        print(f"Completed a lap in {summary.lap_time:.3f} s.")
    else:
        print("No complete lap was detected.")
    return summary, data_file


def _save_trial(repo_root: Path, env: Turtlebot3LineFollowerEnv, parameters: TrialParameters,
                 summary: TrialSummary, logs: list) -> Path:
    data_dir = Path(__file__).resolve().parents[1] / "simulation_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", parameters.trial_label)
    data_file = data_dir / f"visual_line_follower_with_debug_{safe_label}_{timestamp}.pkl"
    with data_file.open("wb") as f:
        pickle.dump({
            "parameters": asdict(parameters),
            "summary": asdict(summary),
            "logs": logs,
        }, f)
    return data_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-time", type=float, default=200.0)
    parser.add_argument("--map-key", type=str, default="")
    parser.add_argument("--trial-label", type=str, default="run")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[5]
    run_turtlebot3_burger_mujoco_visual_line_follower(
        repo_root, stop_time=args.stop_time, map_key=args.map_key, trial_label=args.trial_label,
    )


if __name__ == "__main__":
    main()
