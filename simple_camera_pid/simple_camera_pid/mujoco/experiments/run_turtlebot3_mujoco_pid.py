"""Run the TurtleBot3 PID baseline in MuJoCo — headless smoke-test / demo.

Translated from ``run_turtlebot3_mujoco_pid.m``.

Usage
-----
    python3 -m simple_camera_pid.mujoco.experiments.run_turtlebot3_mujoco_pid --stop-time 10
    python3 -m simple_camera_pid.mujoco.experiments.run_turtlebot3_mujoco_pid --stop-time 2 --headless
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..sim.turtlebot3_mujoco_env import Turtlebot3LineFollowerEnv


def run_turtlebot3_mujoco_pid(
    repo_root: Path, stop_time: float = 10.0, target_lateral_position: float = 0.0,
) -> Turtlebot3LineFollowerEnv:
    env = Turtlebot3LineFollowerEnv(repo_root, map_key="simple")
    env.reset(options={"lateral_target": target_lateral_position})

    n_steps = round(stop_time / env.controller_cfg.ts_control)
    odom = np.zeros((n_steps, 3))
    wheel = np.zeros((n_steps, 2))
    for k in range(n_steps):
        _, _, terminated, truncated, info = env.step(None)
        log = info["log"]
        odom[k] = log.odom_position
        wheel[k] = log.wheel_command
        if terminated or truncated:
            odom, wheel = odom[: k + 1], wheel[: k + 1]
            break

    if not np.all(np.isfinite(odom)):
        raise RuntimeError("MuJoCo odometry contains NaN or Inf.")
    if not np.all(np.isfinite(wheel)):
        raise RuntimeError("Wheel commands contain NaN or Inf.")

    final_position = odom[-1]
    print(f"MuJoCo simulation complete: {env.data.time:.3f} s, {len(odom)} odometry samples.")
    print(f"Final position [x y z] = [{final_position[0]:.4f} {final_position[1]:.4f} "
          f"{final_position[2]:.4f}] m.")
    print(f"Final lateral error = {target_lateral_position - final_position[1]:.4f} m.")
    print(f"Peak wheel command = {np.max(np.abs(wheel)):.4f} rad/s.")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-time", type=float, default=10.0)
    parser.add_argument("--target-lateral", type=float, default=0.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[5]
    run_turtlebot3_mujoco_pid(repo_root, args.stop_time, args.target_lateral)


if __name__ == "__main__":
    main()
