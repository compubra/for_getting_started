"""Run one 200-second trial and record its completion status.

Translated from ``launch_turtlebot3_manual_run.m``. There is no MuJoCo
on-screen viewer wired up here (see the top-level README's "known
limitations" section) — this runs headless and reports the same summary
metrics the MATLAB version prints after a visible run.

Usage
-----
    python3 -m simple_camera_pid.mujoco.experiments.launch_turtlebot3_manual_run
"""
from __future__ import annotations

import pickle
from pathlib import Path

from .run_turtlebot3_burger_mujoco_visual_line_follower import (
    run_turtlebot3_burger_mujoco_visual_line_follower,
)


def launch_turtlebot3_manual_run(repo_root: Path) -> Path:
    summary, data_file = run_turtlebot3_burger_mujoco_visual_line_follower(
        repo_root, stop_time=200.0, trial_label="manual_visible_200s",
    )

    status_file = Path(__file__).resolve().parents[1] / "last_manual_run_status.pkl"
    with status_file.open("wb") as f:
        pickle.dump({"data_file": str(data_file), "summary": summary}, f)

    print(data_file)
    print(summary)
    return data_file


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    launch_turtlebot3_manual_run(repo_root)


if __name__ == "__main__":
    main()
