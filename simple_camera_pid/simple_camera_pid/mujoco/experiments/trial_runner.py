"""Run a fixed-duration PID trial and compute track-following summary metrics.

Shared by ``run_turtlebot3_burger_mujoco_visual_line_follower.py``,
``tune_complex_track_pid.py``, and ``tune_localpath_gains.py`` — translated
from the summary-metric computation duplicated across
``run_turtlebot3_burger_mujoco_visual_line_follower.m`` and
``tune_complex_track_pid.m`` (``tune_localpath_gains.m`` already called the
former rather than duplicating it; this module generalizes that pattern to
all three).

Unlike the MATLAB versions, odometry and vision are logged at the exact same
20 Hz control-tick cadence here (one :class:`StepLog` per
``Turtlebot3LineFollowerEnv.step()`` call), so no resampling/interpolation
between differently-rated ``To Workspace`` logs is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim.turtlebot3_mujoco_env import StepLog, Turtlebot3LineFollowerEnv

LAP_LENGTH_THRESHOLD = 6.0   # m of travel before a return-to-start counts as a lap
LAP_RETURN_RADIUS = 0.25     # m


@dataclass
class TrialSummary:
    completed_simulation_time: float
    path_length: float
    lap_completed: bool
    lap_time: float
    line_found_ratio: float
    longest_line_loss: float
    mean_abs_lateral_error: float
    rms_lateral_error: float
    max_abs_lateral_error: float
    max_abs_heading_error: float
    wheel_saturation_ratio: float
    final_position: np.ndarray

    def score(self) -> float:
        """Composite score (lower is better), matching tune_complex_track_pid.m /
        tune_localpath_gains.m: rewards finding the line, penalizes lateral
        error, wheel saturation, and failing to complete a lap."""
        return (
            3 * (1 - self.line_found_ratio)
            + self.mean_abs_lateral_error
            + 0.35 * self.max_abs_lateral_error
            + 0.25 * self.wheel_saturation_ratio
            + 0.5 * (0.0 if self.lap_completed else 1.0)
        )


def run_trial(env: Turtlebot3LineFollowerEnv, stop_time: float) -> tuple[TrialSummary, list[StepLog]]:
    env.reset()
    n_steps = round(stop_time / env.controller_cfg.ts_control)
    logs: list[StepLog] = []
    for _ in range(n_steps):
        _, _, terminated, truncated, info = env.step(None)
        logs.append(info["log"])
        if terminated or truncated:
            break
    return _summarize(logs, env.robot.max_wheel_speed), logs


def _summarize(logs: list[StepLog], max_wheel_speed: float) -> TrialSummary:
    times = np.array([log.time for log in logs])
    positions = np.array([log.odom_position for log in logs])
    wheel_commands = np.array([log.wheel_command for log in logs])
    found = np.array([log.vision.found for log in logs])
    lateral = np.array([log.vision.lateral_error for log in logs])
    heading = np.array([log.vision.heading_error for log in logs])

    xy = positions[:, :2]
    step_distance = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    cumulative_distance = np.concatenate(([0.0], np.cumsum(step_distance)))
    distance_to_start = np.hypot(xy[:, 0] - xy[0, 0], xy[:, 1] - xy[0, 1])
    lap_index = np.flatnonzero(
        (cumulative_distance > LAP_LENGTH_THRESHOLD) & (distance_to_start < LAP_RETURN_RADIUS)
    )
    lap_completed = lap_index.size > 0
    lap_time = float(times[lap_index[0]]) if lap_completed else float("nan")

    wheel_saturated = np.any(np.abs(wheel_commands) >= max_wheel_speed - 1e-6, axis=1)

    return TrialSummary(
        completed_simulation_time=float(times[-1]),
        path_length=float(cumulative_distance[-1]),
        lap_completed=lap_completed,
        lap_time=lap_time,
        line_found_ratio=float(found.mean()),
        longest_line_loss=_longest_loss(found, times),
        mean_abs_lateral_error=float(np.mean(np.abs(lateral))),
        rms_lateral_error=float(np.sqrt(np.mean(lateral ** 2))),
        max_abs_lateral_error=float(np.max(np.abs(lateral))),
        max_abs_heading_error=float(np.max(np.abs(heading))),
        wheel_saturation_ratio=float(wheel_saturated.mean()),
        final_position=positions[-1],
    )


def _longest_loss(found: np.ndarray, time: np.ndarray) -> float:
    duration = 0.0
    start_index: int | None = None
    for k in range(len(found)):
        if not found[k] and start_index is None:
            start_index = k
        elif found[k] and start_index is not None:
            duration = max(duration, time[k - 1] - time[start_index])
            start_index = None
    if start_index is not None:
        duration = max(duration, time[-1] - time[start_index])
    return duration
