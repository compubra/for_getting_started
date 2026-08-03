"""Plot residual-RL (SAC/PPO) training curves after training.

Translated from ``plot_rl_training.m``. MATLAB's ``rl.train.result.
rlTrainingResult`` (``trainInfo``) has no stable-baselines3 equivalent, so
this reads the ``stable_baselines3.common.monitor.Monitor`` CSV log written
during training (``monitor.csv``, columns ``r`` reward / ``l`` length /
``t`` wall-clock) instead of a saved ``trainInfo`` .mat file.

The MATLAB version's second panel plots the critic's initial-state Q0
estimate, which has no equivalent without extra custom logging (stable-
baselines3 doesn't expose it by default) — this plots episode length instead,
which is informative for this task (a longer episode means the line was
followed longer before a lost-line timeout).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_monitor_log(monitor_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a Monitor CSV log, returning (episode_index, reward, length)."""
    with monitor_csv.open() as f:
        first_line = f.readline()
        if not first_line.startswith("#"):
            f.seek(0)
        reader = csv.DictReader(f)
        rewards, lengths = [], []
        for row in reader:
            rewards.append(float(row["r"]))
            lengths.append(float(row["l"]))
    episode = np.arange(1, len(rewards) + 1)
    return episode, np.array(rewards), np.array(lengths)


def _rolling_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    window = max(1, min(window, values.size))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_rl_training(source: Path | str, window: int = 20) -> None:
    """Plot the learning curve from a ``monitor.csv`` produced during training.

    ``source`` may be the CSV file itself or a directory containing one
    (e.g. ``simulation_data/sac_training/<run>/``).
    """
    import matplotlib.pyplot as plt

    source = Path(source)
    monitor_csv = source if source.is_file() else _find_monitor_csv(source)
    episode, reward, length = load_monitor_log(monitor_csv)
    avg = _rolling_average(reward, window)
    avg_episode = episode[-avg.size:]

    fig, (ax_reward, ax_length) = plt.subplots(2, 1, figsize=(8, 8))
    ax_reward.plot(episode, reward, "-", color=(0.7, 0.7, 0.9), label="Episode reward")
    ax_reward.plot(avg_episode, avg, "-", linewidth=2, color=(0.10, 0.30, 0.75),
                    label=f"{window}-episode average")
    ax_reward.grid(True)
    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Reward")
    final_avg = avg[-1] if avg.size else float("nan")
    ax_reward.set_title(f"Learning curve (final avg = {final_avg:.3f} over {episode[-1]} episodes)")
    ax_reward.legend(loc="best")

    ax_length.plot(episode, length, "-", linewidth=1.2, color=(0.75, 0.35, 0.10))
    ax_length.grid(True)
    ax_length.set_xlabel("Episode")
    ax_length.set_ylabel("Episode length (steps)")
    ax_length.set_title("Episode length (proxy for how long the line was tracked)")
    fig.tight_layout()
    plt.show()


def _find_monitor_csv(root: Path) -> Path:
    candidates = sorted(root.rglob("*.monitor.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No *.monitor.csv found under {root}. Train first, or pass a file.")
    return candidates[0]
