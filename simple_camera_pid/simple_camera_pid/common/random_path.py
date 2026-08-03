"""Generate and (optionally) plot smooth random reference paths.

Translated from ``gen_random_local_path.m``. Used for visualizing/sanity-
checking the pure-pursuit steering geometry against reference curves of
varying "difficulty", independent of the vision pipeline or MuJoCo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class RandomPath:
    x: np.ndarray          # (num_points,) forward coordinate (m)
    y: np.ndarray          # (num_points,) lateral coordinate (m)
    curvature: np.ndarray  # (num_points,) curvature (1/m)
    heading: np.ndarray    # (num_points,) heading angle (rad)
    coeffs: np.ndarray     # 2nd-order polyfit [a, b, c], same convention as vision.py


def gen_random_local_path(
    count: int = 5,
    num_points: int = 30,
    length: float = 2.0,
    difficulty: float = 0.5,
    num_control: int = 5,
    max_lateral: float = 0.6,
    seed: int | None = None,
    plot: bool = True,
) -> list[RandomPath]:
    rng = np.random.default_rng(seed)
    n = round(num_points)
    nc = max(2, round(num_control))
    amp = max_lateral * (0.15 + 0.85 * difficulty ** 2)

    paths: list[RandomPath] = []
    for _ in range(round(count)):
        x = np.linspace(0, length, n)

        xc = np.linspace(0, length, nc)
        yc = np.concatenate(([0.0], (2 * rng.random(nc - 1) - 1) * amp))
        peak = np.max(np.abs(yc))
        if peak > np.finfo(float).eps:
            yc = yc * (amp / peak)
        yc = np.clip(yc, -max_lateral, max_lateral)

        spline = CubicSpline(xc, yc)
        y = spline(x)
        yp = np.gradient(y, x)
        ypp = np.gradient(yp, x)
        heading = np.arctan(yp)
        curvature = ypp / np.maximum(np.finfo(float).eps, (1 + yp ** 2) ** 1.5)

        coeffs = np.polyfit(x, y, 2)
        paths.append(RandomPath(x=x, y=y, curvature=curvature, heading=heading, coeffs=coeffs))

    if plot:
        _plot_paths(paths, difficulty, length)
    return paths


def _plot_paths(paths: list[RandomPath], difficulty: float, length: float) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_path, ax_curv) = plt.subplots(2, 1, figsize=(7, 8))
    for i, path in enumerate(paths):
        ax_path.plot(path.x, path.y, "-", linewidth=1.8, label=f"path {i + 1}")
        ax_curv.plot(path.x, path.curvature, "-", linewidth=1.2)
    ax_path.plot([0, length], [0, 0], "k--", linewidth=0.5)
    ax_path.set_aspect("equal")
    ax_path.grid(True)
    ax_path.set_xlabel("X forward (m)")
    ax_path.set_ylabel("Y lateral (m)")
    ax_path.set_title(f"{len(paths)} random paths | difficulty={difficulty:.2f} | length={length:g} m")
    ax_path.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

    ax_curv.grid(True)
    ax_curv.set_xlabel("X forward (m)")
    ax_curv.set_ylabel("curvature (1/m)")
    ax_curv.set_title("Path curvature")
    fig.tight_layout()
    plt.show()
