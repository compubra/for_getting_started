"""Odometry dead-reckoned memory of the last reliably-seen line geometry.

Ported from ``matlab/runtime/control/lf_memory_propagate.m`` and
``lf_memory_target.m`` (2026-08-09). Both are pure functions -- the state
(the point buffer and its age) lives in :class:`~.line_search.LineSearch`.

Purpose: while the line is lost there is no new observation, but the robot
keeps moving, so the last good observation can still be carried forward
through the wheel odometry. That gives the recovery behavior a geometric
target ("the line was there, and I have since moved this much") instead of
having to scan blind.

**Known limitation, measured 2026-08-09** -- the memory's spatial extent is
capped by how deep the ROI sees. On the MuJoCo hairpin where this was
debugged, the last reliable frame's path points spanned x = [0.163, 0.171] m:
**8 millimetres, containing no turn information at all**. Following that
memory just drives straight. So this is not a fix for losing the line at a
sharp curve; the observation horizon has to grow first. See
``matlab/runtime/control/README.md``.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def propagate(points: np.ndarray, n_valid: int, v_mps: float, omega_radps: float,
              dt: float) -> np.ndarray:
    """Carry remembered ground points forward by one motion increment.

    ``points`` is an (N, 2) array of body-frame ground points ``[x, y]`` with
    X forward and **Y to the right** -- the convention
    ``vision.py``/``originbot_sliding_window_path_generator.m`` emit. Only the
    first ``n_valid`` rows are touched; the rest are returned untouched (and
    callers must not read them).

    The robot first advances ``v*dt`` along its own X, then rotates by
    ``dtheta = omega*dt``. In the *new* body frame a remembered point is
    therefore translated, then rotated::

        xt    = x - v*dt
        x_new = xt*cos(dtheta) - y *sin(dtheta)
        y_new = xt*sin(dtheta) + y *cos(dtheta)

    The rotation's sign is pinned by one physical fact: **omega > 0 is a left
    turn, and when the robot turns left a point straight ahead moves to the
    right (+y) in the body frame.** (The same convention underpins the CBF in
    ``safety_filter.py``; the two must agree.) Check it with the point (L, 0)
    and dtheta > 0: ``y_new = L*sin(dtheta) > 0``, i.e. it does move toward
    +y. Flipping these two sines does not raise anything -- it silently walks
    the memory the wrong way, and while the line is lost there is no vision to
    correct it.

    Pure kinematic integration, no observation update: wheel slip and odometry
    miscalibration accumulate, which is why callers must bound the memory's
    age (``mem_max_age``) rather than trusting it indefinitely.
    """
    if n_valid <= 0:
        return points

    dtheta = omega_radps * dt
    c = math.cos(dtheta)
    s = math.sin(dtheta)

    idx = slice(0, min(int(n_valid), points.shape[0]))
    xt = points[idx, 0] - v_mps * dt
    y = points[idx, 1]
    points[idx, 0] = xt * c - y * s
    points[idx, 1] = xt * s + y * c
    return points


def select_target(points: np.ndarray, n_valid: int,
                  lookahead: float) -> Optional[Tuple[float, float]]:
    """Pick the point to steer toward, or ``None`` if the memory is unusable.

    ``points`` must be ordered **along the path**, near to far -- which is how
    the sliding window produces them (it climbs image rows from the bottom of
    the ROI upward). Returns the first point at least ``lookahead`` away from
    the robot; if every remembered point is nearer than that, returns the one
    furthest along the path.

    **Selection is by path order, not by distance.** Once the robot has turned,
    a nearest-first search picks the point behind it and steers it backwards;
    path order always keeps it heading the way it was originally going, which
    is the behavior that let the untouched baseline recover by crawling
    forward while in-place scanning did not.
    """
    n = min(int(n_valid), points.shape[0])
    if n <= 0:
        return None

    for k in range(n):
        if math.hypot(float(points[k, 0]), float(points[k, 1])) >= lookahead:
            return float(points[k, 0]), float(points[k, 1])

    return float(points[n - 1, 0]), float(points[n - 1, 1])
