"""Lost-line recovery: a state machine plus dead-reckoned line memory.

Ported from ``matlab/runtime/control/lf_line_search.m`` (2026-08-09).

This module **generates behavior** -- it decides where to go once the line is
gone. It is deliberately separate from :mod:`.safety_filter`, which decides
whether a command is allowed. Keeping "what to do" and "what is permitted"
apart is what makes the safety layer's blindness to command origin visible in
the code rather than just intended. In a control chain this runs first and the
filter second.

States (``found`` returns to TRACK from any of them):

===========  ==========================================================
TRACK        line visible; pass the command through, refresh the memory
HOLD         lost <= hold_time; still pass through -- the vision module is
             itself freezing/extrapolating steering_error over this window
             and a momentary occlusion should not interrupt that
RECOVER      memory usable; steer toward the remembered path
BRAKE        memory unusable/expired; ramp v and omega to zero
SCAN         in place (v = 0), sweeping +/-A1, +/-A2, ... progressively
GIVEUP       swept out or timed out; stop and wait
===========  ==========================================================

RECOVER, not scanning, is the primary strategy -- 2026-08-09 closed-loop
measurements found the robot stayed 3.7-4.5 cm from the centreline during a
20 s loss (i.e. sitting on the line the whole time) while the untouched
baseline recovered in ~5 s by continuing to crawl forward. Stopping to scan
was strictly worse. Continuing along the remembered path is the behavior that
works.

**But neither beat doing nothing**, which is why
:class:`~..config.LineSearchConfig` defaults ``enable`` to False. The
bottleneck was measured to be detection, not recovery: the line was inside
the camera's ROI footprint 32.2% of the lost period and simply not detected,
and the last reliable frame's memory spanned only 8 mm -- no turn
information. Fix the observation side before switching this on. Full
writeup: ``matlab/runtime/control/README.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

from ..config import LineSearchConfig
from . import line_memory

MAX_PATH_POINTS = 30


class SearchState(IntEnum):
    """Values match the MATLAB port's state numbering, so logs compare directly."""

    TRACK = 0
    HOLD = 1
    BRAKE = 2
    SCAN = 3
    GIVEUP = 4
    RECOVER = 5


@dataclass
class LineSearchResult:
    v: float                 # desired command, before the safety filter
    omega: float
    state: SearchState
    swept_angle: float       # scan progress (rad), relative to the scan's start heading
    memory_valid: bool
    memory_age: float
    memory_bearing: float    # rad, > 0 = remembered path is to the right


@dataclass
class LineSearch:
    """Stateful recovery behavior. One instance per control chain."""

    config: LineSearchConfig = field(default_factory=LineSearchConfig)

    _state: SearchState = field(default=SearchState.TRACK, init=False, repr=False)
    _t_lost: float = field(default=0.0, init=False, repr=False)
    _t_fallback: float = field(default=0.0, init=False, repr=False)
    _phi: float = field(default=0.0, init=False, repr=False)
    _leg: int = field(default=1, init=False, repr=False)
    _dir_sign: float = field(default=1.0, init=False, repr=False)
    _mem_points: np.ndarray = field(default=None, init=False, repr=False)
    _mem_count: int = field(default=0, init=False, repr=False)
    _mem_age: float = field(default=math.inf, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._state = SearchState.TRACK
        self._t_lost = 0.0
        self._t_fallback = 0.0
        self._phi = 0.0
        self._leg = 1
        self._dir_sign = 1.0
        self._mem_points = np.zeros((MAX_PATH_POINTS, 2))
        self._mem_count = 0
        self._mem_age = math.inf

    # ------------------------------------------------------------------
    def step(self, v_cmd: float, omega_cmd: float, lateral_error: float, found: bool,
             v_meas: float, omega_meas: float,
             path_points: Optional[np.ndarray] = None,
             path_point_count: int = 0) -> LineSearchResult:
        """Advance one control tick.

        ``v_meas``/``omega_meas`` are **measured** body velocities (m/s, rad/s)
        -- from wheel odometry, not the commanded values. They drive both the
        memory dead reckoning and the scan's swept-angle integration. Measured
        is used because the command still has the safety filter and wheel-speed
        saturation downstream of it, so integrating the command overestimates
        how far the robot actually turned.

        ``path_points`` is the vision module's ground-frame path points, an
        (N, 2) body-frame array (X forward, Y right) with the first
        ``path_point_count`` rows valid. **Pass ``None`` when they are not
        available** -- notably ``real/control_node.py``, whose wire format
        (``real/local_path_msg.py``) carries only five scalars and no
        geometry. The memory then never becomes valid and recovery degrades to
        the BRAKE -> SCAN -> GIVEUP chain, which is correct behavior rather
        than an error.
        """
        cfg = self.config
        prev_state = self._state

        self._update_memory(found, v_meas, omega_meas, path_points, path_point_count)
        memory_bearing, memory_usable = self._memory_target()

        if found:
            self._state = SearchState.TRACK
            self._t_lost = 0.0
            self._t_fallback = 0.0
            self._leg = 1
            self._phi = 0.0
            # Latch which way to sweep: path to the right (e > 0) -> scan right
            # (omega < 0) first.
            if abs(lateral_error) > cfg.dir_deadband:
                self._dir_sign = -math.copysign(1.0, lateral_error)
            elif abs(omega_cmd) > cfg.dir_deadband:
                self._dir_sign = math.copysign(1.0, omega_cmd)
            v_des, omega_des = v_cmd, omega_cmd
        else:
            self._t_lost += cfg.ts
            v_des, omega_des = self._lost_branch(
                v_cmd, omega_cmd, memory_bearing, memory_usable, prev_state,
            )

        # Integrate the sweep from the *measured* yaw rate (see the docstring).
        if self._state is SearchState.SCAN:
            self._phi += omega_meas * cfg.ts

        if not cfg.enable:
            # Ablation: the state machine still runs and still reports where it
            # would have been, but it does not take the command over.
            v_des, omega_des = v_cmd, omega_cmd

        return LineSearchResult(
            v=v_des, omega=omega_des, state=self._state, swept_angle=self._phi,
            memory_valid=memory_usable, memory_age=min(self._mem_age, 1e6),
            memory_bearing=memory_bearing,
        )

    # ------------------------------------------------------------------
    def _update_memory(self, found: bool, v_meas: float, omega_meas: float,
                       path_points: Optional[np.ndarray], path_point_count: int) -> None:
        """Overwrite the memory only with a *reliable* frame; otherwise dead-reckon.

        Do not "simplify" this back to storing every frame. A real line loss is
        not abrupt -- the valid-point count decays first. The measured run
        leading into one loss was 21, 19, 18, 16, 15, 13, 11, 9, 7, 5, 2, then
        0. Storing every frame therefore leaves the memory holding the *worst*
        geometry that ever existed (2 points), which is both below the
        usability threshold and no longer trustworthy. Gating on quality keeps
        the 21-point frame -- the last time the robot actually saw the line.
        """
        cfg = self.config
        new_count = int(max(0, min(MAX_PATH_POINTS, path_point_count)))
        if found and path_points is not None and new_count >= cfg.mem_min_points:
            self._mem_points[:] = 0.0
            self._mem_points[:new_count] = np.asarray(path_points, dtype=float)[:new_count]
            self._mem_count = new_count
            self._mem_age = 0.0
        else:
            self._mem_points = line_memory.propagate(
                self._mem_points, self._mem_count, v_meas, omega_meas, cfg.ts,
            )
            self._mem_age += cfg.ts

    def _memory_target(self) -> tuple:
        cfg = self.config
        if self._mem_count < cfg.mem_min_points or self._mem_age > cfg.mem_max_age:
            return 0.0, False
        target = line_memory.select_target(self._mem_points, self._mem_count, cfg.mem_lookahead)
        if target is None:
            return 0.0, False
        x_t, y_t = target
        return math.atan2(y_t, x_t), True   # bearing > 0 -> target is to the right

    def _lost_branch(self, v_cmd: float, omega_cmd: float, memory_bearing: float,
                     memory_usable: bool, prev_state: SearchState) -> tuple:
        cfg = self.config

        if self._t_lost <= cfg.hold_time:
            # Momentary occlusion: leave it to vision.py's own freeze/extrapolate.
            self._state = SearchState.HOLD
            self._t_fallback = 0.0
            return v_cmd, omega_cmd

        if memory_usable and self._t_lost <= cfg.hold_time + cfg.recover_timeout:
            # Directed recovery. **No BRAKE first** -- measurements showed that
            # continuing along the path is what recovers the line; braking and
            # re-accelerating throws that away.
            self._state = SearchState.RECOVER
            self._t_fallback = 0.0
            omega_des = min(cfg.max_omega,
                            max(-cfg.max_omega, -cfg.recover_gain * memory_bearing))
            if abs(memory_bearing) > cfg.recover_align_cone:
                v_des = 0.0     # too far off-axis: turn in place first
            else:
                v_des = cfg.recover_speed_frac * cfg.max_v
            return v_des, omega_des

        # Memory unusable or expired -> BRAKE -> SCAN -> GIVEUP. This chain runs
        # on its own clock: RECOVER may already have consumed several seconds,
        # and timing the chain off t_lost would compress or skip its phases.
        if self._t_fallback < cfg.brake_time:
            self._state = SearchState.BRAKE
            self._t_fallback += cfg.ts
            ramp = 1.0 - self._t_fallback / max(1e-12, cfg.brake_time)
            return v_cmd * ramp, omega_cmd * ramp

        n_legs = 2 * len(cfg.scan_amplitudes)
        if self._t_fallback <= cfg.brake_time + cfg.scan_timeout and self._leg <= n_legs:
            self._state = SearchState.SCAN
            self._t_fallback += cfg.ts
            if prev_state is not SearchState.SCAN:
                # phi is measured from the heading the *scan* starts at, not from
                # the heading at the moment of loss -- the robot is still turning
                # through HOLD/RECOVER.
                self._phi = 0.0
                self._leg = 1
            amplitude = cfg.scan_amplitudes[(self._leg - 1) // 2]
            leg_sign = 1.0 if self._leg % 2 == 1 else -1.0
            target = self._dir_sign * leg_sign * amplitude
            phi_error = target - self._phi
            if abs(phi_error) <= cfg.scan_tolerance:
                self._leg += 1          # leg reached; next target on the next tick
                omega_des = 0.0
            else:
                omega_des = math.copysign(cfg.scan_rate, phi_error)
            return 0.0, omega_des       # v = 0: scanning in place can never leave the track

        self._state = SearchState.GIVEUP
        self._t_fallback += cfg.ts
        return 0.0, 0.0
