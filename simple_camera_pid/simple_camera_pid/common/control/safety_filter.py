"""Inference-time safety layer: discrete CBF + box + rate constraints.

Ported from ``matlab/runtime/control/lf_safety_filter.m`` (2026-08-09).

**This layer is deliberately blind to where the command came from.** PID,
SAC/PPO residual, or the lost-line recovery behavior -- all are treated
identically. That is the whole point of the safety-layer position in Gu et
al., "A Review of Safe Reinforcement Learning: Methods, Theories, and
Applications", IEEE TPAMI 46(12):11216-11235, 2024, section III-B-2 (safety
layer / OptLayer [67]): run the policy's output through a control filter
rather than folding safety into the objective. The practical payoff is that
it **needs no retraining** -- existing SAC checkpoints keep working, whereas
CPO/IPO-style cost-value optimization (section III-B-1) would not.

Constraint type (section II-A): this is an *instantaneous, explicit*
constraint -- closed-form and checkable every step -- not one of the
trajectory-cumulative constraints of equations (1)-(3). Cumulative
constraints cannot be added without changing the training objective.

CBF derivation
--------------
Let ``y_L`` be the lookahead point's lateral offset in the body frame (m, Y
right-positive) and ``psi`` the path tangent angle there. The vision module
reports the normalized ``e = y_L / lateral_norm``.

Lookahead-offset dynamics for a differential drive (path curvature dropped)::

    d(y_L)/dt = v*sin(psi) + L*omega

with L the lookahead distance. The second term's sign: omega > 0 turns left,
and turning left moves a point ahead toward +y, so y_L grows. The first: with
the path tangent pointing right (psi > 0), driving forward moves the lookahead
point right.

The barrier is built on the **lookahead** offset rather than the robot's own
lateral offset because the lookahead point has relative degree 1 in omega --
omega appears directly in the first derivative -- so the CBF condition
collapses to a single affine inequality and its projection is one clamp. The
robot's own offset has relative degree 2 and would need a HOCBF cascade. The
vision module's ``lateral_error`` already *is* the lookahead offset, so
nothing extra has to be estimated.

Barrier (safe set ``h >= 0``, i.e. ``|e| <= lateral_max``)::

    h(e) = lateral_max^2 - e^2

Substituting into ``dh/dt >= -alpha*h`` and multiplying through by
``lateral_norm/2 > 0`` gives ``A*omega >= b`` with::

    A = -e*L
    b = -(alpha*lateral_norm/2)*h + e*v*sin(psi)

For e > 0 (path to the right, robot left of it) A < 0 and the constraint reads
``omega <= b/A`` -- no turning further left. Symmetric for e < 0. At e ~ 0,
A ~ 0 and b < 0, so the constraint is vacuous (0 >= negative); ``cbf_singular_tol``
guards that degeneracy.

``v`` is not part of the CBF -- keeping the decision variable a scalar omega
is what makes the closed form possible. It gets box/rate limits plus a
near-boundary slowdown, and **that slowdown is a heuristic, not part of the
forward-invariance guarantee.** Please keep that distinction in any writeup.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import SafetyFilterConfig


@dataclass
class SafetyFilterResult:
    """Applied command plus everything needed to audit the layer offline."""

    v: float                # command actually issued
    omega: float
    v_desired: float        # what came in, before projection
    omega_desired: float
    barrier: float          # h; < 0 means outside the safe set
    omega_lo: float         # final effective bounds (already intersected with box/rate)
    omega_hi: float
    cbf_active: bool        # a finite CBF bound existed this tick
    infeasible: bool        # CBF demanded more than the actuator/rate limit allows

    @property
    def omega_intervention(self) -> float:
        return self.omega - self.omega_desired

    @property
    def v_intervention(self) -> float:
        return self.v - self.v_desired


@dataclass
class SafetyFilter:
    """Stateful one-step projection. One instance per control chain."""

    config: SafetyFilterConfig = field(default_factory=SafetyFilterConfig)

    _v_prev: float = field(default=0.0, init=False, repr=False)
    _omega_prev: float = field(default=0.0, init=False, repr=False)

    def reset(self) -> None:
        self._v_prev = 0.0
        self._omega_prev = 0.0

    def step(self, v_desired: float, omega_desired: float, lateral_error: float,
             heading_error: float, found: bool) -> SafetyFilterResult:
        """Project one (v, omega) command into the safe set.

        With ``config.enable`` False the bounds and diagnostics are still
        computed and returned -- that is exactly the "how much would this have
        violated without the layer" measurement -- but the command passes
        through untouched.
        """
        cfg = self.config

        # ---- CBF: an affine bound on omega
        v_phys = v_desired * cfg.speed_scale
        h = cfg.lateral_max ** 2 - lateral_error ** 2
        cbf_lo = -math.inf
        cbf_hi = math.inf
        cbf_active = False
        if found:
            # While lost the vision module forces lateral_error to 0, so h is
            # positive and the constraint is vacuous anyway -- gate on found
            # explicitly rather than relying on that implicit behavior.
            a = -lateral_error * cfg.lookahead
            b = (-0.5 * cfg.cbf_alpha * cfg.lateral_norm * h
                 + lateral_error * v_phys * math.sin(heading_error))
            if abs(a) > cfg.cbf_singular_tol:
                bound = b / a
                if a > 0:
                    cbf_lo = bound
                else:
                    cbf_hi = bound
                cbf_active = True

        # ---- box ∩ rate (the hard set; never empty)
        d_omega = cfg.max_accel_omega * cfg.ts
        hard_lo = max(-cfg.max_omega, self._omega_prev - d_omega)
        hard_hi = min(cfg.max_omega, self._omega_prev + d_omega)

        omega_lo = max(hard_lo, cbf_lo)
        omega_hi = min(hard_hi, cbf_hi)

        infeasible = False
        if omega_lo > omega_hi:
            # The CBF wants more turn than the actuator/rate limit can deliver
            # this tick. Fall back to the point of the hard set closest to the
            # CBF bound -- equivalent to slacking the CBF constraint and
            # minimizing the violation. Physical feasibility wins, otherwise we
            # would issue a command the motors simply cannot execute.
            infeasible = True
            omega_safe = hard_hi if cbf_lo > hard_hi else hard_lo
            omega_lo, omega_hi = hard_lo, hard_hi
        else:
            omega_safe = min(max(omega_desired, omega_lo), omega_hi)

        # ---- linear speed
        v_scale = 1.0
        if found and cfg.speed_backoff > 0.0:
            margin = max(0.0, h) / max(1e-12, cfg.lateral_max ** 2)  # 1 at centre, 0 at edge
            v_scale = 1.0 - cfg.speed_backoff * (1.0 - margin)
        v_target = v_desired * v_scale

        d_v = cfg.max_accel_v * cfg.ts
        v_lo = max(0.0, self._v_prev - d_v)          # v >= 0: the residual must not reverse
        v_hi = min(cfg.max_v, self._v_prev + d_v)
        v_safe = min(max(v_target, v_lo), v_hi)

        if not cfg.enable:
            v_safe = v_desired
            omega_safe = omega_desired

        # Track what was actually issued either way, so the rate window stays
        # correct across an enable/disable boundary.
        self._v_prev = v_safe
        self._omega_prev = omega_safe

        return SafetyFilterResult(
            v=v_safe, omega=omega_safe, v_desired=v_desired, omega_desired=omega_desired,
            barrier=h, omega_lo=omega_lo, omega_hi=omega_hi,
            cbf_active=cbf_active, infeasible=infeasible,
        )
