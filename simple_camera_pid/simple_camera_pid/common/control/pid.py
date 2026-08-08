"""Discrete PID with filtered derivative and back-calculation anti-windup.

Translated from the ``LinePID`` block (Simulink ``slpidlib/PID Controller``,
Parallel form, Discrete-time, ``UseFilter=on``, ``IntegratorMethod=Backward
Euler``, ``FilterMethod=Trapezoidal``) inside
``visual_line_follower_with_debug.slx``'s ``Subsystem``. Simulink's PID block
is a closed-source S-function; this reimplements the same parallel-form
transfer function

    C(s) = Kp + Ki/s + Kd*N*s/(s + N)

via a backward-Euler integrator and a trapezoidal discretization of the
derivative filter's state-space realization (state x with dx/dt = -N*x + e,
d = Kd*N*(e - N*x)), plus back-calculation anti-windup. This was numerically
verified against MATLAB's ``pid`` object (``IFormula=BackwardEuler,
DFormula=Trapezoidal``, which implements the same parallel/filtered-
derivative structure as the Simulink PID Controller block) via
``tf(c)``/``lsim`` step responses, and matches to at least 4 significant
figures. It is not a bit-exact port of Simulink's internal PID S-function
numerics (undocumented), particularly right at output saturation, where the
back-calculation anti-windup gain (Kb) is a documented but not fully
specified implementation detail.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FilteredPID:
    kp: float
    ki: float
    kd: float
    n_filter: float
    ts: float
    out_min: float
    out_max: float
    kb: float = 1.0  # back-calculation anti-windup gain (Kb, matches the block default)

    _integrator: float = field(default=0.0, init=False, repr=False)
    _filter_state: float = field(default=0.0, init=False, repr=False)
    _prev_error: float = field(default=0.0, init=False, repr=False)

    def reset(self) -> None:
        self._integrator = 0.0
        self._filter_state = 0.0
        self._prev_error = 0.0

    def step(self, error: float) -> float:
        proportional = self.kp * error

        half_tsn = 0.5 * self.ts * self.n_filter
        new_filter_state = (
            self._filter_state * (1.0 - half_tsn) + 0.5 * self.ts * (error + self._prev_error)
        ) / (1.0 + half_tsn)
        derivative = self.kd * self.n_filter * (error - self.n_filter * new_filter_state)
        self._filter_state = new_filter_state
        self._prev_error = error

        integrator_candidate = self._integrator + self.ts * self.ki * error
        unsaturated = proportional + integrator_candidate + derivative
        saturated = min(self.out_max, max(self.out_min, unsaturated))

        # Back-calculation: feed the saturation error back into the integrator
        # so it stops winding up once the output is clamped. Only meaningful
        # when ki != 0 -- with ki == 0 the integrator never accumulates
        # anything of its own (ts*ki*error is always 0), so it's supposed to
        # be permanently inert; applying the correction anyway still wrote a
        # nonzero value into it on any saturation event, and since nothing
        # with ki == 0 can ever unwind that value again, it leaked into
        # every future output as a permanent bias (reproduced directly: a
        # single saturating step left the steady-state output stuck at a
        # nonzero constant forever after, even with the error at exactly 0).
        #
        # This matters on the real robot specifically: config/real/
        # real_line_follower.yaml runs ki: 0.0, and the 2026-08-04 bag shows
        # the wheel command pinned at saturation (wheel_left=-1.34,
        # wheel_right=3.505 rad/s) -- i.e. exactly the condition that arms
        # this leak. Found and fixed in the simulation copy of this file on
        # 2026-08-07; ported here 2026-08-08, where it had been sitting
        # unfixed the whole time.
        if self.ki != 0.0:
            self._integrator = integrator_candidate + self.ts * self.kb * (saturated - unsaturated)
        else:
            self._integrator = 0.0

        return saturated
