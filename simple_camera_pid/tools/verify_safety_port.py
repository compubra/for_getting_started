#!/usr/bin/env python3
"""Numeric parity check: Python safety layer vs its MATLAB original.

The MATLAB implementation (``matlab/runtime/control/lf_safety_filter.m``) is
the reference this port was written from, and both are pure arithmetic with no
toolbox calls, so unlike the vision port (whose Hough stage genuinely cannot be
bit-exact -- see ``common/vision.py``) this one *can* be checked to machine
precision. This script does that, so "ported" is a measured claim rather than
an assertion.

Usage
-----
Regenerate the reference from MATLAB (needs a MATLAB with this repo's
``runtime/`` on its path), writing 11 columns -- 5 inputs then 6 of the 7
outputs -- one row per tick::

    p = struct("EnableFilter",true,"Ts",0.05,"SpeedScale",0.03,"Lookahead",0.20, ...
      "LateralNorm",0.55,"LateralMax",0.8,"CBFAlpha",2.0,"CBFSingularTol",1e-3, ...
      "SpeedBackoff",0.3,"MaxV",20.0,"MaxOmega",1.5,"MaxAccelV",40.0,"MaxAccelOmega",10.0);
    % U = [v, omega, lateral_error, heading_error, found] per row
    clear lf_safety_filter;
    OUT = zeros(size(U,1),7);
    for k = 1:size(U,1), OUT(k,:) = lf_safety_filter(U(k,:).', p).'; end

then run::

    python3 tools/verify_safety_port.py <reference.csv>

The filter is stateful (rate limits carry ``v_prev``/``omega_prev`` between
ticks), so the rows must be replayed in order -- a row-shuffled reference
would not agree even for a correct port.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simple_camera_pid.common.config import SafetyFilterConfig  # noqa: E402
from simple_camera_pid.common.control.safety_filter import SafetyFilter  # noqa: E402

TOLERANCE = 1e-12

# Column layout of the reference CSV.
IN_COLS = 5      # v, omega, lateral_error, heading_error, found
OUT_NAMES = ("v", "omega", "barrier", "omega_lo", "omega_hi", "cbf_active", "infeasible")


def main(path: str) -> int:
    rows = []
    with open(path, newline="") as handle:
        for raw in csv.reader(handle):
            if raw and raw[-1] == "":
                raw = raw[:-1]
            if raw:
                rows.append([float(x) for x in raw])
    if not rows:
        print(f"no rows in {path}")
        return 2

    # Defaults must match the struct the reference was generated with.
    config = SafetyFilterConfig(
        enable=True, ts=0.05, speed_scale=0.03, lookahead=0.20, lateral_norm=0.55,
        lateral_max=0.8, cbf_alpha=2.0, cbf_singular_tol=1e-3, speed_backoff=0.3,
        max_v=20.0, max_omega=1.5, max_accel_v=40.0, max_accel_omega=10.0,
    )
    filt = SafetyFilter(config)

    worst = 0.0
    worst_at = None
    failures = 0
    for i, row in enumerate(rows):
        v, omega, lateral, heading, found = row[:IN_COLS]
        expected = row[IN_COLS:IN_COLS + len(OUT_NAMES)]
        result = filt.step(v, omega, lateral, heading, found >= 0.5)
        actual = (result.v, result.omega, result.barrier, result.omega_lo,
                  result.omega_hi, float(result.cbf_active), float(result.infeasible))

        for name, got, want in zip(OUT_NAMES, actual, expected):
            # +/-inf can legitimately appear if a bound were unbounded; compare
            # those exactly rather than differencing them into a NaN.
            if got == want:
                continue
            delta = abs(got - want)
            if delta > worst:
                worst, worst_at = delta, (i, name, got, want)
            if delta > TOLERANCE:
                failures += 1
                if failures <= 10:
                    print(f"  row {i:3d}  {name:<10} python={got!r} matlab={want!r} "
                          f"delta={delta:.3e}")

    print(f"\nreplayed {len(rows)} ticks against {Path(path).name}")
    if worst_at is not None:
        i, name, got, want = worst_at
        print(f"largest difference: {worst:.3e} at row {i} ({name}: "
              f"python={got:.17g}, matlab={want:.17g})")
    else:
        print("largest difference: 0 (exact)")

    if failures:
        print(f"FAIL: {failures} value(s) exceeded {TOLERANCE:g}")
        return 1
    print(f"PASS: every value within {TOLERANCE:g}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
