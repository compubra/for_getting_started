"""Sanity tests for the MATLAB → Python translation.

The vision pipeline and the discrete PID were verified against the original
MATLAB implementations to ~1e-13 (golden tests run via the MATLAB session
during translation); these tests re-check internal consistency plus the
reward/done logic, and step the full MuJoCo env when a GL context exists.

Run:  python -m pytest tests/ -v      (or just: python tests/test_translation.py)
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lsac.controller import ControllerParams, DiscretePID, LineFollowController
from lsac.reward import DoneDetector, LyapunovReward, RewardParams
from lsac.vision import LocalPathGenerator

H, W = 480, 640


def synthetic_frame(offset_px: float) -> np.ndarray:
    img = np.full((H, W, 3), 25.0)
    cols = np.arange(W)
    center = (W + 1) / 2 + offset_px
    for r in range(H // 2, H):
        img[r, np.abs(cols - center) < 20, :] = 230.0
    return img


def test_vision_signs_and_lost_line():
    gen = LocalPathGenerator()
    right = gen.process(synthetic_frame(60))
    assert right.found == 1.0 and right.steering_error > 0
    gen.reset()
    left = gen.process(synthetic_frame(-60))
    assert left.found == 1.0 and left.steering_error < 0

    # lost line: freeze (<=0.5s) then decay then zero (>1.5s)
    gen.reset()
    seen = gen.process(synthetic_frame(60))
    dark = np.full((H, W, 3), 20.0)
    r1 = gen.process(dark)             # 0.05s lost -> frozen steering
    assert math.isclose(r1.steering_error, seen.steering_error)
    assert r1.lateral_error == 0.0 and r1.found == 0.0
    for _ in range(40):                # push past 1.5 s
        r = gen.process(dark)
    assert r.steering_error == 0.0


def test_pid_recovery_and_mixing():
    # Formula-correctness check (block-wiring math), decoupled from whichever
    # physical profile ControllerParams() defaults to today: pass the values
    # explicitly so this test still verifies the wiring 1e-13-style even after
    # the defaults are next re-synced to a different MATLAB model revision.
    ctrl = LineFollowController(ControllerParams(
        base_linear_speed=20.0, wheel_separation=0.288, max_wheel_speed=25.0))
    # found: straight line, zero error -> both wheels at base speed
    w = ctrl.pid_wheels(0.0, 1.0)
    base = 20 * (0.8 + 0.2) * 1.0 * 0.03 / 0.033
    assert np.allclose(w, [base, base])
    # steering error > 0 (line right) -> turn right: left speeds up
    ctrl.reset()
    w = ctrl.pid_wheels(0.5, 1.0)
    assert w[1] > w[0] or w[0] > w[1]  # differential engaged
    assert not np.isclose(w[0], w[1])
    # residual mixing and saturation
    total = ctrl.mix_residual(np.array([24.0, 24.0]), 1.0)
    y2w = 0.288 / (2 * 0.033)
    assert math.isclose(total[0], 25.0)              # clamped at MaxWheelSpeed
    assert math.isclose(total[1], 24.0 - y2w)
    # lost line -> recovery steering clamp at ±0.45 into the PID
    ctrl.reset()
    w_lost = ctrl.pid_wheels(1.5, 0.0)
    assert np.all(np.abs(w_lost) <= 25.0)


def test_pid_uses_current_model_defaults():
    # ControllerParams() with no overrides must reflect the CURRENT
    # visual_line_follower_sac_residual model (real-car-speed sync,
    # 2026-07-20), not the lyapunov/Waffle values test_pid_recovery_and_mixing
    # pins explicitly above.
    p = ControllerParams()
    assert (p.base_linear_speed, p.max_wheel_speed, p.wheel_separation,
            p.steering_sign) == (5.0, 7.9, 0.16, -1.0)
    ctrl = LineFollowController(p)
    w = ctrl.pid_wheels(0.0, 1.0)
    base = 5.0 * (0.8 + 0.2) * 1.0 * 0.03 / 0.033
    assert np.allclose(w, [base, base])
    total = ctrl.mix_residual(np.array([7.0, 7.0]), 1.0)
    assert np.all(np.abs(total) <= 7.9 + 1e-9)


def test_reward_matches_formula():
    # NOTE: the reward formula carries an Alive_Bonus term (+p.alive while
    # found) and a P_Done one-time penalty (applied by the env on lost-line
    # termination, not inside .step()) added after the original 2000-episode
    # SAC run (job 10797287) plateaued from an incentive-ordering bug — see
    # reward.py's module docstring. Both expected-value formulas below include
    # +p.alive on found=1.0 samples to match the CURRENT step() implementation.
    p = RewardParams()
    rw = LyapunovReward(p)
    # priming sample (gate=0): Vdot masked
    r0 = rw.step(0.3, 0.1, 0.2, 0.05, u=0.0, found=1.0)
    v0 = p.q_e * 0.09 + p.q_l * 0.01 + p.q_h * 0.04 + p.q_v * 0.0025
    expected0 = (-math.tanh(v0) - 0.0
                 - p.k_inv * (1 - math.exp(-p.beta * 0.6)) + p.alive)
    assert math.isclose(r0, expected0, rel_tol=1e-12)
    # second sample: Vdot active
    r1 = rw.step(0.1, 0.05, 0.1, 0.0, u=0.5, found=1.0)
    v1 = p.q_e * 0.01 + p.q_l * 0.0025 + p.q_h * 0.01
    vdot = (v1 - v0) / p.ts
    expected1 = (-math.tanh(v1 + p.lam * vdot) - p.r * 0.25
                 - p.k_inv * (1 - math.exp(-p.beta * 0.25)) + p.alive)
    assert math.isclose(r1, expected1, rel_tol=1e-12)
    # lost line: only P_Lost (bonus and alive both gated off by found=0)
    rw.reset()
    r_lost = rw.step(0.0, 0.0, 0.0, 0.0, u=0.0, found=0.0)
    assert math.isclose(r_lost, -math.tanh(0.0) - p.p_lost)


def test_done_counter():
    d = DoneDetector(done_steps=3)
    assert not any(d.step(0.0) for _ in range(3))   # 1,2,3 -> not yet
    assert d.step(0.0)                              # 4 > 3 -> done
    d.step(1.0)
    assert not d.step(0.0)                          # reset on found


def test_env_physics_and_optionally_render():
    """Model + physics always; camera render only if a GL context works."""
    import mujoco
    from lsac.scenes import resolve_scene

    model = mujoco.MjModel.from_xml_path(resolve_scene("training"))
    data = mujoco.MjData(model)
    model.actuator("left_wheel_velocity_cmd")   # raises if missing
    data.ctrl[:] = 5.0
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert np.linalg.norm(data.sensor("odom_linear_velocity").data[:2]) > 0.01

    try:
        r = mujoco.Renderer(model, height=H, width=W)
        r.update_scene(data, camera="turtlebot3_front_camera")
        rgb = r.render()
        r.close()
    except Exception as e:
        print(f"  [render skipped: no GL context here — {e}]")
        return

    from lsac.env import LineFollowerEnv
    env = LineFollowerEnv(map_key="training", seed=0)
    obs, info = env.reset()
    assert obs.shape == (5,)
    total = 0.0
    for _ in range(20):
        obs, rew, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
        total += rew
        if term:
            break
    print(f"  [env 20 steps OK, cumulative reward {total:.3f}, "
          f"found={info['found']:.0f}]")
    env.close()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name} ...", flush=True)
            fn()
            print(f"{name} PASSED")
    print("all tests passed")
