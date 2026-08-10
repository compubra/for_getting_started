"""Shared configuration defaults.

Values here mirror the MATLAB model-workspace defaults of
``visual_line_follower_with_debug.slx`` (see
``matlab/runtime/init/configure_turtlebot3_visual_line_follower_paths.m`` and the
``LinePID`` / ``Curve_Speed_Governor`` subsystems), read directly from the
model workspace so the Python port starts from the same numbers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RobotConfig:
    """TurtleBot3 Burger differential-drive parameters."""

    wheel_radius: float = 0.033          # TB3_WheelRadius (m)
    wheel_separation: float = 0.160      # TB3_WheelSeparation (m)
    max_wheel_speed: float = 20.0        # MaxWheelSpeed (rad/s), matches actuator ctrlrange


@dataclass
class ControllerConfig:
    """LinePID + kinematics mixing, translated from the ``Subsystem`` block."""

    kp: float = 5.0                      # Kp_Line
    ki: float = 0.0                      # Ki_Line
    kd: float = 0.35                     # Kd_Line
    n_filter: float = 20.0               # N_Line, derivative filter coefficient
    steering_sign: float = -1.0          # SteeringSign
    base_linear_speed: float = 20.0      # BaseLinearSpeed
    base_speed_scale: float = 0.03       # BaseSpeedScale
    max_angular_speed: float = 1.5       # MaxAngularSpeed, PID output saturation (rad/s)
    ts_control: float = 0.05             # Ts_Control (s), 20 Hz vision/control loop
    # Recovery_Steering_Gain (MATLAB default was 10.0). Lowered to 1.0 after
    # live diagnostics showed the line usually gets lost WHILE angular_cmd is
    # already saturated at max_angular_speed (i.e. the robot is already
    # turning as hard as it can right before losing sight of the line) -- at
    # gain=10, any frozen pre-loss steering_error above recovery_limit/10 =
    # 0.045 (essentially always; real errors were ~0.4-0.5) immediately
    # re-saturates switched_steering at recovery_limit, so "recovery" was in
    # practice always a full-effort turn in whatever direction the line was
    # last seen drifting, regardless of how large that drift actually was --
    # confirmed via a live example where the robot held angular_cmd pinned at
    # -1.5 rad/s (right wheel at 0) for many consecutive ticks after losing
    # the line. At gain=1.0 the recovery response is proportional to the same
    # steering_error scale the already-tuned found-branch uses, only still
    # hitting recovery_limit for genuinely large residual errors instead of
    # nearly any nonzero one.
    #
    # Found and fixed in the simulation copy of this file on 2026-08-07;
    # ported here 2026-08-08. It is a real improvement, NOT a complete fix --
    # the recovery branch was never re-tuned end to end after this change.
    # Deliberately kept alongside (not merged into) lost_speed_freeze_timeout/
    # lost_speed_stop_timeout below, which came from the real robot and which
    # the simulation copy still lacks: the two address different halves of the
    # same incident (this one the steering magnitude, those the forward speed).
    recovery_gain: float = 1.0           # Recovery_Steering_Gain
    recovery_limit: float = 0.45         # Recovery_Steering_Limit
    line_found_speed_gain: float = 0.8   # Line_Found_Speed_Gain
    min_search_speed_bias: float = 0.2   # Minimum_Search_Speed bias
    # Not in the original Simulink model -- added 2026-08-05 after a real-robot
    # incident where repeated brief line losses (each under a second) kept the
    # robot crawling forward at min_search_speed_bias's fixed 20% speed
    # indefinitely while recovery steering (see the docstring above) snapped
    # to +/-recovery_limit each time, producing a sustained side-to-side
    # oscillation across the line rather than a controlled stop. These two
    # taper forward speed toward zero the longer the line has been lost,
    # independent of (and in addition to) vision.py's own steering-only
    # freeze/decay (FREEZE_TIMEOUT/SLOWDOWN_TIMEOUT there): a loss shorter
    # than lost_speed_freeze_timeout still gets the full-speed "search crawl"
    # (matches prior behavior, fine for single-frame noise); past that it
    # ramps down to a full stop by lost_speed_stop_timeout instead of
    # crawling forward blind.
    lost_speed_freeze_timeout: float = 0.5
    lost_speed_stop_timeout: float = 1.5


@dataclass
class CurveSpeedGovernorConfig:
    """Slows the robot down ahead of curves, translated from Curve_Speed_Governor."""

    heading_weight: float = 1.0          # Heading_Weight
    lateral_weight: float = 0.8          # Lateral_Weight
    slowdown_gain: float = -0.9          # Slowdown_Gain
    slowdown_bias: float = 1.0           # Slowdown_Bias
    min_speed_scale: float = 0.1         # Speed_Scale_Limit lower bound
    max_speed_scale: float = 1.0         # Speed_Scale_Limit upper bound


@dataclass
class VisionConfig:
    """Camera-local path generator parameters (LocalPath_* / OriginBot_* workspace vars),
    read from the sliding-window algorithm's export tables
    (``configure_turtlebot3_visual_line_follower_paths.m`` for MuJoCo).

    ``lateral_gain``/``heading_gain``/``curvature_gain`` are platform-independent
    in the current (2026-07-22-unified) algorithm. ``roi_bottom_fraction``/
    ``lookahead_distance`` are not: MuJoCo's defaults here (0.30/0.40) come from
    its camera being flattened to level (see ``camera_geometry.py``), which
    raised the near-visible-ground-distance floor to ~0.31 m and happens to
    reliably keep the line inside that ROI (measured found_rate 100% over a
    25 s run). Gazebo's camera geometry does not bracket its own
    roi_bottom_fraction/lookahead_distance pair the same way -- see
    ``LineFollowerVision.roi_widen_step``'s docstring (``common/vision.py``)
    for the 2026-07-29 root-cause writeup and the ``roi_widen_step``/
    ``roi_widen_max`` fields below that fix it without needing a
    platform-specific static ROI value.
    """

    roi_bottom_fraction: float = 0.30    # LocalPath_ROIFraction (MuJoCo default; Gazebo default is 0.10)
    num_points: int = 30                 # LocalPath_NumPoints
    lookahead_distance: float = 0.40     # LocalPath_LookaheadDistance (MuJoCo default; Gazebo default is 0.20)
    lateral_gain: float = 0.6            # LocalPath_LateralGain
    heading_gain: float = 0.35           # LocalPath_HeadingGain
    curvature_gain: float = 0.04         # LocalPath_CurvatureGain
    min_brightness: float = 70.0         # OriginBot_MinBrightness
    max_saturation: float = 0.30         # OriginBot_MaxSaturation
    min_pixels: float = 30.0             # OriginBot_MinPixels
    error_scale: float = 500.0           # OriginBot_ErrorScale
    # Adaptive-ROI fallback, see LineFollowerVision.roi_widen_step. 0.0 =
    # disabled (restores the pre-2026-07-29 single-attempt behavior).
    roi_widen_step: float = 0.2
    roi_widen_max: float = 0.7


@dataclass
class SafetyFilterConfig:
    """Inference-time safety layer: discrete CBF + box + rate constraints.

    Ported from ``matlab/runtime/control/lf_safety_filter.m`` (2026-08-09);
    defaults mirror ``lf_safety_defaults.m``. See that MATLAB file's header for
    the full CBF derivation -- the short version is that the barrier is built
    on the *lookahead* lateral offset (which the vision module already
    reports as ``lateral_error``) rather than the robot's own lateral offset,
    because the lookahead point has relative degree 1 with respect to omega,
    so the CBF condition collapses to a single affine inequality in omega and
    its projection is one clamp rather than a QP.

    Method reference: Gu et al., "A Review of Safe Reinforcement Learning:
    Methods, Theories, and Applications", IEEE TPAMI 46(12):11216-11235, 2024
    -- section III-B-2 (safety layer / OptLayer) and III-A-2 (CBFs).

    ``speed_scale`` converts the ``v`` this filter is handed into m/s. The two
    deployment families disagree on that unit and the numbers alone cannot
    tell them apart, so it is explicit here rather than inferred:

      * real robot / this Python port -- ``LineFollowerController.command()``
        returns v already in the ``base_linear_speed`` scale, so
        ``speed_scale`` is ``base_speed_scale`` (0.03 by default).
      * MATLAB ``_real.slx`` -- PID_Controller emits ``linear_x`` in m/s
        directly, so its ``signalUnits`` is ``"mps"`` and the scale is 1.0.

    Getting it wrong is silent: the ``v*sin(psi)`` term of the barrier is off
    by 1/base_speed_scale (~33x) and the barrier just becomes too loose or too
    tight without raising anything.
    """

    enable: bool = True                  # False -> compute diagnostics, do not project
    ts: float = 0.05                     # control period (s), Ts_Control
    speed_scale: float = 0.03            # v units -> m/s (see docstring)
    lookahead: float = 0.20              # LocalPath_LookaheadDistance (m)
    # Denominator vision.py normalizes lateral_error by. **vision.py's
    # LATERAL_NORM is the source of truth**; both must agree or the barrier's
    # physical units are wrong.
    lateral_norm: float = 0.55
    lateral_max: float = 0.8             # safe set is |lateral_error| <= this
    cbf_alpha: float = 2.0               # class-K gain (1/s); larger = more aggressive
    cbf_singular_tol: float = 1e-3       # |A| below this -> no CBF bound this tick
    speed_backoff: float = 0.3           # 0..1 slowdown near the barrier (HEURISTIC,
    # explicitly NOT part of the CBF's forward-invariance guarantee)
    max_v: float = 20.0                  # box limit, same units as v in
    max_omega: float = 1.5               # box limit (rad/s)
    max_accel_v: float = 40.0            # rate limit (v units/s)
    max_accel_omega: float = 10.0        # rate limit (rad/s^2)


@dataclass
class LineSearchConfig:
    """Lost-line recovery state machine + odometry-dead-reckoned line memory.

    Ported from ``matlab/runtime/control/lf_line_search.m`` (2026-08-09).

    ``enable`` defaults to **False**. 2026-08-09 closed-loop MuJoCo ablation
    (see ``matlab/runtime/control/README.md``): neither in-place scanning nor
    memory-guided directed recovery beat doing nothing -- the untouched
    baseline (this project's existing ``min_search_speed_bias`` forward crawl)
    reacquired the line and completed a lap, while both recovery strategies
    stopped and never recovered. The bottleneck was measured to be *detection*,
    not the recovery strategy: during a 20 s loss the robot stayed 3.7-4.5 cm
    from the track centreline the whole time and the line was geometrically
    inside the camera's ROI footprint 32.2% of that time, undetected. Turning
    this on before the observation side is fixed only makes things worse.
    """

    enable: bool = False
    ts: float = 0.05
    max_v: float = 20.0                  # used to scale recover_speed_frac
    max_omega: float = 1.5
    hold_time: float = 0.4               # don't interfere for this long after a loss --
    # vision.py is still freezing/extrapolating steering_error over this window
    brake_time: float = 0.3
    scan_rate: float = 0.9               # in-place scan yaw rate (rad/s)
    scan_tolerance: float = 0.05         # swept-angle arrival test (rad)
    scan_timeout: float = 15.0
    dir_deadband: float = 0.02
    # Geometric memory. mem_max_age=8 s: the baseline took ~5.2 s of forward
    # crawl to reacquire, and memory age counts from the last *reliable* frame
    # (~0.5 s before the loss), so the budget has to exceed that with margin.
    mem_max_age: float = 8.0
    mem_min_points: int = 3
    mem_lookahead: float = 0.25          # keep inside the ROI's visible ground band
    recover_gain: float = 1.5            # bearing -> omega (1/s)
    recover_align_cone: float = 0.7853981633974483   # 45 deg
    recover_speed_frac: float = 0.25
    recover_timeout: float = 8.0
    scan_amplitudes: tuple = (0.4363323129985824, 0.8726646259971648, 1.3962634015954636)
    # ~25/50/80 deg. Camera horizontal FOV is ~62 deg, so +/-25 deg already
    # sweeps ~112 deg of ground heading.


@dataclass
class ResidualRewardConfig:
    """Reward/done weights shared by the SAC and PPO residual controllers.

    Matches ``SAC_Q``/``SAC_R``/``SAC_P_Lost``/``SAC_Done_Steps`` (and the
    ``PPO_*`` equivalents) written by ``build_sac_residual_controller.m`` /
    the training scripts.
    """

    q: float = 1.0                       # steering-error penalty
    q_lateral: float = 0.5               # lateral-error penalty
    q_heading: float = 0.5               # heading-error penalty
    r: float = 0.1                       # RL action effort penalty
    p_lost: float = 10.0                 # line-lost penalty
    done_steps: int = 100                # consecutive lost-line steps -> episode done (5 s @ 20 Hz)
    max_delta_omega: float = 1.0         # residual action bound (rad/s)
