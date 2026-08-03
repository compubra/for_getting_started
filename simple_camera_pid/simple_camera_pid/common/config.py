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
    recovery_gain: float = 10.0          # Recovery_Steering_Gain
    recovery_limit: float = 0.45         # Recovery_Steering_Limit
    line_found_speed_gain: float = 0.8   # Line_Found_Speed_Gain
    min_search_speed_bias: float = 0.2   # Minimum_Search_Speed bias


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
