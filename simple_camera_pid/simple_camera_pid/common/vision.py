"""Camera-local path generator, shared by the MuJoCo and Gazebo pipelines.

Translated from ``originbot_sliding_window_path_generator.m``: Hough-seed +
sliding-window + ground-quadratic-fit line following. As of 2026-07-22 MATLAB
unified what used to be two separate algorithms -- a MuJoCo-only
skeletonize + greedy-order + pure-pursuit scheme ("scheme A",
``originbot_local_path_generator.m``) and a Gazebo-only band-scan +
quadratic-fit scheme -- into this single file, shared by both platforms and
auto-detecting which one it's looking at via ``originbot_camera_profile.m``.
Both old schemes are now archived under
``matlab/archive/archive_vision_scheme_a/``. This module replaces this
port's former ``vision_mujoco.py`` (scheme A) and ``gazebo/vision_gazebo.py``
(the old Gazebo band-scan), which mirrored that now-superseded split -- in
practice both this port's MuJoCo and Gazebo nodes already shared one vision
instance before this change, exactly like MATLAB's own history.

Pipeline (mirrors the three-technique split in the MATLAB source):

  1. Hough seed: the dominant line segment in the ROI mask seeds the sliding
     window's starting column and column-drift-per-row. Ported via
     ``skimage.transform.probabilistic_hough_line``, which is NOT a bit-exact
     match for MATLAB's ``hough``/``houghpeaks``/``houghlines`` (different
     peak-finding algorithm under the hood) -- this is a behavioral port,
     picking the longest detected segment the same way the MATLAB function
     does, not a numerically-verified one. See ``pid.py``'s docstring for the
     project's convention on flagging this kind of gap explicitly.
  2. Sliding window: from the ROI bottom upward, each window recenters on its
     white-pixel centroid; empty windows extrapolate via the last known
     column drift. This *is* a direct, faithful port of the MATLAB loop.
  3. Quadratic fit: window centroids, IPM-projected to ground coordinates,
     fit to Y = a*X^2 + b*X + c; the lookahead point is read off the
     polynomial at X=lookahead, heading = tangent angle, curvature = Menger
     formula. Also a direct port.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.transform import probabilistic_hough_line

try:
    # Optional: only used for the uint8 fast path in _channel_extremes. Already
    # a vision-layer dependency (requirements-vision.txt, and debug_frame.py
    # imports it unconditionally), but kept guarded so this module -- which is
    # shared with the MuJoCo and training lines -- still imports without it.
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .camera_geometry import CameraParams, pixel_to_ground, turtlebot3_burger_mujoco_camera

MAX_PATH_POINTS = 30
DEBUG_WIDTH = 7 + 2 * MAX_PATH_POINTS  # 7 scalars + 30 [x, y] path-point slots

# Lost-line state-machine thresholds (control period = Ts_Control = 0.05 s, 20 Hz)
CONTROL_PERIOD = 0.05
FREEZE_TIMEOUT = 0.5     # < 0.5 s: momentary occlusion, extrapolate steering trend (see below)
SLOWDOWN_TIMEOUT = 1.5   # 0.5-1.5 s: linearly decay the extrapolated steering toward zero
# > 1.5 s: genuinely lost; steering -> 0, hand off to parking/search upstream

# 2026-08-06: real-hardware curves were confirmed (operator watching the
# camera view directly) to swing the line outside the camera's fixed ~46 deg
# optical FOV before the robot finishes the turn -- a real blind spot, not a
# detection bug (ScalerCrop can't fix this: it can only crop/magnify the
# existing optical FOV, never widen it beyond what the lens physically
# captures), and not something ROI/lookahead tuning alone reaches either
# since the line is genuinely out of frame, not just outside the ROI band.
# During FREEZE_TIMEOUT, steering_error was previously held flat at
# _last_visible_steering; it now extrapolates linearly using the steering
# trend from the last STEERING_TREND_SAMPLES found=True frames instead, so a
# robot that was already turning into the blind spot keeps turning at
# roughly the rate it was, rather than freezing an instantaneous snapshot
# right as the corner starts -- intended to bridge exactly this kind of
# fixed-FOV blind spot, not general sensor noise. UNVALIDATED on real
# hardware yet -- re-tune (sample count, whether to clamp the slope itself
# rather than relying on step()'s existing +/-1.5 clip) from real test data.
STEERING_TREND_SAMPLES = 5  # ~0.25s of history at 20 Hz

# Platform-independent behavioral constants (originbot_sliding_window_path_generator.m's
# "与平台无关的行为常数" block -- shared by both cameras, change here affects both).
STEER_MIX_PATH = 0.65       # path-steering (lookahead prediction) weight
STEER_MIX_CENTROID = 0.35   # centroid-steering (immediate feedback) weight
LATERAL_NORM = 0.55         # lateral-error full-scale (m): robot half-width + margin
CURVATURE_SCALE = 0.20      # curvature-to-steering scale before summing


@dataclass
class SlidingWindowPath:
    x: np.ndarray             # (MAX_PATH_POINTS,) ground X (forward, m)
    y: np.ndarray             # (MAX_PATH_POINTS,) ground Y (right, m)
    valid: np.ndarray         # (MAX_PATH_POINTS,) bool
    valid_count: int
    total_pixels: int
    near_pixel_error: float


@dataclass
class LocalPathResult:
    steering_error: float
    lateral_error: float
    heading_error: float
    confidence: float
    found: bool
    debug: np.ndarray  # (DEBUG_WIDTH,) see pack_local_path_debug for the layout


@dataclass
class _DetectAttempt:
    """One mask+Hough+sliding-window+projection pass at a given ROI depth.
    See ``LineFollowerVision._detect_attempt`` / the ``roi_widen_step``
    field docstring for why there can be more than one of these per frame."""

    total_pixels: int
    valid_count: int
    found: bool
    confidence: float
    win_cols: np.ndarray
    win_rows: np.ndarray
    win_valid: np.ndarray
    x_points: np.ndarray
    y_points: np.ndarray
    valid: np.ndarray
    near_pixel_error: float
    base_col: float
    line_width: float | None
    seed_excess: float = 0.0
    """How far past its ``seed_max_shift_px`` budget this frame's raw seed
    wanted to move, as a fraction of that budget (0.0 = within budget, or the
    limiter is off). Feeds ``confidence``."""


@dataclass
class LineFollowerVision:
    """Stateful port of ``originbot_sliding_window_path_generator`` (persistent vars)."""

    roi_bottom_fraction: float = 0.40
    waypoint_count: int = 30
    min_brightness: float = 70.0
    max_saturation: float = 0.30
    min_pixels: float = 30.0
    error_scale: float = 500.0
    lookahead_distance: float = 0.40
    lateral_gain: float = 0.6
    heading_gain: float = 0.35
    curvature_gain: float = 0.04
    image_height: int = 480
    image_width: int = 640
    camera: CameraParams = field(default_factory=turtlebot3_burger_mujoco_camera)
    roi_widen_step: float = 0.2
    """2026-07-29 adaptive-ROI fallback. Root cause (see
    the since-removed config/gazebo yaml's comment history): a single
    static ``roi_bottom_fraction`` cannot serve every map/camera combination
    -- Gazebo's default (0.30) left the line sitting mostly ABOVE the ROI's
    far edge on gentle-curve maps like ``ellipse`` (measured found_rate
    16.5% over a 25 s run), but widening the ROI to 0.50 to fix that
    regressed sharp-curve maps like ``track_hard`` (100% -> 68.6%, wider ROI
    picks up confusing/adjacent line segments on tight turns). Neither a
    single fixed value nor a linear interpolation between them (0.40 tested
    worse than both neighbors on track_hard: 34.4%) works.

    Fix: try detection at ``roi_bottom_fraction`` first (the precise, narrow
    window -- keeps sharp-curve maps exactly as accurate as before); only if
    that finds nothing, retry once with the ROI widened by this amount
    (clamped to ``roi_widen_max``). Verified same-map, same-duration:
    ellipse found_rate 16.5% -> matches the static-0.50 result (~97%);
    track_hard stays at its original ~100% (the widen path is rarely/never
    reached there, since the narrow attempt already succeeds almost every
    tick). Costs a second full mask/Hough/sliding-window pass only on frames
    where the first one failed -- for MuJoCo (already ~100% found with its
    tuned roi_bottom_fraction=0.30/lookahead_distance=0.40 pair) this should
    essentially never trigger. Set to 0.0 to disable and restore the old
    single-attempt behavior."""
    adaptive_brightness_fraction: float = 0.0
    """0.0 (default) = ``min_brightness`` is used as a fixed floor. Set to
    0.55 to restore MATLAB's ``min(min_brightness, max(40, 0.55*roi_max))``,
    where it acts as a ceiling instead -- see ``_brightness_threshold`` for
    the real-bag measurements behind the default. Ignored entirely when
    ``otsu_min_contrast`` is on."""
    otsu_min_contrast: float = 0.0
    """0.0 (default) = off, use the ``min_brightness`` floor above. When > 0,
    the brightness cut is chosen per frame by Otsu instead, and a frame whose
    two Otsu classes differ in mean by less than this is reported as no
    detection at all. Set 30.0 for this robot's dark-room track: measured
    2026-08-10 over a recorded lap, frames with a line in view had a
    class-mean gap of 123 (median, min 7 for the line-free ones), so 30 sits
    in a wide empty band between the two populations. See ``_otsu_cut``.
    Requires the gate -- Otsu alone will bisect pure noise and turn a
    line-free frame into a confident false detection, which is the exact
    failure this project already crashed on once."""
    roi_widen_max: float = 0.7
    """Ceiling for the widened retry (see ``roi_widen_step``) -- kept well
    below 0.95 so the widened attempt still can't reach up into the region
    that misdetected Gazebo's oversized default ground_plane as the line
    before this project's per-track boundary curbs existed (see
    the since-deleted track_bringup.launch.py's docstring); the curbs make that less likely
    now, but there's no reason to widen past what was actually validated."""
    seed_mode: str = "hough"
    """Which algorithm supplies the sliding window's starting column and
    column-drift-per-row. ``"hough"`` (default) is the MATLAB-derived
    ``_hough_seed`` and leaves every platform exactly as it was;
    ``"run_midpoint"`` selects the Hough-free ``_run_midpoint_seed``.

    Why the switch exists (2026-08-11, measured on this robot's Pi-side
    workload replayed offline). ``_hough_seed`` is **83% of the entire vision
    pass** at every ROI depth -- 24.7 ms of 29.8 at ``roi_bottom_fraction``
    0.3, 45.2 ms of 53.3 at 0.6 (this PC, 150 live frames, relative numbers
    only). ``probabilistic_hough_line`` costs time proportional to the number
    of *lit* mask pixels, and this track's white line is ~275 px wide on a
    640 px frame, so the mask handed to it is a solid blob covering 44-49% of
    the ROI -- roughly 45k-81k points, where PPHT expects a thin edge image.
    OpenCV's own ``HoughLinesP`` on the same filled mask is barely faster
    (38.7 ms vs 44.0 at roi 0.6), which is what identifies the cost as the
    point count rather than the implementation.

    That expense buys almost nothing. Replaying it against a seed taken
    straight from the mask (``_run_midpoint_seed``): over 1196 frames of the
    2026-07-31 driven lap the two agree **exactly** on 94% of frames, with
    max |delta| 0.15 on heading_error and 0.07 on steering_error, and 0
    found-flag mismatches. The reason is structural -- ``_find_nearest_run``
    re-centres the first window on the measured run's own midpoint, so any
    seed landing inside the line's blob produces the same windows.

    On the stationary 150-frame burst the Hough-free seed is also *better
    conditioned*: heading_error sd 0.029 against 0.052 for ``"hough"`` at
    roi 0.6. That is consistent with ``_hough_seed``'s own documented failure
    mode -- a wide line's two long edges both qualify as "the dominant
    segment", so which one is returned can flip frame to frame.

    NOT verified on a driven robot in either mode; both replay results above
    are offline. See ``tools/diagnose_vision_geometry.py`` (``seed`` mode)."""
    seed_max_shift_px: float = 0.0
    """Rate limit on how far the sliding window's seed column may move between
    consecutive frames, in pixels. ``0.0`` (default) disables it and leaves
    every platform bit-identical to the pre-2026-08-11 behaviour.

    Why. Both seed modes choose, per frame, from the CURRENT frame only:
    ``_hough_seed`` takes the longest segment (tie-broken by ``prev_base_col``)
    and ``_run_midpoint_seed`` takes the heaviest column-run. On a single
    continuous line that is unambiguous. At a junction it is not -- the
    transverse tape is part of the same connected component, so the near
    band's heaviest run is the *merged* blob and its midpoint sits between the
    two lines rather than on either. Nothing in either mode bounds how far
    that answer may move from the last one.

    This bounds it. The seed still comes from the image; it is only prevented
    from teleporting. Measured frame-to-frame |delta base_col| with
    ``run_midpoint``: 3.5 px (p95) / 5.1 px (max) over the 150-frame
    stationary burst at roi 0.6, and 8.8 px (p95) over the 1196-frame
    2026-07-31 driven lap. The kinematic bound is the same order: at
    ``max_angular_speed`` 1.0 rad/s and 20 Hz the robot can rotate 0.05 rad
    per frame, which at the seed row (ground x 0.163 m, ~2780 px/m across the
    line's measured 0.130 m / 361 px near-row footprint) moves the line about
    23 px, plus ~3 px for 0.047 m/s of translation.

    Also removes the ``_hough_seed`` restart bistability documented in
    ``camera_geometry.py``'s ``turtlebot3_burger_real_camera`` -- an
    implausible seed can no longer be adopted in one step, whichever mode
    produced it."""
    max_heading_rate: float = 0.0
    """Reject a frame whose ``heading_error`` disagrees with the last accepted
    frame's by more than this many rad/s. ``0.0`` (default) disables the test
    and leaves every platform bit-identical to the pre-2026-08-11 behaviour.

    A rejected frame reports ``found=False``, which routes through the
    existing lost-line path (freeze/extrapolate, then decay, then hand off to
    search) rather than freezing silently. The comparison is against the last
    ACCEPTED frame and is scaled by the elapsed time since it, so a run of
    rejections widens the allowance instead of latching; and the reference is
    dropped when the lost state passes ``SLOWDOWN_TIMEOUT``, so a scene that
    has genuinely changed is re-accepted after at most 1.5 s.

    This is NOT set at the kinematic bound. The kinematic bound is about
    1.2 rad/s (``max_angular_speed`` 1.0, plus v/x_lookahead = 0.047/0.30 for
    translation), but ``heading_error`` is a fitted slope with its own noise
    floor -- 0.034 sd stationary at roi 0.6 -- so the healthy signal already
    shows |delta| p99 of 0.038-0.060 rad per 20 Hz frame, i.e. 0.8-1.2 rad/s,
    on runs that scored found_rate 1.000. The threshold has to be set from the
    measured distribution, above that noise and below the pathological
    population; see ``config/real/real_line_follower.yaml`` for the numbers
    the shipped value came from."""
    flip_vertical: bool = True
    """MuJoCo's raw OpenGL framebuffer (what the Simulink MuJoCo Plant block
    exposes) is bottom-row-first, hence the unconditional flip in the
    original MATLAB function. The official ``mujoco`` Python bindings'
    ``Renderer.render()`` and ROS ``sensor_msgs/Image`` frames are already
    conventional top-row-first, so callers using either (see
    ``mujoco/sim/turtlebot3_mujoco_env.py``, ``gazebo/line_follower_node.py``)
    construct this class with ``flip_vertical=False``, independent of
    ``camera.needs_flip`` (which documents the *MATLAB* raw-framebuffer fact,
    not this Python image source's own orientation)."""

    _last_visible_steering: float = field(default=0.0, init=False, repr=False)
    _last_local_path_debug: np.ndarray = field(default=None, init=False, repr=False)
    _lost_line_time: float = field(default=0.0, init=False, repr=False)
    _last_base_col: float | None = field(default=None, init=False, repr=False)
    _last_line_width: float | None = field(default=None, init=False, repr=False)
    _steering_history: deque = field(
        default_factory=lambda: deque(maxlen=STEERING_TREND_SAMPLES), init=False, repr=False
    )
    _lost_steering_slope: float = field(default=0.0, init=False, repr=False)
    _last_heading_error: float | None = field(default=None, init=False, repr=False)
    _frames_since_accepted: int = field(default=1, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear persistent state (mirrors ``clear originbot_sliding_window_path_generator``)."""
        self._last_visible_steering = 0.0
        self._last_local_path_debug = np.zeros(DEBUG_WIDTH)
        self._lost_line_time = 0.0
        self._last_base_col = None
        self._last_line_width = None
        self._steering_history.clear()
        self._lost_steering_slope = 0.0
        self._last_heading_error = None
        self._frames_since_accepted = 1

    def _brightness_threshold(self, roi_max: float) -> float:
        """Value-channel cut for the line mask.

        ``min_brightness`` is a FLOOR here. That is a deliberate 2026-08-10
        departure from the MATLAB source
        (``originbot_sliding_window_path_generator.m``, which still runs
        ``min(minBrightness, max(40, 0.55*roiMaximum))`` -- port that change
        across before trusting the two to agree again).

        Why: in the MATLAB form ``min_brightness`` is a *ceiling*, so on
        8-bit frames the applied cut can never exceed ``0.55*255 = 140.25``
        and the configured 170 was unreachable -- the real cut was just
        ``0.55*roi_max``, i.e. proportional to whatever the brightest pixel
        in the ROI happened to be. Measured over three real bag runs
        (2026-08-06/07, 434 sampled frames): the applied cut sat between 40
        and 89, the mask covered 23-48% of the ROI on 10-32% of frames, and
        on 27-50% of frames it fell at or below the ROI's own 90th-percentile
        value -- i.e. the bare carpet was passing as line. The failure is
        self-reinforcing: as the line leaves the ROI ``roi_max`` collapses to
        the ground's own brightness, the cut drops with it, and the ground
        becomes line-candidate at exactly the moment a clean ``found=False``
        was wanted. That is the "confidently latched onto an unrelated white
        object" mode in ``real_line_follower.yaml``'s kp log.

        A fixed floor is only safe once the camera's exposure is locked
        (``robot_bringup.launch.py``'s ``exposure_time``/``analogue_gain``,
        added the same day) -- under auto-exposure a fixed cut blinds the
        robot as soon as AE stops down. The two changes are one fix.

        Set ``adaptive_brightness_fraction`` to 0.55 to restore the MATLAB
        expression exactly.

        Note this is a no-op for any already-tuned bright-scene deployment:
        whenever ``fraction*roi_max >= min_brightness`` the old expression
        already returned ``min_brightness``. For MuJoCo (min_brightness=70,
        rendered white line at/near 255) that holds on every frame, so its
        threshold is 70 before and after.
        """
        fraction = float(self.adaptive_brightness_fraction)
        if fraction > 0.0:
            return min(self.min_brightness, max(40.0, fraction * roi_max))
        return float(self.min_brightness)

    def _estimate_steering_slope(self) -> float:
        """Rate of change of steering_error (per second) over the last
        ``STEERING_TREND_SAMPLES`` found=True frames, via simple
        least-squares -- used to extrapolate a trend into a fresh line loss
        instead of freezing an instantaneous snapshot (see the module-level
        note by STEERING_TREND_SAMPLES for why)."""
        n = len(self._steering_history)
        if n < 2:
            return 0.0
        xs = np.arange(n, dtype=float) * CONTROL_PERIOD
        ys = np.array(self._steering_history, dtype=float)
        x_mean, y_mean = xs.mean(), ys.mean()
        denom = float(((xs - x_mean) ** 2).sum())
        if denom < np.finfo(float).eps:
            return 0.0
        return float(((xs - x_mean) * (ys - y_mean)).sum() / denom)

    def step(self, rgb: np.ndarray) -> LocalPathResult:
        """Process one camera frame.

        Parameters
        ----------
        rgb : (image_height, image_width, 3) array, any real dtype (uint8 or
            float). MuJoCo's raw framebuffer is bottom-row-first
            (upside-down, left-right correct); this method flips it upright
            internally when ``flip_vertical`` is set, exactly like the
            MATLAB block.
        """
        roi_bottom_fraction = float(np.clip(self.roi_bottom_fraction, 0.05, 0.95))
        window_count = int(np.clip(round(self.waypoint_count), 2, MAX_PATH_POINTS))
        error_scale = max(np.finfo(float).eps, float(self.error_scale))

        # Deliberately NOT converted to float64 here. This used to be
        # `np.asarray(rgb, dtype=np.float64)`, which materialised a 7.4 MB
        # copy of the whole frame (3.0 ms on the robot's Pi) when only the
        # bottom `roi_bottom_fraction` of it is ever read, by _detect_mask.
        # That method now does its own arithmetic in the incoming dtype --
        # see its docstring for why the result is unchanged.
        rgb = np.asarray(rgb)
        if rgb.shape != (self.image_height, self.image_width, 3):
            raise ValueError(
                f"expected an ({self.image_height}, {self.image_width}, 3) image, got {rgb.shape}"
            )
        if self.flip_vertical:
            rgb = np.flip(rgb, axis=0)

        # 1-indexed image center column, matching pixel_to_ground's cx convention.
        center_x = 0.5 * (self.image_width + 1)

        # Narrow attempt first (precise; matches every map's previous
        # behavior exactly when it succeeds). Only if that finds nothing,
        # retry once with the ROI widened -- see roi_widen_step's docstring.
        attempt = self._detect_attempt(rgb, roi_bottom_fraction, window_count, center_x)
        widen_step = max(0.0, float(self.roi_widen_step))
        if not attempt.found and widen_step > 0.0:
            widened_fraction = float(np.clip(
                roi_bottom_fraction + widen_step, 0.05, min(0.95, float(self.roi_widen_max))
            ))
            if widened_fraction > roi_bottom_fraction:
                wide_attempt = self._detect_attempt(rgb, widened_fraction, window_count, center_x)
                if wide_attempt.found:
                    attempt = wide_attempt

        total_pixels = attempt.total_pixels
        win_cols, win_rows, win_valid = attempt.win_cols, attempt.win_rows, attempt.win_valid
        x_points, y_points, valid, valid_count = (
            attempt.x_points, attempt.y_points, attempt.valid, attempt.valid_count
        )
        near_pixel_error = attempt.near_pixel_error
        found = attempt.found
        confidence = attempt.confidence

        lookahead_x, lookahead_y, heading_error_raw, curvature, coefficients = _fit_poly_lookahead(
            x_points, y_points, valid, self.lookahead_distance
        )

        lateral_error = float(np.clip(lookahead_y / LATERAL_NORM, -1.0, 1.0))
        heading_error = float(np.clip(heading_error_raw, -1.0, 1.0))
        curvature_for_control = float(np.clip(CURVATURE_SCALE * curvature, -1.0, 1.0))
        centroid_steering = near_pixel_error / error_scale
        path_steering = (
            self.lateral_gain * lateral_error
            + self.heading_gain * heading_error
            + self.curvature_gain * curvature_for_control
        )
        current_steering_error = float(
            np.clip(STEER_MIX_PATH * path_steering + STEER_MIX_CENTROID * centroid_steering, -1.5, 1.5)
        )

        local_path = SlidingWindowPath(x_points, y_points, valid, valid_count, total_pixels,
                                        near_pixel_error)
        local_path_debug = pack_local_path_debug(local_path, lookahead_x, lookahead_y, curvature,
                                                   coefficients)

        # Forward span the fit actually had to work with. Same quantity
        # tools/diagnose_vision_geometry.py's "fit" mode reports as x-span.
        x_fit = x_points[valid]
        fit_span = float(x_fit.max() - x_fit.min()) if x_fit.size > 1 else 0.0

        found, confidence = self._apply_plausibility(
            found, confidence, heading_error, attempt.seed_excess, fit_span,
        )

        if found:
            self._lost_line_time = 0.0
            steering_error = current_steering_error
            self._last_visible_steering = steering_error
            self._last_local_path_debug = local_path_debug
            self._last_base_col = attempt.base_col
            self._last_heading_error = heading_error
            self._frames_since_accepted = 1
            if attempt.line_width is not None:
                self._last_line_width = attempt.line_width
            self._steering_history.append(steering_error)
        else:
            self._frames_since_accepted += 1
            just_lost = self._lost_line_time == 0.0
            self._lost_line_time += CONTROL_PERIOD
            lateral_error = 0.0
            heading_error = 0.0
            local_path_debug = self._last_local_path_debug.copy()
            local_path_debug[0] = 0.0

            if just_lost:
                # Freeze the trend now, at the instant of loss -- not
                # recomputed frame-to-frame while lost, since there's no new
                # data to update it with until the line is seen again.
                self._lost_steering_slope = self._estimate_steering_slope()

            if self._lost_line_time <= FREEZE_TIMEOUT:
                steering_error = float(np.clip(
                    self._last_visible_steering
                    + self._lost_steering_slope * self._lost_line_time,
                    -1.5, 1.5,
                ))
            elif self._lost_line_time <= SLOWDOWN_TIMEOUT:
                # Decay from where the extrapolation above left off at
                # exactly t=FREEZE_TIMEOUT, not from the raw last-visible
                # value, so steering_error stays continuous across the
                # freeze/slowdown boundary instead of jumping.
                extrapolated_at_freeze_end = float(np.clip(
                    self._last_visible_steering + self._lost_steering_slope * FREEZE_TIMEOUT,
                    -1.5, 1.5,
                ))
                decay = 1.0 - (self._lost_line_time - FREEZE_TIMEOUT) / max(
                    np.finfo(float).eps, SLOWDOWN_TIMEOUT - FREEZE_TIMEOUT
                )
                steering_error = extrapolated_at_freeze_end * decay
            else:
                steering_error = 0.0
                self._last_visible_steering = 0.0
                # Stale column prior is worse than none once the line's been
                # gone this long -- let the next sighting pick fresh (longest
                # segment) rather than anchoring to a position that may no
                # longer be anywhere near the line.
                self._last_base_col = None
                self._last_line_width = None
                self._steering_history.clear()
                self._lost_steering_slope = 0.0
                # Same argument for the heading reference the plausibility
                # test compares against: after this long the scene may
                # genuinely be a different one, and refusing to believe it
                # would be a permanent latch rather than a rejection. This is
                # what bounds a run of rejections at SLOWDOWN_TIMEOUT.
                self._last_heading_error = None

        return LocalPathResult(
            steering_error=steering_error,
            lateral_error=lateral_error,
            heading_error=heading_error,
            confidence=confidence,
            found=found,
            debug=local_path_debug,
        )

    # ------------------------------------------------------------------
    def _apply_plausibility(
        self, found: bool, confidence: float, heading_error: float, seed_excess: float,
        fit_span: float,
    ) -> tuple[bool, float]:
        """Decide whether to believe this frame, and turn ``confidence`` into
        a number that says how much of the answer came from the image.

        Returns ``(found, confidence)``, both unchanged when neither
        ``seed_max_shift_px`` nor ``max_heading_rate`` is set -- which is the
        default, so every existing deployment is bit-identical.

        What ``confidence`` meant before, and why that was useless here.
        ``_detect_attempt`` sets it to
        ``min(1, total_pixels / (waypoint_count * min_pixels))``, i.e. lit
        pixels against 900. On this robot at roi 0.6 the line alone covers
        ~80 000 ROI pixels (measured: 44% of the ROI), so the ratio is ~90x
        its own ceiling and the published value is exactly 1.00 on every
        frame -- including all 338 found frames of the 2026-08-11 junction
        run, whose ``heading_error`` was swinging its full +/-1.0 range. A
        saturated constant cannot tell a consumer anything.

        What it means when the continuity checks are on: the product of three
        bounded factors, each 1.0 for a frame with nothing wrong with it.

          pixels       the old term, unchanged. Still saturates on this
                       camera; kept because it is the one that catches a
                       nearly-empty ROI, which is a different failure.
          conditioning the valid points' forward span divided by
                       ``lookahead_distance``, capped at 1. Below 1 the
                       polynomial is being *extrapolated* to the lookahead
                       point over a baseline shorter than the distance being
                       read off, which is precisely the regime commit
                       45a4467 had to retract a conclusion from: at roi 0.3 /
                       lookahead 0.20 the span is 0.071 m, this term is 0.36,
                       and ``heading_error``'s measured sd is 0.678. At
                       roi 0.6 / lookahead 0.30 the span is 0.448 m and the
                       term is a flat 1.0 -- so on a healthy frame it does
                       not move, and it is the *dynamic* collapse of the
                       span that it catches.
          agreement    how much of this frame's per-frame motion budget the
                       estimate consumed -- 1.0 for a frame consistent with
                       the last accepted one, falling linearly to 0.0 at the
                       rejection threshold.

        The two continuity mechanisms are deliberately NOT symmetric:

        - A rate-limited *seed* is still a usable frame. The windows restart
          from the held column and re-measure the line from the image, so the
          answer can be perfectly good; what has changed is that part of it is
          carried by the prior. That shows up as ``1 / (1 + seed_excess)``,
          which falls off but never reaches 0, and never forces a rejection.
        - An implausible *heading step* is not usable. Nothing downstream can
          repair it, so it rejects: ``found=False``, taking the existing
          lost-line path (extrapolate, decay, hand off to search). It is not
          silently frozen, and the caller cannot mistake it for a reading.
        """
        seed_on = float(self.seed_max_shift_px) > 0.0
        rate_on = float(self.max_heading_rate) > 0.0
        if not (seed_on or rate_on):
            return found, confidence

        # Fraction of the heading allowance used; >1 means the frame
        # contradicts the last accepted one by more than motion permits.
        head_used = 0.0
        if rate_on and found and self._last_heading_error is not None:
            allowance = (float(self.max_heading_rate)
                         * CONTROL_PERIOD * max(1, self._frames_since_accepted))
            head_used = abs(heading_error - self._last_heading_error) / max(
                np.finfo(float).eps, allowance)

        agreement = float(np.clip(1.0 - head_used, 0.0, 1.0)) / (1.0 + max(0.0, seed_excess))
        conditioning = min(1.0, fit_span / max(np.finfo(float).eps,
                                               float(self.lookahead_distance)))
        confidence = float(np.clip(confidence * conditioning * agreement, 0.0, 1.0))
        if head_used > 1.0:
            found = False
        return found, confidence

    def _detect_attempt(
        self, rgb: np.ndarray, roi_bottom_fraction: float, window_count: int, center_x: float,
    ) -> _DetectAttempt:
        """One full mask -> Hough-seed -> sliding-window -> ground-projection
        pass at a given ROI depth. Factored out of ``step`` so it can be
        called a second time with a wider ROI when the first (narrow, more
        precise) attempt finds nothing -- see ``roi_widen_step``."""
        path_top = int(np.clip(np.floor((1 - roi_bottom_fraction) * self.image_height), 0,
                                self.image_height - 1))
        path_bottom = self.image_height  # exclusive upper bound for slicing

        mask, total_pixels = self._detect_mask(rgb, path_top, path_bottom)

        seed_limit = max(0.0, float(self.seed_max_shift_px))
        prev_base_col = self._last_base_col if seed_limit > 0.0 else None

        if self.seed_mode == "run_midpoint":
            base_col, drift_per_row, hough_found = _run_midpoint_seed(
                mask, self.camera.max_drift_per_row, prev_base_col=prev_base_col,
            )
        else:
            base_col, drift_per_row, hough_found = _hough_seed(
                mask, self.camera.hough_min_length, self.camera.hough_fill_gap,
                self.camera.max_drift_per_row, prev_base_col=self._last_base_col,
            )

        # How far this frame's seed wanted to move, before the rate limit
        # below is applied -- 0.0 when the limit is off or there is no prior.
        # Reported (normalised) as ``seed_excess`` so ``confidence`` can say
        # how much of the estimate is the image and how much is the prior.
        seed_excess = 0.0
        if hough_found and seed_limit > 0.0 and self._last_base_col is not None:
            shift = base_col - self._last_base_col
            seed_excess = max(0.0, abs(shift) / seed_limit - 1.0)
            base_col = float(np.clip(
                self._last_base_col + np.clip(shift, -seed_limit, seed_limit),
                0.0, self.image_width - 1.0,
            ))

        if not hough_found and total_pixels > 0:
            near_start = max(0, mask.shape[0] - 1 - int(np.floor(mask.shape[0] / 3)))
            near_band = mask[near_start:, :]
            column_hist = near_band.sum(axis=0)
            peak_col = int(np.argmax(column_hist))
            peak_val = column_hist[peak_col]
            base_col = float(peak_col) if peak_val > 0 else 0.5 * (self.image_width - 1)
            drift_per_row = 0.0

        win_cols, win_rows, win_valid, near_pixel_error, line_width = _slide_windows(
            mask, path_top, base_col, drift_per_row, window_count,
            self.camera.window_half_width, self.camera.min_window_pixels, center_x,
            adaptive_window_gain=self.camera.adaptive_window_gain,
            adaptive_radius_min=self.camera.adaptive_radius_min,
            adaptive_radius_max=(self.camera.adaptive_radius_max_fraction
                                 * self.camera.image_width),
            max_drift_per_row=self.camera.max_drift_per_row,
            prev_line_width=self._last_line_width,
        )

        x_points, y_points, valid, valid_count = self._project_windows(
            win_cols, win_rows, win_valid, window_count,
        )

        found = valid_count >= 1 and total_pixels >= self.min_pixels
        confidence = min(1.0, total_pixels / max(1.0, window_count * self.min_pixels))

        return _DetectAttempt(
            total_pixels=total_pixels, valid_count=valid_count, found=found, confidence=confidence,
            win_cols=win_cols, win_rows=win_rows, win_valid=win_valid,
            x_points=x_points, y_points=y_points, valid=valid, near_pixel_error=near_pixel_error,
            base_col=base_col, line_width=line_width, seed_excess=seed_excess,
        )

    def _detect_mask(self, rgb: np.ndarray, path_top: int, path_bottom: int) -> tuple[np.ndarray, int]:
        """Brightness + low-saturation mask, filtered to its largest
        connected component. ``total_pixels`` is measured *before* that
        filter (matches the MATLAB source: it reflects all candidate-line
        pixels, not just the largest blob).

        Arithmetic is done in the ROI's incoming dtype rather than a float64
        copy of the whole frame, and the saturation test is rearranged from
        ``(v - min) / max(v, 1) <= max_saturation`` into the equivalent
        ``(v - min) <= max_saturation * max(v, 1)`` to drop the per-pixel
        divide. ``v >= min`` holds elementwise by construction, so the
        subtraction cannot underflow an unsigned dtype. Measured on the Pi:
        30.1 ms -> 12.0 ms per frame for this block, with **0 differing
        pixels** over 40 real bag frames (~3.7 M pixels) -- an empirical
        check on real data, not a proof of bit-exactness for every possible
        input.
        """
        roi = rgb[path_top:path_bottom, :, :]
        value, minimum_channel = _channel_extremes(roi)
        roi_max = value.max() if value.size else 0.0

        if self.otsu_min_contrast > 0.0:
            brightness_threshold, contrast = _otsu_cut(value)
            if brightness_threshold is None or contrast < self.otsu_min_contrast:
                # Either no split was computable, or the two classes are too
                # close together to be line-vs-ground -- Otsu will happily
                # split pure sensor noise into "bright" and "dark", so
                # without this gate a line-free frame becomes a confident
                # false detection. Report nothing and let the caller's
                # found=False path run.
                return np.zeros(value.shape, dtype=bool), 0
        else:
            brightness_threshold = self._brightness_threshold(roi_max)

        mask = ((value >= brightness_threshold)
                & ((value - minimum_channel) <= self.max_saturation * np.maximum(value, 1)))
        total_pixels = int(mask.sum())

        if total_pixels > 0:
            mask = _largest_component(mask)

        return mask, total_pixels

    def _project_windows(
        self, win_cols: np.ndarray, win_rows: np.ndarray, win_valid: np.ndarray, window_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        x_points = np.zeros(MAX_PATH_POINTS)
        y_points = np.zeros(MAX_PATH_POINTS)
        valid = np.zeros(MAX_PATH_POINTS, dtype=bool)

        # One vectorised projection instead of a per-window call. This loop
        # used to call pixel_to_ground() once per valid window (up to 30
        # times), and each call rebuilds the mount rotation via two
        # scipy Rotation.from_euler() constructions -- 10.0 ms per frame on
        # the Pi, against 0.33 ms for a single call on all 30 at once, with
        # a max absolute difference of exactly 0.0 (checked over 30 random
        # window positions). The compaction order below is unchanged: valid
        # windows in ascending k, NaN projections dropped.
        selected = np.flatnonzero(win_valid[:window_count])
        if selected.size:
            # 0-indexed -> MATLAB's 1-indexed pixel convention
            xg, yg = pixel_to_ground(win_cols[selected] + 1.0, win_rows[selected] + 1.0,
                                      self.camera)
            finite = ~(np.isnan(xg) | np.isnan(yg))
            write_count = int(finite.sum())
            x_points[:write_count] = xg[finite]
            y_points[:write_count] = yg[finite]
            valid[:write_count] = True
        return x_points, y_points, valid, int(valid.sum())


def _otsu_cut(value: np.ndarray) -> tuple[float | None, float]:
    """Per-frame Otsu split of the ROI value channel.

    Returns ``(threshold, contrast)`` where ``contrast`` is the gap between
    the two class means -- the caller uses it to decide whether the split is
    a real line-vs-ground boundary or just noise being bisected. Returns
    ``(None, 0.0)`` when no usable split exists (empty ROI, or one class too
    small to have a meaningful mean).

    Why this exists: a single fixed ``min_brightness`` has to be one
    compromise value for the whole track, and a dark room lights the track
    unevenly enough that no such value is clean everywhere. Measured over a
    whole recorded lap (2026-08-10, 407 sampled frames, lights off): a fixed
    floor at its best setting (100-115) still threw away >10% of the line's
    pixels on 19% of frames, against a 15% floor set by the frames that
    genuinely had no line in view -- i.e. ~4% real misses. Per-frame Otsu
    with this gate scored exactly 15%, i.e. no real misses, at the same zero
    ground leakage.
    """
    if value.size == 0:
        return None, 0.0
    vu = value if value.dtype == np.uint8 else np.clip(value, 0, 255).astype(np.uint8)
    if cv2 is not None:
        threshold, _ = cv2.threshold(vu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Same criterion as cv2's, computed from the 256-bin histogram.
        hist = np.bincount(vu.ravel(), minlength=256).astype(np.float64)
        total = hist.sum()
        omega = np.cumsum(hist) / total
        mu = np.cumsum(hist * np.arange(256)) / total
        mu_t = mu[-1]
        denom = omega * (1.0 - omega)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_b = (mu_t * omega - mu) ** 2 / denom
        sigma_b[~np.isfinite(sigma_b)] = -1.0
        threshold = float(np.argmax(sigma_b))

    high = vu >= threshold
    n_high = int(high.sum())
    if n_high == 0 or n_high == vu.size:
        return None, 0.0
    contrast = float(vu[high].mean()) - float(vu[~high].mean())
    return float(threshold), contrast


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """The 8-connected component of ``mask`` with the most pixels.

    Same rule as the ``scipy.ndimage.label`` + ``ndimage.sum`` + ``argmax``
    it replaces. OpenCV gets the component sizes out of the same pass that
    labels them, so it avoids the extra ``ndimage.sum`` reduction over the
    whole ROI: 4.0 -> 0.9 ms per frame at ``roi_bottom_fraction`` 0.6 on this
    PC. Falls back to scipy when OpenCV is missing, which keeps this module
    importable for the MuJoCo/training lines that do not depend on it.

    The explicit tie-break is NOT decoration. ``scipy.ndimage.label`` numbers
    components in raster order of their first pixel (checked over 500 random
    masks), so ``argmax`` silently resolved equal-size components in favour
    of the topmost-leftmost one. OpenCV's numbering does not match, and a
    naive ``argmax`` over its ``CC_STAT_AREA`` disagreed with the old code on
    5 of 3988 randomised masks -- always and only where two components had
    *exactly* equal size. Reproducing the old preference explicitly costs one
    ``argmax`` per tied component, which is nothing because ties are rare,
    and it is the difference between a measured no-op and a silent change of
    which blob the robot follows when the scene is ambiguous.
    """
    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.ascontiguousarray(mask, dtype=np.uint8), connectivity=8,
        )
        if count > 2:  # 0 is background, so >2 means more than one component
            areas = stats[1:, cv2.CC_STAT_AREA]
            tied = np.flatnonzero(areas == areas.max()) + 1
            if tied.size > 1:
                flat = labels.ravel()
                largest_label = int(min(tied, key=lambda k: int(np.argmax(flat == k))))
            else:
                largest_label = int(tied[0])
            return labels == largest_label
        return mask

    labels, num_components = ndimage.label(mask, structure=np.ones((3, 3)))
    if num_components > 1:
        sizes = ndimage.sum(mask, labels, index=np.arange(1, num_components + 1))
        return labels == int(np.argmax(sizes)) + 1
    return mask


RUN_MASS_GATE = 0.5
"""Mass gate for ``_widest_run_midpoint``'s continuity tie-break: a run must
carry at least this fraction of the heaviest run's pixels before being
preferred for being nearer the prior. Serves the same purpose as
``_hough_seed``'s 0.8 length gate -- a faint smear must not win just because
it happens to sit closer to where the line was."""


def _widest_run_midpoint(band: np.ndarray, prev_col: float | None = None) -> float | None:
    """Column midpoint of the heaviest contiguous run of lit columns in a
    horizontal band of the mask, or ``None`` if the band is empty.

    "Heaviest" (most lit pixels) rather than "widest in columns" so a long
    thin smear cannot outvote the line itself. The midpoint of the measured
    run is used rather than a pixel-mass centroid for the same reason
    ``_find_nearest_run`` does -- see its docstring.

    ``prev_col``, when given, switches the choice among runs carrying at least
    ``RUN_MASS_GATE`` of the heaviest one's pixels from "heaviest" to "nearest
    to ``prev_col``". This mirrors ``_hough_seed``'s length-gate-then-prior
    rule and, like it, falls back to the pure heaviest-run choice when there
    is no prior. With ``prev_col=None`` this function is unchanged, including
    its leftmost-wins tie-break (``mass > best_mass``, scanning left to
    right).

    It matters much less here than the rate limit in ``_detect_attempt``: at
    a four-way junction the transverse tape is *connected* to the followed
    line, so ``_largest_component`` keeps them as one blob and the near band
    has a single run spanning both. This tie-break only reaches the cases
    where they are separate runs -- an approach where the junction's far side
    is disconnected in the mask, or a second line beside the path."""
    col_mass = band.sum(axis=0)
    lit = np.flatnonzero(col_mass > 0)
    if lit.size == 0:
        return None
    gaps = np.flatnonzero(np.diff(lit) > 1)
    run_starts = np.concatenate(([0], gaps + 1))
    run_ends = np.concatenate((gaps, [lit.size - 1]))
    runs = [(0.5 * (float(lit[s]) + float(lit[e])),
             float(col_mass[lit[s]:lit[e] + 1].sum()))
            for s, e in zip(run_starts, run_ends)]

    best_mass = max(mass for _, mass in runs)
    if prev_col is None or len(runs) == 1:
        for mid, mass in runs:      # leftmost wins ties, as before
            if mass >= best_mass:
                return mid
        return None
    gate = RUN_MASS_GATE * best_mass
    return min((r for r in runs if r[1] >= gate), key=lambda r: abs(r[0] - prev_col))[0]


def _run_midpoint_seed(
    mask: np.ndarray, max_drift_per_row: float, prev_base_col: float | None = None,
) -> tuple[float, float, bool]:
    """Hough-free replacement for :func:`_hough_seed`, selected by
    ``LineFollowerVision.seed_mode='run_midpoint'``. Same return contract:
    ``(base_col, drift_per_row, found)``, 0-indexed, ``base_col``
    extrapolated to the mask's bottom row.

    The line's position is read directly off the mask instead of being
    inferred from line segments: the heaviest column-run's midpoint in the
    near third gives the column, and the shift of that same measurement
    between the far third and the near third gives the lean. Cost is two
    column sums over a third of the ROI each (~0.1 ms) against ~45 ms for
    ``probabilistic_hough_line`` on the filled mask at roi 0.6.

    ``prev_base_col`` (2026-08-11) breaks the choice among comparably-heavy
    runs in favour of temporal continuity. This function originally took no
    prior at all, on the argument that ``_hough_seed`` only needs one to
    disambiguate the line's two long edges and a run midpoint has no such
    ambiguity. That argument was about a *single* line, and it does not
    survive a junction: with a transverse tape in the ROI there are genuinely
    several candidate runs, or one merged run whose midpoint is on neither
    line. Passing the prior does not make this seed carry a stale column
    forward on its own -- ``_detect_attempt`` drops the prior once the line
    has been lost past ``SLOWDOWN_TIMEOUT``, exactly as ``_hough_seed``'s
    prior already was.

    See ``seed_mode``'s docstring for what was measured, and for the fact
    that neither mode has been checked on a driven robot."""
    height, width = mask.shape
    band = max(1, height // 3)
    near_mid = _widest_run_midpoint(mask[height - band:, :], prev_base_col)
    if near_mid is None:
        return 0.0, 0.0, False

    # The far band's prior is the near band's own answer, not prev_base_col:
    # the line continues upward from the near band, so that is where the far
    # run should be looked for. Only used when there is a prior at all.
    far_prior = near_mid if prev_base_col is not None else None
    far_mid = _widest_run_midpoint(mask[:band, :], far_prior) if height > band else None
    if far_mid is None:
        return float(np.clip(near_mid, 0.0, width - 1)), 0.0, True

    # Band centre rows, in mask-local coordinates. d_row > 0 by construction
    # (the near band is lower in the image), which matches _hough_seed's
    # convention of drift measured toward increasing row.
    near_row = (height - 1) - 0.5 * (band - 1)
    far_row = 0.5 * (band - 1)
    drift = float(np.clip((near_mid - far_mid) / (near_row - far_row),
                          -max_drift_per_row, max_drift_per_row))
    base_col = near_mid + ((height - 1) - near_row) * drift
    return float(np.clip(base_col, 0.0, width - 1)), drift, True


def _channel_extremes(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel max and min across the three colour channels.

    ``roi.max(axis=2)``/``roi.min(axis=2)`` are surprisingly expensive on the
    robot's Pi (a reduction along the smallest, non-contiguous axis): 11.7 ms
    per frame for the pair at 640x144, against 1.8 ms for OpenCV's SIMD
    two-argument max/min on the split channels. Verified identical output
    (0 differing pixels over 12 real bag frames). Falls back to NumPy when
    OpenCV is missing or the frame is not uint8 -- ``cv2.max``/``cv2.min``
    do not accept float64, and ``step()`` documents float input as allowed.
    """
    if cv2 is not None and roi.dtype == np.uint8 and roi.ndim == 3 and roi.shape[2] == 3:
        r, g, b = cv2.split(roi)
        return cv2.max(cv2.max(r, g), b), cv2.min(cv2.min(r, g), b)
    return roi.max(axis=2), roi.min(axis=2)


def _hough_seed(
    mask: np.ndarray, hough_min_length: float, hough_fill_gap: float, max_drift_per_row: float,
    prev_base_col: float | None = None,
) -> tuple[float, float, bool]:
    """Behavioral (not bit-exact) port of MATLAB's ``houghSeed``: finds the
    dominant line segment in the ROI mask via
    ``skimage.transform.probabilistic_hough_line`` and extrapolates it to the
    mask's bottom row, returning a 0-indexed seed column and column-drift-per-row
    for :func:`_slide_windows`.

    ``prev_base_col``, when given, breaks ties among comparably-long
    candidate segments in favor of temporal continuity -- see the note below
    on why this matters for a real (non-simulated) camera.
    """
    if not mask.any():
        return 0.0, 0.0, False

    # probabilistic_hough_line randomly samples edge pixels internally; a fixed
    # seed keeps this deterministic frame-to-frame (matters for reproducible
    # simulation/RL rollouts -- the same camera frame must always seed the
    # same base_col/drift_per_row). The keyword was renamed seed->rng in a
    # recent scikit-image release; requirements-vision.txt only pins
    # scikit-image>=0.21, so support both.
    hough_kwargs = dict(
        threshold=10,
        line_length=max(2, int(round(hough_min_length))),
        line_gap=max(1, int(round(hough_fill_gap))),
    )
    try:
        segments = probabilistic_hough_line(mask, rng=0, **hough_kwargs)
    except TypeError:
        segments = probabilistic_hough_line(mask, seed=0, **hough_kwargs)
    if not segments:
        return 0.0, 0.0, False

    bottom_row = mask.shape[0] - 1

    def _extrapolate(p0: tuple[float, float], p1: tuple[float, float]) -> tuple[float, float]:
        col, row = p0
        d_col = p1[0] - col
        d_row = p1[1] - row
        if d_row < 0:  # normalize direction to point toward larger row (down/near)
            d_col, d_row = -d_col, -d_row
        if abs(d_row) < 1e-6:
            base = col + 0.5 * d_col
            drift = 0.0
        else:
            drift = float(np.clip(d_col / d_row, -max_drift_per_row, max_drift_per_row))
            base = col + (bottom_row - row) * drift
        return float(np.clip(base, 0.0, mask.shape[1] - 1)), drift

    # A real camera's line is several pixels wide, so its two long edges
    # (left/right) both routinely qualify as "the dominant segment" --
    # single-frame sensor/JPEG noise near either edge can flip which one
    # probabilistic_hough_line happens to return as longest, even with a
    # fixed rng seed (the seed only makes a GIVEN mask's sampling
    # deterministic; it does not make near-identical masks agree with each
    # other). Picking by raw length alone therefore lets the seed column
    # swing across the full line width frame-to-frame (confirmed via
    # synthetic 0.2%-pixel-flip perturbation: base_col jumped ~470 -> ~304,
    # sign of drift_per_row included, on an otherwise-static scene).
    #
    # Fix: among candidates within 80% of the longest one's length (so short
    # spurious segments still can't win), prefer whichever extrapolates
    # closest to last frame's converged base_col -- the robot/line can't
    # jump most of the image width in one ~50-60 ms control tick. Falls back
    # to pure longest-segment selection when there's no prior (first frame,
    # or the line was just reacquired after being lost).
    candidates = [
        (math.hypot(x1 - x0, y1 - y0), (float(x0), float(y0)), (float(x1), float(y1)))
        for (x0, y0), (x1, y1) in segments
    ]
    best_len = max(length for length, _, _ in candidates)
    length_gate = 0.8 * best_len
    survivors = [c for c in candidates if c[0] >= length_gate]

    if prev_base_col is None or len(survivors) == 1:
        _, best_p0, best_p1 = max(survivors, key=lambda c: c[0])
    else:
        def _distance_to_prior(c: tuple[float, tuple[float, float], tuple[float, float]]) -> float:
            base, _ = _extrapolate(c[1], c[2])
            return abs(base - prev_base_col)

        _, best_p0, best_p1 = min(survivors, key=_distance_to_prior)

    base_col, drift_per_row = _extrapolate(best_p0, best_p1)
    return base_col, drift_per_row, True


def _find_nearest_run(
    col_mass: np.ndarray, current_col: float, search_radius: float, min_pixels: float,
) -> tuple[int, int, float] | None:
    """Find the contiguous run of lit columns within +/-``search_radius`` of
    ``current_col`` whose total pixel count meets ``min_pixels``, preferring
    whichever run's midpoint sits closest to ``current_col`` (temporal/
    spatial continuity -- guards against snapping to an unrelated blob
    elsewhere in the search band). Returns 0-indexed ``(left, right,
    pixel_count)`` of the run, or ``None`` if nothing qualifies.

    2026-08-05: replaces the previous fixed-``half_width``-window pixel-mass
    centroid (``window_half_width`` was 20px -- a 40px-wide search box --
    while the real track's tape line measured 130-161px wide on camera: the
    window could only ever see a slice of the line, so its centroid tracked
    wherever that slice happened to land relative to the line rather than
    the line's actual center, drifting between the two edges frame to frame
    -- confirmed via vision_debug_node showing the tracked centroid pinned
    to one edge of the visibly-wider line). Using the *measured* run's own
    midpoint instead of a pixel-mass centroid inside an undersized window
    adapts to whatever width the line actually is at each row, without a
    hand-tuned pixel constant. ``search_radius`` (``window_half_width``) is
    now just a generous gate against picking up a far-away unrelated bright
    region, not the centroid computation window itself.
    """
    width = col_mass.size
    lo = max(0, int(np.floor(current_col - search_radius)))
    hi = min(width - 1, int(np.ceil(current_col + search_radius)))
    if hi < lo:
        return None
    gated = col_mass[lo:hi + 1]
    lit = np.flatnonzero(gated > 0)
    if lit.size == 0:
        return None

    gaps = np.flatnonzero(np.diff(lit) > 1)
    run_starts = np.concatenate(([0], gaps + 1))
    run_ends = np.concatenate((gaps, [lit.size - 1]))

    best = None
    best_dist = np.inf
    for s, e in zip(run_starts, run_ends):
        left, right = int(lit[s] + lo), int(lit[e] + lo)
        pixel_count = float(gated[lit[s]:lit[e] + 1].sum())
        if pixel_count < min_pixels:
            continue
        dist = abs(0.5 * (left + right) - current_col)
        if dist < best_dist:
            best_dist = dist
            best = (left, right, pixel_count)
    return best


def _slide_windows(
    mask: np.ndarray, path_top: int, base_col: float, drift_per_row: float, window_count: int,
    half_width: float, min_window_pixels: float, center_x: float,
    adaptive_window_gain: float = 0.0, adaptive_radius_min: float = 8.0,
    adaptive_radius_max: float = np.inf, max_drift_per_row: float = 3.0,
    prev_line_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float | None]:
    """Bottom-up sliding windows, each row band recentering on the midpoint
    of the nearest measured line-width run (see ``_find_nearest_run``);
    empty windows extrapolate via the last known column delta. Returns
    0-indexed window centroid columns/rows (rows already translated to
    full-image coordinates), validity, the nearest valid window's column
    error from ``center_x`` (1-indexed, for direct use as the PID's
    immediate-feedback term), and the last measured run width (``None`` if
    no window succeeded) for the next frame to seed itself with.

    Adaptive search radius (ported from ``slideWindows`` in
    ``originbot_sliding_window_path_generator.m``, 2026-08-09; see
    ``CameraParams.adaptive_window_gain`` for the rationale). The radius
    carries forward from the **last successful window** rather than a
    per-frame average, because perspective makes the line narrower in pixels
    further away, so inheriting window-to-window shrinks the gate naturally
    with distance. An empty window deliberately does NOT update the width --
    with no new measurement, holding the last good one is better than
    letting the radius drift.
    """
    roi_height, image_width = mask.shape
    window_height = max(2, int(np.floor(roi_height / window_count)))

    win_cols = np.zeros(window_count)
    win_rows = np.zeros(window_count)
    win_valid = np.zeros(window_count, dtype=bool)

    current_col = base_col
    last_delta = -window_height * drift_per_row

    adaptive = adaptive_window_gain > 0.0
    drift_margin = max_drift_per_row * window_height
    line_width = prev_line_width if (prev_line_width is not None and prev_line_width > 0) else None
    # Cross-frame seed: the width of the FIRST (nearest) successful window,
    # not the last. MATLAB returns the running `lineWidth`, which after the
    # loop holds the *last* successful window -- the farthest and, by
    # perspective, the narrowest -- and then uses it to size the next
    # frame's first window, which is the nearest and widest. That is
    # backwards from its own stated purpose ("首窗搜索半径的下一帧种子") and
    # systematically under-gates the first window. Measured on the
    # 2026-08-10 lap: it left the detected run touching the gate edge on
    # 93% of window measurements with the fixed gate and 82% with the
    # adaptive one, i.e. the run was being truncated by the gate rather
    # than measured -- the same class of error as the 2026-08-05 undersized
    # window, which is what this whole mechanism exists to avoid.
    # Within a frame the running inheritance is kept exactly as MATLAB has
    # it: it walks from near to far, so it shrinks with perspective, which
    # is correct.
    seed_width: float | None = None

    for w in range(window_count):
        if adaptive and line_width is not None:
            search_radius = max(adaptive_radius_min,
                                min(adaptive_radius_max,
                                    adaptive_window_gain * line_width + drift_margin))
        else:
            search_radius = half_width  # cold start, or adaptive disabled

        row_high = (roi_height - 1) - w * window_height   # window bottom (near), inclusive
        row_low = max(0, row_high - window_height + 1)    # window top (far), inclusive
        row_band = mask[row_low:row_high + 1, :]
        col_mass = row_band.sum(axis=0)

        run = _find_nearest_run(col_mass, current_col, search_radius, min_window_pixels)
        if run is not None:
            left, right, pixel_count = run
            centroid_col = 0.5 * (left + right)
            row_mass = row_band[:, left:right + 1].sum(axis=1)
            centroid_row = row_low + float((row_mass * np.arange(row_mass.size)).sum()) / pixel_count
            if w > 0:
                last_delta = centroid_col - current_col
            current_col = centroid_col
            win_cols[w] = centroid_col
            win_rows[w] = centroid_row + path_top  # ROI-local -> full-image row
            win_valid[w] = True
            line_width = float(right - left + 1)
            if seed_width is None:
                seed_width = line_width
        else:
            current_col = max(0.0, min(image_width - 1.0, current_col + last_delta))

    near_idx = np.flatnonzero(win_valid)
    if near_idx.size == 0:
        near_pixel_error = 0.0
    else:
        near_pixel_error = float((win_cols[near_idx[0]] + 1.0) - center_x)
    return win_cols, win_rows, win_valid, near_pixel_error, seed_width


def _fit_poly_lookahead(
    x_points: np.ndarray, y_points: np.ndarray, valid: np.ndarray, lookahead_distance: float,
) -> tuple[float, float, float, float, np.ndarray]:
    """Direct port of MATLAB's ``fitPolyLookahead``: ground-coordinate quadratic
    fit + lookahead geometry. Sliding windows are stratified by row, so the
    Y=f(X) single-value assumption holds by construction (unlike the archived
    skeleton-walk scheme, which needed a general ordered-arc-length pursuit)."""
    eps = np.finfo(float).eps
    coefficients = np.zeros(3)
    valid_index = np.flatnonzero(valid)
    if valid_index.size == 0:
        lookahead_x = max(0.05, lookahead_distance)
        return lookahead_x, 0.0, 0.0, 0.0, coefficients

    x_valid = x_points[valid_index]
    y_valid = y_points[valid_index]

    if valid_index.size == 1 or (x_valid.max() - x_valid.min()) < 0.02:
        # Single point or forward span too small to fit: treat the nearest point
        # as the lookahead target.
        lookahead_x = float(x_valid[0])
        lookahead_y = float(y_valid[0])
        heading_error = math.atan2(y_valid[0], max(eps, x_valid[0]))
        curvature = 2 * lookahead_y / max(eps, x_valid[0] ** 2 + y_valid[0] ** 2)
        return lookahead_x, lookahead_y, heading_error, curvature, coefficients

    # Clamp the lookahead distance into the valid points' range to avoid extrapolation error.
    lookahead_x = float(np.clip(lookahead_distance, x_valid.min(), x_valid.max()))

    if valid_index.size >= 3:
        coefficients = np.polyfit(x_valid, y_valid, 2)
    else:
        slope2 = (y_valid[1] - y_valid[0]) / max(eps, x_valid[1] - x_valid[0])
        coefficients = np.array([0.0, slope2, y_valid[0] - slope2 * x_valid[0]])

    lookahead_y = float(np.polyval(coefficients, lookahead_x))
    slope = 2 * coefficients[0] * lookahead_x + coefficients[1]
    heading_error = math.atan(slope)
    curvature = 2 * coefficients[0] / max(eps, (1 + slope ** 2) ** 1.5)
    return lookahead_x, lookahead_y, heading_error, curvature, coefficients


def pack_local_path_debug(
    local_path: SlidingWindowPath, lookahead_x: float, lookahead_y: float, curvature: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    """[valid_count, lookahead_x, lookahead_y, curvature, a, b, c, x1, y1, ..., x30, y30]."""
    debug = np.zeros(DEBUG_WIDTH)
    debug[0] = local_path.valid_count
    debug[1] = lookahead_x
    debug[2] = lookahead_y
    debug[3] = curvature
    debug[4:7] = coefficients
    debug[7::2] = local_path.x
    debug[8::2] = local_path.y
    return debug
