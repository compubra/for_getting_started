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
    roi_widen_max: float = 0.7
    """Ceiling for the widened retry (see ``roi_widen_step``) -- kept well
    below 0.95 so the widened attempt still can't reach up into the region
    that misdetected Gazebo's oversized default ground_plane as the line
    before this project's per-track boundary curbs existed (see
    the since-deleted track_bringup.launch.py's docstring); the curbs make that less likely
    now, but there's no reason to widen past what was actually validated."""
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
    _steering_history: deque = field(
        default_factory=lambda: deque(maxlen=STEERING_TREND_SAMPLES), init=False, repr=False
    )
    _lost_steering_slope: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear persistent state (mirrors ``clear originbot_sliding_window_path_generator``)."""
        self._last_visible_steering = 0.0
        self._last_local_path_debug = np.zeros(DEBUG_WIDTH)
        self._lost_line_time = 0.0
        self._last_base_col = None
        self._steering_history.clear()
        self._lost_steering_slope = 0.0

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

        rgb = np.asarray(rgb, dtype=np.float64)
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

        if found:
            self._lost_line_time = 0.0
            steering_error = current_steering_error
            self._last_visible_steering = steering_error
            self._last_local_path_debug = local_path_debug
            self._last_base_col = attempt.base_col
            self._steering_history.append(steering_error)
        else:
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
                self._steering_history.clear()
                self._lost_steering_slope = 0.0

        return LocalPathResult(
            steering_error=steering_error,
            lateral_error=lateral_error,
            heading_error=heading_error,
            confidence=confidence,
            found=found,
            debug=local_path_debug,
        )

    # ------------------------------------------------------------------
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

        base_col, drift_per_row, hough_found = _hough_seed(
            mask, self.camera.hough_min_length, self.camera.hough_fill_gap,
            self.camera.max_drift_per_row, prev_base_col=self._last_base_col,
        )
        if not hough_found and total_pixels > 0:
            near_start = max(0, mask.shape[0] - 1 - int(np.floor(mask.shape[0] / 3)))
            near_band = mask[near_start:, :]
            column_hist = near_band.sum(axis=0)
            peak_col = int(np.argmax(column_hist))
            peak_val = column_hist[peak_col]
            base_col = float(peak_col) if peak_val > 0 else 0.5 * (self.image_width - 1)
            drift_per_row = 0.0

        win_cols, win_rows, win_valid, near_pixel_error = _slide_windows(
            mask, path_top, base_col, drift_per_row, window_count,
            self.camera.window_half_width, self.camera.min_window_pixels, center_x,
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
            base_col=base_col,
        )

    def _detect_mask(self, rgb: np.ndarray, path_top: int, path_bottom: int) -> tuple[np.ndarray, int]:
        """Adaptive-brightness + low-saturation mask, filtered to its largest
        connected component. ``total_pixels`` is measured *before* that
        filter (matches the MATLAB source: it reflects all candidate-line
        pixels, not just the largest blob)."""
        roi = rgb[path_top:path_bottom, :, :]
        value = roi.max(axis=2)
        minimum_channel = roi.min(axis=2)
        saturation = (value - minimum_channel) / np.maximum(value, 1)
        roi_max = value.max() if value.size else 0.0
        adaptive_brightness = min(self.min_brightness, max(40.0, 0.55 * roi_max))
        mask = (value >= adaptive_brightness) & (saturation <= self.max_saturation)
        total_pixels = int(mask.sum())

        if total_pixels > 0:
            labels, num_components = ndimage.label(mask, structure=np.ones((3, 3)))
            if num_components > 1:
                sizes = ndimage.sum(mask, labels, index=np.arange(1, num_components + 1))
                largest_label = int(np.argmax(sizes)) + 1
                mask = labels == largest_label

        return mask, total_pixels

    def _project_windows(
        self, win_cols: np.ndarray, win_rows: np.ndarray, win_valid: np.ndarray, window_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        x_points = np.zeros(MAX_PATH_POINTS)
        y_points = np.zeros(MAX_PATH_POINTS)
        valid = np.zeros(MAX_PATH_POINTS, dtype=bool)
        write_idx = 0
        for k in range(window_count):
            if not win_valid[k]:
                continue
            u_col = win_cols[k] + 1.0  # 0-indexed -> MATLAB's 1-indexed pixel convention
            v_row = win_rows[k] + 1.0
            xg, yg = pixel_to_ground(np.array([u_col]), np.array([v_row]), self.camera)
            if np.isnan(xg[0]) or np.isnan(yg[0]):
                continue
            x_points[write_idx] = xg[0]
            y_points[write_idx] = yg[0]
            valid[write_idx] = True
            write_idx += 1
        return x_points, y_points, valid, int(valid.sum())


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Bottom-up sliding windows, each row band recentering on the midpoint
    of the nearest measured line-width run (see ``_find_nearest_run``);
    empty windows extrapolate via the last known column delta. Returns
    0-indexed window centroid columns/rows (rows already translated to
    full-image coordinates), validity, and the nearest valid window's column
    error from ``center_x`` (1-indexed, for direct use as the PID's
    immediate-feedback term).
    """
    roi_height, image_width = mask.shape
    window_height = max(2, int(np.floor(roi_height / window_count)))

    win_cols = np.zeros(window_count)
    win_rows = np.zeros(window_count)
    win_valid = np.zeros(window_count, dtype=bool)

    current_col = base_col
    last_delta = -window_height * drift_per_row

    for w in range(window_count):
        row_high = (roi_height - 1) - w * window_height   # window bottom (near), inclusive
        row_low = max(0, row_high - window_height + 1)    # window top (far), inclusive
        row_band = mask[row_low:row_high + 1, :]
        col_mass = row_band.sum(axis=0)

        run = _find_nearest_run(col_mass, current_col, half_width, min_window_pixels)
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
        else:
            current_col = max(0.0, min(image_width - 1.0, current_col + last_delta))

    near_idx = np.flatnonzero(win_valid)
    if near_idx.size == 0:
        near_pixel_error = 0.0
    else:
        near_pixel_error = float((win_cols[near_idx[0]] + 1.0) - center_x)
    return win_cols, win_rows, win_valid, near_pixel_error


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
