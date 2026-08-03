"""Camera-local path generator — translation of
runtime/originbot_sliding_window_path_generator.m (2026-07-19 "scheme B":
Hough seed -> sliding window -> polynomial-fit lookahead; the MATLAB SAC/PPO
residual models switched to this from the older
originbot_local_path_generator.m this file previously translated).

Input is an (480, 640, 3) RGB image with values 0..255 (float or uint8), row 0
at the top of the visual field (near ground = bottom rows), exactly like the
double image the Simulink RGB_To_Double block feeds the MATLAB Fcn block.

Outputs match the MATLAB result vector:
  steering_error, lateral_error, heading_error, confidence, found
(the 67-element local-path debug tail is exposed as .debug for parity checks).

The MATLAB function keeps three persistent variables (last visible steering,
last debug vector, lost-line time); here they live on the class, reset per
episode via reset().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from skimage.measure import label as _cc_label
from skimage.transform import probabilistic_hough_line

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
MAX_PATH_POINTS = 30
DEBUG_WIDTH = 7 + 2 * MAX_PATH_POINTS

# TurtleBot3 Burger MuJoCo front camera (cameraParams() in the MATLAB source;
# mount height differs from the lyapunov/Waffle model's 0.117 m).
CAM_FOVY_DEG = 45.9857
CAM_MOUNT_HEIGHT = 0.133
CAM_PITCH_DEG = 15.0
MAX_GROUND_RANGE = 5.0


def _mround(x: float) -> int:
    """MATLAB round(): half away from zero (numpy rounds half to even)."""
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


@dataclass
class VisionParams:
    """Tunables of the detector; defaults = model-workspace values of the
    CURRENT visual_line_follower_sac_residual model (2026-07-20 real-car-speed
    sync). The lyapunov model this file was originally translated from used
    roi_bottom_fraction=0.10; lookahead_distance=0.72 was the pre-tuning value
    for both models — 0.20 is the real-ground-truth-validated optimum found by
    sweeping Lookahead on the MATLAB side at real-car speed (session notes:
    mean tracking error ~2.6x better than 0.72 on the complex/winding tracks)."""

    roi_bottom_fraction: float = 0.50   # LocalPath_ROIFraction (was 0.10)
    cam_pitch_deg: float = CAM_PITCH_DEG  # IPM pitch used by _pixel_to_ground;
                                         # must match whatever physical pitch
                                         # env.py applies to the MuJoCo camera
                                         # (see LineFollowerEnv's cam_pitch_deg),
                                         # or the ground-plane projection is wrong.
    waypoint_count: int = 30            # LocalPath_NumPoints
    min_brightness: float = 70.0        # OriginBot_MinBrightness (0..255)
    max_saturation: float = 0.30        # OriginBot_MaxSaturation
    min_pixels: float = 30.0            # OriginBot_MinPixels
    error_scale: float = 500.0          # OriginBot_ErrorScale
    lookahead_distance: float = 0.20    # LocalPath_LookaheadDistance (m; was 0.72)
    lateral_gain: float = 0.6           # LocalPath_LateralGain
    heading_gain: float = 0.35          # LocalPath_HeadingGain
    curvature_gain: float = 0.04        # LocalPath_CurvatureGain


@dataclass
class VisionResult:
    steering_error: float
    lateral_error: float
    heading_error: float
    confidence: float
    found: float
    debug: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(DEBUG_WIDTH))


def _roi_mask(rgb: np.ndarray, path_top: int, path_bottom: int,
             min_brightness: float, max_saturation: float) -> tuple[np.ndarray, int]:
    """Whole-ROI white-line mask: HSV-style adaptive brightness + low
    saturation, largest 8-connected component (bwlabel(mask, 8) in MATLAB).

    path_top/path_bottom are 1-based inclusive (MATLAB convention). Returns
    (mask, total_pixels); mask is ROI-local (row 0 = global row path_top).
    """
    roi = rgb[path_top - 1:path_bottom, :, :]
    value = roi.max(axis=2)
    minimum_channel = roi.min(axis=2)
    saturation = (value - minimum_channel) / np.maximum(value, 1.0)
    roi_maximum = value.max() if value.size else 0.0
    adaptive_brightness = min(min_brightness, max(40.0, 0.55 * roi_maximum))
    mask = (value >= adaptive_brightness) & (saturation <= max_saturation)

    total_pixels = int(mask.sum())
    if total_pixels > 0:
        labels = _cc_label(mask, connectivity=2)  # 2 == 8-connectivity in 2-D
        if labels.max() > 1:
            counts = np.bincount(labels.ravel())
            counts[0] = 0  # background label never wins
            mask = labels == int(np.argmax(counts))
    return mask, total_pixels


def _hough_seed(mask: np.ndarray, min_length: int, fill_gap: int,
                max_drift: float) -> tuple[float, float, bool]:
    """houghSeed(): dominant line segment in the ROI mask -> sliding-window
    seed column + column-drift-per-row.

    Returns (base_col, drift_per_row, hough_found); base_col/rows below use
    the module's 1-based pixel convention (skimage's 0-based (x, y) points are
    converted immediately on read).
    """
    if not mask.any():
        return 0.0, 0.0, False
    lines = probabilistic_hough_line(mask, threshold=10, line_length=min_length,
                                     line_gap=fill_gap)
    if not lines:
        return 0.0, 0.0, False

    best_len = -1.0
    best = None
    for (x0, y0), (x1, y1) in lines:
        length = math.hypot(x1 - x0, y1 - y0)
        if length > best_len:
            best_len = length
            best = ((x0 + 1, y0 + 1), (x1 + 1, y1 + 1))  # -> 1-based (col, row)

    p1, p2 = best
    dcol, drow = p2[0] - p1[0], p2[1] - p1[1]
    if drow < 0:  # normalize direction: row component points down/near
        dcol, drow = -dcol, -drow

    bottom_row = mask.shape[0]  # 1-based last row of the ROI-local mask
    if abs(drow) < 1e-6:
        base_col = 0.5 * (2 * p1[0] + dcol)  # near-horizontal: segment midpoint
        drift_per_row = 0.0
    else:
        drift_per_row = max(-max_drift, min(max_drift, dcol / drow))
        base_col = p1[0] + (bottom_row - p1[1]) * drift_per_row
    base_col = max(1.0, min(float(mask.shape[1]), base_col))
    return base_col, drift_per_row, True


def _slide_windows(mask: np.ndarray, path_top: int, base_col: float,
                   drift_per_row: float, window_count: int, half_width: int,
                   min_win_pixels: float,
                   center_x: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """slideWindows(): bottom-up sliding window centroid tracking.

    Returns (win_cols, win_rows, win_valid, near_pixel_error). win_cols are
    1-based columns; win_rows are GLOBAL 1-based rows (ready for
    _pixel_to_ground). Empty windows extrapolate along the last known drift.
    """
    roi_height, image_width = mask.shape
    window_height = max(2, roi_height // window_count)

    win_cols = np.zeros(window_count)
    win_rows = np.zeros(window_count)
    win_valid = np.zeros(window_count, dtype=bool)

    current_col = base_col
    last_delta = -window_height * drift_per_row

    for w in range(window_count):
        row_high0 = (roi_height - 1) - w * window_height       # 0-based, near edge
        row_low0 = max(0, row_high0 - window_height + 1)       # 0-based, far edge
        col_low1 = max(1, _mround(current_col - half_width))   # 1-based
        col_high1 = min(image_width, _mround(current_col + half_width))

        window = mask[row_low0:row_high0 + 1, col_low1 - 1:col_high1]
        pixel_count = int(window.sum())
        if pixel_count >= min_win_pixels:
            col_mass = window.sum(axis=0)
            row_mass = window.sum(axis=1)
            cols_1based = np.arange(col_low1, col_high1 + 1)
            rows_local_1based = np.arange(row_low0 + 1, row_high0 + 2)
            centroid_col = float((col_mass * cols_1based).sum()) / pixel_count
            centroid_row_local1 = float((row_mass * rows_local_1based).sum()) / pixel_count
            if w > 0 or win_valid[0]:
                last_delta = centroid_col - current_col
            current_col = centroid_col
            win_cols[w] = centroid_col
            win_rows[w] = centroid_row_local1 + path_top - 1  # -> global 1-based row
            win_valid[w] = True
        else:
            current_col = max(1.0, min(float(image_width), current_col + last_delta))

    near_idx = np.flatnonzero(win_valid)
    near_pixel_error = (0.0 if near_idx.size == 0
                        else float(win_cols[near_idx[0]] - center_x))
    return win_cols, win_rows, win_valid, near_pixel_error


def _pixel_to_ground(u: float, v: float,
                     cam_pitch_deg: float = CAM_PITCH_DEG) -> tuple[float, float]:
    """pixelToGround(): pinhole + flat-ground IPM. u,v are 1-based pixel coords.

    Returns (X forward, Y right) in metres, or (nan, nan) above the horizon or
    beyond MAX_GROUND_RANGE.
    """
    fy = (IMAGE_HEIGHT / 2) / math.tan(math.radians(CAM_FOVY_DEG / 2))
    fx = fy
    cx = (IMAGE_WIDTH + 1) / 2
    cy = (IMAGE_HEIGHT + 1) / 2

    xc = (u - cx) / fx
    yc = (v - cy) / fy
    zc = 1.0

    ph = math.radians(cam_pitch_deg)
    d_down = math.cos(ph) * yc + math.sin(ph) * zc
    d_forward = -math.sin(ph) * yc + math.cos(ph) * zc
    d_right = xc

    if d_down <= 1e-6:
        return float("nan"), float("nan")
    t = CAM_MOUNT_HEIGHT / d_down
    x_ground = t * d_forward
    y_ground = t * d_right
    if not (x_ground <= MAX_GROUND_RANGE):
        return float("nan"), float("nan")
    return x_ground, y_ground


def _fit_quadratic_lookahead(x_points, y_points, valid, lookahead_distance):
    """fitQuadraticLookahead(): poly fit + lookahead lateral/heading/curvature."""
    valid_index = np.flatnonzero(valid)
    coefficients = np.array([0.0, 0.0, 0.0])
    if valid_index.size == 0:
        lookahead_x = max(0.18, min(0.96, lookahead_distance))
        return lookahead_x, 0.0, 0.0, 0.0, coefficients

    x_valid = x_points[valid_index]
    y_valid = y_points[valid_index]
    lookahead_x = min(max(lookahead_distance, x_valid.min()), x_valid.max())

    if valid_index.size >= 3:
        coefficients = np.polyfit(x_valid, y_valid, 2)
    elif valid_index.size == 2:
        slope = (y_valid[1] - y_valid[0]) / max(np.finfo(float).eps,
                                                x_valid[1] - x_valid[0])
        intercept = y_valid[0] - slope * x_valid[0]
        coefficients = np.array([0.0, slope, intercept])
    else:
        coefficients = np.array([0.0, 0.0, y_valid[0]])

    lookahead_y = float(np.polyval(coefficients, lookahead_x))
    slope = 2 * coefficients[0] * lookahead_x + coefficients[1]
    curvature = 2 * coefficients[0] / max(np.finfo(float).eps,
                                          (1 + slope ** 2) ** 1.5)
    heading_error = math.atan(slope)
    return lookahead_x, lookahead_y, heading_error, float(curvature), coefficients


def _pack_debug(local_path, lookahead_x, lookahead_y, curvature, coefficients):
    debug = np.zeros(DEBUG_WIDTH)
    debug[0] = local_path["ValidCount"]
    debug[1] = lookahead_x
    debug[2] = lookahead_y
    debug[3] = curvature
    debug[4:7] = np.asarray(coefficients).ravel()
    debug[7::2] = local_path["X"]
    debug[8::2] = local_path["Y"]
    return debug


class LocalPathGenerator:
    """Stateful equivalent of the MATLAB function (persistent vars → attrs).

    Detection pipeline: Hough seed -> sliding window -> polynomial-fit
    lookahead (translation of originbot_sliding_window_path_generator.m, the
    MATLAB SAC/PPO residual models' current vision algorithm as of 2026-07-19
    — this replaces an earlier band-scan translation of the deprecated,
    pre-2026-07-18 MATLAB detector, which also carried a "try the image
    upright and 180°-rotated, keep whichever finds more white pixels"
    heuristic that MATLAB deliberately removed for mis-switching orientation
    on V-bends; this translation does not carry that heuristic either).
    """

    # 丢线分级阈值 — same constants as the MATLAB source
    CONTROL_PERIOD = 0.05
    FREEZE_TIMEOUT = 0.5
    SLOWDOWN_TIMEOUT = 1.5

    # Internal detector constants (not exposed via VisionParams — mirrors the
    # MATLAB source, where these are also file-local constants, not
    # model-workspace-tunable). Pixel units.
    WINDOW_HALF_WIDTH = 20     # sliding-window half-width (px)
    MIN_WINDOW_PIXELS = 5      # min white pixels for a window to count valid
    MAX_DRIFT_PER_ROW = 3.0    # clamp on the Hough-seed column drift (px/row)
    HOUGH_MIN_LENGTH = 40      # probabilistic_hough_line line_length
    HOUGH_FILL_GAP = 20        # probabilistic_hough_line line_gap

    def __init__(self, params: VisionParams | None = None):
        self.params = params or VisionParams()
        self.reset()

    def reset(self) -> None:
        self.last_visible_steering = 0.0
        self.last_debug = np.zeros(DEBUG_WIDTH)
        self.lost_line_time = 0.0

    def process(self, rgb: np.ndarray) -> VisionResult:
        p = self.params
        rgb = np.asarray(rgb, dtype=np.float64)
        assert rgb.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3), rgb.shape

        roi_bottom_fraction = max(0.05, min(0.95, p.roi_bottom_fraction))
        window_count = max(2, min(MAX_PATH_POINTS, _mround(p.waypoint_count)))
        error_scale = max(np.finfo(float).eps, p.error_scale)

        # ROI rows (1-based, like MATLAB)
        path_top = max(1, min(IMAGE_HEIGHT,
                              math.floor((1 - roi_bottom_fraction) * IMAGE_HEIGHT) + 1))
        path_bottom = IMAGE_HEIGHT
        center_x = 0.5 * (IMAGE_WIDTH + 1)

        mask, total_pixels = _roi_mask(rgb, path_top, path_bottom,
                                       p.min_brightness, p.max_saturation)

        base_col, drift_per_row, hough_found = _hough_seed(
            mask, self.HOUGH_MIN_LENGTH, self.HOUGH_FILL_GAP, self.MAX_DRIFT_PER_ROW)

        if not hough_found and total_pixels > 0:
            # Fallback: column-histogram peak over the near (bottom) third.
            near_band = mask[-max(1, mask.shape[0] // 3):, :]
            column_hist = near_band.sum(axis=0)
            peak_col0 = int(np.argmax(column_hist))
            peak_val = column_hist[peak_col0]
            base_col = float(peak_col0 + 1) if peak_val > 0 else center_x
            drift_per_row = 0.0

        win_cols, win_rows, win_valid, near_pixel_error = _slide_windows(
            mask, path_top, base_col, drift_per_row, window_count,
            self.WINDOW_HALF_WIDTH, self.MIN_WINDOW_PIXELS, center_x)

        # Project valid window centroids to ground coordinates (IPM).
        x_points = np.zeros(MAX_PATH_POINTS)
        y_points = np.zeros(MAX_PATH_POINTS)
        valid = np.zeros(MAX_PATH_POINTS, dtype=bool)
        write_idx = 0
        for k in range(window_count):
            if not win_valid[k]:
                continue
            xg, yg = _pixel_to_ground(win_cols[k], win_rows[k], p.cam_pitch_deg)
            if math.isnan(xg) or math.isnan(yg):
                continue
            x_points[write_idx] = xg
            y_points[write_idx] = yg
            valid[write_idx] = True
            write_idx += 1
        valid_count = float(valid.sum())

        found = float(valid_count >= 1 and total_pixels >= p.min_pixels)
        confidence = min(1.0, total_pixels / max(1.0, window_count * p.min_pixels))

        lookahead_x, lookahead_y, heading_error, curvature, coefficients = \
            _fit_quadratic_lookahead(x_points, y_points, valid, p.lookahead_distance)

        lateral_error = max(-1.0, min(1.0, lookahead_y / 0.55))
        heading_error = max(-1.0, min(1.0, heading_error))
        curvature_for_control = max(-1.0, min(1.0, 0.20 * curvature))
        centroid_steering = near_pixel_error / error_scale
        path_steering = (p.lateral_gain * lateral_error
                         + p.heading_gain * heading_error
                         + p.curvature_gain * curvature_for_control)
        current_steering_error = max(-1.5, min(1.5, 0.65 * path_steering
                                               + 0.35 * centroid_steering))

        local_path = {"X": x_points, "Y": y_points, "ValidCount": valid_count}
        debug = _pack_debug(local_path, lookahead_x, lookahead_y,
                            curvature, coefficients)

        if found > 0.5:
            self.lost_line_time = 0.0
            steering_error = current_steering_error
            self.last_visible_steering = steering_error
            self.last_debug = debug
        else:
            self.lost_line_time += self.CONTROL_PERIOD
            lateral_error = 0.0
            heading_error = 0.0
            debug = self.last_debug.copy()
            debug[0] = 0.0
            if self.lost_line_time <= self.FREEZE_TIMEOUT:
                steering_error = self.last_visible_steering
            elif self.lost_line_time <= self.SLOWDOWN_TIMEOUT:
                decay = 1 - (self.lost_line_time - self.FREEZE_TIMEOUT) / max(
                    np.finfo(float).eps, self.SLOWDOWN_TIMEOUT - self.FREEZE_TIMEOUT)
                steering_error = self.last_visible_steering * decay
            else:
                steering_error = 0.0
                self.last_visible_steering = 0.0

        return VisionResult(steering_error=float(steering_error),
                            lateral_error=float(lateral_error),
                            heading_error=float(heading_error),
                            confidence=float(confidence),
                            found=found,
                            debug=debug)
