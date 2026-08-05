"""Vision debug overlay frame, for on-screen / logged diagnostics only.

Translated from ``originbot_line_follower_debug_frame.m``, the debug-frame
counterpart of ``vision.py``'s Hough-seed + sliding-window + ground-quadratic-fit
algorithm (unified across platforms since MATLAB's 2026-07-22 vision
unification -- see ``vision.py``'s module docstring). Mirrors the same mask/
Hough/sliding-window pipeline as ``LineFollowerVision.step()`` but has no
persistent state and never affects control; reuses ``vision._hough_seed``
directly (rather than hand-duplicating it, unlike the MATLAB source, which
has to because each Simulink MATLABFcn block needs a self-contained
function) so this can never numerically drift from what the controller
actually does for that step. ``_slide_windows_debug`` below is its own copy
of ``vision._slide_windows`` (mirroring the MATLAB source's own separate
copy) only because it needs an extra return value -- each window's pixel
rectangle, for drawing -- that the control path has no use for. Drawing uses
OpenCV primitives rather than the MATLAB source's manual pixel-loop
rasterization -- visually equivalent, not pixel-identical.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .camera_geometry import CameraParams, ground_to_pixel, pixel_to_ground, turtlebot3_burger_mujoco_camera
from .vision import _find_nearest_run, _hough_seed

ROI_FOUND_COLOR = (0, 120, 255)       # blue ROI border when the line is found (RGB)
ROI_LOST_COLOR = (255, 0, 0)          # red ROI border when the line is lost
MASK_COLOR = (0, 255, 0)              # green overlay for detected line pixels
WINDOW_VALID_COLOR = (0, 255, 255)    # cyan box for a valid sliding window
WINDOW_EMPTY_COLOR = (110, 110, 110)  # gray box for an empty (extrapolated) window
CENTROID_COLOR = (255, 0, 255)        # magenta window-centroid marker
CENTER_COLOR = (255, 255, 255)        # white center-column marker
NEAR_COLOR = (255, 0, 0)              # red nearest-valid-window column marker (PID's immediate-feedback source)
PATH_COLOR = (255, 220, 0)            # yellow ground-quadratic-fit curve


def _to_uint8_rgb(rgb_input: np.ndarray, image_height: int, image_width: int) -> np.ndarray:
    rgb = np.asarray(rgb_input)
    if rgb.ndim != 3:
        rgb = rgb.reshape(image_height, image_width, 3)
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    return rgb


def _detect_mask(
    rgb: np.ndarray, top_row: int, bottom_row: int, min_brightness: float, max_saturation: float
) -> tuple[np.ndarray, int]:
    """Mirrors ``vision.LineFollowerVision._detect_mask``: adaptive-brightness
    + low-saturation test, filtered to the largest connected component.
    ``total_pixels`` is measured before that filter, matching the control path."""
    roi = rgb[top_row:bottom_row, :, :].astype(np.float64)
    value = roi.max(axis=2)
    minimum_channel = roi.min(axis=2)
    saturation = (value - minimum_channel) / np.maximum(value, 1)
    roi_max = value.max() if value.size else 0.0
    adaptive_brightness = min(min_brightness, max(40.0, 0.55 * roi_max))
    mask = (value >= adaptive_brightness) & (saturation <= max_saturation)
    total_pixels = int(mask.sum())

    if total_pixels > 0:
        labels, num_components = ndimage.label(mask, structure=np.ones((3, 3)))
        if num_components > 1:
            sizes = ndimage.sum(mask, labels, index=np.arange(1, num_components + 1))
            largest_label = int(np.argmax(sizes)) + 1
            mask = labels == largest_label

    return mask, total_pixels


def _slide_windows_debug(
    mask: np.ndarray, path_top: int, base_col: float, drift_per_row: float, window_count: int,
    half_width: float, min_window_pixels: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirrors ``vision._slide_windows`` (see its docstring and
    ``_find_nearest_run``'s for the 2026-08-05 adaptive-width change),
    additionally returning each window's full-image pixel rectangle
    (row_low, row_high, col_low, col_high), 0-indexed, for drawing --
    ``col_low``/``col_high`` are the *measured* run's own extent when a run
    is found (so the drawn box reflects the actual detected line width, not
    a fixed search box), and the search gate otherwise."""
    roi_height, image_width = mask.shape
    window_height = max(2, int(np.floor(roi_height / window_count)))

    win_cols = np.zeros(window_count)
    win_rows = np.zeros(window_count)
    win_valid = np.zeros(window_count, dtype=bool)
    win_rects = np.zeros((window_count, 4))

    current_col = base_col
    last_delta = -window_height * drift_per_row

    for w in range(window_count):
        row_high = (roi_height - 1) - w * window_height
        row_low = max(0, row_high - window_height + 1)
        row_band = mask[row_low:row_high + 1, :]
        col_mass = row_band.sum(axis=0)

        run = _find_nearest_run(col_mass, current_col, half_width, min_window_pixels)
        if run is not None:
            left, right, pixel_count = run
            win_rects[w] = (row_low + path_top, row_high + path_top, left, right)
            centroid_col = 0.5 * (left + right)
            row_mass = row_band[:, left:right + 1].sum(axis=1)
            centroid_row = row_low + float((row_mass * np.arange(row_mass.size)).sum()) / pixel_count
            if w > 0:
                last_delta = centroid_col - current_col
            current_col = centroid_col
            win_cols[w] = centroid_col
            win_rows[w] = centroid_row + path_top
            win_valid[w] = True
        else:
            col_low = max(0, int(round(current_col - half_width)))
            col_high = min(image_width - 1, int(round(current_col + half_width)))
            win_rects[w] = (row_low + path_top, row_high + path_top, col_low, col_high)
            current_col = max(0.0, min(image_width - 1.0, current_col + last_delta))

    return win_cols, win_rows, win_valid, win_rects


def _detect_attempt_debug(
    rgb: np.ndarray, roi_bottom_fraction: float, window_count: int,
    min_brightness: float, max_saturation: float, min_pixels: float,
    camera: CameraParams, image_height: int, image_width: int,
):
    """Debug-path twin of ``vision.LineFollowerVision._detect_attempt`` --
    same mask -> Hough-seed -> sliding-window -> ground-projection pass, but
    keeping ``win_rects`` (for drawing) that the control path has no use
    for. See that method's docstring for why there can be more than one of
    these per frame (adaptive-ROI fallback, ``roi_widen_step``)."""
    path_top = int(np.clip(np.floor((1 - roi_bottom_fraction) * image_height), 0, image_height - 1))
    path_bottom = image_height

    mask, total_pixels = _detect_mask(rgb, path_top, path_bottom, min_brightness, max_saturation)

    base_col, drift_per_row, hough_found = _hough_seed(
        mask, camera.hough_min_length, camera.hough_fill_gap, camera.max_drift_per_row
    )
    if not hough_found and total_pixels > 0:
        near_start = max(0, mask.shape[0] - 1 - int(np.floor(mask.shape[0] / 3)))
        near_band = mask[near_start:, :]
        column_hist = near_band.sum(axis=0)
        peak_col = int(np.argmax(column_hist))
        peak_val = column_hist[peak_col]
        base_col = float(peak_col) if peak_val > 0 else 0.5 * (image_width - 1)
        drift_per_row = 0.0

    win_cols, win_rows, win_valid, win_rects = _slide_windows_debug(
        mask, path_top, base_col, drift_per_row, window_count,
        camera.window_half_width, camera.min_window_pixels,
    )

    x_valid, y_valid = [], []
    for k in range(window_count):
        if not win_valid[k]:
            continue
        xg, yg = pixel_to_ground(np.array([win_cols[k] + 1.0]), np.array([win_rows[k] + 1.0]), camera)
        if np.isnan(xg[0]) or np.isnan(yg[0]):
            continue
        x_valid.append(float(xg[0]))
        y_valid.append(float(yg[0]))
    valid_count = len(x_valid)
    found = valid_count >= 1 and total_pixels >= min_pixels

    return path_top, path_bottom, mask, win_cols, win_rows, win_valid, win_rects, x_valid, y_valid, found


def render_debug_frame(
    rgb_input: np.ndarray,
    roi_bottom_fraction: float,
    waypoint_count: int,
    min_brightness: float,
    max_saturation: float,
    min_pixels: float,
    camera: CameraParams | None = None,
    image_height: int = 480,
    image_width: int = 640,
    flip_vertical: bool = True,
    roi_widen_step: float = 0.2,
    roi_widen_max: float = 0.7,
) -> np.ndarray:
    """Build an RGB debug frame (uint8, HxWx3) matching the MATLAB overlay:

      * ROI band (blue=found / red=lost)
      * green tint over every mask pixel (post largest-connected-component filter)
      * sliding-window boxes (cyan=valid, gray=empty/extrapolated)
      * magenta window-centroid markers
      * yellow ground-quadratic-fit curve, reprojected via :func:`ground_to_pixel`
      * white center column
      * red nearest-valid-window column (the PID's immediate-feedback source)

    ``camera`` supplies the platform's Hough/sliding-window pixel-scale
    constants and IPM calibration (see ``camera_geometry.CameraParams``);
    defaults to the MuJoCo preset. ``flip_vertical`` corrects MuJoCo's
    bottom-row-first framebuffer (``True`` for MuJoCo, ``False`` for Gazebo/
    ROS image messages -- same convention as ``vision.LineFollowerVision``).
    ``roi_widen_step``/``roi_widen_max`` mirror
    ``vision.LineFollowerVision.roi_widen_step``'s adaptive-ROI fallback, so
    the drawn ROI band matches whichever attempt the real controller would
    have used for this frame (pass 0.0 to always draw the narrow attempt
    only).
    """
    camera = camera or turtlebot3_burger_mujoco_camera()
    roi_bottom_fraction = float(np.clip(roi_bottom_fraction, 0.05, 0.95))
    window_count = int(np.clip(round(waypoint_count), 2, 30))
    rgb = _to_uint8_rgb(rgb_input, image_height, image_width)
    if flip_vertical:
        rgb = np.flip(rgb, axis=0)

    center_x = 0.5 * (image_width + 1)  # 1-indexed, matches pixel_to_ground's cx convention

    (path_top, path_bottom, mask, win_cols, win_rows, win_valid, win_rects,
     x_valid, y_valid, found) = _detect_attempt_debug(
        rgb, roi_bottom_fraction, window_count,
        min_brightness, max_saturation, min_pixels, camera, image_height, image_width,
    )
    widen_step = max(0.0, float(roi_widen_step))
    if not found and widen_step > 0.0:
        widened_fraction = float(np.clip(
            roi_bottom_fraction + widen_step, 0.05, min(0.95, float(roi_widen_max))
        ))
        if widened_fraction > roi_bottom_fraction:
            wide = _detect_attempt_debug(
                rgb, widened_fraction, window_count,
                min_brightness, max_saturation, min_pixels, camera, image_height, image_width,
            )
            if wide[-1]:  # found
                (path_top, path_bottom, mask, win_cols, win_rows, win_valid, win_rects,
                 x_valid, y_valid, found) = wide

    valid_count = len(x_valid)

    frame = rgb.copy()
    frame = _draw_roi(frame, path_top, path_bottom, found)
    frame = _overlay_mask(frame, mask, path_top)

    for k in range(window_count):
        color = WINDOW_VALID_COLOR if win_valid[k] else WINDOW_EMPTY_COLOR
        frame = _draw_rect_outline(frame, win_rects[k], color)

    if valid_count >= 2 and (max(x_valid) - min(x_valid)) >= 0.02:
        x_arr, y_arr = np.array(x_valid), np.array(y_valid)
        if valid_count >= 3:
            coefficients = np.polyfit(x_arr, y_arr, 2)
        else:
            eps = np.finfo(float).eps
            slope2 = (y_arr[1] - y_arr[0]) / max(eps, x_arr[1] - x_arr[0])
            coefficients = np.array([0.0, slope2, y_arr[0] - slope2 * x_arr[0]])
        xs = np.linspace(x_arr.min(), x_arr.max(), 40)
        ys = np.polyval(coefficients, xs)
        us, vs = ground_to_pixel(xs, ys, camera)
        frame = _draw_pixel_curve(frame, us - 1.0, vs - 1.0, PATH_COLOR)  # back to 0-indexed

    for k in range(window_count):
        if win_valid[k]:
            frame = _draw_marker(frame, int(round(win_rows[k])), int(round(win_cols[k])), CENTROID_COLOR)

    frame = _draw_vertical_line(frame, int(round(center_x - 1.0)), path_top, path_bottom, CENTER_COLOR)
    near_idx = np.flatnonzero(win_valid)
    if near_idx.size:
        frame = _draw_vertical_line(frame, int(round(win_cols[near_idx[0]])), path_top, path_bottom,
                                     NEAR_COLOR)
    return frame


def _draw_roi(frame: np.ndarray, top_row: int, bottom_row: int, found: bool) -> np.ndarray:
    color = ROI_FOUND_COLOR if found else ROI_LOST_COLOR
    cv2.line(frame, (0, top_row), (frame.shape[1] - 1, top_row), color, 3)
    cv2.line(frame, (0, bottom_row - 1), (frame.shape[1] - 1, bottom_row - 1), color, 3)
    return frame


def _overlay_mask(frame: np.ndarray, mask: np.ndarray, top_row: int) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return frame
    rows = np.clip(rows + top_row, 0, frame.shape[0] - 1)
    cols = np.clip(cols, 0, frame.shape[1] - 1)
    frame[rows, cols] = MASK_COLOR
    return frame


def _draw_rect_outline(frame: np.ndarray, rect: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    row_low, row_high, col_low, col_high = rect
    r0 = int(np.clip(row_low, 0, h - 1))
    r1 = int(np.clip(row_high, 0, h - 1))
    c0 = int(np.clip(col_low, 0, w - 1))
    c1 = int(np.clip(col_high, 0, w - 1))
    cv2.rectangle(frame, (c0, r0), (c1, r1), color, 1)
    return frame


def _draw_marker(frame: np.ndarray, row: int, col: int, color: tuple[int, int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    row = int(np.clip(row, 0, h - 1))
    col = int(np.clip(col, 0, w - 1))
    cv2.rectangle(frame, (max(0, col - 1), max(0, row - 1)), (min(w - 1, col + 1), min(h - 1, row + 1)),
                  color, -1)
    return frame


def _draw_pixel_curve(frame: np.ndarray, us: np.ndarray, vs: np.ndarray,
                       color: tuple[int, int, int]) -> np.ndarray:
    """Connects a pixel-coordinate sample sequence into a polyline; NaN/out-of-frame
    samples break the line, mirroring the MATLAB source's drawPixelCurve."""
    h, w = frame.shape[:2]
    pts: list[tuple[int, int]] = []
    for u, v in zip(us, vs):
        ok = np.isfinite(u) and np.isfinite(v) and 0 <= u < w and 0 <= v < h
        if ok:
            pts.append((int(round(u)), int(round(v))))
            continue
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, color, 2)
        elif len(pts) == 1:
            frame = _draw_marker(frame, pts[0][1], pts[0][0], color)
        pts = []
    if len(pts) >= 2:
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, color, 2)
    elif len(pts) == 1:
        frame = _draw_marker(frame, pts[0][1], pts[0][0], color)
    return frame


def _draw_vertical_line(
    frame: np.ndarray, col: int, top_row: int, bottom_row: int, color: tuple[int, int, int]
) -> np.ndarray:
    col = int(np.clip(col, 0, frame.shape[1] - 1))
    cv2.line(frame, (col, top_row), (col, bottom_row - 1), color, 3)
    return frame
