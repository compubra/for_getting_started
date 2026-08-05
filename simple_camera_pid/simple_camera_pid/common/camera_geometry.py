"""Pinhole + flat-ground inverse perspective mapping (IPM).

Translated from ``originbot_pixel_to_ground.m`` / ``originbot_ground_to_pixel.m`` /
``originbot_camera_profile.m``, the shared camera/projection functions used by both
platforms since the 2026-07-22 vision unification (see ``vision.py``). Before that,
each platform's IPM math lived duplicated (and MuJoCo's pitched 15 deg) inside
``originbot_local_path_generator.m`` / ``originbot_local_path_generator_gazebo.m`` --
now archived under ``matlab/archive/archive_vision_scheme_a/``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraParams:
    """Pinhole camera + mount geometry, plus the sliding-window algorithm's
    per-platform pixel-scale constants (``originbot_camera_profile.m`` is the
    single source of truth for both, on the MATLAB side)."""

    image_width: int
    image_height: int
    fovy_deg: float       # MJCF/vertical field of view (degrees)
    mount_height: float   # camera height above ground (m)
    pitch_deg: float      # downward pitch, 0 = horizontal (degrees)
    needs_flip: bool      # True if the raw frame is bottom-row-first (MuJoCo framebuffer)
    window_half_width: float    # sliding-window search radius (px) -- see _find_nearest_run
    min_window_pixels: float    # min white pixels for a window to count as valid
    hough_min_length: float     # houghlines MinLength (px)
    hough_fill_gap: float       # houghlines FillGap (px)
    default_roi: float          # LocalPath_ROIFraction fallback
    default_lookahead: float    # LocalPath_LookaheadDistance fallback (m)
    hough_max_peaks: int = 4        # platform-independent
    max_drift_per_row: float = 3.0  # platform-independent, a ratio (not px-scaled)
    max_ground_range: float = 5.0   # discard projections beyond this forward distance (m)


def turtlebot3_burger_mujoco_camera() -> CameraParams:
    """TurtleBot3 Burger MuJoCo front camera, level mount (pitch=0).

    Source: turtlebot3_burger_vehicle_body.xml + scene <global offwidth/offheight>,
    xyaxes="0 -1 0 0 0 1". Flattened from a 15 deg down-pitch mount on 2026-07-22 to
    match Gazebo's always-level camera (see originbot_camera_profile.m and
    matlab/archive/archive_vision_scheme_a/README_RESTORE.md for the old pitched
    calibration and the scheme-A algorithm it went with).
    """
    return CameraParams(
        image_width=640,
        image_height=480,
        fovy_deg=45.9857,
        mount_height=0.133,
        pitch_deg=0.0,
        needs_flip=True,
        window_half_width=20.0,
        min_window_pixels=5.0,
        hough_min_length=40.0,
        hough_fill_gap=20.0,
        default_roi=0.30,
        default_lookahead=0.40,
    )


def turtlebot3_burger_gazebo_camera() -> CameraParams:
    """TurtleBot3 Burger Gazebo front camera (horizontal, no pitch).

    Official turtlebot3_burger has no built-in camera; this project's local
    Gazebo model (model/gazebo/turtlebot3/models/turtlebot3_burger_line_follower)
    adds one using the same Intel RealSense r200 optics as stock
    turtlebot3_waffle (horizontal_fov=1.02974 rad converted to the equivalent
    vertical FOV) so the vision pipeline's calibration stays valid.

    2026-08-03: image_width/image_height set to 640x480 to match the real
    TurtleBot3's camera (its driver was reconfigured to this resolution) and
    turtlebot3_burger_mujoco_camera() -- all three platforms now share one
    resolution. This was 800x600 from 2026-07-31 to 2026-08-03 (downscaled
    from an original 1920x1080 for render-cost reasons -- see git history/
    model.sdf's <image> comment if reviving that reasoning); 640x480 is 4:3,
    the SAME aspect ratio as that 800x600, so fovy_deg does NOT need
    re-deriving (still 45.9857, derived from the SDF's fixed horizontal_fov
    of 1.02974 rad -- a near-exact match to turtlebot3_burger_mujoco_camera()'s
    own 45.9857, since that camera is also 4:3 -- coincidence of aspect
    ratio, not shared derivation). The four pixel-scale constants below
    (which describe real-world feature sizes in pixels, not angles) scale
    linearly by image_width/640 (the same rule turtlebot3_burger_real_camera()
    below documents) -- at 640 wide that ratio is exactly 1.0, so they're
    back to their original values (window_half_width=20, min_window_pixels=5,
    hough_min_length=40, hough_fill_gap=20), identical to
    turtlebot3_burger_mujoco_camera()'s own numbers. Whenever width/height
    here changes again, fovy_deg MUST be recomputed from horizontal_fov + the
    new aspect ratio if the ratio changes (it does not simply scale), and
    these four constants rescaled by the new image_width/640 ratio.
    """
    return CameraParams(
        image_width=640,
        image_height=480,
        fovy_deg=45.9857,
        mount_height=0.133,
        pitch_deg=0.0,
        needs_flip=False,
        window_half_width=20.0,
        min_window_pixels=5.0,
        hough_min_length=40.0,
        hough_fill_gap=20.0,
        default_roi=0.10,
        default_lookahead=0.20,
    )


def turtlebot3_burger_real_camera(
    image_width: int,
    image_height: int,
    fovy_deg: float,
    mount_height: float,
    pitch_deg: float = 0.0,
) -> CameraParams:
    """Real TurtleBot3 Burger USB/CSI camera -- fill in from an actual measurement,
    not a simulator default.

    ``fovy_deg``: vertical field of view in degrees. Get it from the camera
    module's datasheet (convert horizontal/diagonal FOV via the sensor's aspect
    ratio if that's what's published), or calibrate it empirically (e.g.
    ``cv2.calibrateCamera`` on a checkerboard, then
    ``fovy_deg = 2*degrees(atan(image_height / (2*fy)))``).
    ``mount_height``: camera optical center height above the ground plane (m),
    measured with a tape measure.
    ``pitch_deg``: downward tilt from horizontal (m), 0 if mounted level.

    The four pixel-scale constants (``window_half_width``, ``min_window_pixels``,
    ``hough_min_length``, ``hough_fill_gap``) describe real-world feature sizes
    in pixels, not angles, so they don't carry over from the Gazebo/MuJoCo
    presets unless the resolution matches. This scales
    ``turtlebot3_burger_gazebo_camera()``'s calibrated 640-px-wide values
    linearly by ``image_width`` (the same rule that camera's own docstring
    documents for its 1920->640 rescale) as a starting point -- re-tune
    ``min_pixels``/``roi_bottom_fraction``/``lookahead_distance`` (in
    ``VisionConfig`` / the node's ROS parameters) against real frames before
    trusting the found-rate, the same way the Gazebo camera's roi defaults
    were empirically tuned (see ``vision.py``'s ``roi_widen_step`` docstring).

    ``window_half_width`` (180px here, scaled) is NOT this robot's actual
    line width -- since 2026-08-05 ``_find_nearest_run`` measures that
    directly per row from the mask (see its docstring), so this only needs
    to be a generous search radius around the previous row's column, wide
    enough to comfortably contain the widest the line is ever measured at
    plus margin for base_col drift between rows/frames. 640-px-wide real
    frames on this robot's track measured a 130-161px-wide tape line
    (``python3 -c`` brightness/saturation-threshold column scan on a live
    frame, 2026-08-05) -- 180px clears that with room to spare. Was 20px
    (a stale carry-over from being the actual centroid-window width, back
    when it was one) until that same measurement showed it covered barely
    a eighth of the true line width, letting the sliding window lock onto
    either edge of the line at random rather than its center -- see the
    real-robot spin/wobble incidents this caused, and _find_nearest_run's
    docstring for the fix.
    """
    scale = image_width / 640.0
    return CameraParams(
        image_width=image_width,
        image_height=image_height,
        fovy_deg=fovy_deg,
        mount_height=mount_height,
        pitch_deg=pitch_deg,
        needs_flip=False,
        window_half_width=180.0 * scale,
        min_window_pixels=5.0 * scale,
        hough_min_length=40.0 * scale,
        hough_fill_gap=20.0 * scale,
        default_roi=0.10,
        default_lookahead=0.20,
    )


def pixel_to_ground(u: np.ndarray, v: np.ndarray, p: CameraParams) -> tuple[np.ndarray, np.ndarray]:
    """Project pixel coordinates (u, v) to ground-plane (X forward, Y right) in meters.

    Returns NaN where the ray points above the horizon (no ground intersection)
    or the projected forward distance exceeds ``p.max_ground_range`` (near-horizon
    singularity guard, matching the MATLAB implementation).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    fy = (p.image_height / 2) / math.tan(math.radians(p.fovy_deg) / 2)
    fx = fy
    cx = (p.image_width + 1) / 2
    cy = (p.image_height + 1) / 2

    xc = (u - cx) / fx
    yc = (v - cy) / fy
    zc = np.ones_like(xc)

    ph = math.radians(p.pitch_deg)
    d_down = math.cos(ph) * yc + math.sin(ph) * zc
    d_forward = -math.sin(ph) * yc + math.cos(ph) * zc
    d_right = xc

    with np.errstate(divide="ignore", invalid="ignore"):
        t = p.mount_height / d_down
    t = np.where(d_down <= 1e-6, np.nan, t)

    x_ground = t * d_forward
    y_ground = t * d_right

    beyond = ~(x_ground <= p.max_ground_range)  # NaN also counts as beyond
    x_ground = np.where(beyond, np.nan, x_ground)
    y_ground = np.where(beyond, np.nan, y_ground)
    return x_ground, y_ground


def ground_to_pixel(x_ground: np.ndarray, y_ground: np.ndarray, p: CameraParams) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pixel_to_ground`: ground-plane (X forward, Y right, m) to pixel (u, v).

    Translated from ``originbot_ground_to_pixel.m``. Used only by the debug-frame
    overlay to reproject the fitted ground-coordinate curve back into the camera
    image for drawing; the control path never needs this direction. Returns NaN
    where the ground point is behind the camera (not visible).
    """
    x_ground = np.asarray(x_ground, dtype=np.float64)
    y_ground = np.asarray(y_ground, dtype=np.float64)

    fy = (p.image_height / 2) / math.tan(math.radians(p.fovy_deg) / 2)
    fx = fy
    cx = (p.image_width + 1) / 2
    cy = (p.image_height + 1) / 2

    ph = math.radians(p.pitch_deg)
    d_forward = x_ground
    d_right = y_ground
    d_down = np.full_like(x_ground, p.mount_height)

    yc = math.cos(ph) * d_down - math.sin(ph) * d_forward
    zc = math.sin(ph) * d_down + math.cos(ph) * d_forward
    xc = d_right

    u = np.full_like(x_ground, np.nan)
    v = np.full_like(x_ground, np.nan)
    in_front = zc > 1e-6
    u = np.where(in_front, cx + fx * xc / np.where(in_front, zc, 1.0), u)
    v = np.where(in_front, cy + fy * yc / np.where(in_front, zc, 1.0), v)
    return u, v
