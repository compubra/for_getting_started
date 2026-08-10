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
from scipy.spatial.transform import Rotation


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
    # Adaptive sliding-window search radius, ported from
    # originbot_camera_profile.m's 2026-08-09 block (which sets all three
    # after its platform switch, i.e. they are platform-independent there
    # too). Since _find_nearest_run measures the line's run width on every
    # row, the search gate no longer has to be a hand-tuned per-platform
    # constant -- it is derived from that measurement instead:
    #
    #   radius = clamp(gain * measured_width + max_drift_this_window,
    #                  radius_min, radius_max_fraction * image_width)
    #
    # The drift term is max_drift_per_row * window_height, i.e. the largest
    # column shift the Hough slope clamp permits within one window, so the
    # radius always covers "line width + how far this window may move".
    # The upper bound is a fraction of image width rather than an absolute
    # pixel count so it survives a resolution change untouched.
    # Set adaptive_window_gain=0.0 to disable and fall back to the fixed
    # window_half_width (the pre-2026-08-09 behavior). window_half_width is
    # still used even when this is on: it seeds the very first window of a
    # frame, before there is any measured width to work from.
    adaptive_window_gain: float = 1.5
    adaptive_radius_min: float = 8.0          # px floor, so a thin line can't close the gate
    # x image_width. MATLAB uses 0.30 (640 -> 192); this is 0.40 (-> 256),
    # a deliberate divergence measured on the 2026-08-10 lap (976 frames,
    # 18k window measurements). The number that matters is how often the
    # gate TRUNCATES the run instead of containing it -- when it does, the
    # midpoint collapses toward the gate centre and the window stops
    # re-centring on the line, which is the same failure the 2026-08-05
    # undersized-window fix was about:
    #
    #   fixed 180px gate      43.5% of runs truncated by the gate
    #   adaptive, 0.30w       29.3%
    #   adaptive, 0.35w        8.6%
    #   adaptive, 0.40w        5.0%   <- here, past the knee
    #   adaptive, 0.45w        3.8%
    #   adaptive, 0.60w        2.4%   (gate ~= whole image by then)
    #
    # 0.30 is too low for this track because the white path measures 275 px
    # wide (median) on a 640 px frame, so gain*width alone wants 424 px and
    # the ceiling binds on every near window. Stopping at 0.40 rather than
    # going higher keeps the gate meaningfully narrower than the frame, so
    # it still does its original job of refusing to jump to an unrelated
    # bright region elsewhere in the row -- the failure behind the "locked
    # onto a white object across the room" run in real_line_follower.yaml.
    # Detection was 634/634 with 0 false positives at every value tested,
    # so this is a measurement-fidelity fix, not a detection-rate one.
    adaptive_radius_max_fraction: float = 0.40
    max_ground_range: float = 5.0   # discard projections beyond this forward distance (m)
    roll_deg: float = 0.0    # mount roll about the camera's forward/optical axis (deg), 0 = no roll
    yaw_deg: float = 0.0     # mount yaw about the true-vertical axis (deg), 0 = no yaw


def turtlebot3_burger_mujoco_camera() -> CameraParams:
    """TurtleBot3 Burger MuJoCo front camera: Pi Cam v2 optics, 15 deg down-pitch.

    Source: turtlebot3_burger_vehicle_body.xml + scene <global offwidth/offheight>,
    xyaxes="0 -1 0 0.258819 0 0.965926".

    2026-08-04: two changes, both to match the real robot.

    * Optics: Intel RealSense r200 -> official Raspberry Pi Camera Module v2
      (IMX219), which is what the hardware carries. fovy_deg re-derived from
      the module's official 62.2 deg horizontal FOV at this 4:3 output:
      2*atan((480/640)*tan(62.2deg/2)) = 48.6867 deg (official spec quotes
      48.8 deg vertical; the 0.11 deg gap is spec rounding). The IMX219's
      native 3280x2464 is itself 4:3, so 640x480 is not cropped and the full
      sensor FOV applies -- if the output ever moves to a non-4:3 ratio this
      MUST be re-derived, it does not simply scale.
    * Pitch: restored to 15 deg down, matching config/real/real_line_follower.yaml's
      camera_pitch_deg (confirmed 2026-08-03 against the hardware). It had been
      flattened to 0 deg on 2026-07-22 to match the then-level Gazebo camera;
      Gazebo has now been pitched to 15 deg too, so all three platforms agree.

    default_roi/default_lookahead go back to the pitched-camera values
    (0.50/0.20). At 15 deg down-pitch the nearest visible ground is
    mount_height/tan(pitch + fovy/2) = 0.162 m, so a 0.20 m lookahead is
    reachable; at 0 deg it was 0.294 m, which is why lookahead had been raised
    to 0.40 while flat. These are geometry-derived starting points, NOT the
    product of a ground-truth tuning sweep -- re-tune against real frames.
    """
    return CameraParams(
        image_width=640,
        image_height=480,
        fovy_deg=48.6867,
        mount_height=0.133,
        pitch_deg=15.0,
        needs_flip=True,
        window_half_width=30.0,
        min_window_pixels=5.0,
        hough_min_length=40.0,
        hough_fill_gap=20.0,
        default_roi=0.50,
        default_lookahead=0.20,
    )


def turtlebot3_burger_gazebo_camera() -> CameraParams:
    """TurtleBot3 Burger Gazebo front camera: Pi Cam v2 optics, 15 deg down-pitch.

    Official turtlebot3_burger has no built-in camera; this project's local
    Gazebo model (model/gazebo/turtlebot3/models/turtlebot3_burger_line_follower)
    adds one modelling the official Raspberry Pi Camera Module v2 (IMX219) that
    the real robot carries.

    2026-08-04: switched from Intel RealSense r200 optics (horizontal_fov
    1.02974 rad, fovy 45.9857) to Pi Cam v2 (horizontal_fov 1.0855948 rad =
    62.2 deg, fovy re-derived to 48.6867 at 4:3), and pitched 15 deg down to
    match the real robot -- see turtlebot3_burger_mujoco_camera() above for the
    full derivation and the roi/lookahead consequences, which apply identically
    here since both platforms now share one camera model.

    2026-08-03: image_width/image_height set to 640x480 to match the real
    TurtleBot3's camera (its driver was reconfigured to this resolution) and
    turtlebot3_burger_mujoco_camera() -- all three platforms now share one
    resolution. This was 800x600 from 2026-07-31 to 2026-08-03 (downscaled
    from an original 1920x1080 for render-cost reasons -- see git history/
    model.sdf's <image> comment if reviving that reasoning); 640x480 is 4:3,
    the SAME aspect ratio as that 800x600, so fovy_deg did not need
    re-deriving at that time (it was then 45.9857, from the r200's
    horizontal_fov of 1.02974 rad; superseded by the 2026-08-04 entry above).
    The four pixel-scale constants below
    (which describe real-world feature sizes in pixels, not angles) scale
    linearly by image_width/640 (the same rule turtlebot3_burger_real_camera()
    below documents) -- at 640 wide that ratio is exactly 1.0, so they're
    back to their original values (window_half_width=30, min_window_pixels=5,
    hough_min_length=40, hough_fill_gap=20), identical to
    turtlebot3_burger_mujoco_camera()'s own numbers. Whenever width/height
    here changes again, fovy_deg MUST be recomputed from horizontal_fov + the
    new aspect ratio if the ratio changes (it does not simply scale), and
    these four constants rescaled by the new image_width/640 ratio.

    2026-08-04: window_half_width 20 -> 30 (both this and
    turtlebot3_burger_mujoco_camera(), plus turtlebot3_burger_real_camera()'s
    scale-from-640 base below) -- a real-hardware debug session found the
    sliding window visibly too narrow to reliably straddle the line under
    real camera noise (a 40px-wide window at 640px-wide frames is a small
    target once the line's detected column jitters by even a few px frame to
    frame -- see vision.py's _hough_seed prev_base_col tie-break for the
    related Hough-seed noise-sensitivity fix from the same session). 30 is a
    moderate widen (not the 60 an old archived MATLAB Gazebo-only scheme once
    used, matlab/archive/archive_vision_scheme_a/ -- that was tuned for a
    different, now-superseded camera/resolution and isn't a validated number
    for this camera). Re-verify found_rate on both track_easy and
    track_hard after any further change here (see roi_widen_step's docstring
    for how this project measures that).
    """
    return CameraParams(
        image_width=640,
        image_height=480,
        fovy_deg=48.6867,
        mount_height=0.133,
        pitch_deg=15.0,
        needs_flip=False,
        window_half_width=30.0,
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
    roll_deg: float = 0.0,
    yaw_deg: float = 0.0,
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
    ``roll_deg``/``yaw_deg``: mount roll/yaw (degrees, 0 = none) -- see
    ``_mount_rotation_matrix()``'s docstring for the exact rotation and sign
    convention. Added 2026-08-07 after a real-hardware session found a
    stable (low-noise, repeatable within one sitting) ``heading_error``/
    ``lateral_error`` bias with the robot stationary and centered on a
    straight line -- neither error should be nonzero in that pose if
    ``pitch_deg``/``mount_height`` were the only mount-geometry unknowns, so
    the remaining bias is attributed to roll/yaw the existing single-axis
    model couldn't represent at all. Calibration procedure (no dedicated
    tool in this repo -- matches this project's existing "measure live,
    hardcode the number, comment it" convention, same as ``pitch_deg``'s
    own history below):
      1. Stand the robot stationary, centered and heading-aligned on a
         straight track segment.
      2. Run vision-only, zero connection to ``/cmd_vel`` (safe -- no motors
         involved): ``ros2 run simple_camera_pid line_follower_vision_node
         --ros-args --params-file <real_line_follower.yaml> -p
         camera_roll_deg:=<guess> -p camera_yaw_deg:=<guess>``.
      3. Sample ``/line_follower/local_path`` (``[steering_error,
         lateral_error, heading_error, confidence, found]``) for a few
         seconds; average.

         2026-08-07 real-hardware finding: within one restart, samples are
         very low-noise (stdev ~0.001-0.02 rad) -- but restarting the SAME
         node with the SAME roll/yaw values can give a wildly different
         reading next time (observed swings of ~0.5 rad / ~30deg between
         back-to-back restarts at an identical ``camera_yaw_deg``). Root
         cause: ``_hough_seed()`` has no ``prev_base_col`` to anchor to on a
         fresh process's first frame, so it can lock onto either long edge
         of the (real, non-zero-width) line -- see ``_hough_seed()``'s own
         docstring in ``vision.py`` for that ambiguity. Whichever edge wins
         on frame 1 then stays locked in (temporal continuity keeps it
         there) for the rest of that run, so it looks perfectly stable
         within a session while being essentially a coin flip across
         restarts. **Do 2-3 fresh restarts per candidate value and use their
         mean, not a single restart's reading** -- a single restart's number
         is not trustworthy enough to coordinate-descend on. This is the
         same bistability that caused this project's earlier real-robot
         spin/wobble incidents (see ``_find_nearest_run``'s docstring) --
         fixing ``_hough_seed`` to reject an implausible frame-1 seed (or to
         average both edges instead of picking one) would remove the need
         for multi-restart averaging here, but is a separate, bigger change
         than this calibration procedure.
      4. Coordinate descent: ``heading_error ~= radians(yaw_deg)`` when
         centered (near-exact -- ``y_ground/x_ground = tan(yaw_deg)``), so
         guess ``camera_yaw_deg = -degrees(heading_error)`` first, re-run,
         and keep adjusting in whichever direction shrinks ``heading_error``
         (flip the sign if it grows instead). Then do the same for
         ``camera_roll_deg`` against ``lateral_error``. The two aren't
         perfectly orthogonal (yaw leaks a little into lateral_error via
         ``lookahead_distance*tan(yaw_deg)``; roll's contribution to
         heading_error varies with row) so expect 2-3 rounds alternating
         between them before both settle near the noise floor.
      5. Repeat the whole procedure independently 2-3 times (re-square the
         robot on the line each time). If the converged values drift
         meaningfully between attempts, that's a loose mount (a mechanical
         problem), not a calibration-precision problem -- fix the bracket
         before trusting a number.
      6. Once repeatable, hardcode into ``real_line_follower.yaml`` next to
         ``camera_pitch_deg``, updating its comment from UNCONFIRMED to
         CONFIRMED with the date and this method, mirroring
         ``camera_pitch_deg``'s own CONFIRMED-2026-08-03 comment.

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
        roll_deg=roll_deg,
        yaw_deg=yaw_deg,
    )


def _mount_rotation_matrix(p: CameraParams) -> np.ndarray:
    """3x3 rotation R mapping a camera-frame ray ``(xc, yc, zc)`` (OpenCV
    convention: x=right, y=down, z=forward/optical-axis) to mount-frame
    ``(d_right, d_down, d_forward)``, i.e. ``R @ [xc, yc, zc]``.

    Composed as ``R = R_yaw @ R_pitch @ R_roll`` -- roll innermost (a twist
    about the camera module's own lens barrel, independent of how the
    bracket is subsequently tilted), pitch next (this file's original
    pitch-only rotation, unchanged), yaw outermost (how the whole
    already-pitched bracket is clocked left/right relative to the chassis).
    This order is a physically-motivated choice for a two-source mount
    misalignment, not the only valid one -- but at ``roll_deg=yaw_deg=0.0``
    both ``R_roll``/``R_yaw`` are exactly the identity regardless of where
    they sit in the product, so ``R`` reduces to exactly the pitch matrix
    below for *any* composition order: every existing deployment
    (Gazebo/MuJoCo presets, this robot's own tuned ``pitch_deg``) keeps
    today's exact behavior unchanged.

    The pitch block is hand-rolled (not ``Rotation.from_euler('x', ...)``)
    so it stays byte-identical to this file's original two-line trig --
    scipy's quaternion-based composition differs from it by ~1 ULP at
    nonzero angles, which is physically meaningless but pointless to
    introduce into an otherwise-unchanged, already-tuned rotation.

    Sign convention (with the robot centered/heading-aligned on the line,
    so the line sits near the image's center column): positive
    ``roll_deg`` produces a negative, roughly-distance-independent
    ``lateral_error`` bias; positive ``yaw_deg`` produces a positive
    ``heading_error`` bias that grows with lookahead distance (to close
    approximation ``heading_error ~= radians(yaw_deg)`` when centered) --
    see ``turtlebot3_burger_real_camera()``'s docstring for the calibration
    procedure this is meant to cancel. If a first guess at either sign
    makes the corresponding error grow instead of shrink, flip it -- this
    was derived from the code above, not confirmed against the physical
    mount, so getting the sign right on the first try isn't guaranteed.
    """
    roll = Rotation.from_euler("z", p.roll_deg, degrees=True).as_matrix()
    ph = math.radians(p.pitch_deg)
    pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(ph), math.sin(ph)],
        [0.0, -math.sin(ph), math.cos(ph)],
    ])
    yaw = Rotation.from_euler("y", p.yaw_deg, degrees=True).as_matrix()
    return yaw @ pitch @ roll


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

    R = _mount_rotation_matrix(p)
    d_right, d_down, d_forward = R @ np.stack([xc, yc, zc], axis=0)

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

    d_forward = x_ground
    d_right = y_ground
    d_down = np.full_like(x_ground, p.mount_height)

    R = _mount_rotation_matrix(p)
    xc, yc, zc = R.T @ np.stack([d_right, d_down, d_forward], axis=0)

    u = np.full_like(x_ground, np.nan)
    v = np.full_like(x_ground, np.nan)
    in_front = zc > 1e-6
    u = np.where(in_front, cx + fx * xc / np.where(in_front, zc, 1.0), u)
    v = np.where(in_front, cy + fy * yc / np.where(in_front, zc, 1.0), v)
    return u, v
