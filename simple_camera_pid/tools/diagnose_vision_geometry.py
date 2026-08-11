#!/usr/bin/env python3
"""Why heading_error is unusable on the real robot, and what ROI depth fixes it.

Reproduces the 2026-08-11 diagnosis. With the robot STATIONARY on a straight
line, ``heading_error`` was swinging across its whole -1.000..+1.000 range
frame to frame, while ``lateral_error`` stayed steady. That is not a tuning
problem and not an exposure problem; it is the conditioning of the fit
``heading_error`` is derived from:

``_fit_poly_lookahead`` fits ``y = ax^2 + bx + c`` to the sliding-window
centroids in ground coordinates and returns ``atan(2a*x_la + b)`` -- a SLOPE.
At ``roi_bottom_fraction`` 0.3, this camera (15 deg pitch, 0.133 m mount)
sees only about 0.08 m of ground DEPTH, and ``lookahead_distance`` 0.20 lands
on the near edge of that range. Fitting a slope over an 0.08 m baseline to
centroids scattered ~0.018 m laterally leaves the derivative dominated by
noise: the fitted ``a`` measured mean 4.6 / sd 17.8 across frames that are
pixel-for-pixel nearly identical.

Three modes, each on live frames pulled from the compressed camera topic (the
raw topic does not fit the WiFi link -- see vision_node.py's module
docstring, and note that saturating the link has previously dropped the
operator's SSH session):

    exposure   Is the IMAGE moving, or only the vision output? Reports
               per-frame brightness, then replays the same frames through
               LineFollowerVision. A locked exposure holds the ROI's p90
               within a couple of counts; auto-exposure hunting on this
               near-black-carpet scene moved it by ~140 (the fault 80f951c
               locked the exposure to fix).
    fit        Unpacks the debug vector to show the fit inputs (baseline,
               scatter, point count) next to the coefficients and the
               heading_error they produce.
    sweep      Replays ONE burst of frames through several
               roi_bottom_fraction/lookahead_distance pairs, so the
               comparison is on identical input and needs neither the robot
               to move nor the Pi's node to restart.
    horizon    No camera needed: where the ground actually is, per image row,
               for the current mount geometry.

Usage
-----
    python3 diagnose_vision_geometry.py exposure
    python3 diagnose_vision_geometry.py fit
    python3 diagnose_vision_geometry.py sweep
    python3 diagnose_vision_geometry.py horizon

2026-08-11 results on this robot, stationary, found_rate 1.000 throughout:

    roi  lookahead  x-span   |a| med  heading sd  frame-to-frame
    0.3  0.20       0.071 m  13.84    0.678       0.363
    0.4  0.20       0.166 m   4.35    0.504       0.244
    0.5  0.30       0.317 m   0.97    0.145       0.112
    0.6  0.30       0.448 m   0.29    0.085       0.084
    0.7  0.30       1.111 m   0.03    0.046       0.026

0.6/0.30 was then confirmed live on the robot: heading sd 0.337 -> 0.101, no
more +/-1.0 clamping, steering_error sd 0.091 -> 0.025. It cost frame rate:
/line_follower/local_path fell from 20.0 Hz to 9.3 Hz on the Pi, which is a
real trade (the PID's dt assumption still holds, since control_node's timer is
independent, but the information feeding it is half as fresh).

0.7 scores best here and should still NOT be used: run ``horizon`` to see why.
Its x-span jumps 2.5x for a 0.1 ROI increase because the ROI is reaching
toward the horizon, where one pixel row spans tens of centimetres of ground
and the extra smoothness is not information.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# Mount geometry these diagnostics assume. Keep in step with
# config/real/real_line_follower.yaml.
IMAGE_W, IMAGE_H = 640, 480
FOVY_DEG, MOUNT_H, PITCH_DEG = 48.6867, 0.133, 15.0
DEFAULT_TOPIC = '/camera/image_raw/compressed'


def _camera():
    from simple_camera_pid.common.camera_geometry import turtlebot3_burger_real_camera
    return turtlebot3_burger_real_camera(
        image_width=IMAGE_W, image_height=IMAGE_H, fovy_deg=FOVY_DEG,
        mount_height=MOUNT_H, pitch_deg=PITCH_DEG, roll_deg=0.0, yaw_deg=0.0,
    )


def _vision(roi: float, lookahead: float):
    from simple_camera_pid.common.vision import LineFollowerVision
    return LineFollowerVision(
        roi_bottom_fraction=roi, waypoint_count=30, lookahead_distance=lookahead,
        lateral_gain=0.6, heading_gain=0.35, curvature_gain=0.04,
        min_brightness=105.0, max_saturation=0.20, min_pixels=30.0,
        error_scale=500.0, roi_widen_step=0.2, roi_widen_max=0.7,
        otsu_min_contrast=30.0, adaptive_brightness_fraction=0.0,
        image_height=IMAGE_H, image_width=IMAGE_W, camera=_camera(),
        flip_vertical=False,
    )


def grab(topic: str, count: int):
    """Collect ``count`` decoded RGB frames off the compressed camera topic."""
    import cv2
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage

    frames = []

    class Grabber(Node):
        def __init__(self):
            super().__init__('vision_geometry_diag')
            self.create_subscription(CompressedImage, topic, self.cb, 10)

        def cb(self, msg):
            if len(frames) >= count:
                raise KeyboardInterrupt
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    rclpy.init()
    node = Grabber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if len(frames) < 5:
        raise SystemExit(f'only {len(frames)} frames from {topic} -- is the camera up, '
                         f'and is ROS_DOMAIN_ID right?')
    return frames


def mode_exposure(args):
    import cv2
    frames = grab(args.topic, args.count)
    print(f'{len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}\n')

    stats = []
    for f in frames:
        v = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)[:, :, 2]
        roi = v[int(v.shape[0] * 0.7):, :]
        stats.append([v.mean(), np.percentile(roi, 50), np.percentile(roi, 90),
                      np.percentile(roi, 99)])
    s = np.asarray(stats)
    print('brightness stability (robot must be stationary for this to mean anything):')
    for i, n in enumerate(['whole-frame mean', 'ROI p50', 'ROI p90', 'ROI p99']):
        c = s[:, i]
        print(f'  {n:<17} mean {c.mean():6.1f}  sd {c.std():5.2f}  spread {c.max()-c.min():5.1f}')
    spread = s[:, 2].max() - s[:, 2].min()
    print(f'\n  -> ROI p90 spread {spread:.1f} counts: '
          + ('EXPOSURE IS MOVING' if spread > 15 else 'exposure looks locked'))

    vis = _vision(args.roi, args.lookahead)
    o = np.asarray([[(r := vis.step(f)).heading_error, r.lateral_error,
                     1.0 if r.found else 0.0] for f in frames])
    print(f'\nsame frames through LineFollowerVision (roi {args.roi}, '
          f'lookahead {args.lookahead}):')
    print(f'  found_rate {o[:,2].mean():.3f}')
    print(f'  heading  sd {o[:,0].std():.4f}   range {o[:,0].min():+.3f}..{o[:,0].max():+.3f}')
    print(f'  lateral  sd {o[:,1].std():.4f}   range {o[:,1].min():+.3f}..{o[:,1].max():+.3f}')
    dh = np.abs(np.diff(o[:, 0]))
    print(f'  frame-to-frame |delta heading|: median {np.median(dh):.4f}  max {dh.max():.4f}')
    print('\n  A steady image with an unsteady heading means the instability is in the\n'
          '  fit, not the picture -- run the "fit" mode next.')


def mode_fit(args):
    frames = grab(args.topic, args.count)
    vis = _vision(args.roi, args.lookahead)
    rows = []
    for f in frames:
        r = vis.step(f)
        d = r.debug
        xs, ys = d[7::2], d[8::2]
        m = xs != 0.0
        rows.append([
            d[0], d[4], d[5], d[6], d[1],
            (xs[m].max() - xs[m].min()) if m.sum() > 1 else 0.0,
            ys[m].std() if m.sum() > 1 else 0.0,
            r.heading_error, r.lateral_error,
        ])
    R = np.asarray(rows)
    names = ['valid_count', 'a', 'b', 'c', 'lookahead_x',
             'x-span (m)', 'y scatter (m)', 'heading_error', 'lateral_error']
    print(f'{len(frames)} frames, roi {args.roi}, lookahead {args.lookahead}\n')
    for i, n in enumerate(names):
        v = R[:, i]
        print(f'  {n:<15} mean {v.mean():+9.4f}  sd {v.std():8.4f}  '
              f'min {v.min():+9.4f}  max {v.max():+9.4f}')
    span, scatter = R[:, 5].mean(), R[:, 6].mean()
    print(f'\nconditioning: a slope fitted over {span:.3f} m of baseline to points '
          f'scattered {scatter:.3f} m laterally.')
    if span > 0:
        print(f'  one scatter-sigma across that baseline is '
              f'{np.degrees(np.arctan(scatter/span)):.0f} deg of heading_error.')
    print(f'  heading sd / lateral sd = {R[:,7].std()/max(1e-9, R[:,8].std()):.0f}x')


def mode_sweep(args):
    frames = grab(args.topic, args.count)
    print(f'{len(frames)} frames replayed through each setting. Lower heading sd is\n'
          f'better, but found_rate must stay at 1.000 or the setting is unusable.\n')
    print(f'{"roi":>5} {"look":>5} {"found":>6} {"xspan":>7} {"|a| med":>8} '
          f'{"head sd":>8} {"dhead med":>10} {"lat sd":>7}')
    for roi in (0.3, 0.4, 0.5, 0.6, 0.7):
        for lookahead in (0.20, 0.30):
            vis = _vision(roi, lookahead)
            out = []
            for f in frames:
                r = vis.step(f)
                xs = r.debug[7::2]
                m = xs != 0.0
                out.append([r.heading_error, r.lateral_error, 1.0 if r.found else 0.0,
                            (xs[m].max() - xs[m].min()) if m.sum() > 1 else 0.0,
                            abs(r.debug[4])])
            o = np.asarray(out)
            print(f'{roi:>5.1f} {lookahead:>5.2f} {o[:,2].mean():>6.3f} {o[:,3].mean():>7.3f} '
                  f'{np.median(o[:,4]):>8.2f} {o[:,0].std():>8.3f} '
                  f'{np.median(np.abs(np.diff(o[:,0]))):>10.4f} {o[:,1].std():>7.3f}')


def mode_horizon(args):
    """No robot needed. Where the ground is, per image row."""
    from simple_camera_pid.common.camera_geometry import pixel_to_ground
    cam = _camera()
    rows = np.arange(0.0, IMAGE_H, 1.0)
    X, _ = pixel_to_ground(np.full_like(rows, IMAGE_W / 2.0), rows, cam)
    usable = np.isfinite(X)
    first = int(rows[usable][0]) if usable.any() else None
    print(f'mount: pitch {PITCH_DEG} deg, height {MOUNT_H} m, fovy {FOVY_DEG} deg, '
          f'{IMAGE_W}x{IMAGE_H}')
    print(f'max_ground_range guard: {getattr(cam, "max_ground_range", "n/a")} m\n')
    print(f'first row from the top with usable ground: {first}  '
          f'(everything above it projects above the horizon)')
    if first is not None:
        print(f'  -> hard ceiling roi_bottom_fraction = 1 - {first}/{IMAGE_H} = '
              f'{1 - first/IMAGE_H:.2f}\n')
    print(' roi   top row   ground x there   cm of ground per pixel row')
    for roi in (0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0):
        r = IMAGE_H * (1 - roi)
        x = pixel_to_ground(np.array([IMAGE_W / 2.0]), np.array([r]), cam)[0][0]
        x2 = pixel_to_ground(np.array([IMAGE_W / 2.0]), np.array([r + 1.0]), cam)[0][0]
        if not np.isfinite(x):
            print(f'{roi:5.2f} {int(r):9d}       --- no ground (above horizon / beyond range)')
        else:
            print(f'{roi:5.2f} {int(r):9d}   {x:10.3f} m   {abs(x-x2)*100:8.1f}')
    print('\nDistance diverges hyperbolically toward the horizon, so the useful ceiling\n'
          'is well below the geometric one -- past ~0.7 a single pixel row spans more\n'
          'ground than the line is wide. Raising the camera or reducing its pitch is\n'
          'what changes this table; roi_bottom_fraction only selects rows within it.')


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('exposure', 'fit', 'sweep', 'horizon'))
    ap.add_argument('--count', type=int, default=60, help='frames to collect (default 60)')
    ap.add_argument('--topic', default=DEFAULT_TOPIC)
    ap.add_argument('--roi', type=float, default=0.3)
    ap.add_argument('--lookahead', type=float, default=0.20)
    args = ap.parse_args(argv)
    {'exposure': mode_exposure, 'fit': mode_fit,
     'sweep': mode_sweep, 'horizon': mode_horizon}[args.mode](args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
