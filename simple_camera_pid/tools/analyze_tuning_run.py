#!/usr/bin/env python3
"""Score a real-robot tuning run from ``/line_follower/debug``.

Every real-hardware tuning decision recorded in
``config/real/real_line_follower.yaml`` so far (the kp 5.0 -> 2.0 -> 1.0 ->
1.5 -> 0.7 walk, the 2026-08-04 segmentation root-cause, the recovery_gain
fix) was argued from numbers like "0.70 steering-sign-flips/sec" and
"|wheel_diff| max 4.85" that were computed ad hoc and then thrown away with
the scratch directory they lived in. This file is that computation, kept: one
definition of each metric, applied identically to every trial, so a sweep
produces a comparable table instead of a set of remembered impressions.

Reads the 7-float diagnostic both real-robot deployment shapes publish
(``real/line_follower_node.py`` single-process and ``real/control_node.py``
split -- identical layout, see their ``_publish_diagnostics``)::

    [steering_error, lateral_error, heading_error, confidence, found,
     wheel_left, wheel_right]

It also accepts the 5-float ``/line_follower/local_path`` message
``real/vision_node.py`` publishes (``local_path_msg.py``: the same five
leading fields, no wheel commands). That topic exists with **no motors
involved at all**, which makes it the right input for the vision half of a
tuning session -- validating a ``min_brightness``/``max_saturation`` change
by found_rate before letting any gain sweep depend on it. The wheel-derived
rows report ``n/a`` for those runs.

Usage
-----
    # Live, while the robot drives -- prints a rolling window, then a full
    # summary on Ctrl-C. Needs no bag and no disk on the Pi.
    python3 analyze_tuning_run.py --live

    # Vision only, no motors: push the robot along the line by hand and
    # watch found_rate.
    python3 analyze_tuning_run.py --live --topic /line_follower/local_path

    # One recorded run.
    python3 analyze_tuning_run.py ~/bags/kp0.7_run1

    # A whole sweep, side by side (this is the point of the file).
    python3 analyze_tuning_run.py ~/bags/kp0.7_* ~/bags/kp1.0_* ~/bags/kp1.5_*

Record a bag to analyze later with::

    ros2 bag record -o ~/bags/kp0.7_run1 /line_follower/debug /cmd_vel

(``/camera/image_raw`` is deliberately not in that list: it is what makes
bags huge and what saturates the Pi's WiFi. Add it only when the question is
a *vision* question -- the 2026-08-04 false-positive root-cause needed the
frames, a gain sweep does not.)

Metric definitions
------------------
``sign_flips_per_s``
    Sign changes of ``steering_error``, over ``found`` samples only, ignoring
    samples inside ``--deadband`` so that sensor noise around a
    correctly-centered line does not register as oscillation. This is the
    primary oscillation number. Prior sessions' 0.70-0.78/s (kp 1.5-2.0,
    called oscillating) and 0.08/s (the clean pure-proportional diagnostic)
    were computed by hand and their exact deadband is unrecorded, so treat
    those as approximate landmarks and compare runs *within* one sweep, which
    this file does compute identically. ``sign_flips_raw_per_s`` (no
    deadband) is printed alongside for continuity with them.

``angular_sat_frac``
    Fraction of ``found`` samples with the angular command at its
    ``max_angular_speed`` ceiling. **Read this before concluding
    "underdamped, not saturating"**: the wheel commands cannot reveal
    saturation by their own ``max_wheel_speed`` limit, because the binding
    constraint is the earlier PID output clamp. With this robot's geometry a
    saturated command is exactly

        |wheel_right - wheel_left| = max_angular_speed * L / R
                                   = 1.0 * 0.160 / 0.033 = 4.848 rad/s

    which is where the 2026-08-06 log's "|wheel_diff| max 4.85, limit ~6.5+"
    number actually sits -- at the ceiling, not comfortably below it. That
    run's peak was saturated; what was never measured is how *often*, which
    is what this column answers.

``lateral_bias`` / ``heading_bias``
    Signed means, not magnitudes. On a symmetric track these should average
    out near zero; a persistent offset is the stationary mount bias the
    2026-08-07 session found (heading_error ~+0.21 to +0.34 rad) and is a
    ``camera_roll_deg``/``camera_yaw_deg`` calibration problem, not a gain
    problem -- see ``turtlebot3_burger_real_camera()``'s docstring in
    ``common/camera_geometry.py``. Chasing it with kp/kd will not remove it.

``found_rate`` / ``lost_episodes`` / ``lost_max_s``
    Vision health. A gain comparison is only meaningful between runs with
    comparable found_rate -- if one trial saw the line 60% of the time and
    another 95%, their steering statistics describe different problems.
    ``lost_max_s`` against ``lost_speed_stop_timeout`` (1.5 s) says whether
    the robot was actually coasting to a stop mid-run.

Caveats
-------
- The diagnostic carries no timestamp of its own, so bag mode uses rosbag2's
  receive time and live mode uses arrival time. Both are fine for rates and
  durations at this 20 Hz; neither is a precise loop-latency measurement.
- ``found``/wheel columns come from the controller's view of the frame it
  acted on. In the split deployment a stale-input tick (watchdog) is
  published as ``found=0`` -- indistinguishable here from a real line loss,
  by design, since the controller treats them identically.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# Field layout of the Float32MultiArray both real-robot nodes publish.
STEERING, LATERAL, HEADING, CONFIDENCE, FOUND, WHEEL_L, WHEEL_R = range(7)

DEFAULT_TOPIC = '/line_follower/debug'
# RobotConfig/ControllerConfig defaults for the real Burger. Overridable
# because the metrics that depend on them (angular command, saturation) are
# wrong and silently so if the run used different values.
DEFAULT_WHEEL_RADIUS = 0.033
DEFAULT_WHEEL_SEPARATION = 0.160
DEFAULT_MAX_ANGULAR_SPEED = 1.0


@dataclass
class Samples:
    """One run's raw columns, already time-ordered."""

    name: str
    t: np.ndarray            # seconds, relative to first sample
    data: np.ndarray         # (n, 7); wheel columns are zero when has_wheels
    has_wheels: bool = True  # False for the 5-float vision-only local_path

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size > 1 else 0.0


def _rows_to_samples(name: str, stamps: Sequence[float], rows: Sequence[Sequence[float]],
                     has_wheels: bool) -> Samples:
    t = np.asarray(stamps, dtype=np.float64)
    return Samples(name=name, t=t - t[0], data=np.asarray(rows, dtype=np.float64),
                   has_wheels=has_wheels)


def _unpack(data: Sequence[float]) -> Optional[tuple[List[float], bool]]:
    """A published diagnostic row -> (7 columns, has_wheels), or None if the
    message is neither of the two known layouts."""
    if len(data) >= 7:
        return [float(v) for v in data[:7]], True
    if len(data) == 5:
        return [float(v) for v in data] + [0.0, 0.0], False
    return None


def _sign_flips(values: np.ndarray, deadband: float) -> int:
    """Sign changes of ``values``, skipping anything inside ``deadband``.

    Samples within the deadband are dropped rather than treated as a third
    state, so a slow drift from clearly-left to clearly-right through the
    middle counts as one flip (it is one correction), while jitter that never
    leaves the deadband counts as none.
    """
    significant = values[np.abs(values) > deadband]
    if significant.size < 2:
        return 0
    signs = np.sign(significant)
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _lost_episodes(found: np.ndarray, t: np.ndarray) -> tuple[int, float]:
    """``(episode_count, longest_episode_seconds)`` for contiguous found==0 runs."""
    lost = found < 0.5
    if not np.any(lost):
        return 0, 0.0
    edges = np.diff(lost.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if lost[0]:
        starts.insert(0, 0)
    if lost[-1]:
        ends.append(len(lost) - 1)
    longest = max((t[e] - t[s] for s, e in zip(starts, ends)), default=0.0)
    return len(starts), float(longest)


def compute_metrics(
    samples: Samples,
    *,
    wheel_radius: float = DEFAULT_WHEEL_RADIUS,
    wheel_separation: float = DEFAULT_WHEEL_SEPARATION,
    max_angular_speed: float = DEFAULT_MAX_ANGULAR_SPEED,
    deadband: float = 0.02,
    sat_tolerance: float = 0.98,
) -> Dict[str, float]:
    """All published metrics for one run. Pure function over the columns, so
    it can be exercised on synthetic data without ROS."""
    t, d = samples.t, samples.data
    n = d.shape[0]
    duration = samples.duration
    found = d[:, FOUND]
    is_found = found > 0.5
    n_found = int(np.count_nonzero(is_found))

    steering = d[:, STEERING]
    steering_found = steering[is_found]
    # Only found frames: while lost, vision freezes/decays steering_error and
    # the controller runs its recovery branch, so those samples describe the
    # recovery path, not the closed-loop tracking this metric is about.
    flips = _sign_flips(steering_found, deadband)
    flips_raw = _sign_flips(steering_found, 0.0)

    # Vision-only runs carry no wheel commands; report their wheel-derived
    # rows as n/a rather than as a confident zero (a 0.000 saturation
    # fraction that only means "no data" is exactly the kind of number that
    # gets quoted back later as evidence).
    empty = np.empty(0)
    wheel_diff = d[:, WHEEL_R] - d[:, WHEEL_L] if samples.has_wheels else empty
    angular = wheel_radius * wheel_diff / wheel_separation
    linear = (wheel_radius * (d[:, WHEEL_L] + d[:, WHEEL_R]) / 2.0
              if samples.has_wheels else empty)
    angular_found = angular[is_found] if samples.has_wheels else empty
    sat_level = sat_tolerance * max_angular_speed
    sat_frac = (
        float(np.count_nonzero(np.abs(angular_found) >= sat_level)) / n_found
        if n_found and samples.has_wheels else float('nan')
    )

    lost_count, lost_max = _lost_episodes(found, t)

    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else float('nan')

    def _mean(x: np.ndarray) -> float:
        return float(np.mean(x)) if x.size else float('nan')

    return {
        'samples': float(n),
        'duration_s': duration,
        'rate_hz': (n - 1) / duration if duration > 0 else float('nan'),
        'found_rate': float(n_found) / n if n else float('nan'),
        'lost_episodes': float(lost_count),
        'lost_max_s': lost_max,
        'sign_flips_per_s': flips / duration if duration > 0 else float('nan'),
        'sign_flips_raw_per_s': flips_raw / duration if duration > 0 else float('nan'),
        'steering_rms': _rms(steering_found),
        'steering_p95': (
            float(np.percentile(np.abs(steering_found), 95)) if steering_found.size else float('nan')
        ),
        'lateral_bias': _mean(d[is_found, LATERAL]),
        'lateral_rms': _rms(d[is_found, LATERAL]),
        'heading_bias': _mean(d[is_found, HEADING]),
        'heading_rms': _rms(d[is_found, HEADING]),
        'angular_rms': _rms(angular_found),
        'angular_abs_max': float(np.max(np.abs(angular_found))) if angular_found.size else float('nan'),
        'angular_sat_frac': sat_frac,
        'wheel_diff_abs_max': float(np.max(np.abs(wheel_diff))) if wheel_diff.size else float('nan'),
        'linear_mean': _mean(linear),
        'confidence_mean': _mean(d[is_found, CONFIDENCE]),
    }


# --- report -----------------------------------------------------------------

# (key, label, format). Ordered as a reading order, not alphabetically: run
# sanity first, then vision health, then the oscillation verdict, then the
# supporting detail.
_ROWS: Sequence[tuple[str, str, str]] = (
    ('duration_s', 'duration (s)', '{:.1f}'),
    ('samples', 'samples', '{:.0f}'),
    ('rate_hz', 'rate (Hz)', '{:.1f}'),
    ('found_rate', 'found_rate', '{:.3f}'),
    ('lost_episodes', 'lost episodes', '{:.0f}'),
    ('lost_max_s', 'longest loss (s)', '{:.2f}'),
    ('sign_flips_per_s', 'sign flips /s', '{:.2f}'),
    ('sign_flips_raw_per_s', '  (no deadband)', '{:.2f}'),
    ('steering_rms', 'steering RMS', '{:.3f}'),
    ('steering_p95', 'steering |p95|', '{:.3f}'),
    ('angular_sat_frac', 'angular sat frac', '{:.3f}'),
    ('angular_abs_max', 'angular |max|', '{:.3f}'),
    ('angular_rms', 'angular RMS', '{:.3f}'),
    ('wheel_diff_abs_max', 'wheel_diff |max|', '{:.2f}'),
    ('lateral_bias', 'lateral bias', '{:+.3f}'),
    ('lateral_rms', 'lateral RMS', '{:.3f}'),
    ('heading_bias', 'heading bias', '{:+.3f}'),
    ('heading_rms', 'heading RMS', '{:.3f}'),
    ('linear_mean', 'linear mean (m/s)', '{:.3f}'),
    ('confidence_mean', 'confidence mean', '{:.3f}'),
)


def print_report(names: List[str], metrics: List[Dict[str, float]], max_angular_speed: float) -> None:
    label_width = max(len(label) for _, label, _ in _ROWS) + 2
    col_width = max(max((len(n) for n in names), default=8), 9) + 2

    header = ' ' * label_width + ''.join(n.rjust(col_width) for n in names)
    print(header)
    print('-' * len(header))
    for key, label, fmt in _ROWS:
        cells = []
        for m in metrics:
            value = m.get(key, float('nan'))
            cells.append(('n/a' if isinstance(value, float) and math.isnan(value)
                          else fmt.format(value)).rjust(col_width))
        print(label.ljust(label_width) + ''.join(cells))

    sat_diff = max_angular_speed * DEFAULT_WHEEL_SEPARATION / DEFAULT_WHEEL_RADIUS
    print()
    print(f'angular saturation ceiling: |wheel_diff| = {sat_diff:.3f} rad/s '
          f'at max_angular_speed={max_angular_speed}')
    print('lower sign flips /s = less oscillation; found_rate must be comparable '
          'between runs for the rest to mean anything')


# --- bag input --------------------------------------------------------------

def _detect_storage_id(path: Path) -> str:
    """rosbag2's own metadata is the only reliable source (jazzy defaults to
    mcap, older bags on this project's Pi are sqlite3)."""
    metadata = path / 'metadata.yaml'
    if metadata.is_file():
        try:
            import yaml
            info = yaml.safe_load(metadata.read_text())
            identifier = info['rosbag2_bagfile_information'].get('storage_identifier')
            if identifier:
                return str(identifier)
        except Exception:
            pass
    if any(path.glob('*.mcap')):
        return 'mcap'
    if any(path.glob('*.db3')):
        return 'sqlite3'
    return ''


def read_bag(path: Path, topic: str) -> Samples:
    """Read ``topic`` out of a rosbag2 directory into ``Samples``."""
    try:
        from rclpy.serialization import deserialize_message
        from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
        from std_msgs.msg import Float32MultiArray
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise SystemExit(
            f'bag mode needs a sourced ROS 2 environment ({exc}).\n'
            '  source /opt/ros/jazzy/setup.bash'
        ) from exc

    if not path.is_dir():
        raise SystemExit(f'not a rosbag2 directory: {path}')

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(path), storage_id=_detect_storage_id(path)),
        ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'),
    )
    available = {info.name for info in reader.get_all_topics_and_types()}
    if topic not in available:
        raise SystemExit(
            f'{path.name}: no {topic} in this bag. Recorded topics: '
            f'{", ".join(sorted(available)) or "(none)"}'
        )
    reader.set_filter(StorageFilter(topics=[topic]))

    stamps: List[float] = []
    rows: List[List[float]] = []
    has_wheels = True
    while reader.has_next():
        _, payload, stamp_ns = reader.read_next()
        unpacked = _unpack(deserialize_message(payload, Float32MultiArray).data)
        if unpacked is None:
            continue
        row, row_has_wheels = unpacked
        has_wheels &= row_has_wheels
        stamps.append(stamp_ns / 1e9)
        rows.append(row)

    if not rows:
        raise SystemExit(f'{path.name}: {topic} is present but empty')

    return _rows_to_samples(path.name, stamps, rows, has_wheels)


# --- live input -------------------------------------------------------------

def run_live(topic: str, window_s: float, report_every_s: float, metric_kwargs: dict) -> Samples:
    """Subscribe to ``topic`` and print a rolling window until Ctrl-C.

    The rolling line is the number to watch while the robot is actually
    driving -- it says whether the trial is worth finishing before the battery
    and the track time go into a run you already know is oscillating.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise SystemExit(
            f'live mode needs a sourced ROS 2 environment ({exc}).\n'
            '  source /opt/ros/jazzy/setup.bash'
        ) from exc

    stamps: List[float] = []
    rows: List[List[float]] = []
    state = {'has_wheels': True}

    class _Collector(Node):
        def __init__(self) -> None:
            super().__init__('line_follower_tuning_monitor')
            self.create_subscription(Float32MultiArray, topic, self._on_msg, 10)
            self.create_timer(report_every_s, self._on_report)
            self._t0: Optional[float] = None
            self.get_logger().info(f'watching {topic} (Ctrl-C for the full summary)')

        def _now(self) -> float:
            return self.get_clock().now().nanoseconds / 1e9

        def _on_msg(self, msg: Float32MultiArray) -> None:
            unpacked = _unpack(msg.data)
            if unpacked is None:
                return
            row, row_has_wheels = unpacked
            state['has_wheels'] &= row_has_wheels
            now = self._now()
            if self._t0 is None:
                self._t0 = now
            stamps.append(now - self._t0)
            rows.append(row)

        def _on_report(self) -> None:
            if len(rows) < 2:
                print('waiting for data on ' + topic, flush=True)
                return
            t = np.asarray(stamps)
            recent = t >= t[-1] - window_s
            window = Samples(name='live', t=t[recent], data=np.asarray(rows)[recent],
                             has_wheels=state['has_wheels'])
            if window.duration <= 0:
                return
            m = compute_metrics(window, **metric_kwargs)
            print(
                f'[{t[-1]:6.1f}s] last {window_s:.0f}s: '
                f'flips/s {m["sign_flips_per_s"]:5.2f}  '
                f'found {m["found_rate"]:5.3f}  '
                f'|steer| p95 {m["steering_p95"]:5.3f}  '
                f'sat {m["angular_sat_frac"]:5.3f}  '
                f'lat bias {m["lateral_bias"]:+6.3f}',
                flush=True,
            )

    rclpy.init()
    node = _Collector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if len(rows) < 2:
        raise SystemExit('no data received -- check the topic name and ROS_DOMAIN_ID')
    return _rows_to_samples('live', stamps, rows, state['has_wheels'])


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description='Score a real-robot line-following tuning run.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage\n-----\n', 1)[1] if 'Usage' in __doc__ else None,
    )
    parser.add_argument('bags', nargs='*', type=Path, help='rosbag2 directories to compare')
    parser.add_argument('--live', action='store_true',
                        help='subscribe instead of reading bags')
    parser.add_argument('--topic', default=DEFAULT_TOPIC)
    parser.add_argument('--window', type=float, default=10.0,
                        help='live rolling-window length in seconds (default 10)')
    parser.add_argument('--report-every', type=float, default=2.0,
                        help='live report period in seconds (default 2)')
    parser.add_argument('--deadband', type=float, default=0.02,
                        help='ignore |steering_error| below this when counting sign '
                             'flips (default 0.02)')
    parser.add_argument('--max-angular-speed', type=float, default=DEFAULT_MAX_ANGULAR_SPEED,
                        help='the run\'s max_angular_speed, for the saturation metric')
    parser.add_argument('--wheel-radius', type=float, default=DEFAULT_WHEEL_RADIUS)
    parser.add_argument('--wheel-separation', type=float, default=DEFAULT_WHEEL_SEPARATION)
    args = parser.parse_args(argv)

    if args.live == bool(args.bags):
        parser.error('pass either --live or one or more bag directories')

    metric_kwargs = dict(
        wheel_radius=args.wheel_radius,
        wheel_separation=args.wheel_separation,
        max_angular_speed=args.max_angular_speed,
        deadband=args.deadband,
    )

    if args.live:
        runs = [run_live(args.topic, args.window, args.report_every, metric_kwargs)]
    else:
        runs = [read_bag(bag, args.topic) for bag in args.bags]

    print()
    print_report([r.name for r in runs],
                 [compute_metrics(r, **metric_kwargs) for r in runs],
                 args.max_angular_speed)


if __name__ == '__main__':
    sys.exit(main())
