"""Procedural random-track generators — translation of runtime/scene/
gen_simple_track_scene.m, gen_complex_track_scene.m, and
gen_wandering_track_scene.m (the last two added to the MATLAB side
2026-07-22, ported here 2026-07-23). All three produce a closed-loop white
line on a dark ground texture, rasterized as a PNG, plus a self-contained
MuJoCo scene XML (embeds the turtlebot3 body/actuators/sensors via <include>,
same as the fixed map pool) and a starting pose on the line (tangent-aligned
yaw):

  gen_simple_track_scene    — ellipse / capsule, regular geometry
  gen_complex_track_scene   — polyline + independently-radiused corner arcs,
                              sharp-cornered polygon look
  gen_wandering_track_scene — everywhere-smooth organic winding loop (cubic
                              spline over a periodically-padded random radial
                              profile), no straight segments or sharp corners

Wired into domain_rand.py as the GenTrack option: when enabled, a fresh random
track is generated every episode (overwriting the same file by default) so the
residual policy sees effectively unlimited track variety instead of
memorizing a fixed map pool. domain_rand.py's `gen_track_generator` field
selects among the three (or "random" to pick one each episode), mirroring
MATLAB's rl_dr_defaults.m `dr.GenTrack.Generator`.
"""

from __future__ import annotations

import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from skimage.draw import disk as _sk_disk
from skimage.io import imsave

WORLD_HALF = 2.2       # ground mesh half-width (m) — matches the shared mesh
MARGIN = 0.55          # loop-to-boundary safety margin (m)
IMG_SIZE = 512
LINE_WIDTH_PX = 26.0   # gen_simple_track_scene default (see that function)
N_LOOP_POINTS = 720


def _atomic_replace(dst: Path, write_fn) -> None:
    """Write via write_fn(tmp_path_str) to a same-directory temp file, then
    os.replace() it onto dst. dst is reused every episode when overwrite=True
    (gen_track_name is a fixed filename) and a reader (MjModel.from_xml_path,
    in this same process's next reset()) can otherwise observe a
    partially-written/empty file if it races the in-place write — this
    happened in production (empty PNG -> training crash at ep2400, 2026-07-22
    genpath_roi30 run). os.replace is atomic on POSIX so the reader only ever
    sees the old or the fully-written new file, never a partial one."""
    # keep dst's suffix on the temp name (not just prefixed) so format-sniffing
    # writers like imageio/skimage, which pick a backend from the extension,
    # still recognize it (e.g. "foo.tmp123.png", not "foo.png.tmp123").
    tmp = dst.with_name(f".{dst.stem}.tmp{os.getpid()}{dst.suffix}")
    write_fn(str(tmp))
    os.replace(tmp, dst)


def _resolve_dirs(out_model_dir: str | Path | None,
                  out_tex_dir: str | Path | None) -> tuple[Path, Path]:
    """Shared by all three generators: default to model/mujoco/turtlebot3
    (scenes) and model/mujoco/shared/assets/textures (PNGs), resolved
    relative to wherever lsac/ sits inside the HPC bundle (mirrors
    scenes.py's convention: model/mujoco/... expected 2 directories up)."""
    if out_model_dir is None or out_tex_dir is None:
        mujoco_dir = Path(__file__).resolve().parents[3] / "model" / "mujoco"
        if out_model_dir is None:
            out_model_dir = mujoco_dir / "turtlebot3"
        if out_tex_dir is None:
            out_tex_dir = mujoco_dir / "shared" / "assets" / "textures"
    out_model_dir = Path(out_model_dir)
    out_tex_dir = Path(out_tex_dir)
    if not out_model_dir.is_dir():
        raise FileNotFoundError(f"Scene dir not found: {out_model_dir}")
    if not out_tex_dir.is_dir():
        raise FileNotFoundError(f"Texture dir not found: {out_tex_dir}")
    return out_model_dir, out_tex_dir


@dataclass
class TrackInfo:
    name: str
    map_key: str
    scene_file: str
    png_file: str
    shape: str
    start_pos: tuple[float, float, float]
    start_quat: tuple[float, float, float, float]  # (w, x, y, z)
    semi: tuple[float, float]
    center: tuple[float, float]


@dataclass
class ComplexTrackInfo:
    name: str
    map_key: str
    scene_file: str
    png_file: str
    start_pos: tuple[float, float, float]
    start_quat: tuple[float, float, float, float]
    vertices: np.ndarray       # (num_corners, 2) world-coord polygon vertices
    corner_radii: np.ndarray   # (num_corners,) effective fillet radius used


@dataclass
class WanderingTrackInfo:
    name: str
    map_key: str
    scene_file: str
    png_file: str
    start_pos: tuple[float, float, float]
    start_quat: tuple[float, float, float, float]
    control_angles: np.ndarray
    control_radii: np.ndarray


def _resolve_shape(shape: str, rng: np.random.Generator) -> str:
    s = shape.strip().lower()
    if s == "random":
        return "ellipse" if rng.integers(2) == 0 else "capsule"
    if s in ("ellipse", "capsule"):
        return s
    raise ValueError(f"shape must be ellipse|capsule|random, got {shape!r}")


def _capsule_loop(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Stadium/capsule loop: two straight segments + two half-circle ends.
    a = overall half-length, b = half-width (== end-cap radius)."""
    r = b
    sx = max(0.05, a - r)
    seg = n // 4
    t1x = np.linspace(-sx, sx, seg)
    t1y = np.full(seg, r)
    ph = np.linspace(math.pi / 2, -math.pi / 2, seg)
    t2x = sx + r * np.cos(ph)
    t2y = r * np.sin(ph)
    t3x = np.linspace(sx, -sx, seg)
    t3y = np.full(seg, -r)
    ph2 = np.linspace(-math.pi / 2, -3 * math.pi / 2, seg)
    t4x = -sx + r * np.cos(ph2)
    t4y = r * np.sin(ph2)
    return (np.concatenate([t1x, t2x, t3x, t4x]),
            np.concatenate([t1y, t2y, t3y, t4y]))


def _enforce_min_angle_gap(angles: np.ndarray, min_gap: float) -> np.ndarray:
    """Sorted angle sequence: if a (non-wraparound) adjacent gap is below
    min_gap, push the later point forward — prevents the corner-rounding
    formula from seeing a near-zero edge length. The first/last wraparound
    gap is deliberately left unadjusted (matches gen_complex_track_scene.m's
    `if i == n, continue`), implicitly bounded by the overall 2*pi constraint."""
    angles = angles.copy()
    n = len(angles)
    for i in range(n - 1):
        j = i + 1
        gap = (angles[j] - angles[i]) % (2 * math.pi)
        if gap < min_gap:
            angles[j] = angles[i] + min_gap
    return angles % (2 * math.pi)


def _rasterize_line(wx: np.ndarray, wy: np.ndarray, size: int,
                    line_width: float) -> np.ndarray:
    """World-coord closed loop -> dark-background RGB image with a white
    line (rasterizeLine()/stampDisk() in the MATLAB source)."""
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[..., 0] = 43
    img[..., 1] = 43
    img[..., 2] = 46  # ~#2b2b2e

    u = (wx + WORLD_HALF) / (2 * WORLD_HALF)
    v = (wy + WORLD_HALF) / (2 * WORLD_HALF)
    px = u * size
    py = (1 - v) * size

    mask = np.zeros((size, size), dtype=bool)
    half = max(1.0, round(line_width / 2))

    def stamp_segment(x1: float, y1: float, x2: float, y2: float) -> None:
        d = math.hypot(x2 - x1, y2 - y1)
        n = max(2, math.ceil(d))
        xs = np.linspace(x1, x2, n)
        ys = np.linspace(y1, y2, n)
        for x, y in zip(xs, ys):
            rr, cc = _sk_disk((y, x), half, shape=(size, size))
            mask[rr, cc] = True

    for k in range(len(px) - 1):
        stamp_segment(px[k], py[k], px[k + 1], py[k + 1])
    stamp_segment(px[-1], py[-1], px[0], py[0])  # close the loop

    img[mask] = 235
    return img


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _build_scene_xml(name: str, png_name: str,
                     start_pos: tuple[float, float, float],
                     quat: tuple[float, float, float, float]) -> str:
    q = "%.10f %.10f %.10f %.10f" % quat
    p = "%.6f %.6f %.6f" % start_pos
    return f'''<mujoco model="{name}_turtlebot3_burger_visual_scene">
  <compiler angle="radian" meshdir="../shared/assets/meshes" texturedir="../shared/assets/textures" autolimits="true" />
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" iterations="50" />
  <visual>
    <global offwidth="640" offheight="480" />
    <quality shadowsize="1024" />
    <headlight ambient="0.40 0.40 0.40" diffuse="0.90 0.90 0.90" specular="0.10 0.10 0.10" />
  </visual>
  <asset>
    <texture name="{name}_texture" type="2d" file="{png_name}" />
    <material name="{name}_material" texture="{name}_texture" texrepeat="1 1" texuniform="false" rgba="1 1 1 1" reflectance="0" />
    <mesh name="simple_camera_track_surface" file="simple_camera_track_surface.obj" />
    <include file="turtlebot3_burger_vehicle_assets.xml" />
  </asset>
  <worldbody>
    <light name="overhead_a" pos="-1.0 0 3.5" dir="0.3 0 -1" diffuse="0.95 0.95 0.95" specular="0.05 0.05 0.05" />
    <light name="overhead_b" pos=" 1.0 0 3.5" dir="-0.3 0 -1" diffuse="0.95 0.95 0.95" specular="0.05 0.05 0.05" />
    <geom name="{name}_visual_surface" type="mesh" mesh="simple_camera_track_surface" material="{name}_material" contype="0" conaffinity="0" group="1" />
    <geom name="{name}_floor" type="box" pos="0 0 -0.005" size="2.2 2.2 0.005" rgba="0.04 0.04 0.05 1" friction="1.0 0.005 0.0001" />
    <body name="turtlebot3" pos="{p}" quat="{q}">
      <include file="turtlebot3_burger_vehicle_body.xml" />
    </body>
  </worldbody>
  <actuator>
    <include file="turtlebot3_burger_vehicle_actuators.xml" />
  </actuator>
  <sensor>
    <include file="turtlebot3_burger_vehicle_sensors.xml" />
  </sensor>
</mujoco>
'''


def gen_simple_track_scene(
    name: str = "",
    shape: str = "random",
    difficulty: float = 0.4,
    img_size: int = IMG_SIZE,
    line_width: float = LINE_WIDTH_PX,
    out_model_dir: str | Path | None = None,
    out_tex_dir: str | Path | None = None,
    overwrite: bool = True,
    rng: np.random.Generator | None = None,
) -> TrackInfo:
    """Generate one random closed-loop track scene (ellipse or capsule).

    Mirrors gen_simple_track_scene.m: same geometry formulas, same
    rasterization approach, same scene-XML template (Burger body/actuators/
    sensors via <include>), so tracks generated here load with the same
    lsac.scenes / lsac.vision / lsac.controller code as the fixed map pool.

    difficulty in [0, 1]: higher -> smaller, tighter-turning loop.
    overwrite=True (default) reuses `name` every call so repeated episodes
    don't fill the disk with one file per episode — pass a fresh `name` (or
    overwrite=False) if you actually want to keep every generated track.
    """
    rng = rng or np.random.default_rng()
    out_model_dir, out_tex_dir = _resolve_dirs(out_model_dir, out_tex_dir)

    if not name:
        name = f"simple_rand_{secrets.token_hex(2)}"
    map_key = name.lower()
    png_name = f"{name}_track.png"
    scene_name = f"{name}_turtlebot3_burger_visual_scene.xml"
    png_file = out_tex_dir / png_name
    scene_file = out_model_dir / scene_name
    if not overwrite and scene_file.exists():
        raise FileExistsError(f"Scene exists: {scene_file}")

    resolved_shape = _resolve_shape(shape, rng)
    difficulty = float(np.clip(difficulty, 0.0, 1.0))

    # Ground mesh is WORLD_HALF*2 m square; keep the loop in the central
    # region with a margin, shrinking with difficulty (tighter turns).
    max_semi = WORLD_HALF - MARGIN
    a_hi, a_lo = max_semi, 0.9
    a = a_hi - (a_hi - a_lo) * difficulty * (0.6 + 0.4 * rng.random())
    b = a * (0.62 + 0.33 * rng.random())       # minor/major axis ratio
    theta = (2 * rng.random() - 1) * math.pi   # random overall rotation
    cmax = max(0.0, WORLD_HALF - MARGIN - max(a, b))
    cx = (2 * rng.random() - 1) * cmax * 0.6
    cy = (2 * rng.random() - 1) * cmax * 0.6

    tt = np.linspace(0.0, 2 * math.pi, N_LOOP_POINTS)
    if resolved_shape == "capsule":
        lx, ly = _capsule_loop(a, b, N_LOOP_POINTS)
    else:
        lx, ly = a * np.cos(tt), b * np.sin(tt)

    ct, st = math.cos(theta), math.sin(theta)
    wx = ct * lx - st * ly + cx
    wy = st * lx + ct * ly + cy

    img = _rasterize_line(wx, wy, round(img_size), line_width)
    _atomic_replace(png_file, lambda p: imsave(p, img, check_contrast=False))

    # Start pose: random point on the loop, yaw = tangent direction
    # (parameter-increasing / counter-clockwise, matching the MATLAB source).
    i0 = int(rng.integers(len(wx)))
    sx0, sy0 = float(wx[i0]), float(wy[i0])
    i1 = (i0 + 1) % len(wx)
    yaw = math.atan2(wy[i1] - sy0, wx[i1] - sx0)
    quat = _yaw_to_quat(yaw)
    start_pos = (sx0, sy0, 0.010)

    xml = _build_scene_xml(name, png_name, start_pos, quat)
    _atomic_replace(scene_file, lambda p: Path(p).write_text(xml, encoding="utf-8"))

    return TrackInfo(name=name, map_key=map_key, scene_file=str(scene_file),
                     png_file=str(png_file), shape=resolved_shape,
                     start_pos=start_pos, start_quat=quat,
                     semi=(a, b), center=(cx, cy))


def gen_complex_track_scene(
    name: str = "",
    num_corners: int = 8,
    difficulty: float = 0.5,
    min_corner_radius: float = 0.15,
    max_corner_radius: float = 0.55,
    img_size: int = IMG_SIZE,
    line_width: float = 4.0,
    out_model_dir: str | Path | None = None,
    out_tex_dir: str | Path | None = None,
    overwrite: bool = True,
    rng: np.random.Generator | None = None,
) -> ComplexTrackInfo:
    """Generate one closed-loop track combining straights with
    independently-radiused corner arcs — translation of
    runtime/scene/gen_complex_track_scene.m.

    Unlike gen_simple_track_scene (pure ellipse / two-straight+two-half-circle
    capsule), this splits the loop into num_corners vertices connected by
    straight edges, each corner filleted with its OWN randomly sampled
    radius — sharp and gentle turns mixed on the same track, a much more
    irregular silhouette, with a thinner default line width.

    Algorithm (matches the MATLAB source's 4 steps):
      1) num_corners vertices roughly evenly spread around a center (angle
         step + jitter, then sorted and min-gap-enforced so the polygon
         can't self-intersect), radius per vertex also randomized.
      2) Per-vertex signed turn angle (for the fillet formula) and edge
         lengths from the resulting polygon skeleton.
      3) Per-vertex independent random fillet radius -> tangent length
         (R*tan(|turn|/2)); a 3-pass relax shrinks any pair of adjacent
         tangent lengths that would together exceed 85% of their shared
         edge, preventing neighboring fillets from overlapping/self-crossing.
      4) Straight segments alternating with fillet arcs, sampled and closed
         into the final loop; closure is geometric, no extra stitching.

    difficulty in [0, 1]: higher -> more vertex-angle jitter and corner radii
    skewed toward the lower bound (tighter turns more common).
    """
    rng = rng or np.random.default_rng()
    if max_corner_radius <= min_corner_radius:
        raise ValueError("max_corner_radius must exceed min_corner_radius")
    if num_corners < 4:
        raise ValueError("num_corners must be >= 4")
    out_model_dir, out_tex_dir = _resolve_dirs(out_model_dir, out_tex_dir)

    if not name:
        name = f"complex_rand_{secrets.token_hex(2)}"
    map_key = name.lower()
    png_name = f"{name}_track.png"
    scene_name = f"{name}_turtlebot3_burger_visual_scene.xml"
    png_file = out_tex_dir / png_name
    scene_file = out_model_dir / scene_name
    if not overwrite and scene_file.exists():
        raise FileExistsError(f"Scene exists: {scene_file}")

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    n = num_corners
    eps = float(np.finfo(float).eps)

    # 1) Vertices: n points around the center, angle and radius both random.
    max_r = WORLD_HALF - MARGIN
    base_r = max_r * (0.55 + 0.25 * rng.random())
    radius_jitter_frac = 0.15 + 0.25 * difficulty
    angle_step = 2 * math.pi / n
    angle_jitter_frac = 0.25 + 0.45 * difficulty
    base_angles = np.arange(n) * angle_step
    angles = base_angles + (2 * rng.random(n) - 1) * angle_step * angle_jitter_frac * 0.5
    angles = np.sort(np.mod(angles, 2 * math.pi))
    angles = _enforce_min_angle_gap(angles, math.radians(10.0))

    vert_radii = base_r * (1.0 + (2 * rng.random(n) - 1) * radius_jitter_frac)
    vx = vert_radii * np.cos(angles)
    vy = vert_radii * np.sin(angles)

    theta0 = (2 * rng.random() - 1) * math.pi
    ct0, st0 = math.cos(theta0), math.sin(theta0)
    V = np.stack([ct0 * vx - st0 * vy, st0 * vx + ct0 * vy], axis=1)
    V = V + 0.10 * base_r * (2 * rng.random(2) - 1)

    # 2) Per-vertex signed turn angle + edge length (edge i = vertex i -> i+1).
    turn_angle = np.zeros(n)
    edge_len = np.zeros(n)
    for i in range(n):
        i_prev, i_next = (i - 1) % n, (i + 1) % n
        d_in = V[i] - V[i_prev]
        d_out = V[i_next] - V[i]
        edge_len[i] = float(np.linalg.norm(d_out))
        d_in = d_in / max(eps, float(np.linalg.norm(d_in)))
        d_out = d_out / max(eps, float(np.linalg.norm(d_out)))
        turn_angle[i] = math.atan2(d_in[0] * d_out[1] - d_in[1] * d_out[0],
                                   float(np.dot(d_in, d_out)))

    # 3) Per-vertex random fillet radius -> tangent length, relaxed against
    # neighboring-edge overlap.
    target_radius = min_corner_radius + (max_corner_radius - min_corner_radius) * (
        rng.random(n) ** (1.0 + difficulty))
    tan_dist = np.zeros(n)
    for i in range(n):
        if abs(turn_angle[i]) > math.radians(1.0):
            tan_dist[i] = target_radius[i] * math.tan(abs(turn_angle[i]) / 2.0)
    for _ in range(3):
        for i in range(n):
            i_next = (i + 1) % n
            avail = 0.85 * edge_len[i]
            s = tan_dist[i] + tan_dist[i_next]
            if s > avail and s > eps:
                f = avail / s
                tan_dist[i] *= f
                tan_dist[i_next] *= f
    eff_radius = np.zeros(n)
    for i in range(n):
        if abs(turn_angle[i]) > math.radians(1.0):
            eff_radius[i] = tan_dist[i] / math.tan(abs(turn_angle[i]) / 2.0)

    # 4) "straight -> arc -> straight -> arc ..." sample sequence.
    wx: list[float] = []
    wy: list[float] = []
    for i in range(n):
        i_prev, i_next = (i - 1) % n, (i + 1) % n
        d_in = V[i] - V[i_prev]
        d_in = d_in / max(eps, float(np.linalg.norm(d_in)))
        d_out = V[i_next] - V[i]
        d_out = d_out / max(eps, float(np.linalg.norm(d_out)))

        t_in = V[i] - d_in * tan_dist[i]
        t_out = V[i] + d_out * tan_dist[i]

        wx.append(float(t_in[0])); wy.append(float(t_in[1]))

        if eff_radius[i] > eps:
            perp = np.array([-d_in[1], d_in[0]]) * np.sign(turn_angle[i])
            center = t_in + eff_radius[i] * perp
            start_ang = math.atan2(t_in[1] - center[1], t_in[0] - center[0])
            num_arc_pts = max(4, round(abs(turn_angle[i]) * 24))
            sweep = np.linspace(0.0, turn_angle[i], num_arc_pts)
            arc_ang = start_ang + sweep
            wx.extend((center[0] + eff_radius[i] * np.cos(arc_ang)).tolist())
            wy.extend((center[1] + eff_radius[i] * np.sin(arc_ang)).tolist())
        else:
            wx.append(float(V[i, 0])); wy.append(float(V[i, 1]))

        wx.append(float(t_out[0])); wy.append(float(t_out[1]))

    wx_arr, wy_arr = np.asarray(wx), np.asarray(wy)

    img = _rasterize_line(wx_arr, wy_arr, round(img_size), line_width)
    _atomic_replace(png_file, lambda p: imsave(p, img, check_contrast=False))

    i0 = int(rng.integers(len(wx_arr)))
    sx0, sy0 = float(wx_arr[i0]), float(wy_arr[i0])
    i1 = (i0 + 1) % len(wx_arr)
    yaw = math.atan2(wy_arr[i1] - sy0, wx_arr[i1] - sx0)
    quat = _yaw_to_quat(yaw)
    start_pos = (sx0, sy0, 0.010)

    xml = _build_scene_xml(name, png_name, start_pos, quat)
    _atomic_replace(scene_file, lambda p: Path(p).write_text(xml, encoding="utf-8"))

    return ComplexTrackInfo(name=name, map_key=map_key, scene_file=str(scene_file),
                            png_file=str(png_file), start_pos=start_pos,
                            start_quat=quat, vertices=V, corner_radii=eff_radius)


def gen_wandering_track_scene(
    name: str = "",
    num_control: int = 6,
    difficulty: float = 0.5,
    base_radius: float | None = None,
    img_size: int = IMG_SIZE,
    line_width: float = 4.0,
    out_model_dir: str | Path | None = None,
    out_tex_dir: str | Path | None = None,
    overwrite: bool = True,
    rng: np.random.Generator | None = None,
) -> WanderingTrackInfo:
    """Generate one smooth closed-loop track by cubic-spline-interpolating a
    random radial profile — translation of runtime/scene/
    gen_wandering_track_scene.m.

    Unlike gen_simple_track_scene (ellipse/capsule) and gen_complex_track_scene
    (straight edges + independently-radiused corner arcs, sharp silhouette),
    this produces an everywhere-smooth organic winding loop: no straight
    segments, no sharp corners, curvature varies continuously along the path.

    Algorithm: num_control angles are spaced evenly around the center (NOT
    jittered — jitter would break the angle-to-arclength mapping used for the
    spline; all the irregularity comes from the radius at each angle).
    Radius = base + random offset (offset amplitude set by difficulty,
    renormalized so the largest |offset| exactly hits the target amplitude).
    The first/last 3 control points are periodically padded (moved to
    -2*pi/+2*pi) before a cubic spline interpolates angle -> radius, so the
    curve sees neighbors across the 0/2*pi seam and closes with no visible
    slope discontinuity.

    difficulty in [0, 1]: higher -> larger radial perturbation amplitude.
    """
    rng = rng or np.random.default_rng()
    if num_control < 4:
        raise ValueError("num_control must be >= 4")
    out_model_dir, out_tex_dir = _resolve_dirs(out_model_dir, out_tex_dir)

    if not name:
        name = f"wandering_rand_{secrets.token_hex(2)}"
    map_key = name.lower()
    png_name = f"{name}_track.png"
    scene_name = f"{name}_turtlebot3_burger_visual_scene.xml"
    png_file = out_tex_dir / png_name
    scene_file = out_model_dir / scene_name
    if not overwrite and scene_file.exists():
        raise FileExistsError(f"Scene exists: {scene_file}")

    difficulty = float(np.clip(difficulty, 0.0, 1.0))

    # 1)+2) Control angles evenly spaced; radius perturbed per angle (same
    # amp/normalization trick as the archived gen_random_local_path.m, here
    # applied to a radial offset around a center instead of a lateral offset
    # along forward X).
    max_r = WORLD_HALF - MARGIN
    base_r = float(base_radius) if base_radius is not None else (
        max_r * (0.55 + 0.20 * rng.random()))
    amp = base_r * 0.5 * (0.15 + 0.85 * difficulty ** 2)

    nc = num_control
    control_angles = np.arange(nc) * (2 * math.pi / nc)
    radial_offset = (2 * rng.random(nc) - 1) * amp
    peak = float(np.max(np.abs(radial_offset)))
    if peak > np.finfo(float).eps:
        radial_offset = radial_offset * (amp / peak)
    control_radii = np.maximum(0.3, base_r + radial_offset)

    # 3) Periodic padding + cubic spline, smooth/seamless closure.
    pad_n = min(3, nc - 1)
    angles_ext = np.concatenate([control_angles[-pad_n:] - 2 * math.pi,
                                 control_angles,
                                 control_angles[:pad_n] + 2 * math.pi])
    radii_ext = np.concatenate([control_radii[-pad_n:], control_radii,
                                control_radii[:pad_n]])

    num_samples = 720
    theta = np.linspace(0.0, 2 * math.pi, num_samples + 1)[:-1]  # drop dup 0/2pi
    r_sample = np.maximum(0.3, CubicSpline(angles_ext, radii_ext)(theta))

    # 4) Back to world coordinates (random overall rotation for variety);
    # closure is automatic from the angle parameterization.
    theta0 = (2 * rng.random() - 1) * math.pi
    wx = r_sample * np.cos(theta + theta0)
    wy = r_sample * np.sin(theta + theta0)

    img = _rasterize_line(wx, wy, round(img_size), line_width)
    _atomic_replace(png_file, lambda p: imsave(p, img, check_contrast=False))

    i0 = int(rng.integers(len(wx)))
    sx0, sy0 = float(wx[i0]), float(wy[i0])
    i1 = (i0 + 1) % len(wx)
    yaw = math.atan2(wy[i1] - sy0, wx[i1] - sx0)
    quat = _yaw_to_quat(yaw)
    start_pos = (sx0, sy0, 0.010)

    xml = _build_scene_xml(name, png_name, start_pos, quat)
    _atomic_replace(scene_file, lambda p: Path(p).write_text(xml, encoding="utf-8"))

    return WanderingTrackInfo(name=name, map_key=map_key, scene_file=str(scene_file),
                              png_file=str(png_file), start_pos=start_pos,
                              start_quat=quat, control_angles=control_angles,
                              control_radii=control_radii)
