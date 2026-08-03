"""Procedural random-track generator — translation of
runtime/gen_simple_track_scene.m. Produces a closed-loop white line on a dark
ground texture (ellipse or capsule/stadium shape), rasterized as a PNG, plus a
self-contained MuJoCo scene XML (embeds the turtlebot3 body/actuators/sensors
via <include>, same as the fixed map pool) and a starting pose on the line
(tangent-aligned yaw).

Wired into domain_rand.py as the GenTrack option: when enabled, a fresh random
track is generated every episode (overwriting the same file by default) so the
residual policy sees effectively unlimited track variety instead of
memorizing a fixed map pool. MATLAB's rl_domain_randomization.m equivalent
(GenTrack) is disabled by default there too, and had NOT been ported to this
package until now (domain_rand.py's docstring used to say so explicitly).
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.draw import disk as _sk_disk
from skimage.io import imsave

WORLD_HALF = 2.2       # ground mesh half-width (m) — matches the shared mesh
MARGIN = 0.55          # loop-to-boundary safety margin (m)
IMG_SIZE = 512
LINE_WIDTH_PX = 26.0
N_LOOP_POINTS = 720


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

    if out_model_dir is None or out_tex_dir is None:
        # mirrors scenes.py's convention: wherever lsac/ sits inside the HPC
        # bundle, model/mujoco/... is expected 2 directories up (see that
        # module's docstring — this is a bundle-layout assumption, not
        # auto-detected).
        mujoco_dir = Path(__file__).resolve().parents[2] / "model" / "mujoco"
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
    imsave(str(png_file), img, check_contrast=False)

    # Start pose: random point on the loop, yaw = tangent direction
    # (parameter-increasing / counter-clockwise, matching the MATLAB source).
    i0 = int(rng.integers(len(wx)))
    sx0, sy0 = float(wx[i0]), float(wy[i0])
    i1 = (i0 + 1) % len(wx)
    yaw = math.atan2(wy[i1] - sy0, wx[i1] - sx0)
    quat = _yaw_to_quat(yaw)
    start_pos = (sx0, sy0, 0.010)

    xml = _build_scene_xml(name, png_name, start_pos, quat)
    scene_file.write_text(xml, encoding="utf-8")

    return TrackInfo(name=name, map_key=map_key, scene_file=str(scene_file),
                     png_file=str(png_file), shape=resolved_shape,
                     start_pos=start_pos, start_quat=quat,
                     semi=(a, b), center=(cx, cy))
