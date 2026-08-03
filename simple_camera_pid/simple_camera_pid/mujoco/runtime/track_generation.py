"""Procedurally generate a random simple closed-loop MuJoCo track.

Translated from ``gen_simple_track_scene.m``: renders a smooth closed white
loop (ellipse or capsule) onto a PNG line texture and writes a matching MJCF
scene XML with the robot spawned on the line, tangent to it. Used both for
one-off track generation and, via ``GenTrackConfig``, for per-episode fresh
tracks during RL training (see ``runtime/domain_randomization.py``).

Rasterization uses ``cv2.polylines`` instead of the MATLAB source's manual
disk-stamping loop — visually equivalent, not pixel-identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_SCENE_XML_TEMPLATE = """<mujoco model="{name}_turtlebot3_burger_visual_scene">
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
    <body name="turtlebot3" pos="{pos}" quat="{quat}">
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
"""


@dataclass
class TrackInfo:
    name: str
    map_key: str
    scene_file: Path
    png_file: Path
    shape: str
    start_pos: tuple[float, float, float]
    start_quat: tuple[float, float, float, float]
    semi_axes: tuple[float, float]
    center: tuple[float, float]


def gen_simple_track_scene(
    repo_root: Path,
    name: str = "",
    shape: str = "random",
    rng: np.random.Generator | None = None,
    img_size: int = 512,
    line_width_px: float = 26.0,
    difficulty: float = 0.4,
    out_model_dir: Path | None = None,
    out_tex_dir: Path | None = None,
    overwrite: bool = True,
) -> TrackInfo:
    rng = rng if rng is not None else np.random.default_rng()

    mujoco_dir = repo_root / "model" / "mujoco"
    model_dir = out_model_dir or (mujoco_dir / "turtlebot3")
    tex_dir = out_tex_dir or (mujoco_dir / "shared" / "assets" / "textures")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Scene dir not found: {model_dir}")
    if not tex_dir.is_dir():
        raise FileNotFoundError(f"Texture dir not found: {tex_dir}")

    if not name:
        name = f"simple_rand_{rng.integers(0, 65536):04x}"
    map_key = name.lower()
    png_name = f"{name}_track.png"
    scene_name = f"{name}_turtlebot3_burger_visual_scene.xml"
    png_file = tex_dir / png_name
    scene_file = model_dir / scene_name
    if not overwrite and scene_file.is_file():
        raise FileExistsError(f"Scene exists: {scene_file}")

    resolved_shape = _resolve_shape(shape, rng)

    # Ground mesh is 4.4x4.4 m (world [-2.2, 2.2]); keep the loop centered
    # with a safety margin so the camera line-follow stays in bounds.
    world_half = 2.2
    margin = 0.55
    max_semi = world_half - margin
    a_hi, a_lo = max_semi, 0.9
    a = a_hi - (a_hi - a_lo) * difficulty * (0.6 + 0.4 * rng.random())
    b = a * (0.62 + 0.33 * rng.random())
    theta = (2 * rng.random() - 1) * math.pi
    cmax = max(0.0, world_half - margin - max(a, b))
    cx = (2 * rng.random() - 1) * cmax * 0.6
    cy = (2 * rng.random() - 1) * cmax * 0.6

    t = np.linspace(0, 2 * math.pi, 720)
    if resolved_shape == "capsule":
        lx, ly = _capsule_loop(a, b, t.size)
    else:
        lx = a * np.cos(t)
        ly = b * np.sin(t)

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    wx = lx * cos_t - ly * sin_t + cx
    wy = lx * sin_t + ly * cos_t + cy

    img = _rasterize_line(wx, wy, world_half, img_size, line_width_px)
    cv2.imwrite(str(png_file), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    i0 = int(rng.integers(wx.size))
    sx, sy = wx[i0], wy[i0]
    i_next = (i0 + 1) % wx.size
    yaw = math.atan2(wy[i_next] - sy, wx[i_next] - sx)
    quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    start_pos = (float(sx), float(sy), 0.010)

    xml = _SCENE_XML_TEMPLATE.format(
        name=name, png_name=png_name,
        pos=f"{start_pos[0]:.6f} {start_pos[1]:.6f} {start_pos[2]:.6f}",
        quat=f"{quat[0]:.10f} {quat[1]:.10f} {quat[2]:.10f} {quat[3]:.10f}",
    )
    scene_file.write_text(xml)

    return TrackInfo(
        name=name, map_key=map_key, scene_file=scene_file, png_file=png_file, shape=resolved_shape,
        start_pos=start_pos, start_quat=quat, semi_axes=(float(a), float(b)), center=(float(cx), float(cy)),
    )


def _resolve_shape(shape: str, rng: np.random.Generator) -> str:
    shape = shape.strip().lower()
    if shape == "random":
        return "ellipse" if rng.integers(2) == 0 else "capsule"
    if shape in ("ellipse", "capsule"):
        return shape
    raise ValueError(f"Shape must be ellipse|capsule|random, got '{shape}'.")


def _capsule_loop(a: float, b: float, npts: int) -> tuple[np.ndarray, np.ndarray]:
    """Capsule ring: two straight segments + two semicircular end caps."""
    r = b
    sx = max(0.05, a - r)
    seg = npts // 4
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
    return np.concatenate([t1x, t2x, t3x, t4x]), np.concatenate([t1y, t2y, t3y, t4y])


def _rasterize_line(
    wx: np.ndarray, wy: np.ndarray, world_half: float, sz: int, line_w: float
) -> np.ndarray:
    """Rasterize a closed world-space polyline as a white line on a dark PNG."""
    u = (wx + world_half) / (2 * world_half)
    v = (wy + world_half) / (2 * world_half)
    px = u * sz
    py = (1 - v) * sz

    mask = np.zeros((sz, sz), dtype=np.uint8)
    half = max(1, round(line_w / 2))
    pts = np.stack([px, py], axis=1).astype(np.int32)
    closed_pts = np.vstack([pts, pts[:1]])
    cv2.polylines(mask, [closed_pts], isClosed=True, color=1, thickness=2 * half, lineType=cv2.LINE_8)

    bg = np.empty((sz, sz, 3), dtype=np.uint8)
    bg[..., 0] = 43
    bg[..., 1] = 43
    bg[..., 2] = 46
    img = bg.copy()
    img[mask.astype(bool)] = 235
    return img
