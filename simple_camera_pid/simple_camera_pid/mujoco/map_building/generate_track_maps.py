"""Generate 3 new closed-loop tracks with mixed road types.

Translated from ``generate_track_maps.m``: three smooth closed curves (polar
harmonic sums) with increasing difficulty:
  track_easy   - long straights + gentle curves
  track_medium - straights + curves + two sharp turns
  track_hard   - continuous S-bends + a hairpin

Registers the three new map keys into
``runtime/mujoco_scene.py``'s ``KNOWN_SCENES`` so
``set_turtlebot3_mujoco_scene("track_easy")`` works afterwards, mirroring the
MATLAB script's self-registration into ``resolve_turtlebot3_mujoco_scene.m``.

Usage
-----
    python3 -m simple_camera_pid.mujoco.map_building.generate_track_maps
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..runtime import mujoco_scene

_SCENE_XML_TEMPLATE = """<mujoco model="{key}_turtlebot3_burger_visual_scene">
  <compiler angle="radian" meshdir="../shared/assets/meshes" texturedir="../shared/assets/textures" autolimits="true" />
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" iterations="50" />
  <visual>
    <global offwidth="640" offheight="480" />
    <quality shadowsize="1024" />
    <headlight ambient="0.40 0.40 0.40" diffuse="0.90 0.90 0.90" specular="0.10 0.10 0.10" />
  </visual>
  <asset>
    <texture name="{key}_texture" type="2d" file="{png_name}" />
    <material name="{key}_material" texture="{key}_texture" texrepeat="1 1" texuniform="false" rgba="1 1 1 1" reflectance="0" />
    <mesh name="simple_camera_track_surface" file="simple_camera_track_surface.obj" />
    <include file="turtlebot3_burger_vehicle_assets.xml" />
  </asset>
  <worldbody>
    <light name="overhead_a" pos="-1.0 0 3.5" dir="0.3 0 -1" diffuse="0.95 0.95 0.95" specular="0.05 0.05 0.05" />
    <light name="overhead_b" pos=" 1.0 0 3.5" dir="-0.3 0 -1" diffuse="0.95 0.95 0.95" specular="0.05 0.05 0.05" />
    <geom name="{key}_visual_surface" type="mesh" mesh="simple_camera_track_surface" material="{key}_material" contype="0" conaffinity="0" group="1" />
    <geom name="{key}_floor" type="box" pos="0 0 -0.005" size="{half_span:.1f} {half_span:.1f} 0.005" rgba="0.04 0.04 0.05 1" friction="1.0 0.005 0.0001" />
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
class _TrackDef:
    key: str
    disp: str
    r0: float
    harmonics: list[tuple[float, float, float]]  # (k, amplitude, phase)


TRACKS = [
    _TrackDef("track_easy", "Track Easy (long straights + gentle curves)", 1.55,
              [(2, 0.08, 0.0)]),
    _TrackDef("track_medium", "Track Medium (straights + curves + 2 sharp turns)", 1.50,
              [(2, 0.14, 0.0), (3, 0.10, 1.2), (5, 0.05, 0.4)]),
    _TrackDef("track_hard", "Track Hard (S-bends + hairpin)", 1.45,
              [(2, 0.12, 0.3), (3, 0.16, 0.8), (4, 0.10, 2.1), (6, 0.09, 1.5)]),
]


def generate_track_maps(repo_root: Path) -> None:
    tex_dir = repo_root / "model" / "mujoco" / "shared" / "assets" / "textures"
    scene_dir = repo_root / "model" / "mujoco" / "turtlebot3"
    if not tex_dir.is_dir():
        raise FileNotFoundError(f"Texture dir not found: {tex_dir}")
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene dir not found: {scene_dir}")

    image_size = 2048
    half_span = 2.2
    line_width_px = 45

    registered = []
    for track in TRACKS:
        png_name = f"{track.key}_track.png"
        scene_name = f"{track.key}_turtlebot3_burger_visual_scene.xml"

        wx, wy = _track_curve(track.r0, track.harmonics)
        mask = _rasterize_loop(wx, wy, image_size, half_span, line_width_px)
        rgb = np.repeat((mask * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        cv2.imwrite(str(tex_dir / png_name), rgb)  # grayscale-as-RGB, channel order irrelevant

        pos, quat = _spawn_from_curve(wx, wy)
        xml = _SCENE_XML_TEMPLATE.format(
            key=track.key, png_name=png_name, half_span=half_span,
            pos=f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.3f}",
            quat=f"{quat[0]:.10f} {quat[1]:.1f} {quat[2]:.1f} {quat[3]:.10f}",
        )
        (scene_dir / scene_name).write_text(xml)

        print(f"Generated {track.key:<14} : {png_name}  +  {scene_name}")
        registered.append({"key": track.key, "filename": scene_name, "display": track.disp})
        mujoco_scene.register_scene(track.key, scene_name, track.disp)

    module_path = Path(mujoco_scene.__file__)
    if mujoco_scene.register_scene_keys_in_source(module_path, registered):
        print(f"Registered {len(registered)} new keys into {module_path.name}.")
    else:
        print("KNOWN_SCENES already contains the new keys; skipping registration.")

    print('\nDone. Use e.g. mujoco_scene.resolve_turtlebot3_mujoco_scene(repo_root, requested_map="track_easy")')


def _track_curve(r0: float, harmonics: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * math.pi, 4000)
    r = np.ones_like(theta)
    for k, a, phi in harmonics:
        r = r + a * np.sin(k * theta + phi)
    r = r0 * r
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x[-1], y[-1] = x[0], y[0]  # explicit closure
    return x, y


def _rasterize_loop(wx: np.ndarray, wy: np.ndarray, n: int, half_span: float, line_w_px: float) -> np.ndarray:
    col = (wx + half_span) / (2 * half_span) * n
    row = (1 - (wy + half_span) / (2 * half_span)) * n
    pts = np.stack([col, row], axis=1).astype(np.int32)

    mask = np.zeros((n, n), dtype=np.uint8)
    closed_pts = np.vstack([pts, pts[:1]])
    radius = max(1, round(line_w_px / 2))
    cv2.polylines(mask, [closed_pts], isClosed=True, color=1, thickness=2 * radius, lineType=cv2.LINE_8)
    return mask.astype(bool)


def _spawn_from_curve(wx: np.ndarray, wy: np.ndarray) -> tuple[tuple[float, float, float],
                                                                  tuple[float, float, float, float]]:
    px, py = wx[0], wy[0]
    dx, dy = wx[1] - wx[0], wy[1] - wy[0]
    yaw = math.atan2(dy, dx)
    quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    pos = (float(px), float(py), 0.010)
    return pos, quat


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    generate_track_maps(repo_root)


if __name__ == "__main__":
    main()
