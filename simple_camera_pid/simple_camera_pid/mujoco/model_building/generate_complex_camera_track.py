"""Generate the closed MuJoCo line-following "complex" map texture.

Translated from ``generate_complex_camera_track.m``: a non-self-intersecting
harmonic loop with straights, changing-radius bends, and chicanes, rendered
onto the shared 4.4 m-square ground texture.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class ComplexTrack:
    texture_file: Path
    path_length: float
    minimum_radius: float
    line_width: float
    spawn_position: tuple[float, float, float]
    spawn_yaw: float


def generate_complex_camera_track(repo_root: Path) -> ComplexTrack:
    texture_file = (
        repo_root / "model" / "mujoco" / "shared" / "assets" / "textures" / "complex_camera_track.png"
    )

    image_size = 2048
    world_half_width = 2.2
    line_width = 0.09
    sample_count = 24000

    t = np.linspace(0, 2 * np.pi, sample_count)
    radial_x = 1.35 + 0.25 * np.cos(3 * t) + 0.08 * np.sin(5 * t)
    radial_y = 1.20 + 0.20 * np.cos(3 * t) + 0.06 * np.sin(5 * t)
    x = radial_x * np.cos(t)
    y = -radial_y * np.sin(t)

    column = np.round((x + world_half_width) / (2 * world_half_width) * (image_size - 1)).astype(int)
    row = np.round((world_half_width - y) / (2 * world_half_width) * (image_size - 1)).astype(int)
    column = np.clip(column, 0, image_size - 1)
    row = np.clip(row, 0, image_size - 1)

    centerline = np.zeros((image_size, image_size), dtype=bool)
    centerline[row, column] = True
    line_radius_px = round(0.5 * line_width / (2 * world_half_width) * image_size)
    structure = _disk_structure(line_radius_px)
    track_mask = ndimage.binary_dilation(centerline, structure=structure)

    track_image = np.repeat((track_mask * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    cv2.imwrite(str(texture_file), track_image)  # binary mask -> channel order is irrelevant

    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)
    curvature = np.abs(dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, np.finfo(float).eps)
    segment_length = np.hypot(np.diff(x), np.diff(y))

    track = ComplexTrack(
        texture_file=texture_file,
        path_length=float(segment_length.sum()),
        minimum_radius=1.0 / float(curvature.max()),
        line_width=line_width,
        spawn_position=(-1.227151, -0.446432, 0.010),
        spawn_yaw=np.pi / 2,
    )
    print(f"Generated complex camera track: {texture_file}")
    print(f"Path {track.path_length:.3f} m, minimum radius {track.minimum_radius:.3f} m, "
          f"line width {track.line_width:.3f} m.")
    return track


def _disk_structure(radius: int) -> np.ndarray:
    """Equivalent of MATLAB's ``strel("disk", radius, 0)`` (no line-approximation)."""
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx ** 2 + yy ** 2) <= radius ** 2


def main() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    generate_complex_camera_track(repo_root)


if __name__ == "__main__":
    main()
