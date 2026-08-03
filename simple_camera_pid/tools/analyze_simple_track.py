#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from PIL import Image


def run_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    for row in mask[::4]:
        padded = np.r_[False, row, False]
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        lengths.extend(int(e - s) for s, e in zip(starts, ends) if 3 <= e - s <= 80)
    for col in mask[:, ::4].T:
        padded = np.r_[False, col, False]
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        lengths.extend(int(e - s) for s, e in zip(starts, ends) if 3 <= e - s <= 80)
    return lengths


def main() -> None:
    package_dir = Path(__file__).resolve().parents[1]
    workspace_dir = package_dir.parent.parent
    texture = (
        workspace_dir
        / 'model'
        / 'mujoco'
        / 'shared'
        / 'assets'
        / 'textures'
        / 'simple_camera_track.png'
    )
    image = Image.open(texture).convert('L')
    gray = np.asarray(image)
    mask = gray > 180
    ys, xs = np.nonzero(mask)
    lengths = run_lengths(mask)
    median_width = float(np.median(lengths)) if lengths else float('nan')

    print(f'texture: {texture}')
    print(f'image_size_px: {gray.shape[1]} x {gray.shape[0]}')
    print(f'white_fraction: {mask.mean():.4f}')
    print(f'white_bbox_px: x={xs.min()}..{xs.max()}, y={ys.min()}..{ys.max()}')
    print(f'median_line_width_px: {median_width:.1f}')
    print(f'median_line_width_m: {median_width * 4.4 / gray.shape[1]:.3f}')
    print('recommended_baseline:')
    print('  kp: 1.25')
    print('  ki: 0.0')
    print('  kd: 0.12')
    print('  k_heading: 0.55')
    print('  base_speed: 0.12')
    print('  max_angular_speed: 1.35')


if __name__ == '__main__':
    main()
