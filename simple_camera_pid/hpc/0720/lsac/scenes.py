"""Scene resolution — translation of resolve_turtlebot3_mujoco_scene.m
(built-in map registry only; the Simulink model-workspace/env-var priority
chain collapses to an explicit map key or scene path here)."""

from __future__ import annotations

import os
from pathlib import Path

# python/ sits next to model/ inside train_bundle (mirrors the MATLAB layout
# where scenes resolve via modelDir/../../../model/mujoco).
_DEFAULT_SCENE_DIR = Path(__file__).resolve().parents[2] / "model" / "mujoco" / "turtlebot3"

_KNOWN_SCENES = {
    "simple": "simple_camera_track_turtlebot3_burger_visual_scene.xml",
    "complex": "complex_camera_track_turtlebot3_burger_visual_scene.xml",
    "ellipse": "ellipse_camera_track_turtlebot3_burger_visual_scene.xml",
    "training": "training_oval_track_turtlebot3_burger_visual_scene.xml",
    "track_easy": "track_easy_turtlebot3_burger_visual_scene.xml",
    "track_medium": "track_medium_turtlebot3_burger_visual_scene.xml",
    "track_hard": "track_hard_turtlebot3_burger_visual_scene.xml",
    "winding": "winding_camera_track_turtlebot3_burger_visual_scene.xml",
}

MAP_KEYS = tuple(_KNOWN_SCENES)


def scene_dir() -> Path:
    override = os.environ.get("TURTLEBOT3_MUJOCO_SCENE_DIR", "")
    return Path(override) if override else _DEFAULT_SCENE_DIR


def resolve_scene(map_key_or_path: str) -> str:
    """Map key (e.g. 'training', 'track_hard') or explicit XML path → abs path."""
    key = map_key_or_path.strip().lower()
    if key in _KNOWN_SCENES:
        path = scene_dir() / _KNOWN_SCENES[key]
    else:
        path = Path(map_key_or_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"MuJoCo scene not found: {path}\n"
            f"Valid map keys: {', '.join(MAP_KEYS)}")
    return str(path)
