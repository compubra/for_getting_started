"""Resolve/select the MJCF scene used by the TurtleBot3 simulation.

Translated from ``resolve_turtlebot3_mujoco_scene.m`` and
``set_turtlebot3_mujoco_scene.m``. There is no Simulink model workspace or
``.slx`` Plant block here, so "selecting a scene" just means resolving a map
key or path to an absolute MJCF file that ``sim/turtlebot3_mujoco_env.py``
loads with ``mujoco.MjModel.from_xml_path``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# key -> (scene filename, display name). Extend via register_scene() (used by
# map_building/generate_track_maps.py, mirroring the MATLAB knownScenes()
# self-registration).
KNOWN_SCENES: dict[str, tuple[str, str]] = {
    "simple": ("simple_camera_track_turtlebot3_burger_visual_scene.xml", "Simple camera track"),
    "complex": ("complex_camera_track_turtlebot3_burger_visual_scene.xml", "Complex chicane"),
    "ellipse": ("ellipse_camera_track_turtlebot3_burger_visual_scene.xml", "Ellipse camera track"),
    "training": ("training_oval_track_turtlebot3_burger_visual_scene.xml",
                 "Training oval (compact, RL-optimised)"),
    "track_easy": ("track_easy_turtlebot3_burger_visual_scene.xml",
                   "Track Easy (long straights + gentle curves)"),
    "track_medium": ("track_medium_turtlebot3_burger_visual_scene.xml",
                     "Track Medium (straights + curves + 2 sharp turns)"),
    "track_hard": ("track_hard_turtlebot3_burger_visual_scene.xml",
                   "Track Hard (S-bends + hairpin)"),
    "winding": ("winding_camera_track_turtlebot3_burger_visual_scene.xml",
                "Winding 5-lobe serpentine ring"),
}


def register_scene(key: str, filename: str, display_name: str) -> None:
    """Add a new map key to the in-process registry (does not persist to disk)."""
    KNOWN_SCENES[key.lower()] = (filename, display_name)


def _turtlebot3_dir(repo_root: Path) -> Path:
    return repo_root / "model" / "mujoco" / "turtlebot3"


def resolve_turtlebot3_mujoco_scene(
    repo_root: Path,
    requested_map: str = "",
    requested_scene: str = "",
) -> tuple[Path, str, str]:
    """Resolve a scene file, returning (scene_path, map_key, display_name).

    Selection priority mirrors the MATLAB function (with the Simulink-Plant
    and model-workspace layers removed, since Python has neither):
      1. ``requested_scene`` argument
      2. ``TURTLEBOT3_MUJOCO_SCENE`` environment variable
      3. ``requested_map`` argument
      4. ``TURTLEBOT3_MUJOCO_MAP`` environment variable
      5. the "simple" track (fallback)
    """
    turtlebot3_dir = _turtlebot3_dir(repo_root)

    scene = requested_scene or os.environ.get("TURTLEBOT3_MUJOCO_SCENE", "")
    if scene:
        return _resolve_scene_file(scene, turtlebot3_dir)

    map_key = requested_map or os.environ.get("TURTLEBOT3_MUJOCO_MAP", "")
    if map_key:
        return _resolve_map_key(map_key, turtlebot3_dir)

    return _resolve_map_key("simple", turtlebot3_dir)


def _resolve_map_key(requested_map: str, turtlebot3_dir: Path) -> tuple[Path, str, str]:
    key = requested_map.strip().lower()
    if key not in KNOWN_SCENES:
        raise ValueError(
            f"Unknown TurtleBot3 MuJoCo map '{requested_map}'. "
            f"Valid map keys: {', '.join(sorted(KNOWN_SCENES))}."
        )
    filename, display_name = KNOWN_SCENES[key]
    scene_file = turtlebot3_dir / filename
    if not scene_file.is_file():
        raise FileNotFoundError(f"MuJoCo scene file was not found: {scene_file}")
    return scene_file, key, display_name


def _resolve_scene_file(requested_scene: str, turtlebot3_dir: Path) -> tuple[Path, str, str]:
    requested = Path(requested_scene.strip())
    candidates = [requested, turtlebot3_dir / requested.name]
    scene_file = next((c for c in candidates if c.is_file()), None)
    if scene_file is None:
        raise FileNotFoundError(
            f"MuJoCo scene file was not found: {requested_scene}\n"
            "Use a full XML path, a known file name, or set TURTLEBOT3_MUJOCO_SCENE."
        )

    scene_name = scene_file.name
    for key, (filename, display_name) in KNOWN_SCENES.items():
        if filename == scene_name:
            return scene_file, key, display_name
    return scene_file, "custom", scene_name


def register_scene_keys_in_source(module_path: Path, tracks: list[dict]) -> bool:
    """Append new map keys into this module's ``KNOWN_SCENES`` source text.

    Mirrors ``generate_track_maps.m``'s ``registerScenes`` self-modification
    of ``resolve_turtlebot3_mujoco_scene.m`` — used by
    ``map_building/generate_track_maps.py`` so newly generated tracks are
    importable by key on the next process start. Idempotent: skipped if the
    first track's key is already present. Returns True if the file was
    modified.
    """
    text = module_path.read_text()
    if any(f'"{t["key"]}":' in text for t in tracks):
        return False

    entries = "".join(
        f'    "{t["key"]}": ("{t["filename"]}", "{t["display"]}"),\n' for t in tracks
    )
    new_text = re.sub(
        r"(KNOWN_SCENES: dict\[str, tuple\[str, str\]\] = \{\n)",
        r"\1" + entries,
        text,
        count=1,
    )
    if new_text == text:
        raise RuntimeError(f"Could not find KNOWN_SCENES declaration in {module_path}")
    module_path.write_text(new_text)
    return True
