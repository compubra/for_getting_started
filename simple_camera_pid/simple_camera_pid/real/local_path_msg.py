"""Wire format for the split vision/control real-hardware architecture.

``vision_node.py`` (meant to run on the robot, next to the camera -- e.g. the
TurtleBot3's Raspberry Pi) publishes this; ``control_node.py`` (meant to run
on the PC) subscribes to it. A plain ``Float32MultiArray`` instead of a
custom ``.msg`` -- avoids adding a whole separate rosidl interface package
just for five floats, and matches this project's existing diagnostic-topic
convention (see ``line_follower_node.py``'s ``_publish_diagnostics``).
"""
from __future__ import annotations

from typing import Tuple

from std_msgs.msg import Float32MultiArray

# [steering_error, lateral_error, heading_error, confidence, found]
LOCAL_PATH_LEN = 5


def pack_local_path(steering_error: float, lateral_error: float, heading_error: float,
                     confidence: float, found: bool) -> Float32MultiArray:
    msg = Float32MultiArray()
    msg.data = [
        float(steering_error), float(lateral_error), float(heading_error),
        float(confidence), 1.0 if found else 0.0,
    ]
    return msg


def unpack_local_path(msg: Float32MultiArray) -> Tuple[float, float, float, float, bool]:
    """Returns ``(steering_error, lateral_error, heading_error, confidence, found)``."""
    data = msg.data
    if len(data) != LOCAL_PATH_LEN:
        raise ValueError(f"expected {LOCAL_PATH_LEN} floats, got {len(data)}")
    steering_error, lateral_error, heading_error, confidence, found_f = data
    return steering_error, lateral_error, heading_error, confidence, found_f >= 0.5
