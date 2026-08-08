#!/usr/bin/env python3
"""ROS2 node: vision-only half of the split real-hardware architecture.

Meant to run ON THE ROBOT (next to the camera, e.g. the TurtleBot3's
Raspberry Pi): subscribes to the raw camera topic, runs
``common.vision.LineFollowerVision`` (the same algorithm
``line_follower_node.py`` uses), and publishes just the resulting
steering/lateral/heading error + confidence + found as a small
``Float32MultiArray`` on ``local_path_topic`` (see ``local_path_msg.py``) --
NOT the raw camera frame. Pairs with ``control_node.py`` (meant to run on the
PC), which subscribes to that topic and does the PID(+optional residual RL)
half and publishes ``/cmd_vel``.

Why split this way: streaming raw 640x480 RGB frames over WiFi at any
reasonable rate is on the order of hundreds of Mbps (640*480*3 bytes/frame,
uncompressed -- sensor_msgs/Image carries no compression); the vision result
is 5 floats. Running vision.py on the Pi (where the camera already is) keeps
only that tiny message on the network, at the cost of the vision compute
itself running on the Pi's weaker CPU instead of the PC's.

For the simpler single-machine alternative (vision + PID in one node, no
network hop at all), see ``line_follower_node.py`` /
``real_line_follower.launch.py`` instead -- this split only makes sense once
the control loop is actually meant to run on a different machine than the
camera.

Usage (on the Pi)
------------------
    ros2 run simple_camera_pid line_follower_vision_node --ros-args \\
        --params-file install/simple_camera_pid/share/simple_camera_pid/config/real/real_line_follower.yaml
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from simple_camera_pid.common.camera_geometry import (
    turtlebot3_burger_gazebo_camera, turtlebot3_burger_real_camera,
)
from simple_camera_pid.common.vision import LineFollowerVision
from simple_camera_pid.real.local_path_msg import pack_local_path


def _image_msg_to_rgb(msg: Image) -> Optional[np.ndarray]:
    """Decode a sensor_msgs/Image into an (H, W, 3) uint8 RGB array."""
    encoding = msg.encoding.lower()
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    if height <= 0 or width <= 0 or step <= 0:
        return None
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    try:
        rows = raw.reshape((height, step))
    except ValueError:
        return None

    if encoding in ('rgb8', 'bgr8'):
        channels = rows[:, :width * 3].reshape((height, width, 3))
        if encoding == 'bgr8':
            channels = channels[:, :, ::-1]
        return np.ascontiguousarray(channels)
    if encoding in ('rgba8', 'bgra8'):
        channels = rows[:, :width * 4].reshape((height, width, 4))[:, :, :3]
        if encoding == 'bgra8':
            channels = channels[:, :, ::-1]
        return np.ascontiguousarray(channels)
    return None


class LineFollowerVisionNode(Node):
    def __init__(self) -> None:
        super().__init__("line_follower_vision_node")
        self._declare_parameters()

        camera_profile = self.get_parameter("camera_profile").value
        if camera_profile == "real":
            # Real camera: geometry has to come from an actual measurement of
            # this robot's camera -- see turtlebot3_burger_real_camera()'s
            # docstring (common/camera_geometry.py) for how to get these.
            image_width = int(self.get_parameter("image_width").value)
            image_height = int(self.get_parameter("image_height").value)
            camera_params = turtlebot3_burger_real_camera(
                image_width=image_width,
                image_height=image_height,
                fovy_deg=self.get_parameter("camera_fovy_deg").value,
                mount_height=self.get_parameter("camera_mount_height").value,
                pitch_deg=self.get_parameter("camera_pitch_deg").value,
                roll_deg=self.get_parameter("camera_roll_deg").value,
                yaw_deg=self.get_parameter("camera_yaw_deg").value,
            )
        elif camera_profile == "gazebo":
            camera_params = turtlebot3_burger_gazebo_camera()
            image_width, image_height = camera_params.image_width, camera_params.image_height
        else:
            raise ValueError(f"unknown camera_profile: {camera_profile!r} (expected 'gazebo' or 'real')")

        self.vision = LineFollowerVision(
            roi_bottom_fraction=self.get_parameter("roi_bottom_fraction").value,
            waypoint_count=self.get_parameter("num_points").value,
            lookahead_distance=self.get_parameter("lookahead_distance").value,
            lateral_gain=self.get_parameter("lateral_gain").value,
            heading_gain=self.get_parameter("heading_gain").value,
            curvature_gain=self.get_parameter("curvature_gain").value,
            min_brightness=self.get_parameter("min_brightness").value,
            max_saturation=self.get_parameter("max_saturation").value,
            min_pixels=self.get_parameter("min_pixels").value,
            error_scale=self.get_parameter("error_scale").value,
            roi_widen_step=self.get_parameter("roi_widen_step").value,
            roi_widen_max=self.get_parameter("roi_widen_max").value,
            image_height=image_height,
            image_width=image_width,
            camera=camera_params,
            flip_vertical=False,
        )

        self.local_path_pub = self.create_publisher(
            Float32MultiArray, self.get_parameter("local_path_topic").value, 10
        )
        self.create_subscription(
            Image, self.get_parameter("image_topic").value, self._image_callback, 10
        )
        self.get_logger().info(
            f"Vision node started: camera_profile={camera_profile}, "
            f"{image_width}x{image_height}, "
            f"{self.get_parameter('image_topic').value} -> "
            f"{self.get_parameter('local_path_topic').value}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("local_path_topic", "/line_follower/local_path")
        # Real hardware by default (this node only exists for the split
        # real-robot architecture) -- see turtlebot3_burger_real_camera()'s
        # docstring in common/camera_geometry.py for how to fill these in
        # from an actual measurement.
        self.declare_parameter("camera_profile", "real")
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("camera_fovy_deg", 45.9857)
        self.declare_parameter("camera_mount_height", 0.133)
        self.declare_parameter("camera_pitch_deg", 15.0)
        self.declare_parameter("camera_roll_deg", 0.0)
        self.declare_parameter("camera_yaw_deg", 0.0)
        self.declare_parameter("roi_bottom_fraction", 0.3)
        self.declare_parameter("num_points", 30)
        self.declare_parameter("lookahead_distance", 0.20)
        self.declare_parameter("lateral_gain", 0.6)
        self.declare_parameter("heading_gain", 0.35)
        self.declare_parameter("curvature_gain", 0.04)
        self.declare_parameter("min_brightness", 70.0)
        self.declare_parameter("max_saturation", 0.30)
        self.declare_parameter("min_pixels", 30.0)
        self.declare_parameter("error_scale", 500.0)
        self.declare_parameter("roi_widen_step", 0.2)
        self.declare_parameter("roi_widen_max", 0.7)

    def _image_callback(self, msg: Image) -> None:
        rgb = _image_msg_to_rgb(msg)
        if rgb is None:
            return
        result = self.vision.step(rgb)
        self.local_path_pub.publish(pack_local_path(
            result.steering_error, result.lateral_error, result.heading_error,
            result.confidence, result.found,
        ))


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = LineFollowerVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
