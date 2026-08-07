#!/usr/bin/env python3
"""ROS2 node: visualizes what the line-following vision algorithm actually sees.

Subscribes to a raw camera topic, runs the same
``common.vision.LineFollowerVision`` algorithm (Hough-seed + sliding-window +
ground-quadratic-fit, see ``common/vision.py``'s module docstring) used by
``gazebo_line_follower_node.py``/``mujoco_line_follower_node.py`` on every
frame, and republishes an annotated copy via
``common.debug_frame.render_debug_frame`` -- the full overlay ported from
MATLAB's ``originbot_line_follower_debug_frame.m`` (see that file's
docstring for the color legend), not a hand-rolled subset: ROI band (blue
found / red lost), every candidate-line pixel tinted green, sliding-window
boxes (cyan valid / gray extrapolated), magenta window centroids, the
yellow ground-quadratic-fit **trajectory curve** (reprojected into the
image -- this is the actual computed path, not just a pass/fail direction
cue), the white center column, and the red nearest-valid-window column
(the PID's immediate-feedback source). A text readout of steering_error /
lateral_error / heading_error / confidence / found is drawn on top of that
overlay (MATLAB's version has no text -- Simulink shows those on a scope
instead, but this ROS node has no scope to hand off to).

2026-07-29: generalized from a Gazebo-only node (hardcoded
turtlebot3_burger_gazebo_camera(), lived under gazebo/) to platform-agnostic
-- the ``platform`` parameter below selects the camera profile, moved to
common/ alongside the vision/debug_frame code it wraps since neither the
algorithm nor this node has any Gazebo-specific content left. Image
dimensions default to the Gazebo camera's (480x640 -- see
turtlebot3_burger_gazebo_camera()'s docstring); pass platform:=mujoco (same
480x640, so no override needed -- MuJoCo and Gazebo share one resolution as
of 2026-08-03) or platform:=real with your own image_height/image_width to
point this at a MuJoCo or real-hardware run instead. All platforms publish
already-
upright frames on /camera/image_raw (see vision.LineFollowerVision's
flip_vertical docstring), so flip_vertical=False is correct for either.

This runs its own independent ``LineFollowerVision`` instance (its own
lost-line freeze/decay state machine), so its displayed ``steering_error``
during a momentary line loss can lag the controller's by up to one control
tick -- it is a monitoring tap, not a splice into the control loop.

Usage
-----
    ros2 launch simple_camera_pid gazebo_monitor.launch.py
    ros2 run simple_camera_pid vision_debug_node
    ros2 run simple_camera_pid vision_debug_node --ros-args \\
        -p platform:=mujoco -p image_height:=480 -p image_width:=640 \\
        -p lookahead_distance:=0.40
    ros2 run simple_camera_pid vision_debug_node --ros-args \\
        --params-file install/simple_camera_pid/share/simple_camera_pid/config/real/real_line_follower.yaml \\
        -p platform:=real
        (real robot -- run this BEFORE trusting gazebo_line_follower_node to
        actually drive: reuses the same real_line_follower.yaml you filled in
        for the control node -- image_topic/image_width/image_height/
        camera_fovy_deg/camera_mount_height/camera_pitch_deg all come from
        there, since that file's top-level key is gazebo_line_follower_node
        but unrecognized keys are silently ignored by other nodes, so this
        picks up exactly the subset it declares. Point rqt_image_view at
        /vision_debug/image_raw and confirm found=1/steering_error has the
        expected sign while you slide the robot along the real line by hand)
    ros2 run rqt_image_view rqt_image_view /vision_debug/image_raw
        (rqt_image_view's own Topic dropdown lists both /vision_debug/
        image_raw and its .../compressed sibling separately -- pick
        .../compressed when viewing over a bandwidth-limited link, e.g. this
        project's Pi<->PC WiFi: 2026-08-03, subscribing to the raw feed
        remotely alongside the raw camera panel congested that link badly
        enough to drop an unrelated SSH session. Same JPEG topic also lets
        RViz's Image display pick a "compressed" transport hint instead of
        "raw" for the same reason -- see launch/real/real_monitor.launch.py)
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

from simple_camera_pid.common.camera_geometry import (
    turtlebot3_burger_gazebo_camera, turtlebot3_burger_mujoco_camera,
    turtlebot3_burger_real_camera,
)
from simple_camera_pid.common.debug_frame import render_debug_frame
from simple_camera_pid.common.vision import LocalPathResult, LineFollowerVision


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


def _rgb_to_image_msg(rgb: np.ndarray, header) -> Image:
    msg = Image()
    msg.header = header
    msg.height, msg.width = rgb.shape[0], rgb.shape[1]
    msg.encoding = 'rgb8'
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    return msg


def _rgb_to_compressed_image_msg(rgb: np.ndarray, header, jpeg_quality: int) -> CompressedImage:
    """JPEG-encoded sibling of _rgb_to_image_msg, on the image_transport
    ``<topic>/compressed`` convention (see the module docstring's network
    note) -- a 640x480 raw frame is ~0.9 MB; JPEG brings this to tens of KB,
    which is what makes viewing this feed live over the Pi<->PC WiFi link
    practical at all (2026-08-03: uncompressed /vision_debug/image_raw
    congested that link badly enough to drop an unrelated SSH session)."""
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    msg = CompressedImage()
    msg.header = header
    msg.format = 'jpeg'
    if ok:
        msg.data = buf.tobytes()
    return msg


class VisionDebugNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_debug_node")
        self._declare_parameters()

        platform = str(self.get_parameter("platform").value).lower()
        if platform == "mujoco":
            self.camera = turtlebot3_burger_mujoco_camera()
        elif platform == "real":
            # See turtlebot3_burger_real_camera()'s docstring
            # (common/camera_geometry.py) for how to measure these.
            self.camera = turtlebot3_burger_real_camera(
                image_width=int(self.get_parameter("image_width").value),
                image_height=int(self.get_parameter("image_height").value),
                fovy_deg=self.get_parameter("camera_fovy_deg").value,
                mount_height=self.get_parameter("camera_mount_height").value,
                pitch_deg=self.get_parameter("camera_pitch_deg").value,
                roll_deg=self.get_parameter("camera_roll_deg").value,
                yaw_deg=self.get_parameter("camera_yaw_deg").value,
            )
        else:
            self.camera = turtlebot3_burger_gazebo_camera()

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
            image_height=self.get_parameter("image_height").value,
            image_width=self.get_parameter("image_width").value,
            camera=self.camera,
            flip_vertical=False,
        )

        debug_image_topic = self.get_parameter("debug_image_topic").value
        self.image_pub = self.create_publisher(Image, debug_image_topic, 10)
        # image_transport's "<topic>/compressed" convention -- lets
        # rqt_image_view's transport dropdown (or RViz's Image display, which
        # also offers a transport-hint picker) select this instead of the
        # full-size raw topic when viewing over a bandwidth-limited link
        # (e.g. the Pi<->PC WiFi in this project's real-hardware setup).
        self.compressed_image_pub = self.create_publisher(
            CompressedImage, debug_image_topic + "/compressed", 10
        )
        self._jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.create_subscription(
            Image, self.get_parameter("image_topic").value, self._image_callback, 10
        )
        self.get_logger().info(
            f"Vision debug node started: platform={platform} "
            f"{self.get_parameter('image_topic').value} -> "
            f"{debug_image_topic} (+ {debug_image_topic}/compressed, "
            f"jpeg_quality={self._jpeg_quality}) "
            f"(roi_bottom_fraction={self.vision.roi_bottom_fraction})"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("platform", "gazebo")  # "gazebo" | "mujoco" | "real"
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("debug_image_topic", "/vision_debug/image_raw")
        self.declare_parameter("image_height", 480)
        self.declare_parameter("image_width", 640)
        # Only read when platform:=real -- see turtlebot3_burger_real_camera()'s
        # docstring in common/camera_geometry.py.
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
        # JPEG quality for the .../compressed sibling topic (1-100, higher =
        # bigger/better). 80 is a reasonable default for a debug overlay
        # that's mostly flat color regions (ROI tint, boxes, text) plus a
        # camera frame -- visually near-lossless at this content, and far
        # smaller than raw regardless of exact value chosen.
        self.declare_parameter("jpeg_quality", 80)

    def _image_callback(self, msg: Image) -> None:
        rgb = _image_msg_to_rgb(msg)
        if rgb is None:
            return
        # Stateful pass (freeze/decay-aware found/steering, for the text
        # readout) -- kept independent from the stateless renderer below, see
        # the module docstring.
        result = self.vision.step(rgb)
        annotated = render_debug_frame(
            rgb,
            roi_bottom_fraction=self.vision.roi_bottom_fraction,
            waypoint_count=self.vision.waypoint_count,
            min_brightness=self.vision.min_brightness,
            max_saturation=self.vision.max_saturation,
            min_pixels=self.vision.min_pixels,
            roi_widen_step=self.vision.roi_widen_step,
            roi_widen_max=self.vision.roi_widen_max,
            camera=self.camera,
            image_height=self.get_parameter("image_height").value,
            image_width=self.get_parameter("image_width").value,
            flip_vertical=False,
        )
        annotated = self._draw_text(annotated, result)
        self.image_pub.publish(_rgb_to_image_msg(annotated, msg.header))
        self.compressed_image_pub.publish(
            _rgb_to_compressed_image_msg(annotated, msg.header, self._jpeg_quality)
        )

    def _draw_text(self, img: np.ndarray, result: LocalPathResult) -> np.ndarray:
        found_color = (0, 200, 0) if result.found else (220, 0, 0)
        lines = [
            f"found={result.found}  confidence={result.confidence:.2f}",
            f"steering={result.steering_error:+.3f}  lateral={result.lateral_error:+.3f}  "
            f"heading={result.heading_error:+.3f}",
        ]
        for i, text in enumerate(lines):
            y = 34 + i * 32
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, found_color, 2)
        return img


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = VisionDebugNode()
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
