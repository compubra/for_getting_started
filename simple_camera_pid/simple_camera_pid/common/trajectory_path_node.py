#!/usr/bin/env python3
"""ROS2 node: republishes accumulated /odom poses as a nav_msgs/Path.

RViz2's stock Odometry display (rviz_default_plugins/Odometry, used in
rviz/gazebo_monitor.rviz and rviz/mujoco_line_follower.rviz) only draws
discrete Arrow or Axes markers, one per kept message -- there is no
"continuous line" shape in that plugin. With Keep set high enough to show
useful history it reads as a trail of (by default red) arrows, not a
trajectory curve. rviz_default_plugins/Path is the display that actually
draws a connected line through a pose sequence, but it needs something
publishing nav_msgs/Path -- neither line-follower node does, since their
own control loops only need the latest pose. This node is that something:
platform-agnostic (subscribes to whatever nav_msgs/Odometry topic it's
pointed at, works identically for gazebo_line_follower_node's or
mujoco_line_follower_node's /odom), stateless beyond the capped pose
history below.

Usage
-----
    ros2 run simple_camera_pid trajectory_path_node
    ros2 run simple_camera_pid trajectory_path_node --ros-args -p max_poses:=5000
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node


class TrajectoryPathNode(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_path_node")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("path_topic", "/trajectory")
        # Bounds memory/RViz draw cost on long-running demos; ~2000 poses at
        # the 20 Hz control rate both nodes publish odom at is ~100 s of
        # history, plenty for a lap or two on any of the 4.4 m tracks.
        self.declare_parameter("max_poses", 2000)

        self._max_poses = int(self.get_parameter("max_poses").value)
        self._poses: deque = deque(maxlen=self._max_poses)

        self.path_pub = self.create_publisher(Path, self.get_parameter("path_topic").value, 10)
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self._on_odom, 10
        )
        self.get_logger().info(
            f"Trajectory path node started: {self.get_parameter('odom_topic').value} -> "
            f"{self.get_parameter('path_topic').value} (max_poses={self._max_poses})"
        )

    def _on_odom(self, msg: Odometry) -> None:
        pose_stamped = _pose_stamped_from_odom(msg)
        self._poses.append(pose_stamped)

        path = Path()
        path.header.stamp = msg.header.stamp
        path.header.frame_id = msg.header.frame_id
        path.poses = list(self._poses)
        self.path_pub.publish(path)


def _pose_stamped_from_odom(msg: Odometry) -> PoseStamped:
    pose_stamped = PoseStamped()
    pose_stamped.header = msg.header
    pose_stamped.pose = msg.pose.pose
    return pose_stamped


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = TrajectoryPathNode()
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
