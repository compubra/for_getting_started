"""Monitoring test flow for the real TurtleBot3: camera, vision, velocity, pose.

Meant to run ON THE PC, alongside line_follower_vision_node (on the Pi) and
line_follower_control_node (on the PC) -- see real_line_follower.launch.py
for the single-process alternative, or run the two split nodes directly for
the actual control loop. This file only starts monitoring tools, no control
loop of its own -- same division of labor as gazebo_monitor.launch.py.

Unlike gazebo_monitor.launch.py, vision_debug_node is NOT started here -- it
needs to run ON THE PI (next to the camera), for the same reason
line_follower_vision_node does (see that node's module docstring):
subscribing to the raw 800x600 camera feed from the PC over WiFi measured out
at only ~1-2 Hz (2026-07-31), so running the vision computation remotely
defeats the point of a live debug view. Start it separately on the Pi:

    ros2 run simple_camera_pid vision_debug_node --ros-args \\
        --params-file install/simple_camera_pid/share/simple_camera_pid/config/real/real_line_follower.yaml \\
        -p platform:=real

This file uses rviz/real_monitor.rviz, NOT gazebo_monitor.rviz -- same
layout (annotated image, pose, trajectory), minus the raw-camera Image
panel. Confirmed 2026-07-31: with both gazebo_monitor.rviz's Image panels
subscribed (raw /camera/image_raw + annotated /vision_debug/image_raw, both
full 800x600 frames) on top of an already-marginal WiFi link, the resulting
traffic congested the link badly enough to drop the operator's unrelated SSH
session into the Pi -- not a camera/resolution/power problem, a "two
uncompressed image streams saturating one WiFi link" problem. real_monitor
.rviz only subscribes to /vision_debug/image_raw (the raw feed is still
visible baked into that overlay anyway), roughly halving the image traffic.
If you need the separate raw feed too, add it back as a second Image display
pointed at /camera/image_raw in RViz -- know that doing so reintroduces the
same congestion risk this file was changed to avoid.

Usage
-----
    ros2 launch simple_camera_pid real_monitor.launch.py
    ros2 launch simple_camera_pid real_monitor.launch.py plot:=false
    ros2 launch simple_camera_pid real_monitor.launch.py trajectory:=false

Arguments
---------
    rviz        Launch rviz2 with the annotated vision_debug + pose +
                trajectory panels (default true; no separate raw-camera
                panel, see the module docstring above for why). vision_debug_node
                itself is NOT started by this file -- run it on the Pi.
    plot        Launch rqt_plot windows for commanded-vs-measured linear/
                angular velocity (default true). Field paths assume
                TwistStamped (this robot's turtlebot3_node convention -- see
                cmd_vel_stamped in gazebo_line_follower_node.py/
                control_node.py), unlike gazebo_monitor.launch.py's plain
                Twist paths.
    trajectory  Launch trajectory_path_node, publishing the driven route as
                a nav_msgs/Path on /trajectory (default true).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rviz = LaunchConfiguration('rviz')
    plot = LaunchConfiguration('plot')
    trajectory = LaunchConfiguration('trajectory')

    this_pkg = get_package_share_directory('simple_camera_pid')

    trajectory_path_node = Node(
        package='simple_camera_pid',
        executable='trajectory_path_node',
        name='trajectory_path_node',
        output='screen',
        condition=IfCondition(trajectory),
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='real_monitor_rviz2',
        output='screen',
        arguments=['-d', os.path.join(this_pkg, 'rviz', 'real_monitor.rviz')],
        condition=IfCondition(rviz),
    )

    plot_linear = Node(
        package='rqt_plot',
        executable='rqt_plot',
        name='plot_linear_velocity',
        output='screen',
        arguments=['/cmd_vel/twist/linear/x', '/odom/twist/twist/linear/x'],
        condition=IfCondition(plot),
    )

    plot_angular = Node(
        package='rqt_plot',
        executable='rqt_plot',
        name='plot_angular_velocity',
        output='screen',
        arguments=['/cmd_vel/twist/angular/z', '/odom/twist/twist/angular/z'],
        condition=IfCondition(plot),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='Launch rviz2 (reuses gazebo_monitor.rviz -- its topic names are '
                        'platform-generic, not Gazebo-specific).',
        ),
        DeclareLaunchArgument(
            'plot', default_value='true',
            description='Launch rqt_plot windows for velocity (TwistStamped field paths).',
        ),
        DeclareLaunchArgument(
            'trajectory', default_value='true',
            description='Launch trajectory_path_node.',
        ),
        trajectory_path_node,
        rviz_node,
        plot_linear,
        plot_angular,
    ])
