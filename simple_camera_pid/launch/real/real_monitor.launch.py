"""Monitoring test flow for the real TurtleBot3: camera, vision, velocity, pose.

Meant to run ON THE PC, alongside line_follower_vision_node (on the Pi) and
line_follower_control_node (on the PC) -- see real_line_follower.launch.py
for the single-process alternative, or run the two split nodes directly for
the actual control loop. This file only starts monitoring tools, no control
loop of its own.

vision_debug_node normally runs ON THE PI, not here -- for the same reason
line_follower_vision_node does (see that node's module docstring): subscribing
to the raw camera feed from the PC over WiFi measured out at only ~1-2 Hz
(2026-07-31), so running the vision computation remotely defeats the point of
a live debug view. robot_bringup.launch.py starts it on the Pi by default
(its own ``vision_debug:=true``), and this file on the PC just needs rviz2 (or
an Image display) pointed at the resulting /vision_debug/image_raw/compressed
topic -- which is exactly what rviz/real_monitor.rviz already does.

``vision_debug:=true`` here is kept only as an escape hatch for when the Pi
can't run it (e.g. debugging the Pi-side install itself). It pays the full
WiFi cost described above -- vision_debug_node's own subscription is to the
RAW /camera/image_raw, so this is for a one-off debug session, never for a
normal control-loop run. Default is false.

This file uses rviz/real_monitor.rviz: annotated image, pose, trajectory,
and deliberately no raw-camera Image panel. Confirmed 2026-07-31 (against
the then-current gazebo_monitor.rviz, since deleted): with both of its Image
panels subscribed (raw /camera/image_raw + annotated /vision_debug/image_raw, both
full 800x600 frames) on top of an already-marginal WiFi link, the resulting
traffic congested the link badly enough to drop the operator's unrelated SSH
session into the Pi -- not a camera/resolution/power problem, a "two
uncompressed image streams saturating one WiFi link" problem. real_monitor
.rviz only subscribes to /vision_debug/image_raw/compressed (the raw feed is
still visible baked into that overlay anyway), keeping RViz's own draw off
the raw stream even though vision_debug_node's own subscription upstream is
still raw -- if that's still too much load on the link, fall back to running
vision_debug_node on the Pi as shown above instead.

Usage
-----
    ros2 launch simple_camera_pid real_monitor.launch.py
    ros2 launch simple_camera_pid real_monitor.launch.py plot:=false
    ros2 launch simple_camera_pid real_monitor.launch.py trajectory:=false
    ros2 launch simple_camera_pid real_monitor.launch.py vision_debug:=true

Arguments
---------
    rviz        Launch rviz2 with the annotated vision_debug + pose +
                trajectory panels (default true; no separate raw-camera
                panel, see the module docstring above for why).
    plot        Launch rqt_plot windows for commanded-vs-measured linear/
                angular velocity (default true). Field paths assume
                TwistStamped (this robot's turtlebot3_node convention -- see
                cmd_vel_stamped in line_follower_node.py/
                control_node.py).
    trajectory  Launch trajectory_path_node, publishing the driven route as
                a nav_msgs/Path on /trajectory (default true).
    vision_debug
                Launch vision_debug_node ON THE PC (platform:=real, reading
                camera/vision params from config/real/real_line_follower.yaml)
                publishing the annotated feed on /vision_debug/image_raw (+
                /compressed sibling). Default FALSE -- the Pi runs it via
                robot_bringup.launch.py; see the module docstring's WiFi-cost
                note before turning this on.
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
    vision_debug = LaunchConfiguration('vision_debug')

    this_pkg = get_package_share_directory('simple_camera_pid')
    default_config = os.path.join(
        this_pkg, 'config', 'real', 'real_line_follower.yaml')

    trajectory_path_node = Node(
        package='simple_camera_pid',
        executable='trajectory_path_node',
        name='trajectory_path_node',
        output='screen',
        condition=IfCondition(trajectory),
    )

    vision_debug_node = Node(
        package='simple_camera_pid',
        executable='vision_debug_node',
        name='vision_debug_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file'), {'platform': 'real'}],
        condition=IfCondition(vision_debug),
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
            description='Launch rviz2 (real_monitor.rviz -- its topic names are '
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
        DeclareLaunchArgument(
            'vision_debug', default_value='false',
            description='Launch vision_debug_node on the PC (platform:=real). See module '
                        'docstring for the WiFi-cost tradeoff vs. running it on the Pi instead.',
        ),
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='Params file vision_debug_node reads camera/vision config from.',
        ),
        vision_debug_node,
        trajectory_path_node,
        rviz_node,
        plot_linear,
        plot_angular,
    ])
