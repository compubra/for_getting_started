"""Monitoring test flow for the Gazebo line follower: camera, velocity, pose.

Run alongside track_bringup.launch.py / gazebo_line_follower.launch.py --
this file only starts monitoring tools, no simulation or controller:

    ros2 launch simple_camera_pid track_bringup.launch.py map:=track_hard controller:=false
    ros2 launch simple_camera_pid gazebo_line_follower.launch.py
    ros2 launch simple_camera_pid gazebo_monitor.launch.py

All-ROS-native monitoring stack, no custom scripts (except trajectory_path_node,
see below):
  * vision_debug_node -- runs the actual line-following vision algorithm on
              the live camera feed and republishes an annotated copy on
              /vision_debug/image_raw: ROI band boundary, every pixel
              currently treated as "candidate line" tinted green, sliding-
              window boxes, and the fitted ground-quadratic-fit trajectory
              curve (yellow) -- see common/debug_frame.py. This is what
              actually lets you see *why* the controller is steering the
              way it is, not just that a camera image exists.
  * trajectory_path_node -- 2026-07-29: RViz2's stock Odometry display
              (used for pose below) only draws discrete Arrow/Axes markers,
              no continuous line -- with Keep set high enough to show useful
              history it reads as a trail of red arrows, not a trajectory.
              This node republishes accumulated /odom poses as a
              nav_msgs/Path on /trajectory so RViz's Path display (added to
              gazebo_monitor.rviz) can draw the driven route as an actual
              line; Odometry itself is now Keep:=1 (a single current-pose
              marker, not a trail).
  * rviz2   -- raw camera feed + annotated vision_debug feed side by side,
              current pose (Odometry, one marker), and the full driven route
              (Path, on /trajectory).
  * rqt_plot -- two windows, commanded vs. measured velocity: linear.x
              (/cmd_vel vs. /odom's twist) and angular.z (same pair).
  * ros2 bag record (opt-in via record:=true) -- logs camera/vision_debug/
              odom/trajectory/cmd_vel/diagnostics for offline replay
              (ros2 bag play) or inspection (ros2 bag info / rqt_bag), the
              standard ROS2 test-capture flow.

Usage
-----
    ros2 launch simple_camera_pid gazebo_monitor.launch.py
    ros2 launch simple_camera_pid gazebo_monitor.launch.py plot:=false
    ros2 launch simple_camera_pid gazebo_monitor.launch.py vision_debug:=false
    ros2 launch simple_camera_pid gazebo_monitor.launch.py trajectory:=false
    ros2 launch simple_camera_pid gazebo_monitor.launch.py record:=true
    ros2 launch simple_camera_pid gazebo_monitor.launch.py \\
        record:=true bag_path:=/tmp/my_test_run

Arguments
---------
    rviz        Launch rviz2 with camera + vision_debug + pose + trajectory
                (default true).
    plot        Launch rqt_plot windows for linear/angular velocity
                (default true).
    vision_debug
                Launch vision_debug_node, publishing the annotated feed on
                /vision_debug/image_raw (default true).
    trajectory  Launch trajectory_path_node, publishing the driven route as
                a nav_msgs/Path on /trajectory (default true).
    record      Launch `ros2 bag record` for camera/vision_debug/odom/
                trajectory/cmd_vel/diagnostics (default false).
    bag_path    Output directory for the bag when record:=true (default:
                gazebo_test_<timestamp> in the current directory).
    diagnostic_topic
                Diagnostics topic to record (default
                /gazebo_line_follower/debug, matching gazebo_line_follower_node).
"""
import datetime
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rviz = LaunchConfiguration('rviz')
    plot = LaunchConfiguration('plot')
    vision_debug = LaunchConfiguration('vision_debug')
    trajectory = LaunchConfiguration('trajectory')
    record = LaunchConfiguration('record')
    bag_path = LaunchConfiguration('bag_path')
    diagnostic_topic = LaunchConfiguration('diagnostic_topic')

    this_pkg = get_package_share_directory('simple_camera_pid')
    default_bag_path = 'gazebo_test_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    vision_debug_node = Node(
        package='simple_camera_pid',
        executable='vision_debug_node',
        name='vision_debug_node',
        output='screen',
        condition=IfCondition(vision_debug),
    )

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
        name='gazebo_monitor_rviz2',
        output='screen',
        arguments=['-d', os.path.join(this_pkg, 'rviz', 'gazebo_monitor.rviz')],
        condition=IfCondition(rviz),
    )

    plot_linear = Node(
        package='rqt_plot',
        executable='rqt_plot',
        name='plot_linear_velocity',
        output='screen',
        arguments=['/cmd_vel/linear/x', '/odom/twist/twist/linear/x'],
        condition=IfCondition(plot),
    )

    plot_angular = Node(
        package='rqt_plot',
        executable='rqt_plot',
        name='plot_angular_velocity',
        output='screen',
        arguments=['/cmd_vel/angular/z', '/odom/twist/twist/angular/z'],
        condition=IfCondition(plot),
    )

    bag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record', '-o', bag_path,
            '/camera/image_raw', '/camera/camera_info', '/vision_debug/image_raw',
            '/odom', '/trajectory', '/cmd_vel', '/joint_states', '/tf', diagnostic_topic,
        ],
        output='screen',
        condition=IfCondition(record),
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true',
                               description='Launch rviz2 with camera + vision_debug + pose + '
                                           'trajectory.'),
        DeclareLaunchArgument('plot', default_value='true',
                               description='Launch rqt_plot windows for velocity.'),
        DeclareLaunchArgument('vision_debug', default_value='true',
                               description='Launch vision_debug_node (annotated feed on '
                                           '/vision_debug/image_raw).'),
        DeclareLaunchArgument('trajectory', default_value='true',
                               description='Launch trajectory_path_node (driven route as a '
                                           'nav_msgs/Path on /trajectory).'),
        DeclareLaunchArgument('record', default_value='false',
                               description='Launch ros2 bag record.'),
        DeclareLaunchArgument('bag_path', default_value=default_bag_path,
                               description='Output directory for the bag when record:=true.'),
        DeclareLaunchArgument('diagnostic_topic', default_value='/gazebo_line_follower/debug',
                               description='Diagnostics topic to record.'),
        vision_debug_node,
        trajectory_path_node,
        rviz_node,
        plot_linear,
        plot_angular,
        bag_record,
    ])
