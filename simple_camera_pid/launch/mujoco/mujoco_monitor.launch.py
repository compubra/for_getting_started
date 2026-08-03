"""Monitoring test flow for the MuJoCo line follower: camera, velocity, pose.

Run alongside mujoco_line_follower.launch.py -- this file only starts
monitoring tools, no simulation or controller:

    ros2 launch simple_camera_pid mujoco_line_follower.launch.py use_viewer:=false
    ros2 launch simple_camera_pid mujoco_monitor.launch.py

Mirrors gazebo_monitor.launch.py (see that file for the full rationale of
each piece); the only MuJoCo-specific differences are the diagnostic topic
default, vision_debug_node's platform/image-size/lookahead params (so its
independent LineFollowerVision instance matches the MuJoCo camera instead of
Gazebo's), the rviz config, and a trimmed bag-record topic list (MuJoCo's
node publishes no /camera/camera_info, /joint_states, or /tf -- it has no
Gazebo bridge/TF tree, just image/odom/cmd_vel/diagnostics):

  * vision_debug_node -- runs common.vision.LineFollowerVision on the live
              /camera/image_raw feed and republishes an annotated copy on
              /vision_debug/image_raw (ROI band, candidate-line pixels,
              sliding-window boxes, fitted trajectory curve, text readout).
  * trajectory_path_node -- republishes accumulated /odom poses as a
              nav_msgs/Path on /trajectory so RViz's Path display can draw
              the driven route as a continuous line (Odometry's own display
              only draws discrete arrow markers).
  * rviz2   -- raw camera feed + annotated vision_debug feed, current pose,
              and the full driven route (reuses rviz/mujoco_line_follower.rviz).
  * rqt_plot -- two windows, commanded vs. measured velocity: linear.x
              (/cmd_vel vs. /odom's twist) and angular.z (same pair).
  * ros2 bag record (opt-in via record:=true) -- logs camera/vision_debug/
              odom/trajectory/cmd_vel/diagnostics for offline replay
              (ros2 bag play) or inspection (ros2 bag info / rqt_bag).
              Recording /odom alongside the diagnostic topic matters here:
              lateral_error/heading_error alone don't say *where* on the
              track a bias happened -- correlating them against odom pose/
              heading-rate is what told apart a genuine calibration bias
              from cornering-geometry corner-cutting in the 2026-08-02
              MuJoCo lateral-bias investigation.

Usage
-----
    ros2 launch simple_camera_pid mujoco_monitor.launch.py
    ros2 launch simple_camera_pid mujoco_monitor.launch.py plot:=false
    ros2 launch simple_camera_pid mujoco_monitor.launch.py vision_debug:=false
    ros2 launch simple_camera_pid mujoco_monitor.launch.py trajectory:=false
    ros2 launch simple_camera_pid mujoco_monitor.launch.py record:=true
    ros2 launch simple_camera_pid mujoco_monitor.launch.py \\
        record:=true bag_path:=/tmp/my_test_run

Arguments
---------
    rviz        Launch rviz2 with camera + vision_debug + pose + trajectory
                (default true).
    plot        Launch rqt_plot windows for linear/angular velocity
                (default true).
    vision_debug
                Launch vision_debug_node (platform:=mujoco), publishing the
                annotated feed on /vision_debug/image_raw (default true).
    trajectory  Launch trajectory_path_node, publishing the driven route as
                a nav_msgs/Path on /trajectory (default true).
    record      Launch `ros2 bag record` for camera/vision_debug/odom/
                trajectory/cmd_vel/diagnostics (default false).
    bag_path    Output directory for the bag when record:=true (default:
                mujoco_test_<timestamp> in the current directory).
    diagnostic_topic
                Diagnostics topic to record (default
                /mujoco_line_follower/debug, matching mujoco_line_follower_node).
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
    default_bag_path = 'mujoco_test_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    vision_debug_node = Node(
        package='simple_camera_pid',
        executable='vision_debug_node',
        name='vision_debug_node',
        output='screen',
        parameters=[{
            'platform': 'mujoco',
            'image_height': 480,
            'image_width': 640,
            'lookahead_distance': 0.40,
        }],
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
        name='mujoco_monitor_rviz2',
        output='screen',
        arguments=['-d', os.path.join(this_pkg, 'rviz', 'mujoco_line_follower.rviz')],
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
            '/camera/image_raw', '/vision_debug/image_raw',
            '/odom', '/trajectory', '/cmd_vel', diagnostic_topic,
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
                               description='Launch vision_debug_node (platform:=mujoco, '
                                           'annotated feed on /vision_debug/image_raw).'),
        DeclareLaunchArgument('trajectory', default_value='true',
                               description='Launch trajectory_path_node (driven route as a '
                                           'nav_msgs/Path on /trajectory).'),
        DeclareLaunchArgument('record', default_value='false',
                               description='Launch ros2 bag record.'),
        DeclareLaunchArgument('bag_path', default_value=default_bag_path,
                               description='Output directory for the bag when record:=true.'),
        DeclareLaunchArgument('diagnostic_topic', default_value='/mujoco_line_follower/debug',
                               description='Diagnostics topic to record.'),
        vision_debug_node,
        trajectory_path_node,
        rviz_node,
        plot_linear,
        plot_angular,
        bag_record,
    ])
