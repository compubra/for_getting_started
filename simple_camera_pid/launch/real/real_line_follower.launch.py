"""Launch the plain-PID line follower against a real TurtleBot3.

Starts line_follower_node only (vision + PID + /cmd_vel in one process,
with use_sim_time forced off). Bring up the robot's own
base/motors/odometry (turtlebot3_bringup or equivalent) and a camera driver
publishing sensor_msgs/Image in rgb8/bgr8/rgba8/bgra8 separately -- this
launch file only starts the vision+PID+control node.

Before running this against real motors: fill in config/real/
real_line_follower.yaml's camera_fovy_deg/camera_mount_height/
camera_pitch_deg/image_topic TODOs from an actual measurement (see
turtlebot3_burger_real_camera()'s docstring in common/camera_geometry.py),
then sanity-check the vision pipeline with vision_debug_node BEFORE letting
the robot drive (see that node's module docstring for the exact command).

Usage
-----
    ros2 launch simple_camera_pid real_line_follower.launch.py
        (pure PID baseline against config/real/real_line_follower.yaml --
        the only sane default for a real robot: no sim-trained residual
        policy, no /clock)
    ros2 launch simple_camera_pid real_line_follower.launch.py \\
        residual_model_path:=/path/to/sac_agent.zip residual_algo:=sac
        (only after the PID baseline is verified to track the line reliably
        -- see simple_camera_pid.common.residual_policy for the checkpoint
        table; sim-trained checkpoints are unverified on real hardware)
    ros2 launch simple_camera_pid real_line_follower.launch.py \\
        config_file:=/path/to/your_own_real_config.yaml

Arguments
---------
    config_file            Node parameter YAML (default config/real/
                          real_line_follower.yaml). Copy it if you want a
                          second robot/camera profile instead of editing the
                          shared one in place.
    residual_model_path    Trained SAC/PPO residual policy .zip. Empty
                          (default) = pure PID baseline.
    residual_algo           "sac" or "ppo", matching residual_model_path
                          (default sac).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Matches ~/.bashrc: export ROS_DOMAIN_ID=30 # TURTLEBOT3. The PC and the
# TurtleBot3's own SBC must both be on this domain to discover each other's
# topics at all -- this only checks the PC side launching this file.
EXPECTED_ROS_DOMAIN_ID = '30'


def _warn_if_domain_id_mismatch(context, *args, **kwargs):
    actual = os.environ.get('ROS_DOMAIN_ID')
    if actual != EXPECTED_ROS_DOMAIN_ID:
        return [LogInfo(msg=(
            f"** ROS_DOMAIN_ID={actual!r} in this shell, expected "
            f"{EXPECTED_ROS_DOMAIN_ID!r} (see ~/.bashrc). If the robot's own "
            "SBC is not on the same ROS_DOMAIN_ID, this node will start "
            "cleanly but never see the camera topic or reach /cmd_vel. **"
        ))]
    return []


def generate_launch_description():
    config_file = LaunchConfiguration('config_file')
    this_pkg = get_package_share_directory('simple_camera_pid')

    node = Node(
        package='simple_camera_pid',
        executable='line_follower_node',
        name='line_follower_node',
        output='screen',
        parameters=[
            config_file,
            {
                # Real hardware: never sim time, and never launch this file
                # into an already-mixed use_sim_time state.
                'use_sim_time': False,
                'residual_model_path': LaunchConfiguration('residual_model_path'),
                'residual_algo': LaunchConfiguration('residual_algo'),
            },
        ],
    )

    return LaunchDescription([
        OpaqueFunction(function=_warn_if_domain_id_mismatch),
        DeclareLaunchArgument(
            'config_file',
            default_value=os.path.join(this_pkg, 'config', 'real', 'real_line_follower.yaml'),
            description='Real-robot parameter file -- fill in its '
                        'camera_fovy_deg/camera_mount_height/camera_pitch_deg/'
                        'image_topic TODOs from an actual measurement before '
                        'trusting it.',
        ),
        DeclareLaunchArgument(
            'residual_model_path',
            default_value='',
            description='Trained SAC/PPO residual policy .zip. Empty '
                        '(default) = pure PID baseline -- verify this tracks '
                        'the line reliably before ever pointing this at a '
                        'sim-trained checkpoint.',
        ),
        DeclareLaunchArgument(
            'residual_algo',
            default_value='sac',
            description='"sac" or "ppo", matching residual_model_path.',
        ),
        node,
    ])
