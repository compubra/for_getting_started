"""One-shot bring-up: track world + TurtleBot3 Burger + (optional) controller.

Composes the two building-block launch files:
  * track_world.launch.py          — Gazebo + selected track + spawned robot
  * gazebo_line_follower.launch.py — PID (+ optional SAC/PPO residual) controller

Whether the controller starts is governed by the `controller` argument, so the
same launch covers both "just the sim" and "sim + controller" workflows.
`controller:=true` defaults to pure PID (residual_model_path:='') -- pass a
checkpoint path to layer a trained residual policy on top, same as calling
gazebo_line_follower.launch.py directly.

2026-07-28: this used to include the legacy simple_camera_pid_node (its own
separate, unmaintained image_line_detector.py vision code) by default -- that
node misdetected this world's own oversized default ground_plane as the line
(see track_camera_world.sdf's docstring) and drove straight off the track on
anything but the most trivial map. It has been deleted; this now defaults to
gazebo_line_follower_node's shared common/vision.py pipeline instead, which
does not have that failure mode (verified against track_hard).

Works on modern ROS 2 with the ros_gz stack (Jazzy, Kilted, Lyrical, ...); no
distro name is hardcoded. Check yours with `echo $ROS_DISTRO`.

Usage
-----
    # Full bring-up (default): world + robot + pure PID controller
    ros2 launch simple_camera_pid track_bringup.launch.py map:=track_hard

    # Simulation only, no controller
    ros2 launch simple_camera_pid track_bringup.launch.py map:=complex controller:=false

    # With a trained SAC residual policy on top of the PID
    ros2 launch simple_camera_pid track_bringup.launch.py \\
        map:=track_hard residual_model_path:=/path/to/sac_agent.zip

    # Headless + point Gazebo at the in-repo track models
    ros2 launch simple_camera_pid track_bringup.launch.py \\
        map:=track_medium gui:=false controller:=true \\
        models_path:=<repo>/model/gazebo/tracks

Arguments
---------
    map          Track to load (default track_hard). One of:
                 simple complex ellipse training track_easy track_medium track_hard
    controller   Start the line-following controller node (default true).
    gui          Run the Gazebo GUI (default true).
    use_sim_time Use simulation time for the controller (default true).
    config_file  Controller parameter YAML (default gazebo_line_follower.yaml).
    residual_model_path
                 Trained SAC/PPO residual policy .zip (default '' = pure PID
                 baseline, no residual). See gazebo_line_follower.launch.py
                 for the current best-checkpoint default if you want RL.
    residual_algo, residual_use_wheel_speed_obs, residual_use_2d_action,
    residual_max_delta_v
                 Passed straight through to gazebo_line_follower.launch.py;
                 only relevant when residual_model_path is set.
    models_path  GZ_SIM_RESOURCE_PATH folder for the track models (default:
                 this repo's own model/gazebo/tracks).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


# This workspace's known location on this machine — checked first so
# `ros2 launch` works regardless of the caller's current working directory.
# Kept in sync manually with track_world.launch.py's copy of this constant
# (see there for the full explanation) since each launch file here is exec'd
# standalone by ros2 launch and doesn't import the other as a Python module.
KNOWN_WORKSPACE_TRACKS_PATH = (
    '/media/kevin/ding/final_project/Sheffield/for_getting_started/model/gazebo/tracks'
)


def _default_tracks_path() -> str:
    """Default GZ_SIM_RESOURCE_PATH folder for the track models.

    Search upward from the current working directory for a
    model/gazebo/tracks marker (valid for the standard
    `cd <workspace_root> && ros2 launch ...` invocation), falling back to a
    walk from this file's own location. See track_world.launch.py's
    _default_tracks_path for the full explanation — a plain relative walk
    from an installed (non-symlinked) launch file resolves into install/
    instead of the workspace root and silently breaks map loading. Kept in
    sync manually since each launch file here is exec'd standalone by
    ros2 launch and doesn't import the other as a Python module.
    """
    if os.path.isdir(KNOWN_WORKSPACE_TRACKS_PATH):
        return KNOWN_WORKSPACE_TRACKS_PATH

    here = os.getcwd()
    candidate = here
    while True:
        tracks_dir = os.path.join(candidate, 'model', 'gazebo', 'tracks')
        if os.path.isdir(tracks_dir):
            return tracks_dir
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..', '..', '..', '..', 'model', 'gazebo', 'tracks',
    ))


DEFAULT_TRACKS_PATH = _default_tracks_path()


def generate_launch_description():
    map_arg = LaunchConfiguration('map')
    controller = LaunchConfiguration('controller')
    gui = LaunchConfiguration('gui')
    models_path = LaunchConfiguration('models_path')
    use_sim_time = LaunchConfiguration('use_sim_time')
    config_file = LaunchConfiguration('config_file')
    residual_model_path = LaunchConfiguration('residual_model_path')
    residual_algo = LaunchConfiguration('residual_algo')
    residual_use_wheel_speed_obs = LaunchConfiguration('residual_use_wheel_speed_obs')
    residual_use_2d_action = LaunchConfiguration('residual_use_2d_action')
    residual_max_delta_v = LaunchConfiguration('residual_max_delta_v')

    pkg_share = FindPackageShare('simple_camera_pid')
    this_pkg = get_package_share_directory('simple_camera_pid')

    # 1) Track world + robot (always) — reuse track_world.launch.py.
    world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, 'launch', 'gazebo', 'track_world.launch.py'])
        ),
        launch_arguments={
            'map': map_arg,
            'gui': gui,
            'models_path': models_path,
        }.items(),
    )

    # 2) Controller (optional) — reuse gazebo_line_follower.launch.py, pure
    #    PID by default (residual_model_path defaults to '' below).
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, 'launch', 'gazebo', 'gazebo_line_follower.launch.py'])
        ),
        condition=IfCondition(controller),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'config_file': config_file,
            'residual_model_path': residual_model_path,
            'residual_algo': residual_algo,
            'residual_use_wheel_speed_obs': residual_use_wheel_speed_obs,
            'residual_use_2d_action': residual_use_2d_action,
            'residual_max_delta_v': residual_max_delta_v,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value='track_hard',
            description=(
                'Track to load. One of: simple, complex, ellipse, training, '
                'track_easy, track_medium, track_hard.'
            ),
        ),
        DeclareLaunchArgument(
            'controller',
            default_value='true',
            description='Start the line-following controller node.',
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Run the Gazebo GUI (false = headless server only).',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time for the controller.',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=os.path.join(this_pkg, 'config', 'gazebo', 'gazebo_line_follower.yaml'),
            description='Controller parameter YAML.',
        ),
        DeclareLaunchArgument(
            'residual_model_path',
            default_value='',
            description="Trained SAC/PPO residual policy .zip. Default '' = "
                        'pure PID baseline, no residual. See '
                        'gazebo_line_follower.launch.py for the current '
                        'best-checkpoint default if you want RL.',
        ),
        DeclareLaunchArgument(
            'residual_algo',
            default_value='sac',
            description='"sac" or "ppo", matching residual_model_path.',
        ),
        DeclareLaunchArgument(
            'residual_use_wheel_speed_obs',
            default_value='true',
            description='Passed through to gazebo_line_follower.launch.py.',
        ),
        DeclareLaunchArgument(
            'residual_use_2d_action',
            default_value='true',
            description='Passed through to gazebo_line_follower.launch.py.',
        ),
        DeclareLaunchArgument(
            'residual_max_delta_v',
            default_value='1.0',
            description='Passed through to gazebo_line_follower.launch.py.',
        ),
        DeclareLaunchArgument(
            'models_path',
            default_value=DEFAULT_TRACKS_PATH,
            description=(
                'Folder appended to GZ_SIM_RESOURCE_PATH so Gazebo can find '
                "the track models. Defaults to this repo's own "
                'model/gazebo/tracks.'
            ),
        ),
        world,
        controller_launch,
    ])
