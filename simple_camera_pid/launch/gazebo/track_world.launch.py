"""Launch a camera-track world and spawn a TurtleBot3 Burger on the line.

Parameterised over 7 tracks (map:=<key>). Starts Gazebo (Harmonic via
ros_gz_sim), loads the selected track model as the ground, and spawns the
turtlebot3_burger_line_follower at that track's start pose (on the white
line, facing along it). The PID / line-following controller is NOT started
here — launch it separately with simple_camera_pid.launch.py.

Works on modern ROS 2 with the ros_gz stack (Jazzy, Kilted, Lyrical, ...).
No distro name is hardcoded and the ros_gz_sim interface used here
(gz_sim.launch.py, the `create` node, empty.sdf) is stable across these
releases. Check your distro with `echo $ROS_DISTRO` and pair it with the
matching Gazebo version.

Usage
-----
    ros2 launch simple_camera_pid track_world.launch.py map:=track_hard
    ros2 launch simple_camera_pid track_world.launch.py map:=complex gui:=false

Available maps:
    simple  complex  ellipse  training  track_easy  track_medium  track_hard

Prerequisites
-------------
  * The track model packages must be on the Gazebo resource path. By default
    `models_path` already points at this repo's own model/gazebo/tracks (via
    `colcon build --symlink-install`), so nothing extra is needed. To use a
    different copy instead (e.g. one installed under
    /opt/ros/<distro>/share/turtlebot3_gazebo/models/), pass
    models_path:=<that folder>, or models_path:="" to rely purely on
    whatever is already on GZ_SIM_RESOURCE_PATH.
  * The robot model itself (model/gazebo/turtlebot3/models/
    turtlebot3_burger_line_follower/) is vendored in this repo — official
    turtlebot3_burger ships with no camera, so this project's own model adds
    one (same Intel RealSense r200 optics as stock turtlebot3_waffle; see
    that model's model.config for details). It is spawned by absolute path
    (not `model://...`), so it resolves regardless of GZ_SIM_RESOURCE_PATH.
  * turtlebot3_gazebo must still be installed: the robot model's meshes
    (turtlebot3_common/meshes/...) and LIDAR/IMU/camera sensor plugins are
    referenced from that package, so its models/ folder needs to be on
    GZ_SIM_RESOURCE_PATH too (this launch file appends it automatically from
    the ROS 2 install, see set_turtlebot3_common_path below).

Notes
-----
This uses worlds/track_camera_world.sdf instead of ros_gz_sim's stock
empty.sdf: the stock world omits the Sensors system plugin, so the robot's
camera (and any lidar) never renders or publishes — every topic looks fine
except the one that matters for line following. This also starts a
ros_gz_bridge (config/gazebo/ros_gz_bridge_turtlebot3.yaml) so /cmd_vel, /odom,
/camera/image_raw, /camera/camera_info, /joint_states, /tf and /clock are
actually visible to ROS 2, not just Gazebo's own transport.
"""

import os

from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# This workspace's known location on this machine — checked first so
# `ros2 launch` works regardless of the caller's current working directory
# (the #1 cause of "Error finding file [model://<map>]": see the search
# fallback below for why cwd/install-layout alone isn't reliable).
KNOWN_WORKSPACE_TRACKS_PATH = (
    '/media/kevin/ding/final_project/Sheffield/for_getting_started/model/gazebo/tracks'
)


def _default_tracks_path() -> str:
    """Default GZ_SIM_RESOURCE_PATH folder for the track models.

    Walking up from this file's own installed location only finds the real
    source tree under a working `colcon-symlink-install`; without that
    extension (not always present — it wasn't in the environment this was
    last checked in), `colcon build --symlink-install` silently falls back
    to copying files, and the walk resolves into install/simple_camera_pid
    instead of the workspace root, so model/gazebo/tracks is never found and
    Gazebo can't resolve model://<map> — the track silently fails to spawn.

    Robust alternative: search upward from the current working directory for
    a model/gazebo/tracks marker — valid for the standard
    `cd <workspace_root> && ros2 launch ...` invocation. Falls back to the
    file-location walk (correct for a source checkout or a working
    symlink-install) if that search comes up empty.
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


# Default to this repo's own track models so `map:=` works out of the box,
# without copying packages into turtlebot3_gazebo/models or passing
# models_path by hand. See _default_tracks_path() for why this is a search
# rather than a plain relative-path computation.
DEFAULT_TRACKS_PATH = _default_tracks_path()


def _default_burger_model_file() -> str:
    """Absolute path to this repo's vendored turtlebot3_burger_line_follower model.sdf.

    Derived from DEFAULT_TRACKS_PATH (model/gazebo/tracks) by walking to its
    sibling model/gazebo/turtlebot3/models/ — same repo-root search already
    solved there, so this just reuses its result instead of duplicating the
    cwd-walk/colcon-layout fallback logic.
    """
    tracks_dir = DEFAULT_TRACKS_PATH
    gazebo_dir = os.path.dirname(tracks_dir)
    return os.path.join(
        gazebo_dir, 'turtlebot3', 'models',
        'turtlebot3_burger_line_follower', 'model.sdf',
    )


# Absolute path (not model://...) so the robot model resolves regardless of
# GZ_SIM_RESOURCE_PATH — only its mesh/sensor-plugin dependencies on
# turtlebot3_gazebo still need that env var (see turtlebot3_common models
# path appended in generate_launch_description()).
DEFAULT_BURGER_MODEL_FILE = _default_burger_model_file()

# Per-track robot start pose (on the white line, facing along it).
# x, y, z in metres; yaw in radians. Matches the MuJoCo scenes.
TRACK_START_POSES = {
    'simple':       {'x': -1.419, 'y': -0.025, 'z': 0.010, 'yaw': 1.5708},
    'complex':      {'x': -1.227, 'y': -0.446, 'z': 0.010, 'yaw': 1.5708},
    'ellipse':      {'x': -1.450, 'y':  0.000, 'z': 0.010, 'yaw': 1.5708},
    'training':     {'x': -1.460, 'y':  0.000, 'z': 0.010, 'yaw': 1.5708},
    'track_easy':   {'x':  1.550, 'y':  0.000, 'z': 0.010, 'yaw': 1.4129},
    'track_medium': {'x':  1.669, 'y':  0.000, 'z': 0.010, 'yaw': 1.0648},
    'track_hard':   {'x':  1.923, 'y':  0.000, 'z': 0.010, 'yaw': 1.2819},
}

DEFAULT_MAP = 'track_hard'


def _spawn_setup(context, *args, **kwargs):
    """Resolve the map argument, validate it, and build the spawn action."""
    map_name = LaunchConfiguration('map').perform(context)

    if map_name not in TRACK_START_POSES:
        valid = ', '.join(sorted(TRACK_START_POSES))
        raise RuntimeError(
            f"Unknown map '{map_name}'. Valid maps: {valid}."
        )

    pose = TRACK_START_POSES[map_name]

    # Spawn the track ground plane (model://<map>) at the origin. Raised 1mm
    # above z=0 so it doesn't z-fight with empty.sdf's own ground_plane,
    # which is a same-sized-or-larger plane also sitting exactly at z=0 —
    # coplanar visuals flicker unpredictably as the camera moves. The track
    # model has no collision geometry, so this offset has no physics effect.
    spawn_track = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_track',
        output='screen',
        arguments=[
            '-name', map_name,
            '-file', f'model://{map_name}',
            '-x', '0', '-y', '0', '-z', '0.001',
        ],
    )

    # Spawn the TurtleBot3 Burger at the track's start pose. Spawned by
    # absolute path (this repo's vendored model, see
    # DEFAULT_BURGER_MODEL_FILE) rather than model://turtlebot3_burger since
    # the stock package's turtlebot3_burger has no camera — this project's
    # copy adds one.
    burger_model_file = LaunchConfiguration('burger_model_file').perform(context)
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_turtlebot3',
        output='screen',
        arguments=[
            '-name', 'turtlebot3_burger',
            '-file', burger_model_file,
            '-x', str(pose['x']),
            '-y', str(pose['y']),
            '-z', str(pose['z']),
            '-Y', str(pose['yaw']),
        ],
    )

    return [spawn_track, spawn_robot]


def generate_launch_description():
    gui = LaunchConfiguration('gui')
    models_path = LaunchConfiguration('models_path')

    # Make the track models discoverable by Gazebo. Defaults to this repo's
    # own model/gazebo/tracks (see DEFAULT_TRACKS_PATH above); pass a
    # different folder (e.g. turtlebot3_gazebo/models) or "" to override.
    set_resource_path = AppendEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=models_path,
    )

    # The vendored turtlebot3_burger_line_follower model references
    # turtlebot3_gazebo's shared meshes/sensor plugins via
    # model://turtlebot3_common/... — append that package's models/ folder so
    # those URIs resolve regardless of the caller's own GZ_SIM_RESOURCE_PATH.
    set_turtlebot3_common_path = AppendEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=PathJoinSubstitution([FindPackageShare('turtlebot3_gazebo'), 'models']),
    )

    gz_launch = PathJoinSubstitution([
        FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py',
    ])
    world_file = PathJoinSubstitution([
        FindPackageShare('simple_camera_pid'), 'worlds', 'track_camera_world.sdf',
    ])
    bridge_config = PathJoinSubstitution([
        FindPackageShare('simple_camera_pid'), 'config', 'gazebo', 'ros_gz_bridge_turtlebot3.yaml',
    ])

    # Start Gazebo with the Sensors-enabled world; models are spawned into
    # it below. gui:=true  -> run server + GUI (-r)
    #           gui:=false -> headless server only (-s -r)
    gz_sim_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch),
        condition=IfCondition(gui),
        launch_arguments={
            'gz_args': ['-r -v4 ', world_file],
            'on_exit_shutdown': 'true',
        }.items(),
    )
    gz_sim_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch),
        condition=UnlessCondition(gui),
        launch_arguments={
            'gz_args': ['-s -r -v4 ', world_file],
            'on_exit_shutdown': 'true',
        }.items(),
    )

    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge_turtlebot3',
        output='screen',
        parameters=[{
            'config_file': bridge_config,
            'use_sim_time': True,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=DEFAULT_MAP,
            description=(
                'Track to load. One of: '
                + ', '.join(sorted(TRACK_START_POSES)) + '.'
            ),
        ),
        DeclareLaunchArgument(
            'models_path',
            default_value=DEFAULT_TRACKS_PATH,
            description=(
                'Folder appended to GZ_SIM_RESOURCE_PATH so Gazebo can find '
                "the track models. Defaults to this repo's own "
                'model/gazebo/tracks. Pass an empty string to skip, or '
                'override if the models are installed elsewhere (e.g. '
                'turtlebot3_gazebo/models).'
            ),
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Run the Gazebo GUI (false = headless server only).',
        ),
        DeclareLaunchArgument(
            'burger_model_file',
            default_value=DEFAULT_BURGER_MODEL_FILE,
            description=(
                'Absolute path to the TurtleBot3 Burger model.sdf to spawn. '
                "Defaults to this repo's own vendored "
                'model/gazebo/turtlebot3/models/turtlebot3_burger_line_follower/model.sdf.'
            ),
        ),
        set_resource_path,
        set_turtlebot3_common_path,
        gz_sim_gui,
        gz_sim_headless,
        ros_gz_bridge,
        OpaqueFunction(function=_spawn_setup),
    ])
