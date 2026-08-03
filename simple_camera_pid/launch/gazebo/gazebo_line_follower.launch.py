"""Launch the SAC/PPO-residual (or plain-PID) line follower against Gazebo.

Starts gazebo_line_follower_node only -- bring up the world + robot first
with track_world.launch.py (or track_bringup.launch.py controller:=false),
same division of labor as simple_camera_pid.launch.py.

This same node/launch file is also what to point at a real TurtleBot3 (swap
the image source and cmd_vel sink outside this launch file's scope -- the
node itself is agnostic to whether frames come from Gazebo's camera plugin
or a real camera driver).

Usage
-----
    ros2 launch simple_camera_pid track_bringup.launch.py map:=track_hard controller:=false
    ros2 launch simple_camera_pid gazebo_line_follower.launch.py
        (runs with the default residual_model_path -- the hpc/0724
        param_sweep_0724_ep800/roi0.3 SAC checkpoint, 2-D action, see
        DEFAULT_RESIDUAL_MODEL_PATH below)
    ros2 launch simple_camera_pid gazebo_line_follower.launch.py residual_model_path:=''
        (pure PID baseline, no residual policy)
    ros2 launch simple_camera_pid gazebo_line_follower.launch.py \\
        residual_model_path:=/path/to/other_sac_agent.zip residual_algo:=sac \\
        residual_use_wheel_speed_obs:=true residual_use_2d_action:=false
        (older-generation 7-D-obs/1-D-action checkpoint, e.g. the pre-sweep
        2026-07-12 run -- see simple_camera_pid.common.residual_policy for
        the full checkpoint-generation table)
    ros2 launch simple_camera_pid gazebo_line_follower.launch.py \\
        use_sim_time:=false residual_model_path:='' \\
        config_file:=/path/to/simple_camera_pid/share/config/real/real_line_follower.yaml
        (real TurtleBot3: no /clock, pure PID baseline, real-camera geometry
        via config/real/real_line_follower.yaml's camera_profile:=real --
        fill in that file's camera_fovy_deg/camera_mount_height/
        camera_pitch_deg/image_topic TODOs from an actual measurement first,
        see turtlebot3_burger_real_camera()'s docstring in
        common/camera_geometry.py)

Arguments
---------
    use_sim_time           Use simulation time (default true) -- important:
                          the control loop's fixed Ts_Control timer must run
                          on simulated time, not wall-clock, to stay
                          correctly calibrated if Gazebo runs off real-time.
                          Set false for a real robot (no /clock source).
    config_file            Node parameter YAML (default
                          gazebo_line_follower.yaml).
    residual_model_path    Trained SAC/PPO residual policy .zip (default:
                          DEFAULT_RESIDUAL_MODEL_PATH below -- the best
                          hpc/0724 sweep checkpoint by compare_vs_pid.csv
                          that also matches this node's roi_bottom_fraction/
                          lookahead_distance defaults; see the constant's
                          comment). Pass residual_model_path:='' for the
                          pure PID baseline.
    residual_algo           "sac" or "ppo", matching residual_model_path
                          (default sac).
    residual_use_wheel_speed_obs
                          True (default): observation includes normalized
                          wheel-speed feedback, matching every HPC-trained
                          checkpoint (hpc/0720 through hpc/0724). Set false
                          only for a checkpoint from the in-repo 5-D-obs
                          trainer (simple_camera_pid.training).
    residual_use_2d_action
                          True (default, matches the default checkpoint
                          above): 2-D action [delta_v, delta_omega],
                          matching hpc/0721+/hpc/0724's residual model
                          (post-2026-07-21 Diff_Drive_Kinematics refactor).
                          Set false for an older-generation 1-D-action
                          (delta_omega only) checkpoint -- hpc/0720 and
                          earlier, or the in-repo trainer.
    residual_max_delta_v   Clip bound for the delta_v action component,
                          rad/s-equivalent velocity units (default 1.0,
                          matching HPC's --max-delta-v default). Only used
                          when residual_use_2d_action:=true.
    rviz                   Launch rviz2 with camera + pose (default false).
                          Reuses gazebo_monitor.rviz -- its second Image
                          panel (/vision_debug/image_raw) stays blank unless
                          vision_debug_node is separately running (e.g. via
                          gazebo_monitor.launch.py).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# This workspace's own location on this machine -- gazebo_line_follower_node
# loads residual_model_path directly (no repo_root-relative resolution like
# the MuJoCo node has), so the default below must be absolute.
WORKSPACE_ROOT = '/media/kevin/ding/final_project/Sheffield/for_getting_started'

# Best-scoring checkpoint by hpc/0724/simulation_data/sac_residual_training/
# param_sweep_0724_ep800/compare_vs_pid.csv (2nd place by mean_reward,
# -52.03 vs PID-only -193.95; best mean_wheel_sat_ratio in that sweep). This
# node's own hardcoded vision defaults (gazebo_line_follower_node.py's
# _declare_parameters: roi_bottom_fraction=0.3, lookahead_distance=0.20)
# match this checkpoint's training geometry EXACTLY (roi_fraction=0.3 was
# this run's one deliberate override; lookahead stayed at the lsac script's
# untouched default of 0.20, which happens to equal Gazebo's own default --
# unlike the MuJoCo node/config, which uses lookahead_distance=0.40 and so
# only gets a partial geometry match from this same checkpoint).
#
# Two things had to be fixed in this node before ANY hpc/0721+/hpc/0724
# checkpoint (including this one) was usable here (both verified fixed
# 2026-07-28, see simple_camera_pid/common/residual_policy.py):
#   1. numpy pickle mismatch: these checkpoints were saved under Numpy 2.2.6
#      (see their system_info.txt); SAC.load()/PPO.load() under this node's
#      Numpy 1.26.4 used to segfault or raise "ModuleNotFoundError: No
#      module named 'numpy._core.numeric'" trying to unpickle them --
#      load_residual_model() now routes around the problem fields via SB3's
#      custom_objects mechanism instead (do NOT try aliasing numpy._core ->
#      numpy.core in sys.modules, that segfaults rather than raising).
#   2. these sweeps moved to an 8-D observation / 2-D action
#      [delta_v, delta_omega] space on 2026-07-21 (lsac/env.py's
#      observe_wheel_speeds branch); this node now builds that shape too
#      when residual_use_2d_action:=true (the default below).
# Re-point this at a newer checkpoint as sweeps continue; see
# hpc/*/simulation_data/*/compare_vs_pid.csv for other candidates. The
# pre-sweep 2026-07-12 SAC run (7-D obs/1-D action,
# hpc/sac_residual_training/20260712_210946/sac_residual_agent_converted.zip)
# remains available as an older-generation fallback -- pass it with
# residual_use_2d_action:=false.
DEFAULT_RESIDUAL_MODEL_PATH = os.path.join(
    WORKSPACE_ROOT, 'src', 'simple_camera_pid', 'hpc', '0724', 'simulation_data',
    'sac_residual_training', 'param_sweep_0724_ep800', 'roi0.3',
    'sac_residual_agent_20260727_181625.zip',
)


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    config_file = LaunchConfiguration('config_file')
    rviz = LaunchConfiguration('rviz')

    this_pkg = get_package_share_directory('simple_camera_pid')

    node = Node(
        package='simple_camera_pid',
        executable='gazebo_line_follower_node',
        name='gazebo_line_follower_node',
        output='screen',
        parameters=[
            config_file,
            {
                'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
                'residual_model_path': LaunchConfiguration('residual_model_path'),
                'residual_algo': LaunchConfiguration('residual_algo'),
                'residual_use_wheel_speed_obs': ParameterValue(
                    LaunchConfiguration('residual_use_wheel_speed_obs'), value_type=bool),
                'residual_use_2d_action': ParameterValue(
                    LaunchConfiguration('residual_use_2d_action'), value_type=bool),
                'residual_max_delta_v': ParameterValue(
                    LaunchConfiguration('residual_max_delta_v'), value_type=float),
            },
        ],
    )

    # RViz (optional) -- reuses gazebo_monitor.rviz (raw camera + annotated
    # vision_debug feed + current pose + driven trajectory).
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='gazebo_line_follower_rviz2',
        output='screen',
        arguments=['-d', os.path.join(this_pkg, 'rviz', 'gazebo_monitor.rviz')],
        condition=IfCondition(rviz),
    )

    # gazebo_monitor.rviz's second Image panel (/vision_debug/image_raw)
    # needs this running, and its Path display needs trajectory_path_node
    # publishing nav_msgs/Path on /trajectory -- both tied to the same rviz
    # flag so rviz:=true alone gives the full picture (2026-07-29: this used
    # to require separately remembering to also run gazebo_monitor.launch.py
    # for the annotated feed to show up).
    vision_debug_node = Node(
        package='simple_camera_pid',
        executable='vision_debug_node',
        name='vision_debug_node',
        output='screen',
        condition=IfCondition(rviz),
    )

    trajectory_path_node = Node(
        package='simple_camera_pid',
        executable='trajectory_path_node',
        name='trajectory_path_node',
        output='screen',
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time. Set false for a real robot.',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=os.path.join(this_pkg, 'config', 'gazebo', 'gazebo_line_follower.yaml'),
            description='PID and image-processing parameter file.',
        ),
        DeclareLaunchArgument(
            'residual_model_path',
            default_value=DEFAULT_RESIDUAL_MODEL_PATH,
            description='Trained SAC/PPO residual policy .zip (default: the '
                        'best hpc/0724 sweep checkpoint by compare_vs_pid.csv '
                        'that also matches this node\'s vision defaults -- '
                        'see DEFAULT_RESIDUAL_MODEL_PATH above). Pass '
                        "residual_model_path:='' for the pure PID baseline.",
        ),
        DeclareLaunchArgument(
            'residual_algo',
            default_value='sac',
            description='"sac" or "ppo", matching residual_model_path.',
        ),
        DeclareLaunchArgument(
            'residual_use_wheel_speed_obs',
            default_value='true',
            description='True (default): observation includes normalized '
                        'wheel-speed feedback, matching every HPC-trained '
                        'checkpoint (hpc/0720 through hpc/0724). Set false '
                        'only for a checkpoint from the in-repo 5-D-obs '
                        'trainer (simple_camera_pid.training).',
        ),
        DeclareLaunchArgument(
            'residual_use_2d_action',
            default_value='true',
            description='True (default, matches the default checkpoint '
                        'above): 2-D action [delta_v, delta_omega], matching '
                        'hpc/0721+/hpc/0724 (post-2026-07-21 '
                        'Diff_Drive_Kinematics refactor). Set false for an '
                        'older-generation 1-D-action checkpoint (hpc/0720 '
                        'and earlier, or the in-repo trainer).',
        ),
        DeclareLaunchArgument(
            'residual_max_delta_v',
            default_value='1.0',
            description='Clip bound for the delta_v action component '
                        '(velocity units, default 1.0 matching HPC). Only '
                        'used when residual_use_2d_action:=true.',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='false',
            description='Launch rviz2 (raw + annotated camera, pose, '
                        'trajectory -- reuses gazebo_monitor.rviz) plus the '
                        'vision_debug_node/trajectory_path_node it needs.',
        ),
        node,
        rviz_node,
        vision_debug_node,
        trajectory_path_node,
    ])
