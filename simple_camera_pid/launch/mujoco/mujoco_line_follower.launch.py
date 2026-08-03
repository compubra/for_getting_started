"""Launch the self-contained MuJoCo camera line follower node.

Runs physics + camera render + vision + PID (and optionally a trained
SAC/PPO residual policy) inside one node, independent of Gazebo — see
simple_camera_pid/mujoco_line_follower_node.py. Publishes /camera/image_raw,
/odom, /cmd_vel, and a diagnostics topic for rqt/rviz.

Usage
-----
    ros2 launch simple_camera_pid mujoco_line_follower.launch.py
        (runs with the default residual_model_path -- the hpc/0724
        param_sweep_0724_ep800/roi0.3 SAC checkpoint, 2-D action, see
        DEFAULT_RESIDUAL_MODEL_PATH below)
    ros2 launch simple_camera_pid mujoco_line_follower.launch.py map_key:=track_hard
    ros2 launch simple_camera_pid mujoco_line_follower.launch.py use_viewer:=true
    ros2 launch simple_camera_pid mujoco_line_follower.launch.py rviz:=true
    ros2 launch simple_camera_pid mujoco_line_follower.launch.py residual_model_path:=''
        (pure PID baseline, no residual policy)
    ros2 launch simple_camera_pid mujoco_line_follower.launch.py \\
        residual_model_path:=/path/to/other_sac_agent.zip residual_algo:=sac \\
        residual_use_wheel_speed_obs:=true residual_use_2d_action:=false
        (older-generation 7-D-obs/1-D-action checkpoint, e.g. the pre-sweep
        2026-07-12 run -- see simple_camera_pid.common.residual_policy for
        the full checkpoint-generation table)

Arguments
---------
    map_key              MuJoCo track (default simple). One of: simple
                          complex ellipse training track_easy track_medium
                          track_hard winding.
    repo_root             Workspace root containing model/mujoco/ (default:
                          hardcoded to this workspace's own root -- see
                          WORKSPACE_ROOT below. The node's own cwd-search
                          auto-detect only works with a true
                          `colcon build --symlink-install`; on this machine
                          `--symlink-install` silently falls back to plain
                          copies (modern setuptools removed the `--editable`
                          flag `colcon-core` needs for ament_python symlink
                          installs), so auto-detect resolves into
                          install/simple_camera_pid instead of the workspace
                          root. Override with repo_root:=<path> if you ever
                          move this workspace.
    config_file           Node parameter YAML (default
                          mujoco_line_follower.yaml).
    residual_model_path   Trained SAC/PPO residual policy .zip (default:
                          DEFAULT_RESIDUAL_MODEL_PATH below -- the best
                          hpc/0724 sweep checkpoint by compare_vs_pid.csv
                          that also matches this node's roi_bottom_fraction;
                          see the constant's comment). Pass
                          residual_model_path:='' for the pure PID baseline.
    residual_algo         "sac" or "ppo", matching residual_model_path
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
    residual_max_delta_v  Clip bound for the delta_v action component,
                          rad/s-equivalent velocity units (default 1.0,
                          matching HPC's --max-delta-v default). Only used
                          when residual_use_2d_action:=true.
    use_viewer            Open an on-screen MuJoCo window showing the whole
                          track + car, in addition to the /camera/image_raw
                          onboard-camera topic (default false). Requires a
                          display.
    rviz                  Launch rviz2 with a config showing raw
                          (/camera/image_raw) and annotated
                          (/vision_debug/image_raw) camera feeds, current
                          pose, and the driven trajectory (default false).
                          Requires a display. Also starts vision_debug_node
                          (platform:=mujoco) and trajectory_path_node so
                          those two rviz panels have something to show.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Hardcoded to this workspace's own location (see the repo_root argument's
# description for why: --symlink-install can't be relied on here).
WORKSPACE_ROOT = '/media/kevin/ding/final_project/Sheffield/for_getting_started'

# Best-scoring checkpoint by hpc/0724/simulation_data/sac_residual_training/
# param_sweep_0724_ep800/compare_vs_pid.csv (2nd place by mean_reward,
# -52.03 vs PID-only -193.95; best mean_wheel_sat_ratio in that sweep) that
# also matches this node's roi_bottom_fraction=0.30 default -- the only
# param_sweep_0724_ep800 run trained with --roi-fraction 0.3 (every other
# run used the lsac script's own default of 0.5). Still trained at the lsac
# env's hardcoded lookahead_distance=0.20, not this node's 0.40 default (no
# HPC run has ever swept lookahead) -- the closest match available, not an
# exact one.
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
    this_pkg = get_package_share_directory('simple_camera_pid')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(this_pkg, 'rviz', 'mujoco_line_follower.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    # 2026-07-29: RViz2's stock Odometry display only draws discrete
    # Arrow/Axes markers (a trail of them reads as scattered arrows, not a
    # trajectory) -- this republishes /odom as a nav_msgs/Path on
    # /trajectory so the rviz config's Path display can draw the actual
    # driven route as a line. Only useful alongside rviz, so tied to the
    # same condition rather than its own argument.
    trajectory_path_node = Node(
        package='simple_camera_pid',
        executable='trajectory_path_node',
        name='trajectory_path_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    # Annotated feed for mujoco_line_follower.rviz's second Image panel --
    # vision_debug_node (common/vision_debug_node.py, generalized 2026-07-29)
    # run with the MuJoCo camera profile. Params below mirror
    # config/mujoco/mujoco_line_follower.yaml (vision_debug_node's own
    # declared defaults are the Gazebo node's numbers, notably
    # lookahead_distance=0.20 rather than MuJoCo's 0.40, so they need
    # overriding here rather than being left at the node's built-ins). Only
    # useful alongside rviz, so tied to the same condition rather than its
    # own argument.
    vision_debug_node = Node(
        package='simple_camera_pid',
        executable='vision_debug_node',
        name='vision_debug_node',
        output='screen',
        parameters=[{
            'platform': 'mujoco',
            'image_height': 480,
            'image_width': 640,
            'roi_bottom_fraction': 0.30,
            'lookahead_distance': 0.40,
            'lateral_gain': 0.6,
            'heading_gain': 0.35,
            'curvature_gain': 0.04,
            'roi_widen_step': 0.2,
            'roi_widen_max': 0.7,
        }],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    node = Node(
        package='simple_camera_pid',
        executable='mujoco_line_follower_node',
        name='mujoco_line_follower_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'repo_root': LaunchConfiguration('repo_root'),
                'map_key': LaunchConfiguration('map_key'),
                'residual_model_path': LaunchConfiguration('residual_model_path'),
                'residual_algo': LaunchConfiguration('residual_algo'),
                'residual_use_wheel_speed_obs': ParameterValue(
                    LaunchConfiguration('residual_use_wheel_speed_obs'), value_type=bool),
                'residual_use_2d_action': ParameterValue(
                    LaunchConfiguration('residual_use_2d_action'), value_type=bool),
                'residual_max_delta_v': ParameterValue(
                    LaunchConfiguration('residual_max_delta_v'), value_type=float),
                'use_viewer': ParameterValue(LaunchConfiguration('use_viewer'), value_type=bool),
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map_key',
            default_value='simple',
            description='MuJoCo track: simple, complex, ellipse, training, '
                        'track_easy, track_medium, track_hard, winding.',
        ),
        DeclareLaunchArgument(
            'repo_root',
            default_value=WORKSPACE_ROOT,
            description='Workspace root containing model/mujoco/. Hardcoded '
                        'since --symlink-install (and thus the node\'s own '
                        'auto-detect fallback) is unreliable on this machine.',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=os.path.join(this_pkg, 'config', 'mujoco', 'mujoco_line_follower.yaml'),
            description='Node parameter YAML.',
        ),
        DeclareLaunchArgument(
            'residual_model_path',
            default_value=DEFAULT_RESIDUAL_MODEL_PATH,
            description='Trained SAC/PPO residual policy .zip (default: the '
                        'best hpc/0724 sweep checkpoint by compare_vs_pid.csv '
                        'that also matches this node roi_bottom_fraction -- '
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
                        'trainer (simple_camera_pid.training, ros2 run '
                        'mujoco_train_sac/mujoco_train_ppo).',
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
            'use_viewer',
            default_value='true',
            description='Open an on-screen MuJoCo viewer window showing the '
                        'whole track and car (in addition to the '
                        '/camera/image_raw onboard-camera topic). Requires a '
                        'display.',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='false',
            description='Launch rviz2 (raw + annotated camera, pose, '
                        'trajectory) plus the vision_debug_node/'
                        'trajectory_path_node it needs. Requires a display.',
        ),
        node,
        rviz_node,
        vision_debug_node,
        trajectory_path_node,
    ])
