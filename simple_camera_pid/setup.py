from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'simple_camera_pid'


setup(
    name=package_name,
    version='0.1.0',
    # pipo_verl: an unrelated LLM-RL training framework that ended up nested
    # under simple_camera_pid/training/ by accident (not part of this
    # project) -- excluded so colcon/setuptools doesn't try to package its
    # ~400 files as part of this ROS package.
    packages=find_packages(exclude=[
        'test', 'simple_camera_pid.training.pipo_verl',
        'simple_camera_pid.training.pipo_verl.*',
    ]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'config', 'mujoco'),
            glob(os.path.join('config', 'mujoco', '*.yaml'))),
        (os.path.join('share', package_name, 'config', 'real'),
            glob(os.path.join('config', 'real', '*.yaml'))),
        (os.path.join('share', package_name, 'launch', 'mujoco'),
            glob(os.path.join('launch', 'mujoco', '*.launch.py'))),
        (os.path.join('share', package_name, 'launch', 'real'),
            glob(os.path.join('launch', 'real', '*.launch.py'))),
        (os.path.join('share', package_name, 'rviz'),
            glob(os.path.join('rviz', '*.rviz'))),
        (os.path.join('share', package_name, 'matlab'),
            glob(os.path.join('matlab', '*.m'))),
        (os.path.join('share', package_name, 'tools'),
            glob(os.path.join('tools', '*.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kevin',
    maintainer_email='kevin@todo.todo',
    description='Camera-based PID + optional SAC/PPO residual line follower (real TurtleBot3 and MuJoCo).',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'mujoco_line_follower_node = simple_camera_pid.mujoco.mujoco_line_follower_node:main',
            'line_follower_node = simple_camera_pid.real.line_follower_node:main',
            'line_follower_vision_node = simple_camera_pid.real.vision_node:main',
            'line_follower_control_node = simple_camera_pid.real.control_node:main',
            'vision_debug_node = simple_camera_pid.common.vision_debug_node:main',
            'trajectory_path_node = simple_camera_pid.common.trajectory_path_node:main',
            'mujoco_run_pid_baseline = '
                'simple_camera_pid.mujoco.experiments.run_turtlebot3_mujoco_pid:main',
            'mujoco_run_trial = '
                'simple_camera_pid.mujoco.experiments.'
                'run_turtlebot3_burger_mujoco_visual_line_follower:main',
            'mujoco_tune_pid = simple_camera_pid.mujoco.experiments.tune_complex_track_pid:main',
            'mujoco_tune_localpath = simple_camera_pid.mujoco.experiments.tune_localpath_gains:main',
            'mujoco_train_sac = simple_camera_pid.training.train_sac_residual:main',
            'mujoco_train_ppo = simple_camera_pid.training.train_ppo_residual:main',
        ],
    },
)
