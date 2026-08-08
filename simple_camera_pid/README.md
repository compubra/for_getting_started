# Simple Camera PID

This package drives a camera-based TurtleBot3 Burger line follower in Gazebo
and MuJoCo: a PID controller (`common/control/`) on the vision pipeline's
steering/lateral/heading error (`common/vision.py`), with an optional trained
SAC/PPO residual policy layered on top.

7 tracks are available (`simple`, `complex`, `ellipse`, `training`,
`track_easy`, `track_medium`, `track_hard`), each a self-contained model under
`model/gazebo/tracks/` (Gazebo) / `model/mujoco/turtlebot3/` (MuJoCo), with a
low boundary curb around the 4.4 m edge so the robot can't drive off the track
footprint even if the controller loses the line.

> 2026-07-28: the original `simple_camera_pid_node` baseline (its own
> `image_line_detector.py` vision code, tuned years ago for a since-replaced
> Gazebo world) has been removed — it misdetected that world's own oversized
> default `ground_plane` as the line and drove straight through curves. The
> `common/vision.py`-based pipeline below (shared with MuJoCo) replaces it and
> does not have that failure mode.

## 文件分类一览(按用途:Gazebo / MuJoCo / 真车 / 训练 / 通用)

这份包里同时装着仿真(Gazebo、MuJoCo)、真实 TurtleBot3、RL 训练四条线,共用同一套
`common/` 视觉+PID 代码。下表是当前实际目录结构(2026-07-31)按用途分类,方便按需
定位文件——不确定某个文件属于哪条线时先查这里,而不是猜文件名。

### Gazebo 专用

| 路径 | 用途 |
| --- | --- |
| `config/gazebo/gazebo_line_follower.yaml` | PID/视觉参数(800x600 Gazebo 相机) |
| `config/gazebo/ros_gz_bridge_turtlebot3.yaml` | `ros_gz_bridge` 话题映射 |
| `launch/gazebo/track_world.launch.py` | 起 Gazebo 世界 + 生成机器人 |
| `launch/gazebo/track_bringup.launch.py` | world + controller 一起起(`controller:=false` 只起 world) |
| `launch/gazebo/gazebo_line_follower.launch.py` | 只起控制节点(`camera_profile:=real` 时其实是真车用这个) |
| `launch/gazebo/gazebo_monitor.launch.py` | rviz 监控(annotated 相机 + 轨迹) |
| `simple_camera_pid/gazebo/gazebo_line_follower_node.py` | 控制节点本体——**也是真车用的同一个节点**,见下方"真车"分类 |
| `worlds/track_camera_world.sdf` | Gazebo 世界文件 |
| `rviz/gazebo_monitor.rviz` | rviz 配置 |

### MuJoCo 专用

| 路径 | 用途 |
| --- | --- |
| `config/mujoco/mujoco_line_follower.yaml` | PID/视觉参数 |
| `launch/mujoco/mujoco_line_follower.launch.py` | 启动入口(默认自带训练好的 SAC 残差策略) |
| `simple_camera_pid/mujoco/mujoco_line_follower_node.py` | ROS 节点(内部自跑物理+相机渲染+视觉+PID) |
| `simple_camera_pid/mujoco/sim/turtlebot3_mujoco_env.py` | PID-baseline 物理+视觉+PID 仿真环境 |
| `simple_camera_pid/mujoco/runtime/`、`experiments/`、`map_building/`、`model_building/` | 场景解析、实验脚本、赛道贴图/MJCF 生成 |
| `simple_camera_pid/mujoco/README.md` | 设计说明 + 对照 MATLAB 验证过的内容 |
| `rviz/mujoco_line_follower.rviz` | rviz 配置 |
| `matlab/**` | 原 MATLAB/Simulink 工作包(`.slx` 模型、训练脚本),MuJoCo 端口的对照来源 |

### 真车专用(真实 TurtleBot3)

| 路径 | 用途 |
| --- | --- |
| `config/real/real_line_follower.yaml` | 真机参数,`/**:` 段是三种部署方式共用的视觉/相机/PID 调参,下面每个节点名各自的段只放话题名这类节点专属配置 |
| `launch/real/real_line_follower.launch.py` | 单进程部署的启动入口(见下),`use_sim_time` 恒为 false,带 `ROS_DOMAIN_ID` 校验 |
| `simple_camera_pid/common/camera_geometry.py` 里的 `turtlebot3_burger_real_camera()` | 真实相机几何标定函数(FOV/安装高度/俯仰角需实测填入) |
| `simple_camera_pid/real/vision_node.py` | 拆分部署的视觉半节点——跑在**树莓派**上,订阅相机原始帧,只把 `steering_error`/`lateral_error`/`heading_error`/`confidence`/`found` 这 5 个浮点数发到 `/line_follower/local_path`,不发原始图像 |
| `simple_camera_pid/real/control_node.py` | 拆分部署的控制半节点——跑在 **PC** 上,订阅上面那 5 个浮点数,跑 PID(+可选残差 RL),发布 `/cmd_vel`;带网络掉线看门狗(`watchdog_timeout`,默认 1s 没收到新消息就当作丢线处理,不会一直重放旧指令) |
| `simple_camera_pid/real/local_path_msg.py` | 上面两个节点之间的消息打包/解包(`Float32MultiArray`,没有单独开 `.msg` 接口包) |

真车有**两种部署方式**,共用同一份 `common/` 视觉+PID 代码和同一份 `real_line_follower.yaml`:

- **单进程**(`gazebo_line_follower_node.py` + `real_line_follower.launch.py`):视觉+PID 在同一个节点里,跑在哪台机器都行,只要那台机器能订阅到相机话题。靠 `camera_profile` 参数(`gazebo`/`real`)切换相机几何来源,节点本身对图像来自 Gazebo 插件还是真实相机驱动是无感的。
- **拆分**(`vision_node.py` 在树莓派 + `control_node.py` 在 PC):树莓派本地算视觉(不用把 800x600 原始 RGB 图像走 WiFi 传给 PC,那个带宽扛不住),只把 5 个浮点数传过去;PC 只算 PID/RL,把结果传回来给树莓派的 `turtlebot3_node` 转电机。两台机器之间只靠 ROS2 话题(DDS)通信,`ROS_DOMAIN_ID` 一致即可,不用额外写传输代码。

### 训练专用(RL 训练,不参与在线控制)

| 路径 | 用途 |
| --- | --- |
| `simple_camera_pid/training/residual_env.py` | 包一层 MuJoCo PID 环境,加 SAC/PPO 残差动作+奖励+域随机化 |
| `simple_camera_pid/training/domain_randomization.py` | 每回合随机扰动位姿/动力学/PID/感知(可选随机地图) |
| `simple_camera_pid/training/io_specs.py` | 观测/动作空间的唯一真相源 |
| `simple_camera_pid/training/plotting.py` | 读 `monitor.csv` 画训练曲线 |
| `simple_camera_pid/training/sac_training_config.py`、`ppo_training_config.py` | 超参数(改这里调参) |
| `simple_camera_pid/training/train_sac_residual.py`、`train_ppo_residual.py` | 训练入口 |
| `simple_camera_pid/training/hpc_scripts/` | 独立于本仓库的 HPC 集群训练脚本(依赖外部 `lsac` 包,复制到集群跑,不在本地跑) |
| `simple_camera_pid/training/README.md` | 设计说明 + 已验证内容 |
| `hpc/0720/`、`hpc/0721/`、`hpc/0724/` | 每批 HPC 训练的归档产物(checkpoint、日志、`compare_vs_pid.csv`) |

> ⚠️ `simple_camera_pid/training/pipo_verl/` 是一整套跟本项目无关的 LLM RL 训练框架
> (`verl`,含 megatron/vllm/sglang 等几百个文件),疑似误操作混进来的,已经在
> `setup.py` 的 `find_packages()` 里排除(不会被打包/安装),但文件本身还在磁盘上,
> 建议确认无用后手动删除。

### 通用(Gazebo / MuJoCo / 真车三条线共用)

| 路径 | 用途 |
| --- | --- |
| `simple_camera_pid/common/camera_geometry.py` | 针孔相机 + IPM 投影,三个平台的 `CameraParams` 预设(`mujoco`/`gazebo`/`real`) |
| `simple_camera_pid/common/config.py` | 共享默认参数(`RobotConfig`/`ControllerConfig`/`VisionConfig` 等) |
| `simple_camera_pid/common/vision.py` | 循线视觉算法(霍夫种子 + 滑动窗口 + 地面二次拟合) |
| `simple_camera_pid/common/control/pid.py`、`control/line_follower_controller.py` | 滤波 PID + 恢复转向 + 差速运动学 |
| `simple_camera_pid/common/debug_frame.py` | 视觉调试叠加画面渲染 |
| `simple_camera_pid/common/vision_debug_node.py` | 可视化视觉链路的 ROS 节点,`platform:=gazebo/mujoco/real` |
| `simple_camera_pid/common/trajectory_path_node.py` | 发布行驶轨迹供 rviz 显示 |
| `simple_camera_pid/common/residual_policy.py` | 加载/运行训练好的 SAC/PPO 残差策略(推理用,三条控制线共用) |
| `simple_camera_pid/common/random_path.py` | 纯几何参考曲线生成器,独立于任何平台 |
| `simple_camera_pid/common/README.md` | 本目录设计说明 |
| `package.xml`、`setup.py`、`README.md`(本文件)、`resource/simple_camera_pid` | 包元数据/构建配置 |
| `requirements-vision.txt`、`requirements-residual.txt`、`requirements-sim.txt` | 分层 pip 依赖(逐层包含,按机器装最深那层) |
| `tools/analyze_simple_track.py` | 赛道贴图分析工具 |

## Build

From the workspace root:

```bash
colcon build --packages-select simple_camera_pid
source install/setup.bash
```

## Run the controller (Gazebo)

```bash
# World + robot + pure PID controller (no RL) on track_hard
ros2 launch simple_camera_pid track_bringup.launch.py map:=track_hard

# Simulation only, bring your own controller
ros2 launch simple_camera_pid track_bringup.launch.py map:=complex controller:=false

# With a trained SAC residual policy on top of the PID
ros2 launch simple_camera_pid track_bringup.launch.py \
    map:=track_hard residual_model_path:=/path/to/sac_agent.zip
```

`track_bringup.launch.py` composes `track_world.launch.py` (Gazebo + track +
spawned robot) and `gazebo_line_follower.launch.py` (the controller node,
`residual_model_path:=''` by default = pure PID). Call
`gazebo_line_follower.launch.py` directly instead if you want its own
best-checkpoint default (see that file's docstring).

Useful first tuning knobs are in `config/gazebo/gazebo_line_follower.yaml`
(PID gains, vision ROI/lookahead/gains, robot kinematics). For a live view of
what the controller sees while tracking (annotated camera feed, pose trail,
commanded-vs-measured velocity), run alongside:

```bash
ros2 launch simple_camera_pid gazebo_monitor.launch.py
```

## Analyze the texture

```bash
python3 <package-source>/tools/analyze_simple_track.py
```

## MuJoCo line follower (self-contained, no Gazebo)

`simple_camera_pid/mujoco/` is a Python port of the workspace's `matlab/`
TurtleBot3 visual-line-follower work package (MuJoCo physics, pure-pursuit
vision, PID, and SAC/PPO residual RL) — see
`simple_camera_pid/mujoco/README.md` for the full design notes and what was
verified against MATLAB. It runs independently of the Gazebo-based baseline
above: `mujoco_line_follower_node` drives MuJoCo physics + camera render +
vision + PID internally on its own timer, publishing `/camera/image_raw`,
`/odom`, `/cmd_vel`, and a diagnostics topic purely for rqt/rviz visibility
(the control loop is closed inside the node, not over these topics).

Install its extra dependencies first (not resolvable via rosdep):

```bash
pip install -r requirements-sim.txt
```

Then, from the workspace root:

```bash
colcon build --packages-select simple_camera_pid
source install/setup.bash
ros2 launch simple_camera_pid mujoco_line_follower.launch.py
ros2 launch simple_camera_pid mujoco_line_follower.launch.py map_key:=track_hard
```

Tuning knobs are in `config/mujoco/mujoco_line_follower.yaml` (PID gains, vision ROI/
gains, robot kinematics — read out of the original `.slx` model workspace).

By default the launch file loads a trained SAC residual policy on top of the
PID baseline — the best `hpc/0724/simulation_data/sac_residual_training/
param_sweep_0724_ep800/roi0.3` checkpoint by `compare_vs_pid.csv`, 2-D
action `[delta_v, delta_omega]` (see `DEFAULT_RESIDUAL_MODEL_PATH` in
`launch/mujoco/mujoco_line_follower.launch.py`). Getting these newer HPC sweep
checkpoints usable at all took two fixes, documented in
`simple_camera_pid/common/residual_policy.py`: they're pickled with a newer
Numpy than this machine has (worked around via stable-baselines3's
`custom_objects`), and they use an 8-D observation / 2-D action space from
the HPC-side 2026-07-21 `Diff_Drive_Kinematics` refactor that the ROS nodes
didn't originally build/consume (`residual_use_2d_action:=true`, now the
default). The pre-sweep 2026-07-12 checkpoint (7-D obs / 1-D action) is
still available as an older-generation fallback. To run the plain PID
baseline instead, or point at a different checkpoint:

```bash
ros2 launch simple_camera_pid mujoco_line_follower.launch.py residual_model_path:=''
ros2 launch simple_camera_pid mujoco_line_follower.launch.py \
    residual_model_path:=hpc/sac_residual_training/20260712_210946/sac_residual_agent_converted.zip \
    residual_algo:=sac residual_use_2d_action:=false
```

Train one with (see `simple_camera_pid/mujoco/training/*_training_config.py`
for hyperparameters):

```bash
ros2 run simple_camera_pid mujoco_train_sac
ros2 run simple_camera_pid mujoco_train_ppo
```

Other console entry points: `mujoco_run_pid_baseline`, `mujoco_run_trial`,
`mujoco_tune_pid`, `mujoco_tune_localpath` (headless smoke tests and PID/
LocalPath gain sweeps — see `simple_camera_pid/mujoco/experiments/`).
