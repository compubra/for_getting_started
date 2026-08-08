# TurtleBot3 visual line follower

正常使用时，在 MATLAB 当前目录切到本目录后运行：

```matlab
open_system("visual_line_follower_with_debug")
```

模型打开和仿真开始时会自动配置 MuJoCo 地图路径、缓存目录和运行时函数路径。地图路径不再硬编码为 complex；如果你手动在 MuJoCo Plant 里换了 XML，InitFcn 会优先保留这个有效路径。

## 地图选择方式

推荐使用模型工作区变量或运行脚本参数，Windows 和 Ubuntu 都兼容。

```matlab
% 方式 1：按内置地图 key 选择
workspace = get_param("visual_line_follower_with_debug", ...
    "ModelWorkspace");
workspace.assignin("TurtleBot3MuJoCoMap", "ellipse");
workspace.assignin("TurtleBot3MuJoCoScene", "");

% 方式 2：用完整 XML 路径，适合自定义地图
workspace.assignin("TurtleBot3MuJoCoScene", ...
    fullfile(pwd, "..", "..", "..", "model", "mujoco", ...
    "turtlebot3", "simple_camera_track_turtlebot3_burger_visual_scene.xml"));
workspace.assignin("TurtleBot3MuJoCoMap", "");
```

运行脚本也支持：

```matlab
[out, dataFile, summary] = run_turtlebot3_burger_mujoco_visual_line_follower( ...
    "None", 60, "MapKey", "simple");
```

可用地图 key：

- `simple`
- `complex`
- `ellipse`
- `training`

也可以设置环境变量：

- `TURTLEBOT3_MUJOCO_MAP=ellipse`
- `TURTLEBOT3_MUJOCO_SCENE=/path/to/your_scene.xml`

## 模型文件

| 文件 | 用途 |
| --- | --- |
| `visual_line_follower_with_debug.slx` | 主模型，MuJoCo Plant 内置仿真 |
| `visual_line_follower_with_debug_gazebo.slx` | Gazebo 变体，不含内置 Plant，靠 ROS2 Subscribe/Publish 接 `/camera/image_raw`、`/odom`、`/cmd_vel` |
| `visual_line_follower_with_debug_real.slx` | **真机调试变体**(2026-08-01 由 gazebo 版直接复制而来,话题名相同,原理上无需改动即可指向真实机器人)——**仅创建,未打开验证过**。已知两个待确认的坑,用之前必须自己核实: |
| `visual_line_follower_sac_residual.slx` / `_gazebo.slx` | 带 SAC 残差策略的对应变体 |
| `visual_line_follower_sac_residual_real.slx` | **真机 + RL 残差调试变体**(2026-08-01 由 `_gazebo` 版复制而来)——**仅创建,未打开验证过**。除了上面两条 debug 版的坑之外,还多两条: |

`visual_line_follower_with_debug_real.slx` 的已知坑(详见其 InitFcn 指向的
`runtime/init/configure_visual_line_follower_real_debug.m` 里的完整说明):

1. ~~`originbot_camera_profile.m` 没有真机分支~~ —— **2026-08-03 已修复**，该
   函数改为必填 `platform` 参数(`'mujoco'|'gazebo'|'real'`)，`case 'real'`
   存在且参数已与 Python 侧对齐。此条已过期，保留仅供追溯。

2. **`CmdVel_Publish` 的消息类型仍是 `geometry_msgs/Twist`——这条仍然成立，
   且是跑真机前的硬阻塞。** 从 `~/line_follower_bags/lap_20260731_152709`
   录制的数据看，这台机器人 `ROS2 Jazzy` 的 `turtlebot3_node` 实际收发的
   `/cmd_vel` 是 `geometry_msgs/msg/TwistStamped`，纯 `Twist` 连不上它的
   订阅。**注意**：本条原文说"这个坑同样存在于 Python 真机节点"——**该说法
   现已过时**，Python 侧 `real/line_follower_node.py` 与 `real/control_node.py`
   都有 `cmd_vel_stamped` 参数可切换，MATLAB 是唯一还没解决的一侧。改这条
   需要打开 `.slx` 改 Publish 块的消息类型，2026-08-08 这轮没做(当时没有
   可用的 MATLAB 会话)。

3. **视觉算法曾落后于 Python 侧一次关键修复——2026-08-08 已移植。** Python
   在 `d05f008`(2026-08-05)把滑窗从「固定 halfWidth 窗内像素质心」改成
   「逐行带测量白线游程、取游程中点」，起因是实测 640 宽真车帧上胶带线宽
   130~161px，而当时 `WindowHalfWidth=20` 只有 40px 搜索框，只能看到线的
   一小片，质心逐帧在两条边之间跳——**这就是真车转圈/摆动事故的直接成因**。
   MATLAB 侧一直停在旧算法 + `WindowHalfWidth=20`，也就是说在 2026-08-08
   之前，`visual_line_follower_with_debug_real.slx` 拿去跑真机会重现那些
   事故。现已在 `originbot_sliding_window_path_generator.m` 里补上
   `findNearestRun()` 并改写 `slideWindows()`，`WindowHalfWidth` 同步改为
   gazebo/mujoco 30、real 180(与 Python 侧同名预设一致)。**该移植经
   MATLAB-vs-Python 逐点交叉验证**(6 组合成掩膜 × 30 窗，见提交说明)。

4. **MATLAB 与 Python 的真车参数一度各存一份并已漂移。** 2026-08-08 盘点：
   `MinBrightness` 70 vs 170、`MaxSaturation` 0.30 vs 0.20、`ROIFraction`
   0.10 vs 0.3，`Kp` 也不一致。新增
   `runtime/init/load_real_params_from_yaml.m`，直接从
   `config/real/real_line_follower.yaml`(真车运行时实际读的那份)读参数，
   把它定为唯一真相源。**尚未接进 InitFcn**——
   `configure_visual_line_follower_real_debug.m` 里那张硬编码 `defaults`
   表还在用，接线需要在打开模型的情况下做。

`visual_line_follower_sac_residual_real.slx` 额外的两条坑(详见其 InitFcn
指向的 `runtime/init/configure_visual_line_follower_sac_residual_real_debug.m`):

3. 模型里的 `SAC_Agent`(`rllib/RL Agent` 块)读的是工作区变量
   `trainedAgent`——模型本身和 InitFcn 都不会赋值,运行前必须自己在 base
   workspace 里手动加载一个训练好的 agent 并赋给这个变量名(参考
   `train/sac/train_sac_residual_live.m`),不然仿真一开始就会报错。
4. 项目里现有的所有训练好的 agent(`train/sac/` 自己的产物,以及
   `launch/gazebo/gazebo_line_follower.launch.py`(2026-08-08 已随 Gazebo 线删除)里
   `DEFAULT_RESIDUAL_MODEL_PATH` 指向的 hpc/0720-0724 sweep checkpoint)
   全部是纯仿真里训练出来的,没有一个在真机上跑过、更没验证过
   sim-to-real 是否可信。真要往这个模型里塞真车用的 `trainedAgent`,先按
   Python 那边已经在遵守的规矩来:先跑纯 PID(`visual_line_follower_with_
   debug_real.slx`)确认没问题,残差策略再往后放。

## 目录说明

| 目录 | 用途 | 是否需要手动运行 |
| --- | --- | --- |
| `runtime` | 模型 InitFcn、地图解析、视觉检测函数 | 否，模型自动调用 |
| `experiments` | 仿真、PID 扫描、保存实验数据 | 按实验需要运行 |
| `train` | RL 训练（原 `scripts/`），2026-07-21 起分子目录：`train/sac`（SAC 入口与参数中心）、`train/shared`（`rl_*` 共享函数、`plot_rl_training`）、`train/logs`（训练日志）；PPO 线已归档至 `evaluation/archive_ppo/` | 训练时运行 |
| `model_building` | 重建 `.slx` 或生成地图资源的开发脚本 | 否，除非要重建资源 |

仿真数据会保存到 `simulation_data`。
