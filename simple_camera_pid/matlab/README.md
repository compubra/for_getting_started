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

## 安全约束层与丢线找线（2026-08-09 新增）

`visual_line_follower_sac_residual.slx` 与 `visual_line_follower_sac_residual_real.slx`
里新增了 `Line_Search` + `Safety_Filter` 两个子系统，接在 PID + SAC 残差求和之后、下发给
执行器之前，做两件事：

1. `Safety_Filter` — **约束**：离散控制屏障函数（CBF）对前视点横向偏移做前向
   不变性约束，叠加箱式与速率约束。属于推理期安全层，**不需要重训现有 SAC
   检查点**。默认开启，实测是净收益。
2. `Line_Search` — **丢线找线**：`TRACK→HOLD→RECOVER→BRAKE→SCAN→GIVEUP`
   状态机，主策略是基于几何记忆的定向恢复，记忆失效才退回原地逐级扩大扫描。
   **默认关闭**（`Safety_EnableSearch=0`）——实测两种恢复策略都不如"什么都
   不做"，瓶颈在观测而非恢复策略，详见 `runtime/control/README.md`。

两级刻意拆成两个子系统：前者决定"该往哪走"，后者决定"允不允许这么走"，
安全层对指令来源不可知是设计前提。

方法依据：Gu et al., *A Review of Safe Reinforcement Learning: Methods,
Theories, and Applications*, IEEE TPAMI 46(12):11216-11235, 2024，§II-A
（瞬时约束 vs 累积约束）、§III-A-2 / §III-B-2（CBF 与 safety layer / OptLayer）。

设计说明、参数表、单位约定（两个模型不同，传错静默失效）、以及**已验证/未验证
清单**全部在 `runtime/control/README.md`。调参改模型工作区里的 `Safety_*`
变量即可。两个消融开关 `Safety_EnableFilter` / `Safety_EnableSearch` 关掉时
只统计不干预，用于做对比基线。

验证：`verify_safety_governor`（纯函数级，不需要 `.slx`/MuJoCo，20/20 通过）
＋ MuJoCo 闭环消融对比。`_real` 变体只做了结构编辑、**一次都没跑过**，且继承
下面列的原有阻塞项。

> 本机 MuJoCo 相机渲染需要绕过一个环境问题（MATLAB 自带 libstdc++ 比系统
> Mesa 需要的旧，C++ ABI 不匹配 → `std::bad_alloc` / rendering error 2）。
> 从命令行这样起 MATLAB 即可：
> `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 matlab`
> 根因分析见 `runtime/control/README.md` 末节。

## 视觉侧 Python→MATLAB 移植与新增（2026-08-09）

MATLAB 视觉一度落后 Python 侧若干项鲁棒性改进，本轮补齐，并新增一项 MATLAB
独有的自适应搜索半径。全部集中在
`runtime/vision/originbot_sliding_window_path_generator.m`，验证脚本
`experiments/verify_vision_ports.m`（合成帧，不需要 `.slx`/MuJoCo，7/7 通过）。

| 改动 | 来源 | 说明 |
| --- | --- | --- |
| 自适应 ROI 回退 | 移植 Python `roi_widen_step`/`roi_widen_max` | 窄 ROI 检测失败时加宽 0.2（上限 0.7）重试一次 |
| 霍夫种子时间连续性 | 移植 Python `prev_base_col` | 80% 长度闸门内，取外推后最接近上帧种子列的段 |
| 丢线转向趋势外推 | 移植 Python `_estimate_steering_slope` | 由平冻结改为按最近 5 帧斜率线性外推 |
| **自适应搜索半径** | **本轮新增** | 见下 |

新增两个可选尾参（13/14/15），**默认即启用**，所以四个 `.slx` 的 MATLABFcn
表达式不用改就能拿到修复；传 0 可逐项关闭做消融。

### 自适应搜索半径

`WindowHalfWidth` 自 2026-08-05 改算法后已不是"质心计算窗"，只是"别跳到远处
无关亮区"的闸门，但仍是**手调**的每平台常数（mujoco 30 / real 180，差 6 倍，
差异全来自线在像素上有多宽）。既然新算法每行都会实测白线游程宽度，这个常数就
可以直接推出来：

```
searchRadius = clamp(1.5 × 实测线宽 + MaxDriftPerRow × 窗高,  8,  0.30×图像宽)
```

用**上一个成功窗**的实测宽度而非全帧平均，于是随透视自然收缩（远处的线更窄）。
每帧首窗用上一帧留下的宽度作种子，冷启动才退回 `WindowHalfWidth`。
`AdaptiveWindowGain = 0` 关闭。

实测（合成帧，121 px 宽线偏心放置，与 IPM 反算真值比）：固定半径 30 把横向
偏差**低估 26.5%**（闸门装不下线，游程被非对称截断、中点被拉向 currentCol），
自适应半径误差 **3.5%**。

### 参数定稿：ROI 保持窄，前视改小

同批把 `LocalPath_LookaheadDistance` 由 0.40 改为 **0.20**（0.40 超出 ROI 能
看到的 0.283 m，前视点一直靠外推），`LocalPath_ROIFraction` **保持 0.30**。

**不要**把 ROI 改成 profile 里的 `DefaultROI=0.50`——那只是缺参兜底。实测对比
（simple 图，150 s，只开约束层）：

| 配置 | found | 最长丢线 | 真值横向 RMS | 路径 | 角加速度 RMS | 越界 |
| --- | --- | --- | --- | --- | --- | --- |
| 改前 0.30/0.40 | 96.5% | 5.15 s | 4.82 cm | 14.96 m | 12.59 | 0% |
| 0.50/0.20 | **99.6%** | **0.30 s** | 12.47 cm | 14.56 m | 12.10 | **2.4%** |
| 0.50/0.40 | 84.9% | 17.15 s | 17.71 cm | 12.63 m | 12.70 | 3.0% |
| **定稿 0.30/0.20** | 94.6% | 8.05 s | **5.16 cm** | **15.10 m** | **5.62** | **0%** |

ROI 深到 0.50 确实几乎不再丢线，但会在急弯处捡到赛道**另一条邻近分支**
（simple 是 S 形、会自我折返），污染多项式拟合——出现一段持续 10.6 秒、最大
51 cm 的跑偏，屏障 h 最小 −0.36 真的越界了。这正是 Python 侧
`roi_widen_step` 文档早就写明的现象，也正是"窄 ROI 打底 + 失败才加宽"这个
设计存在的理由。

同批还把 `Recovery_Steering_Gain` 由 **10 改为 1.0**（四个模型），与 Python
侧 2026-08-07 的同一修复对齐；真机两个模型的 `MinBrightness`/`MaxSaturation`
由 70/0.30 改为 **170/0.20**，与 `config/real/real_line_follower.yaml` 对齐
——那对值是真机侧从 2026-08-04 转圈事故的 bag 里逐帧反推出来的，MATLAB 一直
停在会触发该事故的旧值上。

**注意**：自适应搜索半径目前是 MATLAB 独有，Python 侧还没有——本轮消除了四项
分歧，又新造了一项反向分歧。要保持两边同步，需要把它回移到
`common/vision.py`。

## 目录说明

| 目录 | 用途 | 是否需要手动运行 |
| --- | --- | --- |
| `runtime` | 模型 InitFcn、地图解析、视觉检测函数 | 否，模型自动调用 |
| `runtime/control` | 安全约束层（CBF + 安全层投影）与丢线找线状态机，见该目录 `README.md` | 否，模型自动调用 |
| `experiments` | 仿真、PID 扫描、保存实验数据 | 按实验需要运行 |
| `train` | RL 训练（原 `scripts/`），2026-07-21 起分子目录：`train/sac`（SAC 入口与参数中心）、`train/shared`（`rl_*` 共享函数、`plot_rl_training`）、`train/logs`（训练日志）；PPO 线已归档至 `evaluation/archive_ppo/` | 训练时运行 |
| `model_building` | 重建 `.slx` 或生成地图资源的开发脚本 | 否，除非要重建资源 |

仿真数据会保存到 `simulation_data`。
