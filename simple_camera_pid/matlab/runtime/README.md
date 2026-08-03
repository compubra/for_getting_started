# Runtime support

这些文件支持主模型运行，不是实验脚本。2026-07-22 起按功能分了 4 个子目录；
外部脚本仍只需 `addpath(genpath(fullfile(matlabDir, "runtime")))` 一次性
递归引入全部（简单 `addpath` 只加当前目录、不会带上子目录，务必用 genpath）。

## init/ — 模型初始化（InitFcn，模型打开/仿真开始时自动调用）

| 文件 | 作用 |
| --- | --- |
| `configure_turtlebot3_visual_line_follower_paths.m` | MuJoCo 模型用，设地图路径、缓存目录、清 persistent 状态；内部用 `genpath` 把整个 `runtime/` 树加入路径 |
| `configure_visual_line_follower_gazebo_debug.m` | Gazebo 模型用，无 MuJoCo Plant 需重锚，只做参数导出+清理；同样 `genpath` 整个 `runtime/` |
| `configure_visual_line_follower_real_debug.m` | 真机调试模型（`visual_line_follower_with_debug_real.slx`）用，逻辑跟 Gazebo 版一样（参数导出+清理），2026-08-01 由后者复制而来；文件头注释里记着两个未解决的坑（`originbot_camera_profile.m` 平台识别没有真机分支、`CmdVel_Publish` 消息类型疑似应为 `TwistStamped`），用前必读 |
| `configure_visual_line_follower_sac_residual_real_debug.m` | 真机 + RL 残差调试模型（`visual_line_follower_sac_residual_real.slx`）用，2026-08-01 由 `_gazebo` 版复制而来；除了上面两个坑，还多两条（`trainedAgent` 变量没人赋值、现有 agent 全是纯仿真训练没验证过 sim-to-real），用前必读 |

## vision/ — 视觉算法（参与控制，MuJoCo/Gazebo 两平台共用一份）

| 文件 | 作用 |
| --- | --- |
| `originbot_sliding_window_path_generator.m` | 循线主算法（霍夫种子+滑动窗口+地面二次多项式拟合），**参与控制**，按输入图像尺寸自动识别平台 |
| `originbot_camera_profile.m` | 两平台相机参数唯一真相源 |
| `originbot_pixel_to_ground.m` | 像素→地面坐标投影（IPM） |
| `originbot_ground_to_pixel.m` | 反投影，仅调试帧画拟合曲线用 |
| `originbot_line_follower_debug_frame.m` | 调试画面生成（Camera Monitor 用），**不参与控制** |

## scene/ — MuJoCo 场景/地图

| 文件 | 作用 |
| --- | --- |
| `resolve_turtlebot3_mujoco_scene.m` | 地图 key/路径解析（8 级优先级） |
| `set_turtlebot3_mujoco_scene.m` | 供训练/实验脚本调用的设图入口 |
| `gen_simple_track_scene.m` | 闭环赛道生成器 A：椭圆/胶囊，规整几何体 |
| `gen_complex_track_scene.m` | 闭环赛道生成器 B：折线+每个顶点独立随机圆弧倒角，带尖角的多边形观感 |
| `gen_wandering_track_scene.m` | 闭环赛道生成器 C：随机控制点+周期样条，处处光滑的有机蜿蜒闭环 |
| `gen_random_local_path.m` | **孤儿工具**，纯 MATLAB 合成开放曲线（不闭合、不接触 MuJoCo），只用于离线测试视觉算法的拟合数学；`gen_wandering_track_scene.m` 借用了它"随机控制点+样条平滑"的思路改造成闭环生成器，但本文件本身未被改动、仍无调用方 |
| `get_track_centerline.m` | 场景 XML→中心线坐标，离线真值评估工具，不接入训练/控制 |

三个 `gen_*_track_scene.m` 闭环生成器都由 `train/shared/rl_domain_randomization.m`
的 `GenTrack` 域随机化调用（`dr.GenTrack.Generator = "simple"|"complex"|
"wandering"|"random"`，默认 `"random"` 即每回合从三者里随机挑一个），接口/
输出 `info` 结构基本一致，可互相替换。

## ops/ — 运维工具

| 文件 | 作用 |
| --- | --- |
| `recover_mujoco_host.m` | 清理上次崩溃/中断残留的 MuJoCo MEX host 进程 |

---

2026-07-22：MuJoCo 相机已从 15° 下俯拍平为水平安装，与 Gazebo 对齐（见
`model/mujoco/turtlebot3/turtlebot3_burger_vehicle_body.xml`）。旧的方案 A
算法（骨架+纯追踪）以及方案 B 曾经独立的 `_gazebo` 后缀文件均已归档，见
[`archive/archive_vision_scheme_a/README_RESTORE.md`](../archive/archive_vision_scheme_a/README_RESTORE.md)。

⚠️ 拍平后 MuJoCo 侧的 `LocalPath_ROIFraction`/`LocalPath_LookaheadDistance`
默认值改为 0.30/0.40（几何推算的合理起点，尚未跑真值调参 sweep）——细节见
`vision/originbot_camera_profile.m` 内注释与上述归档 README。

⚠️ `matlab/add_local_path_to_visual_model.m`（matlab 根目录下的旧一次性
建模脚本）仍引用已归档的方案 A 函数 `originbot_local_path_generator` 和
更老的 `OriginBot_ROI_Centroid` 块名，本次目录重组只修了它的 addpath、
**没有**修复这处更早就存在的过时引用——直接运行会报函数未找到，需要时
请先确认是否还有人在用这个脚本。
