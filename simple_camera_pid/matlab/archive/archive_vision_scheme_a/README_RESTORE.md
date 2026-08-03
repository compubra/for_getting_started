# 视觉方案 A / 已取代的 Gazebo 独立分支 — 归档

归档日期：2026-07-22。此目录不在 MATLAB 搜索路径上，不会被自动加载。
这是**归档不是删除**——所有文件完整保留，按下表放回原位即可恢复。

归档原因：MuJoCo 与 Gazebo 的视觉算法/调试画面统一为**单文件、按输入图像
尺寸自动识别平台**的方案 B（不再有 `_gazebo` 后缀的姐妹文件），MuJoCo 相机
同时从 15° 下俯拍平为水平安装（与 Gazebo 一致）。方案 A（骨架+纯追踪）与
方案 B 的旧 gazebo 独立实现均被取代。

## 归档内容（5 个文件）

| 文件 | 原始位置 | 说明 |
| --- | --- | --- |
| `code/originbot_local_path_generator.m` | `runtime/` | 方案 A 算法（骨架化+贪心排序+纯追踪），MuJoCo 专用，15° 下俯相机 |
| `code/originbot_local_path_generator_gazebo.m` | `runtime/` | 方案 A 算法的 Gazebo 适配版（分带扫描+正向/180°择优），从未成为主线 |
| `code/originbot_sliding_window_path_generator_gazebo.m` | `runtime/` | 方案 B 算法的**旧 Gazebo 独立实现**（2026-07-19 引入），已被统一后的 `runtime/originbot_sliding_window_path_generator.m`（平台自动识别）取代 |
| `code/originbot_line_follower_debug_frame_gazebo.m` | `runtime/` | 调试画面的**旧 Gazebo 独立实现**，逻辑仍是方案 A 时代的分带扫描（未跟上方案 B），已被统一后的 `runtime/originbot_line_follower_debug_frame.m` 取代 |
| `code/originbot_line_follower_debug_frame_legacy_bandscan.m` | `runtime/`（原文件名 `originbot_line_follower_debug_frame.m`） | 调试画面的**最初实现**（分带扫描，无最大连通块筛选、无霍夫/滑窗/拟合）。⚠️ 本文件内容是从本次归档会话早前一次工具调用输出**逐字重建**的——`runtime/` 里的实际文件在本次归档动作之前已被本会话之外的另一次修改覆盖为方案 B 风格（单平台/15°硬编码，未再单独归档，其逻辑已完整包含在现行统一版本中）。恢复本文件需手动改回原文件名 `originbot_line_follower_debug_frame.m`。 |

## 归档时的关键状态（重要）

- **MuJoCo 相机已物理拍平**：`model/mujoco/turtlebot3/turtlebot3_burger_vehicle_body.xml`
  中 `turtlebot3_front_camera` 的 `xyaxes` 从 15° 下俯编码
  (`"0 -1 0 0.258819 0 0.965926"`) 改为水平 (`"0 -1 0 0 0 1"`)。恢复
  `originbot_local_path_generator.m`（方案 A，内部 `cameraParams().pitchDeg=15`
  硬编码）前，**必须先把该 XML 的 `xyaxes` 改回下俯版本**，否则方案 A 的
  IPM 几何与实际相机不匹配，转向会系统性算错。
- **统一后 MuJoCo 侧 ROI/Lookahead 默认值也变了**：`originbot_camera_profile.m`
  的 MuJoCo 分支现为 `DefaultROI=0.30` / `DefaultLookahead=0.40`（原方案 A/B
  统一前分别是 0.50/0.20 与 0.20），因为拍平后近端可视地面距离下限升到约
  0.31m，旧的 0.20m 前视已物理不可达。这两个新值是几何推算的合理起点，
  **尚未跑过真值调参 sweep**（对照 `vision-scheme-b-tuning` 记忆里 2026-07-19
  对方案 B 真值调参的方法），建议后续用同样方法重新调优。
- 两个 MuJoCo 模型（`visual_line_follower_with_debug.slx`、
  `visual_line_follower_sac_residual.slx`）与两个 Gazebo 模型
  （`_with_debug_gazebo`、`_sac_residual_gazebo`）的 `Camera_Local_Path_Generator`
  / 调试帧 MATLABFcn 表达式均已指向统一后的无后缀函数名；Gazebo 两模型的
  表达式从 `..._gazebo(...)` 改回了 `...(...)`（函数体经 `numel` 自动识别
  平台，调用方式不变）。

## 恢复方法

1. 把需要的文件从 `code/` 移回 `runtime/`（`_legacy_bandscan.m` 需先改名为
   `originbot_line_follower_debug_frame.m`，并会与当前统一版本重名——
   恢复前请先把 `runtime/` 里现行的版本挪开或另存）。
2. 若恢复方案 A 算法：按上文说明先把 MuJoCo 相机 XML 的 `xyaxes` 改回下俯版本。
3. 把对应模型的 `Camera_Local_Path_Generator` / 调试帧 MATLABFcn 表达式里的
   函数名改回目标文件的函数名，重新 `save_system`。
4. 若只是想临时对比方案 A/B 效果而非长期回退，无需改 XML/模型——可以在
   `runtime/` 里同时放一份改了名字的方案 A 副本，用独立函数名调用，不影响
   现行统一版本。

## 对接口的影响

- `runtime/originbot_camera_profile.m` / `originbot_pixel_to_ground.m` /
  `originbot_ground_to_pixel.m` 是新增的共享文件，是当前 MuJoCo/Gazebo 两
  平台相机内参/外参/像素尺度常数与 IPM 投影的**唯一真相源**。恢复本归档
  的任何文件都不会用到它们（各归档文件自带独立的 `cameraParams`/
  `pixelToGround` 硬编码副本），互不干扰。
- `runtime/configure_turtlebot3_visual_line_follower_paths.m`（MuJoCo InitFcn）
  与 `runtime/configure_visual_line_follower_gazebo_debug.m`（Gazebo InitFcn）
  的 `clear` 列表与 ROI/Lookahead 默认导出表已同步到统一后的函数名与数值；
  恢复方案 A 需要自行在 InitFcn 里加回对应的 `clear originbot_local_path_generator`
  等语句（不加也不会报错，只是上次仿真的 persistent 状态可能残留一帧）。
