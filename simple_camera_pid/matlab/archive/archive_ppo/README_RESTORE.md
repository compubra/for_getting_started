# PPO 残差训练线 — 归档

归档日期：2026-07-19。此目录不在 MATLAB 搜索路径上，不会被自动加载。
这是**归档不是删除**——所有文件完整保留，按下表放回原位即可恢复。
归档原因：残差 RL 以 SAC 为主线推进，PPO 线暂停维护。

## 归档内容（8 个文件）

| 子目录 | 文件 | 原始位置（放回这里即恢复） |
| --- | --- | --- |
| `models/` | `visual_line_follower_ppo_residual.slx` | matlab 根目录 |
| `models/` | `visual_line_follower_ppo_residual.slx.original`（2026-07-09 旧备份） | matlab 根目录 |
| `code/` | `ppo_training_config.m`（调参接口/参数中心） | `train/` |
| `code/` | `train_ppo_residual_live.m`（训练入口 live 脚本） | `train/` |
| `data/` | `train_ppo_800x200_*.log` ×4（2026-07-10 训练日志，无已存 agent） | `simulation_data/ppo_training/` |

## 归档时模型状态（重要）

`visual_line_follower_ppo_residual.slx` 归档前刚完成 2026-07-19 的主线同步：

- 视觉已切换到**方案 B**（`originbot_sliding_window_path_generator`，霍夫+滑窗+多项式拟合）
- `LocalPath_LookaheadDistance = 0.20`（真车速度实测最优）
- 观测 7 维（含 v_norm/omega_norm 速度反馈），与 `train/shared/rl_io_specs.m` 一致
- ⚠️ 速度参数仍是旧仿真量级 `MaxWheelSpeed=25 / BaseLinearSpeed=20`，
  未随主线降到 7.9/5——恢复训练前需自行决定是否同步降速
- `.original` 是 2026-07-09 的更早备份（5 维观测时代），仅考古用

## 恢复方法

1. `models/visual_line_follower_ppo_residual.slx` → matlab 根目录
2. `code/` 两个 .m → `train/`（放在 `train/` 顶层，不要放进 `train/sac/`——
   两个文件内部按「比 matlab 根目录低一层」计算路径，与 SAC 那对 live
   脚本/config 放进 `train/sac/` 的两层嵌套约定不同）
3. 从 `train/` 目录运行 `train_ppo_residual_live.m` 即可训练。
   全部依赖（`rl_io_specs` / `rl_dr_defaults` / `rl_domain_randomization` /
   `plot_rl_training`，现位于 `train/shared/`；runtime 函数、方案 B 视觉
   文件）都还在主工作区，无需任何代码改动。`simulation_data/ppo_training/`
   输出目录训练时自动重建。

## 对接口的影响

- SAC 训练线完全不受影响（`train/sac/sac_training_config.m` +
  `train/sac/train_sac_residual_live.m` 照常；2026-07-21 起 `train/` 已按
  `sac/` / `shared/` / `logs/` 分子目录整理）。
- `train/shared/plot_rl_training.m` 无参调用会自动扫 `sac_training/` 与
  `ppo_training/` 两个目录，`ppo_training/` 不存在时安全跳过（`dir()` 返回空）。
- `train/shared/rl_io_specs.m` 等共享函数的注释仍提及 SAC/PPO 双算法契约，
  保留不改——恢复 PPO 时契约仍然成立。

## 未归档：编译缓存

`lf_cache/` / `slprj/` 下 `visual_line_follower_ppo_residual` 的编译缓存是
可再生的，未移动，可安全忽略或清理。
