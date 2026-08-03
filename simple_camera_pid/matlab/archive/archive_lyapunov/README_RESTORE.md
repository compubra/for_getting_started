# 李雅普诺夫（Lyapunov）SAC 训练线 — 归档

归档日期：2026-07-09。此目录不在 MATLAB 搜索路径上，不会被自动加载。
这是**归档不是删除**——所有文件完整保留，按下表放回原位即可恢复。

## 归档内容（21 个文件，约 373 MB）

| 子目录 | 数量 | 原始位置（放回这里即恢复） |
| --- | --- | --- |
| `models/` | 6 | matlab 根目录 `src/simple_camera_pid/matlab/` |
| `code/train_sac_lyapunov_agent.m` | 1 | `experiments/` |
| `code/build_sac_lyapunov_controller.m` | 1 | `model_building/` |
| `code/run_lyapunov_1500.m`, `code/train_sac_lyapunov_live.m` | 2 | `train/`（原 `scripts/`，2026-07-18 改名；放顶层，不进 `train/sac/`——见下方接口说明） |
| `agents/` | 7 | `simulation_data/sac_training/` |
| `logs/` | 3 | `simulation_data/sac_training/` |
| `HANDOVER_RL_LYAPUNOV.md` | 1 | matlab 根目录 |

## 对接口的影响（重要）

- ~~`run_sac_training()` 的默认 residual 路径完全不受影响，照常训练。~~
- **⚠️ 2026-07-11 更新**：`run_sac_training.m` / `train_sac_residual_agent.m`
  已从主工作区移除，`sac_training_config.m` 改为服务 live 脚本入口
  `train_sac_residual_live.m`（`cfg.LyapunovReward` 字段组也一并移除）。
  恢复本归档的 lyapunov 训练时，直接运行 `code/train_sac_lyapunov_agent.m`
  （自带 config，不依赖已删除的入口），或参照新架构自行接一个
  `lyapunov_training_config.m`。
- **⚠️ 2026-07-21 更新**：`train/` 进一步拆分为 `sac/`（SAC 专属入口与参数
  中心）/ `shared/`（`rl_io_specs` / `rl_dr_defaults` / `rl_domain_randomization`
  / `plot_rl_training` 等两算法共用函数）/ `logs/`。`code/train_sac_lyapunov_agent.m`
  已改为 `addpath(fullfile(modelDir, "train", "shared"))`；本归档的
  `code/run_lyapunov_1500.m` 与 `code/train_sac_lyapunov_live.m` 恢复时放
  `train/` 顶层（不是 `train/sac/`——它们按内部相对路径逻辑假设自己在
  matlab 根目录下一层），也已同步改用 `train/shared`。

## 未归档：编译缓存

`lf_cache/` 下若干 `visual_line_follower_sac_lyapunov*` 的 `.slxc` / `slprj`
是可再生的编译缓存，与 residual 模型共用 `slprj` 目录，故未移动。可安全忽略，
Simulink 需要时会自行重建。
