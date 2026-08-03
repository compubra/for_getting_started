# Experiment scripts

这些脚本用于仿真、调参和保存实验数据，不负责建立当前主 SLX。

| 文件 | 作用 | 适用模型 |
| --- | --- | --- |
| `launch_turtlebot3_manual_run.m` | 启动一次可视化长时间实验 | 当前主模型 |
| `run_turtlebot3_burger_mujoco_visual_line_follower.m` | 运行一次实验并保存数据与摘要 | 当前主模型 |
| `tune_complex_track_pid.m` | 在复杂地图上扫描 PID 参数 | 当前主模型 |
| `tune_localpath_gains.m` | 扫描局部路径增益 | 当前主模型 |
| `run_turtlebot3_mujoco_pid.m` | 纯里程计 PID baseline 实验 | 需要 baseline SLX |

如需直接调用实验函数：

```matlab
addpath("experiments")
```

## 残差 RL 训练（SAC）→ 已移至 train/

RL 训练的入口和参数中心都在 `../train/sac/`（`train/` 原名 `scripts/`，
2026-07-18 改名；2026-07-21 起按 `sac/`/`shared/`/`logs/` 分子目录），
本目录不再包含训练代码（旧的 `sac_training_config` → `run_sac_training` →
`train_sac_residual_agent` 函数链已移除，live 脚本是唯一入口）：

| 算法 | 参数中心（先改这里） | 训练入口（然后运行这个） |
| --- | --- | --- |
| SAC | `train/sac/sac_training_config.m` | `train/sac/train_sac_residual_live.m` |

PPO 训练线（模型 + 调参接口 + 训练入口 + 日志）已于 2026-07-19 整体归档至
`../evaluation/archive_ppo/`，恢复方法见其中的 `README_RESTORE.md`。

工作流：编辑参数中心里的字段（训练回合、地图、奖励权重、超参数、域随机化、
输出目录，带中英文注释）→ 在 MATLAB 中把当前目录切到 `train/sac/` → 运行对应
live 脚本。训练好的 agent 与统计存到 `simulation_data/<算法>_training/`，
学习曲线用 `train/shared/plot_rl_training.m` 画（无参调用自动找最新结果）；
两算法共用的 `rl_io_specs` / `rl_dr_defaults` / `rl_domain_randomization` 在
`train/shared/`。

扫参示例（在脚本或命令行里循环改 config 的做法已不适用，直接改文件或
临时在工作区覆盖变量后分次运行 live 脚本）。
