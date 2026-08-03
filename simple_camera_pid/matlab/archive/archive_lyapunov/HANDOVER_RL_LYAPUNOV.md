# RL / Lyapunov-SAC 交接文档（供 HPC 上的 Claude Code 读取）

> 生成日期：2026-07-04
> 作用范围：`src/simple_camera_pid/matlab/` 下的残差强化学习 + Lyapunov-SAC 部分。
> 本文件是**某次代码走查会话的结论固化**，供在另一台机器（HPC）上接手的 Claude Code
> 快速获得上下文，**不必重新逐文件通读**即可开始工作。文中每条结论都已通过读源码 /
> 用 MATLAB MCP 读取 .slx 实际接线核实过；凡是“未验证”的点都已显式标注。

---

## 0. 一句话概览

**残差式 SAC 巡线控制**：保留已有 PID 主控，SAC 只学一个角速度残差 `delta_omega` 叠加在
PID 之上；有两个奖励版本——普通二次型（residual）和**李雅普诺夫奖励**（lyapunov）。
算法与网络**全部用 MATLAB RL Toolbox 默认**，未自定义；**原创性集中在奖励设计和残差架构**。

---

## 1. 强化学习框架

### 1.1 范式：Residual RL（残差强化学习）
RL 不替代控制器，叠加在 PID 之上：

```
left_wheel_total  = pid_left  + delta_omega * YawToWheel
right_wheel_total = pid_right - delta_omega * YawToWheel
YawToWheel = TB3_WheelSeparation / (2*TB3_WheelRadius)
```

模型里对应两个并联子系统 `PID_Controller` 和 `SAC_Residual_Controller`，
最后 `Sum` 相加、`SAC_PID_Sat` 限幅到 `±MaxWheelSpeed`，再送 MuJoCo Plant。

### 1.2 算法：SAC（Soft Actor-Critic）
- MATLAB `rlSACAgent` + `rlSACAgentOptions`（off-policy、连续动作、熵正则）
- 双 Q critic、目标网络软更新 tau=5e-3、经验回放 buffer=2e5、minibatch=256
- gamma=0.99、EntropyWeight=0.2、actor/critic LR=3e-4

### 1.3 网络（重要：无自定义）
代码是 `rlSACAgent(obsInfo, actInfo, agentOpts)`，**让工具箱自动生成默认网络**：
标准高斯 actor（MLP）+ 双 Q critic（MLP）。**层数/宽度/激活均未手写、未控制。**
> ⚠️ 未验证：默认 MLP 具体几层多宽。若论文/报告需要写“网络结构”，需要显式实例化
> actor/critic 网络，或用 `getActor(agent)` / `getModel(...)` 把生成的网络结构 dump 出来读。

### 1.4 输入 / 输出（定义于 `train/shared/rl_io_specs.m`（原 `runtime/`，2026-07-21 起在 `train/shared/`），是唯一真相源）
观测（5 维）：
```
[ steering_error;    ±1.5   视觉巡线误差
  lateral_error;     ±1.0   横向偏差
  heading_error;     ±1.0   航向偏差
  found;             0/1    是否检测到白线
  prev_rl_action ]   ±1.0   上一步残差（动作记忆，来自 UnitDelay）
```
动作（1 维）：`delta_omega ∈ [-MaxDeltaOmega, +MaxDeltaOmega] = [-1, 1] rad/s`

---

## 2. 训练相关文件清单与作用

调用链：`sac_training_config.m` → `run_sac_training.m` → `train_sac_residual_agent.m`
（Lyapunov 版是 `train_sac_lyapunov_agent.m`，自带 config，不走 run_sac_training）

> **⚠️ 2026-07-11 架构更新**：主工作区已移除 `run_sac_training.m` 与
> `train_sac_residual_agent.m`；SAC/PPO 训练统一改为「参数中心 + live 脚本」：
> `train/sac/sac_training_config.m` + `train/sac/train_sac_residual_live.m`（原 `scripts/`，2026-07-18 改名为 `train/`，2026-07-21 再拆分出 `sac/`/`shared/`/`logs/` 子目录；PPO 对应
> `ppo_training_config.m` + `train_ppo_residual_live.m`）。下表描述的是归档时点
> （2026-07-04）的旧架构，恢复本归档代码时请以新入口为准对接。

| 文件 | 作用 | 备注 |
|---|---|---|
| `experiments/sac_training_config.m` | **参数中心**，唯一要手改的文件。5 组：训练流程/地图/奖励权重/SAC 超参/输出 | 只服务 residual 版 |
| `experiments/run_sac_training.m` | 一键入口。读 config + 应用 `"Reward.Q",2.0` 点号覆盖 + printConfig + 展开成 21 个 name-value 转发 | 不建 agent、不校验 |
| `experiments/train_sac_residual_agent.m` | **residual 底层训练**：断言→设地图→写奖励权重进模型工作区→建 SAC→rlSimulinkEnv→train→存 agent | 有 AverageReward 早停 |
| `experiments/train_sac_lyapunov_agent.m` | **lyapunov 底层训练**（residual 的孪生版），见下节差异 | **HPC 上大概率跑这个** |
| `train/shared/rl_io_specs.m`（原 `runtime/`） | 观测/动作规格单一真相源，build 和 train 共用 | |
| `runtime/set_turtlebot3_mujoco_scene.m` | 选 MuJoCo 场景（simple/ellipse/complex/training 或 XML 路径），默认只在内存生效不写 .slx | |
| `model_building/build_sac_residual_controller.m` | 一次性造模型脚本 | ⚠️ **别乱跑**：会用硬编码权重回退模型、破坏调参接口 |

### 2.1 Lyapunov 训练脚本相对 residual 的 4 处差异
（`train_sac_lyapunov_agent.m`，目标模型 `visual_line_follower_sac_lyapunov`）
1. **奖励不同**：写入 8 个 Lyapunov 权重 `SAC_Q_e/Q_l/Q_h/Lambda/R/P_Lost/K_inv/Eps`
   （+`SAC_Done_Steps/MaxDeltaOmega`），用 `assignIfPresent` 只在变量已存在时写。
2. **无奖励早停**：`StopTrainingCriteria="none"`，永远跑满 MaxEpisodes（默认 800）。
   `SaveReward` 只是 checkpoint 存档阈值，不是停止条件。
3. **域随机化**：`env.ResetFcn = rl_domain_randomization(mdl, rl_dr_defaults())`，
   6 组（GenTrack/Map/Pose/Dynamics/Control/Perception）。开换图/生成新赛道时自动关 Fast Restart。
4. **训练/加载二合一 + MuJoCo 崩溃恢复**：
   - `DoTrain=false` → 跳过训练，从 SaveDir 载入最新 `sac_lyapunov_agent_*.mat`，塞进块直接 `sim()`。
   - `RecoverMujoco=true` → 训练前 `recover_mujoco_host(mdl)` 清理上次崩溃残留的 MuJoCo MEX 宿主。

**HPC 上典型用法：**
```matlab
cd src/simple_camera_pid/matlab
addpath("experiments"); addpath("runtime");
% 训练：
[agent, info] = train_sac_lyapunov_agent("MaxEpisodes", 800, "UseFast", true);
% 只评估最新 agent（不训练）：
agent = train_sac_lyapunov_agent("DoTrain", false);
sim("visual_line_follower_sac_lyapunov");
```

---

## 3. 李雅普诺夫奖励（已逐块核对，实现 == 文件头公式）

模型里 `Reward_Calculation` 子系统 = 4 项相加（`Reward_Sum`）：`V_pen + a_cost + l_pen + bonus`。

完整公式：
```
V      = Q_e*e_steer^2 + Q_l*e_lat^2 + Q_h*e_head^2          (Lyapunov_V,  正定候选 V)
Vdot   = (V[k] - V[k-1]) / Ts_Control                        (Lyapunov_Rate)
reward = -(V + Lambda*Vdot_gated)                            (块2 V_pen)
         - R*u^2                                             (块3 Action_Cost)
         - P_Lost*(1 - found)                                (块4 Lost_Penalty)
         + K_inv / (|e_steer|+|e_lat|+|e_head| + Eps)        (块5 Inv_Bonus，唯一正项)
```

逐块核对结论（用 MCP 读实际接线，全部对上）：
- **块1 Lyapunov_V**：三个误差都进 V（不只 steering_error），信息利用比 residual 版充分。✅
- **块2 Lyapunov_Rate**：`-(V+Λ·Vdot)`。**实现比文件头多一个 first-step `gate`**（初值0、之后恒1的
  UnitDelay），作用是第 0 步把虚假的 Vdot 跳变屏蔽掉。这是正确的边界处理，文档没写。✅
- **块3 Action_Cost**：`-R*u^2`。✅
- **块4 Lost_Penalty**：先 `CompareToConstant(found>0.5)` 二值化再算 `-P_Lost*(1-found)`，比 residual 版稳健。✅
- **块5 Inv_Bonus**：`K_inv/(Σ|e|+Eps)`，误差越小越大，抵消“奖励恒为负→早死省分”。✅

9 个工作区变量（8 权重 + Ts_Control）与训练脚本 `assignIfPresent` 写入的完全对应，调参链闭合。

### 3.1 已知潜在问题（供后续排查，非 bug）
- **Vdot 未滤波**：`Vdot=ΔV/Ts` 直接用相邻两步 V 差分，V 依赖视觉误差 → Vdot 放大视觉抖动，
  Λ 项可能引入噪声。若训练 reward 方差大，先查这里。
- **量纲不齐**：e_steer(±1.5) 与 u(±1) 量纲不同，Q 系列与 R 无归一化依据。
- **残差惩罚但不奖励“相对 PID 的改善”**：R*u^2 惩罚动作，可能诱导 SAC 学 u≈0 的“懒惰解”。
  Inv_Bonus 部分缓解了这一点，但没有直接度量“有残差 vs 无残差”的优势。

### 3.2 这是不是李雅普诺夫函数？
V 本身是**合格的正定候选 V**，Λ·Vdot 项把“V̇<0”的下降倾向**编进了奖励**。但这是
**reward-shaping 式 Lyapunov RL，不是形式化稳定性证明**——V̇<0 只是被鼓励、不是硬约束，
没有 CLF 安全过滤层或带约束投影。论文措辞需注意，别声称“保证稳定”。

---

## 4. .slx 文件重复情况（重要，影响维护）

目录下 8 个 .slx 是**同一模型反复复制改**出来的家族：

| 文件 | 关系 |
|---|---|
| `visual_line_follower_with_debug.slx` | baseline（PID，无 RL）|
| `visual_line_follower_sac_residual.slx` | baseline + SAC 残差（二次型奖励）|
| `visual_line_follower_sac_lyapunov.slx` | residual + Lyapunov 奖励（**主力**）|
| `*_with_debug_gazebo.slx` / `*_sac_residual_gazebo.slx` / `*_sac_lyapunov_gazebo.slx` | 各自的 Gazebo Plant 变体 |
| `*_sac_lyapunov_backup_20260703_115027.slx` | ⚠️ 手工备份，git 噪声，建议删 |
| `*_sac_lyapunov_backup_native_20260703_120617.slx` | ⚠️ 手工备份，git 噪声，建议删 |

**重复层级：**
- 文件之间：🔴 高。MuJoCo↔Gazebo 双胞胎（仅 Plant 不同，其余 ~90% 相同）+ 2 个手工 backup。
- 跨模型子系统：🔴 高。residual 与 lyapunov **95%+ 相同**——视觉链、整个 PID_Controller、
  SAC 外壳（ObsMux/Prev_RL_Action/差速 Gain/FinalMux/RateTransition/Done_Detection）**逐块一致**，
  **唯一实质差异就是 `Reward_Calculation` 子系统内部**（residual=3 个 Gain；lyapunov=5 个子块）。
- 单文件内部：🟢 低。左右轮对称是差速必要对称，不算冗余。

**建议（低风险先做）：**删两个 `_backup_` .slx（git 已管版本）。
**结构性（大手术，仅长期迭代才值得）：**用 Model Reference / Library 把共享的视觉链、PID、
SAC 外壳做成引用块，让各版只维护自己不同的奖励/Plant，避免改一处 PID 要同步 6 个文件。

---

## 5. 自定义成分评估

| 维度 | 自定义程度 |
|---|---|
| SAC 算法本身 | ❌ 零（调 RL Toolbox，未改一行）|
| 神经网络结构 | ❌ 零（工具箱自动生成默认 MLP，未手写层）|
| 训练循环 | ❌ 零（一句 `train()`）|
| 奖励函数 | ✅ 高（Lyapunov 奖励原创）|
| 环境/残差控制架构 | ✅ 高（PID+SAC 残差 + MuJoCo 闭环）|
| 训练编排脚本 | 🔶 中（DoTrain/崩溃恢复/写权重等工程脚手架，含大量 residual↔lyapunov 复制粘贴）|

**定位：重奖励设计 + 重系统集成、轻算法。** 学术贡献点应落在
“Lyapunov-guided reward + residual control”，而非“改进 SAC”。

---

## 6. 给 HPC Claude Code 的接手提示

1. **先读本文件，再按需读源码**；具体训练入口见第 2 节。HPC 上大概率跑 lyapunov 版。
2. **MuJoCo 依赖**：需要 `Simulink Blockset for MuJoCo Simulator`（脚本会断言 `mj_initbus` 存在）。
   若无桌面 MATLAB，注意 MuJoCo 渲染——用 `"UseFast", true` 关渲染跑 headless。
3. **崩溃恢复已内置**：训练脚本默认 `RecoverMujoco=true`，会自动清理上次崩溃的 MEX 宿主。
4. **别跑 `build_sac_residual_controller.m`**（会回退模型）。
5. 已知待办优先级：①确认默认网络结构（论文需要）②Vdot 滤波 ③抽公共训练函数消除 residual↔lyapunov 复制。
6. **本文件是快照**：凡引用到具体块名/变量/文件名，动手前用 MCP（`model_read`）或 Read 复核一遍是否仍然成立。
