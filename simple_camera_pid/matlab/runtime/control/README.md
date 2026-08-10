# runtime/control —— 安全约束层 + 丢线找线

2026-08-09 新增。按 Gu et al., *A Review of Safe Reinforcement Learning:
Methods, Theories, and Applications*, IEEE TPAMI 46(12):11216-11235, 2024
给 PID + SAC 残差控制链补两件事：

| 文件 | 作用 |
| --- | --- |
| `lf_line_search.m` | **行为生成**：丢线找线状态机 + 几何记忆。产生期望指令 |
| `lf_safety_filter.m` | **约束**：CBF + 箱式 + 速率约束的一次投影 |
| `lf_memory_propagate.m` / `lf_memory_target.m` | 几何记忆的死推与目标点选择（纯函数，无状态） |
| `lf_safety_defaults.m` | **所有默认值的唯一来源**，两个算法文件共用同一份参数结构体 |

`lf_line_search` / `lf_safety_filter` 有 `persistent` 状态，两个 InitFcn 里都有
对应的 `clear`。

验证脚本 `../../experiments/verify_safety_governor.m`（纯函数级，不需要
打开 `.slx`，也不需要 MuJoCo，20/20 通过）。

## 在信号链里的位置

```
视觉 ──┬─→ PID_Controller ──┐
       │                    ├─→ Σ ─→ [Line_Search] ─→ [Safety_Filter] ─→ 差速运动学 ─→ 轮速
       └─→ SAC 残差策略 ────┘          ↑                  ↑
                              lateral/heading/found、  lateral/heading/found
                              实测 v/omega、路径点
```

**两级都接在残差求和之后**，所以对 PID 和 RL 残差一视同仁——安全层看不到、也
不需要知道指令里哪部分是策略产生的。这正是论文 §III-B-2 里 safety layer /
OptLayer 的位置：把策略输出再过一层控制滤波器。

### 为什么拆成两级（2026-08-09）

原本两件事写在一个 `line_follower_safety_governor.m` 里。拆开的理由：

- **职责不同**：`Line_Search` 决定"该往哪走"（线还在就透传，线丢了就接管）；
  `Safety_Filter` 决定"允不允许这么走"。前者是行为生成，后者是约束投影。
- **"安全层对指令来源不可知"是设计前提**，混在一个函数里就看不出来了。
- **可独立测试/消融**：两个 `Enable*` 开关现在各管各的一级。

**两个残差模型**（`visual_line_follower_sac_residual.slx` 与 `_real.slx`）里
都对应两个子系统 `Line_Search` / `Safety_Filter`，各含一个 Interpreted MATLAB
Function 块；根部 `Safety_Diag_Mux` 把两级的诊断拼成一路 `safety_diag` 写工作区
（14 路，布局见 `experiments/run_safety_governor_trial.m` 的注释）。

> `_real.slx` 在 2026-08-09 整理布局时被发现**已经坏了**：它的 `Safety_Governor`
> 仍调用早先被删除的 `line_follower_safety_governor`，Mux 还是 5 路（加几何记忆
> 后需要 74 路），输出维度还停在 13。原因是加记忆功能那轮只改了 MuJoCo 模型，
> 而 `_real` 跑不了仿真（需要 ROS + 真机）所以没暴露。已一并补齐：视觉子系统加
> `local_path_debug` 输出端口、实测 v/omega 从 `Odometry_Interface` 分支引出、
> 两级子系统按 mps 单位约定接好。**该模型仍然一次都没跑过。**

拆分经 25 s 闭环仿真核对：轨迹、真值横向误差、found 率与拆分前**逐位相同**。

拆分顺带修掉一处：扫掠角 `phi` 原先按安全滤波**之后**的指令积分，现在改按
**实测**偏航角速度积分——既避开了"滤波→状态机"的代数环，也比任何指令值都准
（指令下游还有轮速饱和）。

代价函数式的方法（CPO/IPO/拉格朗日，论文 §III-B-1）要改训练目标、必须重训；
安全层不用，**现有 SAC 检查点直接可用**。

## 选了哪一类约束

论文 §II-A 把约束分成沿轨迹累积的 cumulative constraint（式 (1)-(3)）和每步
生效的 instantaneous constraint。这里实现的是后者里的**显式**一类——每步都有
闭式可校验的表达式。理由就是上面那条：累积约束绕不开重训。

具体两层：

### 1. 离散控制屏障函数（CBF）

对**前视点**横向偏移建屏障，而不是车体质心横向偏差。原因是 relative degree：

- 质心横向偏差对 `omega` 的 relative degree 是 2 → 要上 HOCBF 级联；
- 前视点偏移 `y_L` 满足 `d(y_L)/dt = v*sin(psi) + L*omega`，`omega` 直接出现在
  一阶导里，relative degree 是 1 → CBF 条件退化成**一条关于 omega 的仿射
  不等式**，闭式可解，不需要 QP 求解器。

而视觉模块输出的 `lateral_error` 恰好就是前视点偏移（归一化），不用额外估计。

屏障 `h(e) = LateralMax² − e²`，CBF 条件 `dh/dt ≥ −alpha*h` 整理成
`A*omega ≥ b`，`A = −e*L`，`b = −(alpha*lateralNorm/2)*h + e*v*sin(psi)`。
单条仿射约束 + 标量决策变量，投影就是一次 clamp。完整推导见
`lf_safety_filter.m` 文件头。

### 2. 箱式 + 速率约束

`v ∈ [0, MaxV]`（残差策略不得倒车）、`|omega| ≤ MaxOmega`、每拍变化量受
`MaxAccel*` 限制。与 CBF 边界求交；**交集为空时物理可行性优先**（退回硬集里
最接近 CBF 边界的点，并置 `infeasible=1` 供统计），否则会下发电机做不到的指令。

## 丢线找线状态机

`TRACK → HOLD → RECOVER → BRAKE → SCAN → GIVEUP`，任何状态下 `found=1` 立即回
TRACK。`RECOVER`（记忆有效时的定向恢复）是主策略，记忆失效/过期才退回后面
那条 刹停→原地扫描→放弃 链。

- `HOLD`（默认 0.4 s）不干预：此时视觉模块自己还在冻结/外推 `steering_error`，
  瞬时遮挡不该被打断。
- `BRAKE`（0.3 s）v、omega 一起线性归零，先停稳。
- `SCAN` **原地**扫描（v=0，绝不会因找线冲出赛道），扫掠角目标序列
  `+d*A₁, −d*A₁, +d*A₂, −d*A₂, …` 逐级扩大，默认 `A = [25°, 50°, 80°]`。
  方向 `d` 取最后一次看到线时 `−sign(lateral_error)`。
- `GIVEUP` 扫完或超时后停车等待。

顺带修掉了原 `PID_Controller` 里 `Minimum_Search_Speed`（bias 0.2）的一个实机
故障模式：丢线时 v 恒定保持 20% 一直向前爬，配合 `Recovery_Steering` 反复打满，
实机上表现为跨线来回摆动而不是受控停车。BRAKE 之后 v 被强制归零。

## 参数

全部在**模型工作区**里，名字以 `Safety_` 开头，模型第一次运行时由
`lf_safety_defaults.m` 自动写入默认值。调参改模型工作区即可，不用动代码。

两个消融开关：`Safety_EnableFilter` / `Safety_EnableSearch`。关掉时对应阶段
**不改指令，但诊断量照算照输出**——于是同一次运行里既能拿到基线行为，又能
统计"若开启安全层，它本会在何时、以多大幅度介入"。四种组合就是论文的消融表。

### 单位约定：两类模型不一样

`lf_safety_defaults(model, signalUnits)` 的第二参必须显式传，不做自动推断
（照搬 `originbot_camera_profile` 的做法）：

| 值 | 用于 | 线速度信号含义 |
| --- | --- | --- |
| `"scaled"` | `visual_line_follower_sac_residual.slx` | BaseLinearSpeed 刻度，× `BaseSpeedScale` 才是 m/s |
| `"mps"` | `visual_line_follower_sac_residual_real.slx` | 已经是 m/s（`/cmd_vel` 的 `linear_x`） |

**传错不会报错**，只会让 CBF 里 `v*sin(psi)` 那一项差约 33 倍
（`1/BaseSpeedScale`），屏障变得过松或过紧——静默失效，务必对号入座。

## 诊断输出

根部 `Safety_Diag_Mux` 把两级的诊断拼成一路 14 元素的 `safety_diag` 写工作区：

```
 1-2   VOmegaSafe   最终下发 (Safety_Filter 输出)
 3-4   VOmegaDes    安全滤波**之前**的期望值 (Line_Search 输出)
 5-9   SearchDiag   state  phi  memValid  memAge  memBearing
10-14  FilterDiag   h  omegaLo  omegaHi  cbfActive  infeasible
```

干预量 = 1:2 与 3:4 之差。全部导出是为了让离线分析能自行统计干预率/违约率，
不必改算法文件。

## 闭环测试结果（2026-08-09，地图 simple，150 s，四组消融）

复现：`run_safety_governor_sweep` 跑数据（**需要能渲染的进程**，见末节），
`compare_safety_governor("simple")` 出图和指标表。原始数据与图在
`simulation_data/safety_governor/`。

| 指标 | baseline | filter_only | search_only | both |
| --- | --- | --- | --- | --- |
| 存活时间 (s) | 150.00 | 150.00 | **64.65** | **64.85** |
| 路径长度 (m) | 15.05 | 14.96 | **4.77** | **4.76** |
| 完成整圈 | 是 | 是 | **否** | **否** |
| 真值横向误差 RMS (cm) | 4.76 | 4.82 | 4.62 | 4.81 |
| 视觉 found 率 (%) | 96.37 | 96.50 | 68.93 | 69.03 |
| 重新捕获次数 | 1 | 1 | **0** | **0** |
| 约束违反占比 (%) | 0.00 | 0.00 | 0.00 | 0.00 |
| 最小屏障 h | 0.5497 | 0.5530 | 0.5837 | 0.5864 |
| CBF 介入占比 (%) | 0.00 | 3.83 | 0.00 | 1.70 |
| 角加速度 RMS (rad/s²) | 24.29 | **12.59** | 20.23 | **11.25** |
| 轮速饱和占比 (%) | 0.23 | **0.00** | 0.31 | **0.00** |

### 结论 1：约束层达到设计目标，但这条赛道上约束本身没被需要

`filter_only` 与 `baseline` 的循迹精度、路径长度、完成情况几乎一致（横向
RMS 4.82 vs 4.76 cm），证明**默认参数对既有 PID 调参是透明的**——这正是选
参时的目标。速率约束还顺带把角加速度 RMS 砍了一半（24.3→12.6 rad/s²）、
消除了轮速饱和（0.23%→0）。

但 **h 全程稳在 0.63 附近、最小 0.55，从未接近 0**：`|lateral_error|` 距
`LateralMax=0.8` 一直很远，CBF 的**约束**作用在这条赛道上从未真正被触发。
那 3.83% 的"介入"全部发生在安全集内部（屏障只是给出了比指令更紧的上界），
不是防越界。**不能拿这组数据宣称"CBF 阻止了出界"**——只能说它在不损害
跟踪性能的前提下平滑了控制量。要验证防越界效果，需要构造真正会把
`lateral_error` 推过 0.8 的工况（更快的基础速度、更急的弯、或注入扰动）。

### 结论 2：原地扫描找线在这条赛道上是**负面的**，不要直接上真机

t≈45 s 在右上角发夹弯处四组都丢线。之后：

- `baseline`（不干预）：沿用模型原有的 `Minimum_Search_Speed`（20% 速度
  持续向前爬）+ 冻结转向，**约 5 s 后重新找回线**，继续跑完整圈。
- `search_only` / `both`：HOLD→BRAKE→原地扫描，三级幅度 ±25°/±50°/±80°
  **全部扫完**（SCAN 占 16.1%，约 10.4 s）仍未找到，进 GIVEUP，再未恢复。

排除掉的两个解释（都做过定量核对，别再重复走）：

1. **不是"刹停太早打断了视觉外推"。** 把 `Safety_HoldTime` 从 0.4 s 抬到
   2.0 s（越过视觉自身 1.5 s 的 slowdownTimeout 外推窗口）单独跑过，结果
   不变：64.65 s / 4.81 m / 0 次重捕。
2. **不是"机器人已经离开了线、够不着"。** 用里程计真值查过：丢线瞬间机器人
   距赛道中心线只有 **6.2 cm**，整段丢线期间始终在 3.7~4.5 cm，**全程就压在
   线上**；baseline 位移仅 0.10 m 就恢复了，search 位移 0.05 m 也没离开。

### 真正的原因：线看得见，但检测不出来

把丢线全程（20 s，10001 个采样）的机器人位姿与真值中心线做几何求交，判定线
是否落在 ROI 的地面视场内（ROI 只覆盖前方 **0.163~0.283 m 的一条 12 cm 窄
带**，水平半视场 31.1°，由 `originbot_camera_profile` 的 mujoco 预设与
`LocalPath_ROIFraction=0.30` 推出）：

> **线落在 ROI 地面视场内的时间占比 = 32.2%**，而检测器全程 `found=0`。
> 扫描期间机体航向扫过 97.5°，线反复进出视场。

也就是说原地扫描**确实把线转进了视野**，是**检测环节**没认出来。这与视觉函数
自己文件头写明的已知局限一致：

> 适用性说明：滑窗按图像行向上爬，线走成水平/折返（U 型弯掉头）时窗列会提前停止

发夹弯处线在图像里接近水平，尤其原地旋转时线横着扫过画面，逐行向上爬的滑窗
立刻停摆。

**因此"原地扫描 vs 倒车回溯"根本不是关键**——换成倒车回溯多半也一样，因为
瓶颈在检测而不在把线转/挪进视野。当务之急是让检测器能处理近水平的线，或者
给它一个来自记忆的先验去引导搜索（见下面的"下一步"）。

### 顺带查出的两处与 Python 侧的偏离

- **MATLAB 视觉函数缺 Python 侧 2026-07-29 的自适应 ROI 回退**（Python 的
  `roi_widen_step` / `roi_widen_max`：首次检测失败就把 ROI 加宽重试一次）。
  MATLAB 的 `originbot_sliding_window_path_generator` 签名里没有对应参数，
  只做单次尝试。本次丢线正是"窄 ROI 单次失败"的典型场景。
- **前视距离 0.40 m 超出了 ROI 能看到的最远地面距离 0.283 m**，即前视点始终
  靠多项式**外推**得到，而非落在观测数据范围内。ROI/Lookahead 这对参数当初
  就是"几何推算的合理起点，尚未跑过真值调参 sweep"（见
  `configure_turtlebot3_visual_line_follower_paths.m` 的注释）。

### 又试了一步：几何记忆 + 定向恢复，同样无效

既然线看得见只是检测不出来，那给控制侧一个"线在哪"的先验应该有用——于是实现了
几何记忆（`lf_memory_propagate` / `lf_memory_target` + `RECOVER` 状态）：存下最后
一个**可靠**帧的地面路径点，之后每拍用实测轮速死推到当前机体系，恢复时朝记忆
中的路径点走而不是盲扫。

机制本身是通的（单元测试 20/20，闭环里 `RECOVER` 确实进入并占 11.7%，`GIVEUP`
从 12.9% 降到 1.2%），**但结果依旧**：64.85 s / 4.85 m / 0 次重捕。

原因查清了，而且很硬：

> 丢线前最后一个可靠帧（t=44.75，5 个有效点）的路径点 x 范围是
> **[0.163, 0.171] m —— 整条记忆只有 8 毫米长**，首末点方位 −6.0° 与 −7.0°。
> **记忆里根本不含任何转弯信息。**

因为 ROI 只覆盖 0.163~0.283 m 这条 12 cm 带，而接近发夹弯时检测已先行退化
（`validCount` 从 21 一路衰减到 0），最后能留下的可靠帧已经塌缩到 ROI 最近端的
一小撮点。跟着这样的记忆走就是直行——正好错过弯道。

**结论：记忆解决不了这个失败。**需要的信息（线在盲区之后往哪拐）**从未被观测
到**，任何记忆都变不出来。瓶颈始终在观测：ROI 深度和检测器。

这不意味着记忆代码白写——它是正确且测试过的，等观测horizon修好后会变得有用，
而且新增的 `mem_valid`/`mem_age`/`mem_bearing` 三路诊断已经就位。

### 当前建议的默认值

`Safety_EnableSearch` 的默认值已改为 **0**（关闭整条找线/恢复链），因为实测它
无论哪种策略都不如"什么都不做"。`Safety_EnableFilter` 保持 **1**——约束层是净
收益（角加速度 RMS 减半、消除轮速饱和、跟踪精度不变）。

要重新打开找线，先修上游的观测问题，再重跑 `run_safety_governor_sweep` 验证。

## What was actually verified

**已执行**（2026-08-09，MATLAB R2026a Update 3）：

- `verify_safety_governor.m` 15/15 通过。覆盖 CBF 的符号/边界/奇异保护/闭环
  前向不变性、箱式与速率约束、对纯 PID 基线的透明性、找线状态机完整状态序列
  与扫掠角轨迹、两个消融开关。
- `visual_line_follower_sac_residual.slx` 端到端仿真跑通（20 s，加载
  `simulation_data/sac_training/sac_agent_20260722_191331.mat` 作
  `trainedAgent`）。块接线、`LF_Safety` 参数绑定、信号维度、采样时间都正确；
  找线状态机在真实模型里完整走通 `TRACK→HOLD→BRAKE→SCAN`，扫掠角
  −24°→+48.7°，与前两级幅度（25°/50°）吻合。
- 用模型 InitFcn **实际导出的那份** `LF_Safety`（不是测试脚本里的副本）扫过
  `e ∈ [0, 0.95]`，确认 CBF 边界单调、符号正确、越界时强制反向转。

- 闭环消融测试（地图 `simple`，150 s × 4 组 + 1 组 HoldTime 变体），结果与
  结论见上一节。渲染问题已用 `LD_PRELOAD` 绕过（见末节）。

**未验证，用之前请自己补**：

- **CBF 的"防越界"能力仍未被验证。** 上面那轮测试里 `h` 最小 0.55、离 0 很远，
  屏障约束从未真正生效。要证明它能阻止出界，必须先构造出会把
  `lateral_error` 推过 `LateralMax` 的工况。
- **CBF 的正确性只在开环逐点扫描 + 单元测试里验证过**（符号、单调性、越界时
  强制反向转、闭环前向不变性——但那条单元测试用的是本实现自己假设的横向动力
  学，属于自洽性检查，不是对真实小车的证明）。
- **只测了 `simple` 一张图。** `track_hard` 只跑完 baseline / filter_only 两组
  （found 率 99.97%，几乎不丢线，找线功能在那张图上无从体现）；其余地图未测。
- **真实横向动力学与本实现用的 `d(y_L)/dt = v*sin(psi) + L*omega` 有建模误差**
  （忽略了路径曲率项、轮子打滑、视觉延迟）。
- **默认参数没有跑过调参 sweep**，是按"对现有 PID 基线尽量透明"选的。
- **`visual_line_follower_sac_residual_real.slx` 只做了结构编辑，一次都没跑过。**
  该模型本身在本次改动之前就是"仅创建、未打开验证"状态且有已知阻塞项
  （`CmdVel_Publish` 仍是 `Twist` 而非 `TwistStamped`），见 `matlab/README.md`。
  本次改动没有解决那些问题，也没有引入新的——但它的 `mps` 单位分支同样从未
  在真实运行中验证过。
- 与 SAC 残差策略的交互没有单独评估（安全层看不到指令里哪部分来自策略）。
- Interpreted MATLAB Function 块的执行开销未测；本层不参与代码生成。

### 本机 MuJoCo 渲染问题（与本功能无关，但会挡住闭环验证）

现象：`Rendering has failed in initInThread(). Error code is 2`，视觉恒
`found=0`；更早一步 `mj_initbus` 直接 `std::bad_alloc`，连纯 PID 模型也一样。

根因：MATLAB 自带 `libstdc++.so.6.0.30`（GLIBCXX_3.4.30），而本机 Mesa 25.2.8
的 `radeonsi` 驱动需要系统的 `libstdc++.so.6.0.33`（GLIBCXX_3.4.33）。MEX 主机
进程继承 MATLAB 的 `LD_LIBRARY_PATH`，把 GL 驱动链接到了旧的那份，C++ ABI
不匹配表现为 `std::bad_alloc`。

对 `mj_initbus`（跑在独立 MEX 主机里）可以在 MATLAB 内 `setenv` 后生效：

```matlab
setenv('LD_PRELOAD','/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33');
```

但 MuJoCo 的渲染 S-function 跑在 **MATLAB 进程内**，`setenv` 对已启动的进程
无效。要恢复相机渲染必须从命令行重启 MATLAB：

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 matlab
```

重启后再跑一次 20 s 仿真，`found` 应该恢复正常，CBF 分支才能被闭环触发。
