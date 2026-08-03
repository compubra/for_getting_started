function dr = rl_dr_defaults()
%RL_DR_DEFAULTS Default domain-randomization config for residual RL training.
%
%   残差 RL 域随机化的默认范围，SAC 与 PPO 训练共用（域随机化与算法无关）。
%   改这里的数字即可调节每一类随机化的强度，或把某一组的 Enable 置 false
%   关掉它。传给 rl_domain_randomization 构造 ResetFcn。若同时开多个训练
%   （如 SAC 和 PPO 并发）且启用 GenTrack，请给每个训练把 dr.GenTrack.Name
%   改成各自独立的基名（PPO 脚本已内联设 "ppo_gen_episode"），避免互相
%   覆盖生成的场景文件。
%   This is the ONE place to tune how aggressively each group is randomized,
%   shared by both SAC and PPO training. See rl_domain_randomization for how
%   each field is applied.
%
%   百分比字段 p：每回合从 base*(1 +/- p) 均匀采样。
%   Percent fields (p): sampled uniformly from base*(1 +/- p) per episode.
%
%   See also RL_DOMAIN_RANDOMIZATION, SAC_TRAINING_CONFIG, PPO_TRAINING_CONFIG.

% 总开关：false 时完全不随机化（ResetFcn 退化为恒等）。
dr.Enable = true;

% ── 地图随机化：每回合从 Maps 列表里随机选一张地图（MuJoCo 场景）──
%    通过 SimulationInput.setBlockParameter 覆写 MuJoCo Plant 的 xmlFile，
%    只影响当前回合，不改动 .slx 里持久化的地图。空列表或 Enable=false 时
%    使用模型当前地图，不随机切图。
%    Per-episode random map: pick one of Maps each episode. Valid keys are the
%    ones known to resolve_turtlebot3_mujoco_scene.
dr.Map.Enable = false;
dr.Map.Maps   = ["simple","complex","ellipse","training", ...
                 "track_easy","track_medium","track_hard"];

% ── 现场生成随机赛道：每回合现场新生成一张闭环赛道（写 PNG+XML，复用同一
%    文件名覆盖），再 setBlockParameter 加载。启用后优先于 Map（用新鲜随机
%    赛道，而不是固定 7 图）。因为要改 nontunable 的 xmlFile，训练时必须关
%    Fast Restart（trainer 会自动处理）。
%    Per-episode freshly-generated track. When Enable=true this takes priority
%    over the fixed Map list.
%    2026-07-22 起有 3 种生成算法可选（由 Generator 字段决定，"random" 时
%    每回合从三者里随机挑一个，训练见到的赛道形状更多样）：
%      simple    —— gen_simple_track_scene：椭圆/胶囊，规整几何体
%      complex   —— gen_complex_track_scene：折线+独立随机圆弧倒角，带尖角
%      wandering —— gen_wandering_track_scene：处处光滑的有机蜿蜒闭环
%    三者共用同一个 Difficulty 语义（0..1，越大越难）；Shape 只在
%    Generator=simple 时生效。
dr.GenTrack.Enable     = true;   % 默认关；需要"每回合新图"时置 true
dr.GenTrack.Generator  = "random"; % simple | complex | wandering | random
dr.GenTrack.Shape      = "random";% 仅 Generator=simple 时用：ellipse | capsule | random
dr.GenTrack.Difficulty = 0.4;     % 0..1，越大越难（环越小/弯越急/扰动越大，语义因生成器而异）
dr.GenTrack.Name       = "rl_gen_episode";  % 复用此基名，覆盖写，避免塞满磁盘

% ── 起始位姿：机器人初始横向偏移（同 MJ_TargetLateralPosition 单位）──
dr.Pose.Enable       = true;
dr.Pose.LateralRange = 0.05;   % ±0.05：起点左右偏移，逼策略从不同位置贴线

% ── 机器人动力学：轮子与最大轮速的制造/磨损差异（相对基准 ±百分比）──
dr.Dynamics.Enable         = true;
dr.Dynamics.WheelRadiusPct = 0.05;   % 轮半径 ±5%
dr.Dynamics.WheelSepPct    = 0.05;   % 轴距   ±5%
dr.Dynamics.MaxWheelPct    = 0.10;   % 最大轮速 ±10%

% ── 基础速度 / PID：让残差策略适应不同底层控制器表现 ──
dr.Control.Enable      = true;
% 基础线速度：不用百分比抖动，直接按 TurtleBot3 Burger 官方线速度量程做绝对
% 范围随机——0(静止)~0.22 m/s(Burger 额定最大线速度，ROBOTIS 官方规格)。
% 换算成模型的 BaseLinearSpeed 单位：模型里 wheel_rad_s = BaseLinearSpeed *
% BaseSpeedScale/TB3_WheelRadius，真实 wheel_rad_s = v_real/TB3_WheelRadius，
% 两边令其相等，TB3_WheelRadius 正好约掉 → BaseLinearSpeed = v_real/BaseSpeedScale，
% 好处是不受 dr.Dynamics.WheelRadiusPct 域随机化影响（否则每回合还要按当次
% 采样到的轮半径重新换算，徒增耦合）。
dr.Control.BaseSpeedMin = 0;            % 对应真实 0 m/s（静止/最低速）
dr.Control.BaseSpeedMax = 0.22 / 0.03;  % 对应 Burger 额定最大线速度 0.22 m/s (≈7.33)
dr.Control.KpPct        = 0.15;   % Kp ±15%
dr.Control.KiAbs        = 0.0;    % Ki 基准为 0，用绝对抖动 [0, KiAbs]；0=不动
dr.Control.KdPct        = 0.15;   % Kd ±15%

% ── 感知 / 视觉：光照、阈值、ROI 漂移，模拟相机/环境变化 ──
dr.Perception.Enable           = true;
dr.Perception.ErrorScalePct    = 0.10;   % 误差换算增益 ±10%
dr.Perception.MinBrightnessPct = 0.15;   % 白线亮度阈值 ±15%
dr.Perception.ROIStartPct      = 0.05;   % ROI 起始行 ±5%（结果夹到 [0,1]）
dr.Perception.ROIHeightPct     = 0.10;   % ROI 高度 ±10%（结果取整）
end
