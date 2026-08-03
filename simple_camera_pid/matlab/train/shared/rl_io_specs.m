function [obsInfo, actInfo] = rl_io_specs(maxDeltaOmega, maxDeltaV)
%RL_IO_SPECS Single source of truth for the residual-RL observation/action specs.
%
%   残差 RL 的观测/动作规格只在这里定义一次。build_sac_residual_controller
%   (placeholder agent) 和两个训练入口 train_sac_residual_live /
%   train_ppo_residual_live 都调用本函数——SAC/PPO 残差控制器的 Simulink
%   I/O 契约完全相同，避免在多处手抄、改一处漏一处。
%   This is the ONE place the observation and action specifications live.
%   build_sac_residual_controller (placeholder agent) and both training
%   entries (train_sac_residual_live / train_ppo_residual_live) call this —
%   the SAC and PPO residual controllers share the same Simulink I/O
%   contract, so the dimensions and bounds can never drift apart.
%
%   Observation (8-D):
%     [steering_error; lateral_error; heading_error; found;
%      prev_delta_v; prev_delta_omega; wheel_speed_norm; yaw_rate_norm]
%
%     1-4 视觉。5-6 是上一步 2 维动作的记忆（action memory，来自 UnitDelay，
%     不是学出来的补偿量本身——那是第 2 个动作维度，见下）。顺序 [v; omega]
%     与动作向量、与 PID/SAC 的速度指令输出顺序一致（2026-07-21 起两处
%     "线速度+角速度→左右轮速"的转换已合并成根层共享子系统
%     `Diff_Drive_Kinematics`，输入顺序统一为 [v; omega]）。
%     7-8 车体速度反馈，由左右轮编码器速度经差速运动学归一化而来
%     （模型内计算，真车上编码器同样可测，不破坏 sim-to-real）：
%       wheel_speed_norm = (w_L + w_R) / (2*MaxWheelSpeed)   ∈ [-1, 1]
%                        = v / (r*MaxWheelSpeed)     前向速度归一化
%       yaw_rate_norm    = (w_R - w_L) / (2*MaxWheelSpeed)   ∈ [-1, 1]
%                        = omega / (2*r*MaxWheelSpeed/b)  偏航角速度归一化
%     轮半径 r / 轮距 b 在归一化中约掉，所以域随机化扰动 TB3_WheelRadius /
%     TB3_WheelSeparation 不影响观测的量纲一致性。
%
%   Action (2-D):
%     [delta_v; delta_omega]
%       delta_v      in [-maxDeltaV, +maxDeltaV]
%         agent 自己学出来的线速度补偿，与 PID 的线速度指令（Vision_Enable_Speed）
%         同一量纲，在 `Diff_Drive_Kinematics` 里跟 PID 的 v 相加后才转左右轮速——
%         纯前进速度修正，不产生转向。
%       delta_omega  in [-maxDeltaOmega, +maxDeltaOmega]  rad/s
%         残差偏航角速度，与 PID 的 angular_cmd 同一量纲、同一符号约定
%         （正值=左转/CCW，跟 PID 的 SteeringSign 输出一致），同样在
%         `Diff_Drive_Kinematics` 里跟 PID 的 omega 相加后才转左右轮速差。
%       ⚠️ 2026-07-21 之前 SAC 内部曾自带一套差速增益，其符号约定与 PID 的
%       angular_cmd 恰好相反（PID: left=v-omega_term, right=v+omega_term；
%       旧 SAC: left=+omega_term, right=-omega_term）——这是从未被注意到的
%       历史不一致，现在两者的 v/omega 在同一个下游子系统里相加，已统一
%       为 PID 的符号约定，旧的不一致不复存在。
%
%   Usage
%   -----
%     [obsInfo, actInfo] = rl_io_specs(1.0, 1.0);   % both bounds +/-1.0
%
%   If maxDeltaOmega / maxDeltaV are omitted, default 1.0 (matches the
%   training configs in train/ and the SAC_/PPO_MaxDeltaOmega /
%   SAC_MaxDeltaV model-workspace variables) is used for each.
%
%   NOTE: agents trained against the old 1-D-action / 7-D-obs spec, or
%   against the pre-2026-07-21 [delta_omega; delta_v] element order, are
%   incompatible with this spec (RL Agent block will refuse to load a
%   dimension mismatch; an order swap alone would silently retrain against
%   mislabeled channels) — retrain from scratch after this change.

arguments
    maxDeltaOmega (1,1) double {mustBePositive} = 1.0
    maxDeltaV     (1,1) double {mustBePositive} = 1.0
end

% Observation: [steering_error; lateral_error; heading_error; found;
%               prev_delta_v; prev_delta_omega; wheel_speed_norm; yaw_rate_norm]
obsInfo = rlNumericSpec([8 1], ...
    "LowerLimit", [-1.5; -1.0; -1.0; 0.0; -maxDeltaV; -maxDeltaOmega; -1.0; -1.0], ...
    "UpperLimit",  [ 1.5;  1.0;  1.0; 1.0;  maxDeltaV;  maxDeltaOmega;  1.0;  1.0]);
obsInfo.Name = "LineFollowerObs";
obsInfo.Description = ...
    "steering_error, lateral_error, heading_error, found, " + ...
    "prev_delta_v, prev_delta_omega, wheel_speed_norm, yaw_rate_norm";

% Action: [delta_v; delta_omega]
actInfo = rlNumericSpec([2 1], ...
    "LowerLimit", [-maxDeltaV; -maxDeltaOmega], ...
    "UpperLimit", [ maxDeltaV;  maxDeltaOmega]);
actInfo.Name        = "ResidualAction";
actInfo.Description = ...
    "[delta_v: learned linear-velocity compensation, combined with PID's " + ...
    "v before the shared Diff_Drive_Kinematics conversion; " + ...
    "delta_omega: residual yaw-rate correction (rad/s), combined with " + ...
    "PID's omega the same way]";
end
