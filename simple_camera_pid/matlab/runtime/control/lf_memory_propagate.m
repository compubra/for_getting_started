function points = lf_memory_propagate(points, nValid, vMps, omegaRadps, dt)
%LF_MEMORY_PROPAGATE 把记忆中的地面路径点按一步运动增量推算到新的机体系。
%
%   points = lf_memory_propagate(points, nValid, vMps, omegaRadps, dt)
%
% points  (N×2) 机体系地面点 [x y]，X 前向、**Y 右正**（与
%         originbot_sliding_window_path_generator 的输出约定一致）。
%         只有前 nValid 行会被处理，其余保持原样（调用方不该读它们）。
% vMps    本步实测线速度 (m/s)，omegaRadps 实测偏航角速度 (rad/s)
% dt      步长 (s)
%
% 坐标变换的符号约定（**这里最容易错，改之前先读完**）：
%
% 机体先沿自身 X 前进 v*dt，再绕自身旋转 dtheta = omega*dt。点在**新**机体系
% 里的坐标 = 先减去平移、再做旋转：
%
%     xt    = x - v*dt
%     x_new = xt*cos(dtheta) - y *sin(dtheta)
%     y_new = xt*sin(dtheta) + y *cos(dtheta)
%
% 旋转部分的符号由这条物理事实定死：**omega>0 表示左转，机体左转时正前方的点
% 在机体系里向右(+y)移动**（同一条约定也支撑 lf_safety_filter 里的 CBF
% 推导，两处必须一致）。代入正前方点 (L, 0)、dtheta>0 验证：
% y_new = L*sin(dtheta) > 0，确实移向 +y。若把这里的两个 sin 符号写反，记忆
% 会朝错误方向漂移，而且丢线时没有视觉可以纠正它——**不会报错，只会悄悄
% 把机器人引向反方向**。
%
% 死推误差：本函数只做运动学积分，不含任何观测校正。轮子打滑、里程计标定
% 误差都会累积，所以调用方必须给记忆设年龄上限（见 lf_line_search 的 MemMaxAge），
% 过期就丢弃，不要指望它长期可信。
%
% See also LF_MEMORY_TARGET, LF_LINE_SEARCH.

if nValid <= 0
    return
end

dtheta = omegaRadps * dt;
c = cos(dtheta);
s = sin(dtheta);

idx = 1:min(nValid, size(points, 1));
xt = points(idx, 1) - vMps * dt;
y  = points(idx, 2);

points(idx, 1) = xt * c - y * s;
points(idx, 2) = xt * s + y * c;
end
