function errors = rl_true_track_errors(poseVector, centerline)
%RL_TRUE_TRACK_ERRORS Track-relative lateral/heading error from a pose sample.
%
%   errors = rl_true_track_errors([pos3; quatWXYZ], centerline)
%     -> [lateral_error_m; heading_error_rad]
%
%   离线评估工具（不参与训练——训练端奖励是纯视觉的，不依赖位姿）。
%   用途：对已记录的位姿序列逐帧计算相对赛道中心线的真实横向/航向偏差，
%   例如：
%     - 仿真评估：out.mujoco_odom_position + odom_quaternion 的逐帧数据
%     - 真车评估：/odom 话题记录的位姿（回合从标记点出发并重置里程计，
%       使 odom 系与中心线坐标系对齐）
%
%   Inputs
%   ------
%     poseVector  7x1  [x; y; z; qw; qx; qy; qz]  (MuJoCo 与 ROS 的
%                 nav_msgs/Odometry 四元数均可，注意 wxyz 顺序)
%     centerline  Nx2  [x y] closed-loop polyline in world metres,
%                 from get_track_centerline.
%
%   Outputs (sign conventions match the vision estimates)
%   -------
%     lateral_error  signed distance robot-to-centreline (m).
%                    > 0 when the robot is LEFT of the track direction.
%     heading_error  signed angle from track tangent to robot heading
%                    (rad, wrapped to [-pi, pi]). > 0 when the robot points
%                    left of the track direction. The tangent is flipped to
%                    whichever direction the robot is currently following,
%                    so either lap direction is valid (direction-agnostic).
%
%   All-zero pose samples (e.g. MuJoCo's initialization sample, quaternion
%   norm 0) return [0; 0].
%
%   See also GET_TRACK_CENTERLINE.

% Guards: undefined pose (zero quaternion) or missing centreline.
quatNormSq = poseVector(4)^2 + poseVector(5)^2 + ...
    poseVector(6)^2 + poseVector(7)^2;
if quatNormSq < 0.5 || size(centerline, 1) < 3
    errors = [0; 0];
    return
end

x  = poseVector(1);
y  = poseVector(2);
qw = poseVector(4);
qx = poseVector(5);
qy = poseVector(6);
qz = poseVector(7);
yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy^2 + qz^2));

% Nearest centreline vertex (600-point polyline -> ~2 cm spacing on the
% largest loops).
d2 = (centerline(:, 1) - x).^2 + (centerline(:, 2) - y).^2;
[~, nearest] = min(d2);

% Local tangent from the wrap-around neighbours (closed loop).
n  = size(centerline, 1);
ip = mod(nearest, n) + 1;
im = mod(nearest - 2, n) + 1;
tangent = centerline(ip, :) - centerline(im, :);
tangent = tangent / max(hypot(tangent(1), tangent(2)), eps);

% Direction-agnostic: align the tangent with the robot's current heading.
heading = [cos(yaw), sin(yaw)];
if tangent * heading.' < 0
    tangent = -tangent;
end

% Signed lateral offset: z-component of cross(tangent, robot - nearest).
offset = [x, y] - centerline(nearest, :);
lateralError = tangent(1) * offset(2) - tangent(2) * offset(1);

% Signed heading offset: angle from tangent to heading.
headingError = atan2( ...
    tangent(1) * heading(2) - tangent(2) * heading(1), ...
    tangent(1) * heading(1) + tangent(2) * heading(2));

errors = [lateralError; headingError];
end
