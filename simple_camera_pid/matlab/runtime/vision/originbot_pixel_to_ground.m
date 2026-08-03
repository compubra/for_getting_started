function [Xground, Yground] = originbot_pixel_to_ground(u, v, prof)
%ORIGINBOT_PIXEL_TO_GROUND 针孔+平地假设的逆透视投影（IPM）。
%
% 把像素 (u,v) 投到地面 (X 前向, Y 右正)，单位 m。prof 来自
% originbot_camera_profile，携带该平台的分辨率/FOV/安装高度/俯仰角；公式
% 本身两平台通用，是控制（originbot_sliding_window_path_generator）与调试
% 可视化（originbot_line_follower_debug_frame，经 originbot_ground_to_pixel
% 逆投影画拟合曲线）共用的唯一实现。
%
% 返回 NaN 表示该像素在地平线以上、或投影距离超出 maxGroundRange，与地面
% 无有效交点。

fy = (prof.ImageHeight / 2) / tand(prof.FovyDeg / 2);
fx = fy;
cx = (prof.ImageWidth + 1) / 2;
cy = (prof.ImageHeight + 1) / 2;

% 相机坐标系归一化射线 (x右, y下, z前)
xc = (u - cx) / fx;
yc = (v - cy) / fy;
zc = ones(size(u));

% 绕相机右轴俯仰 pitch，转到车体水平参考系
ph = deg2rad(prof.PitchDeg);
dDown = cos(ph) .* yc + sin(ph) .* zc;     % 车体系向下分量
dForward = -sin(ph) .* yc + cos(ph) .* zc; % 车体系前向分量
dRight = xc;

% 与地面求交：沿射线下降量达到安装高度
t = prof.MountHeight ./ dDown;
t(dDown <= 1e-6) = NaN;                     % 射线未朝下 → 地平线以上，无交点

Xground = t .* dForward;
Yground = t .* dRight;                      % 右正，匹配下游 yPoints 约定

% 近地平线奇点保护：dDown→0 时 t 爆炸，前向距离被病态放大，会污染 polyfit。
% 循线前视 <1~2m 才有意义，超过 maxGroundRange 视为无效。
maxGroundRange = 5.0;                        % m
beyond = ~(Xground <= maxGroundRange);      % 同时把 NaN 也算作 beyond
Xground(beyond) = NaN;
Yground(beyond) = NaN;
end
