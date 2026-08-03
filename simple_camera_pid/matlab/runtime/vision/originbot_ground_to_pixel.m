function [u, v] = originbot_ground_to_pixel(Xg, Yg, prof)
%ORIGINBOT_GROUND_TO_PIXEL originbot_pixel_to_ground 的逆投影。
%
% 地面点 (X 前向, Y 右正，单位 m) → 像素 (u,v)。仅供调试可视化把地面系拟合
% 曲线反投影回相机图使用（originbot_line_follower_debug_frame），控制路径
% 不需要这个方向。prof 来自 originbot_camera_profile，与 originbot_pixel_
% to_ground 使用同一份相机内参/外参，两者互为反函数。
%
% 视线在相机后方（zc<=0，即该地面点不在可视范围内）时返回 NaN。

fy = (prof.ImageHeight / 2) / tand(prof.FovyDeg / 2);
fx = fy;
cx = (prof.ImageWidth + 1) / 2;
cy = (prof.ImageHeight + 1) / 2;

ph = deg2rad(prof.PitchDeg);
dForward = Xg;
dRight = Yg;
dDown = prof.MountHeight .* ones(size(Xg));

% 车体系 → 相机系（originbot_pixel_to_ground 内旋转的转置/逆）
yc = cos(ph) .* dDown - sin(ph) .* dForward;
zc = sin(ph) .* dDown + cos(ph) .* dForward;
xc = dRight;

u = nan(size(Xg));
v = nan(size(Xg));
inFront = zc > 1e-6;
u(inFront) = cx + fx .* xc(inFront) ./ zc(inFront);
v(inFront) = cy + fy .* yc(inFront) ./ zc(inFront);
end
