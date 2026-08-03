function prof = originbot_camera_profile(numPixelsTotal, platform)
%ORIGINBOT_CAMERA_PROFILE 按显式 platform 参数返回对应平台的相机/算法常数。
%
% 2026-08-03 起 platform 为必填参数（'mujoco'|'gazebo'|'real'）——此前靠
% numPixelsTotal（图像元素总数）自动识别平台，MuJoCo 与 Gazebo 图像元素数
% 互质时（640×480×3=921600 vs 800×600×3=1440000）这个技巧零额外开销且可靠；
% 但三平台分辨率统一为 640×480 之后，像素总数已无法区分 MuJoCo/Gazebo/real，
% 必须显式传参。numPixelsTotal 仍保留（仅用于与预期分辨率做一致性校验，
% 不再参与平台判别），调用方不变的地方是 originbot_sliding_window_path_generator
% 与 originbot_line_follower_debug_frame 的表达式已加上平台参数——本函数
% 仍是两者共用的相机内参/外参/像素尺度常数唯一真相源。
%
%   prof = originbot_camera_profile(numel(rgbVector), 'mujoco')
%
% 字段：
%   ImageWidth/ImageHeight  图像尺寸(px)
%   NeedsFlip               是否需要 flip(rgb,1) 翻正（MuJoCo bottom-row-first）
%   FovyDeg/MountHeight/PitchDeg  IPM 用相机内参/外参
%   WindowHalfWidth/MinWindowPixels/HoughMinLength/HoughFillGap
%                            滑窗与霍夫的像素尺度常数（随分辨率缩放）
%   HoughMaxPeaks/MaxDriftPerRow  与分辨率无关，两平台共用同一常数
%   DefaultROI/DefaultLookahead   缺参时的回退值，与各自平台 InitFcn
%                            导出表(实际调参值)一致

expectedPixels = 640 * 480 * 3;   % 三平台 2026-08-03 起统一分辨率

switch platform
    case 'gazebo'
        % ── Gazebo：TurtleBot3 Burger 前置相机（RealSense r200 光学参数，水平安装）──
        % 2026-08-03 起分辨率从 800x600 改回 640x480，与 MuJoCo/real 三平台
        % 统一（此前 2026-07-31~2026-08-03 曾用过 800x600，见 git 历史/
        % model.sdf 的 <image> 注释）。640x480 与之前的 800x600 同为 4:3，
        % FovyDeg 无需重新推导，仍是由 SDF horizontal_fov=1.02974 rad 按 4:3
        % 换算得到的 45.9857（与 MuJoCo 自己的 45.9857 撞了个巧合，两边都是
        % 4:3，不是共用同一个推导）。像素尺度常数随分辨率变化，640 宽是原
        % 800 宽的 640/800=0.8 倍，故按比例缩回：WindowHalfWidth 25->20、
        % MinWindowPixels 6.25->5、HoughMinLength 50->40、HoughFillGap 25->20
        % （正好落回 MuJoCo 自己的数值，因为两者现在都是 640 宽）。
        prof.ImageWidth  = 640;
        prof.ImageHeight = 480;
        prof.NeedsFlip   = false;   % ROS image_raw 顶行在上，帧本就正立
        prof.FovyDeg     = 45.9857; % 由 SDF horizontal_fov=1.02974 rad 按 4:3 换算
        prof.MountHeight = 0.133;   % 相机离地高度(m)
        prof.PitchDeg    = 0;       % 水平安装，无下俯
        prof.WindowHalfWidth = 20;
        prof.MinWindowPixels = 5;
        prof.HoughMinLength  = 40;
        prof.HoughFillGap    = 20;
        % ROIFraction/Lookahead 缺参时的回退值，与当前 Python/Gazebo 侧调参一致
        % （2026-08-02 起 0.5，修复窄 ROI 在部分地图弯道上频繁检测失败的问题，
        % 见 config/gazebo/gazebo_line_follower.yaml 同期改动；该问题是
        % ROI 几何比例的问题，与本次的分辨率改动无关，此值不受影响）。
        prof.DefaultROI       = 0.50;
        prof.DefaultLookahead = 0.20;
    case 'mujoco'
        % ── MuJoCo：TurtleBot3 Burger 前置相机（2026-07-22 起水平安装，原 15° 下俯
        % 已归档，见 archive/archive_vision_scheme_a）──
        prof.ImageWidth  = 640;
        prof.ImageHeight = 480;
        prof.NeedsFlip   = true;    % MuJoCo framebuffer 为 bottom-row-first，需翻正
        prof.FovyDeg     = 45.9857; % MJCF <camera fovy>，垂直 FOV
        prof.MountHeight = 0.133;   % 相机离地高度(m)
        prof.PitchDeg    = 0;       % 水平安装，无下俯（拍平前为 15）
        prof.WindowHalfWidth = 20;
        prof.MinWindowPixels = 5;
        prof.HoughMinLength  = 40;
        prof.HoughFillGap    = 20;
        % 拍平后近端可视地面距离下限≈0.31m(由 MountHeight/FovyDeg 决定，水平相机
        % physically 看不到更近的地面)，故 ROI/Lookahead 不能再沿用旧的 15°-下俯
        % 时代数值(0.50/0.20)——0.20 会被下游 clamp 强制抬到约 0.31m 且不易察觉。
        % 当前为几何推算的合理起点，尚未跑过 2026-07-19 那种真值调参 sweep，
        % 建议后续用同样方法重新调优（见 vision-scheme-b-tuning 记忆）。
        prof.DefaultROI       = 0.30;
        prof.DefaultLookahead = 0.40;
    case 'real'
        % ── 真实 TurtleBot3 Burger 相机 ── 2026-08-03 起本项目支持的第三个
        % 平台。ImageWidth/ImageHeight/PitchDeg 已对真车实测确认；
        % FovyDeg/MountHeight 沿用 Gazebo/MuJoCo 同款数值，尚未对真车单独
        % 标定确认（见 Python 侧 turtlebot3_burger_real_camera() 与
        % config/real/real_line_follower.yaml 里的 UNCONFIRMED 标注）。
        prof.ImageWidth  = 640;   % 2026-08-03 起真车相机驱动确认为此分辨率
        prof.ImageHeight = 480;
        prof.NeedsFlip   = false;   % ROS image_raw 顶行在上，帧本就正立
        prof.FovyDeg     = 45.9857; % 未对真车单独标定，沿用 Gazebo/MuJoCo 数值
        prof.MountHeight = 0.133;   % 未对真车单独标定，沿用 Gazebo/MuJoCo 数值
        prof.PitchDeg    = 15;      % 2026-08-03 对真车实测确认：15° 下俯安装
        prof.WindowHalfWidth = 20;
        prof.MinWindowPixels = 5;
        prof.HoughMinLength  = 40;
        prof.HoughFillGap    = 20;
        % 与当前 config/real/real_line_follower.yaml 的 roi_bottom_fraction/
        % lookahead_distance 默认值一致，未跑过真值调参 sweep。
        prof.DefaultROI       = 0.30;
        prof.DefaultLookahead = 0.20;
    otherwise
        error("OriginBot:UnknownCameraPlatform", ...
            "platform must be 'mujoco', 'gazebo', or 'real', got %s.", ...
            string(platform));
end

if numPixelsTotal ~= expectedPixels
    error("OriginBot:UnexpectedImageSize", ...
        ["Input has %d pixel elements for platform '%s', expected " ...
         "640x480x3=%d. Check the upstream RGB reshape/vectorization " ...
         "matches this platform's configured camera resolution."], ...
        numPixelsTotal, platform, expectedPixels);
end

prof.HoughMaxPeaks  = 4;    % 霍夫峰值数，三平台一致
prof.MaxDriftPerRow = 3;    % 霍夫斜率钳位，比值量纲、不随分辨率缩放
end
