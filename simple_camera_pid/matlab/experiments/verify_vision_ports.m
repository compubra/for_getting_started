function results = verify_vision_ports()
%VERIFY_VISION_PORTS 2026-08-09 视觉侧移植/新增功能的合成帧验证。
%
%   cd matlab; verify_vision_ports
%
% 不需要 .slx、不需要 MuJoCo：自己生成合成相机帧直接调
% originbot_sliding_window_path_generator，跑一次几秒。
%
% 覆盖：
%   - 自适应搜索半径（新增）：宽线 + 偏心起点，固定半径会被闸门截断而跟丢，
%     自适应半径能跟住
%   - 自适应 ROI 回退（移植自 Python）：线只出现在加宽后的 ROI 里
%   - 丢线转向趋势外推（移植自 Python）：冻结段应按趋势继续变化而非平冻结
%   - 向后兼容：只传原来 12 个参数时仍正常工作
%
% **不覆盖**（诚实声明）：
%   - 霍夫种子的时间连续性 tie-break。它针对的是"真实相机上白线两条边都够格
%     当最长段、单帧噪声翻转谁更长"，在合成的干净帧上无法复现（掩码只保留最大
%     连通块，两条边属于同一块，houghlines 是否把它们拆成两段依赖真实边缘噪声）。
%     该项只做了对 Python 原实现的逐行代码核对，**未经数值验证**。
%   - 与 MATLAB 旧版本的逐位一致性：本文件的改动会改变输出（这正是目的）。
%   - 真实相机帧上的表现：全部用合成帧，真车光照/噪声特性未涉及。

addpath(genpath(fullfile(fileparts(mfilename("fullpath")), "..", "runtime")));

results = struct("name", {}, "pass", {}, "detail", {});
results = runCase(results, "adaptive_radius_tracks_a_wide_offset_line", @adaptiveWideLine);
results = runCase(results, "fixed_radius_loses_that_same_line",        @fixedRadiusFails);
results = runCase(results, "adaptive_still_tracks_a_narrow_line",      @adaptiveNarrowLine);
results = runCase(results, "roi_widen_recovers_a_far_only_line",       @roiWidenRecovers);
results = runCase(results, "roi_widen_off_fails_that_same_frame",      @roiWidenOffFails);
results = runCase(results, "lost_line_extrapolates_the_trend",         @trendExtrapolation);
results = runCase(results, "twelve_arg_call_still_works",              @backwardCompatible);

nPass = sum([results.pass]);
fprintf("\n==== verify_vision_ports: %d/%d passed ====\n", nPass, numel(results));
for k = 1:numel(results)
    if results(k).pass
        fprintf("  PASS  %s\n", results(k).name);
    else
        fprintf("  FAIL  %s\n        %s\n", results(k).name, results(k).detail);
    end
end
if nPass < numel(results)
    error("VerifyVisionPorts:Failed", "%d case(s) failed", numel(results) - nPass);
end
end

% ─────────────────────────────────────────────────────────────────────
function v = mkFrame(prof, lineCenterCol, lineWidth, topRow)
%MKFRAME 合成一帧：暗背景 + 指定列中心/宽度的竖直白带，只画在 topRow 以下。
% 函数内部会按 prof.NeedsFlip 翻正，所以这里先反着翻一次，抵消掉它。
img = 30 * ones(prof.ImageHeight, prof.ImageWidth, 3);
lo = max(1, round(lineCenterCol - lineWidth / 2));
hi = min(prof.ImageWidth, round(lineCenterCol + lineWidth / 2));
img(topRow:end, lo:hi, :) = 230;
if prof.NeedsFlip
    img = flip(img, 1);
end
v = img(:);
end

function out = runGenerator(v, roiFrac, lookahead, varargin)
%RUNGENERATOR 用 mujoco 预设跑一帧。varargin 依次是 roiWidenStep /
% roiWidenMax / adaptiveWindowGain。
clear originbot_sliding_window_path_generator;
out = originbot_sliding_window_path_generator(v, roiFrac, 30, 70, 0.30, ...
    30, 500, lookahead, 0.6, 0.35, 0.04, 'mujoco', varargin{:});
end

function [valid, lookaheadY] = unpackResult(out)
valid = out(5 + 1);      % result = [steer; lat; head; conf; found; debug...]
lookaheadY = out(5 + 3); % debug 槽位 3 = 前视点 Y
end

% ── 自适应搜索半径 ───────────────────────────────────────────────────
function [ok, detail] = adaptiveWideLine()
% 宽线(120px) + 霍夫种子会落在线中心，但把线放在偏离图像中心的位置，
% 固定半径 30 的闸门只能框住线的一小片。自适应半径 = 1.5*120 + 漂移余量，
% 足以框住整条线，游程中点才是线的真中心。
prof = originbot_camera_profile(640 * 480 * 3, 'mujoco');
v = mkFrame(prof, 200, 120, 1);
out = runGenerator(v, 0.50, 0.20, [], [], 1.5);     % 自适应开
[validCount, ~] = unpackResult(out);
ok = validCount >= 25;
detail = sprintf("有效窗点数=%d (期望>=25)", validCount);
end

function [ok, detail] = fixedRadiusFails()
% 同一帧，自适应 vs 固定 WindowHalfWidth=30，**都与 IPM 反算的真值比**。
%
% 固定闸门装不下 121px 的线：findNearestRun 用 [currentCol±30] 去截取列质量，
% 线偏离 currentCol 时截断是非对称的，游程中点被拉向 currentCol，于是横向
% 偏差被系统性低估。实测（2026-08-09）：固定 +26.5% 误差，自适应 +3.5%。
prof = originbot_camera_profile(640 * 480 * 3, 'mujoco');
lineCol = 200; lineWidth = 120;
v = mkFrame(prof, lineCol, lineWidth, 1);

% 真值：线中心列在地面 x=lookahead 处对应的 Y
best = inf; bestRow = NaN;
for r = 1:prof.ImageHeight
    [xg, ~] = originbot_pixel_to_ground(lineCol, r, prof);
    if ~isnan(xg) && abs(xg - 0.20) < best, best = abs(xg - 0.20); bestRow = r; end
end
[~, yTrue] = originbot_pixel_to_ground(lineCol, bestRow, prof);

[~, yAdaptive] = unpackResult(runGenerator(v, 0.50, 0.20, [], [], 1.5));
[~, yFixed]    = unpackResult(runGenerator(v, 0.50, 0.20, [], [], 0));
errAdaptive = abs(yAdaptive - yTrue) / abs(yTrue);
errFixed    = abs(yFixed - yTrue) / abs(yTrue);

ok = errAdaptive < 0.10 && errFixed > 0.15;
detail = sprintf("真值 Y=%.4f | 自适应 %.4f (误差 %.1f%%, 需<10%%) | 固定 %.4f (误差 %.1f%%, 需>15%%)", ...
    yTrue, yAdaptive, 100 * errAdaptive, yFixed, 100 * errFixed);
end

function [ok, detail] = adaptiveNarrowLine()
% 回归保护：窄线(20px)在自适应下不能变差——半径会收到下限 8 附近，
% 仍须正常跟踪
prof = originbot_camera_profile(640 * 480 * 3, 'mujoco');
v = mkFrame(prof, 320, 20, 1);
out = runGenerator(v, 0.50, 0.20, [], [], 1.5);
[validCount, ~] = unpackResult(out);
ok = validCount >= 25;
detail = sprintf("有效窗点数=%d (期望>=25)", validCount);
end

% ── 自适应 ROI 回退 ──────────────────────────────────────────────────
function [ok, detail] = roiWidenRecovers()
% 线只画在图像上半部：窄 ROI(0.20，只覆盖底部 20% 行)完全看不到，
% 加宽 0.2 → 0.40 之后才进入视野
prof = originbot_camera_profile(640 * 480 * 3, 'mujoco');
v = mkFrame(prof, 320, 60, 1);
v = zeroBottomRows(v, prof, round(0.30 * prof.ImageHeight));
out = runGenerator(v, 0.20, 0.20, 0.2, 0.7, 1.5);
found = out(5);
detail = sprintf("found=%g (期望 1)", found);
ok = found > 0.5;
end

function [ok, detail] = roiWidenOffFails()
% 同一帧关掉加宽(step=0)：应当找不到，证明上一条确实是加宽救回来的
prof = originbot_camera_profile(640 * 480 * 3, 'mujoco');
v = mkFrame(prof, 320, 60, 1);
v = zeroBottomRows(v, prof, round(0.30 * prof.ImageHeight));
out = runGenerator(v, 0.20, 0.20, 0, 0.7, 1.5);
found = out(5);
ok = found < 0.5;
detail = sprintf("found=%g (期望 0)", found);
end

function v = zeroBottomRows(v, prof, nRows)
% 把（翻正后视角下的）最底部 nRows 行涂黑，制造"近处无线、远处有线"
img = reshape(v, prof.ImageHeight, prof.ImageWidth, 3);
if prof.NeedsFlip, img = flip(img, 1); end
img(end - nRows + 1:end, :, :) = 30;
if prof.NeedsFlip, img = flip(img, 1); end
v = img(:);
end

% ── 丢线趋势外推 ─────────────────────────────────────────────────────
function [ok, detail] = trendExtrapolation()
% 先喂一串线中心稳定右移的帧（转向误差单调上升），然后突然全黑丢线。
% 冻结段(<=0.5s)里 steering_error 应继续沿趋势上升，而不是停在最后一帧的值。
prof = originbot_camera_profile(640 * 480 * 3, 'mujoco');
clear originbot_sliding_window_path_generator;
cols = 320:12:320 + 12 * 7;
lastSteer = 0;
for c = cols
    out = originbot_sliding_window_path_generator(mkFrame(prof, c, 60, 1), ...
        0.50, 30, 70, 0.30, 30, 500, 0.20, 0.6, 0.35, 0.04, 'mujoco');
    lastSteer = out(1);
end
dark = 30 * ones(prof.ImageHeight * prof.ImageWidth * 3, 1);
outLost1 = originbot_sliding_window_path_generator(dark, 0.50, 30, 70, 0.30, ...
    30, 500, 0.20, 0.6, 0.35, 0.04, 'mujoco');
outLost2 = originbot_sliding_window_path_generator(dark, 0.50, 30, 70, 0.30, ...
    30, 500, 0.20, 0.6, 0.35, 0.04, 'mujoco');
s1 = outLost1(1);
s2 = outLost2(1);
% 线一直右移 → 转向误差递增 → 外推应让丢线后的值继续大于最后可见值，
% 且第二拍比第一拍更大（平冻结的话三者会完全相等）
ok = s1 > lastSteer + 1e-9 && s2 > s1 + 1e-9;
detail = sprintf("最后可见=%.6f, 丢线第1拍=%.6f, 第2拍=%.6f (期望严格递增)", ...
    lastSteer, s1, s2);
end

% ── 向后兼容 ─────────────────────────────────────────────────────────
function [ok, detail] = backwardCompatible()
% 四个 .slx 里的 MATLABFcn 表达式仍然只传 12 个参数，必须照常工作
prof = originbot_camera_profile(640 * 480 * 3, 'mujoco');
clear originbot_sliding_window_path_generator;
out = originbot_sliding_window_path_generator(mkFrame(prof, 320, 60, 1), ...
    0.50, 30, 70, 0.30, 30, 500, 0.20, 0.6, 0.35, 0.04, 'mujoco');
ok = numel(out) == 72 && out(5) > 0.5;
detail = sprintf("输出长度=%d (期望 72), found=%g (期望 1)", numel(out), out(5));
end

% ─────────────────────────────────────────────────────────────────────
function results = runCase(results, name, fn)
try
    [ok, detail] = fn();
catch err
    ok = false;
    detail = sprintf("threw %s: %s", err.identifier, err.message);
end
results(end + 1) = struct("name", name, "pass", logical(ok), "detail", detail);
end
