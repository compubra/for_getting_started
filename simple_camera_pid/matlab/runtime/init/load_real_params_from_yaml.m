function params = load_real_params_from_yaml(yamlPath)
%LOAD_REAL_PARAMS_FROM_YAML 从 Python 侧的真车 yaml 读取参数，作为 MATLAB
% 真机模型的参数来源，消除「MATLAB 与 Python 各存一份真车参数」的分叉。
%
%   params = load_real_params_from_yaml()          % 用默认路径
%   params = load_real_params_from_yaml(yamlPath)  % 指定 yaml
%
% 背景：2026-08-08 盘点发现 MATLAB 与 Python 的真车参数已经各自漂移——
% MinBrightness 70 vs 170、MaxSaturation 0.30 vs 0.20、ROIFraction 0.10 vs
% 0.3、Kp 也不一致。两边都在改、都没同步，于是任何一边测出来的结论对另一边
% 都不成立。config/real/real_line_follower.yaml 是真车运行时实际读的文件，
% 因此把它定为唯一真相源，MATLAB 只做消费方。
%
% 用法：在真机模型的 InitFcn 里调用，把返回的结构体写进 model workspace，
% 取代 configure_visual_line_follower_real_debug.m 里那张硬编码 defaults 表。
% 本函数只读、不写任何工作区，方便单独调用核对。
%
% 只解析该 yaml 的 `/**:` 段（三种部署方式共用的调参段）。刻意手写解析而不
% 依赖 R2023b+ 的 readstruct/yaml 支持：该 yaml 里成片的中英文注释和
% `/**` 这种非标准键名容易踩到工具箱解析器的边界，而这里需要的只是
% 「缩进两级的 scalar 键值对」这一种最简单的形式。
%
% 未做的事：不解析 PID 段以外的节点专属段（image_topic 之类话题名，MATLAB
% 侧本来就在模型里单独配）；不处理 list/嵌套；yaml 里出现本函数不认识的键
% 会被忽略而不是报错（新增键时这里要跟着加，见下方 wanted 表）。

arguments
    yamlPath (1,1) string = defaultYamlPath()
end

if ~isfile(yamlPath)
    error("OriginBot:YamlNotFound", ...
        "找不到真车参数文件：%s\n（若工作区不在预期位置，显式传入路径）", yamlPath);
end

% yaml 键 -> MATLAB 模型工作区变量名。左列必须与
% config/real/real_line_follower.yaml `/**:` 段里的键完全一致。
wanted = [
    "roi_bottom_fraction",  "LocalPath_ROIFraction"
    "num_points",           "LocalPath_NumPoints"
    "lookahead_distance",   "LocalPath_LookaheadDistance"
    "lateral_gain",         "LocalPath_LateralGain"
    "heading_gain",         "LocalPath_HeadingGain"
    "curvature_gain",       "LocalPath_CurvatureGain"
    "min_brightness",       "OriginBot_MinBrightness"
    "max_saturation",       "OriginBot_MaxSaturation"
    "min_pixels",           "OriginBot_MinPixels"
    "error_scale",          "OriginBot_ErrorScale"
    "kp",                   "Kp_Line"
    "ki",                   "Ki_Line"
    "kd",                   "Kd_Line"
    "n_filter",             "N_Line"
    % steering_sign 刻意不在此表：它是 ControllerConfig 的常量，Python 侧没
    % 把它开成 ROS 参数，yaml 里也就没有这个键。真要同步得先在 Python 侧
    % declare_parameter 出来，否则这里只会每次调用都告警。
    "base_linear_speed",    "BaseLinearSpeed"
    "base_speed_scale",     "BaseSpeedScale"
    "max_angular_speed",    "MaxAngularSpeed"
    "camera_fovy_deg",      "Camera_FovyDeg"
    "camera_mount_height",  "Camera_MountHeight"
    "camera_pitch_deg",     "Camera_PitchDeg"
    "camera_roll_deg",      "Camera_RollDeg"
    "camera_yaw_deg",       "Camera_YawDeg"
    "image_width",          "Camera_ImageWidth"
    "image_height",         "Camera_ImageHeight"
    ];

raw = readSharedSection(yamlPath);

params = struct();
missing = string.empty;
for k = 1:size(wanted, 1)
    key = wanted(k, 1);
    if isKey(raw, key)
        params.(wanted(k, 2)) = raw(key);
    else
        missing(end+1) = key; %#ok<AGROW>
    end
end

if ~isempty(missing)
    warning("OriginBot:YamlKeysMissing", ...
        "yaml 的 /**: 段里没找到这些键，对应的 MATLAB 变量不会被设置：%s", ...
        strjoin(missing, ", "));
end
end


function p = defaultYamlPath()
% 本文件在 runtime/init/ 下，向上四级才到包根：
%   .../simple_camera_pid/matlab/runtime/init/load_real_params_from_yaml.m
%   -> init -> runtime -> matlab -> simple_camera_pid
pkgRoot = fileparts(fileparts(fileparts(fileparts(mfilename("fullpath")))));
p = string(fullfile(pkgRoot, "config", "real", "real_line_follower.yaml"));
end


function map = readSharedSection(yamlPath)
%READSHAREDSECTION 取 `/**:` -> `ros__parameters:` 下缩进两级的 scalar 键值对。
% 遇到下一个顶格键（如 line_follower_node:）即停止。
map = containers.Map("KeyType", "char", "ValueType", "any");
lines = string(splitlines(fileread(yamlPath)));

inShared = false;
inParams = false;
for i = 1:numel(lines)
    line = lines(i);
    stripped = strtrim(line);
    if stripped == "" || startsWith(stripped, "#")
        continue;
    end

    indent = strlength(line) - strlength(strip(line, "left"));

    if indent == 0
        % 顶格键：进入或离开 /**: 段
        inShared = startsWith(stripped, "/**:");
        inParams = false;
        continue;
    end
    if ~inShared
        continue;
    end
    if indent == 2 && startsWith(stripped, "ros__parameters:")
        inParams = true;
        continue;
    end
    if ~inParams || indent ~= 4
        continue;
    end

    % 形如 `key: value   # 注释`
    tok = regexp(stripped, '^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$', 'tokens', 'once');
    if isempty(tok)
        continue;
    end
    key = tok{1};
    valueText = strtrim(regexprep(tok{2}, '\s*#.*$', ''));
    if valueText == ""
        continue;   % 纯嵌套键，本函数不处理
    end

    num = str2double(valueText);
    if ~isnan(num)
        map(char(key)) = num;
    elseif any(strcmpi(valueText, ["true", "false"]))
        map(char(key)) = strcmpi(valueText, "true");
    else
        map(char(key)) = char(valueText);   % 字符串（如 camera_profile: real）
    end
end
end
