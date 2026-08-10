function [xt, yt, ok] = lf_memory_target(points, nValid, lookahead)
%LF_MEMORY_TARGET 从记忆的路径点里挑一个用于定向恢复的目标点。
%
%   [xt, yt, ok] = lf_memory_target(points, nValid, lookahead)
%
% points 是 lf_memory_propagate 推算到当前机体系的 (N×2) 点列，**按路径方向
% 由近及远排序**（滑窗从 ROI 底部向上爬，所以存进来时索引 1 最近）。
%
% 选点规则：沿路径方向取第一个到机体距离 >= lookahead 的点；若全都比
% lookahead 近，取路径方向上最远的那个（即最后一个有效点）。
%
% **按路径顺序而非按距离排序来选**：机器人转过身之后，按距离排会挑到路径上
% 反方向的点，把机器人往回引；按路径顺序选则始终朝原行进方向前进，这正是
% baseline"继续向前爬"能找回线、而原地扫描找不回的那个行为。
%
% ok=false 表示记忆里没有可用点，调用方应回退到别的恢复策略（原地扫描）。
%
% See also LF_MEMORY_PROPAGATE, LF_LINE_SEARCH.

xt = 0;
yt = 0;
ok = false;

n = min(nValid, size(points, 1));
if n <= 0
    return
end

for k = 1:n
    if hypot(points(k, 1), points(k, 2)) >= lookahead
        xt = points(k, 1);
        yt = points(k, 2);
        ok = true;
        return
    end
end

% 全部都比 lookahead 近：取路径方向最远的一个
xt = points(n, 1);
yt = points(n, 2);
ok = true;
end
