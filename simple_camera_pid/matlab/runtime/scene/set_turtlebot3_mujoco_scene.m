function sceneFile = set_turtlebot3_mujoco_scene(model, mapOrScene, options)
%SET_TURTLEBOT3_MUJOCO_SCENE Select the MJCF scene used by a model.
%
% Examples:
%   set_turtlebot3_mujoco_scene("visual_line_follower_with_debug", "ellipse")
%   set_turtlebot3_mujoco_scene("visual_line_follower_with_debug", ...
%       "D:\project\model\mujoco\turtlebot3\complex_camera_track_turtlebot3_burger_visual_scene.xml")
%   set_turtlebot3_mujoco_scene(model, "complex", "Save", true)  % also write .slx
%
% Valid built-in map keys:
%   simple, ellipse, complex, training
%
% By default the selection is applied in memory only (model workspace
% variables + the Plant xmlFile parameter) and the .slx is NOT saved. This
% avoids baking one map into the saved model, which would pin the default
% map for every later run. Pass "Save", true to persist the change to disk.

arguments
    model (1, 1) string
    mapOrScene (1, 1) string
    options.Save (1, 1) logical = false
end

if endsWith(model, ".slx")
    model = erase(model, ".slx");
end

modelFile = which(model + ".slx");
if strlength(modelFile) == 0
    modelFile = model + ".slx";
end

% 若模型未加载则临时打开，函数退出时自动关闭（onCleanup 保证）
wasLoaded = bdIsLoaded(model);
open_system(modelFile);
if ~wasLoaded
    cleanup = onCleanup(@() close_system(model, 0));
end
modelDir = string(fileparts(get_param(model, "FileName")));

% 判断传入的是文件路径还是地图键名，分别走不同的解析分支
if isfile(mapOrScene)
    % 直接场景文件：保存完整路径到 TurtleBot3MuJoCoScene
    [sceneFile, mapKey] = resolve_turtlebot3_mujoco_scene( ...
        model, modelDir, "", mapOrScene);
    sceneOverride = sceneFile;
else
    % 地图键名：保存 mapKey 到 TurtleBot3MuJoCoMap，不覆盖 Scene
    [sceneFile, mapKey] = resolve_turtlebot3_mujoco_scene( ...
        model, modelDir, mapOrScene, "");
    sceneOverride = "";
end

% 将选择持久化到模型工作区，确保下次 InitFcn 能还原正确场景
workspace = get_param(model, "ModelWorkspace");
assignin(workspace, "TurtleBot3MuJoCoMap", mapKey);
assignin(workspace, "TurtleBot3MuJoCoScene", sceneOverride);

% SID 16 是 MuJoCo Plant 块，更新其 xmlFile 参数（仅内存生效）
plant = Simulink.ID.getFullName(model + ":16");
set_param(plant, "xmlFile", sceneFile, "xmlFileRel", sceneFile);
% 默认不写回 .slx，避免把某张地图固化进模型、钉死后续运行的默认地图。
% 选择已持久化到模型工作区变量，InitFcn 下次仍能还原。
if options.Save
    save_system(model);
    fprintf("Set %s MuJoCo scene to %s (saved to model)\n", model, sceneFile);
else
    fprintf("Set %s MuJoCo scene to %s (in memory; not saved)\n", model, sceneFile);
end
end
