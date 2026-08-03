"""MuJoCo + vision + PID line-follower simulation loop.

This is the Python reimplementation of ``visual_line_follower_with_debug.slx``
(the plain PID baseline model, no RL residual): a Gymnasium ``Env`` that
steps MuJoCo physics, renders the onboard camera, runs the same vision +
control pipeline as the Simulink model, and applies the resulting wheel
velocities as actuator commands.

Unlike the Simulink model there is no block diagram to wire — the control
flow that used to live in ``add_local_path_to_visual_model.m`` and the
``.slx`` connections is expressed directly as this class's ``step()`` method.
Odometry, wheel command, and vision-debug logging (``mujoco_odom_position``,
``mujoco_wheel_command``, ``mujoco_vision_debug`` in the MATLAB
``Simulink.SimulationOutput``) are exposed via ``info`` dicts and the
``StepLog`` returned by :meth:`Turtlebot3LineFollowerEnv.step`, instead of
``To Workspace`` blocks.

Camera geometry (``turtlebot3_front_camera``, fovy=45.9857deg, 640x480) and
control period (``Ts_Control`` = 0.05 s = 20 Hz) match the values read
directly from the ``.slx`` via the MATLAB MCP tools; physics steps at the
scene XML's ``<option timestep>`` (0.002 s) between each control update.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ...common.config import ControllerConfig, CurveSpeedGovernorConfig, RobotConfig, VisionConfig
from ...common.control.line_follower_controller import LineFollowerController, WheelCommand
from ...common.vision import LocalPathResult, LineFollowerVision
from ..runtime import mujoco_scene

CAMERA_NAME = "turtlebot3_front_camera"
BODY_NAME = "turtlebot3"


@dataclass(eq=False)
class StepLog:
    """Per-step diagnostics, replacing the ``To Workspace`` logging blocks.

    ``eq=False`` because the default dataclass field-wise ``==`` breaks on
    the ``np.ndarray`` fields (elementwise comparison isn't a bool) — e.g.
    ``gymnasium.utils.env_checker.check_env``'s determinism check compares
    ``info`` dicts this way. Nothing in this codebase compares ``StepLog``
    instances for equality.
    """

    time: float
    odom_position: np.ndarray      # (3,) world [x, y, z] (mujoco_odom_position)
    odom_quaternion: np.ndarray    # (4,) world [w, x, y, z] (MuJoCo's odom_quaternion sensor)
    wheel_command: np.ndarray      # (2,) [left, right] rad/s (mujoco_wheel_command)
    vision: LocalPathResult        # mujoco_vision_debug (steering/lateral/heading/confidence/found)


class Turtlebot3LineFollowerEnv(gym.Env):
    """PID-only TurtleBot3 visual line follower (no RL residual).

    Not itself meant for RL training (there is no reward/observation space
    tailored to an agent) — it is the physics+vision+PID simulation core
    that :class:`sim.residual_env.ResidualLineFollowerEnv` wraps to add the
    SAC/PPO residual action, observation, reward, and done logic.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        repo_root: str | Path,
        map_key: str = "simple",
        scene_path: str | Path | None = None,
        robot: RobotConfig | None = None,
        controller: ControllerConfig | None = None,
        governor: CurveSpeedGovernorConfig | None = None,
        vision: VisionConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.repo_root = Path(repo_root)
        self.render_mode = render_mode
        self.robot = robot or RobotConfig()
        self.controller_cfg = controller or ControllerConfig()
        self.governor_cfg = governor or CurveSpeedGovernorConfig()
        self.vision_cfg = vision or VisionConfig()

        self.controller = LineFollowerController(self.robot, self.controller_cfg, self.governor_cfg)
        self.vision = LineFollowerVision(
            roi_bottom_fraction=self.vision_cfg.roi_bottom_fraction,
            waypoint_count=self.vision_cfg.num_points,
            min_brightness=self.vision_cfg.min_brightness,
            max_saturation=self.vision_cfg.max_saturation,
            min_pixels=self.vision_cfg.min_pixels,
            error_scale=self.vision_cfg.error_scale,
            lookahead_distance=self.vision_cfg.lookahead_distance,
            lateral_gain=self.vision_cfg.lateral_gain,
            heading_gain=self.vision_cfg.heading_gain,
            curvature_gain=self.vision_cfg.curvature_gain,
            roi_widen_step=self.vision_cfg.roi_widen_step,
            roi_widen_max=self.vision_cfg.roi_widen_max,
            flip_vertical=False,  # mujoco.Renderer.render() is already upright; see vision.py
        )

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._renderer: mujoco.Renderer | None = None
        self._camera_id: int = -1
        self._body_id: int = -1
        self._n_substeps: int = 1
        self.map_key = map_key
        self.map_display_name = ""
        self.scene_path: Path | None = None
        self._last_vision: LocalPathResult | None = None
        self._last_frame: np.ndarray | None = None

        self.load_scene(scene_path=scene_path, map_key=map_key)

        self.observation_space = spaces.Dict({
            "steering_error": spaces.Box(-1.5, 1.5, (1,), np.float32),
            "lateral_error": spaces.Box(-1.0, 1.0, (1,), np.float32),
            "heading_error": spaces.Box(-1.0, 1.0, (1,), np.float32),
            "confidence": spaces.Box(0.0, 1.0, (1,), np.float32),
            "found": spaces.Box(0.0, 1.0, (1,), np.float32),
        })
        self.action_space = spaces.Box(low=0, high=0, shape=(0,), dtype=np.float32)  # no external action

    # ------------------------------------------------------------------
    def load_scene(self, scene_path: str | Path | None = None, map_key: str | None = None) -> None:
        """(Re)load a MuJoCo scene, e.g. between RL episodes for map randomization."""
        if scene_path is not None:
            path = Path(scene_path)
            resolved_key = map_key or "custom"
            display_name = path.stem
        else:
            path, resolved_key, display_name = mujoco_scene.resolve_turtlebot3_mujoco_scene(
                self.repo_root, requested_map=map_key or self.map_key
            )
        self.scene_path = path
        self.map_key = resolved_key
        self.map_display_name = display_name

        if self._renderer is not None:
            # A stale mujoco.Renderer left bound to the previous self.model silently
            # renders black frames once self.model/self.data are replaced below — it
            # must be closed before constructing a new one for the new model.
            self._renderer.close()

        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(self.model, height=self.vision.image_height,
                                          width=self.vision.image_width)
        self._camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        self._body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
        self._n_substeps = max(1, round(self.controller_cfg.ts_control / self.model.opt.timestep))

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "scene_path" in options or "map_key" in options:
            self.load_scene(scene_path=options.get("scene_path"), map_key=options.get("map_key"))

        mujoco.mj_resetData(self.model, self.data)
        lateral_target = options.get("lateral_target", 0.0)
        if lateral_target:
            self.data.qpos[1] += lateral_target  # perturb the spawn body's y position
        mujoco.mj_forward(self.model, self.data)

        self.controller.reset()
        self.vision.reset()

        frame = self._render_camera_frame()
        self._last_frame = frame
        self._last_vision = self.vision.step(frame)
        observation = self._observation(self._last_vision)
        info = {"map_key": self.map_key, "map_display_name": self.map_display_name}
        return observation, info

    def step(self, action: float | Sequence[float] | np.ndarray | None = None):
        """Advance one control tick (``Ts_Control`` = 0.05 s of physics).

        Uses the vision result from the frame captured at the *start* of
        this tick (from ``reset()`` or the previous ``step()``) to drive the
        PID controller, then renders a fresh frame after physics has
        advanced for the next call — one control-period sample-and-hold
        between camera and controller, matching the Simulink model's
        discrete signal flow (the ``Camera_Local_Path_Generator`` output
        feeds the ``PID_Controller`` at the same ``Ts_Control`` tick it was
        produced on, one 0.05 s frame behind the physics it observes).

        ``action``, if given, is a SAC/PPO residual action summed with the PID
        (v, omega) command before wheel mixing -- see
        :meth:`control.line_follower_controller.LineFollowerController.step`.
        A scalar (or 1-element array) is the older 1-D-action convention,
        ``delta_omega`` (rad/s) only. A 2-element array/sequence is the
        2026-07-21+ 2-D-action convention, ``[delta_v, delta_omega]``
        (velocity-units / rad/s). Plain PID baseline usage leaves this
        ``None`` (equivalent to zero for both).
        """
        if action is None:
            residual_delta_v, residual_delta_omega = 0.0, 0.0
        else:
            flat = np.asarray(action, dtype=np.float64).reshape(-1)
            if flat.size >= 2:
                residual_delta_v, residual_delta_omega = float(flat[0]), float(flat[1])
            else:
                residual_delta_v, residual_delta_omega = 0.0, float(flat[0])
        wheel_cmd = self.controller.step(
            steering_error=self._last_vision.steering_error,
            found=self._last_vision.found,
            heading_error=self._last_vision.heading_error,
            lateral_error=self._last_vision.lateral_error,
            residual_delta_omega=residual_delta_omega,
            residual_delta_v=residual_delta_v,
        )
        self._apply_wheel_command(wheel_cmd)
        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        frame = self._render_camera_frame()
        self._last_frame = frame
        self._last_vision = self.vision.step(frame)
        observation = self._observation(self._last_vision)

        log = StepLog(
            time=float(self.data.time),
            odom_position=self.data.xpos[self._body_id].copy(),
            odom_quaternion=self.data.xquat[self._body_id].copy(),
            wheel_command=np.array([wheel_cmd.left, wheel_cmd.right]),
            vision=self._last_vision,
        )
        info = {"log": log}
        terminated = False
        truncated = False
        reward = 0.0
        return observation, reward, terminated, truncated, info

    @property
    def last_vision(self) -> LocalPathResult:
        """The vision result produced by the most recent ``reset()``/``step()``."""
        return self._last_vision

    @property
    def last_frame(self) -> np.ndarray:
        """The (image_height, image_width, 3) uint8 camera frame ``last_vision`` was computed from."""
        return self._last_frame

    # ------------------------------------------------------------------
    def _apply_wheel_command(self, cmd: WheelCommand) -> None:
        self.data.ctrl[0] = cmd.left
        self.data.ctrl[1] = cmd.right

    def _render_camera_frame(self) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=self._camera_id)
        return self._renderer.render()  # (H, W, 3) uint8, MuJoCo's native (upright) orientation

    def _observation(self, vision_result: LocalPathResult) -> dict[str, np.ndarray]:
        return {
            "steering_error": np.array([vision_result.steering_error], dtype=np.float32),
            "lateral_error": np.array([vision_result.lateral_error], dtype=np.float32),
            "heading_error": np.array([vision_result.heading_error], dtype=np.float32),
            "confidence": np.array([vision_result.confidence], dtype=np.float32),
            "found": np.array([1.0 if vision_result.found else 0.0], dtype=np.float32),
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_camera_frame()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
