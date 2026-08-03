"""Gymnasium environment replacing rlSimulinkEnv + the Simulink closed loop.

Two distinct pipelines share this class, selected by observe_wheel_speeds —
matching TWO DIFFERENT, INDEPENDENTLY-MAINTAINED MATLAB models (see
controller.py's module docstring for the full kinematics-side explanation):

  observe_wheel_speeds=False — lyapunov model (archived, untouched):
    Observation (5): [steering_error, lateral_error, heading_error, found,
                      prev_action]
    Action      (1): delta_omega in ±max_delta_omega rad/s, mixed with the
                     PID's ALREADY-saturated wheel-space output via the old
                     controller.mix_residual() (own YawToWheel gain, own
                     ±[1,-1] sign split).

  observe_wheel_speeds=True — residual model (ACTIVE line, refactored
  2026-07-21 to extract the shared Diff_Drive_Kinematics conversion):
    Observation (8): [steering_error, lateral_error, heading_error, found,
                      prev_delta_v, prev_delta_omega, v_norm, omega_norm]
      v_norm     = (w_left + w_right) / (2*MaxWheelSpeed)
      omega_norm = (w_right - w_left) / (2*MaxWheelSpeed)
      using the MEASURED jointvel sensors and the (possibly DR-perturbed)
      MaxWheelSpeed, exactly like the model-workspace gain 1/(2*MaxWheelSpeed).
    Action      (2): [delta_v, delta_omega] — delta_v in ±max_delta_v (same
                     units as the PID's own v command, BEFORE the
                     BaseSpeedScale/WheelRadius gain), delta_omega in
                     ±max_delta_omega rad/s (yaw-rate residual, BEFORE the
                     WheelSeparation/(2*WheelRadius) gain). Both are summed
                     with the PID's (v_pid, omega_pid) in velocity space and
                     converted to wheel speeds ONCE via
                     controller.mix_residual_velocity() — there is only one
                     sign convention for the omega term now, shared by PID
                     and the residual (previously the residual model's own
                     wheel-space mixing used the OPPOSITE sign for omega from
                     the PID's, an unnoticed bug fixed the same day the action
                     space grew to 2-D; see controller.py).

One env step = one control tick (Ts_Control = 0.05 s):
  1. PID command from the CURRENT vision sample (same sample the agent's
     observation was built from — matches Simulink's sorted execution order),
  2. mix in the SAC residual action (pipeline-dependent, see above), clamp to
     ±MaxWheelSpeed exactly once,
  3. hold the command for Ts_Control/timestep physics substeps (ZOH, exactly
     like the Command_Rate_Transition into the MuJoCo Plant),
  4. render the front camera, run the line detector,
  5. reward (Lyapunov or Quadratic, whichever reward_fn was passed in) +
     consecutive-lost-line termination.

Rendering the camera requires a working MuJoCo offscreen GL context
(MUJOCO_GL=egl on the GPU nodes; the login node cannot render).

mirror_camera (default True): the Simulink MuJoCo blockset delivered the
camera image horizontally mirrored relative to the true view (raw OpenGL
buffer path), and the whole control loop — SteeringSign=1, wheel mixing —
was tuned against that mirrored image. Verified empirically: with the
mirrored image the PID laps the training oval (16.1 s, 100% line found);
with the upright image it steers away from the line and dies in ~7 s.
Keep True to reproduce the MATLAB system. (The residual model uses the
UPRIGHT image with SteeringSign=-1 instead — pass mirror_camera=False.)
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from .controller import ControllerParams, LineFollowController
from .domain_rand import DRConfig, sample_episode
from .reward import DoneDetector, LyapunovReward, RewardParams
from .scenes import resolve_scene
from .vision import IMAGE_HEIGHT, IMAGE_WIDTH, LocalPathGenerator, VisionParams

TS_CONTROL = 0.05
CAMERA_NAME = "turtlebot3_front_camera"
FREEJOINT_NAME = "turtlebot3_freejoint"
LEFT_ACTUATOR = "left_wheel_velocity_cmd"
RIGHT_ACTUATOR = "right_wheel_velocity_cmd"
ODOM_LINVEL_SENSOR = "odom_linear_velocity"
LEFT_WHEEL_VEL_SENSOR = "left_wheel_velocity"
RIGHT_WHEEL_VEL_SENSOR = "right_wheel_velocity"


class LineFollowerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        map_key: str = "training",
        max_delta_omega: float = 1.0,       # SAC_MaxDeltaOmega
        max_delta_v: float = 1.0,           # SAC_MaxDeltaV (residual model only)
        done_steps: int = 100,              # SAC_Done_Steps
        controller_params: ControllerParams | None = None,
        vision_params: VisionParams | None = None,
        reward_params: RewardParams | None = None,
        reward_fn=None,                     # custom reward object (reset/step);
                                            # default LyapunovReward(reward_params)
        domain_rand: DRConfig | None = None,  # None = randomization OFF
        mirror_camera: bool = True,
        observe_wheel_speeds: bool = False,  # True = residual model (8-D obs,
                                             # 2-D action); False = lyapunov
        cam_pitch_deg: float | None = None,  # None = keep the scene XML's
                                             # baked-in 15 deg camera tilt
                                             # untouched; otherwise overrides
                                             # model.cam_quat at scene-load
                                             # time (see _apply_cam_pitch).
                                             # Caller must pass the SAME value
                                             # to VisionParams.cam_pitch_deg
                                             # or the IPM ground-projection
                                             # will desync from the render.
        seed: int | None = None,
    ):
        super().__init__()
        self.base_map_key = map_key
        self.max_delta_omega = float(max_delta_omega)
        self.max_delta_v = float(max_delta_v)
        self.base_ctrl = controller_params or ControllerParams()
        self.base_vision = vision_params or VisionParams()
        self.reward_params = reward_params or RewardParams()
        self.dr = domain_rand
        self.mirror_camera = mirror_camera
        self.observe_wheel_speeds = observe_wheel_speeds
        self.cam_pitch_deg = cam_pitch_deg
        self.rng = np.random.default_rng(seed)

        # rl_io_specs.m: 5-D (lyapunov, unchanged) / 8-D (residual model,
        # 2026-07-21 — prev_delta_v/prev_delta_omega replaced the old scalar
        # prev_action slot when the action grew from 1-D to 2-D)
        if observe_wheel_speeds:
            self.observation_space = gym.spaces.Box(
                low=np.array([-1.5, -1.0, -1.0, 0.0,
                             -self.max_delta_v, -self.max_delta_omega,
                             -1.0, -1.0], dtype=np.float32),
                high=np.array([1.5, 1.0, 1.0, 1.0,
                              self.max_delta_v, self.max_delta_omega,
                              1.0, 1.0], dtype=np.float32),
            )
            self.action_space = gym.spaces.Box(
                low=np.array([-self.max_delta_v, -self.max_delta_omega],
                             dtype=np.float32),
                high=np.array([self.max_delta_v, self.max_delta_omega],
                              dtype=np.float32),
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=np.array([-1.5, -1.0, -1.0, 0.0, -1.0], dtype=np.float32),
                high=np.array([1.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            )
            self.action_space = gym.spaces.Box(
                low=-self.max_delta_omega, high=self.max_delta_omega,
                shape=(1,), dtype=np.float32,
            )

        self.controller = LineFollowController(self.base_ctrl)
        self.vision = LocalPathGenerator(self.base_vision)
        self.reward_fn = reward_fn or LyapunovReward(self.reward_params)
        self.done_fn = DoneDetector(done_steps)

        self._model_cache: dict[str, mujoco.MjModel] = {}
        self._renderer: mujoco.Renderer | None = None
        self._renderer_model_path: str | None = None
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._scene_path: str | None = None
        self._substeps = 0
        self._prev_action = 0.0            # lyapunov pipeline (scalar)
        self._prev_delta_v = 0.0           # residual pipeline
        self._prev_delta_omega = 0.0       # residual pipeline
        self._last_vision = None
        self._last_rgb = None

        self._load_scene(resolve_scene(map_key))

    # ── scene / model management ────────────────────────────────────────────
    def _load_scene(self, scene_path: str, force_reload: bool = False) -> None:
        """force_reload=True bypasses the path-keyed cache: GenTrack reuses
        the SAME scene_path every episode but overwrites its file content
        (fresh geometry), so caching by path alone would silently keep
        serving the first episode's stale MjModel/renderer forever."""
        reload_model = force_reload or scene_path not in self._model_cache
        if reload_model:
            model = mujoco.MjModel.from_xml_path(scene_path)
            self._apply_cam_pitch(model)
            self._model_cache[scene_path] = model
        self.model = self._model_cache[scene_path]
        self.data = mujoco.MjData(self.model)
        self._scene_path = scene_path
        self._substeps = max(1, round(TS_CONTROL / self.model.opt.timestep))
        self._i_left = self.model.actuator(LEFT_ACTUATOR).id
        self._i_right = self.model.actuator(RIGHT_ACTUATOR).id
        # Renderer must be rebuilt whenever the MODEL OBJECT changes (not just
        # when the path string changes) — a mujoco.Renderer is bound to the
        # specific MjModel instance it was constructed with.
        if self._renderer is None or reload_model:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=IMAGE_HEIGHT,
                                             width=IMAGE_WIDTH)
            self._renderer_model_path = scene_path

    def _apply_cam_pitch(self, model: mujoco.MjModel) -> None:
        """Override the front camera's baked-in orientation to cam_pitch_deg.

        Reproduces the same xyaxes convention as the 15 deg mount baked into
        turtlebot3_burger_vehicle_body.xml (x=(0,-1,0), y=(sin ph,0,cos ph),
        z=x cross y) so cam_pitch_deg=15.0 exactly matches the untouched XML;
        avoids duplicating that shared body file across every map/gen-track
        scene just to sweep the tilt angle.
        """
        if self.cam_pitch_deg is None:
            return
        cam_id = model.camera(CAMERA_NAME).id
        ph = math.radians(self.cam_pitch_deg)
        s, c = math.sin(ph), math.cos(ph)
        mat = np.array([0.0, s, -c,
                        -1.0, 0.0, 0.0,
                        0.0, c, s])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        model.cam_quat[cam_id] = quat

    def _render_camera(self) -> np.ndarray:
        assert self._renderer is not None
        self._renderer.update_scene(self.data, camera=CAMERA_NAME)
        rgb = self._renderer.render()
        if self.mirror_camera:
            rgb = rgb[:, ::-1]
        self._last_rgb = rgb
        return rgb

    def _apply_lateral_offset(self, lateral: float) -> None:
        """Shift the spawn pose sideways in the robot frame (Pose group).

        (In Simulink this was MJ_TargetLateralPosition; here it is applied as a
        true spawn offset perpendicular to the initial heading.)
        """
        if lateral == 0.0:
            return
        adr = self.model.joint(FREEJOINT_NAME).qposadr[0]
        quat = self.data.qpos[adr + 3:adr + 7].copy()
        offset_local = np.array([0.0, lateral, 0.0])
        offset_world = np.zeros(3)
        mujoco.mju_rotVecQuat(offset_world, offset_local, quat)
        self.data.qpos[adr:adr + 3] += offset_world

    # ── gym API ──────────────────────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # per-episode domain randomization (env.ResetFcn equivalent)
        lateral = 0.0
        if self.dr is not None:
            ep = sample_episode(self.rng, self.dr, self.base_ctrl, self.base_vision)
            if ep.map_key is not None:
                self._load_scene(resolve_scene(ep.map_key),
                                 force_reload=ep.is_generated)
            lateral = ep.lateral_offset
            self.controller.params = ep.controller
            self.vision.params = ep.vision
        else:
            self.controller.params = self.base_ctrl
            self.vision.params = self.base_vision

        mujoco.mj_resetData(self.model, self.data)
        self._apply_lateral_offset(lateral)
        mujoco.mj_forward(self.model, self.data)

        self.controller.reset()
        self.vision.reset()
        self.reward_fn.reset()
        self.done_fn.reset()
        self._prev_action = 0.0
        self._prev_delta_v = 0.0
        self._prev_delta_omega = 0.0

        rgb = self._render_camera()
        vis = self.vision.process(rgb)
        self._last_vision = vis

        # Prime reward/done exactly like the k=0 Simulink sample: the reward
        # block runs once (gate=0 masks the bogus first Vdot, V_prev := V_0)
        # and the lost-line counter sees found_0. The k=0 reward value itself
        # never enters an experience tuple, so it is discarded.
        self.reward_fn.step(vis.steering_error, vis.lateral_error,
                            vis.heading_error, self._v_lat(vis.heading_error),
                            u=0.0, found=vis.found)
        self.done_fn.step(vis.found)

        return self._obs(vis), {"scene": self._scene_path}

    def step(self, action: np.ndarray):
        vis = self._last_vision

        if self.observe_wheel_speeds:
            # residual model: 2-D action [delta_v, delta_omega], mixed with
            # the PID's (v, omega) in velocity space before the ONE shared
            # Diff_Drive_Kinematics conversion (see controller.py).
            act = np.asarray(action).reshape(-1)
            delta_v = float(np.clip(act[0], -self.max_delta_v, self.max_delta_v))
            delta_omega = float(np.clip(act[1], -self.max_delta_omega,
                                        self.max_delta_omega))
            v_pid, omega_pid = self.controller.pid_velocity_command(
                vis.steering_error, vis.found)
            wheels = self.controller.mix_residual_velocity(
                v_pid, omega_pid, delta_v, delta_omega)
            action_u = (delta_v, delta_omega)
        else:
            # lyapunov model: 1-D action delta_omega, mixed in wheel space
            # (PID converts + saturates itself here — unchanged).
            delta_omega = float(np.clip(action, -self.max_delta_omega,
                                        self.max_delta_omega)[0])
            pid = self.controller.pid_wheels(vis.steering_error, vis.found)
            wheels = self.controller.mix_residual(pid, delta_omega)
            action_u = delta_omega

        self.data.ctrl[self._i_left] = wheels[0]
        self.data.ctrl[self._i_right] = wheels[1]
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)

        rgb = self._render_camera()
        vis = self.vision.process(rgb)
        self._last_vision = vis
        v_lat = self._v_lat(vis.heading_error)

        reward = self.reward_fn.step(vis.steering_error, vis.lateral_error,
                                     vis.heading_error, v_lat,
                                     u=action_u, found=vis.found)
        terminated = self.done_fn.step(vis.found)
        if terminated:  # lost-line done must never beat finishing the episode
            reward -= getattr(self.reward_fn.params, "p_done", 0.0)

        if self.observe_wheel_speeds:
            self._prev_delta_v, self._prev_delta_omega = action_u
        else:
            self._prev_action = action_u

        info = {
            "found": vis.found,
            "confidence": vis.confidence,
            "v_lat": v_lat,
            "wheels": wheels,
            "position": self.body_position(),
        }
        return self._obs(vis), float(reward), bool(terminated), False, info

    def render(self):
        return self._last_rgb

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ── helpers ──────────────────────────────────────────────────────────────
    def _obs(self, vis) -> np.ndarray:
        if self.observe_wheel_speeds:
            base = [vis.steering_error, vis.lateral_error, vis.heading_error,
                    vis.found, self._prev_delta_v, self._prev_delta_omega]
            base += list(self._wheel_speed_norm())
        else:
            base = [vis.steering_error, vis.lateral_error, vis.heading_error,
                    vis.found, self._prev_action]
        return np.array(base, dtype=np.float32)

    def _wheel_speed_norm(self) -> tuple[float, float]:
        """Wheel_Speed_Feedback subsystem: measured jointvel normalized by the
        (DR-perturbed) MaxWheelSpeed — v_norm=(wL+wR)/(2M), omega=(wR-wL)/(2M)."""
        w_l = float(self.data.sensor(LEFT_WHEEL_VEL_SENSOR).data[0])
        w_r = float(self.data.sensor(RIGHT_WHEEL_VEL_SENSOR).data[0])
        m = 2.0 * self.controller.params.max_wheel_speed
        return (w_l + w_r) / m, (w_r - w_l) / m

    def _v_lat(self, heading_error: float) -> float:
        """v_lat = hypot(vx, vy) * sin(e_head)  (Ground_Speed × Sin_Heading)."""
        vel = self.data.sensor(ODOM_LINVEL_SENSOR).data
        return float(np.hypot(vel[0], vel[1]) * np.sin(heading_error))

    def body_position(self) -> np.ndarray:
        return self.data.sensor("odom_position").data.copy()
