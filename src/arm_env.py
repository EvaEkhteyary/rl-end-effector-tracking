"""Gymnasium environment: residual-RL end-effector trajectory tracking.

The environment wraps a MuJoCo Sawyer arm (Rethink Robotics) fitted with a
Robotiq-85 gripper, and frames the problem as *residual reinforcement
learning*:

    joint_velocity_command = base_controller(trajectory)  +  RL_residual(obs)

The model-based ``ResolvedRateController`` already does coarse tracking.  The
RL policy only has to output a small bounded correction.  This is what makes
the system (a) train fast, (b) stay smooth -- it can never command violent
motion -- and (c) put the *learning* exactly where it is needed: compensating
control delay, sensor noise, dynamics lag and singular / unreachable regions.

Uncertainty injected (all three requested kinds, randomised during training):
  * observation noise -- gaussian noise on the joint encoders / EE sensing
  * action noise      -- gaussian noise on the applied joint command
  * control delay     -- the command is buffered and applied N steps late
  * unreachable goals -- the trajectory is sometimes scaled beyond the
                         dexterous workspace
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .base_controller import ResolvedRateController
from .config import EnvConfig
from .math_utils import quat_angle, quat_error
from .trajectories import make_trajectory

_MODEL_PATH = Path(__file__).resolve().parent.parent / "assets" / "sawyer.xml"
_EE_SITE = "gripper0_grip_site"          # Robotiq-85 grasp centre
_TRAJECTORIES = ("circle", "figure8", "random")


class ArmTrackingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: EnvConfig = None, render_mode: str = None,
                 render_size=(720, 960)):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.render_mode = render_mode
        self._render_h, self._render_w = render_size

        # ---- MuJoCo model -------------------------------------------------
        self.model = mujoco.MjModel.from_xml_path(str(_MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self._kin = mujoco.MjData(self.model)        # scratch for FK / Jacobian / IK
        self.ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, _EE_SITE)
        self.home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")

        self.dt_ctrl = 1.0 / self.cfg.control_hz
        self.substeps = max(1, round((1.0 / self.model.opt.timestep) / self.cfg.control_hz))
        self.j_low = self.model.jnt_range[:7, 0].copy()
        self.j_high = self.model.jnt_range[:7, 1].copy()
        self.j_mid = 0.5 * (self.j_low + self.j_high)
        self.j_span = 0.5 * (self.j_high - self.j_low)

        self.controller = ResolvedRateController(self.cfg)

        # Anchor the trajectories on the home end-effector position so every
        # path is guaranteed to start inside the workspace.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        mujoco.mj_forward(self.model, self.data)
        self.traj_center = self.data.site_xpos[self.ee_site].copy()

        # ---- spaces -------------------------------------------------------
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        k = self.cfg.preview_count
        self.obs_dim = 40 + 6 * k if self.cfg.track_orientation else 37 + 3 * k
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(self.obs_dim,),
                                            dtype=np.float32)

        # ---- episode state ------------------------------------------------
        self.traj = None
        self.t = 0.0
        self.step_count = 0
        self.q_cmd = self.data.qpos[:7].copy()
        self.qdot_filt = np.zeros(7)
        self.prev_action = np.zeros(7)
        self.cmd_buffer = deque()
        self.delay_steps = self.cfg.delay_steps
        self.noise_scale = 1.0
        self._base_qdot = np.zeros(7)
        self._renderer = None
        self._trail = deque(maxlen=110)
        self._path_pts = np.zeros((0, 3))

    # ======================================================================
    # Gymnasium API
    # ======================================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}
        evaluating = options.get("eval", False)

        # -- sample the episode: trajectory + uncertainty -------------------
        if self.cfg.domain_rand and not evaluating:
            name = _TRAJECTORIES[rng.integers(3)]
            speed = rng.uniform(0.8, 1.4)
            # 22% of episodes use an over-sized path -> partly unreachable
            scale = rng.uniform(1.7, 2.5) if rng.random() < 0.22 else 1.0
            self.delay_steps = int(rng.integers(self.cfg.delay_range[0],
                                                self.cfg.delay_range[1] + 1))
            self.noise_scale = rng.uniform(*self.cfg.noise_scale_range)
            start_jitter = 0.05
        else:
            name = options.get("traj", "circle")
            speed = options.get("speed", 1.0)
            scale = options.get("scale", 1.0)
            self.delay_steps = int(options.get("delay", self.cfg.delay_steps))
            self.noise_scale = float(options.get("noise_scale", 1.0))
            start_jitter = 0.0

        self.traj = make_trajectory(name, rng, self.cfg, center=self.traj_center,
                                    scale=scale, speed=speed)
        self.t = 0.0
        self.step_count = 0

        # -- place the arm at the trajectory start via iterative IK ---------
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        p0, quat0 = self.traj.pose(0.0)
        q_start = self._inverse_kinematics(p0, quat0)
        if start_jitter:
            q_start = q_start + rng.normal(0.0, start_jitter, 7)
        q_start = np.clip(q_start, self.j_low, self.j_high)
        self.data.qpos[:7] = q_start
        self.data.qvel[:7] = 0.0
        self.data.ctrl[:7] = q_start
        mujoco.mj_forward(self.model, self.data)

        self.q_cmd = q_start.copy()
        self.qdot_filt = np.zeros(7)
        self.prev_action = np.zeros(7)
        self.cmd_buffer = deque([np.zeros(7) for _ in range(self.delay_steps)])
        self._trail.clear()
        if self.render_mode is not None:
            self._path_pts = self.traj.sample(np.linspace(0, self.traj.duration, 60))

        obs = self._observe(np.zeros(7))
        return obs, {"trajectory": name}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # -- compose command: base controller + bounded RL residual ---------
        residual = action * self.cfg.residual_vel_scale
        qdot = np.clip(self._base_qdot + residual,
                       -self.cfg.max_joint_vel, self.cfg.max_joint_vel)

        # -- uncertainty: action noise + control delay ----------------------
        if self.cfg.act_noise_std > 0:
            qdot = qdot + self.np_random.normal(
                0.0, self.cfg.act_noise_std * self.noise_scale, 7)
        self.cmd_buffer.append(qdot)
        applied = self.cmd_buffer.popleft()

        # -- low-pass filter the command: this is the hard smoothness ------
        #    guarantee -- noise / jitter is attenuated before the actuators.
        beta = self.cfg.cmd_filter_beta
        self.qdot_filt = beta * self.qdot_filt + (1.0 - beta) * applied

        # -- integrate the velocity command into a position setpoint --------
        self.q_cmd = np.clip(self.q_cmd + self.qdot_filt * self.dt_ctrl,
                             self.j_low, self.j_high)
        self.data.ctrl[:7] = self.q_cmd
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self.t += self.dt_ctrl
        self.step_count += 1

        # -- reward is computed from the TRUE state -------------------------
        ee_pos, ee_quat = self._site_pose(self.data)
        p_des, quat_des = self.traj.pose(self.t)
        pos_err = float(np.linalg.norm(p_des - ee_pos))
        ori_err = quat_angle(ee_quat, quat_des) if self.cfg.track_orientation else 0.0
        qd_true = self.data.qvel[:7].copy()
        reward, rew_terms = self._reward(pos_err, ori_err, action, qd_true)

        self._trail.append(ee_pos.copy())
        terminated = not np.all(np.isfinite(self.data.qpos))
        if terminated:
            reward -= 5.0
        truncated = self.step_count >= self.cfg.max_steps

        obs = self._observe(action)
        self.prev_action = action

        info = {
            "pos_err": pos_err,
            "ori_err_deg": np.degrees(ori_err),
            "ee_pos": ee_pos,
            "des_pos": p_des,
            "ee_quat": ee_quat,
            "des_quat": quat_des,
            "joint_vel": qd_true,
            "action": action.copy(),
            "delay_steps": self.delay_steps,
            **rew_terms,
        }
        return obs, reward, terminated, truncated, info

    # ======================================================================
    # Observation
    # ======================================================================
    def _observe(self, last_action):
        cfg = self.cfg
        ns = self.noise_scale

        # -- true joint state ----------------------------------------------
        q_true = self.data.qpos[:7].copy()
        qd_true = self.data.qvel[:7].copy()

        # -- noisy "sensor" reading the controller + policy operate on ------
        q_obs = q_true + self.np_random.normal(0.0, cfg.obs_noise_std * ns, 7)
        qd_obs = qd_true + self.np_random.normal(0.0, cfg.obs_noise_std * 4 * ns, 7)

        # The controller's kinematic view: FK + Jacobian from the *noisy*
        # joint angles -> a self-consistent but imperfect model of reality.
        self._kin.qpos[:7] = q_obs
        self._kin.qvel[:7] = 0.0
        mujoco.mj_forward(self.model, self._kin)
        ee_pos, ee_quat = self._site_pose(self._kin)
        ee_pos = ee_pos + self.np_random.normal(0.0, cfg.ee_obs_noise_std * ns, 3)
        jac_lin, jac_ang = self._jacobian(self._kin)

        # -- desired pose / twist at the current time ----------------------
        p_des, quat_des = self.traj.pose(self.t)
        v_des = self.traj.velocity(self.t)
        w_des = self.traj.angular_velocity(self.t)

        # -- base controller command (cached for the next step) ------------
        self._base_qdot = self.controller.compute(
            ee_pos, ee_quat, jac_lin, jac_ang, p_des, quat_des, v_des, w_des)

        # -- assemble the observation vector -------------------------------
        q_norm = (q_obs - self.j_mid) / self.j_span
        parts = [q_norm, qd_obs / 2.5, (p_des - ee_pos) * 10.0]
        if cfg.track_orientation:
            parts.append(quat_error(ee_quat, quat_des))

        # look-ahead preview: the agent literally sees the upcoming targets
        for kk in range(1, cfg.preview_count + 1):
            tk = self.t + kk * cfg.preview_dt
            parts.append((self.traj.position(tk) - ee_pos) * 5.0)
            if cfg.track_orientation:
                parts.append(quat_error(ee_quat, self.traj.orientation(tk)))

        parts += [v_des * 2.0, w_des,
                  self._base_qdot / self.cfg.max_joint_vel, last_action]
        obs = np.clip(np.concatenate(parts), -10.0, 10.0)
        return obs.astype(np.float32)

    # ======================================================================
    # Reward
    # ======================================================================
    def _reward(self, pos_err, ori_err, action, qd_true):
        cfg = self.cfg
        # Tracking: smooth, bounded gaussian kernels in [0, 1].
        r_pos = np.exp(-(pos_err / cfg.pos_sigma) ** 2)
        r_ori = np.exp(-(ori_err / cfg.ori_sigma) ** 2) if cfg.track_orientation else 0.0
        reward = cfg.w_pos * r_pos + cfg.w_ori * r_ori
        if pos_err < cfg.precision_thresh:
            reward += cfg.precision_bonus

        # Smoothness: penalise large residuals, residual *changes* (jerk) and
        # raw joint velocity.  This is what suppresses jitter.
        pen_action = cfg.w_action * float(np.sum(action ** 2))
        pen_rate = cfg.w_action_rate * float(np.sum((action - self.prev_action) ** 2))
        pen_jvel = cfg.w_jvel * float(np.sum(qd_true ** 2))
        reward -= pen_action + pen_rate + pen_jvel

        return reward, {
            "r_pos": float(r_pos),
            "r_ori": float(r_ori),
            "pen_smooth": float(pen_action + pen_rate + pen_jvel),
        }

    # ======================================================================
    # Kinematics helpers
    # ======================================================================
    def _site_pose(self, data):
        pos = data.site_xpos[self.ee_site].copy()
        quat_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(quat_wxyz, data.site_xmat[self.ee_site])
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return pos, quat_xyzw

    def _jacobian(self, data):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.ee_site)
        return jacp[:, :7], jacr[:, :7]

    def _inverse_kinematics(self, target_pos, target_quat, iters=140):
        """Damped-least-squares IK seeded from the current configuration."""
        q = self.data.qpos[:7].copy()
        for _ in range(iters):
            self._kin.qpos[:7] = q
            self._kin.qvel[:7] = 0.0
            mujoco.mj_forward(self.model, self._kin)
            ee_pos, ee_quat = self._site_pose(self._kin)
            e_pos = target_pos - ee_pos
            e_ori = quat_error(ee_quat, target_quat)
            if np.linalg.norm(e_pos) < 1e-4 and np.linalg.norm(e_ori) < 1e-3:
                break
            jac_lin, jac_ang = self._jacobian(self._kin)
            J = np.vstack([jac_lin, jac_ang])
            err = np.concatenate([e_pos, e_ori])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = np.clip(q + 0.5 * dq, self.j_low, self.j_high)
        return q

    # ======================================================================
    # Rendering
    # ======================================================================
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self._render_h,
                                             width=self._render_w, max_geom=2000)
            self._camera = mujoco.MjvCamera()
            self._camera.lookat = np.array([0.4, 0.0, 0.5])
            self._camera.distance = 2.05
            self._camera.azimuth = 141.0
            self._camera.elevation = -13.0
        self._renderer.update_scene(self.data, camera=self._camera)
        self._draw_overlays(self._renderer.scene)
        return self._renderer.render()

    def _draw_overlays(self, scene):
        ident = np.eye(3).flatten()

        def add(pos, size, rgba):
            if scene.ngeom >= scene.maxgeom:
                return
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([size, 0.0, 0.0]), np.asarray(pos, dtype=np.float64),
                ident, np.asarray(rgba, dtype=np.float32))
            scene.ngeom += 1

        for p in self._path_pts:                       # desired path (blue)
            add(p, 0.006, [0.25, 0.5, 1.0, 0.5])
        n = len(self._trail)
        for i, p in enumerate(self._trail):            # actual trail (green)
            add(p, 0.007, [0.15, 0.85, 0.3, 0.25 + 0.6 * i / max(1, n)])
        if self.traj is not None:                      # current target (red)
            add(self.traj.position(self.t), 0.022, [1.0, 0.25, 0.2, 0.55])

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(cfg: EnvConfig = None, render_mode=None, seed=None):
    """Picklable factory used to build (vectorised) environments."""
    def _init():
        env = ArmTrackingEnv(cfg=cfg, render_mode=render_mode)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init
