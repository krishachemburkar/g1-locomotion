"""G1 velocity-tracking walking environment (MuJoCo + Gymnasium).

The recipe follows Unitree's own G1 setup in unitree_rl_gym (legged_gym style):

- policy runs at 50 Hz, physics at 200 Hz (decimation 4),
- action = 12 leg joint offsets; target = default_pose + 0.25 * action,
  turned into torques by a PD loop at every physics step,
- a 0.8 s gait clock (left/right half a period apart) is part of the
  observation and of the contact reward, which gives a clean alternating gait,
- the observation uses only what the real robot measures (IMU + joint
  encoders + command + clock), so the policy can later run on hardware.

Command = (vx, vy, yaw_rate) in the robot's heading frame, resampled
during an episode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets" / "unitree_g1" / "g1_loco.xml"

LEG_JOINTS = [
    f"{side}_{name}_joint"
    for side in ("left", "right")
    for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]


@dataclass
class Config:
    sim_dt: float = 0.005
    decimation: int = 4
    episode_seconds: float = 20.0
    action_scale: float = 0.25

    # Per-joint defaults, order: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    default_pose: tuple = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0)
    kp: tuple = (100.0, 100.0, 100.0, 150.0, 40.0, 40.0)
    kd: tuple = (2.0, 2.0, 2.0, 4.0, 2.0, 2.0)

    gait_period: float = 0.8
    stance_fraction: float = 0.55
    base_height_target: float = 0.78
    swing_height_peak: float = 0.08  # sole height at mid-swing
    swing_height_sigma: float = 0.0025
    air_time_target: float = 0.25  # s; touchdowns after a longer swing are rewarded

    # Command ranges and resampling
    cmd_vx: tuple = (-0.5, 1.0)
    cmd_vy: tuple = (-0.4, 0.4)
    cmd_yaw: tuple = (-0.8, 0.8)
    cmd_resample_seconds: float = 10.0
    standing_prob: float = 0.1

    # Domain randomization
    randomize: bool = True
    friction_range: tuple = (0.3, 1.25)
    added_mass_range: tuple = (-1.0, 3.0)  # kg on the torso
    push_interval_seconds: float = 5.0
    push_max_vel: float = 0.5
    obs_noise: bool = True

    # Reward weights (multiplied by policy dt, as in legged_gym)
    reward_scales: dict = field(default_factory=lambda: {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.5,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "orientation": -1.0,
        "base_height": -10.0,
        "dof_acc": -2.5e-7,
        "dof_vel": -1e-3,
        "action_rate": -0.01,
        "torques": -1e-5,
        "dof_pos_limits": -5.0,
        "alive": 0.15,
        "hip_pos": -1.0,
        "contact_no_vel": -0.2,
        # Gait terms. Feet glued to the floor must not score here, otherwise
        # standing still is a comfortable local optimum (seen in run g1_walk).
        "feet_phase": 1.0,      # swing foot follows a lift-and-place height curve
        "contact": 0.3,         # +1 per leg when contact matches the clock, -1 when not
        "feet_air_time": 1.0,   # reward real steps when asked to move
        "feet_distance": -1.0,
    })
    tracking_sigma: float = 0.25


def _yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class G1WalkEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, cfg: Config | None = None, render_mode: str | None = None, seed: int | None = None):
        self.cfg = cfg or Config()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.model.opt.timestep = self.cfg.sim_dt
        self.data = mujoco.MjData(self.model)
        self.dt = self.cfg.sim_dt * self.cfg.decimation
        self.max_steps = int(self.cfg.episode_seconds / self.dt)
        self.rng = np.random.default_rng(seed)

        m = self.model
        self.qpos_idx = np.array([m.jnt_qposadr[m.joint(j).id] for j in LEG_JOINTS])
        self.qvel_idx = np.array([m.jnt_dofadr[m.joint(j).id] for j in LEG_JOINTS])
        self.act_idx = np.array([m.actuator(j).id for j in LEG_JOINTS])
        self.default_q = np.array(self.cfg.default_pose * 2)
        self.kp = np.array(self.cfg.kp * 2)
        self.kd = np.array(self.cfg.kd * 2)
        self.torque_limit = np.array([m.jnt_actfrcrange[m.joint(j).id][1] for j in LEG_JOINTS])
        lo, hi = m.jnt_range[[m.joint(j).id for j in LEG_JOINTS]].T
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        self.soft_lo, self.soft_hi = mid - 0.9 * half, mid + 0.9 * half
        self.hip_roll_yaw = np.array([1, 2, 7, 8])

        self.foot_sites = [m.site("left_foot").id, m.site("right_foot").id]
        self.floor_geom = m.geom("floor").id
        self.torso_body = m.body("torso_link").id
        self.base_torso_mass = m.body_mass[self.torso_body]
        self.foot_geoms = [
            np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{s}_ankle_roll_link").id and m.geom_contype[g]])
            for s in ("left", "right")
        ]

        # Standing height: put feet on the floor in the default pose.
        mujoco.mj_resetData(m, self.data)
        self.data.qpos[self.qpos_idx] = self.default_q
        mujoco.mj_forward(m, self.data)
        sole = min(self.data.geom_xpos[g][2] for gs in self.foot_geoms for g in gs) - 0.005
        self.init_height = self.data.qpos[2] - sole + 0.002

        obs_dim = 3 + 3 + 3 + 12 + 12 + 12 + 2
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)  # rescaled below
        self.action_clip = 5.0  # policy outputs are scaled by 5 so +-1 maps to +-1.25 rad

        self._rot9 = np.zeros(9)
        self._renderer = None
        self.command = np.zeros(3)
        self.fixed_command: np.ndarray | None = None

    # ------------------------------------------------------------------ helpers
    def set_command(self, vx: float, vy: float, yaw: float) -> None:
        """Pin the command (for evaluation / teleop). Pass through reset as well."""
        self.fixed_command = np.array([vx, vy, yaw], dtype=float)
        self.command = self.fixed_command.copy()

    def _sample_command(self) -> None:
        if self.fixed_command is not None:
            self.command = self.fixed_command.copy()
            return
        c = self.cfg
        if self.rng.random() < c.standing_prob:
            self.command = np.zeros(3)
            return
        self.command = np.array([
            self.rng.uniform(*c.cmd_vx), self.rng.uniform(*c.cmd_vy), self.rng.uniform(*c.cmd_yaw)
        ])
        if np.linalg.norm(self.command[:2]) < 0.2:
            self.command[:2] = 0.0

    def _phase(self) -> float:
        return (self.step_count * self.dt % self.cfg.gait_period) / self.cfg.gait_period

    def _base_state(self):
        """Pelvis linear velocity, angular velocity and gravity direction, all in the body frame."""
        d = self.data
        mujoco.mju_quat2Mat(self._rot9, d.qpos[3:7])
        rot = self._rot9.reshape(3, 3)
        lin_vel = rot.T @ d.qvel[0:3]
        ang_vel = d.qvel[3:6].copy()  # free joint angular velocity is already in the body frame
        gravity = -rot[2, :]
        return d.qpos[3:7], lin_vel, ang_vel, gravity

    def _feet(self):
        pos = self.data.site_xpos[self.foot_sites].copy()
        z = self.data.geom_xpos[:, 2]
        contact = np.array([z[gs].min() < 0.005 + 0.004 for gs in self.foot_geoms])
        return pos, contact

    def _obs(self) -> np.ndarray:
        _, _, ang_vel, gravity = self._base_state()
        q = self.data.qpos[self.qpos_idx] - self.default_q
        dq = self.data.qvel[self.qvel_idx]
        if self.cfg.obs_noise:
            ang_vel = ang_vel + self.rng.uniform(-0.2, 0.2, 3)
            gravity = gravity + self.rng.uniform(-0.05, 0.05, 3)
            q = q + self.rng.uniform(-0.01, 0.01, 12)
            dq = dq + self.rng.uniform(-1.5, 1.5, 12)
        phase = self._phase()
        obs = np.concatenate([
            ang_vel * 0.25,
            gravity,
            self.command * np.array([2.0, 2.0, 0.25]),
            q,
            dq * 0.05,
            self.last_action,
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
        ])
        return np.clip(obs, -100, 100).astype(np.float32)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = self.np_random
        m, d, c = self.model, self.data, self.cfg
        mujoco.mj_resetData(m, d)

        if c.randomize:
            m.geom_friction[self.floor_geom, 0] = self.rng.uniform(*c.friction_range)
            m.body_mass[self.torso_body] = self.base_torso_mass + self.rng.uniform(*c.added_mass_range)

        yaw = self.rng.uniform(-np.pi, np.pi)
        d.qpos[2] = self.init_height
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[self.qpos_idx] = self.default_q + self.rng.uniform(-0.05, 0.05, 12)
        d.qvel[0:2] = self.rng.uniform(-0.1, 0.1, 2)
        mujoco.mj_forward(m, d)

        self.step_count = 0
        self.action = np.zeros(12)
        self.last_action = np.zeros(12)
        self.last_dq = np.zeros(12)
        self.last_feet_pos, _ = self._feet()
        self.air_time = np.zeros(2)
        self.episode_terms = {k: 0.0 for k in c.reward_scales}
        self.ep_vel_err = 0.0
        self._sample_command()
        return self._obs(), {}

    def step(self, action):
        m, d, c = self.model, self.data, self.cfg
        self.action = np.clip(np.asarray(action, dtype=float) * self.action_clip, -10, 10)
        target = self.default_q + c.action_scale * self.action

        torques_sum = np.zeros(12)
        for _ in range(c.decimation):
            q = d.qpos[self.qpos_idx]
            dq = d.qvel[self.qvel_idx]
            tau = np.clip(self.kp * (target - q) - self.kd * dq, -self.torque_limit, self.torque_limit)
            d.ctrl[self.act_idx] = tau
            torques_sum += tau * tau
            mujoco.mj_step(m, d)
        self.step_count += 1

        if c.randomize and self.step_count % int(c.push_interval_seconds / self.dt) == 0:
            d.qvel[0:2] += self.rng.uniform(-c.push_max_vel, c.push_max_vel, 2)

        if self.step_count % int(c.cmd_resample_seconds / self.dt) == 0:
            self._sample_command()

        base = self._base_state()
        reward, terms = self._reward(torques_sum / c.decimation, base)
        for k, v in terms.items():
            self.episode_terms[k] += v

        gravity = base[3]
        terminated = bool(d.qpos[2] < 0.4 or abs(gravity[0]) > 0.84 or abs(gravity[1]) > 0.7)
        truncated = self.step_count >= self.max_steps

        self.last_action = self.action.copy()
        self.last_dq = d.qvel[self.qvel_idx].copy()
        self.ep_vel_err += float(np.linalg.norm(base[1][:2] - self.command[:2]))
        info = {}
        if terminated or truncated:
            info["episode_terms"] = dict(self.episode_terms)
            info["fell"] = terminated
            info["vel_error"] = self.ep_vel_err / self.step_count  # mean m/s off the commanded velocity
        return self._obs(), float(reward), terminated, truncated, info

    def _reward(self, torques_sq: np.ndarray, base):
        d, c = self.data, self.cfg
        _, lin_vel, ang_vel, gravity = base
        q = d.qpos[self.qpos_idx]
        dq = d.qvel[self.qvel_idx]
        feet_pos, contact = self._feet()
        feet_vel = (feet_pos - self.last_feet_pos) / self.dt
        self.last_feet_pos = feet_pos

        phase = self._phase()
        leg_phase = np.array([phase, (phase + 0.5) % 1.0])
        should_stance = leg_phase < c.stance_fraction
        swing_progress = np.clip((leg_phase - c.stance_fraction) / (1 - c.stance_fraction), 0, 1)
        swing_target = c.swing_height_peak * np.sin(np.pi * swing_progress)
        feet_phase = np.exp(-((feet_pos[:, 2] - swing_target) ** 2) / c.swing_height_sigma) * ~should_stance

        # Air time, rewarded at touchdown, only when the command asks to move.
        self.air_time += self.dt
        touchdown = contact & (self.air_time > self.dt)
        moving = np.linalg.norm(self.command[:2]) > 0.1 or abs(self.command[2]) > 0.1
        air_time_rew = np.sum((np.minimum(self.air_time, 0.5) - c.air_time_target) * touchdown) * moving
        self.air_time[contact] = 0.0

        # Lateral foot spacing in the heading frame; keeps the legs from crossing.
        yaw = _yaw_from_quat(d.qpos[3:7])
        rel = feet_pos[0, :2] - feet_pos[1, :2]
        lateral = -np.sin(yaw) * rel[0] + np.cos(yaw) * rel[1]

        raw = {
            "tracking_lin_vel": np.exp(-np.sum((self.command[:2] - lin_vel[:2]) ** 2) / c.tracking_sigma),
            "tracking_ang_vel": np.exp(-((self.command[2] - ang_vel[2]) ** 2) / c.tracking_sigma),
            "lin_vel_z": lin_vel[2] ** 2,
            "ang_vel_xy": np.sum(ang_vel[:2] ** 2),
            "orientation": np.sum(gravity[:2] ** 2),
            "base_height": (d.qpos[2] - c.base_height_target) ** 2,
            "dof_acc": np.sum(((dq - self.last_dq) / self.dt) ** 2),
            "dof_vel": np.sum(dq ** 2),
            "action_rate": np.sum((self.action - self.last_action) ** 2),
            "torques": np.sum(torques_sq),
            "dof_pos_limits": np.sum(np.clip(self.soft_lo - q, 0, None) + np.clip(q - self.soft_hi, 0, None)),
            "alive": 1.0,
            "hip_pos": np.sum(q[self.hip_roll_yaw] ** 2),
            "contact_no_vel": np.sum(np.sum(feet_vel ** 2, axis=1) * contact),
            "feet_phase": np.sum(feet_phase),
            "contact": np.sum(np.where(contact == should_stance, 1.0, -1.0)),
            "feet_air_time": air_time_rew,
            "feet_distance": np.clip(0.18 - lateral, 0, None),
        }
        terms = {k: raw[k] * s * self.dt for k, s in c.reward_scales.items()}
        total = max(sum(terms.values()), 0.0)  # only-positive rewards, as in legged_gym
        return total, terms

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="track")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
