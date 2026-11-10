"""Experimental scheduled MuJoCo-generator regularization for Qflex.

Adaptation of arXiv:2608.17117, equations 39–44. Deterministic finite-step
generator, not the paper's continuous stochastic PDE or a safety certificate.
"""
from dataclasses import dataclass

import gymnasium as gym
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax

from g1_env import Config, G1WalkEnv


@dataclass(frozen=True)
class PhysicsConfig:
    start: int = 100_000
    ramp: int = 100_000
    every: int = 100
    batch_size: int = 8
    pde_weight: float = 0.01
    boundary_weight: float = 1.0
    goal_distance: float = 1.0
    goal_radius: float = 0.25

    def __post_init__(self):
        if self.start < 0 or min(self.ramp, self.every, self.batch_size) <= 0:
            raise ValueError('Invalid physics schedule or batch size')
        if min(self.pde_weight, self.boundary_weight) < 0:
            raise ValueError('Physics weights must be nonnegative')
        if not 0 < self.goal_radius < self.goal_distance:
            raise ValueError('Require 0 < goal radius < goal distance')


def continuation(step, config):
    """TD-only warmup, followed by a normalized sigmoid continuation."""
    t = np.clip((step - config.start) / config.ramp, 0.0, 1.0)
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    return float((sigmoid(12 * (t - 0.5)) - sigmoid(-6)) /
                 (sigmoid(6) - sigmoid(-6)))


class ReachAvoidEnv(G1WalkEnv):
    """Privileged deterministic state, absorbing goal/fall/horizon terminals.

    Observation: [qpos, qvel, remaining seconds, goal world x/y]. No noisy
    sensor projection, random pushes, payload changes or hidden commands.
    These observations are for simulation research, not hardware deployment.
    """
    def __init__(self, physics=None, **kwargs):
        self.physics = physics or PhysicsConfig()
        super().__init__(cfg=Config(randomize=False, obs_noise=False), **kwargs)
        self.goal = np.zeros(2)
        self.tau_index = self.model.nq + self.model.nv
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.tau_index + 3,), np.float32)
        self.fixed_command = np.zeros(3)

    def _obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel,
                               [max(0, self.cfg.episode_seconds - self.step_count * self.dt)],
                               self.goal]).astype(np.float32)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        q = self.data.qpos[3:7]
        yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
        self.goal = self.data.qpos[:2] + self.physics.goal_distance * np.array([np.cos(yaw), np.sin(yaw)])
        return self._obs(), {}

    def step(self, action):
        _, _, fell, timeout, info = super().step(action)
        reached = bool(not fell and np.linalg.norm(self.data.qpos[:2] - self.goal) <= self.physics.goal_radius)
        ended = fell or reached or timeout
        # Horizon is part of state and is a real terminal, not a bootstrap truncation.
        if ended:
            info.update(fell=fell, reached=reached,
                        vel_error=self.ep_vel_err / self.step_count)
        return self._obs(), float(reached), ended, False, info

    def restore(self, obs):
        """Restore physical state in the independent collocation simulator."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = obs[:m.nq]
        d.qpos[3:7] /= np.linalg.norm(d.qpos[3:7])
        d.qvel[:] = obs[m.nq:self.tau_index]
        self.goal = obs[-2:].copy()
        self.step_count = int(round((self.cfg.episode_seconds - obs[self.tau_index]) / self.dt))
        mujoco.mj_forward(m, d)

    def fall(self):
        gravity = self._base_state()[3]
        return self.data.qpos[2] < 0.4 or abs(gravity[0]) > 0.84 or abs(gravity[1]) > 0.7

    def advance_physics(self, action):
        """Only deterministic PD/MuJoCo dynamics; no reward or episode bookkeeping."""
        target = self.default_q + self.cfg.action_scale * np.clip(action * self.action_clip, -10, 10)
        for _ in range(self.cfg.decimation):
            tau = self.kp * (target - self.data.qpos[self.qpos_idx]) - self.kd * self.data.qvel[self.qvel_idx]
            self.data.ctrl[self.act_idx] = np.clip(tau, -self.torque_limit, self.torque_limit)
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        return self._obs()


class PhysicsRegularizer:
    """Extra critic optimizer steps; Qflex's policy sees these critics next update."""
    def __init__(self, algorithm, config, seed=0):
        self.algorithm, self.config = algorithm, config
        self.rng = np.random.default_rng(seed)
        self.sim = ReachAvoidEnv(config)
        self.sim.reset(seed=seed)
        self._update = jax.jit(self._critic_step)

    def _critic_step(self, params, states, optimizer_states, x, a, drift, boundary, labels, weights):
        algorithm = self.algorithm

        def one(qparams, qstate, optstate):
            def loss(p):
                # JVP differentiates state only; the collocation action is held fixed.
                def residual(s, u, f):
                    def q(y):
                        return algorithm.agent.q(p, qstate, y[None], u[None], is_training=False)[0].reshape(())
                    return jax.jvp(q, (s,), (f,))[1]
                residuals = jax.vmap(residual)(x, a, drift)
                qbc, _ = algorithm.agent.q(p, qstate, boundary, jnp.tile(a, (3, 1)), is_training=False)
                pde = jnp.mean(residuals ** 2)
                bc = jnp.mean((qbc.reshape(-1) - labels) ** 2)
                return weights[0] * pde + weights[1] * bc, (pde, bc)

            (_, components), gradients = jax.value_and_grad(loss, has_aux=True)(qparams)
            change, optstate = algorithm.q_optim.update(gradients, optstate)
            return optax.apply_updates(qparams, change), optstate, components

        p1, o1, c1 = one(params[0], states[0], optimizer_states[0])
        p2, o2, c2 = one(params[1], states[1], optimizer_states[1])
        return (p1, p2), (o1, o2), (jnp.mean(jnp.array([c1[0], c2[0]])), jnp.mean(jnp.array([c1[1], c2[1]])))

    def update(self, key, replay_obs, step):
        cfg, sim, algorithm = self.config, self.sim, self.algorithm
        weight = continuation(step, cfg)
        if weight == 0 or (cfg.pde_weight == 0 and cfg.boundary_weight == 0):
            return dict(physics_weight=0.0, physics_pde_loss=0.0, physics_boundary_loss=0.0, physics_points=0)
        points, actions, directions = [], [], []
        # Independently perturb replay states, then obtain drift from fresh simulator calls.
        for obs in replay_obs:
            candidate = np.array(obs, copy=True)
            candidate[sim.model.nq:sim.tau_index] += self.rng.normal(0, 0.02, sim.model.nv)
            sim.restore(candidate)
            if sim.fall() or candidate[sim.tau_index] <= sim.dt or np.linalg.norm(candidate[:2]-candidate[-2:]) <= cfg.goal_radius:
                continue
            key, subkey = jax.random.split(key)
            action = np.clip(algorithm.get_action(subkey, candidate), -1, 1).reshape(12)
            # Skip detected contact-count changes; this is only a smoothness heuristic.
            contacts = sim.data.ncon
            nxt = sim.advance_physics(action)
            if sim.data.ncon != contacts or sim.fall():
                continue
            points.append(candidate)
            actions.append(action)
            directions.append((nxt - candidate) / sim.dt)
        if not points:
            return dict(physics_weight=weight, physics_pde_loss=0.0, physics_boundary_loss=0.0, physics_points=0)
        x, actions, drift = map(np.asarray, (points, actions, directions))
        # Boundary samples: goal (1), fallen pelvis (0), expired horizon outside goal (0).
        goal, avoid, horizon = x.copy(), x.copy(), x.copy()
        goal[:, -2:] = goal[:, :2]
        avoid[:, 2] = 0.39
        horizon[:, sim.tau_index] = 0
        bc = np.concatenate([goal, avoid, horizon])
        labels = np.concatenate([np.ones(len(x)), np.zeros(2 * len(x))]).astype(np.float32)
        s = algorithm.state
        qp, op, losses = self._update((s.params.q1, s.params.q2),
                                     (s.params.q1_state, s.params.q2_state),
                                     (s.opt_state.q1, s.opt_state.q2),
                                     x, actions, drift, bc, labels,
                                     np.array([cfg.pde_weight, cfg.boundary_weight]) * weight)
        algorithm.state = s._replace(params=s.params._replace(q1=qp[0], q2=qp[1]),
                                     opt_state=s.opt_state._replace(q1=op[0], q2=op[1]))
        return dict(physics_weight=weight, physics_pde_loss=float(losses[0]),
                    physics_boundary_loss=float(losses[1]), physics_points=len(x))

    def close(self):
        self.sim.close()
