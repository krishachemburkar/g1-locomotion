"""Run with: python -m unittest test_physics_informed -v."""
import unittest
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import optax

from physics_informed import PhysicsConfig, PhysicsRegularizer, ReachAvoidEnv, continuation


class PhysicsTests(unittest.TestCase):
    def test_continuation(self):
        config = PhysicsConfig(start=10, ramp=10)
        self.assertEqual(continuation(0, config), 0)
        self.assertEqual(continuation(10, config), 0)
        self.assertAlmostEqual(continuation(15, config), 0.5)
        self.assertEqual(continuation(20, config), 1)

    def test_simulator_restore(self):
        env, clone = ReachAvoidEnv(), ReachAvoidEnv()
        try:
            obs, _ = env.reset(seed=4)
            clone.reset(seed=9)
            clone.restore(obs)
            action = np.linspace(-0.1, 0.1, 12)
            expected = env.step(action)[0]
            actual = clone.advance_physics(action)
            np.testing.assert_allclose(actual, expected, atol=3e-6)
        finally:
            env.close()
            clone.close()

    def test_goal_and_horizon_are_absorbing(self):
        env = ReachAvoidEnv()
        try:
            env.reset(seed=3)
            env.goal = env.data.qpos[:2].copy()
            _, reward, done, truncated, info = env.step(np.zeros(12))
            self.assertEqual(reward, 1)
            self.assertTrue(done and info['reached'])
            self.assertFalse(truncated)
            env.reset(seed=3)
            env.step_count = env.max_steps - 1
            _, reward, done, truncated, info = env.step(np.zeros(12))
            self.assertEqual(reward, 0)
            self.assertTrue(done)
            self.assertFalse(truncated or info['reached'])
        finally:
            env.close()

    def test_generator_loss_and_gradient(self):
        # Affine critic Q=p*x has directional derivative p*f = 2*3 = 6.
        regularizer = object.__new__(PhysicsRegularizer)
        optimizer = optax.sgd(0.01)
        agent = SimpleNamespace(q=lambda p, state, x, a, is_training: (x @ p, state))
        regularizer.algorithm = SimpleNamespace(agent=agent, q_optim=optimizer)
        params = jnp.array([2.0])
        state = optimizer.init(params)
        qp, _, losses = regularizer._critic_step(
            (params, params), ({}, {}), (state, state),
            jnp.array([[1.0]]), jnp.zeros((1, 1)), jnp.array([[3.0]]),
            jnp.ones((3, 1)), jnp.zeros(3), jnp.array([1.0, 0.0]))
        self.assertAlmostEqual(float(losses[0]), 36.0)
        self.assertAlmostEqual(float(losses[1]), 4.0)
        # d(p*f)^2/dp = 2*p*f^2 = 36; SGD should yield 1.64.
        self.assertAlmostEqual(float(qp[0][0]), 1.64, places=5)


if __name__ == '__main__':
    unittest.main()
