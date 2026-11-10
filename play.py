"""Run a trained walking policy.

Interactive viewer (macOS needs mjpython for the viewer):
    .venv/bin/mjpython play.py runs/g1_walk/final.zip
      Up/Down    forward speed  +-0.1 m/s
      Left/Right turn rate      +-0.2 rad/s
      , / .      sideways speed +-0.1 m/s
      Space      stop (zero command)

When the turn command is 0 the robot holds its current heading (a small
P-controller on the IMU yaw), so it walks straight instead of slowly drifting.
Use --no-heading-hold to see the raw policy.

Headless video + tracking report:
    .venv/bin/python play.py runs/g1_walk/final.zip --video out.mp4 --vx 0.5 --seconds 10
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from stable_baselines3 import PPO

from g1_env import Config, G1WalkEnv


def heading(env: G1WalkEnv) -> float:
    w, x, y, z = env.data.qpos[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class HeadingHold:
    """Turn a yaw-rate command of 0 into "keep facing the same way"."""

    def __init__(self, enabled: bool, gain: float = 1.0, max_rate: float = 0.5):
        self.enabled, self.gain, self.max_rate = enabled, gain, max_rate
        self.target: float | None = None

    def yaw_command(self, env: G1WalkEnv, yaw_cmd: float) -> float:
        if not self.enabled or yaw_cmd != 0.0:
            self.target = None  # steering: follow the turn rate, re-latch when released
            return yaw_cmd
        if self.target is None:
            self.target = heading(env)
        err = (self.target - heading(env) + np.pi) % (2 * np.pi) - np.pi
        return float(np.clip(self.gain * err, -self.max_rate, self.max_rate))


def make_env(args) -> G1WalkEnv:
    env = G1WalkEnv(Config(randomize=False, obs_noise=False, episode_seconds=1e9), seed=0)
    env.set_command(args.vx, args.vy, args.yaw)
    return env


def run_video(model, args) -> None:
    import imageio

    env = make_env(args)
    obs, _ = env.reset(seed=0)
    hold = HeadingHold(not args.no_heading_hold)
    start_heading, start_xy = heading(env), env.data.qpos[:2].copy()
    frames, vels = [], []
    steps = int(args.seconds / env.dt)
    for i in range(steps):
        env.set_command(args.vx, args.vy, hold.yaw_command(env, args.yaw))
        obs = env._obs()
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        _, lin_vel, ang_vel, _ = env._base_state()
        vels.append([lin_vel[0], lin_vel[1], ang_vel[2]])
        if i % 2 == 0:
            frames.append(env.render())
        if terminated:
            print(f"fell after {i * env.dt:.1f} s")
            break
    imageio.mimsave(args.video, frames, fps=25)
    v = np.array(vels[len(vels) // 4:])  # skip the start-up transient
    print(f"command  vx={args.vx:+.2f} vy={args.vy:+.2f} yaw={args.yaw:+.2f}")
    print(f"achieved vx={v[:, 0].mean():+.2f} vy={v[:, 1].mean():+.2f} yaw={v[:, 2].mean():+.2f}")
    d = env.data.qpos[:2] - start_xy
    side = np.array([-np.sin(start_heading), np.cos(start_heading)])
    turned = np.degrees((heading(env) - start_heading + np.pi) % (2 * np.pi) - np.pi)
    print(f"drift: {d @ side:+.2f} m sideways, heading changed {turned:+.0f} deg")
    print(f"wrote {args.video}")


def run_viewer(model, args) -> None:
    import mujoco.viewer

    env = make_env(args)
    obs, _ = env.reset(seed=0)
    cmd = np.array([args.vx, args.vy, args.yaw])
    hold = HeadingHold(not args.no_heading_hold)

    def on_key(key: int) -> None:
        steps = {265: (0, 0.1), 264: (0, -0.1), 263: (2, 0.2), 262: (2, -0.2), 44: (1, 0.1), 46: (1, -0.1)}
        if key == 32:
            cmd[:] = 0
        elif key in steps:
            i, d = steps[key]
            cmd[i] = np.round(cmd[i] + d, 2)
        print(f"command vx={cmd[0]:+.1f} vy={cmd[1]:+.1f} yaw={cmd[2]:+.1f}")

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.model.body("pelvis").id
        viewer.cam.distance = 3.0
        while viewer.is_running():
            start = time.time()
            env.set_command(cmd[0], cmd[1], hold.yaw_command(env, cmd[2]))
            obs = env._obs()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                print("fell, resetting")
                obs, _ = env.reset()
            viewer.sync()
            time.sleep(max(0.0, env.dt - (time.time() - start)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--no-heading-hold", action="store_true")
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    if args.video:
        run_video(model, args)
    else:
        run_viewer(model, args)


if __name__ == "__main__":
    main()
