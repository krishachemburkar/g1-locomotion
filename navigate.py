"""Walk the G1 from a random point A to a random point B.

The trained walking policy is unchanged. A small navigator sits on top of it:
every policy step (50 Hz) it looks at where B is relative to the robot and
turns that into the same (vx, vy, yaw_rate) command the policy was trained
to follow.

  - far from B and facing it   -> walk forward, steering toward B
  - B behind or to the side     -> turn on the spot first
  - close to B                  -> slow down, then stop (zero command)

Success = the robot ends within --tolerance of B, without falling, and is
still standing there after --settle seconds.

Evaluate on many random A/B pairs:
    .venv/bin/python navigate.py runs/walk_v2_20261029-231058/checkpoints/ppo_30000000_steps.zip --episodes 50

Save a video of a few episodes (A = blue marker, B = green marker):
    .venv/bin/python navigate.py <model.zip> --episodes 3 --video nav.mp4

The navigator reads the robot's position straight from the simulator. On the
real robot this would come from odometry or localization instead.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import mujoco
import numpy as np
from stable_baselines3 import PPO

from g1_env import Config, G1WalkEnv


def wrap(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def heading(env: G1WalkEnv) -> float:
    w, x, y, z = env.data.qpos[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


@dataclass
class Navigator:
    """Turn "where is B" into a velocity command the walking policy understands."""

    max_speed: float = 0.8       # m/s, top forward speed
    min_speed: float = 0.3       # m/s; the policy was trained with speeds under 0.2 set to zero
    slow_radius: float = 1.0     # m, start slowing down inside this distance
    max_turn: float = 0.8        # rad/s, same limit as in training
    turn_gain: float = 1.5       # turn rate per rad of heading error
    turn_in_place: float = 0.8   # rad (~45 deg); larger heading error -> turn before walking
    tolerance: float = 0.25      # m, close enough to B

    def command(self, pos_xy: np.ndarray, yaw: float, goal_xy: np.ndarray) -> tuple[np.ndarray, float]:
        delta = goal_xy - pos_xy
        dist = float(np.linalg.norm(delta))
        if dist < self.tolerance:
            return np.zeros(3), dist

        # Goal direction relative to where the robot faces.
        bearing = wrap(np.arctan2(delta[1], delta[0]) - yaw)
        yaw_rate = float(np.clip(self.turn_gain * bearing, -self.max_turn, self.max_turn))

        if abs(bearing) > self.turn_in_place:
            return np.array([0.0, 0.0, yaw_rate]), dist

        speed = self.max_speed * min(1.0, dist / self.slow_radius)
        speed = max(speed, self.min_speed) * np.cos(bearing)  # slower while still turning
        return np.array([speed, 0.0, yaw_rate]), dist


def sample_points(rng: np.random.Generator, area: float, min_dist: float, max_dist: float):
    """Random start A (position + facing) and goal B at a distance in [min_dist, max_dist]."""
    a = rng.uniform(-area, area, 2)
    a_yaw = rng.uniform(-np.pi, np.pi)
    dist = rng.uniform(min_dist, max_dist)
    direction = rng.uniform(-np.pi, np.pi)  # any direction, so B can be behind the robot
    b = a + dist * np.array([np.cos(direction), np.sin(direction)])
    return a, a_yaw, b


def place_robot(env: G1WalkEnv, xy: np.ndarray, yaw: float) -> None:
    d = env.data
    d.qpos[0:2] = xy
    d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qvel[:] = 0.0
    mujoco.mj_forward(env.model, d)
    env.last_feet_pos, _ = env._feet()


class Video:
    def __init__(self, env: G1WalkEnv):
        self.env = env
        self.renderer = mujoco.Renderer(env.model, height=480, width=640)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.elevation = -35
        self.cam.azimuth = 90
        self.frames: list[np.ndarray] = []

    def _marker(self, xy: np.ndarray, rgba) -> None:
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.array([0.15, 0.15, 0.005]), np.array([xy[0], xy[1], 0.005]),
            np.eye(3).flatten(), np.array(rgba, dtype=np.float32),
        )
        scn.ngeom += 1

    def frame(self, a: np.ndarray, b: np.ndarray) -> None:
        # Camera looks at the midpoint of the robot and the goal, far enough to see both.
        pos = self.env.data.qpos[:2]
        mid = (pos + b) / 2
        self.cam.lookat[:] = [mid[0], mid[1], 0.5]
        self.cam.distance = max(3.5, 1.3 * np.linalg.norm(pos - b) + 2.0)
        self.renderer.update_scene(self.env.data, self.cam)
        self._marker(a, [0.2, 0.4, 1.0, 0.8])
        self._marker(b, [0.1, 0.9, 0.2, 0.9])
        self.frames.append(self.renderer.render())


def run_episode(env, model, nav: Navigator, rng, args, video: Video | None = None) -> dict:
    env.reset(seed=int(rng.integers(1 << 31)))
    a, a_yaw, b = sample_points(rng, args.area, args.min_dist, args.max_dist)
    place_robot(env, a, a_yaw)

    max_steps = int(args.timeout / env.dt)
    settle_steps = int(args.settle / env.dt)
    reached_at, in_zone, path_len, last_xy = None, 0, 0.0, a.copy()
    result = {"fell": False, "reached": False}

    for i in range(max_steps):
        pos = env.data.qpos[:2].copy()
        path_len += float(np.linalg.norm(pos - last_xy))
        last_xy = pos
        cmd, dist = nav.command(pos, heading(env), b)
        env.set_command(*cmd)
        action, _ = model.predict(env._obs(), deterministic=True)
        _, _, terminated, _, _ = env.step(action)
        if video is not None and i % 2 == 0:
            video.frame(a, b)
        if terminated:
            result["fell"] = True
            break
        # Success needs settle_steps in a row inside the tolerance. If a push
        # moves the robot out, the navigator walks it back and the count restarts.
        if dist < nav.tolerance:
            if reached_at is None:
                reached_at = i
            in_zone += 1
            if in_zone >= settle_steps:
                break
        else:
            in_zone = 0

    final_err = float(np.linalg.norm(env.data.qpos[:2] - b))
    result.update(
        straight_dist=float(np.linalg.norm(b - a)),
        path_len=path_len,
        final_err=final_err,
        time=None if reached_at is None else reached_at * env.dt,
        reached=(not result["fell"]) and in_zone >= settle_steps,
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--area", type=float, default=5.0, help="A is sampled in [-area, area]^2 (m)")
    p.add_argument("--min-dist", type=float, default=2.0, help="min A-to-B distance (m)")
    p.add_argument("--max-dist", type=float, default=6.0, help="max A-to-B distance (m)")
    p.add_argument("--tolerance", type=float, default=0.25, help="success radius around B (m)")
    p.add_argument("--settle", type=float, default=2.0, help="must stay standing at B this long (s)")
    p.add_argument("--timeout", type=float, default=40.0, help="give up after this long (s)")
    p.add_argument("--randomize", action="store_true", help="training randomization: friction, mass, pushes, noise")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", type=str, default=None)
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    cfg = Config(randomize=args.randomize, obs_noise=args.randomize, episode_seconds=1e9)
    env = G1WalkEnv(cfg, seed=args.seed)
    nav = Navigator(tolerance=args.tolerance)
    rng = np.random.default_rng(args.seed)
    video = Video(env) if args.video else None

    results = []
    for ep in range(args.episodes):
        r = run_episode(env, model, nav, rng, args, video)
        results.append(r)
        status = "reached" if r["reached"] else ("FELL" if r["fell"] else "missed")
        t = f"{r['time']:.1f} s" if r["time"] is not None else "  -  "
        print(f"ep {ep:3d}  {status:7s}  dist {r['straight_dist']:.1f} m  path {r['path_len']:.1f} m  "
              f"time {t}  final error {r['final_err']:.2f} m")

    ok = [r for r in results if r["reached"]]
    print(f"\nsuccess {len(ok)}/{len(results)}  fell {sum(r['fell'] for r in results)}")
    if ok:
        print(f"mean time to B {np.mean([r['time'] for r in ok]):.1f} s, "
              f"mean final error {np.mean([r['final_err'] for r in ok]):.2f} m, "
              f"path / straight line {np.mean([r['path_len'] / r['straight_dist'] for r in ok]):.2f}")

    if video is not None:
        import imageio
        imageio.mimsave(args.video, video.frames, fps=25)
        print(f"wrote {args.video}")


if __name__ == "__main__":
    main()
