"""Train a PPO walking policy for the G1.

    .venv/bin/python train.py --name walk                # default run, 50M steps
    .venv/bin/python train.py --steps 5e6 --name quick
    .venv/bin/python train.py --name walk_more --resume runs/<run>/checkpoints/ppo_10000000_steps.zip

Every run gets its own folder, runs/<name>_<date-time>/, so nothing is ever
overwritten. It holds checkpoints/, final.zip, config.json, the TensorBoard
events, progress.csv and log.txt (watch with: .venv/bin/tensorboard --logdir runs).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from torch import nn

from g1_env import Config, G1WalkEnv

ROOT = Path(__file__).resolve().parent


class EpisodeStatsLogger(BaseCallback):
    """Log success rate, velocity error and each reward term once per rollout.

    success_rate = share of finished episodes that lasted the full 20 s
    without falling. vel_error = mean m/s between achieved and commanded
    velocity. Reward terms are per-episode sums divided by episode seconds.
    """

    def __init__(self):
        super().__init__()
        self.terms: dict[str, list[float]] = {}
        self.fell: list[bool] = []
        self.vel_err: list[float] = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "episode_terms" in info:
                self.fell.append(info["fell"])
                self.vel_err.append(info["vel_error"])
                length = info.get("episode", {}).get("l", 1) or 1
                for k, v in info["episode_terms"].items():
                    self.terms.setdefault(k, []).append(v / (length * 0.02))
        return True

    def _on_rollout_end(self) -> None:
        if self.fell:
            self.logger.record("walk/success_rate", 1.0 - float(np.mean(self.fell)))
            self.logger.record("walk/vel_error", float(np.mean(self.vel_err)))
            self.logger.record("walk/episodes", len(self.fell))
        for k, vals in self.terms.items():
            self.logger.record(f"reward_terms/{k}", float(np.mean(vals)))
        self.terms.clear()
        self.fell.clear()
        self.vel_err.clear()


def make_env(rank: int, seed: int):
    def _init():
        env = G1WalkEnv(seed=seed + rank)
        env.reset(seed=seed + rank)
        return env
    return _init


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="g1_walk")
    p.add_argument("--steps", type=float, default=50e6)
    p.add_argument("--envs", type=int, default=os.cpu_count() or 8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    run_dir = ROOT / "runs" / f"{args.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps({
        "args": vars(args),
        "command": " ".join(sys.argv),
        "env_config": dataclasses.asdict(Config()),
    }, indent=2))
    print(f"run folder: {run_dir}")

    env = VecMonitor(SubprocVecEnv([make_env(i, args.seed) for i in range(args.envs)]))

    n_steps = 1024  # per env, so one update sees envs * 1024 transitions
    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=n_steps * args.envs // 4,  # 4 minibatches, as in legged_gym
            n_epochs=5,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # 0.003 let exploration collapse in run g1_walk
            vf_coef=1.0,
            max_grad_norm=1.0,
            target_kl=0.03,
            policy_kwargs=dict(
                net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                activation_fn=nn.ELU,
                # env scales actions by 5 * 0.25, so std 0.2 = 0.25 rad of joint noise
                log_std_init=math.log(0.2),
            ),
            seed=args.seed,
            device="cpu",
            verbose=1,
        )

    model.set_logger(configure(str(run_dir), ["stdout", "log", "csv", "tensorboard"]))

    callbacks = [
        EpisodeStatsLogger(),
        CheckpointCallback(
            save_freq=max(2_000_000 // args.envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
        ),
    ]
    try:
        model.learn(
            total_timesteps=int(args.steps),
            callback=callbacks,
            reset_num_timesteps=args.resume is None,
        )
    finally:
        model.save(run_dir / "final")
        print(f"saved {run_dir / 'final.zip'}")
        env.close()


if __name__ == "__main__":
    main()
