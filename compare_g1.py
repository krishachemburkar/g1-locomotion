"""Evaluate PPO checkpoints on the same seeded G1 episodes as Qflex."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=100000)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.project))
    from g1_env import G1WalkEnv
    torch.set_num_threads(1)
    base = a.project / 'runs/walk_v2_20261029-231058'
    checkpoints = [(2_000_000, base / 'checkpoints/ppo_2000000_steps.zip'),
                   (50_000_000, base / 'checkpoints/ppo_50000000_steps.zip')]
    a.output.mkdir(parents=True, exist_ok=True)
    results = []
    for steps, path in checkpoints:
        model = PPO.load(path, device='cpu')
        rows = []
        env = G1WalkEnv()
        started = time.monotonic()
        try:
            for episode in range(a.episodes):
                obs, _ = env.reset(seed=a.seed + episode)
                total, length = 0.0, 0
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
                    total += reward
                    length += 1
                    if terminated or truncated:
                        rows.append(dict(seed=a.seed + episode, episode_return=total,
                                         fell=info['fell'], vel_error=info['vel_error'], steps=length))
                        break
        finally:
            env.close()
        with (a.output / f'ppo_{steps}_episodes.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        result = dict(algorithm='PPO', training_steps=steps, checkpoint=str(path),
                      episodes=a.episodes, evaluation_seed=a.seed,
                      success_rate=1 - float(np.mean([r['fell'] for r in rows])),
                      mean_return=float(np.mean([r['episode_return'] for r in rows])),
                      mean_vel_error=float(np.mean([r['vel_error'] for r in rows])),
                      evaluation_seconds=time.monotonic() - started)
        results.append(result)
        (a.output / 'ppo_summary.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
