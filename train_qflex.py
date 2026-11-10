"""Train upstream Qflex on the existing G1 environment; see docs/QFLEX.md."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qflex-root', type=Path, required=True)
    p.add_argument('--resume-state', type=Path)
    p.add_argument('--resume-step', type=int, default=0)
    p.add_argument('--steps', type=int, default=5_000_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--name', default='qflex_walk')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--buffer-size', type=int, default=1_000_000)
    p.add_argument('--start-step', type=int, default=30_000)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--flow-steps', type=int, default=20)
    p.add_argument('--grad-step-num', type=int, default=20)
    p.add_argument('--grad-step-size', type=float, default=0.01)
    p.add_argument('--eval-every', type=int, default=100_000)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--save-every', type=int, default=100_000)
    a = p.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, (int, float)) and key not in {'seed', 'resume_step'} and value <= 0:
            p.error(f'--{key.replace("_", "-")} must be positive')
    if a.buffer_size < max(a.batch_size, a.start_step):
        p.error('--buffer-size must cover --batch-size and --start-step')
    if Path(a.name).name != a.name or a.name in {'.', '..'}:
        p.error('--name must be a single folder name')
    if a.resume_step < 0 or a.resume_step >= a.steps:
        p.error('--resume-step must be nonnegative and less than --steps')
    if bool(a.resume_state) != bool(a.resume_step):
        p.error('--resume-state and a positive --resume-step must be supplied together')
    return a


def evaluate(algorithm, env, episodes, seed):
    """Use the same held-out seeds at each evaluation for comparable results."""
    import jax
    import numpy as np
    returns, falls, errors = [], [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        key = jax.random.key(seed + episode)
        while True:
            key, action_key = jax.random.split(key)
            action = np.clip(algorithm.get_action(action_key, obs), -1, 1).reshape(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                returns.append(total)
                falls.append(bool(info['fell']))
                errors.append(info['vel_error'])
                break
    return float(np.mean(returns)), 1 - float(np.mean(falls)), float(np.mean(errors))


def main():
    args = parse_args()
    source = args.qflex_root.expanduser().resolve()
    if not (source / 'relax/algorithm/flow_exp.py').is_file():
        raise SystemExit('--qflex-root must point to a checkout of LNSGroup/Qflex')
    sys.path.insert(0, str(source))
    import jax
    import numpy as np
    from relax.algorithm.flow_exp import FlowExp
    from relax.buffer import TreeBuffer
    from relax.network.flow import create_flow_net
    from relax.utils.experience import Experience
    from g1_env import Config, G1WalkEnv

    run = ROOT / 'runs' / f'{args.name}_{datetime.now():%Y%m%d-%H%M%S-%f}'
    run.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    (run / 'config.json').write_text(json.dumps({
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'qflex_commit': revision, 'env_config': dataclasses.asdict(Config()),
        'command': sys.argv, 'jax_devices': [str(d) for d in jax.devices()],
    }, indent=2))
    env, eval_env = G1WalkEnv(seed=args.seed), G1WalkEnv(seed=args.seed + 100_000)
    try:
        env.action_space.seed(args.seed)
        obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        key, init_key = jax.random.split(jax.random.key(args.seed))
        agent, params = create_flow_net(init_key, obs_dim, act_dim,
                                        [args.hidden_dim] * 3,
                                        num_timesteps=args.flow_steps,
                                        use_bn=True, learn_reference_gn=True)
        algorithm = FlowExp(agent, params, grad_step_num=args.grad_step_num,
                            grad_step_size=args.grad_step_size)
        buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=args.buffer_size,
                                            seed=args.seed)
        algorithm.warmup(Experience.create_example(obs_dim, act_dim, args.batch_size))
        if args.resume_state:
            algorithm.load(args.resume_state)
        print(f'run folder: {run}', flush=True)
        obs, _ = env.reset(seed=args.seed)
        with (run / 'evaluation.csv').open('w', newline='') as ef, \
                (run / 'updates.csv').open('w', newline='') as uf:
            evaluation = csv.writer(ef)
            evaluation.writerow(['step', 'return', 'success_rate', 'vel_error', 'wall_seconds'])
            updates = None
            started = time.monotonic()
            try:
                for step in range(args.resume_step + 1, args.steps + 1):
                    key, action_key, update_key = jax.random.split(key, 3)
                    action = (env.action_space.sample() if len(buffer) < args.start_step else
                              np.clip(algorithm.get_action(action_key, obs), -1, 1).reshape(env.action_space.shape))
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    # Store the terminal observation BEFORE resetting. Time limits bootstrap;
                    # falls do not, matching upstream Experience.create semantics.
                    buffer.add(Experience.create(obs, action, reward, terminated,
                                                 truncated, next_obs, info))
                    obs = env.reset()[0] if terminated or truncated else next_obs
                    if len(buffer) >= max(args.start_step, args.batch_size):
                        metrics, _ = algorithm.update(update_key, buffer.sample(args.batch_size))
                        if updates is None:
                            updates = csv.DictWriter(uf, fieldnames=['step', *metrics])
                            updates.writeheader()
                        if step % 1000 == 0:
                            if not all(np.isfinite(v) for v in metrics.values()):
                                raise FloatingPointError(f'Nonfinite Qflex metrics at step {step}: {metrics}')
                            print(f'training step={step} elapsed={time.monotonic() - started:.1f}s', flush=True)
                            updates.writerow({'step': step, **metrics})
                            uf.flush()
                    if step % args.save_every == 0:
                        algorithm.save(run / f'state-{step}.pkl')
                        algorithm.save_policy(run / f'policy-{step}.pkl')
                    if step % args.eval_every == 0 or step == args.steps:
                        result = evaluate(algorithm, eval_env, args.eval_episodes,
                                          args.seed + 100_000)
                        evaluation.writerow([step, *result, time.monotonic() - started])
                        ef.flush()
                        print(f'step={step} return={result[0]:.3f} '
                              f'success={result[1]:.3f} vel_error={result[2]:.3f} '
                              f'wall_seconds={time.monotonic() - started:.1f}', flush=True)
            finally:
                algorithm.save(run / 'state-final.pkl')
                algorithm.save_policy(run / 'policy-final.pkl')
    finally:
        env.close()
        eval_env.close()


if __name__ == '__main__':
    main()
