"""Point-to-point success rate across training checkpoints, then plot it.

Each checkpoint is tested on the same random A/B pairs (same seed), once in
the clean sim and once with training randomization (friction, torso mass,
pushes, sensor noise). Success is defined in navigate.py: within 0.25 m of B
and standing there for 2 s in a row, without falling, inside 40 s.

    .venv/bin/python nav_eval.py runs/walk_v2_20261029-231058 --episodes 50
    .venv/bin/python nav_eval.py runs/walk_v2_20261029-231058 --plot-only

Writes <run>/nav_eval.json and <run>/nav_success.png.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

NAV_DEFAULTS = dict(area=5.0, min_dist=2.0, max_dist=6.0, tolerance=0.25, settle=2.0, timeout=40.0)


def evaluate(job: tuple[str, bool, int, int]) -> dict:
    from stable_baselines3 import PPO

    from g1_env import Config, G1WalkEnv
    from navigate import Navigator, run_episode

    path, randomize, episodes, seed = job
    args = SimpleNamespace(**NAV_DEFAULTS)
    model = PPO.load(path, device="cpu")
    env = G1WalkEnv(Config(randomize=randomize, obs_noise=randomize, episode_seconds=1e9), seed=seed)
    nav = Navigator(tolerance=args.tolerance)
    rng = np.random.default_rng(seed)
    results = [run_episode(env, model, nav, rng, args) for _ in range(episodes)]
    ok = [r for r in results if r["reached"]]
    return {
        "steps": int(re.search(r"(\d+)_steps", path).group(1)),
        "randomize": randomize,
        "episodes": episodes,
        "success": len(ok),
        "fell": sum(r["fell"] for r in results),
        "mean_time": float(np.mean([r["time"] for r in ok])) if ok else None,
    }


def plot(data: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    series = [(False, "Clean sim", "#2a78d6"), (True, "With randomization", "#eb6834")]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for randomize, label, color in series:
        rows = sorted((d for d in data if d["randomize"] == randomize), key=lambda d: d["steps"])
        x = [d["steps"] / 1e6 for d in rows]
        y = [100 * d["success"] / d["episodes"] for d in rows]
        ax.plot(x, y, color=color, lw=2, marker="o", ms=5, label=label,
                markeredgecolor=surface, markeredgewidth=1.5, zorder=3)
        ax.annotate(f"{label}  {y[-1]:.0f}%", (x[-1], y[-1]), xytext=(8, -14 if randomize else 6),
                    textcoords="offset points", color=ink, fontsize=8.5, va="center")

    n = data[0]["episodes"]
    ax.set_title(f"Point A to B success rate during training ({n} random A/B pairs per point)",
                 loc="left", fontsize=11, color=ink, pad=12)
    ax.set_xlabel("Training steps (millions)", color=muted, fontsize=9)
    ax.set_ylabel("Success rate (%)", color=muted, fontsize=9)
    ax.set_ylim(-3, 105)
    ax.set_xlim(0, max(d["steps"] for d in data) / 1e6 * 1.22)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.tick_params(colors=muted, labelsize=8.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=ink)
    fig.text(0.01, 0.01, "Success = within 0.25 m of B for 2 s in a row, no fall, within 40 s.",
             color=muted, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, facecolor=surface)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--every", type=int, default=4, help="use every Nth million-step checkpoint (multiple of 2)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    out_json = args.run / "nav_eval.json"
    if not args.plot_only:
        ckpts = sorted((args.run / "checkpoints").glob("ppo_*_steps.zip"),
                       key=lambda f: int(re.search(r"(\d+)_steps", f.name).group(1)))
        ckpts = [c for c in ckpts
                 if (int(re.search(r"(\d+)_steps", c.name).group(1)) // 1_000_000 - 2) % args.every == 0
                 or c == ckpts[-1]]
        jobs = [(str(c), r, args.episodes, args.seed) for c in ckpts for r in (False, True)]
        print(f"{len(ckpts)} checkpoints x 2 conditions x {args.episodes} episodes")
        data = []
        with ProcessPoolExecutor(args.workers) as pool:
            for d in pool.map(evaluate, jobs):
                data.append(d)
                tag = "random" if d["randomize"] else "clean "
                print(f"{d['steps'] / 1e6:5.0f}M {tag} success {d['success']}/{d['episodes']} fell {d['fell']}")
        out_json.write_text(json.dumps(data, indent=1))
    data = json.loads(out_json.read_text())
    plot(data, args.run / "nav_success.png")


if __name__ == "__main__":
    main()
