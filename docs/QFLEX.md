# Qflex on G1

`train_qflex.py` calls the official Qflex `FlowExp` algorithm and flow network
directly. It uses the existing G1 rewards, observations, PD controller and
randomization. `train.py` remains the PPO baseline. The initial runner uses one
simulation environment and one update per transition after random warmup.
It does not use upstream's Linux-specific vector environment extensions.

## Setup

From the g1-locomotion directory, use a separate environment (Python 3.10–3.12):

```sh
git clone https://github.com/LNSGroup/Qflex.git vendor/Qflex
git -C vendor/Qflex checkout a028cfa76525a62bfb90d76c6f1535f697121383
python3 -m venv .venv-qflex
.venv-qflex/bin/pip install -r requirements-qflex.txt
# Only if assets/unitree_g1/g1_loco.xml has not been built:
.venv-qflex/bin/python build_model.py
```

The runner imports the checkout directly; do not run Qflex's `setup.py` on macOS.
The pinned JAX wheels use CPU on macOS. For an NVIDIA Linux machine, replace
the JAX install with `pip install 'jax[cuda12]==0.4.35'` inside this environment.

## Smoke run and training

```sh
.venv-qflex/bin/python train_qflex.py --qflex-root vendor/Qflex \
  --steps 40 --start-step 16 --batch-size 8 --buffer-size 64 \
  --hidden-dim 32 --flow-steps 2 --grad-step-num 2 \
  --eval-every 40 --eval-episodes 1 --save-every 40 --name qflex_smoke

.venv-qflex/bin/python train_qflex.py --qflex-root vendor/Qflex \
  --steps 5000000 --seed 0 --name qflex_walk
```

JAX compiles the first update; a smoke run checks plumbing, not walking quality.
`--steps` counts all environment transitions, including warmup. A warmup longer
than the run deliberately produces no learning updates.

Each unique `runs/qflex_walk_<timestamp>/` contains:

- `config.json`: arguments, environment config, upstream commit, devices.
- `updates.csv`: scalar update metrics every 1000 transitions.
- `evaluation.csv`: seeded stochastic rollout return, fraction of episodes without
  falls, and velocity tracking error; held-out reset seeds repeat across evaluations.
- `state-*.pkl`: upstream algorithm state, including optimizer state.
- `policy-*.pkl`: upstream policy parameters/state; final files are also saved
  when the loop exits, including interruption. These are not SB3 ZIP files and
  cannot be opened with the existing PPO `play.py`.

Only falls mask bootstrapping. Time-limit transitions keep their final observation
and bootstrap before resetting. The upstream deterministic-action helper contains a `jnp.random` typo, so this
runner evaluates stochastic actions with repeatable, independent JAX seeds.
Evaluation uses its own environment and does not
change the training environment's random state. Resume and standalone playback
are not implemented in this initial runner.

## Experiment scope

Compare PPO and Qflex at equal transition budgets and multiple training seeds,
with the same held-out evaluation reset seeds and environment settings. Report
return, velocity error, fall-free episode rate and wall time. The existing PPO
training success metric uses training episodes, so it is not by itself an equal
held-out comparison with this runner's evaluation metric.

This implements Qflex exploration for locomotion. The scheduled PDE and boundary
losses from https://arxiv.org/abs/2608.17117 are a separate research extension.
They require a reach-avoid objective, a suitable state/dynamics/noise model and
handling of contact transitions; the current dense tracking return is not a
reach-avoid probability.
