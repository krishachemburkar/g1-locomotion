# G1 Locomotion (MuJoCo + PPO)

Trains the Unitree G1 (29 DoF + Dex3 hands, same model as `harvest_scene`) to walk
by following a velocity command (forward, sideways, turn).

## Why PPO (not SAC)

PPO is the standard for legged locomotion (legged_gym, Unitree's unitree_rl_gym,
MuJoCo Playground). It scales with many parallel sims, trains stably, and gives
smoother gaits. SAC is a better fit for single-env manipulation like pick-and-place.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python build_model.py      # writes assets/unitree_g1/g1_loco.xml
```

## Files

| File | What it does |
|---|---|
| `build_model.py` | Makes the locomotion model: 12 leg joints move, upper body + hands rigid (mass kept), only foot spheres collide |
| `g1_env.py` | Gymnasium env: 50 Hz policy, 200 Hz physics, PD control, gait clock, Unitree-style rewards, domain randomization |
| `train.py` | SB3 PPO with one env per CPU core |
| `play.py` | Viewer with arrow-key steering, or headless video |

## Train

```sh
.venv/bin/python train.py --name g1_walk --steps 50e6
.venv/bin/tensorboard --logdir runs          # watch progress
```

Each run gets its own folder `runs/<name>_<date-time>/` (checkpoints, `final.zip`,
`config.json` with the exact reward settings, `log.txt`, `progress.csv`). Nothing is overwritten.

Signs it is learning: `walk/success_rate` (episodes lasting 20 s without falling) climbs
toward 1, and `walk/vel_error` (m/s off the command) drops toward ~0.1–0.2.

Run history:
- `runs/g1_walk/` — first run, stopped at 10M steps. Learned to stand still and ignore commands.
- `walk_v2_*` — gait rewards no longer pay feet that stay on the floor; higher entropy bonus.

## Watch it

```sh
.venv/bin/mjpython play.py runs/g1_walk/final.zip                     # viewer, arrow keys steer
.venv/bin/python play.py runs/g1_walk/final.zip --video walk.mp4 --vx 0.5
```

Checkpoints are saved every 2M steps in `runs/<name>/checkpoints/`.

## Design

- **Action**: 12 leg joint offsets; target = default pose + 0.25 × action → PD torques (Unitree's G1 gains).
- **Observation (47)**: gyro, gravity direction, command, joint pos/vel, last action, gait clock sin/cos.
  All of these are measured on the real robot (no base linear velocity), so the policy can move to hardware later.
- **Rewards**: velocity tracking, plus penalties for tilt, height error, jerky actions, foot slip, hip splay,
  and a reward for foot contacts that match the gait clock (0.8 s period, alternating legs).
- **Randomization**: floor friction, torso payload (−1 to +3 kg), random pushes every 5 s, sensor noise.

## Next steps

1. Get a stable walk on flat ground.
2. Let the arms move (or randomize arm poses) so walking still works while the arms carry the basket or reach for apples.
3. Add orchard terrain (slopes, bumps) and curriculum.
4. Combine: walk to the tree → stand → run the pick policy.

## Qflex experiment

A separate official-Qflex training runner is available in `train_qflex.py`.
See [docs/QFLEX.md](docs/QFLEX.md) for isolated setup, a smoke run,
training commands, outputs and comparison guidance. PPO remains available
through `train.py`.
