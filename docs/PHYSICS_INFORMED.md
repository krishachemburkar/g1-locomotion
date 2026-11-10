# Experimental physics-informed Qflex

This opt-in experiment adapts the scheduled critic training in
[Hoshino and Nakahira, arXiv:2608.17117](https://arxiv.org/abs/2608.17117),
equations 39–44. It adds a MuJoCo dynamics-generator residual and boundary
losses to Qflex's two actual critics. The next Qflex update uses those modified
critics when learning its flow/reference policy. This is executable training
code, not just reward shaping or a separately trained diagnostic model.

## Run

Use the isolated environment and pinned Qflex checkout in QFLEX.md:

```sh
.venv-qflex/bin/python train_qflex.py --qflex-root vendor/Qflex \
  --physics-informed --steps 2000000 --seed 0 --name qflex_pirl \
  --physics-start 100000 --physics-ramp 100000 \
  --physics-every 100 --physics-batch-size 8

.venv-qflex/bin/python -m unittest test_physics_informed -v
```

`--physics-informed` selects a **different task and observation space**:

- Reach a goal 1 m ahead of the initial heading, within a 0.25 m radius,
  before falling or the 20-second deadline. Reward is 1 on successful reach,
  otherwise 0. A fall takes precedence over reaching the goal.
- All episode endings, including the deadline, are Bellman terminals. Gamma is
  1; remaining time is an explicit observation.
- Observations are privileged `[qpos, qvel, remaining_seconds, goal_x, goal_y]`.
  This is a simulator experiment, not the hardware-compatible 47-input policy.
- Payload/friction randomization, pushes and observation noise are disabled.
  The initial pose/velocity still varies by reset seed; dynamics are deterministic
  conditional on the state and action.

Leaving the flag off retains the velocity-tracking Qflex experiment. Running
this command does not convert or resume an existing tracking checkpoint. The
already-running Mac baseline process is not restarted by installing this code.

## Losses and schedule

Qflex performs its normal TD/flow update. Every `--physics-every` transitions,
an additional critic optimizer step minimizes:

```
eta(step) * [pde_weight * mean((grad_state Q . drift)^2)
            + boundary_weight * mean((Q(boundary, action) - label)^2)]
```

`eta` is zero through `--physics-start`, follows a normalized sigmoid during
`--physics-ramp`, then remains one. TD keeps weight 1 throughout. The two losses
use sequential optimizer steps rather than the paper's single combined TD/PDE/BC
gradient. Defaults are experimental starting values, not tuned paper settings.

Collocation points come from independently sampled replay states with small
velocity perturbations. A separate MuJoCo simulator restores each point, holds
the candidate Qflex action fixed, advances 20 ms under the existing PD controller,
and computes `drift = (next_state - state) / dt`. JAX differentiates Q with respect
to state via JVP and then with respect to critic parameters. The time component
of drift is -1, implementing the remaining-time derivative sign.

Boundary samples are derived from these points: set the goal at the current
position (label 1); set pelvis height below the fall threshold (label 0); or set
remaining time to zero outside the goal (label 0). Boundary actions are held fixed.
The probability boundaries apply to all actions. This first implementation uses
hard target/avoid sets instead of the paper's smoothed target construction.

## Scope and interpretation

This is a **deterministic finite-step generator approximation**, not a complete
implementation of the paper's stochastic continuous-time PDE. It has no diffusion
covariance/Hessian term, LLM schedule search, or numerical safety certificate.
Finite-step drift is biased by integration, contact dynamics and quaternion
coordinates. Samples with changed contact counts or a fallen next state are
skipped as a heuristic; unchanged contact counts do not guarantee smoothness.
Collocation is local to visited states, not uniform coverage of the entire domain.

The default Qflex critics are unbounded, so their outputs must not be presented
as calibrated probabilities without separate rollout calibration and validation.
Sparse reach-avoid learning may need a goal curriculum; successful training is
not established by a smoke test.

`updates.csv` adds `physics_weight`, `physics_pde_loss`,
`physics_boundary_loss`, and `physics_points` (the number of accepted points).
These are the most recent regularizer metrics, recorded every 1000 transitions.
If all points are rejected, that update is skipped and `physics_points` is zero.
`evaluation.csv` adds `reach_rate`. In reach-avoid mode, `success_rate` is only
the no-fall fraction and includes timeouts; use **reach_rate** as task success.
`vel_error` remains a diagnostic against a zero command and is not the objective.

For a fair ablation, run this same reach-avoid task with both physics weights zero
versus their configured values, using multiple training seeds and held-out reset
seeds. Neither result is directly comparable to the existing PPO tracking scores.
The regularizer shares the critic optimizer state, which is included in normal
algorithm checkpoints. Physics-run resumption is deliberately rejected until
environment/replay/schedule restoration is implemented.
