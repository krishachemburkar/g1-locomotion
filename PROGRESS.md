# G1 Locomotion: Progress Update

*Updated 2026-10-30. MuJoCo + PPO, Unitree G1 (29 DoF + Dex3 hands)*

## Summary

**Progress so far**

1. **Walking policy trained in MuJoCo.** Trained a PPO policy that controls the G1's 12 leg joints using only sensors the real robot has. It walks at the commanded speed, turns and side-steps, and stayed upright in 99 of 100 test episodes with randomized friction, payload, pushes and sensor noise.
2. **Fixed the reward and improved the gait.** The first run learned to stand still because the gait reward paid it for never lifting its feet. Reworked the gait rewards and retrained: clean alternating walk, and a heading correction cuts drift from about 30° to 3° over 15 s.

**Next steps**

3. **Realistic motors and randomization.** Add motor delay, varied motor strength and gains, mass changes and joint calibration offsets, to close the main gap between sim and the real robot.
4. **Better gait and orchard terrain.** Remove a slight limp with left-right symmetry, add a stand-still mode, and train on rough and sloped ground that gets harder over time.
5. **Upper body and carrying.** Simulate the waist, arms and hands as moving joints instead of fixed ones, and add a basket load so the robot can walk while carrying apples.
6. **Point-to-point navigation.** Right now the robot follows random speed and turn commands. Add a goal position so it walks from point A to point B, steering and stopping on its own.
7. **Testing and deployment.** Check the policy in a second simulator (Isaac Sim), write the real-robot control loop with safety stops, then test on hardware: gantry, standing, slow walk.

---

## Reward function

Every 20 ms policy step, the reward is a weighted sum of the terms below. Each term is multiplied by its weight and by the step time `dt = 0.02 s` (as in legged_gym). The total is clipped at zero, so falling early never pays.

```
r_t = max(0, dt · Σ_i  w_i · r_i)
```

An episode lasts 20 s (1000 steps). It ends early with a **fall** if the pelvis drops below 0.4 m, the body pitches more than about 57°, or it rolls more than about 44°.

**Notation:**
- **v, ω**: pelvis linear and angular velocity, in the body frame.
- **c**: command (vₓ, v_y, ω_z).
- **g**: gravity direction seen by the IMU.
- **q, q̇, τ**: leg joint angles, velocities and torques.
- **a**: action.

### 1. Task: follow the command

| Term | Formula | Weight | Purpose |
|---|---|---|---|
| Linear velocity tracking | exp(−‖c_xy − v_xy‖² / 0.25) | **+1.0** | Walk at the commanded forward/sideways speed |
| Yaw rate tracking | exp(−(c_yaw − ω_z)² / 0.25) | **+0.5** | Turn at the commanded rate |
| Alive | 1 | **+0.15** | Small bonus for every step without falling |

### 2. Gait: alternating steps on a clock

A 0.8 s gait clock drives both legs, with the right leg half a period behind the left. For each leg, the first 55% of its cycle is **stance** (foot down) and the last 45% is **swing** (foot in the air).

| Term | Formula | Weight | Purpose |
|---|---|---|---|
| Contact timing | Σ over legs: +1 if the foot contact matches the clock, −1 if not | **+0.3** | Feet down in stance and up in swing. The −1 stops "never lift a foot" from scoring. |
| Swing height | Σ over swinging legs: exp(−(z_foot − h(s))² / 0.0025), with h(s) = 0.08 · sin(π s) and s = swing progress from 0 to 1 | **+1.0** | The foot follows a smooth lift-and-place arc that peaks at 8 cm. Paid only during swing. |
| Air time | At each touchdown: min(t_air, 0.5) − 0.25 s. Only when told to move. | **+1.0** | Rewards real steps, not shuffling |
| Foot slip | Σ over feet in contact: ‖v_foot‖² | **−0.2** | Planted feet should not slide |
| Foot spacing | max(0, 0.18 − lateral distance between feet) | **−1.0** | Keeps the feet at least 18 cm apart so the legs don't cross |
| Hip splay | Σ (hip roll² + hip yaw²) | **−1.0** | Legs stay straight under the body; no frog-legging or toe-in |

### 3. Stability: keep the body steady

| Term | Formula | Weight | Purpose |
|---|---|---|---|
| Orientation | g_x² + g_y² | **−1.0** | Torso stays upright |
| Body height | (z_pelvis − 0.78)² | **−10.0** | Pelvis stays at walking height, no crouching |
| Vertical bounce | v_z² | **−2.0** | No bobbing up and down |
| Roll/pitch rate | ω_x² + ω_y² | **−0.05** | No wobbling |

### 4. Smoothness and hardware safety: motor-friendly motions

| Term | Formula | Weight | Purpose |
|---|---|---|---|
| Action rate | Σ (a_t − a_{t−1})² | **−0.01** | No jittery commands from one step to the next |
| Joint acceleration | Σ q̈² | **−2.5e-7** | Smooth joint motion |
| Joint velocity | Σ q̇² | **−1e-3** | No unnecessarily fast joints |
| Torque | Σ τ² | **−1e-5** | Less energy and motor heating |
| Joint limits | Σ amount past 90% of each joint's range | **−5.0** | Stay clear of the mechanical stops |

### How the policy turns actions into torques

```
target_q = default_pose + 0.25 · (5 · a)          a ∈ [−1, 1], 12 leg joints
τ        = Kp · (target_q − q) − Kd · q̇           at 200 Hz, clipped to motor limits
Kp = 100 (hip), 150 (knee), 40 (ankle);  Kd = 2, 4, 2
```

**Lesson from run 1:** the original gait reward gave +1 per leg whenever contact matched the clock. A robot that never lifted its feet was already "right" 55% of the time, so standing still was a safe local optimum. Paying swing height only during swing, giving −1 for a wrong contact, and adding the air-time bonus removed that optimum.
