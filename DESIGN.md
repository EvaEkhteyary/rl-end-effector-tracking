# Design Note — RL End-Effector Trajectory Tracking

Robot: **Sawyer** 7-DOF + **Robotiq-85** gripper, simulated in **MuJoCo**.

---

## 1. Core idea: residual RL

```
trajectory ──▶ base controller (DLS Jacobian) ──▶ q̇_base ──┐
observation ──▶ RL policy (PPO, bounded residual) ──────────┤▶ ⊕ ──▶ low-pass ──▶ arm
                                                  q̇_resid ──┘
   q̇_command = lowpass( q̇_base  +  0.6 · π(observation) )
```

The **base controller** (resolved-rate, DLS) handles kinematics. The **RL policy** outputs a bounded residual (±0.6 rad/s) to compensate for what the base controller cannot: control delay, dynamics lag, and near-singular behaviour. A **low-pass filter** on the final command guarantees smooth motion by construction. The policy anticipates lag; the filter removes jitter — the two halves are complementary.

---

## 2. Trajectory representation

Analytic, time-parameterised `f(t) → 6-DOF pose` (`src/trajectories.py`):

| Name      | Definition |
|-----------|------------|
| `circle`  | circle in a tilted plane (genuinely 3-D) |
| `figure8` | Gerono lemniscate (sharp curvature reversals) |
| `random`  | low-frequency sum of sinusoids per axis |

Each exposes `position(t)`, `velocity(t)`, `orientation(t)`, `angular_velocity(t)`. Analytic representation gives exact feed-forward velocity and free look-ahead for the policy preview window.

---

## 3. MDP design

**Observation — 70D:**

| Block | Dim |
|-------|-----|
| joint angles (normalised) + velocities | 14 |
| EE position + orientation error (axis-angle) | 6 |
| preview: 5 future target positions + orientations | 30 |
| desired EE linear + angular velocity (feed-forward) | 6 |
| base controller's proposed joint velocity | 7 |
| previous residual action | 7 |

The policy sees the base controller's output (so it reasons about a correction) and its own previous action (to penalise jerk). All observations pass through the noisy sensor model.

**Action:** 7-D continuous in `[-1, 1]`, scaled by 0.6 rad/s, added to base controller output, then low-pass filtered.

**Reward:**
```
r =  1.0 · exp(−(e_pos/0.05)²)     # position tracking
   + 0.35 · exp(−(e_ori/0.35)²)    # orientation tracking
   + 0.5  · 1[e_pos < 1 cm]        # precision bonus
   − 0.015 · ‖a‖²                  # penalise large residuals
   − 0.05  · ‖a − a_prev‖²         # penalise residual jerk  ← key anti-jitter term
   − 0.004 · ‖q̇‖²                  # penalise joint speed
```

Gaussian kernels keep the reward bounded when targets are briefly unreachable. Zero residual (pure base controller) is always the safe fallback.

---

## 4. Uncertainty model

Randomised every episode:

| Source | Implementation |
|--------|----------------|
| Observation noise | Gaussian on encoders + EE pose; propagates through FK and Jacobian |
| Action noise | Gaussian on applied joint-velocity command |
| Control delay | 0–3 steps late (0–150 ms) |
| Unreachable goals | 22% of episodes scale path 1.7–2.5× outside dexterous workspace |

---

## 5. Training

PPO (Stable-Baselines3), 8 parallel envs, 2M steps, ~12 min on 8-core CPU.
`γ=0.98`, GAE `λ=0.95`, 256×256 MLP, linear LR decay, `log_std_init=−1.0`.

---

## 6. Evaluation

Each scenario run twice under identical seeded noise — base controller alone vs. base + RL residual:

- **Accuracy:** position RMSE / mean / max / final error; orientation RMSE / max
- **Smoothness:** mean joint jerk (rad/s³), EE speed-profile jitter
- **Robustness:** control-delay sweep (0 / 50 / 100 / 200 ms)
- **Graceful failure:** path scaled beyond workspace
