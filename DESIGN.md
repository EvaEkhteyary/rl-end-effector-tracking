# Design Note — RL End-Effector Trajectory Tracking

This note explains *how the system thinks*: the core idea, the MDP design, the
trajectory representation, and how tracking is evaluated.

The robot is a **Sawyer** 7-DOF arm (Rethink Robotics) fitted with a
**Robotiq-85** gripper, simulated in **MuJoCo**. The model is vendored into
`assets/` (validated kinematics + meshes) so the project is fully
self-contained.

---

## 1. The core idea: residual reinforcement learning

A naive RL formulation makes the policy output raw joint commands and learn the
whole controller from scratch. For *trajectory tracking* that is wasteful and
fragile: the policy spends most of its capacity re-discovering inverse
kinematics, and the resulting motion is hard to keep jitter-free.

Instead, the controller is split into two halves:

```
trajectory ─▶┌ model-based base controller ─┐─ q̇_base ──┐
             │ (resolved-rate, DLS Jacobian)│           │
             └──────────────────────────────┘           ├▶ ⊕ ─▶ low-pass ─▶ arm
observation ▶┌ RL policy (PPO) ─────────────┐─ q̇_resid ─┘       filter
             │ learns a bounded *correction*│
             └──────────────────────────────┘

   q̇_command = lowpass( q̇_base  +  0.6 · π(observation) )
```

* **The base controller** is a resolved-rate motion controller. It inverts the
  geometric Jacobian with damped least squares (DLS) to turn a desired
  end-effector twist into joint velocities. On its own it already tracks a
  trajectory *roughly* — but it lags, because it is a kinematic controller
  driving an arm that has real second-order dynamics, sensor noise and control
  delay.

* **The RL policy** outputs only a small, bounded residual (±0.6 rad/s). It
  cannot re-learn kinematics and it cannot command violent motion. Its entire
  job is to compensate for what the base controller structurally cannot handle:
  control delay, dynamics lag, sensor noise, and behaviour near singular /
  unreachable configurations.

* **A first-order low-pass filter** on the *final* joint-velocity command is
  the hard guarantee of smooth motion: sensor noise, action noise and any
  high-frequency content in the residual are attenuated before they ever reach
  the actuators. The filter introduces a small lag — and compensating lag is
  precisely what the RL policy (with its trajectory preview) is good at. The
  two halves are complementary: the filter removes jitter, the policy removes
  the lag the filter (and the dynamics) introduce.

**Why this is the right design for the brief:**

| Requirement            | How residual RL addresses it                          |
|------------------------|-------------------------------------------------------|
| Accuracy over time     | RL closes the lag of the (filtered) base controller   |
| Smoothness / no jitter | bounded residual **+ low-pass command filter** → smooth *by construction* |
| Robustness             | RL is *trained on* delay/noise; base controller gives a stable fallback |
| Clarity & simplicity   | Each half is simple and individually testable         |

It also yields a clean, honest evaluation: run every scenario with the residual
**off** (pure base controller) and **on**. The gap is exactly what RL bought —
no hand-waving.

---

## 2. Trajectory representation

Trajectories are **analytic, time-parameterised functions** `f(t) → 6-DOF pose`
(`src/trajectories.py`). Three families:

| Name      | Definition                                              |
|-----------|---------------------------------------------------------|
| `circle`  | circle in a *tilted* plane (genuinely 3-D, not axis-aligned) |
| `figure8` | Gerono lemniscate — the hardest case (sharp curvature reversals) |
| `random`  | smooth "moving target": a low-frequency sum of sinusoids per axis |

Each trajectory exposes `position(t)`, `velocity(t)` (analytic derivative),
`orientation(t)` (quaternion), and `angular_velocity(t)`. The orientation
slowly oscillates around a downward-pointing pose so orientation tracking is
non-trivial yet always reachable.

Two consequences of choosing an *analytic* representation:

1. **Exact feed-forward velocity** — the base controller gets a clean
   `ẋ_desired`, no numerical-differentiation noise.
2. **Free look-ahead** — the trajectory can be sampled at any future time, so
   the policy is given a **preview window** of upcoming targets (see §3). This
   is what lets the policy *anticipate* instead of *chase* — the single biggest
   lever for low-lag tracking.

Trajectories are anchored on the arm's home end-effector position so the start
is always inside the workspace.

---

## 3. MDP design

### State — 70-dimensional observation

| Block | Dim | Why it is included |
|-------|-----|--------------------|
| joint angles (normalised)            | 7  | configuration |
| joint velocities                     | 7  | dynamics state |
| current EE **position** error        | 3  | what to fix now |
| current EE **orientation** error     | 3  | axis-angle, what to fix now |
| **preview**: 5 future target positions   | 15 | anticipation |
| **preview**: 5 future target orientations | 15 | anticipation |
| desired EE linear velocity (feed-forward) | 3  | where the target is heading |
| desired EE angular velocity (feed-forward)| 3  | where the target is heading |
| base controller's proposed joint velocity | 7  | *what the residual is correcting* |
| previous residual action             | 7  | lets the policy stay smooth |

Two deliberate, slightly unusual choices: the policy is told **the base
controller's output** (so it reasons about a *correction*, not an absolute
command) and **its own previous action** (so it can penalise its own jerk).

All observations are read through the *noisy* sensor model — the policy never
sees ground truth.

### Action — bounded residual joint velocity

7-D continuous action in `[-1, 1]`, scaled by `0.6 rad/s` and **added** to the
base controller's joint-velocity command. The sum is low-pass filtered,
integrated into a joint position setpoint, and tracked by the arm's PD
actuators. Bounding the residual and filtering the command are the two
structural guarantees of smooth, stable motion — no reward tuning required for
the policy to be *unable* to produce violent motion.

### Reward

```
r =  w_pos · exp(−(e_pos/σ_pos)²)        # position tracking      (0..1)
   + w_ori · exp(−(e_ori/σ_ori)²)        # orientation tracking   (0..1)
   + bonus · 1[e_pos < 1 cm]             # precision incentive
   − w_a  · ‖a‖²                         # penalise large residuals
   − w_Δa · ‖a − a_prev‖²               # penalise residual *jerk*  ← anti-jitter
   − w_v  · ‖q̇‖²                         # penalise raw joint speed
```

with `w_pos=1.0, w_ori=0.35, σ_pos=5 cm, σ_ori=0.35 rad, bonus=0.5,
w_a=0.015, w_Δa=0.05, w_v=0.004`.

Design reasoning:

* **Gaussian kernels**, not raw `−error`. They are bounded and smooth, so the
  reward never explodes when the target is briefly unreachable — the policy
  degrades gracefully instead of going unstable.
* The **precision bonus** sharpens behaviour in the last millimetre, where a
  Gaussian kernel is nearly flat.
* Three **smoothness penalties** target jitter directly. The residual-jerk term
  `‖a − a_prev‖²` is the most important one. Because the worst case for that
  term is exactly the safe fallback (zero residual = pure base controller),
  smoothness shaping can never destabilise training.

---

## 4. Uncertainty model

All three requested kinds of uncertainty are implemented and **randomised every
episode** during training (`domain_rand`):

| Source            | Implementation                                            |
|-------------------|-----------------------------------------------------------|
| Observation noise | Gaussian noise on joint encoders + EE pose sensing; the base controller's FK **and Jacobian** are computed from the noisy angles, so model mismatch propagates realistically |
| Action noise      | Gaussian noise added to the applied joint-velocity command |
| Control delay     | the command is buffered and applied **0–3 control steps late** (0–150 ms) |
| Unreachable goals | 22 % of episodes scale the path 1.7–2.5× so part of it leaves the dexterous workspace |

Domain randomisation over delay, noise magnitude, trajectory type, speed and
scale is what makes the single trained policy robust to all of them at once.

---

## 5. Training

* **Algorithm:** PPO (Stable-Baselines3). The residual problem has a dense,
  well-shaped reward and a benign bounded action space, so throughput matters
  more than sample efficiency — PPO parallelises across CPU cores and yields
  stable, smooth policies.
* 8 parallel environments, 2 M steps, ~12 min on an 8-core CPU.
* `γ=0.98`, GAE `λ=0.95`, 256×256 MLP, learning rate linearly decayed to 0,
  `log_std_init=−1.0` (gentle initial exploration → the policy starts close to
  the base controller and improves from there).

---

## 6. Evaluation methodology

Tracking is evaluated along several axes (`src/evaluate.py`):

* **Accuracy over time** — position RMSE / mean / max / final error, and
  orientation RMSE / max, over a full trajectory (not a single set-point).
* **Smoothness** — mean joint **jerk** (`rad/s³`) and end-effector
  speed-profile jitter. Lower = smoother.
* **Robustness** — a control-delay sweep (0 / 50 / 100 / 200 ms).
* **Graceful failure** — an "unreachable" stress test where the path is scaled
  far beyond the workspace.

The discipline that makes the numbers trustworthy: **every scenario is run
twice under identical, seeded noise** — base controller alone vs. base + RL
residual. The generated figures and `results/metrics.json` are summarised in
the README.

---

## 7. What I would do next

* **Torque-level residual** instead of velocity-level, for true dynamic
  compensation.
* **Recurrent policy** to *estimate* the control delay online rather than
  relying on domain randomisation to average over it.
* **Sim-to-real**: the base controller is a real, deployable controller, so
  only the residual would need careful transfer — a natural advantage of this
  architecture.
