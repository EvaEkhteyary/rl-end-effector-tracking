# RL End-Effector Trajectory Tracking

Teaching a **Sawyer** arm (Rethink Robotics) with a **Robotiq-85** gripper to
follow a moving 3-D Cartesian trajectory — accurately, smoothly, and robustly
to noise and control delay — using **residual reinforcement learning**.

![Sawyer tracking a circle](results/tracking_circle.gif)

> Sawyer + Robotiq-85 tracking a circle — blue ribbon = desired path,
> red sphere = current target, green trail = actual end-effector path.

### ▶ Full demo video — circle, figure-8 & moving target

<video src="https://github.com/EvaEkhteyary/rl-end-effector-tracking/raw/main/results/tracking_demo.mp4" controls muted width="720"></video>

If the embedded player does not load, view
[`results/tracking_demo.mp4`](results/tracking_demo.mp4) directly (it plays in
GitHub's file viewer).

---

## TL;DR — the idea

Instead of asking RL to learn a whole controller from scratch, the policy
learns only a **small bounded correction** on top of a classic model-based
controller:

```
joint command  =  resolved-rate base controller(trajectory)  +  RL residual(observation)
```

The base controller (damped-least-squares Jacobian inverse) does coarse
tracking. The **RL policy (PPO)** learns exactly the part a model-based
controller cannot: compensating control delay, dynamics lag, sensor noise and
near-singular / unreachable configurations.

This buys three things the challenge explicitly asks for:

* **Accuracy** — RL closes the tracking lag of the (filtered) base controller.
* **Smoothness** — a low-pass command filter plus a bounded residual make
  motion jitter-free *by construction*, reinforced by jerk penalties in the reward.
* **Robustness** — the policy is *trained on* randomised delay/noise, and the
  base controller is always a stable fallback.

Full reasoning in **[DESIGN.md](DESIGN.md)**.

---

## Quickstart

Requires **Python 3.10+** (developed on 3.12) and a few hundred MB of pip
dependencies (MuJoCo, Stable-Baselines3, PyTorch — all prebuilt wheels).

```bash
# 1. set up the environment  (use PYTHON=python3.12 etc. if `python3` is older)
make setup PYTHON=python3.12

# 2a. reproduce everything from scratch  (~15 min on an 8-core CPU)
make all

# 2b. ...or just regenerate plots + videos from the included trained model
make demo
```

That's it. `make all` runs training → evaluation → video rendering. The
repository already ships a trained policy (`models/`) and results
(`results/`), so `make demo` works immediately.

Manual invocation (equivalent):

```bash
.venv/bin/python -m src.train      --timesteps 2000000 --n-envs 8
.venv/bin/python -m src.evaluate   # -> results/*.png, results/metrics.json
.venv/bin/python -m src.render     # -> results/tracking_demo.mp4
.venv/bin/python -m src.render --gui   # live interactive viewer
```

---

## Results

Every scenario is evaluated **twice under identical, seeded noise**: the pure
model-based base controller vs. base + RL residual. The gap is what RL adds.

### Tracking accuracy — position RMSE over a full trajectory

| Trajectory      | Base controller | Base + RL residual | Improvement |
|-----------------|----------------:|-------------------:|------------:|
| Circle          | 24.8 mm         | **6.7 mm**         | **−73 %**   |
| Figure-8        | 35.1 mm         | **8.4 mm**         | **−76 %**   |
| Moving target   | 5.3 mm          | 6.2 mm             | −18 %       |
| Unreachable \*  | 288.4 mm        | **51.8 mm**        | **−82 %**   |

\* a circle scaled past the edge of the workspace. The base controller flails
at the joint limits (288 mm); the residual policy degrades gracefully, stays
controlled, and is also ~2.4× smoother there. On the slow *moving target* the
base controller is already near-optimal (~5 mm) so the residual has almost
nothing to add — an honest wash, both well under 1 cm.

### Robustness to control delay — figure-8

| Control delay | 0 ms | 50 ms | 100 ms | 150 ms | 200 ms |
|---------------|-----:|------:|-------:|-------:|-------:|
| Base RMSE     | 29.7 | 35.1  | 41.2   | 49.6   | 66.6   |
| RL RMSE       | 7.6  | 8.4   | 11.4   | 17.7   | 30.8   |
| Improvement   |−74 % |−76 %  | −72 %  | −64 %  | −54 %  |

The residual policy keeps tracking error far below the base controller at
every delay it was trained on (0–200 ms), and joint jerk stays on par with or
below the base controller — accuracy is *not* bought with jitter.

Figures written to `results/`:

| File | Shows |
|------|-------|
| `tracking_3d.png`      | desired vs. actual end-effector path, all 4 scenarios |
| `error_over_time.png`  | position error vs. time, base vs. RL |
| `xyz_tracking.png`     | per-axis tracking of the figure-8 |
| `robustness.png`       | RMSE vs. control delay, base vs. RL |
| `smoothness.png`       | EE speed profile + joint-jerk comparison |
| `training_curve.png`   | PPO learning curve |
| `tracking_demo.mp4`    | rendered video of all three trajectories |

---

## How it works

| Component | File | Role |
|-----------|------|------|
| Trajectories      | `src/trajectories.py`   | analytic circle / figure-8 / moving-target paths (6-DOF) |
| Base controller   | `src/base_controller.py`| resolved-rate DLS Jacobian controller |
| Environment       | `src/arm_env.py`        | MuJoCo sim, residual-RL wrapper, uncertainty injection, reward |
| Robot model       | `assets/sawyer.xml`     | vendored Sawyer + Robotiq-85 — **self-contained, no downloads** |
| Model build       | `tools/build_sawyer_model.py` | one-off script that generated the vendored model |
| Training          | `src/train.py`          | PPO (Stable-Baselines3) |
| Evaluation        | `src/evaluate.py`       | metrics + plots, base-vs-RL ablation |
| Rendering         | `src/render.py`         | MP4 / GIF videos, live viewer |

**State (70-D):** joint angles & velocities, current EE pose error, a 5-step
**preview** of upcoming targets (so the policy *anticipates*), feed-forward
desired twist, the base controller's command, and the previous action.

**Action (7-D):** residual joint velocity, bounded to ±0.6 rad/s, added to the
base command and low-pass filtered before reaching the actuators.

**Reward:** Gaussian position + orientation tracking kernels, a sub-centimetre
precision bonus, and three smoothness penalties (residual magnitude, residual
jerk, joint speed).

**Uncertainty:** observation noise, action noise, control delay (0–150 ms) and
over-sized unreachable paths — all randomised per episode during training.

See [DESIGN.md](DESIGN.md) for the complete rationale.

---

## Repository layout

```
rl-end-effector-tracking/
├── assets/
│   ├── sawyer.xml            self-contained Sawyer + Robotiq-85 model
│   └── sawyer_meshes/        vendored robot meshes
├── src/
│   ├── trajectories.py       desired-trajectory definitions
│   ├── base_controller.py    model-based resolved-rate controller
│   ├── arm_env.py            Gymnasium environment (residual RL)
│   ├── train.py              PPO training
│   ├── evaluate.py           metrics + plots
│   ├── render.py             video rendering
│   ├── config.py             all hyper-parameters in one dataclass
│   └── math_utils.py         quaternion helpers
├── tools/build_sawyer_model.py   regenerates the vendored Sawyer model
├── models/                   trained policy
├── results/                  plots, metrics, videos
├── DESIGN.md                 design note (state/action/reward, trajectory, evaluation)
├── Makefile
└── requirements.txt
```

---

## Notes

* The Sawyer + Robotiq-85 model is **vendored** into `assets/` (validated
  kinematics and meshes), so the project is 100 % self-contained — no asset
  downloads at run time. `tools/build_sawyer_model.py` documents exactly how it
  was generated (it uses `robosuite` purely as a build-time source).
* Contacts are disabled: this is a free-space tracking task, so only the arm's
  gravity, inertia and actuator dynamics matter.

---
