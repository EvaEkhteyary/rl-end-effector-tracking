"""Evaluate the trained policy and produce the figures / metrics.

Everything here answers three questions the challenge asks about:

  1. Tracking accuracy over time  -> RMSE / max / final position & orientation
  2. Smoothness and stability     -> joint-jerk and EE-speed-jitter metrics
  3. Robustness to noise/mismatch -> a control-delay sweep, base vs. residual

Every scenario is run twice under *identical* (seeded) noise: once with the
RL residual disabled (pure model-based base controller) and once with it
enabled.  The gap between the two is exactly what the RL component buys.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .arm_env import ArmTrackingEnv
from .config import EnvConfig

SCENARIOS = [
    ("circle",  dict(traj="circle")),
    ("figure8", dict(traj="figure8")),
    ("random",  dict(traj="random")),
    ("unreachable", dict(traj="circle", scale=1.6)),   # path leaves the workspace
]
EVAL_SEED = 12345


# ---------------------------------------------------------------------------
# Rollout + metrics
# ---------------------------------------------------------------------------
def rollout(env, model, options):
    """Run one full episode. ``model=None`` -> base controller only."""
    obs, _ = env.reset(seed=EVAL_SEED, options=options)
    log = defaultdict(list)
    done = False
    while not done:
        if model is None:
            action = np.zeros(7, dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        for key in ("pos_err", "ori_err_deg", "ee_pos", "des_pos",
                    "joint_vel", "action"):
            log[key].append(info[key])
        log["reward"].append(reward)
        done = term or trunc
    return {k: np.asarray(v) for k, v in log.items()}


def metrics(log, dt):
    perr = log["pos_err"]
    oerr = log["ori_err_deg"]
    vel = log["joint_vel"]
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    ee_speed = np.linalg.norm(np.diff(log["ee_pos"], axis=0) / dt, axis=1)
    return {
        "pos_rmse_mm": float(np.sqrt(np.mean(perr ** 2)) * 1e3),
        "pos_mean_mm": float(np.mean(perr) * 1e3),
        "pos_max_mm": float(np.max(perr) * 1e3),
        "pos_final_mm": float(perr[-1] * 1e3),
        "ori_rmse_deg": float(np.sqrt(np.mean(oerr ** 2))),
        "ori_max_deg": float(np.max(oerr)),
        "joint_jerk": float(np.mean(np.linalg.norm(jerk, axis=1))),
        "ee_speed_jitter": float(np.std(np.diff(ee_speed))),
        "return": float(np.sum(log["reward"])),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_3d(runs, path):
    fig = plt.figure(figsize=(15, 11))
    for i, (name, _) in enumerate(SCENARIOS):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        des = runs[name]["rl"]["des_pos"]
        rl = runs[name]["rl"]["ee_pos"]
        base = runs[name]["base"]["ee_pos"]
        ax.plot(*des.T, color="#1f77b4", lw=3, label="desired", alpha=0.9)
        ax.plot(*base.T, color="#999999", lw=1.4, ls="--", label="base controller")
        ax.plot(*rl.T, color="#ff7f0e", lw=1.8, label="base + RL residual")
        ax.scatter(*des[0], color="green", s=40)
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.legend(fontsize=8, loc="upper left")
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    fig.suptitle("End-effector trajectory tracking (3D)", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_error_over_time(runs, dt, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (name, _) in zip(axes.flat, SCENARIOS):
        base = runs[name]["base"]["pos_err"] * 1e3
        rl = runs[name]["rl"]["pos_err"] * 1e3
        t = np.arange(len(rl)) * dt
        ax.plot(t, base, color="#999999", lw=1.4, label="base controller")
        ax.plot(t, rl, color="#ff7f0e", lw=1.8, label="base + RL residual")
        ax.axhline(np.sqrt(np.mean((rl / 1e3) ** 2)) * 1e3, color="#ff7f0e",
                   ls=":", lw=1, alpha=0.7)
        ax.set_title(f"{name}", fontweight="bold")
        ax.set_xlabel("time [s]"); ax.set_ylabel("position error [mm]")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Tracking error over time", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_xyz(run_base, run_rl, dt, path):
    des = run_rl["des_pos"]
    base = run_base["ee_pos"]
    rl = run_rl["ee_pos"]
    t = np.arange(len(des)) * dt
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i, lab in enumerate("xyz"):
        ax = axes[i]
        ax.plot(t, des[:, i], color="#1f77b4", lw=2.5, label="desired")
        ax.plot(t, base[:, i], color="#999999", lw=1.3, ls="--", label="base")
        ax.plot(t, rl[:, i], color="#ff7f0e", lw=1.6, label="base + RL")
        ax.set_ylabel(f"{lab} [m]"); ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9, ncol=3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Per-axis position tracking (figure-8)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_robustness(delays, base_rmse, rl_rmse, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(delays))
    ax.bar(x - 0.2, base_rmse, 0.4, color="#999999", label="base controller")
    ax.bar(x + 0.2, rl_rmse, 0.4, color="#ff7f0e", label="base + RL residual")
    for i, (b, r) in enumerate(zip(base_rmse, rl_rmse)):
        ax.text(i - 0.2, b + 0.5, f"{b:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.2, r + 0.5, f"{r:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d} step{'' if d == 1 else 's'}\n({d*50} ms)"
                        for d in delays])
    ax.set_xlabel("control delay"); ax.set_ylabel("position RMSE [mm]")
    ax.set_title("Robustness to control delay (figure-8 trajectory)",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_smoothness(runs, dt, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # left: EE speed profile (circle) -- visual smoothness
    for label, color in (("base", "#999999"), ("rl", "#ff7f0e")):
        ee = runs["circle"][label]["ee_pos"]
        speed = np.linalg.norm(np.diff(ee, axis=0) / dt, axis=1)
        t = np.arange(len(speed)) * dt
        name = "base controller" if label == "base" else "base + RL residual"
        axes[0].plot(t, speed, color=color, lw=1.6, label=name)
    axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("EE speed [m/s]")
    axes[0].set_title("End-effector speed profile (circle)", fontweight="bold")
    axes[0].grid(alpha=0.3); axes[0].legend()
    # right: joint-jerk metric per scenario
    names = [s[0] for s in SCENARIOS]
    base_j = [runs[n]["base"]["metrics"]["joint_jerk"] for n in names]
    rl_j = [runs[n]["rl"]["metrics"]["joint_jerk"] for n in names]
    x = np.arange(len(names))
    axes[1].bar(x - 0.2, base_j, 0.4, color="#999999", label="base controller")
    axes[1].bar(x + 0.2, rl_j, 0.4, color="#ff7f0e", label="base + RL residual")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_ylabel("mean joint jerk [rad/s³]")
    axes[1].set_title("Motion smoothness (lower = smoother)", fontweight="bold")
    axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_training_curve(logdir, path):
    csv = Path(logdir) / "progress.csv"
    if not csv.exists():
        return
    import csv as _csv
    rows = list(_csv.DictReader(open(csv)))
    if not rows:
        return

    def col(key):
        xs, ys = [], []
        for r in rows:
            v = r.get(key, "")
            if v not in ("", None):
                xs.append(float(r["time/total_timesteps"]))
                ys.append(float(v))
        return np.array(xs), np.array(ys)

    fig, ax = plt.subplots(figsize=(9, 5))
    xr, yr = col("rollout/ep_rew_mean")
    xe, ye = col("eval/mean_reward")
    if len(yr):
        ax.plot(xr, yr, color="#ff7f0e", lw=1.8, label="train episode reward")
    if len(ye):
        ax.plot(xe, ye, color="#1f77b4", lw=2, marker="o", ms=3,
                label="eval reward")
    ax.set_xlabel("environment steps"); ax.set_ylabel("episode reward")
    ax.set_title("PPO training curve", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate the tracking policy.")
    ap.add_argument("--model", type=str, default="models/best_model.zip")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--logdir", type=str, default="runs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = None
    if Path(args.model).exists():
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        print(f"[eval] loaded policy: {args.model}")
    else:
        print(f"[eval] no model at {args.model} -> evaluating base controller only")

    cfg = EnvConfig(domain_rand=False)
    env = ArmTrackingEnv(cfg)
    dt = env.dt_ctrl

    # ---- per-scenario rollouts: base vs. RL -------------------------------
    runs = {}
    print(f"\n{'scenario':12s} {'controller':20s} "
          f"{'RMSE':>9s} {'max':>9s} {'final':>9s} {'ori RMSE':>10s} {'jerk':>9s}")
    print("-" * 84)
    for name, opts in SCENARIOS:
        runs[name] = {}
        for label, mdl in (("base", None), ("rl", model)):
            log = rollout(env, mdl, dict(opts, eval=True))
            log["metrics"] = metrics(log, dt)
            runs[name][label] = log
            m = log["metrics"]
            tag = "base controller" if label == "base" else "base + RL residual"
            print(f"{name:12s} {tag:20s} "
                  f"{m['pos_rmse_mm']:7.2f}mm {m['pos_max_mm']:7.2f}mm "
                  f"{m['pos_final_mm']:7.2f}mm {m['ori_rmse_deg']:8.2f}° "
                  f"{m['joint_jerk']:9.1f}")
        if model is None:
            runs[name]["rl"] = runs[name]["base"]

    # ---- robustness: control-delay sweep on the figure-8 ------------------
    delays = [0, 1, 2, 3, 4]
    base_rmse, rl_rmse = [], []
    print(f"\n{'delay':>10s} {'base RMSE':>12s} {'RL RMSE':>12s} {'improvement':>13s}")
    print("-" * 50)
    for d in delays:
        lb = rollout(env, None, dict(traj="figure8", delay=d, eval=True))
        lr = rollout(env, model, dict(traj="figure8", delay=d, eval=True))
        rb = metrics(lb, dt)["pos_rmse_mm"]
        rr = metrics(lr, dt)["pos_rmse_mm"]
        base_rmse.append(rb)
        rl_rmse.append(rr)
        imp = 100.0 * (rb - rr) / rb if rb else 0.0
        print(f"{d*50:8d}ms {rb:10.2f}mm {rr:10.2f}mm {imp:11.1f}%")

    # ---- figures ----------------------------------------------------------
    plot_3d(runs, out / "tracking_3d.png")
    plot_error_over_time(runs, dt, out / "error_over_time.png")
    plot_xyz(runs["figure8"]["base"], runs["figure8"]["rl"], dt,
             out / "xyz_tracking.png")
    plot_robustness(delays, base_rmse, rl_rmse, out / "robustness.png")
    plot_smoothness(runs, dt, out / "smoothness.png")
    plot_training_curve(args.logdir, out / "training_curve.png")
    print(f"\n[eval] figures written to {out}/")

    # ---- metrics.json -----------------------------------------------------
    summary = {
        name: {lab: runs[name][lab]["metrics"] for lab in ("base", "rl")}
        for name, _ in SCENARIOS
    }
    summary["robustness_delay"] = {
        "delay_ms": [d * 50 for d in delays],
        "base_rmse_mm": base_rmse,
        "rl_rmse_mm": rl_rmse,
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] metrics written to {out / 'metrics.json'}")
    env.close()


if __name__ == "__main__":
    main()
