"""Train the residual tracking policy with PPO.

Why PPO?  The residual task has a *dense, well-shaped* reward and a benign
action space (a small bounded correction), so the hard part is wall-clock
throughput, not sample efficiency.  PPO parallelises across CPU cores
extremely well and produces stable, smooth policies -- a good match here.

Usage
-----
    python -m src.train                       # full run (~1.5M steps)
    python -m src.train --timesteps 20000     # quick smoke run
    python -m src.train --n-envs 8 --seed 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .arm_env import make_env
from .config import EnvConfig


def linear_schedule(initial: float):
    """Linearly decay a hyper-parameter to 0 over training (smoother endgame)."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial
    return schedule


def main():
    ap = argparse.ArgumentParser(description="Train the residual tracking policy.")
    ap.add_argument("--timesteps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="models")
    ap.add_argument("--logdir", type=str, default="runs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- environments -----------------------------------------------------
    train_cfg = EnvConfig(seed=args.seed, domain_rand=True)   # uncertainty ON
    eval_cfg = EnvConfig(seed=args.seed + 999, domain_rand=False)  # clean circle

    env_fns = [make_env(train_cfg) for _ in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)
    eval_env = DummyVecEnv([make_env(eval_cfg)])

    # ---- agent ------------------------------------------------------------
    model = PPO(
        "MlpPolicy", vec_env,
        n_steps=512, batch_size=512, n_epochs=10,
        gamma=0.98, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        learning_rate=linear_schedule(3e-4),
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                           log_std_init=-1.0),
        device="cpu", seed=args.seed, verbose=1,
    )
    model.set_logger(configure(args.logdir, ["stdout", "csv"]))

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=str(out), log_path=args.logdir,
        eval_freq=max(20_000 // args.n_envs, 1), n_eval_episodes=4,
        deterministic=True, render=False,
    )

    print(f"[train] {args.timesteps:,} steps | {args.n_envs} envs | device=cpu")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=eval_cb,
                progress_bar=False)
    dt = time.time() - t0

    final_path = out / "ppo_tracker"
    model.save(final_path)
    vec_env.close()
    eval_env.close()
    print(f"[train] done in {dt/60:.1f} min")
    print(f"[train] final  model -> {final_path}.zip")
    print(f"[train] best   model -> {out / 'best_model'}.zip")


if __name__ == "__main__":
    main()
