"""Render tracking videos of the trained policy.

Produces an MP4 (and a short GIF preview) showing the Sawyer arm following
the desired trajectory.  Overlays:

  * blue ribbon  -- desired path
  * red sphere   -- current target pose
  * green trail  -- where the end-effector actually went
  * HUD text     -- live position error and running RMSE

    python -m src.render                       # circle+figure8+random -> mp4
    python -m src.render --traj circle --gui   # live interactive viewer
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .arm_env import ArmTrackingEnv
from .config import EnvConfig

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]


def _font(size):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def annotate(frame, lines, font_big, font_small):
    """Draw a HUD onto an RGB frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    y = 16
    for i, text in enumerate(lines):
        font = font_big if i == 0 else font_small
        draw.text((20, y), text, fill=(255, 255, 255), font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0))
        y += (40 if i == 0 else 30)
    return np.asarray(img)


def render_episode(env, model, traj_name, font_big, font_small):
    obs, _ = env.reset(seed=7, options=dict(traj=traj_name, eval=True))
    frames, errs = [], []
    done = False
    while not done:
        action = (np.zeros(7) if model is None
                  else model.predict(obs, deterministic=True)[0])
        obs, _, term, trunc, info = env.step(action)
        errs.append(info["pos_err"])
        rmse = np.sqrt(np.mean(np.square(errs))) * 1e3
        hud = [
            f"Trajectory: {traj_name}",
            f"position error : {info['pos_err']*1e3:5.1f} mm",
            f"running RMSE    : {rmse:5.1f} mm",
            f"orientation err : {info['ori_err_deg']:5.1f} deg",
            f"control delay   : {info['delay_steps']*50} ms",
        ]
        frames.append(annotate(env.render(), hud, font_big, font_small))
        done = term or trunc
    return frames


def run_gui(traj_name):
    """Live interactive viewer (requires a display)."""
    import mujoco.viewer

    cfg = EnvConfig(domain_rand=False)
    env = ArmTrackingEnv(cfg)
    model = _load_model()
    obs, _ = env.reset(seed=7, options=dict(traj=traj_name, eval=True))
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        done = False
        while viewer.is_running() and not done:
            action = (np.zeros(7) if model is None
                      else model.predict(obs, deterministic=True)[0])
            obs, _, term, trunc, _ = env.step(action)
            viewer.sync()
            done = term or trunc
    env.close()


def _load_model(path="models/best_model.zip"):
    if Path(path).exists():
        from stable_baselines3 import PPO
        print(f"[render] loaded policy: {path}")
        return PPO.load(path, device="cpu")
    print("[render] no model found -> rendering base controller only")
    return None


def main():
    ap = argparse.ArgumentParser(description="Render tracking videos.")
    ap.add_argument("--model", type=str, default="models/best_model.zip")
    ap.add_argument("--traj", type=str, default="all",
                    choices=["all", "circle", "figure8", "random"])
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--gui", action="store_true", help="live interactive viewer")
    args = ap.parse_args()

    if args.gui:
        run_gui("circle" if args.traj == "all" else args.traj)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = _load_model(args.model)
    font_big, font_small = _font(34), _font(26)

    cfg = EnvConfig(domain_rand=False)
    env = ArmTrackingEnv(cfg, render_mode="rgb_array")
    trajs = (["circle", "figure8", "random"] if args.traj == "all"
             else [args.traj])

    all_frames = []
    for name in trajs:
        print(f"[render] rolling out '{name}' ...")
        frames = render_episode(env, model, name, font_big, font_small)
        all_frames += frames
        # short GIF preview of the first trajectory for the README
        if name == trajs[0]:
            gif = [Image.fromarray(f).resize((480, 360)) for f in frames[::2]]
            gif_path = out / f"tracking_{name}.gif"
            gif[0].save(gif_path, save_all=True, append_images=gif[1:],
                        duration=1000 // (args.fps // 2), loop=0)
            print(f"[render] wrote {gif_path}")

    mp4 = out / "tracking_demo.mp4"
    imageio.mimsave(mp4, all_frames, fps=args.fps, quality=8,
                    macro_block_size=1)
    print(f"[render] wrote {mp4}  ({len(all_frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
