#!/usr/bin/env python3
"""Ephemeral env viewer — run anytime to visually inspect the env. Close = gone.

Opens WindowAviary in the interactive 3rd-person viewer. Fly it with the keyboard,
watch a trained policy, or just let it hover. Auto-resets each episode. No
recording, no training.

Examples:
    python rl/view_env.py                       # teleop (hold keys to fly)
    python rl/view_env.py --model runs/window/best_model.zip   # watch a policy
    python rl/view_env.py --mode idle           # just hover / watch physics
"""

import argparse
import numpy as np

from rl.window_aviary import WindowAviary
from rl._viewer import run_viewer, KeyTeleop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teleop", "policy", "idle"], default="teleop")
    ap.add_argument("--model", type=str, default=None, help="policy .zip (implies --mode policy)")
    ap.add_argument("--dolls", action="store_true")
    args = ap.parse_args()
    if args.model:
        args.mode = "policy"

    env = WindowAviary(include_dolls=args.dolls)

    if args.mode == "policy":
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        def get_action(env, obs):
            a, _ = model.predict(obs, deterministic=True)
            return a
        on_key = None
        print(f"Watching policy: {args.model}  (episodes auto-reset; close window to quit)")
    elif args.mode == "idle":
        def get_action(env, obs):
            return np.zeros(4, dtype=np.float32)
        on_key = None
        print("Idle hover (close window to quit)")
    else:  # teleop
        teleop = KeyTeleop()
        def get_action(env, obs):
            return teleop.action(env.CTRL_TIMESTEP)   # already normalized [-1,1]
        on_key = teleop.on_key
        print("Teleop: hold W/S A/D (fwd/strafe), Up/Down (alt), Left/Right (yaw), "
              "H hover. Close window to quit.")

    def post_step(env, obs, r, term, trunc, info):
        if term or trunc:
            if info.get("is_success"):
                print("  PASSED ✓")
            else:
                reason = info.get("terminal_reason") or ("timeout" if trunc else "unknown")
                print(f"  FAILED: {reason}")

    run_viewer(env, get_action, post_step=post_step, on_key=on_key)
    env.close()


if __name__ == "__main__":
    main()
