#!/usr/bin/env python3
"""Watch a trained policy fly the window: interactive 3rd-person viewer + onboard RGB-D.

    python rl/eval_window.py --model runs/window/best_model.zip

The 3rd-person view is the interactive mujoco.viewer; the onboard RGB-D (what the
CNN sees) shows in a separate-process window (handled by run_viewer).
"""

import argparse

from stable_baselines3 import PPO

from rl.window_aviary import WindowAviary
from rl._viewer import run_viewer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    args = ap.parse_args()

    env = WindowAviary()
    model = PPO.load(args.model, device="cpu")

    results = {"pass": 0, "crash": 0}

    def get_action(env, obs):
        a, _ = model.predict(obs, deterministic=True)
        return a

    def post_step(env, obs, r, term, trunc, info):
        if term or trunc:
            if info.get("is_success"):
                results["pass"] += 1
            else:
                results["crash"] += 1
            print(f"  {info.get('terminal_reason')}  "
                  f"(passed={results['pass']}, failed={results['crash']})")

    print(f"Evaluating {args.model}. Close the viewer to quit.")
    run_viewer(env, get_action, post_step=post_step)   # onboard RGB-D on by default
    env.close()


if __name__ == "__main__":
    main()
