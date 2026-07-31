#!/usr/bin/env python3
"""Teleop the window task and record demos for behavior cloning.

Fly WindowAviary by hand in the interactive viewer; every step's
(image, proprio, action) is logged. Only SUCCESSFUL episodes (drone passed
through the window) are kept, so the demos are clean. Saves to demos.npz.

Crucially, it prints the LIVE traversal status using the env's own checker
(the same WindowGate the reward uses) — so hand-flying validates that the
pass/crash detection is correct.

    python rl/teleop_record.py --out demos.npz
Fly through the window a handful of times, then close the window to save.
"""

import argparse
import numpy as np

from rl.window_aviary import WindowAviary
from rl._viewer import run_viewer, KeyTeleop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="demos.npz")
    ap.add_argument("--keep-crashes", action="store_true",
                    help="also keep crashed episodes (default: successes only)")
    args = ap.parse_args()

    env = WindowAviary()
    teleop = KeyTeleop()

    master = {"image": [], "proprio": [], "action": []}
    ep = {"image": [], "proprio": [], "action": []}
    stats = {"episodes": 0, "kept": 0, "last_print_dy": 1e9}

    def get_action(env, obs):
        a = teleop.action(env.CTRL_TIMESTEP)          # already normalized [-1,1]
        ep["image"].append(obs["image"])
        ep["proprio"].append(obs["proprio"])
        ep["action"].append(a)
        return a

    def post_step(env, obs, r, term, trunc, info):
        # live approach readout (so you can watch the checker fire)
        dy = env.gate.Y - env.pos[0, 1]
        if abs(dy - stats["last_print_dy"]) > 0.25:
            stats["last_print_dy"] = dy
            side = "before" if dy > 0 else "past"
            print(f"    window {side} plane: dy={dy:+.2f} m   (NOT PASSED)")
        if term or trunc:
            stats["episodes"] += 1
            reason = info.get("terminal_reason")
            success = info.get("is_success")
            tag = "PASSED ✓" if success else f"{str(reason).upper()}"
            keep = success or args.keep_crashes
            if keep and len(ep["action"]) > 0:
                master["image"].extend(ep["image"])
                master["proprio"].extend(ep["proprio"])
                master["action"].extend(ep["action"])
                stats["kept"] += 1
            print(f"  >>> episode {stats['episodes']}: {tag}  "
                  f"({len(ep['action'])} steps, kept={keep})  "
                  f"total demo steps={len(master['action'])}")
            ep["image"].clear(); ep["proprio"].clear(); ep["action"].clear()
            stats["last_print_dy"] = 1e9

    print("Teleop-record: HOLD keys to fly through the RED window.\n"
          "  W/S fwd/back  A/D left/right  Up/Down alt  Left/Right yaw  H hover.\n"
          "  Only successful passes are saved. Close the window when done.\n")
    run_viewer(env, get_action, post_step=post_step, on_key=teleop.on_key)
    env.close()

    n = len(master["action"])
    if n == 0:
        print("No demo steps recorded — nothing saved.")
        return
    np.savez_compressed(
        args.out,
        image=np.asarray(master["image"], dtype=np.uint8),
        proprio=np.asarray(master["proprio"], dtype=np.float32),
        action=np.asarray(master["action"], dtype=np.float32),
    )
    print(f"\nSaved {n} demo steps from {stats['kept']} kept episodes -> {args.out}")


if __name__ == "__main__":
    main()
