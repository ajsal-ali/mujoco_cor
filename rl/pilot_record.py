#!/usr/bin/env python3
"""Generate clean expert demos with a scripted pilot (privileged window pose).

Human hold-to-fly teleop demos are NOT a consistent function of the camera image
(while a key is held the action stays constant regardless of what's seen; taps are
intent-driven), so behavior cloning can only predict their mean -> the policy just
flies straight. A scripted P-controller that flies to the window center IS a clean
function of state, so BC learns it and GENERALIZES (validated: val MSE ~ 0 for the
strafe/climb corrections).

This is the privileged-teacher -> vision-student route: the pilot uses the known
window pose to fly; the STUDENT policy it trains (via bc_pretrain -> train_window)
uses only vision + proprioception and never sees the pose.

    python -m rl.pilot_record --episodes 40 --out demos.npz
    python -m rl.bc_pretrain  --demos demos.npz --out runs/bc_init.zip
    python -m rl.train_window --init runs/bc_init.zip --n-envs 8 --timesteps 2000000

Steers with forward + strafe + climb and keeps yaw fixed on the window normal
(matches the alignment reward) — no yaw, the dimension humans make noisy.
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")   # headless render for fast collection

import argparse
import math

import numpy as np

from rl.window_aviary import WindowAviary


def pilot_action(env, kp=1.5):
    """Clean expert action: velocity toward a point just past the window center."""
    pos = env.pos[0]
    yaw = env.rpy[0, 2]
    target = env.gate.CENTER + np.array([0.0, 0.9, 0.0])     # past the opening
    v_des = np.clip(kp * (target - pos), -1.0, 1.0)          # desired world velocity
    cz, sz = math.cos(yaw), math.sin(yaw)
    vbx = cz * v_des[0] + sz * v_des[1]                      # world -> body
    vby = -sz * v_des[0] + cz * v_des[1]
    a = np.array([vbx / env.V_XY, vby / env.V_XY, v_des[2] / env.V_Z, 0.0], np.float32)
    return np.clip(a, -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40, help="successful episodes to keep")
    ap.add_argument("--out", type=str, default="demos.npz")
    ap.add_argument("--noise", type=float, default=0.05,
                    help="std of exploration noise added to the EXECUTED action "
                         "(the recorded label stays the clean expert action -> DAgger-like "
                         "state diversity for a more robust clone)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = WindowAviary(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    imgs, props, acts = [], [], []
    kept, tried = 0, 0
    max_tries = args.episodes * 5
    while kept < args.episodes and tried < max_tries:
        obs, info = env.reset()
        tried += 1
        ei, ep, ea = [], [], []
        for _ in range(500):
            a_clean = pilot_action(env)
            ei.append(obs["image"]); ep.append(obs["proprio"]); ea.append(a_clean)
            a_exec = a_clean
            if args.noise > 0:
                a_exec = np.clip(a_clean + rng.normal(0, args.noise, 4).astype(np.float32),
                                 -1.0, 1.0)
            obs, r, term, trunc, info = env.step(a_exec)
            if term or trunc:
                if info["is_success"]:
                    imgs += ei; props += ep; acts += ea; kept += 1
                break
        if tried % 10 == 0:
            print(f"  tried {tried}, kept {kept}, demo steps {len(acts)}")

    if kept == 0:
        print("No successful episodes — pilot failed to pass the window. "
              "Check the window geometry / spawn config.")
        env.close(); return

    np.savez_compressed(
        args.out,
        image=np.asarray(imgs, np.uint8),
        proprio=np.asarray(props, np.float32),
        action=np.asarray(acts, np.float32),
    )
    print(f"Saved {len(acts)} demo steps from {kept} successful episodes "
          f"(of {tried} tries) -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
