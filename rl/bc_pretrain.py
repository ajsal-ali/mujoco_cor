#!/usr/bin/env python3
"""Behavior-clone a PPO policy from teleop demos, so RL doesn't start from noise.

Builds a PPO model with the SAME policy (light CNN + MultiInput) as training,
then supervised-trains its policy to imitate the recorded teleop actions (MSE on
the deterministic action mean). Saves a PPO .zip that train_window.py loads with
--init to warm-start.

    python rl/bc_pretrain.py --demos demos.npz --out runs/bc_init.zip
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.window_aviary import WindowAviary
from rl.light_cnn import POLICY_KWARGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=str, default="demos.npz")
    ap.add_argument("--out", type=str, default="runs/bc_init.zip")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    data = np.load(args.demos)
    images, proprio, actions = data["image"], data["proprio"], data["action"]
    n = len(actions)
    print(f"Loaded {n} demo steps: image{images.shape} proprio{proprio.shape} action{actions.shape}")

    # one env just to construct a PPO with the right spaces + policy
    venv = DummyVecEnv([lambda: WindowAviary()])
    model = PPO("MultiInputPolicy", venv, policy_kwargs=POLICY_KWARGS, device="cuda", verbose=0)
    policy = model.policy
    dev = model.device

    act_t = th.as_tensor(actions, dtype=th.float32, device=dev)
    opt = th.optim.Adam(policy.parameters(), lr=args.lr)

    policy.set_training_mode(True)
    for epoch in range(args.epochs):
        idx = np.random.permutation(n)
        tot, nb = 0.0, 0
        for s in range(0, n, args.batch_size):
            b = idx[s:s + args.batch_size]
            obs = {"image": images[b], "proprio": proprio[b]}
            obs_t, _ = policy.obs_to_tensor(obs)
            features = policy.extract_features(obs_t)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            mean_action = policy.action_net(latent_pi)
            loss = th.nn.functional.mse_loss(mean_action, act_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        final_loss = tot / max(nb, 1)
        print(f"  epoch {epoch + 1:2d}/{args.epochs}  bc_mse={final_loss:.4f}")

    # sanity: if the loss stalled near the action variance, the net collapsed to
    # mean-prediction (image-ignoring) -> the policy will just fly straight.
    mean_var = float(actions.var(0).mean())
    if final_loss > 0.5 * mean_var:
        print(f"\n[WARN] bc_mse ({final_loss:.4f}) is close to the action variance "
              f"({mean_var:.4f}) — the policy likely collapsed to mean-prediction "
              f"(it will just fly straight). Are the demos clean/consistent? "
              f"Prefer scripted demos from `pilot_record.py` over hand teleop.")
    else:
        print(f"\n[ok] learned (bc_mse {final_loss:.4f} << action variance {mean_var:.4f}).")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    venv.close()
    print(f"[done] BC-pretrained policy saved -> {args.out}")


if __name__ == "__main__":
    main()
