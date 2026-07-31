#!/usr/bin/env python3
"""Headless PPO training for WindowAviary (vision + proprioception).

Fully headless (EGL offscreen render, SubprocVecEnv). Optionally warm-started
from a behavior-cloned policy (rl/bc_pretrain.py). Visual inspection is only ever
via rl/view_env.py or rl/eval_window.py.

Examples:
    # from scratch
    python rl/train_window.py --n-envs 8 --timesteps 2000000
    # warm-started from BC
    python rl/train_window.py --init runs/bc_init.zip --n-envs 8 --timesteps 2000000
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")   # headless GL for every worker

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from rl.window_aviary import WindowAviary
from rl.light_cnn import POLICY_KWARGS


def make_env(rank: int, seed: int = 0, dolls: bool = False):
    def _init():
        return WindowAviary(seed=seed + rank, include_dolls=dolls)
    return _init


class LivePlotCallback(BaseCallback):
    """Live training curves, updated once per PPO rollout (cheap, ~every few s).

    Plots mean episode reward and success rate vs timesteps. Degrades to a no-op
    if no display / matplotlib backend is available (so headless runs never break).
    Saves training_curve.png at the end.
    """

    def __init__(self, out_dir: str):
        super().__init__()
        self.out_dir = out_dir
        self._ok = False
        self._t, self._rew, self._succ = [], [], []

    def _on_training_start(self) -> None:
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            self._plt = plt
            plt.ion()
            self._fig, (self._ax1, self._ax2) = plt.subplots(1, 2, figsize=(10, 4))
            self._fig.canvas.manager.set_window_title("WindowAviary training")
            (self._l1,) = self._ax1.plot([], [], "b-")
            (self._l2,) = self._ax2.plot([], [], "g-")
            self._ax1.set_title("episode reward (mean)"); self._ax1.set_xlabel("timesteps")
            self._ax2.set_title("success rate"); self._ax2.set_xlabel("timesteps")
            self._ax2.set_ylim(-0.02, 1.02)
            self._fig.tight_layout(); plt.show(block=False)
            self._ok = True
        except Exception as e:
            print(f"[live-plot] disabled (no display?): {e}")
            self._ok = False

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if not self._ok:
            return
        buf = self.model.ep_info_buffer
        if not buf:
            return
        self._t.append(self.num_timesteps)
        self._rew.append(float(np.mean([e["r"] for e in buf])))
        self._succ.append(float(np.mean([e.get("is_success", 0.0) for e in buf])))
        self._l1.set_data(self._t, self._rew); self._ax1.relim(); self._ax1.autoscale_view()
        self._l2.set_data(self._t, self._succ); self._ax2.relim()
        self._ax2.autoscale_view(scaley=False)
        self._fig.canvas.draw_idle(); self._fig.canvas.flush_events()

    def _on_training_end(self) -> None:
        if self._ok:
            try:
                self._fig.savefig(str(Path(self.out_dir) / "training_curve.png"))
            except Exception:
                pass


class WarmStartStabilizer(BaseCallback):
    """Stops a BC-warm-started PPO from destroying the good policy.

    bc_pretrain trains only the ACTOR; the value net (critic) starts random, so
    PPO's first advantages are garbage and shove the good actor into a bad basin
    (e.g. flying backwards). Two guards:
      * freeze the BC-learned CNN features (keep the hard-won vision) — PPO only
        refines the small policy/value MLP heads on top;
      * critic warm-up: freeze the actor for the first `warmup_steps` so the value
        net catches up on the good policy's rollouts before it can move the actor.
    """

    def __init__(self, warmup_steps: int = 30000, freeze_cnn: bool = True):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.freeze_cnn = freeze_cnn
        self._actor_frozen = False

    def _actor_params(self):
        pol = self.model.policy
        params = list(pol.mlp_extractor.policy_net.parameters())
        params += list(pol.action_net.parameters())
        params += [pol.log_std]
        return params

    def _on_training_start(self) -> None:
        pol = self.model.policy
        if self.freeze_cnn:
            for p in pol.features_extractor.parameters():
                p.requires_grad = False
            print("[stabilizer] CNN features frozen — PPO refines the MLP heads on BC vision")
        if self.warmup_steps > 0:
            for p in self._actor_params():
                p.requires_grad = False
            self._actor_frozen = True
            print(f"[stabilizer] critic warm-up: actor frozen for {self.warmup_steps} steps")

    def _on_step(self) -> bool:
        if self._actor_frozen and self.num_timesteps >= self.warmup_steps:
            for p in self._actor_params():
                p.requires_grad = True
            self._actor_frozen = False
            print(f"[stabilizer] warm-up done at {self.num_timesteps} steps — actor unfrozen")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--init", type=str, default=None, help="BC-pretrained .zip to warm-start from")
    ap.add_argument("--out", type=str, default="runs/window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dolls", action="store_true")
    ap.add_argument("--plot", action="store_true",
                    help="live matplotlib curves (updated once per rollout; needs a display)")
    ap.add_argument("--critic-warmup", type=int, default=30000,
                    help="steps to train only the critic before unfreezing the actor "
                         "(warm-start only; 0 disables)")
    ap.add_argument("--no-freeze-cnn", action="store_true",
                    help="also fine-tune the CNN during warm-start (less stable)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    venv = SubprocVecEnv([make_env(i, args.seed, args.dolls) for i in range(args.n_envs)])
    venv = VecMonitor(venv, info_keywords=("is_success",))
    eval_env = VecMonitor(DummyVecEnv([make_env(9999, args.seed, args.dolls)]),
                          info_keywords=("is_success",))

    ppo_kwargs = dict(
        policy="MultiInputPolicy",
        env=venv,
        policy_kwargs=POLICY_KWARGS,
        n_steps=512, batch_size=256, n_epochs=4,
        gamma=0.995, gae_lambda=0.95, clip_range=0.1,
        ent_coef=0.0, learning_rate=1e-4, target_kl=0.03,
        device="cuda", verbose=1,
        tensorboard_log=str(out / "tb"),
    )

    if args.init:
        print(f"Warm-starting from {args.init}")
        model = PPO.load(args.init, env=venv, device="cuda")
        model.tensorboard_log = str(out / "tb")
    else:
        model = PPO(**ppo_kwargs)

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=str(out), log_path=str(out),
        eval_freq=max(20000 // args.n_envs, 1), n_eval_episodes=20,
        deterministic=True, render=False)
    ckpt_cb = CheckpointCallback(
        save_freq=max(100000 // args.n_envs, 1), save_path=str(out), name_prefix="ckpt")

    callbacks = [eval_cb, ckpt_cb]
    if args.init:                       # warm-started -> stabilize against critic collapse
        callbacks.append(WarmStartStabilizer(
            warmup_steps=args.critic_warmup, freeze_cnn=not args.no_freeze_cnn))
    if args.plot:
        callbacks.append(LivePlotCallback(str(out)))

    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(str(out / "final_model"))
    print(f"[done] saved to {out}/final_model.zip")
    venv.close(); eval_env.close()


if __name__ == "__main__":
    main()
