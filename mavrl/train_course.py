#!/usr/bin/env python3
"""Stages 1 and 4: PPO on the course.

Stage 1  --stage 0 --frozen-encoder     random frozen encoder, entry->exit only.
                                        Produces the policy that collects data.
Stage 4  --curriculum --init ckpt/...   full curriculum, warm-started from BC.

The curriculum switches layout for **every env at once**, at a rollout boundary.
Switching mid-rollout would invalidate the value estimates already collected
against the old geometry.

Two artefacts land in `--out` as the run goes: `log.jsonl`, one row per rollout,
and `curves.png`, that log plotted. The PNG is rewritten every `--plot-every`
rollouts and again on exit, so a run on a remote box can be watched by pulling
one file -- no TensorBoard process to keep alive alongside it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mavrl import config as C                                    # noqa: E402
from mavrl.course_world import STAGE_STATIONS                    # noqa: E402
from mavrl.curriculum import CourseSampler, broadcast_layout     # noqa: E402
from mavrl.policy import MavrlActorCritic, load_policy_state     # noqa: E402
from mavrl.ppo import PPOConfig, RecurrentPPO                    # noqa: E402
from mavrl.sensor_noise import NoiseConfig                       # noqa: E402
from mavrl.vecenv import build_venv                              # noqa: E402
from mavrl.visualize import save_training_curves                 # noqa: E402


class WarmStartFreeze:
    """Freeze the actor while the critic catches up.

    Ported from rl/train_window.py:95. After BC the actor is good and the value
    function is random; letting them train together immediately lets garbage
    advantages destroy a policy that already works.
    """

    def __init__(self, policy: MavrlActorCritic, steps: int):
        self.policy = policy
        self.steps = steps
        self.released = steps <= 0
        if not self.released:
            self._set(False)

    def _set(self, on: bool) -> None:
        for module in (self.policy.pi,):
            for prm in module.parameters():
                prm.requires_grad = on
        self.policy.log_std.requires_grad = on

    def maybe_release(self, num_timesteps: int) -> bool:
        if not self.released and num_timesteps >= self.steps:
            self._set(True)
            self.released = True
            return True
        return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/course"))
    p.add_argument("--timesteps", type=int, default=500_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--memory-type", default="lstm",
                   choices=("lstm", "attention", "none"))
    p.add_argument("--mem-tokens", type=int, default=None)
    p.add_argument("--stage", type=int, default=None,
                   help="fix the curriculum stage (no advancement)")
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--frozen-encoder", action="store_true")
    p.add_argument("--init", type=Path, default=None)
    p.add_argument("--critic-warmup", type=int, default=30_000)
    p.add_argument("--sensor-noise", type=float, default=1.0)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-subproc", action="store_true")
    p.add_argument("--plot-every", type=int, default=10,
                   help="rollouts between redraws of curves.png; 0 = only at the end")
    p.add_argument("--plot-smooth", type=int, default=15,
                   help="trailing-mean width, in rollouts, for the curves")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    noise = (NoiseConfig().scaled(args.sensor_noise)
             if args.sensor_noise > 0 else NoiseConfig.disabled())

    start_stage = args.stage if args.stage is not None else 0
    sampler = CourseSampler(seed=args.seed, split="train",
                            start_stage=start_stage)
    venv = build_venv(sampler.layout, args.n_envs, args.seed, noise,
                      subproc=not args.no_subproc)

    # "none" is the no-memory ablation: an LSTM with a 1-step window would still
    # be recurrent, so instead we keep the LSTM module but zero its state every
    # step, making the policy genuinely Markovian.
    memory_type = "lstm" if args.memory_type == "none" else args.memory_type
    policy = MavrlActorCritic(memory_type=memory_type,
                              mem_tokens=args.mem_tokens,
                              freeze_encoder=args.frozen_encoder)
    if args.memory_type == "none":
        policy.memory_type = "none"

    if args.init and args.init.exists():
        ckpt = torch.load(args.init, map_location="cpu")
        load_policy_state(policy, ckpt["policy"], ckpt.get("aux_segments"))
        print("warm-started from", args.init)

    cfg = PPOConfig(n_steps=args.n_steps, learning_rate=args.lr,
                    envs_per_batch=max(1, args.n_envs // 2))
    algo = RecurrentPPO(policy, venv, cfg, device=args.device)

    warmup = WarmStartFreeze(policy, args.critic_warmup if args.init else 0)
    log_path = args.out / "log.jsonl"
    plot_path = args.out / "curves.png"
    # 200 episodes is ~25 rollouts at 8 envs: long enough that the success rate
    # is not quantized into eighths, short enough to still move within a stage.
    recent = {k: deque(maxlen=200)
              for k in ("success", "return", "agv", "ep_len", "gates")}

    def draw_curves() -> None:
        """A broken plot must never take down a run that is otherwise fine."""
        try:
            save_training_curves(log_path, plot_path, args.plot_smooth)
        except Exception as exc:                      # pragma: no cover
            print(f"[plot] skipped: {exc}", flush=True)

    def on_rollout_end(algo: RecurrentPPO, rollout: int, stats: dict) -> None:
        infos = algo.episode_infos
        algo.episode_infos = []
        for info in infos:
            recent["success"].append(float(info.get("is_success", False)))
            recent["return"].append(float(info.get("ep_return", 0.0)))
            recent["agv"].append(float(info.get("agv", 0.0)))
            recent["ep_len"].append(float(info.get("ep_len", 0)))
            # Partial credit. Success is all-or-nothing, so early in a stage it
            # sits at zero while the policy is in fact learning gate 1 of 3.
            n_gates = max(1, int(info.get("n_gates", 1)))
            recent["gates"].append(float(info.get("gates_cleared", 0)) / n_gates)
        sampler.record_episodes(infos)

        if warmup.maybe_release(algo.num_timesteps):
            algo.optimizer = torch.optim.Adam(
                [q for q in policy.parameters() if q.requires_grad], lr=args.lr)
            print(f"[{algo.num_timesteps}] actor released")

        row = {
            "rollout": rollout,
            "timesteps": algo.num_timesteps,
            "stage": sampler.stage,
            "episodes": len(infos),
            **{k: float(np.mean(v)) if v else 0.0 for k, v in recent.items()},
            **stats,
        }
        with log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        if rollout % 5 == 0:
            print(f"[{algo.num_timesteps:>9}] {sampler.describe()} "
                  f"ret={row['return']:+.2f} agv={row['agv']:.2f} "
                  f"pg={stats['pg']:+.4f} "
                  f"vf={stats['vf']:.4f} kl={stats['kl']:.4f}", flush=True)

        if args.plot_every and rollout % args.plot_every == 0:
            draw_curves()

        if args.curriculum:
            new_layout = sampler.on_rollout_end()
            if new_layout is not None:
                broadcast_layout(algo.venv, new_layout)

        if rollout % 50 == 0:
            algo.save(args.out / f"ckpt_{algo.num_timesteps}.pt")

    try:
        algo.learn(args.timesteps, on_rollout_end=on_rollout_end)
    finally:
        algo.save(args.out / "final.pt")
        draw_curves()          # also covers Ctrl-C and a crash mid-run
        venv.close()
    print("saved", args.out / "final.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
