#!/usr/bin/env python3
"""Sequence-aware behaviour cloning onto the recurrent policy.

Unlike rl/bc_pretrain.py, which shuffles frames i.i.d., this trains over whole
windows so the recurrent state means something. It also warm-starts from the
already-trained SeVAE and memory checkpoints rather than from scratch.

Only the actor is trained. The critic stays random, which is exactly why
train_course applies a warm-start freeze on the actor for the first stretch of
PPO -- otherwise a random value function destroys the cloned policy before it
learns anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mavrl.config import AUX_SEGMENTS
from mavrl.dataset import SequenceDataset
from mavrl.memory import AuxReconstructionHead
from mavrl.policy import MavrlActorCritic


def load_pretrained(policy: MavrlActorCritic, sevae_path, memory_path,
                    device) -> None:
    if sevae_path and Path(sevae_path).exists():
        policy.sevae.load_state_dict(
            torch.load(sevae_path, map_location=device)["model"])
        print("loaded SeVAE   ", sevae_path)
    if memory_path and Path(memory_path).exists():
        ckpt = torch.load(memory_path, map_location=device)
        policy.memory.load_state_dict(ckpt["memory"])
        # The aux head's width is set by the lambda configuration stage 3 ran
        # with, which the policy has no way to know. Rebuild it to match rather
        # than dropping the checkpoint on a shape mismatch.
        segments = tuple(ckpt.get("aux_segments", AUX_SEGMENTS))
        if segments != policy.aux.segments:
            policy.aux = AuxReconstructionHead(segments=segments).to(device)
        policy.aux.load_state_dict(ckpt["aux"])
        print("loaded memory  ", memory_path, f"(aux={list(segments)})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--sevae", type=Path, default=Path("ckpt/sevae.pt"))
    p.add_argument("--memory", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("ckpt/bc_init.pt"))
    p.add_argument("--memory-type", default="lstm",
                   choices=("lstm", "attention"))
    p.add_argument("--mem-tokens", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--freeze-encoder", action="store_true", default=True)
    p.add_argument("--train-encoder", dest="freeze_encoder",
                   action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    memory_path = args.memory or Path(
        f"ckpt/memory_{args.memory_type}"
        f"{'' if args.mem_tokens is None else f'_m{args.mem_tokens}'}.pt")

    policy = MavrlActorCritic(memory_type=args.memory_type,
                              mem_tokens=args.mem_tokens,
                              freeze_encoder=args.freeze_encoder).to(device)
    load_pretrained(policy, args.sevae, memory_path, device)

    trainable = [prm for prm in policy.parameters() if prm.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr)

    ds = SequenceDataset(args.data, seq_len=args.seq_len,
                         keys=("image", "proprio", "action"))
    print(f"{len(ds)} windows | memory={args.memory_type} "
          f"encoder={'frozen' if args.freeze_encoder else 'trainable'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    history = []
    action_var = None

    for epoch in range(args.epochs):
        total, n = 0.0, 0
        for batch in ds.iter_batches(args.batch_size, rng):
            img = torch.as_tensor(batch["image"], device=device)
            img = img.permute(0, 1, 4, 2, 3).contiguous()
            proprio = torch.as_tensor(batch["proprio"], device=device)
            act = torch.as_tensor(batch["action"], device=device)

            state = policy.initial_state(img.shape[0], device)
            dist, _, _, _ = policy.sequence(img, proprio, state, sample=False)
            loss = F.mse_loss(dist.mean, act)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            total += loss.item()
            n += 1
            if action_var is None:
                action_var = float(act.reshape(-1, act.shape[-1]).var(0).mean())

        if n == 0:
            raise SystemExit(
                f"epoch {epoch} ran zero batches: no shard holds "
                f"{args.batch_size} windows of {args.seq_len} frames. Lower "
                f"--batch-size or --seq-len.")

        mse = total / max(1, n)
        history.append({"epoch": epoch, "mse": mse})
        print(f"epoch {epoch:3d}  mse={mse:.5f}", flush=True)
        torch.save({"policy": policy.state_dict(),
                    "memory_type": args.memory_type,
                    "mem_tokens": args.mem_tokens,
                    "aux_segments": list(policy.aux.segments)}, args.out)

    # Mean-collapse detector, same idea as rl/bc_pretrain.py:70 -- a policy that
    # just predicts the dataset mean scores ~= the action variance.
    if action_var and mse > 0.5 * action_var:
        print(f"\nWARNING: bc mse {mse:.4f} vs action variance {action_var:.4f} "
              "-- the policy may have collapsed to predicting the mean.")

    (args.out.parent / "bc_history.json").write_text(json.dumps(history, indent=2))
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
