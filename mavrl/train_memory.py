#!/usr/bin/env python3
"""Stage 3: train the memory module with a frozen SeVAE.

The objective is the paper's Eq. (2): from z_t (plus the state vector x_t),
reconstruct several images at once -- `I_{t-T}`, `I_t`, `I_{t+T}` -- all through
the same frozen decoder, with `lambda_i in {0,1}` selecting which. Note this is
the *only* multi-frame stage: the VAE itself (Eq. (1)) is trained per-frame with
no time index and no recurrence, which is why stage 2 uses `FrameDataset` and
this stage uses `SequenceDataset`.

`--aux-segments` is that lambda configuration. Default `past,current` = (1,1,0);
`past,current,future` reproduces the full three-way head.

Remember what T is. It is the *prediction offset*, not a memory window -- an
LSTM's hidden state has no horizon, and T=20 is a supervision target that
pressures it into retaining ~20 steps. For the attention backbone the window K
genuinely is a cap, which is why K=16 < T=20: if the window still contained step
t-T the loss would be satisfiable by copying rather than remembering, and only
the LSTM would face a real memory task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mavrl.config import AUX_SEGMENTS, AUX_T, LATENT_DIM
from mavrl.dataset import SequenceDataset
from mavrl.memory import AuxReconstructionHead, aux_valid_range, build_memory
from mavrl.sevae import SeVAE


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--sevae", type=Path, default=Path("ckpt/sevae.pt"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--memory-type", default="lstm",
                   choices=("lstm", "attention"))
    p.add_argument("--mem-tokens", type=int, default=None,
                   help="attention only; 0 = plain windowed attention")
    p.add_argument("--T", type=int, default=AUX_T)
    p.add_argument("--aux-segments", default=",".join(AUX_SEGMENTS),
                   help="paper's lambda_i, by name: comma-separated subset of "
                        "past,current,future")
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--samples", type=Path, default=None,
                   help="where to write the past/current reconstruction grid "
                        "(default: alongside --out)")
    p.add_argument("--no-samples", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    segments = tuple(s.strip() for s in args.aux_segments.split(",") if s.strip())
    aux = AuxReconstructionHead(segments=segments)
    offsets = aux.offsets(args.T)
    lo, hi = aux_valid_range(args.seq_len, offsets)
    if hi <= lo:
        raise SystemExit(
            f"--seq-len ({args.seq_len}) is too short for segments "
            f"{list(segments)} at T={args.T}: offsets {list(offsets)} leave no "
            f"timestep whose targets all lie inside the window "
            f"(need seq_len > {max(offsets) - min(offsets)})")

    tag = args.memory_type + ("" if args.mem_tokens is None
                              else f"_m{args.mem_tokens}")
    if segments != AUX_SEGMENTS:
        tag += "_" + "".join(s[0] for s in segments)
    out = args.out or Path(f"ckpt/memory_{tag}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    sevae = SeVAE().to(device)
    sevae.load_state_dict(torch.load(args.sevae, map_location=device)["model"])
    sevae.eval()
    for prm in sevae.parameters():
        prm.requires_grad = False

    mem_kwargs = {} if args.mem_tokens is None else {"n_mem": args.mem_tokens}
    memory = build_memory(args.memory_type, **mem_kwargs).to(device)
    aux = aux.to(device)
    opt = torch.optim.Adam(
        list(memory.parameters()) + list(aux.parameters()), lr=args.lr)

    ds = SequenceDataset(args.data, seq_len=args.seq_len,
                         keys=("image", "image_gt", "proprio"))
    if len(ds) == 0:
        raise SystemExit(
            f"no windows of {args.seq_len} frames in {args.data} -- every "
            f"episode is shorter than that. Lower --seq-len, or collect longer "
            f"episodes.")
    print(f"{len(ds)} windows of {args.seq_len} frames | memory={tag} T={args.T}")
    print(f"segments={list(segments)} offsets={list(offsets)} "
          f"supervised timesteps [{lo},{hi}) of {args.seq_len}")

    history = []
    for epoch in range(args.epochs):
        agg = {s: 0.0 for s in segments} | {"total": 0.0}
        n = 0
        for batch in ds.iter_batches(args.batch_size, rng):
            img = torch.as_tensor(batch["image"], device=device)
            gt = torch.as_tensor(batch["image_gt"], device=device)
            proprio = torch.as_tensor(batch["proprio"], device=device)
            b, t = img.shape[:2]

            with torch.no_grad():
                z, _, _ = sevae.encode(
                    img.permute(0, 1, 4, 2, 3).reshape(b * t, 4, *img.shape[2:4]),
                    sample=False)
                z = z.view(b, t, -1)

            z_seq, _ = memory(z, None)                       # (B,T,256)

            # Only timesteps whose every target lies inside the window count.
            idx = torch.arange(lo, min(hi, t), device=device)
            pred = aux(z_seq[:, idx], proprio[:, idx])       # (B,T',n_seg,latent)

            def decode(latent):
                rgb, depth, _ = sevae.decoder(latent.reshape(-1, LATENT_DIM))
                return torch.cat([rgb, depth], dim=1)

            tgt = gt.permute(0, 1, 4, 2, 3).float() / 255.0
            per_seg = {}
            loss = 0.0
            for i, (name, off) in enumerate(zip(segments, offsets)):
                target = tgt[:, idx + off].reshape(-1, 4, *img.shape[2:4])
                l = F.mse_loss(decode(pred[..., i, :]), target)
                per_seg[name] = l
                loss = loss + l

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(memory.parameters()) + list(aux.parameters()), 1.0)
            opt.step()

            for name, l in per_seg.items():
                agg[name] += l.item()
            agg["total"] += loss.item()
            n += 1

        # `iter_batches` skips any shard holding fewer than batch_size windows,
        # so an epoch can silently do nothing and report a loss of 0.0.
        if n == 0:
            raise SystemExit(
                f"epoch {epoch} ran zero batches: no shard holds "
                f"{args.batch_size} windows of {args.seq_len} frames. Lower "
                f"--batch-size or --seq-len.")

        row = {k: v / max(1, n) for k, v in agg.items()} | {"epoch": epoch}
        history.append(row)
        print(f"epoch {epoch:3d}  " +
              "  ".join(f"{s}={row[s]:.5f}" for s in segments), flush=True)
        torch.save({"memory": memory.state_dict(), "aux": aux.state_dict(),
                    "memory_type": args.memory_type,
                    "mem_tokens": args.mem_tokens, "T": args.T,
                    "aux_segments": list(segments)}, out)

    # A `past` MSE is not a picture. Render one -- it is the only way to see the
    # difference between remembering a bar and learning the mean corridor.
    if not args.no_samples:
        from mavrl.visualize import save_memory_samples
        sample_batch = next(ds.iter_batches(min(4, args.batch_size), rng), None)
        if sample_batch is not None:
            save_memory_samples(
                sevae, memory, aux, sample_batch,
                args.samples or out.parent / f"memory_{tag}_samples.png",
                T=args.T, n=4)

    (out.parent / f"memory_{tag}_history.json").write_text(
        json.dumps(history, indent=2))
    print("saved", out)
    print("\nNote: `past` is the number that matters. It is the one the "
          "attention window cannot satisfy by copying when K < T. `current` is "
          "near-free for any backbone, and `future` is partly unlearnable -- "
          "the paper's Fig. 2(b) shows it as the blurriest of the three.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
