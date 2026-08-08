#!/usr/bin/env python3
"""Stage 2: train the SeVAE on collected RGB-D + segmentation.

Input is the noise-corrupted frame, target is the clean one -- a denoising VAE.
Reconstruction is proximity-weighted; the segmentation cross-entropy is not
(see mavrl.sevae for why).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mavrl.config import BETA_KL, LAMBDA_SEG
from mavrl.course_world import N_SEM_CLASSES, SemClass
from mavrl.dataset import FrameDataset
from mavrl.sevae import SeVAE, seg_class_weights, sevae_loss


def to_tensors(batch: dict, device):
    img = torch.as_tensor(batch["image"], device=device).permute(0, 3, 1, 2)
    gt = torch.as_tensor(batch["image_gt"], device=device).permute(0, 3, 1, 2)
    depth = torch.as_tensor(batch["depth_m"].astype(np.float32),
                            device=device).unsqueeze(1)
    seg = torch.as_tensor(batch["seg"].astype(np.int64), device=device)
    return img, gt, depth, seg


@torch.no_grad()
def seg_iou(logits: torch.Tensor, target: torch.Tensor, n_classes: int):
    """Per-class IoU. Bar classes are the ones that will actually be hard --
    a bar is 1-4% of the frame, so a model that never predicts one still scores
    well on pixel accuracy."""
    pred = logits.argmax(1)
    ious = []
    for c in range(n_classes):
        p, t = pred == c, target == c
        union = (p | t).sum().item()
        ious.append((p & t).sum().item() / union if union else float("nan"))
    return ious


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("ckpt/sevae.pt"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta", type=float, default=BETA_KL)
    p.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG)
    p.add_argument("--seg-weight", choices=("uniform", "proximity"),
                   default="uniform")
    p.add_argument("--samples", type=Path, default=None,
                   help="where to write the reconstruction grid "
                        "(default: alongside --out)")
    p.add_argument("--no-samples", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ds = FrameDataset(args.data)
    counts = ds.class_counts(N_SEM_CLASSES)
    class_w = seg_class_weights(torch.as_tensor(counts)).to(device)
    print("class pixel share:",
          {SemClass(i).name: f"{c / counts.sum():.4f}"
           for i, c in enumerate(counts)})
    print("class weights   :",
          {SemClass(i).name: round(w, 2)
           for i, w in enumerate(class_w.tolist())})

    model = SeVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(args.epochs):
        model.train()
        agg, n = {}, 0
        for batch in ds.iter_batches(args.batch_size, rng):
            img, gt, depth, seg = to_tensors(batch, device)
            out = model(img)
            loss, parts = sevae_loss(
                out, gt, depth, seg, class_weights=class_w,
                beta=args.beta, lambda_seg=args.lambda_seg,
                weight_seg_by_proximity=(args.seg_weight == "proximity"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1

        if n == 0:
            raise SystemExit(
                f"epoch {epoch} ran zero batches: no shard holds "
                f"{args.batch_size} frames. Lower --batch-size.")

        row = {k: v / max(1, n) for k, v in agg.items()} | {"epoch": epoch}
        history.append(row)
        print(f"epoch {epoch:3d}  " +
              "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"),
              flush=True)
        torch.save({"model": model.state_dict(), "epoch": epoch}, args.out)

    # Final IoU on one batch, reported per class so bar mIoU is visible.
    model.eval()
    batch = next(ds.iter_batches(args.batch_size, rng))
    img, gt, depth, seg = to_tensors(batch, device)
    out = model(img, sample=False)
    ious = seg_iou(out.seg_logits, seg, N_SEM_CLASSES)
    print("\nseg IoU:", {SemClass(i).name: (None if np.isnan(v) else round(v, 3))
                         for i, v in enumerate(ious)})

    # The losses above cannot tell you whether the bars survived. Look at the
    # picture.
    if not args.no_samples:
        from mavrl.visualize import save_sevae_samples
        save_sevae_samples(model, batch,
                           args.samples or args.out.parent / "sevae_samples.png")

    (args.out.parent / "sevae_history.json").write_text(json.dumps(history, indent=2))
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
