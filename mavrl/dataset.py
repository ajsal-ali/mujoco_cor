#!/usr/bin/env python3
"""Sharded sequence dataset for SeVAE, memory and BC training.

The existing rl/ dataset (`demos.npz`) is 8042 flat i.i.d. frames with no
episode boundaries, which cannot train a sequence model at all. This format
keeps them:

    image      (N,128,128,4) uint8    encoder input (noise-corrupted)
    image_gt   (N,128,128,4) uint8    CLEAN reconstruction target
    seg        (N,128,128)   uint8    semantic class ids
    depth_m    (N,128,128)   float16  metres, drives the proximity weight
    proprio    (N,16)        float32
    action     (N,4)         float32
    ep_start   (M,)          int64    episode boundaries into N
    ep_len     (M,)          int64
    ep_layout  (M,)          <U64     layout descriptor, for stratification
    ep_source  (M,)          <U16     "scripted" | "manual" -- survives a merge
    gate_flags (N,)          uint8    gates cleared as of this frame

Sharded per batch of episodes: at 128^2 with RGB-D, clean RGB-D, seg and depth,
one frame is ~115 kB, so 2000 episodes lands in the tens of GB if written as a
single file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np

from mavrl.config import IMG_RES


class ShardWriter:
    """Accumulates episodes and flushes a shard every `episodes_per_shard`."""

    def __init__(self, out_dir, episodes_per_shard: int = 50, prefix: str = "shard"):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.episodes_per_shard = episodes_per_shard
        self.prefix = prefix
        self.shard_idx = 0
        self._reset()

    def _reset(self) -> None:
        self.buf = {k: [] for k in
                    ("image", "image_gt", "seg", "depth_m", "proprio",
                     "action", "gate_flags")}
        self.ep_len: List[int] = []
        self.ep_layout: List[str] = []
        self.ep_source: List[str] = []

    def add_episode(self, frames: dict, layout_desc: str,
                    source: str = "scripted") -> None:
        n = len(frames["image"])
        if n == 0:
            return
        for k in self.buf:
            self.buf[k].append(np.asarray(frames[k]))
        self.ep_len.append(n)
        self.ep_layout.append(layout_desc)
        self.ep_source.append(source)
        if len(self.ep_len) >= self.episodes_per_shard:
            self.flush()

    def flush(self) -> Optional[Path]:
        if not self.ep_len:
            return None
        ep_len = np.asarray(self.ep_len, dtype=np.int64)
        ep_start = np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64)
        path = self.dir / f"{self.prefix}_{self.shard_idx:05d}.npz"
        np.savez_compressed(
            path,
            ep_start=ep_start, ep_len=ep_len,
            ep_layout=np.asarray(self.ep_layout, dtype="<U64"),
            ep_source=np.asarray(self.ep_source, dtype="<U16"),
            **{k: np.concatenate(v, axis=0) for k, v in self.buf.items()})
        self.shard_idx += 1
        self._reset()
        return path

    def close(self) -> Optional[Path]:
        return self.flush()


class SequenceDataset:
    """Yields fixed-length windows that never cross an episode boundary.

    Iterates **shard at a time**, decompressing each shard exactly once per
    epoch. The obvious per-window `np.load` is catastrophically slow -- a
    compressed npz is fully decompressed on every access, so a 2-shard, 1621
    frame set took longer to iterate than it took to generate.
    """

    def __init__(self, root, seq_len: int = 32,
                 keys: Optional[Sequence[str]] = None, stride: int = 4):
        self.root = Path(root)
        self.shards = sorted(self.root.glob("*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"no .npz shards under {self.root}")
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.keys = tuple(keys) if keys else (
            "image", "image_gt", "seg", "depth_m", "proprio", "action")
        self._windows = self._build_index()

    def _shard_windows(self, ep_start, ep_len):
        """Window start offsets that stay inside a single episode."""
        out = []
        for s, ln in zip(ep_start, ep_len):
            if ln < self.seq_len:
                continue
            out.extend(range(int(s), int(s) + int(ln) - self.seq_len + 1,
                             self.stride))
        return out

    def _build_index(self):
        """Per-shard window lists, read from the small header arrays only."""
        windows = []
        for path in self.shards:
            with np.load(path) as z:
                windows.append(self._shard_windows(z["ep_start"], z["ep_len"]))
        return windows

    def __len__(self) -> int:
        return sum(len(w) for w in self._windows)

    def iter_batches(self, batch_size: int, rng: np.random.Generator,
                     shuffle: bool = True) -> Iterator[dict]:
        order = (rng.permutation(len(self.shards)) if shuffle
                 else np.arange(len(self.shards)))
        for si in order:
            starts = list(self._windows[si])
            if len(starts) < batch_size:
                continue
            with np.load(self.shards[si]) as z:
                data = {k: z[k] for k in self.keys}
            if shuffle:
                rng.shuffle(starts)
            for b in range(0, len(starts) - batch_size + 1, batch_size):
                idx = starts[b:b + batch_size]
                yield {k: np.stack([v[s:s + self.seq_len] for s in idx])
                       for k, v in data.items()}


class FrameDataset:
    """Flat frame view over the same shards, for the SeVAE (no time axis)."""

    def __init__(self, root, keys: Optional[Sequence[str]] = None):
        self.root = Path(root)
        self.shards = sorted(self.root.glob("*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"no .npz shards under {self.root}")
        self.keys = tuple(keys) if keys else (
            "image", "image_gt", "seg", "depth_m")

    def iter_batches(self, batch_size: int, rng: np.random.Generator,
                     shuffle: bool = True) -> Iterator[dict]:
        for path in (rng.permutation(self.shards) if shuffle else self.shards):
            with np.load(path) as z:
                data = {k: z[k] for k in self.keys}
            n = len(next(iter(data.values())))
            order = rng.permutation(n) if shuffle else np.arange(n)
            for s in range(0, n - batch_size + 1, batch_size):
                idx = order[s:s + batch_size]
                yield {k: v[idx] for k, v in data.items()}

    def class_counts(self, n_classes: int) -> np.ndarray:
        counts = np.zeros(n_classes, dtype=np.int64)
        for path in self.shards:
            with np.load(path) as z:
                counts += np.bincount(z["seg"].ravel(), minlength=n_classes)
        return counts


def summarize(root) -> dict:
    """Shape/consistency report -- the assertions the notebook runs after collect."""
    root = Path(root)
    shards = sorted(root.glob("*.npz"))
    total_frames = total_eps = 0
    layouts, bad = [], []
    by_source: dict = {}
    for path in shards:
        with np.load(path) as z:
            n = len(z["image"])
            ep_start, ep_len = z["ep_start"], z["ep_len"]
            if int(ep_len.sum()) != n:
                bad.append(f"{path.name}: ep_len sums to {ep_len.sum()}, N={n}")
            expected = np.concatenate([[0], np.cumsum(ep_len)[:-1]])
            if not np.array_equal(ep_start, expected):
                bad.append(f"{path.name}: ep_start does not match cumsum(ep_len)")
            total_frames += n
            total_eps += len(ep_len)
            layouts.extend(z["ep_layout"].tolist())
            # Pre-merge shards predate ep_source; treat them as scripted.
            src = (z["ep_source"].tolist() if "ep_source" in z.files
                   else ["scripted"] * len(ep_len))
            for s_, ln in zip(src, ep_len):
                by_source[s_] = by_source.get(s_, 0) + int(ln)
    return {"shards": len(shards), "frames": total_frames, "episodes": total_eps,
            "unique_layouts": len(set(layouts)), "frames_by_source": by_source,
            "problems": bad}
