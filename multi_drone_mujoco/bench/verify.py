#!/usr/bin/env python3
"""Prove the shared renderer produces the same pixels as the per-env renderer.

This must pass before any speed number from compare.py means anything. A shared
renderer that mixes up which image belongs to which env does not raise -- it
silently feeds the policy the wrong observation and training just gets worse.

Two checks:

  1. **Parity** -- K envs stepped through identical action sequences, once with
     per-env renderers and once through one shared renderer. Every RGB and
     depth frame must match exactly.

  2. **No cross-talk** -- the same test, but the order in which the shared
     renderer serves envs is shuffled every step. If any per-env state leaks
     across calls, differing service order changes the output and this fails
     while check 1 passes.

Run:
    python -m multi_drone_mujoco.bench.verify --envs 4 --steps 60
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys

import numpy as np

from multi_drone_mujoco.bench._common import DEFAULT_ENV, resolve_env_class
from multi_drone_mujoco.rendering import SharedStaticRenderer


def _build(env_cls, k: int, seed: int):
    return [env_cls(seed=seed + i) for i in range(k)]


def _images(env):
    """Raw (rgb, dep, seg) straight from the env's render path."""
    return env._getDroneImages(0)


def run_check(env_cls, k: int, steps: int, seed: int, shuffle: bool,
              want_seg: bool) -> dict:
    base_envs = _build(env_cls, k, seed)
    shared_envs = _build(env_cls, k, seed)

    # Baseline envs keep their own renderers; make them render the same passes
    # the shared path will, so the comparison is like-for-like.
    for e in base_envs:
        e._render_seg = want_seg

    head = shared_envs[0]
    renderer = SharedStaticRenderer(
        head.model, width=int(head.IMG_RES[0]), height=int(head.IMG_RES[1]),
        want_seg=want_seg,
    )
    for e in shared_envs:
        renderer.assert_compatible(e.model, "drone0_cam")
        e._external_renderer = renderer
        e._render_seg = want_seg

    for e in base_envs + shared_envs:
        e.reset(seed=seed)

    rng = np.random.default_rng(seed)
    order_rng = np.random.default_rng(seed + 9999)

    worst_rgb = 0
    worst_dep = 0.0
    mismatched_steps = 0
    compared = 0

    try:
        for t in range(steps):
            actions = [rng.uniform(-1, 1, size=base_envs[0].action_space.shape
                                   ).astype(np.float32) for _ in range(k)]

            for e, a in zip(base_envs, actions):
                e.step(a)
            for e, a in zip(shared_envs, actions):
                e.step(a)

            order = list(range(k))
            if shuffle:
                order_rng.shuffle(order)

            step_bad = False
            for i in order:
                b_rgb, b_dep, _ = _images(base_envs[i])
                s_rgb, s_dep, _ = _images(shared_envs[i])

                d_rgb = int(np.abs(b_rgb.astype(np.int32) - s_rgb.astype(np.int32)).max())
                d_dep = float(np.abs(b_dep - s_dep).max())
                worst_rgb = max(worst_rgb, d_rgb)
                worst_dep = max(worst_dep, d_dep)
                compared += 1
                if d_rgb != 0 or d_dep != 0.0:
                    step_bad = True

            if step_bad:
                mismatched_steps += 1
    finally:
        renderer.close()
        for e in base_envs + shared_envs:
            e.close()

    return {
        "n_envs": k,
        "steps": steps,
        "shuffled_service_order": shuffle,
        "want_seg": want_seg,
        "frames_compared": compared,
        "max_rgb_abs_diff": worst_rgb,
        "max_depth_abs_diff": worst_dep,
        "mismatched_steps": mismatched_steps,
        "passed": worst_rgb == 0 and worst_dep == 0.0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default=DEFAULT_ENV)
    p.add_argument("--envs", type=int, default=4, help="envs sharing one renderer")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seg", action="store_true",
                   help="also render and compare the segmentation pass")
    args = p.parse_args()

    env_cls = resolve_env_class(args.env)

    print(f"env={args.env}  k={args.envs}  steps={args.steps}  seg={args.seg}\n")

    results = []
    for shuffle in (False, True):
        name = "cross-talk (shuffled order)" if shuffle else "parity (in order)"
        print(f"[{name}] ...", end="", flush=True)
        r = run_check(env_cls, args.envs, args.steps, args.seed, shuffle, args.seg)
        results.append((name, r))
        verdict = "PASS" if r["passed"] else "FAIL"
        print(f" {verdict}  rgb_maxdiff={r['max_rgb_abs_diff']}"
              f"  depth_maxdiff={r['max_depth_abs_diff']:.6g}"
              f"  ({r['frames_compared']} frames)")

    print()
    if all(r["passed"] for _, r in results):
        print("ALL CHECKS PASSED — shared renderer is pixel-identical.")
        return 0

    print("FAILED — do not trust any speed comparison until this passes.")
    print("\nLikely causes, in order:")
    print("  * worlds are not actually identical across envs")
    print("  * something in the scene moves outside qpos/mocap (shared.py copies")
    print("    only those, so anything else stays at its scratch-data default)")
    print("  * env<->image association is wrong in the shared path")
    return 1


if __name__ == "__main__":
    sys.exit(main())
