# Shared static-scene rendering — what to run

Today each env builds its own `mujoco.Renderer`, so N envs = **N GL contexts and
N identical copies of the arena in VRAM**. For a static world those copies never
change. `SharedRenderVecEnv` runs the N envs in **M** worker processes with one
renderer each, so VRAM scales with M instead of N — which is what lets N grow.

Total rasterisation work is unchanged (still N images per step). What drops is
context count, VRAM and CPU-side scene setup.

Run everything on the **training box** (Linux/EGL, with the GPU), from the repo
root. Numbers from any other machine are not comparable.

---

## 0. Setup

```bash
conda activate rl_mujoco
pip install pynvml psutil matplotlib     # VRAM, host RAM, plots
```

All optional, but without `pynvml` you lose the VRAM curve — which is the main
result. Install them.

### Which env gets benchmarked

Everything defaults to `multi_drone_mujoco.bench.env:BenchAviary` — a
self-contained static world with a procedurally textured arena. No external
asset files, no imports from `rl/`.

That matters right now because **`rl.window_aviary` cannot be imported**: it
imports `imav_play` and `imav_teleop` (deleted in commit `fa05718`), and the
`Files(3)/` asset tree holding the IMAV world is gone from the working tree
too. Both `.py` files are recoverable from commit `1241352` if you want them
back; the SDF/texture assets would need to come from wherever you keep them.

**Scene weight is the knob that matters.** VRAM per GL context is driven by how
much geometry and texture the world holds, and that is exactly what shared
rendering saves. `BenchAviary(n_clutter=N)` sets it — default 24 textured
boxes. A light scene *understates* the saving. Before trusting the numbers as a
proxy for your real arena, raise `n_clutter` until per-env `vram_peak_mb` in
step 2 is in the same range as that arena's.

To benchmark a real task env instead, pass `--env 'module:Class'` to any script.

---

## 1. Verify correctness FIRST (~1 min)

Nothing below means anything until this passes. A shared renderer that hands
env 7's image to env 12 does not raise — it silently corrupts training.

```bash
python -m multi_drone_mujoco.bench.verify --envs 4 --steps 60
```

Runs two checks: **parity** (shared vs per-env pixels) and **cross-talk** (same
test with the service order shuffled every step, which catches state leaking
between envs).

**Pass criteria are asymmetric, on purpose:**

- **Depth must match exactly.** It encodes geometry and camera pose, so any
  difference is a real defect — a misplaced camera or a wrong env/image
  association shows up here immediately and hugely.
- **RGB is judged against a measured floor.** Two independent GL contexts do
  not produce bit-identical colour; the script first renders the same fixed
  state repeatedly through each path to measure that GPU's own repeatability,
  then requires the shared-vs-baseline difference to sit at or below it.

Output also reports **what fraction of pixels differ** and the mean, because a
max alone can't separate rounding from a defect: a few pixels off by 1 is
quantisation, most of the frame off by 1 is systematic. Raise `--rgb-tol` only
with a reason.

On failure the script localises the divergence to one of three causes (A:
physics already diverged, B: state didn't reach the renderer, C: identical
state but different pixels) instead of just printing diffs. If it fails, stop
and send me that block — do not run the benchmarks.

---

## 2. Baseline — the current path (~10–20 min)

```bash
python -m multi_drone_mujoco.bench.baseline --envs 1,2,4,8,16,30 \
       --out runs/bench/baseline
```

Before starting: close the viewer and any training run — `nvidia-smi` should
show no other python processes — and don't use the machine while it runs.

| Output | Meaning |
|---|---|
| `mj_step` | physics cost (CPU; unaffected by this work) |
| `rgb` / `depth` / `seg` | per-pass render cost |
| `seg_share_of_render_pct` | share spent on the segmentation pass the RL obs discards |
| `update_scene_saving_pct` | what dropping the redundant `update_scene` would save |
| `single_renderer_rgbd_fps` | frames/s one renderer sustains |
| `env_steps_per_sec` vs N | where throughput flattens = today's ceiling |
| `vram_peak_mb` vs N | slope = per-env context cost — **the ceiling being removed** |
| `vec_step.p99_ms` | latency tail |

---

## 3. Calibrate — pick M (~15–30 min)

```bash
python -m multi_drone_mujoco.bench.calibrate --n-envs 30 \
       --workers 1,2,3,5,6,10,15,30 --out runs/bench/calibrate
```

Sweeps M and picks the **smallest M whose p99 step latency hasn't started
climbing** — i.e. the fewest contexts that don't make envs wait. That's a
measured curve, not a formula, because the whole question is how much M
contexts contend with each other.

It also tells you which architecture you're in:

- **M at or above your core count** → the current grouped-worker design fits;
  nothing more to build.
- **M below your core count** → grouping ties up physics parallelism you could
  otherwise use, because here M is *both* the context count and the process
  count. Decoupling them needs a render-server design (N physics workers, M
  render processes, images over shared memory). Not built yet — tell me if the
  numbers land here.

---

## 4. Compare — old vs new (~15 min)

```bash
python -m multi_drone_mujoco.bench.compare --n-envs 30 \
       --workers <M from step 3> --repeats 3 --out runs/bench/compare
```

Runs the two configs **interleaved** (old, new, old, new, …) rather than
all-old-then-all-new — GPU clocks drift as the card heats, and a sequential
layout would systematically penalise whichever ran second. It waits between
configs so the driver actually releases VRAM before the next peak reading.

Read `run-to-run spread` in the summary: **if the spread is larger than the
difference between the two configs, the result is inconclusive** — raise
`--repeats`.

The last block extrapolates how many envs fit in your VRAM under each path.
That's the number this whole exercise is about.

---

## 5. Then push N up

The point of the VRAM saving is more envs. With M fixed from step 3:

```bash
python -m multi_drone_mujoco.bench.calibrate --n-envs 60 --workers <M>
python -m multi_drone_mujoco.bench.calibrate --n-envs 100 --workers <M>
```

Watch where `env_steps_per_sec` stops rising — that's your new ceiling, and it
should now be CPU/GPU-compute bound rather than VRAM bound.

---

## 6. Train with it

```bash
python -m rl.train_window --n-envs 30 --shared-renderers <M> --timesteps 2000000
```

`--shared-renderers 0` (default) keeps the original path unchanged.

This step needs `rl.window_aviary` importable — see the note in §0. The flag
and the VecEnv are wired up and independent of that; only this command is
blocked by the missing `imav_play` / `Files(3)`.

Note: more envs changes PPO's batch (`n_envs × n_steps`), so throughput gains
don't convert one-for-one into faster convergence without retuning `n_steps` /
`batch_size` / LR. Out of scope here, but it's why "3× the env-steps/sec" won't
read as "3× faster to a good policy".

---

## What to send me

`runs/bench/*.json`, `runs/bench/*.png`, `nvidia-smi`, and `nproc`.

---

## Known gaps

- **The redundant `update_scene`** in `_getDroneImages` is still there
  deliberately. `update_scene_saving_pct` prices it; removing it needs a pixel
  test proving depth is unaffected, since a stale scene would corrupt depth
  silently.
- **Render server (decoupled N physics / M render)** is not built — only worth
  it if step 3 says M ≪ core count.
- **`rl/README.md`** still claims no files in `multi_drone_mujoco/` are
  modified. That's no longer true: `base_aviary.py` gained the shared-renderer
  hook and the segmentation opt-out. Needs updating.
- **`rl/window_aviary.py` and `rl/_viewer.py` are broken** independently of this
  work — they import `imav_play` / `imav_teleop`, which no longer exist. The
  benchmarks route around it; training does not.
