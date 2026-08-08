# `mavrl/` — memory-augmented vision-only flight through a bar course

A MAVRL-style pipeline (Yu et al., *MAVRL: Learn to Fly in Cluttered Environments With
Varying Speed*, IEEE RA-L, Feb 2025) for a task the paper does not have: a corridor with a
**red entry window** (a blue window beside it is a decoy), **0–3 horizontal bars** — red means
*fly above*, blue means *fly below* — and an **exit window**. The policy sees RGB-D and its own
body state. It gets no goal pose, no bearing, and no privileged geometry: the colour of the
bar in front of it is the only thing that says which way to go.

`MAVRL_PLAN.md` at the repo root is the design document — why each number is what it is. This
file is how to run it.

---

## What differs from the paper

| | Paper | Here |
|---|---|---|
| Encoder input | depth only | **RGB-D** — the task is colour-conditioned, so depth alone is unsolvable |
| VAE loss | plain MSE | **proximity-weighted** MSE + a **segmentation head** (SeVAE) |
| Memory | LSTM | LSTM **or** windowed attention with RMT memory tokens, behind `--memory-type` |
| Simulator | AvoidBench + SGM stereo | MuJoCo, with a synthetic depth/RGB noise model in its place |
| Baselines | fixed-speed comparison, Pareto sweep | dropped — the goal is a working policy |

Everything else — 64-d latent, 256-d memory, the six-conv encoder, `concat(z_t, x_t)` at both
the aux head and PPO, the frozen-decoder aux loss, `T = 20` — is the paper's.

## Which stage is multi-frame

Worth being precise about, because it is easy to get backwards:

- **The VAE is per-frame.** Paper Eq. (1) is `MSE(I_t, I_recon_t)` — no time index, no
  recurrence, no state input. Its own recipe says to train it "skipping the LSTM phase in this
  step". Stage 2 here reads the shards through `FrameDataset`.
- **The LSTM stage is multi-frame.** Paper Eq. (2): one FC layer emits `3 · N_e`, split into
  past / current / future segments (`Î_{t-T}, Î_t, Î_{t+T}`), all pushed through the *same
  frozen decoder*, with `λ_i ∈ {0,1}` choosing which are supervised. Stage 3 reads through
  `SequenceDataset` and `--aux-segments` is that λ.

`T = 20` is a **prediction offset, not a memory window**. An LSTM hidden state has no horizon;
`T` is the supervision target that pressures it into retaining ~20 steps of history.

---

## Install

```bash
pip install -e ".[mavrl]"        # torch, stable-baselines3, imageio, tqdm, matplotlib
```

Offline stages (SeVAE, memory, BC, all the geometry) need only numpy + torch. Only
`train_course`, `ald` and `vecenv` touch stable-baselines3, and that import is deferred inside
`build_venv` so the rest of the package imports without it.

Set the GL backend **before** any mujoco import — every script does
`os.environ.setdefault("MUJOCO_GL", "egl")`, so exporting it first wins:

```bash
export MUJOCO_GL=egl      # headless Linux / Modal
export MUJOCO_GL=glfw     # local machine with a display
```

## Pipeline

Seven stages, in order. Stages 1–5 mirror the paper's own recipe; 6–7 are the task.

```
1  train_course --stage 0 --frozen-encoder   PPO with a random frozen encoder (smoke test)
2  collect                                   scripted pilot -> sharded dataset
3  train_sevae                               per-frame VAE, encoder frozen after this
4  train_memory                              memory backbone, frozen encoder + decoder
5  bc                                        sequence-aware behaviour cloning
6  train_course --curriculum --init ...      PPO, warm-started
7  evaluate                                  per-stage success + held-out heights
```

```bash
# 1. validates the whole PPO stack on the easiest course before anything expensive.
#    The paper collects its VAE dataset with this policy; we collect with the scripted pilot
#    instead, since the same trajectories also have to serve as BC demonstrations.
python -m mavrl.train_course --stage 0 --frozen-encoder --timesteps 1000000

# 2. data. Does NOT end an episode on a missed gate -- the SeVAE needs failure states, or the
#    latent space only ever models clean traversals. A different layout every episode.
python -m mavrl.collect --episodes 2000 --out data --split all

# 3. the semantically-enhanced VAE
python -m mavrl.train_sevae --data data --epochs 50

# 4. the memory backbone (see below for the variants worth running)
python -m mavrl.train_memory --memory-type lstm --T 20

# 5. behaviour cloning onto the recurrent policy, warm-started from 3 and 4
python -m mavrl.bc --data data --memory-type lstm

# 6. the real run
python -m mavrl.train_course --curriculum --init ckpt/bc_init.pt \
    --memory-type lstm --timesteps 5000000

# 7. evaluation, including the held-out bar heights
python -m mavrl.evaluate --model runs/course/final.pt --memory-type lstm
```

`--split all` in step 2 is deliberate: the *encoder* is allowed to have seen a bar height the
*policy* never trains on. Training uses red {1.20, 1.98} and blue {0.40, 1.20}; red 1.60 and
blue 0.80 are held out, and step 7 reports the gap. A large gap means four heights were
memorized rather than the rule "red → above".

## Stage 4 in detail — the memory ablation

```bash
python -m mavrl.train_memory --memory-type lstm                        # the paper's backbone
python -m mavrl.train_memory --memory-type attention --mem-tokens 4    # windowed + RMT memory
python -m mavrl.train_memory --memory-type attention --mem-tokens 0    # the control
```

The `--mem-tokens 0` run is the point of the comparison. `ATTN_WINDOW = 16 < AUX_T = 20`, so a
plain window **physically cannot see** step `t-20`; only the recurrent memory tokens can carry
it. Watch the `past` column: if `--mem-tokens 0` is not clearly worse there, the memory tokens
are not doing their job (or the task never needed memory — see the caveat below).

`current` is near-free for any backbone and tells you very little.

Each run also writes `ckpt/memory_<tag>_samples.png` — see *Seeing whether it works*.

### λ — `--aux-segments`

```bash
python -m mavrl.train_memory --aux-segments past,current                 # default, λ = (1,1,0)
python -m mavrl.train_memory --aux-segments past,current,future --seq-len 64   # paper's full head
```

The default drops `future` because the paper's Fig. 2(b) shows it reconstructing far more
blurrily than the other two — the drone cannot see 20 steps ahead, so that segment is partly
unlearnable and mostly contributes gradient noise. The flag is here so that is a default and
not an assumption.

**Raise `--seq-len` when you add `future`.** Only timesteps whose every target lies inside the
window are supervised, so `future` costs `T` steps off the *end* as well as the `T` that `past`
already costs off the front. At the default `--seq-len 48`: `past,current` supervises steps
[20, 48); `past,current,future` supervises only [20, 28). The script prints the range it
actually used — read it.

The λ configuration is written into the checkpoint, and `bc.py` / `train_course.py` /
`evaluate.py` resize the aux head to match on load. The head is training-only and dropped at
deploy, so a mismatch resizes rather than fails.

## Seeing whether it works

Losses do not tell you whether the bars survived. Both offline stages write a PNG when they
finish, next to the checkpoint:

| file | what it shows |
|---|---|
| `ckpt/sevae_samples.png` | noisy input / clean target / reconstruction, for RGB, depth and segmentation |
| `ckpt/memory_<tag>_samples.png` | ground truth vs reconstruction for every aux segment |

Regenerate either at any time, for any checkpoint:

```bash
python -m mavrl.visualize --data data --sevae ckpt/sevae.pt     --memory ckpt/memory_lstm.pt --out samples
```

`--no-samples` on the training scripts turns it off.

**In `sevae_samples.png`** — near geometry should be visibly sharper than far. That asymmetry
*is* the proximity weight; if near and far are equally blurry the weight is not reaching the
loss. Then check `pred seg` contains bars at all: a bar is 1–4 % of the frame, so a model that
never predicts one still scores well on pixel accuracy and the loss will not tell you.

**In `memory_*_samples.png`** — this is the one that answers your question about the memory.
`Î_{t-20}` is reconstructed from `z_t` alone, so anything recognizable in it is information the
recurrent state *carried* after the frame left the field of view. The `current` pair below it
is the control: any backbone gets that nearly for free. A sharp `current` beside a featureless
smear for `past` means the backbone is describing, not remembering.

## The env

| | |
|---|---|
| Rates | 240 Hz sim → 60 Hz PID → **10 Hz policy** (24 sim steps per `env.step`) |
| Action | `[a_x, a_y, a_z, ψ̇]`, body frame, integrated into a velocity setpoint the PID tracks |
| Observation | `image (128,128,4) uint8` + `proprio (16,)` |
| Proprio | `[grav_body(3), gyro(3), vel_body(3), v_cmd_body(3), prev_action(4)]` — **no goal terms** |
| Render | 512², downsampled to 128 (area-mean RGB, **min-pool** depth, nearest seg) |
| Depth | clipped to 12 m |

Min-pool for depth, not area-mean: a bar 5 px wide at 2 m and ~1 px at 8 m is averaged out of
existence by a mean filter. `tests/test_mavrl_geometry.py` pins that.

`collect_mode=True` adds `image_gt`, `seg` and `depth_m` from the **clean** render, so the
dataset trains reconstruction against ground truth while the encoder sees noise — a denoising
VAE, and the reason `image` and `image_gt` are separate arrays.

## Layout and randomization

All parallel envs share one layout at a time; the curriculum broadcasts a new one at a
**rollout boundary** via `venv.env_method("set_layout", …)`. Switching mid-rollout would
invalidate value estimates already collected against the old geometry.

Stations are 4.0 m apart (constant), red bars at 1.20 / 1.60 / 1.98 m, blue at 0.40 / 0.80 /
1.20 m, spawn ~2 m before the entry window so the drone can see it. Advancement is on a rolling
200-episode success rate above 80 %, with a forced advance after 400 rollouts so a stalled
stage cannot deadlock the run.

## Tests

```bash
pytest tests/test_mavrl_geometry.py -q      # 22 tests, no simulator, no GPU
```

Geometry, downsamplers, proximity weight, noise model. These caught the two worst bugs in the
build — a waypoint placed *past* the gate plane (correct pursuit, `WRONG_SIDE` outcome), and a
blue-bar waypoint that landed exactly on the ground-termination threshold.

## Modal

`notebooks/mavrl_modal.ipynb` drives all of it — install via `%uv pip install`, a render smoke
test that fails loudly if EGL is broken rather than 40 minutes into training, layout and noise
previews, one cell per stage with `nohup … &` plus a log-tail helper so a dropped connection
does not kill a run, and inline eval video.

## Two things to know before reading results

**The blue decoy window is a weak distractor.** Spawn is centred on the red opening, so at
2 m with a 60° FOV the blue window is off-frame; it is 0.24 % of pixels across the whole
dataset. Colour discrimination at the entry is close to trivial. The bars are where the rule
actually gets tested.

**The memory ablation may come out flat.** The paper's obstacles fill the field of view; a
thin horizontal bar does not, and it stays visible for most of the approach. If LSTM,
attention and `none` all score the same on stage 3, that is a finding about this course, not a
broken implementation — and it means ALD (`mavrl/ald.py`) has nothing worth distilling.
