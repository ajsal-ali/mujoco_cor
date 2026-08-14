# `mavrl/` — memory-augmented vision-only flight through a bar course

A MAVRL-style pipeline (Yu et al., *MAVRL: Learn to Fly in Cluttered Environments With
Varying Speed*, IEEE RA-L, Feb 2025) flown in the **real IMAV2026 arena** —
`Files(3)/imav2026_scaled.sdf`, converted to MJCF by `imav_teleop.SdfToMjcf` and injected the
way `imav_play.make_world_injector` does it. Nothing about the arena is invented.

The task: through the **red entry window** (a blue window beside it is a decoy), past the
fixed obstacles, through **0–3 horizontal bars** — red means *fly above*, blue means *fly
below* — and out through the **exit window**. The policy sees RGB-D and its own body state. It
gets no goal pose, no bearing, and no privileged geometry: the colour of the bar in front of
it is the only thing that says which way to go.

## The arena

Travel is along **+y**, starting from the arena's own `takeoff_platform` plane. Only the bar
stations vary between episodes.

| y | what | varies? |
|---|---|---|
| −14.30 | spawn — the `takeoff_platform` plane, one station spacing back | fixed |
| **−12.10** | entry wall — red opening (x 0.44–1.32, z 2.97–3.85) + blue decoy | fixed |
| −9.90 / −7.70 / −5.50 | **bar stations** (slot 0 first) | **count, colour, height** |
| −2.20 | `tube_B` — posts at x=±1.10, a horizontal at z=1.01, a diagonal | fixed |
| 0.00 | `tube_A` — vertical post at x=0 | fixed |
| **+3.30** | exit wall — identical to the entry wall | fixed |

Plus three ground boxes, a turbine, a ring board and four platforms, all fixed.

The SDF's model names run the other way — what it calls `exit_wall_*` is the wall you enter
through, because the takeoff platform sits beyond it. The two walls are geometrically
identical, so the course is symmetric and direction is pure convention; this is the
competition's convention, and it puts the bars *before* the tubes.

`course_gates.TRAVEL_SIGN` is the single source of truth for that direction. Crossings,
progress, AGV, spawn side, the pilot's approach points and the tests all derive from it —
nothing hardcodes a y comparison, so flipping the course is a one-line change.

**`tube_B` is the reason the scripted pilot has via-points.** Its right post sits at x = 1.10
and the exit window's red opening is centred at x = 0.88 — a straight run from the last bar
(x = 0) to that window passes about 0.12 from the post, inside the pilot's own lateral
overshoot, and collides. The pilot threads x = 0.35 at `tube_B` and closes the lateral gap
afterwards. The *policy* gets no such hint; it can see the tubes.

### Units: the arena is the competition course × 2.2

The filename says "scaled" and the numbers agree exactly — `red_bar` at z = 4.356 is
2.2 × 1980 mm, `blue_bar` at 0.880 is 2.2 × 400 mm, stations 2.20 apart is 2.2 × 1 m.
**Every constant in this package is in SDF units**; competition millimetres appear only in
comments. Bar heights are therefore red {2.640, 3.520, 4.356} and blue {0.880, 1.760, 2.640}.

Mixing the two frames is the one mistake this codebase is built to prevent, because it
already happened: the window constants came from the SDF and the bar heights from the spec
sheet, which put a 3 m dive out of every window and got mis-diagnosed as a spacing problem.
`test_scale_relates_sdf_units_to_competition_millimetres` pins it.

`MAVRL_PLAN.md` at the repo root is the design document — why each number is what it is. This
file is how to run it.

---

## What differs from the paper

| | Paper | Here |
|---|---|---|
| Encoder input | depth only | **RGB-D** — the task is colour-conditioned, so depth alone is unsolvable |
| VAE loss | plain MSE | **proximity-weighted** MSE + a **segmentation head** (SeVAE) |
| Memory | LSTM | LSTM **or** windowed attention with RMT memory tokens, behind `--memory-type` |
| Simulator | AvoidBench + SGM stereo | MuJoCo on the IMAV2026 arena, with a synthetic depth/RGB noise model in place of SGM stereo |
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
export MUJOCO_GL=egl      # headless Linux server
export MUJOCO_GL=glfw     # local machine with a display
```

## Headless GPU rendering

A server has no monitor, and MuJoCo's default GL backend wants one. `MUJOCO_GL=egl` is the
answer — EGL creates a **surfaceless** context straight on the GPU, with no X server, no
display and no `xvfb` in the way. `osmesa` also works headless but rasterizes on the CPU,
which for this project is not a fallback so much as a way to make a two-day run into a
two-month one.

Check the box before trusting it with a run:

```bash
python -m mavrl.glcheck
```

Four tiers — environment, context, *which* GL implementation answered, then the real
512² course render, timed. The third tier is the one worth having. A software rasterizer
does not raise anything; it returns correct frames at 1/50th the rate, so the only symptom
is a run that never finishes, and `glcheck` names `GL_RENDERER` rather than leaving you to
infer it. It exits non-zero on software, so it can gate a job script.

**The three things that actually break, in order of how often:**

| Symptom | Cause | Fix |
|---|---|---|
| `import mujoco` → `'NoneType' object has no attribute 'eglQueryString'` | no `libEGL.so.1` on the box, so PyOpenGL hands MuJoCo `None` | `apt-get install libegl1 libglvnd0 libgles2 libglx0`, or the container |
| `GL_RENDERER = llvmpipe` | only Mesa's ICD is listed, so libglvnd never reaches NVIDIA's EGL | write `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` (see below) — installing `libegl1` brings `50_mesa.json` with it, and 10 sorts first |
| context creation fails outright | no EGL device — headless VM with no GPU passthrough, or a container without the GPU | `nvidia-smi` first; if that fails nothing else matters |
| `GL_FRAMEBUFFER_UNSUPPORTED (0x8CDD)` | NVIDIA surfaceless contexts reject MuJoCo's default 4× multisampled offscreen buffer | already handled — `patch_offscreen_framebuffer` sets `offsamples="0"` |

### Picking the NVIDIA vendor, not Mesa

Under EGL, *which* GL implementation you get is decided entirely by the JSON files in
`/usr/share/glvnd/egl_vendor.d/`, in filename order. `libEGL_nvidia.so.0` being present on
the box means nothing if no ICD names it — libglvnd falls to Mesa and renders on the CPU,
with no error anywhere. Installing `libegl1` makes this *more* likely, because it ships
`50_mesa.json`.

```bash
printf '%s' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

`10` sorts before `50`, so NVIDIA wins. To bypass the directory scan entirely — useful when
you cannot write there — set `__EGL_VENDOR_LIBRARY_FILENAMES` to that file's path.

Measured on an A10: llvmpipe gives ~17 policy steps/s for one env; the same box on the GPU
is two orders of magnitude past that. `glcheck` fails below 50.

### No root on the server

A GPU box often has the NVIDIA driver's `libEGL_nvidia.so.0` but not libglvnd's `libEGL.so.1`
— the thin shim that routes an EGL call to the vendor library. `glcheck` names that case
specifically. With root it is `apt-get install -y libegl1 libopengl0`; without root, the same
two libraries unpack into `$HOME`:

```bash
bash scripts/gl_no_root.sh
. ~/.mavrl_gl_env          # `.`, not `source` -- the server's /bin/sh is dash
python -m mavrl.glcheck
```

`apt-get download` and `dpkg-deb -x` are both unprivileged, so this installs nothing
system-wide and `rm -rf ~/.local/gl` undoes it. The script also makes the unversioned
`libEGL.so` symlink, which the runtime package omits and which `ctypes.util.find_library` —
what PyOpenGL calls — needs before it will return anything but `None`.

### In a container

`docker/Dockerfile` builds a CUDA image with the GL loader and the EGL vendor ICD in place:

```bash
docker build -t mavrl -f docker/Dockerfile .        # from the repo root
docker run --rm --gpus all mavrl                    # CMD is glcheck
docker run --rm --gpus all -v "$PWD/runs:/workspace/runs" mavrl \
    python3 -m mavrl.train_course --stage 3 --curriculum --n-envs 60
```

The line that matters is `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`. The default
is `compute,utility`, which mounts CUDA and `nvidia-smi` but **not** the GL libraries — so
`--gpus all` gives a container that trains on the GPU and renders on the CPU, with nothing
anywhere reporting a problem. Most CUDA Dockerfiles omit it because most CUDA workloads
never draw anything. This one does, once per env per policy step.

Build from the repo root, not from `docker/`: the context has to include the IMAV arena
under `Files(3)`, which `course_world.py` refuses to run without.

## Pipeline

Seven stages, in order. Stages 1–5 mirror the paper's own recipe; 6–7 are the task.

```
1  train_course --stage 0 --frozen-encoder   PPO with a random frozen encoder (smoke test)
2  collect                                   scripted pilot -> sharded dataset
2b teleop / merge_data (optional)            fly it yourself, then combine
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

### Step 2 has two collectors

```bash
# scripted pilot -- headless, thousands of episodes, unattended
python -m mavrl.collect --episodes 2000 --out data --split all

# ...the same thing with windows open, so you can watch it
python -m mavrl.collect --episodes 20 --out data --gui

# you fly it yourself
python -m mavrl.teleop --episodes 30 --out data_manual

# combine them into one trainable directory
python -m mavrl.merge_data --out data_all data data_manual
```

Every stage after step 2 points at `data_all` instead of `data`.

**Manual controls** (feed window must be focused):

| key | |
|---|---|
| `W` / `S` | forward / back (body +x / −x) |
| `A` / `D` | left / right |
| `SPACE` / `L-SHIFT` | up / down |
| `Q` / `E` | yaw — only with `--free-yaw` |
| `[` / `]` | speed scale down / up |
| `ENTER` | end episode, **keep** it |
| `BACKSPACE` | end episode, **discard** it |
| `P` | pause |
| `ESC` | quit, saving everything so far |

Hold-to-fly: while a key is held that axis commands `±V_MAX × scale`, and the
instant it is released that axis's target goes to **zero** — a velocity command, so
the drone brakes to a stop rather than drifting. Axes are independent: releasing
`SPACE` while still holding `W` zeroes the climb and keeps the forward speed.

Two windows open for both `--gui` and `teleop`: a third-person MuJoCo viewer, and a
feed window showing RGB / depth / segmentation at the 128 px the encoder actually
receives — not the 512 px raster. `--no-seg` drops the third panel.

Both need a display, so on your machine: `MUJOCO_GL=glfw`. `pip install pygame` (or
`pip install -e ".[mavrl-gui]"`); headless collection never imports it.

**The recorded label is an acceleration either way.** The env's action space is body
acceleration + yaw rate, so teleop converts your velocity target the same way the
scripted pilot does — `a = (v_target − v_cmd)/dt`. If it recorded raw velocities the
`action` column would mean different things in different rows and the two datasets
could not be mixed.

Yaw is **locked** to the corridor heading by default so manual data matches the
scripted set. `--free-yaw` hands `Q`/`E` to you, which is better for the SeVAE
(stranger viewpoints) and worse for BC (inconsistent with the rest).

One honest caveat, and it's the one `rl/pilot_record.py` already documents: hold-to-fly
demos are not a clean function of the camera image — the action stays constant while a
key is held, regardless of what comes into view — so BC on human data tends to regress
toward the mean. For the **SeVAE** the opposite is true; human flying visits states no
scripted controller reaches, which is exactly what the encoder wants. Every episode
carries `ep_source` (`"scripted"` / `"manual"`) so you can filter by it later.

`merge_data` guards the two things that actually break a hand-merge: filename
collisions between sources (shards are renamed `<source-dir>__<original>`), and
mismatched columns — a shard from a different `IMG_RES` or an older format loads fine
alone and explodes mid-epoch. It refuses rather than mixing; `--force` skips the bad
shard instead. `--link` hard-links rather than copying.

`--split all` in step 2 is deliberate: the *encoder* is allowed to have seen a bar height the
*policy* never trains on. Training uses red {2.640, 4.356} and blue {0.880, 2.640}; red 3.520
and blue 1.760 are held out, and step 7 reports the gap. A large gap means four heights were
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
| Depth | clipped to 16 (SDF units) — keeps the next two stations in the sensor |
| Semantics | 8 classes: free, wall, red/blue window, red/blue bar, floor, **obstacle** (tubes, boxes, turbine, ring board) |

Min-pool for depth, not area-mean: a thin bar covers a couple of pixels at range and a mean
filter averages it out of existence. `tests/test_mavrl_geometry.py` pins that.

Segmentation is classified by **body** name, not geom name: `SdfToMjcf` emits `g1, g2, ...`
for geoms and puts the meaningful name on the enclosing body, so the LUT goes through
`model.geom_bodyid`. That one rule covers both the converted arena and the generated bars.

`collect_mode=True` adds `image_gt`, `seg` and `depth_m` from the **clean** render, so the
dataset trains reconstruction against ground truth while the encoder sees noise — a denoising
VAE, and the reason `image` and `image_gt` are separate arrays.

## Layout and randomization

All parallel envs share one layout at a time; the curriculum broadcasts a new one at a
**rollout boundary** via `venv.env_method("set_layout", …)`. Switching mid-rollout would
invalidate value estimates already collected against the old geometry.

That one-layout rule is also what makes **shared rendering** legal, and `build_venv` uses it by
default: N envs run in M = `ceil(N/4)` processes, each holding a single GL context and a single
copy of the arena, instead of one context and one copy per env. Rasterisation work is unchanged
— still N frames per vec step — but VRAM scales with M, which is what lets `--n-envs` grow past
the point where the card fills up at 20 % utilisation. Tune it with `--render-workers`, or fall
back to one context per env with `--no-shared-render` when a rendering bug is suspected.
Segmentation is not rendered on this path: the training env never runs `collect_mode`, so that
pass was a full render per frame thrown away.

A layout swap compiles a new `MjModel` per env, so the shared renderer re-adopts it
(`SharedStaticRenderer.rebind`) immediately after `set_layout` returns and re-asserts that every
env in the group took the same geometry. Without that it would keep rasterising the *previous*
course — wrong pixels, no error.

Stations sit on the SDF's own planes, in travel order (y = −9.90 / −7.70 / −5.50, 2.20 apart); red bars at
2.640 / 3.520 / 4.356, blue at 0.880 / 1.760 / 2.640; spawn 2.20 before the entry wall,
centred on the red opening so the target is the first thing the camera sees. Advancement is on
a rolling 200-episode success rate above 80 %, with a forced advance after 400 rollouts so a
stalled stage cannot deadlock the run.

Within a stage **only the bar heights are redrawn**, every 50 rollouts; station count and colour
order change when the stage does and not otherwise. Heights come from `--split`: the default
`train` holds red 3.520 and blue 1.760 back so `evaluate.py` can ask whether the colour rule
generalized, and `--split all` trains on all three per colour, which gives that question up.

### Pinning one stage

`--stage N` starts there, and stage 3 is `max_stage`, so the advancement branch can never fire
— the run stays on the full course. Keep `--curriculum` anyway: it is the only thing that calls
`broadcast_layout`, so without it the heights never get redrawn either and the policy trains
against one frozen course for the entire run.

```bash
python -m mavrl.train_course --stage 3 --curriculum --split all \
    --init runs/course/final.pt --memory-type attention --mem-tokens 4 \
    --frozen-encoder --n-envs 60 --envs-per-batch 8 --n-steps 64 --out runs/stage3
```

`--init` takes a PPO checkpoint as readily as `bc_init.pt` — `RecurrentPPO.save` writes the same
`"policy"` key. Leave `--critic-warmup` at its default when jumping stages: `R_TOTAL` is split
across gates, so per-gate credit drops from 75 at stage 0 to 30 at stage 3 and the value head
arrives calibrated to the wrong scale even though the actor does not.

That same split is why **stage 3 returns are lower by construction** and not a symptom: five
gates instead of two, longer episodes, more accumulated `K_TIME`. Read the `gates` panel in
`curves.png` instead — it is the fraction cleared, so it compares across stages in a way raw
return does not. Gates climbing while return is flat is a run that is working.

The trap in a pinned stage is that colour order is drawn **once**, at `CourseSampler`
construction, from `--seed`, and never redrawn — nothing advances the stage. With only heights
varying, red→blue→red is a fixed sequence the policy can fly as *up, down, up* without ever
reading colour, which looks like success until the arrangement changes. Check it with
`evaluate.py` on a different colour order before believing a stage-3 success rate; if it
collapses there, rotate `--seed` across runs.

**Consecutive stations are only 2.20 apart**, so a blue bar at 0.880 followed by a red at
4.356 is a 3.5-unit swing inside one spacing. That is not flyable at full forward speed —
which is the point. The scripted pilot throttles its forward speed by the vertical error still
outstanding, and measured AGV falls from 1.61 (no bars) to 0.79 on a red/blue/red whipsaw.
That spread is the varying-speed signal the policy is meant to reproduce.

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
