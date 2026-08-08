# MAVRL-style pipeline (`mavrl/`)

## Context

`rl/` today trains a Crazyflie to fly through **one** window using stock SB3 PPO with a
stateless single-frame RGB-D CNN. `rl/window_world.py` builds one static wall at `y=3.30`
with two openings (red `x∈[0.44,1.32]`, blue `x∈[-1.32,0]`), both at `z∈[2.97,3.85]`, plus
three background posts. There is no memory, no VAE, no course, and no randomization beyond
spawn jitter.

We want a MAVRL-style pipeline (Yu et al., RA-L 2025) adapted to a **multi-station obstacle
course**: entry window → randomized horizontal bars → exit window. On top of the paper we
add (a) RGB-D rather than depth-only encoding, (b) a *semantically-enhanced* VAE whose
reconstruction loss is proximity-weighted and which also predicts a segmentation map, and
(c) a switchable memory backbone — LSTM vs temporal attention with recurrent memory tokens —
with Actor-Learner Distillation scaffolding.

The goal is a **working policy**, not a publishable comparison against the paper: the
fixed-speed baseline and Pareto sweep are deliberately out of scope.

## Decisions already taken

| Question | Decision |
|---|---|
| Memory backbone | LSTM **and** temporal attention behind one `memory_type` flag; ALD scaffolding built now |
| Resolution | render 512×512, downsample to 128×128×4 for the encoder |
| SeVAE semantics | proximity-weighted RGBD reconstruction **plus** a segmentation decoder head |
| Action space | paper's form: body acceleration + yaw rate |
| Observation | vision + proprio only — **no** goal pose/bearing (corridor is straight, goal vector is near-constant) |
| Course randomization | station count, station order, bar height only. No spacing jitter, no texture/colour jitter |
| Bar width | same as the existing wall, `x∈[-2,2]` |
| Layout switching | all envs share one layout; switches on success-rate threshold **and** on a period |
| Vertical profile | window at `z≈3.4 m`, bars at `0.4–2.0 m` — the descent is intended |
| Fixed-speed baseline | **not built** — goal is a working varying-speed policy, not a published comparison. AGV is still logged |
| Sensor noise | small depth + RGB noise on the encoder input, clean reconstruction targets (denoising VAE) |
| Generalization eval | one bar height per colour held out of training |

## Course geometry

Corridor `x∈[-2,2]`, `z∈[0,5]`, stations spaced **ΔY = 4.0 m** along +y.

- **Station 0** — the existing wall at `y=3.30`, unchanged. Must pass the **red** opening;
  crossing the blue opening is a `wrong_gate` failure.
- **Intermediate stations** (0–3 of them, at `y = 3.30 + k·4.0`) — one horizontal bar each,
  spanning the full corridor width, `hy=0.05`, `hz=0.04`, collidable:
  - `RED_BAR` — centre height ∈ {1.20, 1.60, 1.98} m, must pass **above**
  - `BLUE_BAR` — centre height ∈ {0.40, 0.80, 1.20} m, must pass **below**
  Free space exists on both sides; colour alone says which side is legal. Wrong side =
  `wrong_side` failure (terminate), hitting the bar = `collision` crash.
- **Exit window** — a second wall at `y = 3.30 + (N+1)·4.0`, single centred opening.
- **Lateral containment** — no new geometry. `_computeReward`'s out-of-bounds check tightens
  to `|x| > 2.0` once past station 0, so flying around a bar terminates the episode.
- **Spawn standoff** — `SPAWN_STANDOFF = 2.0 m` in front of station 0's plane, jittered to
  stay within `[1.7, 2.3] m`, rather than `SpawnConfig.BASE`'s hardcoded `y = 0.4`
  (`window_aviary.py:84`). Defined relative to the first station so it survives the course
  being repositioned. At 2 m with a 60° FOV the camera covers a 2.3 × 2.3 m patch — the full
  0.88 m opening plus surrounding wall for context, so the drone can see what it is aiming at
  from the first frame rather than starting with its nose in the wall.

## Files

Everything lands in a new **`mavrl/`** package. **`rl/` is not touched at all** — it stays
as the working single-window baseline, and `mavrl/` carries its own copy of anything it
needs (the `_box` XML helper, the entry-wall constants). That costs a little duplication and
buys total independence: breaking `mavrl/` cannot break a task that already trains.

### Status — all written and exercised

| File | Contents |
|---|---|
| `mavrl/config.py` | Every shared constant: resolutions, rates, action limits, reward weights, network widths, curriculum thresholds |
| `mavrl/course_gates.py` | `Opening`, `WallGate` (target + decoy), `BarGate` (above/below), `GateSequence`, `Outcome`. No mujoco, no torch |
| `mavrl/course_world.py` | `StationSpec`/`CourseLayout`, curriculum sampling with held-out heights, `SemClass` + `classify_geom_name`, named-geom XML, injector, `build_geom_class_lut` |
| `mavrl/imageproc.py` | 512→128 downsamplers (area RGB / min-pool depth / nearest seg), depth packing, proximity weight |
| `mavrl/sensor_noise.py` | Depth + RGB noise, encoder input only |
| `mavrl/course_aviary.py` | The env: 10 Hz policy over 60 Hz PID over 240 Hz sim, acceleration actions, per-gate reward, `set_layout` |
| `mavrl/sevae.py` | Encoder + rgb/depth/seg heads, weighted losses, class weights |
| `mavrl/memory.py` | `LSTMMemory`, `TemporalAttentionMemory` (windowed + RMT tokens), `AuxReconstructionHead` |
| `mavrl/policy.py` | `MavrlActorCritic` — encoder → memory → policy/value heads |
| `mavrl/ppo.py` | Recurrent PPO with truncated BPTT, return normalization |
| `mavrl/vecenv.py` | Vec-env construction; SB3 import deferred so offline stages don't need it |
| `mavrl/curriculum.py` | `CourseSampler`, synchronized `broadcast_layout` |
| `mavrl/dataset.py` | `ShardWriter`, `SequenceDataset`, `FrameDataset`, `summarize` |
| `mavrl/collect.py` | `CascadedPilot` + collection loop |
| `mavrl/teleop.py` | Manual keyboard collection: hold-to-fly velocity commands, recorded as accelerations |
| `mavrl/gui.py` | `ViewerWindow` (3-D, re-attaches after `set_layout`) + `FeedWindow` (pygame RGB/depth/seg + held-key state) |
| `mavrl/merge_data.py` | Combines shard directories; guards filename collisions and mismatched columns |
| `mavrl/visualize.py` | Sample-image dumps for the SeVAE and the memory aux head; matplotlib imported lazily on Agg |
| `mavrl/bc.py`, `train_sevae.py`, `train_memory.py`, `train_course.py`, `ald.py`, `evaluate.py` | Training stages |
| `tests/test_mavrl_geometry.py` | 22 tests, no simulator required |
| `notebooks/mavrl_modal.ipynb` | 57 cells |
| `mavrl/README.md` | How to run it: pipeline order, every command, what each number means |
| `pyproject.toml` | `mavrl` extra, `mavrl*` packaged, `tests` on the pytest path |

### What implementation changed from the plan

Each of these came out of a measurement, not a preference.

- **`STATION_SPACING` 2.5 → 4.0 m.** The vertical profile does not fit in 2.5 m: leaving the
  entry window at z ≈ 3.4 and reaching the exit at 1.5 is a 2 m descent in ~1.7 s. The pilot
  clipped the exit sill at z = 1.06 on essentially every run. At 4.0 m it exits dead centre
  (z = 1.49 against a 1.50 target). Spacing is still constant, as specified.
- **`MIN_FLIGHT_Z = 0.25`** in `course_gates`. `height − clearance` for a blue@0.40 bar is
  z = 0.05 — exactly the env's ground-termination threshold, so the waypoint was the kill
  line. Stage-3 runs died with `ground` while tracking their target correctly.
- **The pilot drives the env's velocity integrator directly** (`a = (v_des − v_cmd)/dt`)
  rather than running a second P loop on velocity error. The command passes through three
  lags — pilot → integrator → PID — and a P-P cascade overshot laterally to x = −1.02
  against an opening edge at −0.44. Per-axis braking conservatism `(0.5, 0.5, 0.30)`, with z
  tightest because arresting a descent needs thrust above weight.
- **Offscreen framebuffer patch.** MuJoCo defaults to 640×480, so a 512² render fails
  outright. `patch_offscreen_framebuffer` edits the existing `<global>` tag inside
  `<visual>` — a second `<global>` element is rejected.
- **`state_batch_dim` is declared by each memory module, never inferred.** LSTM state is
  `(layers, B, H)` and attention state is `(B, …)`; when `n_envs == n_mem` those are
  indistinguishable by shape, and the shape-sniffing version silently sliced the wrong axis.
- **`SequenceDataset` iterates shard-at-a-time.** The per-window `np.load` fully decompressed
  the shard on every access: one epoch over 1621 frames went from >10 min to 1.1 s.
- **Custom `mavrl/ppo.py` instead of `sb3_contrib.RecurrentPPO`** — see §6.
- **Depth is clipped to `DEPTH_MAX`** in `build_observation`; MuJoCo returns the far-clip
  distance (~1130 m) for sky pixels, which is both out of the declared observation space and
  wasteful in float16 storage.
- **`ep_source` added to the shard format.** Two collectors now write into the dataset
  (scripted and manual), and once merged there is otherwise no way to tell which frames came
  from where — which matters, because human demos are good for the SeVAE and questionable for
  BC. `summarize` reports the frame mix.
- **pygame rather than OpenCV + a keyboard library.** `key.get_pressed()` is a state poll, so
  "hold to move, release to stop" is exact; a press-event callback (MuJoCo's viewer
  `key_callback`, `cv2.waitKey`) has no release event and would have to reconstruct hold from
  OS key repeat, complete with its ~500 ms initial delay.
- **An epoch that runs zero batches now raises.** `iter_batches` skips any shard holding
  fewer than `batch_size` windows, so `--seq-len` above the episode length (or an oversized
  batch) produced a full training run reporting `past=0.00000` and saving an untrained
  checkpoint. Found by a stub dataset whose episodes were shorter than the window.
- **Both offline stages write sample PNGs** (`mavrl/visualize.py`). A SeVAE recon loss of 0.026
  says nothing about whether bars survived, and a memory `past` loss says nothing about whether
  `Î_{t-T}` is a remembered bar or the mean corridor. `--no-samples` opts out.
- **The aux head's λ is a flag, not a constant.** It was hard-wired to two segments (past +
  current) on a claim about the paper's λ sweep that the paper text does not actually support.
  `--aux-segments` now selects any subset of `past,current,future`, the head is sized to the
  selection, and the choice rides in the checkpoint so warm starts resize rather than fail.
  The default is still `past,current` — Fig. 2(b)'s future reconstruction is the blurriest of
  the three — but it is now a default rather than an assumption.

### Measured on an RTX 2050 (laptop)

- **11.8 policy-steps/s** at 512² with three render passes. Modal will be substantially faster;
  `RENDER_RES` is one constant if it still hurts.
- **Scripted pilot: 5/5 on every curriculum stage**, including the tight blue@0.40 gate.
- **Class pixel shares**: FREE 49.7 %, WALL 40.0 %, FLOOR 5.4 %, RED_WINDOW 2.4 %,
  RED_BAR 1.15 %, BLUE_BAR 1.13 %, BLUE_WINDOW 0.24 % — confirming the 1–4 % estimate for
  bars and the need for CE class weights.
- **The blue decoy is a weak distractor.** Spawn is centred on the red opening, so at 2 m with
  a 60° FOV the blue window is off-frame (0.24 % of pixels, absent entirely at spawn). The
  colour-discrimination task at the entry is therefore much easier than at the bars. Worth
  revisiting if you want the decoy to actually bite.

### Reused from `rl/` by reading, not importing

- `rl/window_aviary.py:170` `_preprocessAction` — the PID velocity-servo call pattern.
- `rl/train_window.py:95` `WarmStartStabilizer` — ported as `WarmStartFreeze`.
- `rl/light_cnn.py` — the no-memory ablation baseline.


## Implementation

### 1. Env: rates, action space, observation

**Fixed planner rate.** `sim_freq=240`, `ctrl_freq=60`, **policy 10 Hz** → `CTRL_PER_POLICY = 6`.
Override `step()` to hold one acceleration command across 6 inner `BaseAviary.step` calls
(`base_aviary.py:528`), summing reward and rendering **only on the last** — a `_skip_render`
flag makes `_computeObs` return the cached dict on inner ticks. Gate crossings are still
checked every inner tick, so fast passes aren't missed. This also cuts render calls ~5×
versus today's every-control-step rendering, which pays for the 512² raster.

**Action → acceleration.** Replace `_preprocessAction` (`window_aviary.py:170`):

```
a_body = action[:3] * A_MAX          # A_MAX ≈ [3.0, 3.0, 2.0] m/s²
yaw_rate = action[3] * V_YAW
v_cmd_body += a_body * POLICY_DT     # integrate, clip to V_MAX
```

then feed `v_cmd` world-rotated into the existing `PIDControl` velocity servo exactly as
today (`target_pos = cur_pos`, `target_vel = v_world`). `v_cmd_body` resets to zero on
`reset()`.

**Observation.**
- `image`: `(128, 128, 4)` uint8 — RGB + depth-normalized-by-`DEPTH_MAX` (`= 12.0 m`, raised
  from 8.0 so the depth channel spans the whole course; see §4).
  Render at `RENDER_RES = 512`, then downsample: **area-average for RGB**, **min-pool for
  depth** (preserves thin bars — a mean would blur a bar into the background), **nearest for
  seg**. `RENDER_RES` is a constant so it can be dropped to 256 if throughput hurts.
- `proprio`: `(16,)` = `[grav_body(3), gyro(3), vel_body(3), v_cmd_body(3), prev_action(4)]`.
  All onboard-observable; `v_cmd_body` is the integrator state the policy needs now that
  actions are accelerations.
- `seg`: `(128,128)` uint8 class map, **collection-time only** — added to the obs dict only
  when `env.collect_mode=True`, so PPO rollout buffers don't carry it.

**Segmentation classes.** `mujoco.Renderer` in segmentation mode returns `(H,W,2)` int32
(`objid`, `objtype`) — note `base_aviary.py:863`'s zero fallback is `(H,W)`, so shapes differ
between branches and the consumer must handle both. Map `objid → class` where `objtype ==
mjOBJ_GEOM`, via `mavrl.course_world.SemClass` / `classify_geom_name` (name-prefix based,
which is why `_box` names the geom and not just the body):
`{0 free, 1 wall, 2 red_window, 3 blue_window, 4 red_bar, 5 blue_bar, 6 floor}`.
Segmentation is **already rendered and discarded** in the existing env
(`window_aviary.py:197`), so the seg head is nearly free.

**Background posts are dropped.** `rl/window_world.py:126` places three at `y = 5.2` and
`6.4`, which is inside this corridor and would collide with stations 1 and 2. They existed
so the view through the opening was not a flat void; the course itself now supplies that
depth structure.

### 2. Course randomization + curriculum

`CourseLayout = (station_types, bar_heights)`. Stages:

| Stage | Intermediate stations |
|---|---|
| 0 | none — entry → exit window |
| 1 | 1 bar, either colour, any height |
| 2 | 2 bars, both orders |
| 3 | 3 bars |

**Held-out heights.** Training samples only two of the three heights per colour (red
{1.20, 1.98}, blue {0.40, 1.20}); the middle value of each is reserved for evaluation. Costs
nothing and turns the result into a generalization claim rather than a memorization one.

Stage advances when rolling success over the last `N=200` episodes ≥ `0.8`; within a stage
the layout resamples every `K` PPO rollouts. **All envs share one layout at a time** — a
`SetLayoutCallback` broadcasts via `venv.env_method("set_layout", layout)`, and each env
rebuilds its model/data/renderer. Because layouts stay synchronized, the
`SharedStaticRenderer` path (`multi_drone_mujoco/rendering/shared.py:60`) remains valid; it
just needs re-attaching after each rebuild. Start with plain `SubprocVecEnv` and treat the
shared path as an optional follow-up.

### 3. Dataflow (and exactly where it differs from the paper)

The paper's VAE is **depth-image-only** — no state, no velocity, no action inside it. The
state vector `x_t` enters in two other places: the LSTM's auxiliary reconstruction head
(training-only, Fig. 2(a)'s red dotted box), and PPO's input. The LSTM's *input* is `z_vae`
alone.

Note also **which stage is multi-frame**. The VAE (Eq. (1)) is trained strictly per-frame:
`MSE(I_t, I_recon_t)`, no time index, no recurrence — "train the VAE, skipping the LSTM
phase in this step". Multi-frame reconstruction belongs to the *LSTM* stage (Eq. (2)), where
one FC layer emits `3 · N_e`, split into past / current / future segments, all decoded by the
same decoder, with `λ_i ∈ {0,1}` selecting which are supervised. Stage 2 here therefore reads
the shards through `FrameDataset` and stage 3 through `SequenceDataset`.

```
image ──[SeVAE enc]──► z_vae (64) ──[memory]──► z_t (256)
                                                  │
       ┌──────────────────────────────────────────┴────────────────┐
       │ TRAIN ONLY (discarded at deploy)                          │ DEPLOY
       ▼                                                           ▼
 concat(z_t, proprio) ─FC─► k·64 ─split─► ẑ_{t-T}, ẑ_t, ẑ_{t+T}   concat(z_t, proprio) ─► PPO ─► a_t
                                            │
                            each ──[SAME frozen SeVAE decoder]──► Î
```

`k` is the number of segments `λ` selects, set by `--aux-segments` (`AUX_SEGMENTS` in
`config.py`). The head is sized to the *selected* segments — an unsupervised segment gets no
gradient, so emitting it would only be dead weights. Default is `past,current` (λ = 1,1,0);
`past,current,future` reproduces the paper's full three-way head. The default follows the
paper's Fig. 2(b): the future reconstruction is visibly the blurriest of the three ("highest
MAE loss"), which is what you would expect when the drone cannot see 20 steps ahead — that
segment is partly unlearnable and mostly contributes gradient noise. The paper's own λ sweep
is in its Section V; the flag is here so it can be reproduced rather than assumed.

**Our only deviations from the paper are inside the SeVAE box**: the input is RGB-D rather
than depth, and the loss gains proximity weighting plus a segmentation head. Latent width,
memory width, aux-head structure, decoder sharing, and the `concat(z, x)` points are all the
paper's.

### 4. SeVAE (`mavrl/sevae.py`)

Encoder: 6 conv layers (paper's depth), input `(B,4,128,128)`, → `μ, log σ²`, `N_e = 64`.
Decoders share a trunk, three heads: RGB (3ch, sigmoid), depth (1ch, sigmoid), seg (`C`
logits). No state input — matching the paper.

```
t   = clip((d − D_NEAR) / (D_FAR − D_NEAR), 0, 1)     # D_NEAR = 0.5 m, D_FAR = 12.0 m
w   = W_MIN + (1 − W_MIN) · (1 − t)²                  # W_MIN = 0.05, never exactly 0
L   = mean(w ⊙ (MSE_rgb + MSE_depth))
    + λ_seg * mean(CE(seg_logits, seg_target))        # seg CE is NOT proximity-weighted
    + β_norm * KL
```

Weight profile: `1.00` at contact, `0.77` at 2 m, `0.31` at 6 m, `0.05` floor from 12 m out.
Quadratic rather than exponential so the decay is gradual through the near field and only
collapses in the last few metres — an `exp(−d/2)` would already be at 0.05 by 6 m, throwing
away the mid-field where the *next* station first becomes visible. The `W_MIN` floor keeps a
non-zero gradient at all ranges, so the far field is de-emphasised rather than discarded.

**This forces `DEPTH_MAX = 12.0 m`** (was 8.0). A 12 m rolloff is meaningless if the depth
channel clips at 8 m, and the course is ~13 m end to end (entry `y = 3.3`, exit up to
`y = 13.3`), so at 8 m the drone cannot see the far half of it at all. Cost: uint8
quantization coarsens from 3.1 cm to 4.7 cm per level, which is still finer than the sensor
noise model in §4b.

The proximity weight `w` makes the latent spend capacity on near geometry; the seg head is
what makes it *semantic* — weighting alone cannot teach "red bar" vs "blue bar". Class
weights in the CE compensate for bars being a small pixel fraction.

**KL sign.** Eq. (1) as printed in the paper reads `L_KL = ½Σ(1 − μ² − σ² + log σ²)`, which
is the *negative* of the KL divergence — minimizing it as written would blow the posterior
up. Implement the standard `−0.5 · Σ(1 + logvar − μ² − exp(logvar))`. This is silent if you
transcribe the paper literally, so it is called out here.

**Segmentation loss is uniform, not proximity-weighted** (`--seg-weight {uniform,proximity}`,
default `uniform`). With τ = 2 m the weight falls to 0.018 at 8 m, which would zero out
gradients exactly where the *next* bar's colour cue lives — and colour is what decides
above-vs-below. Proximity weighting belongs on RGB/depth reconstruction, where near geometry
genuinely matters more; on the class map it works against the semantic goal.

**Class imbalance.** A 0.08 m bar at 60° FOV over 128 px subtends `9.78/d` pixels: ~5 px at
2 m, 2 px at 5 m, ~1 px at 8 m — 1–4 % of the frame. CE class weights need to be ~25–100×,
and **bar mIoU is the metric that will actually be hard**. (At `RENDER_RES = 512` the bar is
20/8/5 px before min-pooling, which is why the 512 render earns its cost — it is preserving
real signal, not just antialiasing.)

### 4b. Sensor noise (`mavrl/sensor_noise.py`)

MuJoCo gives perfect ray-traced depth. The paper never trained on that — AvoidBench runs SGM
stereo matching *specifically* "to replicate realistic depth errors, reducing the gap between
simulation and reality." Perfect depth lets the VAE encode sub-pixel-crisp edges no real
sensor produces, and the policy will come to depend on them.

Applied to the **encoder input only; reconstruction targets stay clean** — i.e. a denoising
VAE. Corrupting both would spend latent capacity modelling noise.

| Channel | Model | Default |
|---|---|---|
| Depth | range-dependent Gaussian, `σ(d) = k·d²` (D435i-class: ~1 % at 2 m, quadratic growth) | `k = 0.001` |
| Depth | edge dropout — invalidate pixels at depth discontinuities, the dominant real stereo failure | 2 % of edge pixels |
| Depth | random invalid/zero pixels | 0.5 % |
| RGB | additive Gaussian + mild brightness/gamma jitter | `σ = 2/255`, gamma ±5 % |

This is *sensor* noise, distinct from the arena texture/colour randomization you ruled out —
the world stays visually identical, only the measurement of it is corrupted.

Two constraints: the same noise model must be active during collection, SeVAE training, and
RL, or a train/deploy gap opens; and note the existing 8-bit depth quantization
(`DEPTH_MAX = 8 m` → 3.1 cm/level, ~0.9 cm RMS) is already comparable to a RealSense at 1 m,
so some noise exists whether or not this is enabled. All magnitudes are config so the whole
thing can be ablated to zero.

### 5. Memory (`mavrl/memory.py`)

Both variants map a stream of `z_vae ∈ R^64` → `z_t ∈ R^256`:

- `LSTMMemory` — single-layer LSTM, hidden 256.
- `TemporalAttentionMemory` — ring buffer of the last `K = 16` latents, learned positional
  encoding, 2 layers × 4 heads, plus `n_mem = 4` **recurrent memory tokens** (`--mem-tokens`,
  0 = plain windowed attention). Policy output is the last latent's token.

**What T actually is.** `T = 20` is *not* a memory window — it is the **prediction offset in
the auxiliary loss**: from `z_t`, reconstruct the image seen 20 steps ago. The LSTM's `h_t`
rolls forward with no horizon; T is a supervision target that pressures `h_t` into retaining
~20 steps, not a cap on what it can hold. The real gradient limit is BPTT truncation
(`n_steps`), not T.

**Why memory tokens, and why K < T.** Plain windowed attention *is* hard-capped at K, unlike
the LSTM — so the two are not equal-capacity baselines. Recurrent memory tokens (Bulatov et
al. 2022, RMT; cf. Transformer-XL segment recurrence) fix that: the tokens' output at step
`t` becomes their input at `t+1`, so forward memory is unbounded while attention keeps sharp
within-window access.

```
step t:  [ m_{t-1} | z_{t-K+1} … z_t ] ──attn──► [ m_t | … | z_t^out ]
               │                                     │          │
               └────────── carried to t+1 ───────────┘          └──► policy
```

This also repairs the aux objective. With `K = 32 > T = 20` the attention variant could
satisfy `MSE(I_{t-20}, Î_{t-20})` by **copying `z_{t-20}` straight out of its window** — no
compression, while the LSTM faced a genuine memory task. Setting **K = 16 < T = 20** puts
step `t-20` outside the window, so the only route to it is through the memory tokens. Both
backbones now face real compression pressure. Gradient horizon is still BPTT-bounded for
both, so the comparison is fair rather than free.

Memory input is `z_vae` only by default. A `--memory-inputs {latent,latent+state}` flag
reproduces the paper's alternative variant that also feeds `proprio` in (which already
carries `a_{t-1}`, so no separate action input is needed).

**Where the state vector is injected, in full** — three sites, none of them the SeVAE:

| Site | Expression | Live at deploy? |
|---|---|---|
| Aux reconstruction head | `Linear(256 + 16 → k·64)`, k = len(λ) | no — training only |
| PPO feature vector | `concat(z_t, proprio)` → 272 | yes |
| Memory input (optional flag) | `concat(z_vae, proprio)` → 80 | yes, if enabled |

Note this is **not** the paper's `x_t = [d_hor, v_hor, β′, d_z, v_z, χ′, ψ]`, which is mostly
goal-relative. We occupy the same architectural slot with
`[grav_body(3), gyro(3), vel_body(3), v_cmd_body(3), prev_action(4)]` — no goal terms, per
the straight-corridor decision.

Aux training head (paper §III-B, Eq. (2)): `Linear(256 + 16 → k·64)` split into the segments
`λ` selects from `ẑ_{t-T}, ẑ_t, ẑ_{t+T}`, each decoded by the **frozen** SeVAE decoder; loss
is the sum of `MSE(I_{t+iT}, Î_{t+iT})` over them, with **T = 20**. Encoder frozen during this
stage.

Only timesteps whose every target lies inside the training window are supervised, so adding
`future` costs T steps off the *end* of each window as well as the T `past` already costs off
the front: at the default `--seq-len 48`, `past,current` supervises steps [20, 48) and
`past,current,future` only [20, 28). Raise `--seq-len` to ~64 when running the three-way
config, or the epoch is doing a third of the work it looks like it is.

### 6. PPO wiring

Feature extractor output is `concat(z_t, proprio)` = `256 + 16 = 272`, straight into the
policy/value MLP — the paper's `concat(z_t, x_t) → PPO`.

Use `sb3_contrib.RecurrentPPO` so sequence-aware rollout buffers and BPTT come for free.
`RecurrentActorCriticPolicy` carries state as `RNNStates(pi=(h,c), vf=(h,c))`:
- LSTM — the natural fit.
- Attention — `h` holds the `(K, 64)` ring buffer plus write index, `c` holds the
  `(n_mem, d)` recurrent memory tokens. Both slots used, which maps onto the API more
  honestly than the flat-pack hack; still document it at the definition so it isn't
  mistaken for an LSTM.

Reuse `WarmStartStabilizer` (`train_window.py:95`) when initializing from BC — it already
freezes the extractor and delays the actor so a random critic can't destroy the BC policy.

### 7. Reward

Per-gate rather than single-gate. Keep `K_PROG`, `K_TIME`, `K_SMOOTH` from
`window_aviary.py:114`; retarget progress at the **next uncleared gate's** waypoint.

**The progress waypoint must lie *in* the gate plane, not past it.** `rl/`'s single-window
task uses `WP = [0.88, 3.80, 3.41]`, half a metre beyond the wall at `y = 3.30`, which is
harmless there because the approach is nearly axial. On this course it is not: gates sit
metres apart in z, so straight-line pursuit toward a point beyond a bar crosses that bar's
plane while still descending toward the target altitude. Measured on a dead-centre pursuit
trajectory: crossing a blue bar at `z = 1.36` when the bar sits at `1.20` — a `WRONG_SIDE`
failure on a trajectory that was aiming correctly. `WallGate.waypoint()` and
`BarGate.waypoint()` therefore default to `standoff = 0.0`. The scripted pilot in §8 keeps
its own approach→exit waypoint chain, which is a separate concern from the dense reward.

| Term | Value | When |
|---|---|---|
| progress toward next gate | `K_PROG = 1.0` | every policy step |
| time | `−K_TIME = −0.03` | every step |
| jerk `‖a_t − a_{t−1}‖` | `−K_SMOOTH` | every step |
| centering near a gate plane | `−K_CENTER` | `|y − y_gate| < 1.5` |
| **per-gate pass** | `+R_TOTAL / n_gates`, `R_TOTAL = 150` | each gate cleared, in order |
| **course-complete** | `+R_FINISH = 100` | all gates cleared |
| **time bonus** | `+K_TBONUS · max(0, T_nom − t)`, `T_nom = 4 + 2·N_stations` | **only** on course-complete |
| wrong gate / wrong side | `−R_CRASH`, terminate | blue opening, or wrong side of a bar |
| collision / ground / flip / `\|x\|>2` | `−R_CRASH`, terminate | as today |

The time bonus firing only on full completion is what you asked for: speed pays nothing
until the whole course is solved, so the policy learns to clear gates before it learns to
rush. Speed variation then emerges from `K_TIME` + the completion bonus traded against
collision risk, which is MAVRL's varying-speed mechanism.

**Constant return scale across curriculum stages.** Per-gate reward is
`R_TOTAL / n_gates_in_this_course` — a constant per layout, **not** divided by gates
*passed*. On a 6-gate course each gate pays `R_TOTAL/6`, so clearing 1 earns 1/6 and
clearing all 6 earns the lot; the incentive to pass more is untouched. What it buys is that
a 2-gate and a 6-gate course both cap at the same return, so the value function does not
have to rescale every time the curriculum advances — which would otherwise destabilize PPO
at exactly the moment the task also changes. `T_nom` scales with `N_stations` for the same
reason. Wrap with `VecNormalize(norm_reward=True, norm_obs=False)`.

Episode limit scales with course length: `T_max = 8 + 4·N_stations` seconds.

**Jerk penalty is evaluated at the policy boundary**, not per inner control tick — otherwise
`prev_action` updates inside `_computeReward` (`window_aviary.py:265`) make the delta zero on
5 of every 6 ticks.

**Logged metrics.** Success rate per stage, plus **average goal velocity (AGV)** and the
per-episode speed profile. No fixed-speed baseline is built, but AGV is what shows whether
the policy is genuinely modulating speed or just flying flat out.

### 8. Data collection (`mavrl/collect.py`)

Scripted privileged pilot, structurally like `pilot_action` (`rl/pilot_record.py:34`) but
following a waypoint **chain**: for each gate, an approach point ~0.8 m before the plane at
the legal altitude, then the gate centre itself. The approach point is what makes the pilot
arrive at the right height *before* crossing — the same failure mode described in §7, and
the reason the pilot's chain is richer than the dense reward's single waypoint.

**Acceleration labels are produced natively, not by differentiation.** A cascaded
controller — P on position error → velocity target, P on velocity error → acceleration —
emits the acceleration command directly. Differentiating a velocity command instead would
amplify controller noise straight into the BC targets.

DAgger-style noise on the executed action, clean label recorded.

Two behaviours you specifically asked for:
- **Episodes do not terminate on a missed gate.** In `collect_mode` a `wrong_gate` /
  `wrong_side` flags the info dict and flies on, so SeVAE still gets frames from failure
  states. Only collisions and the time limit end an episode.
- **A different layout per episode**, sampled across all curriculum stages, so the SeVAE
  sees every station type and bar height.

Dataset format — `rl/`'s existing `demos.npz` is 8042 flat i.i.d. frames with **no episode
boundaries**, which cannot train a sequence model. New format (and a `layout` descriptor per
episode, so the set can be stratified and debugged):

```
image      (N, 128, 128, 4) uint8
seg        (N, 128, 128)    uint8
depth_m    (N, 128, 128)    float16   # metres, for the proximity weight
proprio    (N, 16)          float32
action     (N, 4)           float32
ep_start   (M,)             int64     # episode boundaries
ep_len     (M,)             int64
ep_layout  (M,)             object    # CourseLayout descriptor per episode
gate_flags (N,)             uint8     # which gates were cleared by this frame
```

Sharded `.npz` per episode batch — a single array at 128² with seg and depth gets large fast.

### 9. Training stages

1. **Stage 1** — PPO with frozen random SeVAE + memory, stage-0 course. The paper's step 1
   collects its VAE dataset with this policy; we collect with the privileged scripted pilot
   instead (`collect.py` takes no policy), because the same trajectories have to serve as BC
   demonstrations and a barely-trained policy makes poor ones. Stage 1's role here is to
   validate the full PPO stack on the easiest course before spending compute on the rest.
2. **Collect** — `mavrl/collect.py`, mixed layouts, non-terminating on missed gates.
3. **Stage 2** — `mavrl/train_sevae.py`: SeVAE on the collected RGBD + seg + depth.
4. **Stage 3** — `mavrl/train_memory.py`: memory module with frozen encoder, multi-frame
   reconstruction through the frozen decoder; `--aux-segments` is the paper's λ.
5. **BC** — sequence-aware `mavrl/bc.py` on the pilot data.
6. **Stage 4** — `RecurrentPPO` with curriculum, warm-started from BC.
7. **ALD** (`mavrl/ald.py`) — transformer learner + LSTM actor, actor collects, learner runs
   PPO, actor minimizes `KL(π_learner ‖ π_actor)`. Run **after** stage 4 establishes whether
   attention actually beats LSTM here; if it doesn't, ALD has nothing to distill.

### 10. Modal notebook

**`notebooks/mavrl_modal.ipynb`** drives the whole pipeline. Assumes the repo is already
cloned and the working directory is the repo root.

| # | Cell | Contents |
|---|---|---|
| 1 | Install | `%uv pip install "mujoco>=3.0" "gymnasium>=0.29" "stable-baselines3>=2.0" sb3-contrib torch numpy imageio imageio-ffmpeg tensorboard tqdm` |
| 2 | Runtime | `os.environ["MUJOCO_GL"] = "egl"` **before** any mujoco import; `sys.path` to repo root; print `torch.cuda.is_available()`, GPU name, `mujoco.__version__`, `nvidia-smi` |
| 3 | Volume | bind `runs/`, `data/`, `ckpt/` to a Modal volume mount; assert they are writable and **not** on the container's ephemeral disk |
| 4 | Geometry check | CPU-only: sample layouts, print `layout.describe()`, assert station y-spacing and that gate planes are strictly increasing. Runs before any GL work, so a geometry bug is caught without a GPU |
| 5 | Render smoke test | build one `CourseAviary`, render at 512², show RGB / depth / seg side by side. **Fails loudly here if EGL is broken**, rather than 40 minutes into training |
| 6 | Layout preview | third-person render of every curriculum stage (0–3 bars) so the course is eyeballed before committing compute |
| 7 | Noise preview | same frame with `sensor_noise` on and off, side by side, to sanity-check magnitudes |
| 8 | Stage 1 | `!python -m mavrl.train_course --stage 0 --frozen-encoder --timesteps 1_000_000` |
| 9 | Collect | `!python -m mavrl.collect --episodes 2000 --mixed-layouts --out data/` + assert `ep_start`/`ep_len` partition `N` and missed-gate episodes are present |
| 10 | Stage 2 | `!python -m mavrl.train_sevae --data data/ --epochs 50` + inline reconstruction grid + bar mIoU |
| 11 | Stage 3 | `!python -m mavrl.train_memory --memory-type lstm --T 20`, then `--memory-type attention --mem-tokens 4`, plus the `--mem-tokens 0` control |
| 12 | BC | `!python -m mavrl.bc --data data/ --sequence` |
| 13 | Stage 4 | `!python -m mavrl.train_course --init ckpt/bc_init.zip --curriculum --timesteps 5_000_000` |
| 14 | ALD | `!python -m mavrl.ald --learner attention --actor lstm` |
| 15 | Eval | success rate + AGV per stage; held-out-height eval; rollout video via imageio shown inline |
| 16 | TensorBoard | `%load_ext tensorboard` / `%tensorboard --logdir runs/` |

**Long-running cells** (8, 9, 13, 14) launch detached — `nohup … > runs/<stage>.log 2>&1 &`
— followed by a separate tail-and-poll cell. A dropped notebook connection then costs you
the log view, not the run. Every such cell prints its PID so it can be killed deliberately.

**Ordering matters**: cells 4–7 are all cheap and all catch a different class of failure
(geometry, GL, visual sanity, noise scale). They come before anything that costs GPU-hours
on purpose.

## Verification

The Windows checkout can't run any of it — `mujoco`, `gymnasium`, `stable_baselines3` are
not importable and `torch.cuda.is_available() == False`; the code hardcodes `device="cuda"`
and `MUJOCO_GL=egl`. Everything below runs on Modal / the Linux+CUDA box.

Pure-python checks that **do** run on Windows — `mavrl.course_gates` and
`mavrl.course_world` deliberately import no mujoco and no torch at module level, which is
what makes this possible:

1. `python -c "from mavrl.course_world import *; print(world_xml(sample_layout(...)))"` —
   inspect generated XML for a few layouts; assert bar heights and station y-positions.
   **Already run**: 3-station layout produced 20 named bodies / 20 named geoms, stations at
   `y = 5.8 / 8.3 / 10.8`, exit at `13.3`.
2. Unit-test `course_gates.py`: forward vs backward crossings, red-above vs blue-below,
   the decoy opening, `GateSequence` ordering. **This is what caught the waypoint-past-the-
   plane bug in §7** — a pursuit trajectory aiming at the correct gate still scored
   `WRONG_SIDE`, which is exactly the failure a unit test should surface before a GPU does.
3. Unit-test the 512→128 downsamplers on synthetic images: a 1-pixel-wide bright bar must
   survive min-pool depth downsampling, and seg must contain no interpolated class ids.

On Modal:

4. `python -m mavrl.course_aviary` self-test — print obs shapes, step 50 times, confirm
   reward is finite and the policy rate is 10 Hz (`step_counter` advances by 24 sim steps
   per `env.step`).
5. Layout-preview cell — render every curriculum stage third-person; confirm gates register
   in order and wrong-side passes are flagged.
6. `python -m mavrl.collect --episodes 200` then assert the npz shapes, that
   `ep_start`/`ep_len` partition `N` exactly, and that missed-gate episodes are present.
7. `python -m mavrl.train_sevae` — check reconstructions, and specifically that near-field
   detail is sharper than far-field (that's the proximity weight working); report bar mIoU.
8. `python -m mavrl.train_memory --memory-type lstm|attention` — check `Î_{t-20}` recovers a
   bar that has left the field of view. For the attention variant, verify `--mem-tokens 0`
   does **worse** than `--mem-tokens 4`: with `K = 16 < T = 20` a plain window physically
   cannot reach `t-20`, so a gap there confirms the memory tokens are carrying it. Optionally
   rerun with `--aux-segments past,current,future --seq-len 64` to reproduce the paper's λ
   sweep; expect `future` to sit well above the other two and `past` to be no better for it.
9. `python -m mavrl.train_course --memory-type lstm` vs `--memory-type attention` vs
   `--memory-type none`, ≥3 seeds each; compare success rate and AGV per curriculum stage.
   This is the ablation that decides whether ALD is worth running.
10. **Held-out generalization** — evaluate on the reserved bar heights (red 1.60, blue 0.80),
    which never appear in training. A large train/held-out gap means the policy memorized
    heights rather than learning the above/below rule.
11. **Noise ablation** — retrain with `--sensor-noise 0`. If success collapses when noise is
    later re-enabled, the policy was leaning on perfect depth edges.

## Risks

- **512² render cost.** 3 passes × 512² per policy step. The 10 Hz decimation buys back ~5×,
  but if throughput is still bad, drop `RENDER_RES` to 256 — it's a single constant.
- **Curriculum stalling.** If stage 2→3 never clears 80%, the run silently sits at stage 2.
  Log the stage to TensorBoard and add a max-dwell fallback.
- **Attention-in-`RNNStates`.** Ring buffer in `h`, memory tokens in `c` is still a
  reinterpretation of SB3's API; if it fights back, the fallback is a custom PPO loop for the
  attention variant only.
- **Memory tokens can be ignored.** Nothing forces the policy to write anything useful into
  them early in training, and a collapsed memory token is silent. `K < T` in the aux loss is
  the forcing function; verification step 8 is the check that it worked.
- **Flat memory ablation.** The genuine memory demand here is gate counting (~12 steps), not
  the paper's getting-stuck-on-large-obstacles scenario — a thin bar never fills the FOV. If
  LSTM, attention and `none` all score the same, that is the likely cause, and it is a real
  result rather than a bug.
- **Action-space change invalidates existing artifacts.** `demos.npz`, `runs/bc_init.zip`,
  and `runs/window/best_model.zip` are all velocity-action; none carry over.
