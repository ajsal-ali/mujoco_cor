# `rl/` — Vision-based RL for window / obstacle traversal

Reinforcement-learning stack for flying the repo's Crazyflie **through the red
window** of the IMAV2026 arena, designed to generalize to the wider course
later. Built entirely on top of the package's `BaseAviary` (real CF2 drone +
`PIDControl` velocity engine + onboard-camera renderer) — **no files in
`multi_drone_mujoco/` are modified**. The IMAV world is injected at runtime via
`imav_play.make_world_injector` (same mechanism as `imav_play.py`).

Run everything from the repo root inside the `rl_mujoco` conda env:

```bash
conda activate rl_mujoco
cd ~/Documents/donttouch/MuJoCo-drones-gym
```

---

## Design in one screen

| Aspect | Choice | Why |
|---|---|---|
| **Observation** | `Dict{ image: RGB-D 96×96×4 (uint8), proprio: 9 }` | Vision + IMU-realizable state only — **no privileged window pose**, so it can transfer to a real drone. |
| `proprio` (9) | `[gravity-dir-in-body(3), body angular-vel(3), body linear-vel(3)]` | What an IMU / VIO gives on the real bird. |
| **Action** | continuous velocity `[vx, vy, vz, yaw_rate] ∈ [-1,1]` | Scaled to m/s and driven through `PIDControl` as a pure velocity set-point (the validated `imav_play` controller). Smooth, agile, real-transferable. |
| **Reward / termination** | privileged sim geometry (allowed) | Only the *checker* uses the known window geometry — the *policy* never sees it. |
| **Algorithm** | PPO (SB3), `SubprocVecEnv`, GPU | Proven for gate flight; scales headless. |
| **Bootstrap** | teleop → behavior cloning → PPO fine-tune | So RL starts from a competent policy, not noise. |

The observation deliberately contains **no window pose** — the CNN must *see* the
opening. The reward is free to use the exact geometry because it only runs in sim.

---

## Files

| File | Role |
|---|---|
| `window_aviary.py` | `WindowAviary(BaseAviary)` — the gym env. Also holds `WindowGate` (geometry + traversal check) and `SpawnConfig`. |
| `light_cnn.py` | `LightCNNExtractor` — a small (~0.3 M param) RGB-D CNN + proprio MLP for SB3. `POLICY_KWARGS` wires it into PPO. |
| `_viewer.py` | `run_viewer(...)` — shared interactive `mujoco.viewer` loop (3rd-person, prop spin, auto-reset). |
| `view_env.py` | **Ephemeral env viewer** — look at / fly / watch the env anytime; close the window and it's gone. |
| `teleop_record.py` | Hand-fly and record demos → `demos.npz`. Prints live traversal status from the same checker the reward uses. |
| `bc_pretrain.py` | Behavior-clone a PPO policy from `demos.npz` → a warm-start `.zip`. |
| `train_window.py` | Headless PPO training (EGL + `SubprocVecEnv`), optional `--init` warm-start. |
| `eval_window.py` | Watch a trained policy: interactive viewer + onboard RGB-D window. |

---

## Typical workflow

```bash
# 1. Look at the environment (teleop by default; --model to watch a policy; --mode idle to hover)
python -m rl.view_env

# 2. Collect demos. PREFERRED: a scripted expert (clean, consistent, learnable).
#    Human hold-to-fly teleop demos are NOT a consistent function of the image
#    (held key => constant action regardless of view) so BC can't clone them —
#    the policy just flies straight. Use the scripted pilot instead:
python -m rl.pilot_record --episodes 40 --out demos.npz
#    (teleop_record still exists for watching/validating the checker by hand.)

# 3. Behavior-clone a policy from the demos (so PPO doesn't start from trash)
python -m rl.bc_pretrain --demos demos.npz --out runs/bc_init.zip --epochs 15

# 4. Train PPO, warm-started from BC, fully headless across many envs
python -m rl.train_window --init runs/bc_init.zip --n-envs 8 --timesteps 2000000

# 5. Watch the result (3rd-person viewer + live onboard RGB-D)
python -m rl.eval_window --model runs/window/best_model.zip
```

You can skip steps 2–3 and train from scratch (`python -m rl.train_window ...`),
but expect much slower first success.

### Controls (view_env / teleop_record)
Hold a key to fly; release and the drone brakes to rest.

```
W / S   forward / back      Up / Down    up / down
A / D   left / right         Left / Right yaw
H       hover (zero)         (mouse: orbit / pan / zoom)
```

Click the 3rd-person window first so it has keyboard focus.

---

## The traversal checker (`WindowGate`)

The red window is the obstacle wall at **y = 3.30**, with a flyable
opening **x ∈ [0.44, 1.32], z ∈ [2.97, 3.85]** (center ≈ `(0.88, 3.30, 3.41)`).
(The blue window is the mirror opening at `x ∈ [-1.32, 0.0]`.)

Detection is a **plane crossing + in-opening bounds check** (not a sphere test):

1. signed distance to the plane `s = pos_y - 3.30`, tracked step to step;
2. on a forward crossing (`s_prev < 0 ≤ s_cur`), interpolate the `(x, z)` where
   the path meets the plane;
3. inside the opening (minus a `MARGIN` of 0.06 m) ⇒ **passed**; outside ⇒ **hit
   wall** (crash).

Collisions with *any* obstacle (posts, bars, tubes, frame, ground) are detected
via MuJoCo contacts on the drone's collision geom — so this generalizes beyond
windows. `teleop_record.py` prints this checker's result live, so hand-flying
verifies the exact logic that trains the policy.

---

## Reward (privileged, sim-only)

Per step, `r =`
- `+ K_PROG · (d_prev − d)` — progress toward the waypoint just past the window (`d = ‖pos − WP‖`)
- `− K_TIME` — constant time penalty (encourages speed)
- `− K_SMOOTH · ‖Δaction‖` — smoothness (helps sim-to-real)

Near the window plane (`|y − 3.30| < 1.5 m`), two shaping terms:
- `+ K_ALIGN_STEP · (sin(yaw) − 1)` — face the window normal (+y); 0 when aligned, negative otherwise
- `− K_CENTER · off` — stay on the center-line (`off = ‖(x,z) − center‖`); grows as it drifts off-center

Terminal:
- **pass** ⇒ `+ R_PASS + K_TBONUS · time_left` + `K_ALIGN · max(0, sin(yaw))` (aligned crossing)
  + `K_CENTER_CROSS · max(0, 1 − off/0.44)` (centered crossing)
- **crash** (hit wall / collision / ground / flip / out-of-bounds) ⇒ `− R_CRASH`

Defaults (top of `window_aviary.py`): `K_PROG=1.0, K_TIME=0.03, K_SMOOTH=0.01,
R_PASS=100, R_CRASH=25, K_TBONUS=5, K_ALIGN=15, K_ALIGN_STEP=0.05, K_CENTER=0.1,
K_CENTER_CROSS=10`.

---

## Key knobs (edit `window_aviary.py`)

- `IMG_W, IMG_H` — onboard image size (default 96). Offscreen buffer supports up to 640×480.
- `DEPTH_MAX` — depth clip (m) used to pack depth into the 4th image channel.
- `V_XY, V_Z, V_YAW` — velocity action scaling (m/s, rad/s).
- `EPISODE_LEN_SEC` — episode time limit (default 8 s).
- reward weights `K_*, R_*` — shaping.
- `SpawnConfig.BASE / JIT / YAW_JIT` — the slightly-randomized start pose that
  always faces the window (used identically in teleop, training, eval).
- `WindowGate.*` — window geometry; change these to target a different opening.

CNN: `light_cnn.py` is **early-fusion** (RGB-D stacked into one CNN), the lightest
baseline. To upgrade to two-stream (separate RGB and depth encoders), split
`self.cnn` and concat features before `img_head` — see the note in that file.

---

## Notes & caveats

- **Fully headless training.** `train_window.py` sets `MUJOCO_GL=egl` and renders
  offscreen. Visual inspection is only ever via `view_env.py` / `eval_window.py`
  (ephemeral windows — close = gone). Don't open a viewer inside training.
- **GLFW + Tk deadlock**: an interactive `mujoco.viewer` and a matplotlib window
  cannot coexist in one process, so `eval_window.py` runs the onboard RGB-D window
  in a **separate process** (`imav_play._onboard_window`).
- **Validated end-to-end**: a policy behavior-cloned from pilot demos passes the
  window **~83%** using only vision + proprioception — the observation carries
  enough signal without privileged pose.
- The cosmetic spinning propellers are injected by `imav_play.make_world_injector`
  and are visual-only (contype 0); they don't affect physics or training.
- **Sim-to-real / generalization (later):** the observation is already onboard-
  realizable. Extending to the whole course = add more waypoints (collision
  handles the obstacles) and randomize the target; an optional privileged
  teacher → vision-student distillation can speed that up. See the project plan.
