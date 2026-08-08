#!/usr/bin/env python3
"""Data collection with a scripted privileged pilot.

Two behaviours this needs that ordinary rollout collection does not:

* **Episodes do not end on a missed gate.** The SeVAE has to see failure states
  -- approaching a bar on the wrong side, drifting toward the decoy window -- or
  the latent space only ever models clean traversals. Only collisions and the
  time limit end an episode here.
* **A different layout every episode**, drawn across all curriculum stages and
  every bar height (including the eval-only ones, since the encoder is allowed
  to have seen a height the *policy* has not).

Acceleration labels are produced **natively** by a cascaded controller
(P on position error -> velocity target, P on velocity error -> acceleration).
Differentiating a velocity command to get acceleration would amplify controller
noise straight into the BC targets.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mavrl import config as C                                    # noqa: E402
from mavrl.course_aviary import CourseAviary                     # noqa: E402
from mavrl.course_gates import BarGate, BarSide, WallGate        # noqa: E402
from mavrl.course_world import (                                 # noqa: E402
    STAGE_STATIONS, sample_layout,
)
from mavrl.dataset import ShardWriter, summarize                 # noqa: E402
from mavrl.sensor_noise import NoiseConfig                       # noqa: E402


APPROACH_DIST = 0.8      # metres before a gate plane to hit the legal altitude


def gate_targets(gate) -> list:
    """Approach point then gate centre, both on the legal side.

    The approach point is the whole trick: it forces the altitude to be correct
    *before* the plane is crossed. Aiming straight at a bar's centre from above
    would cross the plane while still descending, which is exactly the
    wrong-side failure the dense reward's waypoint placement also guards against.
    """
    c = gate.center
    return [np.array([c[0], gate.y - APPROACH_DIST, c[2]]), c.copy()]


class CascadedPilot:
    """P on position -> velocity target; P on velocity -> acceleration.

    The velocity target is **deceleration-limited**: per axis it never exceeds
    `sqrt(2 * a_max * |error|)`, the fastest approach from which the drone can
    still stop on the target.

    Without that limit a plain P-P cascade overshoots badly here, because the
    command passes through three lags -- pilot acceleration, the env's
    integration into a velocity setpoint, and the PID tracking that setpoint.
    Measured overshoot before the fix: lateral excursion to x = 0.45 against an
    opening edge at 0.44, and arrival at the exit window at z = 1.06 against a
    sill at 1.06. Both scraped the frame.
    """

    #: Per-axis braking conservatism (world x, y, z). Z is tightest because a
    #: descent has to be arrested by thrust exceeding weight, so the achievable
    #: deceleration is well below A_MAX -- and the blue@0.40 gate leaves only
    #: ~0.3 m between the bar's underside and the ground cutoff.
    BRAKE_SAFETY = (0.5, 0.5, 0.30)

    def __init__(self, kp_pos: float = 1.6, kp_vel: float = 2.5,
                 kp_yaw: float = 2.0, brake_safety=None):
        self.kp_pos = kp_pos
        self.kp_vel = kp_vel
        self.kp_yaw = kp_yaw
        self.brake_safety = np.asarray(
            brake_safety if brake_safety is not None else self.BRAKE_SAFETY,
            dtype=float)

    def _velocity_target(self, err: np.ndarray) -> np.ndarray:
        a_max = np.array(C.A_MAX)
        v_max = np.array(C.V_MAX)
        v_brake = self.brake_safety * np.sqrt(2.0 * a_max * np.abs(err))
        speed = np.minimum(np.abs(self.kp_pos * err),
                           np.minimum(v_max, v_brake))
        return np.sign(err) * speed

    def target(self, env) -> np.ndarray:
        gate = env.gates.current
        if gate is None:
            return np.array([0.0, env.layout.exit_y + 2.0, 1.5])
        approach, centre = gate_targets(gate)
        return approach if env.pos[0][1] < approach[1] else centre

    def __call__(self, env) -> np.ndarray:
        pos = env.pos[0]
        vel = env.vel[0]
        yaw = env.rpy[0, 2]

        tgt = self.target(env)
        v_des = self._velocity_target(tgt - pos)

        cz, sz = math.cos(-yaw), math.sin(-yaw)
        v_des_body = np.array([cz * v_des[0] - sz * v_des[1],
                               sz * v_des[0] + cz * v_des[1],
                               v_des[2]])

        # Drive the env's velocity integrator directly rather than through a
        # second P loop. The env holds v_cmd and adds `action * A_MAX * dt` to
        # it each policy step; asking for exactly the acceleration that lands
        # v_cmd on v_des removes an entire lag from the chain. Guessing at it
        # with P-on-velocity-error instead is what produced the lateral
        # overshoot to x = -1.02 against an opening edge at -0.44.
        a_body = (v_des_body - env.v_cmd_body) / C.POLICY_DT

        yaw_err = math.atan2(math.sin(math.pi / 2 - yaw),
                             math.cos(math.pi / 2 - yaw))
        action = np.empty(4, dtype=np.float32)
        action[:3] = np.clip(a_body / np.array(C.A_MAX), -1.0, 1.0)
        action[3] = np.clip(self.kp_yaw * yaw_err / C.YAW_RATE_MAX, -1.0, 1.0)
        return action


FRAME_KEYS = ("image", "image_gt", "seg", "depth_m", "proprio", "action",
              "gate_flags")


def new_frames() -> dict:
    """Empty per-episode buffer. Shared with mavrl.teleop so the two collectors
    cannot drift into writing different columns."""
    return {k: [] for k in FRAME_KEYS}


def record_frame(frames: dict, obs: dict, action, env) -> None:
    frames["image"].append(obs["image"])
    frames["image_gt"].append(obs["image_gt"])
    frames["seg"].append(obs["seg"])
    frames["depth_m"].append(obs["depth_m"].astype(np.float16))
    frames["proprio"].append(obs["proprio"])
    frames["action"].append(np.asarray(action, dtype=np.float32))
    frames["gate_flags"].append(np.uint8(env.gates_cleared))


def stack_frames(frames: dict) -> dict:
    return {k: np.asarray(v) for k, v in frames.items()}


def run_episode(env, pilot, rng, noise_std: float, max_steps: int,
                on_step=None):
    frames = new_frames()
    obs, info = env.reset()
    for _ in range(max_steps):
        clean = pilot(env)
        record_frame(frames, obs, clean, env)

        # DAgger-style: execute a perturbed action, record the clean label, so
        # the dataset covers states a slightly-wrong policy would reach.
        noisy = np.clip(clean + rng.normal(0.0, noise_std, 4), -1.0, 1.0)
        obs, _, term, trunc, info = env.step(noisy.astype(np.float32))
        if on_step is not None and on_step(env, obs, info) is False:
            break
        if term or trunc:
            break
    return stack_frames(frames), info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.06,
                   help="DAgger action noise std")
    p.add_argument("--sensor-noise", type=float, default=1.0,
                   help="sensor noise scale (0 disables)")
    p.add_argument("--episodes-per-shard", type=int, default=50)
    p.add_argument("--split", default="all",
                   choices=("train", "eval", "all"),
                   help="bar-height split; 'all' lets the encoder see held-out "
                        "heights the policy never trains on")
    p.add_argument("--max-stations", type=int, default=max(STAGE_STATIONS))
    p.add_argument("--gui", action="store_true",
                   help="watch it fly: 3-D viewer + live RGB/depth/seg feed. "
                        "Slower, and it needs a display -- use MUJOCO_GL=glfw.")
    p.add_argument("--prefix", default="shard",
                   help="shard filename prefix; keep distinct per source so "
                        "merged directories stay traceable")
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    noise = (NoiseConfig().scaled(args.sensor_noise)
             if args.sensor_noise > 0 else NoiseConfig.disabled())

    layout = sample_layout(rng, args.max_stations, args.split)
    env = CourseAviary(layout=layout, seed=args.seed, noise=noise,
                       collect_mode=True)
    pilot = CascadedPilot()
    writer = ShardWriter(args.out, args.episodes_per_shard, prefix=args.prefix)

    viewer = feed = on_step = None
    if args.gui:
        from mavrl.gui import FeedWindow, ViewerWindow, hud_lines
        viewer, feed = ViewerWindow(env), FeedWindow("mavrl -- scripted pilot")

        def on_step(env, obs, info):
            feed.poll()
            feed.draw(obs, hud_lines(env, "scripted pilot (read-only)"))
            feed.tick()
            viewer.sync(env)
            return feed.alive and viewer.running

    kept = completed = 0
    for ep in range(args.episodes):
        n_stations = int(rng.integers(0, args.max_stations + 1))
        layout = sample_layout(rng, n_stations, args.split)
        env.set_layout(layout)
        if viewer is not None:
            viewer.attach(env)          # set_layout built a new MjModel

        max_steps = int(C.t_max(n_stations) * C.POLICY_FREQ) + 5
        frames, info = run_episode(env, pilot, rng, args.noise, max_steps,
                                   on_step=on_step)
        if len(frames["image"]) == 0:
            continue

        writer.add_episode(frames, layout.describe(), source="scripted")
        kept += 1
        completed += int(info.get("is_success", False))
        if (ep + 1) % 50 == 0 or args.gui:
            print(f"[{ep + 1}/{args.episodes}] kept={kept} "
                  f"completed={completed} ({completed / max(1, kept):.0%}) "
                  f"last={layout.describe()}", flush=True)
        if args.gui and not (feed.alive and viewer.running):
            print("window closed -- stopping and saving")
            break

    writer.close()
    if viewer is not None:
        viewer.close()
        feed.close()
    env.close()

    report = summarize(args.out)
    print("\ndataset:", report)
    if report["problems"]:
        print("PROBLEMS:", *report["problems"], sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
