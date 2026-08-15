#!/usr/bin/env python3
"""Watch a trained policy fly the full course, live.

    MUJOCO_GL=glfw python -m mavrl.fly

Defaults to `runs/course/final.pt` -- the weights `train_course.py` writes when
it finishes -- so with no arguments this just shows you the finished policy on
the full 3-bar course. Point `--model` at any other checkpoint to compare:

    python -m mavrl.fly --model runs/course/ckpt_2000000.pt
    python -m mavrl.fly --model ckpt/bc_init.pt --stations 1

Two windows open, the same pair teleop uses: a third-person MuJoCo viewer, and a
feed window showing the RGB / depth / segmentation the encoder actually receives
at 128 px. **Keys only register while the feed window is focused.**

    SPACE       pause / resume
    N           skip to the next episode
    R           replay this same layout from the start
    Y           toggle yaw lock (see below)
    ESC         quit

A note on yaw, because it surprises people
------------------------------------------
The action space is body acceleration (3) **plus a yaw rate** (`a[3]`), and
`yaw_sp` integrates that rate with no restoring term and no penalty anywhere in
the reward. Every source of training data pins yaw with a P controller onto
`YAW_DOWN_COURSE` (collect.py:177, teleop.py:107), so a policy that puts any
constant bias on `a[3]` will simply spin up heading error all episode -- it was
never shown a reason not to. The HUD reports heading error in degrees; `Y`
substitutes the collector's yaw controller so you can see whether that drift is
causing the failures or just riding along with them.

This is the sibling of two files that answer different questions. `evaluate.py`
gives you the success rate over many episodes; `preview.py` writes mp4s on a
headless box. This one is for sitting and watching, which is the only one of the
three that catches "it clears the gate but wobbles the whole way there".

Needs a display -- MUJOCO_GL defaults to glfw here, not egl.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "glfw")     # a live window needs a display
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mavrl import config as C                                    # noqa: E402
from mavrl.course_aviary import CourseAviary                     # noqa: E402
from mavrl.course_world import (                                 # noqa: E402
    STAGE_STATIONS, YAW_DOWN_COURSE, sample_layout,
)
from mavrl.policy import (                                       # noqa: E402
    MavrlActorCritic, load_policy_state, obs_to_tensors,
)
from mavrl.sensor_noise import NoiseConfig                       # noqa: E402


def load_policy(path: Path, device, memory_type=None, mem_tokens=None):
    """Build the net the checkpoint was trained as, and load it.

    `ppo.save` records `memory_type`, so the usual case needs no flags at all --
    which matters because a silent lstm/attention mismatch loads a subset of the
    weights and flies badly rather than erroring.
    """
    ckpt = torch.load(path, map_location=device)
    saved = ckpt.get("memory_type")
    wanted = memory_type or saved or "lstm"
    if memory_type and saved and memory_type != saved:
        print(f"warning: checkpoint says memory_type={saved!r}, "
              f"you asked for {memory_type!r}")

    policy = MavrlActorCritic(memory_type="lstm" if wanted == "none" else wanted,
                              mem_tokens=mem_tokens).to(device)
    load_policy_state(policy, ckpt.get("policy", ckpt.get("actor")),
                      ckpt.get("aux_segments"))
    policy.eval()
    if wanted == "none":
        policy.memory_type = "none"

    steps = ckpt.get("num_timesteps")
    print(f"loaded {path}  (memory={wanted}, aux={ckpt.get('aux_segments')}"
          + (f", trained {steps:,} steps)" if steps else ")"))
    return policy


def yaw_lock_action(env, action, kp_yaw: float = 2.0) -> np.ndarray:
    """Overwrite the policy's yaw rate with the collector's P controller.

    `a[3]` is a yaw *rate* feeding `yaw_sp`, which is a plain integrator with no
    restoring term (course_aviary.py:189) and no penalty anywhere in the reward.
    So any constant bias the policy puts on that dim accumulates all episode.
    Every source of training data locks yaw this way (collect.py:177,
    teleop.py:107), so the policy has never been shown a reason to hold heading.

    Diagnostic only -- it tells you whether the drift is *causing* the failures
    or just riding along with them. It does not fix the policy.
    """
    yaw = env.rpy[0, 2]
    err = math.atan2(math.sin(YAW_DOWN_COURSE - yaw),
                     math.cos(YAW_DOWN_COURSE - yaw))
    out = action.copy()
    out[3] = np.clip(kp_yaw * err / C.YAW_RATE_MAX, -1.0, 1.0)
    return out


def _hud(env, action, paused: bool, ep: int, n_ep: int, layout,
         lock_yaw: bool = False) -> list:
    from mavrl.gui import hud_lines
    head = (f"{'PAUSED  ' if paused else ''}episode {ep + 1}/{n_ep}   "
            f"[SPACE pause] [N next] [R replay] [Y yaw-lock] [ESC quit]")
    lines = hud_lines(env, head)
    a = action
    # Heading error is the number to watch: yaw_sp integrates a[3] with nothing
    # pulling it back, so this grows monotonically under a biased policy.
    err = math.degrees(math.atan2(math.sin(YAW_DOWN_COURSE - env.rpy[0, 2]),
                                  math.cos(YAW_DOWN_COURSE - env.rpy[0, 2])))
    lines.append(f"action a=({a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f}) "
                 f"yaw={a[3]:+.2f}   heading err {err:+.0f} deg"
                 + ("  [LOCKED]" if lock_yaw else ""))
    lines.append(layout.describe())
    return lines


@torch.no_grad()
def fly_episode(env, policy, device, feed, viewer, max_steps: int,
                ep: int, n_ep: int, layout, deterministic: bool = True,
                lock_yaw: bool = False):
    """One episode under the policy. Returns (info, verdict, lock_yaw).

    verdict: "done" | "next" | "replay" | "quit"
    `lock_yaw` is returned because Y toggles it live and the setting should
    survive into the next episode.
    """
    pg = feed.pygame
    obs, info = env.reset()
    state = policy.initial_state(1, device)
    action = np.zeros(C.N_ACTIONS, dtype=np.float32)
    paused = False

    for _ in range(max_steps):
        held, tapped = feed.poll()
        if not feed.alive or not viewer.running or pg.K_ESCAPE in tapped:
            return info, "quit", lock_yaw
        if pg.K_n in tapped:
            return info, "next", lock_yaw
        if pg.K_r in tapped:
            return info, "replay", lock_yaw
        if pg.K_SPACE in tapped:
            paused = not paused
        if pg.K_y in tapped:
            lock_yaw = not lock_yaw

        if paused:
            feed.draw(obs, _hud(env, action, paused, ep, n_ep, layout,
                                lock_yaw))
            feed.tick()
            viewer.sync(env)
            continue

        batched = {k: np.asarray(v)[None] for k, v in obs.items()
                   if k in ("image", "proprio")}
        image, proprio = obs_to_tensors(batched, device)
        act, _, _, state = policy.act(image, proprio, state,
                                      deterministic=deterministic)
        action = act.cpu().numpy()[0].astype(np.float32)
        if lock_yaw:
            action = yaw_lock_action(env, action)

        obs, _, term, trunc, info = env.step(action)

        feed.draw(obs, _hud(env, action, paused, ep, n_ep, layout, lock_yaw))
        feed.tick()                       # hold to POLICY_FREQ -> real time
        viewer.sync(env)
        if term or trunc:
            break

    return info, "done", lock_yaw


def hold(feed, viewer, seconds: float = 1.2) -> bool:
    """Keep the last frame and its result banner up long enough to read.

    Without this the next episode resets on the very next frame and the outcome
    is gone before you can see which gate it lost. Events are still pumped, so
    the window stays responsive and ESC still quits.
    """
    for _ in range(max(1, int(seconds * C.POLICY_FREQ))):
        _, tapped = feed.poll()
        if not feed.alive or not viewer.running \
                or feed.pygame.K_ESCAPE in tapped:
            return False
        feed.tick()
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=Path("runs/course/final.pt"),
                   help="checkpoint to fly (default: the finished PPO weights)")
    p.add_argument("--memory-type", default=None,
                   choices=("lstm", "attention", "none"),
                   help="override; normally read from the checkpoint")
    p.add_argument("--mem-tokens", type=int, default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--stations", type=int, default=max(STAGE_STATIONS),
                   help="bars in the course; default is the full course")
    p.add_argument("--split", default="train", choices=("train", "eval", "all"),
                   help="'eval' uses the held-out bar heights")
    p.add_argument("--sensor-noise", type=float, default=1.0)
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking the mean")
    p.add_argument("--lock-yaw", action="store_true",
                   help="override the policy's yaw rate with the collector's P "
                        "controller (Y toggles live). Diagnostic: shows whether "
                        "heading drift is causing the failures or riding along")
    p.add_argument("--no-seg", action="store_true",
                   help="hide the segmentation panel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    if not args.model.exists():
        print(f"no checkpoint at {args.model}\n"
              f"train one first (python -m mavrl.train_course) or pass --model",
              file=sys.stderr)
        return 1

    from mavrl.gui import FeedWindow, ViewerWindow

    device = torch.device(args.device)
    policy = load_policy(args.model, device, args.memory_type, args.mem_tokens)

    noise = (NoiseConfig().scaled(args.sensor_noise)
             if args.sensor_noise > 0 else NoiseConfig.disabled())
    rng = np.random.default_rng(args.seed)

    # collect_mode gives the panels clean seg and metric depth; the encoder is
    # still fed the noisy `image`, so watching does not flatter the policy.
    layout = sample_layout(rng, args.stations, args.split)
    env = CourseAviary(layout=layout, seed=args.seed, noise=noise,
                       collect_mode=True)
    feed = FeedWindow("mavrl -- trained policy", show_seg=not args.no_seg)
    viewer = ViewerWindow(env)

    max_steps = int(C.t_max(args.stations) * C.POLICY_FREQ) + 5
    outcomes = []
    lock_yaw = args.lock_yaw
    ep = 0
    try:
        while ep < args.episodes:
            env.set_layout(layout)
            viewer.attach(env)               # set_layout rebuilt the model
            if not viewer.running or not feed.alive:
                break

            info, verdict, lock_yaw = fly_episode(
                env, policy, device, feed, viewer, max_steps, ep,
                args.episodes, layout, deterministic=not args.stochastic,
                lock_yaw=lock_yaw)

            if verdict == "quit":
                print("quit requested")
                break
            if verdict == "replay":
                continue                     # same layout, fresh reset

            if verdict == "done":
                ok = bool(info.get("is_success", False))
                reason = info.get("terminal_reason") or "timeout"
                outcomes.append(ok)
                print(f"ep{ep:02d}  {'PASS' if ok else 'FAIL':4s}  "
                      f"{reason:16s} gates {info['gates_cleared']}/"
                      f"{info['n_gates']}  agv {info['agv']:.2f} m/s")
                feed.message(f"{'SUCCESS' if ok else reason.upper()} -- "
                             f"gates {info['gates_cleared']}/{info['n_gates']}",
                             (60, 200, 90) if ok else (230, 90, 90))
                if not hold(feed, viewer):
                    break

            ep += 1
            layout = sample_layout(rng, args.stations, args.split)
    finally:
        viewer.close()
        feed.close()
        env.close()

    if outcomes:
        print(f"\n{sum(outcomes)}/{len(outcomes)} succeeded on "
              f"{args.stations} stations ({args.split} heights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
