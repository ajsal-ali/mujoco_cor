#!/usr/bin/env python3
"""Watch the examples/pid.py demos live in the MuJoCo viewer.

pid.py itself runs headless (it only prints numbers). This runs the exact same
PID demos but opens the standard MuJoCo viewer and paces to real time so you can
actually see the drone fly.

Run inside the rl_mujoco conda env:
    conda run -n rl_mujoco python watch_pid.py            # all three, in order
    conda run -n rl_mujoco python watch_pid.py --demo hover
    conda run -n rl_mujoco python watch_pid.py --demo square
    conda run -n rl_mujoco python watch_pid.py --demo multi

Close the viewer window (or ESC) to advance to the next demo / quit.
"""

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.control.pid_control import PIDControl
from multi_drone_mujoco.control.dsl_pid_control import DSLPIDControl
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType


def _track_first_drone(viewer, model, distance=1.5):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone0")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = bid
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = distance, 135, -20


def _run(env, controllers, targets_fn, steps, label, cam_dist=1.5):
    """Drive `env` with per-drone PID controllers and show it in the viewer.

    targets_fn(step, env) -> list of target positions, one per drone.
    """
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    n = env.NUM_DRONES
    dt = env.CTRL_TIMESTEP
    env.reset()
    for c in controllers:
        c.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        _track_first_drone(viewer, env.model, cam_dist)
        for step in range(steps):
            if not viewer.is_running():
                break
            tic = time.perf_counter()

            targets = targets_fn(step, env)
            rpms = np.zeros((n, 4))
            for i in range(n):
                rpms[i], _, _ = controllers[i].computeControl(
                    control_timestep=dt,
                    cur_pos=env.pos[i], cur_quat=env.quat[i],
                    cur_vel=env.vel[i], cur_ang_vel=env.ang_v[i],
                    target_pos=targets[i],
                )

            _, _, terminated, _, _ = env.step(rpms.flatten())
            viewer.sync()

            if step % 48 == 0:
                errs = [np.linalg.norm(env.pos[i] - targets[i]) for i in range(n)]
                print(f"  t={step*dt:5.1f}s  err(m)=" +
                      " ".join(f"{e:.3f}" for e in errs))
            if terminated:
                print("  TERMINATED (crash)")
                break

            slack = dt - (time.perf_counter() - tic)
            if slack > 0:
                time.sleep(slack)
    env.close()


# --- the three pid.py demos -------------------------------------------------

def demo_hover():
    """One drone holds z=1.0 (pid.py: pid_hover)."""
    env = BaseAviary(drone_model=DroneModel.CF2X, num_drones=1,
                     physics=Physics.MJC, sim_freq=240, ctrl_freq=48,
                     act_type=ActionType.RPM)
    target = np.array([0.0, 0.0, 1.0])
    _run(env, [PIDControl(env)],
         targets_fn=lambda s, e: [target],
         steps=480, label="PID Hover  (hold z = 1.0 m)")


def demo_square():
    """One drone flies a 1 m square via DSL PID waypoints (pid.py: pid_velocity)."""
    env = BaseAviary(drone_model=DroneModel.CF2X, num_drones=1,
                     physics=Physics.MJC, sim_freq=240, ctrl_freq=48,
                     act_type=ActionType.RPM,
                     initial_xyzs=np.array([[0, 0, 0.5]]))
    waypoints = [np.array([1.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]),
                 np.array([0.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0])]
    state = {"idx": 0}

    def targets_fn(step, e):
        wp = waypoints[state["idx"]]
        if np.linalg.norm(e.pos[0] - wp) < 0.1 and state["idx"] < len(waypoints) - 1:
            state["idx"] += 1
            print(f"  reached waypoint {state['idx']}")
        return [waypoints[state["idx"]]]

    _run(env, [DSLPIDControl(env)], targets_fn, steps=960,
         label="PID Square  (DSL PID, 4 waypoints)")


def demo_multi():
    """Three drones hover at different heights, full aero (pid.py: multi_drone_pid).

    NOTE: pid.py stacks the drones at x=0,0.3,0.6 — nearly on top of each other —
    which makes the package's downwash model blow up (NaN) so they never take off.
    Here they're spread to x=-1,0,1 so downwash stays stable and you actually see
    three drones holding three heights.
    """
    n = 3
    targets = [np.array([-1.0, 0.0, 0.8]), np.array([0.0, 0.0, 1.0]),
               np.array([1.0, 0.0, 1.2])]
    env = BaseAviary(drone_model=DroneModel.CF2X, num_drones=n,
                     physics=Physics.MJC_GND_DRAG_DW, sim_freq=240, ctrl_freq=48,
                     act_type=ActionType.RPM,
                     initial_xyzs=np.array([[-1, 0, 0.1], [0, 0, 0.1], [1, 0, 0.1]]))
    _run(env, [PIDControl(env) for _ in range(n)],
         targets_fn=lambda s, e: targets, steps=480,
         label="Multi-Drone PID  (3 drones, full aero)", cam_dist=3.5)


DEMOS = {"hover": demo_hover, "square": demo_square, "multi": demo_multi}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--demo", choices=list(DEMOS) + ["all"], default="all")
    args = ap.parse_args()

    if args.demo == "all":
        for fn in (demo_hover, demo_square, demo_multi):
            fn()
    else:
        DEMOS[args.demo]()


if __name__ == "__main__":
    main()
