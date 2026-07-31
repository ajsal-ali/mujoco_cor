#!/usr/bin/env python3
"""Shared interactive-viewer loop for the RL tools (view_env / teleop_record / eval).

Opens the standard interactive mujoco.viewer (3rd-person, mouse orbit) tracking the
drone, spins the cosmetic props, drives the env from a `get_action` callback, and
auto-resets on episode end. Optionally shows the onboard RGB + depth in a separate
process (the GLFW viewer and matplotlib deadlock in one process). Ephemeral: close
the window and it's gone.
"""

import math
import queue
import time
import multiprocessing as mp

import numpy as np
import mujoco
import mujoco.viewer

from imav_play import _onboard_window

_SPIN_DIRS = [1.0, -1.0, 1.0, -1.0]
_VIS_SPIN = 55.0


class KeyTeleop:
    """Hold-to-fly teleop that outputs a NORMALIZED action directly in [-1,1].

    Unlike converting an m/s velocity command (which saturates once divided by a
    small V_* scale), this returns the action the env expects, so V_XY/V_Z/V_YAW
    in WindowAviary set the actual speed while the control feel stays proportional.

    On release: translation axes decay smoothly to rest; the YAW axis stops
    instantly (so you can aim a heading precisely instead of over-rotating).
    """
    LEVEL = 1.0            # normalized command while a key is held
    HOLD_TIMEOUT = 0.55    # s without a key event -> treat as released
    DECAY_TAU = 0.15       # s, translation brake after release

    W, A, S, D, H, R = 87, 65, 83, 68, 72, 82
    UP, DOWN, LEFT, RIGHT = 265, 264, 263, 262

    def __init__(self):
        self.cmd = np.zeros(4)
        self.t_axis = np.full(4, -1e9)
        self.reset_requested = False
        self._map = {
            self.W: (0, +self.LEVEL), self.S: (0, -self.LEVEL),
            self.A: (1, +self.LEVEL), self.D: (1, -self.LEVEL),
            self.UP: (2, +self.LEVEL), self.DOWN: (2, -self.LEVEL),
            self.LEFT: (3, +self.LEVEL), self.RIGHT: (3, -self.LEVEL),
        }

    def on_key(self, k):
        if k == self.H:
            self.cmd[:] = 0.0; self.t_axis[:] = -1e9
        elif k == self.R:
            self.reset_requested = True
            self.cmd[:] = 0.0; self.t_axis[:] = -1e9
        elif k in self._map:
            ax, lvl = self._map[k]
            self.cmd[ax] = lvl
            self.t_axis[ax] = time.perf_counter()

    def action(self, dt):
        now = time.perf_counter()
        decay = math.exp(-dt / self.DECAY_TAU)
        for i in range(4):
            if now - self.t_axis[i] > self.HOLD_TIMEOUT:
                if i == 3:
                    self.cmd[i] = 0.0                    # yaw: instant stop
                else:
                    self.cmd[i] *= decay
                    if abs(self.cmd[i]) < 0.02:
                        self.cmd[i] = 0.0
        return self.cmd.astype(np.float32).copy()


def _push_onboard(env, obs, frame_q):
    """Split the env's RGB-D obs image and push (rgb, depth_m) to the window."""
    img = obs["image"]
    rgb = img[..., :3]
    depth_m = img[..., 3].astype(np.float32) / 255.0 * env.DEPTH_MAX
    try:
        frame_q.put_nowait((rgb.copy(), depth_m))
    except queue.Full:
        pass


def _prop_joints(model):
    qpos, dofs = [], []
    for i in range(4):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"drone0_prop{i}_spin")
        if jid < 0:
            return None, None
        qpos.append(model.jnt_qposadr[jid])
        dofs.append(model.jnt_dofadr[jid])
    return qpos, dofs


def run_viewer(env, get_action, post_step=None, on_key=None, cam_dist=2.5,
               realtime=True, show_onboard=True, onboard_every=3):
    """Drive `env` in the interactive viewer.

    get_action(env, obs) -> action (normalized, passed to env.step).
    post_step(env, obs, reward, terminated, truncated, info) -> None (optional).
    on_key(keycode) -> None (optional viewer key callback).
    show_onboard: also open a separate-process window with onboard RGB + depth.
    """
    model, data = env.model, env.data
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone0")
    pq, pd = _prop_joints(model)
    ang = [0.0, 0.0, 0.0, 0.0]
    dt = env.CTRL_TIMESTEP

    # onboard RGB + depth window in its own process (avoids GLFW/Tk deadlock)
    frame_q = img_proc = None
    if show_onboard:
        frame_q = mp.Queue(maxsize=2)
        img_proc = mp.Process(target=_onboard_window, args=(frame_q,), daemon=True)
        img_proc.start()

    obs, info = env.reset()
    step = 0
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = bid
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam_dist, 90, -20

        while viewer.is_running():
            tic = time.perf_counter()

            # cosmetic prop spin (set before env.step so its FK reflects it)
            if pq is not None:
                rpm = getattr(env, "last_rpm", None)
                for i in range(4):
                    rate = _VIS_SPIN * (rpm[i] / env.HOVER_RPM if rpm is not None else 1.0)
                    ang[i] += _SPIN_DIRS[i] * rate * dt
                    data.qpos[pq[i]] = ang[i]
                    data.qvel[pd[i]] = 0.0

            action = get_action(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()

            if frame_q is not None and step % onboard_every == 0:
                _push_onboard(env, obs, frame_q)

            if post_step is not None:
                post_step(env, obs, reward, terminated, truncated, info)

            if terminated or truncated:
                obs, info = env.reset()
                ang = [0.0, 0.0, 0.0, 0.0]
            step += 1

            if realtime:
                slack = dt - (time.perf_counter() - tic)
                if slack > 0:
                    time.sleep(slack)

    if img_proc is not None:
        try:
            frame_q.put_nowait(None)
        except queue.Full:
            pass
        img_proc.join(timeout=1.0)
        if img_proc.is_alive():
            img_proc.terminate()
