#!/usr/bin/env python3
"""WindowAviary — vision-based RL env for traversing the red window.

Built entirely on the repo's BaseAviary (real CF2 drone + PIDControl velocity
engine + onboard-camera renderer). The arena is injected at runtime via
rl.window_world.make_window_world_injector — generated geometry, no SDF and no
external asset files, so this runs from a clean checkout.

Design (see plan):
  * Observation = Dict{ image: onboard RGB-D 84x84x4 (uint8),
                        proprio: [grav_dir_body(3), angvel_body(3), vel_body(3)] }
    -> NO privileged window pose in the observation (vision only + IMU-realizable).
  * Action = continuous velocity [vx, vy, vz, yaw_rate] in [-1,1], scaled to m/s
    and driven through PIDControl as a pure velocity set-point.
  * Reward / termination MAY use the known window geometry (privileged, sim-only):
    waypoint-progress + traversal check + collision + time. See WindowGate.

The WindowGate helper (geometry + traversal check) is imported by teleop_record.py
so the checker you validate by hand is exactly the one that trains the policy.
"""

import os
import sys
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# make the parent dir (with the package) importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.window_world import make_window_world_injector          # noqa: E402
from multi_drone_mujoco.envs import base_aviary as _BA          # noqa: E402
from multi_drone_mujoco.envs.base_aviary import BaseAviary      # noqa: E402
from multi_drone_mujoco.control.pid_control import PIDControl   # noqa: E402
from multi_drone_mujoco.utils.enums import (                    # noqa: E402
    DroneModel, Physics, ActionType, ObservationType)


# ---------------------------------------------------------------------------
# Window geometry + traversal check (privileged; shared with teleop_record)
# ---------------------------------------------------------------------------

class WindowGate:
    """The RED window (obstacle wall at y=3.30).

    These values are the ground truth for both the reward and the arena:
    rl.window_world builds the wall to match them exactly. Change them and the
    world changes with them -- they must never drift apart, or the reward would
    score traversals against geometry the drone cannot see.

    Flyable opening: x in [0.44, 1.32], z in [2.97, 3.85], plane y = 3.30.
    (The blue window is the mirror opening at x in [-1.32, 0.0].)
    """
    Y = 3.30
    X_LO, X_HI = 0.44, 1.32
    Z_LO, Z_HI = 2.97, 3.85
    CENTER = np.array([0.88, 3.30, 3.41])
    MARGIN = 0.06        # require the drone center this far inside the opening
    NORMAL = np.array([0.0, 1.0, 0.0])   # approach-side normal (drone flies +y)

    def crossed(self, prev_pos, cur_pos):
        """Detect a forward plane crossing between two consecutive positions.

        Returns (crossed, through): `crossed` if the +y plane was crossed this
        step; `through` if the interpolated crossing point is inside the opening.
        """
        s0 = prev_pos[1] - self.Y
        s1 = cur_pos[1] - self.Y
        if s0 < 0.0 <= s1 and s1 != s0:
            a = -s0 / (s1 - s0)
            xc = prev_pos[0] + a * (cur_pos[0] - prev_pos[0])
            zc = prev_pos[2] + a * (cur_pos[2] - prev_pos[2])
            through = (self.X_LO + self.MARGIN <= xc <= self.X_HI - self.MARGIN and
                       self.Z_LO + self.MARGIN <= zc <= self.Z_HI - self.MARGIN)
            return True, bool(through)
        return False, False


class SpawnConfig:
    """Slightly-randomized start pose that always faces the window (+y)."""
    BASE = np.array([0.88, 0.4, 3.05])      # in front of the RED window
    JIT = np.array([0.45, 0.4, 0.4])        # +/- x, y, z
    YAW = math.pi / 2                        # face +y
    YAW_JIT = math.radians(15)

    def sample(self, rng):
        pos = self.BASE + rng.uniform(-self.JIT, self.JIT)
        yaw = self.YAW + rng.uniform(-self.YAW_JIT, self.YAW_JIT)
        return pos, yaw


# ---------------------------------------------------------------------------

class WindowAviary(BaseAviary):
    """Single-window traversal task, vision + proprioception, velocity action."""

    IMG_W, IMG_H = 96, 96
    DEPTH_MAX = 8.0                          # m, depth clip for the image channel
    EPISODE_LEN_SEC = 20.0                   # time limit (generous — time to line up)
    T_NOMINAL = 8.0                          # reference time for the speed bonus

    # MAX velocity per axis (m/s, rad/s). The action is in [-1,1] and the commanded
    # velocity = action * V_max, i.e. the action is a proportion of the max speed.
    # Same scaling is used for both teleop and the trained policy (env.step).
    V_XY, V_Z, V_YAW = 1, .5, .3

    # waypoint just past the window center (dense progress target)
    WP = np.array([0.88, 3.8, 3.41])

    # reward weights
    K_PROG, K_TIME, K_SMOOTH = 1.0, 0.03, 0.01
    R_PASS, R_CRASH, K_TBONUS = 100.0, 25.0, 5.0
    # heading alignment with the window normal (+y): sin(yaw)=1 when facing through.
    K_ALIGN = 15.0        # bonus at the crossing for facing the normal
    K_ALIGN_STEP = 0.05   # gentle per-step nudge to face the normal near the plane
    # centering: penalize lateral (x,z) offset from the window center, so the drone
    # aims for the middle of the opening. Grows as it moves off-center.
    K_CENTER = 0.1        # per-step, applied near the plane
    K_CENTER_CROSS = 10.0 # bonus at the crossing for a well-centered pass

    def __init__(self, seed=None, posts=True):
        self.gate = WindowGate()
        self.spawn = SpawnConfig()
        self._rng = np.random.default_rng(seed)
        # task state (set properly in reset)
        self._success = False
        self._crashed = False
        self._terminal_reason = None
        self.cur_action = np.zeros(4)
        self.prev_action = np.zeros(4)

        original, patched = make_window_world_injector(posts=posts)
        _BA._generate_aviary_xml = patched
        try:
            super().__init__(
                drone_model=DroneModel.CF2X, num_drones=1,
                initial_xyzs=np.array([self.spawn.BASE], dtype=float),
                initial_rpys=np.array([[0.0, 0.0, self.spawn.YAW]]),
                physics=Physics.MJC, sim_freq=240, ctrl_freq=48,
                vision_attributes=True,
                obs_type=ObservationType.KIN, act_type=ActionType.RPM,
            )
        finally:
            _BA._generate_aviary_xml = original

        # onboard camera resolution (renderer built lazily from IMG_RES)
        self.IMG_RES = np.array([self.IMG_W, self.IMG_H])

        self.pid = PIDControl(self)
        self.drone_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone0")
        self.drone_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "drone0_collision")
        self.yaw_sp = float(self.spawn.YAW)

    # -- spaces -------------------------------------------------------------

    def _actionSpace(self):
        return spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        return spaces.Dict({
            "image": spaces.Box(0, 255, shape=(self.IMG_H, self.IMG_W, 4), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
        })

    # -- action: velocity -> PIDControl (pure velocity set-point) ------------

    def _preprocessAction(self, action):
        a = np.clip(np.asarray(action, dtype=float).flatten(), -1.0, 1.0)
        self.cur_action = a.copy()
        v_body = np.array([a[0] * self.V_XY, a[1] * self.V_XY, a[2] * self.V_Z])
        yaw_rate = a[3] * self.V_YAW

        yaw = self.rpy[0, 2]
        cz, sz = math.cos(yaw), math.sin(yaw)
        v_world = np.array([cz * v_body[0] - sz * v_body[1],
                            sz * v_body[0] + cz * v_body[1],
                            v_body[2]])
        self.yaw_sp += yaw_rate * self.CTRL_TIMESTEP

        rpm, _, _ = self.pid.computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=self.pos[0], cur_quat=self.quat[0],
            cur_vel=self.vel[0], cur_ang_vel=self.ang_v[0],
            target_pos=self.pos[0],
            target_rpy=np.array([0.0, 0.0, self.yaw_sp]),
            target_vel=v_world)
        rpm = np.clip(rpm, 0, self.MAX_RPM).flatten()
        self.last_rpm = rpm                              # for cosmetic prop spin
        return rpm.reshape(1, 4)

    # -- observation: onboard RGB-D + proprioception -------------------------

    def _computeObs(self):
        rgb, dep, _ = self._getDroneImages(0)                  # (H,W,3) uint8, (H,W) float
        depth_u8 = (np.clip(dep, 0.0, self.DEPTH_MAX) / self.DEPTH_MAX * 255.0).astype(np.uint8)
        image = np.concatenate([rgb, depth_u8[..., None]], axis=2)   # (H,W,4)

        R = self.data.xmat[self.drone_bid].reshape(3, 3)       # body->world
        grav_body = R.T @ np.array([0.0, 0.0, -1.0])           # accelerometer-like
        angv_body = R.T @ self.ang_v[0]                        # gyro
        vel_body = R.T @ self.vel[0]                           # VIO/flow
        proprio = np.concatenate([grav_body, angv_body, vel_body]).astype(np.float32)

        return {"image": image, "proprio": proprio}

    # -- reward / termination (privileged geometry allowed) ------------------

    def _check_collision(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.drone_gid or c.geom2 == self.drone_gid:
                return True
        return False

    def _computeReward(self):
        pos = self.pos[0]
        rpy = self.rpy[0]

        d = float(np.linalg.norm(pos - self.WP))
        reward = self.K_PROG * (self.d_prev - d)
        self.d_prev = d
        reward -= self.K_TIME
        reward -= self.K_SMOOTH * float(np.linalg.norm(self.cur_action - self.prev_action))

        # heading alignment with the window normal (+y). sin(yaw) = 1 when the
        # drone faces straight through; gently penalize misalignment near the plane.
        align = math.sin(rpy[2])
        # lateral (x,z) offset from the window center-line.
        off = math.hypot(pos[0] - self.gate.CENTER[0], pos[2] - self.gate.CENTER[2])
        if abs(pos[1] - self.gate.Y) < 1.5:
            reward += self.K_ALIGN_STEP * (align - 1.0)
            reward -= self.K_CENTER * off                    # closer to center = higher

        crash, reason = False, None
        crossed, through = self.gate.crossed(self.prev_pos, pos)
        if crossed and through:
            self._success = True
            self._terminal_reason = "passed"
            t = self.step_counter / self.SIM_FREQ
            reward += self.R_PASS + self.K_TBONUS * max(0.0, self.T_NOMINAL - t)
            reward += self.K_ALIGN * max(0.0, align)         # aligned-crossing bonus
            reward += self.K_CENTER_CROSS * max(0.0, 1.0 - off / 0.44)  # centered-pass bonus
        elif crossed and not through:
            crash, reason = True, "hit_wall"

        if not self._success and not crash:
            if self._check_collision():
                crash, reason = True, "collision"
            elif pos[2] < 0.05:
                crash, reason = True, "ground"
            elif abs(rpy[0]) > math.pi / 2 or abs(rpy[1]) > math.pi / 2:
                crash, reason = True, "flip"
            elif abs(pos[0]) > 7.0 or pos[1] < -13.0 or pos[1] > 15.0 or pos[2] > 6.0:
                crash, reason = True, "out_of_bounds"

        if crash:
            self._crashed = True
            self._terminal_reason = reason
            reward -= self.R_CRASH

        self.prev_pos = pos.copy()
        self.prev_action = self.cur_action.copy()
        return float(reward)

    def _computeTerminated(self):
        return bool(self._success or self._crashed)

    def _computeTruncated(self):
        return bool(self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC)

    def _computeInfo(self):
        return {
            "is_success": bool(self._success),
            "crashed": bool(self._crashed),
            "terminal_reason": self._terminal_reason,
            "dist_to_wp": float(np.linalg.norm(self.pos[0] - self.WP)),
        }

    # -- reset --------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        pos, yaw = self.spawn.sample(self._rng)
        self.INIT_XYZS = np.array([pos], dtype=float)
        self.INIT_RPYS = np.array([[0.0, 0.0, yaw]])

        self._success = False
        self._crashed = False
        self._terminal_reason = None
        self.cur_action = np.zeros(4)
        self.prev_action = np.zeros(4)

        obs, info = super().reset(seed=seed, options=options)

        self.yaw_sp = float(yaw)
        self.pid.reset()
        self.prev_pos = self.pos[0].copy()
        self.d_prev = float(np.linalg.norm(self.pos[0] - self.WP))
        return obs, info


def make_window_env(seed=None):
    """Factory for SubprocVecEnv / make_vec_env."""
    return WindowAviary(seed=seed)


if __name__ == "__main__":
    # quick self-test
    env = WindowAviary()
    obs, info = env.reset(seed=0)
    print("obs image", obs["image"].shape, obs["image"].dtype,
          "proprio", obs["proprio"].shape)
    print("action_space", env.action_space)
    total = 0.0
    for t in range(50):
        obs, r, term, trunc, info = env.step(np.array([0, 1, 0.2, 0], dtype=np.float32))
        total += r
        if term or trunc:
            print("episode end", info, "steps", t)
            break
    print("ran ok, cum_reward=%.2f pos=%s" % (total, np.round(env.pos[0], 2)))
    env.close()
