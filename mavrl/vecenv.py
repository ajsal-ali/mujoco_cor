#!/usr/bin/env python3
"""Vectorized env construction.

Separate module so `ald` and `train_course` do not import each other, and so the
stable-baselines3 import is **deferred into the function**. Data collection and
the offline training stages (SeVAE, memory, BC) need neither SB3 nor a vec env;
making it a module-level import would force it on them anyway.
"""

from __future__ import annotations

from typing import Optional

from mavrl.course_world import CourseLayout
from mavrl.sensor_noise import NoiseConfig


def make_env(layout: Optional[CourseLayout], seed: int,
             noise: Optional[NoiseConfig]):
    def _init():
        from mavrl.course_aviary import CourseAviary
        return CourseAviary(layout=layout, seed=seed, noise=noise)
    return _init


def build_venv(layout: Optional[CourseLayout], n_envs: int, seed: int,
               noise: Optional[NoiseConfig] = None, subproc: bool = True):
    """SubprocVecEnv for n_envs > 1, DummyVecEnv otherwise."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [make_env(layout, seed + i, noise) for i in range(n_envs)]
    return SubprocVecEnv(fns) if (subproc and n_envs > 1) else DummyVecEnv(fns)
