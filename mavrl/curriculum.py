#!/usr/bin/env python3
"""Curriculum over course complexity, with synchronized layout switching.

Two triggers, as specified:
  * a stage advances once rolling success clears a threshold -- progression is
    earned, not timed;
  * within a stage only the **bar heights** resample, every K rollouts, so the
    policy sees variety without the difficulty running ahead of it.

Station count and colour order change only when the stage does. Resampling them
every K rollouts as well would move three things at once, and a course that is
suddenly blue-first is a different task, not a variation on the same one.

**All envs share one layout at a time.** The sampler produces a layout and it is
broadcast to every worker at a rollout boundary. That is what keeps a
SharedStaticRenderer path viable later -- per-env randomization would make every
worker's model different and invalidate it outright.

A max-dwell fallback exists because a stalled stage is otherwise silent: the run
sits at stage 2 forever and the only symptom is a success curve that stopped
moving.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from mavrl.config import (
    MAX_DWELL_ROLLOUTS, RESAMPLE_EVERY_ROLLOUTS, SUCCESS_THRESHOLD,
    SUCCESS_WINDOW,
)
from mavrl.course_world import (
    STAGE_STATIONS, CourseLayout, resample_heights, sample_layout_for_stage,
)


@dataclass
class CurriculumState:
    stage: int = 0
    rollouts_in_stage: int = 0
    layout: Optional[CourseLayout] = None
    advanced_at: List[int] = field(default_factory=list)


class CourseSampler:
    def __init__(self, seed: Optional[int] = None, split: str = "train",
                 start_stage: int = 0,
                 success_window: int = SUCCESS_WINDOW,
                 success_threshold: float = SUCCESS_THRESHOLD,
                 resample_every: int = RESAMPLE_EVERY_ROLLOUTS,
                 max_dwell: int = MAX_DWELL_ROLLOUTS):
        self.rng = np.random.default_rng(seed)
        self.split = split
        self.success_threshold = success_threshold
        self.resample_every = resample_every
        self.max_dwell = max_dwell
        self.successes = deque(maxlen=success_window)
        self.state = CurriculumState(stage=start_stage)
        self.state.layout = self._sample()

    # -- bookkeeping ---------------------------------------------------------

    def _sample(self) -> CourseLayout:
        return sample_layout_for_stage(self.rng, self.state.stage, self.split)

    @property
    def layout(self) -> CourseLayout:
        return self.state.layout

    @property
    def stage(self) -> int:
        return self.state.stage

    @property
    def max_stage(self) -> int:
        return len(STAGE_STATIONS) - 1

    @property
    def success_rate(self) -> float:
        if not self.successes:
            return 0.0
        return float(np.mean(self.successes))

    def record_episodes(self, infos) -> None:
        for info in infos:
            if info is None:
                continue
            self.successes.append(bool(info.get("is_success", False)))

    # -- the two triggers ----------------------------------------------------

    def on_rollout_end(self) -> Optional[CourseLayout]:
        """Returns a new layout if one should be broadcast, else None."""
        self.state.rollouts_in_stage += 1

        ready = (len(self.successes) >= self.successes.maxlen
                 and self.success_rate >= self.success_threshold)
        stalled = self.state.rollouts_in_stage >= self.max_dwell

        if (ready or stalled) and self.state.stage < self.max_stage:
            self.state.stage += 1
            self.state.rollouts_in_stage = 0
            self.state.advanced_at.append(self.state.stage)
            self.successes.clear()
            self.state.layout = self._sample()
            return self.state.layout

        if self.state.rollouts_in_stage % self.resample_every == 0:
            if not self.state.layout.stations:
                # Stage 0 has no bars, so there is no height to redraw. Sending
                # the identical layout anyway would rebuild every worker's
                # MjModel -- and its GL context -- for nothing.
                return None
            if self.state.stage >= self.max_stage:
                # Terminal stage: resample the colour order too.
                #
                # Heights-only is right while the curriculum is still moving --
                # a stage change is what varies colour order, and varying both
                # at once makes an improvement unattributable. But at the top
                # stage no further advance ever comes, so heights-only freezes
                # the colour sequence for the entire run and the policy can
                # memorise "up, down, up" instead of learning that red means
                # above. Anyone training with --stage 3 sits here from step 0.
                self.state.layout = self._sample()
            else:
                self.state.layout = resample_heights(
                    self.rng, self.state.layout, self.split)
            return self.state.layout

        return None

    # -- reporting -----------------------------------------------------------

    def describe(self) -> str:
        return (f"stage {self.state.stage}/{self.max_stage} "
                f"({STAGE_STATIONS[self.state.stage]} bars) "
                f"success={self.success_rate:.2f} "
                f"n={len(self.successes)} "
                f"layout={self.state.layout.describe()}")


def broadcast_layout(venv, layout: CourseLayout) -> None:
    """Give every worker the same layout.

    Called at a rollout boundary, never mid-rollout: swapping geometry inside a
    rollout would invalidate the value estimates already collected against the
    old course.
    """
    venv.env_method("set_layout", layout)
