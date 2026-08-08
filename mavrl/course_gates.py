#!/usr/bin/env python3
"""Gate geometry and traversal checks for the obstacle course.

Pure geometry -- no mujoco, no torch, no gymnasium. That keeps it importable and
unit-testable from a clean checkout (and from the Windows box, which has no
mujoco), exactly like rl.window_world.

These values are the ground truth for both the reward and the arena:
rl.course_world builds geometry to match them. They must never drift apart, or
the reward would score traversals against geometry the drone cannot see.

Two gate kinds:

* `WallGate` -- a wall plane with a *target* opening and an optional *decoy*
  opening. Passing the decoy is a distinct outcome from missing both, because the
  policy has to learn the colour cue, not just "find a hole".
* `BarGate`  -- a horizontal bar spanning the corridor. Free space exists both
  above and below; the colour says which side is legal. The bar's centreline
  divides the two, so "above" and "below" are unambiguous and a grazing pass is
  resolved by the physical collision check rather than by a margin here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Sequence

import numpy as np


class Outcome(Enum):
    """Result of testing one gate against one position step."""

    NONE = auto()        # the gate plane was not crossed forward this step
    PASS = auto()        # crossed legally
    WRONG_GATE = auto()  # crossed through the decoy opening
    WRONG_SIDE = auto()  # crossed on the illegal side of a bar
    BLOCKED = auto()     # crossed the plane but through no opening at all


class BarSide(Enum):
    """Which side of a bar is the legal one."""

    ABOVE = auto()
    BELOW = auto()


def _forward_crossing(prev_pos, cur_pos, plane_y: float) -> Optional[float]:
    """Interpolation factor for a forward (+y) crossing of `plane_y`, else None.

    Returns `a` in [0, 1) such that the crossing point is
    `prev_pos + a * (cur_pos - prev_pos)`. Backward crossings do not count -- the
    drone retreating through a gate must not re-trigger it.
    """
    s0 = prev_pos[1] - plane_y
    s1 = cur_pos[1] - plane_y
    if s0 < 0.0 <= s1 and s1 != s0:
        return -s0 / (s1 - s0)
    return None


@dataclass(frozen=True)
class Opening:
    """A rectangular hole in a wall."""

    x_lo: float
    x_hi: float
    z_lo: float
    z_hi: float

    @property
    def center(self) -> np.ndarray:
        return np.array([0.5 * (self.x_lo + self.x_hi),
                         0.5 * (self.z_lo + self.z_hi)])

    @property
    def half_width(self) -> float:
        return 0.5 * (self.x_hi - self.x_lo)

    def contains(self, x: float, z: float, margin: float = 0.0) -> bool:
        return (self.x_lo + margin <= x <= self.x_hi - margin and
                self.z_lo + margin <= z <= self.z_hi - margin)


@dataclass(frozen=True)
class WallGate:
    """A wall plane with a target opening and an optional decoy opening."""

    y: float
    target: Opening
    decoy: Optional[Opening] = None
    margin: float = 0.06     # require the drone centre this far inside the opening

    def check(self, prev_pos, cur_pos) -> Outcome:
        a = _forward_crossing(prev_pos, cur_pos, self.y)
        if a is None:
            return Outcome.NONE
        xc = prev_pos[0] + a * (cur_pos[0] - prev_pos[0])
        zc = prev_pos[2] + a * (cur_pos[2] - prev_pos[2])
        if self.target.contains(xc, zc, self.margin):
            return Outcome.PASS
        if self.decoy is not None and self.decoy.contains(xc, zc, self.margin):
            return Outcome.WRONG_GATE
        return Outcome.BLOCKED

    def waypoint(self, standoff: float = 0.0) -> np.ndarray:
        """Dense-progress target: the opening centre, *in* the gate plane.

        Deliberately not past the plane. A target beyond the gate makes
        straight-line pursuit cross the plane while still converging laterally,
        so the drone arrives off-centre -- harmless on a short axial approach,
        fatal on this course where gates sit metres apart in z.
        """
        cx, cz = self.target.center
        return np.array([cx, self.y + standoff, cz])

    @property
    def center(self) -> np.ndarray:
        cx, cz = self.target.center
        return np.array([cx, self.y, cz])


#: Never aim below this altitude. The env terminates on z < 0.05, and the
#: lowest blue bar sits at 0.40 -- a naive `height - clearance` puts the target
#: at exactly 0.05, i.e. on the kill line. Measured effect: stage-3 pilot runs
#: ending in `ground` while tracking their waypoint correctly.
MIN_FLIGHT_Z = 0.25


@dataclass(frozen=True)
class BarGate:
    """A horizontal bar spanning the corridor; one side of it is legal."""

    y: float
    height: float            # bar centreline height, metres
    side: BarSide
    clearance: float = 0.35  # nominal offset from the bar on the legal side

    def check(self, prev_pos, cur_pos) -> Outcome:
        a = _forward_crossing(prev_pos, cur_pos, self.y)
        if a is None:
            return Outcome.NONE
        zc = prev_pos[2] + a * (cur_pos[2] - prev_pos[2])
        if self.side is BarSide.ABOVE:
            legal = zc > self.height
        else:
            legal = zc < self.height
        return Outcome.PASS if legal else Outcome.WRONG_SIDE

    @property
    def target_z(self) -> float:
        """Altitude to cross the plane at, floored above the ground cutoff."""
        if self.side is BarSide.ABOVE:
            return self.height + self.clearance
        return max(self.height - self.clearance, MIN_FLIGHT_Z)

    def waypoint(self, standoff: float = 0.0) -> np.ndarray:
        """Corridor centre at the legal-side altitude, *in* the bar's plane.

        See WallGate.waypoint -- the altitude has to be reached before the plane
        is crossed, not after, or the drone clips the wrong side of the bar.
        """
        return np.array([0.0, self.y + standoff, self.target_z])

    @property
    def center(self) -> np.ndarray:
        return np.array([0.0, self.y, self.target_z])


class GateSequence:
    """Ordered gates, tracking which one the drone must clear next.

    Only the *current* gate is tested each step. That is sound here because the
    gates are sequential planes along +y with lateral containment, so the drone
    cannot reach gate k's plane without first crossing gate k-1's.
    """

    def __init__(self, gates: Sequence[object]):
        self.gates = list(gates)
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    @property
    def done(self) -> bool:
        return self.index >= len(self.gates)

    @property
    def current(self):
        return None if self.done else self.gates[self.index]

    @property
    def n_gates(self) -> int:
        return len(self.gates)

    def update(self, prev_pos, cur_pos) -> Outcome:
        """Test the current gate; advance the index on a legal pass."""
        if self.done:
            return Outcome.NONE
        outcome = self.current.check(prev_pos, cur_pos)
        if outcome is Outcome.PASS:
            self.index += 1
        return outcome

    def waypoint(self, standoff: float = 0.5) -> Optional[np.ndarray]:
        """Progress target: the next uncleared gate's waypoint."""
        if self.done:
            return None
        return self.current.waypoint(standoff)

    def distance_to_plane(self, pos) -> float:
        """Signed distance to the current gate's plane (for the centering term)."""
        if self.done:
            return math.inf
        return float(pos[1] - self.current.y)
