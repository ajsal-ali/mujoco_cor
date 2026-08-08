#!/usr/bin/env python3
"""Procedural obstacle-course geometry for MAVRL.

Self-contained: no imports from `rl/`, no mujoco at module level, no external
assets. Everything below is string XML plus MuJoCo builtin texture generators, so
this module is importable and unit-testable from a clean checkout on any OS.

Course layout (all stations are planes normal to +y, spaced STATION_SPACING):

    y = 3.30            entry wall, two openings -- RED is the target,
                        BLUE is a decoy the policy must learn to reject
    y = 3.30 + k*2.50   0..3 bar stations. A bar spans the full corridor width;
                        free space exists above and below it, and the colour
                        says which side is legal (red -> above, blue -> below)
    y = exit_y          exit wall, single centred opening

Lateral containment is *not* geometric -- the env terminates on |x| > 2.0 past
the entry wall. That keeps the obstacle set exactly as specified rather than
adding corridor walls.

Note on the background posts: rl.window_world puts three decorative posts at
y = 5.2 and 6.4, which is inside this corridor and would collide with stations 1
and 2. They are dropped here; the course itself now provides the depth structure
they existed to supply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence, Tuple

import numpy as np

from mavrl.config import RENDER_RES
from mavrl.course_gates import (
    BarGate, BarSide, GateSequence, Opening, WallGate,
)


# --------------------------------------------------------------------------
# Geometry constants. These are the ground truth shared with course_gates --
# the gate checks and the rendered arena are built from the same numbers, so
# they cannot drift apart.
# --------------------------------------------------------------------------

X_MIN, X_MAX = -2.00, 2.00
Z_MIN, Z_MAX = 0.00, 5.00
WALL_HALF_THICK = 0.05

# Entry wall -- identical to the arena rl/ already trains on.
ENTRY_Y = 3.30
RED_X_LO, RED_X_HI = 0.44, 1.32
BLUE_X_LO, BLUE_X_HI = -1.32, 0.00
ENTRY_Z_LO, ENTRY_Z_HI = 2.97, 3.85

# Exit wall -- single centred opening, down in the bar band so the descent
# through the course is monotone rather than up-down-up.
EXIT_X_LO, EXIT_X_HI = -0.44, 0.44
EXIT_Z_LO, EXIT_Z_HI = 1.06, 1.94

#: Constant gap between consecutive gate planes.
#:
#: 4.0 rather than 2.5 because of the vertical profile: the drone leaves the
#: entry window at z ~= 3.4 and must reach the exit at z = 1.5, or dip under a
#: blue bar at 0.4. At 2.5 m spacing and ~1.5 m/s that is a 2 m descent in 1.7 s
#: (~1.2 m/s sustained, arriving with no margin) -- measured result was the
#: pilot clipping the exit sill at z = 1.06 on essentially every run. At 4.0 m
#: the same descent has 2.7 s and lands comfortably inside the opening.
STATION_SPACING = 4.00

# Bars: thin horizontal boxes spanning the full corridor width.
BAR_HY = 0.05
BAR_HZ = 0.04

RED_BAR_HEIGHTS = (1.20, 1.60, 1.98)
BLUE_BAR_HEIGHTS = (0.40, 0.80, 1.20)

# One height per colour is held out of training entirely, so evaluation on it
# measures whether the above/below rule generalized or the heights were memorized.
TRAIN_RED_HEIGHTS = (1.20, 1.98)
EVAL_RED_HEIGHTS = (1.60,)
TRAIN_BLUE_HEIGHTS = (0.40, 1.20)
EVAL_BLUE_HEIGHTS = (0.80,)

# Spawn standoff from the entry wall, so the drone opens its eyes with the
# opening comfortably in frame rather than with its nose against the wall.
SPAWN_STANDOFF = 2.00
SPAWN_STANDOFF_JIT = 0.30

_FRAME_BAR = 0.04      # half-width of the coloured frame bars
_FRAME_Y_OFF = 0.07    # frames sit on the approach (-y) side, so they are visible

RED_RGBA = "0.85 0.12 0.12 1"
BLUE_RGBA = "0.12 0.25 0.85 1"


# --------------------------------------------------------------------------
# Semantic classes for the SeVAE segmentation head
# --------------------------------------------------------------------------

class SemClass(IntEnum):
    FREE = 0
    WALL = 1
    RED_WINDOW = 2
    BLUE_WINDOW = 3
    RED_BAR = 4
    BLUE_BAR = 5
    FLOOR = 6


N_SEM_CLASSES = len(SemClass)


def classify_geom_name(name: Optional[str]) -> SemClass:
    """Map a geom name to its semantic class.

    Unknown and unnamed geoms fall through to FREE -- that covers the drone's
    own geoms and anything BaseAviary adds, none of which the policy needs to
    distinguish.
    """
    if not name:
        return SemClass.FREE
    if name.startswith("entry_frame_red") :
        return SemClass.RED_WINDOW
    if name.startswith("entry_frame_blue"):
        return SemClass.BLUE_WINDOW
    if name.startswith("bar_red"):
        return SemClass.RED_BAR
    if name.startswith("bar_blue"):
        return SemClass.BLUE_BAR
    if name.startswith(("entry_wall", "exit_wall")):
        return SemClass.WALL
    if any(k in name for k in ("floor", "ground", "plane")):
        return SemClass.FLOOR
    return SemClass.FREE


# --------------------------------------------------------------------------
# Layout description
# --------------------------------------------------------------------------

class StationType(IntEnum):
    RED_BAR = 0     # legal side: above
    BLUE_BAR = 1    # legal side: below


@dataclass(frozen=True)
class StationSpec:
    kind: StationType
    height: float

    @property
    def side(self) -> BarSide:
        return BarSide.ABOVE if self.kind is StationType.RED_BAR else BarSide.BELOW

    @property
    def rgba(self) -> str:
        return RED_RGBA if self.kind is StationType.RED_BAR else BLUE_RGBA

    @property
    def tag(self) -> str:
        return "red" if self.kind is StationType.RED_BAR else "blue"


@dataclass(frozen=True)
class CourseLayout:
    """An ordered course. `stations` may be empty (entry -> exit directly)."""

    stations: Tuple[StationSpec, ...] = ()

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    def station_y(self, k: int) -> float:
        """Plane of the k-th bar station (0-indexed, first one past the entry)."""
        return ENTRY_Y + (k + 1) * STATION_SPACING

    @property
    def exit_y(self) -> float:
        return ENTRY_Y + (self.n_stations + 1) * STATION_SPACING

    @property
    def n_gates(self) -> int:
        return self.n_stations + 2      # entry + bars + exit

    def describe(self) -> str:
        if not self.stations:
            return "entry -> exit"
        mid = " -> ".join(f"{s.tag}@{s.height:.2f}" for s in self.stations)
        return f"entry -> {mid} -> exit"


# --------------------------------------------------------------------------
# Layout sampling / curriculum
# --------------------------------------------------------------------------

#: Curriculum stage -> number of intermediate bar stations.
STAGE_STATIONS = (0, 1, 2, 3)


def heights_for(kind: StationType, split: str = "train") -> Tuple[float, ...]:
    if split == "train":
        return TRAIN_RED_HEIGHTS if kind is StationType.RED_BAR else TRAIN_BLUE_HEIGHTS
    if split == "eval":
        return EVAL_RED_HEIGHTS if kind is StationType.RED_BAR else EVAL_BLUE_HEIGHTS
    if split == "all":
        return RED_BAR_HEIGHTS if kind is StationType.RED_BAR else BLUE_BAR_HEIGHTS
    raise ValueError(f"unknown split {split!r} (expected train/eval/all)")


def sample_layout(rng: np.random.Generator, n_stations: int,
                  split: str = "train") -> CourseLayout:
    """Random station types and heights; count is fixed by the curriculum stage."""
    stations = []
    for _ in range(n_stations):
        kind = StationType(int(rng.integers(0, 2)))
        choices = heights_for(kind, split)
        stations.append(StationSpec(kind, float(rng.choice(choices))))
    return CourseLayout(tuple(stations))


def sample_layout_for_stage(rng: np.random.Generator, stage: int,
                            split: str = "train") -> CourseLayout:
    stage = int(np.clip(stage, 0, len(STAGE_STATIONS) - 1))
    return sample_layout(rng, STAGE_STATIONS[stage], split)


def all_layouts(n_stations: int, split: str = "all") -> Tuple[CourseLayout, ...]:
    """Every (type, height) combination for a given station count.

    Used by the collector to cover the space exhaustively rather than by
    sampling, and by tests.
    """
    from itertools import product

    per_slot = [
        StationSpec(kind, h)
        for kind in (StationType.RED_BAR, StationType.BLUE_BAR)
        for h in heights_for(kind, split)
    ]
    return tuple(CourseLayout(combo)
                 for combo in product(per_slot, repeat=n_stations))


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

def build_gates(layout: CourseLayout) -> GateSequence:
    """The gate sequence matching `layout`'s geometry, in flight order."""
    entry = WallGate(
        y=ENTRY_Y,
        target=Opening(RED_X_LO, RED_X_HI, ENTRY_Z_LO, ENTRY_Z_HI),
        decoy=Opening(BLUE_X_LO, BLUE_X_HI, ENTRY_Z_LO, ENTRY_Z_HI),
    )
    gates = [entry]
    for k, st in enumerate(layout.stations):
        gates.append(BarGate(y=layout.station_y(k), height=st.height, side=st.side))
    gates.append(WallGate(
        y=layout.exit_y,
        target=Opening(EXIT_X_LO, EXIT_X_HI, EXIT_Z_LO, EXIT_Z_HI),
        decoy=None,
    ))
    return GateSequence(gates)


def spawn_pose(rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """Start pose: SPAWN_STANDOFF metres in front of the entry wall, facing +y."""
    import math

    standoff = SPAWN_STANDOFF + rng.uniform(-SPAWN_STANDOFF_JIT, SPAWN_STANDOFF_JIT)
    cx = 0.5 * (RED_X_LO + RED_X_HI)
    cz = 0.5 * (ENTRY_Z_LO + ENTRY_Z_HI)
    pos = np.array([
        cx + rng.uniform(-0.45, 0.45),
        ENTRY_Y - standoff,
        cz + rng.uniform(-0.40, 0.40),
    ])
    yaw = math.pi / 2 + rng.uniform(-math.radians(15), math.radians(15))
    return pos, float(yaw)


# --------------------------------------------------------------------------
# XML generation
# --------------------------------------------------------------------------

def _box(name: str, cx: float, cy: float, cz: float,
         hx: float, hy: float, hz: float,
         material: Optional[str] = None, rgba: Optional[str] = None,
         collide: bool = True) -> str:
    """One named box body containing one *named* geom.

    Segmentation rendering returns geom ids, so the geom must carry the name --
    naming only the body would force a detour through `model.geom_bodyid` to
    recover the semantic class.
    """
    look = f'material="{material}"' if material else f'rgba="{rgba}"'
    con = '' if collide else ' contype="0" conaffinity="0"'
    return (f'    <body name="{name}" pos="{cx:.4f} {cy:.4f} {cz:.4f}">\n'
            f'      <geom name="{name}" type="box" '
            f'size="{hx:.4f} {hy:.4f} {hz:.4f}" {look}{con}/>\n'
            f'    </body>')


def assets_xml() -> str:
    return """
    <texture name="mavrl_wall_tex" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.55 0.55 0.58" rgb2="0.42 0.42 0.46"/>
    <material name="mavrl_wall_mat" texture="mavrl_wall_tex" texrepeat="6 6"
              specular="0.2" shininess="0.3"/>"""


def _frame_xml(prefix: str, x_lo: float, x_hi: float,
               z_lo: float, z_hi: float, y: float, rgba: str) -> str:
    """Four thin coloured bars outlining one opening.

    Visual only (contype 0) -- the traversal test is the geometric plane crossing
    in course_gates, so these must not narrow the flyable aperture.
    """
    cx = 0.5 * (x_lo + x_hi)
    hx = 0.5 * (x_hi - x_lo)
    cz = 0.5 * (z_lo + z_hi)
    hz = 0.5 * (z_hi - z_lo)
    return "\n".join([
        _box(f"{prefix}_bot", cx, y, z_lo, hx, 0.02, _FRAME_BAR, rgba=rgba, collide=False),
        _box(f"{prefix}_top", cx, y, z_hi, hx, 0.02, _FRAME_BAR, rgba=rgba, collide=False),
        _box(f"{prefix}_left", x_lo, y, cz, _FRAME_BAR, 0.02, hz, rgba=rgba, collide=False),
        _box(f"{prefix}_right", x_hi, y, cz, _FRAME_BAR, 0.02, hz, rgba=rgba, collide=False),
    ])


def _wall_xml(prefix: str, wall_y: float,
              openings: Sequence[Tuple[float, float]],
              z_lo: float, z_hi: float) -> str:
    """A wall split into segments that leave `openings` (x ranges) hollow.

    `openings` must be sorted and non-overlapping.
    """
    ht = WALL_HALF_THICK
    x_half = 0.5 * (X_MAX - X_MIN)
    x_mid = 0.5 * (X_MIN + X_MAX)
    band_cz = 0.5 * (z_lo + z_hi)
    band_hz = 0.5 * (z_hi - z_lo)

    segs = [
        _box(f"{prefix}_bottom", x_mid, wall_y, 0.5 * (Z_MIN + z_lo),
             x_half, ht, 0.5 * (z_lo - Z_MIN), material="mavrl_wall_mat"),
        _box(f"{prefix}_top", x_mid, wall_y, 0.5 * (z_hi + Z_MAX),
             x_half, ht, 0.5 * (Z_MAX - z_hi), material="mavrl_wall_mat"),
    ]

    # Pillars filling the band between openings.
    edges = [X_MIN]
    for lo, hi in openings:
        edges += [lo, hi]
    edges.append(X_MAX)
    for i in range(0, len(edges) - 1, 2):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo <= 1e-9:
            continue
        segs.append(_box(f"{prefix}_pillar{i // 2}", 0.5 * (lo + hi), wall_y, band_cz,
                         0.5 * (hi - lo), ht, band_hz, material="mavrl_wall_mat"))
    return "\n".join(segs)


def world_xml(layout: CourseLayout) -> str:
    """Full course worldbody XML for one layout."""
    parts = [
        _wall_xml("entry_wall", ENTRY_Y,
                  [(BLUE_X_LO, BLUE_X_HI), (RED_X_LO, RED_X_HI)],
                  ENTRY_Z_LO, ENTRY_Z_HI),
        _frame_xml("entry_frame_red", RED_X_LO, RED_X_HI,
                   ENTRY_Z_LO, ENTRY_Z_HI, ENTRY_Y - _FRAME_Y_OFF, RED_RGBA),
        _frame_xml("entry_frame_blue", BLUE_X_LO, BLUE_X_HI,
                   ENTRY_Z_LO, ENTRY_Z_HI, ENTRY_Y - _FRAME_Y_OFF, BLUE_RGBA),
    ]

    for k, st in enumerate(layout.stations):
        parts.append(_box(
            f"bar_{st.tag}_{k}",
            0.5 * (X_MIN + X_MAX), layout.station_y(k), st.height,
            0.5 * (X_MAX - X_MIN), BAR_HY, BAR_HZ,
            rgba=st.rgba, collide=True))

    parts.append(_wall_xml("exit_wall", layout.exit_y,
                           [(EXIT_X_LO, EXIT_X_HI)],
                           EXIT_Z_LO, EXIT_Z_HI))
    return "\n".join(parts)


def patch_offscreen_framebuffer(xml: str, res: int) -> str:
    """Enlarge MuJoCo's offscreen framebuffer to `res` x `res`.

    The default is 640x480, so any render above 480 px tall fails outright with
    "Image height N > framebuffer height 480". BaseAviary's XML already carries a
    `<global azimuth=... elevation=.../>` inside `<visual>`, so the size
    attributes are added to that tag rather than a second one being introduced --
    MuJoCo rejects duplicate `<global>` elements.
    """
    m = re.search(r"<global\b([^>]*)/>", xml)
    if m:
        attrs = m.group(1)
        attrs = re.sub(r'\s*off(width|height)="[^"]*"', "", attrs)
        return (xml[:m.start()]
                + f'<global{attrs} offwidth="{res}" offheight="{res}"/>'
                + xml[m.end():])
    if "<visual>" in xml:
        return xml.replace(
            "<visual>",
            f'<visual>\n    <global offwidth="{res}" offheight="{res}"/>', 1)
    raise RuntimeError(
        "course_world: no <visual> block to size the offscreen framebuffer in")


def make_course_world_injector(layout: CourseLayout,
                               render_res: int = RENDER_RES):
    """Return (original, patched) for `BaseAviary._generate_aviary_xml`.

    Same patch-and-restore mechanism rl.window_world uses, so the env's call site
    is a two-line context. The package import is deferred to here so everything
    above stays importable without mujoco.
    """
    from multi_drone_mujoco.envs import base_aviary as _BA

    original = _BA._generate_aviary_xml

    def patched(*args, **kwargs):
        xml = original(*args, **kwargs)
        if "</asset>" not in xml or "</worldbody>" not in xml:
            raise RuntimeError(
                "course_world: generated XML has no <asset>/<worldbody> to "
                "splice into -- the base XML layout changed.")
        xml = xml.replace("</asset>", assets_xml() + "\n  </asset>", 1)
        xml = xml.replace("</worldbody>", world_xml(layout) + "\n  </worldbody>", 1)
        return patch_offscreen_framebuffer(xml, render_res)

    return original, patched


def build_geom_class_lut(model) -> np.ndarray:
    """geom_id -> SemClass lookup for the segmentation buffer.

    Needs a live mujoco model, so it is called from the env, not from here.
    """
    import mujoco

    lut = np.full(model.ngeom, SemClass.FREE, dtype=np.uint8)
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        lut[gid] = int(classify_geom_name(name))
    return lut
