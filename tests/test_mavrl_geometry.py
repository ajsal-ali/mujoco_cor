#!/usr/bin/env python3
"""Geometry, gate and image-pipeline tests.

Runs anywhere -- mavrl.course_gates, mavrl.course_world and mavrl.imageproc
import no mujoco and no torch, which is the whole reason they are separate
modules. Run with `pytest tests/` or `python tests/test_mavrl_geometry.py`.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mavrl.course_gates import (
    BarGate, BarSide, GateSequence, Opening, Outcome, WallGate,
)
from mavrl.course_world import (
    BLUE_BAR_HEIGHTS, ENTRY_Y, EVAL_BLUE_HEIGHTS, EVAL_RED_HEIGHTS,
    RED_BAR_HEIGHTS, STATION_SPACING, TRAIN_BLUE_HEIGHTS, TRAIN_RED_HEIGHTS,
    CourseLayout, SemClass, StationSpec, StationType, all_layouts, build_gates,
    classify_geom_name, sample_layout, spawn_pose, world_xml,
)
from mavrl.imageproc import (
    downsample_depth, downsample_rgb, downsample_seg, proximity_weight,
    seg_to_classes,
)
from mavrl.sensor_noise import NoiseConfig, corrupt


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def test_wall_gate_target_decoy_and_blocked():
    g = WallGate(y=3.3,
                 target=Opening(0.44, 1.32, 2.97, 3.85),
                 decoy=Opening(-1.32, 0.0, 2.97, 3.85))
    assert g.check([0.88, 3.2, 3.41], [0.88, 3.4, 3.41]) is Outcome.PASS
    assert g.check([-0.66, 3.2, 3.41], [-0.66, 3.4, 3.41]) is Outcome.WRONG_GATE
    assert g.check([0.0, 3.2, 1.0], [0.0, 3.4, 1.0]) is Outcome.BLOCKED
    assert g.check([0.88, 3.2, 3.41], [0.88, 3.25, 3.41]) is Outcome.NONE


def test_backward_crossing_does_not_trigger():
    """Retreating through a gate must not re-score it."""
    g = WallGate(y=3.3, target=Opening(0.44, 1.32, 2.97, 3.85))
    assert g.check([0.88, 3.4, 3.41], [0.88, 3.2, 3.41]) is Outcome.NONE


def test_bar_gate_sides():
    red = BarGate(y=6.0, height=1.6, side=BarSide.ABOVE)
    assert red.check([0, 5.9, 2.2], [0, 6.1, 2.2]) is Outcome.PASS
    assert red.check([0, 5.9, 1.0], [0, 6.1, 1.0]) is Outcome.WRONG_SIDE

    blue = BarGate(y=6.0, height=0.8, side=BarSide.BELOW)
    assert blue.check([0, 5.9, 0.4], [0, 6.1, 0.4]) is Outcome.PASS
    assert blue.check([0, 5.9, 1.5], [0, 6.1, 1.5]) is Outcome.WRONG_SIDE


def test_bar_gate_uses_interpolated_crossing_height():
    """The z at the plane decides, not the z at either endpoint."""
    bar = BarGate(y=6.0, height=1.2, side=BarSide.BELOW)
    # starts above the bar, ends below it, but crosses the plane at z=1.5
    assert bar.check([0, 5.5, 2.0], [0, 6.5, 1.0]) is Outcome.WRONG_SIDE
    # same endpoints in z but crossing later, at z=1.1
    assert bar.check([0, 5.9, 2.0], [0, 6.1, 0.2]) is Outcome.PASS


def test_waypoints_lie_in_the_gate_plane():
    """Regression: a waypoint past the plane makes pursuit cross at the wrong
    altitude. Found by flying a straight line at a blue bar at 1.20 and
    crossing at 1.36."""
    layout = CourseLayout((StationSpec(StationType.BLUE_BAR, 1.20),))
    for gate in build_gates(layout).gates:
        assert gate.waypoint()[1] == gate.y


def test_gate_sequence_orders_and_completes():
    layout = CourseLayout((StationSpec(StationType.RED_BAR, 1.20),
                           StationSpec(StationType.BLUE_BAR, 0.80)))
    gs = build_gates(layout)
    assert gs.n_gates == 4

    pos = np.array([0.88, ENTRY_Y - 2.0, 3.41])
    outcomes = []
    for _ in range(2000):
        wp = gs.waypoint()
        if wp is None:
            break
        prev = pos.copy()
        step = wp - pos
        n = np.linalg.norm(step)
        pos = pos + step / max(n, 1e-9) * min(0.05, n)
        o = gs.update(prev, pos)
        if o is not Outcome.NONE:
            outcomes.append(o)
        if n < 1e-3:                       # reached the waypoint; nudge past it
            pos = pos + np.array([0.0, 0.05, 0.0])
            o = gs.update(pos - np.array([0.0, 0.05, 0.0]), pos)
            if o is not Outcome.NONE:
                outcomes.append(o)
    assert gs.done, f"course not completed; outcomes={outcomes}"
    assert all(o is Outcome.PASS for o in outcomes), outcomes


# --------------------------------------------------------------------------
# world
# --------------------------------------------------------------------------

def test_station_spacing_and_exit():
    layout = CourseLayout(tuple(
        StationSpec(StationType.RED_BAR, 1.2) for _ in range(3)))
    ys = [layout.station_y(k) for k in range(3)]
    assert ys == [ENTRY_Y + STATION_SPACING * (k + 1) for k in range(3)]
    assert layout.exit_y == ENTRY_Y + STATION_SPACING * 4
    assert all(b > a for a, b in zip(ys, ys[1:]))


def test_gate_planes_strictly_increasing():
    rng = np.random.default_rng(0)
    for n in range(4):
        gates = build_gates(sample_layout(rng, n)).gates
        ys = [g.y for g in gates]
        assert ys == sorted(ys) and len(set(ys)) == len(ys), ys


def test_every_geom_is_named_and_classified():
    layout = CourseLayout((StationSpec(StationType.RED_BAR, 1.98),
                           StationSpec(StationType.BLUE_BAR, 0.40)))
    xml = world_xml(layout)
    assert xml.count("<geom") == xml.count("<body")
    assert xml.count('<geom name=') == xml.count("<geom")

    names = [line.split('"')[1] for line in xml.splitlines()
             if "<body name=" in line]
    classes = {classify_geom_name(n) for n in names}
    for expected in (SemClass.WALL, SemClass.RED_WINDOW, SemClass.BLUE_WINDOW,
                     SemClass.RED_BAR, SemClass.BLUE_BAR):
        assert expected in classes, (expected, sorted(c.name for c in classes))


def test_bars_span_the_full_corridor():
    """A bar you can fly around laterally is not an obstacle."""
    layout = CourseLayout((StationSpec(StationType.RED_BAR, 1.6),))
    line = [ln for ln in world_xml(layout).splitlines() if 'name="bar_red_0"' in ln]
    size = [ln for ln in world_xml(layout).splitlines()
            if 'name="bar_red_0" type' in ln][0]
    hx = float(size.split('size="')[1].split()[0])
    assert math.isclose(hx, 2.0), hx


def test_held_out_heights_are_disjoint():
    assert not set(TRAIN_RED_HEIGHTS) & set(EVAL_RED_HEIGHTS)
    assert not set(TRAIN_BLUE_HEIGHTS) & set(EVAL_BLUE_HEIGHTS)
    assert set(TRAIN_RED_HEIGHTS) | set(EVAL_RED_HEIGHTS) == set(RED_BAR_HEIGHTS)
    assert set(TRAIN_BLUE_HEIGHTS) | set(EVAL_BLUE_HEIGHTS) == set(BLUE_BAR_HEIGHTS)


def test_sample_layout_respects_split():
    rng = np.random.default_rng(1)
    for _ in range(50):
        for st in sample_layout(rng, 3, "train").stations:
            allowed = (TRAIN_RED_HEIGHTS if st.kind is StationType.RED_BAR
                       else TRAIN_BLUE_HEIGHTS)
            assert st.height in allowed


def test_all_layouts_covers_both_orders():
    combos = all_layouts(2, "train")
    assert len(combos) == 16          # 4 options per slot, 2 slots
    tags = {tuple(s.tag for s in c.stations) for c in combos}
    assert ("red", "blue") in tags and ("blue", "red") in tags


def test_spawn_standoff_is_in_range():
    rng = np.random.default_rng(3)
    for _ in range(200):
        pos, yaw = spawn_pose(rng)
        assert 1.7 <= ENTRY_Y - pos[1] <= 2.3
        assert abs(yaw - math.pi / 2) <= math.radians(15) + 1e-9


# --------------------------------------------------------------------------
# image pipeline
# --------------------------------------------------------------------------

def test_min_pool_preserves_a_one_pixel_bar():
    depth = np.full((512, 512), 10.0, np.float32)
    depth[256, :] = 2.0
    out = downsample_depth(depth, 128)
    assert (out.min(axis=1) < 3.0).sum() == 1
    mean_pooled = depth.reshape(128, 4, 128, 4).mean(axis=(1, 3))
    assert mean_pooled.min() > 7.0        # a mean would erase the bar


def test_seg_downsample_invents_no_classes():
    seg = np.zeros((512, 512), np.int32)
    seg[100:140] = 4
    seg[300:340] = 5
    out = downsample_seg(seg, 128)
    assert set(np.unique(out).tolist()) <= {0, 4, 5}


def test_rgb_downsample_shape_and_range():
    rgb = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    out = downsample_rgb(rgb, 128)
    assert out.shape == (128, 128, 3) and out.dtype == np.uint8


def test_proximity_weight_profile():
    w = proximity_weight(np.array([0.0, 2.0, 6.0, 12.0, 30.0]))
    assert math.isclose(w[0], 1.0, abs_tol=1e-6)
    assert math.isclose(w[1], 0.768, abs_tol=2e-3)
    assert math.isclose(w[2], 0.309, abs_tol=2e-3)
    assert math.isclose(w[3], 0.05, abs_tol=1e-6)
    assert math.isclose(w[4], 0.05, abs_tol=1e-6)      # floored, never zero
    d = np.linspace(0.5, 12.0, 200)
    assert np.all(np.diff(proximity_weight(d)) <= 1e-9)   # monotone decreasing


def test_seg_to_classes_handles_both_buffer_shapes():
    lut = np.array([0, 1, 4, 5], dtype=np.uint8)
    flat = np.full((512, 512), 2, dtype=np.int32)
    stacked = np.stack([flat, np.zeros_like(flat)], axis=-1)
    assert np.array_equal(seg_to_classes(flat, lut, 128),
                          seg_to_classes(stacked, lut, 128))
    assert seg_to_classes(flat, lut, 128)[0, 0] == 4


def test_seg_to_classes_ignores_out_of_range_ids():
    lut = np.array([0, 1], dtype=np.uint8)
    buf = np.full((512, 512), -1, dtype=np.int32)     # sky
    assert seg_to_classes(buf, lut, 128).max() == 0


def test_noise_disabled_is_a_noop():
    rng = np.random.default_rng(0)
    rgb = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    depth = np.random.rand(64, 64).astype(np.float32) * 10
    r, d = corrupt(rgb, depth, NoiseConfig.disabled(), rng)
    assert np.array_equal(r, rgb) and np.array_equal(d, depth)


def test_noise_is_bounded_and_scales():
    rng = np.random.default_rng(0)
    depth = np.full((256, 256), 4.0, np.float32)
    strong = corrupt(np.zeros((256, 256, 3), np.uint8), depth,
                     NoiseConfig(), rng)[1]
    weak = corrupt(np.zeros((256, 256, 3), np.uint8), depth,
                   NoiseConfig().scaled(0.1), rng)[1]
    clean = np.abs(strong - 4.0)
    assert np.median(clean[clean < 1.0]) < 0.1        # sigma(4m) ~ 1.6 cm
    assert (np.abs(weak - 4.0).mean() < np.abs(strong - 4.0).mean())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"  FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
