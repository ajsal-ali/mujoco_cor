#!/usr/bin/env python3
"""The window arena, built procedurally — no SDF, no external assets.

Replaces the old `imav_play.make_world_injector`, which loaded the IMAV SDF from
`Files(3)/` and no longer exists. The geometry here reproduces what the task
actually depends on: an obstacle wall at y = 3.30 with two rectangular openings,
red and blue, at the coordinates `WindowGate` checks against.

Everything is generated: box geoms plus MuJoCo's builtin texture generators. The
env is therefore importable and trainable from a clean checkout with nothing but
the package installed.

What is preserved from the SDF world
------------------------------------
* wall plane and both opening rectangles, to the exact coordinates in WindowGate
* the red/blue colour cue -- the policy sees only pixels, so it has to be able to
  tell the two openings apart
* a collidable wall, so the contact-based crash detector still fires

What is not
-----------
* the SDF's meshes, aruco textures and decorative dolls
* the cosmetic spinning propellers (they were visual-only; `_viewer._prop_joints`
  already returns None when the joints are absent, so the viewer just skips the
  spin). Dropping them also avoids adding four hinge DOFs to the drone.

The arena is **static and identical across envs**, which is what makes it valid
for `multi_drone_mujoco.rendering.SharedStaticRenderer`.
"""

from __future__ import annotations


# Wall extent and opening geometry. The openings must match rl.window_aviary's
# WindowGate exactly -- it is the reward's ground truth, and a mismatch would
# score traversals against geometry the drone cannot see.
WALL_Y = 3.30
WALL_HALF_THICK = 0.05
X_MIN, X_MAX = -2.00, 2.00
Z_MIN, Z_MAX = 0.00, 5.00

RED_X_LO, RED_X_HI = 0.44, 1.32
BLUE_X_LO, BLUE_X_HI = -1.32, 0.00
OPEN_Z_LO, OPEN_Z_HI = 2.97, 3.85

_FRAME_BAR = 0.04      # half-width of the coloured frame bars
_FRAME_Y_OFF = 0.07    # frames sit on the approach (-y) side, so they are visible


def _box(name: str, cx: float, cy: float, cz: float,
         hx: float, hy: float, hz: float,
         material: str = None, rgba: str = None, collide: bool = True) -> str:
    look = f'material="{material}"' if material else f'rgba="{rgba}"'
    con = '' if collide else ' contype="0" conaffinity="0"'
    return (f'    <body name="{name}" pos="{cx:.4f} {cy:.4f} {cz:.4f}">\n'
            f'      <geom type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}" '
            f'{look}{con}/>\n'
            f'    </body>')


def _assets_xml() -> str:
    return """
    <texture name="win_wall_tex" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.55 0.55 0.58" rgb2="0.42 0.42 0.46"/>
    <material name="win_wall_mat" texture="win_wall_tex" texrepeat="6 6"
              specular="0.2" shininess="0.3"/>
    <texture name="win_post_tex" type="2d" builtin="gradient" width="256" height="256"
             rgb1="0.30 0.32 0.35" rgb2="0.18 0.19 0.22"/>
    <material name="win_post_mat" texture="win_post_tex" texrepeat="2 4"/>"""


def _frame_xml(tag: str, x_lo: float, x_hi: float, rgba: str) -> str:
    """Four thin coloured bars outlining one opening.

    Visual only (contype 0). The traversal test is WindowGate's geometric
    plane-crossing check, so these must not narrow the flyable aperture.
    """
    cx = 0.5 * (x_lo + x_hi)
    hx = 0.5 * (x_hi - x_lo)
    cz = 0.5 * (OPEN_Z_LO + OPEN_Z_HI)
    hz = 0.5 * (OPEN_Z_HI - OPEN_Z_LO)
    y = WALL_Y - _FRAME_Y_OFF
    parts = [
        _box(f"win_{tag}_bar_bot", cx, y, OPEN_Z_LO, hx, 0.02, _FRAME_BAR,
             rgba=rgba, collide=False),
        _box(f"win_{tag}_bar_top", cx, y, OPEN_Z_HI, hx, 0.02, _FRAME_BAR,
             rgba=rgba, collide=False),
        _box(f"win_{tag}_bar_left", x_lo, y, cz, _FRAME_BAR, 0.02, hz,
             rgba=rgba, collide=False),
        _box(f"win_{tag}_bar_right", x_hi, y, cz, _FRAME_BAR, 0.02, hz,
             rgba=rgba, collide=False),
    ]
    return "\n".join(parts)


def _world_xml(posts: bool = True) -> str:
    """The wall, split into segments that leave the two openings hollow."""
    ht = WALL_HALF_THICK
    x_half = 0.5 * (X_MAX - X_MIN)
    x_mid = 0.5 * (X_MIN + X_MAX)
    band_hz = 0.5 * (OPEN_Z_HI - OPEN_Z_LO)
    band_cz = 0.5 * (OPEN_Z_LO + OPEN_Z_HI)

    segs = [
        # below both openings
        _box("win_wall_bottom", x_mid, WALL_Y, 0.5 * (Z_MIN + OPEN_Z_LO),
             x_half, ht, 0.5 * (OPEN_Z_LO - Z_MIN), material="win_wall_mat"),
        # above both openings
        _box("win_wall_top", x_mid, WALL_Y, 0.5 * (OPEN_Z_HI + Z_MAX),
             x_half, ht, 0.5 * (Z_MAX - OPEN_Z_HI), material="win_wall_mat"),
        # pillars within the opening band: outside blue, between blue and red,
        # outside red
        _box("win_wall_pillar_l", 0.5 * (X_MIN + BLUE_X_LO), WALL_Y, band_cz,
             0.5 * (BLUE_X_LO - X_MIN), ht, band_hz, material="win_wall_mat"),
        _box("win_wall_pillar_m", 0.5 * (BLUE_X_HI + RED_X_LO), WALL_Y, band_cz,
             0.5 * (RED_X_LO - BLUE_X_HI), ht, band_hz, material="win_wall_mat"),
        _box("win_wall_pillar_r", 0.5 * (RED_X_HI + X_MAX), WALL_Y, band_cz,
             0.5 * (X_MAX - RED_X_HI), ht, band_hz, material="win_wall_mat"),
        _frame_xml("red", RED_X_LO, RED_X_HI, "0.85 0.12 0.12 1"),
        _frame_xml("blue", BLUE_X_LO, BLUE_X_HI, "0.12 0.25 0.85 1"),
    ]

    if posts:
        # A little static scenery beyond the wall, so the view through the
        # opening is not a flat void and the depth channel carries signal.
        for i, (px, py) in enumerate(((-1.4, 5.2), (1.4, 5.2), (0.0, 6.4))):
            segs.append(_box(f"win_post_{i}", px, py, 1.25, 0.09, 0.09, 1.25,
                             material="win_post_mat"))

    return "\n".join(segs)


def make_window_world_injector(posts: bool = True):
    """Return (original, patched) for `_BA._generate_aviary_xml`.

    Same patch-and-restore mechanism the env used with the SDF injector, so the
    call site in WindowAviary barely changes.

    The package import is deferred to here so the geometry helpers above stay
    importable (and testable) without pulling in mujoco/gymnasium.
    """
    from multi_drone_mujoco.envs import base_aviary as _BA

    original = _BA._generate_aviary_xml

    def patched(*args, **kwargs):
        xml = original(*args, **kwargs)
        if "</asset>" not in xml or "</worldbody>" not in xml:
            raise RuntimeError(
                "window_world: generated XML has no <asset>/<worldbody> to "
                "splice into — the base XML layout changed.")
        xml = xml.replace("</asset>", _assets_xml() + "\n  </asset>", 1)
        xml = xml.replace("</worldbody>", _world_xml(posts) + "\n  </worldbody>", 1)
        return xml

    return original, patched
