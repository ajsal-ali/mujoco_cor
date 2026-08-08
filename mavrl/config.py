#!/usr/bin/env python3
"""Shared constants for the MAVRL pipeline.

One module so the env, the dataset writer, the SeVAE and the notebook cannot
drift apart on image size, depth scaling or the proximity-weight curve. Nothing
here imports mujoco or torch.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Vision
# --------------------------------------------------------------------------

#: MuJoCo renders at this resolution...
RENDER_RES = 512
#: ...and is downsampled to this before it reaches the encoder.
#: The gap is not vanity: an 0.08 m bar subtends 9.78/d pixels at 128 px
#: (~5 px at 2 m, ~1 px at 8 m), so rendering natively at 128 aliases thin
#: bars away. Rendering at 512 and min-pooling preserves them.
IMG_RES = 128

#: Depth clip, metres. 12 rather than 8 because the course is ~13 m end to end
#: and the proximity weight below runs out to 12 -- a shorter clip would make
#: the far half of the course invisible and the weight tail meaningless.
DEPTH_MAX = 12.0

IMG_CHANNELS = 4        # RGB + depth

# --------------------------------------------------------------------------
# Proximity weight for the SeVAE reconstruction loss
#
#   t = clip((d - D_NEAR) / (D_FAR - D_NEAR), 0, 1)
#   w = W_MIN + (1 - W_MIN) * (1 - t)**2
#
# 1.00 at contact, 0.92 @1m, 0.77 @2m, 0.63 @3m, 0.51 @4m, 0.31 @6m,
# 0.17 @8m, 0.08 @10m, 0.05 floor from 12 m out. Quadratic rather than
# exponential so the decay stays gradual through the mid-field where the *next*
# station first becomes visible; exp(-d/2) would already be at 0.05 by 6 m.
# --------------------------------------------------------------------------

W_MIN = 0.05
D_NEAR = 0.5
D_FAR = 12.0

# --------------------------------------------------------------------------
# Control rates
#
# 240 Hz physics / 60 Hz PID / 10 Hz policy. The policy holds one acceleration
# command across CTRL_PER_POLICY inner ticks; rendering happens once per policy
# step, which is what makes the 512 raster affordable.
# --------------------------------------------------------------------------

SIM_FREQ = 240
CTRL_FREQ = 60
POLICY_FREQ = 10
CTRL_PER_POLICY = CTRL_FREQ // POLICY_FREQ          # 6
SIM_STEPS_PER_POLICY = SIM_FREQ // POLICY_FREQ      # 24
POLICY_DT = 1.0 / POLICY_FREQ

# --------------------------------------------------------------------------
# Action space: body acceleration + yaw rate, integrated to a velocity setpoint
#
# Z authority is deliberately generous. The course drops from the entry window
# at z ~= 3.41 to below a 0.40 m blue bar within one 2.5 m station spacing: at
# 1.5 m/s forward that needs ~2 m/s of descent, which the original env's
# V_Z = 0.5 could not deliver.
# --------------------------------------------------------------------------

A_MAX = (3.0, 3.0, 4.0)      # m/s^2, body frame
V_MAX = (2.5, 2.5, 2.5)      # m/s, body frame
YAW_RATE_MAX = 1.0           # rad/s

N_PROPRIO = 16               # grav(3) gyro(3) vel(3) v_cmd(3) prev_action(4)
N_ACTIONS = 4

# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------

K_PROG = 1.0
K_TIME = 0.03
K_SMOOTH = 0.01
K_CENTER = 0.1
CENTER_BAND = 1.5            # apply the centering term within this |dy| of a gate

R_TOTAL = 150.0              # split evenly across the course's gates
R_FINISH = 100.0
R_CRASH = 25.0
K_TBONUS = 5.0

CORRIDOR_X_LIMIT = 2.0       # terminate past the entry wall if |x| exceeds this


def t_nominal(n_stations: int) -> float:
    """Reference time for the completion speed bonus."""
    return 4.0 + 2.0 * n_stations


def t_max(n_stations: int) -> float:
    """Episode time limit, seconds."""
    return 8.0 + 4.0 * n_stations


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------

LATENT_DIM = 64              # N_e, VAE latent
MEMORY_DIM = 256             # N_l, memory output
ATTN_WINDOW = 16             # K -- deliberately < AUX_T, see below
N_MEM_TOKENS = 4             # RMT recurrent memory tokens
ATTN_HEADS = 4
ATTN_LAYERS = 2

#: Auxiliary reconstruction offset. NOT a memory window -- it is the prediction
#: target ("from z_t, rebuild the image from T steps ago"), which pressures the
#: recurrent state into retaining that much history.
#:
#: AUX_T > ATTN_WINDOW on purpose: if the attention window could still see step
#: t-T, the aux loss would be satisfiable by copying rather than remembering,
#: and only the LSTM would face a real memory task.
AUX_T = 20

#: Paper Eq. (2): the head emits 3*N_e and splits it into past / current /
#: future segments, with lambda_i in {0,1} choosing which are supervised. This
#: is the lambda configuration, by name.
#:
#: Default is (past, current) -- lambda = (1, 1, 0). Fig. 2(b) of the paper shows
#: the future reconstruction is by far the blurriest of the three ("highest MAE
#: loss"), which is what you would expect: the drone cannot see what is 20 steps
#: ahead, so that segment is partly unlearnable and mostly adds gradient noise.
#: Set `--aux-segments past,current,future` to reproduce the full three-way loss.
AUX_SEGMENTS = ("past", "current")

#: Offset in policy steps of each segment relative to t.
AUX_OFFSETS = {"past": -AUX_T, "current": 0, "future": AUX_T}

BETA_KL = 1e-4
LAMBDA_SEG = 1.0

# --------------------------------------------------------------------------
# Curriculum
# --------------------------------------------------------------------------

SUCCESS_WINDOW = 200         # episodes in the rolling success estimate
SUCCESS_THRESHOLD = 0.8      # advance a stage above this
RESAMPLE_EVERY_ROLLOUTS = 10 # resample the layout this often within a stage
MAX_DWELL_ROLLOUTS = 400     # force-advance if a stage stalls
