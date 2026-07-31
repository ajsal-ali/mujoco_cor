#!/usr/bin/env python3
"""Onboard RGB + depth window, run in its own process.

Replaces `imav_play._onboard_window`. It lives in a separate process because an
interactive `mujoco.viewer` (GLFW) and a matplotlib window cannot coexist in one
process -- they deadlock on the UI thread.

Protocol: the parent pushes `(rgb_uint8_HxWx3, depth_m_HxW)` tuples onto the
queue and a single `None` to shut down.
"""

from __future__ import annotations

import queue


def onboard_window(frame_q, depth_max: float = 8.0) -> None:
    """Consume frames from `frame_q` and display them until told to stop.

    Silently returns if no GUI backend is available, so headless callers that
    forget to pass show_onboard=False still run rather than crashing.
    """
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    fig, (ax_rgb, ax_dep) = plt.subplots(1, 2, figsize=(7, 3.2))
    fig.canvas.manager.set_window_title("onboard RGB-D")
    im_rgb = im_dep = None
    ax_rgb.set_title("RGB"); ax_dep.set_title("depth (m)")
    for ax in (ax_rgb, ax_dep):
        ax.set_xticks([]); ax.set_yticks([])
    plt.ion()
    plt.show(block=False)

    try:
        while True:
            try:
                item = frame_q.get(timeout=0.5)
            except queue.Empty:
                # Keep the UI responsive between frames.
                try:
                    fig.canvas.flush_events()
                except Exception:
                    return
                continue

            if item is None:
                break

            rgb, depth = item
            if im_rgb is None:
                im_rgb = ax_rgb.imshow(rgb)
                im_dep = ax_dep.imshow(depth, cmap="viridis",
                                       vmin=0.0, vmax=depth_max)
                fig.colorbar(im_dep, ax=ax_dep, fraction=0.046)
                fig.tight_layout()
            else:
                im_rgb.set_data(rgb)
                im_dep.set_data(depth)

            try:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except Exception:
                break
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass


# Backwards-compatible alias: this was imav_play._onboard_window.
_onboard_window = onboard_window
