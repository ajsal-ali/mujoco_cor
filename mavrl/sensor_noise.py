#!/usr/bin/env python3
"""Sensor noise for the encoder input.

MuJoCo hands back perfect ray-traced depth. MAVRL never trained on that:
AvoidBench runs SGM stereo matching precisely "to replicate realistic depth
errors, reducing the gap between simulation and reality". Perfect depth lets the
VAE encode sub-pixel-crisp edges no real sensor produces, and the policy then
leans on them.

Applied to the **encoder input only** -- reconstruction targets stay clean, i.e.
this is a denoising VAE. Corrupting both sides instead would spend latent
capacity modelling the noise.

This is *sensor* noise, not domain randomization: the world looks identical
frame to frame, only the measurement of it is corrupted.

Note the depth channel is already quantized to 8 bits over DEPTH_MAX = 12 m,
which is ~4.7 cm per level (~1.4 cm RMS) -- comparable to a RealSense at short
range. So some noise exists even with this disabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mavrl.config import DEPTH_MAX


@dataclass
class NoiseConfig:
    """All magnitudes; scale to zero with `NoiseConfig.disabled()`."""

    #: Range-dependent depth sigma, sigma(d) = depth_k * d**2.
    #: 0.001 gives ~4 mm at 2 m and ~10 cm at 10 m, matching a D435i's
    #: roughly-1%-of-range-growing-quadratically behaviour.
    depth_k: float = 0.001
    #: Fraction of depth-discontinuity pixels invalidated. Edge dropout is the
    #: dominant real stereo failure -- matching surfaces across an occlusion
    #: boundary is exactly where SGM gives up.
    edge_dropout: float = 0.02
    #: Depth gradient (metres) above which a pixel counts as an edge.
    edge_threshold: float = 0.25
    #: Fraction of pixels invalidated at random, anywhere.
    invalid_frac: float = 0.005
    #: Value written into invalidated pixels (max range reads as "no return").
    invalid_value: float = DEPTH_MAX
    #: Additive RGB sigma, in [0,1] units.
    rgb_sigma: float = 2.0 / 255.0
    #: Per-frame gamma jitter, multiplicative +/- this fraction.
    gamma_jit: float = 0.05

    @classmethod
    def disabled(cls) -> "NoiseConfig":
        return cls(depth_k=0.0, edge_dropout=0.0, invalid_frac=0.0,
                   rgb_sigma=0.0, gamma_jit=0.0)

    @property
    def is_enabled(self) -> bool:
        return any((self.depth_k, self.edge_dropout, self.invalid_frac,
                    self.rgb_sigma, self.gamma_jit))

    def scaled(self, factor: float) -> "NoiseConfig":
        """All magnitudes scaled -- for `--sensor-noise 0.5` style ablations."""
        return NoiseConfig(
            depth_k=self.depth_k * factor,
            edge_dropout=self.edge_dropout * factor,
            edge_threshold=self.edge_threshold,
            invalid_frac=self.invalid_frac * factor,
            invalid_value=self.invalid_value,
            rgb_sigma=self.rgb_sigma * factor,
            gamma_jit=self.gamma_jit * factor,
        )


def _edge_mask(depth_m: np.ndarray, threshold: float) -> np.ndarray:
    """Pixels sitting on a depth discontinuity."""
    gy = np.zeros_like(depth_m)
    gx = np.zeros_like(depth_m)
    gy[:-1, :] = np.abs(np.diff(depth_m, axis=0))
    gx[:, :-1] = np.abs(np.diff(depth_m, axis=1))
    return np.maximum(gx, gy) > threshold


def corrupt_depth(depth_m: np.ndarray, cfg: NoiseConfig,
                  rng: np.random.Generator) -> np.ndarray:
    """Metric depth -> noisy metric depth."""
    out = depth_m.astype(np.float32, copy=True)

    if cfg.depth_k > 0.0:
        sigma = cfg.depth_k * np.square(out)
        out = out + rng.normal(0.0, 1.0, out.shape).astype(np.float32) * sigma

    if cfg.edge_dropout > 0.0:
        edges = _edge_mask(depth_m, cfg.edge_threshold)
        drop = edges & (rng.random(out.shape) < cfg.edge_dropout)
        out[drop] = cfg.invalid_value

    if cfg.invalid_frac > 0.0:
        drop = rng.random(out.shape) < cfg.invalid_frac
        out[drop] = cfg.invalid_value

    return np.clip(out, 0.0, cfg.invalid_value)


def corrupt_rgb(rgb: np.ndarray, cfg: NoiseConfig,
                rng: np.random.Generator) -> np.ndarray:
    """uint8 RGB -> noisy uint8 RGB."""
    if not (cfg.rgb_sigma > 0.0 or cfg.gamma_jit > 0.0):
        return rgb
    x = rgb.astype(np.float32) / 255.0

    if cfg.gamma_jit > 0.0:
        gamma = 1.0 + rng.uniform(-cfg.gamma_jit, cfg.gamma_jit)
        x = np.power(np.clip(x, 0.0, 1.0), gamma)

    if cfg.rgb_sigma > 0.0:
        x = x + rng.normal(0.0, cfg.rgb_sigma, x.shape).astype(np.float32)

    return (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def corrupt(rgb: np.ndarray, depth_m: np.ndarray, cfg: NoiseConfig,
            rng: np.random.Generator) -> tuple:
    """Both channels at once. Returns (rgb, depth_m)."""
    if not cfg.is_enabled:
        return rgb, depth_m
    return corrupt_rgb(rgb, cfg, rng), corrupt_depth(depth_m, cfg, rng)
