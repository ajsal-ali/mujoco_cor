#!/usr/bin/env python3
"""Lightweight CNN feature extractor for the WindowAviary Dict observation.

Deliberately small so it can run on onboard compute, but deep enough to read the
84x84 RGB-D image. Image branch = 3 conv layers (~250k params total); proprio
branch = a tiny MLP; the two are concatenated and fused.

Used by rl/train_window.py (PPO MultiInputPolicy) and rl/bc_pretrain.py.
"""

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LightCNNExtractor(BaseFeaturesExtractor):
    """Early-fusion RGB-D extractor. SB3 wraps Dict image obs in VecTransposeImage,
    so the image space arrives channels-first (C,H,W). `forward` also accepts raw
    channels-last (H,W,C) tensors (e.g. from BC) and transposes them.

    NOTE: to switch to two-stream fusion later, split `self.cnn` into an rgb CNN
    (channels 0:3) and a depth CNN (channel 3:4) and concat their flattened
    features before `img_head` — everything else stays the same.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        shape = observation_space["image"].shape            # channels-first (C,H,W)
        self.n_channels = int(min(shape))                   # 4 (RGB-D); H,W >> C
        c = self.n_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # build a dummy in the space's own layout to size the linear layer
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, *shape)).shape[1]
        # LayerNorm after each branch: keeps activations well-scaled so the net
        # can't collapse to constant (image-ignoring) mean-prediction. RL-safe
        # (no running stats, unlike BatchNorm). Fixes the BC mean-plateau trap.
        self.img_head = nn.Sequential(nn.Linear(n_flat, 128), nn.LayerNorm(128), nn.ReLU())

        p = observation_space["proprio"].shape[0]           # 9
        self.prop = nn.Sequential(nn.Linear(p, 64), nn.LayerNorm(64), nn.ReLU())
        self.fuse = nn.Sequential(
            nn.Linear(128 + 64, features_dim), nn.LayerNorm(features_dim), nn.ReLU())

    def forward(self, obs):
        img = obs["image"].float()
        if img.shape[1] != self.n_channels:                 # (B,H,W,C) -> (B,C,H,W)
            img = img.permute(0, 3, 1, 2)
        img = img / 255.0
        f_img = self.img_head(self.cnn(img))
        f_prop = self.prop(obs["proprio"].float())
        return self.fuse(th.cat([f_img, f_prop], dim=1))


# policy kwargs to hand PPO / the BC model.
# normalize_images=False: the extractor does its own /255, so SB3 must NOT also
# divide by 255 (that would double-normalize the image to ~0).
# log_std_init=-2.0: start with SMALL action noise (std ~0.14). BC only trains the
# action mean; without this the default std=1.0 makes PPO rollouts near-random,
# which both looks jerky and destroys the BC policy during fine-tuning.
POLICY_KWARGS = dict(
    features_extractor_class=LightCNNExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[128, 128],
    normalize_images=False,
    log_std_init=-2.0,
)


if __name__ == "__main__":
    import numpy as np
    # channels-first, as SB3 hands it after VecTransposeImage
    space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (4, 96, 96), np.uint8),
        "proprio": gym.spaces.Box(-np.inf, np.inf, (9,), np.float32),
    })
    ext = LightCNNExtractor(space)
    n_params = sum(p.numel() for p in ext.parameters())
    dummy = {
        "image": th.zeros(2, 4, 96, 96),
        "proprio": th.zeros(2, 9),
    }
    out = ext(dummy)
    print(f"extractor params: {n_params:,}  output: {tuple(out.shape)}")
