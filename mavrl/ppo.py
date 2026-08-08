#!/usr/bin/env python3
"""Recurrent PPO with truncated BPTT.

Written rather than taken from sb3-contrib because RecurrentPPO sizes its state
buffers from `lstm.hidden_size` and feeds that same number into the MLP
extractor. The attention backbone needs a 1024-wide state (a 16x64 ring buffer
plus 4x256 memory tokens) but a 256-wide output, and those cannot both be
`hidden_size` without patching SB3 internals. ALD needs a custom two-network
loop regardless, so one loop serves both.

Minibatching is over **envs, not timesteps**: each minibatch takes a subset of
environments and their full n_steps of history, so BPTT runs over an unbroken
sequence and the recurrent state stays valid. Episode boundaries inside a
segment are handled by resetting state mid-sequence (see
MavrlActorCritic.sequence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from mavrl.policy import MavrlActorCritic, obs_to_tensors


@dataclass
class PPOConfig:
    n_steps: int = 128           # per env, per rollout
    n_epochs: int = 4
    envs_per_batch: int = 4      # minibatch size, in envs
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    target_kl: Optional[float] = 0.03
    normalize_advantage: bool = True
    #: Running return normalization, the cheap half of VecNormalize. Matters
    #: here because the curriculum changes episode length and gate count, so raw
    #: return scale drifts between stages even with a constant per-course total.
    normalize_returns: bool = True


class RunningNorm:
    """Welford running mean/variance of the discounted return."""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self._ret = None

    def update(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        if self._ret is None:
            self._ret = np.zeros(rewards.shape[-1], dtype=np.float64)
        for r, d in zip(rewards, dones):
            self._ret = self._ret * self.gamma + r
            self._ret[d.astype(bool)] = 0.0
            batch = self._ret
            bm, bv, bc = batch.mean(), batch.var(), batch.size
            delta = bm - self.mean
            tot = self.count + bc
            self.mean += delta * bc / tot
            m_a = self.var * self.count
            m_b = bv * bc
            self.var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
            self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + 1e-8)


class RolloutBuffer:
    """(T, N, ...) storage for one rollout, plus the state each env started in."""

    def __init__(self, n_steps: int, n_envs: int, obs_shape, n_proprio: int,
                 n_actions: int, device, state_batch_dim: int = 0):
        self.n_steps, self.n_envs, self.device = n_steps, n_envs, device
        # Which axis of a state tensor indexes the environment. Declared by the
        # memory module, never inferred -- see MavrlActorCritic._reset_state.
        self.state_batch_dim = state_batch_dim
        self.images = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.uint8)
        self.proprio = np.zeros((n_steps, n_envs, n_proprio), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, n_actions), dtype=np.float32)
        self.logprobs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.ep_starts = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.initial_state = None
        self.ptr = 0

    def reset(self, initial_state) -> None:
        self.ptr = 0
        self.initial_state = tuple(s.detach().clone() for s in initial_state)

    def add(self, image, proprio, action, logprob, value, reward, done, ep_start):
        i = self.ptr
        self.images[i] = image
        self.proprio[i] = proprio
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = done
        self.ep_starts[i] = ep_start
        self.ptr += 1

    def compute_returns(self, last_values: np.ndarray, last_dones: np.ndarray,
                        gamma: float, gae_lambda: float, ret_std: float = 1.0):
        adv = np.zeros_like(self.rewards)
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        rewards = self.rewards / ret_std
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_nonterminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_nonterminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = (rewards[t] + gamma * next_values * next_nonterminal
                     - self.values[t])
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        self.advantages = adv
        self.returns = adv + self.values

    def env_batch(self, idx: np.ndarray):
        """(B, T, ...) tensors for a subset of envs -- time axis intact."""
        def swap(a):
            return torch.as_tensor(a[:, idx], device=self.device).transpose(0, 1)
        return {
            "images": swap(self.images),
            "proprio": swap(self.proprio),
            "actions": swap(self.actions),
            "logprobs": swap(self.logprobs),
            "advantages": swap(self.advantages),
            "returns": swap(self.returns),
            "ep_starts": swap(self.ep_starts),
            "state": tuple(
                s[:, idx] if self.state_batch_dim == 1 else s[idx]
                for s in self.initial_state),
        }


class RecurrentPPO:
    def __init__(self, policy: MavrlActorCritic, venv, cfg: PPOConfig,
                 device: str = "cuda", aux_loss_fn: Optional[Callable] = None):
        self.policy = policy.to(device)
        self.venv = venv
        self.cfg = cfg
        self.device = torch.device(device)
        self.aux_loss_fn = aux_loss_fn

        self.n_envs = venv.num_envs
        img_space = venv.observation_space["image"]
        # Stored NHWC as the env produces it; permuted to NCHW at use.
        self.obs_shape = img_space.shape
        self.buffer = RolloutBuffer(
            cfg.n_steps, self.n_envs, self.obs_shape,
            venv.observation_space["proprio"].shape[0],
            venv.action_space.shape[0], self.device,
            state_batch_dim=self.policy.state_batch_dim)

        self.optimizer = torch.optim.Adam(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=cfg.learning_rate)
        self.ret_norm = RunningNorm(cfg.gamma) if cfg.normalize_returns else None

        self._obs = venv.reset()
        self._state = self.policy.initial_state(self.n_envs, self.device)
        self._ep_start = np.ones(self.n_envs, dtype=np.float32)
        self.num_timesteps = 0
        self.episode_infos = []

    # -- rollout -------------------------------------------------------------

    @torch.no_grad()
    def collect(self) -> None:
        self.buffer.reset(self._state)
        for _ in range(self.cfg.n_steps):
            image, proprio = obs_to_tensors(self._obs, self.device)
            action, logp, value, self._state = self.policy.act(
                image, proprio, self._state)

            a = action.cpu().numpy()
            obs, reward, done, infos = self.venv.step(np.clip(a, -1.0, 1.0))

            self.buffer.add(
                self._obs["image"], self._obs["proprio"], a,
                logp.cpu().numpy(), value.cpu().numpy(),
                reward, done.astype(np.float32), self._ep_start)

            for i, d in enumerate(done):
                if d:
                    self.episode_infos.append(infos[i])
            self._ep_start = done.astype(np.float32)
            self._obs = obs
            self.num_timesteps += self.n_envs

            # Reset the recurrent state for envs that just finished, so the next
            # episode does not inherit the previous one's memory.
            if done.any():
                mask = torch.as_tensor(done.astype(bool), device=self.device)
                fresh = self.policy.initial_state(self.n_envs, self.device)
                self._state = self.policy._reset_state(self._state, mask, fresh)

        image, proprio = obs_to_tensors(self._obs, self.device)
        _, last_values, _ = self.policy.step(image, proprio, self._state)

        if self.ret_norm is not None:
            self.ret_norm.update(self.buffer.rewards, self.buffer.dones)
            std = self.ret_norm.std
        else:
            std = 1.0
        self.buffer.compute_returns(
            last_values.cpu().numpy(), self._ep_start,
            self.cfg.gamma, self.cfg.gae_lambda, std)

    # -- update --------------------------------------------------------------

    def update(self) -> dict:
        cfg = self.cfg
        stats = {"pg": [], "vf": [], "ent": [], "kl": [], "aux": []}
        stop = False

        for _ in range(cfg.n_epochs):
            order = np.random.permutation(self.n_envs)
            for start in range(0, self.n_envs, cfg.envs_per_batch):
                idx = order[start:start + cfg.envs_per_batch]
                if len(idx) == 0:
                    continue
                b = self.buffer.env_batch(idx)

                images = b["images"].permute(0, 1, 4, 2, 3).contiguous()
                logp, values, entropy, _, z_seq = self.policy.evaluate(
                    images, b["proprio"], b["state"], b["actions"],
                    b["ep_starts"])

                adv = b["advantages"]
                if cfg.normalize_advantage:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                ratio = torch.exp(logp - b["logprobs"])
                pg = -torch.min(
                    adv * ratio,
                    adv * torch.clamp(ratio, 1 - cfg.clip_range,
                                      1 + cfg.clip_range)).mean()
                vf = nn.functional.mse_loss(values, b["returns"])
                ent = entropy.mean()
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent

                aux_val = 0.0
                if self.aux_loss_fn is not None:
                    aux = self.aux_loss_fn(self.policy, b, z_seq)
                    loss = loss + aux
                    aux_val = float(aux.detach())

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1) - (logp - b["logprobs"])).mean())
                stats["pg"].append(float(pg.detach()))
                stats["vf"].append(float(vf.detach()))
                stats["ent"].append(float(ent.detach()))
                stats["kl"].append(approx_kl)
                stats["aux"].append(aux_val)

                if cfg.target_kl is not None and approx_kl > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break

        return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}

    # -- driver --------------------------------------------------------------

    def learn(self, total_timesteps: int,
              on_rollout_end: Optional[Callable] = None) -> "RecurrentPPO":
        rollout = 0
        while self.num_timesteps < total_timesteps:
            self.collect()
            stats = self.update()
            rollout += 1
            if on_rollout_end is not None:
                on_rollout_end(self, rollout, stats)
        return self

    # -- checkpointing -------------------------------------------------------

    def save(self, path) -> None:
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "memory_type": self.policy.memory_type,
            "aux_segments": list(self.policy.aux.segments),
            "num_timesteps": self.num_timesteps,
        }, path)

    def load(self, path, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"], strict=strict)
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                pass          # param groups changed (e.g. encoder unfrozen)
        self.num_timesteps = ckpt.get("num_timesteps", 0)
