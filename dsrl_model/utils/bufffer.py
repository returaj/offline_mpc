from __future__ import annotations

import torch
import numpy as np


class PriorityBuffer:
    """
    TD-MPC code modified.

    Storage and sampling functionality for training TD-MPC / TOLD.
    The replay buffer is stored in GPU memory when training from state.
    Uses prioritized experience replay by default.
    """

    def __init__(
        self, obs_dim, act_dim, data_size, horizon, batch_size, device, ep_len=1000
    ):
        self.horizon = horizon
        self.batch_size = batch_size
        self.capacity = data_size
        self.ep_len = ep_len
        self.device = torch.device(device)
        dtype = torch.float32
        self._obs = torch.empty(
            (self.capacity + 1, obs_dim), dtype=dtype, device=self.device
        )
        self._last_obs = torch.empty(
            (self.capacity // ep_len, obs_dim), dtype=dtype, device=self.device
        )
        self._action = torch.empty(
            (self.capacity, act_dim), dtype=dtype, device=self.device
        )
        self._cost = torch.empty((self.capacity,), dtype=dtype, device=self.device)
        self._priorities = torch.ones(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self._eps = 1e-6
        self._full = False
        self.idx = 0

    def add(self, obs: np.array, act: np.array, cost: np.array):
        self._obs[self.idx : self.idx + self.ep_len] = obs
        self._last_obs[self.idx // self.ep_len] = obs[-1]
        self._action[self.idx : self.idx + self.ep_len] = act
        self._cost[self.idx : self.idx + self.ep_len] = cost
        max_priority = 1.0
        mask = torch.arange(self.ep_len) >= self.ep_len - self.horizon
        new_priorities = torch.full((self.ep_len,), max_priority, device=self.device)
        new_priorities[mask] = 0
        self._priorities[self.idx : self.idx + self.ep_len] = new_priorities
        self.idx = (self.idx + self.ep_len) % self.capacity
        self._full = self._full or self.idx == 0

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.to(self.device) + self._eps

    def _get_obs(self, arr, idxs):
        return arr[idxs]

    def sample(self):
        probs = (
            self._priorities if self._full else self._priorities[: self.idx]
        ) ** 0.6  # per_alpha = 0.6, value from TD-MPC implementation
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(
            np.random.choice(
                total,
                self.batch_size,
                p=probs.cpu().numpy(),
                replace=not self._full,
            )
        ).to(self.device)
        weights = (total * probs[idxs]) ** (-0.4)  # per_beta = 0.4, value from TD-MPC
        weights /= weights.max()

        obs = self._get_obs(self._obs, idxs)
        next_obs_shape = self._last_obs.shape[1:]
        next_obs = torch.empty(
            (self.horizon + 1, self.batch_size, *next_obs_shape),
            dtype=obs.dtype,
            device=obs.device,
        )
        action = torch.empty(
            (self.horizon + 1, self.batch_size, *self._action.shape[1:]),
            dtype=torch.float32,
            device=self.device,
        )
        cost = torch.empty(
            (self.horizon + 1, self.batch_size),
            dtype=torch.float32,
            device=self.device,
        )
        for t in range(self.horizon + 1):
            _idxs = idxs + t
            next_obs[t] = self._get_obs(self._obs, _idxs + 1)
            action[t] = self._action[_idxs]
            cost[t] = self._cost[_idxs]

        mask = (_idxs + 1) % self.ep_len == 0
        next_obs[-1, mask] = (
            self._last_obs[_idxs[mask] // self.ep_len].to(obs.device).float()
        )
        if not action.is_cuda:
            action, cost, idxs, weights = (
                action.to(obs.device),
                cost.to(obs.device),
                idxs.to(obs.device),
                weights.to(obs.device),
            )

        return obs, next_obs, action, cost, idxs, weights


class OnPolicyBuffer:
    def __init__(
        self,
        obs_dim,
        act_dim,
        neg_data_size,
        union_data_size,
        horizon,
        batch_size,
        device,
        ep_len=1000,
    ):
        self.horizon = horizon
        self.batch_size = batch_size
        self.neg_capacity = neg_data_size
        self.union_capacity = union_data_size
        self.ep_len = ep_len
        self.device = torch.device(device)
        dtype = torch.float32
        self._neg_obs = torch.empty(
            (self.neg_capacity + 1, obs_dim), dtype=dtype, device=self.device
        )
        self._neg_act = torch.empty(
            (self.neg_capacity, act_dim), dtype=dtype, device=self.device
        )
        self._union_obs = torch.empty(
            (self.union_capacity + 1, obs_dim), dtype=dtype, device=self.device
        )
        self._union_act = torch.empty(
            (self.union_capacity, act_dim), dtype=dtype, device=self.device
        )
        self._neg_priorities = torch.ones(
            (self.neg_capacity,), dtype=torch.float32, device=self.device
        )
        self._union_priorities = torch.ones(
            (self.union_capacity,), dtype=torch.float32, device=self.device
        )
        self._eps = 1e-6
        self._neg_idx = 0
        self._union_idx = 0

    def add(self, obs, act, done=None, is_negative=False):
        max_priority = 1.0
        done_sum = np.sum(done) or 1.0
        true_ep_len = self.ep_len - done_sum + 1
        mask = torch.arange(self.ep_len) >= true_ep_len - self.horizon
        new_priorities = torch.full((self.ep_len,), max_priority, device=self.device)
        new_priorities[mask] = 0.0

        if is_negative:
            self._neg_obs[self._neg_idx : self._neg_idx + self.ep_len] = obs
            self._neg_act[self._neg_idx : self._neg_idx + self.ep_len] = act
            self._neg_priorities[self._neg_idx : self._neg_idx + self.ep_len] = (
                new_priorities
            )
            self._neg_idx = (self._neg_idx + self.ep_len) % self.neg_capacity
        else:
            self._union_obs[self._union_idx : self._union_idx + self.ep_len] = obs
            self._union_act[self._union_idx : self._union_idx + self.ep_len] = act
            self._union_priorities[self._union_idx : self._union_idx + self.ep_len] = (
                new_priorities
            )
            self._union_idx = (self._union_idx + self.ep_len) % self.union_capacity

    def sample(self):
        batch_size = self.batch_size
        steps_per_epoch = self.union_capacity // batch_size

        union_probs = self._union_priorities
        union_probs /= union_probs.sum()
        union_total = len(union_probs)
        union_idxs = torch.from_numpy(
            np.random.choice(
                union_total,
                (steps_per_epoch, batch_size),
                p=union_probs.cpu().numpy(),
                replace=True,
            )
        ).to(self.device)

        neg_probs = self._neg_priorities
        neg_probs /= neg_probs.sum()
        neg_total = len(neg_probs)
        neg_idxs = torch.from_numpy(
            np.random.choice(
                neg_total,
                (steps_per_epoch, batch_size),
                p=neg_probs.cpu().numpy(),
                replace=True,
            )
        ).to(self.device)

        # tensor([12566703, 12468216, 12362093]) tensor([26191119, 25599700, 26147546])

        for n_idx, u_idx in zip(neg_idxs, union_idxs):
            h_neg_obs = torch.empty(
                (self.horizon, batch_size, *self._neg_obs.shape[1:]),
                dtype=torch.float32,
                device=self.device,
            )
            h_neg_act = torch.empty(
                (self.horizon, batch_size, *self._neg_act.shape[1:]),
                dtype=torch.float32,
                device=self.device,
            )
            h_union_obs = torch.empty_like(
                h_neg_obs, dtype=torch.float32, device=self.device
            )
            h_union_act = torch.empty_like(
                h_neg_act, dtype=torch.float32, device=self.device
            )

            for t in range(self.horizon):
                _n_idx, _u_idx = n_idx + t, u_idx + t
                h_neg_obs[t] = self._neg_obs[_n_idx]
                h_neg_act[t] = self._neg_act[_n_idx]
                h_union_obs[t] = self._union_obs[_u_idx]
                h_union_act[t] = self._union_act[_u_idx]

            # Horizon X Batch X obs/act_dim

            yield (h_neg_obs, h_neg_act, h_union_obs, h_union_act)
