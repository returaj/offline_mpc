# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import annotations

import torch
import numpy as np


def get_shape_from_obs_space(obs_space):
    return obs_space.shape


def get_shape_from_act_space(act_space):
    return act_space.shape[0]


class VectorizedOnPolicyBuffer:
    """
    A buffer for storing vectorized on-policy data for reinforcement learning.

    Args:
        obs_space (gymnasium.Space): The observation space.
        act_space (gymnasium.Space): The action space.
        size (int): The maximum size of the buffer.
        gamma (float, optional): The discount factor for rewards. Defaults to 0.99.
        lam (float, optional): The lambda parameter for GAE computation. Defaults to 0.95.
        lam_c (float, optional): The lambda parameter for cost GAE computation. Defaults to 0.95.
        standardized_adv_r (bool, optional): Whether to standardize advantage rewards. Defaults to True.
        standardized_adv_c (bool, optional): Whether to standardize advantage costs. Defaults to True.
        device (torch.device, optional): The device to store tensors on. Defaults to "cpu".
        num_envs (int, optional): The number of parallel environments. Defaults to 1.
    """

    def __init__(
        self,
        obs_space,
        act_space,
        size: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        lam_c: float = 0.95,
        standardized_adv_r: bool = True,
        standardized_adv_c: bool = True,
        device: torch.device = "cpu",
        num_envs: int = 1,
    ) -> None:
        self.buffers: list[dict[str, torch.tensor]] = [
            {
                "obs": torch.zeros(
                    (size, *obs_space.shape), dtype=torch.float32, device=device
                ),
                "act": torch.zeros(
                    (size, *act_space.shape), dtype=torch.float32, device=device
                ),
                "reward": torch.zeros(size, dtype=torch.float32, device=device),
                "cost": torch.zeros(size, dtype=torch.float32, device=device),
                "done": torch.zeros(size, dtype=torch.float32, device=device),
                "value_r": torch.zeros(size, dtype=torch.float32, device=device),
                "value_c": torch.zeros(size, dtype=torch.float32, device=device),
                "adv_r": torch.zeros(size, dtype=torch.float32, device=device),
                "adv_c": torch.zeros(size, dtype=torch.float32, device=device),
                "target_value_r": torch.zeros(size, dtype=torch.float32, device=device),
                "target_value_c": torch.zeros(size, dtype=torch.float32, device=device),
                "log_prob": torch.zeros(size, dtype=torch.float32, device=device),
            }
            for _ in range(num_envs)
        ]
        self._gamma = gamma
        self._lam = lam
        self._lam_c = lam_c
        self._standardized_adv_r = standardized_adv_r
        self._standardized_adv_c = standardized_adv_c
        self.ptr_list = [0] * num_envs
        self.path_start_idx_list = [0] * num_envs
        self._device = device
        self.num_envs = num_envs

    def store(self, **data: torch.Tensor) -> None:
        """
        Store vectorized data into the buffer.

        Args:
            **data: Keyword arguments specifying data tensors to be stored.
        """
        for i, buffer in enumerate(self.buffers):
            assert self.ptr_list[i] < buffer["obs"].shape[0], "Buffer overflow"
            for key, value in data.items():
                buffer[key][self.ptr_list[i]] = value[i]
            self.ptr_list[i] += 1

    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_c: torch.Tensor | None = None,
        idx: int = 0,
    ) -> None:
        """
        Finalize the trajectory path and compute advantages and value targets.

        Args:
            last_value_r (torch.Tensor, optional): The last value estimate for rewards. Defaults to None.
            last_value_c (torch.Tensor, optional): The last value estimate for costs. Defaults to None.
            idx (int, optional): Index of the environment. Defaults to 0.
        """
        if last_value_r is None:
            last_value_r = torch.zeros(1, device=self._device)
        if last_value_c is None:
            last_value_c = torch.zeros(1, device=self._device)
        path_slice = slice(self.path_start_idx_list[idx], self.ptr_list[idx])
        last_value_r = last_value_r.to(self._device)
        last_value_c = last_value_c.to(self._device)
        rewards = torch.cat([self.buffers[idx]["reward"][path_slice], last_value_r])
        costs = torch.cat([self.buffers[idx]["cost"][path_slice], last_value_c])
        values_r = torch.cat([self.buffers[idx]["value_r"][path_slice], last_value_r])
        values_c = torch.cat([self.buffers[idx]["value_c"][path_slice], last_value_c])

        adv_r, target_value_r = calculate_adv_and_value_targets(
            values_r,
            rewards,
            lam=self._lam,
            gamma=self._gamma,
        )
        adv_c, target_value_c = calculate_adv_and_value_targets(
            values_c,
            costs,
            lam=self._lam_c,
            gamma=self._gamma,
        )
        self.buffers[idx]["adv_r"][path_slice] = adv_r
        self.buffers[idx]["adv_c"][path_slice] = adv_c
        self.buffers[idx]["target_value_r"][path_slice] = target_value_r
        self.buffers[idx]["target_value_c"][path_slice] = target_value_c

        self.path_start_idx_list[idx] = self.ptr_list[idx]

    def get(self) -> dict[str, torch.Tensor]:
        """
        Retrieve collected data from the buffer.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing collected data tensors.
        """
        data_pre = {k: [v] for k, v in self.buffers[0].items()}
        for buffer in self.buffers[1:]:
            for k, v in buffer.items():
                data_pre[k].append(v)
        data = {k: torch.cat(v, dim=0) for k, v in data_pre.items()}
        adv_mean = data["adv_r"].mean()
        adv_std = data["adv_r"].std()
        cadv_mean = data["adv_c"].mean()
        if self._standardized_adv_r:
            data["adv_r"] = (data["adv_r"] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data["adv_c"] = data["adv_c"] - cadv_mean
        self.ptr_list = [0] * self.num_envs
        self.path_start_idx_list = [0] * self.num_envs

        return data


def discount_cumsum(vector_x: torch.Tensor, discount: float) -> torch.Tensor:
    """
    Compute the discounted cumulative sum of a tensor along its first dimension.

    This function computes the discounted cumulative sum of the input tensor `vector_x` along
    its first dimension. The discount factor `discount` is applied to compute the weighted sum
    of future values. The resulting tensor has the same shape as the input tensor.

    Args:
        vector_x (torch.Tensor): Input tensor with shape `(length, ...)`.
        discount (float): Discount factor for future values.

    Returns:
        torch.Tensor: Tensor containing the discounted cumulative sum of `vector_x`.
    """
    length = vector_x.shape[0]
    vector_x = vector_x.type(torch.float64)
    cumsum = vector_x[-1]
    for idx in reversed(range(length - 1)):
        cumsum = vector_x[idx] + discount * cumsum
        vector_x[idx] = cumsum
    return vector_x


def calculate_adv_and_value_targets(
    values: torch.Tensor,
    rewards: torch.Tensor,
    lam: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    adv = discount_cumsum(deltas, gamma * lam)
    target_value = adv + values[:-1]
    return adv, target_value


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
