import functools

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def update_arr_jit(arr, idxs, val):
    return arr.at[idxs].set(val)


class OnPolicyBuffer:
    def __init__(
        self,
        rngs,
        obs_dim,
        act_dim,
        neg_data_size,
        union_data_size,
        horizon,
        batch_size,
        ep_len=1000,
        priorities_alpha=1.0,
    ):
        self.rngs = rngs
        self.horizon = horizon
        self.batch_size = batch_size
        self.neg_capacity = neg_data_size
        self.union_capacity = union_data_size
        self.ep_len = ep_len
        self._priorities_alpha = priorities_alpha

        self.dtype = np.float32
        self._neg_obs = np.empty((self.neg_capacity + 1, obs_dim), dtype=self.dtype)
        self._neg_act = np.empty((self.neg_capacity, act_dim), dtype=self.dtype)
        self._neg_cost = np.empty((self.neg_capacity,), dtype=self.dtype)
        self._neg_priorities = np.ones((self.neg_capacity,), dtype=self.dtype)
        self._union_obs = np.empty((self.union_capacity + 1, obs_dim), dtype=self.dtype)
        self._union_act = np.empty((self.union_capacity, act_dim), dtype=self.dtype)
        self._union_cost = np.empty((self.union_capacity,), dtype=self.dtype)
        self._union_priorities = np.ones((self.union_capacity,), dtype=self.dtype)

        self._eps = 1e-6
        self._neg_idx = 0
        self._union_idx = 0

    def _update(
        self,
        obs_store,
        act_store,
        priority_store,
        cost_store,
        idx,
        obs,
        act,
        priority,
        cost,
        capacity,
    ):
        obs_store[idx : idx + self.ep_len] = np.array(obs)
        act_store[idx : idx + self.ep_len] = np.array(act)
        priority_store[idx : idx + self.ep_len] = np.array(priority)
        cost_store[idx : idx + self.ep_len] = np.array(cost)
        return (idx + self.ep_len) % capacity

    def add(self, obs, act, done, cost=None, is_negative=False):
        assert cost is not None, "cost field cannot be none"
        max_priority = 1.0
        done_sum = np.sum(done) or 1.0
        true_ep_len = self.ep_len - done_sum + 1
        mask = np.arange(self.ep_len) >= true_ep_len - self.horizon
        new_priorities = np.full((self.ep_len,), max_priority)
        new_priorities[mask] = 0.0

        if is_negative:
            self._neg_idx = self._update(
                obs_store=self._neg_obs,
                act_store=self._neg_act,
                priority_store=self._neg_priorities,
                cost_store=self._neg_cost,
                idx=self._neg_idx,
                obs=obs,
                act=act,
                priority=new_priorities,
                cost=cost,
                capacity=self.neg_capacity,
            )
        else:
            self._union_idx = self._update(
                obs_store=self._union_obs,
                act_store=self._union_act,
                priority_store=self._union_priorities,
                cost_store=self._union_cost,
                idx=self._union_idx,
                obs=obs,
                act=act,
                priority=new_priorities,
                cost=cost,
                capacity=self.union_capacity,
            )

    def to_jax_ndarray(self):
        self.dtype = jnp.float32
        self._neg_obs = jnp.array(self._neg_obs, dtype=self.dtype)
        self._neg_act = jnp.array(self._neg_act, dtype=self.dtype)
        self._neg_cost = jnp.array(self._neg_cost, dtype=self.dtype)
        self._neg_priorities = jnp.array(self._neg_priorities, dtype=self.dtype)
        self._union_obs = jnp.array(self._union_obs, dtype=self.dtype)
        self._union_act = jnp.array(self._union_act, dtype=self.dtype)
        self._union_cost = jnp.array(self._union_cost, dtype=self.dtype)
        self._union_priorities = jnp.array(self._union_priorities, dtype=self.dtype)

    def update_priorities(self, idxs, priorities):
        priorities = jnp.array(priorities, dtype=self.dtype) + self._eps
        self._union_priorities = update_arr_jit(
            self._union_priorities, idxs, priorities
        )

    @functools.partial(jax.jit, static_argnums=0)
    def sample_batch(self, n_idx, u_idx):
        horizon_offsets = jnp.arange(self.horizon)
        n_h_idx = n_idx[..., None] + horizon_offsets
        u_h_idx = u_idx[..., None] + horizon_offsets

        h_neg_obs = self._neg_obs[n_h_idx]
        h_neg_act = self._neg_act[n_h_idx]
        h_neg_cost = self._neg_cost[n_h_idx]
        h_union_obs = self._union_obs[u_h_idx]
        h_union_act = self._union_act[u_h_idx]
        h_union_cost = self._union_cost[u_h_idx]

        return (
            h_neg_obs,
            h_neg_act,
            h_neg_cost,
            h_union_obs,
            h_union_act,
            h_union_cost,
            u_idx,
        )

    def sample(self):
        batch_size = self.batch_size
        steps_per_epoch = self.union_capacity // batch_size

        union_probs = self._union_priorities**self._priorities_alpha
        union_probs /= union_probs.sum()
        union_total = len(union_probs)
        union_idxs = jax.random.choice(
            key=self.rngs(),
            a=union_total,
            shape=(steps_per_epoch, batch_size),
            p=union_probs,
            replace=True,
        )

        neg_probs = self._neg_priorities
        neg_probs /= neg_probs.sum()
        neg_total = len(neg_probs)
        neg_idxs = jax.random.choice(
            key=self.rngs(),
            a=neg_total,
            shape=(steps_per_epoch, batch_size),
            p=neg_probs,
            replace=True,
        )

        for n_idx, u_idx in zip(neg_idxs, union_idxs):
            yield self.sample_batch(n_idx, u_idx)


class SafeCLBuffer(OnPolicyBuffer):
    def __init__(
        self,
        rngs,
        obs_dim,
        act_dim,
        neg_data_size,
        union_data_size,
        horizon,
        batch_size,
        ep_len=1000,
        priorities_alpha=1,
    ):
        super().__init__(
            rngs,
            obs_dim,
            act_dim,
            neg_data_size,
            union_data_size,
            horizon,
            batch_size,
            ep_len,
            priorities_alpha,
        )
        self._union_labels = -np.ones((self.union_capacity,), dtype=self.dtype)

    def to_jax_ndarray(self):
        super().to_jax_ndarray()
        self._union_labels = jnp.array(self._union_labels, dtype=self.dtype)

    def update_labels(self, idxs, labels):
        labels = jnp.array(labels, dtype=self.dtype)
        self._union_labels = update_arr_jit(self._union_labels, idxs, labels)

    @functools.partial(jax.jit, static_argnums=0)
    def sample_batch(self, n_idx, u_idx):
        batch = super().sample_batch(n_idx, u_idx)
        label = self._union_labels[u_idx]
        return (*batch, label)
