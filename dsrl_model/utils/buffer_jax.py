import functools

import jax
import jax.numpy as jnp
import numpy as np


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

    @functools.partial(jax.jit, static_argnums=0)
    def update_priorities(self, idxs, priorities):
        priorities = jnp.array(priorities, dtype=self.dtype) + self._eps
        self._union_priorities = self._union_priorities.at[idxs].set(priorities)

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
            h_neg_obs, h_neg_act, h_neg_cost = [], [], []
            h_union_obs, h_union_act, h_union_cost = [], [], []

            for t in range(self.horizon):
                _nidx, _uidx = n_idx + t, u_idx + t
                h_neg_obs.append(self._neg_obs[_nidx])
                h_neg_act.append(self._neg_act[_nidx])
                h_neg_cost.append(self._neg_cost[_nidx])
                h_union_obs.append(self._union_obs[_uidx])
                h_union_act.append(self._union_act[_uidx])
                h_union_cost.append(self._union_cost[_uidx])

            yield (
                jnp.array(h_neg_obs),
                jnp.array(h_neg_act),
                jnp.array(h_neg_cost),
                jnp.array(h_union_obs),
                jnp.array(h_union_act),
                jnp.array(h_union_cost),
                jnp.array(u_idx),
            )


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

    @functools.partial(jax.jit, static_argnums=0)
    def update_labels(self, idxs, labels):
        labels = jnp.array(labels, dtype=self.dtype)
        self._union_labels = self._union_labels.at[idxs].set(labels)

    def sample(self):
        for buffer in super().sample():
            uidx = buffer[-1]
            sample_label = jnp.array(self._union_labels[uidx])
            yield (*buffer, sample_label)
