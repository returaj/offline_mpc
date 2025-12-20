import functools

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def update_arr_jit(arr, idxs, val):
    return arr.at[idxs].set(val)


@functools.partial(jax.jit, static_argnames=["horizon"])
def batched_data(
    pos_obs,
    pos_act,
    pos_reward,
    pos_cost,
    neg_obs,
    neg_act,
    neg_reward,
    neg_cost,
    union_obs,
    union_act,
    union_reward,
    union_cost,
    horizon,
    p_idx,
    n_idx,
    u_idx,
):
    horizon_offsets = jnp.arange(horizon)
    p_h_idx = p_idx[..., None] + horizon_offsets
    n_h_idx = n_idx[..., None] + horizon_offsets
    u_h_idx = u_idx[..., None] + horizon_offsets

    h_pos_obs = pos_obs[p_h_idx]
    h_pos_act = pos_act[p_h_idx]
    h_pos_reward = pos_reward[p_h_idx]
    h_pos_cost = pos_cost[p_h_idx]

    h_neg_obs = neg_obs[n_h_idx]
    h_neg_act = neg_act[n_h_idx]
    h_neg_reward = neg_reward[n_h_idx]
    h_neg_cost = neg_cost[n_h_idx]

    h_union_obs = union_obs[u_h_idx]
    h_union_act = union_act[u_h_idx]
    h_union_reward = union_reward[u_h_idx]
    h_union_cost = union_cost[u_h_idx]

    return (
        h_pos_obs,
        h_pos_act,
        h_pos_reward,
        h_pos_cost,
        h_neg_obs,
        h_neg_act,
        h_neg_reward,
        h_neg_cost,
        h_union_obs,
        h_union_act,
        h_union_reward,
        h_union_cost,
        u_idx,
    )


@jax.jit
def sample_label(labels, idx):
    return labels[idx]


class OnPolicyBuffer:
    def __init__(
        self,
        rngs,
        obs_dim,
        act_dim,
        pos_data_size,
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
        self.pos_capacity = pos_data_size
        self.neg_capacity = neg_data_size
        self.union_capacity = union_data_size
        self.ep_len = ep_len
        self._priorities_alpha = priorities_alpha
        self.dtype = np.float32

        self._pos_obs = np.empty((self.pos_capacity + 1, obs_dim), dtype=self.dtype)
        self._pos_act = np.empty((self.pos_capacity, act_dim), dtype=self.dtype)
        self._pos_reward = np.empty((self.pos_capacity,), dtype=self.dtype)
        self._pos_cost = np.empty((self.pos_capacity,), dtype=self.dtype)
        self._pos_priorities = np.ones((self.pos_capacity,), dtype=self.dtype)

        self._neg_obs = np.empty((self.neg_capacity + 1, obs_dim), dtype=self.dtype)
        self._neg_act = np.empty((self.neg_capacity, act_dim), dtype=self.dtype)
        self._neg_reward = np.empty((self.neg_capacity,), dtype=self.dtype)
        self._neg_cost = np.empty((self.neg_capacity,), dtype=self.dtype)
        self._neg_priorities = np.ones((self.neg_capacity,), dtype=self.dtype)

        self._union_obs = np.empty((self.union_capacity + 1, obs_dim), dtype=self.dtype)
        self._union_act = np.empty((self.union_capacity, act_dim), dtype=self.dtype)
        self._union_reward = np.empty((self.union_capacity,), dtype=self.dtype)
        self._union_cost = np.empty((self.union_capacity,), dtype=self.dtype)
        self._union_priorities = np.ones((self.union_capacity,), dtype=self.dtype)

        self._eps = 1e-6
        self._pos_idx = 0
        self._neg_idx = 0
        self._union_idx = 0

    def _update(
        self,
        obs_store,
        act_store,
        priority_store,
        reward_store,
        cost_store,
        idx,
        obs,
        act,
        priority,
        reward,
        cost,
        capacity,
    ):
        obs_store[idx : idx + self.ep_len] = np.array(obs)
        act_store[idx : idx + self.ep_len] = np.array(act)
        priority_store[idx : idx + self.ep_len] = np.array(priority)
        reward_store[idx : idx + self.ep_len] = np.array(reward)
        cost_store[idx : idx + self.ep_len] = np.array(cost)
        return (idx + self.ep_len) % capacity

    def add(
        self, obs, act, done, reward, cost, is_pos=False, is_neg=False, is_union=False
    ):
        assert cost is not None, "cost field cannot be none"
        max_priority = 1.0
        done_sum = np.sum(done) or 1.0
        true_ep_len = self.ep_len - done_sum + 1
        mask = np.arange(self.ep_len) >= true_ep_len - self.horizon
        new_priorities = np.full((self.ep_len,), max_priority)
        new_priorities[mask] = 0.0

        if is_pos:
            self._pos_idx = self._update(
                obs_store=self._pos_obs,
                act_store=self._pos_act,
                priority_store=self._pos_priorities,
                reward_store=self._pos_reward,
                cost_store=self._pos_cost,
                idx=self._pos_idx,
                obs=obs,
                act=act,
                priority=new_priorities,
                reward=reward,
                cost=cost,
                capacity=self.pos_capacity,
            )
        elif is_neg:
            self._neg_idx = self._update(
                obs_store=self._neg_obs,
                act_store=self._neg_act,
                priority_store=self._neg_priorities,
                reward_store=self._neg_reward,
                cost_store=self._neg_cost,
                idx=self._neg_idx,
                obs=obs,
                act=act,
                priority=new_priorities,
                reward=reward,
                cost=cost,
                capacity=self.neg_capacity,
            )
        elif is_union:
            self._union_idx = self._update(
                obs_store=self._union_obs,
                act_store=self._union_act,
                priority_store=self._union_priorities,
                reward_store=self._union_reward,
                cost_store=self._union_cost,
                idx=self._union_idx,
                obs=obs,
                act=act,
                priority=new_priorities,
                reward=reward,
                cost=cost,
                capacity=self.union_capacity,
            )
        else:
            raise ValueError(
                "Select either positive, negative or union dataset to add."
            )

    def to_jax_ndarray(self):
        self.dtype = jnp.float32

        self._pos_obs = jnp.array(self._pos_obs, dtype=self.dtype)
        self._pos_act = jnp.array(self._pos_act, dtype=self.dtype)
        self._pos_reward = jnp.array(self._pos_reward, dtype=self.dtype)
        self._pos_cost = jnp.array(self._pos_cost, dtype=self.dtype)
        self._pos_priorities = jnp.array(self._pos_priorities, dtype=self.dtype)

        self._neg_obs = jnp.array(self._neg_obs, dtype=self.dtype)
        self._neg_act = jnp.array(self._neg_act, dtype=self.dtype)
        self._neg_reward = jnp.array(self._neg_reward, dtype=self.dtype)
        self._neg_cost = jnp.array(self._neg_cost, dtype=self.dtype)
        self._neg_priorities = jnp.array(self._neg_priorities, dtype=self.dtype)

        self._union_obs = jnp.array(self._union_obs, dtype=self.dtype)
        self._union_act = jnp.array(self._union_act, dtype=self.dtype)
        self._union_reward = jnp.array(self._union_reward, dtype=self.dtype)
        self._union_cost = jnp.array(self._union_cost, dtype=self.dtype)
        self._union_priorities = jnp.array(self._union_priorities, dtype=self.dtype)

    def update_priorities(self, idxs, priorities):
        priorities = jnp.array(priorities, dtype=self.dtype) + self._eps
        self._union_priorities = update_arr_jit(
            self._union_priorities, idxs, priorities
        )

    def sample_batch(self, p_idx, n_idx, u_idx):
        return batched_data(
            pos_obs=self._neg_obs,
            pos_act=self._neg_act,
            pos_reward=self._neg_reward,
            pos_cost=self._neg_cost,
            neg_obs=self._neg_obs,
            neg_act=self._neg_act,
            neg_reward=self._neg_reward,
            neg_cost=self._neg_cost,
            union_obs=self._union_obs,
            union_act=self._union_act,
            union_reward=self._union_reward,
            union_cost=self._union_cost,
            horizon=self.horizon,
            p_idx=p_idx,
            n_idx=n_idx,
            u_idx=u_idx,
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

        pos_probs = self._pos_priorities
        pos_probs /= pos_probs.sum()
        pos_total = len(pos_probs)
        pos_idxs = jax.random.choice(
            key=self.rngs(),
            a=pos_total,
            shape=(steps_per_epoch, batch_size),
            p=pos_probs,
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

        for p_idx, n_idx, u_idx in zip(pos_idxs, neg_idxs, union_idxs):
            yield self.sample_batch(p_idx, n_idx, u_idx)


class SafeCLBuffer(OnPolicyBuffer):
    def __init__(
        self,
        rngs,
        obs_dim,
        act_dim,
        pos_data_size,
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
            pos_data_size,
            neg_data_size,
            union_data_size,
            horizon,
            batch_size,
            ep_len,
            priorities_alpha,
        )
        self._union_labels = np.zeros((self.union_capacity,), dtype=self.dtype)

    def to_jax_ndarray(self):
        super().to_jax_ndarray()
        self._union_labels = jnp.array(self._union_labels, dtype=self.dtype)

    def update_labels(self, idxs, labels):
        labels = jnp.array(labels, dtype=self.dtype)
        self._union_labels = update_arr_jit(self._union_labels, idxs, labels)

    def sample_batch(self, p_idx, n_idx, u_idx):
        batch = super().sample_batch(p_idx, n_idx, u_idx)
        label = sample_label(self._union_labels, u_idx)
        return (*batch, label)
