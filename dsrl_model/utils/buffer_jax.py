import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class BatchData:
    horizon: int = struct.field(pytree_node=False)

    pos_obs: jnp.ndarray
    pos_act: jnp.ndarray
    pos_reward: jnp.ndarray
    pos_cost: jnp.ndarray

    neg_obs: jnp.ndarray
    neg_act: jnp.ndarray
    neg_reward: jnp.ndarray
    neg_cost: jnp.ndarray

    union_obs: jnp.ndarray
    union_act: jnp.ndarray
    union_reward: jnp.ndarray
    union_cost: jnp.ndarray
    union_idx: jnp.ndarray
    union_done: jnp.ndarray = None
    union_weight: jnp.ndarray = None


@jax.jit
def get_batch_data_obj(obj, p_idx, n_idx, u_idx):
    horizon_offsets = jnp.arange(obj.horizon)
    p_h_idx = p_idx[..., None] + horizon_offsets
    n_h_idx = n_idx[..., None] + horizon_offsets
    u_h_idx = u_idx[..., None] + horizon_offsets

    return BatchData(
        horizon=obj.horizon,
        pos_obs=obj.pos_obs[p_h_idx],
        pos_act=obj.pos_act[p_h_idx],
        pos_reward=obj.pos_reward[p_h_idx],
        pos_cost=obj.pos_cost[p_h_idx],
        neg_obs=obj.neg_obs[n_h_idx],
        neg_act=obj.neg_act[n_h_idx],
        neg_reward=obj.neg_reward[n_h_idx],
        neg_cost=obj.neg_cost[n_h_idx],
        union_obs=obj.union_obs[u_h_idx],
        union_act=obj.union_act[u_h_idx],
        union_reward=obj.union_reward[u_h_idx],
        union_cost=obj.union_cost[u_h_idx],
        union_idx=u_idx,
    )


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


@functools.partial(jax.jit, static_argnames=["horizon"])
def sample_horizon_arr(arr, idx, horizon):
    horizon_offsets = jnp.arange(horizon)
    h_idx = idx[..., None] + horizon_offsets
    return arr[h_idx]


@jax.jit
def sample_arr(arr, idx):
    return arr[idx]


@jax.jit
def update_arr_jit(arr, idxs, val):
    return arr.at[idxs].set(val)


@struct.dataclass
class OnPolicyDataBuffer:
    horizon: int = struct.field(pytree_node=False)
    batch_size: int = struct.field(pytree_node=False)

    pos_obs: jnp.ndarray
    pos_act: jnp.ndarray
    pos_reward: jnp.ndarray
    pos_cost: jnp.ndarray
    pos_priorities: jnp.ndarray

    neg_obs: jnp.ndarray
    neg_act: jnp.ndarray
    neg_reward: jnp.ndarray
    neg_cost: jnp.ndarray
    neg_priorities: jnp.ndarray

    union_obs: jnp.ndarray
    union_act: jnp.ndarray
    union_reward: jnp.ndarray
    union_cost: jnp.ndarray
    union_priorities: jnp.ndarray

    @staticmethod
    def sample_idxs(obj, key, num_steps):
        key1, key2, key3 = jax.random.split(key, 3)

        pos_probs = obj.pos_priorities / obj.pos_priorities.sum()
        pos_total = len(pos_probs)
        pos_idxs = jax.random.choice(
            key=key1,
            a=pos_total,
            shape=(num_steps, obj.batch_size),
            p=pos_probs,
            replace=True,
        )

        neg_probs = obj.neg_priorities / obj.neg_priorities.sum()
        neg_total = len(neg_probs)
        neg_idxs = jax.random.choice(
            key=key2,
            a=neg_total,
            shape=(num_steps, obj.batch_size),
            p=neg_probs,
            replace=True,
        )

        union_probs = obj.union_priorities / obj.union_priorities.sum()
        union_total = len(union_probs)
        union_idxs = jax.random.choice(
            key=key3,
            a=union_total,
            shape=(num_steps, obj.batch_size),
            p=union_probs,
            replace=True,
        )

        return pos_idxs, neg_idxs, union_idxs

    @staticmethod
    def sample_batch(obj, p_idx, n_idx, u_idx):
        return get_batch_data_obj(
            obj=obj,
            p_idx=p_idx,
            n_idx=n_idx,
            u_idx=u_idx,
        )


@struct.dataclass
class OSILDataBuffer(OnPolicyDataBuffer):
    union_done: jnp.ndarray

    @staticmethod
    def sample_batch(obj, p_idx, n_idx, u_idx):
        batch = super().sample_batch(obj, p_idx, n_idx, u_idx)
        h_union_done = sample_horizon_arr(obj.union_done, u_idx, obj.horizon)
        batch = batch.replace(union_done=h_union_done)
        return batch


@struct.dataclass
class SafeCLDataBuffer(OnPolicyDataBuffer):
    union_weight: jnp.ndarray

    @staticmethod
    def update_union_weight(obj, idx, weight):
        weight = jnp.array(weight, dtype=obj.union_weight.dtype)
        union_weight = update_arr_jit(obj.union_weight, idx, weight)
        return obj.replace(union_weight=union_weight)

    @staticmethod
    def sample_batch(obj, p_idx, n_idx, u_idx):
        batch = super().sample_batch(obj, p_idx, n_idx, u_idx)
        union_weight = sample_arr(obj.union_weight, u_idx)
        return batch.replace(union_weight=union_weight)


@struct.dataclass
class ILIDDataBuffer(OnPolicyDataBuffer):
    union_weight: jnp.ndarray

    @staticmethod
    def sample_batch(obj, p_idx, n_idx, u_idx):
        batch = super().sample_batch(obj, p_idx, n_idx, u_idx)
        union_weight = sample_horizon_arr(obj.union_weight, u_idx, obj.horizon)
        batch = batch.replace(union_weight=union_weight)
        return batch


@functools.partial(jax.jit, static_argnames="rollback")
def reshuffle_union_data(obj: ILIDDataBuffer, pred_expert_mask, rollback, decay):
    weight_init = 1.0
    N = obj.union_obs.shape[0]
    dtype = obj.union_obs.dtype

    done = 1.0 - obj.union_priorities
    ep_id = jnp.cumsum(jnp.concat([jnp.array([0.0]), done[:-1]]))

    # -------------------------------------------------
    # Vectorized rollback
    # -------------------------------------------------
    ks = jnp.arange(rollback)
    indx_arr = jnp.arange(N)

    def shifted_mask(k):
        shifted = indx_arr + k + 1
        valid = shifted < N
        same_ep = jnp.where(valid, ep_id[indx_arr] == ep_id[shifted], False)
        kstep_expert = jnp.where(valid, pred_expert_mask[shifted], False)
        return kstep_expert & same_ep

    rollback_masks = jax.vmap(shifted_mask)(ks)
    # union all rollback masks
    total_mask = jnp.any(rollback_masks, axis=0)
    new_priorities = jnp.array(total_mask, dtype=dtype)

    # -------------------------------------------------
    # Weight decay (vectorized)
    # -------------------------------------------------
    decay_weights = weight_init * (decay**ks)
    weight_candidates = rollback_masks * decay_weights[:, None]
    new_weight = jnp.max(weight_candidates, axis=0)
    new_weight = jnp.where(total_mask, new_weight, 0.0)

    obj = obj.replace(union_priorities=new_priorities, union_weight=new_weight)
    return obj


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

    def get_data_buffer(self, data_buffer_cls=OnPolicyDataBuffer, **kwargs):
        return data_buffer_cls(
            horizon=self.horizon,
            batch_size=self.batch_size,
            pos_obs=self._pos_obs,
            pos_act=self._pos_act,
            pos_reward=self._pos_reward,
            pos_cost=self._pos_cost,
            pos_priorities=self._pos_priorities,
            neg_obs=self._neg_obs,
            neg_act=self._neg_act,
            neg_reward=self._neg_reward,
            neg_cost=self._neg_cost,
            neg_priorities=self._neg_priorities,
            union_obs=self._union_obs,
            union_act=self._union_act,
            union_reward=self._union_reward,
            union_cost=self._union_cost,
            union_priorities=self._union_priorities,
            **kwargs,
        )

    def sample_batch(self, p_idx, n_idx, u_idx):
        return batched_data(
            pos_obs=self._pos_obs,
            pos_act=self._pos_act,
            pos_reward=self._pos_reward,
            pos_cost=self._pos_cost,
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


class OSILBuffer(OnPolicyBuffer):
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
        self._union_done = np.zeros((self.union_capacity,), dtype=self.dtype)

    def add(
        self, obs, act, done, reward, cost, is_pos=False, is_neg=False, is_union=False
    ):
        if is_union:
            idx = self._union_idx
            self._union_done[idx : idx + self.ep_len] = np.array(done)

        super().add(obs, act, done, reward, cost, is_pos, is_neg, is_union)

    def to_jax_ndarray(self):
        super().to_jax_ndarray()
        self._union_done = jnp.array(self._union_done)

    def get_data_buffer(self):
        return super().get_data_buffer(OSILDataBuffer, union_done=self._union_done)


class ILIDBuffer(OnPolicyBuffer):
    def to_jax_ndarray(self):
        super().to_jax_ndarray()
        self._union_weight = jnp.zeros((self.union_capacity,), dtype=self.dtype)

    def get_data_buffer(self):
        return super().get_data_buffer(ILIDDataBuffer, union_weight=self._union_weight)


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
        priorities_alpha=1.0,
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
        self._union_weight = np.zeros((self.union_capacity,), dtype=self.dtype)

    def to_jax_ndarray(self):
        super().to_jax_ndarray()
        self._union_weight = jnp.array(self._union_weight, dtype=self.dtype)

    def get_data_buffer(self):
        return super().get_data_buffer(
            SafeCLDataBuffer, union_weight=self._union_weight
        )

    def update_labels(self, idxs, labels):
        labels = jnp.array(labels, dtype=self.dtype)
        self._union_weight = update_arr_jit(self._union_weight, idxs, labels)

    def sample_batch(self, p_idx, n_idx, u_idx):
        batch = super().sample_batch(p_idx, n_idx, u_idx)
        label = sample_arr(self._union_weight, u_idx)
        return (*batch, label)
