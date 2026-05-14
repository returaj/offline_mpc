import functools
import os
import random
import re
import sys
import time
from collections import deque
from copy import deepcopy

import dsrl.infos as dsrl_infos
import dsrl.offline_safety_gymnasium  # type: ignore
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx, struct
from jax import debug

from dsrl_model.utils.buffer_jax import SafeCLBuffer
from dsrl_model.utils.dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_nonpref_mean_value,
    get_normalized_data,
    get_pos_neg_and_union_data,
)
from dsrl_model.utils.models_jax import (
    EnsembleValue,
    SafeDiceTanhMixtureActor,
    TransformerEmbedding,
    get_tree_norm,
)
from dsrl_model.utils.native_logger import EpochLogger
from dsrl_model.utils.utils import make_static_config_from_dict, single_agent_args

# jax.config.update("jax_disable_jit", True)

EPS = 1e-6

default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "eval_episode_freq": 1,  # use saved bc_policy to run evaluatation
    "hidden_size": 256,
    "embd_size": 128,
    "max_grad_norm": 5.0,
    "gamma": 0.99,
    "action_repeat": 1,  # set to 2, min value is 1
    "train_horizon": 500,  # 20
    "embd_freq": int(5e2),
    "warmup_steps": int(5e4),
    "decay": 0.999,
    "value_temp": 0.1,
    "value_limit": 0.85,
    "value_th": 0.85,
    "update_tau": 0.01,
    "weight_decay": 0.01,
    "total_iteration": int(1e6),
}

trajectory_cfg = {
    "density": 1.0,
    "target_cost": 25.0,
    # ((low_cost, low_reward), (high_cost, low_reward), (medium_cost, high_reward))
    "inpaint_ranges": None,
    "num_positive_trajectories": 0,
    "num_negative_trajectories": 50,
    "num_union_trajectories": -1,
}

trajectory_data = {
    "full": None,
    "reward_only": None,
    "cost_only": ((0.0, 1.0, 0.0, 0.5),),
}


class Models(nnx.Module):
    def __init__(
        self, curriculum_embedding, embedding_target, embedding, value, policy
    ):
        self.curriculum_embedding = curriculum_embedding
        self.embedding_target = embedding_target
        self.embedding = embedding
        self.value = value
        self.policy = policy


class Optimizers(nnx.Module):
    def __init__(self, embedding, value, policy):
        self.embedding = embedding
        self.value = value
        self.policy = policy


class TrainingState(nnx.Module):
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers


@struct.dataclass
class TrainAux:
    weight: float = 0.0
    num_embd_updates: float = 0.0
    embd_freq: float = 0.0
    num_itr: float = 0.0


@struct.dataclass
class DataAux:
    pos_reward: float = 0.0
    neg_reward: float = 0.0
    union_reward: float = 0.0
    pos_cost: float = 0.0
    neg_cost: float = 0.0
    union_cost: float = 0.0


@struct.dataclass
class UnionSinkAux:
    mean_score: float = 0.0
    std_score: float = 0.0
    pos_sink_percent: float = 0.0
    neg_sink_percent: float = 0.0
    sink_percent: float = 0.0
    sink_percent_ema: float = 0.0
    sink_bimodality: float = 0.0
    sink_bimodality_ema: float = 0.0
    pos_sink_reward: float = 0.0
    neg_sink_reward: float = 0.0
    source_reward: float = 0.0
    pos_sink_cost: float = 0.0
    neg_sink_cost: float = 0.0
    source_cost: float = 0.0
    pos_sink_nonpref: float = 0.0
    neg_sink_nonpref: float = 0.0
    source_nonpref: float = 0.0


@struct.dataclass
class ValueAux:
    loss: float = 0.0
    pos_loss: float = 0.0
    neg_loss: float = 0.0
    random_loss: float = 0.0
    union_loss: float = 0.0
    pos_value: float = 0.0
    neg_value: float = 0.0
    random_value: float = 0.0
    union_value: float = 0.0


@struct.dataclass
class EmbeddingAux:
    loss: float = 0.0
    pos_loss: float = 0.0
    neg_loss: float = 0.0
    union_loss: float = 0.0
    random_loss: float = 0.0
    pos_score: float = 0.0
    neg_score: float = 0.0
    union_score: float = 0.0
    random_score: float = 0.0


@struct.dataclass
class PolicyAux:
    loss: float = 0.0
    q: float = 0.0
    v: float = 0.0
    weight: float = 0.0
    entropy: float = 0.0


def evaluate_bc_policy(eval_env, bc_policy, mu_obs, std_obs):
    eval_done = False
    eval_obs, _ = eval_env.reset()
    eval_obs = (eval_obs - mu_obs) / (std_obs + EPS)
    eval_obs = jnp.expand_dims(jnp.array(eval_obs, dtype=jnp.float32), axis=0)
    eval_reward, eval_cost, eval_len = 0.0, 0.0, 0.0
    while not eval_done:
        act = bc_policy(eval_obs)
        next_obs, reward, terminated, truncated, info = eval_env.step(
            np.array(act[0].squeeze())
        )
        cost = info["cost"]
        next_obs = (next_obs - mu_obs) / (std_obs + EPS)
        next_obs = jnp.expand_dims(jnp.array(next_obs, dtype=jnp.float32), axis=0)
        eval_obs = next_obs
        eval_reward += reward
        eval_cost += cost
        eval_len += 1
        eval_done = terminated or truncated
    return eval_reward, eval_cost, eval_len


@jax.jit
def index_fun(arr, all_labels, label_range):
    # For each value in arr, find which interval of `ranges` it falls into
    # range[i] >= x > range[i+1] → assign labels[i]
    idx = jnp.sum(arr[..., None] <= label_range[1:], axis=-1)
    return all_labels[idx]


@jax.jit
def kernel_density_entropy(samples, mask, sigma=0.2):
    diffs = samples[:, None] - samples[None, :]
    kernel = jnp.exp(-0.5 * (diffs / sigma) ** 2)
    mask2d = mask[:, None] * mask[None, :]
    kernel = kernel * mask2d
    n_effective = jnp.sum(mask)
    p_est = jnp.sum(kernel, axis=1) / (n_effective * sigma * jnp.sqrt(2 * jnp.pi) + EPS)
    entropy = -jnp.sum(mask * jnp.log(p_est + EPS)) / (n_effective + EPS)
    return entropy


def get_multimodel_score(union, target_z, embedding_model):
    num_models = 5

    @nnx.scan(length=num_models, in_axes=nnx.Carry, out_axes=(nnx.Carry, 0))
    def multimodel(carry):
        x, target_z, model = carry
        # Batch X embd_dim
        z = model(x)
        # Batch
        score = jnp.einsum("ij,ij->i", z, target_z)
        return carry, score

    # 5 X Batch
    _, union_scores = multimodel((union, target_z, embedding_model))
    # use mean of 5 models to estimate the union score
    union_score = jnp.mean(union_scores, axis=0)
    return union_score


@functools.partial(jax.jit, static_argnames=["num_trajs"])
def get_cost_reward_weight_matrix(data_buffer, num_trajs):
    costs = data_buffer.union_cost
    rewards = data_buffer.union_reward
    weights = data_buffer.union_weight
    priorities = data_buffer.union_priorities

    nonzero = priorities != 0
    # start is where current priority is nonzero and previous is zero
    start = nonzero & jnp.concatenate([jnp.array([True]), ~nonzero[:-1]])

    segment_ids = jnp.cumsum(start) - 1
    num_segments = num_trajs

    # estimates the total cost/reward/weight of the trajectory
    costs = jax.ops.segment_sum(costs, segment_ids, num_segments)
    rewards = jax.ops.segment_sum(rewards, segment_ids, num_segments)
    weights_sum = jax.ops.segment_sum(weights, segment_ids, num_segments)

    # estimates the expected weight of the trajectory
    lengths = jax.ops.segment_sum(jnp.ones_like(weights), segment_ids, num_segments)
    weights = weights_sum / lengths

    return costs, rewards, weights


def plot_weighted_trajectory_data(data_buffer, num_trajs, env_name, save_plot):
    # plot cost, reward weight
    costs, rewards, weights = get_cost_reward_weight_matrix(data_buffer, num_trajs)
    # Custom colormap: purple to orange
    neg_limit, pos_limit = min(min(weights), -0.01), max(max(weights), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=neg_limit, vcenter=0, vmax=pos_limit)
    fig, ax = plt.subplots()
    sc = ax.scatter(costs, rewards, c=weights, cmap="PuOr", norm=norm)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Preference")

    ax.set_xlabel("Traj. Cost")
    ax.set_ylabel("Traj. Reward")
    ax.set_title(f"Preference Plot ({env_name})")
    fig.savefig(save_plot, dpi=300, bbox_inches="tight")


@jax.jit
def ema(target, curr, decay):
    return decay * target + (1 - decay) * curr


@nnx.jit
def polyak_update(target_model, curr_model, tau):
    target_param = nnx.state(target_model, nnx.Param)
    curr_param = nnx.state(curr_model, nnx.Param)

    new_target_param = jax.tree_util.tree_map(
        lambda t, c: (1 - tau) * t + tau * c, target_param, curr_param
    )
    nnx.update(target_model, new_target_param)
    return target_model


@jax.jit
def discounted_sum(vector_x, gamma):
    dtype = vector_x.dtype
    horizon = vector_x.shape[0]

    def body_fun(t, cumsum):
        cumsum = vector_x[horizon - 1 - t] + gamma * cumsum
        return cumsum

    init_cumsum = jnp.zeros_like(vector_x[0], dtype=dtype)
    cumsum = jax.lax.fori_loop(0, horizon, body_fun, init_cumsum)
    return cumsum


@jax.jit
def range_loss(scores, target_scores, scale):
    loss = (scores - target_scores) ** 2
    return scale * loss


def get_sink_source_mean_values(value_arr, pos_mask, neg_mask):
    pos_count, neg_count = pos_mask.sum(), neg_mask.sum()
    batch_horizon_value = jnp.sum(value_arr, axis=-1)
    total_pos_value = jnp.einsum("i,i->", batch_horizon_value, pos_mask)
    total_neg_value = jnp.einsum("i,i->", batch_horizon_value, neg_mask)
    mean_pos_value = total_pos_value / jnp.clip(pos_count, min=1)
    mean_neg_value = total_neg_value / jnp.clip(neg_count, min=1)
    source_mask = 1.0 - pos_mask - neg_mask
    source_count = source_mask.sum()
    total_source_value = jnp.einsum("i,i->", batch_horizon_value, source_mask)
    mean_source_value = total_source_value / jnp.clip(source_count, min=1)
    return mean_pos_value, mean_neg_value, mean_source_value


def get_union_sink(
    curriculum_embedding_model, embedding_model, data, weight, do_warmup, config
):
    dtype = data.union_obs.dtype
    batch, horizon, _ = data.union_obs.shape

    mid_traj = horizon // 2
    select_mid_traj = 1 / mid_traj * (jnp.arange(horizon) > mid_traj).astype(dtype)

    # Batch X Horizon X obs_act_dim
    target_union = jnp.concat([data.union_obs, data.union_act], axis=-1)

    # Batch X Horizon X embd_dim
    target_union_z = curriculum_embedding_model(target_union, training=False)

    # # multimodel estimation of the union score
    # union_score = get_multimodel_score(target_union, target_union_z, embedding_model)

    # Batch X Horizon X embd_dim
    union_z = embedding_model(target_union, training=False)
    # Batch X Horizon
    traj_union_score = do_warmup * jnp.einsum("ijk,ijk->ij", union_z, target_union_z)
    # Batch
    union_score = jnp.einsum("ij,j->i", traj_union_score, select_mid_traj)

    pos_sink_mask = (union_score > config.value_limit).astype(dtype)
    neg_sink_mask = (union_score < -config.value_limit).astype(dtype)
    source_mask = 1.0 - pos_sink_mask - neg_sink_mask

    pos_sink_percent = pos_sink_mask.sum() / batch
    neg_sink_percent = neg_sink_mask.sum() / batch
    sink_percent = pos_sink_percent + neg_sink_percent
    sink_bimodality = jnp.mean(jnp.abs(union_score) ** 2)

    sink_score = pos_sink_mask - neg_sink_mask

    # Batch
    mean_pos_sink_reward, mean_neg_sink_reward, mean_source_reward = (
        get_sink_source_mean_values(data.union_reward, pos_sink_mask, neg_sink_mask)
    )
    mean_pos_sink_cost, mean_neg_sink_cost, mean_source_cost = (
        get_sink_source_mean_values(data.union_cost, pos_sink_mask, neg_sink_mask)
    )

    mean_pos_sink_nonpref = (pos_sink_percent > 0.0) * get_nonpref_mean_value(
        mean_pos_sink_reward,
        mean_pos_sink_cost,
        horizon,
        config.env_name,
        rscale=config.nonpref_reward_scale,
        cscale=config.nonpref_cost_scale,
    )
    mean_neg_sink_nonpref = (neg_sink_percent > 0.0) * get_nonpref_mean_value(
        mean_neg_sink_reward,
        mean_neg_sink_cost,
        horizon,
        config.env_name,
        rscale=config.nonpref_reward_scale,
        cscale=config.nonpref_cost_scale,
    )
    mean_source_nonpref = get_nonpref_mean_value(
        mean_source_reward,
        mean_source_cost,
        horizon,
        config.env_name,
        rscale=config.nonpref_reward_scale,
        cscale=config.nonpref_cost_scale,
    )

    # union_weight = jnp.where(sink_mask, weight, weight * union_score)
    # union_weight = jnp.maximum(data.union_weight, union_weight)

    # only add weights to those identified and reset those trajectories
    # which have not been identified to zero.
    pos_union_weight = pos_sink_mask * jnp.maximum(data.union_weight, weight)
    neg_union_weight = neg_sink_mask * jnp.minimum(data.union_weight, -weight)
    # use union score for outside mask
    source_weight = source_mask * weight * union_score
    union_weight = pos_union_weight + neg_union_weight + source_weight

    union_aux = UnionSinkAux(
        mean_score=union_score.mean(),
        std_score=union_score.std(),
        pos_sink_percent=pos_sink_percent,
        neg_sink_percent=neg_sink_percent,
        sink_percent=sink_percent,
        sink_bimodality=sink_bimodality,
        pos_sink_reward=mean_pos_sink_reward,
        neg_sink_reward=mean_neg_sink_reward,
        source_reward=mean_source_reward,
        pos_sink_cost=mean_pos_sink_cost,
        neg_sink_cost=mean_neg_sink_cost,
        source_cost=mean_source_cost,
        pos_sink_nonpref=mean_pos_sink_nonpref,
        neg_sink_nonpref=mean_neg_sink_nonpref,
        source_nonpref=mean_source_nonpref,
    )

    return sink_score, union_weight, union_aux


def embedding_grad_aux_fun(
    curriculum_embedding_model,
    embedding_model,
    data,
    target_union_score,
    config,
    pos_scale,
    neg_scale,
    union_scale,
    key,
):
    dtype = data.union_obs.dtype
    batch, horizon, _ = data.union_obs.shape

    mid_traj = horizon // 2
    select_mid_traj = 1 / mid_traj * (jnp.arange(horizon) > mid_traj).astype(dtype)

    # shape: Batch X Horizon X obs_act_dim
    target_pos = jnp.concat([data.pos_obs, data.pos_act], axis=-1)
    target_neg = jnp.concat([data.neg_obs, data.neg_act], axis=-1)
    target_union = jnp.concat([data.union_obs, data.union_act], axis=-1)

    ## we were able to do this if it is guareented
    ## that there will be negative trajectories too.
    # mix_p1 = jax.random.uniform(key=key1, shape=target_neg.shape)
    # target_random1 = mix_p1 * target_neg + (1 - mix_p1) * target_union

    key1, key2 = jax.random.split(key, num=2)
    mix_p1 = jax.random.uniform(key=key1, shape=target_union.shape)
    target_shuffle_union = jax.random.permutation(key2, target_union, axis=0)
    target_random = mix_p1 * target_shuffle_union + (1 - mix_p1) * target_union

    # Batch
    target_pos_score = config.pos_label * jnp.ones(shape=(batch,), dtype=dtype)
    target_neg_score = config.neg_label * jnp.ones(shape=(batch,), dtype=dtype)
    target_random_score = jnp.zeros(shape=(batch,), dtype=dtype)

    union_sink_mask = (target_union_score != 0).astype(dtype)

    # Batch X Horizon X embd_dim
    target_pos_z = curriculum_embedding_model(target_pos, training=False)
    target_neg_z = curriculum_embedding_model(target_neg, training=False)
    target_union_z = curriculum_embedding_model(target_union, training=False)
    target_random_z = curriculum_embedding_model(target_random, training=False)

    def loss_fun(embedding_model):
        # Batch X Horizon X embd_dim
        pos_z = embedding_model(target_pos)
        # Batch X Horizon
        traj_pos_score = jnp.einsum("ijk,ijk->ij", pos_z, target_pos_z)
        # Batch
        pos_score = jnp.einsum("ij,j->i", traj_pos_score, select_mid_traj)
        pos_mean_loss = jnp.mean(range_loss(pos_score, target_pos_score, pos_scale))
        pos_mean_score = pos_scale * jnp.mean(pos_score)

        neg_z = embedding_model(target_neg)
        traj_neg_score = jnp.einsum("ijk,ijk->ij", neg_z, target_neg_z)
        neg_score = jnp.einsum("ij,j->i", traj_neg_score, select_mid_traj)
        neg_mean_loss = jnp.mean(range_loss(neg_score, target_neg_score, neg_scale))
        neg_mean_score = neg_scale * jnp.mean(neg_score)

        union_z = embedding_model(target_union)
        traj_union_score = jnp.einsum("ijk,ijk->ij", union_z, target_union_z)
        union_score = jnp.einsum("ij,j->i", traj_union_score, select_mid_traj)
        union_loss = range_loss(union_score, target_union_score, union_scale)
        union_sink_count = jnp.clip(union_sink_mask.sum(), min=1.0)
        union_mean_loss = (
            jnp.einsum("i,i->", union_sink_mask, union_loss) / union_sink_count
        )
        union_mean_score = (
            jnp.einsum("i,i->", union_sink_mask, union_score) / union_sink_count
        )

        random_z = embedding_model(target_random)
        traj_random_score = jnp.einsum("ijk,ijk->ij", random_z, target_random_z)
        random_score = jnp.einsum("ij,j->i", traj_random_score, select_mid_traj)
        random_mean_loss = jnp.mean(range_loss(random_score, target_random_score, 1.0))
        random_mean_score = jnp.mean(random_score)

        loss = (
            pos_mean_loss
            + neg_mean_loss
            + config.pos_neg_ratio * union_mean_loss
            + random_mean_loss
        )

        return loss, EmbeddingAux(
            loss=loss,
            pos_loss=pos_mean_loss,
            neg_loss=neg_mean_loss,
            union_loss=union_mean_loss,
            random_loss=random_mean_loss,
            pos_score=pos_mean_score,
            neg_score=neg_mean_score,
            union_score=union_mean_score,
            random_score=random_mean_score,
        )

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, embd_aux), grads = grad_fun(embedding_model)

    return grads, embd_aux


def value_grad_aux_fun(
    value_model,
    union_mask,
    union_weight,
    data,
    config,
    pos_scale,
    neg_scale,
    union_scale,
    key,
):
    # Value model learns the preferred state-action pair score
    pref_sign = jnp.where(config.pos_label > 0, 1.0, -1.0)

    # Batch X obs/act_dim
    target_pos = jnp.concat([data.pos_obs[:, 0], data.pos_act[:, 0]], axis=-1)
    target_neg = jnp.concat([data.neg_obs[:, 0], data.neg_act[:, 0]], axis=-1)

    # union data
    target_union_obs, target_union_act = data.union_obs[:, 0], data.union_act[:, 0]
    target_union = jnp.concat([target_union_obs, target_union_act], axis=-1)

    key1, key2 = jax.random.split(key, num=2)
    mix_p1 = jax.random.uniform(key=key1, shape=target_union.shape)
    target_shuffle_union = jax.random.permutation(key2, target_union, axis=0)
    target_random = mix_p1 * target_shuffle_union + (1 - mix_p1) * target_union

    pos_weight = pref_sign * config.pos_label * jnp.ones_like(union_weight)
    neg_weight = pref_sign * config.neg_label * jnp.ones_like(union_weight)
    random_weight = jnp.zeros_like(union_weight)
    union_weight = pref_sign * union_weight

    full_mask = jnp.ones_like(union_mask)

    def xql_rescale_loss(value_model, mask, score, x, scale):
        # Batch
        v1, v2 = value_model(x)
        v1_z, v2_z = (score - v1) / config.value_temp, (score - v2) / config.value_temp

        max_z = jnp.maximum(v1_z, v2_z).max()
        max_z = jnp.where(max_z < -1.0, -1.0, max_z)
        # scale by e^max_z
        # Detach the gradients is important as loss function is getting changed
        max_z = jax.lax.stop_gradient(max_z)
        loss_v1 = jnp.exp(v1_z - max_z) - v1_z * jnp.exp(-max_z) - jnp.exp(-max_z)
        loss_v2 = jnp.exp(v2_z - max_z) - v2_z * jnp.exp(-max_z) - jnp.exp(-max_z)

        mask_count = jnp.clip(mask.sum(), min=1.0)
        loss = scale * (mask * (loss_v1 + loss_v2)).sum() / mask_count
        value = scale * (mask * jnp.minimum(v1, v2)).sum() / mask_count
        return loss, value

    def loss_fun(value_model):
        pos_loss, pos_value = xql_rescale_loss(
            value_model, full_mask, pos_weight, target_pos, pos_scale
        )
        neg_loss, neg_value = xql_rescale_loss(
            value_model, full_mask, neg_weight, target_neg, neg_scale
        )
        random_loss, random_value = xql_rescale_loss(
            value_model, full_mask, random_weight, target_random, 1.0
        )
        union_loss, union_value = xql_rescale_loss(
            value_model, union_mask, union_weight, target_union, union_scale
        )
        loss = pos_loss + config.pos_neg_ratio * neg_loss + union_loss
        return loss, ValueAux(
            loss=loss,
            pos_loss=pos_loss,
            neg_loss=neg_loss,
            random_loss=random_loss,
            union_loss=union_loss,
            pos_value=pos_value,
            neg_value=neg_value,
            random_value=random_value,
            union_value=union_value,
        )

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux), grads = grad_fun(value_model)
    return grads, aux


def policy_grad_aux_fun(policy_model, value_model, data, union_mask, do_warmup, config):
    batch, horizon, _ = data.union_obs.shape

    # B X obs/act_dim
    # only consider the first state-action pair as
    # our value function may not be trained enough to
    # judge the state-action pair for later trajectory pair

    # BH X obs/act_dim
    target_union_obs = data.union_obs.reshape(batch * horizon, -1)
    target_union_act = data.union_act.reshape(batch * horizon, -1)

    union_mask = union_mask.repeat(horizon)

    def loss_fun(policy_model):
        pred_union_act, log_pi, *_ = policy_model(target_union_obs)

        q = do_warmup * jnp.minimum(
            *value_model(jnp.concat([target_union_obs, target_union_act], axis=-1))
        )
        v = do_warmup * jnp.minimum(
            *value_model(jnp.concat([target_union_obs, pred_union_act], axis=-1))
        )

        union_count = jnp.clip(union_mask.sum(), min=1.0)

        weight = jnp.exp(jnp.clip(q / config.value_temp, max=5.0))
        if config.policy_loss_type == "forward_kl":
            union_loss = optax.l2_loss(pred_union_act, target_union_act).sum(axis=-1)
            loss = (union_mask * weight * union_loss).sum() / union_count
        else:
            union_loss = -v + 0.001 * log_pi
            loss = (union_mask * union_loss).sum() / union_count

        qmean = (union_mask * q).sum() / union_count
        vmean = (union_mask * v).sum() / union_count
        weightmean = (union_mask * weight).sum() / union_count
        entropymean = -(union_mask * log_pi).sum() / union_count

        return loss, PolicyAux(
            loss=loss,
            q=qmean,
            v=vmean,
            weight=weightmean,
            entropy=entropymean,
        )

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, policy_aux), grads = grad_fun(policy_model)

    return grads, policy_aux


def train_step(
    state, batch_data, config, train_aux, has_positive, has_negative, do_warmup, key
):
    key1, key2 = jax.random.split(key, num=2)

    union_score, union_weight, union_sink_aux = get_union_sink(
        curriculum_embedding_model=state.models.curriculum_embedding,
        embedding_model=state.models.embedding_target,
        data=batch_data,
        weight=train_aux.weight,
        do_warmup=1.0 * do_warmup,
        config=config,
    )

    embedding_grads, embedding_aux = embedding_grad_aux_fun(
        curriculum_embedding_model=state.models.curriculum_embedding,
        embedding_model=state.models.embedding,
        data=batch_data,
        target_union_score=union_score,
        config=config,
        pos_scale=1.0 * has_positive,
        neg_scale=1.0 * has_negative,
        union_scale=1.0 * do_warmup,  # convert into float type
        key=key1,
    )
    state.optimizers.embedding.update(embedding_grads)

    # Allow full union dataset to learn value
    union_mask = jnp.ones_like(union_weight)
    value_grads, value_aux = value_grad_aux_fun(
        value_model=state.models.value,
        union_mask=union_mask,
        union_weight=union_weight,
        data=batch_data,
        config=config,
        pos_scale=1.0 * has_positive,
        neg_scale=1.0 * has_negative,
        union_scale=1.0 * do_warmup,
        key=key2,
    )
    state.optimizers.value.update(value_grads)

    policy_grads, policy_aux = policy_grad_aux_fun(
        policy_model=state.models.policy,
        value_model=state.models.value,
        data=batch_data,
        union_mask=union_mask,  # full union dataset to learn policy
        do_warmup=1.0 * do_warmup,
        config=config,
    )
    state.optimizers.policy.update(policy_grads)

    data_aux = DataAux(
        pos_reward=has_positive * batch_data.pos_reward.sum(-1).mean(),
        pos_cost=has_positive * batch_data.pos_cost.sum(-1).mean(),
        neg_reward=has_negative * batch_data.neg_reward.sum(-1).mean(),
        neg_cost=has_negative * batch_data.neg_cost.sum(-1).mean(),
        union_reward=batch_data.union_reward.sum(-1).mean(),
        union_cost=batch_data.union_cost.sum(-1).mean(),
    )

    return (
        union_weight,
        union_sink_aux,
        embedding_aux,
        value_aux,
        policy_aux,
        data_aux,
    )


@nnx.jit
def train_n_steps(
    state,
    data_buffer,
    train_aux,
    config,
    has_positive,
    has_negative,
    do_warmup,
    key,
    steps,
):
    num_steps = config.log_freq

    key1, key2 = jax.random.split(key, 2)

    pos_idxs, neg_idxs, union_idxs = data_buffer.sample_idxs(
        data_buffer, key1, num_steps
    )

    def body_fun(i, carry):
        (
            key,
            state,
            data_buffer,
            train_aux,
            val_aux,
        ) = carry

        prev_sink_aux, *_ = val_aux

        key, subkey = jax.random.split(key, 2)

        batch_data = data_buffer.sample_batch(
            data_buffer, pos_idxs[i], neg_idxs[i], union_idxs[i]
        )

        union_weight, sink_aux, *other_aux = train_step(
            state=state,
            batch_data=batch_data,
            config=config,
            train_aux=train_aux,
            has_positive=has_positive,
            has_negative=has_negative,
            do_warmup=do_warmup,
            key=subkey,
        )

        embedding_cond = 1.0 * ((steps + i + 1) % train_aux.embd_freq == 0)
        polyak_update(
            state.models.embedding_target,
            state.models.embedding,
            embedding_cond * config.update_tau,
        )

        sink_percent_ema = ema(
            prev_sink_aux.sink_percent_ema, sink_aux.sink_percent, config.decay
        )
        sink_bimodality_ema = ema(
            prev_sink_aux.sink_bimodality_ema, sink_aux.sink_bimodality, config.decay
        )
        sink_aux = sink_aux.replace(
            sink_percent_ema=sink_percent_ema, sink_bimodality_ema=sink_bimodality_ema
        )

        update_weight = do_warmup * embedding_cond
        embd_freq = jnp.maximum(
            250, (100 * sink_bimodality_ema * config.embd_freq) // 1
        )

        train_aux = train_aux.replace(
            embd_freq=jnp.where(embedding_cond, embd_freq, train_aux.embd_freq),
            num_embd_updates=train_aux.num_embd_updates + embedding_cond,
            weight=jnp.where(
                update_weight, 1.0 - sink_bimodality_ema, train_aux.weight
            ),
        )

        # do_warmup: False: union_weight = 0
        # do_warmup: True: update union_weight
        data_buffer = data_buffer.update_union_weight(
            data_buffer, union_idxs[i], union_weight
        )

        return (
            key,
            state,
            data_buffer,
            train_aux,
            (sink_aux, *other_aux),
        )

    init_carry = (
        key2,
        state,
        data_buffer,
        train_aux,
        (UnionSinkAux(), EmbeddingAux(), ValueAux(), PolicyAux(), DataAux()),
    )
    _, _, data_buffer, train_aux, val_aux = nnx.fori_loop(
        0, num_steps, body_fun, init_carry
    )

    train_aux = train_aux.replace(num_itr=num_steps)
    return data_buffer, train_aux, *val_aux


def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    rngs = nnx.Rngs(
        default=args.seed,
        params=args.seed + 3,
        dropout=args.seed + 5,
        random_sample=args.seed + 7,
    )
    curriculum_rngs = nnx.Rngs(params=args.seed + 42, dropout=args.seed + 47)

    # set default device id
    jax.default_device = jax.devices(args.device)[args.device_id]

    has_positive = float(args.num_preferred > 0)
    has_negative = float(args.num_non_preferred > 0)

    assert (
        has_positive or has_negative
    ), "Both preferred and non-preferred trajectory dataset cannot be empty together."

    trajectory_cfg["num_positive_trajectories"] = args.num_preferred
    trajectory_cfg["num_negative_trajectories"] = args.num_non_preferred
    trajectory_cfg["num_union_trajectories"] = args.num_union
    trajectory_cfg["non_pref_noise"] = args.non_pref_noise
    trajectory_cfg["data_inpaint"] = args.data_inpaint
    trajectory_cfg["inpaint_ranges"] = trajectory_data[args.data_inpaint]

    config = {**default_cfg, **trajectory_cfg}
    config["train_horizon"] = args.train_horizon or config.get("train_horizon")
    config["policy_loss_type"] = args.policy_loss_type
    config["normalize_observation"] = args.normalize_observation
    config["value_temp"] = args.value_weight_temp or config["value_temp"]
    config["value_limit"] = args.value_weight_limit or config["value_limit"]
    config["embd_freq"] = args.embd_freq or config["embd_freq"]
    config["pos_label"] = args.preferred_label
    config["neg_label"] = args.non_preferred_label
    config["lr"] = args.lr
    config["pos_neg_ratio"] = jnp.maximum(args.num_preferred, 1.0) / jnp.maximum(
        args.num_non_preferred, 1.0
    )

    # set training steps
    batch_size = args.batch_size or config.get("batch_size")
    config["batch_size"] = batch_size

    # env name
    env_name = re.search(r"Offline(.*?)(?:Gymnasium)?-v[0-9]", args.task).group(1)
    config["env_name"] = env_name

    # nonpref_reward_scale and nonpref_cost_scale
    config["nonpref_reward_scale"] = 1.0
    config["nonpref_cost_scale"] = 1.0
    if args.data_inpaint == "cost_only":
        config["nonpref_reward_scale"] = 0.0
    if args.data_inpaint == "reward_only":
        config["nonpref_cost_scale"] = 0.0

    config_data = make_static_config_from_dict(name="State", d=config)()

    # evaluation environment
    eval_env = gym.make(args.task)
    eval_env.set_target_cost(config["target_cost"])
    eval_env.reset(seed=args.seed)

    # set model
    obs_space, act_space = eval_env.observation_space, eval_env.action_space
    policy_model = SafeDiceTanhMixtureActor(
        rngs=rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_size=config["hidden_size"],
    )
    policy_optimizer = nnx.Optimizer(
        model=policy_model,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adamw(
                learning_rate=config["lr"], weight_decay=config["weight_decay"]
            ),
        ),
    )

    embd_size = config["embd_size"]
    embedding_model = TransformerEmbedding(
        rngs=rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        horizon=config["train_horizon"],
        embd_dim=embd_size,
        num_attentions=2,
    )
    embedding_optimizer = nnx.Optimizer(
        model=embedding_model,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adamw(
                learning_rate=config["lr"], weight_decay=config["weight_decay"]
            ),
        ),
    )
    embedding_model_target = deepcopy(embedding_model)
    curriculum_embedding_model = TransformerEmbedding(
        rngs=curriculum_rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        horizon=config["train_horizon"],
        embd_dim=embd_size,
        num_attentions=3,
        do_residual=False,
    )

    value_model = EnsembleValue(
        rngs=rngs,
        x_dim=obs_space.shape[0] + act_space.shape[0],
        hidden_size=config["hidden_size"],
    )
    value_optimizer = nnx.Optimizer(
        model=value_model,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adamw(
                learning_rate=config["lr"], weight_decay=config["weight_decay"]
            ),
        ),
    )

    state = TrainingState(
        models=Models(
            curriculum_embedding=curriculum_embedding_model,
            embedding_target=embedding_model_target,
            embedding=embedding_model,
            value=value_model,
            policy=policy_model,
        ),
        optimizers=Optimizers(
            embedding=embedding_optimizer,
            value=value_optimizer,
            policy=policy_optimizer,
        ),
    )

    # data
    ep_len = dsrl_infos.DEFAULT_MAX_EPISODE_STEPS[env_name]
    data = get_dataset_in_d4rl_format(
        eval_env, trajectory_cfg, args.task, ep_len, config["action_repeat"]
    )
    pos_data, neg_data, union_data = get_pos_neg_and_union_data(
        data, trajectory_cfg, save_dir=args.log_dir, seed=args.seed
    )
    mu_obs, std_obs = 0.0, 1.0
    if config["normalize_observation"]:
        pos_data, neg_data, union_data, mu_obs, std_obs = get_normalized_data(
            pos_data, neg_data, union_data
        )

    union_observations = union_data["observations"]
    union_actions = union_data["actions"]
    union_dones = union_data["timeouts"] | union_data["terminals"]
    union_rewards = union_data["rewards"]
    union_costs = union_data["costs"]

    ep_len = ep_len // config["action_repeat"] + (ep_len % config["action_repeat"] > 0)
    assert (
        union_observations.shape[1] == ep_len
    ), f"{union_observations.shape[1]} episode length is different from {ep_len}"

    pos_data_size = 1
    if has_positive:
        pos_data_size = np.prod(pos_data["observations"].shape[:-1])

    neg_data_size = 1
    if has_negative:
        neg_data_size = np.prod(neg_data["observations"].shape[:-1])

    buffer = SafeCLBuffer(
        rngs=rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        pos_data_size=pos_data_size,
        neg_data_size=neg_data_size,
        union_data_size=np.prod(union_observations.shape[:-1]),
        horizon=config["train_horizon"],
        batch_size=batch_size,
        ep_len=ep_len,
    )

    if has_positive:
        pos_observations = pos_data["observations"]
        pos_actions = pos_data["actions"]
        pos_dones = pos_data["timeouts"] | pos_data["terminals"]
        pos_rewards = pos_data["rewards"]
        pos_costs = pos_data["costs"]

        for obs, act, done, reward, cost in zip(
            pos_observations, pos_actions, pos_dones, pos_rewards, pos_costs
        ):
            buffer.add(obs, act, done, reward, cost, is_pos=True)

    if has_negative:
        neg_observations = neg_data["observations"]
        neg_actions = neg_data["actions"]
        neg_dones = neg_data["timeouts"] | neg_data["terminals"]
        neg_rewards = neg_data["rewards"]
        neg_costs = neg_data["costs"]

        for obs, act, done, reward, cost in zip(
            neg_observations, neg_actions, neg_dones, neg_rewards, neg_costs
        ):
            buffer.add(obs, act, done, reward, cost, is_neg=True)

    for obs, act, done, reward, cost in zip(
        union_observations, union_actions, union_dones, union_rewards, union_costs
    ):
        buffer.add(obs, act, done, reward, cost, is_union=True)

    buffer.to_jax_ndarray()
    data_buffer = buffer.get_data_buffer()

    # set logger
    dict_args = config
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)

    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config(dict_args)
    logger.log("Start embedding, cost and bc_policy model training.")

    train_aux = TrainAux(
        weight=1.0,
        num_embd_updates=0,
        embd_freq=config_data.embd_freq,
        num_itr=0,
    )
    steps = 0
    while steps < config["total_iteration"]:
        do_warmup = steps >= config_data.warmup_steps

        (
            data_buffer,
            train_aux,
            sink_aux,
            embd_aux,
            value_aux,
            policy_aux,
            data_aux,
        ) = train_n_steps(
            state=state,
            data_buffer=data_buffer,
            train_aux=train_aux,
            config=config_data,
            has_positive=has_positive,
            has_negative=has_negative,
            do_warmup=do_warmup,
            key=rngs.random_sample(),
            steps=steps,
        )

        steps += train_aux.num_itr

        logger.logged = False

        logger.log_tabular("Train/Steps", steps)

        logger.log_tabular("Loss/Loss_embedding", embd_aux.loss.item())
        logger.log_tabular("Loss/Loss_embd_pos", embd_aux.pos_loss.item())
        logger.log_tabular("Loss/Loss_embd_neg", embd_aux.neg_loss.item())
        logger.log_tabular("Loss/Loss_embd_union", embd_aux.union_loss.item())
        logger.log_tabular("Loss/Loss_embd_random_loss", embd_aux.random_loss.item())
        logger.log_tabular("Loss/Loss_value", value_aux.loss.item())
        logger.log_tabular("Loss/Loss_value_pos", value_aux.pos_loss.item())
        logger.log_tabular("Loss/Loss_value_neg", value_aux.neg_loss.item())
        logger.log_tabular("Loss/Loss_value_random", value_aux.random_loss.item())
        logger.log_tabular("Loss/Loss_value_union", value_aux.union_loss.item())

        logger.log_tabular("Loss/Loss_policy", policy_aux.loss.item())
        logger.log_tabular("Loss/Loss_policy_q", policy_aux.q.item())
        logger.log_tabular("Loss/Loss_policy_v", policy_aux.v.item())
        logger.log_tabular("Loss/Loss_policy_weight", policy_aux.weight.item())
        logger.log_tabular("Loss/Loss_policy_entropy", policy_aux.entropy.item())

        logger.log_tabular(
            "Num/target_embedding_updates", train_aux.num_embd_updates.item()
        )
        logger.log_tabular("Num/target_embedding_stale", train_aux.embd_freq.item())

        logger.log_tabular("Mean/union_score", sink_aux.mean_score.item())
        logger.log_tabular("Mean/union_std", sink_aux.std_score.item())
        logger.log_tabular("Mean/union_bimodality", sink_aux.sink_bimodality.item())
        logger.log_tabular(
            "Mean/union_bimodality_ema", sink_aux.sink_bimodality_ema.item()
        )
        logger.log_tabular("Mean/embd_pos_score", embd_aux.pos_score.item())
        logger.log_tabular("Mean/embd_neg_score", embd_aux.neg_score.item())
        logger.log_tabular("Mean/embd_union_score", embd_aux.union_score.item())
        logger.log_tabular("Mean/embd_random_score", embd_aux.random_score.item())
        logger.log_tabular("Mean/pos_value", value_aux.pos_value.item())
        logger.log_tabular("Mean/neg_value", value_aux.neg_value.item())
        logger.log_tabular("Mean/random_value", value_aux.random_value.item())
        logger.log_tabular("Mean/union_value", value_aux.union_value.item())
        logger.log_tabular("Mean/decay_weight", train_aux.weight.item())

        logger.log_tabular(
            "NonPrefScore/union_pos_sink", sink_aux.pos_sink_nonpref.item()
        )
        logger.log_tabular(
            "NonPrefScore/union_neg_sink", sink_aux.neg_sink_nonpref.item()
        )
        logger.log_tabular("NonPrefScore/union_source", sink_aux.source_nonpref.item())

        logger.log_tabular(
            "Percentage/union_pos_sink", sink_aux.pos_sink_percent.item()
        )
        logger.log_tabular(
            "Percentage/union_neg_sink", sink_aux.neg_sink_percent.item()
        )
        logger.log_tabular("Percentage/union_sink", sink_aux.sink_percent.item())
        logger.log_tabular(
            "Percentage/union_sink_ema", sink_aux.sink_percent_ema.item()
        )

        logger.log_tabular("Reward/pos", data_aux.pos_reward.item())
        logger.log_tabular("Reward/neg", data_aux.neg_reward.item())
        logger.log_tabular("Reward/union", data_aux.union_reward)
        logger.log_tabular("Reward/union_pos_sink", sink_aux.pos_sink_reward.item())
        logger.log_tabular("Reward/union_neg_sink", sink_aux.neg_sink_reward.item())
        logger.log_tabular("Reward/union_source", sink_aux.source_reward.item())

        logger.log_tabular("Cost/pos", data_aux.pos_cost.item())
        logger.log_tabular("Cost/neg", data_aux.neg_cost.item())
        logger.log_tabular("Cost/union", data_aux.union_cost.item())
        logger.log_tabular("Cost/union_pos_sink", sink_aux.pos_sink_cost.item())
        logger.log_tabular("Cost/union_neg_sink", sink_aux.neg_sink_cost.item())
        logger.log_tabular("Cost/union_source", sink_aux.source_cost.item())

        logger.log_tabular(
            "Norm/embedding_model",
            get_tree_norm(nnx.state(embedding_model, nnx.Param)),
        )
        logger.log_tabular(
            "Norm/value_model",
            get_tree_norm(nnx.state(value_model, nnx.Param)),
        )
        logger.log_tabular(
            "Norm/policy_model",
            get_tree_norm(nnx.state(policy_model, nnx.Param)),
        )
        logger.dump_tabular()

        if steps % config["save_freq"] == 0:
            plot_weighted_trajectory_data(
                data_buffer=data_buffer,
                num_trajs=args.num_union,
                env_name=env_name,
                save_plot=f"{args.log_dir}/weight_dataset_{steps}_{args.seed}.png",
            )
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=embedding_model,
                prefix="embedding",
            )
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=value_model,
                prefix="value",
            )
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=policy_model,
                prefix="bc_policy",
            )

        if steps >= config["total_iteration"]:
            break

    logger.nn_model_save(
        itr=steps,
        nn_model_saver_element=curriculum_embedding_model,
        prefix="curriculum_embedding",
    )
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=embedding_model, prefix="embedding"
    )
    logger.nn_model_save(itr=steps, nn_model_saver_element=value_model, prefix="value")
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=policy_model, prefix="bc_policy"
    )
    if config["normalize_observation"]:
        logger.save_state(
            state_dict={"mu_obs": mu_obs, "std_obs": std_obs}, dirname="norm"
        )

    plot_weighted_trajectory_data(
        data_buffer=data_buffer,
        num_trajs=args.num_union,
        env_name=env_name,
        save_plot=f"{args.log_dir}/weight_dataset_{args.seed}.png",
    )

    logger.close()


if __name__ == "__main__":
    args, cfg_env = single_agent_args()
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    if not args.write_terminal:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(
                f"{args.log_dir}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{args.log_dir}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                main(args, cfg_env)
    else:
        main(args, cfg_env)
