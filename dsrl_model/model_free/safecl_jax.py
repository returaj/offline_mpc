import functools
import os
import os.path as osp
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
import numpy as np
import optax
from flax import nnx
from jax import debug

from dsrl_model.utils.buffer_jax import SafeCLBuffer
from dsrl_model.utils.dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_normalized_data,
    get_pos_neg_and_union_data,
)
from dsrl_model.utils.models_jax import (
    SafeDiceTanhMixtureActor,
    TransformerEmbedding,
    bce_loss,
    get_tree_norm,
    l2_normalize,
    sample_von_mises_fisher_samples,
)
from dsrl_model.utils.native_logger import EpochLogger
from dsrl_model.utils.utils import single_agent_args

EPS = 1e-6

default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "eval_episode_freq": 1,  # use saved bc_policy to run evaluatation
    "hidden_size": 256,
    "embd_size": 128,
    "max_grad_norm": 1.0,
    "gamma": 0.99,
    "action_repeat": 1,  # set to 2, min value is 1
    "train_horizon": 500,  # 20
    "update_bc_freq": 1,
    "update_embd_freq": int(1e3),
    "decay": 0.85,
    "warmup_steps": int(3e4),
    "value_temp": 0.1,
    "value_limit": 0.85,
    "update_tau": 0.01,
    "weight_decay": 0.01,
    "grad_reg_coeffs": 10.0,
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

labels_cfg = {
    "num_modes": 3,
    "concentration_factor": 3.5,
    "labels": [3.0, 2.0, 1.0, 0.0] + [-1.0],  # -1 denotes invalid label
    "range": [1.0, 0.75, 0.5, 0.25, -0.25] + [-2.0],  # -2 denotes invalid range
    "distance": [1.0, 0.5, 0.25, 0.0] + [-2.0],  # -2 denotes invalid distance
}


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
def kernel_density_entropy(samples, mask, sigma=0.2):
    diffs = samples[:, None] - samples[None, :]
    kernel = jnp.exp(-0.5 * (diffs / sigma) ** 2)
    mask2d = mask[:, None] * mask[None, :]
    kernel = kernel * mask2d
    n_effective = jnp.sum(mask)
    p_est = jnp.sum(kernel, axis=1) / (n_effective * sigma * jnp.sqrt(2 * jnp.pi) + EPS)
    entropy = -jnp.sum(mask * jnp.log(p_est + EPS)) / (n_effective + EPS)
    return entropy


@jax.jit
def range_loss(scores, target_scores, scale):
    loss = (scores - target_scores) ** 2
    return scale * loss


@nnx.jit
def train_embedding_model(
    curriculum_embedding_model,
    embedding_model,
    embedding_optimizer,
    target_pos_obs,
    target_pos_act,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    target_union_trainable,
    union_scale,
    has_positive,
    has_negative,
    pos_label,
    key,
):
    dtype = target_neg_obs.dtype
    batch = target_neg_obs.shape[0]

    # shape: Batch X Horizon X obs_act_dim
    target_pos = jnp.concat([target_pos_obs, target_pos_act], axis=-1)
    target_neg = jnp.concat([target_neg_obs, target_neg_act], axis=-1)
    target_union = jnp.concat([target_union_obs, target_union_act], axis=-1)

    ## we were able to do this if it is guareented 
    ## that there will be negative trajectories too.
    # mix_p1 = jax.random.uniform(key=key1, shape=target_neg.shape)
    # target_random1 = mix_p1 * target_neg + (1 - mix_p1) * target_union

    key1, key2 = jax.random.split(key, num=2)
    mix_p1 = jax.random.uniform(key=key1, shape=target_neg.shape)
    target_shuffle_union = jax.random.permutation(key2, target_union, axis=0)
    target_random = mix_p1 * target_shuffle_union + (1 - mix_p1) * target_union

    # Batch
    target_pos_score = pos_label * jnp.ones(shape=(batch,), dtype=dtype)
    target_neg_score = jnp.ones(shape=(batch,), dtype=dtype)
    target_random_score = jnp.zeros(shape=(batch,), dtype=dtype)

    # Batch X embd_dim
    target_pos_z = curriculum_embedding_model(target_pos, training=False)
    target_neg_z = curriculum_embedding_model(target_neg, training=False)
    target_union_z = curriculum_embedding_model(target_union, training=False)
    target_random_z = curriculum_embedding_model(target_random, training=False)

    def loss_fun(embedding_model):
        default_scale = 1.0

        pos_z = embedding_model(target_pos)
        # Batch
        pos_score = jnp.einsum("ij,ij->i", pos_z, target_pos_z)
        pos_mean_loss = has_positive * jnp.mean(
            range_loss(pos_score, target_pos_score, default_scale)
        )
        pos_mean_score = has_positive * jnp.mean(pos_score)

        neg_z = embedding_model(target_neg)
        # Batch
        neg_score = jnp.einsum("ij,ij->i", neg_z, target_neg_z)
        neg_mean_loss = has_negative * jnp.mean(
            range_loss(neg_score, target_neg_score, default_scale)
        )
        neg_mean_score = has_negative * jnp.mean(neg_score)

        union_z = embedding_model(target_union)
        union_score = jnp.einsum("ij,ij->i", union_z, target_union_z)
        union_loss = range_loss(union_score, target_neg_score, union_scale)
        trainable_scores = (target_union_trainable > 0).astype(dtype)
        trainable_count = jnp.clip(trainable_scores.sum(), min=1.0)
        union_mean_loss = (
            jnp.einsum("i,i->", trainable_scores, union_loss) / trainable_count
        )
        union_mean_score = (
            jnp.einsum("i,i->", trainable_scores, union_score) / trainable_count
        )

        random_z = embedding_model(target_random)
        random_score = jnp.einsum("ij,ij->i", random_z, target_random_z)
        random_mean_loss = jnp.mean(
            range_loss(random_score, target_random_score, default_scale)
        )
        random_mean_score = jnp.mean(random_score)

        loss = pos_mean_loss + neg_mean_loss + union_mean_loss + random_mean_loss

        return loss, (
            pos_mean_loss,
            neg_mean_loss,
            union_mean_loss,
            random_mean_loss,
            pos_mean_score,
            neg_mean_score,
            union_mean_score,
            random_mean_score,
        )

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_values), grads = grad_fun(embedding_model)
    embedding_optimizer.update(grads)

    return loss, *aux_values


@jax.jit
def index_fun(arr, all_labels, label_range):
    # For each value in arr, find which interval of `ranges` it falls into
    # range[i] >= x > range[i+1] → assign labels[i]
    idx = jnp.sum(arr[..., None] <= label_range[1:], axis=-1)
    return all_labels[idx]


@jax.jit
def get_trainable_mean_values(value_arr, trainable_mask):
    batch = trainable_mask.shape[0]
    trainable_count = trainable_mask.sum()
    batch_horizon_value = jnp.sum(value_arr, axis=-1)
    total_trainable_value = jnp.einsum("i,i->", batch_horizon_value, trainable_mask)
    mean_trainable_value = total_trainable_value / (trainable_count + 1)
    total_non_trainable_value = batch_horizon_value.sum() - total_trainable_value
    mean_non_trainable_value = total_non_trainable_value / (batch - trainable_count + 1)
    return mean_trainable_value, mean_non_trainable_value


@nnx.jit
def get_union_trainable(
    curriculum_embedding_model,
    embedding_model,
    target_union_obs,
    target_union_act,
    target_union_reward,
    target_union_cost,
    value_limit,
):
    num_models = 5
    batch = target_union_obs.shape[0]
    dtype = target_union_obs.dtype

    # Batch X Horizon X obs_act_dim
    target_union = jnp.concat([target_union_obs, target_union_act], axis=-1)

    # Batch X embd_dim
    target_union_z = curriculum_embedding_model(target_union, training=False)

    @nnx.scan(length=num_models, in_axes=nnx.Carry, out_axes=(nnx.Carry, 0))
    def multimodel(carry):
        x, target_z, model = carry
        # Batch X embd_dim
        z = model(x)
        # Batch
        score = jnp.einsum("ij,ij->i", z, target_z)
        return carry, score

    _, union_scores = multimodel((target_union, target_union_z, embedding_model))
    mean_union_score = jnp.mean(union_scores, axis=0)
    baseline_mean, baseline_std = mean_union_score.mean(), mean_union_score.std()

    baseline = value_limit
    trainable_mask = (mean_union_score > baseline).astype(dtype)
    trainable_count = trainable_mask.sum()
    trainable_percent = trainable_count / batch

    # Batch
    mean_trainable_reward, mean_non_trainable_reward = get_trainable_mean_values(
        target_union_reward, trainable_mask
    )
    mean_trainable_cost, mean_non_trainable_cost = get_trainable_mean_values(
        target_union_cost, trainable_mask
    )
    return (
        trainable_mask,
        mean_union_score,
        baseline_mean,
        baseline_std,
        baseline,
        trainable_percent,
        mean_trainable_reward,
        mean_non_trainable_reward,
        mean_trainable_cost,
        mean_non_trainable_cost,
    )


@nnx.jit
def train_policy_model(
    bc_policy,
    bc_optimizer,
    target_pos_obs,
    target_pos_act,
    target_union_obs,
    target_union_act,
    target_union_score,
    gamma,
    value_limit,
    value_temp,
    has_positive,
    pos_label,
):
    dtype = target_union_obs.dtype
    batch, horizon, _ = target_union_obs.shape

    # Batch_Horizon X obs/act_dim
    target_pos_obs = target_pos_obs.reshape(batch * horizon, -1)
    target_pos_act = target_pos_act.reshape(batch * horizon, -1)

    target_union_obs = target_union_obs.reshape(batch * horizon, -1)
    target_union_act = target_union_act.reshape(batch * horizon, -1)

    non_trainable = (target_union_score <= value_limit).astype(dtype)
    non_trainable_count = jnp.clip(non_trainable.sum(), min=1.0)
    union_weight = non_trainable * jnp.exp(-target_union_score / value_temp)

    pos_weight = (
        has_positive * jnp.exp(-pos_label / value_temp) * jnp.ones_like(union_weight)
    )

    norm_weight = (union_weight.sum() + pos_weight.sum()) / (
        non_trainable_count + has_positive * batch
    )
    norm_weight = jnp.clip(norm_weight, min=EPS)

    final_union_weight = union_weight / norm_weight
    final_pos_weight = pos_weight / norm_weight

    def bc_policy_trajectory_loss_fun(bc_policy):
        pred_union_act, *_ = bc_policy(target_union_obs)
        flat_union_loss = optax.l2_loss(pred_union_act, target_union_act).sum(axis=-1)
        batch_horizon_union_loss = flat_union_loss.reshape(batch, horizon)
        batch_union_loss = jax.vmap(discounted_sum, in_axes=(0, None))(
            batch_horizon_union_loss, gamma
        ).squeeze()
        union_loss = jnp.mean(final_union_weight * batch_union_loss)

        pred_pos_act, *_ = bc_policy(target_pos_obs)
        flat_pos_loss = optax.l2_loss(pred_pos_act, target_pos_act).sum(axis=-1)
        batch_horizon_pos_loss = flat_pos_loss.reshape(batch, horizon)
        batch_pos_loss = jax.vmap(discounted_sum, in_axes=(0, None))(
            batch_horizon_pos_loss, gamma
        ).squeeze()
        pos_loss = jnp.mean(final_pos_weight * batch_pos_loss)

        loss = pos_loss + union_loss
        return loss, (pos_loss, union_loss)

    bc_grad_fun = nnx.value_and_grad(bc_policy_trajectory_loss_fun, has_aux=True)
    (bc_loss, aux_values), bc_grad = bc_grad_fun(bc_policy)
    bc_grad = jax.tree.map(lambda g: g / horizon, bc_grad)
    bc_optimizer.update(bc_grad)
    return bc_loss, *aux_values


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

    config = {**default_cfg, **trajectory_cfg, **labels_cfg}
    config["train_horizon"] = args.train_horizon or config.get("train_horizon")
    config["policy_type"] = args.policy_type
    config["normalize_observation"] = args.normalize_observation
    config["value_temp"] = args.value_weight_temp or config["value_temp"]
    config["value_limit"] = args.value_weight_limit or config["value_limit"]
    config["use_vonmisesfisher_mode"] = args.use_vonmisesfisher_mode
    config["pos_label"] = args.preferred_label
    config["lr"] = args.lr

    # set training steps
    batch_size = args.batch_size or config.get("batch_size")
    config["batch_size"] = batch_size

    # evaluation environment
    eval_env = gym.make(args.task)
    eval_env.set_target_cost(config["target_cost"])
    eval_env.reset(seed=args.seed)

    # set model
    obs_space, act_space = eval_env.observation_space, eval_env.action_space
    bc_policy = SafeDiceTanhMixtureActor(
        rngs=rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_size=config["hidden_size"],
    )
    bc_optimizer = nnx.Optimizer(
        model=bc_policy,
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
    target_embedding_model = deepcopy(embedding_model)
    curriculum_embedding_model = TransformerEmbedding(
        rngs=curriculum_rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        horizon=config["train_horizon"],
        embd_dim=embd_size,
        num_attentions=3,
    )

    # data
    agent_task = re.search(r"Offline(.*?)Gymnasium-v[0-9]", args.task).group(1)
    ep_len = dsrl_infos.DEFAULT_MAX_EPISODE_STEPS[agent_task]
    data = get_dataset_in_d4rl_format(
        eval_env, trajectory_cfg, args.task, ep_len, config["action_repeat"]
    )
    pos_data, neg_data, union_data = get_pos_neg_and_union_data(
        data, trajectory_cfg, save_dir=args.log_dir
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

    # set logger
    eval_rew_deque = deque(maxlen=config["eval_episode_freq"])
    eval_cost_deque = deque(maxlen=config["eval_episode_freq"])
    eval_len_deque = deque(maxlen=config["eval_episode_freq"])
    dict_args = config
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)

    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config(dict_args)
    logger.log("Start embedding, cost and bc_policy model training.")

    steps, union_scale = 0, 0.0
    while steps < config["total_iteration"]:
        # shape: Batch X Horizon X obs/act_dim
        for (
            target_pos_obs,
            target_pos_act,
            target_pos_reward,
            target_pos_cost,
            target_neg_obs,
            target_neg_act,
            target_neg_reward,
            target_neg_cost,
            target_union_obs,
            target_union_act,
            target_union_reward,
            target_union_cost,
            _,
            _,
        ) in buffer.sample():

            steps += 1

            (
                union_trainable,
                union_score,
                baseline_mean,
                baseline_std,
                baseline_score,
                union_trainable_percent,
                mean_trainable_reward,
                mean_non_trainable_reward,
                mean_trainable_cost,
                mean_non_trainable_cost,
            ) = get_union_trainable(
                curriculum_embedding_model=curriculum_embedding_model,
                embedding_model=target_embedding_model,
                target_union_obs=target_union_obs,
                target_union_act=target_union_act,
                target_union_reward=target_union_reward,
                target_union_cost=target_union_cost,
                value_limit=config["value_limit"],
            )

            (
                embedding_loss,
                pos_mean_loss,
                neg_mean_loss,
                union_mean_loss,
                random_mean_loss,
                pos_mean_score,
                neg_mean_score,
                union_mean_score,
                random_mean_score,
            ) = train_embedding_model(
                curriculum_embedding_model=curriculum_embedding_model,
                embedding_model=embedding_model,
                embedding_optimizer=embedding_optimizer,
                target_pos_obs=target_pos_obs,
                target_pos_act=target_pos_act,
                target_neg_obs=target_neg_obs,
                target_neg_act=target_neg_act,
                target_union_obs=target_union_obs,
                target_union_act=target_union_act,
                target_union_trainable=union_trainable,
                union_scale=union_scale,
                has_positive=has_positive,
                has_negative=has_negative,
                pos_label=config["pos_label"],
                key=rngs.random_sample(),
            )

            if steps % config["update_embd_freq"] == 0:
                target_embedding_model = polyak_update(
                    target_embedding_model, embedding_model, config["update_tau"]
                )

            bc_loss = bc_pos_loss = bc_union_loss = jnp.array(0.0)
            if (steps > config["warmup_steps"]) and (
                steps % config["update_bc_freq"] == 0
            ):
                union_scale = 1.0

                bc_loss, bc_pos_loss, bc_union_loss = train_policy_model(
                    bc_policy=bc_policy,
                    bc_optimizer=bc_optimizer,
                    target_pos_obs=target_pos_obs,
                    target_pos_act=target_pos_act,
                    target_union_obs=target_union_obs,
                    target_union_act=target_union_act,
                    target_union_score=union_score,
                    gamma=config["gamma"],
                    value_limit=config["value_limit"],
                    value_temp=config["value_temp"],
                    has_positive=has_positive,
                    pos_label=config["pos_label"],
                )

            logger.logged = False

            if (steps % config["log_freq"] == 0) and (not logger.logged):
                eval_episodes = config["eval_episode_freq"]
                if args.use_eval:
                    eval_start_time = time.time()
                    for id in range(eval_episodes):
                        (eval_reward, eval_cost, eval_len) = evaluate_bc_policy(
                            eval_env, bc_policy.action, mu_obs, std_obs
                        )
                        eval_rew_deque.append(eval_reward)
                        eval_cost_deque.append(eval_cost)
                        eval_len_deque.append(eval_len)
                    eval_end_time = time.time()

                    logger.log_tabular("Metrics/EvalEpRet", np.mean(eval_rew_deque))
                    logger.log_tabular("Metrics/EvalEpCost", np.mean(eval_cost_deque))
                    logger.log_tabular("Metrics/EvalEpLen", np.mean(eval_len_deque))
                    logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)

                logger.log_tabular("Train/Steps", steps)

                logger.log_tabular("Loss/Loss_embedding", embedding_loss.item())
                logger.log_tabular("Loss/Loss_embd_pos_mean_loss", pos_mean_loss.item())
                logger.log_tabular("Loss/Loss_embd_neg_mean_loss", neg_mean_loss.item())
                logger.log_tabular(
                    "Loss/Loss_embd_union_mean_loss", union_mean_loss.item()
                )
                logger.log_tabular(
                    "Loss/Loss_embd_random_mean_loss", random_mean_loss.item()
                )
                logger.log_tabular("Loss/Loss_bc_policy", bc_loss.item())
                logger.log_tabular("Loss/Loss_bc_pos_policy", bc_pos_loss.item())
                logger.log_tabular("Loss/Loss_bc_union_policy", bc_union_loss.item())

                logger.log_tabular("Mean/embd_pos_score", pos_mean_score.item())
                logger.log_tabular("Mean/embd_neg_score", neg_mean_score.item())
                logger.log_tabular("Mean/embd_union_score", union_mean_score.item())
                logger.log_tabular("Mean/embd_random_score", random_mean_score.item())
                logger.log_tabular("Mean/union_baseline_mean", baseline_mean.item())
                logger.log_tabular("Mean/union_baseline_std", baseline_std.item())
                logger.log_tabular("Mean/union_baseline_score", baseline_score.item())

                logger.log_tabular(
                    f"Percentage/new_union_trainable", union_trainable_percent.item()
                )

                logger.log_tabular(
                    "Reward/pos", jnp.sum(target_pos_reward, axis=-1).mean().item()
                )
                logger.log_tabular(
                    "Reward/neg", jnp.sum(target_neg_reward, axis=-1).mean().item()
                )
                logger.log_tabular(
                    "Reward/union",
                    jnp.sum(target_union_reward, axis=-1).mean().item(),
                )
                logger.log_tabular(
                    "Reward/union_trainable", mean_trainable_reward.item()
                )
                logger.log_tabular(
                    "Reward/union_non_trainable", mean_non_trainable_reward.item()
                )

                logger.log_tabular(
                    "Cost/pos", jnp.sum(target_pos_cost, axis=-1).mean().item()
                )
                logger.log_tabular(
                    "Cost/neg", jnp.sum(target_neg_cost, axis=-1).mean().item()
                )
                logger.log_tabular(
                    "Cost/union", jnp.sum(target_union_cost, axis=-1).mean().item()
                )
                logger.log_tabular("Cost/union_trainable", mean_trainable_cost.item())
                logger.log_tabular(
                    "Cost/union_non_trainable", mean_non_trainable_cost.item()
                )

                logger.log_tabular(
                    "Norm/embedding_model",
                    get_tree_norm(nnx.state(embedding_model, nnx.Param)),
                )
                logger.log_tabular(
                    "Norm/bc_policy",
                    get_tree_norm(nnx.state(bc_policy, nnx.Param)),
                )
                logger.dump_tabular()

            if steps % config["save_freq"] == 0:
                logger.nn_model_save(
                    itr=steps,
                    nn_model_saver_element=embedding_model,
                    prefix="embedding",
                )
                logger.nn_model_save(
                    itr=steps,
                    nn_model_saver_element=bc_policy,
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
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=bc_policy, prefix="bc_policy"
    )
    if config["normalize_observation"]:
        logger.save_state(
            state_dict={"mu_obs": mu_obs, "std_obs": std_obs}, dirname="norm"
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
