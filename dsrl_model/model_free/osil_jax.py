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

from dsrl_model.utils.buffer_jax import OSILBuffer
from dsrl_model.utils.dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_normalized_data,
    get_pos_neg_and_union_data,
)
from dsrl_model.utils.models_jax import (
    ContrastiveCostModel,
    EnsembleValue,
    SafeDiceTanhMixtureActor,
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
    "max_grad_norm": 1.0,
    "gamma": 0.99,
    "action_repeat": 1,  # set to 2, min value is 1
    "update_bc_freq": 2,
    "update_cost_freq": 5,
    "update_tau": 0.005,
    "value_weight_temp": 2.5,  # TD3-BC coef
    "train_horizon": 5,  # 5
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


def polyak_update(target_model, curr_model, tau):
    target_param = nnx.state(target_model, nnx.Param)
    curr_param = nnx.state(curr_model, nnx.Param)

    new_target_param = jax.tree_util.tree_map(
        lambda t, c: (1 - tau) * t + tau * c, target_param, curr_param
    )
    nnx.update(target_model, new_target_param)
    return target_model


@jax.jit
def discounted_sum(arr, gamma):
    dtype = arr.dtype
    horizon = arr.shape[0]

    def body_fun(t, cumsum):
        cumsum = arr[horizon - 1 - t] + gamma * cumsum
        return cumsum

    init_cumsum = jnp.zeros_like(arr[0], dtype=dtype)
    cumsum = jax.lax.fori_loop(0, horizon, body_fun, init_cumsum)
    return cumsum


@jax.jit
def pref_cost_loss_fun(pos_cost, neg_cost, union_cost, has_positive, gamma):
    # neg_cost shape: Batch X Horizon

    # batch
    pos_traj_cost = discounted_sum(pos_cost.T, gamma)
    neg_traj_cost = discounted_sum(neg_cost.T, gamma)
    union_traj_cost = discounted_sum(union_cost.T, gamma)

    # min L = - log[ exp(neg) / (exp(neg) + exp(pos)) ] = log[ 1 + exp(pos - neg) ]
    # L = nn.softplus(pos-neg)
    pos_neg_loss = jnp.mean(
        has_positive * jax.nn.softplus(pos_traj_cost - neg_traj_cost)
    )
    pos_union_loss = jnp.mean(
        has_positive * jax.nn.softplus(pos_traj_cost - union_traj_cost)
    )
    union_neg_loss = jnp.mean(jax.nn.softplus(union_traj_cost - neg_traj_cost))

    loss = pos_neg_loss + pos_union_loss + union_neg_loss

    return loss, pos_neg_loss, pos_union_loss, union_neg_loss


@jax.jit
def compute_contrastive_ce_loss(p, q):
    q = jax.nn.log_softmax(q, axis=1)
    p = p / jnp.clip(p.sum(axis=1, keepdims=True), min=1.0)
    loss = -jnp.sum(p * q, axis=1)
    return jnp.mean(loss)


@jax.jit
def contrastive_cost_loss_fun(pos_zs, neg_zs, union_zs, gamma):
    del pos_zs

    temp = 0.1  # value from SupContrast

    dtype = neg_zs.dtype
    # batch X horizon X zdim
    batch, horizon, _ = neg_zs.shape

    # logits for union and non-pref
    neg_zs = jnp.concat(neg_zs, axis=0)  # BH X zdim
    union_zs = jnp.concat(union_zs, axis=0)  # BH X zdim
    combined_zs = jnp.concat([neg_zs, union_zs], axis=0)  # 2BH X zdim
    combined_logits = combined_zs @ combined_zs.T / temp  # 2BH X 2BH
    # remove the self instance from the loss fn
    diag_mask = -1e9 * jnp.eye(combined_logits.shape[0], dtype=dtype)
    combined_logits = combined_logits + diag_mask

    # state-action pairs in the same trajectory are gamma discounted closer to each other
    # H_mask = [[1, gamma, gamma^2, ..., gamma^(H-1)]] * H
    discounted_mask = gamma ** jnp.arange(horizon, dtype=dtype)
    horizon_mask = jnp.ones((horizon, 1), dtype=dtype) * discounted_mask
    mask = jnp.eye(2 * batch, dtype=dtype)
    combined_mask = jnp.kron(mask, horizon_mask)  # 2BH X 2BH
    # remove the self instance from the loss fn
    combined_mask = combined_mask * (1.0 - jnp.eye(combined_mask.shape[0], dtype=dtype))

    loss = compute_contrastive_ce_loss(combined_mask, combined_logits)

    return loss


def cost_loss_grad_fun(cost_model, data, has_positive, gamma):
    # Batch X Horizon X obs_act_dim
    target_pos = jnp.concat([data.pos_obs, data.pos_act], axis=-1)
    target_neg = jnp.concat([data.neg_obs, data.neg_act], axis=-1)
    target_union = jnp.concat([data.union_obs, data.union_act], axis=-1)

    def loss_fun(cost_model):
        # Batch x Horizon X zdim/,
        pos_zs, pos_cost = cost_model(target_pos)
        neg_zs, neg_cost = cost_model(target_neg)
        union_zs, union_cost = cost_model(target_union)

        (
            pref_loss,
            pos_neg_pref_loss,
            pos_union_pref_loss,
            union_neg_pref_loss,
        ) = pref_cost_loss_fun(
            pos_cost=pos_cost,
            neg_cost=neg_cost,
            union_cost=union_cost,
            has_positive=has_positive,
            gamma=gamma,
        )

        contrastive_loss = contrastive_cost_loss_fun(
            pos_zs=pos_zs,
            neg_zs=neg_zs,
            union_zs=union_zs,
            gamma=gamma,
        )

        loss = pref_loss + contrastive_loss

        return loss, (
            pos_neg_pref_loss,
            pos_union_pref_loss,
            union_neg_pref_loss,
            pref_loss,
            contrastive_loss,
        )

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_values), grads = grad_fun(cost_model)
    grads = jax.tree.map(lambda g: g / data.horizon, grads)
    return loss, grads, *aux_values


def compute_target_value(cost_model, bc_policy, value_model, obs, act, done, gamma):
    dtype = obs.dtype

    # Batch X Horizon X obs_dim
    batch, horizon, _ = obs.shape

    # Batch X Horizon X act_dim
    act_next, *_ = bc_policy(obs.reshape(batch * horizon, -1))
    act_next = act_next.reshape(batch, horizon, -1)

    # mask last horizon
    mask_last_horizon = jnp.concat(
        [jnp.ones(horizon - 1, dtype=dtype), jnp.zeros(1, dtype=dtype)]
    )
    # This is a permutation matrix to shift one timestep ahead
    shift_one_timestep = jnp.eye(horizon, k=-1, dtype=dtype)

    def batch_fun(obs, act, act_next, done):
        # Batch X Horizon
        _, cost = cost_model(jnp.concat([obs, act], axis=-1))
        cost = mask_last_horizon * cost
        # Batch X Horizon
        v_next = jnp.maximum(*value_model(jnp.concat([obs, act_next], axis=-1)))
        v_next = v_next @ shift_one_timestep

        target = cost + gamma * (1 - done) * v_next
        return target

    target_value = jax.vmap(batch_fun, in_axes=(0, 0, 0, 0))(obs, act, act_next, done)
    return target_value


def value_loss_grad_fun(
    cost_model,
    bc_policy,
    target_value_model,
    value_model,
    data,
    gamma,
):
    dtype = data.union_obs.dtype
    horizon = data.horizon

    # mask the horizon-1 element
    mask = jnp.concat([jnp.ones(horizon - 1, dtype=dtype), jnp.zeros(1, dtype=dtype)])

    # Batch X Horizon
    target_v = compute_target_value(
        cost_model=cost_model,
        bc_policy=bc_policy,
        value_model=target_value_model,
        obs=data.union_obs,
        act=data.union_act,
        done=data.union_done,
        gamma=gamma,
    )

    def loss_fun(value_model):
        # Batch X Horizon X obs_act_dim
        target_oa = jnp.concat([data.union_obs, data.union_act], axis=-1)
        # Batch X Horizon
        pred_v1, pred_v2 = value_model(target_oa)
        v1_loss = mask * optax.huber_loss(pred_v1, target_v, delta=2.0)
        v2_loss = mask * optax.huber_loss(pred_v2, target_v, delta=2.0)
        discounted_v1_loss = discounted_sum(v1_loss.T, gamma)
        discounted_v2_loss = discounted_sum(v2_loss.T, gamma)

        loss = jnp.mean(discounted_v1_loss) + jnp.mean(discounted_v2_loss)
        return loss

    grad_fun = nnx.value_and_grad(loss_fun)
    loss, grads = grad_fun(value_model)
    grads = jax.tree.map(lambda g: g / horizon, grads)

    return loss, grads


def compute_value_weight(
    value_model,
    bc_policy,
    obs,
    act,
    alpha,
    beta,
    use_osil_weight,
):
    pred_act, *_ = bc_policy(obs)

    # Batch_Horizon
    q = jnp.maximum(*value_model(jnp.concat([obs, act], axis=-1)))
    v = jnp.maximum(*value_model(jnp.concat([obs, pred_act], axis=-1)))

    def per_transition_weight():
        neg_adv = jnp.clip(-(q - v) / beta, max=2.0)
        union_weight = jnp.exp(neg_adv) / jnp.mean(jnp.abs(v))
        return union_weight

    def constant_adaptive_weight():
        # osil type union weight
        neg_adv = -(q - v) / beta
        log_Z = jax.nn.logsumexp(neg_adv) - jnp.log(neg_adv.shape[0]) + EPS
        union_weight = alpha * jnp.exp(jnp.clip(log_Z, max=5.0))
        return union_weight

    union_weight = jnp.where(
        use_osil_weight, constant_adaptive_weight(), per_transition_weight()
    )
    return union_weight, jnp.abs(q).mean(), jnp.abs(v).mean()


def policy_loss_grad_fun(
    value_model,
    bc_policy,
    data,
    config,
    has_positive,
):
    # Batch X Horizon X obs_dim
    batch, horizon, _ = data.union_obs.shape

    # Batch_Horizon X obs/act_dim
    target_pos_obs = data.pos_obs.reshape(batch * horizon, -1)
    target_pos_act = data.pos_act.reshape(batch * horizon, -1)

    target_union_obs = data.union_obs.reshape(batch * horizon, -1)
    target_union_act = data.union_act.reshape(batch * horizon, -1)

    union_weight, qmean, vmean = compute_value_weight(
        value_model=value_model,
        bc_policy=bc_policy,
        obs=target_union_obs,
        act=target_union_act,
        alpha=config.alpha,
        beta=config.beta,
        use_osil_weight=config.use_osil_weight,
    )

    def loss_fun(bc_policy):
        pred_pos_act, *_ = bc_policy(target_pos_obs)
        pos_loss = optax.l2_loss(pred_pos_act, target_pos_act).sum(axis=-1)
        pos_bc_loss = has_positive * jnp.mean(pos_loss)

        pred_union_act, *_ = bc_policy(target_union_obs)
        union_loss = optax.l2_loss(pred_union_act, target_union_act).sum(axis=-1)
        union_bc_loss = jnp.mean(union_loss)

        union_value = jnp.maximum(
            *value_model(jnp.concat([target_union_obs, pred_union_act], axis=-1))
        )
        union_value_loss = jnp.mean(union_weight * union_value)

        loss = pos_bc_loss + union_bc_loss + union_value_loss
        return loss, (pos_bc_loss, union_bc_loss, union_value_loss)

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_values), grads = grad_fun(bc_policy)

    return loss, grads, *aux_values, qmean, vmean


def train_step(
    cost_model,
    cost_optimizer,
    value_model_target,
    value_model,
    value_optimizer,
    bc_policy_target,
    bc_policy,
    bc_optimizer,
    batch_data,
    config,
    has_positive,
    steps,
):
    cost_cond = (steps % config.update_cost_freq) == 0
    cost_loss, cost_grads, *cost_aux = cost_loss_grad_fun(
        cost_model=cost_model,
        data=batch_data,
        has_positive=has_positive,
        gamma=config.gamma,
    )
    cost_grads = jax.tree.map(
        lambda g: jnp.where(cost_cond, g, jnp.zeros_like(g)),
        cost_grads,
    )
    cost_optimizer.update(cost_grads)

    value_loss, value_grads = value_loss_grad_fun(
        cost_model=cost_model,
        bc_policy=bc_policy_target,
        target_value_model=value_model_target,
        value_model=value_model,
        data=batch_data,
        gamma=config.gamma,
    )
    value_optimizer.update(value_grads)

    policy_cond = (steps % config.update_bc_freq) == 0
    policy_loss, policy_grads, *policy_aux = policy_loss_grad_fun(
        value_model=value_model,
        bc_policy=bc_policy,
        data=batch_data,
        config=config,
        has_positive=has_positive,
    )
    policy_grads = jax.tree.map(
        lambda g: jnp.where(policy_cond, g, jnp.zeros_like(g)),
        policy_grads,
    )
    bc_optimizer.update(policy_grads)

    value_model_target = polyak_update(
        value_model_target, value_model, policy_cond * config.update_tau
    )
    bc_policy_target = polyak_update(
        bc_policy_target, bc_policy, policy_cond * config.update_tau
    )

    mean_pos_reward = batch_data.pos_reward.sum(-1).mean()
    mean_neg_reward = batch_data.neg_reward.sum(-1).mean()
    mean_union_reward = batch_data.union_reward.sum(-1).mean()

    mean_pos_cost = batch_data.pos_cost.sum(-1).mean()
    mean_neg_cost = batch_data.neg_cost.sum(-1).mean()
    mean_union_cost = batch_data.union_cost.sum(-1).mean()

    return (
        cost_loss,
        *cost_aux,
        value_loss,
        policy_loss,
        *policy_aux,
        mean_pos_reward,
        mean_neg_reward,
        mean_union_reward,
        mean_pos_cost,
        mean_neg_cost,
        mean_union_cost,
    )


@nnx.jit
def train_n_steps(
    cost_model,
    cost_optimizer,
    value_model_target,
    value_model,
    value_optimizer,
    bc_policy_target,
    bc_policy,
    bc_optimizer,
    data_buffer,
    config,
    has_positive,
    key,
):
    num_steps = config.log_freq

    pos_idxs, neg_idxs, union_idxs = data_buffer.sample_idxs(
        data_buffer, key, num_steps
    )

    def body_fun(i, carry):
        (
            _,
            cost_model,
            cost_optimizer,
            value_model_target,
            value_model,
            value_optimizer,
            bc_policy_target,
            bc_policy,
            bc_optimizer,
        ) = carry

        batch_data = data_buffer.sample_batch(
            data_buffer, pos_idxs[i], neg_idxs[i], union_idxs[i]
        )

        val = train_step(
            cost_model=cost_model,
            cost_optimizer=cost_optimizer,
            value_model_target=value_model_target,
            value_model=value_model,
            value_optimizer=value_optimizer,
            bc_policy_target=bc_policy_target,
            bc_policy=bc_policy,
            bc_optimizer=bc_optimizer,
            batch_data=batch_data,
            config=config,
            has_positive=has_positive,
            steps=i,
        )

        return (
            val,
            cost_model,
            cost_optimizer,
            value_model_target,
            value_model,
            value_optimizer,
            bc_policy_target,
            bc_policy,
            bc_optimizer,
        )

    init_val = (jnp.zeros((), dtype=jnp.float32),) * 19
    init_carry = (
        init_val,
        cost_model,
        cost_optimizer,
        value_model_target,
        value_model,
        value_optimizer,
        bc_policy_target,
        bc_policy,
        bc_optimizer,
    )
    val, *_ = nnx.fori_loop(0, num_steps, body_fun, init_carry)

    return val, num_steps


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

    # set default device id
    jax.default_device = jax.devices(args.device)[args.device_id]

    has_positive = float(args.num_preferred > 0)
    trajectory_cfg["num_positive_trajectories"] = args.num_preferred
    trajectory_cfg["num_negative_trajectories"] = args.num_non_preferred
    trajectory_cfg["num_union_trajectories"] = args.num_union
    trajectory_cfg["non_pref_noise"] = args.non_pref_noise
    trajectory_cfg["data_inpaint"] = args.data_inpaint
    trajectory_cfg["inpaint_ranges"] = trajectory_data[args.data_inpaint]

    config = {**default_cfg, **trajectory_cfg}
    config["train_horizon"] = args.train_horizon or config.get("train_horizon")
    config["alpha"] = args.value_weight_temp or config["value_weight_temp"]
    config["beta"] = args.bc_weight_temp
    config["use_osil_weight"] = args.use_osil_weight
    config["policy_type"] = args.policy_type
    config["normalize_observation"] = args.normalize_observation
    config["lr"] = args.lr

    # set training steps
    batch_size = args.batch_size or config.get("batch_size")
    config["batch_size"] = batch_size

    config_data = make_static_config_from_dict(name="State", d=config)()

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
    bc_policy_target = deepcopy(bc_policy)

    cost_model = ContrastiveCostModel(
        rngs=rngs,
        x_dims=obs_space.shape[0] + act_space.shape[0],
        hidden_size=config["hidden_size"],
    )
    cost_optimizer = nnx.Optimizer(
        model=cost_model,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adamw(
                learning_rate=config["lr"], weight_decay=config["weight_decay"]
            ),
        ),
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
    value_model_target = deepcopy(value_model)

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

    neg_observations = neg_data["observations"]
    neg_actions = neg_data["actions"]
    neg_dones = neg_data["timeouts"] | neg_data["terminals"]
    neg_rewards = neg_data["rewards"]
    neg_costs = neg_data["costs"]

    union_observations = union_data["observations"]
    union_actions = union_data["actions"]
    union_dones = union_data["timeouts"] | union_data["terminals"]
    union_rewards = union_data["rewards"]
    union_costs = union_data["costs"]

    ep_len = ep_len // config["action_repeat"] + (ep_len % config["action_repeat"] > 0)
    assert (
        neg_observations.shape[1] == ep_len
    ), f"{neg_observations.shape[1]} episode length is different from {ep_len}"

    pos_data_size = 1
    if has_positive:
        pos_data_size = np.prod(pos_data["observations"].shape[:-1])

    buffer = OSILBuffer(
        rngs=rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        pos_data_size=pos_data_size,
        neg_data_size=np.prod(neg_observations.shape[:-1]),
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
    eval_rew_deque = deque(maxlen=config["eval_episode_freq"])
    eval_cost_deque = deque(maxlen=config["eval_episode_freq"])
    eval_len_deque = deque(maxlen=config["eval_episode_freq"])
    dict_args = config
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)

    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config(dict_args)
    logger.log("Start cost and bc_policy model training.")

    steps = 0
    while steps < config["total_iteration"]:
        val, num_itr = train_n_steps(
            cost_model=cost_model,
            cost_optimizer=cost_optimizer,
            value_model_target=value_model_target,
            value_model=value_model,
            value_optimizer=value_optimizer,
            bc_policy_target=bc_policy_target,
            bc_policy=bc_policy,
            bc_optimizer=bc_optimizer,
            data_buffer=data_buffer,
            config=config_data,
            has_positive=has_positive,
            key=rngs.random_sample(),
        )

        (
            cost_loss,
            pos_neg_pref_loss,
            pos_union_pref_loss,
            union_neg_pref_loss,
            pref_loss,
            contrastive_loss,
            value_loss,
            policy_loss,
            policy_pos_bc_loss,
            policy_union_bc_loss,
            policy_union_value_loss,
            policy_qmean,
            policy_vmean,
            mean_pos_reward,
            mean_neg_reward,
            mean_union_reward,
            mean_pos_cost,
            mean_neg_cost,
            mean_union_cost,
        ) = val

        steps += num_itr

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

            logger.log_tabular("Loss/Loss_cost", cost_loss.item())
            logger.log_tabular("Loss/Loss_pos_neg_pref_cost", pos_neg_pref_loss.item())
            logger.log_tabular(
                "Loss/Loss_pos_union_pref_cost", pos_union_pref_loss.item()
            )
            logger.log_tabular(
                "Loss/Loss_union_neg_pref_cost", union_neg_pref_loss.item()
            )
            logger.log_tabular("Loss/Loss_pref_cost", pref_loss.item())
            logger.log_tabular("Loss/Loss_contrastive_cost", contrastive_loss.item())

            logger.log_tabular("Loss/Loss_value", value_loss.item())

            logger.log_tabular("Loss/Loss_policy", policy_loss.item())
            logger.log_tabular("Loss/Loss_policy_pos_bc", policy_pos_bc_loss.item())
            logger.log_tabular("Loss/Loss_policy_union_bc", policy_union_bc_loss.item())
            logger.log_tabular(
                "Loss/Loss_policy_union_value", policy_union_value_loss.item()
            )
            logger.log_tabular("Loss/policy_q_value", policy_qmean.item())
            logger.log_tabular("Loss/policy_v_value", policy_vmean.item())

            logger.log_tabular("Reward/pos", mean_pos_reward.item())
            logger.log_tabular("Reward/neg", mean_neg_reward.item())
            logger.log_tabular("Reward/union", mean_union_reward.item())

            logger.log_tabular("Cost/pos", mean_pos_cost.item())
            logger.log_tabular("Cost/neg", mean_neg_cost.item())
            logger.log_tabular("Cost/union", mean_union_cost.item())

            logger.log_tabular(
                "Norm/cost_model",
                get_tree_norm(nnx.state(cost_model, nnx.Param)),
            )
            logger.log_tabular(
                "Norm/value_model",
                get_tree_norm(nnx.state(value_model, nnx.Param)),
            )
            logger.log_tabular(
                "Norm/bc_policy",
                get_tree_norm(nnx.state(bc_policy, nnx.Param)),
            )
            logger.dump_tabular()

        if steps % config["save_freq"] == 0:
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=cost_model,
                prefix="cost",
            )
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=value_model,
                prefix="value",
            )
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=bc_policy,
                prefix="bc_policy",
            )

        if steps >= config["total_iteration"]:
            break

    logger.nn_model_save(itr=steps, nn_model_saver_element=cost_model, prefix="cost")
    logger.nn_model_save(itr=steps, nn_model_saver_element=value_model, prefix="value")
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
