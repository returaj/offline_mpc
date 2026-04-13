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

from dsrl_model.utils.buffer_jax import ILIDBuffer, reshuffle_union_data
from dsrl_model.utils.dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_normalized_data,
    get_pos_neg_and_union_data,
)
from dsrl_model.utils.models_jax import (
    ExpCostModel,
    SafeDiceTanhMixtureActor,
    Scalar,
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
    "max_grad_norm": 10.0,
    "decay": 0.9,
    "alpha": 1.0,
    "lmbda": 0.85,
    "pu_nu": 0.5,
    "rollback": 5,
    "action_repeat": 1,  # set to 2, min value is 1
    "train_horizon": 1,
    "weight_decay": 0.005,
    "total_iteration_disc": int(1e5),
    "warmup_iteration_policy": int(2e5),
    "total_iteration_policy": int(1e6),
}

trajectory_cfg = {
    "density": 1.0,
    "target_cost": 25.0,
    # ((low_cost, low_reward), (high_cost, low_reward), (medium_cost, high_reward))
    "inpaint_ranges": None,
    "num_positive_trajectories": 5,
    "num_negative_trajectories": 0,
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


def get_pos_data_log_prob(policy_model, data_buffer):
    obs, act = data_buffer.pos_obs[:-1], data_buffer.pos_act
    valid = data_buffer.pos_priorities

    assert (
        obs.shape[0] == act.shape[0] == valid.shape[0]
    ), "Shape of obs, act or valid does not match."

    log_prob = policy_model.get_log_prob(obs, act)
    baseline = jnp.sum(log_prob * valid) / jnp.sum(valid)
    return baseline


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
def discounted_sum(arr, gamma):
    dtype = arr.dtype
    horizon = arr.shape[0]

    def body_fun(t, cumsum):
        cumsum = arr[horizon - 1 - t] + gamma * cumsum
        return cumsum

    init_cumsum = jnp.zeros_like(arr[0], dtype=dtype)
    cumsum = jax.lax.fori_loop(0, horizon, body_fun, init_cumsum)
    return cumsum


def policy_loss_grads_fun(
    policy_model,
    data,
    alpha,
):
    # Batch X Horizon X obs_dim
    batch, horizon, _ = data.pos_obs.shape

    pos_obs = data.pos_obs.reshape(batch * horizon, -1)
    pos_act = data.pos_act.reshape(batch * horizon, -1)

    union_obs = data.union_obs.reshape(batch * horizon, -1)
    union_act = data.union_act.reshape(batch * horizon, -1)
    union_weight = data.union_weight.reshape((batch * horizon,))

    def loss_fun(policy_model):
        # Batch_Horizon,
        logp_pos = policy_model.get_log_prob(pos_obs, pos_act)
        logp_union = union_weight * policy_model.get_log_prob(union_obs, union_act)
        loss = -alpha * jnp.mean(logp_pos) - jnp.mean(logp_union)
        return loss, (union_weight.min(), union_weight.mean(), union_weight.max())

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_value), grads = grad_fun(policy_model)
    return loss, grads, *aux_value


def alpha_loss_grads_fun(log_alpha_model, policy_model, data, log_pi_baseline):
    ep = 0.01

    # Batch X Horizon X obs_dim
    batch, horizon, _ = data.pos_obs.shape

    obs = data.pos_obs.reshape(batch * horizon, -1)
    act = data.pos_act.reshape(batch * horizon, -1)

    log_pi = policy_model.get_log_prob(obs, act)
    weight = jax.lax.stop_gradient(log_pi.mean() + ep - log_pi_baseline)

    def loss_fun(log_alpha_model):
        alpha = jax.lax.stop_gradient(jnp.exp(log_alpha_model()))
        loss = jnp.exp(log_alpha_model()) * weight
        return loss, (alpha,)

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_value), grads = grad_fun(log_alpha_model)
    return loss, grads, *aux_value


def train_steps_policy(
    policy_model,
    policy_optimizer,
    log_alpha_model,
    log_alpha_optimizer,
    batch_data,
    log_pi_baseline,
    is_train_alpha,
    config,
):
    alpha_loss, alpha_grads, alpha = alpha_loss_grads_fun(
        log_alpha_model=log_alpha_model,
        policy_model=policy_model,
        data=batch_data,
        log_pi_baseline=log_pi_baseline,
    )
    alpha_grads = jax.tree.map(
        lambda g: jnp.where(is_train_alpha, g, jnp.zeros_like(g)),
        alpha_grads,
    )
    log_alpha_optimizer.update(alpha_grads)

    alpha = jnp.where(is_train_alpha, alpha, config.alpha)
    policy_loss, policy_grads, *policy_aux = policy_loss_grads_fun(
        policy_model=policy_model,
        data=batch_data,
        alpha=alpha,
    )
    policy_optimizer.update(policy_grads)

    return is_train_alpha * alpha_loss, alpha, policy_loss, *policy_aux


@nnx.jit
def train_n_steps_policy(
    policy_model,
    policy_optimizer,
    log_alpha_model,
    log_alpha_optimizer,
    data_buffer,
    config,
    log_pi_baseline,
    is_train_alpha,
    key,
):
    num_steps = config.log_freq

    pos_idxs, neg_idxs, union_idxs = data_buffer.sample_idxs(
        data_buffer, key, num_steps
    )

    def body_fun(i, carry):
        (
            _,
            policy_model,
            policy_optimizer,
            log_alpha_model,
            log_alpha_optimizer,
        ) = carry

        batch_data = data_buffer.sample_batch(
            data_buffer, pos_idxs[i], neg_idxs[i], union_idxs[i]
        )

        val = train_steps_policy(
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            log_alpha_model=log_alpha_model,
            log_alpha_optimizer=log_alpha_optimizer,
            batch_data=batch_data,
            log_pi_baseline=log_pi_baseline,
            is_train_alpha=is_train_alpha,
            config=config,
        )

        return (
            val,
            policy_model,
            policy_optimizer,
            log_alpha_model,
            log_alpha_optimizer,
        )

    init_val = (jnp.zeros((), dtype=jnp.float32),) * 6
    init_carry = (
        init_val,
        policy_model,
        policy_optimizer,
        log_alpha_model,
        log_alpha_optimizer,
    )
    val, *_ = nnx.fori_loop(0, num_steps, body_fun, init_carry)

    return val, num_steps


@nnx.jit
def train_n_steps_disc(
    discriminator_model,
    discriminator_optimizer,
    data_buffer,
    config,
    key,
):
    num_steps = config.log_freq

    pos_idxs, neg_idxs, union_idxs = data_buffer.sample_idxs(
        data_buffer, key, num_steps
    )

    def body_fun(i, carry):
        _, model, optimizer = carry

        batch_data = data_buffer.sample_batch(
            data_buffer, pos_idxs[i], neg_idxs[i], union_idxs[i]
        )

        # Batch X Horizon X obs_dim
        pos_obs = batch_data.pos_obs
        union_obs = batch_data.union_obs

        def loss_fun(model):
            p_pos, p_union = model(pos_obs), 1.0 - model(union_obs)
            # standard discriminator loss fun
            # loss_pos, loss_union = -jnp.log(p_pos), -jnp.log(p_union)

            # PU-Learning loss fun
            loss_pos = -jnp.log(p_pos)
            loss_union = -jnp.log(p_union) / config.pu_nu + jnp.log(1 - p_pos)
            loss = jnp.mean(loss_pos + loss_union)

            return loss

        grad_fun = nnx.value_and_grad(loss_fun)
        loss, grads = grad_fun(model)
        optimizer.update(grads)

        return (loss, model, optimizer)

    init_val = jnp.zeros((), dtype=jnp.float32)
    init_carry = (init_val, discriminator_model, discriminator_optimizer)
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

    trajectory_cfg["num_positive_trajectories"] = args.num_preferred
    trajectory_cfg["num_negative_trajectories"] = 0
    trajectory_cfg["num_union_trajectories"] = args.num_union
    trajectory_cfg["non_pref_noise"] = args.non_pref_noise
    trajectory_cfg["data_inpaint"] = args.data_inpaint
    trajectory_cfg["inpaint_ranges"] = trajectory_data[args.data_inpaint]

    config = {**default_cfg, **trajectory_cfg}
    config["train_horizon"] = args.train_horizon or config.get("train_horizon")
    config["lmbda"] = args.lmbda or config.get("lmbda")
    config["policy_type"] = args.policy_type
    config["normalize_observation"] = args.normalize_observation
    config["lr"] = args.lr
    config["warmup_iteration_policy"] = args.warmup_bc
    do_alpha_warmup = config["warmup_iteration_policy"] > 0

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

    expert_policy_model = SafeDiceTanhMixtureActor(
        rngs=rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_size=config["hidden_size"],
    )
    expert_policy_optimizer = nnx.Optimizer(
        model=expert_policy_model,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adamw(
                learning_rate=config["lr"], weight_decay=config["weight_decay"]
            ),
        ),
    )

    discriminator_model = ExpCostModel(
        rngs=rngs,
        x_dims=obs_space.shape[0],
        hidden_size=config["hidden_size"],
        clip_range=(0.1, 0.9),
    )
    discriminator_optimizer = nnx.Optimizer(
        model=discriminator_model,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adamw(
                learning_rate=config["lr"], weight_decay=config["weight_decay"]
            ),
        ),
    )

    log_alpha_model = Scalar(0.0)
    log_alpha_optimizer = nnx.Optimizer(
        model=log_alpha_model,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adamw(
                learning_rate=config["lr"], weight_decay=config["weight_decay"]
            ),
        ),
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

    pos_observations = pos_data["observations"]
    pos_actions = pos_data["actions"]
    pos_dones = pos_data["timeouts"] | pos_data["terminals"]
    pos_rewards = pos_data["rewards"]
    pos_costs = pos_data["costs"]

    union_observations = union_data["observations"]
    union_actions = union_data["actions"]
    union_dones = union_data["timeouts"] | union_data["terminals"]
    union_rewards = union_data["rewards"]
    union_costs = union_data["costs"]

    ep_len = ep_len // config["action_repeat"] + (ep_len % config["action_repeat"] > 0)
    assert (
        pos_observations.shape[1] == ep_len
    ), f"{pos_observations.shape[1]} episode length is different from {ep_len}"

    buffer = ILIDBuffer(
        rngs=rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        pos_data_size=np.prod(pos_observations.shape[:-1]),
        neg_data_size=1,  # dummy negative data
        union_data_size=np.prod(union_observations.shape[:-1]),
        horizon=config["train_horizon"],
        batch_size=batch_size,
        ep_len=ep_len,
    )

    for obs, act, done, reward, cost in zip(
        pos_observations, pos_actions, pos_dones, pos_rewards, pos_costs
    ):
        buffer.add(obs, act, done, reward, cost, is_pos=True)

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
    logger.log("Start discriminator model training.")

    disc_loss = jnp.array(0.0)
    steps = 0
    while steps < config["total_iteration_disc"]:

        disc_loss, num_itr = train_n_steps_disc(
            discriminator_model=discriminator_model,
            discriminator_optimizer=discriminator_optimizer,
            data_buffer=data_buffer,
            config=config_data,
            key=rngs.random_sample(),
        )

        steps += num_itr

        if steps % config["log_freq"] == 0:
            logger.log(
                f"Loss discriminator {steps}/{config['total_iteration_disc']}: {disc_loss.item():.3f}"
            )
            norm_discriminator = get_tree_norm(
                nnx.state(discriminator_model, nnx.Param)
            )
            logger.log(f"Norm discriminator_model: {norm_discriminator.item():.3f}")

        if steps >= config["total_iteration_disc"]:
            break

    log_pi_baseline = jnp.array(0.0)
    steps = 0
    if do_alpha_warmup:
        logger.log("Start warmup training of expert policy model.")
    while steps < config["warmup_iteration_policy"]:

        val, num_itr = train_n_steps_policy(
            policy_model=expert_policy_model,
            policy_optimizer=expert_policy_optimizer,
            log_alpha_model=log_alpha_model,
            log_alpha_optimizer=log_alpha_optimizer,
            data_buffer=data_buffer,
            config=config_data,
            log_pi_baseline=log_pi_baseline,
            is_train_alpha=False,  # no alpha training when warmup
            key=rngs.random_sample(),
        )

        (
            alpha_loss,
            alpha,
            policy_loss,
            policy_weight_min,
            policy_weight_mean,
            policy_weight_max,
        ) = val

        steps += num_itr

        logger.logged = False

        if steps % config["log_freq"] == 0:
            logger.log_tabular("Loss/loss_discriminator", disc_loss.item())
            logger.log_tabular("Loss/loss_alpha", alpha_loss.item())
            logger.log_tabular("Loss/loss_policy", policy_loss.item())

            logger.log_tabular("Value/alpha", alpha.item())
            logger.log_tabular("Value/log_pi_baseline", log_pi_baseline.item())
            logger.log_tabular(
                "Value/policy_union_weight_min", policy_weight_min.item()
            )
            logger.log_tabular(
                "Value/policy_union_weight_mean", policy_weight_mean.item()
            )
            logger.log_tabular(
                "Value/policy_union_weight_max", policy_weight_max.item()
            )

            logger.log_tabular(
                "Norm/discriminator_model",
                get_tree_norm(nnx.state(discriminator_model, nnx.Param)),
            )
            logger.log_tabular(
                "Norm/policy_model",
                get_tree_norm(nnx.state(expert_policy_model, nnx.Param)),
            )

            logger.dump_tabular()

        if steps >= config["warmup_iteration_policy"]:
            break

    if do_alpha_warmup:
        logger.log("Estimate log_pi_baseline from trained expert_policy_model")
        log_pi_baseline = jax.lax.stop_gradient(
            get_pos_data_log_prob(expert_policy_model, data_buffer)
        )

        logger.log("Initialize policy_model with expert_policy_model")
        policy_model = polyak_update(policy_model, expert_policy_model, 1.0)

    logger.log(
        "Reshuffle union dataset based on discriminator next state expert prediction."
    )
    pred_expert_mask = discriminator_model(data_buffer.union_obs) > config_data.lmbda
    logger.log(
        f"Number of predicted expert state in union dataset: {pred_expert_mask.sum()} / {pred_expert_mask.shape[0]}"
    )
    data_buffer = reshuffle_union_data(
        obj=data_buffer,
        pred_expert_mask=pred_expert_mask,
        rollback=config_data.rollback,
        decay=config_data.decay,
    )

    logger.log("Start training policy model.")
    steps = 0
    while steps < config["total_iteration_policy"]:

        val, num_itr = train_n_steps_policy(
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            log_alpha_model=log_alpha_model,
            log_alpha_optimizer=log_alpha_optimizer,
            data_buffer=data_buffer,
            config=config_data,
            log_pi_baseline=log_pi_baseline,
            is_train_alpha=do_alpha_warmup,
            key=rngs.random_sample(),
        )

        (
            alpha_loss,
            alpha,
            policy_loss,
            policy_weight_min,
            policy_weight_mean,
            policy_weight_max,
        ) = val

        steps += num_itr

        logger.logged = False

        if (steps % config["log_freq"] == 0) and (not logger.logged):
            logger.log_tabular("Loss/loss_discriminator", disc_loss.item())
            logger.log_tabular("Loss/loss_alpha", alpha_loss.item())
            logger.log_tabular("Loss/loss_policy", policy_loss.item())

            logger.log_tabular("Value/alpha", alpha.item())
            logger.log_tabular("Value/log_pi_baseline", log_pi_baseline.item())
            logger.log_tabular(
                "Value/policy_union_weight_min", policy_weight_min.item()
            )
            logger.log_tabular(
                "Value/policy_union_weight_mean", policy_weight_mean.item()
            )
            logger.log_tabular(
                "Value/policy_union_weight_max", policy_weight_max.item()
            )

            logger.log_tabular(
                "Norm/discriminator_model",
                get_tree_norm(nnx.state(discriminator_model, nnx.Param)),
            )
            logger.log_tabular(
                "Norm/policy_model",
                get_tree_norm(nnx.state(policy_model, nnx.Param)),
            )

            logger.dump_tabular()

        if steps % config["save_freq"] == 0:
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=policy_model,
                prefix="bc_policy",
            )

        if steps >= config["total_iteration_policy"]:
            break

    logger.nn_model_save(
        itr=steps, nn_model_saver_element=discriminator_model, prefix="discriminator"
    )
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=policy_model, prefix="bc_policy"
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
