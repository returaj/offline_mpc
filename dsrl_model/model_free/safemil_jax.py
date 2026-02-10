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

from dsrl_model.utils.buffer_jax import OnPolicyBuffer
from dsrl_model.utils.dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_normalized_data,
    get_pos_neg_and_union_data,
)
from dsrl_model.utils.models_jax import (
    ExpCostModel,
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
    "max_grad_norm": 10.0,
    "bag_size": 1,
    "gamma": 0.99,
    "action_repeat": 1,  # set to 2, min value is 1
    "update_freq": 1,
    "temp": 0.6,
    "train_horizon": 20,  # 20
    "weight_decay": 0.01,
    "bc_weight_binary": None,
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
    cost_model,
    policy_model,
    data,
    config,
    has_positive,
):
    gamma = config.gamma
    temp = config.temp

    # Batch_Bag X Horizon X obs
    batch_bag, horizon, _ = data.union_obs.shape

    # Batch_Bag_Horizon X obs/act_dim
    target_pos_obs = data.pos_obs.reshape(batch_bag * horizon, -1)
    target_pos_act = data.pos_act.reshape(batch_bag * horizon, -1)

    target_union_obs = data.union_obs.reshape(batch_bag * horizon, -1)
    target_union_act = data.union_act.reshape(batch_bag * horizon, -1)

    # Batch_Bag_Horizon
    union_cost = cost_model(jnp.concat([target_union_obs, target_union_act], axis=-1))
    # Batch_Bag X Horizon
    horizon_union_cost = union_cost.reshape((batch_bag, horizon))
    # Batch_Bag
    batch_bag_union_cost = discounted_sum(horizon_union_cost.T, gamma)
    # weight = exp(-C / temp) / Mean [ exp(-C / temp) ]
    exp_union_weight = jnp.exp(-batch_bag_union_cost / temp)
    union_weight = exp_union_weight / exp_union_weight.mean()

    def loss_fun(policy_model):
        pred_pos_act, *_ = policy_model(target_pos_obs)
        flat_pos_loss = optax.l2_loss(pred_pos_act, target_pos_act).sum(axis=-1)
        horizon_pos_loss = has_positive * flat_pos_loss.reshape((batch_bag, horizon))
        batch_bag_pos_loss = discounted_sum(horizon_pos_loss.T, gamma)
        pos_loss = jnp.mean(batch_bag_pos_loss)

        pred_union_act, *_ = policy_model(target_union_obs)
        flat_union_loss = optax.l2_loss(pred_union_act, target_union_act).sum(axis=-1)
        horizon_union_loss = flat_union_loss.reshape((batch_bag, horizon))
        batch_bag_union_loss = discounted_sum(horizon_union_loss.T, gamma)
        union_loss = jnp.mean(union_weight * batch_bag_union_loss)

        loss = pos_loss + union_loss
        return loss, (pos_loss, union_loss)

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_values), grads = grad_fun(policy_model)
    return loss, grads, *aux_values


def cost_loss_grads_fun(cost_model, data, config, has_positive):
    gamma = config.gamma
    batch_size, bag_size = config.batch_size, config.bag_size

    # Batch_Bag X Horizon X obs_act_dim
    target_pos = jnp.concat([data.pos_obs, data.pos_act], axis=-1)
    target_neg = jnp.concat([data.neg_obs, data.neg_act], axis=-1)
    target_union = jnp.concat([data.union_obs, data.union_act], axis=-1)

    def loss_fun(cost_model):
        # Batch_Bag x Horizon
        batch_bag_pos_cost = cost_model(target_pos)
        batch_bag_neg_cost = cost_model(target_neg)
        batch_bag_union_cost = cost_model(target_union)

        # Batch_Bag
        pos_traj_cost = discounted_sum(batch_bag_pos_cost.T, gamma)
        neg_traj_cost = discounted_sum(batch_bag_neg_cost.T, gamma)
        union_traj_cost = discounted_sum(batch_bag_union_cost.T, gamma)

        # Batch
        bag_pos_cost = pos_traj_cost.reshape((batch_size, bag_size)).mean(axis=1)
        bag_neg_cost = neg_traj_cost.reshape((batch_size, bag_size)).mean(axis=1)
        bag_union_cost = union_traj_cost.reshape((batch_size, bag_size)).mean(axis=1)

        # min L = - log[ exp(neg) / (exp(neg) + exp(pos)) ] = log[ 1 + exp(pos - neg) ]
        # L = nn.softplus(pos-neg)
        pos_neg_loss = has_positive * jax.nn.softplus(bag_pos_cost - bag_neg_cost)
        pos_union_loss = has_positive * jax.nn.softplus(bag_pos_cost - bag_union_cost)
        union_neg_loss = jax.nn.softplus(bag_union_cost - bag_neg_cost)

        loss = (pos_neg_loss + pos_union_loss + union_neg_loss).mean()

        return loss, (
            pos_neg_loss.mean(),
            pos_union_loss.mean(),
            union_neg_loss.mean(),
            has_positive * bag_pos_cost.mean(),
            bag_neg_cost.mean(),
            bag_union_cost.mean(),
        )

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_values), grads = grad_fun(cost_model)
    return loss, grads, *aux_values


def train_step(
    cost_model,
    cost_optimizer,
    policy_model,
    policy_optimizer,
    batch_data,
    config,
    has_positive,
):
    cost_loss, cost_grads, *cost_aux = cost_loss_grads_fun(
        cost_model=cost_model,
        data=batch_data,
        config=config,
        has_positive=has_positive,
    )
    cost_optimizer.update(cost_grads)

    policy_loss, policy_grads, *policy_aux = policy_loss_grads_fun(
        cost_model=cost_model,
        policy_model=policy_model,
        data=batch_data,
        config=config,
        has_positive=has_positive,
    )
    policy_optimizer.update(policy_grads)

    mean_pos_reward = has_positive * batch_data.pos_reward.sum(-1).mean()
    mean_neg_reward = batch_data.neg_reward.sum(-1).mean()
    mean_union_reward = batch_data.union_reward.sum(-1).mean()

    mean_pos_cost = has_positive * batch_data.pos_cost.sum(-1).mean()
    mean_neg_cost = batch_data.neg_cost.sum(-1).mean()
    mean_union_cost = batch_data.union_cost.sum(-1).mean()

    return (
        cost_loss,
        *cost_aux,
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
    policy_model,
    policy_optimizer,
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
            policy_model,
            policy_optimizer,
        ) = carry

        batch_data = data_buffer.sample_batch(
            data_buffer, pos_idxs[i], neg_idxs[i], union_idxs[i]
        )

        val = train_step(
            cost_model=cost_model,
            cost_optimizer=cost_optimizer,
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            batch_data=batch_data,
            config=config,
            has_positive=has_positive,
        )

        return (
            val,
            cost_model,
            cost_optimizer,
            policy_model,
            policy_optimizer,
        )

    init_val = (jnp.zeros((), dtype=jnp.float32),) * 16
    init_carry = (
        init_val,
        cost_model,
        cost_optimizer,
        policy_model,
        policy_optimizer,
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
    config["bag_size"] = args.bag_size or config["bag_size"]
    config["temp"] = args.cost_weight_temp or config["temp"]
    config["bc_weight_binary"] = args.bc_weight_binary
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

    cost_model = ExpCostModel(
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

    # data
    agent_task = re.search(r"Offline(.*?)Gymnasium-v[0-9]", args.task).group(1)
    ep_len = dsrl_infos.DEFAULT_MAX_EPISODE_STEPS[agent_task]
    data = get_dataset_in_d4rl_format(
        eval_env, trajectory_cfg, args.task, ep_len, config["action_repeat"]
    )
    pos_data, neg_data, union_data = get_pos_neg_and_union_data(data, trajectory_cfg)
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

    buffer = OnPolicyBuffer(
        rngs=rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        pos_data_size=pos_data_size,
        neg_data_size=np.prod(neg_observations.shape[:-1]),
        union_data_size=np.prod(union_observations.shape[:-1]),
        horizon=config["train_horizon"],
        batch_size=batch_size * config["bag_size"],
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
    logger.log("Start cost and policy model training.")

    steps = 0
    while steps < config["total_iteration"]:

        val, num_itr = train_n_steps(
            cost_model=cost_model,
            cost_optimizer=cost_optimizer,
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            data_buffer=data_buffer,
            config=config_data,
            has_positive=has_positive,
            key=rngs.random_sample(),
        )

        (
            cost_loss,
            cost_pos_neg_loss,
            cost_pos_union_loss,
            cost_union_neg_loss,
            mean_cost_pos_bag,
            mean_cost_neg_bag,
            mean_cost_union_bag,
            policy_loss,
            policy_pos_loss,
            policy_union_loss,
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
                        eval_env, policy_model.action, mu_obs, std_obs
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
            logger.log_tabular("Loss/Loss_cost_pos_neg", cost_pos_neg_loss.item())
            logger.log_tabular("Loss/Loss_cost_pos_union", cost_pos_union_loss.item())
            logger.log_tabular("Loss/Loss_cost_union_neg", cost_union_neg_loss.item())

            logger.log_tabular("Loss/Loss_policy", policy_loss.item())
            logger.log_tabular("Loss/Loss_policy_pos", policy_pos_loss.item())
            logger.log_tabular("Loss/Loss_policy_union", policy_union_loss.item())

            logger.log_tabular("CostPred/pos_bag_cost", mean_cost_pos_bag.item())
            logger.log_tabular("CostPred/neg_bag_cost", mean_cost_neg_bag.item())
            logger.log_tabular("CostPred/union_bag_cost", mean_cost_union_bag.item())

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
                "Norm/policy_model",
                get_tree_norm(nnx.state(policy_model, nnx.Param)),
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
                nn_model_saver_element=policy_model,
                prefix="bc_policy",
            )

        if steps >= config["total_iteration"]:
            break

    logger.nn_model_save(itr=steps, nn_model_saver_element=cost_model, prefix="cost")
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
