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
from dsrl_model.utils.utils import single_agent_args

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
    "cost_weight_temp": 0.6,
    "train_horizon": 20,  # 20
    "weight_decay": 0.01,
    "grad_reg_coeffs": 10.0,
    "total_iteration": int(1e6),
    "bc_weight_binary": None,
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


@functools.partial(nnx.jit, static_argnames=["bag_size"])
def train_cost_model(
    cost_model,
    cost_optimizer,
    target_pos_obs,
    target_pos_act,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    has_positive,
    gamma,
    bag_size,
):
    # Batch_Bag x Horizon x obs/act_dim
    batch_bag_size = target_neg_obs.shape[0]
    batch_size = batch_bag_size // bag_size

    # Batch_Bag X Horizon X obs_act_dim
    target_pos = jnp.concat([target_pos_obs, target_pos_act], axis=-1)
    target_neg = jnp.concat([target_neg_obs, target_neg_act], axis=-1)
    target_union = jnp.concat([target_union_obs, target_union_act], axis=-1)

    def loss_fun(cost_model):
        # Batch_Bag x Horizon
        batch_bag_pos_cost = cost_model(target_pos)
        batch_bag_neg_cost = cost_model(target_neg)
        batch_bag_union_cost = cost_model(target_union)

        # Batch_Bag
        pos_traj_cost = discounted_sum(batch_bag_pos_cost.T, gamma)
        neg_traj_cost = discounted_sum(batch_bag_neg_cost.T, gamma)
        union_traj_cost = discounted_sum(batch_bag_union_cost.T, gamma)

        bag_pos_cost = pos_traj_cost.reshape((batch_size, bag_size)).mean(axis=1)
        bag_neg_cost = neg_traj_cost.reshape((batch_size, bag_size)).mean(axis=1)
        bag_union_cost = union_traj_cost.reshape((batch_size, bag_size)).mean(axis=1)

        # min L = - log[ exp(neg) / (exp(neg) + exp(pos)) ] = log[ 1 + exp(pos - neg) ]
        # L = nn.softplus(pos-neg)
        pos_neg_loss = has_positive * jax.nn.softplus(bag_pos_cost - bag_neg_cost)
        pos_union_loss = has_positive * jax.nn.softplus(bag_pos_cost - bag_union_cost)
        union_neg_loss = jax.nn.softplus(bag_union_cost - bag_neg_cost)

        loss = pos_neg_loss + pos_union_loss + union_neg_loss

        return loss.mean(), (
            has_positive * bag_pos_cost.mean(),
            bag_neg_cost.mean(),
            bag_union_cost.mean(),
            pos_neg_loss.mean(),
            pos_union_loss.mean(),
            union_neg_loss.mean(),
        )

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_values), grads = grad_fun(cost_model)
    cost_optimizer.update(grads)

    return loss, *aux_values


@nnx.jit
def train_policy_model(
    cost_model,
    bc_policy,
    bc_optimizer,
    target_pos_obs,
    target_pos_act,
    target_union_obs,
    target_union_act,
    has_positive,
    gamma,
    temp,
):
    batch_bag, horizon, _ = target_union_obs.shape

    # Batch_Bag_Horizon X obs/act_dim
    target_pos_obs = target_pos_obs.reshape(batch_bag * horizon, -1)
    target_pos_act = target_pos_act.reshape(batch_bag * horizon, -1)

    target_union_obs = target_union_obs.reshape(batch_bag * horizon, -1)
    target_union_act = target_union_act.reshape(batch_bag * horizon, -1)

    # Batch_Bag_Horizon
    union_cost = cost_model(jnp.concat([target_union_obs, target_union_act], axis=-1))
    horizon_union_cost = union_cost.reshape((batch_bag, horizon))
    # Batch_Bag
    batch_bag_union_cost = discounted_sum(horizon_union_cost.T, gamma)
    # weight = exp(-C / temp) / Mean [ exp(-C / temp) ]
    exp_union_weight = jnp.exp(-batch_bag_union_cost / temp)
    union_weight = exp_union_weight / exp_union_weight.mean()

    def loss_fun(bc_policy):
        pred_pos_act, *_ = bc_policy(target_pos_obs)
        flat_pos_loss = optax.l2_loss(pred_pos_act, target_pos_act).sum(axis=-1)
        horizon_pos_loss = has_positive * flat_pos_loss.reshape((batch_bag, horizon))
        batch_bag_pos_loss = discounted_sum(horizon_pos_loss.T, gamma)
        pos_loss = jnp.mean(batch_bag_pos_loss)

        pred_union_act, *_ = bc_policy(target_union_obs)
        flat_union_loss = optax.l2_loss(pred_union_act, target_union_act).sum(axis=-1)
        horizon_union_loss = flat_union_loss.reshape((batch_bag, horizon))
        batch_bag_union_loss = discounted_sum(horizon_union_loss.T, gamma)
        union_loss = jnp.mean(union_weight * batch_bag_union_loss)

        loss = pos_loss + union_loss
        return loss, (pos_loss, union_loss)

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_values), grads = grad_fun(bc_policy)
    bc_optimizer.update(grads)

    return loss, *aux_values


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
    config["cost_weight_temp"] = args.cost_weight_temp or config["cost_weight_temp"]
    config["bc_weight_binary"] = args.bc_weight_binary
    config["policy_type"] = args.policy_type
    config["normalize_observation"] = args.normalize_observation

    # evaluation environment
    eval_env = gym.make(args.task)
    eval_env.set_target_cost(config["target_cost"])
    eval_env.reset(seed=args.seed)

    # set training steps
    batch_size = args.batch_size or config.get("batch_size")

    # set model
    obs_space, act_space = eval_env.observation_space, eval_env.action_space
    config["lr"] = args.lr
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
        ) in buffer.sample():

            steps += 1

            (
                cost_loss,
                mean_pos_bag_cost,
                mean_neg_bag_cost,
                mean_union_bag_cost,
                pos_neg_cost_loss,
                pos_union_cost_loss,
                union_neg_cost_loss,
            ) = train_cost_model(
                cost_model=cost_model,
                cost_optimizer=cost_optimizer,
                target_pos_obs=target_pos_obs,
                target_pos_act=target_pos_act,
                target_neg_obs=target_neg_obs,
                target_neg_act=target_neg_act,
                target_union_obs=target_union_obs,
                target_union_act=target_union_act,
                has_positive=has_positive,
                gamma=config["gamma"],
                bag_size=config["bag_size"],
            )

            bc_loss, bc_pos_loss, bc_union_loss = train_policy_model(
                cost_model=cost_model,
                bc_policy=bc_policy,
                bc_optimizer=bc_optimizer,
                target_pos_obs=target_pos_obs,
                target_pos_act=target_pos_act,
                target_union_obs=target_union_obs,
                target_union_act=target_union_act,
                has_positive=has_positive,
                gamma=config["gamma"],
                temp=config["cost_weight_temp"],
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

                logger.log_tabular("Loss/Loss_cost", cost_loss.item())
                logger.log_tabular("Loss/Loss_pos_neg_cost", pos_neg_cost_loss.item())
                logger.log_tabular(
                    "Loss/Loss_pos_union_cost", pos_union_cost_loss.item()
                )
                logger.log_tabular(
                    "Loss/Loss_union_neg_cost", union_neg_cost_loss.item()
                )

                logger.log_tabular("Loss/Loss_bc_policy", bc_loss.item())
                logger.log_tabular("Loss/Loss_bc_pos_policy", bc_pos_loss.item())
                logger.log_tabular("Loss/Loss_bc_union_policy", bc_union_loss.item())

                logger.log_tabular("CostPred/pos_bag_cost", mean_pos_bag_cost.item())
                logger.log_tabular("CostPred/neg_bag_cost", mean_neg_bag_cost.item())
                logger.log_tabular(
                    "CostPred/union_bag_cost", mean_union_bag_cost.item()
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
                    "Cost/pos", jnp.sum(target_pos_cost, axis=-1).mean().item()
                )
                logger.log_tabular(
                    "Cost/neg", jnp.sum(target_neg_cost, axis=-1).mean().item()
                )
                logger.log_tabular(
                    "Cost/union", jnp.sum(target_union_cost, axis=-1).mean().item()
                )

                logger.log_tabular(
                    "Norm/cost_model",
                    get_tree_norm(nnx.state(cost_model, nnx.Param)),
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
                    nn_model_saver_element=bc_policy,
                    prefix="bc_policy",
                )

            if steps >= config["total_iteration"]:
                break

    logger.nn_model_save(itr=steps, nn_model_saver_element=cost_model, prefix="cost")
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
