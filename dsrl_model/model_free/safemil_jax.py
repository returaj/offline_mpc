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
    get_neg_and_union_data,
    get_normalized_data,
)
from dsrl_model.utils.models_jax import (
    ExpCostModel,
    SafeDiceTanhMixtureActor,
    TransformerEmbedding,
    bce_loss,
    get_tree_norm,
    l2_normalize,
)
from dsrl_model.utils.native_logger import EpochLogger
from dsrl_model.utils.utils import single_agent_args

EPS = 1e-6

default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "eval_episode_freq": 1,  # use saved bc_policy to run evaluatation
    "hidden_size": 256,
    "bag_size": 1,
    "latent_obs_dim": 50,
    "max_grad_norm": 10.0,
    "gamma": 0.99,
    "cost_lambda": 0.0,
    "action_repeat": 1,  # set to 2, min value is 1
    "cost_weight_temp": 0.6,
    "train_horizon": 20,  # 20
    "weight_decay": 0.01,
    "grad_reg_coeffs": 10.0,
    "total_iteration": int(1e6),
    "bc_weight_binary": None,
    "use_validation": None,
}

trajectory_cfg = {
    "density": 1.0,
    # ((low_cost, low_reward), (high_cost, low_reward), (medium_cost, high_reward))
    "inpaint_ranges": (
        (0.0, 0.5, 0.0, 0.5),
        (0.5, 1.0, 0.0, 0.5),
        (0.25, 0.75, 0.0, 1.0),
    ),
    "target_cost": 25.0,
    "alpha": 0.5,  # dU = alpha * dN + (1-alpha) * dP
    "num_negative_trajectories": 50,
    "num_union_negative_trajectories": 100,
    "num_union_positive_trajectories": 100,
}


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


@nnx.jit
def train_policy_model(
    cost_model,
    bc_policy,
    bc_optimizer,
    target_obs,
    target_act,
    gamma,
    beta,
    temp,
):

    def loss_fun(bc_policy):

        def trajectory_bcloss_cost(obs, act):
            pred_act, *_ = bc_policy(obs)
            bcloss = jnp.sum((pred_act - act) ** 2, axis=1)
            trajectory_bcloss = discounted_sum(bcloss, gamma)

            cost = cost_model(jnp.concat([obs, act], axis=1))
            trajectory_cost = discounted_sum(cost, gamma)
            return trajectory_bcloss, trajectory_cost

        trajectory_bcloss, trajectory_cost = jax.vmap(
            trajectory_bcloss_cost, in_axes=(0, 0)
        )(target_obs, target_act)

        weight = jnp.clip(jnp.exp((beta - trajectory_cost) / temp), max=5.0)
        loss = jnp.mean(weight * trajectory_bcloss)
        return loss

    grad_fun = nnx.value_and_grad(loss_fun)
    loss, grads = grad_fun(bc_policy)
    bc_optimizer.update(grads)
    return loss


@nnx.jit
def train_cost_model(
    cost_model,
    cost_optimizer,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    gamma,
    bag_size,
):
    # Batch_Bag x Horizon x obs/act_dim
    batch_bag_size, *_ = target_neg_obs.shape
    batch_size = batch_bag_size // bag_size

    def loss_fun(cost_model):
        vmap_discounted_sum = jax.vmap(discounted_sum, in_axes=(0, None))

        def bag_cost(obs, act):
            x = jnp.concat([obs, act], axis=-1)
            costs = cost_model(x)
            total_cost = vmap_discounted_sum(costs, gamma)
            total_cost = total_cost.reshape((batch_size, bag_size))
            bag_cost = jnp.mean(total_cost, axis=1)
            return bag_cost

        neg_bag_cost = bag_cost(target_neg_obs, target_neg_act)
        union_bag_cost = bag_cost(target_union_obs, target_union_act)

        max_bag_cost = jnp.maximum(neg_bag_cost, union_bag_cost)
        exp_neg = jnp.exp(neg_bag_cost - max_bag_cost)
        exp_union = jnp.exp(union_bag_cost - max_bag_cost)
        p_neg = exp_neg / (exp_neg + exp_union)
        loss = -jnp.mean(jnp.log(p_neg))
        return loss, jnp.mean(neg_bag_cost), jnp.mean(union_bag_cost)

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_values), grads = grad_fun(cost_model)
    cost_optimizer.update(grads)
    return loss, *aux_values


def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    rngs = nnx.Rngs(args.seed)

    # set default device id
    jax.default_device = jax.devices(args.device)[args.device_id]

    trajectory_cfg["num_negative_trajectories"] = args.num_non_preferred

    config = {**default_cfg, **trajectory_cfg}
    config["train_horizon"] = args.train_horizon or config.get("train_horizon")
    config["bag_size"] = args.bag_size or config["bag_size"]
    config["cost_weight_temp"] = args.cost_weight_temp or config["cost_weight_temp"]
    config["bc_weight_binary"] = args.bc_weight_binary
    config["normalize_observation"] = args.normalize_observation
    config["use_validation"] = args.use_validation
    if not config["use_validation"]:
        config["cost_validation_freq"] = 1

    # evaluation environment
    eval_env = gym.make(args.task)
    eval_env.set_target_cost(config["target_cost"])
    eval_env.reset(seed=args.seed)

    # set training steps
    batch_size = args.batch_size or config.get("batch_size")

    # set model
    obs_space, act_space = eval_env.observation_space, eval_env.action_space
    config["bc_lr"] = args.lr
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
                learning_rate=config["bc_lr"], weight_decay=config["weight_decay"]
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
            optax.adamw(learning_rate=args.lr, weight_decay=config["weight_decay"]),
        ),
    )

    # data
    agent_task = re.search(r"Offline(.*?)Gymnasium-v[0-9]", args.task).group(1)
    ep_len = dsrl_infos.DEFAULT_MAX_EPISODE_STEPS[agent_task]
    data = get_dataset_in_d4rl_format(
        eval_env, trajectory_cfg, args.task, ep_len, config["action_repeat"]
    )
    neg_data, union_data = get_neg_and_union_data(data, trajectory_cfg)
    mu_obs, std_obs = 0.0, 1.0
    if config["normalize_observation"]:
        neg_data, union_data, mu_obs, std_obs = get_normalized_data(
            neg_data, union_data
        )

    neg_observations = neg_data["observations"]
    neg_actions = neg_data["actions"]
    neg_dones = neg_data["timeouts"] | neg_data["terminals"]
    neg_costs = neg_data["costs"]

    union_observations = union_data["observations"]
    union_actions = union_data["actions"]
    union_dones = union_data["timeouts"] | union_data["terminals"]
    union_costs = union_data["costs"]

    ep_len = ep_len // config["action_repeat"] + (ep_len % config["action_repeat"] > 0)
    assert (
        neg_observations.shape[1] == ep_len
    ), f"{neg_observations.shape[1]} episode length is different from {ep_len}"

    buffer = OnPolicyBuffer(
        rngs=rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        neg_data_size=np.prod(neg_observations.shape[:-1]),
        union_data_size=np.prod(union_observations.shape[:-1]),
        horizon=config["train_horizon"],
        batch_size=batch_size * config["bag_size"],
        ep_len=ep_len,
    )
    for obs, act, done, cost in zip(
        neg_observations, neg_actions, neg_dones, neg_costs
    ):
        buffer.add(obs, act, done, cost=cost, is_negative=True)
    for obs, act, done, cost in zip(
        union_observations, union_actions, union_dones, union_costs
    ):
        buffer.add(obs, act, done, cost=cost, is_negative=False)

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
            target_neg_obs,
            target_neg_act,
            _,
            target_union_obs,
            target_union_act,
            _,
            _,
        ) in buffer.sample():

            steps += 1

            cost_loss, mean_neg_cost, mean_union_cost = train_cost_model(
                cost_model=cost_model,
                cost_optimizer=cost_optimizer,
                target_neg_obs=target_neg_obs,
                target_neg_act=target_neg_act,
                target_union_obs=target_union_obs,
                target_union_act=target_union_act,
                gamma=config["gamma"],
                bag_size=config["bag_size"],
            )

            bc_loss = train_policy_model(
                cost_model=cost_model,
                bc_policy=bc_policy,
                bc_optimizer=bc_optimizer,
                target_obs=target_union_obs,
                target_act=target_union_act,
                gamma=config["gamma"],
                beta=mean_union_cost,
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
                logger.log_tabular("Loss/Loss_embedding", embedding_loss.item())
                logger.log_tabular("Loss/Loss_embd_mode_entropy", mode_entropy.item())
                logger.log_tabular("Loss/Loss_bc_policy", bc_loss.item())

                for i in range(labels_cfg["num_modes"]):
                    logger.log_tabular(
                        f"Percentage/embd_mode_{i}", mode_percent[i].item()
                    )

                logger.log_tabular("Mean/embd_neg_score", neg_mean_score.mean().item())
                for l, lms in zip(all_labels, union_mean_score):
                    logger.log_tabular(f"Mean/embd_union_score_label_{l}", lms.item())

                for l, nul in zip(all_labels, new_union_labels_percent):
                    logger.log_tabular(f"Percentage/new_union_label_{l}", nul.item())

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
        itr=steps, nn_model_saver_element=embedding_model, prefix="embedding"
    )
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=bc_policy, prefix="bc_policy"
    )
    logger.save_state(state_dict={"target_neg_z": target_neg_z}, dirname="neg_z")
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
