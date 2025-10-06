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

from dsrl_model.utils.buffer_jax import SafeCLBuffer
from dsrl_model.utils.dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_neg_and_union_data_2,
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
    "log_freq": int(1e1),
    "save_freq": int(2e1),
    "eval_episode_freq": 1,  # use saved bc_policy to run evaluatation
    "hidden_size": 256,
    "embd_size": 128,
    "max_grad_norm": 1.0,
    "gamma": 0.99,
    "action_repeat": 1,  # set to 2, min value is 1
    "train_horizon": 5,  # 20
    "update_freq": 2,
    "decay": 0.85,
    "max_label": 4.0,
    "warmup_steps": int(1e1),
    "cost_weight_temp": 0.6,
    "update_tau": 0.01,
    "weight_decay": 0.01,
    "grad_reg_coeffs": 10.0,
    "total_iteration": int(1e2),
}

trajectory_cfg = {
    "density": 1.0,
    "target_cost": 25.0,
    # ((low_cost, low_reward), (high_cost, low_reward), (medium_cost, high_reward))
    "inpaint_ranges": ((0.0, 1.0, 0.0, 0.5),),
    "num_negative_trajectories": 50,
    "num_union_trajectories": -1,
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
def discounted_sum(vector_x, gamma):
    horizon = vector_x.shape[0]
    cumsum = 0

    def body_fun(t, cumsum):
        cumsum = vector_x[horizon - 1 - t] + gamma * cumsum
        return cumsum

    cumsum = jax.lax.fori_loop(0, horizon, body_fun, cumsum)
    return cumsum


@jax.jit
def compute_contrastive_ce_loss(p, q, decay):
    q = jax.nn.log_softmax(q, axis=1)
    p = p / jnp.clip(p.sum(axis=1, keepdims=True), min=1.0)
    loss = jnp.sum(p * q, axis=1) * decay
    return -jnp.mean(loss)


@nnx.jit
def train_embedding_model(
    embedding_model,
    embedding_optimizer,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    target_union_label,
    decay,
    max_label,
):
    _, batch_size, _ = target_neg_obs.shape

    # shape: Horizon X Batch X obs_act_dim
    target_neg = jnp.concat([target_neg_obs, target_neg_act], axis=-1)
    target_union = jnp.concat([target_union_obs, target_union_act], axis=-1)

    def loss_fun(embedding_model):
        # Batch X embd_dim
        neg_z = embedding_model(target_neg, horizon_axis=0, normalize_z=True)
        union_z = embedding_model(target_union, horizon_axis=0, normalize_z=True)

        temperature = 0.1  # value from SupContrast
        combined_z = jnp.concat([neg_z, union_z], axis=0)
        combined_logits = (combined_z @ combined_z.T) / temperature
        # remove the self instance from the logits
        combined_logits = jnp.fill_diagonal(combined_logits, -1e9, inplace=False)

        neg_mask = jnp.ones((batch_size, batch_size), dtype=jnp.float32)
        union_mask = (
            jnp.expand_dims(target_union_label, axis=0)
            == jnp.expand_dims(target_union_label, axis=1)
        ).astype(jnp.float32)
        valid_mask = (target_union_label >= 0).astype(jnp.float32)
        # remove -1 labels from all the union_mask
        union_mask = (
            union_mask
            * jnp.expand_dims(valid_mask, axis=0)
            * jnp.expand_dims(valid_mask, axis=1)
        )
        combined_mask = jax.scipy.linalg.block_diag(neg_mask, union_mask)
        # zero out diagoals
        combined_mask = jnp.fill_diagonal(combined_mask, 0.0, inplace=False)

        neg_decay_rate = jnp.ones((batch_size,), dtype=jnp.float32)
        union_decay_rate = decay ** (max_label - target_union_label)
        decay_rate = jnp.concat([neg_decay_rate, union_decay_rate])

        loss = compute_contrastive_ce_loss(combined_mask, combined_logits, decay_rate)
        return loss

    grad_fun = nnx.value_and_grad(loss_fun)
    loss, grads = grad_fun(embedding_model)
    embedding_optimizer.update(grads)

    return loss


def index_fun(v, max_label):
    return jnp.select(
        condlist=[
            v >= 0.8,
            (v >= 0.6) & (v < 0.8),
            (v >= 0.4) & (v < 0.6),
            (v >= 0.2) & (v < 0.4),
        ],
        choicelist=[max_label, 3.0, 2.0, 1.0],
        default=0.0,
    )


@functools.partial(jax.jit, static_argnums=0)
def get_new_union_labels(
    embedding_model,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    target_neg_z,
    tau,
    max_label,
):
    target_neg = jnp.concat([target_neg_obs, target_neg_act], axis=-1)
    target_union = jnp.concat([target_union_obs, target_union_act], axis=-1)

    # Batch X embd_dim
    neg_z = embedding_model(
        target_neg, horizon_axis=0, normalize_z=True, training=False
    )
    union_z = embedding_model(
        target_union, horizon_axis=0, normalize_z=True, training=False
    )

    rep_neg_z = l2_normalize(neg_z.sum(axis=0, keepdims=True), axis=-1)
    target_neg_z = (1 - tau) * target_neg_z + tau * rep_neg_z

    union_pred = (union_z @ target_neg_z.T).squeeze()
    new_label = index_fun(union_pred, max_label)
    return target_neg_z, new_label


def cost_loss_fun(
    cost_model,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    target_union_label,
    gamma,
    decay,
    max_label,
):
    discount = 1.0
    # Horizon X Batch X obs/act_dim
    horizon, batch_size, *_ = target_neg_obs.shape
    dtype = target_neg_obs.dtype

    total_neg_cost = jnp.zeros((batch_size,), dtype=dtype)
    total_union_cost = jnp.zeros((batch_size,), dtype=dtype)

    target_neg = jnp.concat([target_neg_obs, target_neg_act], axis=-1)
    target_union = jnp.concat([target_union_obs, target_union_act], axis=-1)

    def estimate_loss(t, val):
        discount, total_neg_cost, total_union_cost = val
        tn, tu = target_neg[t], target_union[t]
        total_neg_cost += discount * cost_model(tn)
        total_union_cost += discount * cost_model(tu)
        discount *= gamma
        return (discount, total_neg_cost, total_union_cost)

    _, total_neg_cost, total_union_cost = jax.lax.fori_loop(
        0, horizon, estimate_loss, (discount, total_neg_cost, total_union_cost)
    )

    # compare neg and union trajectory
    logit_neg = total_neg_cost - total_union_cost
    target_ones = jnp.ones_like(logit_neg, dtype=dtype)
    neg_union_loss = bce_loss(logit_neg, target_ones)

    # compare union and union trajectory
    # ensure batch size is of 2s multiple
    part_uu_cost = total_union_cost.reshape((2, -1))
    part_uu_label = target_union_label.reshape((2, -1))

    logit_uu = part_uu_cost[0] - part_uu_cost[1]
    target_uu = (part_uu_label[0] > part_uu_label[1]).astype(dtype)
    target_uu = jnp.where(part_uu_label[0] == part_uu_label[1], 0.5, target_uu)

    valid_compare = ((part_uu_label[0] > 0) & (part_uu_label[1] > 0)).astype(dtype)
    decay_weight = decay ** (max_label - jnp.abs(part_uu_label[0] - part_uu_label[1]))
    weight = decay_weight * valid_compare

    union_union_loss = bce_loss(logit_uu, target_uu, weight)

    loss = neg_union_loss + union_union_loss
    return jnp.mean(loss)


def bc_policy_transition_loss_fun(bc_policy, cost_model, target_obs, target_act):
    horizon, batch_size, _ = target_obs.shape

    target_obs = target_obs.reshape((horizon * batch_size, -1))
    target_act = target_act.reshape((horizon * batch_size, -1))

    weight = 1 - cost_model(jnp.concat([target_obs, target_act], axis=-1))

    pred_act, *_ = bc_policy(target_obs)
    loss = optax.l2_loss(pred_act, target_act).sum(axis=-1)
    loss = jnp.mean(weight * loss)
    return loss


@nnx.jit
def train_cost_and_policy_model(
    cost_model,
    bc_policy,
    cost_optimizer,
    bc_optimizer,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    target_union_label,
    gamma,
    decay,
    max_label,
):
    horizon = target_neg_obs.shape[0]

    cost_grad_fun = nnx.value_and_grad(cost_loss_fun)
    cost_loss, cost_grad = cost_grad_fun(
        cost_model,
        target_neg_obs,
        target_neg_act,
        target_union_obs,
        target_union_act,
        target_union_label,
        gamma,
        decay,
        max_label,
    )
    cost_grad = jax.tree.map(lambda g: g / horizon, cost_grad)
    cost_optimizer.update(cost_grad)

    bc_grad_fun = nnx.value_and_grad(bc_policy_transition_loss_fun)
    bc_loss, bc_grad = bc_grad_fun(
        bc_policy, cost_model, target_union_obs, target_union_act
    )
    bc_optimizer.update(bc_grad)

    return cost_loss, bc_loss


def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    rngs = nnx.Rngs(args.seed)

    # set default device id
    jax.default_device = jax.devices(args.device)[args.device_id]

    trajectory_cfg["num_negative_trajectories"] = args.num_non_preferred
    trajectory_cfg["num_union_trajectories"] = args.num_union
    trajectory_cfg["non_pref_noise"] = args.non_pref_noise

    config = {**default_cfg, **trajectory_cfg}
    config["train_horizon"] = args.train_horizon or config.get("train_horizon")
    config["policy_type"] = args.policy_type
    config["normalize_observation"] = args.normalize_observation
    config["cost_weight_temp"] = args.cost_weight_temp or config["cost_weight_temp"]
    config["use_bc_trajectory"] = args.use_bc_trajectory

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
            optax.adamw(
                learning_rate=config["bc_lr"], weight_decay=config["weight_decay"]
            ),
        ),
    )

    # data
    agent_task = re.search(r"Offline(.*?)Gymnasium-v[0-9]", args.task).group(1)
    ep_len = dsrl_infos.DEFAULT_MAX_EPISODE_STEPS[agent_task]
    data = get_dataset_in_d4rl_format(
        eval_env, trajectory_cfg, args.task, ep_len, config["action_repeat"]
    )
    neg_data, union_data = get_neg_and_union_data_2(data, trajectory_cfg)
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

    buffer = SafeCLBuffer(
        rngs=rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        neg_data_size=np.prod(neg_observations.shape[:-1]),
        union_data_size=np.prod(union_observations.shape[:-1]),
        horizon=config["train_horizon"],
        batch_size=batch_size,
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
    logger.log("Start embedding, cost and bc_policy model training.")

    steps = 0
    target_neg_z = jnp.zeros((1, embd_size), dtype=jnp.float32)
    while steps < config["total_iteration"]:
        # shape: Horizon X Batch X obs/act_dim
        for (
            target_neg_obs,
            target_neg_act,
            _,
            target_union_obs,
            target_union_act,
            _,
            target_union_idx,
            target_union_label,
        ) in buffer.sample():

            steps += 1

            embedding_loss = train_embedding_model(
                embedding_model=embedding_model,
                embedding_optimizer=embedding_optimizer,
                target_neg_obs=target_neg_obs,
                target_neg_act=target_neg_act,
                target_union_obs=target_union_obs,
                target_union_act=target_union_act,
                target_union_label=target_union_label,
                decay=config["decay"],
                max_label=config["max_label"],
            )

            cost_loss = bc_loss = jnp.array(0.0)
            if (steps > config["warmup_steps"]) and (
                steps % config["update_freq"] == 0
            ):
                target_neg_z, new_union_labels = get_new_union_labels(
                    embedding_model=embedding_model,
                    target_neg_obs=target_neg_obs,
                    target_neg_act=target_neg_act,
                    target_union_obs=target_union_obs,
                    target_union_act=target_union_act,
                    target_neg_z=target_neg_z,
                    tau=config["update_tau"],
                    max_label=config["max_label"],
                )
                buffer.update_labels(target_union_idx, new_union_labels)

                cost_loss, bc_loss = train_cost_and_policy_model(
                    cost_model=cost_model,
                    bc_policy=bc_policy,
                    cost_optimizer=cost_optimizer,
                    bc_optimizer=bc_optimizer,
                    target_neg_obs=target_neg_obs,
                    target_neg_act=target_neg_act,
                    target_union_obs=target_union_obs,
                    target_union_act=target_union_act,
                    target_union_label=new_union_labels,
                    gamma=config["gamma"],
                    decay=config["decay"],
                    max_label=config["max_label"],
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
                logger.log_tabular("Loss/Loss_embedding", embedding_loss.mean().item())
                logger.log_tabular("Loss/Loss_cost", cost_loss.mean().item())
                logger.log_tabular("Loss/Loss_bc_policy", bc_loss.mean().item())
                logger.log_tabular(
                    "Norm/embedding_model",
                    get_tree_norm(nnx.state(embedding_model, nnx.Param)),
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
                    nn_model_saver_element=embedding_model,
                    prefix="embedding",
                )
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

    logger.nn_model_save(
        itr=steps, nn_model_saver_element=embedding_model, prefix="embedding"
    )
    logger.nn_model_save(itr=steps, nn_model_saver_element=cost_model, prefix="cost")
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
