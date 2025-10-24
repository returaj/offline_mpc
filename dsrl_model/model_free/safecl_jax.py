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
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "eval_episode_freq": 1,  # use saved bc_policy to run evaluatation
    "hidden_size": 256,
    "embd_size": 128,
    "max_grad_norm": 5.0,
    "gamma": 0.99,
    "action_repeat": 1,  # set to 2, min value is 1
    "train_horizon": 500,  # 20
    "update_freq": 2,
    "decay": 0.85,
    "warmup_steps": int(1e2),
    "cost_weight_temp": 0.6,
    "update_tau": 0.01,
    "weight_decay": 0.01,
    "grad_reg_coeffs": 10.0,
    "total_iteration": int(1e6),
}

trajectory_cfg = {
    "density": 1.0,
    "target_cost": 25.0,
    # ((low_cost, low_reward), (high_cost, low_reward), (medium_cost, high_reward))
    "inpaint_ranges": ((0.0, 1.0, 0.0, 0.5),),
    "num_negative_trajectories": 50,
    "num_union_trajectories": -1,
}

labels_cfg = {
    "labels": [3.0, 2.0, 1.0, 0.0] + [-1.0],  # -1 denotes invalid label
    "range": [1.0, 0.5, 0.0, -0.5, -1.0] + [-2.0],  # -2 denotes invalid range
    "distance": [1.0, 0.5, -0.5, -1.0] + [-2.0],  # -2 denotes invalid distance
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
    dtype = vector_x.dtype
    horizon = vector_x.shape[0]

    def body_fun(t, cumsum):
        cumsum = vector_x[horizon - 1 - t] + gamma * cumsum
        return cumsum

    init_cumsum = jnp.zeros_like(vector_x[0], dtype=dtype)
    cumsum = jax.lax.fori_loop(0, horizon, body_fun, init_cumsum)
    return cumsum


@jax.jit
def compute_contrastive_ce_loss(p, q, decay):
    q = jax.nn.log_softmax(q, axis=1)
    p = p / jnp.clip(p.sum(axis=1, keepdims=True), min=1.0)
    loss = jnp.sum(p * q, axis=1) * decay
    return -jnp.mean(loss)


@jax.jit
def supervised_contrastive_loss(
    neg_z,
    union_z,
    target_union_label,
    all_labels,
    decay,
):
    dtype = neg_z.dtype
    batch_size, *_ = neg_z.shape
    max_label = all_labels[0] + 1

    temperature = 0.1  # value from SupContrast
    combined_z = jnp.concat([neg_z, union_z], axis=0)
    combined_logits = (combined_z @ combined_z.T) / temperature
    # remove the self instance from the logits
    combined_logits = jnp.fill_diagonal(combined_logits, -1e9, inplace=False)

    neg_mask = jnp.ones((batch_size, batch_size), dtype=dtype)
    union_mask = (
        jnp.expand_dims(target_union_label, axis=0)
        == jnp.expand_dims(target_union_label, axis=1)
    ).astype(dtype)
    valid_mask = (target_union_label >= 0).astype(dtype)
    # remove -1 labels from all the union_mask
    union_mask = (
        union_mask
        * jnp.expand_dims(valid_mask, axis=0)
        * jnp.expand_dims(valid_mask, axis=1)
    )
    combined_mask = jax.scipy.linalg.block_diag(neg_mask, union_mask)
    # zero out diagoals
    combined_mask = jnp.fill_diagonal(combined_mask, 0.0, inplace=False)

    neg_decay_rate = jnp.ones((batch_size,), dtype=dtype)
    union_decay_rate = decay ** (max_label - target_union_label)
    decay_rate = jnp.concat([neg_decay_rate, union_decay_rate])

    loss = compute_contrastive_ce_loss(combined_mask, combined_logits, decay_rate)
    return loss


@jax.jit
def distance_loss(
    neg_z,
    union_z,
    target_union_label,
    target_neg_z,
    all_labels,
    label_distance,
    decay,
):
    temp1, lambda1 = 1.0, 0.5

    dtype = neg_z.dtype
    max_label = all_labels[0] + 1

    neg_score = neg_z @ target_neg_z.T
    neg_mean_loss = jnp.mean(((1 - neg_score) / temp1) ** 2)
    neg_mean_score = jnp.mean(neg_score)

    union_score = union_z @ target_neg_z.T
    union_label_onehot = (target_union_label[:, None] == all_labels).astype(dtype)
    target_union_score = jnp.take(
        label_distance, jnp.argmax(union_label_onehot, axis=-1)
    )
    union_loss = ((target_union_score - union_score.squeeze()) / temp1) ** 2
    # remove invalid labels
    valid_weight = (target_union_label >= 0).astype(dtype)
    # high decay factor for lower labels
    decay_weight = decay ** (max_label - target_union_label)
    union_mean_loss = jnp.mean(valid_weight * decay_weight * union_loss)

    union_label_count = jnp.clip(jnp.sum(union_label_onehot, axis=0), min=1.0)
    union_mean_score = (
        jnp.sum(union_score * union_label_onehot, axis=0) / union_label_count
    )

    loss = neg_mean_loss + lambda1 * union_mean_loss
    return loss, (neg_mean_score, union_mean_score)


@nnx.jit
def train_embedding_model(
    embedding_model,
    embedding_optimizer,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    target_union_label,
    target_neg_z,
    all_labels,
    label_distance,
    decay,
):
    horizon = target_neg_obs.shape[1]

    # shape: Batch X Horizon X obs_act_dim
    target_neg = jnp.concat([target_neg_obs, target_neg_act], axis=-1)
    target_union = jnp.concat([target_union_obs, target_union_act], axis=-1)

    def loss_fun(embedding_model):
        # Batch X embd_dim
        neg_z = embedding_model(target_neg, normalize_z=True)
        union_z = embedding_model(target_union, normalize_z=True)

        supcon_loss = supervised_contrastive_loss(
            neg_z=neg_z,
            union_z=union_z,
            target_union_label=target_union_label,
            all_labels=all_labels,
            decay=decay,
        )

        dist_loss, aux_values = distance_loss(
            neg_z=neg_z,
            union_z=union_z,
            target_union_label=target_union_label,
            target_neg_z=target_neg_z,
            all_labels=all_labels,
            label_distance=label_distance,
            decay=decay,
        )

        loss = supcon_loss + dist_loss

        return loss, (supcon_loss, dist_loss, *aux_values)

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_values), grads = grad_fun(embedding_model)
    grads = jax.tree.map(lambda g: g / horizon, grads)
    embedding_optimizer.update(grads)

    return loss, *aux_values


@jax.jit
def index_fun(arr, all_labels, label_range):
    # For each value in arr, find which interval of `ranges` it falls into
    # range[i] >= x > range[i+1] → assign labels[i]
    idx = jnp.sum(arr[:, None] <= label_range[1:], axis=-1)
    return all_labels[idx]


@nnx.jit
def get_new_union_labels(
    embedding_model,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    target_neg_z,
    all_labels,
    label_range,
    tau,
):
    target_neg = jnp.concat([target_neg_obs, target_neg_act], axis=-1)
    target_union = jnp.concat([target_union_obs, target_union_act], axis=-1)

    # Batch X embd_dim
    neg_z = embedding_model(target_neg, normalize_z=True, training=False)
    union_z = embedding_model(target_union, normalize_z=True, training=False)

    rep_neg_z = neg_z.mean(axis=0, keepdims=True)
    target_neg_z = (1 - tau) * target_neg_z + tau * rep_neg_z
    target_neg_z = l2_normalize(target_neg_z, axis=1)

    union_score = (union_z @ target_neg_z.T).squeeze()
    new_label = index_fun(union_score, all_labels, label_range)
    new_label_count = (new_label[:, None] == all_labels).sum(axis=0)
    new_label_percent = new_label_count / jnp.sum(new_label_count)
    return target_neg_z, union_score, new_label, new_label_percent


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
    dtype = target_neg_obs.dtype

    target_neg = jnp.concat([target_neg_obs, target_neg_act], axis=-1)
    target_union = jnp.concat([target_union_obs, target_union_act], axis=-1)
    cost_neg = cost_model(target_neg)
    cost_union = cost_model(target_union)

    total_neg_cost = jax.vmap(discounted_sum, in_axes=(0, None))(cost_neg, gamma)
    total_union_cost = jax.vmap(discounted_sum, in_axes=(0, None))(cost_union, gamma)

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


@nnx.jit
def bc_policy_trajectory_loss_fun(
    bc_policy,
    target_obs,
    target_act,
    target_score,
    gamma,
):
    batch_size, horizon, _ = target_obs.shape

    target_obs = target_obs.reshape((batch_size * horizon, -1))
    target_act = target_act.reshape((batch_size * horizon, -1))

    pred_act, *_ = bc_policy(target_obs)
    loss = optax.l2_loss(pred_act, target_act).sum(axis=-1)
    loss = loss.reshape(batch_size, horizon)
    traj_loss = jax.vmap(discounted_sum, in_axes=(0, None))(loss, gamma)

    weight = jnp.clip(jnp.exp((-0.5 - target_score) / 0.1), max=5.0)

    return jnp.mean(weight * traj_loss)


@nnx.jit
def train_policy_model(
    bc_policy,
    bc_optimizer,
    target_obs,
    target_act,
    target_score,
    gamma,
):
    horizon = target_obs.shape[1]

    bc_grad_fun = nnx.value_and_grad(bc_policy_trajectory_loss_fun)
    bc_loss, bc_grad = bc_grad_fun(
        bc_policy, target_obs, target_act, target_score, gamma
    )
    bc_grad = jax.tree.map(lambda g: g / horizon, bc_grad)
    bc_optimizer.update(bc_grad)

    return bc_loss


def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    rngs = nnx.Rngs(args.seed)

    # set default device id
    jax.default_device = jax.devices(args.device)[args.device_id]

    # label configs:
    all_labels = jnp.array(labels_cfg["labels"], dtype=jnp.float32)
    label_distance = jnp.array(labels_cfg["distance"], dtype=jnp.float32)
    label_range = jnp.array(labels_cfg["range"], dtype=jnp.float32)

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
        # shape: Batch X Horizon X obs/act_dim
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

            (
                embedding_loss,
                supcon_loss,
                dist_loss,
                neg_mean_score,
                union_mean_score,
            ) = train_embedding_model(
                embedding_model=embedding_model,
                embedding_optimizer=embedding_optimizer,
                target_neg_obs=target_neg_obs,
                target_neg_act=target_neg_act,
                target_union_obs=target_union_obs,
                target_union_act=target_union_act,
                target_union_label=target_union_label,
                target_neg_z=target_neg_z,
                all_labels=all_labels,
                label_distance=label_distance,
                decay=config["decay"],
            )

            bc_loss = jnp.array(0.0)
            new_union_labels_percent = jnp.zeros_like(all_labels, dtype=jnp.float32)
            if (steps > config["warmup_steps"]) and (
                steps % config["update_freq"] == 0
            ):
                (
                    target_neg_z,
                    new_union_score,
                    new_union_labels,
                    new_union_labels_percent,
                ) = get_new_union_labels(
                    embedding_model=embedding_model,
                    target_neg_obs=target_neg_obs,
                    target_neg_act=target_neg_act,
                    target_union_obs=target_union_obs,
                    target_union_act=target_union_act,
                    target_neg_z=target_neg_z,
                    all_labels=all_labels,
                    label_range=label_range,
                    tau=config["update_tau"],
                )
                buffer.update_labels(target_union_idx, new_union_labels)

                bc_loss = train_policy_model(
                    bc_policy=bc_policy,
                    bc_optimizer=bc_optimizer,
                    target_obs=target_union_obs,
                    target_act=target_union_act,
                    target_score=new_union_score,
                    gamma=config["gamma"],
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
                logger.log_tabular("Loss/Loss_embd_supcon_loss", supcon_loss.item())
                logger.log_tabular("Loss/Loss_embd_dist_loss", dist_loss.item())
                logger.log_tabular("Loss/Loss_bc_policy", bc_loss.item())

                logger.log_tabular("Mean/embd_neg_score", neg_mean_score.item())
                for l, lms in zip(all_labels, union_mean_score):
                    logger.log_tabular(f"Mean/embd_union_score_label_{l}", lms.item())
                for l, nul in zip(all_labels, new_union_labels_percent):
                    logger.log_tabular(f"Percentage/new_union_label_{l}", nul.item())

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
