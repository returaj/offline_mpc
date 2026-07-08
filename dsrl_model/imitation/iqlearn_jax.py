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
    "bag_size": 1,
    "gamma": 0.99,
    "action_repeat": 1,  # set to 2, min value is 1
    "update_critic_freq": 2,
    "update_tau": 0.005,
    "train_horizon": 5,  # min horizon 2 is required
    "weight_decay": 0.01,
    "grad_reg_coeffs": 10.0,
    "total_iteration": int(1e6),
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
    critic_model,
    policy_model,
    data,
    alpha,
):
    # Batch X Horizon X obs_dim
    batch, horizon, _ = data.union_obs.shape

    # Batch_(Horizon-1) X obs_dim: mask the last horizon step
    target_pos_obs = data.pos_obs[:, :-1].reshape(batch * (horizon - 1), -1)
    target_union_obs = data.union_obs[:, :-1].reshape(batch * (horizon - 1), -1)
    target_obs = jnp.concat([target_pos_obs, target_union_obs], axis=0)

    def loss_fun(policy_model):
        pred_act, log_prob, *_ = policy_model(target_obs)
        q = jnp.minimum(*critic_model(jnp.concat([target_obs, pred_act], axis=-1)))
        loss = -(q - alpha * log_prob).mean()
        return loss, (
            q.mean(),
            -log_prob.mean(),
        )

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_values), grads = grad_fun(policy_model)
    return loss, grads, *aux_values


def critic_loss_grads_fun(
    critic_model_target,
    critic_model,
    policy_model,
    data,
    gamma,
    alpha,
    lmbda,
):
    # using target network was making the critic learning very unstable
    del critic_model_target

    dtype = data.union_obs.dtype
    batch, horizon, _ = data.pos_obs.shape

    # mask last horizon
    mask_last_horizon = jnp.concat(
        [jnp.ones(horizon - 1, dtype=dtype), jnp.zeros(1, dtype=dtype)]
    )
    # This is a permutation matrix to shift one timestep ahead
    shift_one_timestep = jnp.eye(horizon, k=-1, dtype=dtype)

    # 2Batch
    # pos_idx = jnp.ones(batch, dtype=dtype)
    pos_idx = jnp.concat([jnp.ones(batch, dtype=dtype), jnp.zeros(batch, dtype=dtype)])
    # 2Batch X 1
    pos_idx = pos_idx[:, None]
    # 2Batch X Horizon X obs/act_dim
    # target_obs, target_act = data.pos_obs, data.pos_act
    target_obs = jnp.concat([data.pos_obs, data.union_obs], axis=0)
    target_act = jnp.concat([data.pos_act, data.union_act], axis=0)

    b2, h, _ = target_obs.shape
    # 2Batch X act_dim
    pi_act, pi_log_prob, *_ = jax.lax.stop_gradient(
        policy_model(target_obs.reshape(b2 * h, -1))
    )

    def get_vpi(critic, obs):
        # 2Batch X obs_dim
        obs = obs.reshape(b2 * h, -1)
        # 2Batch
        q = jnp.minimum(*critic(jnp.concat([obs, pi_act], axis=-1)))
        v = q - alpha * pi_log_prob
        v = v.reshape(b2, h)
        return v

    def iqlearn_loss(q, v, vnext):
        # Batch X (Horizon-1): mask the last horizon step
        reward = (q - gamma * vnext)[:, :-1]
        total_pos_reward = batch * (horizon - 1)
        reward_pos_loss = -jnp.sum(pos_idx * reward) / total_pos_reward
        # Batch X (Horizon-1)
        v0 = (v - gamma * vnext)[:, :-1]
        value_loss = v0.mean()
        # Batch X (Horizon-1)
        chi_loss = 1 / (4 * lmbda) * (reward**2).mean()
        loss = reward_pos_loss + value_loss + chi_loss
        return loss, reward_pos_loss, value_loss, chi_loss

    def loss_fun(critic_model):
        q1, q2 = critic_model(jnp.concat([target_obs, target_act], axis=-1))
        v = get_vpi(critic_model, target_obs)
        # vnext = get_vpi(critic_model_target, target_obs) @ shift_one_timestep
        vnext = v @ shift_one_timestep
        qloss1, *aux_values1 = iqlearn_loss(q1, v, vnext)
        qloss2, *aux_values2 = iqlearn_loss(q2, v, vnext)
        loss = (qloss1 + qloss2) / 2
        aux_values = jnp.array((aux_values1, aux_values2)).mean(axis=0)
        return loss, tuple(aux_values)

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, aux_values), grads = grad_fun(critic_model)
    return loss, grads, *aux_values


def train_step(
    critic_model_target,
    critic_model,
    critic_optimizer,
    policy_model,
    policy_optimizer,
    batch_data,
    config,
    steps,
):
    critic_cond = (steps % config.update_critic_freq) == 0
    critic_loss, critic_grads, *critic_aux = critic_loss_grads_fun(
        critic_model_target=critic_model_target,
        critic_model=critic_model,
        policy_model=policy_model,
        data=batch_data,
        gamma=config.gamma,
        alpha=config.alpha,
        lmbda=config.lmbda,
    )
    critic_grads = jax.tree.map(
        lambda g: jnp.where(critic_cond, g, jnp.zeros_like(g)),
        critic_grads,
    )
    critic_optimizer.update(critic_grads)

    policy_loss, policy_grads, *policy_aux = policy_loss_grads_fun(
        critic_model=critic_model,
        policy_model=policy_model,
        data=batch_data,
        alpha=config.alpha,
    )
    policy_optimizer.update(policy_grads)

    critic_model_target = polyak_update(
        critic_model_target, critic_model, config.update_tau
    )

    mean_pos_reward = batch_data.pos_reward.sum(-1).mean()
    mean_neg_reward = jnp.array(0.0)
    mean_union_reward = batch_data.union_reward.sum(-1).mean()

    mean_pos_cost = batch_data.pos_cost.sum(-1).mean()
    mean_neg_cost = jnp.array(0.0)
    mean_union_cost = batch_data.union_cost.sum(-1).mean()

    return (
        critic_loss,
        *critic_aux,
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
    critic_model_target,
    critic_model,
    critic_optimizer,
    policy_model,
    policy_optimizer,
    data_buffer,
    config,
    key,
):
    num_steps = config.log_freq

    pos_idxs, neg_idxs, union_idxs = data_buffer.sample_idxs(
        data_buffer, key, num_steps
    )

    def body_fun(i, carry):
        (
            _,
            critic_model_target,
            critic_model,
            critic_optimizer,
            policy_model,
            policy_optimizer,
        ) = carry

        batch_data = data_buffer.sample_batch(
            data_buffer, pos_idxs[i], neg_idxs[i], union_idxs[i]
        )

        val = train_step(
            critic_model_target=critic_model_target,
            critic_model=critic_model,
            critic_optimizer=critic_optimizer,
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            batch_data=batch_data,
            config=config,
            steps=i,
        )

        return (
            val,
            critic_model_target,
            critic_model,
            critic_optimizer,
            policy_model,
            policy_optimizer,
        )

    init_val = (jnp.zeros((), dtype=jnp.float32),) * 13
    init_carry = (
        init_val,
        critic_model_target,
        critic_model,
        critic_optimizer,
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

    trajectory_cfg["num_positive_trajectories"] = args.num_preferred
    trajectory_cfg["num_negative_trajectories"] = 0
    trajectory_cfg["num_union_trajectories"] = args.num_union
    trajectory_cfg["non_pref_noise"] = args.non_pref_noise
    trajectory_cfg["data_inpaint"] = args.data_inpaint
    trajectory_cfg["inpaint_ranges"] = trajectory_data[args.data_inpaint]

    config = {**default_cfg, **trajectory_cfg}
    config["train_horizon"] = args.train_horizon or config.get("train_horizon")
    config["alpha"] = args.bc_weight_temp
    config["lmbda"] = args.lmbda
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

    critic_model = EnsembleValue(
        rngs=rngs,
        x_dim=obs_space.shape[0] + act_space.shape[0],
        hidden_size=config["hidden_size"],
    )
    critic_optimizer = nnx.Optimizer(
        model=critic_model,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adamw(
                learning_rate=config["lr"], weight_decay=config["weight_decay"]
            ),
        ),
    )
    critic_model_target = deepcopy(critic_model)

    # data
    agent_task = re.search(r"Offline(.*?)(?:Gymnasium)?-v[0-9]", args.task).group(1)
    ep_len = dsrl_infos.DEFAULT_MAX_EPISODE_STEPS[agent_task]
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

    buffer = OnPolicyBuffer(
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
    eval_rew_deque = deque(maxlen=config["eval_episode_freq"])
    eval_cost_deque = deque(maxlen=config["eval_episode_freq"])
    eval_len_deque = deque(maxlen=config["eval_episode_freq"])
    dict_args = config
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)

    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    logger.save_config(dict_args)
    logger.log("Start critic, value and policy model training.")

    steps = 0
    while steps < config["total_iteration"]:

        val, num_itr = train_n_steps(
            critic_model_target=critic_model_target,
            critic_model=critic_model,
            critic_optimizer=critic_optimizer,
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            data_buffer=data_buffer,
            config=config_data,
            key=rngs.random_sample(),
        )

        (
            critic_loss,
            critic_reward_loss,
            critic_value_loss,
            critic_chi_loss,
            policy_loss,
            policy_q,
            policy_entropy,
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
                    eval_reward, eval_cost, eval_len = evaluate_bc_policy(
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

            logger.log_tabular("Loss/Loss_critic", critic_loss.item())
            logger.log_tabular(
                "Loss/Loss_critic_reward_loss", critic_reward_loss.item()
            )
            logger.log_tabular("Loss/Loss_critic_value_loss", critic_value_loss.item())
            logger.log_tabular("Loss/Loss_critic_chi_loss", critic_chi_loss.item())

            logger.log_tabular("Loss/Loss_policy", policy_loss.item())
            logger.log_tabular("Loss/policy_qvalue", policy_q.item())
            logger.log_tabular("Loss/policy_entropy", policy_entropy.item())

            logger.log_tabular("Reward/pos", mean_pos_reward.item())
            logger.log_tabular("Reward/neg", mean_neg_reward.item())
            logger.log_tabular("Reward/union", mean_union_reward.item())

            logger.log_tabular("Cost/pos", mean_pos_cost.item())
            logger.log_tabular("Cost/neg", mean_neg_cost.item())
            logger.log_tabular("Cost/union", mean_union_cost.item())

            logger.log_tabular(
                "Norm/critic_model",
                get_tree_norm(nnx.state(critic_model, nnx.Param)),
            )
            logger.log_tabular(
                "Norm/policy_model",
                get_tree_norm(nnx.state(policy_model, nnx.Param)),
            )
            logger.dump_tabular()

        if steps % config["save_freq"] == 0:
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=critic_model,
                prefix="critic",
            )
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=policy_model,
                prefix="bc_policy",
            )

        if steps >= config["total_iteration"]:
            break

    logger.nn_model_save(
        itr=steps, nn_model_saver_element=critic_model, prefix="critic"
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
