import os
import os.path as osp
import random
import re
import sys
import time
from collections import deque
from copy import deepcopy
from functools import partial

import dsrl.infos as dsrl_infos
import dsrl.offline_safety_gymnasium  # type: ignore
import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR

from dsrl_model.utils.bufffer import OnPolicyBuffer
from dsrl_model.utils.dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_neg_and_union_data,
)
from dsrl_model.utils.logger import EpochLogger
from dsrl_model.utils.models import (
    BcqVAE,
    Encoder,
    EnsembleValue,
    ExpCostModel,
    SafeDiceTanhMixtureActor,
    TdmpcCostModel,
    TdmpcDynamics,
)
from dsrl_model.utils.save_video_with_value import save_video
from dsrl_model.utils.utils import ActionRepeater, get_params_norm, single_agent_args

EP = 1e-6

default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "eval_episode_freq": 1,  # use saved bc_policy to run evaluatation
    "hidden_sizes": [256, 256],
    "latent_obs_dim": 50,
    "max_grad_norm": 10.0,
    "gamma": 0.99,
    "lambda_c": 0.95,
    "action_repeat": 1,  # set to 2, min value is 1
    "explore_noise_std": 0.5,
    "update_freq": 1,
    "update_tau": 0.005,
    "dynamics_coef": 2.0,  # TDMPC update coef
    "bc_coef": 0.5,
    "cost_coef": 0.5,  # TDMPC update coef
    "value_coef": 0.1,  # TDMPC update coef
    "cost_weight_temp": 0.5,  # TDMPC temperature coef
    "elite_portion": 0.1,  # 0.1
    "num_samples": 400,  # 400
    "inference_horizon": 5,  # 5
    "train_horizon": 5,  # 5
    "weight_decay": 0.01,
    "total_iteration": int(1e6),
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
    "percentage_validation_trajectories": 0.2,
}


def ema(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    # implementation from td-mpc
    # target_params = (1-tau) * target_params + tau * params
    # tau is generally a small number.
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)


@torch.no_grad
def mcem_policy(encoder, dynamics, cost_model, value, bc_policy, config, device, obs):
    horizon = config["inference_horizon"]
    num_samples = config["num_samples"]
    num_elite = int(config["elite_portion"] * num_samples)
    gamma = config["gamma"]
    policy_type = config["policy_type"]
    # mask = torch.arange(num_samples) >= num_samples // 2
    # noise_std = mask.to(device).unsqueeze(1) * config["explore_noise_std"]
    z = encoder(obs)
    sample_next_zs = torch.zeros(
        (horizon, num_samples, z.shape[-1]), dtype=torch.float32, device=device
    )
    sample_acts = torch.zeros(
        (horizon, num_samples, bc_policy.act_dim),
        dtype=torch.float32,
        device=device,
    )

    init_zs = zs = z.repeat(num_samples, 1)
    for t in range(horizon):
        if policy_type == "vae":
            acts = bc_policy.decode_bc(zs)
        else:
            acts = bc_policy.action(zs, deterministic=False)
        zs = dynamics(zs, acts)
        sample_acts[t] = acts
        sample_next_zs[t] = zs

    # size: num_samples
    advantage = calculate_target_value(
        cost_model=cost_model,
        value=value,
        target_z=init_zs,
        target_acts=sample_acts,
        target_next_zs=sample_next_zs,
        config=config,
    )[0]

    best_control_idx = torch.argsort(advantage)[:num_elite]
    elite_controls = sample_acts[0][best_control_idx]
    elite_costs = advantage[best_control_idx]
    # weights = torch.exp(-config["cost_weight_temp"] * elite_costs)
    # weights /= weights.sum()
    # weighted_cost = (elite_costs * weights).sum() / (weights.sum() + EP)
    # weighted_controls = torch.sum(
    #     weights.view(-1, 1, 1).repeat(1, horizon, 1) * elite_controls, dim=0
    # ) / (weights.sum() + EP)

    return (
        elite_controls.mean(dim=0),
        elite_costs.mean(),
        # elite_controls.mean(dim=0),
        elite_costs.min(),
        elite_costs.max(),
        advantage.mean(),
        advantage.max() - advantage.min(),
    )


@torch.no_grad
def evaluate_bc_policy(eval_env, policy, device):
    eval_done = False
    eval_obs, _ = eval_env.reset()
    # eval_obs = (eval_obs - mu_obs) / (std_obs + EP)
    eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(
        0
    )
    eval_reward, eval_cost, eval_pred_cost, eval_len = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    while not eval_done:
        act, pred_cost, *_ = policy(eval_obs)
        next_obs, reward, terminated, truncated, info = eval_env.step(
            act.detach().squeeze().cpu().numpy()
        )
        cost = info["cost"]
        # next_obs = (next_obs - mu_obs) / (std_obs + EP)
        next_obs = torch.as_tensor(
            next_obs, dtype=torch.float32, device=device
        ).unsqueeze(0)
        eval_obs = next_obs
        eval_reward += reward
        eval_cost += cost
        eval_pred_cost += pred_cost
        eval_len += 1
        eval_done = terminated or truncated
    return eval_reward, eval_cost, eval_pred_cost, eval_len


def bc_policy_loss_fn(bc_policy, dynamics, encoder, target_o, target_acts, config):
    gamma = config["gamma"]
    horizon = target_acts.shape[0]
    pred_tz = encoder(target_o)
    loss, discount = 0.0, 1.0
    for t in range(horizon):
        ta = target_acts[t]
        if config["policy_type"] == "vae":
            pred_act, bc_mean, bc_std = bc_policy(pred_tz, ta)
            recon_loss = F.mse_loss(pred_act, ta, reduction="none").sum(dim=1)
            kl_loss = -0.5 * (
                1 + torch.log(bc_std.pow(2)) - bc_mean.pow(2) - bc_std.pow(2)
            ).sum(dim=1)
            # 0.5 weight is from BCQ implementation See @aviralkumar implementation
            loss += discount * (recon_loss + 0.5 * kl_loss)
        else:
            pred_act, *_ = bc_policy(pred_tz)
            recon_loss = F.mse_loss(pred_act, ta, reduction="none").sum(dim=1)
            loss += discount * recon_loss

        pred_tz = dynamics(pred_tz, ta)
        discount *= gamma
    return torch.mean(loss)


def dynamics_loss_fn(
    dynamics,
    encoder,
    encoder_target,
    target_o,
    target_acts,
    target_next_os,
    config,
):
    gamma = config["gamma"]
    horizon = target_acts.shape[0]
    discount, dynamics_loss = 1.0, 0.0
    pred_tz = encoder(target_o)
    for t in range(horizon):
        ta, tno = target_acts[t], target_next_os[t]
        with torch.no_grad():
            tnz = encoder_target(tno)
        pred_nz = dynamics(pred_tz, ta)
        dynamics_loss += discount * F.mse_loss(pred_nz, tnz, reduction="none").sum(
            dim=1
        )
        discount *= gamma
        pred_tz = pred_nz
    return torch.mean(dynamics_loss)


def cost_loss_fn(
    cost_model,
    dynamics,
    encoder,
    target_neg_o,
    target_neg_acts,
    target_union_o,
    target_union_acts,
    config,
):
    gamma = config["gamma"]
    horizon = target_neg_acts.shape[0]
    discount, total_neg_pref, total_union_pref = 1.0, 0.0, 0.0
    device = target_neg_o.device

    tnz, tuz = encoder(target_neg_o), encoder(target_union_o)

    for t in range(horizon):
        tna, tua = target_neg_acts[t], target_union_acts[t]
        tn, tu = torch.cat([tnz, tna], dim=1), torch.cat([tuz, tua], dim=1)
        total_neg_pref += discount * cost_model(tn, use_sigmoid=True)
        total_union_pref += discount * cost_model(tu, use_sigmoid=True)
        discount *= gamma
        tnz, tuz = dynamics(tnz, tna), dynamics(tuz, tua)

    exp_neg, exp_union = torch.exp(total_neg_pref), torch.exp(total_union_pref)
    sum_exp = exp_neg + exp_union
    p_neg, p_union = exp_neg / sum_exp, exp_union / sum_exp
    target_zeros = torch.zeros_like(p_union, device=device)
    target_ones = torch.ones_like(p_neg, device=device)
    loss = F.binary_cross_entropy(p_union, target_zeros)
    loss += F.binary_cross_entropy(p_neg, target_ones)

    return torch.mean(loss)


@torch.no_grad
def calculate_target_value(
    cost_model, value, target_z, target_acts, target_next_zs, config
):
    gamma, lambda_c = config["gamma"], config["lambda_c"]

    device = target_z.device
    horizon, batch_size, _ = target_next_zs.shape

    advantage_c = torch.zeros((horizon, batch_size), dtype=torch.float32, device=device)
    value_c = torch.zeros((horizon, batch_size), dtype=torch.float32, device=device)

    z = target_z
    v = torch.min(*value.V(z))
    for t in range(horizon):
        a, z_next = target_acts[t], target_next_zs[t]
        c = cost_model(torch.cat([z, a], dim=1), use_sigmoid=True)
        v_next = torch.min(*value.V(z_next))
        advantage_c[t] = c + gamma * v_next - v
        value_c[t] = v
        v = v_next
    cumsum = advantage_c[-1]
    for t in reversed(range(horizon - 1)):
        cumsum = advantage_c[t] + gamma * lambda_c * cumsum
        advantage_c[t] = cumsum

    return advantage_c + value_c


def value_loss_fn(
    cost_model,
    dynamics,
    encoder,
    value,
    value_target,
    target_o,
    target_acts,
    target_next_os,
    config,
):
    gamma = config["gamma"]
    horizon = target_acts.shape[0]
    target_value = calculate_target_value(
        cost_model=cost_model,
        value=value_target,
        target_z=encoder(target_o),
        target_acts=target_acts,
        target_next_zs=encoder(target_next_os),
        config=config,
    )
    z = encoder(target_o)
    discount, value_loss, priority_loss = 1.0, 0.0, 0.0
    for t in range(horizon):
        pred_v1, pred_v2 = value.V(z)
        value_loss += discount * (
            F.mse_loss(pred_v1, target_value[t], reduction="none")
            + F.mse_loss(pred_v2, target_value[t], reduction="none")
        )
        priority_loss += discount * (
            F.l1_loss(pred_v1, target_value[t], reduction="none")
            + F.l1_loss(pred_v2, target_value[t], reduction="none")
        )
        discount *= gamma
        z = dynamics(z, target_acts[t])

    return torch.mean(value_loss), torch.mean(priority_loss)


def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(f"{args.device}:{args.device_id}")
    config = {**default_cfg, **trajectory_cfg}
    config["train_horizon"] = args.train_horizon or config.get("train_horizon")
    config["cost_weight_temp"] = args.cost_weight_temp or config["cost_weight_temp"]
    config["policy_type"] = args.policy_type

    # evaluation environment
    eval_env = gym.make(args.task)
    if args.save_video:
        eval_env.render_parameters.mode = "rgb_array"
        eval_env.render_parameters.camera_name = "track"
    eval_env.set_target_cost(config["target_cost"])
    eval_env = ActionRepeater(eval_env, num_repeats=config["action_repeat"])
    eval_env.reset(seed=args.seed)

    # set training steps
    num_epochs = config.get("num_epochs", args.num_epochs)
    batch_size = config.get("batch_size", args.batch_size)

    # set model
    obs_space, act_space = eval_env.observation_space, eval_env.action_space
    encoder = Encoder(
        obs_dim=obs_space.shape[0], latent_dim=config["latent_obs_dim"]
    ).to(device)
    encoder_target = deepcopy(encoder)
    encoder_optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=args.lr, weight_decay=config["weight_decay"]
    )
    config["bc_lr"] = args.lr
    if config["policy_type"] == "vae":
        # See BEAR implementation from @aviralkumar
        bc_latent_dim = config.get("latent_dim", act_space.shape[0] * 2)
        config["bc_latent_dim"] = bc_latent_dim
        bc_policy = BcqVAE(
            obs_dim=config["latent_obs_dim"],
            act_dim=act_space.shape[0],
            latent_dim=bc_latent_dim,
            device=device,
        ).to(device)
    else:
        bc_policy = SafeDiceTanhMixtureActor(
            obs_dim=config["latent_obs_dim"],
            act_dim=act_space.shape[0],
            hidden_size=config["hidden_sizes"][0],
        ).to(device)
    bc_policy_optimizer = torch.optim.AdamW(
        bc_policy.parameters(), lr=config["bc_lr"], weight_decay=config["weight_decay"]
    )
    # bc_scheduler = LinearLR(
    #     bc_policy_optimizer,
    #     start_factor=1.0,
    #     end_factor=0.0,
    #     total_iters=config["total_iteration"],
    # )
    cost_model = ExpCostModel(
        # (s,a)
        obs_dim=config["latent_obs_dim"] + act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    cost_optimizer = torch.optim.AdamW(
        cost_model.parameters(), lr=args.lr, weight_decay=config["weight_decay"]
    )
    dynamics = TdmpcDynamics(
        obs_dim=config["latent_obs_dim"],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    dynamics_optimizer = torch.optim.AdamW(
        dynamics.parameters(), lr=args.lr, weight_decay=config["weight_decay"]
    )
    value_cost = EnsembleValue(
        obs_dim=config["latent_obs_dim"],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    value_cost_target = deepcopy(value_cost)
    value_cost_optimizer = torch.optim.AdamW(
        value_cost.parameters(), lr=args.lr, weight_decay=config["weight_decay"]
    )

    # data
    agent_task = re.search(r"Offline(.*?)Gymnasium-v[0-9]", args.task).group(1)
    ep_len = dsrl_infos.DEFAULT_MAX_EPISODE_STEPS[agent_task]
    data = get_dataset_in_d4rl_format(
        eval_env, trajectory_cfg, args.task, ep_len, config["action_repeat"]
    )
    neg_data, union_data = get_neg_and_union_data(data, trajectory_cfg)
    # neg_data, union_data, mu_obs, std_obs = get_normalized_data(neg_data, union_data)
    neg_observations = torch.as_tensor(
        neg_data["observations"], dtype=torch.float32, device=device
    )
    neg_actions = torch.as_tensor(
        neg_data["actions"], dtype=torch.float32, device=device
    )
    neg_dones = neg_data["timeouts"] | neg_data["terminals"]
    union_observations = torch.as_tensor(
        union_data["observations"], dtype=torch.float32, device=device
    )
    union_actions = torch.as_tensor(
        union_data["actions"], dtype=torch.float32, device=device
    )
    union_dones = union_data["timeouts"] | union_data["terminals"]

    # create negative validation dataset
    valid_neg_size = int(
        neg_observations.shape[0] * trajectory_cfg["percentage_validation_trajectories"]
    )
    valid_neg_observations = neg_observations[:valid_neg_size]
    valid_neg_actions = neg_actions[:valid_neg_size]
    valid_neg_dones = neg_dones[:valid_neg_size]
    neg_observations = neg_observations[valid_neg_size:]
    neg_actions = neg_actions[valid_neg_size:]
    neg_dones = neg_dones[valid_neg_size:]

    # create union validation dataset
    valid_union_size = int(
        union_observations.shape[0]
        * trajectory_cfg["percentage_validation_trajectories"]
    )
    valid_union_observations = union_observations[:valid_union_size]
    valid_union_actions = union_actions[:valid_union_size]
    valid_union_dones = union_dones[:valid_union_size]
    union_observations = union_observations[valid_union_size:]
    union_actions = union_actions[valid_union_size:]
    union_dones = union_dones[valid_union_size:]

    ep_len = ep_len // config["action_repeat"] + (ep_len % config["action_repeat"] > 0)
    assert (
        neg_observations.shape[1] == ep_len
    ), f"{neg_observations.shape[1]} episode length is different from {ep_len}"

    buffer = OnPolicyBuffer(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        neg_data_size=np.prod(neg_observations.shape[:-1]),
        union_data_size=np.prod(union_observations.shape[:-1]),
        horizon=config["train_horizon"],
        batch_size=batch_size,
        device=device,
        ep_len=ep_len,
    )
    for obs, act, done in zip(neg_observations, neg_actions, neg_dones):
        buffer.add(obs, act, done, is_negative=True)
    for obs, act, done in zip(union_observations, union_actions, union_dones):
        buffer.add(obs, act, done, is_negative=False)

    # set logger
    eval_rew_deque = deque(maxlen=config["eval_episode_freq"])
    eval_cost_deque = deque(maxlen=config["eval_episode_freq"])
    eval_norm_rew_deque = deque(maxlen=config["eval_episode_freq"])
    eval_norm_cost_deque = deque(maxlen=config["eval_episode_freq"])
    eval_pred_cost_deque = deque(maxlen=config["eval_episode_freq"])
    eval_len_deque = deque(maxlen=config["eval_episode_freq"])
    dict_args = vars(args)
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    logger.save_config(dict_args)
    logger.log("Start with bc_policy, dynamics and cost training.")

    # train critic and dynamics model
    steps = 0
    while steps < config["total_iteration"]:
        # shape: Horizon X Batch X obs/act_dim
        for (
            target_neg_os,
            target_neg_acts,
            target_union_os,
            target_union_acts,
        ) in buffer.sample():

            steps += 1

            target_neg_acts = target_neg_acts[:-1]
            target_union_acts = target_union_acts[:-1]
            target_union_next_os = target_union_os[1:]

            dynamics_loss = dynamics_loss_fn(
                dynamics=dynamics,
                encoder=encoder,
                encoder_target=encoder_target,
                target_o=target_union_os[0],
                target_acts=target_union_acts,
                target_next_os=target_union_next_os,
                config=config,
            )

            bc_policy_loss = bc_policy_loss_fn(
                bc_policy=bc_policy,
                dynamics=dynamics,
                encoder=encoder,
                target_o=target_union_os[0],
                target_acts=target_union_acts,
                config=config,
            )

            cost_loss = cost_loss_fn(
                cost_model=cost_model,
                dynamics=dynamics,
                encoder=encoder,
                target_neg_o=target_neg_os[0],
                target_neg_acts=target_neg_acts,
                target_union_o=target_union_os[0],
                target_union_acts=target_union_acts,
                config=config,
            )

            value_loss, priority_loss = value_loss_fn(
                cost_model=cost_model,
                dynamics=dynamics,
                encoder=encoder,
                value=value_cost,
                value_target=value_cost_target,
                target_o=target_union_os[0],
                target_acts=target_union_acts,
                target_next_os=target_union_next_os,
                config=config,
            )

            total_loss = (
                config["dynamics_coef"] * dynamics_loss
                + config["bc_coef"] * bc_policy_loss
                + config["cost_coef"] * cost_loss
                + config["value_coef"] * value_loss
            )
            total_loss.register_hook(lambda grad: grad * (1 / config["train_horizon"]))

            encoder_optimizer.zero_grad()
            dynamics_optimizer.zero_grad()
            cost_optimizer.zero_grad()
            value_cost_optimizer.zero_grad()
            bc_policy_optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(encoder.parameters(), config["max_grad_norm"])
            clip_grad_norm_(dynamics.parameters(), config["max_grad_norm"])
            clip_grad_norm_(bc_policy.parameters(), config["max_grad_norm"])
            clip_grad_norm_(cost_model.parameters(), config["max_grad_norm"])
            clip_grad_norm_(value_cost.parameters(), config["max_grad_norm"])
            encoder_optimizer.step()
            dynamics_optimizer.step()
            cost_optimizer.step()
            value_cost_optimizer.step()
            bc_policy_optimizer.step()
            # bc_scheduler.step()

            if (steps % config["update_freq"]) == 0:
                ema(encoder, encoder_target, config["update_tau"])
                ema(value_cost, value_cost_target, config["update_tau"])

            logger.logged = False

            if (steps % config["log_freq"] == 0) and (not logger.logged):
                eval_episodes = config["eval_episode_freq"]
                if args.use_eval:
                    eval_start_time = time.time()
                    cem_policy = partial(
                        mcem_policy,
                        encoder,
                        dynamics,
                        cost_model,
                        value_cost,
                        bc_policy,
                        config,
                        device,
                    )
                    for id in range(eval_episodes):
                        (
                            eval_reward,
                            eval_cost,
                            eval_pred_cost,
                            eval_len,
                        ) = evaluate_bc_policy(
                            eval_env=eval_env,
                            policy=cem_policy,
                            device=device,
                        )
                        norm_reward, norm_cost = eval_env.get_normalized_score(
                            eval_reward, eval_cost
                        )
                        eval_norm_rew_deque.append(norm_reward)
                        eval_norm_cost_deque.append(norm_cost)
                        eval_rew_deque.append(eval_reward)
                        eval_cost_deque.append(eval_cost)
                        eval_pred_cost_deque.append(eval_pred_cost)
                        eval_len_deque.append(eval_len)

                    eval_end_time = time.time()

                    logger.log_tabular("Metrics/EvalEpRet", np.mean(eval_rew_deque))
                    logger.log_tabular("Metrics/EvalEpCost", np.mean(eval_cost_deque))
                    logger.log_tabular(
                        "Metrics/EvalEpPredCost", np.mean(eval_pred_cost_deque)
                    )
                    logger.log_tabular(
                        "Metrics/EvalEpNormRet", np.mean(eval_norm_rew_deque)
                    )
                    logger.log_tabular(
                        "Metrics/EvalEpNormCost", np.mean(eval_norm_cost_deque)
                    )
                    logger.log_tabular("Metrics/EvalEpLen", np.mean(eval_len_deque))

                logger.log_tabular("Train/Steps", steps)
                logger.log_tabular("Loss/Loss_bc_policy", bc_policy_loss.mean().item())
                logger.log_tabular("Loss/Loss_dynamics", dynamics_loss.mean().item())
                logger.log_tabular("Loss/Loss_cost", cost_loss.mean().item())
                logger.log_tabular("Loss/Loss_value_cost", value_loss.mean().item())
                logger.log_tabular("Loss/Loss_total", total_loss.mean().item())

                logger.log_tabular(
                    "Norm/encoder",
                    get_params_norm(encoder.parameters(), grads=False),
                )
                logger.log_tabular(
                    "Norm/bc_policy",
                    get_params_norm(bc_policy.parameters(), grads=False),
                )
                logger.log_tabular(
                    "Norm/dynamics",
                    get_params_norm(dynamics.parameters(), grads=False),
                )
                logger.log_tabular(
                    "Norm/cost_model",
                    get_params_norm(cost_model.parameters(), grads=False),
                )
                logger.log_tabular(
                    "Norm/value_cost",
                    get_params_norm(value_cost.parameters(), grads=False),
                )
                if args.use_eval:
                    logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
                logger.dump_tabular()

            if steps % config["save_freq"] == 0:
                logger.torch_save(
                    itr=steps,
                    torch_saver_elements=encoder,
                    prefix="tdmpc_encoder",
                )
                logger.torch_save(
                    itr=steps,
                    torch_saver_elements=bc_policy,
                    prefix="bc_policy",
                )
                logger.torch_save(
                    itr=steps,
                    torch_saver_elements=dynamics,
                    prefix="tdmpc_dynamics",
                )
                logger.torch_save(
                    itr=steps,
                    torch_saver_elements=cost_model,
                    prefix="cost_model",
                )
                logger.torch_save(
                    itr=steps,
                    torch_saver_elements=value_cost,
                    prefix="value_cost",
                )

            if steps >= config["total_iteration"]:
                break

    logger.torch_save(itr=steps, torch_saver_elements=encoder, prefix="tdmpc_encoder")
    logger.torch_save(itr=steps, torch_saver_elements=bc_policy, prefix="bc_policy")
    logger.torch_save(itr=steps, torch_saver_elements=dynamics, prefix="tdmpc_dynamics")
    logger.torch_save(itr=steps, torch_saver_elements=cost_model, prefix="cost_model")
    logger.torch_save(itr=steps, torch_saver_elements=value_cost, prefix="value_cost")
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
