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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR

from dsrl_model.utils.bufffer import SafeTD3Buffer
from dsrl_model.utils.dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_neg_and_union_data_2,
    get_normalized_data,
)
from dsrl_model.utils.logger import EpochLogger
from dsrl_model.utils.models import (
    BcqVAE,
    ContrastiveCostModel,
    Encoder,
    EnsembleValue,
    ExpCostModel,
    SafeDiceTanhMixtureActor,
    SafeTransformerCritic,
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
    "max_grad_norm_cost": 1.0,
    "max_grad_norm_critic": 1.0,
    "max_grad_norm_bc": 1.0,
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
    "inpaint_ranges": ((0.0, 1.0, 0.0, 0.5),),
    "num_negative_trajectories": 50,
    "num_union_trajectories": -1,
}


def ema(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    # implementation from td-mpc
    # target_params = (1-tau) * target_params + tau * params
    # tau is generally a small number.
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)


def normalize_observation(mu_obs, std_obs, obs):
    if mu_obs is None:
        return obs
    return (obs - mu_obs) / (std_obs + EP)


@torch.no_grad
def evaluate_bc_policy(eval_env, policy, cost_model, device, norm_fn):
    eval_done = False
    eval_obs, _ = eval_env.reset()
    eval_obs = torch.as_tensor(
        norm_fn(eval_obs), dtype=torch.float32, device=device
    ).unsqueeze(0)
    eval_reward, eval_cost, eval_pred_cost, eval_len = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    while not eval_done:
        act = policy(eval_obs)
        next_obs, reward, terminated, truncated, info = eval_env.step(
            act.detach().squeeze().cpu().numpy()
        )
        cost = info["cost"]
        next_obs = torch.as_tensor(
            norm_fn(next_obs), dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            pred_cost = cost_model(torch.cat([eval_obs, act], dim=1)).item()
        eval_obs = next_obs
        eval_reward += reward
        eval_cost += cost
        eval_pred_cost += pred_cost
        eval_len += 1
        eval_done = terminated or truncated
    return eval_reward, eval_cost, eval_pred_cost, eval_len


def bc_policy_loss_fn(bc_policy, value, target_os, target_acts, config):
    alpha, beta = config["value_weight_temp"], config["bc_weight_temp"]

    # Horizon X Batch X obs/act_dim
    horizon, batch_size, _ = target_os.shape
    # Horizon_Batch X obs/act_dim
    target_os = target_os.view(horizon * batch_size, -1)
    target_acts = target_acts.view(horizon * batch_size, -1)

    bc_loss = 0.0

    pred_acts = None
    if config["policy_type"] == "vae":
        pred_acts, bc_mean, bc_std = bc_policy(target_os, target_acts)
        recon_loss = F.mse_loss(pred_acts, target_acts, reduction="none").sum(dim=1)
        kl_loss = -0.5 * (
            1 + torch.log(bc_std.pow(2)) - bc_mean.pow(2) - bc_std.pow(2)
        ).sum(dim=1)
        # 0.5 weight is from BCQ implementation See @aviralkumar implementation
        bc_loss = recon_loss + 0.5 * kl_loss
    else:
        pred_acts, *_ = bc_policy(target_os)
        recon_loss = F.mse_loss(pred_acts, target_acts, reduction="none").sum(dim=1)
        bc_loss = recon_loss

    v = torch.max(*value.V(torch.cat([target_os, pred_acts], dim=1)))
    q = torch.max(*value.V(torch.cat([target_os, target_acts], dim=1)))

    # bc_lambda = torch.exp(-(q - v) / alpha).detach()
    # bc_lambda /= torch.mean(weight) + EP
    # the following lines calculate the same as above
    neg_adv = (-(q - v) / beta).detach()
    log_Z = torch.logsumexp(neg_adv, dim=0) - np.log(neg_adv.shape[0]) + EP
    bc_beta = torch.exp(neg_adv - log_Z)
    v_beta = torch.exp(log_Z.clamp(max=5.0))

    if config["use_td3_style_bc"]:
        # v_lambda = (alpha * bc_loss.mean() / (torch.mean(torch.abs(v)) + EP)).detach()
        v_alpha = alpha
        loss = bc_loss + v_alpha * v_beta * v
    else:
        loss = bc_beta * bc_loss
    return torch.mean(loss), torch.mean(torch.abs(q)), torch.mean(torch.abs(v))


def compute_contrastive_ce_loss(p, q):
    q = F.log_softmax(q, dim=1)
    p = p / p.sum(dim=1, keepdim=True).clamp(min=1.0)
    loss = torch.sum(p * q, dim=1)
    return -torch.mean(loss)


def cost_contrastive_loss_fn(
    cost_model,
    target_neg_os,
    target_neg_acts,
    target_union_os,
    target_union_acts,
    config,
    bootstrap_lambda=0.5,
):
    del bootstrap_lambda

    horizon, batch_size, _ = target_neg_acts.shape
    device = target_neg_os.device

    temperature = 0.1  # value from SupContrast
    num_neg_extra_traj = config["num_neg_extra_traj"]

    tn = torch.cat([target_neg_os, target_neg_acts], dim=-1)
    tu = torch.cat([target_union_os, target_union_acts], dim=-1)
    (neg_zs, _), (union_zs, _) = cost_model(tn), cost_model(tu)

    # horizon mask
    horizon_mask = torch.ones((horizon, horizon), device=device, dtype=torch.float32)
    mask = torch.eye(batch_size, device=device, dtype=torch.float32)

    # logits for union and non-pref
    union_zs = torch.concat(torch.unbind(union_zs, dim=1), dim=0)  # HB x obs
    neg_zs = torch.concat(torch.unbind(neg_zs, dim=1), dim=0)  # HB x obs
    combined_zs = torch.concat([union_zs, neg_zs], dim=0)  # 2HB x obs
    combined_logits = torch.matmul(combined_zs, combined_zs.T) / temperature
    # remove the self instance from the loss fn
    diag_mask = torch.eye(combined_logits.shape[0], device=device).bool()
    combined_logits = combined_logits.masked_fill(diag_mask, -1e9)
    combined_logits_max, _ = torch.max(combined_logits, dim=1, keepdim=True)
    combined_logits = combined_logits - combined_logits_max.detach()

    # mask for union and non-pref
    union_mask = torch.kron(mask, horizon_mask)
    if num_neg_extra_traj > 0:
        zeros_pos = (mask == 0).to(torch.float32)
        indx = torch.multinomial(zeros_pos, num_neg_extra_traj, replacement=False)
        neg_mask = mask.scatter(1, indx, 1)
        # neg_mask = torch.ones((batch_size, batch_size), device=device, dtype=torch.float32)
        neg_mask = torch.kron(neg_mask, horizon_mask)
    else:
        neg_mask = torch.kron(mask, horizon_mask)
    combined_mask = torch.block_diag(union_mask, neg_mask)
    # remove the self instance from the loss fn
    combined_mask.fill_diagonal_(0.0)

    loss = compute_contrastive_ce_loss(combined_mask, combined_logits)
    return loss


def cost_pref_loss_fn(
    cost_model,
    target_neg_os,
    target_neg_acts,
    target_union_os,
    target_union_acts,
    config,
    bootstrap_lambda=0.5,
):
    gamma = config["gamma"]
    horizon = target_neg_acts.shape[0]
    device = target_neg_os.device
    tn = torch.cat([target_neg_os, target_neg_acts], dim=-1)
    tu = torch.cat([target_union_os, target_union_acts], dim=-1)

    if config["use_cost_attention"]:
        p_neg = cost_model(tn, use_sigmoid=True)
        p_union = cost_model(tu, use_sigmoid=True)
    else:
        discount, total_neg_pref, total_union_pref = 1.0, 0.0, 0.0
        for t in range(horizon):
            if config["use_cost_contrastive"] or config["pretrain_cost_contrastive"]:
                (_, cost_neg), (_, cost_union) = cost_model(tn[t]), cost_model(tu[t])
            else:
                cost_neg = cost_model(tn[t], use_sigmoid=True)
                cost_union = cost_model(tu[t], use_sigmoid=True)
            total_neg_pref += discount * cost_neg
            total_union_pref += discount * cost_union
            discount *= gamma

        if config["use_expected_cost_pref"]:
            bag_size = 32
            total_neg_pref = total_neg_pref.view(-1, bag_size).mean(dim=-1)
            total_union_pref = total_union_pref.view(-1, bag_size).mean(dim=-1)

        exp_neg, exp_union = torch.exp(total_neg_pref), torch.exp(total_union_pref)
        sum_exp = exp_neg + exp_union
        p_neg, p_union = exp_neg / sum_exp, exp_union / sum_exp

    if config["use_expected_cost_pref"]:
        target_union = torch.zeros_like(p_union, device=device)
    else:
        target_union = bootstrap_lambda * p_union.detach()

    loss = F.binary_cross_entropy(p_union, target_union)
    # # the following lines are not needed.
    # as addining this lines makes the total loss as:
    # say p = p_union,
    # loss = - (\lambda p) log(p) - (1-\lambda p) log(1-p) - log(1-p)
    # the last term -log(1-p) is not needed, as the gradient update can make it unstable.
    # target_neg = torch.ones_like(p_neg, device=device)
    # loss += F.binary_cross_entropy(p_neg, target_neg)

    return torch.mean(loss)


@torch.no_grad
def get_cost(cost_model, obs, acts, config):
    horizon = obs.shape[0]

    oa = torch.cat([obs, acts], dim=-1)

    if config["use_cost_contrastive"] or config["pretrain_cost_contrastive"]:
        _, cost = cost_model(oa)
    else:
        cost = cost_model(oa, use_sigmoid=True)
        if config["use_cost_attention"]:
            # for attention based cost size is: #batch
            # convert to size: horizon x batch
            cost = (
                cost.unsqueeze(1)
                .repeat(1, horizon)
                .permute(1, 0)
                .contiguous()
                .view(horizon, -1)
            )

    return cost


@torch.no_grad
def calculate_target_value(
    value,
    bc_policy,
    target_next_os,
    target_done,
    target_cost,
    gamma,
):
    policy_a = bc_policy.sample_action(target_next_os)
    v_next = torch.max(*value.V(torch.cat([target_next_os, policy_a], dim=1)))
    value_c = target_cost + gamma * (1 - target_done) * v_next
    return value_c


def discounted_sum(gamma, matrix):
    discount = 1.0
    vec = 0.0
    for v in matrix:
        vec += discount * v
        discount *= gamma
    return vec


def value_loss_fn(
    bc_policy,
    value,
    value_target,
    target_os,
    target_acts,
    target_next_os,
    target_done,
    target_cost,
    config,
):
    gamma = config["gamma"]

    horizon, batch_size, _ = target_os.shape

    # Horizon_Batch X obs/act_dim
    target_os = target_os.view(horizon * batch_size, -1)
    target_acts = target_acts.view(horizon * batch_size, -1)
    target_next_os = target_next_os.view(horizon * batch_size, -1)
    target_done = target_done.view(horizon * batch_size)
    target_cost = target_cost.view(horizon * batch_size)

    target_value = calculate_target_value(
        value=value_target,
        bc_policy=bc_policy,
        target_next_os=target_next_os,
        target_done=target_done,
        target_cost=target_cost,
        gamma=gamma,
    )

    value_loss, priority_loss = 0.0, 0.0
    o, a = target_os, target_acts
    pred_v1, pred_v2 = value.V(torch.cat([o, a], dim=1))
    value_loss += F.huber_loss(pred_v1, target_value, delta=2.0, reduction="none")
    value_loss += F.huber_loss(pred_v1, target_value, delta=2.0, reduction="none")
    value_loss = value_loss.view(horizon, batch_size)
    value_loss = discounted_sum(gamma, value_loss)

    priority_loss += F.l1_loss(pred_v1, target_value, reduction="none")
    priority_loss += F.l1_loss(pred_v2, target_value, reduction="none")
    priority_loss = priority_loss.view(horizon, batch_size)
    priorities = discounted_sum(gamma, priority_loss)

    return torch.mean(value_loss), priorities


def pretrain_cost_contrastive_model(cost_model, cost_optimizer, buffer, config):
    steps = 0
    while steps < config["total_iteration"]:
        for buffer_sample in buffer.sample():
            steps += 1

            (
                target_neg_os,
                target_neg_acts,
                target_union_os,
                target_union_acts,
                _,
                _,
                _,
            ) = buffer_sample

            cost_contrast_loss = cost_contrastive_loss_fn(
                cost_model=cost_model,
                target_neg_os=target_neg_os,
                target_neg_acts=target_neg_acts,
                target_union_os=target_union_os,
                target_union_acts=target_union_acts,
                config=config,
            )
            cost_optimizer.zero_grad()
            cost_contrast_loss.backward()
            clip_grad_norm_(cost_model.parameters(), config["max_grad_norm_cost"])
            cost_optimizer.step()

            if steps % config["log_freq"] == 0:
                print(f"Contrastive cost loss: {cost_contrast_loss:.3f}")

            if steps >= config["total_iteration"]:
                break

    return cost_model


def train_cost_model(cost_model, cost_optimizer, buffer_sample, config, steps):
    # train cost model
    (
        target_neg_os,
        target_neg_acts,
        target_union_os,
        target_union_acts,
        _,
        _,
        _,
    ) = buffer_sample

    cost_loss = cost_pref_loss = cost_contrast_loss = torch.tensor(0.0)

    if config["pretrain_cost_contrastive"]:
        return cost_loss, cost_pref_loss, cost_contrast_loss

    if (steps % config["update_cost_freq"]) == 0:
        max_bootstrap_lambda = config["bootstrap_lambda"]
        cost_optimizer.zero_grad()
        bootstrap_lambda = (
            3 * max_bootstrap_lambda / config["total_iteration"]
        ) * steps
        bootstrap_lambda = min(max_bootstrap_lambda, bootstrap_lambda)
        cost_pref_loss = cost_pref_loss_fn(
            cost_model=cost_model,
            target_neg_os=target_neg_os,
            target_neg_acts=target_neg_acts,
            target_union_os=target_union_os,
            target_union_acts=target_union_acts,
            config=config,
            bootstrap_lambda=bootstrap_lambda,
        )
        if config["use_cost_contrastive"] and config["use_contrastive_loss"]:
            cost_contrast_loss = cost_contrastive_loss_fn(
                cost_model=cost_model,
                target_neg_os=target_neg_os,
                target_neg_acts=target_neg_acts,
                target_union_os=target_union_os,
                target_union_acts=target_union_acts,
                config=config,
                bootstrap_lambda=bootstrap_lambda,
            )
        cost_loss = cost_pref_loss + cost_contrast_loss
        cost_loss.register_hook(lambda grad: grad * (1 / config["train_horizon"]))
        cost_loss.backward()
        clip_grad_norm_(cost_model.parameters(), config["max_grad_norm_cost"])
        cost_optimizer.step()

    return cost_loss, cost_pref_loss, cost_contrast_loss


def train_value_and_policy_model(
    cost_model,
    value_cost,
    value_cost_target,
    bc_policy,
    bc_policy_target,
    value_cost_optimizer,
    bc_policy_optimizer,
    bc_scheduler,
    buffer_sample,
    config,
    steps,
):
    (
        _,
        _,
        target_union_os,
        target_union_acts,
        target_union_next_os,
        target_union_done,
        _,
    ) = buffer_sample

    target_cost = get_cost(cost_model, target_union_os, target_union_acts, config)

    value_cost_optimizer.zero_grad()
    value_loss, priorities = value_loss_fn(
        bc_policy=bc_policy_target,
        value=value_cost,
        value_target=value_cost_target,
        target_os=target_union_os,
        target_acts=target_union_acts,
        target_next_os=target_union_next_os,
        target_done=target_union_done,
        target_cost=target_cost,
        config=config,
    )
    value_loss.register_hook(lambda grad: grad * (1 / config["train_horizon"]))
    value_loss.backward()
    clip_grad_norm_(value_cost.parameters(), config["max_grad_norm_critic"])
    value_cost_optimizer.step()

    bc_policy_loss = q_loss = v_loss = torch.tensor(0.0)
    if (steps % config["update_bc_freq"]) == 0:
        # to ensure that when the value fn is used in policy loss
        # then the value_cost parameters grad are zero initially.
        value_cost.zero_grad()
        bc_policy_optimizer.zero_grad()
        bc_policy_loss, q_loss, v_loss = bc_policy_loss_fn(
            bc_policy=bc_policy,
            value=value_cost,
            target_os=target_union_os,
            target_acts=target_union_acts,
            config=config,
        )
        bc_policy_loss.backward()
        clip_grad_norm_(bc_policy.parameters(), config["max_grad_norm_bc"])
        bc_policy_optimizer.step()
        # bc_scheduler.step()

        ema(value_cost, value_cost_target, config["update_tau"])
        ema(bc_policy, bc_policy_target, config["update_tau"])

    return value_loss, bc_policy_loss, q_loss, v_loss, priorities


def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device_name = "cpu" if args.device == "cpu" else f"{args.device}:{args.device_id}"
    device = torch.device(device_name)

    trajectory_cfg["num_negative_trajectories"] = args.num_non_preferred
    trajectory_cfg["num_union_trajectories"] = args.num_union
    trajectory_cfg["non_pref_noise"] = args.non_pref_noise

    config = {**default_cfg, **trajectory_cfg}
    config["train_horizon"] = args.train_horizon or config.get("train_horizon")
    config["bootstrap_lambda"] = args.bootstrap_lambda
    config["value_weight_temp"] = args.value_weight_temp or config["value_weight_temp"]
    config["bc_weight_temp"] = args.bc_weight_temp
    config["policy_type"] = args.policy_type
    config["update_priority_buffer"] = args.update_priority_buffer
    config["normalize_observation"] = args.normalize_observation
    config["cost_model_path"] = args.cost_model_path
    config["use_cost_attention"] = args.use_cost_attention
    config["use_cost_contrastive"] = args.use_cost_contrastive
    config["use_contrastive_loss"] = args.use_contrastive_loss
    config["pretrain_cost_contrastive"] = args.pretrain_cost_contrastive
    config["use_td3_style_bc"] = args.use_td3_style_bc
    config["num_neg_extra_traj"] = int(args.num_neg_extra_traj)
    config["use_expected_cost_pref"] = args.use_expected_cost_pref

    # evaluation environment
    eval_env = gym.make(args.task)
    if args.save_video:
        eval_env.render_parameters.mode = "rgb_array"
        eval_env.render_parameters.camera_name = "track"
    eval_env.set_target_cost(config["target_cost"])
    eval_env = ActionRepeater(eval_env, num_repeats=config["action_repeat"])
    eval_env.reset(seed=args.seed)

    # set training steps
    batch_size = config.get("batch_size", args.batch_size)

    # set model
    obs_space, act_space = eval_env.observation_space, eval_env.action_space
    config["bc_lr"] = args.lr
    if config["policy_type"] == "vae":
        # See BEAR implementation from @aviralkumar
        bc_latent_dim = config.get("latent_dim", act_space.shape[0] * 2)
        config["bc_latent_dim"] = bc_latent_dim
        bc_policy = BcqVAE(
            obs_dim=obs_space.shape[0],
            act_dim=act_space.shape[0],
            latent_dim=bc_latent_dim,
            device=device,
        ).to(device)
    else:
        bc_policy = SafeDiceTanhMixtureActor(
            obs_dim=obs_space.shape[0],
            act_dim=act_space.shape[0],
            hidden_size=config["hidden_sizes"][0],
        ).to(device)
    bc_policy_target = deepcopy(bc_policy)
    bc_policy_optimizer = torch.optim.AdamW(
        bc_policy.parameters(), lr=config["bc_lr"], weight_decay=config["weight_decay"]
    )
    bc_scheduler = LinearLR(
        bc_policy_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=config["total_iteration"] // config["update_bc_freq"],
    )

    if config["use_cost_attention"]:
        cost_model = SafeTransformerCritic(
            obs_dim=obs_space.shape[0],
            act_dim=act_space.shape[0],
            horizon=config["train_horizon"],
            latent_dim=config["hidden_sizes"][0],
            num_attentions=1,
            device=device,
        ).to(device)
    elif config["use_cost_contrastive"] or config["pretrain_cost_contrastive"]:
        cost_model = ContrastiveCostModel(
            obs_dim=obs_space.shape[0] + act_space.shape[0],
            hidden_sizes=config["hidden_sizes"],
        ).to(device)
    else:
        cost_model = ExpCostModel(
            # (s,a)
            obs_dim=obs_space.shape[0] + act_space.shape[0],
            hidden_sizes=config["hidden_sizes"],
        ).to(device)
    cost_optimizer = torch.optim.AdamW(
        cost_model.parameters(), lr=args.lr, weight_decay=config["weight_decay"]
    )

    value_cost = EnsembleValue(
        obs_dim=obs_space.shape[0] + act_space.shape[0],
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
    neg_data, union_data = get_neg_and_union_data_2(data, trajectory_cfg)
    mu_obs, std_obs = None, None
    if config["normalize_observation"]:
        neg_data, union_data, mu_obs, std_obs = get_normalized_data(
            neg_data, union_data
        )

    neg_observations = torch.as_tensor(
        neg_data["observations"], dtype=torch.float32, device=device
    )
    neg_actions = torch.as_tensor(
        neg_data["actions"], dtype=torch.float32, device=device
    )
    neg_dones = neg_data["timeouts"] | neg_data["terminals"]
    neg_dones = torch.as_tensor(neg_dones, dtype=torch.float32, device=device)

    union_observations = torch.as_tensor(
        union_data["observations"], dtype=torch.float32, device=device
    )
    union_actions = torch.as_tensor(
        union_data["actions"], dtype=torch.float32, device=device
    )
    union_dones = union_data["timeouts"] | union_data["terminals"]
    union_dones = torch.as_tensor(union_dones, dtype=torch.float32, device=device)

    ep_len = ep_len // config["action_repeat"] + (ep_len % config["action_repeat"] > 0)
    assert (
        neg_observations.shape[1] == ep_len
    ), f"{neg_observations.shape[1]} episode length is different from {ep_len}"

    buffer = SafeTD3Buffer(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        neg_data_size=np.prod(neg_observations.shape[:-1]),
        union_data_size=np.prod(union_observations.shape[:-1]),
        horizon=config["train_horizon"],
        batch_size=batch_size,
        device=device,
        ep_len=ep_len,
        priorities_alpha=0.6,
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

    # pretrain cost_contrastive
    if config["pretrain_cost_contrastive"]:
        cost_model_path = args.cost_model_path
        if (cost_model_path is not None) and (osp.exists(osp.abspath(cost_model_path))):
            logger.log("Load cost contrastive model.")
            cost_model_path = osp.abspath(cost_model_path)
            cost_model.load_state_dict(
                torch.load(cost_model_path, weights_only=True, map_location=device)
            )
        else:
            logger.log("Start cost contrastive training.")
            cost_model = pretrain_cost_contrastive_model(
                cost_model=cost_model,
                cost_optimizer=cost_optimizer,
                buffer=buffer,
                config=config,
            )

        cost_model.freeze_encoder()

    # train cost, value and policy model
    logger.log("Start cost, value and bc_policy training.")
    steps = 0
    while steps < config["total_iteration"]:
        # shape: Horizon X Batch X obs/act_dim
        for buffer_sample in buffer.sample():

            steps += 1

            cost_loss, cost_pref_loss, cost_contrast_loss = train_cost_model(
                cost_model=cost_model,
                cost_optimizer=cost_optimizer,
                buffer_sample=buffer_sample,
                config=config,
                steps=steps,
            )

            value_loss, bc_policy_loss, q_loss, v_loss, priorities = (
                train_value_and_policy_model(
                    cost_model=cost_model,
                    value_cost=value_cost,
                    value_cost_target=value_cost_target,
                    value_cost_optimizer=value_cost_optimizer,
                    bc_policy=bc_policy,
                    bc_policy_target=bc_policy_target,
                    bc_policy_optimizer=bc_policy_optimizer,
                    bc_scheduler=bc_scheduler,
                    buffer_sample=buffer_sample,
                    config=config,
                    steps=steps,
                )
            )

            if config["update_priority_buffer"]:
                buffer.update_priorities(
                    buffer_sample[-1], priorities.clamp(max=1e4).detach()
                )

            logger.logged = False

            if (steps % config["log_freq"] == 0) and (not logger.logged):
                eval_episodes = config["eval_episode_freq"]
                if args.use_eval:
                    eval_start_time = time.time()
                    eval_policy = (
                        bc_policy.decode_bc
                        if config["policy_type"] == "vae"
                        else bc_policy.action
                    )
                    eval_cost_fn = lambda *_: torch.tensor(0.0)
                    eval_cost_model = (
                        eval_cost_fn if config["use_cost_attention"] else cost_model
                    )
                    for id in range(eval_episodes):
                        (
                            eval_reward,
                            eval_cost,
                            eval_pred_cost,
                            eval_len,
                        ) = evaluate_bc_policy(
                            eval_env=eval_env,
                            policy=eval_policy,
                            cost_model=eval_cost_model,
                            device=device,
                            norm_fn=partial(normalize_observation, mu_obs, std_obs),
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
                logger.log_tabular("Loss/Loss_cost", cost_loss.item())
                logger.log_tabular("Loss/Loss_cost_pref", cost_pref_loss.item())
                logger.log_tabular("Loss/Loss_cost_contrast", cost_contrast_loss.item())
                logger.log_tabular("Loss/Loss_value_cost", value_loss.item())
                logger.log_tabular("Loss/Loss_bc_policy", bc_policy_loss.item())
                logger.log_tabular("Loss/bc_q_value", q_loss.item())
                logger.log_tabular("Loss/bc_v_value", v_loss.item())

                logger.log_tabular(
                    "Norm/cost",
                    get_params_norm(cost_model.parameters(), grads=False),
                )
                logger.log_tabular(
                    "Norm/bc_policy",
                    get_params_norm(bc_policy.parameters(), grads=False),
                )
                logger.log_tabular(
                    "Norm/value_cost",
                    get_params_norm(value_cost.parameters(), grads=False),
                )
                # logger.log_tabular(
                #     "Norm/grads/cost",
                #     get_params_norm(cost_model.parameters(), grads=True),
                # )
                # logger.log_tabular(
                #     "Norm/grads/bc_policy",
                #     get_params_norm(bc_policy.parameters(), grads=True),
                # )
                # logger.log_tabular(
                #     "Norm/grads/value_cost",
                #     get_params_norm(value_cost.parameters(), grads=True),
                # )
                if args.use_eval:
                    logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
                logger.dump_tabular()

            if steps % config["save_freq"] == 0:
                logger.torch_save(
                    itr=steps,
                    torch_saver_elements=bc_policy,
                    prefix="bc_policy",
                )
                logger.torch_save(
                    itr=steps,
                    torch_saver_elements=cost_model,
                    prefix="cost",
                )
                logger.torch_save(
                    itr=steps,
                    torch_saver_elements=value_cost,
                    prefix="value_cost",
                )

            if steps >= config["total_iteration"]:
                break

    logger.torch_save(itr=steps, torch_saver_elements=bc_policy, prefix="bc_policy")
    logger.torch_save(itr=steps, torch_saver_elements=cost_model, prefix="cost")
    logger.torch_save(itr=steps, torch_saver_elements=value_cost, prefix="value_cost")
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
