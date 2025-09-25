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
import numpy as np
import torch
import torch.distributions as td
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR

from dsrl_model.utils.bufffer import SafeCLBuffer
from dsrl_model.utils.dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_neg_and_union_data_2,
    get_normalized_data,
)
from dsrl_model.utils.logger import EpochLogger
from dsrl_model.utils.models import (
    BcqVAE,
    Encoder,
    ExpCostModel,
    SafeDiceTanhMixtureActor,
    SafeTransformerCritic,
    horizon_gradient_panelty,
)
from dsrl_model.utils.save_video_with_value import save_video
from dsrl_model.utils.utils import ActionRepeater, get_params_norm, single_agent_args

EP = 1e-6
EP2 = 1e-3

default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "eval_episode_freq": 1,  # use saved bc_policy to run evaluatation
    "hidden_sizes": [256, 256],
    "max_grad_norm": 1.0,
    "gamma": 0.99,
    "action_repeat": 1,  # set to 2, min value is 1
    "train_horizon": 5,  # 20
    "update_label_freq": 5,
    "update_cost_bc_freq": 10,
    "decay": 0.85,
    "max_label": 4.0,
    "warmup_steps": int(1e4),
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
    "percentage_validation_trajectories": 0.2,
}


@torch.no_grad
def evaluate_bc_policy(eval_env, bc_policy, device):
    eval_done = False
    eval_obs, _ = eval_env.reset()
    # eval_obs = (eval_obs - mu_obs) / (std_obs + EP)
    eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(
        0
    )
    eval_reward, eval_cost, eval_pred_reward, eval_len = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    while not eval_done:
        act = bc_policy(eval_obs)
        next_obs, reward, terminated, truncated, info = eval_env.step(
            act[0].detach().squeeze().cpu().numpy()
        )
        cost = info["cost"]
        # next_obs = (next_obs - mu_obs) / (std_obs + EP)
        next_obs = torch.as_tensor(
            next_obs, dtype=torch.float32, device=device
        ).unsqueeze(0)
        eval_obs = next_obs
        eval_reward += reward
        eval_cost += cost
        eval_len += 1
        eval_done = terminated or truncated
    return eval_reward, eval_cost, eval_pred_reward, eval_len


def ema(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    # implementation from td-mpc
    # target_params = (1-tau) * target_params + tau * params
    # tau is generally a small number.
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)


def discounted_sum(vector_x, gamma):
    horizon = vector_x.shape[0]
    cumsum = vector_x[-1]
    for t in reversed(range(horizon - 1)):
        cumsum = vector_x[t] + gamma * cumsum
    return cumsum


@torch.no_grad
def compute_bc_weight(cost_model, target_obs, target_act, margin, config):
    gamma, tau = config["gamma"], config["update_tau"]

    cost_weight = cost_model(
        torch.cat([target_obs, target_act], dim=-1), use_sigmoid=True
    )
    weight = discounted_sum(cost_weight, gamma)
    batch_margin = torch.mean(weight)
    margin = (1 - tau) * margin + tau * batch_margin
    weight = torch.exp((margin - weight) / config["cost_weight_temp"])
    final_weight = weight / (torch.mean(weight) + EP)
    return final_weight, margin


def bc_policy_loss_fn(bc_policy, target_obs, target_act, weight, config):
    gamma = config["gamma"]
    discount, loss = 1.0, 0.0
    # Horizon X Batch X obs/act_dim
    horizon, *_ = target_obs.shape

    for t in range(horizon):
        to, ta = target_obs[t], target_act[t]
        if config["policy_type"] == "vae":
            pred_act, bc_mean, bc_std = bc_policy(to, ta)
            recon_loss = F.mse_loss(pred_act, ta, reduction="none").sum(dim=1)
            kl_loss = -0.5 * (
                1 + torch.log(bc_std.pow(2)) - bc_mean.pow(2) - bc_std.pow(2)
            ).sum(dim=1)
            # 0.5 weight is from BCQ implementation See @aviralkumar implementation
            loss += discount * (recon_loss + 0.5 * kl_loss)
        else:
            pred_act, *_ = bc_policy(to)
            recon_loss = F.mse_loss(pred_act, ta, reduction="none").sum(dim=1)
            loss += discount * recon_loss
        discount *= gamma

    loss = weight * loss
    return torch.mean(loss)


def compute_contrastive_ce_loss(p, q, decay):
    q = F.log_softmax(q, dim=1)
    p = p / p.sum(dim=1, keepdim=True).clamp(min=1.0)
    loss = torch.sum(p * q, dim=1) * decay
    return -torch.mean(loss)


def train_embedding_model(
    embedding_model,
    embedding_optimizer,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    target_union_label,
    config,
):
    device = target_neg_obs.device
    _, batch_size, _ = target_neg_obs.shape

    # shape: Horizon X Batch X obs_act_dim
    target_neg = torch.concat([target_neg_obs, target_neg_act], dim=-1)
    target_union = torch.concat([target_union_obs, target_union_act], dim=-1)

    # Batch X embd_dim
    _, neg_z = embedding_model(target_neg, use_sigmoid=False)
    _, union_z = embedding_model(target_union, use_sigmoid=False)

    temperature = 0.1  # value from SupContrast
    combined_z = torch.concat([neg_z, union_z], dim=0)
    combined_logits = torch.matmul(combined_z, combined_z.T) / temperature
    # remove the self instance from the logits
    combined_logits.fill_diagonal_(-1e9)

    neg_mask = torch.ones((batch_size, batch_size), device=device, dtype=torch.float32)
    union_mask = (
        target_union_label.unsqueeze(0) == target_union_label.unsqueeze(1)
    ).float()
    valid_mask = (target_union_label >= 0).float()
    # remove -1 labels from all the union_mask
    union_mask = union_mask * valid_mask.unsqueeze(0) * valid_mask.unsqueeze(1)
    combined_mask = torch.block_diag(neg_mask, union_mask)
    # zero out diagoals
    combined_mask.fill_diagonal_(0.0)

    neg_decay_rate = torch.ones((batch_size,), dtype=torch.float32, device=device)
    union_decay_rate = config["decay"] ** (config["max_label"] - target_union_label)
    decay_rate = torch.concat([neg_decay_rate, union_decay_rate])

    loss = compute_contrastive_ce_loss(combined_mask, combined_logits, decay_rate)

    embedding_optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(embedding_model.parameters(), config["max_grad_norm"])
    embedding_optimizer.step()

    return loss


def index_fn(pred_vec, config):
    device = pred_vec.device
    new_label = config["max_label"] * torch.ones_like(
        pred_vec, dtype=torch.float32, device=device
    )
    new_label[pred_vec < 0.8] = 3.0
    new_label[pred_vec < 0.6] = 2.0
    new_label[pred_vec < 0.4] = 1.0
    new_label[pred_vec < 0.2] = 0.0
    return new_label


@torch.no_grad()
def get_new_union_labels(
    embedding_model,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    target_neg_z,
    config,
):
    # shape: Horizon X Batch X obs_act_dim
    target_neg = torch.concat([target_neg_obs, target_neg_act], dim=-1)
    target_union = torch.concat([target_union_obs, target_union_act], dim=-1)

    # Batch X embd_dim
    _, neg_z = embedding_model(target_neg, use_sigmoid=False)
    _, union_z = embedding_model(target_union, use_sigmoid=False)

    rep_neg_z = F.normalize(neg_z.sum(dim=0, keepdim=True), dim=-1, p=2.0)
    target_neg_z.lerp_(rep_neg_z, config["update_tau"])

    union_pred = torch.matmul(union_z, target_neg_z.T).squeeze()
    new_label = index_fn(union_pred, config)
    return target_neg_z, new_label


def cost_loss_fn(
    cost_model,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    target_union_label,
    config,
):
    decay, max_label = config["decay"], config["max_label"]
    gamma, discount = config["gamma"], 1.0
    total_neg_cost, total_union_cost = 0.0, 0.0
    device = target_neg_obs.device

    # Horizon X Batch X obs/act_dim
    horizon, batch_size, _ = target_neg_obs.shape

    target_neg = torch.cat([target_neg_obs, target_neg_act], dim=-1)
    target_union = torch.cat([target_union_obs, target_union_act], dim=-1)
    for t in range(horizon):
        tn, tu = target_neg[t], target_union[t]
        total_neg_cost += discount * cost_model(tn, use_sigmoid=True)
        total_union_cost += discount * cost_model(tu, use_sigmoid=True)
        discount *= gamma
    exp_neg, exp_union = torch.exp(total_neg_cost), torch.exp(total_union_cost)

    # neg and union loss
    p_neg = exp_neg / (exp_neg + exp_union)
    target_ones = torch.ones_like(p_neg, device=device)
    neg_union_loss = F.binary_cross_entropy(p_neg, target_ones)

    # union and union loss
    # ensure batch size is of 2s multiple
    uu_exp = exp_union.view((2, -1))
    uu_label = target_union_label.view((2, -1))

    p_uu = uu_exp[0] / uu_exp.sum(dim=0)
    target_uu = (uu_label[0] > uu_label[1]).float()
    target_uu[uu_label[0] == uu_label[1]] = 0.5

    valid_compare = torch.logical_and(uu_label[0] > 0, uu_label[1] > 0).float()
    decay_weight = decay ** (max_label - torch.abs(uu_label[0] - uu_label[1]))
    weight = decay_weight * valid_compare

    union_union_loss = F.binary_cross_entropy(p_uu, target_uu, weight=weight)

    # final loss
    loss = neg_union_loss + union_union_loss
    return torch.mean(loss)


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
    margin,
    config,
):
    cost_loss = cost_loss_fn(
        cost_model=cost_model,
        target_neg_obs=target_neg_obs,
        target_neg_act=target_neg_act,
        target_union_obs=target_union_obs,
        target_union_act=target_union_act,
        target_union_label=target_union_label,
        config=config,
    )
    cost_optimizer.zero_grad()
    cost_loss.register_hook(lambda grad: grad * (1 / config["train_horizon"]))
    cost_loss.backward()
    clip_grad_norm_(cost_model.parameters(), config["max_grad_norm"])
    cost_optimizer.step()

    bc_weight, margin = compute_bc_weight(
        cost_model=cost_model,
        target_obs=target_union_obs,
        target_act=target_union_act,
        margin=margin,
        config=config,
    )

    bc_loss = bc_policy_loss_fn(
        bc_policy=bc_policy,
        target_obs=target_union_obs,
        target_act=target_union_act,
        weight=bc_weight,
        config=config,
    )
    bc_optimizer.zero_grad()
    bc_loss.register_hook(lambda grad: grad * (1 / config["train_horizon"]))
    bc_loss.backward()
    clip_grad_norm_(bc_policy.parameters(), config["max_grad_norm"])
    bc_optimizer.step()

    return cost_loss, bc_loss, margin


def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cpu.deterministic = True
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(f"{args.device}:{args.device_id}")

    trajectory_cfg["num_negative_trajectories"] = args.num_non_preferred
    trajectory_cfg["num_union_trajectories"] = args.num_union
    trajectory_cfg["non_pref_noise"] = args.non_pref_noise

    config = {**default_cfg, **trajectory_cfg}
    config["train_horizon"] = args.train_horizon or config.get("train_horizon")
    config["policy_type"] = args.policy_type
    config["normalize_observation"] = args.normalize_observation
    config["cost_weight_temp"] = args.cost_weight_temp or config["cost_weight_temp"]

    # evaluation environment
    eval_env = gym.make(args.task)
    if args.save_video:
        eval_env.render_parameters.mode = "rgb_array"
        eval_env.render_parameters.camera_name = "track"
    eval_env.set_target_cost(config["target_cost"])
    eval_env = ActionRepeater(eval_env, num_repeats=config["action_repeat"])
    eval_env.reset(seed=args.seed)

    # set training steps
    batch_size = args.batch_size or config.get("batch_size")

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
    bc_optimizer = torch.optim.AdamW(
        bc_policy.parameters(), lr=config["bc_lr"], weight_decay=config["weight_decay"]
    )
    bc_scheduler = LinearLR(
        bc_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=config["total_iteration"],
    )

    embd_dim = config["hidden_sizes"][0]
    embedding_model = SafeTransformerCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        horizon=config["train_horizon"],
        latent_dim=embd_dim,
        num_attentions=2,
        device=device,
    ).to(device)
    embedding_optimizer = torch.optim.AdamW(
        embedding_model.parameters(), lr=args.lr, weight_decay=config["weight_decay"]
    )

    cost_model = ExpCostModel(
        obs_dim=obs_space.shape[0] + act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    cost_optimizer = torch.optim.AdamW(
        cost_model.parameters(), lr=args.lr, weight_decay=config["weight_decay"]
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

    union_observations = torch.as_tensor(
        union_data["observations"], dtype=torch.float32, device=device
    )
    union_actions = torch.as_tensor(
        union_data["actions"], dtype=torch.float32, device=device
    )
    union_dones = union_data["timeouts"] | union_data["terminals"]

    ep_len = ep_len // config["action_repeat"] + (ep_len % config["action_repeat"] > 0)
    assert (
        neg_observations.shape[1] == ep_len
    ), f"{neg_observations.shape[1]} episode length is different from {ep_len}"

    buffer = SafeCLBuffer(
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
    eval_pred_reward_deque = deque(maxlen=config["eval_episode_freq"])
    eval_len_deque = deque(maxlen=config["eval_episode_freq"])
    dict_args = config
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    logger.save_config(dict_args)
    logger.log("Start embedding, cost and bc_policy model training.")

    steps = 0
    target_neg_z = torch.zeros((1, embd_dim), dtype=torch.float32, device=device)
    margin = 0.0
    while steps < config["total_iteration"]:
        # shape: Horizon X Batch X obs/act_dim
        for (
            target_neg_obs,
            target_neg_act,
            target_union_obs,
            target_union_act,
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
                config=config,
            )

            if (steps > config["warmup_steps"]) and (
                steps % config["update_label_freq"] == 0
            ):
                target_neg_z, new_labels = get_new_union_labels(
                    embedding_model=embedding_model,
                    target_neg_obs=target_neg_obs,
                    target_neg_act=target_neg_act,
                    target_union_obs=target_union_obs,
                    target_union_act=target_union_act,
                    target_neg_z=target_neg_z,
                    config=config,
                )
                buffer.update_labels(target_union_idx, new_labels)

            cost_loss = bc_loss = torch.tensor(0.0)
            if (steps > config["warmup_steps"]) and (
                steps % config["update_cost_bc_freq"] == 0
            ):
                cost_loss, bc_loss, margin = train_cost_and_policy_model(
                    cost_model=cost_model,
                    bc_policy=bc_policy,
                    cost_optimizer=cost_optimizer,
                    bc_optimizer=bc_optimizer,
                    target_neg_obs=target_neg_obs,
                    target_neg_act=target_neg_act,
                    target_union_obs=target_union_obs,
                    target_union_act=target_union_act,
                    target_union_label=target_union_label,
                    margin=margin,
                    config=config,
                )

            logger.logged = False

            if (steps % config["log_freq"] == 0) and (not logger.logged):
                eval_episodes = config["eval_episode_freq"]
                if args.use_eval:
                    eval_start_time = time.time()
                    for id in range(eval_episodes):
                        (
                            eval_reward,
                            eval_cost,
                            eval_pred_reward,
                            eval_len,
                        ) = evaluate_bc_policy(
                            eval_env=eval_env,
                            bc_policy=(
                                bc_policy.decode_bc
                                if config["policy_type"] == "vae"
                                else bc_policy.action
                            ),
                            device=device,
                        )
                        norm_reward, norm_cost = eval_env.get_normalized_score(
                            eval_reward, eval_cost
                        )
                        eval_norm_rew_deque.append(norm_reward)
                        eval_norm_cost_deque.append(norm_cost)
                        eval_rew_deque.append(eval_reward)
                        eval_cost_deque.append(eval_cost)
                        eval_pred_reward_deque.append(eval_pred_reward)
                        eval_len_deque.append(eval_len)
                    logger.store(
                        **{
                            "Metrics/EvalEpRet": np.mean(eval_rew_deque),
                            "Metrics/EvalEpCost": np.mean(eval_cost_deque),
                            "Metrics/EvalEpPredReward": np.mean(eval_pred_reward_deque),
                            "Metrics/EvalEpNormRet": np.mean(eval_norm_rew_deque),
                            "Metrics/EvalEpNormCost": np.mean(eval_norm_cost_deque),
                            "Metrics/EvalEpLen": np.mean(eval_len_deque),
                        }
                    )
                    eval_end_time = time.time()

                    logger.log_tabular("Metrics/EvalEpRet")
                    logger.log_tabular("Metrics/EvalEpCost")
                    logger.log_tabular("Metrics/EvalEpPredReward")
                    logger.log_tabular("Metrics/EvalEpNormRet")
                    logger.log_tabular("Metrics/EvalEpNormCost")
                    logger.log_tabular("Metrics/EvalEpLen")

                logger.log_tabular("Train/Steps", steps)
                logger.log_tabular("Loss/Loss_embedding", embedding_loss.mean().item())
                logger.log_tabular("Loss/Loss_cost", cost_loss.mean().item())
                logger.log_tabular("Loss/Loss_bc_policy", bc_loss.mean().item())
                logger.log_tabular(
                    "Norm/embedding_model",
                    get_params_norm(embedding_model.parameters(), grads=False),
                )
                logger.log_tabular(
                    "Norm/cost_model",
                    get_params_norm(cost_model.parameters(), grads=False),
                )
                logger.log_tabular(
                    "Norm/bc_policy",
                    get_params_norm(bc_policy.parameters(), grads=False),
                )
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
                    torch_saver_elements=embedding_model,
                    prefix="embedding",
                )
                logger.torch_save(
                    itr=steps,
                    torch_saver_elements=cost_model,
                    prefix="cost",
                )

            if steps >= config["total_iteration"]:
                break

    logger.torch_save(itr=steps, torch_saver_elements=bc_policy, prefix="bc_policy")
    logger.torch_save(
        itr=steps, torch_saver_elements=embedding_model, prefix="embedding"
    )
    logger.torch_save(itr=steps, torch_saver_elements=cost_model, prefix="cost")
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
