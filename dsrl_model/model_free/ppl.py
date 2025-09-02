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

from dsrl_model.utils.bufffer import OnPolicyBuffer
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
    "max_grad_norm": 10.0,
    "bag_size": 1,
    "gamma": 0.99,
    "action_repeat": 1,  # set to 2, min value is 1
    "train_horizon": 5,  # 20
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
def evaluate_bc_policy(eval_env, bc_policy, reward_model, device):
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
        with torch.no_grad():
            pred_reward = reward_model(
                torch.cat([eval_obs, act], dim=1), use_sigmoid=True
            ).item()
        eval_obs = next_obs
        eval_reward += reward
        eval_cost += cost
        eval_pred_reward += pred_reward
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


def bc_policy_loss_fn(
    bc_policy,
    reward_model,
    target_obs,
    target_act,
    config,
):
    loss = 0.0
    # Horizon X Batch X obs/act_dim
    horizon, batch_size, _ = target_obs.shape

    target_obs = target_obs.view(horizon * batch_size, -1)
    target_act = target_act.view(horizon * batch_size, -1)

    with torch.no_grad():
        weight = 1 - reward_model(
            torch.cat([target_obs, target_act], dim=1), use_sigmoid=True
        )

    if config["policy_type"] == "vae":
        pred_act, bc_mean, bc_std = bc_policy(target_obs, target_act)
        recon_loss = F.mse_loss(pred_act, target_act, reduction="none").sum(dim=1)
        kl_loss = -0.5 * (
            1 + torch.log(bc_std.pow(2)) - bc_mean.pow(2) - bc_std.pow(2)
        ).sum(dim=1)
        loss = recon_loss + 0.5 * kl_loss
    else:
        pred_act, *_ = bc_policy(target_obs)
        recon_loss = F.mse_loss(pred_act, target_act, reduction="none").sum(dim=1)
        loss = recon_loss

    loss = weight * loss
    return torch.mean(loss)


def reward_loss_fn(
    reward_model,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    config,
):
    gamma = config["gamma"]
    discount, total_neg_cost, total_union_cost, total_mix_cost = 1.0, 0.0, 0.0, 0.0
    device = target_neg_obs.device

    # Horizon X Batch X obs/act_dim
    horizon, batch_size, _ = target_neg_obs.shape

    target_neg = torch.cat([target_neg_obs, target_neg_act], dim=-1)
    target_union = torch.cat([target_union_obs, target_union_act], dim=-1)

    unif_rand = torch.rand(size=(horizon, batch_size, 1)).to(device)
    target_mixed = unif_rand * target_neg + (1 - unif_rand) * target_union
    target_mixed_input = Variable(target_mixed, requires_grad=True).to(device=device)

    for t in range(horizon):
        tn, tu, tm = target_neg[t], target_union[t], target_mixed_input[t]
        total_neg_cost += discount * reward_model(tn, use_sigmoid=True)
        total_union_cost += discount * reward_model(tu, use_sigmoid=True)
        total_mix_cost += discount * reward_model(tm, use_sigmoid=True)
        discount *= gamma

    exp_neg, exp_union = torch.exp(total_neg_cost), torch.exp(total_union_cost)
    sum_exp = exp_neg + exp_union
    p_neg, p_union = exp_neg / sum_exp, exp_union / sum_exp
    target_ones = torch.ones_like(p_neg, device=device)
    target_zeros = torch.zeros_like(p_union, device=device)
    loss = F.binary_cross_entropy(p_union, target_zeros)
    loss += F.binary_cross_entropy(p_neg, target_ones)

    grad_loss = 0.0  # horizon_gradient_panelty(target_mixed_input, total_mix_cost)
    return torch.mean(loss) + config["grad_reg_coeffs"] * grad_loss


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
    config["bag_size"] = 1
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
    bc_policy_optimizer = torch.optim.AdamW(
        bc_policy.parameters(),
        lr=config["bc_lr"],
        weight_decay=config["weight_decay"],
    )
    bc_scheduler = LinearLR(
        bc_policy_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=config["total_iteration"],
    )
    # bc_policy_optimizer = AdamWScheduleFree(
    #     bc_policy.parameters(),
    #     lr=config["bc_lr"],
    #     weight_decay=config["weight_decay"],
    #     warmup_steps=1000,
    # )
    reward_model = ExpCostModel(
        obs_dim=obs_space.shape[0] + act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    reward_model_optimizer = torch.optim.AdamW(
        reward_model.parameters(),
        lr=args.lr,
        weight_decay=config["weight_decay"],
    )

    # data
    agent_task = re.search(r"Offline(.*?)Gymnasium-v[0-9]", args.task).group(1)
    ep_len = dsrl_infos.DEFAULT_MAX_EPISODE_STEPS[agent_task]
    data = get_dataset_in_d4rl_format(
        eval_env, trajectory_cfg, args.task, ep_len, config["action_repeat"]
    )
    neg_data, union_data = get_neg_and_union_data_2(data, trajectory_cfg)
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
        batch_size=batch_size * config["bag_size"],
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
    logger.log("Start with bc_policy, cost model training.")

    steps = 0
    while steps < config["total_iteration"]:
        # shape: Horizon X Batch X obs/act_dim
        for (
            target_neg_obs,
            target_neg_act,
            target_union_obs,
            target_union_act,
            _,
        ) in buffer.sample():

            reward_loss = reward_loss_fn(
                reward_model=reward_model,
                target_neg_obs=target_neg_obs,
                target_neg_act=target_neg_act,
                target_union_obs=target_union_obs,
                target_union_act=target_union_act,
                config=config,
            )
            reward_loss.register_hook(lambda grad: grad * (1 / config["train_horizon"]))
            reward_model_optimizer.zero_grad()
            reward_loss.backward()
            clip_grad_norm_(reward_model.parameters(), config["max_grad_norm"])
            reward_model_optimizer.step()

            # bc_policy.train()
            # bc_policy_optimizer.train()
            bc_policy_loss = bc_policy_loss_fn(
                bc_policy=bc_policy,
                reward_model=reward_model,
                target_obs=target_union_obs,
                target_act=target_union_act,
                config=config,
            )
            bc_policy_optimizer.zero_grad()
            bc_policy_loss.backward()
            clip_grad_norm_(bc_policy.parameters(), config["max_grad_norm"])
            bc_policy_optimizer.step()
            bc_scheduler.step()

            logger.logged = False

            steps += 1

            if (steps % config["log_freq"] == 0) and (not logger.logged):
                eval_episodes = config["eval_episode_freq"]
                if args.use_eval:
                    eval_start_time = time.time()
                    # bc_policy.eval()
                    # bc_policy_optimizer.eval()
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
                            reward_model=reward_model,
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
                logger.log_tabular("Loss/Loss_bc_policy", bc_policy_loss.mean().item())
                logger.log_tabular("Loss/Loss_reward", reward_loss.mean().item())
                logger.log_tabular(
                    "Norm/bc_policy",
                    get_params_norm(bc_policy.parameters(), grads=False),
                )
                logger.log_tabular(
                    "Norm/reward_model",
                    get_params_norm(reward_model.parameters(), grads=False),
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
                    torch_saver_elements=reward_model,
                    prefix="reward",
                )

            if steps >= config["total_iteration"]:
                break

    logger.torch_save(itr=steps, torch_saver_elements=bc_policy, prefix="bc_policy")
    logger.torch_save(itr=steps, torch_saver_elements=reward_model, prefix="reward")
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
