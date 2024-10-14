import os
import os.path as osp
import sys
import time
from copy import deepcopy
from collections import deque

import re
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR

import gymnasium as gym
import dsrl.offline_safety_gymnasium  # type: ignore
import dsrl.infos as dsrl_infos

from dsrl_model.utils.models import (
    BcqVAE,
    Encoder,
    ExpCostModel,
)
from dsrl_model.utils.bufffer import OnPolicyBuffer
from dsrl_model.utils.logger import EpochLogger
from dsrl_model.utils.save_video_with_value import save_video
from dsrl_model.utils.dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_neg_and_union_data,
    get_normalized_data,
)
from dsrl_model.utils.utils import ActionRepeater
from dsrl_model.utils.utils import single_agent_args, get_params_norm


EP = 1e-6
EP2 = 1e-3

default_cfg = {
    "save_freq": 20,
    "warmup_bc": -1,  # set -1 if no warmup is needed in bc learning
    "hidden_sizes": [512, 512],
    "latent_obs_dim": 50,
    "max_grad_norm": 10.0,
    "bag_size": 1,
    "gamma": 0.99,
    "cost_lambda": 0.0,
    "action_repeat": 2,  # set to 2, min value is 1
    "update_freq": 1,
    "update_tau": 0.005,
    "bc_coef": 0.1,
    "cost_coef": 0.9,
    "cost_weight_temp": 0.6,
    "train_horizon": 20,  # 20
    "weight_decay_bc": 0.001,
    "weight_decay_cost": 0.001,
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
    "alpha": 0.0,  # dU = alpha * dN + (1-alpha) * dP
    "num_negative_trajectories": 50,
    "num_union_negative_trajectories": 0,
    "num_union_positive_trajectories": 50,
}


def ema(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    # implementation from td-mpc
    # target_params = (1-tau) * target_params + tau * params
    # tau is generally a small number.
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)


def bc_policy_loss_fn(bc_policy, encoder, cost_model, target_obs, target_act, config):
    gamma = config["gamma"]
    discount, loss = 1.0, 0.0
    # Horizon X Batch_Bag X obs/act_dim
    horizon, batch_bag_size, _ = target_obs.shape
    with torch.no_grad():
        target = encoder(torch.cat([target_obs, target_act], dim=-1))
        weight = cost_model(target, use_sigmoid=True).sum(dim=0)
        inv_weight = (1 / (weight + EP2)) ** config["cost_weight_temp"]
        inv_weight = torch.clip(inv_weight, max=10.0)
    for t in range(horizon):
        to, ta = target_obs[t], target_act[t]
        pred_act, bc_mean, bc_std = bc_policy(to, ta)
        recon_loss = F.mse_loss(pred_act, ta, reduction="none").sum(dim=1)
        kl_loss = -0.5 * (
            1 + torch.log(bc_std.pow(2)) - bc_mean.pow(2) - bc_std.pow(2)
        ).sum(dim=1)
        # 0.5 weight is from BCQ implementation See @aviralkumar implementation
        loss += discount * inv_weight * (recon_loss + 0.5 * kl_loss)
        discount *= gamma
    policy_loss = 0.0
    if config.get("use_policy_norm", False):
        for params in bc_policy.parameters():
            policy_loss += params.pow(2).sum() * 0.001
    return torch.mean(loss) + policy_loss


def cost_loss_fn(
    encoder,
    cost_model,
    target_neg_obs,
    target_neg_act,
    target_union_obs,
    target_union_act,
    config,
):
    gamma, bag_size = config["gamma"], config["bag_size"]
    alpha, cost_lambda = config["alpha"], config["cost_lambda"]
    discount, total_neg_cost, total_union_cost = 1.0, 0.0, 0.0

    # Horizon X Batch_Bag X obs/act_dim
    horizon, batch_bag_size, _ = target_neg_obs.shape
    batch_size = batch_bag_size // bag_size

    # encoded s,a pair
    target_neg = encoder(torch.cat([target_neg_obs, target_neg_act], dim=-1))
    target_union = encoder(torch.cat([target_union_obs, target_union_act], dim=-1))

    for t in range(horizon):
        tn, tu = target_neg[t], target_union[t]
        total_neg_cost += discount * cost_model(tn, use_sigmoid=True)
        total_union_cost += discount * cost_model(tu, use_sigmoid=True)
        discount *= gamma

    total_neg_cost = total_neg_cost.view(batch_size, bag_size)
    total_union_cost = total_union_cost.view(batch_size, bag_size)
    expected_neg_cost = torch.mean(total_neg_cost, dim=1)
    expected_union_cost = torch.mean(total_union_cost, dim=1)
    # expected_pos_cost = (1 / (1 - alpha)) * (
    #     expected_union_cost - alpha * expected_neg_cost
    # ).clamp(min=EP)
    # z = torch.log(expected_neg_cost + expected_pos_cost + EP)
    # # print(
    # #     expected_neg_cost.item(), expected_union_cost.item(), expected_pos_cost.item()
    # # )
    # loss = -torch.log(expected_neg_cost + EP) + torch.log(expected_union_cost + EP) + z
    loss = -torch.log(expected_neg_cost + EP) + torch.log(expected_union_cost + EP)
    cost_loss = 0.0
    if config.get("use_cost_norm", False):
        for params in cost_model.parameters():
            cost_loss += params.pow(2).sum() * 0.001
        for params in encoder.parameters():
            cost_loss += params.pow(2).sum() * 0.001
    return torch.mean(loss) + cost_loss


def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cpu.deterministic = True
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(f"{args.device}:{args.device_id}")
    config = {**default_cfg, **trajectory_cfg}
    config["train_horizon"] = args.train_horizon or config.get("train_horizon")
    config["warmup_bc"] = args.warmup_bc or config["warmup_bc"]
    config["bag_size"] = args.bag_size or config["bag_size"]
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
    num_epochs = args.num_epochs or config.get("num_epochs")
    batch_size = args.batch_size or config.get("batch_size")

    # set model
    obs_space, act_space = eval_env.observation_space, eval_env.action_space
    # See BEAR implementation from @aviralkumar
    bc_latent_dim = config.get("latent_dim", act_space.shape[0] * 2)
    config["bc_latent_dim"] = bc_latent_dim
    config["bc_lr"] = 1e-5
    bc_policy = BcqVAE(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        latent_dim=bc_latent_dim,
        device=device,
    ).to(device)
    bc_policy_optimizer = torch.optim.Adam(
        bc_policy.parameters(),
        lr=config["bc_lr"],
        weight_decay=config["weight_decay_bc"],
    )
    encoder = Encoder(
        # (s,a)
        obs_dim=obs_space.shape[0] + act_space.shape[0],
        latent_dim=config["latent_obs_dim"],
    ).to(device)
    encoder_optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=args.lr,
        weight_decay=config["weight_decay_cost"],
    )
    cost_model = ExpCostModel(
        obs_dim=config["latent_obs_dim"],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    cost_model_optimizer = torch.optim.Adam(
        cost_model.parameters(),
        lr=args.lr,
        weight_decay=config["weight_decay_cost"],
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
    eval_rew_deque = deque(maxlen=5)
    eval_cost_deque = deque(maxlen=5)
    eval_norm_rew_deque = deque(maxlen=5)
    eval_norm_cost_deque = deque(maxlen=5)
    eval_pred_cost_deque = deque(maxlen=5)
    eval_len_deque = deque(maxlen=5)
    dict_args = config
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    logger.save_config(dict_args)
    logger.log("Start with bc_policy, cost model training.")

    # train model
    step = 0
    for epoch in range(num_epochs):
        training_start_time = time.time()
        # assert (
        #     len(bc_scheduler.get_last_lr()) == 1
        # ), f"multiple learning rates found {bc_scheduler.get_last_lr()}"
        # lr = bc_scheduler.get_last_lr()[0]

        # shape: Horizon X Batch_Bag X obs/act_dim
        for (
            target_neg_obs,
            target_neg_act,
            target_union_obs,
            target_union_act,
        ) in buffer.sample():

            step += 1

            cost_loss = cost_loss_fn(
                encoder=encoder,
                cost_model=cost_model,
                target_neg_obs=target_neg_obs,
                target_neg_act=target_neg_act,
                target_union_obs=target_union_obs,
                target_union_act=target_union_act,
                config=config,
            )

            bc_policy_loss = torch.as_tensor(0.0)
            if (epoch + 1) > config["warmup_bc"]:
                bc_policy_loss = bc_policy_loss_fn(
                    bc_policy=bc_policy,
                    encoder=encoder,
                    cost_model=cost_model,
                    target_obs=target_union_obs,
                    target_act=target_union_act,
                    config=config,
                )

            total_loss = (
                config["bc_coef"] * bc_policy_loss + config["cost_coef"] * cost_loss
            )
            total_loss.register_hook(lambda grad: grad * (1 / config["train_horizon"]))
            bc_policy_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            cost_model_optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(bc_policy.parameters(), config["max_grad_norm"])
            clip_grad_norm_(encoder.parameters(), config["max_grad_norm"])
            clip_grad_norm_(cost_model.parameters(), config["max_grad_norm"])
            bc_policy_optimizer.step()
            encoder_optimizer.step()
            cost_model_optimizer.step()

            # if (step % config["update_freq"]) == 0:
            #     ema(encoder, encoder_target, config["update_tau"])
            #     ema(value_cost, value_cost_target, config["update_tau"])

            logger.store(
                **{
                    "Loss/Loss_bc_policy": bc_policy_loss.mean().item(),
                    "Loss/Loss_cost": cost_loss.mean().item(),
                    "Loss/Loss_total": total_loss.mean().item(),
                }
            )
            logger.logged = False

        # bc_scheduler.step()
        training_end_time = time.time()

        eval_start_time = time.time()
        is_save = (epoch + 1) % config["save_freq"] == 0 or epoch == 0
        is_last_epoch = epoch >= num_epochs - 1
        eval_episodes = 1 if is_last_epoch else 1
        if args.use_eval:
            for id in range(eval_episodes):
                eval_done = False
                eval_obs, _ = eval_env.reset()
                # eval_obs = (eval_obs - mu_obs) / (std_obs + EP)
                eval_obs = torch.as_tensor(
                    eval_obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                eval_reward, eval_cost, eval_pred_cost, eval_len = (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                ep_frames, ep_pred_cost = [], []
                while not eval_done:
                    act = bc_policy.decode_bc(eval_obs)
                    next_obs, reward, terminated, truncated, info = eval_env.step(
                        act[0].detach().squeeze().cpu().numpy()
                    )
                    cost = info["cost"]
                    # next_obs = (next_obs - mu_obs) / (std_obs + EP)
                    next_obs = torch.as_tensor(
                        next_obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        pred_cost = cost_model(
                            encoder(torch.cat([eval_obs, act], dim=1)), use_sigmoid=True
                        ).item()
                    eval_obs = next_obs
                    eval_reward += reward
                    eval_cost += cost
                    eval_pred_cost += pred_cost
                    eval_len += 1
                    eval_done = terminated or truncated
                    if args.save_video and (is_save or is_last_epoch):
                        ep_frames.append(eval_env.render())
                        ep_pred_cost.append(
                            {
                                "C(s,a)": cost,
                                "pC(s,a)": pred_cost,
                                "tC(s,a)": eval_cost,
                            }
                        )
                if args.save_video and (is_save or is_last_epoch):
                    save_video(
                        ep_frames,
                        ep_pred_cost,
                        prefix_name=f"video_{epoch}_{id}",
                        video_dir=osp.join(args.log_dir, "video"),
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
            logger.store(
                **{
                    "Metrics/EvalEpRet": np.mean(eval_rew_deque),
                    "Metrics/EvalEpCost": np.mean(eval_cost_deque),
                    "Metrics/EvalEpPredCost": np.mean(eval_pred_cost_deque),
                    "Metrics/EvalEpNormRet": np.mean(eval_norm_rew_deque),
                    "Metrics/EvalEpNormCost": np.mean(eval_norm_cost_deque),
                    "Metrics/EvalEpLen": np.mean(eval_len_deque),
                }
            )
        eval_end_time = time.time()

        if not logger.logged:
            if args.use_eval:
                logger.log_tabular("Metrics/EvalEpRet")
                logger.log_tabular("Metrics/EvalEpCost")
                logger.log_tabular("Metrics/EvalEpPredCost")
                logger.log_tabular("Metrics/EvalEpNormRet")
                logger.log_tabular("Metrics/EvalEpNormCost")
                logger.log_tabular("Metrics/EvalEpLen")
            logger.log_tabular("Train/Epoch", epoch + 1)
            # logger.log_tabular("Train/learning_rate", lr)
            logger.log_tabular("Loss/Loss_bc_policy")
            logger.log_tabular("Loss/Loss_cost")
            logger.log_tabular("Loss/Loss_total")
            logger.log_tabular(
                "Norm/bc_policy", get_params_norm(bc_policy.parameters(), grads=False)
            )
            logger.log_tabular(
                "Norm/cost_model", get_params_norm(cost_model.parameters(), grads=False)
            )
            if args.use_eval:
                logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
            logger.log_tabular(
                "Time/TrainingUpdate", training_end_time - training_start_time
            )
            logger.log_tabular("Time/Total", eval_end_time - training_start_time)
            logger.dump_tabular()
            if is_save:
                logger.torch_save(
                    itr=epoch, torch_saver_elements=bc_policy, prefix="bc_vae_policy"
                )
                logger.torch_save(
                    itr=epoch, torch_saver_elements=encoder, prefix="encoder"
                )
                logger.torch_save(
                    itr=epoch, torch_saver_elements=cost_model, prefix="cost_model"
                )
    logger.torch_save(itr=epoch, torch_saver_elements=bc_policy, prefix="bc_vae_policy")
    logger.torch_save(itr=epoch, torch_saver_elements=encoder, prefix="encoder")
    logger.torch_save(itr=epoch, torch_saver_elements=cost_model, prefix="cost_model")
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
