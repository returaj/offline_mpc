import os
import os.path as osp
import sys
import time
import re
from copy import deepcopy
from collections import deque

import random
import numpy as np
import h5py

import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR

import gymnasium as gym
import dsrl.offline_safety_gymnasium

from dsrl_model.utils.models import (
    TdmpcDynamics,
    BcqVAE,
    TdmpcCostModel,
    EnsembleValue,
    Encoder,
)
from dsrl_model.utils.bufffer import PriorityBuffer
from dsrl_model.utils.logger import EpochLogger
from dsrl_model.utils.save_video_with_value import save_video
from dsrl_model.utils.dsrl_dataset import get_dataset_in_d4rl_format
from dsrl_model.model_based.utils import ActionRepeater
from dsrl_model.model_based.utils import single_agent_args


EP = 1e-6

default_cfg = {
    "hidden_sizes": [512, 512],
    "latent_obs_dim": 50,
    "max_grad_norm": 10.0,
    "gamma": 0.99,
    "lambda_c": 0.95,
    "action_repeat": 2,  # set to 2, min value is 1
    "explore_noise_std": 0.5,
    "update_freq": 1,
    "update_tau": 0.005,
    "dynamics_coef": 2.0,  # TDMPC update coef
    "bc_coef": 0.5,
    "cost_coef": 0.5,  # TDMPC update coef
    "value_coef": 0.1,  # TDMPC update coef
    "cost_weight_temp": 0.5,  # TDMPC temperature coef
    "elite_portion": 0.1,  # 0.1
    "num_samples": 40,  # 400
    "inference_horizon": 2,  # 5
    "train_horizon": 2,  # 5
}

trajectory_cfg = {
    "density": 0.5,
    # ((low_cost, low_reward), (high_cost, low_reward), (medium_cost, high_reward))
    "inpaint_ranges": (
        (0.0, 0.5, 0.0, 0.5),
        (0.5, 1.0, 0.0, 0.5),
        (0.25, 0.75, 0.0, 1.0),
    ),
    "target_cost": 25.0,
}


def ema(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    # implementation from td-mpc
    # target_params = (1-tau) * target_params + tau * params
    # tau is generally a small number.
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)


def bc_policy_loss_fn(bc_policy, dynamics, encoder, target_obs, target_act, config):
    gamma, horizon = config["gamma"], config["train_horizon"]
    pred_tz = encoder(target_obs)
    recon_loss, kl_loss = 0.0, 0.0
    for t in range(horizon):
        ta = target_act[t]
        pred_act, bc_mean, bc_std = bc_policy(pred_tz, ta)
        recon_loss += (gamma**t) * F.mse_loss(pred_act, ta, reduction="none")
        kl_loss += (gamma**t) * (
            -0.5 * (1 + torch.log(bc_std.pow(2)) - bc_mean.pow(2) - bc_std.pow(2))
        )
        pred_tz = dynamics(pred_tz, ta)
    # 0.5 weight is from BCQ implementation See @aviralkumar implementation
    return recon_loss.sum(-1) + 0.5 * kl_loss.sum(-1)


def dynamics_loss_fn(
    dynamics,
    encoder,
    encoder_target,
    target_obs,
    target_act,
    target_next_obs,
    config,
):
    gamma, horizon = config["gamma"], config["train_horizon"]
    dynamics_loss = 0.0
    pred_tz = encoder(target_obs)
    for t in range(horizon):
        ta, tno = target_act[t], target_next_obs[t]
        with torch.no_grad():
            tnz = encoder_target(tno)
        pred_nz = dynamics(pred_tz, ta)
        dynamics_loss += (gamma**t) * F.mse_loss(pred_nz, tnz, reduction="none")
        pred_tz = pred_nz
    return dynamics_loss.sum(-1)


def cost_loss_fn(
    critic,
    dynamics,
    encoder,
    target_obs,
    target_act,
    target_label,
    config,
):
    gamma, horizon = config["gamma"], config["train_horizon"]
    critic_loss = 0.0
    pred_tz = encoder(target_obs)
    for t in range(horizon):
        ta, tl = target_act[t], target_label[t]
        pred_c = critic(torch.cat([pred_tz, ta], dim=1))
        critic_loss += (gamma**t) * F.mse_loss(pred_c, tl, reduction="none")
        pred_tz = dynamics(pred_tz, ta)
    return critic_loss.sum(-1)


@torch.no_grad
def calculate_target_value(config, value, encoder, obs, next_obs, cost):
    gamma, lambda_c = config["gamma"], config["lambda_c"]
    horizon = next_obs.shape[0]
    advantage_c = torch.zeros_like(cost, dtype=torch.float32, device=cost.device)
    value_zs = torch.zeros_like(cost, dtype=torch.float32, device=cost.device)
    value_z = torch.min(*value.V(encoder(obs)))
    for idx in range(horizon):
        value_next_z = torch.min(*value.V(encoder(next_obs[idx])))
        advantage_c[idx] = cost[idx] + gamma * value_next_z - value_z
        value_zs[idx] = value_z
        value_z = value_next_z
    cumsum = advantage_c[-1]
    for idx in reversed(range(horizon - 1)):
        cumsum = advantage_c[idx] + gamma * lambda_c * cumsum
        advantage_c[idx] = cumsum
    return advantage_c + value_zs


def value_cost_loss_fn(
    value,
    value_target,
    dynamics,
    encoder,
    target_obs,
    target_act,
    target_next_obs,
    target_label,
    config,
):
    gamma, horizon = config["gamma"], config["train_horizon"]
    with torch.no_grad():
        target_value = calculate_target_value(
            config, value_target, encoder, target_obs, target_next_obs, target_label
        )
    pred_tz = encoder(target_obs)
    value_loss, priority_loss = 0.0, 0.0
    for t in range(horizon):
        pred_v1, pred_v2 = value.V(pred_tz)
        value_loss += (gamma**t) * (
            F.mse_loss(pred_v1, target_value[t], reduction="none")
            + F.mse_loss(pred_v2, target_value[t], reduction="none")
        )
        priority_loss += (gamma**t) * (
            F.l1_loss(pred_v1, target_value[t], reduction="none")
            + F.l1_loss(pred_v2, target_value[t], reduction="none")
        )
        pred_tz = dynamics(pred_tz, target_act[t])

    return value_loss, priority_loss


def mcem_policy(dynamics, critic, policy, value, encoder, obs, config, device):
    horizon = config["inference_horizon"]
    num_samples = config["num_samples"]
    num_elite = int(config["elite_portion"] * num_samples)
    gamma = config["gamma"]
    mask = torch.arange(num_samples) >= num_samples // 2
    noise_std = mask.to(device).unsqueeze(1) * config["explore_noise_std"]
    samples = []
    costs = torch.zeros(num_samples, dtype=torch.float32, device=device)
    all_z = encoder(obs.repeat(num_samples, 1))
    with torch.no_grad():
        for t in range(horizon):
            all_act = policy.decode_bc(all_z)
            noise_act = noise_std * torch.randn_like(
                all_act, device=device, dtype=torch.float32
            )
            all_act = all_act + noise_act
            costs += (gamma**t) * critic(torch.cat([all_z, all_act], dim=1))
            all_z = dynamics(all_z, all_act)
            samples.append(all_act.unsqueeze(1))
        # costs += (gamma**t) * torch.min(
        #     *value.V(torch.cat([all_z, policy.decode_bc(all_z)], dim=1))
        # )
        costs += (gamma**t) * torch.min(*value.V(all_z))
    samples = torch.cat(samples, dim=1)
    best_control_idx = torch.argsort(costs)[:num_elite]
    elite_controls = samples[best_control_idx]
    elite_costs = costs[best_control_idx]
    weights = torch.exp(-config["cost_weight_temp"] * elite_costs)
    weights /= weights.sum()
    weighted_cost = (elite_costs * weights).sum() / (weights.sum() + EP)
    weighted_controls = torch.sum(
        weights.view(-1, 1, 1).repeat(1, horizon, 1) * elite_controls, dim=0
    ) / (weights.sum() + EP)
    return (
        weighted_controls,
        weighted_cost,
        # elite_controls.mean(dim=0),
        elite_costs.mean(),
        elite_costs.min(),
        elite_costs.max(),
        costs.mean(),
        costs.max() - costs.min(),
    )


def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(f"{args.device}:{args.device_id}")
    config = {**default_cfg, **trajectory_cfg}

    # evaluation environment
    eval_env = gym.make(args.task, render_mode="rgb_array", camera_name="track")
    eval_env.set_target_cost(config["target_cost"])
    eval_env = ActionRepeater(eval_env, num_repeats=config["action_repeat"])
    eval_env.reset(seed=None)

    # set training steps
    num_epochs = config.get("num_epochs", args.num_epochs)
    batch_size = config.get("batch_size", args.batch_size)

    # set model
    obs_space, act_space = eval_env.observation_space, eval_env.action_space
    encoder = Encoder(
        obs_dim=obs_space.shape[0], latent_dim=config["latent_obs_dim"]
    ).to(device)
    encoder_target = deepcopy(encoder)
    # [TODO] linearly decreasing encoder learning rate is a possible exploration idea.
    # keeping all the other lr same.
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-5)
    # See BEAR implementation from @aviralkumar
    bc_latent_dim = config.get("latent_dim", act_space.shape[0] * 2)
    config["bc_latent_dim"] = bc_latent_dim
    bc_vae_policy = BcqVAE(
        obs_dim=config["latent_obs_dim"],
        act_dim=act_space.shape[0],
        latent_dim=bc_latent_dim,
        device=device,
    ).to(device)
    bc_vae_policy_optimizer = torch.optim.Adam(bc_vae_policy.parameters(), lr=3e-4)
    bc_vae_scheduler = LinearLR(
        bc_vae_policy_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=num_epochs,
    )
    critic = TdmpcCostModel(
        # (s,a)
        obs_dim=config["latent_obs_dim"] + act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    dynamics = TdmpcDynamics(
        obs_dim=config["latent_obs_dim"],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    dynamics_optimizer = torch.optim.Adam(dynamics.parameters(), lr=3e-4)
    value_cost = EnsembleValue(
        obs_dim=config["latent_obs_dim"],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    value_cost_target = deepcopy(value_cost)
    value_cost_optimizer = torch.optim.Adam(value_cost.parameters(), lr=3e-4)

    # data
    data = get_dataset_in_d4rl_format(
        eval_env, trajectory_cfg, args.task, config["action_repeat"]
    )

    observations = torch.as_tensor(
        data["observations"], dtype=torch.float32, device=device
    )
    actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=device)
    labels = torch.as_tensor(data["costs"], dtype=torch.float32, device=device)

    ep_len = 1000 // config["action_repeat"] + (1000 % config["action_repeat"] > 0)
    assert (
        observations.shape[1] == ep_len
    ), f"{observations.shape[1]} episode length is different from {ep_len}"
    buffer = PriorityBuffer(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        data_size=np.prod(labels.shape),
        horizon=config["train_horizon"],
        batch_size=batch_size,
        device=device,
        ep_len=ep_len,
    )
    for obs, act, label in zip(observations, actions, labels):
        buffer.add(obs, act, label)

    # set logger
    eval_rew_deque = deque(maxlen=5)
    eval_cost_deque = deque(maxlen=5)
    eval_critic_deque = deque(maxlen=5)
    eval_len_deque = deque(maxlen=5)
    dict_args = vars(args)
    dict_args.update(config)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    logger.save_config(dict_args)
    logger.log("Start with bc_policy, dynamics and critic training.")

    # train critic and dynamics model
    step = 0
    for epoch in range(num_epochs):
        training_start_time = time.time()
        idxs, priorities = [], []
        assert (
            len(bc_vae_scheduler.get_last_lr()) == 1
        ), f"multiple learning rates found {bc_vae_scheduler.get_last_lr()}"
        lr = bc_vae_scheduler.get_last_lr()[0]
        for _ in range(buffer.capacity // batch_size):
            (
                target_obs,
                target_next_obs,
                target_act,
                target_label,
                target_idxs,
                target_weights,
            ) = buffer.sample()

            step += 1

            recon_loss = dynamics_loss_fn(
                dynamics=dynamics,
                encoder=encoder,
                encoder_target=encoder_target,
                target_obs=target_obs,
                target_act=target_act,
                target_next_obs=target_next_obs,
                config=config,
            )

            bc_policy_loss = bc_policy_loss_fn(
                bc_policy=bc_vae_policy,
                dynamics=dynamics,
                encoder=encoder,
                target_obs=target_obs,
                target_act=target_act,
                config=config,
            )

            critic_loss = cost_loss_fn(
                critic=critic,
                dynamics=dynamics,
                encoder=encoder,
                target_obs=target_obs,
                target_act=target_act,
                target_label=target_label,
                config=config,
            )

            value_loss, priority_loss = value_cost_loss_fn(
                value=value_cost,
                value_target=value_cost_target,
                dynamics=dynamics,
                encoder=encoder,
                target_obs=target_obs,
                target_act=target_act,
                target_next_obs=target_next_obs,
                target_label=target_label,
                config=config,
            )

            total_loss = (
                config["dynamics_coef"] * recon_loss
                + config["bc_coef"] * bc_policy_loss
                + config["cost_coef"] * critic_loss
                + config["value_coef"] * value_loss
            )
            weighted_loss = (total_loss * target_weights).mean()
            weighted_loss.register_hook(
                lambda grad: grad * (1 / config["train_horizon"])
            )
            encoder_optimizer.zero_grad()
            dynamics_optimizer.zero_grad()
            bc_vae_policy_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            value_cost_optimizer.zero_grad()
            weighted_loss.backward()
            clip_grad_norm_(encoder.parameters(), config["max_grad_norm"])
            clip_grad_norm_(dynamics.parameters(), config["max_grad_norm"])
            clip_grad_norm_(bc_vae_policy.parameters(), config["max_grad_norm"])
            clip_grad_norm_(critic.parameters(), config["max_grad_norm"])
            clip_grad_norm_(value_cost.parameters(), config["max_grad_norm"])
            encoder_optimizer.step()
            dynamics_optimizer.step()
            bc_vae_policy_optimizer.step()
            critic_optimizer.step()
            value_cost_optimizer.step()

            buffer.update_priorities(
                target_idxs, torch.zeros_like(weighted_loss, dtype=torch.float32)
            )
            idxs.append(target_idxs)
            priorities.append(priority_loss.clamp(max=1e4).detach())

            if (step % config["update_freq"]) == 0:
                ema(encoder, encoder_target, config["update_tau"])
                ema(value_cost, value_cost_target, config["update_tau"])

            logger.store(
                **{
                    "Loss/Loss_bc_policy": bc_policy_loss.mean().item(),
                    "Loss/Loss_dynamics": recon_loss.mean().item(),
                    "Loss/Loss_critic": critic_loss.mean().item(),
                    "Loss/Loss_value_cost": value_loss.mean().item(),
                    "Loss/Loss_total": weighted_loss.mean().item(),
                }
            )
            logger.logged = False

        idxs, priorities = torch.cat(idxs), torch.cat(priorities)
        buffer.update_priorities(idxs, priorities)
        bc_vae_scheduler.step()
        training_end_time = time.time()

        eval_start_time = time.time()
        is_last_epoch = epoch >= num_epochs - 1
        eval_episodes = 5 if is_last_epoch else 1
        if args.use_eval:
            for id in range(eval_episodes):
                eval_done = False
                eval_obs, _ = eval_env.reset()
                # eval_obs = (eval_obs - mu_obs) / (std_obs + EP)
                eval_obs = torch.as_tensor(
                    eval_obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                eval_reward, eval_cost, eval_critic, eval_len = (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                ep_frames, ep_values = [], []
                while not eval_done:
                    (
                        act,
                        hcost_weight,
                        hcost_mu,
                        hcost_min,
                        hcost_max,
                        samples_mu,
                        samples_gap,
                    ) = mcem_policy(
                        dynamics=dynamics,
                        critic=critic,
                        policy=bc_vae_policy,
                        value=value_cost,
                        encoder=encoder,
                        obs=eval_obs,
                        config=config,
                        device=device,
                    )
                    next_obs, reward, terminated, truncated, info = eval_env.step(
                        act[0].detach().squeeze().cpu().numpy()
                    )
                    cost = info["cost"]
                    # next_obs = (next_obs - mu_obs) / (std_obs + EP)
                    next_obs = torch.as_tensor(
                        next_obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        tmp_za = torch.cat(
                            [encoder(eval_obs), act[0].unsqueeze(0)], dim=1
                        )
                        critic_cost = critic(tmp_za)
                        value = torch.min(*value_cost.V(encoder(eval_obs))).item()
                    eval_obs = next_obs
                    eval_reward += reward
                    eval_cost += cost
                    eval_critic += critic_cost.item()
                    eval_len += 1
                    eval_done = terminated or truncated
                    logger.store(
                        **{
                            "Metrics/EvalHorizonCostWeight": hcost_weight.item(),
                            "Metrics/EvalHorizonCostMean": hcost_mu.item(),
                            "Metrics/EvalHorizonCostMin": hcost_min.item(),
                            "Metrics/EvalHorizonCostMax": hcost_max.item(),
                            "Metrics/EvalHorizonCostGap": (
                                hcost_max - hcost_min
                            ).item(),
                            "Metrics/EvalSamplesCostMean": samples_mu.item(),
                            "Metrics/EvalSamplesCostGap": samples_gap.item(),
                        }
                    )
                    if is_last_epoch:
                        ep_frames.append(eval_env.render())
                        ep_values.append(value)
                if is_last_epoch:
                    save_video(
                        ep_frames,
                        ep_values,
                        prefix_name=f"video_{id}",
                        video_dir=osp.join(args.log_dir, "video"),
                    )
                eval_rew_deque.append(eval_reward)
                eval_cost_deque.append(eval_cost)
                eval_critic_deque.append(eval_critic)
                eval_len_deque.append(eval_len)
            logger.store(
                **{
                    "Metrics/EvalEpRet": np.mean(eval_rew_deque),
                    "Metrics/EvalEpCost": np.mean(eval_cost_deque),
                    "Metrics/EvalCriticCost": np.mean(eval_critic_deque),
                    "Metrics/EvalEpLen": np.mean(eval_len_deque),
                }
            )
        eval_end_time = time.time()

        if not logger.logged:
            if args.use_eval:
                logger.log_tabular("Metrics/EvalEpRet")
                logger.log_tabular("Metrics/EvalEpCost")
                logger.log_tabular("Metrics/EvalCriticCost")
                logger.log_tabular("Metrics/EvalEpLen")
                logger.log_tabular(
                    "Metrics/EvalHorizonCostWeight", min_and_max=True, std=True
                )
                logger.log_tabular(
                    "Metrics/EvalHorizonCostMean", min_and_max=True, std=True
                )
                logger.log_tabular(
                    "Metrics/EvalHorizonCostMin", min_and_max=True, std=True
                )
                logger.log_tabular(
                    "Metrics/EvalHorizonCostMax", min_and_max=True, std=True
                )
                logger.log_tabular(
                    "Metrics/EvalHorizonCostGap", min_and_max=True, std=True
                )
                logger.log_tabular(
                    "Metrics/EvalSamplesCostMean", min_and_max=True, std=True
                )
                logger.log_tabular(
                    "Metrics/EvalSamplesCostGap", min_and_max=True, std=True
                )
            logger.log_tabular("Train/Epoch", epoch + 1)
            logger.log_tabular("Train/learning_rate", lr)
            logger.log_tabular("Loss/Loss_bc_policy")
            logger.log_tabular("Loss/Loss_dynamics")
            logger.log_tabular("Loss/Loss_critic")
            logger.log_tabular("Loss/Loss_value_cost")
            logger.log_tabular("Loss/Loss_total")
            if args.use_eval:
                logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
            logger.log_tabular(
                "Time/TrainingUpdate", training_end_time - training_start_time
            )
            logger.log_tabular("Time/Total", eval_end_time - training_start_time)
            logger.dump_tabular()
            if (epoch + 1) % 20 == 0 or epoch == 0:
                logger.torch_save(
                    itr=epoch,
                    torch_saver_elements=bc_vae_policy,
                    prefix="bc_vae_policy",
                )
                logger.torch_save(
                    itr=epoch, torch_saver_elements=dynamics, prefix="morel_dynamics"
                )
                logger.torch_save(
                    itr=epoch, torch_saver_elements=critic, prefix="critic"
                )
                logger.torch_save(
                    itr=epoch,
                    torch_saver_elements=value_cost,
                    prefix="action_value_cost",
                )
    logger.torch_save(
        itr=epoch, torch_saver_elements=bc_vae_policy, prefix="bc_vae_policy"
    )
    logger.torch_save(itr=epoch, torch_saver_elements=dynamics, prefix="morel_dynamics")
    logger.torch_save(itr=epoch, torch_saver_elements=critic, prefix="critic")
    logger.torch_save(
        itr=epoch, torch_saver_elements=value_cost, prefix="action_value_cost"
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
