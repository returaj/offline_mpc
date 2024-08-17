import os
import os.path as osp
import sys
import time
import re
from copy import deepcopy
from collections import deque

from functools import partial
import random
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import safety_gymnasium
from baselines.utils.models import (
    TdmpcDynamics,
    BcqVAE,
    VCritic,
    EnsembleValue,
    Encoder,
)
from baselines.utils.logger import EpochLogger
from baselines.utils.save_video_with_value import save_video
from baselines.model_based.utils import single_agent_args


EP = 1e-6

default_cfg = {
    "hidden_sizes": [512, 512],
    "latent_obs_dim": 50,
    "max_grad_norm": 10.0,
    "gamma": 0.99,
    "lambda_c": 0.95,
    "update_freq": 1,
    "update_tau": 0.005,
    "dynamics_coef": 2.0,  # TDMPC update coef
    "bc_coef": 0.5,
    "cost_coef": 0.5,  # TDMPC update coef
    "value_coef": 0.1,  # TDMPC update coef
    "use_dynamics_norm": False,
    "use_critic_norm": False,
    "elite_portion": 0.01,  # 0.01
    "num_samples": 400,  # 400
    "inference_horizon": 5,  # 5
    "train_horizon": 5,  # 5
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
    target_act = target_act.permute(1, 0, 2)
    pred_tz = encoder(target_obs[:, 0, :])
    recon_loss, kl_loss = 0.0, 0.0
    for t in range(horizon):
        ta = target_act[t]
        pred_act, bc_mean, bc_std = bc_policy(pred_tz, ta)
        recon_loss += (gamma**t) * F.mse_loss(pred_act, ta)
        kl_loss += (gamma**t) * (
            -0.5
            * (1 + torch.log(bc_std.pow(2)) - bc_mean.pow(2) - bc_std.pow(2))
            .sum(-1)
            .mean()
        )
        pred_tz = dynamics(pred_tz, ta)
    # 0.5 weight is from BCQ implementation See @aviralkumar implementation
    return recon_loss + 0.5 * kl_loss


def dynamics_loss_fn(
    dynamics,
    encoder,
    encoder_target,
    target_obs,
    target_act,
    target_next_obs,
    config,
):
    target_obs = target_obs.permute(1, 0, 2)
    target_act = target_act.permute(1, 0, 2)
    target_next_obs = target_next_obs.permute(1, 0, 2)
    gamma, horizon = config["gamma"], config["train_horizon"]
    dynamics_loss = 0.0
    pred_tz = encoder(target_obs[0])
    for t in range(horizon):
        ta, tno = target_act[t], target_next_obs[t]
        with torch.no_grad():
            tnz = encoder_target(tno)
        pred_nz = dynamics(pred_tz, ta)
        dynamics_loss += (gamma**t) * F.mse_loss(pred_nz, tnz)
        pred_tz = pred_nz

    if config.get("use_dynamics_norm", False):
        for params in dynamics.parameters():
            dynamics_loss += params.pow(2).sum() * 0.001
    return dynamics_loss


def critic_loss_fn(
    critic,
    dynamics,
    encoder,
    target_obs,
    target_act,
    target_next_obs,
    target_label,
    config,
):
    target_obs = target_obs.permute(1, 0, 2)
    target_act = target_act.permute(1, 0, 2)
    target_next_obs = target_next_obs.permute(1, 0, 2)
    target_label = target_label.permute(1, 0)
    gamma, horizon = config["gamma"], config["train_horizon"]
    critic_loss = 0.0

    pred_tz = encoder(target_obs[0])
    for t in range(horizon):
        ta, tl = target_act[t], target_label[t]
        pred_nz = dynamics(pred_tz, ta)
        logits = critic(pred_nz)
        critic_loss += (gamma**t) * F.binary_cross_entropy_with_logits(logits, tl)
        pred_tz = pred_nz

    if config.get("use_critic_norm", False):
        for params in critic.parameters():
            critic_loss += params.pow(2).sum() * 0.001
    return critic_loss


def calculate_target_value(config, value, encoder, obs, next_obs, label):
    gamma, lambda_c = config["gamma"], config["lambda_c"]
    with torch.no_grad():
        z, next_z = encoder(obs), encoder(next_obs)
        value_z = torch.min(*value.V(z))
        value_next_z = torch.min(*value.V(next_z))
    # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
    adavantage_c = label + gamma * value_next_z - value_z
    length = adavantage_c.shape[0]
    cumsum = adavantage_c[-1]
    for idx in reversed(range(length - 1)):
        cumsum = adavantage_c[idx] + gamma * lambda_c * cumsum
        adavantage_c[idx] = cumsum
    return adavantage_c + value_z


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
    target_value_fn = partial(calculate_target_value, config, value_target, encoder)
    target_value = torch.vmap(target_value_fn, in_dims=(0, 0, 0))(
        target_obs, target_next_obs, target_label
    )
    target_act = target_act.permute(1, 0, 2)
    target_value = target_value.permute(1, 0)
    pred_tz = encoder(target_obs[:, 0, :])
    value_loss = 0.0
    for t in range(horizon):
        pred_v1, pred_v2 = value.V(pred_tz)
        value_loss += (gamma**t) * (
            F.mse_loss(pred_v1, target_value[t]) + F.mse_loss(pred_v2, target_value[t])
        )
        pred_tz = dynamics(pred_tz, target_act[t])

    return value_loss


def mcem_policy(dynamics, critic, policy, value, encoder, obs, config, device):
    horizon = config["inference_horizon"]
    num_samples = config["num_samples"]
    num_elite = int(config["elite_portion"] * num_samples)
    gamma = config["gamma"]
    samples = []
    costs = torch.zeros(num_samples, dtype=torch.float32, device=device)
    all_z = encoder(obs.repeat(num_samples, 1))
    with torch.no_grad():
        for t in range(horizon):
            all_act = policy.decode_bc(all_z)
            all_z = dynamics(all_z, all_act)
            costs += (gamma**t) * nn.functional.sigmoid(critic(all_z))
            samples.append(all_act.unsqueeze(1))
        costs += (gamma**t) * torch.min(*value.V(all_z))
    samples = torch.cat(samples, dim=1)
    best_control_idx = torch.argsort(costs)[:num_elite]
    elite_controls = samples[best_control_idx]
    elite_costs = costs[best_control_idx]
    return (
        elite_controls.mean(dim=0),
        elite_costs.mean(),
        elite_costs.min(),
        elite_costs.max(),
        costs.mean(),
        costs.max() - costs.min(),
    )


def sample_data_in_sequence(data: np.array, timestep=1):
    resample_data = []
    excluding_self_timestep = timestep - 1
    for d in data:
        for i in range(d.shape[0] - excluding_self_timestep):
            resample_data.append(d[i : i + timestep])
    return np.array(resample_data)


def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(f"{args.device}:{args.device_id}")
    config = default_cfg

    # evaluation environment
    eval_env = safety_gymnasium.make(
        args.task, render_mode="rgb_array", camera_name="track"
    )
    eval_env.reset(seed=None)

    # set model
    obs_space, act_space = eval_env.observation_space, eval_env.action_space
    encoder = Encoder(
        obs_dim=obs_space.shape[0], latent_dim=config["latent_obs_dim"]
    ).to(device)
    encoder_target = deepcopy(encoder)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=3e-4)
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
    critic = VCritic(
        obs_dim=config["latent_obs_dim"],
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

    # set training steps
    num_epochs = config.get("num_epochs", args.num_epochs)
    batch_size = config.get("batch_size", args.batch_size)

    # data
    for file in os.listdir(args.data_path):
        negative_matched = re.search("^.*?neg_.*$", file)
        positive_matched = re.search("^.*?pos_.*$", file)
        if not (negative_matched or positive_matched):
            continue
        filepath = os.path.join(args.data_path, file)
        with h5py.File(filepath, "r") as d:
            observations = sample_data_in_sequence(
                np.array(d["obs"]).squeeze(), timestep=config["train_horizon"]
            )
            actions = sample_data_in_sequence(
                np.array(d["act"]).squeeze(), timestep=config["train_horizon"]
            )
            next_observations = sample_data_in_sequence(
                np.array(d["next_obs"]).squeeze(), timestep=config["train_horizon"]
            )
            costs = sample_data_in_sequence(
                np.array(d["cost"]).squeeze(), timestep=config["train_horizon"]
            )
            avg_reward = np.mean(np.sum(d["reward"], axis=1))
            avg_cost = np.mean(np.sum(d["cost"], axis=1))
        print(f"Added data: {filepath}")
        print(f"Avg reward: {avg_reward}")
        print(f"Avg cost: {avg_cost}")
        if negative_matched:
            neg_observations = observations
            neg_actions = actions
            neg_next_observations = next_observations
            neg_costs = costs
        else:
            union_observations = observations
            union_actions = actions
            union_next_observations = next_observations
            union_costs = costs
    observations = np.concatenate([neg_observations, union_observations], axis=0)
    mu_obs = observations.mean(axis=(0, 1))
    std_obs = observations.std(axis=(0, 1))
    observations = (observations - mu_obs) / (std_obs + EP)
    observations = torch.as_tensor(observations, dtype=torch.float32, device=device)
    actions = np.concatenate([neg_actions, union_actions], axis=0)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    next_observations = np.concatenate(
        [neg_next_observations, union_next_observations], axis=0
    )
    next_observations = (next_observations - mu_obs) / (std_obs + EP)
    next_observations = torch.as_tensor(
        next_observations, dtype=torch.float32, device=device
    )
    labels = np.concatenate([neg_costs, union_costs])  # use costs as labels
    labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
    label_one_count = (labels.sum(axis=1) > 0).sum()
    label_zero_count = labels.shape[0] - label_one_count
    label_weights = [1 / label_zero_count, 1 / label_one_count]
    sample_weights = torch.as_tensor(
        [label_weights[int(i > 0)] for i in labels.sum(axis=1)],
        dtype=torch.float32,
        device=device,
    )
    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=labels.shape[0], replacement=True
    )
    dataloader = DataLoader(
        dataset=TensorDataset(observations, actions, next_observations, labels),
        batch_size=batch_size,
        sampler=weighted_sampler,
    )

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
        for target_obs, target_act, target_next_obs, target_label in dataloader:
            step += 1
            config["train_horizon"] = 5 if epoch > 5 else 1

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

            critic_loss = critic_loss_fn(
                critic=critic,
                dynamics=dynamics,
                encoder=encoder,
                target_obs=target_obs,
                target_act=target_act,
                target_next_obs=target_next_obs,
                target_label=target_label,
                config=config,
            )

            value_loss = value_cost_loss_fn(
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
            total_loss.register_hook(lambda grad: grad * (1 / config["train_horizon"]))

            encoder_optimizer.zero_grad()
            dynamics_optimizer.zero_grad()
            bc_vae_policy_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            value_cost_optimizer.zero_grad()
            total_loss.backward()
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

            if (step % config["update_freq"]) == 0:
                ema(encoder, encoder_target, config["update_tau"])
                ema(value_cost, value_cost_target, config["update_tau"])

            logger.store(
                **{
                    "Loss/Loss_bc_policy": bc_policy_loss.item(),
                    "Loss/Loss_dynamics": recon_loss.item(),
                    "Loss/Loss_critic": critic_loss.item(),
                    "Loss/Loss_value_cost": value_loss.item(),
                    "Loss/Loss_total": total_loss.item(),
                }
            )
            logger.logged = False
        training_end_time = time.time()

        eval_start_time = time.time()
        is_last_epoch = epoch >= num_epochs - 1
        eval_episodes = 5 if is_last_epoch else 1
        if args.use_eval:
            for id in range(eval_episodes):
                eval_done = False
                eval_obs, _ = eval_env.reset()
                eval_obs = (eval_obs - mu_obs) / (std_obs + EP)
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
                    act, hcost_mu, hcost_min, hcost_max, samples_mu, samples_gap = (
                        mcem_policy(
                            dynamics=dynamics,
                            critic=critic,
                            policy=bc_vae_policy,
                            value=value_cost,
                            encoder=encoder,
                            obs=eval_obs,
                            config=config,
                            device=device,
                        )
                    )
                    next_obs, reward, cost, terminated, truncated, _ = eval_env.step(
                        act[0].detach().squeeze().cpu().numpy()
                    )
                    next_obs = (next_obs - mu_obs) / (std_obs + EP)
                    next_obs = torch.as_tensor(
                        next_obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        critic_cost = nn.functional.sigmoid(critic(encoder(next_obs)))
                    eval_obs = next_obs
                    eval_reward += reward
                    eval_cost += cost
                    eval_critic += critic_cost.item()
                    eval_len += 1
                    eval_done = terminated or truncated
                    logger.store(
                        **{
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
                        value = torch.min(*value_cost.V(encoder(eval_obs))).item()
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
                    itr=epoch, torch_saver_elements=value_cost, prefix="value_cost"
                )
                logger.save_state(
                    state_dict={
                        "mu_obs": mu_obs,
                        "std_obs": std_obs,
                    },
                    itr=epoch,
                )
    logger.torch_save(
        itr=epoch, torch_saver_elements=bc_vae_policy, prefix="bc_vae_policy"
    )
    logger.torch_save(itr=epoch, torch_saver_elements=dynamics, prefix="morel_dynamics")
    logger.torch_save(itr=epoch, torch_saver_elements=critic, prefix="critic")
    logger.torch_save(itr=epoch, torch_saver_elements=value_cost, prefix="value_cost")
    logger.save_state(
        state_dict={
            "mu_obs": mu_obs,
            "std_obs": std_obs,
        },
        itr=epoch,
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
