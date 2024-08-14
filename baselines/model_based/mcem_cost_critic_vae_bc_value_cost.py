import os
import os.path as osp
import sys
import time
import re
from collections import deque

from functools import partial
import random
import numpy as np
import h5py

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torcheval.metrics.functional import binary_confusion_matrix

import safety_gymnasium
from baselines.utils.models import MorelDynamics, BcqVAE, VCritic
from baselines.utils.logger import EpochLogger
from baselines.utils.save_video import save_video
from baselines.model_based.utils import single_agent_args


EP = 1e-6

default_cfg = {
    "hidden_sizes": [512, 512],
    "max_grad_norm": 40.0,
    "gamma": 0.99,
    "lambda_c": 0.95,
    "elite_portion": 0.01,  # 0.01
    "num_samples": 400,  # 400
    "horizon": 10,  # 10
}


def bc_policy_loss_fn(bc_policy, target_obs, target_act):
    reshaped_target_obs = target_obs.view(-1, target_obs.shape[-1])
    reshaped_target_act = target_act.view(-1, target_act.shape[-1])
    reshaped_pred_act, bc_mean, bc_std = bc_policy(
        reshaped_target_obs, reshaped_target_act
    )
    recon_loss = nn.functional.mse_loss(reshaped_pred_act, reshaped_target_act)
    kl_loss = (
        -0.5
        * (1 + torch.log(bc_std.pow(2)) - bc_mean.pow(2) - bc_std.pow(2)).sum(-1).mean()
    )
    # 0.5 weight is from BCQ implementation See @aviralkumar implementation
    return recon_loss + 0.5 * kl_loss


def dynamics_loss_fn(
    dynamics, target_obs, target_act, target_next_obs, config, self_force=False
):
    target_obs = target_obs.permute(1, 0, 2)
    target_act = target_act.permute(1, 0, 2)
    target_next_obs = target_next_obs.permute(1, 0, 2)
    dynamics_loss = 0.0
    for to, ta, tno in zip(target_obs, target_act, target_next_obs):
        pred_no = dynamics(to, ta)
        dynamics_loss += nn.functional.mse_loss(pred_no, tno)

    if self_force:
        pred_to = target_obs[0]
        for ta, tno in zip(target_act, target_next_obs):
            pred_no = dynamics(pred_to, ta)
            dynamics_loss += nn.functional.mse_loss(pred_no, tno)
            pred_to = pred_no

    if config.get("use_dynamics_norm", True):
        for params in dynamics.parameters():
            dynamics_loss += params.pow(2).sum() * 0.001
    return dynamics_loss


def critic_loss_fn(
    critic,
    dynamics,
    target_obs,
    target_act,
    target_next_obs,
    target_label,
    config,
    self_force=False,
):
    target_obs = target_obs.permute(1, 0, 2)
    target_act = target_act.permute(1, 0, 2)
    target_next_obs = target_next_obs.permute(1, 0, 2)
    target_label = target_label.permute(1, 0)
    critic_loss = 0.0
    for tno, tl in zip(target_next_obs, target_label):
        critic_loss += nn.functional.binary_cross_entropy_with_logits(critic(tno), tl)

    if self_force:
        pred_to = target_obs[0]
        for ta, tl in zip(target_act, target_label):
            with torch.no_grad():
                pred_no = dynamics(pred_to, ta)
            logits = critic(pred_no)
            critic_loss += nn.functional.binary_cross_entropy_with_logits(logits, tl)
            pred_to = pred_no

    if config.get("use_critic_norm", True):
        for params in critic.parameters():
            critic_loss += params.pow(2).sum() * 0.001
    return critic_loss


def calculate_target_value(config, value, obs, next_obs, label):
    gamma, lambda_c = config["gamma"], config["lambda_c"]
    with torch.no_grad():
        value_obs, value_next_obs = value(obs), value(next_obs)
    # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
    adavantage_c = label + gamma * value_next_obs - value_obs
    length = adavantage_c.shape[0]
    cumsum = adavantage_c[-1]
    for idx in reversed(range(length - 1)):
        cumsum = adavantage_c[idx] + gamma * lambda_c * cumsum
        adavantage_c[idx] = cumsum
    return adavantage_c + value_obs


def value_cost_loss_fn(
    value,
    dynamics,
    target_obs,
    target_act,
    target_next_obs,
    target_label,
    config,
    self_force=False,
):
    target_value_fn = partial(calculate_target_value, config, value)
    true_target_value = torch.vmap(target_value_fn, in_dims=(0, 0, 0))(
        target_obs, target_next_obs, target_label
    )
    true_pred_value = torch.vmap(value, in_dims=(0,))(target_obs)
    recon_loss = nn.functional.mse_loss(true_pred_value, true_target_value)
    return recon_loss


def mcem_policy(dynamics, critic, policy, value, obs, config, device):
    horizon = config["horizon"]
    num_samples = config["num_samples"]
    num_elite = int(config["elite_portion"] * num_samples)
    samples = []
    costs = torch.zeros(num_samples, dtype=torch.float32, device=device)
    all_obs = obs.repeat(num_samples, 1)
    with torch.no_grad():
        for _ in range(horizon - 1):
            all_act = policy.decode_bc(all_obs)
            all_obs = dynamics(all_obs, all_act)
            costs += nn.functional.sigmoid(critic(all_obs))
            samples.append(all_act.unsqueeze(1))
        costs += value(all_obs)
    samples = torch.cat(samples, dim=1)
    best_control_idx = torch.argsort(costs)[:num_elite]
    elite_controls = samples[best_control_idx]
    elite_costs = costs[best_control_idx]
    return (
        elite_controls.mean(dim=0),
        elite_costs.mean(),
        elite_costs.min(),
        elite_costs.max(),
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
    # See BEAR implementation from @aviralkumar
    latent_dim = config.get("latent_dim", act_space.shape[0] * 2)
    config["latent_dim"] = latent_dim
    bc_vae_policy = BcqVAE(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        latent_dim=latent_dim,
        device=device,
    ).to(device)
    bc_vae_policy_optimizer = torch.optim.Adam(bc_vae_policy.parameters(), lr=3e-4)
    critic = VCritic(
        obs_dim=obs_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    dynamics = MorelDynamics(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    dynamics_optimizer = torch.optim.Adam(dynamics.parameters(), lr=3e-4)
    value_cost = VCritic(
        obs_dim=obs_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
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
                np.array(d["obs"]).squeeze(), timestep=config["horizon"]
            )
            actions = sample_data_in_sequence(
                np.array(d["act"]).squeeze(), timestep=config["horizon"]
            )
            next_observations = sample_data_in_sequence(
                np.array(d["next_obs"]).squeeze(), timestep=config["horizon"]
            )
            costs = sample_data_in_sequence(
                np.array(d["cost"]).squeeze(), timestep=config["horizon"]
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

    # set state diff standard deviation
    state_diff_std = (next_observations - observations).std(dim=(0, 1))
    dynamics.set_state_diff_std(state_diff_std)

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
    for epoch in range(num_epochs):
        training_start_time = time.time()
        for target_obs, target_act, target_next_obs, target_label in dataloader:
            bc_policy_loss = bc_policy_loss_fn(bc_vae_policy, target_obs, target_act)
            bc_vae_policy_optimizer.zero_grad()
            bc_policy_loss.backward()
            clip_grad_norm_(bc_vae_policy.parameters(), config["max_grad_norm"])
            bc_vae_policy_optimizer.step()

            self_force = True if epoch > 10 else False
            dynamics_loss = dynamics_loss_fn(
                dynamics=dynamics,
                target_obs=target_obs,
                target_act=target_act,
                target_next_obs=target_next_obs,
                config=config,
                self_force=self_force,
            )
            dynamics_optimizer.zero_grad()
            dynamics_loss.backward()
            clip_grad_norm_(dynamics.parameters(), config["max_grad_norm"])
            dynamics_optimizer.step()

            critic_loss = critic_loss_fn(
                critic=critic,
                dynamics=dynamics,
                target_obs=target_obs,
                target_act=target_act,
                target_next_obs=target_next_obs,
                target_label=target_label,
                config=config,
                self_force=self_force,
            )
            critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(critic.parameters(), config["max_grad_norm"])
            critic_optimizer.step()

            value_cost_loss = value_cost_loss_fn(
                value=value_cost,
                dynamics=dynamics,
                target_obs=target_obs,
                target_act=target_act,
                target_next_obs=target_next_obs,
                target_label=target_label,
                config=config,
                self_force=self_force,
            )
            value_cost_optimizer.zero_grad()
            value_cost_loss.backward()
            clip_grad_norm_(value_cost.parameters(), config["max_grad_norm"])
            value_cost_optimizer.step()

            logger.store(
                **{
                    "Loss/Loss_bc_policy": bc_policy_loss.item(),
                    "Loss/Loss_dynamics": dynamics_loss.item(),
                    "Loss/Loss_critic": critic_loss.item(),
                    "Loss/Loss_value_cost": value_cost_loss.item(),
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
                ep_frames = []
                while not eval_done:
                    act, hcost_mu, hcost_min, hcost_max = mcem_policy(
                        dynamics=dynamics,
                        critic=critic,
                        policy=bc_vae_policy,
                        value=value_cost,
                        obs=eval_obs,
                        config=config,
                        device=device,
                    )
                    next_obs, reward, cost, terminated, truncated, _ = eval_env.step(
                        act[0].detach().squeeze().cpu().numpy()
                    )
                    next_obs = (next_obs - mu_obs) / (std_obs + EP)
                    next_obs = torch.as_tensor(
                        next_obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        critic_cost = nn.functional.sigmoid(critic(next_obs))
                    eval_obs = next_obs
                    eval_reward += reward
                    eval_cost += cost
                    eval_critic += critic_cost.item()
                    eval_len += 1
                    eval_done = terminated or truncated
                    logger.store(
                        **{
                            "Metrics/EvalHorizonCostMean": hcost_mu,
                            "Metrics/EvalHorizonCostMin": hcost_min,
                            "Metrics/EvalHorizonCostMax": hcost_max,
                        }
                    )
                    if is_last_epoch:
                        ep_frames.append(eval_env.render())
                if is_last_epoch:
                    save_video(
                        ep_frames,
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
            logger.log_tabular("Train/Epoch", epoch + 1)
            logger.log_tabular("Loss/Loss_bc_policy")
            logger.log_tabular("Loss/Loss_dynamics")
            logger.log_tabular("Loss/Loss_critic")
            logger.log_tabular("Loss/Loss_value_cost")
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
