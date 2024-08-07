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
from torch.utils.data import DataLoader, TensorDataset

import safety_gymnasium
from baselines.utils.models import Actor, VCritic, MLP
from baselines.utils.logger import EpochLogger
from baselines.utils.save_video import save_video
from baselines.model_based.utils import single_agent_args


EP = 1e-6

default_cfg = {
    "hidden_sizes": [512, 512],
    "max_grad_norm": 40.0,
    "sampling_smoothing": 0.0,
    "evolution_smoothing": 0.1,
    "elite_portion": 0.1,
    "max_iter": 10,  # 10
    "num_samples": 400,  # 400
    "horizon": 10,  # 10
}


def get_initial_action(dynamics, bc_policy, init_state, horizon):
    actions = []
    state = init_state
    with torch.no_grad():
        for _ in range(horizon):
            act = bc_policy(state).mean
            actions.append(act)
            state = state + dynamics(torch.cat([state, act], dim=1))
    return torch.cat(actions, dim=0)


def get_horizon_cost(dynamics, critic, init_state, controls):
    next_states = []
    state = init_state
    controls = controls.unsqueeze(1)
    with torch.no_grad():
        for act in controls:
            state = state + dynamics(torch.cat([state, act], dim=1))
            next_states.append(state)
        next_states = torch.cat(next_states, dim=0)
        states = torch.cat([init_state, next_states[:-1]], dim=0)
        state_next_state = torch.cat([states, next_states], dim=1)
        costs = torch.vmap(critic, in_dims=0)(state_next_state)
    return torch.sum(nn.functional.sigmoid(costs))


def cem_policy(dynamics, critic, eval_obs, act, config, device):
    mean = act
    # act_high, act_low = config["act_high"], config["act_low"]
    act_high = torch.as_tensor(config["act_high"], dtype=torch.float32, device=device)
    act_low = torch.as_tensor(config["act_low"], dtype=torch.float32, device=device)
    std = (act_high - act_low) / 2.0
    std = std.repeat(act.shape[0], 1).to(dtype=torch.float32)

    num_elits = int(config["num_samples"] * config["elite_portion"])
    smoothing = config["evolution_smoothing"]
    cost_fn = partial(get_horizon_cost, dynamics, critic, eval_obs)
    for _ in range(config["max_iter"]):
        gaussian_dist = torch.distributions.Normal(mean, std)
        samples = gaussian_dist.sample((config["num_samples"],))
        samples = torch.clip(samples, act_low, act_high)
        costs = torch.vmap(cost_fn, in_dims=0)(samples)
        best_control_idx = torch.argsort(costs)[:num_elits]
        elite_controls = samples[best_control_idx]
        new_mean = torch.mean(elite_controls, dim=0)
        new_std = torch.std(elite_controls, dim=0)
        mean = smoothing * mean + (1 - smoothing) * new_mean
        std = smoothing * std + (1 - smoothing) * new_std
    horizon_cost = cost_fn(mean)
    return mean, horizon_cost


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
    bc_policy = Actor(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    bc_policy_optimizer = torch.optim.Adam(bc_policy.parameters(), lr=3e-4)
    critic = VCritic(
        obs_dim=2 * obs_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    dynamics = MLP(
        in_dim=obs_space.shape[0] + act_space.shape[0],
        out_dim=obs_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    dynamics_optimizer = torch.optim.Adam(dynamics.parameters(), lr=3e-4)

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
            observations = np.concatenate(np.array(d["obs"]).squeeze(), axis=0)
            actions = np.concatenate(np.array(d["act"]).squeeze(), axis=0)
            next_observations = np.concatenate(
                np.array(d["next_obs"]).squeeze(), axis=0
            )
            costs = np.concatenate(np.array(d["cost"]).squeeze())
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
    mu_obs = (
        union_observations.mean(axis=0) * union_observations.shape[0]
        + neg_observations.mean(axis=0) * neg_observations.shape[0]
    ) / (union_observations.shape[0] + neg_observations.shape[0])
    std_obs = (
        union_observations.std(axis=0) * union_observations.shape[0]
        + neg_observations.std(axis=0) * neg_observations.shape[0]
    ) / (union_observations.shape[0] + neg_observations.shape[0])
    observations = np.concatenate([neg_observations, union_observations], axis=0)
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
    dataloader = DataLoader(
        dataset=TensorDataset(observations, actions, next_observations, labels),
        batch_size=batch_size,
        shuffle=True,
    )

    # set logger
    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_critic_deque = deque(maxlen=50)
    eval_horizon_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    dict_args = vars(args)
    dict_args.update(config)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    logger.save_config(dict_args)
    logger.log("Start with critic and dynamics training.")

    # train critic and dynamics model
    for epoch in range(num_epochs):
        training_start_time = time.time()
        for target_obs, target_act, target_next_obs, label in dataloader:
            bc_policy_optimizer.zero_grad()
            dynamics_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            pred_act = bc_policy(target_obs).rsample()
            bc_policy_loss = nn.functional.mse_loss(target_act, pred_act)
            if config.get("use_critic_norm", True):
                for params in bc_policy.parameters():
                    bc_policy_loss += params.pow(2).sum() * 0.001
            pred_next_obs = target_obs + dynamics(
                torch.cat([target_obs, target_act], dim=1)
            )
            dynamics_loss = nn.functional.mse_loss(target_next_obs, pred_next_obs)
            if config.get("use_critic_norm", True):
                for params in dynamics.parameters():
                    dynamics_loss += params.pow(2).sum() * 0.001
            pos_weight = (label.shape[0] - label.sum()) / (
                label.sum() + EP
            )  # num_neg_samples / num_pos_samples
            true_logits = critic(torch.cat([target_obs, target_next_obs], dim=1))
            critic_loss = nn.functional.binary_cross_entropy_with_logits(
                true_logits, label, pos_weight=pos_weight
            )
            if config.get("use_critic_norm", True):
                for params in critic.parameters():
                    critic_loss += params.pow(2).sum() * 0.001
            cyclic_logits = critic(torch.cat([target_obs, pred_next_obs], dim=1))
            cyclic_loss = nn.functional.binary_cross_entropy_with_logits(
                cyclic_logits, label, pos_weight=pos_weight
            )
            loss = bc_policy_loss + dynamics_loss + critic_loss + cyclic_loss
            loss.backward()
            clip_grad_norm_(bc_policy.parameters(), config["max_grad_norm"])
            clip_grad_norm_(critic.parameters(), config["max_grad_norm"])
            clip_grad_norm_(dynamics.parameters(), config["max_grad_norm"])
            bc_policy_optimizer.step()
            critic_optimizer.step()
            dynamics_optimizer.step()
            logger.store(
                **{
                    "Loss/Loss_bc_policy": bc_policy_loss.mean().item(),
                    "Loss/Loss_dynamics": dynamics_loss.mean().item(),
                    "Loss/Loss_critic": critic_loss.mean().item(),
                    "Loss/Loss_cyclic": cyclic_loss.mean().item(),
                    "Loss/Loss_dynamics_critic": loss.mean().item(),
                }
            )
            logger.logged = False
        training_end_time = time.time()

        eval_start_time = time.time()
        is_last_epoch = epoch >= num_epochs - 1
        eval_episodes = 1 if is_last_epoch else 1
        if args.use_eval:
            config["act_high"] = act_space.high
            config["act_low"] = act_space.low
            for id in range(eval_episodes):
                eval_done = False
                eval_obs, _ = eval_env.reset()
                eval_obs = (eval_obs - mu_obs) / (std_obs + EP)
                eval_obs = torch.as_tensor(
                    eval_obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                eval_reward, eval_cost, eval_critic, eval_horizon_cost, eval_len = (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                ep_frames = []
                while not eval_done:
                    init_act = get_initial_action(
                        dynamics=dynamics,
                        bc_policy=bc_policy,
                        init_state=eval_obs,
                        horizon=config["horizon"],
                    )
                    act, horizon_cost = cem_policy(
                        dynamics=dynamics,
                        critic=critic,
                        eval_obs=eval_obs,
                        act=init_act,
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
                        critic_cost = nn.functional.sigmoid(
                            critic(torch.cat([eval_obs, next_obs], dim=1))
                        )
                    eval_obs = next_obs
                    eval_reward += reward
                    eval_cost += cost
                    eval_critic += critic_cost.item()
                    eval_horizon_cost += horizon_cost.item()
                    eval_len += 1
                    eval_done = terminated or truncated
                    if is_last_epoch:
                        ep_frames.append(eval_env.render())
                save_video(
                    ep_frames,
                    prefix_name=f"video_{id}",
                    video_dir=osp.join(args.log_dir, "video"),
                )
                eval_rew_deque.append(eval_reward)
                eval_cost_deque.append(eval_cost)
                eval_critic_deque.append(eval_critic)
                eval_horizon_cost_deque.append(eval_horizon_cost)
                eval_len_deque.append(eval_len)
            logger.store(
                **{
                    "Metrics/EvalEpRet": np.mean(eval_rew_deque),
                    "Metrics/EvalEpCost": np.mean(eval_cost_deque),
                    "Metrics/EvalCriticCost": np.mean(eval_critic_deque),
                    "Metrics/EvalHorizonCost": np.mean(eval_horizon_cost_deque),
                    "Metrics/EvalEpLen": np.mean(eval_len_deque),
                }
            )
        eval_end_time = time.time()

        if not logger.logged:
            if args.use_eval:
                logger.log_tabular("Metrics/EvalEpRet")
                logger.log_tabular("Metrics/EvalEpCost")
                logger.log_tabular("Metrics/EvalCriticCost")
                logger.log_tabular("Metrics/EvalHorizonCost")
                logger.log_tabular("Metrics/EvalEpLen")
            logger.log_tabular("Train/Epoch", epoch + 1)
            logger.log_tabular("Loss/Loss_bc_policy")
            logger.log_tabular("Loss/Loss_dynamics")
            logger.log_tabular("Loss/Loss_critic")
            logger.log_tabular("Loss/Loss_cyclic")
            logger.log_tabular("Loss/Loss_dynamics_critic")
            if args.use_eval:
                logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
            logger.log_tabular(
                "Time/Update_dynamics_critic", training_end_time - training_start_time
            )
            logger.log_tabular("Time/Total", eval_end_time - training_start_time)
            logger.dump_tabular()
            if (epoch + 1) % 20 == 0 or epoch == 0:
                logger.torch_save(
                    itr=epoch, torch_saver_elements=bc_policy, prefix="bc_policy"
                )
                logger.torch_save(
                    itr=epoch, torch_saver_elements=dynamics, prefix="dynamics"
                )
                logger.torch_save(
                    itr=epoch, torch_saver_elements=critic, prefix="critic"
                )
                logger.save_state(
                    state_dict={
                        "mu_obs": mu_obs,
                        "std_obs": std_obs,
                    },
                    itr=epoch,
                )
    logger.torch_save(itr=epoch, torch_saver_elements=bc_policy, prefix="bc_policy")
    logger.torch_save(itr=epoch, torch_saver_elements=dynamics, prefix="dynamics")
    logger.torch_save(itr=epoch, torch_saver_elements=critic, prefix="critic")
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
