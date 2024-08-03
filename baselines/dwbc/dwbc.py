import os
import sys
import time
import re
from collections import deque

import random
import numpy as np
import h5py

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

import safety_gymnasium
from baselines.dwbc.utils import single_agent_args
from baselines.utils.models import Actor, VCritic
from baselines.utils.logger import EpochLogger


EP = 1e-6

default_cfg = {
    "hidden_sizes": [512, 512],
    "gamma": 0.99,
    "target_kl": 0.02,
    "max_grad_norm": 40.0,
}


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
    eval_env = safety_gymnasium.make(args.task)
    eval_env.reset(seed=None)

    # set model
    obs_space, act_space = eval_env.observation_space, eval_env.action_space
    policy = Actor(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    critic = VCritic(
        obs_dim=(obs_space.shape[0] + act_space.shape[0]),
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    # set training steps
    num_epochs = config.get("num_epochs", args.num_epochs)
    batch_size = config.get("batch_size", args.batch_size)

    # data
    for file in os.listdir(args.data_path):
        negative_matched = re.search("^.*?neg_.*$", file)
        positive_matched = re.search("^.*?pos_.*$", file)
        if not (negative_matched or positive_matched):
            pass
        filepath = os.path.join(args.data_path, file)
        with h5py.File(filepath, "r") as d:
            observations = np.concatenate(np.array(d["obs"]).squeeze(), axis=0)
            actions = np.concatenate(np.array(d["act"]).squeeze(), axis=0)
        if negative_matched:
            neg_observations = observations
            neg_actions = actions
        else:
            union_observations = observations
            union_actions = actions
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
    labels = np.concatenate(
        [
            np.zeros(neg_observations.shape[0]),
            np.ones(union_observations.shape[0]),
        ],
    )
    labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
    dataloader = DataLoader(
        dataset=TensorDataset(observations, actions, labels),
        batch_size=batch_size,
        shuffle=True,
    )

    # set critic logger
    dict_args = vars(args)
    dict_args.update(config)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    logger.save_config(dict_args)
    logger.log("Start with critic training.")

    # train critic model
    for epoch in range(num_epochs):
        training_start_time = time.time()
        for obs, act, label in dataloader:
            critic_optimizer.zero_grad()
            logits = critic(torch.cat([obs, act], dim=1))
            loss_critic = nn.functional.binary_cross_entropy_with_logits(logits, label)
            if config.get("use_critic_norm", True):
                for params in critic.parameters():
                    loss_critic += params.pow(2).sum() * 0.001
            loss_critic.backward()
            clip_grad_norm_(critic.parameters(), config["max_grad_norm"])
            critic_optimizer.step()
            logger.store(**{"Loss/Loss_critic": loss_critic.mean().item()})
            logger.logged = False
        training_end_time = time.time()
        if not logger.logged:
            logger.log_tabular("Train/Epoch_critic", epoch + 1)
            logger.log_tabular("Loss/Loss_critic")
            logger.log_tabular(
                "Time/Update_critic", training_end_time - training_start_time
            )
            logger.dump_tabular()
    logger.close()

    # setup policy logger
    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    logger.setup_torch_saver(policy)
    logger.log("Start with policy training.")

    # train weighted bc
    old_distribution = policy(observations)
    final_kl = torch.ones_like(old_distribution.loc)
    for epoch in range(num_epochs):
        training_start_time = time.time()
        for obs, act, _ in dataloader:
            with torch.no_grad():
                weights = nn.functional.sigmoid(critic(torch.cat([obs, act], dim=1)))
            policy_optimizer.zero_grad()
            dist = policy(obs)
            log_prob = dist.log_prob(act).sum(axis=-1)
            loss_policy = torch.mean(weights * log_prob)
            if config.get("use_critic_norm", True):
                for params in policy.parameters():
                    loss_policy += params.pow(2).sum() * 0.001
            loss_policy.backward()
            clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
            policy_optimizer.step()
            logger.store(**{"Loss/Loss_policy": loss_policy.mean().item()})
            logger.logged = False
        training_end_time = time.time()

        new_distribution = policy(observations)
        final_kl = (
            torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
            .sum(-1, keepdim=True)
            .mean()
            .item()
        )
        old_distribution = new_distribution

        eval_start_time = time.time()
        eval_episodes = 1 if epoch < num_epochs - 1 else 10
        if args.use_eval:
            for _ in range(eval_episodes):
                eval_done = False
                eval_obs, _ = eval_env.reset()
                eval_obs = (eval_obs - mu_obs) / (std_obs + EP)
                eval_obs = torch.as_tensor(
                    eval_obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                eval_reward, eval_cost, eval_len = 0.0, 0.0, 0.0
                while not eval_done:
                    with torch.no_grad():
                        act = policy(eval_obs).mean
                    next_obs, reward, cost, terminated, truncated, _ = eval_env.step(
                        act.detach().squeeze().cpu().numpy()
                    )
                    next_obs = (next_obs - mu_obs) / (std_obs + EP)
                    eval_obs = torch.as_tensor(
                        next_obs, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    eval_reward += reward
                    eval_cost += cost
                    eval_len += 1
                    eval_done = terminated or truncated
                eval_rew_deque.append(eval_reward)
                eval_cost_deque.append(eval_cost)
                eval_len_deque.append(eval_len)
            logger.store(
                **{
                    "Metrics/EvalEpRet": np.mean(eval_rew_deque),
                    "Metrics/EvalEpCost": np.mean(eval_cost_deque),
                    "Metrics/EvalEpLen": np.mean(eval_len_deque),
                }
            )
        eval_end_time = time.time()

        if not logger.logged:
            if args.use_eval:
                logger.log_tabular("Metrics/EvalEpRet")
                logger.log_tabular("Metrics/EvalEpCost")
                logger.log_tabular("Metrics/EvalEpLen")
            logger.log_tabular("Train/Epoch_policy", epoch + 1)
            logger.log_tabular(
                "Train/Total_policy", training_end_time - training_start_time
            )
            logger.log_tabular("Train/KL", final_kl)
            logger.log_tabular("Loss/Loss_policy")
            logger.log_tabular(
                "Time/Update_policy", training_end_time - training_start_time
            )
            if args.use_eval:
                logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
            logger.log_tabular("Time/Total", eval_end_time - training_start_time)
            logger.dump_tabular()
            if (epoch + 1) % 2 == 0 or epoch == 0:
                logger.torch_save(itr=epoch)
                logger.save_state(
                    state_dict={
                        "mu_obs": mu_obs,
                        "std_obs": std_obs,
                    },
                    itr=epoch,
                )
    logger.torch_save(itr=epoch)
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
