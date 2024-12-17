import argparse
import os
import os.path as osp
import re

import dsrl.offline_safety_gymnasium  # type: ignore
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dsrl_model.utils.models import SafeDiceTanhMixtureActor
from dsrl_model.utils.utils import ActionRepeater

EP = 1e-6

default_cfg = {
    "hidden_size": 256,
    "action_repeat": 1,  # set to 2, min value is 1
}


def create_arguments():
    custom_parameters = [
        {"name": "--seed", "type": int, "default": 0, "help": "seed information"},
        {
            "name": "--task",
            "type": str,
            "default": "OfflineHopperVelocityGymnasium-v1",
            "help": "The task to run",
        },
        {
            "name": "--device",
            "type": str,
            "default": "cpu",
            "help": "The device to run the model on",
        },
        {
            "name": "--device-id",
            "type": int,
            "default": 0,
            "help": "The device id to run the model on",
        },
        {
            "name": "--model-path",
            "type": str,
            "default": "../runs",
            "help": "path of saved bc agents",
        },
        {
            "name": "--state-path",
            "type": str,
            "default": None,
            "help": "path of additional state informations",
        },
        {
            "name": "--log-dir",
            "type": str,
            "default": None,
            "help": "path to save video",
        },
        {
            "name": "--num-evals",
            "type": int,
            "default": 5,
            "help": "number of evaluations",
        },
    ]
    parser = argparse.ArgumentParser(description="RL Policy")
    for param in custom_parameters:
        param_name = param.pop("name")
        parser.add_argument(param_name, **param)

    args = parser.parse_args()
    return args


def load_model(obs_dim, act_dim, hidden_size, path, device):
    bc = SafeDiceTanhMixtureActor(
        obs_dim=obs_dim, act_dim=act_dim, hidden_size=hidden_size
    ).to(device)
    bc.load_state_dict(torch.load(path, weights_only=True))
    bc.eval()
    return bc


def evaluate(eval_env, bc_policy, device, num_evals):
    ep_rewards, ep_costs, ep_lens = [], [], []
    for _ in range(num_evals):
        done = False
        obs, _ = eval_env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        rewards, costs, lens = 0, 0, 0
        while not done:
            act = bc_policy.action(obs)
            next_obs, reward, terminated, truncated, info = eval_env.step(
                act[0].detach().squeeze().cpu().numpy()
            )
            cost = info["cost"]
            next_obs = torch.as_tensor(
                next_obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            obs = next_obs
            rewards += reward
            costs += cost
            lens += 1
            done = terminated or truncated
        ep_rewards.append(rewards)
        ep_costs.append(costs)
        ep_lens.append(lens)
    return np.mean(ep_rewards), np.mean(ep_costs), np.mean(ep_lens)


def save_csv(steps, values, path):
    dicts = {"Step": steps, "Value": values}
    df = pd.DataFrame(dicts)
    df.to_csv(path, header=True, index=False)


def main(args):
    config = default_cfg

    eval_env = gym.make(args.task)
    eval_env = ActionRepeater(eval_env, num_repeats=config["action_repeat"])
    eval_env.reset(seed=args.seed)

    obs_space, act_space = eval_env.observation_space, eval_env.action_space
    device = torch.device(f"{args.device}:{args.device_id}")

    path = args.model_path
    bc_model_files = [
        f for f in os.listdir(path) if re.search(r"bc_policy_model_[0-9]+.pt", f)
    ]
    ids = sorted(
        [
            int(re.search(r"bc_policy_model_([0-9]+).pt", f).group(1))
            for f in bc_model_files
        ]
    )
    reward_values, cost_values, length_values = [], [], []
    for id in ids:
        bc_policy = load_model(
            obs_dim=obs_space.shape[0],
            act_dim=act_space.shape[0],
            hidden_size=config["hidden_size"],
            path=osp.join(path, f"bc_policy_model_{id}.pt"),
            device=device,
        )
        reward, cost, length = evaluate(
            eval_env=eval_env,
            bc_policy=bc_policy,
            device=device,
            num_evals=args.num_evals,
        )
        reward_values.append(reward)
        cost_values.append(cost)
        length_values.append(length)

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = osp.join(args.model_path, "..")
    save_csv(ids, reward_values, osp.join(log_dir, "ep_reward.csv"))
    save_csv(ids, cost_values, osp.join(log_dir, "ep_cost.csv"))
    save_csv(ids, length_values, osp.join(log_dir, "ep_length.csv"))


if __name__ == "__main__":
    args = create_arguments()
    main(args)
