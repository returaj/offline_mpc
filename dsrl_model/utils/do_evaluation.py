import argparse
import os
import os.path as osp
import re
import time
from distutils.util import strtobool
from functools import partial

import dsrl.offline_safety_gymnasium  # type: ignore
import gymnasium as gym
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dsrl_model.utils.models import (
    ContrastiveCostModel,
    ExpCostModel,
    SafeDiceTanhMixtureActor,
)
from dsrl_model.utils.utils import ActionRepeater

EP = 1e-6
WORST_COST_EVALS = [0.1, 0.2, 0.25, 0.3, 0.5]

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
        {
            "name": "--cost-model-path",
            "type": str,
            "default": "cost_model_model_0.pt",
            "help": "cost model file name to add if it exists",
        },
        {
            "name": "--add-predicted-cost",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "whether to add predicted cost information",
        },
        {
            "name": "--is-contrastive",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "whether to use contrastive cost for prediction",
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
    bc.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    bc.eval()
    return bc


def load_cost_model(obs_dim, act_dim, hidden_size, path, device, is_contrastive):
    if is_contrastive:
        cost_model = ContrastiveCostModel(
            obs_dim=obs_dim + act_dim, hidden_sizes=[hidden_size, hidden_size]
        ).to(device)
    else:
        cost_model = ExpCostModel(
            obs_dim=obs_dim + act_dim, hidden_sizes=[hidden_size, hidden_size]
        ).to(device)
    cost_model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    cost_model.eval()
    return cost_model


def timeit(func):
    def wrapped_func(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        total_time = time.time() - start_time
        return total_time, *ret

    return wrapped_func


def normalize(mu_obs, std_obs, obs):
    if mu_obs is None:
        return obs
    return (obs - mu_obs) / (std_obs + EP)


@timeit
def evaluate(
    eval_env,
    bc_policy,
    device,
    num_evals,
    norm_fn,
    cost_model=None,
    is_contrastive=False,
):
    num_cost_percent = [int(num_evals * per) for per in WORST_COST_EVALS]

    ep_rewards, ep_costs, ep_lens, ep_pred_costs = [], [], [], []
    for _ in range(num_evals):
        done = False
        obs, _ = eval_env.reset()
        obs = torch.as_tensor(
            norm_fn(obs), dtype=torch.float32, device=device
        ).unsqueeze(0)
        rewards, costs, lens, pred_cost = 0, 0, 0, 0
        while not done:
            act = bc_policy.action(obs)
            next_obs, reward, terminated, truncated, info = eval_env.step(
                act[0].detach().squeeze().cpu().numpy()
            )
            cost = info["cost"]
            next_obs = torch.as_tensor(
                norm_fn(next_obs), dtype=torch.float32, device=device
            ).unsqueeze(0)
            if cost_model is not None:
                oa = torch.cat([obs, act], dim=1)
                if is_contrastive:
                    pred_cost += cost_model(oa)[-1].item()
                else:
                    pred_cost += cost_model(oa, use_sigmoid=True).item()
            obs = next_obs
            rewards += reward
            costs += cost
            lens += 1
            done = terminated or truncated
        ep_rewards.append(rewards)
        ep_costs.append(costs)
        ep_lens.append(lens)
        ep_pred_costs.append(pred_cost)
    ep_costs = sorted(ep_costs)
    mean_worst_costs = [np.mean(ep_costs[-num:]) for num in num_cost_percent]
    return (
        np.mean(ep_rewards),
        np.mean(ep_costs),
        mean_worst_costs,
        np.mean(ep_lens),
        np.mean(ep_pred_costs),
    )


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
    device_name = "cpu" if args.device == "cpu" else f"{args.device}:{args.device_id}"
    device = torch.device(device_name)

    path = args.model_path
    bc_model_files = [
        f for f in os.listdir(path) if re.search(r"bc_policy_model_[0-9]+.pt", f)
    ]
    state_path = osp.join(path, "../norm/state.pkl")
    mu_obs, std_obs = None, None
    if osp.exists(state_path):
        state_dict = joblib.load(state_path, mmap_mode="r")
        mu_obs, std_obs = state_dict["mu_obs"], state_dict["std_obs"]

    is_contrastive = args.is_contrastive
    cost_model = None
    cost_model_path = osp.join(path, args.cost_model_path)
    to_add_cost_pred = args.add_predicted_cost and osp.exists(cost_model_path)
    if to_add_cost_pred:
        cost_model = load_cost_model(
            obs_dim=obs_space.shape[0],
            act_dim=act_space.shape[0],
            hidden_size=config["hidden_size"],
            path=cost_model_path,
            device=device,
            is_contrastive=is_contrastive,
        )

    ids = sorted(
        [
            int(re.search(r"bc_policy_model_([0-9]+).pt", f).group(1))
            for f in bc_model_files
        ]
    )
    reward_values, length_values = [], []
    cost_values, pred_cost_values = [], []
    worst_cost_values = [[] for _ in WORST_COST_EVALS]
    for id in ids:
        bc_policy = load_model(
            obs_dim=obs_space.shape[0],
            act_dim=act_space.shape[0],
            hidden_size=config["hidden_size"],
            path=osp.join(path, f"bc_policy_model_{id}.pt"),
            device=device,
        )
        total_time, reward, cost, worst_costs, length, pred_cost = evaluate(
            eval_env=eval_env,
            bc_policy=bc_policy,
            device=device,
            num_evals=args.num_evals,
            norm_fn=partial(normalize, mu_obs, std_obs),
            cost_model=cost_model,
            is_contrastive=is_contrastive,
        )
        print(
            f"task: {args.task}, seed: {args.seed}, id: {id}, time: {total_time:.2f}sec"
        )
        reward_values.append(reward)
        cost_values.append(cost)
        length_values.append(length)
        pred_cost_values.append(pred_cost)
        for i, worst_cost in enumerate(worst_costs):
            worst_cost_values[i].append(worst_cost)

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = osp.join(args.model_path, "..")
    save_csv(
        ids,
        reward_values,
        osp.join(log_dir, f"ep_reward_{args.num_evals}_{args.seed}.csv"),
    )
    save_csv(
        ids,
        cost_values,
        osp.join(log_dir, f"ep_cost_{args.num_evals}_{args.seed}.csv"),
    )
    save_csv(
        ids,
        length_values,
        osp.join(log_dir, f"ep_length_{args.num_evals}_{args.seed}.csv"),
    )
    for i, per in enumerate(WORST_COST_EVALS):
        save_csv(
            ids,
            worst_cost_values[i],
            osp.join(log_dir, f"ep_worst_cost_{per}_{args.num_evals}_{args.seed}.csv"),
        )

    if to_add_cost_pred:
        save_csv(
            ids,
            pred_cost_values,
            osp.join(log_dir, f"ep_pred_cost_{args.num_evals}_{args.seed}.csv"),
        )


if __name__ == "__main__":
    args = create_arguments()
    main(args)
