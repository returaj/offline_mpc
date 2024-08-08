import argparse
import os
import os.path as osp
import joblib
import numpy as np
import safety_gymnasium
import torch
import torcheval
import torcheval.metrics

from baselines.utils.models import VCritic


EP = 1e-6

default_cfg = {
    "hidden_sizes": [512, 512],
    "num_trajectories": 100,
}


def create_arguments():
    custom_parameters = [
        {
            "name": "--seed",
            "type": int,
            "default": 0,
            "help": "Random seed",
        },
        {
            "name": "--task",
            "type": str,
            "default": "SafetyPointGoal1-v0",
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
            "name": "--num-trajectories",
            "type": int,
            "default": default_cfg["num_trajectories"],
            "help": "num of trajectories",
        },
        {
            "name": "--model-path",
            "type": str,
            "default": "../runs",
            "help": "model path used for loading the model",
        },
        {
            "name": "--state-path",
            "type": str,
            "default": "../run/state.pkl",
            "help": "path of saved environment state",
        },
    ]
    parser = argparse.ArgumentParser(description="RL Policy")
    for param in custom_parameters:
        param_name = param.pop("name")
        parser.add_argument(param_name, **param)

    args = parser.parse_args()
    return args


def load_critic_model(obs_dim, hidden_sizes, path, device):
    critic = VCritic(obs_dim=obs_dim, hidden_sizes=hidden_sizes)
    critic.load_state_dict(torch.load(path))
    critic = critic.to(device)
    critic.eval()
    return critic


def load_state(path):
    obs_norm = joblib.load(path, "r")
    return obs_norm


def calculate_critic_confusion_metric(args):
    env = safety_gymnasium.make(args.task)

    obs_space, act_space = env.observation_space, env.action_space
    device = torch.device(f"{args.device}:{args.device_id}")
    critic = load_critic_model(
        obs_dim=2 * obs_space.shape[0],
        hidden_sizes=default_cfg["hidden_sizes"],
        path=args.model_path,
        device=device,
    )
    obs_norm = load_state(args.state_path)

    true_cost, pred_cost = [], []
    for id in range(args.num_trajectories):
        done = False
        eval_config = {"total_return": 0.0, "total_cost": 0.0, "total_critic": 0.0}
        obs, _ = env.reset(seed=args.seed + id)
        obs = (obs - obs_norm["mu_obs"]) / (obs_norm["std_obs"] + EP)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        while not done:
            next_obs, reward, cost, terminated, truncated, _ = env.step(
                act_space.sample()
            )
            next_obs = (next_obs - obs_norm["mu_obs"]) / (obs_norm["std_obs"] + EP)
            next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            x = torch.cat([obs, next_obs]).unsqueeze(0)
            pcost = torch.nn.functional.sigmoid(critic(x)).item()
            pred_cost.append(pcost)
            true_cost.append(cost)
            eval_config["total_critic"] += pcost
            eval_config["total_return"] += reward
            eval_config["total_cost"] += cost
            done = terminated or truncated
            obs = next_obs
        print(eval_config)
    env.close()

    metric = torcheval.metrics.BinaryConfusionMatrix()
    metric.update(pred_cost, true_cost)
    cm = metric.compute()
    print(cm)
    return cm


if __name__ == "__main__":
    args = create_arguments()
    calculate_critic_confusion_metric(args)
