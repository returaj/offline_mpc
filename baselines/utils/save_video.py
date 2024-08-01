import argparse
import os
import numpy as np
import joblib
import torch
import safety_gymnasium
from gymnasium.wrappers.record_video import RecordVideo

from baselines.utils.models import Actor


EP = 1e-6

default_cfg = {
    "hidden_sizes": [512, 512],
    "num_trajectories": 100,
    "min_return": 20,
    "max_cost": 14,
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
            "name": "--num-videos",
            "type": int,
            "default": 5,
            "help": "Number of videos to save",
        },
        {
            "name": "--model-path",
            "type": str,
            "default": "../runs",
            "help": "path of saved agents",
        },
        {
            "name": "--state-path",
            "type": str,
            "default": "../runs",
            "help": "path of saved environment state",
        },
        {
            "name": "--video-dir",
            "type": str,
            "default": "../runs",
            "help": "path to save video",
        },
    ]
    parser = argparse.ArgumentParser(description="RL Policy")
    for param in custom_parameters:
        param_name = param.pop("name")
        parser.add_argument(param_name, **param)

    args = parser.parse_args()
    return args


def load_model(obs_dim, act_dim, hidden_sizes, path, device):
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes)
    actor.load_state_dict(torch.load(path))
    actor = actor.to(device)
    actor.eval()
    return actor


def load_state(path):
    obs_norm = joblib.load(path, "r")["Normalizer"]
    return obs_norm


def main(args):
    config = default_cfg

    env = safety_gymnasium.make(args.task, render_mode="rgb_array")
    env = RecordVideo(
        env=env,
        video_folder=args.video_dir,
        name_prefix="test-video",
        episode_trigger=lambda x: x % 1 == 0,
    )

    obs_space, act_space = env.observation_space, env.action_space
    device = torch.device(f"{args.device}:{args.device_id}")
    actor = load_model(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
        path=args.model_path,
        device=device,
    )
    obs_norm = load_state(args.state_path)

    eval_configs = {}
    for id in range(args.num_videos):
        done = False
        eval_config = {"total_return": 0.0, "total_cost": 0.0}
        obs, _ = env.reset(seed=args.seed + id)
        while not done:
            obs = (obs - obs_norm.mean) / np.sqrt(obs_norm.var + EP)
            obs = torch.as_tensor(obs.unsqueeze(), dtype=torch.float32, device=device)
            dist = actor(obs)
            act = dist.mean.detach().squeeze().cpu().numpy()
            obs, reward, cost, terminated, truncated, _ = env.step(act)
            eval_config["total_return"] += reward
            eval_config["total_cost"] += cost
            done = terminated or truncated
            env.render()
        print(eval_config)
        eval_configs[id] = eval_config
    env.close()


if __name__ == "__main__":
    args = create_arguments()
    main(args)
