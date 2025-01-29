import argparse
import os
import os.path as osp
import imageio
import numpy as np
import cv2
import joblib
import json
import torch
import torch.nn as nn
import safety_gymnasium

from baselines.utils.models import MorelDynamics, BcqVAE, VCritic, EnsembleValue


EP = 1e-6

default_cfg = {
    "hidden_sizes": [512, 512],
    "gamma": 0.99,
    "elite_portion": 0.01,  # 0.01
    "num_samples": 400,  # 400
    "inference_horizon": 5,
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


def mcem_policy(dynamics, critic, policy, value, obs, config, device):
    horizon = config["inference_horizon"]
    num_samples = config["num_samples"]
    num_elite = int(config["elite_portion"] * num_samples)
    gamma = config["gamma"]
    samples = []
    costs = torch.zeros(num_samples, dtype=torch.float32, device=device)
    all_obs = obs.repeat(num_samples, 1)
    with torch.no_grad():
        for t in range(horizon):
            all_act = policy.decode_bc(all_obs)
            all_obs = dynamics(all_obs, all_act)
            costs += (gamma**t) * nn.functional.sigmoid(critic(all_obs))
            samples.append(all_act.unsqueeze(1))
        costs += (gamma**t) * torch.min(*value.V(all_obs))
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


def load_model(obs_dim, act_dim, hidden_sizes, state_diff_std, path, device):
    bc = BcqVAE(
        obs_dim=obs_dim,
        act_dim=act_dim,
        latent_dim=2 * act_dim,
        device=device,
    ).to(device)
    dynamics = MorelDynamics(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden_sizes,
        state_diff_std=state_diff_std,
    ).to(device)
    critic = VCritic(obs_dim=obs_dim, hidden_sizes=hidden_sizes).to(device)
    value_cost = EnsembleValue(obs_dim=obs_dim, hidden_sizes=hidden_sizes).to(device)

    dynamics_path = osp.join(path, "morel_dynamicsmodel0.pt")
    critic_path = osp.join(path, "criticmodel0.pt")
    value_path = osp.join(path, "value_costmodel0.pt")
    bc_path = osp.join(path, "bc_vae_policymodel0.pt")

    dynamics.load_state_dict(torch.load(dynamics_path))
    critic.load_state_dict(torch.load(critic_path))
    value_cost.load_state_dict(torch.load(value_path))
    bc.load_state_dict(torch.load(bc_path))

    bc.eval()
    dynamics.eval()
    critic.eval()
    value_cost.eval()
    return bc, dynamics, critic, value_cost


def load_state(path):
    return joblib.load(path, "r")


def save_video(frames, values, prefix_name, video_dir, fps=20):
    os.makedirs(video_dir, exist_ok=True)
    video_path = osp.join(video_dir, f"{prefix_name}.mp4")
    writer = imageio.get_writer(video_path, fps=fps)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for f, v in zip(frames, values):
        f = cv2.putText(
            np.array(f),
            f"V(s) = {v:.4f}",
            (5, 20),
            font,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_4,
        )
        writer.append_data(f)
    writer.close()


def main(args):
    config = default_cfg

    env = safety_gymnasium.make(args.task, render_mode="rgb_array", camera_name="track")

    obs_space, act_space = env.observation_space, env.action_space
    device = torch.device(f"{args.device}:{args.device_id}")

    obs_norm = load_state(args.state_path)
    state_diff_std = np.array(
        load_state(osp.join(osp.dirname(args.state_path), "state_diff_std.pkl"))[
            "state_diff_std"
        ]
    )
    bc, dynamics, critic, value_cost = load_model(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
        state_diff_std=torch.as_tensor(
            state_diff_std, dtype=torch.float32, device=device
        ),
        path=args.model_path,
        device=device,
    )

    eval_configs = {}
    for id in range(args.num_videos):
        done = False
        frames, values = [], []
        eval_config = {"total_return": 0.0, "total_cost": 0.0}
        obs, _ = env.reset(seed=args.seed + id)
        obs = (obs - obs_norm["mu_obs"]) / (obs_norm["std_obs"] + EP)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        while not done:
            act, *_ = mcem_policy(
                policy=bc,
                critic=critic,
                dynamics=dynamics,
                value=value_cost,
                obs=obs,
                config=config,
                device=device,
            )
            act = act[0].detach().squeeze().cpu().numpy()
            obs, reward, cost, terminated, truncated, _ = env.step(act)
            obs = (obs - obs_norm["mu_obs"]) / (obs_norm["std_obs"] + EP)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            value = torch.min(*value_cost.V(obs)).item()
            eval_config["total_return"] += reward
            eval_config["total_cost"] += cost
            done = terminated or truncated
            frames.append(env.render())
            values.append(value)
        save_video(
            frames=frames,
            values=values,
            fps=30,
            prefix_name=f"video_{id}",
            video_dir=args.video_dir,
        )
        print(eval_config)
        eval_configs[id] = eval_config
    env.close()
    with open(osp.join(args.video_dir, "eval_config.json"), "w") as f:
        json.dump(eval_configs, f, indent=4)


if __name__ == "__main__":
    args = create_arguments()
    main(args)
