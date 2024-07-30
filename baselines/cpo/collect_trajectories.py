import os.path as osp
import h5py
import joblib
import torch
import numpy as np
from baselines.cpo.utils import make_sa_mujoco_env, single_agent_args
from baselines.utils.models import Actor


EP = 1e-6

default_cfg = {
    "hidden_sizes": [512, 512],
    "num_trajectories": 100,
    "min_return": 20,
    "max_cost": 14,
}


def load_model(obs_dim, act_dim, hidden_sizes, path, device):
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes)
    actor.load_state_dict(torch.load(path))
    actor = actor.to(device)
    actor.eval()
    return actor


def load_state(path):
    obs_norm = joblib.load(path, "r")["Normalize"]
    return obs_norm


class Trajectories:
    def __init__(self):
        self.trajectories = {
            "obs": [],
            "act": [],
            "reward": [],
            "cost": [],
            "next_obs": [],
            "done": [],
        }
        self.trajectory = {k: [] for k in self.trajectories.keys()}

    def store(self, **kwargs):
        for k, v in kwargs.items():
            self.trajectory[k].append(v)

    def merge(self, min_return, max_cost):
        merged = False
        ep_return, ep_cost = np.sum(self.trajectory["reward"]), np.sum(
            self.trajectory["cost"]
        )
        if ep_return >= min_return and ep_cost < max_cost:
            for k, v in self.trajectory.items():
                self.trajectories[k].append(np.array(v))
            merged = True
        self.trajectory = {k: [] for k in self.trajectories.keys()}
        return merged, ep_return, ep_cost

    def get_trajectories(self):
        return self.trajectories


def collect_trajectories(env, obs_norm, actor, config, device):
    trajectories = Trajectories()
    obs, info = env.reset()
    unnorm_obs = info["unnormalized_obs"]
    accpeted_traj, cnt_traj = 0, 0
    while accpeted_traj < config["num_trajectories"]:
        obs = (obs - obs_norm.mean) / np.sqrt(obs_norm.var + EP)
        dist = actor(torch.tensor(obs, dtype=torch.float32, device=device))
        act = dist.mean.detach().squeeze().cpu().numpy()
        next_obs, reward, cost, terminated, truncated, next_info = env.step(act)
        next_unnorm_obs = next_info["unnormalized_obs"]
        done = terminated or truncated
        trajectories.store(
            obs=unnorm_obs,
            act=act,
            reward=reward,
            cost=cost,
            done=done,
            next_obs=next_info["unnormalized_final_obs"] if done else next_unnorm_obs,
        )
        obs, unnorm_obs = next_obs, next_unnorm_obs
        if done:
            merged, ep_return, ep_cost = trajectories.merge(
                min_return=config["min_return"], max_cost=config["max_cost"]
            )
            cnt_traj += 1
            if merged:
                accpeted_traj += 1
                print(
                    "total_trajectories: {:d}, accpeted_trajectories: {:d}, episode return: {:.3f}, episode cost: {:.3f}".format(
                        cnt_traj, accpeted_traj, ep_return, ep_cost
                    )
                )
    return trajectories.get_trajectories()


def main(args):
    env, obs_space, act_space = make_sa_mujoco_env(
        num_envs=1, env_id=args.task, seed=None, monitor=True
    )
    config = default_cfg
    device = torch.device(f"{args.device}:{args.device_id}")
    actor = load_model(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
        path=args.model_path,
        device=device,
    )
    obs_norm = load_state(args.state_path)
    print("Start collecting trajectories")
    trajectories = collect_trajectories(
        env=env, obs_norm=obs_norm, actor=actor, config=config, device=device
    )
    filepath = osp.join(osp.dirname(args.log_dir), "..", "trajectories.h5")
    hf = h5py.File(filepath, "w")
    for k, v in trajectories.items():
        hf.create_dataset(k, data=v)
    print("Done collecting trajectories")


if __name__ == "__main__":
    args, _ = single_agent_args()
    main(args)
