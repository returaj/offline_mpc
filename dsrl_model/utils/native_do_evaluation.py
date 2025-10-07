import os

os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
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

from dsrl_model.utils.utils import ActionRepeater

EPS = 1e-6
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
        {
            "name": "--is-flax",
            "type": lambda x: bool(strtobool(x)),
            "default": True,
            "help": "whether to uses flax model for prediction",
        },
    ]
    parser = argparse.ArgumentParser(description="RL Policy")
    for param in custom_parameters:
        param_name = param.pop("name")
        parser.add_argument(param_name, **param)

    args = parser.parse_args()
    return args


def load_torch_model(obs_dim, act_dim, hidden_size, path, device):
    bc = SafeDiceTanhMixtureActor(
        obs_dim=obs_dim, act_dim=act_dim, hidden_size=hidden_size
    ).to(device)
    bc.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    bc.eval()
    return bc


def load_flax_model(obs_dim, act_dim, hidden_size, path, device):
    del device
    restored_pure_dict = joblib.load(path)

    model = SafeDiceTanhMixtureActor(
        rngs=nnx.Rngs(0), obs_dim=obs_dim, act_dim=act_dim, hidden_size=hidden_size
    )
    abstract_model = nnx.eval_shape(lambda: model)
    graphdef, abstract_state = nnx.split(abstract_model)
    nnx.replace_by_pure_dict(abstract_state, restored_pure_dict)
    bc = nnx.merge(graphdef, abstract_state)
    return bc


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
    return (obs - mu_obs) / (std_obs + EPS)


def map_obs(obs, norm_fn, is_flax=True, device=None):
    obs = norm_fn(obs)
    if is_flax:
        obs = jnp.expand_dims(jnp.array(obs, dtype=jnp.float32), axis=0)
    else:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    return obs


def map_act(act, is_flax):
    act = act[0].squeeze()
    if is_flax:
        act = np.array(act)
    else:
        act = act.detach().cpu().numpy()
    return act


@timeit
def evaluate(
    eval_env,
    bc_policy,
    norm_fn,
    num_evals,
    device=None,
    is_flax=True,
):
    num_cost_percent = [int(num_evals * per) for per in WORST_COST_EVALS]

    ep_rewards, ep_costs, ep_lens = [], [], []
    for _ in range(num_evals):
        done = False
        obs, _ = eval_env.reset()
        obs = map_obs(obs, norm_fn, is_flax, device)
        rewards, costs, lens, pred_cost = 0, 0, 0, 0
        while not done:
            act = map_act(bc_policy.action(obs), is_flax)
            next_obs, reward, terminated, truncated, info = eval_env.step(act)
            cost = info["cost"]
            next_obs = map_obs(next_obs, norm_fn, is_flax, device)
            obs = next_obs
            rewards += reward
            costs += cost
            lens += 1
            done = terminated or truncated
        ep_rewards.append(rewards)
        ep_costs.append(costs)
        ep_lens.append(lens)
    ep_costs = sorted(ep_costs)
    mean_worst_costs = [np.mean(ep_costs[-num:]) for num in num_cost_percent]
    return (
        np.mean(ep_rewards),
        np.mean(ep_costs),
        mean_worst_costs,
        np.mean(ep_lens),
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

    if args.is_flax:
        jax.default_device = jax.devices(args.device)[args.device_id]
        device = args.device
        load_model = load_flax_model
    else:
        device_name = (
            "cpu" if args.device == "cpu" else f"{args.device}:{args.device_id}"
        )
        device = torch.device(device_name)
        load_model = load_torch_model

    path = args.model_path
    bc_model_files = [
        f for f in os.listdir(path) if re.search(r"bc_policy_model_[0-9]+.pt", f)
    ]
    state_path = osp.join(path, "../norm/state.pkl")
    mu_obs, std_obs = None, None
    if osp.exists(state_path):
        state_dict = joblib.load(state_path, mmap_mode="r")
        mu_obs, std_obs = state_dict["mu_obs"], state_dict["std_obs"]

    ids = sorted(
        [
            int(re.search(r"bc_policy_model_([0-9]+).pt", f).group(1))
            for f in bc_model_files
        ]
    )
    reward_values, length_values = [], []
    cost_values = []
    worst_cost_values = [[] for _ in WORST_COST_EVALS]
    for id in ids:
        bc_policy = load_model(
            obs_dim=obs_space.shape[0],
            act_dim=act_space.shape[0],
            hidden_size=config["hidden_size"],
            path=osp.join(path, f"bc_policy_model_{id}.pt"),
            device=device,
        )
        total_time, reward, cost, worst_costs, length = evaluate(
            eval_env=eval_env,
            bc_policy=bc_policy,
            norm_fn=partial(normalize, mu_obs, std_obs),
            num_evals=args.num_evals,
            device=device,
            is_flax=args.is_flax,
        )
        print(
            f"task: {args.task}, seed: {args.seed}, id: {id}, time: {total_time:.2f}sec"
        )
        reward_values.append(reward)
        cost_values.append(cost)
        length_values.append(length)
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


if __name__ == "__main__":
    args = create_arguments()
    if args.is_flax:
        import jax
        import jax.numpy as jnp
        from flax import nnx

        from dsrl_model.utils.models_jax import SafeDiceTanhMixtureActor
    else:
        import torch

        from dsrl_model.utils.models import SafeDiceTanhMixtureActor

    main(args)
