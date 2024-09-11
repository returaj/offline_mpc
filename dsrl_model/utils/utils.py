import argparse
from distutils.util import strtobool

import gymnasium


class ActionRepeater(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    def __init__(self, env: gymnasium.Env, num_repeats: int, discount: int = 1.0):
        gymnasium.utils.RecordConstructorArgs.__init__(self)
        gymnasium.Wrapper.__init__(self, env)
        self._num_repeats = num_repeats
        self._discount = discount

    def step(self, action):
        # Check the equivalent code from TD-MPC
        reward, cost = 0.0, 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            obs, r, term, trun, infos = self.env.step(action)
            reward += discount * (r or 0.0)
            cost += discount * (infos["cost"] or 0.0)
            discount *= self._discount
            if term or trun:
                break
        infos["cost"] = cost
        return obs, reward, term, trun, infos

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def get_params_norm(params, grads=False):
    total_norm = 0.0
    for p in params:
        if grads:
            total_norm += p.grad.detach().data.norm(2).item()
        else:
            total_norm += p.data.norm(2).item()
    return total_norm


def single_agent_args():
    custom_parameters = [
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed"},
        {
            "name": "--use-eval",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "Use evaluation environment for testing",
        },
        {
            "name": "--task",
            "type": str,
            "default": "OfflinePointGoal1Gymnasium-v0",
            "help": "The task to run",
        },
        {
            "name": "--experiment",
            "type": str,
            "default": "equal",
            "help": "Experiment name",
        },
        {
            "name": "--log-dir",
            "type": str,
            "default": "../runs",
            "help": "directory to save agent logs",
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
            "name": "--write-terminal",
            "type": lambda x: bool(strtobool(x)),
            "default": True,
            "help": "Toggles terminal logging",
        },
        {
            "name": "--num-epochs",
            "type": int,
            "default": 1000,
            "help": "Total timesteps of the experiments",
        },
        {
            "name": "--batch-size",
            "type": int,
            "default": 128,
            "help": "The number of steps to run in each environment per policy rollout",
        },
        {
            "name": "--lr",
            "type": float,
            "default": 3e-4,  # 1e-3 performs better
            "help": "Default common learning rate for the models",
        },
    ]
    # Create argument parser
    parser = argparse.ArgumentParser(description="RL Policy")
    for param in custom_parameters:
        param_name = param.pop("name")
        parser.add_argument(param_name, **param)

    # Parse arguments

    args = parser.parse_args()
    cfg_env = {}
    return args, cfg_env
