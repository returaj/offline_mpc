import argparse
from distutils.util import strtobool


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
            "default": "SafetyPointGoal1-v0",
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
            "name": "--data-path",
            "type": str,
            "default": "../../data",
            "help": "directory for trajectory dataset",
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
