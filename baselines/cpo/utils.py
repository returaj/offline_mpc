# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
from distutils.util import strtobool
import numpy as np
from typing import Callable
import gymnasium
from gymnasium.wrappers.normalize import NormalizeObservation
import safety_gymnasium
from safety_gymnasium.wrappers import (
    SafeAutoResetWrapper,
    SafeRescaleAction,
    SafeUnsqueeze,
)
from safety_gymnasium.vector.async_vector_env import SafetyAsyncVectorEnv


class SafeMonitor(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    def __init__(self, env: gymnasium.Env) -> None:
        gymnasium.utils.RecordConstructorArgs.__init__(self)
        gymnasium.Wrapper.__init__(self, env)

    def update_info(self, obs, info):
        assert (
            "unnormalized_obs" not in info
        ), 'info dict cannot contain key "unormalized_obs"'
        info["unnormalized_obs"] = obs
        return info

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = self.update_info(obs, info)
        return obs, reward, cost, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = self.update_info(obs, info)
        return obs, info


class SafeNormalizeObservation(NormalizeObservation):
    """This wrapper will normalize observations as Gymnasium's NormalizeObservation wrapper does."""

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, costs, terminateds, truncateds, infos = self.env.step(action)
        obs = (
            self.normalize(obs)
            if self.is_vector_env
            else self.normalize(np.array([obs]))[0]
        )
        for k in infos.keys():
            if k in ("final_observation", "unnormalized_obs"):
                if self.is_vector_env:
                    infos[k] = np.array(
                        [
                            (
                                self.normalize(np.array([array]))[0]
                                if array is not None
                                else np.zeros(obs.shape[-1])
                            )
                            for array in infos[k]
                        ],
                    )
                else:
                    array = infos[k]
                    infos[k] = (
                        self.normalize(np.array([array]))[0]
                        if array is not None
                        else np.zeros(obs.shape[-1])
                    )
        return obs, rews, costs, terminateds, truncateds, infos


def make_sa_mujoco_env(
    num_envs: int, env_id: str, seed: int | None = None, monitor: bool = False
):
    """
    Creates and wraps an environment based on the specified parameters.

    Args:
        num_envs (int): Number of parallel environments.
        env_id (str): ID of the environment to create.
        seed (int or None, optional): Seed for the random number generator. Default is None.
        monitor: monitor unnormalized observation

    Returns:
        env: The created and wrapped environment.
        obs_space: The observation space of the environment.
        act_space: The action space of the environment.

    Examples:
        >>> from safepo.common.env import make_sa_mujoco_env
        >>>
        >>> env, obs_space, act_space = make_sa_mujoco_env(
        >>>     num_envs=1,
        >>>     env_id="SafetyPointGoal1-v0",
        >>>     seed=0
        >>> )
    """
    if num_envs > 1:

        def create_env() -> Callable:
            """Creates an environment that can enable or disable the environment checker."""
            env = safety_gymnasium.make(env_id)
            env = SafeRescaleAction(env, -1.0, 1.0)
            return env

        env_fns = [create_env for _ in range(num_envs)]
        env = SafetyAsyncVectorEnv(env_fns)
        env = SafeNormalizeObservation(env)
        env.reset(seed=seed)
        obs_space = env.single_observation_space
        act_space = env.single_action_space
    else:
        env = safety_gymnasium.make(env_id)
        env.reset(seed=seed)
        obs_space = env.observation_space
        act_space = env.action_space
        env = SafeAutoResetWrapper(env)
        env = SafeRescaleAction(env, -1.0, 1.0)
        if monitor:
            env = SafeMonitor(env)
        env = SafeNormalizeObservation(env)
        env = SafeUnsqueeze(env)

    return env, obs_space, act_space


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
            "name": "--num-envs",
            "type": int,
            "default": 10,
            "help": "The number of parallel game environments",
        },
        {
            "name": "--experiment",
            "type": str,
            "default": "single_agent_exp",
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
            "name": "--headless",
            "type": lambda x: bool(strtobool(x)),
            "default": False,
            "help": "Toggles headless mode",
        },
        {
            "name": "--total-steps",
            "type": int,
            "default": 10000000,
            "help": "Total timesteps of the experiments",
        },
        {
            "name": "--steps-per-epoch",
            "type": int,
            "default": 20000,
            "help": "The number of steps to run in each environment per policy rollout",
        },
        {
            "name": "--randomize",
            "type": bool,
            "default": False,
            "help": "Wheather to randomize the environments' initial states",
        },
        {"name": "--cost-limit", "type": float, "default": 25.0, "help": "cost_lim"},
        {
            "name": "--lagrangian-multiplier-init",
            "type": float,
            "default": 0.001,
            "help": "initial value of lagrangian multiplier",
        },
        {
            "name": "--lagrangian-multiplier-lr",
            "type": float,
            "default": 0.035,
            "help": "learning rate of lagrangian multiplier",
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
