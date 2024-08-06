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


from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.distributions import Normal


def build_mlp_network(sizes, activation=None):
    """
    Build a multi-layer perceptron (MLP) neural network.

    This function constructs an MLP network with the specified layer sizes and activation functions.

    Args:
        sizes (list of int): List of integers representing the sizes of each layer in the network.
        activation: activation funciton
    Returns:
        nn.Sequential: An instance of PyTorch's Sequential module representing the constructed MLP.
    """
    layers = list()
    for j in range(len(sizes) - 1):
        activation = activation or nn.Tanh
        act = activation if j < len(sizes) - 2 else nn.Identity
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        nn.init.kaiming_uniform_(affine_layer.weight, a=np.sqrt(5))
        layers += [affine_layer, act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Actor network for policy-based reinforcement learning.

    This class represents an actor network that outputs a distribution over actions given observations.

    Args:
        obs_dim (int): Dimensionality of the observation space.
        act_dim (int): Dimensionality of the action space.

    Attributes:
        mean (nn.Sequential): MLP network representing the mean of the action distribution.
        log_std (nn.Parameter): Learnable parameter representing the log standard deviation of the action distribution.

    Example:
        obs_dim = 10
        act_dim = 2
        actor = Actor(obs_dim, act_dim)
        observation = torch.randn(1, obs_dim)
        action_distribution = actor(observation)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list = [64, 64]):
        super().__init__()
        self.mean = build_mlp_network([obs_dim] + hidden_sizes + [act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim), requires_grad=True)

    def forward(self, obs: torch.Tensor):
        mean = self.mean(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)


class VCritic(nn.Module):
    """
    Critic network for value-based reinforcement learning.

    This class represents a critic network that estimates the value function for input observations.

    Args:
        obs_dim (int): Dimensionality of the observation space.

    Attributes:
        critic (nn.Sequential): MLP network representing the critic function.

    Example:
        obs_dim = 10
        critic = VCritic(obs_dim)
        observation = torch.randn(1, obs_dim)
        value_estimate = critic(observation)
    """

    def __init__(self, obs_dim, hidden_sizes: list = [64, 64]):
        super().__init__()
        self.critic = build_mlp_network([obs_dim] + hidden_sizes + [1])

    def forward(self, obs):
        return torch.squeeze(self.critic(obs), -1)


class ActorVCritic(nn.Module):
    """
    Actor-critic policy for reinforcement learning.

    This class represents an actor-critic policy that includes an actor network, two critic networks for reward
    and cost estimation, and provides methods for taking policy steps and estimating values.

    Args:
        obs_dim (int): Dimensionality of the observation space.
        act_dim (int): Dimensionality of the action space.

    Example:
        obs_dim = 10
        act_dim = 2
        actor_critic = ActorVCritic(obs_dim, act_dim)
        observation = torch.randn(1, obs_dim)
        action, log_prob, reward_value, cost_value = actor_critic.step(observation)
        value_estimate = actor_critic.get_value(observation)
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes: list = [64, 64]):
        super().__init__()
        self.reward_critic = VCritic(obs_dim, hidden_sizes)
        self.cost_critic = VCritic(obs_dim, hidden_sizes)
        self.actor = Actor(obs_dim, act_dim, hidden_sizes)

    def get_value(self, obs):
        """
        Estimate the value of observations using the critic network.

        Args:
            obs (torch.Tensor): Input observation tensor.

        Returns:
            torch.Tensor: Estimated value for the input observation.
        """
        return self.critic(obs)

    def step(self, obs, deterministic=False):
        """
        Take a policy step based on observations.

        Args:
            obs (torch.Tensor): Input observation tensor.
            deterministic (bool): Flag indicating whether to take a deterministic action.

        Returns:
            tuple: Tuple containing action tensor, log probabilities of the action, reward value estimate,
                   and cost value estimate.
        """

        dist = self.actor(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        value_r = self.reward_critic(obs)
        value_c = self.cost_critic(obs)
        return action, log_prob, value_r, value_c


class MLP(nn.Module):
    """
    Standard MLP network.

    This class represents a standard mlp network.

    Args:
        in_dim (int): Dimensionality of the input space.
        out_dim (int): Dimensionality of the output space.
    Attributes:
        vector_value (nn.Sequential): MLP network representing the output function.

    Example:
        in_dim, out_dim = 10, 5
        value = MLP(in_dim, out_dim)
        x = torch.randn(1, obs_dim)
        value_estimate = value(x)
    """

    def __init__(self, in_dim, out_dim, hidden_sizes: list = [64, 64]):
        super().__init__()
        self.model = build_mlp_network([in_dim] + hidden_sizes + [out_dim], nn.ReLU)

    def forward(self, x):
        return self.model(x)


class GaussianMLP(nn.Module):
    """
    Standard MLP network which outputs gaussian distributions mean and std.

    Args:
        in_dim (int): Dimensionality of the input space.
        out_dim (int): Dimensionality of the output space.
    Attributes:
        mean, std (nn.Sequential): MLP network representing the gaussian distribution.

    Example:
        in_dim, out_dim = 10, 5
        value = GaussianMLP(in_dim, out_dim)
        x = torch.randn(1, obs_dim)
        mean, std = value(x)
    """

    def __init__(self, in_dim, out_dim, hidden_sizes: list = [64, 64]):
        super().__init__()
        self.mean = MLP(in_dim, out_dim, hidden_sizes)
        self.log_std = nn.Parameter(torch.zeros(out_dim), requires_grad=True)

    def forward(self, x: torch.Tensor):
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
