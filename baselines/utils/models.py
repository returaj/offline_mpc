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


EPS = 1e-7


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


def atanh(x):
    one_plus_x = (1 + x).clamp(min=EPS)
    one_minus_x = (1 - x).clamp(min=EPS)
    return 0.5 * torch.log(one_plus_x / one_minus_x)


class TanhActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list = [64, 64]):
        super().__init__()
        self.mean = build_mlp_network([obs_dim] + hidden_sizes + [act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim), requires_grad=True)

    def forward(self, obs: torch.Tensor):
        mu = self.mean(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        raw_action = dist.rsample()
        return torch.tanh(raw_action), mu, raw_action

    def log_prob(self, obs, action=None, raw_action=None):
        mu = self.mean(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        if raw_action is None:
            raw_action = atanh(action)
        if action is None:
            action = torch.tanh(raw_action)
        log_normal = dist.log_prob(raw_action).sum(-1)
        log_prob = log_normal - (1.0 - action.pow(2)).clamp(min=EPS).log().sum(-1)
        return log_prob


class BcqVAE(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        latent_dim: int = 750,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.pre_encoder_layer = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 750), nn.ReLU(), nn.Linear(750, 750), nn.ReLU()
        )
        self.encoder_mean = nn.Linear(750, latent_dim)
        self.encoder_log_std = nn.Linear(750, latent_dim)

        self.pre_decoder_layer = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, 750),
            nn.ReLU(),
            nn.Linear(750, 750),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(750, act_dim)

        self.latent_dim = latent_dim
        self.device = device

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        z = self.pre_encoder_layer(torch.cat([obs, act], dim=1))
        mean = self.encoder_mean(z)
        # clamped for numerical stability
        # see BEAR algo implementation by @aviralkumar
        log_std = self.encoder_log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = Normal(mean, std).rsample()

        u = self.decode(obs, z)
        return u, mean, std

    def decode(self, obs, z):
        action = self.pre_decoder_layer(torch.cat([obs, z], dim=1))
        return torch.tanh(self.decoder(action))

    def decode_bc(self, obs, z=None):
        if z is None:
            z = torch.normal(
                0,
                1,
                size=(obs.size(0), self.latent_dim),
                dtype=torch.float32,
                device=self.device,
            )
        return self.decode(obs, z)

    def decode_bc_multiple(self, obs, z=None, num_decodes=10):
        if z is None:
            z = torch.normal(
                0,
                1,
                size=(obs.size(0) * num_decodes, self.latent_dim),
                dtype=torch.float32,
                device=self.device,
            )
        # repeats obs and form a size of Batch X NumDecodes X obs_dim
        repeat_obs = obs.unsqueeze(1).repeat(1, num_decodes, 1).view(-1, obs.size(1))
        action = self.decode(repeat_obs, z)
        batch_action = action.view(obs.size(0), num_decodes, -1)
        return batch_action


class MorelDynamics(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list = [64, 64],
        state_diff_std: float = 0.01,
    ):
        super().__init__()
        self.model = MLP(obs_dim + act_dim, obs_dim, hidden_sizes)
        self._state_diff_std = state_diff_std

    def set_state_diff_std(self, state_diff_std):
        self._state_diff_std = state_diff_std

    def forward(self, obs, act):
        next_obs = obs + self._state_diff_std * self.model(torch.cat([obs, act], dim=1))
        return next_obs


class EnsembleDynamics(nn.Module):
    def __init__(
        self,
        num_ds: int,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list = [64, 64],
        state_diff_std: float = 0.01,
    ):
        super().__init__()
        self.models = [
            MorelDynamics(obs_dim, act_dim, hidden_sizes, state_diff_std)
            for _ in range(num_ds)
        ]

    def set_state_diff_std(self, state_diff_std):
        for m in self.models:
            m.set_state_diff_std(state_diff_std)

    def forward(self, obs, act, with_var=False):
        all_next_obs = [m(obs, act) for m in self.models]
        all_next_obs = torch.cat([no.unsqueeze(0) for no in all_next_obs], dim=0)
        if with_var:
            assert (
                len(self.models) > 1
            ), "There is only one model and std needs atleast two models to be defined."
            std = torch.std(all_next_obs, dim=0, unbiased=False)
            return all_next_obs, std
        return all_next_obs

    def m1(self, obs, act):
        m1 = self.models[0]
        return m1(obs, act)

    def m_all(self, obs, act, with_var=False):
        return self.forward(obs, act, with_var)


class EnsembleValue(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=[64, 64]):
        super().__init__()
        self._V1 = VCritic(obs_dim=obs_dim, hidden_sizes=hidden_sizes)
        self._V2 = VCritic(obs_dim=obs_dim, hidden_sizes=hidden_sizes)

    def V(self, obs):
        return self._V1(obs), self._V2(obs)


class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, enc_dim=256):
        super().__init__()
        # TDMPC encoder
        self.model = nn.Sequential(
            nn.Linear(obs_dim, enc_dim), nn.ELU(), nn.Linear(enc_dim, latent_dim)
        )

    def forward(self, obs):
        return self.model(obs)


class TdmpcDynamics(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[64, 64]):
        super().__init__()
        sizes = [obs_dim + act_dim] + hidden_sizes + [obs_dim]
        layers = list()
        for j in range(len(sizes) - 1):
            act = nn.ELU() if j < len(sizes) - 2 else nn.Identity()
            affine_layer = nn.Linear(sizes[j], sizes[j + 1])
            layers += [affine_layer, act]
        self.model = nn.Sequential(*layers)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.model(x)
