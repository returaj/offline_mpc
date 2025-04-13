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

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
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
        del hidden_sizes
        self.pre_encoder_layer = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU()
        )
        self.mean = nn.Linear(512, act_dim)
        self.log_std = nn.Linear(512, act_dim)

    def forward(self, obs: torch.Tensor):
        z = self.pre_encoder_layer(obs)
        mu = self.mean(z)
        # clamped for numerical stability
        std = torch.exp(self.log_std(z).clamp(-4, 15))
        dist = Normal(mu, std)
        raw_action = dist.rsample()
        return torch.tanh(raw_action), mu, raw_action

    def log_prob(self, obs, action=None, raw_action=None):
        z = self.pre_encoder_layer(obs)
        mu = self.mean(z)
        # clamped for numerical stability
        std = torch.exp(self.log_std(z).clamp(-4, 15))
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
        self.act_dim = act_dim
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

    def sample_action(self, obs):
        return self.decode_bc(obs)


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


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ContrastiveCostModel(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=[64, 64]):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [128]
        layers = list()
        for j in range(len(sizes) - 1):
            act = nn.ELU() if j < len(sizes) - 2 else nn.Identity()
            affine_layer = nn.Linear(sizes[j], sizes[j + 1])
            layers += [affine_layer, act]
        self.encoder = nn.Sequential(*layers)
        self.projection = nn.Linear(sizes[-1], 1)
        self.apply(orthogonal_init)

    def forward(self, obs):
        z = F.normalize(self.encoder(obs), dim=-1, p=2.0)
        cost = torch.sigmoid(self.projection(z))
        return z, cost

    def freeze_encoder(self):
        for params in self.encoder.parameters():
            params.requires_grad = False


class ExpCostModel(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=[64, 64]):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [1]
        layers = list()
        for j in range(len(sizes) - 1):
            act = nn.ELU() if j < len(sizes) - 2 else nn.Identity()
            affine_layer = nn.Linear(sizes[j], sizes[j + 1])
            layers += [affine_layer, act]
        self.model = nn.Sequential(*layers)
        self.apply(orthogonal_init)
        self.model[-2].weight.data.fill_(0)
        self.model[-2].bias.data.fill_(0)

    def forward(self, obs, use_sigmoid=False):
        raw_val = torch.squeeze(self.model(obs), -1)
        if use_sigmoid:
            return torch.sigmoid(raw_val)
        # clamped for numerical stability
        return torch.exp(raw_val.clamp(-4, 3))


class TdmpcCostModel(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=[64, 64]):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [1]
        layers = list()
        for j in range(len(sizes) - 1):
            act = nn.ELU() if j < len(sizes) - 2 else nn.Identity()
            affine_layer = nn.Linear(sizes[j], sizes[j + 1])
            layers += [affine_layer, act]
        self.model = nn.Sequential(*layers)
        self.apply(orthogonal_init)
        self.model[-2].weight.data.fill_(0)
        self.model[-2].bias.data.fill_(0)

    def forward(self, obs):
        return torch.squeeze(self.model(obs), -1)


class TdmpcValue(nn.Module):
    def __init__(self, obs_dim, hidden_size=512, last_act=nn.Identity()):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
            last_act,
        )
        self.apply(orthogonal_init)
        self.model[-2].weight.data.fill_(0)
        self.model[-2].bias.data.fill_(0)

    def forward(self, obs):
        return torch.squeeze(self.model(obs), -1)


class EnsembleValue(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=[64, 64], last_act=nn.Identity()):
        super().__init__()
        self._V1 = TdmpcValue(
            obs_dim=obs_dim, hidden_size=hidden_sizes[0], last_act=last_act
        )
        self._V2 = TdmpcValue(
            obs_dim=obs_dim, hidden_size=hidden_sizes[0], last_act=last_act
        )

    def V(self, obs):
        return self._V1(obs), self._V2(obs)


class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, enc_dim=256):
        super().__init__()
        # TDMPC encoder
        self.model = nn.Sequential(
            nn.Linear(obs_dim, enc_dim), nn.ELU(), nn.Linear(enc_dim, latent_dim)
        )
        self.apply(orthogonal_init)

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
        self.apply(orthogonal_init)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.model(x)


class ScaledDotProductAttention(nn.Module):
    """
    Code from: https://github.com/sooftware/attentions/blob/master/attentions.py
    """

    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float("Inf"))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class MultiHeadAttention(nn.Module):
    """
    Code from: https://github.com/sooftware/attentions/blob/master/attentions.py
    """

    """
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)

    Args:
        d_model (int): The dimension of keys / values / quries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): In transformer, three different ways:
            Case 1: come from previoys decoder layer
            Case 2: come from the input embedding
            Case 3: come from the output embedding (masked)

        - **key** (batch, k_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - **value** (batch, v_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - **mask** (-): tensor containing indices to be masked

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(
            batch_size, -1, self.num_heads, self.d_head
        )  # BxQ_LENxNxD
        key = self.key_proj(key).view(
            batch_size, -1, self.num_heads, self.d_head
        )  # BxK_LENxNxD
        value = self.value_proj(value).view(
            batch_size, -1, self.num_heads, self.d_head
        )  # BxV_LENxNxD

        query = (
            query.permute(2, 0, 1, 3)
            .contiguous()
            .view(batch_size * self.num_heads, -1, self.d_head)
        )  # BNxQ_LENxD
        key = (
            key.permute(2, 0, 1, 3)
            .contiguous()
            .view(batch_size * self.num_heads, -1, self.d_head)
        )  # BNxK_LENxD
        value = (
            value.permute(2, 0, 1, 3)
            .contiguous()
            .view(batch_size * self.num_heads, -1, self.d_head)
        )  # BNxV_LENxD

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = (
            context.permute(1, 2, 0, 3)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_head)
        )  # BxTxND

        return context, attn


def positionalencoding1d(d_model, length):
    """
    Code: https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    """
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask=None):  # x shape: horizon x batch x d_model
        x_norm = self.ln1(x)
        attn_x, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        # residual connection
        x = x + attn_x
        # residual connection
        x = x + self.ff(self.ln2(x))
        return x


class SafeTransformerCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        horizon,
        device,
        latent_dim=256,
        num_heads=4,
        num_attentions=1,
    ):
        super().__init__()
        self.encoder = nn.Linear(obs_dim + act_dim, latent_dim)
        self.pos_encoding = positionalencoding1d(latent_dim, horizon)
        self.pos_encoding = self.pos_encoding.unsqueeze(1).to(device)
        self.mask = torch.triu(
            torch.ones(horizon, horizon, dtype=torch.bool, device=device), diagonal=1
        )
        d_ff = 4 * latent_dim
        self.transformers = nn.ModuleList(
            [
                TransformerEncoderBlock(latent_dim, d_ff, num_heads)
                for _ in range(num_attentions)
            ]
        )
        self.cost_pred = nn.Linear(latent_dim, 1)

    def forward(self, x, use_sigmoid=True):  # shape x: horizon X batch X obs_act_dim
        x = self.encoder(x)
        x += self.pos_encoding
        for attn in self.transformers:
            x = attn(x, self.mask)
        x = x[-1]
        c = torch.squeeze(self.cost_pred(x), -1)
        if use_sigmoid:
            return torch.sigmoid(c)
        return c, x


class SafeDiceCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_size,
        out_activation_fn=None,
        use_last_layer_bias=False,
        out_dim=None,
    ):
        super().__init__()
        sizes = [obs_dim + act_dim] + [hidden_size, hidden_size]
        layers = list()
        for j in range(len(sizes) - 1):
            affine_layer = nn.Linear(sizes[j], sizes[j + 1])
            nn.init.kaiming_normal_(affine_layer.weight, nonlinearity="relu")
            layers += [affine_layer, nn.ReLU()]

        out_dim = out_dim or 1
        out_activation_fn = out_activation_fn or nn.Identity()
        if use_last_layer_bias:
            affine_layer = nn.Linear(hidden_size, out_dim)
            nn.init.uniform_(affine_layer.weight, -3e-3, 3e-3)
            nn.init.uniform_(affine_layer.bias, -3e-3, 3e-3)
            layers += [affine_layer, out_activation_fn]
        else:
            affine_layer = nn.Linear(hidden_size, out_dim, bias=False)
            nn.init.kaiming_normal_(affine_layer.weight, nonlinearity="relu")
            layers += [affine_layer, out_activation_fn]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.squeeze(self.model(x), -1)


def xavier_init(m):
    """Xavier layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def minmax_discriminator_loss(dx, dgz, label_smoothing=0.0):
    """code from torchgan"""
    target_ones = torch.ones_like(dgz) * (1.0 - label_smoothing)
    target_zeros = torch.zeros_like(dx)
    loss = F.binary_cross_entropy_with_logits(dx, target_ones)
    loss += F.binary_cross_entropy_with_logits(dgz, target_zeros)
    return loss


def horizon_gradient_panelty(interpolate, d_interpolate):
    "code from torchgan"
    horizon, batch_size, _ = interpolate.shape
    device = interpolate.device
    grad_output = torch.ones_like(d_interpolate, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolate,
        inputs=interpolate,
        grad_outputs=grad_output,
        retain_graph=True,
        create_graph=True,
    )[0]
    gradients = gradients.view(horizon, batch_size, -1)
    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=-1) + 1e-12)
    penalty = (gradients_norm - 1) ** 2
    return torch.mean(penalty)


def gradient_panelty(interpolate, d_interpolate):
    "code from torchgan"
    batch_size = interpolate.shape[0]
    device = interpolate.device
    grad_output = torch.ones_like(d_interpolate, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolate,
        inputs=interpolate,
        grad_outputs=grad_output,
        retain_graph=True,
        create_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
    penalty = (gradients_norm - 1) ** 2
    return torch.mean(penalty)


class SafeDiceTanhMixtureActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_size=256,
        num_components=2,
        mean_range=(-7.0, 7.0),
        logstd_range=(-5.0, 2.0),
        eps=EPS,
        mdn_temperature=1.0,
    ):
        super().__init__()

        self.act_dim = act_dim
        self.num_components = num_components
        self.mdn_temp = mdn_temperature

        self.pre_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # self.pre_encoder.apply(xavier_init)

        self.means = nn.Linear(hidden_size, num_components * act_dim)
        # nn.init.xavier_uniform_(self.means.weight)
        # nn.init.uniform_(self.means.bias, -1e-3, 1e-3)

        self.logstds = nn.Linear(hidden_size, num_components * act_dim)
        # nn.init.uniform_(self.logstds.weight, -1e-3, 1e-3)
        # nn.init.uniform_(self.logstds.bias, -1e-3, 1e-3)

        self.logits = nn.Linear(hidden_size, num_components)
        # nn.init.xavier_uniform_(self.logits.weight)
        # nn.init.uniform_(self.logits.bias, -1e-3, 1e-3)

        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def forward(self, obs):
        x = self.pre_encoder(obs)

        means = self.means(x).clamp(self.mean_min, self.mean_max)
        means = means.view(-1, self.num_components, self.act_dim)
        logstds = self.logstds(x).clamp(self.logstd_min, self.logstd_max)
        logstds = logstds.view(-1, self.num_components, self.act_dim)
        stds = torch.exp(logstds)
        mixture_logits = self.logits(x) / self.mdn_temp

        mixture_dist = td.Categorical(logits=mixture_logits)
        component_dist = td.Normal(means, stds)
        component_dist = td.Independent(component_dist, 1)
        pretanh_actions_dist = td.MixtureSameFamily(mixture_dist, component_dist)

        mixture_sample = F.gumbel_softmax(logits=mixture_logits, tau=1.0, hard=True)
        component_sample = component_dist.rsample()
        pretanh_actions = torch.einsum("ij,ijk->ik", mixture_sample, component_sample)
        actions = torch.tanh(pretanh_actions)

        logprob, pretanh_logprob = self.logprob(
            pretanh_actions_dist,
            pretanh_actions,
            is_pretanh_actions=True,
        )

        return (
            actions,
            pretanh_actions,
            logprob,
            pretanh_logprob,
            pretanh_actions_dist,
        )

    def logprob(self, pretanh_actions_dist, actions, is_pretanh_actions=True):
        if is_pretanh_actions:
            pretanh_actions = actions
            actions = torch.tanh(pretanh_actions)
        else:
            pretanh_actions = torch.atanh(actions.clamp(-1 + self.eps, 1 - self.eps))

        pretanh_logprob = pretanh_actions_dist.log_prob(pretanh_actions).sum(-1)
        logprob = pretanh_logprob - (1.0 - actions.pow(2)).clamp(
            min=self.eps
        ).log().sum(-1)
        return logprob, pretanh_logprob

    def get_logprob(self, obs, actions):
        """
        Args:
            obs: A batch of observations.
            actions: A batch of actions to evaluate log probs on.
        Returns:
            Log probabilities of actions.
        """
        x = self.pre_encoder(obs)

        means = self.means(x).clamp(self.mean_min, self.mean_max)
        means = means.view(-1, self.num_components, self.act_dim)
        logstds = self.logstds(x).clamp(self.logstd_min, self.logstd_max)
        logstds = logstds.view(-1, self.num_components, self.act_dim)
        stds = torch.exp(logstds)

        pretanh_actions_dist = td.Independent(td.Normal(means, stds), 1)
        pretanh_actions = torch.atanh(actions.clamp(-1 + self.eps, 1 - self.eps))
        pretanh_actions = torch.stack([pretanh_actions, pretanh_actions], dim=1)
        pretanh_log_prob = pretanh_actions_dist.log_prob(pretanh_actions)
        log_probs = torch.mean(pretanh_log_prob, dim=-1) - torch.sum(
            torch.log(1 - actions**2 + self.eps), dim=-1
        )
        return log_probs

    def true_get_logprob(self, obs, actions):
        """
        Args:
            obs: A batch of observations.
            actions: A batch of actions to evaluate log probs on.
        Returns:
            Log probabilities of actions.
        """
        x = self.pre_encoder(obs)

        means = self.means(x).clamp(self.mean_min, self.mean_max)
        means = means.view(-1, self.num_components, self.act_dim)
        logstds = self.logstds(x).clamp(self.logstd_min, self.logstd_max)
        logstds = logstds.view(-1, self.num_components, self.act_dim)
        stds = torch.exp(logstds)
        mixture_logits = self.logits(x) / self.mdn_temp

        mixture_dist = td.Categorical(logits=mixture_logits)
        component_dist = td.Normal(means, stds)
        component_dist = td.Independent(component_dist, 1)
        pretanh_actions_dist = td.MixtureSameFamily(mixture_dist, component_dist)

        actions = actions.clamp(-1 + self.eps, 1 - self.eps)
        pretanh_actions = torch.atanh(actions)
        pretanh_logprob = pretanh_actions_dist.log_prob(pretanh_actions)
        jacobian_det = torch.sum(torch.log(1 - actions**2 + self.eps), dim=1)
        # logprob = pretanh_logprob - (1.0 - actions.pow(2)).clamp(
        #     min=self.eps
        # ).log().sum(-1)
        return pretanh_logprob + jacobian_det

    def action(self, obs, deterministic=True):
        x = self.pre_encoder(obs)

        means = self.means(x).clamp(self.mean_min, self.mean_max)
        means = means.view(-1, self.num_components, self.act_dim)
        logstds = self.logstds(x).clamp(self.logstd_min, self.logstd_max)
        logstds = logstds.view(-1, self.num_components, self.act_dim)
        stds = torch.exp(logstds)
        mixture_logits = self.logits(x) / self.mdn_temp

        mixture_dist = td.Categorical(logits=mixture_logits)

        device = means.device

        if deterministic:
            mixture_id = mixture_dist.sample()
            idx = torch.vstack(
                [torch.arange(0, obs.shape[0], device=device), mixture_id]
            )
            pretanh_actions = means[idx.tolist()]
        else:
            component_dist = td.Normal(means, stds)
            component_dist = td.Independent(component_dist, 1)
            pretanh_actions_dist = td.MixtureSameFamily(mixture_dist, component_dist)
            pretanh_actions = pretanh_actions_dist.sample()

        return torch.tanh(pretanh_actions)

    def sample_action(self, obs):
        return self.action(obs, deterministic=False)


class DwbcDiscriminator(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.fc1_1 = nn.Linear(obs_dim + act_dim, 128)
        self.fc1_2 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, act, log_pi):
        sa = torch.cat([obs, act], 1)
        d1 = F.relu(self.fc1_1(sa))
        d2 = F.relu(self.fc1_2(log_pi))
        d = torch.cat([d1, d2], 1)
        d = F.relu(self.fc2(d))
        d = F.sigmoid(self.fc3(d))
        d = torch.clip(d, 0.1, 0.9)
        return d
