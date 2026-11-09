"""
DualCOIL: Offline Imitation Learning from Contrasting Demonstrations
(Hoang, Mai, Varakantham, Verma, ICML 2026), ported to flax.nnx.

Faithful re-implementation of the authors' official JAX/linen code
(https://github.com/hmhuy0/DualCOIL, method folder sources/algos/ContraDICE)
inside the PSIL experimental pipeline (data construction, observation normalisation,
policy network, logger, saved checkpoints).

Mapping to the authors' files
-----------------------------
networks/{critic,value,discriminator}.py -> DoubleCritic, ValueCritic, Discriminator
                                            (2 x 256 ReLU MLPs, orthogonal(sqrt 2) init,
                                             discriminator output sigmoid clipped to
                                             [0.05, 0.95])
algos/ContraDICE/disc.py                 -> disc_grad_fun:
                                            -E_high[log d] - E_low[log(1 - d)]
algos/ContraDICE/algo.py get_ratio       -> get_ratio (implicit reward of unlabeled
                                            transitions from both discriminators at s')
algos/ContraDICE/critic.py update_value  -> value_grad_fun (reverse-KL / Extreme-V loss)
algos/ContraDICE/critic.py update_critic -> critic_grad_fun (IQ-style loss with reward
                                            weights +r_scale (good), -r_scale*r_alpha
                                            (bad), ratio (unlabeled))
algos/ContraDICE/actor.py update_actor   -> actor_grad_fun (Q-weighted log-likelihood
                                            on the unlabeled data)
algos/ContraDICE/algo.py update order    -> value -> actor -> critic -> soft target update
train.py                                 -> discriminators trained first (num_disc_train
                                            steps each), then 1M joint updates
parse.py / configs/*.py                  -> dualcoil_cfg defaults

What is kept from PSIL
----------------------
* Data construction (get_dataset_in_d4rl_format / get_pos_neg_and_union_data /
  get_normalized_data) with the same seeding order, so D_P, D_N and D_U are identical
  to PSIL for the same seed and arguments. D_P -> good dataset, D_N -> bad dataset,
  D_U -> unlabeled (mixed) dataset.
* Policy network: SafeDiceTanhMixtureActor, trained with DualCOIL's own objective
  (Q-weighted log-likelihood through get_log_prob) and DualCOIL's actor optimiser.
  It is saved with the "bc_policy" prefix and the "norm" state, so your PSIL
  evaluation script can load it unchanged.
* Logger: EpochLogger (progress.csv for the joint training, disc_progress.csv for the
  discriminators).

Settings
--------
* LfP+N: DualCOIL (good, bad and unlabeled data).
* LfP:   DualCOIL-G of the paper (alpha = 0, no bad dataset, no bad discriminator).
* LfN:   not supported; DualCOIL needs expert (good) demonstrations.

Deviations (only where the DSRL pipeline differs from D4RL)
----------------------------------------------------------
* Transitions are built from the padded DSRL trajectories: (s_t, a_t, s_{t+1}) for the
  valid steps, mask = 1 - terminal_t; the final step of a time-out episode has no
  successor and is dropped (as in d4rl.qlearning_dataset). Actions are clipped to
  +-(1 - 1e-5) (D4RLDataset clip_to_eps).
* Observations are normalised by the PSIL pipeline (get_normalized_data, statistics of
  D_U), which replaces the authors' state_norm (same statistics up to 1e-3).
* The policy network is SafeDiceTanhMixtureActor instead of the authors'
  NormalTanhPolicy (Gaussian with tanh mean, state-independent std).
* Only the options used in the authors' runs are implemented: v_update='rkl_loss',
  exp_r=False (the exp_r branch reads an argument that does not exist in their code).
  adv_policy_extraction and learn_expert_data are available as flags.

Usage (LfP+N):
    python dualcoil_jax.py --task OfflinePointCircle2Gymnasium-v0 --num-preferred 1 \
        --num-non-preferred 10 --num-union 500 [--dualcoil-* flags]
DualCOIL-specific flags are prefixed with --dualcoil- and are removed from sys.argv
before single_agent_args() parses the shared PSIL arguments.
"""

import argparse
import os
import random
import re
import sys
import time
from copy import deepcopy

import dsrl.infos as dsrl_infos
import dsrl.offline_safety_gymnasium  # type: ignore
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx, struct

from dsrl_model.utils.dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_normalized_data,
    get_pos_neg_and_union_data,
)
from dsrl_model.utils.models_jax import SafeDiceTanhMixtureActor, get_tree_norm
from dsrl_model.utils.native_logger import EpochLogger
from dsrl_model.utils.utils import make_static_config_from_dict, single_agent_args

EPS = 1e-6

# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------

# Logging / checkpointing: same frequency as PSIL (also the authors' eval_interval).
default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "hidden_size": 256,  # policy (SafeDiceTanhMixtureActor), as in PSIL
    "total_iteration": int(1e6),  # authors' max_steps
}

# Data construction: identical to PSIL.
trajectory_cfg = {
    "density": 1.0,
    "target_cost": 25.0,
    # ((low_cost, low_reward), (high_cost, low_reward), (medium_cost, high_reward))
    "inpaint_ranges": None,
    "num_positive_trajectories": 0,
    "num_negative_trajectories": 50,
    "num_union_trajectories": -1,
}

trajectory_data = {
    "full": None,
    "reward_only": None,
    "cost_only": ((0.0, 1.0, 0.0, 0.5),),
}

# DualCOIL configuration.
#   parse.py defaults: batch_size 1024, lr 3e-4 (all networks), discount 0.99,
#       2 x 256 MLPs, double Q, v_update rkl_loss, max_clip 7, num_disc_train 1e5,
#       adv_policy_extraction False, learn_expert_data False.
#   configs/*.py (per D4RL task; DSRL is not among them): the most frequent values of
#       the released configs are used here: r_scale 3.0, tau 0.005, v_beta 20,
#       actor_Q_scale 2.0, r_alpha 0.6, weight_decay 0.01.
dualcoil_cfg = {
    "batch_size": 1024,  # per dataset (good / bad / unlabeled)
    "num_layers": 2,
    "mlp_hidden_size": 256,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "value_lr": 3e-4,
    "disc_lr": 3e-4,
    "discount": 0.99,
    "tau": 0.005,
    "weight_decay": 0.01,
    "actor_Q_scale": 2.0,
    "v_beta": 20.0,
    "r_scale": 3.0,
    "r_alpha": 0.6,
    "max_clip": 7.0,
    "double_q": True,
    "num_disc_train": int(1e5),
    "adv_policy_extraction": False,
    "learn_expert_data": False,
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes", "y", "t")


def dualcoil_args():
    """Parse DualCOIL-only flags and strip them from sys.argv."""
    parser = argparse.ArgumentParser(add_help=False)
    types = {bool: str2bool, int: int, float: float}
    for key, val in dualcoil_cfg.items():
        flag = "--dualcoil-" + key.replace("_", "-")
        parser.add_argument(flag, type=types[type(val)], default=None)
    parsed, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    cfg = dict(dualcoil_cfg)
    for key, val in vars(parsed).items():
        if val is not None:
            cfg[key[len("dualcoil_"):]] = val
    return cfg


# ----------------------------------------------------------------------------------
# Networks (authors' sources/networks and sources/utils/common.MLP)
# ----------------------------------------------------------------------------------


def default_init(scale=np.sqrt(2)):
    return nnx.initializers.orthogonal(scale)


class MLP(nnx.Module):
    """Dense layers with orthogonal(sqrt 2) kernels, zero biases, ReLU between layers
    and no activation after the final layer (activate_final=False)."""

    def __init__(self, in_dim, hidden_dims, out_dim, rngs):
        dims = [in_dim, *hidden_dims, out_dim]
        self.layers = tuple(
            [
                nnx.Linear(dims[i], dims[i + 1], kernel_init=default_init(), rngs=rngs)
                for i in range(len(dims) - 1)
            ]
        )

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers):
                x = nnx.relu(x)
        return x


class DoubleCritic(nnx.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims, rngs):
        self.q1 = MLP(obs_dim + act_dim, hidden_dims, 1, rngs)
        self.q2 = MLP(obs_dim + act_dim, hidden_dims, 1, rngs)

    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)
        return jnp.squeeze(self.q1(x), -1), jnp.squeeze(self.q2(x), -1)


class ValueCritic(nnx.Module):
    def __init__(self, obs_dim, hidden_dims, rngs):
        self.v = MLP(obs_dim, hidden_dims, 1, rngs)

    def __call__(self, obs):
        return jnp.squeeze(self.v(obs), -1)


class Discriminator(nnx.Module):
    """Discriminator_state_only: sigmoid output clipped to [0.05, 0.95]."""

    def __init__(self, obs_dim, hidden_dims, rngs):
        self.d = MLP(obs_dim, hidden_dims, 1, rngs)

    def __call__(self, obs):
        return jnp.squeeze(jax.nn.sigmoid(self.d(obs)).clip(min=0.05, max=0.95), -1)


class Models(nnx.Module):
    def __init__(self, actor, critic, target_critic, value, good_disc, bad_disc):
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.value = value
        self.good_disc = good_disc
        self.bad_disc = bad_disc


class Optimizers(nnx.Module):
    def __init__(self, actor, critic, value, good_disc, bad_disc):
        self.actor = actor
        self.critic = critic
        self.value = value
        self.good_disc = good_disc
        self.bad_disc = bad_disc


class TrainingState(nnx.Module):
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers


# ----------------------------------------------------------------------------------
# Datasets of transitions
# ----------------------------------------------------------------------------------


@struct.dataclass
class TransitionData:
    observations: jnp.ndarray
    actions: jnp.ndarray
    next_observations: jnp.ndarray
    masks: jnp.ndarray

    def sample(self, key, batch_size):
        # jax_d4rl_dataset.sample (observations are already normalised)
        indx = jax.random.randint(
            key=key, minval=0, maxval=self.observations.shape[0], shape=(batch_size,)
        )
        return TransitionData(
            observations=self.observations[indx],
            actions=self.actions[indx],
            next_observations=self.next_observations[indx],
            masks=self.masks[indx],
        )


@struct.dataclass
class AllData:
    good: TransitionData
    bad: TransitionData
    mix: TransitionData


def get_transitions(data, ep_len, eps=1e-5):
    """(s_t, a_t, s_{t+1}, 1 - terminal_t) for the valid steps of padded trajectories.

    Returns the transitions and, for logging, the trajectory index of each transition.
    """
    obs = np.asarray(data["observations"], dtype=np.float32)
    act = np.clip(np.asarray(data["actions"], dtype=np.float32), -1 + eps, 1 - eps)
    terminals = np.asarray(data["terminals"]).astype(bool)
    dones = terminals | np.asarray(data["timeouts"]).astype(bool)

    done_sum = dones.sum(axis=1)
    done_sum = np.where(done_sum == 0, 1, done_sum)
    true_len = ep_len - done_sum + 1  # same rule as OnPolicyBuffer.add
    t = np.arange(ep_len)[None, :]
    last = true_len[:, None] - 1
    # every step with a successor + the final step if the episode terminated
    include = (t < last) | ((t == last) & terminals)

    next_obs = np.concatenate([obs[:, 1:], obs[:, -1:]], axis=1)
    masks = 1.0 - terminals.astype(np.float32)
    traj_idx = np.broadcast_to(np.arange(obs.shape[0])[:, None], include.shape)

    trans = TransitionData(
        observations=jnp.asarray(obs[include]),
        actions=jnp.asarray(act[include]),
        next_observations=jnp.asarray(next_obs[include]),
        masks=jnp.asarray(masks[include]),
    )
    return trans, traj_idx[include]


def empty_transitions(obs_dim, act_dim):
    # placeholder for the missing bad dataset in LfP (never sampled)
    return TransitionData(
        observations=jnp.zeros((1, obs_dim)),
        actions=jnp.zeros((1, act_dim)),
        next_observations=jnp.zeros((1, obs_dim)),
        masks=jnp.ones((1,)),
    )


# ----------------------------------------------------------------------------------
# DualCOIL losses (authors' algos/ContraDICE)
# ----------------------------------------------------------------------------------


@struct.dataclass
class DiscAux:
    loss: float = 0.0
    d_high: float = 0.0
    d_low: float = 0.0


@struct.dataclass
class TrainAux:
    value_loss: float = 0.0
    value: float = 0.0
    q: float = 0.0
    critic_loss: float = 0.0
    actor_loss: float = 0.0
    actor_weight: float = 0.0
    actor_good_logp: float = 0.0
    actor_mixed_logp: float = 0.0
    mix_ratio: float = 0.0
    entropy: float = 0.0


def disc_grad_fun(disc, high_obs, low_obs):
    def loss_fun(disc):
        d_high = disc(high_obs)
        d_low = disc(low_obs)
        loss = -jnp.log(d_high).mean() - jnp.log(1 - d_low).mean()
        return loss, DiscAux(loss=loss, d_high=d_high.mean(), d_low=d_low.mean())

    (_, aux), grads = nnx.value_and_grad(loss_fun, has_aux=True)(disc)
    return grads, aux


def get_ratio(good_disc, bad_disc, observations, config):
    good_out = config.r_scale * good_disc(observations)
    if config.has_bad:
        bad_out = config.r_scale * bad_disc(observations)
        frac = (1 - config.r_alpha) if config.r_alpha < 1 else 1
        out = jnp.exp((jnp.log(good_out) - config.r_alpha * jnp.log(bad_out)) / frac)
    else:
        # DualCOIL-G (alpha = 0)
        out = jnp.exp(jnp.log(good_out))
    return jax.lax.stop_gradient(
        out.clip(max=config.good_reward_coeff, min=config.bad_reward_coeff)
    )


def reverse_kl_loss(diff, v_beta, max_clip):
    z = diff / v_beta
    if max_clip is not None:
        z = jnp.minimum(z, max_clip)  # clip max value
    max_z = jnp.max(z, axis=0)
    max_z = jnp.where(max_z < -1.0, -1.0, max_z)
    max_z = jax.lax.stop_gradient(max_z)  # Detach the gradients
    loss = jnp.exp(z - max_z) - z * jnp.exp(-max_z) - jnp.exp(-max_z)  # scale by e^max_z
    return loss


def value_grad_fun(target_critic, value, obs, acts, config):
    q1, q2 = target_critic(obs, acts)
    q = jnp.minimum(q1, q2) if config.double_q else q1

    def loss_fun(value):
        v = value(obs)
        loss = reverse_kl_loss(
            jax.lax.stop_gradient(q) - v, config.v_beta, config.max_clip
        ).mean()
        return loss, (loss, v.mean(), q.mean())

    (_, aux), grads = nnx.value_and_grad(loss_fun, has_aux=True)(value)
    return grads, aux


def actor_grad_fun(actor, target_critic, t_v, good_obs, good_act, mix_obs, mix_act, config):
    q1, q2 = target_critic(mix_obs, mix_act)
    q = jnp.minimum(q1, q2) if config.double_q else q1
    q = jax.lax.stop_gradient(q)
    if not config.adv_policy_extraction:
        weight = jnp.exp(config.actor_temperature_Q * q).clip(min=0, max=100)
    else:
        adv = (q - t_v).clip(max=7.0)
        weight = jnp.exp(adv * 3).clip(min=0, max=100)
    weight = jax.lax.stop_gradient(weight)

    obs = jnp.concatenate([good_obs, mix_obs], axis=0)
    act = jnp.concatenate([good_act, mix_act], axis=0)
    num_good = good_obs.shape[0]

    def loss_fun(actor):
        log_probs = actor.get_log_prob(obs, act)
        expert_log_probs = log_probs[:num_good]
        mixed_log_probs = log_probs[num_good:]
        loss = -(weight * mixed_log_probs).mean()
        if config.learn_expert_data:
            loss += -(100 * expert_log_probs.mean())
        return loss, (loss, weight.mean(), expert_log_probs.mean(), mixed_log_probs.mean())

    (_, aux), grads = nnx.value_and_grad(loss_fun, has_aux=True)(actor)
    return grads, aux


def critic_grad_fun(critic, value, obs, acts, next_obs, masks, reward_weight, config):
    lo = config.bad_reward_coeff / (1 - config.discount)
    hi = config.good_reward_coeff / (1 - config.discount)
    next_v = value(next_obs)
    next_v = config.discount * masks * next_v.clip(max=hi, min=lo)
    next_v = jax.lax.stop_gradient(next_v)

    def iq_loss(q):
        reward = q - next_v
        transformed_reward = -reward  # exp_r = False
        reward_loss = (reward_weight * transformed_reward).mean()
        regularizer_loss = 0.5 * (reward**2).mean()
        return reward_loss + regularizer_loss

    def loss_fun(critic):
        q1, q2 = critic(obs, acts)
        if config.double_q:
            loss = (iq_loss(q1) + iq_loss(q2)).mean()
        else:
            loss = iq_loss(q1)
        return loss, loss

    (_, loss), grads = nnx.value_and_grad(loss_fun, has_aux=True)(critic)
    return grads, loss


@nnx.jit
def soft_target_update(target_model, curr_model, tau):
    target_param = nnx.state(target_model, nnx.Param)
    curr_param = nnx.state(curr_model, nnx.Param)
    new_target_param = jax.tree_util.tree_map(
        lambda t, c: c * tau + t * (1 - tau), target_param, curr_param
    )
    nnx.update(target_model, new_target_param)
    return target_model


def dualcoil_step(state, data, key, config):
    """One update of _update_ContraDICE (value -> actor -> critic -> target)."""
    m, o = state.models, state.optimizers
    batch_size = config.batch_size

    # the authors draw the three batches with the same key
    good = data.good.sample(key, batch_size)
    mix = data.mix.sample(key, batch_size)
    mix_ratio = get_ratio(m.good_disc, m.bad_disc, mix.next_observations, config)

    parts = [good, mix]
    weights = [config.good_reward_coeff * jnp.ones(batch_size), mix_ratio]
    if config.has_bad:
        bad = data.bad.sample(key, batch_size)
        parts = [bad] + parts
        weights = [config.bad_reward_coeff * jnp.ones(batch_size)] + weights

    obs = jnp.concatenate([p.observations for p in parts], axis=0)
    acts = jnp.concatenate([p.actions for p in parts], axis=0)
    next_obs = jnp.concatenate([p.next_observations for p in parts], axis=0)
    masks = jnp.concatenate([p.masks for p in parts], axis=0)
    reward_weight = jnp.concatenate(weights, axis=0)

    # t_V with the value network before its update (used by adv_policy_extraction)
    lo = config.bad_reward_coeff / (1 - config.discount)
    hi = config.good_reward_coeff / (1 - config.discount)
    t_v = jax.lax.stop_gradient(m.value(mix.observations)).clip(max=hi, min=lo)

    value_grads, (value_loss, v_mean, q_mean) = value_grad_fun(
        m.target_critic, m.value, obs, acts, config
    )
    o.value.update(value_grads)

    actor_grads, (actor_loss, actor_weight, good_logp, mixed_logp) = actor_grad_fun(
        m.actor,
        m.target_critic,
        t_v,
        good.observations,
        good.actions,
        mix.observations,
        mix.actions,
        config,
    )
    o.actor.update(actor_grads)

    critic_grads, critic_loss = critic_grad_fun(
        m.critic, m.value, obs, acts, next_obs, masks, reward_weight, config
    )
    o.critic.update(critic_grads)

    soft_target_update(m.target_critic, m.critic, config.tau)

    return TrainAux(
        value_loss=value_loss,
        value=v_mean,
        q=q_mean,
        critic_loss=critic_loss,
        actor_loss=actor_loss,
        actor_weight=actor_weight,
        actor_good_logp=good_logp,
        actor_mixed_logp=mixed_logp,
        mix_ratio=mix_ratio.mean(),
        entropy=-mixed_logp,
    )


@nnx.jit
def train_n_steps(state, data, config, key):
    def body_fun(i, carry):
        state, key, _ = carry
        key, subkey = jax.random.split(key)
        aux = dualcoil_step(state, data, subkey, config)
        return state, key, aux

    _, _, aux = nnx.fori_loop(0, config.log_freq, body_fun, (state, key, TrainAux()))
    return aux


@nnx.jit
def train_disc_n_steps(disc, optimizer, high, low, config, key):
    # ContraDICE.update_disc: two keys per step (high / low batch)
    def body_fun(i, carry):
        disc, optimizer, key, _ = carry
        key1, key2, key = jax.random.split(key, 3)
        high_batch = high.sample(key1, config.batch_size)
        low_batch = low.sample(key2, config.batch_size)
        grads, aux = disc_grad_fun(disc, high_batch.observations, low_batch.observations)
        optimizer.update(grads)
        return disc, optimizer, key, aux

    _, _, _, aux = nnx.fori_loop(
        0, config.log_freq, body_fun, (disc, optimizer, key, DiscAux())
    )
    return aux


@nnx.jit
def batched_ratio(good_disc, bad_disc, observations, config):
    return get_ratio(good_disc, bad_disc, observations, config)


def chunked_ratio(state, observations, config, chunk_size=65536):
    out = []
    for start in range(0, observations.shape[0], chunk_size):
        out.append(
            np.asarray(
                batched_ratio(
                    state.models.good_disc,
                    state.models.bad_disc,
                    observations[start : start + chunk_size],
                    config,
                )
            )
        )
    return np.concatenate(out)


def plot_trajectory_ratio(costs, rewards, ratio, save_plot):
    fig, ax = plt.subplots(figsize=(2.5, 1.6))
    sc = ax.scatter(
        costs, rewards, c=ratio, cmap="Blues", s=25, edgecolors="black", linewidths=0.5
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("ratio", fontsize=10)
    ax.set_xlabel("Traj. Cost", fontsize=10)
    ax.set_ylabel("Traj. Reward", fontsize=10)
    fig.savefig(save_plot, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def safe_mean(x):
    x = np.asarray(x)
    return float(x.mean()) if x.size > 0 else 0.0


# ----------------------------------------------------------------------------------
# Utilities shared with PSIL
# ----------------------------------------------------------------------------------


def evaluate_bc_policy(eval_env, bc_policy, mu_obs, std_obs):
    eval_done = False
    eval_obs, _ = eval_env.reset()
    eval_obs = (eval_obs - mu_obs) / (std_obs + EPS)
    eval_obs = jnp.expand_dims(jnp.array(eval_obs, dtype=jnp.float32), axis=0)
    eval_reward, eval_cost, eval_len = 0.0, 0.0, 0.0
    while not eval_done:
        act = bc_policy(eval_obs)
        next_obs, reward, terminated, truncated, info = eval_env.step(
            np.array(act[0].squeeze())
        )
        cost = info["cost"]
        next_obs = (next_obs - mu_obs) / (std_obs + EPS)
        next_obs = jnp.expand_dims(jnp.array(next_obs, dtype=jnp.float32), axis=0)
        eval_obs = next_obs
        eval_reward += reward
        eval_cost += cost
        eval_len += 1
        eval_done = terminated or truncated
    return eval_reward, eval_cost, eval_len


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------


def main(args, dc_config, cfg_env=None):
    # set the random seed, device and number of threads (same order as PSIL)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rngs = nnx.Rngs(
        default=args.seed,
        params=args.seed + 3,
        dropout=args.seed + 5,
        random_sample=args.seed + 7,
    )
    # separate streams so the critic / discriminators do not change the policy
    # initialisation
    critic_rngs = nnx.Rngs(params=args.seed + 11)
    disc_rngs = nnx.Rngs(params=args.seed + 13)

    # set default device id
    jax.default_device = jax.devices(args.device)[args.device_id]

    has_positive = args.num_preferred > 0
    has_negative = args.num_non_preferred > 0
    assert has_positive, (
        "DualCOIL needs preferred (good) demonstrations: use LfP+N, or LfP for "
        "DualCOIL-G. The LfN setting is not supported by the method."
    )
    if not has_negative and dc_config["r_alpha"] != 0.0:
        dc_config["r_alpha"] = 0.0  # DualCOIL-G

    trajectory_cfg["num_positive_trajectories"] = args.num_preferred
    trajectory_cfg["num_negative_trajectories"] = args.num_non_preferred
    trajectory_cfg["num_union_trajectories"] = args.num_union
    trajectory_cfg["non_pref_noise"] = args.non_pref_noise
    trajectory_cfg["data_inpaint"] = args.data_inpaint
    trajectory_cfg["inpaint_ranges"] = trajectory_data[args.data_inpaint]

    config = {**default_cfg, **trajectory_cfg}
    config["dualcoil"] = dict(dc_config)
    config["normalize_observation"] = args.normalize_observation

    # env name
    env_name = re.search(r"Offline(.*?)(?:Gymnasium)?-v[0-9]", args.task).group(1)
    config["env_name"] = env_name

    # derived values (train.py)
    actor_temperature_Q = dc_config["actor_Q_scale"] / dc_config["v_beta"]
    good_reward_coeff = dc_config["r_scale"]
    bad_reward_coeff = -dc_config["r_scale"] * dc_config["r_alpha"]

    config_data = make_static_config_from_dict(
        name="State",
        d={
            "log_freq": config["log_freq"],
            "batch_size": dc_config["batch_size"],
            "discount": dc_config["discount"],
            "tau": dc_config["tau"],
            "v_beta": dc_config["v_beta"],
            "max_clip": dc_config["max_clip"],
            "r_scale": dc_config["r_scale"],
            "r_alpha": dc_config["r_alpha"],
            "good_reward_coeff": good_reward_coeff,
            "bad_reward_coeff": bad_reward_coeff,
            "actor_temperature_Q": actor_temperature_Q,
            "double_q": dc_config["double_q"],
            "adv_policy_extraction": dc_config["adv_policy_extraction"],
            "learn_expert_data": dc_config["learn_expert_data"],
            "has_bad": has_negative,
        },
    )()

    # evaluation environment
    eval_env = gym.make(args.task)
    eval_env.set_target_cost(config["target_cost"])
    eval_env.reset(seed=args.seed)

    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    hidden_dims = tuple([dc_config["mlp_hidden_size"]] * dc_config["num_layers"])

    # policy network (identical to PSIL), DualCOIL actor optimiser
    policy_model = SafeDiceTanhMixtureActor(
        rngs=rngs,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=config["hidden_size"],
    )
    actor_optimizer = nnx.Optimizer(
        model=policy_model,
        tx=optax.chain(
            optax.add_decayed_weights(dc_config["weight_decay"]),
            optax.adam(learning_rate=dc_config["actor_lr"]),
        ),
    )

    critic = DoubleCritic(obs_dim, act_dim, hidden_dims, critic_rngs)
    target_critic = deepcopy(critic)  # same initialisation as the critic
    value = ValueCritic(obs_dim, hidden_dims, critic_rngs)
    good_disc = Discriminator(obs_dim, hidden_dims, disc_rngs)
    bad_disc = Discriminator(obs_dim, hidden_dims, disc_rngs)

    def disc_tx():
        return optax.chain(
            optax.add_decayed_weights(dc_config["weight_decay"]),
            optax.adam(learning_rate=dc_config["disc_lr"]),
        )

    state = TrainingState(
        models=Models(
            actor=policy_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            good_disc=good_disc,
            bad_disc=bad_disc,
        ),
        optimizers=Optimizers(
            actor=actor_optimizer,
            # optax.adamw default weight decay (1e-4), as in the authors' code
            critic=nnx.Optimizer(model=critic, tx=optax.adamw(dc_config["critic_lr"])),
            value=nnx.Optimizer(model=value, tx=optax.adamw(dc_config["value_lr"])),
            good_disc=nnx.Optimizer(model=good_disc, tx=disc_tx()),
            bad_disc=nnx.Optimizer(model=bad_disc, tx=disc_tx()),
        ),
    )

    # data (identical to PSIL)
    ep_len = dsrl_infos.DEFAULT_MAX_EPISODE_STEPS[env_name]
    data = get_dataset_in_d4rl_format(
        eval_env, trajectory_cfg, args.task, ep_len, 1
    )
    pos_data, neg_data, union_data = get_pos_neg_and_union_data(
        data, trajectory_cfg, save_dir=args.log_dir, seed=args.seed
    )
    mu_obs, std_obs = 0.0, 1.0
    if config["normalize_observation"]:
        pos_data, neg_data, union_data, mu_obs, std_obs = get_normalized_data(
            pos_data, neg_data, union_data
        )
    assert (
        union_data["observations"].shape[1] == ep_len
    ), f"{union_data['observations'].shape[1]} episode length is different from {ep_len}"

    good_trans, _ = get_transitions(pos_data, ep_len)
    mix_trans, mix_traj_idx = get_transitions(union_data, ep_len)
    if has_negative:
        bad_trans, _ = get_transitions(neg_data, ep_len)
    else:
        bad_trans = empty_transitions(obs_dim, act_dim)
    all_data = AllData(good=good_trans, bad=bad_trans, mix=mix_trans)

    # set logger
    dict_args = config
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)

    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    disc_logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
        output_fname="disc_progress.csv",
        use_tensorboard=False,
    )
    logger.save_config(dict_args)
    logger.log(
        f"DualCOIL{'' if has_negative else '-G'}: good={good_trans.observations.shape[0]}, "
        f"bad={bad_trans.observations.shape[0] if has_negative else 0}, "
        f"unlabeled={mix_trans.observations.shape[0]} transitions"
    )

    # analysis only: unlabeled trajectories split by the cost threshold
    traj_costs = np.asarray(union_data["costs"]).sum(1)
    traj_rewards = np.asarray(union_data["rewards"]).sum(1)
    low_cost = traj_costs[mix_traj_idx] <= config["target_cost"]

    # ------------------------------------------------------------------------------
    # Step 1: discriminators (train.py: trained from scratch before the main loop)
    # ------------------------------------------------------------------------------
    logger.log("Start discriminator training.")
    num_chunks = max(dc_config["num_disc_train"] // config["log_freq"], 1)
    for chunk in range(num_chunks):
        good_aux = train_disc_n_steps(
            state.models.good_disc,
            state.optimizers.good_disc,
            all_data.good,
            all_data.mix,
            config_data,
            rngs.random_sample(),
        )
        disc_logger.log_tabular("Disc/Steps", (chunk + 1) * config["log_freq"])
        disc_logger.log_tabular("Disc/good_loss", good_aux.loss.item())
        disc_logger.log_tabular("Disc/good_on_good", good_aux.d_high.item())
        disc_logger.log_tabular("Disc/good_on_unlabeled", good_aux.d_low.item())
        if has_negative:
            bad_aux = train_disc_n_steps(
                state.models.bad_disc,
                state.optimizers.bad_disc,
                all_data.bad,
                all_data.mix,
                config_data,
                rngs.random_sample(),
            )
            disc_logger.log_tabular("Disc/bad_loss", bad_aux.loss.item())
            disc_logger.log_tabular("Disc/bad_on_bad", bad_aux.d_high.item())
            disc_logger.log_tabular("Disc/bad_on_unlabeled", bad_aux.d_low.item())

        ratio = chunked_ratio(state, all_data.mix.next_observations, config_data)
        disc_logger.log_tabular("Ratio/unlabeled", float(ratio.mean()))
        disc_logger.log_tabular("Ratio/unlabeled_low_cost", safe_mean(ratio[low_cost]))
        disc_logger.log_tabular("Ratio/unlabeled_high_cost", safe_mean(ratio[~low_cost]))
        disc_logger.dump_tabular()
    disc_logger.close()

    logger.nn_model_save(nn_model_saver_element=state.models.good_disc, prefix="good_disc")
    if has_negative:
        logger.nn_model_save(nn_model_saver_element=state.models.bad_disc, prefix="bad_disc")

    traj_ratio = np.zeros(traj_costs.shape[0])
    np.add.at(traj_ratio, mix_traj_idx, ratio)
    traj_ratio /= np.maximum(np.bincount(mix_traj_idx, minlength=traj_costs.shape[0]), 1)
    np.save(f"{args.log_dir}/dualcoil_traj_ratio_{args.seed}.npy", traj_ratio)
    plot_trajectory_ratio(
        traj_costs,
        traj_rewards,
        traj_ratio,
        save_plot=f"{args.log_dir}/ratio_dataset_{args.seed}.svg",
    )

    # ------------------------------------------------------------------------------
    # Step 2: Q, V and policy
    # ------------------------------------------------------------------------------
    logger.log("Start Q, V and bc_policy training.")
    steps = 0
    while steps < config["total_iteration"]:
        aux = train_n_steps(
            state=state,
            data=all_data,
            config=config_data,
            key=rngs.random_sample(),
        )
        steps += config["log_freq"]

        logger.logged = False
        logger.log_tabular("Train/Steps", steps)
        logger.log_tabular("Loss/value", aux.value_loss.item())
        logger.log_tabular("Loss/critic", aux.critic_loss.item())
        logger.log_tabular("Loss/actor", aux.actor_loss.item())
        logger.log_tabular("Mean/value", aux.value.item())
        logger.log_tabular("Mean/q", aux.q.item())
        logger.log_tabular("Mean/mix_ratio", aux.mix_ratio.item())
        logger.log_tabular("Mean/actor_weight", aux.actor_weight.item())
        logger.log_tabular("Mean/actor_good_logp", aux.actor_good_logp.item())
        logger.log_tabular("Mean/actor_mixed_logp", aux.actor_mixed_logp.item())
        logger.log_tabular(
            "Norm/policy_model",
            get_tree_norm(nnx.state(policy_model, nnx.Param)),
        )
        logger.dump_tabular()

        if steps % config["save_freq"] == 0:
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=policy_model,
                prefix="bc_policy",
            )

        if steps >= config["total_iteration"]:
            break

    logger.nn_model_save(
        itr=steps, nn_model_saver_element=policy_model, prefix="bc_policy"
    )
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=state.models.critic, prefix="critic"
    )
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=state.models.value, prefix="value"
    )
    if config["normalize_observation"]:
        logger.save_state(
            state_dict={"mu_obs": mu_obs, "std_obs": std_obs}, dirname="norm"
        )

    logger.close()


if __name__ == "__main__":
    dc_config = dualcoil_args()
    args, cfg_env = single_agent_args()
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    if not args.write_terminal:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(
                f"{args.log_dir}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{args.log_dir}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                main(args, dc_config, cfg_env)
    else:
        main(args, dc_config, cfg_env)
