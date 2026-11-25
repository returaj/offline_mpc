"""
PUBC: Positive Unlabeled Behavioural Cloning (Wang et al., CoRL 2023), ported to JAX.

Faithful re-implementation of the authors' official code
(https://github.com/RedmondLabUCD/Positive-Unlabeled-Behavioural-Cloning)
inside the PSIL experimental pipeline, so the only difference between PSIL and
PUBC is how the unlabeled trajectories are selected / weighted.

Mapping to the authors' files
-----------------------------
models.py   FilterNet                 -> FilterNet (same widths, BatchNorm, softmax head,
                                          PyTorch default Linear initialisation)
trainer.py  PUTrainer.reset/train     -> fresh ensemble + Adam(lr=1e-3, L2 wd=3e-3) every
                                          iteration, BCELoss on softmax outputs,
                                          `epochs_per_iteration` passes over each member's
                                          positive subset, batch size 1024
buffer.py   TorchDatasetHandler       -> PUSampler (50% positive / 50% synthetic negative,
                                          'part' and 'full' negative samplers, including
                                          the authors' sequential-draw logic for 'full')
buffer.py   set_seed_positive         -> random 95% subset of seed transitions per member
buffer.py   classifier_validate       -> per-trajectory mean positive probability,
                                          adaptive threshold, ensemble 'vote' / 'avg'
buffer.py   update_pos                -> relabelled positives split into K contiguous
                                          chunks of floor(n / K) transitions (in data order)
utils.py    adap_probs                -> adap_probs (histogram + polyfit local minimum)
main.py     main_filter               -> run_pubc_filter + BC on the filtered trajectories

What is kept from PSIL (for a controlled comparison)
----------------------------------------------------
* Data construction: get_dataset_in_d4rl_format / get_pos_neg_and_union_data /
  get_normalized_data with the same seeding order, so D_P and D_U are identical to PSIL
  for the same seed and arguments.
* Policy: SafeDiceTanhMixtureActor, AdamW + global-norm clipping, horizon-H windows,
  discounted L2 BC loss (gamma), same batch size, learning rate and 1M gradient steps.
  PUBC trains this policy only on the trajectories the filter labels as positive
  (uniform weight), exactly like the authors' BC on the filtered dataset.
* Buffer: OnPolicyBuffer from dsrl_model.utils.buffer_jax. Filtering is applied by
  zeroing the sampling priorities of rejected trajectories.
* Logger: EpochLogger (progress.csv for policy learning, filter_progress.csv for the
  PU filter iterations).

Saved artifacts (only what is actually used)
--------------------------------------------
* Filter: only the ensemble of the LAST iteration (the one whose selection is used for
  BC) is saved, together with its trajectory probabilities, the adaptive-threshold plot
  and the selected-trajectory plot. Intermediate iterations are only logged in
  filter_progress.csv.
* pubc_selected_<seed>.npy: the trajectory mask actually used to train the policy
  (after any fallback to the seed positives).
* Policy: only the final bc_policy is saved. Periodic checkpoints can be re-enabled
  with --pubc-save-freq N (N > 0).

Deviations (only where the original code would crash or cannot apply)
---------------------------------------------------------------------
* DSRL trajectories are padded to a fixed episode length; padded steps are masked out
  everywhere (seed subsets, negative sampling, per-trajectory probabilities).
* The authors draw the seed positives from the raw dataset, so seeds are filtered
  together with the rest. Here D_P is held out of D_U by the PSIL data pipeline, so by
  default (--pubc-seed-in-raw=True) D_P is concatenated to D_U to form the "raw"
  dataset that is filtered and cloned. Coverage statistics are reported on D_U only.
* adap_probs raises IndexError in the original code when the fitted polynomial has no
  local minimum; here it falls back to the high bound (th_high_bound) and logs a warning.
* If the filter selects too few transitions to give every ensemble member data (or no
  trajectory at all for BC), the original code crashes; here it falls back to the seed
  positives and logs a warning.
* Ground-truth expert labels do not exist in DSRL, so instead of classification accuracy
  the filter logs the reward/cost of selected vs. rejected trajectories.

Usage (PUBC uses only preferred demonstrations, i.e. the LfP setting):
    python pubc_jax.py --task OfflinePointCircle2Gymnasium-v0 --num-preferred 1 \
        --num-non-preferred 0 --num-union 500 [--pubc-iterations 2 ...]
PUBC-specific flags are prefixed with --pubc- and are removed from sys.argv before
single_agent_args() parses the shared PSIL arguments.
"""

import argparse
import os
import random
import re
import sys
import time

import dsrl.infos as dsrl_infos
import dsrl.offline_safety_gymnasium  # type: ignore
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import scipy.signal as sg
from flax import nnx, struct

from dsrl_model.utils.buffer_jax import OnPolicyBuffer
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

# Policy-learning configuration: identical to PSIL so that the policy is the same.
default_cfg = {
    "log_freq": int(1e4),
    # 0: save only the final policy. N > 0: also save a checkpoint every N steps.
    "save_freq": int(2e4),
    "hidden_size": 256,
    "max_grad_norm": 5.0,
    "action_repeat": 1,
    "weight_decay": 0.01,
    "total_iteration": int(1e6),
    # Policy learning: values of the PSIL paper (Appendix B and C).
    "train_horizon": 50,
    "normalize_observation": True,
    "lr": 3e-4,
    "gamma": 0.9,  # for trajectory based policy learning
    "batch_size": 64,
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

# PU filter configuration: defaults of the authors' main.py.
# (The paper reports 3 iterations for MuJoCo and 4 for TriFinger, and describes the
#  seven-type negative sampler, which corresponds to --pubc-negative-sampler full.)
pubc_cfg = {
    "models_in_ensemble": 3,
    "ensemble_method": "vote",  # 'vote' or 'avg'
    "iterations": 2,
    "negative_sampler": "part",  # 'part' or 'full'
    "epochs_per_iteration": 20,
    "pos_seed_subset": 0.95,  # buffer.sub_seed_positive
    "th_bins": 100,
    "th_high_bound": 0.96,
    "th_fit_pow": 10,
    "filter_batch_size": 1024,
    "filter_lr": 1e-3,
    "filter_weight_decay": 3e-3,  # torch.optim.Adam(weight_decay=3e-3): coupled L2
    "seed_in_raw": True,
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes", "y", "t")


def parse_prefixed_flags(prefix, cfg_dicts):
    """Create one --<prefix>-<key> flag for every entry of the given config dicts,
    update the dicts in place with the values passed on the command line, and strip
    these flags from sys.argv before single_agent_args() parses the PSIL arguments.

    The dicts in this file are the single source of truth for every hyperparameter.
    """
    parser = argparse.ArgumentParser(add_help=False)
    flag_map = {}
    for cfg in cfg_dicts:
        for key, val in cfg.items():
            if isinstance(val, bool):
                ftype = str2bool
            elif isinstance(val, int):
                ftype = int
            elif isinstance(val, float) or val is None:
                ftype = float
            elif isinstance(val, str):
                ftype = str
            else:
                continue  # tuples / None-only entries are not exposed
            dest = f"{prefix}_{key}"
            parser.add_argument(
                f"--{prefix}-{key.replace('_', '-')}",
                dest=dest,
                type=ftype,
                default=None,
            )
            flag_map[dest] = (cfg, key)
    parsed, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    for dest, val in vars(parsed).items():
        if val is not None:
            cfg, key = flag_map[dest]
            cfg[key] = val


# PSIL command-line arguments that are NOT used by this baseline: the values come
# from the dicts above. A warning is logged if they are passed with another value.
IGNORED_PSIL_ARGS = {
    "lr": "lr",
    "batch_size": "batch_size",
    "gamma": "gamma",
    "train_horizon": "train_horizon",
    "normalize_observation": "normalize_observation",
    "policy_baseline_type": "pi_baseline_type",
    "alpha": "pi_alpha",
    "bc_weight_temp": "pi_temp",
}


def warn_ignored_args(args, config, logger, prefix):
    for arg_name, cfg_key in IGNORED_PSIL_ARGS.items():
        if cfg_key not in config:
            continue
        val = getattr(args, arg_name, None)
        if val is not None and val != config[cfg_key]:
            logger.log(
                f"--{arg_name.replace('_', '-')}={val} is ignored; using "
                f"{cfg_key}={config[cfg_key]} from the config dict "
                f"(override with --{prefix}-{cfg_key.replace('_', '-')}).",
                color="red",
            )


def pubc_args():
    """PUBC flags: --pubc-<key> for every entry of default_cfg and pubc_cfg."""
    parse_prefixed_flags("pubc", [default_cfg, pubc_cfg])
    return dict(pubc_cfg)


# ----------------------------------------------------------------------------------
# Auxiliary structures
# ----------------------------------------------------------------------------------


@struct.dataclass
class PolicyAux:
    loss: float = 0.0
    entropy: float = 0.0


class TrainingState(nnx.Module):
    def __init__(self, policy, optimizer):
        self.policy = policy
        self.optimizer = optimizer


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


@jax.jit
def discounted_sum(vector_x, gamma):
    dtype = vector_x.dtype
    horizon = vector_x.shape[0]

    def body_fun(t, cumsum):
        cumsum = vector_x[horizon - 1 - t] + gamma * cumsum
        return cumsum

    init_cumsum = jnp.zeros_like(vector_x[0], dtype=dtype)
    cumsum = jax.lax.fori_loop(0, horizon, body_fun, init_cumsum)
    return cumsum


def get_valid_mask(dones, ep_len):
    """True for real (non-padded) steps; same episode-length rule as OnPolicyBuffer.add."""
    dones = np.asarray(dones).astype(bool)
    done_sum = dones.sum(axis=1)
    done_sum = np.where(done_sum == 0, 1, done_sum)
    true_ep_len = ep_len - done_sum + 1
    return np.arange(ep_len)[None, :] < true_ep_len[:, None]


def safe_mean(x):
    x = np.asarray(x)
    return float(x.mean()) if x.size > 0 else 0.0


# ----------------------------------------------------------------------------------
# PU filter network (authors' models.py)
# ----------------------------------------------------------------------------------


def torch_linear(in_features, out_features, rngs):
    """nn.Linear with PyTorch's default initialisation.

    weight ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)) (kaiming_uniform with a=sqrt(5)),
    bias   ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
    Note: models.init_weights is defined by the authors but never applied.
    """
    bound = 1.0 / np.sqrt(in_features)

    def bias_init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return nnx.Linear(
        in_features,
        out_features,
        kernel_init=nnx.initializers.variance_scaling(1.0 / 3.0, "fan_in", "uniform"),
        bias_init=bias_init,
        rngs=rngs,
    )


def torch_batchnorm(num_features, rngs):
    # nn.BatchNorm1d: momentum 0.1 (flax momentum = 1 - 0.1), eps 1e-5.
    # The train/eval mode is passed explicitly at call time (see FilterNet).
    return nnx.BatchNorm(
        num_features,
        momentum=0.9,
        epsilon=1e-5,
        rngs=rngs,
    )


class FilterNet(nnx.Module):
    def __init__(self, obs_dim, act_dim, rngs):
        # net_obs: Linear-BN-ReLU x4 (512, 512, 256, 128) -> Linear(action_dim * 2)
        obs_sizes = [obs_dim, 512, 512, 256, 128]
        self.obs_linears = tuple(
            [
                torch_linear(obs_sizes[i], obs_sizes[i + 1], rngs)
                for i in range(len(obs_sizes) - 1)
            ]
        )
        self.obs_norms = tuple(
            [torch_batchnorm(obs_sizes[i + 1], rngs) for i in range(len(obs_sizes) - 1)]
        )
        self.obs_out = torch_linear(obs_sizes[-1], act_dim * 2, rngs)

        # net: Linear-BN-ReLU x2 (128, 128) -> Linear(2) -> Softmax
        head_sizes = [act_dim * 3, 128, 128]
        self.head_linears = tuple(
            [
                torch_linear(head_sizes[i], head_sizes[i + 1], rngs)
                for i in range(len(head_sizes) - 1)
            ]
        )
        self.head_norms = tuple(
            [
                torch_batchnorm(head_sizes[i + 1], rngs)
                for i in range(len(head_sizes) - 1)
            ]
        )
        self.head_out = torch_linear(head_sizes[-1], 2, rngs)

    def logits(self, obs, act, training=True):
        # training=True : model.train() in PyTorch (batch statistics, update running stats)
        # training=False: model.eval() in PyTorch (running statistics)
        use_running_average = not training

        x = obs
        for linear, norm in zip(self.obs_linears, self.obs_norms):
            x = nnx.relu(norm(linear(x), use_running_average=use_running_average))
        o = self.obs_out(x)

        x = jnp.concatenate([o, act], axis=-1)
        for linear, norm in zip(self.head_linears, self.head_norms):
            x = nnx.relu(norm(linear(x), use_running_average=use_running_average))
        return self.head_out(x)

    def __call__(self, obs, act, training=True):
        # nn.Softmax(dim=1): column 0 = P(positive), column 1 = P(negative)
        return jax.nn.softmax(self.logits(obs, act, training=training), axis=-1)


def torch_bce_loss(logits, labels):
    """torch.nn.BCELoss(Softmax(logits), labels): mean reduction, log clamped at -100.

    Computed from the logits with log_softmax instead of log(softmax(.)). In float32
    the softmax saturates to exactly 0 or 1 as soon as the two logits differ by ~17,
    so log(1 - p) = -inf and the gradient 1 / (1 - p) = inf; the clamp at -100 then
    multiplies that inf by 0 and the loss becomes NaN. (PyTorch avoids this because its
    BCELoss backward divides by max(p * (1 - p), 1e-12).) For a two-class softmax,
    log(1 - p_j) = log p_{1-j}, so both terms come from the same stable log_softmax.
    """
    log_p = jnp.clip(jax.nn.log_softmax(logits, axis=-1), min=-100.0)
    log_1mp = log_p[..., ::-1]
    return -jnp.mean(labels * log_p + (1.0 - labels) * log_1mp)


def make_filter_optimizer(model, lr, weight_decay):
    # torch.optim.Adam(lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8): the L2 term
    # is added to the gradient before the Adam moments (not decoupled AdamW).
    return nnx.Optimizer(
        model=model,
        tx=optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.adam(learning_rate=lr, b1=0.9, b2=0.999, eps=1e-8),
        ),
    )


@nnx.jit
def filter_train_step(model, optimizer, obs, act, labels):
    def loss_fun(model):
        logits = model.logits(obs, act, training=True)
        return torch_bce_loss(logits, labels)

    loss, grads = nnx.value_and_grad(loss_fun)(model)
    optimizer.update(grads)
    return loss


@nnx.jit
def filter_positive_prob(model, obs, act):
    return model(obs, act, training=False)[..., 0]


# ----------------------------------------------------------------------------------
# Positive / synthetic-negative sampler (authors' buffer.TorchDatasetHandler)
# ----------------------------------------------------------------------------------


class PUSampler:
    def __init__(
        self, raw_obs, raw_act, max_action, max_observation, negative_sampler, rng
    ):
        # raw_obs / raw_act: all valid (non-padded) transitions of the raw dataset
        self.raw_obs = raw_obs
        self.raw_act = raw_act
        self.raw_len = raw_obs.shape[0]
        self.obs_dim = raw_obs.shape[-1]
        self.act_dim = raw_act.shape[-1]
        self.max_action = max_action
        self.max_observation = max_observation
        self.negative_sampler = negative_sampler
        self.rng = rng

    def _uniform_action(self, n):
        return self.rng.uniform(
            -self.max_action, self.max_action, size=(n, self.act_dim)
        ).astype(np.float32)

    def _uniform_observation(self, n):
        return self.rng.uniform(
            -self.max_observation, self.max_observation, size=(n, self.obs_dim)
        ).astype(np.float32)

    def _random_raw(self, store, n):
        return store[self.rng.integers(0, self.raw_len, size=n)]

    def _random_raw_differing(self, store, reference):
        # `while True: ... if not (pos == raw[random]).all(): break`
        out = self._random_raw(store, reference.shape[0])
        same = np.all(out == reference, axis=-1)
        while same.any():
            out[same] = self._random_raw(store, int(same.sum()))
            same = np.all(out == reference, axis=-1)
        return out

    def batch(self, pos_obs, pos_act):
        batch_size = pos_obs.shape[0]
        obs = np.array(pos_obs, dtype=np.float32, copy=True)
        act = np.array(pos_act, dtype=np.float32, copy=True)

        # `if np.random.uniform(0, 1) <= 0.5`: positive, label [1, 0]; else [0, 1]
        is_pos = self.rng.uniform(size=batch_size) <= 0.5
        neg = ~is_pos
        labels = np.zeros((batch_size, 2), dtype=np.float32)
        labels[is_pos, 0] = 1.0
        labels[neg, 1] = 1.0

        if self.negative_sampler == "part":
            # observation = pos_observation; action uniform (p=0.5) or another raw action
            u = self.rng.uniform(size=batch_size)
            uni = neg & (u < 0.5)
            raw = neg & ~(u < 0.5)
            act[uni] = self._uniform_action(int(uni.sum()))
            act[raw] = self._random_raw_differing(self.raw_act, pos_act[raw])

        elif self.negative_sampler == "full":
            # The authors draw a fresh np.random.uniform(0, 1) in every elif condition,
            # so the seven branches are not equiprobable. This reproduces that logic.
            u = self.rng.uniform(size=(6, batch_size))
            taken = np.zeros(batch_size, dtype=bool)
            branches = []
            for k in range(6):
                lo, hi = k / 7.0, (k + 1) / 7.0
                cond = (u[k] < hi) if k == 0 else ((u[k] >= lo) & (u[k] < hi))
                b = neg & ~taken & cond
                taken |= b
                branches.append(b)
            b1, b2, b3, b4, b5, b6 = branches
            b7 = neg & ~taken

            # 1: pos obs + uniform action
            act[b1] = self._uniform_action(int(b1.sum()))
            # 2: pos obs + different raw action
            act[b2] = self._random_raw_differing(self.raw_act, pos_act[b2])
            # 3: pos action + different raw observation
            obs[b3] = self._random_raw_differing(self.raw_obs, pos_obs[b3])
            # 4: random raw observation + uniform action
            obs[b4] = self._random_raw(self.raw_obs, int(b4.sum()))
            act[b4] = self._uniform_action(int(b4.sum()))
            # 5: pos action + uniform observation
            obs[b5] = self._uniform_observation(int(b5.sum()))
            # 6: random raw action + uniform observation
            act[b6] = self._random_raw(self.raw_act, int(b6.sum()))
            obs[b6] = self._uniform_observation(int(b6.sum()))
            # 7: uniform observation + uniform action
            obs[b7] = self._uniform_observation(int(b7.sum()))
            act[b7] = self._uniform_action(int(b7.sum()))
        else:
            raise ValueError("negative_sampler should be 'part' or 'full'.")

        return obs, act, labels


# ----------------------------------------------------------------------------------
# Adaptive threshold (authors' utils.adap_probs)
# ----------------------------------------------------------------------------------


def adap_probs(
    probs, bins=100, fit_pow=8, prob_th=0.96, save_plot_path=None, logger=None
):
    def get_minima(values):
        min_index = sg.argrelmin(values)[0]
        return min_index, values[min_index]

    probs = np.asarray(probs, dtype=np.float64)
    finite = np.isfinite(probs)
    if not finite.all():
        if logger is not None:
            logger.log(
                f"adap_probs: {int((~finite).sum())} non-finite trajectory probabilities "
                "ignored for the threshold fit.",
                color="red",
            )
        probs = probs[finite]
    if probs.size == 0:
        raise ValueError("adap_probs: no finite trajectory probability.")
    if probs.max() == probs.min():
        if logger is not None:
            logger.log(
                "adap_probs: all trajectory probabilities are equal; using th_high_bound.",
                color="red",
            )
        return prob_th

    unit = (probs.max() - probs.min()) / bins
    xdata = np.array([probs.min() + (i * unit) for i in range(bins)])
    ydata = np.array(np.histogram(probs, bins=bins)[0])

    coeffi = np.polyfit(xdata, ydata, fit_pow)
    pln = np.poly1d(coeffi)
    y_pred = pln(xdata)

    idxs, values = get_minima(y_pred)

    adap_v = None
    if idxs.shape[0] > 1:
        if xdata[idxs[-1]] >= prob_th:
            adap_p = xdata[idxs[-2]]
            adap_v = values[-2]
        else:
            adap_p = xdata[idxs[-1]]
            adap_v = values[-1]
    elif idxs.shape[0] == 1:
        if xdata[idxs[-1]] >= prob_th:
            adap_p = prob_th
        else:
            adap_p = xdata[idxs[-1]]
            adap_v = values[-1]
    else:
        # The original code raises IndexError here (no local minimum).
        adap_p = prob_th
        if logger is not None:
            logger.log(
                "adap_probs: no local minimum in the fitted histogram; using th_high_bound.",
                color="red",
            )

    if save_plot_path:
        fig, ax = plt.subplots()
        ax.plot(xdata, ydata, "*", color="gold", label="original values")
        ax.plot(xdata, y_pred, color="r", label="polyfit values")
        if adap_v is not None:
            ax.scatter(
                adap_p, adap_v, s=40, marker="D", color="b", label="adapted point"
            )
        ax.legend(loc=0)
        fig.savefig(save_plot_path, format="svg", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return float(adap_p)


# ----------------------------------------------------------------------------------
# PU filter loop (authors' main.main_filter + trainer.PUTrainer + buffer.PUBuffer)
# ----------------------------------------------------------------------------------


def make_filter_ensemble(num_models, obs_dim, act_dim, rngs, lr, weight_decay):
    # PUTrainer.reset(): new networks and optimisers at every iteration
    models = [FilterNet(obs_dim, act_dim, rngs) for _ in range(num_models)]
    optimizers = [make_filter_optimizer(m, lr, weight_decay) for m in models]
    return models, optimizers


def train_filter_member(
    model,
    optimizer,
    pos_obs,
    pos_act,
    sampler,
    epochs,
    batch_size,
    rng,
    iteration,
    member,
):
    num = pos_obs.shape[0]
    epoch_loss = 0.0
    for epoch in range(1, epochs + 1):
        # DataLoader(shuffle=True, drop_last=False)
        perm = rng.permutation(num)
        losses = []
        for start in range(0, num, batch_size):
            idx = perm[start : start + batch_size]
            if idx.shape[0] < 2:
                # BatchNorm cannot use batch statistics on a single sample (PyTorch
                # raises an error here); skip the leftover sample of this epoch.
                continue
            obs, act, labels = sampler.batch(pos_obs[idx], pos_act[idx])
            loss = filter_train_step(
                model,
                optimizer,
                jnp.asarray(obs),
                jnp.asarray(act),
                jnp.asarray(labels),
            )
            loss = float(loss)
            if not np.isfinite(loss):
                raise FloatingPointError(
                    f"PUBC filter loss became {loss} (iteration {iteration + 1}, "
                    f"member {member}, epoch {epoch}). The filter diverged; try a "
                    "lower --pubc-filter-lr."
                )
            losses.append(loss)
        epoch_loss = safe_mean(losses)
        print(
            f"Iteration: {iteration + 1}   Model: {member}   Epoch: {epoch}   Loss: {epoch_loss}"
        )
    return epoch_loss


def trajectory_positive_probs(models, raw_obs, raw_act, valid, chunk_size=65536):
    """Per classifier, the mean P(positive) over each trajectory's (valid) steps.

    Only valid (non-padded) steps are passed through the network. Masking by
    multiplication is not enough: padded steps can hold NaN/inf, and NaN * 0 = NaN,
    which made every trajectory probability NaN.
    """
    num_trajs, _ = valid.shape
    valid_obs = raw_obs[valid]
    valid_act = raw_act[valid]
    # trajectory index of every valid step (boolean indexing keeps row-major order)
    traj_idx = np.nonzero(valid)[0]
    steps_per_traj = valid.sum(axis=1)

    traj_probs = []
    for model in models:
        step_probs = []
        for start in range(0, valid_obs.shape[0], chunk_size):
            step_probs.append(
                np.asarray(
                    filter_positive_prob(
                        model,
                        jnp.asarray(valid_obs[start : start + chunk_size]),
                        jnp.asarray(valid_act[start : start + chunk_size]),
                    )
                )
            )
        step_probs = np.concatenate(step_probs).astype(np.float64)
        sums = np.bincount(traj_idx, weights=step_probs, minlength=num_trajs)
        # trajectories without any valid step get probability 0 (never selected)
        traj_probs.append(
            np.where(steps_per_traj > 0, sums / np.maximum(steps_per_traj, 1), 0.0)
        )
    # num_models X num_trajs
    return np.stack(traj_probs, axis=0)


def seed_positive_subsets(seed_obs, seed_act, num_models, subset, rng):
    # buffer.sub_seed_positive: random.sample(range(n), int(n * 0.95)) per member
    num = seed_obs.shape[0]
    subsets = []
    for _ in range(num_models):
        idx = rng.choice(num, size=int(num * subset), replace=False)
        subsets.append((seed_obs[idx], seed_act[idx]))
    return subsets


def relabelled_positive_subsets(raw_obs, raw_act, valid, selected, num_models):
    # buffer.update_pos: walk the relabelled transitions in data order and cut them into
    # num_models contiguous chunks of floor(n / num_models) transitions.
    mask = valid & selected[:, None]
    pos_obs = raw_obs[mask]
    pos_act = raw_act[mask]
    num_in_each = pos_obs.shape[0] // num_models
    if num_in_each == 0:
        return None
    return [
        (
            pos_obs[i * num_in_each : (i + 1) * num_in_each],
            pos_act[i * num_in_each : (i + 1) * num_in_each],
        )
        for i in range(num_models)
    ]


def plot_selected_trajectory_data(costs, rewards, selected, is_seed, save_plot):
    fig, ax = plt.subplots(figsize=(2.5, 1.6))
    rejected = ~selected & ~is_seed
    chosen = selected & ~is_seed
    ax.scatter(
        costs[rejected],
        rewards[rejected],
        c="#FFFFFF",
        marker="o",
        s=25,
        edgecolors="black",
        linewidths=0.5,
        label="Rejected",
    )
    ax.scatter(
        costs[chosen],
        rewards[chosen],
        c="#0072B2",
        marker="o",
        s=25,
        edgecolors="black",
        linewidths=0.5,
        label="Selected",
    )
    if is_seed.any():
        ax.scatter(
            costs[is_seed],
            rewards[is_seed],
            c="#E69F00",
            marker="*",
            s=60,
            edgecolors="black",
            linewidths=0.5,
            label="Seed",
        )
    ax.set_xlabel("Traj. Cost", fontsize=10)
    ax.set_ylabel("Traj. Reward", fontsize=10)
    fig.savefig(save_plot, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_pubc_filter(
    raw,
    seed,
    is_seed,
    cfg,
    filter_rngs,
    np_rng,
    filter_logger,
    logger,
    log_dir,
    run_seed,
):
    raw_obs, raw_act, valid = raw["observations"], raw["actions"], raw["valid"]
    seed_obs, seed_act = seed["observations"], seed["actions"]
    obs_dim, act_dim = raw_obs.shape[-1], raw_act.shape[-1]
    num_models = cfg["models_in_ensemble"]
    num_iterations = cfg["iterations"]
    assert num_iterations >= 1, "PUBC needs at least one filter iteration."

    traj_rewards = raw["rewards"].sum(axis=1)
    traj_costs = raw["costs"].sum(axis=1)
    union_mask = ~is_seed

    # args.max_action / args.max_observation in PUBuffer (scalars over the raw dataset)
    valid_obs = raw_obs[valid]
    valid_act = raw_act[valid]

    # Padded steps may contain NaN/inf (they are never used); real steps must be finite.
    for name, arr in (
        ("raw observations", valid_obs),
        ("raw actions", valid_act),
        ("seed observations", seed_obs),
        ("seed actions", seed_act),
    ):
        bad = ~np.isfinite(arr)
        if bad.any():
            bad_dims = np.nonzero(bad.any(axis=0))[0].tolist()
            raise ValueError(
                f"PUBC filter: {int(bad.any(axis=1).sum())} non-padded {name} contain "
                f"NaN/inf (dims {bad_dims}). Check the observation normalisation "
                "(e.g. a constant dimension with zero std)."
            )
    if num_trajs_without_steps := int((valid.sum(axis=1) == 0).sum()):
        logger.log(
            f"PUBC filter: {num_trajs_without_steps} trajectories have no valid step "
            "and will never be selected.",
            color="red",
        )
    sampler = PUSampler(
        raw_obs=valid_obs,
        raw_act=valid_act,
        max_action=float(valid_act.max()),
        max_observation=float(valid_obs.max()),
        negative_sampler=cfg["negative_sampler"],
        rng=np_rng,
    )

    # buffer.set_seed_positive()
    member_data = seed_positive_subsets(
        seed_obs, seed_act, num_models, cfg["pos_seed_subset"], np_rng
    )

    selected = np.zeros(raw_obs.shape[0], dtype=bool)
    for iteration in range(num_iterations):
        # Only the last iteration's ensemble/selection is used for BC, so only its
        # artifacts (models, probabilities, plots) are written to disk.
        is_last = iteration == num_iterations - 1

        models, optimizers = make_filter_ensemble(
            num_models,
            obs_dim,
            act_dim,
            filter_rngs,
            cfg["filter_lr"],
            cfg["filter_weight_decay"],
        )

        member_losses = []
        for member, (model, optimizer, (pos_obs, pos_act)) in enumerate(
            zip(models, optimizers, member_data)
        ):
            member_losses.append(
                train_filter_member(
                    model,
                    optimizer,
                    pos_obs,
                    pos_act,
                    sampler,
                    cfg["epochs_per_iteration"],
                    cfg["filter_batch_size"],
                    np_rng,
                    iteration,
                    member,
                )
            )

        # buffer.classifier_validate()
        member_probs = trajectory_positive_probs(models, raw_obs, raw_act, valid)
        probs = member_probs.mean(axis=0)
        threshold = adap_probs(
            probs,
            bins=cfg["th_bins"],
            fit_pow=cfg["th_fit_pow"],
            prob_th=cfg["th_high_bound"],
            save_plot_path=(
                f"{log_dir}/pubc_adap_probs_{run_seed}.svg" if is_last else None
            ),
            logger=logger,
        )
        if cfg["ensemble_method"] == "vote":
            votes = (member_probs >= threshold).sum(axis=0)
            selected = votes >= num_models / 2
        else:
            selected = probs >= threshold

        # logging (the authors log accuracy against ground-truth labels instead)
        sel_u = selected & union_mask
        rej_u = ~selected & union_mask
        filter_logger.log_tabular("Filter/Iteration", iteration + 1)
        filter_logger.log_tabular("Filter/Threshold", threshold)
        filter_logger.log_tabular("Filter/LossMean", float(np.mean(member_losses)))
        filter_logger.log_tabular("Filter/NumSelected", int(sel_u.sum()))
        filter_logger.log_tabular(
            "Filter/Coverage", 100.0 * sel_u.sum() / max(int(union_mask.sum()), 1)
        )
        filter_logger.log_tabular(
            "Filter/SeedSelected",
            safe_mean(selected[is_seed]) if is_seed.any() else 0.0,
        )
        filter_logger.log_tabular(
            "Filter/TransitionsSelected", int((valid & selected[:, None]).sum())
        )
        filter_logger.log_tabular("Prob/mean", float(probs.mean()))
        filter_logger.log_tabular("Prob/std", float(probs.std()))
        filter_logger.log_tabular("Prob/min", float(probs.min()))
        filter_logger.log_tabular("Prob/max", float(probs.max()))
        filter_logger.log_tabular("Reward/selected", safe_mean(traj_rewards[sel_u]))
        filter_logger.log_tabular("Reward/rejected", safe_mean(traj_rewards[rej_u]))
        filter_logger.log_tabular("Cost/selected", safe_mean(traj_costs[sel_u]))
        filter_logger.log_tabular("Cost/rejected", safe_mean(traj_costs[rej_u]))
        filter_logger.dump_tabular()

        if is_last:
            # Final filter: the one whose selection is used to train the policy.
            np.save(f"{log_dir}/pubc_probs_{run_seed}.npy", member_probs)
            plot_selected_trajectory_data(
                traj_costs,
                traj_rewards,
                selected,
                is_seed,
                save_plot=f"{log_dir}/pubc_selected_{run_seed}.svg",
            )
            for member, model in enumerate(models):
                logger.nn_model_save(
                    itr=iteration + 1,
                    nn_model_saver_element=model,
                    prefix=f"pubc_filter{member + 1}",
                )
            break

        # buffer.update_pos() (only needed if another iteration follows)
        new_member_data = relabelled_positive_subsets(
            raw_obs, raw_act, valid, selected, num_models
        )
        if new_member_data is None:
            logger.log(
                "PUBC filter selected too few transitions for the ensemble; "
                "reusing the seed-positive subsets for the next iteration.",
                color="red",
            )
            member_data = seed_positive_subsets(
                seed_obs, seed_act, num_models, cfg["pos_seed_subset"], np_rng
            )
        else:
            member_data = new_member_data

    return selected


# ----------------------------------------------------------------------------------
# Policy learning on the filtered trajectories (same policy and loss as PSIL)
# ----------------------------------------------------------------------------------


def policy_grad_aux_fun(policy_model, data, config):
    batch, horizon, _ = data.union_obs.shape

    # BH X obs/act_dim
    target_obs = data.union_obs.reshape(batch * horizon, -1)
    target_act = data.union_act.reshape(batch * horizon, -1)

    def loss_fun(policy_model):
        pred_act, log_pi, *_ = policy_model(target_obs)
        # BH
        l2_loss = optax.l2_loss(pred_act, target_act).sum(axis=-1)  # forward kl
        l2_loss = l2_loss.reshape(batch, horizon)  # B X H
        l2_loss = discounted_sum(l2_loss.T, config.gamma)  # B
        # PUBC: every selected trajectory has weight 1 (plain BC on the filtered data)
        loss = l2_loss.mean()
        return loss, PolicyAux(loss=loss, entropy=-log_pi.mean())

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, policy_aux), grads = grad_fun(policy_model)
    return grads, policy_aux


@nnx.jit
def train_n_steps(state, data_buffer, config, key):
    num_steps = config.log_freq

    key1, _ = jax.random.split(key, 2)
    pos_idxs, neg_idxs, union_idxs = data_buffer.sample_idxs(
        data_buffer, key1, num_steps
    )

    def body_fun(i, carry):
        state, data_buffer, _ = carry
        batch_data = data_buffer.sample_batch(
            data_buffer, pos_idxs[i], neg_idxs[i], union_idxs[i]
        )
        policy_grads, policy_aux = policy_grad_aux_fun(
            policy_model=state.policy, data=batch_data, config=config
        )
        state.optimizer.update(policy_grads)
        return state, data_buffer, policy_aux

    init_carry = (state, data_buffer, PolicyAux())
    _, _, policy_aux = nnx.fori_loop(0, num_steps, body_fun, init_carry)
    return policy_aux


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------


def main(args, pubc_config, cfg_env=None):
    """
    PUBC on DSRL, LfP setting: a PU filter is trained from the preferred (seed-positive)
    demonstrations, the unlabeled dataset is filtered, and the PSIL policy is trained
    with behaviour cloning on the selected trajectories only.
    """

    # set the random seed, device and number of threads (same order as PSIL)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rngs = nnx.Rngs(
        default=args.seed,
        params=args.seed + 3,
        dropout=args.seed + 5,
        random_sample=args.seed + 7,
    )
    # separate streams so the filter does not change the policy initialisation
    # or the PSIL data split
    filter_rngs = nnx.Rngs(params=args.seed + 101)
    np_rng = np.random.default_rng(args.seed + 1001)

    # set default device id
    jax.default_device = jax.devices(args.device)[args.device_id]

    has_positive = float(args.num_preferred > 0)
    assert has_positive, "PUBC requires preferred (seed-positive) demonstrations."

    trajectory_cfg["num_positive_trajectories"] = args.num_preferred
    trajectory_cfg["num_negative_trajectories"] = args.num_non_preferred
    trajectory_cfg["num_union_trajectories"] = args.num_union
    trajectory_cfg["non_pref_noise"] = args.non_pref_noise
    trajectory_cfg["data_inpaint"] = args.data_inpaint
    trajectory_cfg["inpaint_ranges"] = trajectory_data[args.data_inpaint]

    config = {**default_cfg, **trajectory_cfg}
    config["pubc"] = dict(pubc_config)
    batch_size = config["batch_size"]

    # env name
    env_name = re.search(r"Offline(.*?)(?:Gymnasium)?-v[0-9]", args.task).group(1)
    config["env_name"] = env_name

    config_data = make_static_config_from_dict(
        name="State",
        d={"log_freq": config["log_freq"], "gamma": config["gamma"]},
    )()

    # evaluation environment
    eval_env = gym.make(args.task)
    eval_env.set_target_cost(config["target_cost"])
    eval_env.reset(seed=args.seed)

    # policy (identical to PSIL)
    obs_space, act_space = eval_env.observation_space, eval_env.action_space
    policy_model = SafeDiceTanhMixtureActor(
        rngs=rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_size=config["hidden_size"],
    )
    policy_optimizer = nnx.Optimizer(
        model=policy_model,
        tx=optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adamw(
                learning_rate=config["lr"], weight_decay=config["weight_decay"]
            ),
        ),
    )
    state = TrainingState(policy=policy_model, optimizer=policy_optimizer)

    # data (identical to PSIL)
    ep_len = dsrl_infos.DEFAULT_MAX_EPISODE_STEPS[env_name]
    data = get_dataset_in_d4rl_format(
        eval_env, trajectory_cfg, args.task, ep_len, config["action_repeat"]
    )
    pos_data, neg_data, union_data = get_pos_neg_and_union_data(
        data, trajectory_cfg, save_dir=args.log_dir, seed=args.seed
    )
    mu_obs, std_obs = 0.0, 1.0
    if config["normalize_observation"]:
        pos_data, neg_data, union_data, mu_obs, std_obs = get_normalized_data(
            pos_data, neg_data, union_data
        )

    ep_len = ep_len // config["action_repeat"] + (ep_len % config["action_repeat"] > 0)
    assert (
        union_data["observations"].shape[1] == ep_len
    ), f"{union_data['observations'].shape[1]} episode length is different from {ep_len}"

    # set logger (config values take precedence over the PSIL command-line arguments)
    dict_args = dict(config)
    dict_args["psil_args"] = {k: v for k, v in vars(args).items() if v is not None}

    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    filter_logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
        output_fname="filter_progress.csv",
        use_tensorboard=False,
    )
    logger.save_config(dict_args)
    warn_ignored_args(args, config, logger, "pubc")

    if args.num_non_preferred > 0:
        logger.log(
            f"PUBC does not use non-preferred demonstrations; the {args.num_non_preferred} "
            "non-preferred trajectories are excluded from D_U by the data pipeline and ignored.",
            color="red",
        )

    # raw (mixed) dataset that the PU filter labels and BC learns from
    keys = ["observations", "actions", "rewards", "costs", "terminals", "timeouts"]
    num_pos = pos_data["observations"].shape[0]
    num_union = union_data["observations"].shape[0]
    if pubc_config["seed_in_raw"]:
        raw = {k: np.concatenate([pos_data[k], union_data[k]], axis=0) for k in keys}
        is_seed = np.concatenate([np.ones(num_pos, bool), np.zeros(num_union, bool)])
    else:
        raw = {k: np.array(union_data[k]) for k in keys}
        is_seed = np.zeros(num_union, bool)
    raw_dones = raw["timeouts"].astype(bool) | raw["terminals"].astype(bool)
    raw["observations"] = raw["observations"].astype(np.float32)
    raw["actions"] = raw["actions"].astype(np.float32)
    raw["valid"] = get_valid_mask(raw_dones, ep_len)

    pos_dones = pos_data["timeouts"].astype(bool) | pos_data["terminals"].astype(bool)
    pos_valid = get_valid_mask(pos_dones, ep_len)
    seed = {
        "observations": pos_data["observations"][pos_valid].astype(np.float32),
        "actions": pos_data["actions"][pos_valid].astype(np.float32),
    }

    logger.log("Start PUBC filter training.")
    selected = run_pubc_filter(
        raw=raw,
        seed=seed,
        is_seed=is_seed,
        cfg=pubc_config,
        filter_rngs=filter_rngs,
        np_rng=np_rng,
        filter_logger=filter_logger,
        logger=logger,
        log_dir=args.log_dir,
        run_seed=args.seed,
    )
    filter_logger.close()

    # policy buffer: every raw trajectory is stored; rejected ones get zero priority
    buffer = OnPolicyBuffer(
        rngs=rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        pos_data_size=1,
        neg_data_size=1,
        union_data_size=raw["observations"].shape[0] * ep_len,
        horizon=config["train_horizon"],
        batch_size=batch_size,
        ep_len=ep_len,
    )
    for obs, act, done, reward, cost in zip(
        raw["observations"], raw["actions"], raw_dones, raw["rewards"], raw["costs"]
    ):
        buffer.add(obs, act, done, reward, cost, is_union=True)
    buffer.to_jax_ndarray()
    data_buffer = buffer.get_data_buffer()

    traj_mask = np.repeat(selected.astype(np.float32), ep_len)
    filtered_priorities = data_buffer.union_priorities * jnp.asarray(traj_mask)
    if float(filtered_priorities.sum()) <= 0.0:
        logger.log(
            "PUBC filter selected no usable trajectory; falling back to the seed positives.",
            color="red",
        )
        if not is_seed.any():
            raise RuntimeError(
                "No trajectory selected and the seed is not part of the raw dataset."
            )
        selected = is_seed.copy()
        traj_mask = np.repeat(selected.astype(np.float32), ep_len)
        filtered_priorities = data_buffer.union_priorities * jnp.asarray(traj_mask)
    data_buffer = data_buffer.replace(union_priorities=filtered_priorities)

    # the trajectory mask actually used to train the policy
    np.save(f"{args.log_dir}/pubc_selected_{args.seed}.npy", selected)

    logger.log("Start bc_policy training on the PUBC-filtered trajectories.")
    save_freq = int(config["save_freq"])
    steps = 0
    while steps < config["total_iteration"]:
        policy_aux = train_n_steps(
            state=state,
            data_buffer=data_buffer,
            config=config_data,
            key=rngs.random_sample(),
        )
        steps += config["log_freq"]

        logger.logged = False
        logger.log_tabular("Train/Steps", steps)
        logger.log_tabular("Loss/pi_union", policy_aux.loss.item())
        logger.log_tabular("Mean/pi_union_entropy", policy_aux.entropy.item())
        logger.log_tabular(
            "Percentage/union_selected",
            100.0 * float((selected & ~is_seed).sum()) / max(num_union, 1),
        )
        logger.log_tabular(
            "Norm/policy_model",
            get_tree_norm(nnx.state(policy_model, nnx.Param)),
        )
        logger.dump_tabular()

        if steps >= config["total_iteration"]:
            break

        # optional intermediate checkpoints (disabled by default: save_freq = 0)
        if save_freq > 0 and steps % save_freq == 0:
            logger.nn_model_save(
                itr=steps,
                nn_model_saver_element=policy_model,
                prefix="bc_policy",
            )

    # final policy (the only policy checkpoint saved by default)
    logger.nn_model_save(
        itr=steps, nn_model_saver_element=policy_model, prefix="bc_policy"
    )
    if config["normalize_observation"]:
        logger.save_state(
            state_dict={"mu_obs": mu_obs, "std_obs": std_obs}, dirname="norm"
        )

    logger.close()


if __name__ == "__main__":
    pubc_config = pubc_args()
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
                main(args, pubc_config, cfg_env)
    else:
        main(args, pubc_config, cfg_env)
