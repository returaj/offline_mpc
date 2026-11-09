"""
PNU: Semi-supervised classification based on classification from positive and
unlabeled data (Sakai, du Plessis, Niu, Sugiyama, ICML 2017), ported to JAX.

Faithful re-implementation of the authors' official code (pywsl by Tomoya Sakai,
https://github.com/t-sakai-kure/pywsl):
    pywsl/ssl/pnu_mr.py   PNU_SL_FastCV, make_mat, solve, calc_risk, calc_etab, ker,
                          sqdist, make_func
    pywsl/cpe/cpe_ene.py  cpe (class-prior estimation based on energy distance)
inside the PSIL experimental pipeline, so the only difference between PSIL and PNU
is how the preference-weights of the unlabeled trajectories are obtained.

PNU in a nutshell (squared loss, linear-in-parameter model g(x) = K(x, C) theta)
---------------------------------------------------------------------------
* Gaussian basis: b = min(200, n_u) centres C drawn from the unlabeled data,
  K = exp(-||x - c||^2 / (2 sigma^2)); or a linear model K = [x, 1] ('lm').
* Sufficient statistics (make_mat):
      H_p = pi E_P[k k^T],  H_n = (1 - pi) E_N[k k^T],  H_u = E_U[k k^T]
      h_p = pi E_P[k],      h_n = (1 - pi) E_N[k],      h_u = E_U[k]
* Closed-form solution (solve), eta in [-1, 1]:
      eta >= 0 (PN + PU): ((1-eta)(H_p+H_n) + eta H_u + lam I) theta
                              = (1-eta)(h_p - h_n) + eta (2 h_p - h_u)
      eta <  0 (PN + NU): ((1-|eta|)(H_p+H_n) + |eta| H_u + lam I) theta
                              = (1-|eta|)(h_p - h_n) + |eta| (h_u - 2 h_n)
* Model selection (PNU_SL_FastCV): n_fold cross-validation over sigma, lambda and
  eta; every candidate is scored with the zero-one PNU risk using the fixed
  combination eta_hat = calc_etab(n_p, n_n, prior), not the candidate's own eta.
  For each eta the best (sigma, lambda) is taken, then the eta with the smallest
  risk; the classifier is refitted on all data with the selected values.
* Class prior pi: estimated with cpe (energy distance) from P, N and U.

How PNU is plugged into the PSIL pipeline
-----------------------------------------
* Inputs are state-action pairs (normalised observation, action), as in PUBC:
  P = transitions of D_P, N = transitions of D_N, U = transitions of D_U
  (padded steps of the DSRL episodes are excluded).
* With the squared loss the PNU minimiser estimates g(x) = p(+|x) - p(-|x), which
  lives in [-1, 1] like PSIL's score s_theta. The preference-weight of a window
  (sub-trajectory of length H starting at step t) is the mean decision value over
  the window, clipped to [-1, 1]:  w = clip(mean_{t..t+H-1} g(s, a), -1, 1).
* These weights replace PSIL's progressive weights in exactly the same weighted
  behaviour-cloning objective as PSIL (same policy, same baseline/temperature,
  same additional BC term on D_P). This adaptation step is ours; the authors only
  provide the classifier.

Settings
--------
* LfP+N (D_P and D_N given): the method of the paper; eta is selected by CV and the
  class prior is estimated with cpe unless --pnu-prior is given.
* LfP (only D_P): eta = +1 (pure PU learning). The authors' pu_sl in pul/pu_mr.py
  references undefined variables, so the PNU solver at eta = +1 is used, which is
  exactly PU_SL.fit: (H_u + lam I) theta = 2 pi E_P[k] - E_U[k]; validation uses
  the PU risk. cpe needs both classes, so --pnu-prior is required.
* LfN (only D_N): eta = -1 (pure NU learning), symmetric to LfP; --pnu-prior required.

Deviations (only where the original code cannot apply directly)
--------------------------------------------------------------
* DSRL datasets contain hundreds of thousands of unlabeled transitions; the
  classifier is fitted on a random subset of at most --pnu-max-unlabeled unlabeled
  transitions (the decision function is then evaluated on all of D_U).
* Linear algebra runs in float32 (JAX default) instead of NumPy float64; squared
  distances are clipped at 0 before the kernel to avoid round-off.
* The estimated class prior is clipped to [1e-3, 1 - 1e-3] (cpe can return 0 or 1).
* With a single (sigma, lambda, eta) combination the original code sets the best
  index to 1 (out of range); here it is 0.

Usage:
    python pnu_jax.py --task OfflinePointCircle2Gymnasium-v0 --num-preferred 1 \
        --num-non-preferred 10 --num-union 500 [PSIL policy flags] [--pnu-* flags]
PNU-specific flags are prefixed with --pnu- and are removed from sys.argv before
single_agent_args() parses the shared PSIL arguments.
"""

import argparse
import functools
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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx, struct
from matplotlib.colors import LinearSegmentedColormap

from dsrl_model.utils.buffer_jax import SafeCLBuffer
from dsrl_model.utils.dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_normalized_data,
    get_pos_neg_and_union_data,
)
from dsrl_model.utils.models_jax import (
    SafeDiceTanhMixtureActor,
    get_tree_norm,
    softer_max,
)
from dsrl_model.utils.native_logger import EpochLogger
from dsrl_model.utils.utils import make_static_config_from_dict, single_agent_args

EPS = 1e-6

# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------

# Policy-learning configuration: identical to PSIL so that the policy is the same.
default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "hidden_size": 256,
    "max_grad_norm": 5.0,
    "gamma": 0.9,  # for trajectory based policy learning
    "action_repeat": 1,
    "train_horizon": 50,
    "pi_temp": 0.2,
    "weight_decay": 0.01,
    "total_iteration": int(1e6),
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

# PNU configuration: defaults of the authors' code.
#   PNU_SL_FastCV: n_fold=5, basis='gauss', n_basis=200,
#                  sigma_list = sqrt(median d_u) * logspace(-1, 1, 11),
#                  lambda_list = logspace(-3, 1, 10)
#   demo_pnu_mr_cv.py: eta_list = mgrid[-.9:.9:19j]
pnu_cfg = {
    "basis": "gauss",  # 'gauss' or 'lm'
    "n_basis": 200,
    "n_fold": 5,
    "sigma_num": 11,
    "lambda_min": -3.0,  # log10
    "lambda_max": 1.0,  # log10
    "lambda_num": 10,
    "eta_min": -0.9,
    "eta_max": 0.9,
    "eta_num": 19,
    "prior": None,  # None: estimate with cpe (LfP+N only)
    "max_unlabeled": 10000,
}


def pnu_args():
    """Parse PNU-only flags and strip them from sys.argv."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pnu-basis", type=str, default=None, choices=["gauss", "lm"])
    parser.add_argument("--pnu-n-basis", type=int, default=None)
    parser.add_argument("--pnu-n-fold", type=int, default=None)
    parser.add_argument("--pnu-sigma-num", type=int, default=None)
    parser.add_argument("--pnu-lambda-min", type=float, default=None)
    parser.add_argument("--pnu-lambda-max", type=float, default=None)
    parser.add_argument("--pnu-lambda-num", type=int, default=None)
    parser.add_argument("--pnu-eta-min", type=float, default=None)
    parser.add_argument("--pnu-eta-max", type=float, default=None)
    parser.add_argument("--pnu-eta-num", type=int, default=None)
    parser.add_argument("--pnu-prior", type=float, default=None)
    parser.add_argument("--pnu-max-unlabeled", type=int, default=None)
    pnu, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    cfg = dict(pnu_cfg)
    for key, val in vars(pnu).items():
        if val is not None:
            cfg[key[len("pnu_"):]] = val
    return cfg


# ----------------------------------------------------------------------------------
# Auxiliary structures (same as PSIL)
# ----------------------------------------------------------------------------------


class TrainingState(nnx.Module):
    def __init__(self, policy, optimizer):
        self.policy = policy
        self.optimizer = optimizer


@struct.dataclass
class PolicyAux:
    @struct.dataclass
    class Value:
        loss: float = 0.0
        weight: float = 0.0
        entropy: float = 0.0

    pos: Value = Value()
    union: Value = Value()
    pi_baseline: float = 0.0


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


def get_sa_pairs(data, ep_len):
    """Valid (observation, action) pairs of a d4rl-format trajectory dataset."""
    if data is None:
        return None, None
    dones = data["timeouts"].astype(bool) | data["terminals"].astype(bool)
    valid = get_valid_mask(dones, ep_len)
    x = np.concatenate([data["observations"], data["actions"]], axis=-1)
    return x[valid].astype(np.float32), valid


def safe_mean(x):
    x = np.asarray(x)
    return float(x.mean()) if x.size > 0 else 0.0


# ----------------------------------------------------------------------------------
# PNU classifier (authors' pywsl/ssl/pnu_mr.py)
# ----------------------------------------------------------------------------------


@jax.jit
def sqdist(x, c):
    d = (
        jnp.sum(x**2, axis=1, keepdims=True)
        + jnp.sum(c**2, axis=1, keepdims=True).T
        - 2.0 * x @ c.T
    )
    return jnp.maximum(d, 0.0)


def homo_coord(x):
    return jnp.concatenate([x, jnp.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)


def ker(d, sigma, basis):
    if basis == "gauss":
        return jnp.exp(-d / (2.0 * sigma**2))
    return d  # 'lm': d already holds the homogeneous coordinates


def make_mat(K_p, K_n, K_u, prior):
    b = K_u.shape[1]
    zeros_H, zeros_h = jnp.zeros((b, b)), jnp.zeros((b,))
    n_p, n_n, n_u = K_p.shape[0], K_n.shape[0], K_u.shape[0]
    H_p = prior * (K_p.T @ K_p) / n_p if n_p != 0 else zeros_H
    H_n = (1 - prior) * (K_n.T @ K_n) / n_n if n_n != 0 else zeros_H
    H_u = (K_u.T @ K_u) / n_u if n_u != 0 else zeros_H
    h_p = prior * jnp.mean(K_p, axis=0) if n_p != 0 else zeros_h
    h_n = (1 - prior) * jnp.mean(K_n, axis=0) if n_n != 0 else zeros_h
    h_u = jnp.mean(K_u, axis=0) if n_u != 0 else zeros_h
    return H_p, H_n, H_u, h_p, h_n, h_u


@functools.partial(jax.jit, static_argnames=["basis"])
def solve(H_p, H_n, H_u, h_p, h_n, h_u, lam, eta, basis):
    b = H_u.shape[0]
    reg_diag = jnp.ones((b,))
    if basis == "lm":
        reg_diag = reg_diag.at[b - 1].set(0.0)  # bias is not regularised
    Reg = lam * jnp.diag(reg_diag)

    H_pn = H_p + H_n
    h_pn = h_p - h_n

    abs_eta = jnp.abs(eta)
    h_pu = 2 * h_p - h_u  # PNPU (eta >= 0)
    h_nu = h_u - 2 * h_n  # PNNU (eta < 0)
    h_xu = jnp.where(eta >= 0, h_pu, h_nu)
    return jnp.linalg.solve(
        (1 - abs_eta) * H_pn + abs_eta * H_u + Reg,
        (1 - abs_eta) * h_pn + abs_eta * h_xu,
    )


@functools.partial(jax.jit, static_argnames=["basis"])
def solve_grid(H_p, H_n, H_u, h_p, h_n, h_u, lams, etas, basis):
    # eta X lambda X b
    solve_lam = jax.vmap(
        lambda lam, eta: solve(H_p, H_n, H_u, h_p, h_n, h_u, lam, eta, basis),
        in_axes=(0, None),
    )
    return jax.vmap(lambda eta: solve_lam(lams, eta))(etas)


def calc_risk(g_p, g_n, g_u, prior, eta):
    """Zero-one PNU risk; g_* have the samples on the last axis."""
    n_p, n_n, n_u = g_p.shape[-1], g_n.shape[-1], g_u.shape[-1]

    f_n = jnp.mean(g_p <= 0, axis=-1) if n_p != 0 else 0.0
    f_p = jnp.mean(g_n >= 0, axis=-1) if n_n != 0 else 0.0
    risk_pn = prior * f_n + (1 - prior) * f_p

    if eta >= 0:  # PNPU
        f_pu = jnp.mean(g_u >= 0, axis=-1) if n_u != 0 else 0.0
        risk_pu = prior * f_n + jnp.maximum(f_pu + prior * f_n - prior, 0)
        risk = (1 - eta) * risk_pn + eta * risk_pu
    else:  # PNNU
        eta = -eta
        f_nu = jnp.mean(g_u <= 0, axis=-1) if n_u != 0 else 0.0
        risk_nu = jnp.maximum(f_nu + (1 - prior) * f_p - (1 - prior), 0) + (
            1 - prior
        ) * f_p
        risk = (1 - eta) * risk_pn + eta * risk_nu

    return risk


def calc_etab(n_p, n_n, prior):
    psi_p, psi_n = prior**2 / n_p, (1 - prior) ** 2 / n_n
    return (psi_n - psi_p) / (psi_p + psi_n)


def cv_index(n, n_fold, np_rng):
    index = (np.arange(n, dtype=np.int_) * n_fold) // max(n, 1)
    return index[np_rng.permutation(n)]


def pnu_sl_fast_cv(
    x_p,
    x_n,
    x_u,
    prior,
    eta_list,
    eta_val,
    n_fold,
    basis,
    lambda_list,
    sigma_num,
    n_basis,
    np_rng,
):
    """PNU_SL_FastCV (nargout=2). x_p / x_n may have zero rows (LfN / LfP)."""
    n_p, n_n, n_u = x_p.shape[0], x_n.shape[0], x_u.shape[0]
    dim = x_u.shape[1]
    x_p, x_n, x_u = jnp.asarray(x_p), jnp.asarray(x_n), jnp.asarray(x_u)

    if basis == "gauss":
        b = int(np.minimum(n_basis, n_u))
        center_index = np_rng.permutation(n_u)[:b]
        x_c = x_u[center_index, :]
        d_p = sqdist(x_p, x_c) if n_p else jnp.zeros((0, b))
        d_n = sqdist(x_n, x_c) if n_n else jnp.zeros((0, b))
        d_u = sqdist(x_u, x_c)
        med = float(jnp.median(d_u.ravel()))
        sigma_list = np.sqrt(med) * np.logspace(-1, 1, sigma_num)
    elif basis == "lm":
        b = dim + 1
        x_c = None
        d_p = homo_coord(x_p) if n_p else jnp.zeros((0, b))
        d_n = homo_coord(x_n) if n_n else jnp.zeros((0, b))
        d_u = homo_coord(x_u)
        sigma_list = np.array([1.0])
    else:
        raise ValueError("Invalid basis type.")

    cv_index_p = cv_index(n_p, n_fold, np_rng)
    cv_index_n = cv_index(n_n, n_fold, np_rng)
    cv_index_u = cv_index(n_u, n_fold, np_rng)

    lams = jnp.asarray(lambda_list, dtype=jnp.float32)
    etas = jnp.asarray(eta_list, dtype=jnp.float32)

    score_cv_fold = np.empty((len(sigma_list), len(lambda_list), len(eta_list), n_fold))
    if len(eta_list) == 1 and len(sigma_list) == 1 and len(lambda_list) == 1:
        score_cv = np.empty((1, 1, 1))
        score_cv[0, 0, 0] = -np.inf
        best_sigma_index, best_lambda_index, best_eta_index = 0, 0, 0
    else:
        for ite_sigma, sigma in enumerate(sigma_list):
            K_p, K_n, K_u = ker(d_p, sigma, basis), ker(d_n, sigma, basis), ker(d_u, sigma, basis)

            for ite_fold in range(n_fold):
                mats = make_mat(
                    K_p[cv_index_p != ite_fold, :],
                    K_n[cv_index_n != ite_fold, :],
                    K_u[cv_index_u != ite_fold, :],
                    prior,
                )
                K_pte = K_p[cv_index_p == ite_fold, :]
                K_nte = K_n[cv_index_n == ite_fold, :]
                K_ute = K_u[cv_index_u == ite_fold, :]

                # eta X lambda X b
                theta_cv = solve_grid(*mats, lams, etas, basis)
                gp = jnp.einsum("elb,nb->eln", theta_cv, K_pte)
                gn = jnp.einsum("elb,nb->eln", theta_cv, K_nte)
                gu = jnp.einsum("elb,nb->eln", theta_cv, K_ute)
                # eta X lambda
                risk = np.asarray(calc_risk(gp, gn, gu, prior, eta_val))
                score_cv_fold[ite_sigma, :, :, ite_fold] = np.broadcast_to(
                    risk, (len(eta_list), len(lambda_list))
                ).T

        score_cv = np.mean(score_cv_fold, axis=3)

    score_list = np.empty(len(eta_list))
    score_best = np.inf
    for ite_eta in range(len(eta_list)):
        sub_score_cv = score_cv[:, :, ite_eta]
        tmp = np.argmin(sub_score_cv.ravel())
        sigma_index, lambda_index = np.unravel_index(tmp, sub_score_cv.shape)
        score_list[ite_eta] = sub_score_cv[sigma_index, lambda_index]
        if score_list[ite_eta] < score_best:
            score_best = score_list[ite_eta]
            best_sigma_index = sigma_index
            best_lambda_index = lambda_index
            best_eta_index = ite_eta

    sigma = float(sigma_list[best_sigma_index])
    lam = float(lambda_list[best_lambda_index])
    eta = float(eta_list[best_eta_index])

    K_p, K_n, K_u = ker(d_p, sigma, basis), ker(d_n, sigma, basis), ker(d_u, sigma, basis)
    mats = make_mat(K_p, K_n, K_u, prior)
    theta = solve(*mats, lam, eta, basis)

    outs = {
        "sigma": sigma,
        "lambda": lam,
        "eta": eta,
        "sigma_index": int(best_sigma_index),
        "lambda_index": int(best_lambda_index),
        "eta_index": int(best_eta_index),
        "score_best": float(score_best),
        "score_table": score_cv,
        "score_list": score_list,
        "sigma_list": np.asarray(sigma_list),
        "w": np.asarray(theta),
    }
    return theta, x_c, outs


def make_func(theta, x_c, sigma, basis, x_t, chunk_size=65536):
    """Decision values g(x_t) = K(x_t) theta, evaluated in chunks."""
    out = []
    for start in range(0, x_t.shape[0], chunk_size):
        x = jnp.asarray(x_t[start : start + chunk_size])
        if basis == "gauss":
            K = jnp.exp(-sqdist(x, x_c) / (2.0 * sigma**2))
        else:
            K = homo_coord(x)
        out.append(np.asarray(K @ theta))
    return np.concatenate(out) if out else np.zeros((0,), np.float32)


# ----------------------------------------------------------------------------------
# Class-prior estimation (authors' pywsl/cpe/cpe_ene.py)
# ----------------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=["alpha"])
def _alpha_dist_sum(x, c, alpha):
    dist = (
        jnp.sum(x**2, axis=1, keepdims=True)
        + jnp.sum(c**2, axis=1, keepdims=True).T
        - 2 * x @ c.T
    )
    if alpha == 1:
        return jnp.sum(jnp.sqrt(jnp.maximum(dist, 0)))
    return jnp.sum(jnp.power(jnp.maximum(dist, 0), alpha / 2))


def mean_alpha_dist(x, c, alpha=1, chunk_size=2048):
    """np.mean(alpha_dist(x, c, alpha)) computed in chunks."""
    total = 0.0
    for start in range(0, x.shape[0], chunk_size):
        total += float(
            _alpha_dist_sum(jnp.asarray(x[start : start + chunk_size]), jnp.asarray(c), alpha)
        )
    return total / (x.shape[0] * c.shape[0])


def cpe(xl, y, xu, alpha=1):
    c = np.sort(np.unique(y))
    n_c = len(c)

    x = []
    for i in range(n_c):
        x.append(xl[y == c[i], :])

    b = np.empty(n_c)
    for i in range(n_c):
        b[i] = 2 * mean_alpha_dist(x[i], xu, alpha)

    A = np.empty((n_c, n_c))
    for i in range(n_c):
        for j in range(n_c):
            A[i, j] = -mean_alpha_dist(x[i], x[j], alpha)

    As = A[: (n_c - 1), : (n_c - 1)]
    a = A[: (n_c - 1), n_c - 1]
    Ac = A[n_c - 1, n_c - 1]
    bs = b[: (n_c - 1)]
    bc = b[n_c - 1]

    Anew = ((As - a[:, None].T) - a) + Ac
    Anew = (Anew + Anew.T) / 2
    bnew = 2 * a - 2 * Ac + bs - bc

    x0 = -np.linalg.solve(2 * Anew, bnew[:, None].T)
    x = np.minimum(np.maximum(0, x0), 1)
    theta = 1 - np.sum(x)

    return theta


# ----------------------------------------------------------------------------------
# From PNU decision values to PSIL preference-weights
# ----------------------------------------------------------------------------------


def window_weights(g, valid, horizon):
    """w(t) = clip(mean of g over the valid steps of [t, t + H), -1, 1).

    g, valid: num_trajs X ep_len. Windows starting at t > true_len - H are never
    sampled by the buffer (zero priority); their value is the mean over the
    remaining valid steps.
    """
    num_trajs, ep_len = g.shape
    g = np.where(valid, g, 0.0)
    v = valid.astype(np.float64)
    cg = np.concatenate([np.zeros((num_trajs, 1)), np.cumsum(g, axis=1)], axis=1)
    cv = np.concatenate([np.zeros((num_trajs, 1)), np.cumsum(v, axis=1)], axis=1)
    start = np.arange(ep_len)
    end = np.minimum(start + horizon, ep_len)
    window_sum = cg[:, end] - cg[:, start]
    window_cnt = cv[:, end] - cv[:, start]
    w = window_sum / np.maximum(window_cnt, 1.0)
    return np.clip(w, -1.0, 1.0).astype(np.float32)


def plot_weighted_trajectory_data(costs, rewards, weights, save_plot):
    # same style as PSIL's plot_weighted_trajectory_data
    neg_limit, pos_limit = min(min(weights), -0.01), max(max(weights), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=neg_limit, vcenter=0, vmax=pos_limit)
    fig, ax = plt.subplots(figsize=(2.5, 1.6))

    pos_mask = weights > 0
    neg_mask = weights <= 0

    cmap = LinearSegmentedColormap.from_list(
        "preference",
        [
            "#D97662",  # Non-Preferred (coral)
            "#F8E9E5",  # very light coral
            "#FFFFFF",  # zero
            "#DCEAF7",  # very light blue
            "#0072B2",  # Preferred (blue)
        ],
        N=256,
    )

    sc_pos = ax.scatter(
        np.array(costs)[pos_mask],
        np.array(rewards)[pos_mask],
        c=weights[pos_mask],
        cmap=cmap,
        norm=norm,
        marker="o",
        s=25,
        edgecolors="black",
        linewidths=0.5,
        label="Positive",
    )
    ax.scatter(
        np.array(costs)[neg_mask],
        np.array(rewards)[neg_mask],
        c=weights[neg_mask],
        cmap=cmap,
        norm=norm,
        marker="x",
        s=25,
        linewidths=1.0,
        label="Negative",
    )
    cbar = fig.colorbar(sc_pos, ax=ax)
    cbar.set_label("weights", fontsize=10)
    ax.set_xlabel("Traj. Cost", fontsize=10)
    ax.set_ylabel("Traj. Reward", fontsize=10)
    fig.savefig(save_plot, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------
# Policy learning: PSIL's weighted behaviour cloning (copied from PSIL)
# ----------------------------------------------------------------------------------


def policy_grad_aux_fun(
    policy_model, data, union_weight, pos_scale, union_scale, config
):
    batch, horizon, _ = data.union_obs.shape

    # BH X obs/act_dim
    target_union_obs = data.union_obs.reshape(batch * horizon, -1)
    target_union_act = data.union_act.reshape(batch * horizon, -1)
    target_pos_obs = data.pos_obs.reshape(batch * horizon, -1)
    target_pos_act = data.pos_act.reshape(batch * horizon, -1)

    # scalar
    if config.pi_baseline_type == "softer_max":
        pi_baseline = softer_max(union_weight, config.pi_temp)
    elif config.pi_baseline_type == "mean_std":
        pi_baseline = union_weight.mean() + config.pi_alpha * union_weight.std()
    else:
        pi_baseline = config.pi_alpha

    def fwd_kl_loss(policy_model, target_obs, target_act, logw, scale):
        pred_act, log_pi, *_ = policy_model(target_obs)
        # BH
        l2_loss = optax.l2_loss(pred_act, target_act).sum(axis=-1)  # forward kl
        l2_loss = l2_loss.reshape(batch, horizon)  # B X H
        l2_loss = discounted_sum(l2_loss.T, config.gamma)  # B

        # B
        weight = jnp.exp(jnp.clip((logw - pi_baseline) / config.pi_temp, max=5.0))
        weight = scale * jax.lax.stop_gradient(weight)

        loss = (weight * l2_loss).mean()
        return loss, PolicyAux.Value(
            loss=loss,
            weight=weight.mean(),
            entropy=-log_pi.mean(),
        )

    def loss_fun(policy_model):
        union_loss, union_aux = fwd_kl_loss(
            policy_model,
            target_union_obs,
            target_union_act,
            union_weight,
            union_scale,
        )
        pos_weight = jnp.ones_like(union_weight)
        pos_loss, pos_aux = fwd_kl_loss(
            policy_model,
            target_pos_obs,
            target_pos_act,
            pos_weight,
            pos_scale,
        )
        loss = union_loss + pos_loss
        return loss, PolicyAux(
            union=union_aux,
            pos=pos_aux,
            pi_baseline=pi_baseline,
        )

    grad_fun = nnx.value_and_grad(loss_fun, has_aux=True)
    (loss, policy_aux), grads = grad_fun(policy_model)

    return grads, policy_aux


@nnx.jit
def train_n_steps(state, data_buffer, config, has_positive, key):
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
        # PNU preference-weights are fixed: read them from the buffer
        policy_grads, policy_aux = policy_grad_aux_fun(
            policy_model=state.policy,
            data=batch_data,
            union_weight=batch_data.union_weight,
            pos_scale=1.0 * has_positive,
            union_scale=1.0,
            config=config,
        )
        state.optimizer.update(policy_grads)
        return state, data_buffer, policy_aux

    init_carry = (state, data_buffer, PolicyAux())
    _, _, policy_aux = nnx.fori_loop(0, num_steps, body_fun, init_carry)
    return policy_aux


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------


def main(args, pnu_config, cfg_env=None):
    """
    PNU on DSRL: a PNU (or PU / NU) classifier on state-action pairs gives
    preference-weights to the unlabeled sub-trajectories, which are then used in
    PSIL's weighted behaviour-cloning objective.
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
    # separate stream so PNU does not change the PSIL data split
    np_rng = np.random.default_rng(args.seed + 2002)

    # set default device id
    jax.default_device = jax.devices(args.device)[args.device_id]

    has_positive = float(args.num_preferred > 0)
    has_negative = float(args.num_non_preferred > 0)

    assert (
        has_positive or has_negative
    ), "Both preferred and non-preferred trajectory dataset cannot be empty together."

    trajectory_cfg["num_positive_trajectories"] = args.num_preferred
    trajectory_cfg["num_negative_trajectories"] = args.num_non_preferred
    trajectory_cfg["num_union_trajectories"] = args.num_union
    trajectory_cfg["non_pref_noise"] = args.non_pref_noise
    trajectory_cfg["data_inpaint"] = args.data_inpaint
    trajectory_cfg["inpaint_ranges"] = trajectory_data[args.data_inpaint]

    config = {**default_cfg, **trajectory_cfg}
    config["pnu"] = dict(pnu_config)
    config["pi_baseline_type"] = args.policy_baseline_type
    config["pi_alpha"] = args.alpha
    config["train_horizon"] = args.train_horizon or config.get("train_horizon")
    config["normalize_observation"] = args.normalize_observation
    config["pi_temp"] = args.bc_weight_temp or config["pi_temp"]
    config["lr"] = args.lr
    config["gamma"] = args.gamma or config["gamma"]

    # set training steps
    batch_size = args.batch_size or config.get("batch_size")
    config["batch_size"] = batch_size

    # env name
    env_name = re.search(r"Offline(.*?)(?:Gymnasium)?-v[0-9]", args.task).group(1)
    config["env_name"] = env_name

    config_data = make_static_config_from_dict(
        name="State",
        d={
            "log_freq": config["log_freq"],
            "gamma": config["gamma"],
            "pi_temp": config["pi_temp"],
            "pi_alpha": config["pi_alpha"],
            "pi_baseline_type": config["pi_baseline_type"],
        },
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

    union_observations = union_data["observations"]
    union_actions = union_data["actions"]
    union_dones = union_data["timeouts"] | union_data["terminals"]
    union_rewards = union_data["rewards"]
    union_costs = union_data["costs"]

    ep_len = ep_len // config["action_repeat"] + (ep_len % config["action_repeat"] > 0)
    assert (
        union_observations.shape[1] == ep_len
    ), f"{union_observations.shape[1]} episode length is different from {ep_len}"

    pos_data_size = 1
    if has_positive:
        pos_data_size = np.prod(pos_data["observations"].shape[:-1])

    neg_data_size = 1
    if has_negative:
        neg_data_size = np.prod(neg_data["observations"].shape[:-1])

    # buffer (identical to PSIL)
    buffer = SafeCLBuffer(
        rngs=rngs,
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        pos_data_size=pos_data_size,
        neg_data_size=neg_data_size,
        union_data_size=np.prod(union_observations.shape[:-1]),
        horizon=config["train_horizon"],
        batch_size=batch_size,
        ep_len=ep_len,
    )

    if has_positive:
        pos_dones = pos_data["timeouts"] | pos_data["terminals"]
        for obs, act, done, reward, cost in zip(
            pos_data["observations"],
            pos_data["actions"],
            pos_dones,
            pos_data["rewards"],
            pos_data["costs"],
        ):
            buffer.add(obs, act, done, reward, cost, is_pos=True)

    if has_negative:
        neg_dones = neg_data["timeouts"] | neg_data["terminals"]
        for obs, act, done, reward, cost in zip(
            neg_data["observations"],
            neg_data["actions"],
            neg_dones,
            neg_data["rewards"],
            neg_data["costs"],
        ):
            buffer.add(obs, act, done, reward, cost, is_neg=True)

    for obs, act, done, reward, cost in zip(
        union_observations, union_actions, union_dones, union_rewards, union_costs
    ):
        buffer.add(obs, act, done, reward, cost, is_union=True)

    buffer.to_jax_ndarray()
    data_buffer = buffer.get_data_buffer()

    # set logger
    dict_args = config
    dict_args.update((k, v) for k, v in vars(args).items() if v is not None)

    logger = EpochLogger(log_dir=args.log_dir, seed=str(args.seed))
    pnu_logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
        output_fname="pnu_progress.csv",
        use_tensorboard=False,
    )
    logger.save_config(dict_args)

    # ------------------------------------------------------------------------------
    # PNU classifier on state-action pairs
    # ------------------------------------------------------------------------------
    x_p, _ = get_sa_pairs(pos_data if has_positive else None, ep_len)
    x_n, _ = get_sa_pairs(neg_data if has_negative else None, ep_len)
    x_u_all, union_valid = get_sa_pairs(union_data, ep_len)
    sa_dim = x_u_all.shape[1]
    if x_p is None:
        x_p = np.zeros((0, sa_dim), np.float32)
    if x_n is None:
        x_n = np.zeros((0, sa_dim), np.float32)

    num_u = x_u_all.shape[0]
    if pnu_config["max_unlabeled"] and num_u > pnu_config["max_unlabeled"]:
        u_idx = np_rng.choice(num_u, size=pnu_config["max_unlabeled"], replace=False)
        x_u = x_u_all[u_idx]
    else:
        x_u = x_u_all

    # class prior and eta grid for the setting
    if has_positive and has_negative:
        setting = "LfP+N"
        if pnu_config["prior"] is None:
            x_l = np.concatenate([x_p, x_n], axis=0)
            y_l = np.concatenate([np.ones(x_p.shape[0]), -np.ones(x_n.shape[0])])
            prior_raw = float(cpe(x_l, y_l, x_u))
        else:
            prior_raw = float(pnu_config["prior"])
        prior = float(np.clip(prior_raw, 1e-3, 1 - 1e-3))
        eta_list = np.linspace(
            pnu_config["eta_min"], pnu_config["eta_max"], pnu_config["eta_num"]
        )
        eta_val = calc_etab(x_p.shape[0], x_n.shape[0], prior)
    else:
        setting = "LfP" if has_positive else "LfN"
        assert pnu_config["prior"] is not None, (
            f"{setting}: cpe needs both preferred and non-preferred samples; "
            "pass the class prior with --pnu-prior."
        )
        prior_raw = float(pnu_config["prior"])
        prior = float(np.clip(prior_raw, 1e-3, 1 - 1e-3))
        # PU (eta = +1) or NU (eta = -1); validation with the same risk
        eta_list = np.array([1.0 if has_positive else -1.0])
        eta_val = float(eta_list[0])

    if prior != prior_raw:
        logger.log(
            f"PNU class prior {prior_raw:.4f} clipped to {prior:.4f}.", color="red"
        )
    logger.log(
        f"PNU ({setting}): n_p={x_p.shape[0]}, n_n={x_n.shape[0]}, n_u={x_u.shape[0]}, "
        f"prior={prior:.4f}, validation eta={eta_val:.4f}"
    )

    lambda_list = np.logspace(
        pnu_config["lambda_min"], pnu_config["lambda_max"], pnu_config["lambda_num"]
    )
    theta, x_c, outs = pnu_sl_fast_cv(
        x_p=x_p,
        x_n=x_n,
        x_u=x_u,
        prior=prior,
        eta_list=eta_list,
        eta_val=eta_val,
        n_fold=pnu_config["n_fold"],
        basis=pnu_config["basis"],
        lambda_list=lambda_list,
        sigma_num=pnu_config["sigma_num"],
        n_basis=pnu_config["n_basis"],
        np_rng=np_rng,
    )

    # decision values on every (valid) unlabeled transition -> window weights
    g_flat = make_func(theta, x_c, outs["sigma"], pnu_config["basis"], x_u_all)
    num_trajs = union_valid.shape[0]
    g = np.zeros((num_trajs, ep_len), np.float32)
    g[union_valid] = g_flat
    weights = window_weights(g, union_valid, config["train_horizon"])

    data_buffer = data_buffer.replace(
        union_weight=jnp.asarray(weights.reshape(-1), dtype=data_buffer.union_weight.dtype)
    )

    # trajectory-level summaries (mean weight over the sampled window starts)
    priorities = np.asarray(data_buffer.union_priorities).reshape(num_trajs, ep_len)
    traj_weight = (weights * (priorities != 0)).sum(1) / np.maximum(
        (priorities != 0).sum(1), 1
    )
    traj_rewards = np.asarray(union_rewards).sum(1)
    traj_costs = np.asarray(union_costs).sum(1)
    pos_traj, neg_traj = traj_weight > 0, traj_weight <= 0
    sampled = priorities != 0

    pnu_logger.log_tabular("PNU/prior_estimated", prior_raw)
    pnu_logger.log_tabular("PNU/prior", prior)
    pnu_logger.log_tabular("PNU/eta_validation", eta_val)
    pnu_logger.log_tabular("PNU/sigma", outs["sigma"])
    pnu_logger.log_tabular("PNU/lambda", outs["lambda"])
    pnu_logger.log_tabular("PNU/eta", outs["eta"])
    pnu_logger.log_tabular("PNU/cv_risk", outs["score_best"])
    pnu_logger.log_tabular("PNU/n_p", x_p.shape[0])
    pnu_logger.log_tabular("PNU/n_n", x_n.shape[0])
    pnu_logger.log_tabular("PNU/n_u", x_u.shape[0])
    pnu_logger.log_tabular("Percentage/union_pos_sa", 100.0 * float((g_flat > 0).mean()))
    pnu_logger.log_tabular(
        "Percentage/union_pos_window", 100.0 * float((weights[sampled] > 0).mean())
    )
    pnu_logger.log_tabular("Percentage/union_pos_traj", 100.0 * float(pos_traj.mean()))
    pnu_logger.log_tabular("Mean/union_weight", float(weights[sampled].mean()))
    pnu_logger.log_tabular("Mean/union_weight_std", float(weights[sampled].std()))
    pnu_logger.log_tabular("Reward/union_pos_traj", safe_mean(traj_rewards[pos_traj]))
    pnu_logger.log_tabular("Reward/union_neg_traj", safe_mean(traj_rewards[neg_traj]))
    pnu_logger.log_tabular("Cost/union_pos_traj", safe_mean(traj_costs[pos_traj]))
    pnu_logger.log_tabular("Cost/union_neg_traj", safe_mean(traj_costs[neg_traj]))
    pnu_logger.dump_tabular()
    pnu_logger.close()

    np.save(f"{args.log_dir}/pnu_window_weights_{args.seed}.npy", weights)
    np.save(f"{args.log_dir}/pnu_score_table_{args.seed}.npy", outs["score_table"])
    logger.save_state(
        state_dict={
            "theta": outs["w"],
            "centers": None if x_c is None else np.asarray(x_c),
            "sigma": outs["sigma"],
            "lambda": outs["lambda"],
            "eta": outs["eta"],
            "prior": prior,
            "basis": pnu_config["basis"],
        },
        dirname="pnu",
    )
    plot_weighted_trajectory_data(
        costs=traj_costs,
        rewards=traj_rewards,
        weights=traj_weight,
        save_plot=f"{args.log_dir}/weight_dataset_{args.seed}.svg",
    )

    # ------------------------------------------------------------------------------
    # Policy learning (PSIL's weighted behaviour cloning)
    # ------------------------------------------------------------------------------
    logger.log("Start bc_policy training with PNU preference-weights.")
    steps = 0
    while steps < config["total_iteration"]:
        policy_aux = train_n_steps(
            state=state,
            data_buffer=data_buffer,
            config=config_data,
            has_positive=has_positive,
            key=rngs.random_sample(),
        )
        steps += config["log_freq"]

        logger.logged = False
        logger.log_tabular("Train/Steps", steps)
        logger.log_tabular("Loss/pi_union", policy_aux.union.loss.item())
        logger.log_tabular("Loss/pi_pos", policy_aux.pos.loss.item())
        logger.log_tabular("Mean/pi_union_weight", policy_aux.union.weight.item())
        logger.log_tabular("Mean/pi_union_entropy", policy_aux.union.entropy.item())
        logger.log_tabular("Mean/pi_pos_weight", policy_aux.pos.weight.item())
        logger.log_tabular("Mean/pi_pos_entropy", policy_aux.pos.entropy.item())
        logger.log_tabular("Mean/pi_baseline", policy_aux.pi_baseline.item())
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
    if config["normalize_observation"]:
        logger.save_state(
            state_dict={"mu_obs": mu_obs, "std_obs": std_obs}, dirname="norm"
        )

    logger.close()


if __name__ == "__main__":
    pnu_config = pnu_args()
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
                main(args, pnu_config, cfg_env)
    else:
        main(args, pnu_config, cfg_env)
