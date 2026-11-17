"""
dsrl_safety_evals.py  --  DSRL-protocol safety evaluation for the PSIL rebuttal (R1 / R2)
========================================================================================

Notebook usage (each step is its own cell; nothing is saved to disk unless you do it):

    import dsrl_safety_evals as E

    hyperparams = "neg_union_full/five_seed/only_for_reporting"
    key = "LfPN"
    ALGOS = E.SETTINGS[key]          # or your own [(path, label), ...]
    tasks = E.TASKS

    # 1) compute metrics (uses your notebook's load_data_as_array)
    res = E.evaluate_setting(hyperparams, ALGOS, tasks, load_data_as_array)

    # 2) tables -> dict of LaTeX strings
    tables = E.generate_tables(res, setting_name=key)
    E.save_tex(tables["main"], "table_main.tex")

    # 2b) sanity check: reproduces Tables 5-10 (raw reward / cost) exactly
    print(E.latex_table_raw(res["metrics"]))

    # 3) figures -> (fig, axs)
    fig, ax = E.plot_reward_vs_cost(res["metrics"], level="mid")
    fig, ax = E.plot_fbeta_kappa_aggregate(res["fbeta_agg"])
    fig, ax = E.plot_fbeta_kappa_per_task(res["fbeta"], level="strict")

Statistics (why this version exists)
------------------------------------
Tables 5-10 of the paper come from the "Line/Bar Plot" cells of safecl_plotting.ipynb:
`bootstrap_confidence_interval(data)` with `np.random.seed(0)` and `np.random.choice`, reseeded
for every (method, task, metric), and the printed pair `m[-1], max(m[-1]-l[-1], u[-1]-m[-1])`.
So those tables report
    centre = mean of the 1,000 bootstrap means (final evaluation step)
    spread = max one-sided half-width of the 95% percentile bootstrap CI   (NOT the std)
and reward is normalized AFTER bootstrapping (clip + scale applied to m, l, u).

The previous version of this module used the plain seed mean, per-seed clipping and np.std,
which is why Tables 11-15 did not line up with Tables 5-10. With STAT_MODE = "notebook"
(default) every number here is derived from exactly the same bootstrap as Tables 5-10:
    R_norm   = clip-normalize(m_R)            +- max(m-l, u-m) of the normalized CI
    C/kappa  = m_C / kappa                    +- halfwidth_C / kappa
Set STAT_MODE = "std" to get plain mean +- std instead (then also regenerate Tables 5-10).

Definitions
-----------
  R_norm   = (R - R_min) / (R_max - R_min)                 DSRL reward bounds, clipped
  C_norm   = C / kappa                                     <= 1  <=>  constraint satisfied
  S_kappa  = 1 if C <= kappa else kappa / C                safety score for F-beta
  F_beta^k = (1 + b^2) R_norm S_kappa / (b^2 R_norm + S_kappa)

Thresholds follow the OSRL/DSRL protocol and are printed as strict / medium / loose:
Safety-Gymnasium {20, 40, 80}, Bullet-Gym {10, 20, 40}. The numbers appear only in the suite
header rows of the main table and in the one-sentence LEVEL_DEFINITION added to every caption.
Internal level keys remain "strict" / "mid" / "loose" (use level="mid" in the plot calls).
LaTeX tables need \\usepackage{booktabs, xcolor, multirow, graphicx}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EP = 1e-8

# ======================================================================================
# Configuration
# ======================================================================================
TASKS = [
    "PointCircle2",
    "PointGoal1",
    "CarCircle2",
    "CarGoal1",
    "AntVelocity",
    "SwimmerVelocity",
    "AntRun",
    "DroneRun",
    "AntCircle",
    "DroneCircle",
]

PSIL_LFPN = "safecl/embd_64_piconst_0.9_wdecay_fulltrans/best_maps"
SETTINGS = {
    "LfPN": [
        ("safedice", "SafeDICE"),
        ("uniq", "UNIQ"),
        ("osil", "OSIL (LfP+N)"),
        ("ipl", "IPL (LfP+N)"),
        ("dualcoil", "DualCoil"),
        ("pnu_lfpn", "PNU (LfP+N)"),
        (PSIL_LFPN, "PSIL (LfP+N)"),
    ],
    "LfP": [
        ("bc_union_kl", "BC-Comp"),
        ("iqlearn", "IQLearn"),
        ("ilid_mse_nonorm", "ILID"),
        ("ipl_lfp", "IPL (LfP)"),
        ("pubc", "PUBC"),
        ("safecl_lfp/traj", "PSIL (LfP)"),
    ],
    "LfN": [
        ("safedice", "SafeDICE"),
        ("uniq", "UNIQ"),
        ("osil_lfn", "OSIL (LfN)"),
        ("safecl_lfn/traj/itr_1000", "PSIL (LfN)"),
    ],
}

SUITE_THRESHOLDS = {"safetygym": (20.0, 40.0, 80.0), "bullet": (10.0, 20.0, 40.0)}
LEVEL_NAMES = ("strict", "mid", "loose")
# Internal keys stay "strict" / "mid" / "loose"; everything printed uses these names.
LEVEL_LABELS = {"strict": "strict", "mid": "medium", "loose": "loose"}
LEVEL_TITLES = {"strict": "Strict", "mid": "Medium", "loose": "Loose"}
# One sentence, used in every caption, that maps the names to the DSRL thresholds.
LEVEL_DEFINITION = (
    "Strict, medium and loose denote the DSRL cost thresholds "
    "$\\kappa=20, 40, 80$ for Safety-Gymnasium and $\\kappa=10, 20, 40$ for Bullet-Gym."
)

TASK_SUITE = {
    "PointCircle2": "safetygym",
    "CarCircle2": "safetygym",
    "PointGoal1": "safetygym",
    "CarGoal1": "safetygym",
    "AntVelocity": "safetygym",
    "SwimmerVelocity": "safetygym",
    "AntRun": "bullet",
    "DroneRun": "bullet",
    "AntCircle": "bullet",
    "DroneCircle": "bullet",
}

# Table 2 of the paper (dsrl/infos.py); used only if `dsrl` cannot be imported.
FALLBACK_REWARD_BOUNDS = {
    "PointCircle2": (20.48, 54.02),
    "PointGoal1": (0.01, 30.07),
    "CarCircle2": (5.11, 25.95),
    "CarGoal1": (0.01, 39.91),
    "AntVelocity": (6.15, 2976.28),
    "SwimmerVelocity": (0.07, 238.96),
    "AntRun": (0.0, 955.48),
    "DroneRun": (10.56, 682.83),
    "AntCircle": (0.02, 460.71),
    "DroneCircle": (207.79, 996.39),
}

# R_norm must exceed this to count as a success / to be bolded (Appendix M.1 uses 0.1).
MIN_REWARD = 0.1

# "notebook": bootstrap mean +- 95% CI half-width, identical to Tables 5-10 (default)
# "std":      plain seed mean +- std over seeds
STAT_MODE = "notebook"
STD_DDOF = 0  # used only for the *_std columns / STAT_MODE="std"

N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95
BOOT_SEED = 0

BETAS = (0.5, 1, 1.5, 2, 5)

# Same palette / markers as safecl_plotting.ipynb
_CMKY = {
    "cornflowerblue": "#4F81BD",
    "orange": "#E69F00",
    "coral": "#D97662",
    "orchid": "#B57BB3",
    "olive": "#7A8F00",
    "slate": "#6C757D",
    "brown": "#8C564B",
    "vermillion": "#D55E00",
}
BASE_COLORS = [
    "black",
    _CMKY["cornflowerblue"],
    _CMKY["orange"],
    _CMKY["olive"],
    _CMKY["orchid"],
    _CMKY["slate"],
    _CMKY["cornflowerblue"],
    _CMKY["brown"],
    _CMKY["vermillion"],
    _CMKY["coral"],
]
MARKERS = [None, "o", "s", "D", "p", "X", "*", "P", "h", ">", "d", "<", "+", "x", "H", "v", "^"]


def notebook_colors_markers(n):
    """Reproduces `BASE_COLORS[2:n+1] + [BASE_COLORS[-1]]` from the notebook."""
    return BASE_COLORS[2 : n + 1] + [BASE_COLORS[-1]], MARKERS[2 : n + 1] + [MARKERS[-1]]


# ======================================================================================
# Data access (load_fn = load_data_as_array from your notebook)
# ======================================================================================
def load_final(hyperparams, algo, task, load_fn, step=-1):
    """Final evaluation (after 1M steps): per-seed reward and cost."""
    r = load_fn(hyperparams, algo, task, "ep_reward")[:, step]
    c = load_fn(hyperparams, algo, task, "ep_cost")[:, step]
    return np.asarray(r, dtype=float), np.asarray(c, dtype=float)


# ======================================================================================
# Core helpers
# ======================================================================================
def _key(task):
    return task.replace("_", "")


def suite_of(task):
    return TASK_SUITE[_key(task)]


def thresholds_of(task, thresholds=None):
    return (
        tuple(float(k) for k in thresholds)
        if thresholds is not None
        else SUITE_THRESHOLDS[suite_of(task)]
    )


def get_reward_bounds(task):
    key = _key(task)
    try:
        import dsrl.infos as dsrl_infos

        return float(dsrl_infos.MIN_EPISODE_REWARD[key]), float(
            dsrl_infos.MAX_EPISODE_REWARD[key]
        )
    except Exception:
        return FALLBACK_REWARD_BOUNDS[key]


def normalize_reward(rewards, task, clip=True):
    r = np.asarray(rewards, dtype=float)
    r_min, r_max = get_reward_bounds(task)
    if clip:
        r = np.clip(r, r_min, r_max)
    return (r - r_min) / (r_max - r_min)


def normalized_cost(costs, kappa):
    return np.asarray(costs, dtype=float) / float(kappa)


def violation_rate(costs, kappa):
    return float(np.mean(np.asarray(costs, dtype=float) > kappa))


def kappa_safety_score(costs, kappa):
    c = np.asarray(costs, dtype=float)
    return np.where(c <= kappa, 1.0, kappa / np.maximum(c, EP))


# --------------------------------------------------------------------------------------
# Bootstrap identical to `bootstrap_confidence_interval` in safecl_plotting.ipynb.
# The notebook calls np.random.seed(seed) and then np.random.choice(n, (n,)) n_bootstrap
# times; RandomState(seed).choice draws the same stream without touching global state.
# The indices do not depend on T, so bootstrapping only the final column gives the same
# value as taking [-1] of the full time-series bootstrap.
# --------------------------------------------------------------------------------------
def notebook_bootstrap_indices(n, n_bootstrap=N_BOOTSTRAP, seed=BOOT_SEED):
    rs = np.random.RandomState(seed)
    return np.stack([rs.choice(n, (n,), replace=True) for _ in range(n_bootstrap)])


def notebook_bootstrap(x, idx, confidence_level=CONFIDENCE_LEVEL):
    """Returns (mean of bootstrap means, lower, upper), as in the notebook."""
    boot = np.asarray(x, dtype=float)[idx].mean(axis=1)
    a = (1 - confidence_level) / 2
    return (
        float(boot.mean()),
        float(np.percentile(boot, 100 * a)),
        float(np.percentile(boot, 100 * (1 - a))),
    )


def halfwidth(m, lo, hi):
    """The '+-' printed for Tables 5-10: max(m[-1] - l[-1], u[-1] - m[-1])."""
    return float(max(m - lo, hi - m))


# ======================================================================================
# Constraint satisfaction (R1)
# ======================================================================================
def evaluate_task(
    rewards,
    costs,
    task,
    thresholds=None,
    use_upper_ci=False,
    n_bootstrap=N_BOOTSTRAP,
    seed=BOOT_SEED,
    stat_mode=None,
):
    """Rows for each kappa level plus the DSRL 'avg' row (C/kappa averaged over thresholds).

    Columns used by the tables: norm_reward / norm_reward_err and norm_cost / norm_cost_err.
    In "notebook" mode these are the bootstrap mean and CI half-width (= Tables 5-10 after
    normalization); in "std" mode the plain mean and std over seeds.
    """
    stat_mode = stat_mode or STAT_MODE
    rewards, costs = np.asarray(rewards, float), np.asarray(costs, float)
    kappas = thresholds_of(task, thresholds)

    # one set of indices per (method, task), shared by reward and cost (as in the notebook,
    # which reseeds with the same seed for every metric)
    idx = notebook_bootstrap_indices(len(rewards), n_bootstrap, seed)

    # raw statistics (exactly the numbers of Tables 5-10)
    raw_r = notebook_bootstrap(rewards, idx)
    raw_c = notebook_bootstrap(costs, idx)

    # normalized reward: notebook normalizes m, l, u AFTER bootstrapping
    nr_m, nr_lo, nr_hi = (float(v) for v in normalize_reward(np.array(raw_r), task))
    nr_seed = normalize_reward(rewards, task)  # per-seed (only for std mode / std column)

    if stat_mode == "notebook":
        r_centre, r_err = nr_m, halfwidth(nr_m, nr_lo, nr_hi)
    elif stat_mode == "std":
        r_centre, r_err = float(nr_seed.mean()), float(np.std(nr_seed, ddof=STD_DDOF))
    else:
        raise ValueError(f"unknown stat_mode {stat_mode!r}")

    def cost_row(level, kappa, inv_k):
        # C/kappa is linear in C, so the bootstrap CI scales by 1/kappa (inv_k = mean 1/kappa
        # for the 'avg' row)
        c_m, c_lo, c_hi = (v * inv_k for v in raw_c)
        nc_seed = costs * inv_k
        if stat_mode == "notebook":
            c_centre, c_err = c_m, halfwidth(c_m, c_lo, c_hi)
        else:
            c_centre, c_err = float(nc_seed.mean()), float(np.std(nc_seed, ddof=STD_DDOF))
        vr = (
            violation_rate(costs, kappa)
            if not np.isnan(kappa)
            else float(np.mean([violation_rate(costs, k) for k in kappas]))
        )
        return dict(
            task=_key(task),
            suite=suite_of(task),
            level=level,
            kappa=kappa,
            norm_reward=r_centre,
            norm_reward_err=r_err,
            norm_reward_lo=nr_lo,
            norm_reward_hi=nr_hi,
            norm_reward_std=float(np.std(nr_seed, ddof=STD_DDOF)),
            norm_cost=c_centre,
            norm_cost_err=c_err,
            norm_cost_lo=c_lo,
            norm_cost_hi=c_hi,
            norm_cost_std=float(np.std(nc_seed, ddof=STD_DDOF)),
            raw_reward=raw_r[0],
            raw_reward_err=halfwidth(*raw_r),
            raw_cost=raw_c[0],
            raw_cost_err=halfwidth(*raw_c),
            raw_cost_std=float(np.std(costs, ddof=STD_DDOF)),
            violation_rate=vr,
            safe=bool((c_hi if use_upper_ci else c_centre) <= 1.0),
        )

    rows = [cost_row(lvl, k, 1.0 / k) for lvl, k in zip(LEVEL_NAMES, kappas)]
    rows.append(cost_row("avg", np.nan, float(np.mean([1.0 / k for k in kappas]))))
    return rows


METRIC_COLS = [
    "algo", "task", "suite", "level", "kappa",
    "norm_reward", "norm_reward_err", "norm_reward_lo", "norm_reward_hi", "norm_reward_std",
    "norm_cost", "norm_cost_err", "norm_cost_lo", "norm_cost_hi", "norm_cost_std",
    "raw_reward", "raw_reward_err", "raw_cost", "raw_cost_err", "raw_cost_std",
    "violation_rate", "safe",
]


def evaluate_algorithms(hyperparams, algos, tasks, load_fn, thresholds=None, **kwargs):
    rows = []
    for path, label in algos:
        for task in tasks:
            r, c = load_final(hyperparams, path, task, load_fn)
            for rw in evaluate_task(r, c, task, thresholds, **kwargs):
                rw["algo"] = label
                rows.append(rw)
    return pd.DataFrame(rows)[METRIC_COLS]


def safety_summary(df, algo_order=None, min_reward=MIN_REWARD):
    algos = algo_order or list(dict.fromkeys(df["algo"]))
    rows = []
    for a in algos:
        for lvl in list(LEVEL_NAMES) + ["avg"]:
            g = df[(df["algo"] == a) & (df["level"] == lvl)]
            safe = g["safe"].to_numpy()
            succ = (g["safe"] & (g["norm_reward"] > min_reward)).to_numpy()
            rows.append(
                dict(
                    algo=a,
                    level=lvl,
                    n_success=int(succ.sum()),
                    success_tasks=f"{int(succ.sum())}/{len(g)}",
                    reward_on_success_tasks=(
                        g.loc[succ, "norm_reward"].mean() if succ.any() else np.nan
                    ),
                    n_safe=int(safe.sum()),
                    n_tasks=len(g),
                    safe_tasks=f"{int(safe.sum())}/{len(g)}",
                    reward_on_safe_tasks=(
                        g.loc[g["safe"], "norm_reward"].mean() if safe.any() else np.nan
                    ),
                    reward_unsafe_as_zero=float(np.where(safe, g["norm_reward"], 0.0).mean()),
                    # mean over tasks of R_norm, counting tasks without a success as 0
                    safe_reward=float(np.where(succ, g["norm_reward"], 0.0).mean()),
                    mean_norm_cost=g["norm_cost"].mean(),
                    mean_violation_rate=g["violation_rate"].mean(),
                )
            )
    return pd.DataFrame(rows)


def _competition_rank(values, i, tol=1e-9):
    """1 = best (higher is better); tied methods share the rank. Returns (rank, tied)."""
    better = sum(v > values[i] + tol for v in values)
    tied = sum(abs(v - values[i]) <= tol for v in values) > 1
    return better + 1, tied


def success_table(df, algo_order=None, min_reward=MIN_REWARD):
    """Per (method, level): n_success, safe_reward, n_best_rows and their ranks among methods.
    n_best_rows = number of task rows in which the method is bolded (highest reward among the
    methods that succeed)."""
    d = df[df["level"].isin(LEVEL_NAMES)]
    algos = algo_order or list(dict.fromkeys(d["algo"]))
    tasks = list(dict.fromkeys(d["task"]))
    rows = []
    for lvl in LEVEL_NAMES:
        g = d[d["level"] == lvl]
        n_succ, s_rew, n_best = [], [], []
        for a in algos:
            ga = g[g["algo"] == a]
            succ = ga["safe"].to_numpy() & (ga["norm_reward"].to_numpy() > min_reward)
            n_succ.append(int(succ.sum()))
            s_rew.append(float(np.where(succ, ga["norm_reward"], 0.0).sum() / max(len(tasks), 1)))
        best_by_task = [_best_safe(g[g["task"] == t].set_index("algo"), algos, min_reward) for t in tasks]
        n_best = [sum(b == a for b in best_by_task) for a in algos]
        for j, a in enumerate(algos):
            rows.append(
                dict(
                    algo=a, level=lvl, n_tasks=len(tasks),
                    n_success=n_succ[j], rank_success=_competition_rank(n_succ, j),
                    safe_reward=s_rew[j], rank_safe_reward=_competition_rank(s_rew, j),
                    n_best_rows=n_best[j], rank_best_rows=_competition_rank(n_best, j),
                )
            )
    return pd.DataFrame(rows)


# ======================================================================================
# Threshold-aware F-beta (same conventions as the F-beta cells of the notebook:
#   per task  -> bootstrap_fbeta_of_means            (default_rng(seed) per method/task)
#   aggregate -> stratified_bootstrap_fbeta_of_means (default_rng(seed) per method)
# point estimate from the plain seed means, CI from the bootstrap)
# ======================================================================================
def _fbeta_kappa_of_means(r_mean, c_mean, kappa, betas):
    b = np.asarray(betas, dtype=float).reshape(-1, *([1] * np.ndim(r_mean)))
    s = kappa_safety_score(c_mean, kappa)
    return (1 + b**2) * r_mean * s / np.maximum(b**2 * r_mean + s, EP)


def evaluate_fbeta_kappa(
    hyperparams,
    algos,
    tasks,
    load_fn,
    betas=BETAS,
    thresholds=None,
    n_bootstrap=N_BOOTSTRAP,
    confidence_level=CONFIDENCE_LEVEL,
    seed=BOOT_SEED,
):
    a_ = (1 - confidence_level) / 2
    rows, store = [], {}
    for path, label in algos:
        for task in tasks:
            r, c = load_final(hyperparams, path, task, load_fn)
            nr = normalize_reward(r, task)
            store[(label, _key(task))] = (nr, c)
            rng = np.random.default_rng(seed)  # reseeded per (method, task), as in notebook
            idx = rng.integers(0, len(nr), size=(n_bootstrap, len(nr)))
            for lvl, k in zip(LEVEL_NAMES, thresholds_of(task, thresholds)):
                point = _fbeta_kappa_of_means(nr.mean(), c.mean(), k, betas)
                boot = _fbeta_kappa_of_means(nr[idx].mean(1), c[idx].mean(1), k, betas)
                lo = np.percentile(boot, 100 * a_, axis=1)
                hi = np.percentile(boot, 100 * (1 - a_), axis=1)
                se = boot.std(axis=1)
                for bi, b in enumerate(betas):
                    rows.append(
                        dict(
                            algo=label,
                            task=_key(task),
                            suite=suite_of(task),
                            level=lvl,
                            kappa=k,
                            beta=b,
                            fbeta=float(point[bi]),
                            fbeta_lo=float(lo[bi]),
                            fbeta_hi=float(hi[bi]),
                            fbeta_se=float(se[bi]),
                        )
                    )
    return pd.DataFrame(rows), store


def aggregate_fbeta_kappa(
    store,
    algo_labels,
    tasks,
    betas=BETAS,
    thresholds=None,
    n_bootstrap=N_BOOTSTRAP,
    confidence_level=CONFIDENCE_LEVEL,
    seed=BOOT_SEED,
):
    """Mean over tasks of per-task F-beta; stratified bootstrap (seeds resampled within task)."""
    a_ = (1 - confidence_level) / 2
    rows = []
    for label in algo_labels:
        NR = np.array([store[(label, _key(t))][0] for t in tasks])  # (tasks, seeds)
        C = np.array([store[(label, _key(t))][1] for t in tasks])
        n_tasks, n_seeds = NR.shape
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_seeds, size=(n_bootstrap, n_tasks, n_seeds))
        rows_ix = np.arange(n_tasks)[None, :, None]
        NR_b, C_b = NR[rows_ix, idx].mean(-1), C[rows_ix, idx].mean(-1)  # (B, tasks)
        for li, lvl in enumerate(LEVEL_NAMES):
            ks = np.array([thresholds_of(t, thresholds)[li] for t in tasks])
            point = _fbeta_kappa_of_means(NR.mean(1), C.mean(1), ks, betas).mean(-1)
            boot = _fbeta_kappa_of_means(NR_b, C_b, ks[None, :], betas).mean(-1)  # (K, B)
            lo = np.percentile(boot, 100 * a_, axis=1)
            hi = np.percentile(boot, 100 * (1 - a_), axis=1)
            se = boot.std(axis=1)
            for bi, b in enumerate(betas):
                rows.append(
                    dict(
                        algo=label,
                        level=lvl,
                        beta=b,
                        fbeta=float(point[bi]),
                        fbeta_lo=float(lo[bi]),
                        fbeta_hi=float(hi[bi]),
                        fbeta_se=float(se[bi]),
                    )
                )
    return pd.DataFrame(rows)


# ======================================================================================
# LaTeX tables
# ======================================================================================
_WORDS = {3: "three", 4: "four", 5: "five", 6: "six", 8: "eight", 10: "ten"}


def _spread_phrase(n_seeds=5, stat_mode=None):
    """Caption text describing what '+-' means."""
    stat_mode = stat_mode or STAT_MODE
    seeds = f"{_WORDS.get(n_seeds, n_seeds)} seeds" if n_seeds else "seeds"
    if stat_mode == "notebook":
        return (
            f"Values are bootstrap mean$\\pm$half-width of the 95\\% bootstrap confidence "
            f"interval over {seeds} ({N_BOOTSTRAP:,} resamples)."
        ).replace(",", "{,}", 1)
    return f"Values are mean$\\pm$std over {seeds}."


def _pm(m, sd, show_err):
    return f"{m:.2f}$\\pm${sd:.2f}" if show_err else f"{m:.2f}"


def _cell(row, a, best, show_err=True):
    txt = (
        f"{_pm(row.loc[a, 'norm_reward'], row.loc[a, 'norm_reward_err'], show_err)} / "
        f"{_pm(row.loc[a, 'norm_cost'], row.loc[a, 'norm_cost_err'], show_err)}"
    )
    if not row.loc[a, "safe"]:
        return f"\\textcolor{{gray}}{{{txt}}}"
    return f"\\textbf{{{txt}}}" if a == best else txt


def _is_success(r, min_reward=MIN_REWARD):
    """Constraint satisfied (C/kappa <= 1) AND non-trivial reward (R_norm > min_reward)."""
    return bool(r["safe"]) and float(r["norm_reward"]) > min_reward


def _best_safe(row, algos, min_reward=MIN_REWARD):
    """Highest-reward method among those that satisfy the constraint with R_norm > min_reward."""
    safe_r = {
        a: row.loc[a, "norm_reward"]
        for a in algos
        if a in row.index and _is_success(row.loc[a], min_reward)
    }
    return max(safe_r, key=safe_r.get) if safe_r else None


def latex_table_raw(
    df,
    algo_order=None,
    task_order=None,
    setting_name="",
    label="tab:raw_reward_cost",
    n_seeds=5,
):
    """Raw reward / cost per task (the format of Tables 5-10). Use this to cross-check that
    the numbers below are derived from exactly the same statistics as the appendix tables."""
    d = df[df["level"] == LEVEL_NAMES[0]]
    algos = algo_order or list(dict.fromkeys(d["algo"]))
    tasks = task_order or list(dict.fromkeys(d["task"]))
    L = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * len(algos) + "}",
        "\\toprule",
        "Environment & Metric & " + " & ".join(algos) + " \\\\",
        "\\midrule",
    ]
    for ti, t in enumerate(tasks):
        row = d[d["task"] == t].set_index("algo")
        for mi, (name, m, e) in enumerate(
            [("Reward $\\uparrow$", "raw_reward", "raw_reward_err"),
             ("Cost $\\downarrow$", "raw_cost", "raw_cost_err")]
        ):
            cells = [
                f"{row.loc[a, m]:.2f} $\\pm$ {row.loc[a, e]:.2f}" if a in row.index else "--"
                for a in algos
            ]
            first = f"\\multirow{{2}}{{*}}{{{t}}}" if mi == 0 else ""
            L.append(f"{first} & {name} & " + " & ".join(cells) + " \\\\")
        if ti < len(tasks) - 1:
            L.append("\\midrule")
    caption = (
        f"{setting_name + ': ' if setting_name else ''}raw reward and cost. "
        + _spread_phrase(n_seeds, "notebook")
    )
    L += ["\\bottomrule", "\\end{tabular}", "}", f"\\caption{{{caption}}}",
          f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(L)


def latex_table_all_thresholds(
    df,
    algo_order=None,
    task_order=None,
    setting_name="",
    label="tab:dsrl_all_thresholds",
    show_err=True,
    min_reward=MIN_REWARD,
    n_seeds=5,
):
    algos = algo_order or list(dict.fromkeys(df["algo"]))
    tasks = task_order or list(dict.fromkeys(df["task"]))
    caption = (
        f"{setting_name + ': ' if setting_name else ''}normalized reward / normalized cost "
        "($C/\\kappa$) at the strict, medium and loose thresholds. " + LEVEL_DEFINITION + " Each method is a single policy evaluated against every "
        "threshold, so its reward is identical across the rows of a task. Gray: constraint violated "
        "($C/\\kappa>1$). Bold: highest reward among methods satisfying the constraint at that threshold "
        f"(only methods with non-trivial reward, $R>{min_reward:g}$)."
    )
    if show_err:
        caption += " " + _spread_phrase(n_seeds)
    L = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * len(algos) + "}",
        "\\toprule",
        "Task & Threshold & " + " & ".join(algos) + " \\\\",
        "\\midrule",
    ]
    for ti, t in enumerate(tasks):
        for li, lvl in enumerate(LEVEL_NAMES):
            row = df[(df["task"] == t) & (df["level"] == lvl)].set_index("algo")
            best = _best_safe(row, algos, min_reward)
            cells = [_cell(row, a, best, show_err) if a in row.index else "--" for a in algos]
            name = f"\\multirow{{{len(LEVEL_NAMES)}}}{{*}}{{{t}}}" if li == 0 else ""
            L.append(f"{name} & {LEVEL_LABELS[lvl]} & " + " & ".join(cells) + " \\\\")
        if ti < len(tasks) - 1:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}", "}", f"\\caption{{{caption}}}",
          f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(L)


def latex_table_avg(
    df,
    algo_order=None,
    task_order=None,
    setting_name="",
    label="tab:dsrl_avg",
    show_err=True,
    min_reward=MIN_REWARD,
    n_seeds=5,
):
    algos = algo_order or list(dict.fromkeys(df["algo"]))
    tasks = task_order or list(dict.fromkeys(df["task"]))
    d = df[df["level"] == "avg"]
    caption = (
        f"{setting_name + ': ' if setting_name else ''}normalized reward / normalized cost averaged "
        "over the strict, medium and loose thresholds ($\\frac{1}{3}\\sum_\\kappa C/\\kappa$). "
        + LEVEL_DEFINITION + " Gray: constraint violated. "
        f"Bold: highest reward among methods satisfying the constraint with non-trivial reward ($R>{min_reward:g}$)."
    )
    if show_err:
        caption += " " + _spread_phrase(n_seeds)
    L = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(algos) + "}",
        "\\toprule",
        "Task & " + " & ".join(algos) + " \\\\",
        "\\midrule",
    ]
    for t in tasks:
        row = d[d["task"] == t].set_index("algo")
        best = _best_safe(row, algos, min_reward)
        L.append(
            f"{t} & "
            + " & ".join(_cell(row, a, best, show_err) if a in row.index else "--" for a in algos)
            + " \\\\"
        )
    L += ["\\bottomrule", "\\end{tabular}", "}", f"\\caption{{{caption}}}",
          f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(L)


def latex_safety_summary(
    summary,
    algo_order=None,
    setting_name="",
    label="tab:dsrl_safety_summary",
    min_reward=MIN_REWARD,
):
    algos = algo_order or list(dict.fromkeys(summary["algo"]))
    s = summary.set_index(["algo", "level"])
    best = {lvl: max(s.loc[(a, lvl), "n_success"] for a in algos) for lvl in LEVEL_NAMES}
    caption = (
        f"{setting_name + ': ' if setting_name else ''}number of tasks on which each method succeeds at each "
        "threshold (strict, medium, loose): the constraint is satisfied "
        f"($C/\\kappa\\le 1$) and the normalized reward is non-trivial ($R>{min_reward:g}$); mean normalized "
        "reward over those tasks in parentheses. " + LEVEL_DEFINITION + " Bold: most successes."
    )
    L = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(LEVEL_NAMES) + "}",
        "\\toprule",
        "Method & " + " & ".join(LEVEL_TITLES[l] for l in LEVEL_NAMES) + " \\\\",
        "\\midrule",
    ]
    for a in algos:
        cells = []
        for lvl in LEVEL_NAMES:
            r = s.loc[(a, lvl)]
            cnt = (
                r["success_tasks"]
                if r["n_success"] != best[lvl]
                else f"\\textbf{{{r['success_tasks']}}}"
            )
            rew = f"{r['reward_on_success_tasks']:.2f}" if r["n_success"] else "--"
            cells.append(f"{cnt} ({rew})")
        L.append(f"{a} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}",
          f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(L)


def latex_fbeta_kappa_table(
    agg,
    betas=(1, 2),
    algo_order=None,
    setting_name="",
    label="tab:fbeta_kappa",
    show_err=True,
):
    algos = algo_order or list(dict.fromkeys(agg["algo"]))
    cols = [(lvl, b) for lvl in LEVEL_NAMES for b in betas]
    v = agg.set_index(["algo", "level", "beta"])["fbeta"]
    se = agg.set_index(["algo", "level", "beta"])["fbeta_se"]
    best = {c: max(v.loc[(a, *c)] for a in algos) for c in cols}
    caption = (
        f"{setting_name + ': ' if setting_name else ''}threshold-aware F$_\\beta$-score averaged over "
        "tasks, with safety score $S_\\kappa=\\min(1,\\kappa/C)$ at each threshold. "
        + LEVEL_DEFINITION + " Bold: best per column."
    )
    if show_err:
        caption += " Values are mean$\\pm$bootstrap standard error over seeds."
    head1 = " & ".join(
        f"\\multicolumn{{{len(betas)}}}{{c}}{{{LEVEL_TITLES[l]}}}" for l in LEVEL_NAMES
    )
    head2 = " & ".join(f"$\\beta$={b:g}" for _, b in cols)
    rule = " ".join(
        f"\\cmidrule(lr){{{2 + i * len(betas)}-{1 + (i + 1) * len(betas)}}}"
        for i in range(len(LEVEL_NAMES))
    )
    L = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\toprule",
        f"Method & {head1} \\\\",
        rule,
        f" & {head2} \\\\",
        "\\midrule",
    ]
    for a in algos:
        cells = []
        for c in cols:
            x = v.loc[(a, *c)]
            txt = _pm(x, se.loc[(a, *c)], show_err)
            cells.append(f"\\textbf{{{txt}}}" if np.isclose(x, best[c]) else txt)
        L.append(f"{a} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "}", f"\\caption{{{caption}}}",
          f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(L)


def latex_fbeta_kappa_per_task(
    fb,
    beta=1,
    algo_order=None,
    task_order=None,
    setting_name="",
    label="tab:fbeta_kappa_per_task",
    show_err=True,
):
    """Per-task threshold-aware F-beta at every DSRL kappa (one table per beta)."""
    d = fb[np.isclose(fb["beta"].astype(float), float(beta))]
    algos = algo_order or list(dict.fromkeys(d["algo"]))
    tasks = task_order or list(dict.fromkeys(d["task"]))
    caption = (
        f"{setting_name + ': ' if setting_name else ''}per-task threshold-aware F$_\\beta$-score "
        f"($\\beta={beta:g}$) at the strict, medium and loose thresholds, with safety score $S_\\kappa=\\min(1,\\kappa/C)$ computed from the "
        "seed-averaged cost. " + LEVEL_DEFINITION + " Bold: best per row."
    )
    if show_err:
        caption += " Values are mean$\\pm$bootstrap standard error over seeds."
    L = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * len(algos) + "}",
        "\\toprule",
        "Task & Threshold & " + " & ".join(algos) + " \\\\",
        "\\midrule",
    ]
    for ti, t in enumerate(tasks):
        for li, lvl in enumerate(LEVEL_NAMES):
            row = d[(d["task"] == t) & (d["level"] == lvl)].set_index("algo")
            if row.empty:
                continue
            vals = {a: row.loc[a, "fbeta"] for a in algos if a in row.index}
            best = max(vals.values())
            cells = []
            for a in algos:
                if a not in vals:
                    cells.append("--")
                    continue
                x = _pm(vals[a], row.loc[a, "fbeta_se"], show_err)
                cells.append(f"\\textbf{{{x}}}" if np.isclose(vals[a], best) else x)
            name = f"\\multirow{{{len(LEVEL_NAMES)}}}{{*}}{{{t}}}" if li == 0 else ""
            L.append(f"{name} & {LEVEL_LABELS[lvl]} & " + " & ".join(cells) + " \\\\")
        if ti < len(tasks) - 1:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}", "}", f"\\caption{{{caption}}}",
          f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(L)


def _display_name(label, display_names=None, ours="PSIL"):
    """Header text: optional mapping, else strip a trailing '(LfP+N)' / '(LfP)' / '(LfN)' suffix.
    The method containing `ours` is shown bold as '<name> (ours)'."""
    if display_names and label in display_names:
        name = display_names[label]
    else:
        name = label
        for sfx in (" (LfP+N)", " (LfP)", " (LfN)"):
            if name.endswith(sfx):
                name = name[: -len(sfx)]
    if ours and ours in label:
        return f"\\textbf{{{name} (ours)}}"
    return name


def latex_table_main(
    df,
    algo_order=None,
    task_order=None,
    setting_name="LfP+N",
    min_reward=MIN_REWARD,
    display_names=None,
    ours="PSIL",
    n_seeds=5,
    label="tab:dsrl_all_lfpn",
    show_safe_reward=True,
):
    """
    Main per-threshold table (Tables 11-13): one block per suite, then a combined success count.
      * per task: row R = normalized reward, then one row C/kappa per DSRL kappa;
      * bold C/kappa: highest reward among methods with C/kappa <= 1 AND R_norm > min_reward;
      * '# successes': per threshold level, tasks with C/kappa <= 1 AND R_norm > min_reward.
    """
    d = df[df["level"].isin(LEVEL_NAMES)]
    algos = algo_order or list(dict.fromkeys(d["algo"]))
    tasks = task_order or list(dict.fromkeys(d["task"]))
    N = len(algos)

    def pm(m, s, bold=False):
        mean = f"\\textbf{{{m:.2f}}}" if bold else f"{m:.2f}"
        return f"{mean}{{\\scriptsize$\\pm${s:.2f}}}"

    suites = [("safetygym", "Safety-Gymnasium"), ("bullet", "Bullet-Gym")]
    success = {lvl: [0] * N for lvl in LEVEL_NAMES}
    n_tasks = 0

    L = [
        "% Requires: \\usepackage{booktabs, multirow, graphicx}",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * N + "}",
        "\\toprule",
        "Task & & " + " & ".join(_display_name(a, display_names, ours) for a in algos) + " \\\\",
    ]
    for suite, title in suites:
        st = [t for t in tasks if TASK_SUITE.get(_key(t)) == suite]
        if not st:
            continue
        kappas = SUITE_THRESHOLDS[suite]
        L += [
            "\\midrule",
            f"\\multicolumn{{{N + 2}}}{{l}}{{\\textit{{{title}}} ($\\kappa\\in\\{{{', '.join(f'{k:g}' for k in kappas)}\\}}$)}} \\\\",
            "\\midrule",
        ]
        for ti, t in enumerate(st):
            n_tasks += 1
            dt = d[d["task"] == _key(t)]
            first = dt[dt["level"] == LEVEL_NAMES[0]].set_index("algo")
            L.append(
                f"\\multirow{{{len(LEVEL_NAMES) + 1}}}{{*}}{{{_key(t)}}} & $R$ & "
                + " & ".join(
                    pm(first.loc[a, "norm_reward"], first.loc[a, "norm_reward_err"])
                    if a in first.index
                    else "--"
                    for a in algos
                )
                + " \\\\"
            )
            for lvl in LEVEL_NAMES:
                row = dt[dt["level"] == lvl].set_index("algo")
                best = _best_safe(row, algos, min_reward)
                cells = []
                for j, a in enumerate(algos):
                    if a not in row.index:
                        cells.append("--")
                        continue
                    r = row.loc[a]
                    if _is_success(r, min_reward):
                        success[lvl][j] += 1
                    cells.append(pm(r["norm_cost"], r["norm_cost_err"], bold=(a == best)))
                L.append(
                    f" & $C/\\kappa$, {LEVEL_LABELS[lvl]} & " + " & ".join(cells) + " \\\\"
                )
            if ti < len(st) - 1:
                L.append(f"\\cmidrule(lr){{1-{N + 2}}}")

    L += [
        "\\midrule",
        f"\\multicolumn{{{N + 2}}}{{l}}{{\\textit{{Number of successful tasks: $C/\\kappa\\le 1$ and "
        f"$R>{min_reward:g}$ (Safety-Gymnasium + Bullet-Gym)}}}} \\\\",
        "\\midrule",
    ]
    for i, lvl in enumerate(LEVEL_NAMES):
        best = max(success[lvl])
        cells = [
            f"\\textbf{{{n}/{n_tasks}}}" if n == best else f"{n}/{n_tasks}" for n in success[lvl]
        ]
        first = f"\\multirow{{{len(LEVEL_NAMES)}}}{{*}}{{\\# successes}}" if i == 0 else ""
        L.append(f"{first} & {LEVEL_LABELS[lvl]} & " + " & ".join(cells) + " \\\\")

    if show_safe_reward:
        st = success_table(df, algos, min_reward)
        L.append(f"\\cmidrule(lr){{1-{N + 2}}}")
        for i, lvl in enumerate(LEVEL_NAMES):
            vals = [round(float(st[(st.algo == a) & (st.level == lvl)]["safe_reward"].iloc[0]), 2) for a in algos]
            best = max(vals)
            cells = [f"\\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}" for v in vals]
            first = f"\\multirow{{{len(LEVEL_NAMES)}}}{{*}}{{Safe reward}}" if i == 0 else ""
            L.append(f"{first} & {LEVEL_LABELS[lvl]} & " + " & ".join(cells) + " \\\\")

    words = _WORDS.get(n_tasks, n_tasks)
    seeds = _WORDS.get(n_seeds, n_seeds)
    caption = (
        f"\\small Constraint satisfaction at every DSRL threshold in the {setting_name} setting. "
        "For each task, row $R$ gives the normalized reward and the rows below give the normalized cost "
        "$C/\\kappa$ at the strict, medium and loose thresholds. "
        "$C/\\kappa\\le 1$ means the constraint is satisfied. "
        f"Mean$\\pm$95\\% CI over {seeds} seeds. "
        "\\textit{A method succeeds on a task at a given threshold if it satisfies the constraint "
        f"($C/\\kappa\\le 1$) with non-trivial reward ($R>{min_reward:g}$); this excludes trivially safe "
        "policies that do not perform the task.} "
        "At each threshold, bold marks the highest-reward method among those that succeed. "
        f"The bottom rows summarize performance over all {words} tasks: \\#successes counts the tasks on "
        "which a method succeeds"
        + (
            f", and safe reward is the normalized reward averaged over the {words} tasks, with unsuccessful "
            "tasks counted as 0, so the value is high only if the method is safe and can perform the task."
            if show_safe_reward
            else "."
        )
    )
    L += ["\\bottomrule", "\\end{tabular}", "}", f"\\caption{{{caption}}}",
          f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(L) + "\n"


def _ordinal_rank(rank, tied):
    return f"{rank}{'=' if tied else ''}"


def latex_rank_summary(
    results,
    ours="PSIL",
    min_reward=MIN_REWARD,
    label="tab:psil_rank_summary",
):
    """
    One-glance summary across settings. `results` = {"LfP+N": res, "LfP": res, "LfN": res}
    (outputs of evaluate_setting). For each setting and threshold, shows PSIL's number of
    successful tasks and safe reward, with PSIL's rank among all methods of that setting in
    parentheses ('=' marks a tie), and the best competing value for reference.
    """
    L = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "c" * (2 * len(LEVEL_NAMES)) + "}",
        "\\toprule",
        " & \\multicolumn{3}{c}{\\# successful tasks (out of 10)} & \\multicolumn{3}{c}{Safe reward} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
        "Setting (\\# methods) & "
        + " & ".join(LEVEL_TITLES[l] for l in LEVEL_NAMES) + " & "
        + " & ".join(LEVEL_TITLES[l] for l in LEVEL_NAMES) + " \\\\",
        "\\midrule",
    ]
    for setting, res in results.items():
        st = success_table(res["metrics"], res["labels"], min_reward)
        psil = [a for a in res["labels"] if ours in a]
        if len(psil) != 1:
            raise ValueError(f"{setting}: expected exactly one method containing {ours!r}")
        psil = psil[0]
        cells_s, cells_r = [], []
        for lvl in LEVEL_NAMES:
            g = st[st.level == lvl]
            me = g[g.algo == psil].iloc[0]
            others = g[g.algo != psil]
            bs = int(others["n_success"].max())
            br = float(others["safe_reward"].max())
            rk, tie = me["rank_success"]
            v = f"\\textbf{{{me['n_success']}}}" if rk == 1 else f"{me['n_success']}"
            cells_s.append(f"{v} ({_ordinal_rank(rk, tie)}) {{\\scriptsize vs {bs}}}")
            rk, tie = me["rank_safe_reward"]
            v = f"\\textbf{{{me['safe_reward']:.2f}}}" if rk == 1 else f"{me['safe_reward']:.2f}"
            cells_r.append(f"{v} ({_ordinal_rank(rk, tie)}) {{\\scriptsize vs {br:.2f}}}")
        L.append(f"{setting} ({len(res['labels'])}) & " + " & ".join(cells_s + cells_r) + " \\\\")
    caption = (
        f"Summary of {ours} across the three settings. Each cell gives {ours}'s value, its rank among "
        "all methods of that setting in parentheses ($=$: tied), and, in small type, the best value "
        "among the other methods. A task is a success if $C/\\kappa\\le 1$ and "
        f"$R>{min_reward:g}$. Safe reward is the normalized reward averaged over the ten tasks, "
        "counting tasks without a success as 0. " + LEVEL_DEFINITION + " Bold: rank 1."
    )
    L += ["\\bottomrule", "\\end{tabular}", "}", f"\\caption{{{caption}}}",
          f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(L) + "\n"


def save_tex(text, path):
    with open(path, "w") as f:
        f.write(text)


# ======================================================================================
# Figures
# ======================================================================================
def plot_reward_vs_cost(df, level="mid", algo_order=None, task_order=None, ncols=5, fontsize=9):
    import matplotlib.pyplot as plt

    d = df[df["level"] == level]
    algos = algo_order or list(dict.fromkeys(d["algo"]))
    tasks = task_order or list(dict.fromkeys(d["task"]))
    colors, markers = notebook_colors_markers(len(algos))
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.2 * nrows), squeeze=False)
    for i, t in enumerate(tasks):
        ax = axs[i // ncols, i % ncols]
        sub = d[d["task"] == t].set_index("algo")
        xmax = max(1.5, float(sub["norm_cost_hi"].max()) * 1.1)
        ax.axvspan(1.0, xmax, color="0.9", zorder=0)
        ax.axvline(1.0, color="0.4", ls="--", lw=1)
        for j, a in enumerate(algos):
            if a not in sub.index:
                continue
            s = sub.loc[a]
            ax.errorbar(
                s["norm_cost"],
                s["norm_reward"],
                xerr=[[max(s["norm_cost"] - s["norm_cost_lo"], 0)],
                      [max(s["norm_cost_hi"] - s["norm_cost"], 0)]],
                yerr=[[max(s["norm_reward"] - s["norm_reward_lo"], 0)],
                      [max(s["norm_reward_hi"] - s["norm_reward"], 0)]],
                fmt=markers[j],
                ms=5,
                capsize=2,
                label=a,
                color=colors[j],
            )
        ax.set_xlim(0, xmax)
        kappa = sub["kappa"].iloc[0]
        ax.set_title(t if np.isnan(kappa) else f"{t} ({LEVEL_LABELS[level]})", fontsize=fontsize)
        ax.set_xlabel(r"Normalized cost $C/\kappa$", fontsize=fontsize)
        if i % ncols == 0:
            ax.set_ylabel("Normalized reward", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 1)
        ax.grid(alpha=0.3)
    for k in range(len(tasks), nrows * ncols):
        axs[k // ncols, k % ncols].axis("off")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(algos), fontsize=fontsize)
    fig.tight_layout()
    return fig, axs


def _beta_lines(ax, d, algos, betas):
    colors, markers = notebook_colors_markers(len(algos))
    x = np.arange(len(betas))
    for j, a in enumerate(algos):
        s = d[d["algo"] == a].set_index("beta").loc[list(betas)]
        ax.errorbar(
            x,
            s["fbeta"],
            yerr=[s["fbeta"] - s["fbeta_lo"], s["fbeta_hi"] - s["fbeta"]],
            label=a,
            color=colors[j],
            marker=markers[j],
            capsize=5,
            lw=2,
            alpha=0.8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.grid(alpha=0.3)


def plot_fbeta_kappa_aggregate(agg, algo_order=None, fontsize=10):
    import matplotlib.pyplot as plt

    algos = algo_order or list(dict.fromkeys(agg["algo"]))
    betas = sorted(agg["beta"].unique())
    fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.3), sharey=True)
    for ax, lvl in zip(axs, LEVEL_NAMES):
        _beta_lines(ax, agg[agg["level"] == lvl], algos, betas)
        ax.set_title(LEVEL_TITLES[lvl], fontsize=fontsize)
        ax.set_xlabel(r"$\beta$", fontsize=fontsize)
    axs[0].set_ylabel(r"F$_\beta^\kappa$-score", fontsize=fontsize)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(algos), fontsize=fontsize - 1)
    fig.tight_layout()
    return fig, axs


def plot_fbeta_kappa_per_task(fb, level="mid", algo_order=None, task_order=None, ncols=5, fontsize=9):
    import matplotlib.pyplot as plt

    d = fb[fb["level"] == level]
    algos = algo_order or list(dict.fromkeys(d["algo"]))
    tasks = task_order or list(dict.fromkeys(d["task"]))
    betas = sorted(d["beta"].unique())
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.0 * nrows), squeeze=False)
    for i, t in enumerate(tasks):
        ax = axs[i // ncols, i % ncols]
        dt = d[d["task"] == t]
        _beta_lines(ax, dt, algos, betas)
        ax.set_title(f"{t} ({LEVEL_LABELS[level]})", fontsize=fontsize)
        ax.set_xlabel(r"$\beta$", fontsize=fontsize)
        if i % ncols == 0:
            ax.set_ylabel(r"F$_\beta^\kappa$-score", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 1)
    for k in range(len(tasks), nrows * ncols):
        axs[k // ncols, k % ncols].axis("off")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(algos), fontsize=fontsize)
    fig.tight_layout()
    return fig, axs


# ======================================================================================
# R2: threshold-conditioned D_P / D_N / D_U for the kappa-sweep experiment
# ======================================================================================
def build_threshold_datasets(
    traj_returns,
    traj_costs,
    kappa,
    n_neg=10,
    n_unlabeled=500,
    low_reward_frac=0.01,
    seed=0,
):
    """
    Appendix-A construction made relative to kappa:
      D_P: highest-return trajectory with cost <= kappa
      D_N: half from the lowest-return `low_reward_frac`, half from trajectories with cost > kappa
      D_U: uniform sample of the remaining trajectories
    Returns index arrays into the full trajectory pool.
    """
    rng = np.random.default_rng(seed)
    R, C = np.asarray(traj_returns, float), np.asarray(traj_costs, float)
    idx = np.arange(len(R))
    feasible = idx[C <= kappa]
    if len(feasible) == 0:
        raise ValueError(f"No trajectory satisfies cost <= {kappa}.")
    pos = np.array([feasible[np.argmax(R[feasible])]])
    n_low, n_unsafe = n_neg // 2, n_neg - n_neg // 2
    low_pool = np.setdiff1d(
        np.argsort(R)[: max(n_low, int(np.ceil(low_reward_frac * len(R))))], pos
    )
    unsafe_pool = np.setdiff1d(idx[C > kappa], np.concatenate([pos, low_pool]))
    if len(unsafe_pool) < n_unsafe:
        raise ValueError(f"Only {len(unsafe_pool)} trajectories with cost > {kappa}.")
    neg = np.concatenate(
        [
            rng.choice(low_pool, n_low, replace=False),
            rng.choice(unsafe_pool, n_unsafe, replace=False),
        ]
    )
    rest = np.setdiff1d(idx, np.concatenate([pos, neg]))
    return {
        "preferred": pos,
        "non_preferred": neg,
        "unlabeled": rng.choice(rest, min(n_unlabeled, len(rest)), replace=False),
    }


# ======================================================================================
# Notebook entry points
# ======================================================================================
def evaluate_setting(
    hyperparams, algos, tasks, load_fn, betas=BETAS, thresholds=None, stat_mode=None
):
    """
    Compute everything for one setting. Returns a dict of DataFrames:
      metrics    per (method, task, level incl. 'avg'): R_norm, C/kappa (+ err, CI, std),
                 raw reward/cost (= Tables 5-10), violation rate, safe
      summary    #successes / #safe tasks per (method, level)
      fbeta      threshold-aware F-beta per (method, task, level, beta)
      fbeta_agg  same, averaged over tasks with stratified bootstrap CI
    plus 'labels' (method order) and 'tasks'.
    """
    labels = [l for _, l in algos]
    metrics = evaluate_algorithms(
        hyperparams, algos, tasks, load_fn, thresholds, stat_mode=stat_mode
    )
    fb, store = evaluate_fbeta_kappa(hyperparams, algos, tasks, load_fn, betas, thresholds)
    return dict(
        metrics=metrics,
        summary=safety_summary(metrics, labels),
        fbeta=fb,
        fbeta_agg=aggregate_fbeta_kappa(store, labels, tasks, betas, thresholds),
        labels=labels,
        tasks=[_key(t) for t in tasks],
    )


def generate_tables(
    res,
    setting_name="",
    fbeta_betas=(1, 2),
    label_suffix="",
    show_err=True,
    min_reward=MIN_REWARD,
    display_names=None,
    ours="PSIL",
    n_seeds=5,
):
    """
    LaTeX strings from the output of `evaluate_setting`.
      main             per-suite blocks: R row + C/kappa row per kappa, combined #successes
      raw              raw reward / cost (format of Tables 5-10, for cross-checking)
      all_thresholds   per task, one row per DSRL kappa: R_norm / C_norm
      summary          #successes (mean reward on those tasks) per kappa level
      fbeta_kappa      aggregated threshold-aware F-beta per kappa level
      fbeta_kappa_per_task_b<beta>
      avg_threshold    DSRL average-over-thresholds table
    """
    labels, tasks = res["labels"], res["tasks"]
    sfx = label_suffix or setting_name.lower().replace("+", "").replace(" ", "")
    sfx = f"_{sfx}" if sfx else ""
    betas = tuple(b for b in fbeta_betas if b in set(res["fbeta_agg"]["beta"]))
    out = {
        "main": latex_table_main(
            res["metrics"], labels, tasks, setting_name or "LfP+N", min_reward,
            display_names, ours, n_seeds, f"tab:dsrl_main{sfx}",
        ),
        "raw": latex_table_raw(
            res["metrics"], labels, tasks, setting_name, f"tab:raw{sfx}", n_seeds
        ),
        "all_thresholds": latex_table_all_thresholds(
            res["metrics"], labels, tasks, setting_name, f"tab:dsrl_all{sfx}",
            show_err, min_reward, n_seeds,
        ),
        "summary": latex_safety_summary(
            safety_summary(res["metrics"], labels, min_reward), labels, setting_name,
            f"tab:dsrl_summary{sfx}", min_reward,
        ),
        "fbeta_kappa": latex_fbeta_kappa_table(
            res["fbeta_agg"], betas, labels, setting_name, f"tab:fbeta_kappa{sfx}", show_err,
        ),
        "avg_threshold": latex_table_avg(
            res["metrics"], labels, tasks, setting_name, f"tab:dsrl_avg{sfx}",
            show_err, min_reward, n_seeds,
        ),
    }
    for b in betas:
        out[f"fbeta_kappa_per_task_b{b:g}"] = latex_fbeta_kappa_per_task(
            res["fbeta"], b, labels, tasks, setting_name, f"tab:fbeta_kappa_task_b{b:g}{sfx}", show_err,
        )
    return out
