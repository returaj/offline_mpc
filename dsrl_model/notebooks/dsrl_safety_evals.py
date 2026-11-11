"""
dsrl_safety_eval.py  --  DSRL-protocol safety evaluation for the PSIL rebuttal (R1 / R2)
=======================================================================================

Notebook usage (each step is its own cell; nothing is saved to disk unless you do it):

    import dsrl_safety_eval as E

    hyperparams = "neg_union_full/five_seed/only_for_reporting"
    key = "LfPN"
    ALGOS = E.SETTINGS[key]          # or your own [(path, label), ...]
    tasks = E.TASKS

    # 1) compute metrics (uses your notebook's load_data_as_array)
    res = E.evaluate_setting(hyperparams, ALGOS, tasks, load_data_as_array)
    res["summary"]                   # pandas DataFrames: metrics, summary, fbeta, fbeta_agg

    # 2) tables -> dict of LaTeX strings (print them, or E.save_tex(text, path))
    tables = E.generate_tables(res, setting_name="LfP+N")
    print(tables["all_thresholds"])

    # 3) figures -> (fig, axs), exactly like your notebook; save in your save-fig cell
    fig, ax = E.plot_reward_vs_cost(res["metrics"], level="mid")
    fig, ax = E.plot_fbeta_kappa_aggregate(res["fbeta_agg"])
    fig, ax = E.plot_fbeta_kappa_per_task(res["fbeta"], level="mid")

Definitions
-----------
  R_norm   = (R - R_min) / (R_max - R_min)                 DSRL reward bounds, clipped
  C_norm   = C / kappa                                     <= 1  <=>  constraint satisfied
  S_kappa  = 1 if C <= kappa else kappa / C                safety score for F-beta
  F_beta^k = (1 + b^2) R_norm S_kappa / (b^2 R_norm + S_kappa)

Thresholds follow the OSRL/DSRL protocol: Safety-Gymnasium {20, 40, 80}, Bullet-Gym
{10, 20, 40}. Verify against the OSRL configs you cite (edit SUITE_THRESHOLDS if needed).
LaTeX tables need \\usepackage{booktabs, xcolor, multirow, graphicx}.

The logs store per seed the mean cost over the evaluation episodes, so 'violation_rate'
is the fraction of seeds with cost > kappa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EP = 1e-8

# ======================================================================================
# Configuration
# ======================================================================================
TASKS = ["PointCircle2", "PointGoal1", "CarCircle2", "CarGoal1", "AntVelocity",
         "SwimmerVelocity", "AntRun", "DroneRun", "AntCircle", "DroneCircle"]

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
        ("safecl_lfn/traj", "PSIL (LfN)"),
    ],
}

SUITE_THRESHOLDS = {"safetygym": (20.0, 40.0, 80.0), "bullet": (10.0, 20.0, 40.0)}
LEVEL_NAMES = ("strict", "mid", "loose")
LEVEL_TITLES = {"strict": r"Strict ($\kappa$=20 / 10)", "mid": r"Mid ($\kappa$=40 / 20)",
                "loose": r"Loose ($\kappa$=80 / 40)"}

TASK_SUITE = {
    "PointCircle2": "safetygym", "CarCircle2": "safetygym", "PointGoal1": "safetygym",
    "CarGoal1": "safetygym", "AntVelocity": "safetygym", "SwimmerVelocity": "safetygym",
    "AntRun": "bullet", "DroneRun": "bullet", "AntCircle": "bullet", "DroneCircle": "bullet",
}

# Table 2 of the paper (dsrl/infos.py); used only if `dsrl` cannot be imported.
FALLBACK_REWARD_BOUNDS = {
    "PointCircle2": (20.48, 54.02), "PointGoal1": (0.01, 30.07),
    "CarCircle2": (5.11, 25.95), "CarGoal1": (0.01, 39.91),
    "AntVelocity": (6.15, 2976.28), "SwimmerVelocity": (0.07, 238.96),
    "AntRun": (0.0, 955.48), "DroneRun": (10.56, 682.83),
    "AntCircle": (0.02, 460.71), "DroneCircle": (207.79, 996.39),
}

MIN_REWARD = 0.05  # R_norm must exceed this to count as a success / to be bolded (trivially safe policies excluded)

STD_DDOF = 0   # np.std default (population std over seeds); set 1 for sample std

BETAS = (0.5, 1, 1.5, 2, 5)

# Same palette / markers as safecl_plotting.ipynb
_CMKY = {"cornflowerblue": "#4F81BD", "orange": "#E69F00", "coral": "#D97662",
         "orchid": "#B57BB3", "olive": "#7A8F00", "slate": "#6C757D", "brown": "#8C564B",
         "vermillion": "#D55E00"}
BASE_COLORS = ["black", _CMKY["cornflowerblue"], _CMKY["orange"], _CMKY["olive"],
               _CMKY["orchid"], _CMKY["slate"], _CMKY["cornflowerblue"], _CMKY["brown"],
               _CMKY["vermillion"], _CMKY["coral"]]
MARKERS = [None, "o", "s", "D", "p", "X", "*", "P", "h", ">", "d", "<", "+", "x", "H", "v", "^"]


def notebook_colors_markers(n):
    """Reproduces `BASE_COLORS[2:n+1] + [BASE_COLORS[-1]]` from the notebook."""
    return BASE_COLORS[2:n + 1] + [BASE_COLORS[-1]], MARKERS[2:n + 1] + [MARKERS[-1]]


# ======================================================================================
# Data access (load_fn = load_data_as_array from your notebook)
# ======================================================================================
def load_final(hyperparams, algo, task, load_fn, step=-1):
    """Final evaluation (after 1M steps): per-seed reward and cost."""
    r = load_fn(hyperparams, algo, task, "ep_reward")[:, step]
    c = load_fn(hyperparams, algo, task, "ep_cost")[:, step]
    return np.asarray(r, dtype=float), np.asarray(c, dtype=float)


# ======================================================================================
# Core metrics
# ======================================================================================
def _key(task):
    return task.replace("_", "")


def suite_of(task):
    return TASK_SUITE[_key(task)]


def thresholds_of(task, thresholds=None):
    return tuple(float(k) for k in thresholds) if thresholds is not None else SUITE_THRESHOLDS[suite_of(task)]


def get_reward_bounds(task):
    key = _key(task)
    try:
        import dsrl.infos as dsrl_infos
        return float(dsrl_infos.MIN_EPISODE_REWARD[key]), float(dsrl_infos.MAX_EPISODE_REWARD[key])
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


def bootstrap_mean_ci(x, n_bootstrap=1000, confidence_level=0.95, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, len(x), size=(n_bootstrap, len(x)))].mean(axis=1)
    a = (1 - confidence_level) / 2
    return float(x.mean()), float(np.percentile(boot, 100 * a)), float(np.percentile(boot, 100 * (1 - a)))


# ======================================================================================
# Constraint satisfaction (R1)
# ======================================================================================
def evaluate_task(rewards, costs, task, thresholds=None, use_upper_ci=False, n_bootstrap=1000, seed=0):
    """Rows for each kappa level plus the DSRL 'avg' row (C/kappa averaged over thresholds)."""
    rewards, costs = np.asarray(rewards, float), np.asarray(costs, float)
    kappas = thresholds_of(task, thresholds)
    nr = normalize_reward(rewards, task)
    r_m, r_lo, r_hi = bootstrap_mean_ci(nr, n_bootstrap, seed=seed)

    def row(level, kappa, nc, vr):
        c_m, c_lo, c_hi = bootstrap_mean_ci(nc, n_bootstrap, seed=seed)
        return dict(task=_key(task), suite=suite_of(task), level=level, kappa=kappa,
                    norm_reward=r_m, norm_reward_lo=r_lo, norm_reward_hi=r_hi,
                    norm_cost=c_m, norm_cost_lo=c_lo, norm_cost_hi=c_hi,
                    norm_reward_std=float(np.std(nr, ddof=STD_DDOF)),
                    norm_cost_std=float(np.std(nc, ddof=STD_DDOF)),
                    raw_cost=float(costs.mean()), raw_cost_std=float(np.std(costs, ddof=STD_DDOF)), violation_rate=vr,
                    safe=bool((c_hi if use_upper_ci else c_m) <= 1.0))

    rows, per_k = [], []
    for lvl, k in zip(LEVEL_NAMES, kappas):
        nc = normalized_cost(costs, k)
        per_k.append(nc)
        rows.append(row(lvl, k, nc, violation_rate(costs, k)))
    rows.append(row("avg", np.nan, np.mean(per_k, axis=0),
                    float(np.mean([violation_rate(costs, k) for k in kappas]))))
    return rows


def evaluate_algorithms(hyperparams, algos, tasks, load_fn, thresholds=None, **kwargs):
    rows = []
    for path, label in algos:
        for task in tasks:
            r, c = load_final(hyperparams, path, task, load_fn)
            for rw in evaluate_task(r, c, task, thresholds, **kwargs):
                rw["algo"] = label
                rows.append(rw)
    cols = ["algo", "task", "suite", "level", "kappa", "norm_reward", "norm_reward_lo",
            "norm_reward_hi", "norm_reward_std", "norm_cost", "norm_cost_lo", "norm_cost_hi",
            "norm_cost_std", "raw_cost", "raw_cost_std", "violation_rate", "safe"]
    return pd.DataFrame(rows)[cols]


def safety_summary(df, algo_order=None, min_reward=MIN_REWARD):
    algos = algo_order or list(dict.fromkeys(df["algo"]))
    rows = []
    for a in algos:
        for lvl in list(LEVEL_NAMES) + ["avg"]:
            g = df[(df["algo"] == a) & (df["level"] == lvl)]
            safe = g["safe"].to_numpy()
            succ = (g["safe"] & (g["norm_reward"] > min_reward)).to_numpy()
            rows.append(dict(
                n_success=int(succ.sum()), success_tasks=f"{int(succ.sum())}/{len(g)}",
                reward_on_success_tasks=g.loc[succ, "norm_reward"].mean() if succ.any() else np.nan,
                algo=a, level=lvl, n_safe=int(safe.sum()), n_tasks=len(g),
                safe_tasks=f"{int(safe.sum())}/{len(g)}",
                reward_on_safe_tasks=g.loc[g["safe"], "norm_reward"].mean() if safe.any() else np.nan,
                reward_unsafe_as_zero=float(np.where(safe, g["norm_reward"], 0.0).mean()),
                mean_norm_cost=g["norm_cost"].mean(),
                mean_violation_rate=g["violation_rate"].mean()))
    return pd.DataFrame(rows)


# ======================================================================================
# Threshold-aware F-beta (complementary trade-off summary)
# ======================================================================================
def _fbeta_kappa_of_means(r_mean, c_mean, kappa, betas):
    b = np.asarray(betas, dtype=float).reshape(-1, *([1] * np.ndim(r_mean)))
    s = kappa_safety_score(c_mean, kappa)
    return (1 + b**2) * r_mean * s / np.maximum(b**2 * r_mean + s, EP)


def evaluate_fbeta_kappa(hyperparams, algos, tasks, load_fn, betas=BETAS, thresholds=None,
                         n_bootstrap=1000, confidence_level=0.95, seed=0):
    """F-beta from seed-mean R and C; CI by bootstrapping seeds (shared indices)."""
    rng = np.random.default_rng(seed)
    a_ = (1 - confidence_level) / 2
    rows, store = [], {}
    for path, label in algos:
        for task in tasks:
            r, c = load_final(hyperparams, path, task, load_fn)
            nr = normalize_reward(r, task)
            store[(label, _key(task))] = (nr, c)
            idx = rng.integers(0, len(nr), size=(n_bootstrap, len(nr)))
            for lvl, k in zip(LEVEL_NAMES, thresholds_of(task, thresholds)):
                point = _fbeta_kappa_of_means(nr.mean(), c.mean(), k, betas)
                boot = _fbeta_kappa_of_means(nr[idx].mean(1), c[idx].mean(1), k, betas)
                lo, hi = np.percentile(boot, 100 * a_, axis=1), np.percentile(boot, 100 * (1 - a_), axis=1)
                se = boot.std(axis=1)
                for bi, b in enumerate(betas):
                    rows.append(dict(algo=label, task=_key(task), suite=suite_of(task), level=lvl,
                                     kappa=k, beta=b, fbeta=float(point[bi]),
                                     fbeta_lo=float(lo[bi]), fbeta_hi=float(hi[bi]),
                                     fbeta_se=float(se[bi])))
    return pd.DataFrame(rows), store


def aggregate_fbeta_kappa(store, algo_labels, tasks, betas=BETAS, thresholds=None,
                          n_bootstrap=1000, confidence_level=0.95, seed=0):
    """Mean over tasks of per-task F-beta; stratified bootstrap (seeds resampled within task)."""
    rng = np.random.default_rng(seed)
    a_ = (1 - confidence_level) / 2
    rows = []
    for label in algo_labels:
        for li, lvl in enumerate(LEVEL_NAMES):
            pts, boots = [], []
            for task in tasks:
                nr, c = store[(label, _key(task))]
                k = thresholds_of(task, thresholds)[li]
                idx = rng.integers(0, len(nr), size=(n_bootstrap, len(nr)))
                pts.append(_fbeta_kappa_of_means(nr.mean(), c.mean(), k, betas))
                boots.append(_fbeta_kappa_of_means(nr[idx].mean(1), c[idx].mean(1), k, betas))
            point, boot = np.mean(pts, axis=0), np.mean(boots, axis=0)
            lo, hi = np.percentile(boot, 100 * a_, axis=1), np.percentile(boot, 100 * (1 - a_), axis=1)
            se = boot.std(axis=1)
            for bi, b in enumerate(betas):
                rows.append(dict(algo=label, level=lvl, beta=b, fbeta=float(point[bi]),
                                 fbeta_lo=float(lo[bi]), fbeta_hi=float(hi[bi]),
                                     fbeta_se=float(se[bi])))
    return pd.DataFrame(rows)


# ======================================================================================
# LaTeX tables
# ======================================================================================
def _pm(m, sd, show_std):
    return f"{m:.2f}$\\pm${sd:.2f}" if show_std else f"{m:.2f}"


def _cell(row, a, best, show_std=True):
    txt = (f"{_pm(row.loc[a, 'norm_reward'], row.loc[a, 'norm_reward_std'], show_std)} / "
           f"{_pm(row.loc[a, 'norm_cost'], row.loc[a, 'norm_cost_std'], show_std)}")
    if not row.loc[a, "safe"]:
        return f"\\textcolor{{gray}}{{{txt}}}"
    return f"\\textbf{{{txt}}}" if a == best else txt


def _is_success(r, min_reward=MIN_REWARD):
    """Constraint satisfied (C/kappa <= 1) AND non-trivial reward (R_norm > min_reward)."""
    return bool(r["safe"]) and float(r["norm_reward"]) > min_reward


def _best_safe(row, algos, min_reward=MIN_REWARD):
    """Highest-reward method among those that satisfy the constraint with R_norm > min_reward."""
    safe_r = {a: row.loc[a, "norm_reward"] for a in algos if a in row.index and _is_success(row.loc[a], min_reward)}
    return max(safe_r, key=safe_r.get) if safe_r else None


def latex_table_all_thresholds(df, algo_order=None, task_order=None, setting_name="",
                               label="tab:dsrl_all_thresholds", show_std=True, min_reward=MIN_REWARD):
    algos = algo_order or list(dict.fromkeys(df["algo"]))
    tasks = task_order or list(dict.fromkeys(df["task"]))
    caption = (f"{setting_name + ': ' if setting_name else ''}normalized reward / normalized cost "
               "($C/\\kappa$) at each DSRL threshold ($\\kappa\\in\\{20,40,80\\}$ for Safety-Gymnasium, "
               "$\\{10,20,40\\}$ for Bullet-Gym). Each method is a single policy evaluated against every "
               "threshold, so its reward is identical across the rows of a task. Gray: constraint violated "
               "($C/\\kappa>1$). Bold: highest reward among methods satisfying the constraint at that threshold "
               f"(only methods with non-trivial reward, $R>{min_reward:g}$).")
    caption += (" Values are mean$\\pm$std over seeds." if show_std else " Values are means over seeds.")
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\resizebox{\\linewidth}{!}{%",
         "\\begin{tabular}{ll" + "c" * len(algos) + "}", "\\toprule",
         "Task & $\\kappa$ & " + " & ".join(algos) + " \\\\", "\\midrule"]
    for ti, t in enumerate(tasks):
        for li, lvl in enumerate(LEVEL_NAMES):
            row = df[(df["task"] == t) & (df["level"] == lvl)].set_index("algo")
            best = _best_safe(row, algos, min_reward)
            cells = [_cell(row, a, best, show_std) if a in row.index else "--" for a in algos]
            name = f"\\multirow{{{len(LEVEL_NAMES)}}}{{*}}{{{t}}}" if li == 0 else ""
            L.append(f"{name} & {row['kappa'].iloc[0]:g} & " + " & ".join(cells) + " \\\\")
        if ti < len(tasks) - 1:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}", "}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
          "\\end{table}"]
    return "\n".join(L)


def latex_table_avg(df, algo_order=None, task_order=None, setting_name="", label="tab:dsrl_avg",
                    show_std=True, min_reward=MIN_REWARD):
    algos = algo_order or list(dict.fromkeys(df["algo"]))
    tasks = task_order or list(dict.fromkeys(df["task"]))
    d = df[df["level"] == "avg"]
    caption = (f"{setting_name + ': ' if setting_name else ''}normalized reward / normalized cost averaged "
               "over the DSRL thresholds ($\\frac{1}{3}\\sum_\\kappa C/\\kappa$). Gray: constraint violated. "
               f"Bold: highest reward among methods satisfying the constraint with non-trivial reward ($R>{min_reward:g}$).")
    caption += (" Values are mean$\\pm$std over seeds." if show_std else " Values are means over seeds.")
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\resizebox{\\linewidth}{!}{%",
         "\\begin{tabular}{l" + "c" * len(algos) + "}", "\\toprule",
         "Task & " + " & ".join(algos) + " \\\\", "\\midrule"]
    for t in tasks:
        row = d[d["task"] == t].set_index("algo")
        best = _best_safe(row, algos, min_reward)
        L.append(f"{t} & " + " & ".join(_cell(row, a, best, show_std) if a in row.index else "--" for a in algos) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
          "\\end{table}"]
    return "\n".join(L)


def latex_safety_summary(summary, algo_order=None, setting_name="", label="tab:dsrl_safety_summary",
                         min_reward=MIN_REWARD):
    algos = algo_order or list(dict.fromkeys(summary["algo"]))
    s = summary.set_index(["algo", "level"])
    best = {lvl: max(s.loc[(a, lvl), "n_success"] for a in algos) for lvl in LEVEL_NAMES}
    caption = (f"{setting_name + ': ' if setting_name else ''}number of tasks on which each method succeeds at each "
               "DSRL threshold (Safety-Gymnasium / Bullet-Gym $\\kappa$): the constraint is satisfied "
               f"($C/\\kappa\\le 1$) and the normalized reward is non-trivial ($R>{min_reward:g}$); mean normalized "
               "reward over those tasks in parentheses. Bold: most successes.")
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{l" + "c" * len(LEVEL_NAMES) + "}",
         "\\toprule", "Method & " + " & ".join(LEVEL_TITLES[l] for l in LEVEL_NAMES) + " \\\\", "\\midrule"]
    for a in algos:
        cells = []
        for lvl in LEVEL_NAMES:
            r = s.loc[(a, lvl)]
            cnt = r["success_tasks"] if r["n_success"] != best[lvl] else f"\\textbf{{{r['success_tasks']}}}"
            rew = f"{r['reward_on_success_tasks']:.2f}" if r["n_success"] else "--"
            cells.append(f"{cnt} ({rew})")
        L.append(f"{a} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(L)


def latex_fbeta_kappa_table(agg, betas=(1, 2), algo_order=None, setting_name="", label="tab:fbeta_kappa",
                            show_std=True):
    algos = algo_order or list(dict.fromkeys(agg["algo"]))
    cols = [(lvl, b) for lvl in LEVEL_NAMES for b in betas]
    v = agg.set_index(["algo", "level", "beta"])["fbeta"]
    se = agg.set_index(["algo", "level", "beta"])["fbeta_se"]
    best = {c: max(v.loc[(a, *c)] for a in algos) for c in cols}
    caption = (f"{setting_name + ': ' if setting_name else ''}threshold-aware F$_\\beta$-score averaged over "
               "tasks, with safety score $S_\\kappa=\\min(1,\\kappa/C)$ at each DSRL threshold "
               "(Safety-Gymnasium / Bullet-Gym $\\kappa$). Bold: best per column.")
    caption += (" Values are mean$\\pm$bootstrap standard error over seeds." if show_std else "")
    head1 = " & ".join(f"\\multicolumn{{{len(betas)}}}{{c}}{{{LEVEL_TITLES[l]}}}" for l in LEVEL_NAMES)
    head2 = " & ".join(f"$\\beta$={b:g}" for _, b in cols)
    rule = " ".join(f"\\cmidrule(lr){{{2 + i * len(betas)}-{1 + (i + 1) * len(betas)}}}"
                    for i in range(len(LEVEL_NAMES)))
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\resizebox{\\linewidth}{!}{%",
         "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
         f"Method & {head1} \\\\", rule, f" & {head2} \\\\", "\\midrule"]
    for a in algos:
        cells = []
        for c in cols:
            x = v.loc[(a, *c)]
            txt = _pm(x, se.loc[(a, *c)], show_std)
            cells.append(f"\\textbf{{{txt}}}" if np.isclose(x, best[c]) else txt)
        L.append(f"{a} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
          "\\end{table}"]
    return "\n".join(L)


def latex_fbeta_kappa_per_task(fb, beta=1, algo_order=None, task_order=None, setting_name="",
                               label="tab:fbeta_kappa_per_task", show_std=True):
    """
    Per-task threshold-aware F-beta at every DSRL kappa (one table per beta):
    rows = task x kappa, columns = methods. Bold: best per row.
    """
    d = fb[np.isclose(fb["beta"].astype(float), float(beta))]
    algos = algo_order or list(dict.fromkeys(d["algo"]))
    tasks = task_order or list(dict.fromkeys(d["task"]))
    caption = (f"{setting_name + ': ' if setting_name else ''}per-task threshold-aware F$_\\beta$-score "
               f"($\\beta={beta:g}$) at each DSRL threshold ($\\kappa\\in\\{{20,40,80\\}}$ for Safety-Gymnasium, "
               "$\\{10,20,40\\}$ for Bullet-Gym), with safety score $S_\\kappa=\\min(1,\\kappa/C)$ computed from the "
               "seed-averaged cost. Bold: best per row.")
    caption += (" Values are mean$\\pm$bootstrap standard error over seeds." if show_std else "")
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\resizebox{\\linewidth}{!}{%",
         "\\begin{tabular}{ll" + "c" * len(algos) + "}", "\\toprule",
         "Task & $\\kappa$ & " + " & ".join(algos) + " \\\\", "\\midrule"]
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
                x = _pm(vals[a], row.loc[a, "fbeta_se"], show_std)
                cells.append(f"\\textbf{{{x}}}" if np.isclose(vals[a], best) else x)
            name = f"\\multirow{{{len(LEVEL_NAMES)}}}{{*}}{{{t}}}" if li == 0 else ""
            L.append(f"{name} & {row['kappa'].iloc[0]:g} & " + " & ".join(cells) + " \\\\")
        if ti < len(tasks) - 1:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}", "}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
          "\\end{table}"]
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


def latex_table_main(df, algo_order=None, task_order=None, setting_name="LfP+N", min_reward=MIN_REWARD,
                     display_names=None, ours="PSIL", n_seeds=5, label="tab:dsrl_all_lfpn"):
    """
    Main per-threshold table (one block per suite, then a combined success count):
      * per task: row R = normalized reward (mean+-std), then one row C/kappa (mean+-std) per DSRL kappa;
      * bold C/kappa: highest reward among methods with C/kappa <= 1 AND R_norm > min_reward at that kappa;
      * '# successes': over all tasks, per threshold level (strict / mid / loose), counts tasks with
        C/kappa <= 1 AND R_norm > min_reward (trivially safe, non-performing policies are not counted).
    Needs booktabs, multirow and graphicx.
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

    L = ["% Requires: \\usepackage{booktabs, multirow, graphicx}",
         "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
         "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{ll" + "c" * N + "}", "\\toprule",
         "Task & & " + " & ".join(_display_name(a, display_names, ours) for a in algos) + " \\\\"]
    for suite, title in suites:
        st = [t for t in tasks if TASK_SUITE.get(_key(t)) == suite]
        if not st:
            continue
        kappas = SUITE_THRESHOLDS[suite]
        L += ["\\midrule",
              f"\\multicolumn{{{N + 2}}}{{l}}{{\\textit{{{title}}} ($\\kappa\\in\\{{{', '.join(f'{k:g}' for k in kappas)}\\}}$)}} \\\\",
              "\\midrule"]
        for ti, t in enumerate(st):
            n_tasks += 1
            dt = d[d["task"] == _key(t)]
            first = dt[dt["level"] == LEVEL_NAMES[0]].set_index("algo")
            L.append(f"\\multirow{{{len(LEVEL_NAMES) + 1}}}{{*}}{{{_key(t)}}} & $R$ & " +
                     " & ".join(pm(first.loc[a, "norm_reward"], first.loc[a, "norm_reward_std"])
                                if a in first.index else "--" for a in algos) + " \\\\")
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
                    cells.append(pm(r["norm_cost"], r["norm_cost_std"], bold=(a == best)))
                L.append(f" & $C/\\kappa$, {row['kappa'].iloc[0]:g} & " + " & ".join(cells) + " \\\\")
            if ti < len(st) - 1:
                L.append(f"\\cmidrule(lr){{1-{N + 2}}}")

    L += ["\\midrule",
          f"\\multicolumn{{{N + 2}}}{{l}}{{\\textit{{Number of successful tasks: $C/\\kappa\\le 1$ and "
          f"$R>{min_reward:g}$ (Safety-Gymnasium + Bullet-Gym)}}}} \\\\",
          "\\midrule"]
    for i, lvl in enumerate(LEVEL_NAMES):
        kl = " / ".join(f"{SUITE_THRESHOLDS[s][i]:g}" for s, _ in suites)
        best = max(success[lvl])
        cells = [f"\\textbf{{{n}/{n_tasks}}}" if n == best else f"{n}/{n_tasks}" for n in success[lvl]]
        first = f"\\multirow{{{len(LEVEL_NAMES)}}}{{*}}{{\\# successes}}" if i == 0 else ""
        L.append(f"{first} & $\\kappa$={kl} & " + " & ".join(cells) + " \\\\")
    words = {3: "three", 4: "four", 5: "five", 6: "six", 8: "eight", 10: "ten"}
    seeds = f"{words.get(n_seeds, n_seeds)} seeds" if n_seeds else "seeds"
    caption = (f"Constraint satisfaction at every DSRL threshold in the {setting_name} setting. "
               "For each task, row $R$ gives the normalized reward and the rows below give the normalized cost "
               "$C/\\kappa$ at each threshold. $C/\\kappa\\le 1$ means the constraint is satisfied. "
               f"Values are mean{{\\scriptsize$\\pm$std}} over {seeds}. "
               "Bold: highest reward among methods that succeed at that threshold, i.e.\\ satisfy the constraint "
               f"($C/\\kappa\\le 1$) with non-trivial reward ($R>{min_reward:g}$), so trivially safe policies that "
               "do not perform the task are never bolded or counted. "
               f"The last rows count, over all {words.get(n_tasks, n_tasks)} tasks, how often each method succeeds at the strict, middle and "
               "loose thresholds ($\\kappa$ for Safety-Gymnasium / Bullet-Gym); a task counts as a success only if "
               "it meets the same criterion. Bold marks the most successes.")
    L += ["\\bottomrule", "\\end{tabular}", "}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
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
            ax.errorbar(s["norm_cost"], s["norm_reward"],
                        xerr=[[s["norm_cost"] - s["norm_cost_lo"]], [s["norm_cost_hi"] - s["norm_cost"]]],
                        yerr=[[s["norm_reward"] - s["norm_reward_lo"]], [s["norm_reward_hi"] - s["norm_reward"]]],
                        fmt=markers[j], ms=5, capsize=2, label=a, color=colors[j])
        ax.set_xlim(0, xmax)
        kappa = sub["kappa"].iloc[0]
        ax.set_title(t if np.isnan(kappa) else f"{t} ($\\kappa$={kappa:g})", fontsize=fontsize)
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
        ax.errorbar(x, s["fbeta"], yerr=[s["fbeta"] - s["fbeta_lo"], s["fbeta_hi"] - s["fbeta"]],
                    label=a, color=colors[j], marker=markers[j], capsize=5, lw=2, alpha=0.8)
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
    axs[0].set_ylabel(r"F$\beta$-score", fontsize=fontsize)
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
        ax.set_title(f"{t} ($\\kappa$={dt['kappa'].iloc[0]:g})", fontsize=fontsize)
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
def build_threshold_datasets(traj_returns, traj_costs, kappa, n_neg=10, n_unlabeled=500,
                             low_reward_frac=0.01, seed=0):
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
    low_pool = np.setdiff1d(np.argsort(R)[: max(n_low, int(np.ceil(low_reward_frac * len(R))))], pos)
    unsafe_pool = np.setdiff1d(idx[C > kappa], np.concatenate([pos, low_pool]))
    if len(unsafe_pool) < n_unsafe:
        raise ValueError(f"Only {len(unsafe_pool)} trajectories with cost > {kappa}.")
    neg = np.concatenate([rng.choice(low_pool, n_low, replace=False),
                          rng.choice(unsafe_pool, n_unsafe, replace=False)])
    rest = np.setdiff1d(idx, np.concatenate([pos, neg]))
    return {"preferred": pos, "non_preferred": neg,
            "unlabeled": rng.choice(rest, min(n_unlabeled, len(rest)), replace=False)}


# ======================================================================================
# Notebook entry points
# ======================================================================================
def evaluate_setting(hyperparams, algos, tasks, load_fn, betas=BETAS, thresholds=None):
    """
    Compute everything for one setting. Returns a dict of DataFrames:
      metrics    per (method, task, level incl. 'avg'): R_norm, C/kappa, violation rate, safe
      summary    #safe tasks and reward on safe tasks per (method, level)
      fbeta      threshold-aware F-beta per (method, task, level, beta)
      fbeta_agg  same, averaged over tasks with stratified bootstrap CI
    plus 'labels' (method order) and 'tasks'.
    """
    labels = [l for _, l in algos]
    metrics = evaluate_algorithms(hyperparams, algos, tasks, load_fn, thresholds)
    fb, store = evaluate_fbeta_kappa(hyperparams, algos, tasks, load_fn, betas, thresholds)
    return dict(metrics=metrics, summary=safety_summary(metrics, labels), fbeta=fb,
                fbeta_agg=aggregate_fbeta_kappa(store, labels, tasks, betas, thresholds),
                labels=labels, tasks=[_key(t) for t in tasks])


def generate_tables(res, setting_name="", fbeta_betas=(1, 2), label_suffix="", show_std=True,
                    min_reward=MIN_REWARD, display_names=None, ours="PSIL", n_seeds=5):
    """
    LaTeX strings from the output of `evaluate_setting`.
    min_reward (default MIN_REWARD = 0.05): a method is bolded / counted as a success at a kappa only if
    C/kappa <= 1 AND R_norm > min_reward. Applies to the main, all_thresholds, avg_threshold and summary tables.
      all_thresholds   per task, one row per DSRL kappa: R_norm / C_norm  (appendix)
      main             per-suite blocks: R row + C/kappa row per kappa, combined #successes
                       (C/kappa <= 1 and R > min_reward)  (main table)
      summary          #safe tasks (mean reward on safe tasks) per kappa level  (main text)
      fbeta_kappa      aggregated threshold-aware F-beta per kappa level
      fbeta_kappa_per_task_b<beta>   per task, one row per DSRL kappa: F-beta (one table per beta)
      avg_threshold    DSRL average-over-thresholds table (optional)
    """
    labels, tasks = res["labels"], res["tasks"]
    sfx = label_suffix or setting_name.lower().replace("+", "").replace(" ", "")
    sfx = f"_{sfx}" if sfx else ""
    betas = tuple(b for b in fbeta_betas if b in set(res["fbeta_agg"]["beta"]))
    out = {
        "main": latex_table_main(res["metrics"], labels, tasks, setting_name or "LfP+N", min_reward,
                                 display_names, ours, n_seeds, f"tab:dsrl_main{sfx}"),
        "all_thresholds": latex_table_all_thresholds(res["metrics"], labels, tasks, setting_name,
                                                     f"tab:dsrl_all{sfx}", show_std, min_reward),
        "summary": latex_safety_summary(safety_summary(res["metrics"], labels, min_reward), labels,
                                        setting_name, f"tab:dsrl_summary{sfx}", min_reward),
        "fbeta_kappa": latex_fbeta_kappa_table(res["fbeta_agg"], betas, labels, setting_name, f"tab:fbeta_kappa{sfx}", show_std),
        "avg_threshold": latex_table_avg(res["metrics"], labels, tasks, setting_name, f"tab:dsrl_avg{sfx}", show_std,
                                         min_reward),
    }
    for b in betas:
        out[f"fbeta_kappa_per_task_b{b:g}"] = latex_fbeta_kappa_per_task(
            res["fbeta"], b, labels, tasks, setting_name, f"tab:fbeta_kappa_task_b{b:g}{sfx}", show_std)
    return out
