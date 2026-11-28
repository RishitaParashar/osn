"""Figures and tables for Part A, all from results/partA_roc.npz (run partA_run.py first).

Outputs
  figures/partA_roc.png           averaged ROC curves, one panel per network x Q method
  figures/partA_edge_strength.png |q~_ij| on the true edges of the Q's used in the runs
  figures/partA_f1.png            F1 of the network picked by CV-min, CV-1SE and BIC, vs N
  results/partA_Q_table.md        edges and edge-strength summary of each Q
  results/partA_f1_table.md/.csv  TPR at FPR = 0.05, F1 and edge count for each rule,
                                  oracle F1 (reference only), glasso failures
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from partA_generate import unit_diag_strength

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG, RES = os.path.join(ROOT, "figures"), os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)
d = np.load(os.path.join(RES, "partA_roc.npz"), allow_pickle=True)
NETS, METHODS = list(d["NETS"]), list(d["METHODS"])
ESTS, N_LIST = list(d["ESTIMATORS"]), [int(n) for n in d["N_LIST"]]
P, K = int(d["P"]), int(d["K_FOLDS"])
R_RUNS = d["glasso_TPR"].shape[3]
RULES = ["CV-min", "CV-1SE", "BIC"]
RULE_MARK = {"CV-min": "o", "CV-1SE": "^", "BIC": "s"}
RULE_ALPHA = {"CV-min": 1.0, "CV-1SE": 0.8, "BIC": 0.45}
LABEL = {"glasso": "glasso", "neigh_OR": "neigh. sel. (OR)", "neigh_AND": "neigh. sel. (AND)"}
STYLE = {"glasso": "-", "neigh_OR": "--", "neigh_AND": "-."}
EST_COLOR = {"glasso": "#2471a3", "neigh_OR": "#d35400", "neigh_AND": "#1e8449"}
N_COLOR = dict(zip(N_LIST, ["#c0392b", "#2471a3", "#1e8449"]))
METHOD_NAME = {"DD": "diagonal dominance", "PO": "partial orthogonalization"}


def mean_curve(a, b, c, est):
    """Average TPR and FPR over runs at each lambda of the grid."""
    T, F = d[f"{est}_TPR"][a, b, c], d[f"{est}_FPR"][a, b, c]
    return np.nanmean(F, 0), np.nanmean(T, 0), np.nanstd(T, 0)


def tpr_at(fpr_target, mf, mt):
    """TPR of the averaged ROC curve at a given FPR (linear interpolation)."""
    ok = ~np.isnan(mf)
    order = np.argsort(mf[ok])
    x, y = mf[ok][order], mt[ok][order]
    return np.interp(fpr_target, x, y) if x.size and x.max() >= fpr_target else np.nan


def pick(rule, crit, se):
    """Grid index chosen by a rule (the grid runs from the largest lambda to the smallest).
    CV-min, BIC: the lambda with the smallest score.
    CV-1SE:      the largest lambda whose CV score is within one standard error of the minimum.
    """
    cr = np.where(np.isfinite(crit), crit, np.nan)
    if np.all(np.isnan(cr)):
        return None
    g = int(np.nanargmin(cr))
    if rule == "CV-1SE" and np.isfinite(se[g]):
        g = int(np.flatnonzero(cr <= cr[g] + se[g])[0])
    return g


def selected(a, b, c, est, rule):
    """F1 and edge count of the network picked by `rule` in each run: (mean, sd) each."""
    crit = d[f"{est}_{'BIC' if rule == 'BIC' else 'CV'}"][a, b, c]
    se = d[f"{est}_CV_SE"][a, b, c]
    F1, NE = d[f"{est}_F1"][a, b, c], d[f"{est}_NEDGES"][a, b, c]
    f1, ne = [], []
    for run in range(crit.shape[0]):
        g = pick(rule, crit[run], se[run])
        if g is None:
            continue
        f1.append(F1[run, g]); ne.append(NE[run, g])
    return np.nanmean(f1), np.nanstd(f1), np.nanmean(ne), np.nanstd(ne)


# ---------- 1. ROC curves ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
for a, net in enumerate(NETS):
    for b, method in enumerate(METHODS):
        ax = axes[a, b]
        for est in ESTS:
            for c, N in enumerate(N_LIST):
                mf, mt, st = mean_curve(a, b, c, est)
                ax.plot(mf, mt, STYLE[est], color=N_COLOR[N], lw=1.6)
                ax.fill_between(mf, mt - st, mt + st, color=N_COLOR[N], alpha=0.07)
        ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
        ax.set_title(f"{net} network ({d['adjs'][a].sum()//2} edges), "
                     f"{METHOD_NAME[method]}", fontsize=11)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel("FPR (non-edges reported as edges)")
for ax in axes[:, 0]:
    ax.set_ylabel("TPR (true edges found)")
handles = [plt.Line2D([], [], color=N_COLOR[N], lw=2, label=f"N = {N}") for N in N_LIST]
handles += [plt.Line2D([], [], color="k", ls=STYLE[e], label=LABEL[e]) for e in ESTS]
axes[0, 0].legend(handles=handles, fontsize=9, loc="lower right")
fig.suptitle(f"Part A: ROC curves, p = {P}, averaged over {R_RUNS} runs "
             f"(band: ±1 SD of TPR over runs)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "partA_roc.png"), dpi=150)

# ---------- 2. edge strengths of the Q's used, and the Q table ----------
q_rows = ["| Network | Method | Edges | Edge strength mean | median | range |",
          "|---|---|---|---|---|---|"]
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a, net in enumerate(NETS):
    adj = d["adjs"][a]
    for b, (method, col) in enumerate(zip(METHODS, ["#2471a3", "#d35400"])):
        s = unit_diag_strength(d["Qs"][a, b], adj)
        q_rows.append(f"| {net} | {method} | {adj.sum()//2} | {s.mean():.3f} | "
                      f"{np.median(s):.3f} | {s.min():.3f} – {s.max():.3f} |")
        ax2[a].hist(s, bins=20, range=(0, 0.8), alpha=0.55, color=col,
                    label=f"{METHOD_NAME[method]}: mean {s.mean():.2f}")
    ax2[a].set_title(f"{net} network ({adj.sum()//2} edges)")
    ax2[a].set_xlabel("edge strength |q~_ij| (Q rescaled to unit diagonal)")
    ax2[a].legend(fontsize=9)
ax2[0].set_ylabel("number of edges")
fig2.suptitle("Edge strengths of the Q used in the experiment")
fig2.tight_layout()
fig2.savefig(os.path.join(FIG, "partA_edge_strength.png"), dpi=150)
open(os.path.join(RES, "partA_Q_table.md"), "w").write("\n".join(q_rows) + "\n")

# ---------- 3. lambda chosen by CV and BIC: F1 figure and table ----------
f1_rows = ["| Network | Method | N | Estimator | TPR @ FPR=0.05 | "
           + " | ".join(f"F1 ({r}) | edges ({r})" for r in RULES)
           + " | F1 oracle* | glasso failures |",
           "|" + "---|" * (7 + 2 * len(RULES))]
csv = ["network,method,N,estimator,true_edges,tpr_at_fpr005,"
       + ",".join(f"f1_{r}_mean,f1_{r}_sd,edges_{r}_mean" for r in RULES)
       + ",f1_oracle,glasso_failures"]
fig3, axes3 = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for a, net in enumerate(NETS):
    n_true = d["adjs"][a].sum() // 2
    for b, method in enumerate(METHODS):
        ax = axes3[a, b]
        for est in ESTS:
            line = {r: [] for r in RULES + ["oracle"]}
            for c, N in enumerate(N_LIST):
                mf, mt, _ = mean_curve(a, b, c, est)
                t05 = tpr_at(0.05, mf, mt)
                sel = {r: selected(a, b, c, est, r) for r in RULES}
                fo = np.nanmax(np.nanmean(d[f"{est}_F1"][a, b, c], 0))
                nfail = int(d["fails"][a, b, c]) if est == "glasso" else 0
                for r in RULES:
                    line[r].append(sel[r][0])
                line["oracle"].append(fo)
                f1_rows.append(
                    f"| {net} ({n_true}) | {method} | {N} | {LABEL[est]} | {t05:.2f} | "
                    + " | ".join(f"{sel[r][0]:.2f} ± {sel[r][1]:.2f} | {sel[r][2]:.0f}"
                                 for r in RULES)
                    + f" | {fo:.2f} | {nfail if est == 'glasso' else '–'} |")
                csv.append(f"{net},{method},{N},{est},{n_true},{t05:.4f},"
                           + ",".join(f"{sel[r][0]:.4f},{sel[r][1]:.4f},{sel[r][2]:.2f}"
                                      for r in RULES)
                           + f",{fo:.4f},{nfail}")
            col = EST_COLOR[est]
            for r in ["CV-min", "CV-1SE"]:          # BIC is in the table only
                ax.plot(N_LIST, line[r], STYLE[est], marker=RULE_MARK[r], color=col,
                        alpha=RULE_ALPHA[r], label=f"{LABEL[est]}, {r}")
            ax.plot(N_LIST, line["oracle"], ":", color=col, alpha=0.6,
                    label=f"{LABEL[est]}, oracle*")
        ax.axvline(P, color="grey", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xticks(N_LIST); ax.set_xticklabels(N_LIST)
        ax.minorticks_off()
        ax.set_ylim(0, 1)
        ax.set_title(f"{net} network, {METHOD_NAME[method]}", fontsize=11); ax.grid(alpha=0.3)
for ax in axes3[1]:
    ax.set_xlabel("sample size N (dotted line: N = p)")
for ax in axes3[:, 0]:
    ax.set_ylabel("F1 of the selected network")
axes3[0, 0].legend(fontsize=7, loc="upper left")
fig3.suptitle(f"F1 of the network chosen by {K}-fold CV (min and 1-SE rule), "
              f"mean over {R_RUNS} runs; "
              "*oracle uses the true network (reference only)", fontsize=11)
fig3.tight_layout()
fig3.savefig(os.path.join(FIG, "partA_f1.png"), dpi=150)

note = ("\nThe number in brackets after the network is its true number of edges. "
        "Edges is the mean number of edges in the selected network. CV-min: lambda with the "
        "smallest 5-fold CV score. CV-1SE: the largest lambda whose CV score is within one "
        "standard error of that minimum (a sparser network with almost the same CV score).\n"
        "*Oracle F1: the F1 of the single best fixed λ/λ_max on the grid, averaged over runs. "
        "It uses the true network, so it is not a usable rule; it is a reference. The rules "
        "choose λ separately in each run, so they can occasionally beat it.\n")
open(os.path.join(RES, "partA_f1_table.md"), "w").write("\n".join(f1_rows) + "\n" + note)
open(os.path.join(RES, "partA_f1_table.csv"), "w").write("\n".join(csv) + "\n")

print("\n".join(q_rows))
print()
print("\n".join(f1_rows))
print("\nsaved figures/partA_roc.png, figures/partA_edge_strength.png, figures/partA_f1.png")
print("saved results/partA_Q_table.md, results/partA_f1_table.md, results/partA_f1_table.csv")
