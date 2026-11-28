"""Figures and tables for Part A, all from results/partA_roc.npz (run partA_run.py first).

Outputs
  figures/partA_roc.png            averaged ROC curves: one panel per network x Q method,
                                   one curve per estimator and N (handout layout)
  figures/partA_edge_strength.png  histogram of |q~_ij| on the edges of each Q
  results/partA_Q_table.md         edges and edge-strength summary of each Q
  results/partA_roc_table.md/.csv  TPR read off the averaged ROC curve at FPR = 0.05 and 0.10
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
P = int(d["P"])
R_RUNS = d["glasso_TPR"].shape[3]
LABEL = {"glasso": "graphical lasso", "neigh_AND": "neighbourhood sel. (AND)"}
STYLE = {"glasso": "-", "neigh_AND": "--"}
N_COLOR = dict(zip(N_LIST, ["#c0392b", "#2471a3", "#1e8449"]))
METHOD_NAME = {"DD": "diagonal dominance", "PO": "partial orthogonalization"}
FPR_AT = [0.05, 0.10]


def mean_curve(a, b, c, est):
    """Average TPR and FPR over runs at each lambda of the grid, and the SD of TPR."""
    T, F = d[f"{est}_TPR"][a, b, c], d[f"{est}_FPR"][a, b, c]
    return np.nanmean(F, 0), np.nanmean(T, 0), np.nanstd(T, 0)


def tpr_at(fpr_target, mf, mt):
    """TPR of the averaged ROC curve at a given FPR (linear interpolation; the curve starts
    at (0, 0) when lambda = lambda_max)."""
    x = np.r_[0.0, mf]; y = np.r_[0.0, mt]
    ok = ~np.isnan(x) & ~np.isnan(y)
    order = np.argsort(x[ok], kind="stable")
    x, y = x[ok][order], y[ok][order]
    return np.interp(fpr_target, x, y) if x.max() >= fpr_target else np.nan


# ---------- 1. ROC curves ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
for a, net in enumerate(NETS):
    for b, method in enumerate(METHODS):
        ax = axes[a, b]
        for est in ESTS:
            for c, N in enumerate(N_LIST):
                mf, mt, st = mean_curve(a, b, c, est)
                mf, mt, st = np.r_[0, mf], np.r_[0, mt], np.r_[0, st]
                ax.plot(mf, mt, STYLE[est], color=N_COLOR[N], lw=1.7)
                ax.fill_between(mf, mt - st, mt + st, color=N_COLOR[N], alpha=0.07)
        ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
        ax.set_title(f"{net} network ({d['adjs'][a].sum()//2} edges), "
                     f"{METHOD_NAME[method]}", fontsize=11)
        ax.set_xlim(-0.005, 0.5); ax.set_ylim(-0.01, 1.01); ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel("FPR (non-edges reported as edges)")
for ax in axes[:, 0]:
    ax.set_ylabel("TPR (true edges found)")
handles = [plt.Line2D([], [], color=N_COLOR[N], lw=2, label=f"N = {N}" + (" (< p)" if N < P else ""))
           for N in N_LIST]
handles += [plt.Line2D([], [], color="k", ls=STYLE[e], label=LABEL[e]) for e in ESTS]
axes[0, 0].legend(handles=handles, fontsize=9, loc="lower right")
fig.suptitle(f"Part A: averaged ROC curves, p = {P}, {R_RUNS} runs per curve "
             f"(band: ±1 SD of TPR; FPR axis cut at 0.5)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "partA_roc.png"), dpi=150)

# ---------- 2. edge strengths of each Q, and the Q table ----------
q_rows = ["| Network | Q method | Edges | Edge strength mean | median | range |",
          "|---|---|---|---|---|---|"]
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a, net in enumerate(NETS):
    adj = d["adjs"][a]
    for b, (method, col) in enumerate(zip(METHODS, ["#2471a3", "#d35400"])):
        s = unit_diag_strength(d["Qs"][a, b], adj)
        q_rows.append(f"| {net} | {method} | {adj.sum()//2} | {s.mean():.3f} | "
                      f"{np.median(s):.3f} | {s.min():.3f} – {s.max():.3f} |")
        ax2[a].hist(s, bins=20, range=(0, 0.9), alpha=0.55, color=col,
                    label=f"{METHOD_NAME[method]}: mean {s.mean():.2f}")
    ax2[a].set_title(f"{net} network ({adj.sum()//2} edges)")
    ax2[a].set_xlabel("edge strength |q~_ij| (Q rescaled to unit diagonal)")
    ax2[a].legend(fontsize=9)
ax2[0].set_ylabel("number of edges")
fig2.suptitle("Edge strengths of the Q used in the experiment")
fig2.tight_layout()
fig2.savefig(os.path.join(FIG, "partA_edge_strength.png"), dpi=150)
open(os.path.join(RES, "partA_Q_table.md"), "w").write("\n".join(q_rows) + "\n")

# ---------- 3. TPR read off the ROC curves at fixed FPR ----------
rows = ["| Network | Q method | N | " + " | ".join(
            f"TPR @ FPR={f:.2f} ({LABEL[e]})" for f in FPR_AT for e in ESTS) + " |",
        "|" + "---|" * (3 + len(FPR_AT) * len(ESTS))]
csv = ["network,method,N," + ",".join(f"tpr_fpr{f:.2f}_{e}" for f in FPR_AT for e in ESTS)]
for a, net in enumerate(NETS):
    for b, method in enumerate(METHODS):
        for c, N in enumerate(N_LIST):
            vals = []
            for f in FPR_AT:
                for est in ESTS:
                    mf, mt, _ = mean_curve(a, b, c, est)
                    vals.append(tpr_at(f, mf, mt))
            rows.append(f"| {net} | {method} | {N} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
            csv.append(f"{net},{method},{N}," + ",".join(f"{v:.4f}" for v in vals))
note = (f"\nTPR of the averaged ROC curve ({R_RUNS} runs) at a fixed FPR, read by linear "
        "interpolation; higher is better. It summarises the curves in figures/partA_roc.png; "
        "no lambda is chosen. glasso solver failures: "
        f"{int(d['fails'].sum())} of {d['fails'].size * R_RUNS * len(d['R_GRID'])} fits.\n")
open(os.path.join(RES, "partA_roc_table.md"), "w").write("\n".join(rows) + "\n" + note)
open(os.path.join(RES, "partA_roc_table.csv"), "w").write("\n".join(csv) + "\n")

print("\n".join(q_rows)); print(); print("\n".join(rows)); print(note)
print("saved figures/partA_roc.png, figures/partA_edge_strength.png")
print("saved results/partA_Q_table.md, results/partA_roc_table.md, results/partA_roc_table.csv")
