"""Part C2 table and figure from results/partC2.npz (run partC2_run.py first).

Table (v2 handout: "choose lambda for each so that both networks have about the same number of
edges"): in each run the reweighted glasso is taken at its BIC lambda, and glasso at the lambda
whose edge count is closest to it. F1 / TPR / FPR / edges / bias, mean ± sd over runs. Figure: bias and F1 against the number of selected edges,
averaged over runs at each lambda, so the two methods are compared at matched edge counts.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
d = np.load(os.path.join(ROOT, "results", "partC2.npz"))
N, eps = int(d["N"]), float(d["eps"])
G, Rw = d["glasso_grid"], d["reweighted_grid"]       # runs x lambdas x (tpr,fpr,f1,edges,bias,bic)
runs = G.shape[0]
cols = ["TPR", "FPR", "F1", "edges", "bias"]

def matched(Gg, Rg):
    """Per run: reweighted at its BIC lambda; glasso at the lambda with the closest edge count."""
    rg, rr = [], []
    for r in range(Rg.shape[0]):
        bic = Rg[r, :, 5]
        if np.all(np.isnan(bic)):
            continue
        k = int(np.nanargmin(bic))
        target = Rg[r, k, 3]
        diff = np.abs(Gg[r, :, 3] - target)
        if np.all(np.isnan(diff)):
            continue
        rr.append(Rg[r, k, :5]); rg.append(Gg[r, int(np.nanargmin(diff)), :5])
    return np.array(rg), np.array(rr)

ag, ar = matched(G, Rw)
lines = ["| Method (matched edge counts) | TPR | FPR | F1 | edges | bias on true edges |",
         "|---|---|---|---|---|---|"]
for name, a in [("glasso", ag), ("reweighted glasso (log-sum)", ar)]:
    cells = [f"{a[:, i].mean():.3f} ± {a[:, i].std(ddof=1):.3f}" if i != 3
             else f"{a[:, i].mean():.0f} ± {a[:, i].std(ddof=1):.0f}" for i in range(5)]
    lines.append(f"| {name} | " + " | ".join(cells) + " |")
note = (f"\nchain network (59 edges), DD Q with lambda_min = 0.1, N = {N}, {runs} runs; "
        f"reweighted glasso: log-sum penalty, eps = {eps}, up to 5 MM steps from the glasso fit; "
        "its lambda is chosen by BIC, and glasso's lambda gives the closest number of edges. "
        "bias = mean over true edges of (|q_hat| - |q|), negative = shrunk toward 0.\n")
# sensitivity to eps: reweighted glasso at the BIC lambda for each eps that has been run
eps_files = [(eps, "partC2.npz")] + [
    (float(fn[len("partC2_eps"):-4]), fn) for fn in sorted(os.listdir(os.path.join(ROOT, "results")))
    if fn.startswith("partC2_eps") and fn.endswith(".npz")]
eps_lines = ["", "Sensitivity to eps (same matching; glasso columns are at the matched edge count):", "",
             "| eps | edges (reweighted / glasso) | F1 (reweighted / glasso) | bias (reweighted / glasso) |",
             "|---|---|---|---|"]
for e_, fn in sorted(eps_files):
    z = np.load(os.path.join(ROOT, "results", fn))
    g_, r_ = matched(z["glasso_grid"], z["reweighted_grid"])
    eps_lines.append(f"| {e_:g} | {r_[:, 3].mean():.0f} / {g_[:, 3].mean():.0f} | "
                     f"{r_[:, 2].mean():.3f} / {g_[:, 2].mean():.3f} | "
                     f"{r_[:, 4].mean():.3f} / {g_[:, 4].mean():.3f} |")
note += "\n".join(eps_lines) + "\n"
open(os.path.join(ROOT, "results", "partC2.md"), "w").write("\n".join(lines) + "\n" + note)
print("\n".join(lines)); print(note)

# matched-edge curves: average over runs at each lambda
def curve(grid, i):
    return np.nanmean(grid[:, :, 3], 0), np.nanmean(grid[:, :, i], 0), np.nanstd(grid[:, :, i], 0)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for grid, name, col in [(G, "glasso", "#2471a3"), (Rw, "reweighted glasso", "#d35400")]:
    e, b, sb = curve(grid, 4)
    ax[0].plot(e, b, "-o", ms=3, color=col, label=name)
    ax[0].fill_between(e, b - sb, b + sb, color=col, alpha=0.12)
    e, f, sf = curve(grid, 2)
    ax[1].plot(e, f, "-o", ms=3, color=col, label=name)
    ax[1].fill_between(e, f - sf, f + sf, color=col, alpha=0.12)
ax[0].axhline(0, color="grey", lw=0.8, ls=":")
ax[0].set_xlabel("selected edges"); ax[0].set_ylabel("bias on true edges  (|q_hat| - |q|)")
ax[0].set_title("Bias against number of edges (closer to 0 = less shrinkage)")
ax[1].axhline(0, color="grey", lw=0.8, ls=":")
ax[1].set_xlabel("selected edges"); ax[1].set_ylabel("F1")
ax[1].set_title("F1 against number of edges")
for a_ in ax:
    a_.set_xscale("log"); a_.grid(alpha=0.3); a_.legend(fontsize=9)
    a_.axvline(59, color="grey", lw=0.8, ls="--")
    a_.set_xlim(left=1)
fig.suptitle(f"Part C2: chain network, DD, N = {N}, mean ± sd over {runs} runs "
             "(dashed line: 59 true edges)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "partC2_bias_f1.png"), dpi=150)
print("saved figures/partC2_bias_f1.png, results/partC2.md")
