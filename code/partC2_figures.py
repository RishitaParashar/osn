"""Part C2 figures: networks, edge strengths, and bias across eps.

Figure 1  partC2_network.png   true network vs glasso vs reweighted glasso (one run, drawn on a circle)
Figure 2  partC2_strength.png  true |q| vs estimated |q_hat| on every pair (one run)
Figure 3  partC2_bias_eps.png  bias on true edges for each eps (uses results/partC2*.npz; no new fits)

Figures 1 and 2 use run 0 of the BIC-selected fits (eps = 0.01, chain, DD, N = 100) and
the same seeds as partC2_run.py.
Usage: python code/partC2_figures.py
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.covariance import graphical_lasso
from partA_generate import sample_gaussian
from weighted_glasso import reweighted_glasso, support
import partA_run as R

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "figures")
EPS = 0.01
d = np.load(os.path.join(ROOT, "results", "partA_roc.npz"), allow_pickle=True)
Q_TRUE, ADJ = d["Qs"][1, 0], d["adjs"][1].astype(bool)     # chain, DD
N = int(d["N_LIST"][1])                                    # 100
p = R.P
iu = np.triu_indices(p, 1)
rng = np.random.default_rng([R.SEED, 1, 0, 1, 0])          # run 0 of partC2_run.py
X = sample_gaussian(Q_TRUE, N, rng)
Xc = X - X.mean(0); S = Xc.T @ Xc / N
lams = R.R_GRID * np.abs(S - np.diag(np.diag(S))).max()

def bic(Q):
    _, ld = np.linalg.slogdet(Q)
    return N * (-ld + np.trace(S @ Q)) + np.log(N) * (support(Q).sum() // 2)

G, RW = [], []
for lam in lams:
    _, Qg = graphical_lasso(S, alpha=lam, mode="cd", max_iter=300)
    Qr, _ = reweighted_glasso(S, lam, eps=EPS, n_steps=5, Q0=Qg)
    G.append(Qg); RW.append(Qr)
k = int(np.argmin([bic(Q) for Q in RW]))                  # reweighted: BIC lambda
ne = support(RW[k]).sum() // 2
g = int(np.argmin([abs(support(Q).sum() // 2 - ne) for Q in G]))   # glasso: same edge count
Q_g, Q_r = G[g], RW[k]
print(f"run 0: reweighted lambda/max {R.R_GRID[k]:.3f} ({ne} edges); glasso lambda/max "
      f"{R.R_GRID[g]:.3f} ({support(Q_g).sum()//2} edges)")

# ---------- figure 1: networks ----------
ang = np.linspace(0, 2 * np.pi, p, endpoint=False)
pos = np.c_[np.cos(ang), np.sin(ang)]
def draw(ax, E_est, title, truth=ADJ):
    for i, j in zip(*iu):
        t, e = truth[i, j], E_est[i, j]
        if t and e:
            ax.plot(*pos[[i, j]].T, color="#1e8449", lw=2.2, zorder=2)
        elif e and not t:
            ax.plot(*pos[[i, j]].T, color="#c0392b", lw=1.6, zorder=2)
        elif t and not e:
            ax.plot(*pos[[i, j]].T, color="#7f8c8d", lw=1.2, ls="--", zorder=1)
    ax.scatter(*pos.T, s=40, color="black", zorder=3)
    ax.set_title(title, fontsize=10); ax.set_aspect("equal"); ax.axis("off")
fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
draw(ax[0], ADJ, f"True network ({ADJ.sum()//2} edges)", truth=ADJ)
ax[0].cla(); ax[0].set_aspect("equal"); ax[0].axis("off")
for i, j in zip(*iu):
    if ADJ[i, j]:
        ax[0].plot(*pos[[i, j]].T, color="#1e8449", lw=2.2)
ax[0].scatter(*pos.T, s=40, color="black", zorder=3)
ax[0].set_title(f"True network ({ADJ.sum()//2} edges)", fontsize=10)
draw(ax[1], support(Q_g), f"glasso ({support(Q_g).sum()//2} edges)")
draw(ax[2], support(Q_r), f"Reweighted glasso ({support(Q_r).sum()//2} edges)")
from matplotlib.lines import Line2D
leg = [Line2D([], [], color="#1e8449", lw=2.2, label="true edge found"),
       Line2D([], [], color="#c0392b", lw=1.6, label="false edge"),
       Line2D([], [], color="#7f8c8d", lw=1.2, ls="--", label="true edge missed")]
ax[2].legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
fig.suptitle(f"Chain network, DD Q, N = {N}, one run (run 0), BIC lambda", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "partC2_network.png"), dpi=150)

# ---------- figure 2: true vs estimated edge strength ----------
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True, sharey=True)
for a_, Q, name, col in [(ax[0], Q_g, "glasso", "#2471a3"), (ax[1], Q_r, "reweighted glasso", "#d35400")]:
    qt = np.abs(Q_TRUE[iu]); qh = np.abs(Q[iu]); t = ADJ[iu]
    a_.scatter(qt[~t], qh[~t], s=10, color="#7f8c8d", alpha=0.5, label="non-edges (should be 0)")
    a_.scatter(qt[t], qh[t], s=28, color=col, label="true edges")
    a_.plot([0, 1], [0, 1], ":", color="black", lw=1)
    a_.set_title(name, fontsize=11); a_.set_xlabel("true |q_ij|"); a_.grid(alpha=0.3)
    a_.set_xlim(-0.02, 1.0); a_.set_ylim(-0.02, 1.0)
    b_ = np.mean(qh[t] - qt[t])
    a_.text(0.03, 0.9, f"bias on true edges = {b_:+.3f}", transform=a_.transAxes, fontsize=9)
ax[0].set_ylabel("estimated |q_hat_ij|"); ax[0].legend(fontsize=8, loc="lower right")
fig.suptitle("True vs estimated edge strengths (dotted line: no bias), one run, chain DD N = 100",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "partC2_strength.png"), dpi=150)

# ---------- figure 3: bias for each eps (from saved results, no new fits) ----------
def matched_bias(npz):
    z = np.load(npz)
    G_, R_ = z["glasso_grid"], z["reweighted_grid"]
    out = []
    for r in range(R_.shape[0]):
        bic_ = R_[r, :, 5]
        if np.all(np.isnan(bic_)):
            continue
        kk = int(np.nanargmin(bic_)); tgt = R_[r, kk, 3]
        diff = np.abs(G_[r, :, 3] - tgt)
        if np.all(np.isnan(diff)):
            continue
        out.append((G_[r, int(np.nanargmin(diff)), 4], R_[r, kk, 4]))
    return np.array(out)
files = [(0.001, "partC2_eps0.001.npz"), (0.01, "partC2.npz"), (0.1, "partC2_eps0.1.npz")]
labels = [f"ε = {e:g}" for e, _ in files]
gb = [matched_bias(os.path.join(ROOT, "results", f))[:, 0] for _, f in files]
rb = [matched_bias(os.path.join(ROOT, "results", f))[:, 1] for _, f in files]
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(files)); w = 0.36
for off, data, name, col in [(-w/2, gb, "glasso", "#2471a3"), (w/2, rb, "reweighted glasso", "#d35400")]:
    m = [v.mean() for v in data]; s = [v.std(ddof=1) for v in data]
    ax.bar(x + off, m, w, yerr=s, capsize=4, color=col, alpha=0.85, label=name)
    for xi, mi in zip(x + off, m):
        ax.text(xi, mi / 2, f"{mi:.2f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("bias on true edges  (|q_hat| − |q|)")
ax.set_title("Bias at the same number of edges, 35 runs (chain, DD, N = 100)\n"
             "reweighted glasso stays close to 0 for every ε tried", fontsize=10)
ax.set_ylim(-0.42, 0.0)
ax.legend(loc="lower left", fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "partC2_bias_eps.png"), dpi=150)
print("saved figures/partC2_network.png, partC2_strength.png, partC2_bias_eps.png")
