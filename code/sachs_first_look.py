"""Part B: Sachs protein data, following the v2 handout.

Estimators: graphical lasso (sklearn, existing software) and neighbourhood selection with the
AND rule (our own lasso in solvers.py). Steps:
  1. log-transform and standardise the data
  2. choose lambda by simulation: data with the same p and N as the real data, drawn from a Q
     whose network is the Sachs skeleton (20 edges); take the relative lambda with the best F1
  3. estimate the network on the real data with the chosen lambda (both estimators)
  4. stability: 50 random halves of the samples, same lambda, selection frequency of each pair
  5. stable edges: pairs selected in at least 80% of the halves
  6. trust: simulate from the stable network (edge strengths from the glasso fit on all data,
     rescaled to unit diagonal; built as in Part A if not positive definite), estimate with the
     same relative lambda, report TPR and FPR

Lambda grid: r * lambda_max with r from 1 to 0.01 (the same grid as Part A); lambda_max is
computed from the data set being analysed (simulated or real).

Outputs: results/partB_sachs.md, figures/partB_selection.png
Usage: python code/sachs_first_look.py
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.covariance import graphical_lasso
from solvers import neighbourhood_path
from partA_generate import make_Q, sample_gaussian

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEED = 2026
R_GRID = np.logspace(0, -2, 25)
N_SIM = 20           # simulated data sets for choosing lambda
N_HALVES = 50        # random halves for stability
N_TRUST = 20         # simulated data sets for the trust check
THRESH = 0.80
LAM_MIN = 0.1
ESTS = ["glasso", "neigh_AND"]
NAME = {"glasso": "graphical lasso", "neigh_AND": "neighbourhood sel. (AND)"}

# ---------- data and ground truth ----------
df = pd.read_csv(os.path.join(ROOT, "data", "sachs_continuous.txt"), sep="\t")
nodes = list(df.columns)
X = np.log(df.values)                          # right-skewed measurements: log first
X = (X - X.mean(0)) / X.std(0)                 # then standardise
N, p = X.shape
gt = np.zeros((p, p), bool)                    # the 20 edges of the Sachs skeleton
for line in open(os.path.join(ROOT, "data", "sachs_ground_truth.txt")):
    if "-->" in line:
        a, b = [s.strip() for s in line.split(".", 1)[1].split("-->")]
        i, j = nodes.index(a), nodes.index(b)
        gt[i, j] = gt[j, i] = True
iu = np.triu_indices(p, 1)
n_true = gt[iu].sum()


def estimate(Xd, rel_lams, est):
    """Edge sets (boolean p x p) of one estimator at each relative lambda of the grid."""
    Xc = Xd - Xd.mean(0)
    S = Xc.T @ Xc / Xd.shape[0]
    lams = rel_lams * np.abs(S - np.diag(np.diag(S))).max()
    if est == "glasso":
        out = []
        for lam in lams:
            _, prec = graphical_lasso(S, alpha=lam, mode="cd", max_iter=500)
            E = np.abs(prec) > 1e-8
            np.fill_diagonal(E, False)
            out.append(E)
        return out
    return neighbourhood_path(Xd, lams, rule="and")


def scores(E, truth):
    e, t = E[iu], truth[iu]
    tp = np.sum(e & t); fp = np.sum(e & ~t); fn = np.sum(~e & t)
    tpr = tp / max(t.sum(), 1); fpr = fp / max((~t).sum(), 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return tpr, fpr, f1, int(e.sum())


rng = np.random.default_rng(SEED)
out_lines = []
def say(s=""):
    print(s); out_lines.append(s)

say(f"Sachs data: N = {N}, p = {p}, true undirected edges = {n_true} (Sachs skeleton)")

# ---------- 2. choose lambda by simulation (same p and N, network = Sachs skeleton) ----------
Q_sim = make_Q(gt.astype(int), "DD", rng, lam_min=LAM_MIN)
f1_sum = {e: np.zeros(len(R_GRID)) for e in ESTS}
for s in range(N_SIM):
    Xs = sample_gaussian(Q_sim, N, rng)
    for est in ESTS:
        Es = estimate(Xs, R_GRID, est)
        f1_sum[est] += np.array([scores(E, gt)[2] for E in Es])
f1_mean = {e: f1_sum[e] / N_SIM for e in ESTS}
best_r = {e: R_GRID[int(np.argmax(f1_mean[e]))] for e in ESTS}
say(f"\nStep 2. lambda by simulation ({N_SIM} data sets, F1 against the Sachs skeleton):")
for e in ESTS:
    say(f"  {NAME[e]}: best lambda/lambda_max = {best_r[e]:.3f}, mean F1 there = "
        f"{f1_mean[e].max():.2f}")

# ---------- 3. estimate on the real data ----------
lam_real = {}
Xc = X - X.mean(0); S_real = Xc.T @ Xc / N
lam_max_real = np.abs(S_real - np.diag(np.diag(S_real))).max()
est_real = {}
say("\nStep 3. estimates on the Sachs data with the chosen lambda:")
say("| estimator | lambda/lambda_max | lambda | edges | TP | FP | F1 vs skeleton |")
say("|---|---|---|---|---|---|---|")
for e in ESTS:
    lam_real[e] = best_r[e] * lam_max_real
    E = estimate(X, np.array([best_r[e]]), e)[0]
    est_real[e] = E
    tpr, fpr, f1, ne = scores(E, gt)
    tp = int(round(tpr * n_true)); fp = ne - tp
    say(f"| {NAME[e]} | {best_r[e]:.3f} | {lam_real[e]:.4f} | {ne} | {tp} | {fp} | {f1:.2f} |")

# ---------- 4. stability: 50 random halves ----------
freq = {e: np.zeros((p, p)) for e in ESTS}
for h in range(N_HALVES):
    idx = rng.permutation(N)[: N // 2]
    Xh = X[idx]
    for e in ESTS:
        E = estimate(Xh, np.array([best_r[e]]), e)[0]
        freq[e] += E
for e in ESTS:
    freq[e] /= N_HALVES
say(f"\nStep 4. stability: {N_HALVES} random halves of {N // 2} samples each, same lambda.")

# ---------- 5. stable edges ----------
say(f"\nStep 5. stable edges (selected in at least {THRESH:.0%} of the halves):")
stable_adj = {}
for e in ESTS:
    F = freq[e]
    stable = (F >= THRESH) & (F.T >= THRESH)
    np.fill_diagonal(stable, False)
    stable_adj[e] = stable
    say(f"\n{NAME[e]}: {stable[iu].sum()} stable edges")
    rows = []
    for i, j in zip(*iu):
        if stable[i, j]:
            rows.append((F[i, j], nodes[i], nodes[j], "yes" if gt[i, j] else "no"))
    for fq, a, b, t in sorted(rows, reverse=True):
        say(f"  {a} - {b}: selected {fq:.0%} of halves, in Sachs skeleton: {t}")
    if not rows:
        say("  (none)")
    edges_all = [(nodes[i], nodes[j], F[i, j]) for i, j in zip(*iu) if F[i, j] > 0.5]
    say(f"  pairs selected in more than half of the halves: {len(edges_all)}")

# ---------- 6. trust: simulate from the stable network ----------
say(f"\nStep 6. trust: simulate {N_TRUST} data sets from the stable network (N = {N}, p = {p}),"
    " estimate with the same relative lambda, compare with the stable network.")
say("| estimator | stable edges (truth) | TPR | FPR | F1 |")
say("|---|---|---|---|---|")
for e in ESTS:
    # edge strengths: glasso precision on all data, restricted to the stable edges, unit diagonal
    _, prec = graphical_lasso(S_real, alpha=lam_real["glasso"], mode="cd", max_iter=500)
    A = stable_adj[e].astype(int)
    Qs = np.where(A.astype(bool), prec, 0.0)
    np.fill_diagonal(Qs, np.diag(prec))
    d_ = np.sqrt(np.diag(Qs)); Qs = Qs / np.outer(d_, d_)
    try:
        np.linalg.cholesky(Qs)
        Q_true = Qs
        built = "restricted glasso estimate"
    except np.linalg.LinAlgError:
        Q_true = make_Q(A, "DD", rng, lam_min=LAM_MIN)
        built = "DD on the stable network (not PD)"
    truth = A.astype(bool)
    tprs, fprs, f1s = [], [], []
    for s in range(N_TRUST):
        Xs = sample_gaussian(Q_true, N, rng)
        E = estimate(Xs, np.array([best_r[e]]), e)[0]
        t_, f_, f1_, _ = scores(E, truth)
        tprs.append(t_); fprs.append(f_); f1s.append(f1_)
    say(f"| {NAME[e]} | {truth[iu].sum()} ({built}) | {np.mean(tprs):.2f} ± {np.std(tprs, ddof=1):.2f}"
        f" | {np.mean(fprs):.3f} ± {np.std(fprs, ddof=1):.3f} | {np.mean(f1s):.2f} |")

# ---------- outputs ----------
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
open(os.path.join(ROOT, "results", "partB_sachs.md"), "w").write("\n".join(out_lines) + "\n")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a_, e in zip(ax, ESTS):
    im = a_.imshow(freq[e], vmin=0, vmax=1, cmap="viridis")
    a_.set_xticks(range(p)); a_.set_xticklabels(nodes, rotation=60, fontsize=8)
    a_.set_yticks(range(p)); a_.set_yticklabels(nodes, fontsize=8)
    a_.set_title(f"{NAME[e]}: selection frequency", fontsize=10)
    for i, j in zip(*iu):
        if gt[i, j]:
            a_.plot(j, i, "o", mfc="none", mec="red", ms=5)
fig.colorbar(im, ax=ax, fraction=0.03)
fig.suptitle(f"Sachs: selection frequency over {N_HALVES} random halves (red circles: "
             "Sachs skeleton edges)", fontsize=10)
fig.savefig(os.path.join(ROOT, "figures", "partB_selection.png"), dpi=150, bbox_inches="tight")
print("\nsaved results/partB_sachs.md, figures/partB_selection.png")
