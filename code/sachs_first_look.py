"""Part B: Sachs protein data, step by step as in the v2 handout.

Estimators (the Part A estimators): graphical lasso (sklearn) and neighbourhood selection with
the AND rule (solvers.py).

1. Preprocess and estimate.
   Preprocessing: log-transform each protein (the raw values are right-skewed), then
   standardise each variable to mean 0 and variance 1.
   Choose lambda from a simulation: data generated as in Part A (Erdos-Renyi and chain
   networks; Q by diagonal dominance and by partial orthogonalization, lambda_min(Q) = 0.1)
   with the same p and N as the Sachs data. Five values of lambda are tried, lambda = r *
   lambda_max with r in R_LIST; the r with the best F1 (averaged over the simulations) is
   kept for each estimator and used to estimate the network on the Sachs data.
2. Stability: 50 random halves of the samples, same lambda; selection frequency of each pair.
3. Stable edges: pairs with selection frequency >= 80%.
4. Trust: the stable network with its estimated edge strengths (glasso precision on all the
   data, restricted to the stable edges, unit diagonal) is the truth; if it is not positive
   definite, Q is built on the stable network as in Part A (DD). Data with the same p and N
   are simulated from it, estimated with the same lambda, and TPR and FPR are reported.

The Sachs literature network (20 edges) is NOT used to choose lambda; it is only printed next
to the results as a reference.

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
from partA_generate import er_adj, chain_adj, make_Q, sample_gaussian

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEED = 2026
R_LIST = np.array([0.5, 0.2, 0.1, 0.05, 0.02])   # the five lambda values, as lambda / lambda_max
PI_ER = 0.3          # Erdos-Renyi edge probability for p = 11 (about 16 edges)
LAM_MIN = 0.1        # as in Part A
N_SIM = 10           # simulated data sets per (network, Q method)
N_HALVES = 50        # random halves for stability
THRESH = 0.80        # stable-edge threshold
N_TRUST = 20         # simulated data sets for the trust check
ESTS = ["glasso", "neigh_AND"]
NAME = {"glasso": "graphical lasso", "neigh_AND": "neighbourhood selection (AND)"}

rng = np.random.default_rng(SEED)
out_lines = []
def say(s=""):
    print(s); out_lines.append(s)


def lam_max(Xd):
    Xc = Xd - Xd.mean(0)
    S = Xc.T @ Xc / Xd.shape[0]
    return np.abs(S - np.diag(np.diag(S))).max(), S


def estimate(Xd, lam, est):
    """Edge set (boolean p x p) of one estimator at one lambda."""
    if est == "glasso":
        _, S = lam_max(Xd)
        _, prec = graphical_lasso(S, alpha=lam, mode="cd", max_iter=500)
        E = np.abs(prec) > 1e-8
        np.fill_diagonal(E, False)
        return E
    return neighbourhood_path(Xd, [lam], rule="and")[0]


def rates(E, truth):
    iu = np.triu_indices(truth.shape[0], 1)
    e, t = E[iu], truth[iu]
    tp = np.sum(e & t); fp = np.sum(e & ~t); fn = np.sum(~e & t)
    return tp / max(t.sum(), 1), fp / max((~t).sum(), 1), 2 * tp / max(2 * tp + fp + fn, 1)


# ================= 1. preprocess =================
df = pd.read_csv(os.path.join(ROOT, "data", "sachs_continuous.txt"), sep="\t")
nodes = list(df.columns)
X = np.log(df.values)                         # log-transform (right-skewed measurements)
X = (X - X.mean(0)) / X.std(0)                # standardise each variable
N, p = X.shape
iu = np.triu_indices(p, 1)
ref = np.zeros((p, p), bool)                  # literature network, reference only
for line in open(os.path.join(ROOT, "data", "sachs_ground_truth.txt")):
    if "-->" in line:
        a, b = [s.strip() for s in line.split(".", 1)[1].split("-->")]
        i, j = nodes.index(a), nodes.index(b)
        ref[i, j] = ref[j, i] = True
say(f"Data: Sachs et al., N = {N} cells, p = {p} proteins: {', '.join(nodes)}")
say("Preprocessing: log-transform each protein, then standardise each to mean 0, variance 1.")

# ================= 1. choose lambda by simulation (as in Part A, same p and N) =================
settings = []
for net in ["ER", "chain"]:
    adj = er_adj(p, PI_ER, rng) if net == "ER" else chain_adj(p)
    for method in ["DD", "PO"]:
        settings.append((net, method, adj, make_Q(adj, method, rng, lam_min=LAM_MIN)))

f1 = {e: np.zeros((len(settings), len(R_LIST))) for e in ESTS}
for k, (net, method, adj, Q) in enumerate(settings):
    for _ in range(N_SIM):
        Xs = sample_gaussian(Q, N, rng)
        Xs = (Xs - Xs.mean(0)) / Xs.std(0)    # same preprocessing as the real data
        lmax, _ = lam_max(Xs)
        for e in ESTS:
            for g, r in enumerate(R_LIST):
                f1[e][k, g] += rates(estimate(Xs, r * lmax, e), adj.astype(bool))[2] / N_SIM

say(f"\nStep 1. Choosing lambda by simulation: p = {p}, N = {N}; networks ER (pi = {PI_ER}) and "
    f"chain; Q by DD and PO (lambda_min = {LAM_MIN}); {N_SIM} data sets per setting.")
say("Mean F1 of the estimated network (rows: lambda = r * lambda_max):")
head = "| r | " + " | ".join(f"{n} {m} glasso | {n} {m} AND" for n, m, _, _ in settings) \
       + " | mean glasso | mean AND |"
say(head)
say("|" + "---|" * (3 + 2 * len(settings)))
for g, r in enumerate(R_LIST):
    cells = []
    for k in range(len(settings)):
        cells += [f"{f1['glasso'][k, g]:.3f}", f"{f1['neigh_AND'][k, g]:.3f}"]
    say(f"| {r:g} | " + " | ".join(cells) + f" | {f1['glasso'][:, g].mean():.3f} | "
        f"{f1['neigh_AND'][:, g].mean():.3f} |")
best_r = {e: R_LIST[int(np.argmax(f1[e].mean(0)))] for e in ESTS}
lmax_real, S_real = lam_max(X)
lam = {e: best_r[e] * lmax_real for e in ESTS}
say(f"\nChosen lambda (best mean F1); lambda_max of the Sachs data = {lmax_real:.4f}:")
for e in ESTS:
    say(f"  {NAME[e]}: r = {best_r[e]:g}, lambda = {lam[e]:.4f}, mean F1 in simulation = "
        f"{f1[e].mean(0).max():.3f}")

# ================= 1. estimate the network on the Sachs data =================
say("\nNetwork estimated on the Sachs data with the chosen lambda:")
say("| estimator | lambda | edges | edges also in the literature network (20) |")
say("|---|---|---|---|")
E_real = {}
for e in ESTS:
    E_real[e] = estimate(X, lam[e], e)
    say(f"| {NAME[e]} | {lam[e]:.4f} | {E_real[e][iu].sum()} | {(E_real[e] & ref)[iu].sum()} |")

# ================= 2. stability: 50 random halves =================
freq = {e: np.zeros((p, p)) for e in ESTS}
for _ in range(N_HALVES):
    idx = rng.choice(N, size=N // 2, replace=False)
    for e in ESTS:
        freq[e] += estimate(X[idx], lam[e], e)
for e in ESTS:
    freq[e] /= N_HALVES
say(f"\nStep 2. Stability: {N_HALVES} random halves of {N // 2} cells, same lambda.")

# ================= 3. stable edges: threshold chosen from the data =================
# (a) Gap rule: selection frequencies pile up near 0 and near 1. Among frequencies of at least
#     0.5 (an edge must appear in most halves), the threshold is the midpoint of the widest gap,
#     so it separates "almost never" from "almost always" without a fixed number.
# (b) Sweep: number of edges that pass each threshold from 0.5 to 1 (robustness).
# (c) Consensus: edges stable under both estimators form the high-confidence core.
allf = np.concatenate([freq[e][iu] for e in ESTS])
vals = np.unique(np.r_[0.5, allf[allf >= 0.5], 1.0])
gap_k = int(np.argmax(np.diff(vals)))
THRESH = float((vals[gap_k] + vals[gap_k + 1]) / 2)
say(f"\nStep 3. Stable edges. Threshold from the gap rule: the widest gap in the selection "
    f"frequencies above 0.5 is between {vals[gap_k]:.2f} and {vals[gap_k + 1]:.2f}, so the "
    f"threshold is {THRESH:.2f}.")
sweep = np.round(np.arange(0.50, 1.001, 0.01), 2)
say("Edges passing each threshold (robustness check):")
say("| threshold | " + " | ".join(NAME[e] for e in ESTS) + " |")
say("|---|" + "---|" * len(ESTS))
for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
    say(f"| {t:.2f} | " + " | ".join(str(int((freq[e][iu] >= t).sum())) for e in ESTS) + " |")
stable = {}
for e in ESTS:
    stable[e] = freq[e] >= THRESH
    np.fill_diagonal(stable[e], False)
    say(f"\n{NAME[e]}: {stable[e][iu].sum()} stable edges")
    say("| edge | selection frequency | in literature network |")
    say("|---|---|---|")
    rows = sorted(((freq[e][i, j], nodes[i], nodes[j], ref[i, j])
                   for i, j in zip(*iu) if stable[e][i, j]), reverse=True)
    for fq, a, b, t in rows:
        say(f"| {a} - {b} | {fq:.0%} | {'yes' if t else 'no'} |")

core = stable["glasso"] & stable["neigh_AND"]
say(f"\nConsensus (high-confidence core): {core[iu].sum()} edges stable under both estimators, "
    f"{(core & ref)[iu].sum()} of them in the literature network.")
say("| edge | freq. glasso | freq. AND | in literature network |")
say("|---|---|---|---|")
for i, j in zip(*iu):
    if core[i, j]:
        say(f"| {nodes[i]} - {nodes[j]} | {freq['glasso'][i, j]:.0%} | "
            f"{freq['neigh_AND'][i, j]:.0%} | {'yes' if ref[i, j] else 'no'} |")
for e, other in [("glasso", "neigh_AND"), ("neigh_AND", "glasso")]:
    only = [f"{nodes[i]}-{nodes[j]}" for i, j in zip(*iu) if stable[e][i, j] and not stable[other][i, j]]
    say(f"Stable only under {NAME[e]} ({len(only)}): {', '.join(only) if only else 'none'}")

# ================= 4. trust =================
say(f"\nStep 4. Trust: the stable network with its estimated edge strengths is the truth; "
    f"{N_TRUST} data sets with N = {N}, p = {p}; same lambda.")
say("| estimator | stable edges | truth Q | TPR | FPR |")
say("|---|---|---|---|---|")
_, prec_all = graphical_lasso(S_real, alpha=lam["glasso"], mode="cd", max_iter=500)
for e in ESTS:
    A = stable[e]
    Qt = np.where(A, prec_all, 0.0)
    np.fill_diagonal(Qt, np.diag(prec_all))
    dg = np.sqrt(np.diag(Qt)); Qt = Qt / np.outer(dg, dg)      # unit diagonal
    if np.linalg.eigvalsh(Qt).min() > 0:
        built = "estimated strengths"
    else:
        Qt = make_Q(A.astype(int), "DD", rng, lam_min=LAM_MIN)
        built = "built as in Part A (DD)"
    tpr, fpr = [], []
    for _ in range(N_TRUST):
        Xs = sample_gaussian(Qt, N, rng)
        Xs = (Xs - Xs.mean(0)) / Xs.std(0)
        a_, b_, _ = rates(estimate(Xs, lam[e], e), A)
        tpr.append(a_); fpr.append(b_)
    say(f"| {NAME[e]} | {A[iu].sum()} | {built} | {np.mean(tpr):.2f} ± {np.std(tpr, ddof=1):.2f} | "
        f"{np.mean(fpr):.3f} ± {np.std(fpr, ddof=1):.3f} |")

# ================= outputs =================
open(os.path.join(ROOT, "results", "partB_sachs.md"), "w").write("\n".join(out_lines) + "\n")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a_, e in zip(ax, ESTS):
    im = a_.imshow(freq[e], vmin=0, vmax=1, cmap="viridis")
    a_.set_xticks(range(p)); a_.set_xticklabels(nodes, rotation=60, fontsize=8)
    a_.set_yticks(range(p)); a_.set_yticklabels(nodes, fontsize=8)
    a_.set_title(f"{NAME[e]}, r = {best_r[e]:g}", fontsize=10)
    for i, j in zip(*np.where(stable[e])):
        a_.plot(j, i, "s", mfc="none", mec="white", ms=9)
fig.colorbar(im, ax=ax, fraction=0.03, label="selection frequency")
fig.suptitle(f"Sachs: selection frequency over {N_HALVES} random halves "
             f"(white squares: stable edges, >= {THRESH:.0%})", fontsize=10)
fig.savefig(os.path.join(ROOT, "figures", "partB_selection.png"), dpi=150, bbox_inches="tight")

# threshold figure: histogram of selection frequencies, and edges passing each threshold
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 3.8))
cols = {"glasso": "#2471a3", "neigh_AND": "#d35400"}
for e in ESTS:
    ax2[0].hist(freq[e][iu], bins=np.linspace(0, 1, 21), alpha=0.55, color=cols[e], label=NAME[e])
    ax2[1].plot(sweep, [(freq[e][iu] >= t).sum() for t in sweep], color=cols[e], lw=2, label=NAME[e])
for a_ in ax2:
    a_.axvline(THRESH, color="black", ls="--", lw=1)
    a_.grid(alpha=0.3); a_.legend(fontsize=8)
ax2[0].set_xlabel("selection frequency over 50 halves"); ax2[0].set_ylabel("number of pairs")
ax2[0].set_title("Frequencies pile up at 0 and 1", fontsize=10)
ax2[1].set_xlabel("threshold"); ax2[1].set_ylabel("edges passing the threshold")
ax2[1].set_title(f"Edges kept vs threshold (dashed: gap-rule threshold {THRESH:.2f})", fontsize=10)
fig2.tight_layout()
fig2.savefig(os.path.join(ROOT, "figures", "partB_threshold.png"), dpi=150)
print("saved figures/partB_threshold.png")
print("\nsaved results/partB_sachs.md, figures/partB_selection.png")
