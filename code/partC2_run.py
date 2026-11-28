"""Part C2: graphical lasso vs reweighted (log-sum) graphical lasso, chain network, DD Q, N = 100.

Same Q and the same data as the main Part A run (run seeds [SEED, a, b, c, run], with
a = chain, b = DD, c = N = 100), so the glasso rows can be compared with the Part A results.
lambda is chosen by BIC for the reweighted fit, and glasso is taken at the lambda with the
closest number of edges (the v2 handout asks for matched edge counts).
Metrics: F1 / TPR / FPR of the support, and the bias on the true edges
    bias = mean over true edges (i<j) of (|q_hat_ij| - |q_ij|).

Usage: python code/partC2_run.py [n_runs] [n_workers] [eps]   (defaults 35, all cores, 0.01)
Writes results/partC2.npz (eps = 0.01) or results/partC2_eps<eps>.npz (other eps)
"""
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")
import sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.covariance import graphical_lasso
from partA_generate import sample_gaussian
from weighted_glasso import reweighted_glasso, support
import partA_run as R

warnings.filterwarnings("ignore")
A_NET, B_METH, C_N = 1, 0, 1              # chain, DD, N = 100 (indices into partA_roc.npz)
MM_STEPS = 5
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
d = np.load(os.path.join(ROOT, "results", "partA_roc.npz"), allow_pickle=True)
Q_TRUE = d["Qs"][A_NET, B_METH]
ADJ = d["adjs"][A_NET].astype(bool)
N = int(d["N_LIST"][C_N])
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 35
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
EPS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
OUT_NAME = "partC2.npz" if EPS == 0.01 else f"partC2_eps{EPS:g}.npz"
iu = np.triu_indices(R.P, k=1)
ntrue = ADJ[iu].sum()


def metrics(Q, lam_edges):
    E = support(Q)
    tp = np.sum(E[iu] & ADJ[iu]); fp = np.sum(E[iu] & ~ADJ[iu]); fn = np.sum(~E[iu] & ADJ[iu])
    tpr = tp / ntrue; fpr = fp / (iu[0].size - ntrue); f1 = 2 * tp / (2 * tp + fp + fn)
    bias = np.mean(np.abs(Q[iu][ADJ[iu]]) - np.abs(Q_TRUE[iu][ADJ[iu]]))
    return tpr, fpr, f1, E.sum() // 2, bias


def one_run(run):
    rng = np.random.default_rng([R.SEED, A_NET, B_METH, C_N, run])
    X = sample_gaussian(Q_TRUE, N, rng)
    Xc = X - X.mean(0); S = Xc.T @ Xc / N
    lams = R.R_GRID * np.abs(S - np.diag(np.diag(S))).max()
    out = {k: np.full((len(lams), 6), np.nan) for k in ["glasso", "reweighted"]}   # tpr,fpr,f1,edges,bias,BIC
    for g, lam in enumerate(lams):
        try:
            _, prec = graphical_lasso(S, alpha=lam, mode="cd", max_iter=300)
            Qg = prec
        except Exception:
            Qg = None
        try:
            Qr, _ = reweighted_glasso(S, lam, eps=EPS, n_steps=MM_STEPS, Q0=Qg)
        except Exception:
            Qr = None
        for name, Q in [("glasso", Qg), ("reweighted", Qr)]:
            if Q is None:
                continue
            tpr, fpr, f1, ne, bias = metrics(Q, lam)
            _, logdet = np.linalg.slogdet(Q)
            bic = N * (-logdet + np.trace(S @ Q)) + np.log(N) * ne
            out[name][g] = [tpr, fpr, f1, ne, bias, bic]
    return run, out


def main():
    print(f"chain, DD, N = {N}, {RUNS} runs, {WORKERS} workers, eps = {EPS}, MM steps <= {MM_STEPS}",
          flush=True)
    t0 = time.time()
    results = {"glasso": np.full((RUNS, len(R.R_GRID), 6), np.nan),
               "reweighted": np.full((RUNS, len(R.R_GRID), 6), np.nan)}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for k, (run, out) in enumerate(pool.map(one_run, range(RUNS)), 1):
            for name in results:
                results[name][run] = out[name]
            print(f"  {k}/{RUNS} runs, {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(ROOT, "results", OUT_NAME), N=N, eps=EPS, lambdas=R.R_GRID,
             **{f"{k}_grid": v for k, v in results.items()})
    print("saved results/" + OUT_NAME)


if __name__ == "__main__":
    main()
