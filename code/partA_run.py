"""Part A experiment: ROC curves of two estimators on data with a known network.

Design (v2 handout, steps 1-6):
  networks    Erdos-Renyi (pi = 0.05) and chain (i -- i+1), p = 60
  Q methods   diagonal dominance (DD) and partial orthogonalization (PO), Oct 30 lecture
  N           40, 100, 200   (N = 40 < p)
  estimators  graphical lasso, and neighbourhood selection with the AND rule
  lambda      common relative grid lambda = r * lambda_max, r from 1 to 0.01 (25 values)
  runs        35 per setting; Q fixed, fresh data in every run
At every lambda the TPR and FPR are recorded; partA_plot.py averages them over runs (ROC).

Every Q is shifted to lambda_min(Q) = 0.1 (Oct 30 lecture, slide 25) so the two generators
are comparable and glasso does not fail on near-singular Q.

For both estimators lambda_max = max_{j != k} |S_jk| (glasso: Check 1 of the glasso slides;
lasso of x_j on X_{-j}: max_k |x_k^T x_j| / N, the same number).

Run (a, b, c, run) uses its own seed [SEED, a, b, c, run], so results do not depend on the
number of cores.

Usage: python code/partA_run.py [n_runs] [n_workers]      (defaults 35, all cores)
"""
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")         # one BLAS thread per worker process
import sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.covariance import graphical_lasso
from partA_generate import er_adj, chain_adj, make_Q, sample_gaussian, checks
from solvers import neighbourhood_path

# At the smallest lambda glasso can stop at max_iter; that is expected (the graph there is
# almost complete) and is not a failure. Set at import so every worker inherits it.
warnings.filterwarnings("ignore")

P = 60
PI_ER = 0.05
LAM_MIN = 0.1
N_LIST = [40, 100, 200]                    # N/p = 0.67, 1.67, 3.33
R_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 35
N_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
R_GRID = np.logspace(0, -2, 25)            # decreasing lambda
GLASSO_MAX_ITER = 300
NETS = ["ER", "chain"]
METHODS = ["DD", "PO"]
ESTIMATORS = ["glasso", "neigh_AND"]
FIELDS = ["TPR", "FPR", "NEDGES"]
SEED = 2026
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "partA_roc.npz")


def rates(E, truth):
    """TPR, FPR and number of edges over unordered pairs i<j (handout definitions)."""
    iu = np.triu_indices(truth.shape[0], k=1)
    e, t = E[iu], truth[iu]
    tp = np.sum(e & t); fp = np.sum(e & ~t)
    return tp / t.sum(), fp / (t.size - t.sum()), e.sum()


def glasso_edges(S, lams):
    """Edge sets of the graphical lasso at each lambda (None where the solver fails)."""
    out = []
    for lam in lams:
        try:
            _, prec = graphical_lasso(S, alpha=lam, mode="cd", max_iter=GLASSO_MAX_ITER)
            E = np.abs(prec) > 1e-8
            np.fill_diagonal(E, False)
            out.append(E)
        except Exception:
            out.append(None)
    return out


def neigh_and_edges(Xc, lams):
    """Neighbourhood selection with the AND rule, one lasso per node (solvers.neighbourhood_path)."""
    return neighbourhood_path(Xc, lams, rule="and")


def one_run(a, b, c, run, Q, adj, N):
    rng = np.random.default_rng([SEED, a, b, c, run])
    X = sample_gaussian(Q, N, rng)
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / N
    lams = R_GRID * np.abs(S - np.diag(np.diag(S))).max()
    truth = adj.astype(bool)
    out, fails = {}, 0
    for est, E_list in [("glasso", glasso_edges(S, lams)), ("neigh_AND", neigh_and_edges(Xc, lams))]:
        arr = {f: np.full(len(lams), np.nan) for f in FIELDS}
        for g, E in enumerate(E_list):
            if E is None:
                fails += 1
                continue
            arr["TPR"][g], arr["FPR"][g], arr["NEDGES"][g] = rates(E, truth)
        out[est] = arr
    return (a, b, c, run), out, fails


def main():
    rng = np.random.default_rng(SEED)
    adjs = np.zeros((len(NETS), P, P), dtype=int)
    Qs = np.zeros((len(NETS), len(METHODS), P, P))
    for a, net in enumerate(NETS):
        adjs[a] = er_adj(P, PI_ER, rng) if net == "ER" else chain_adj(P)
        for b, method in enumerate(METHODS):
            Qs[a, b] = make_Q(adjs[a], method, rng, lam_min=LAM_MIN)
            ch = checks(Qs[a, b], adjs[a])
            print(f"{net} {method}: {ch['n_edges']} edges, positive definite "
                  f"{ch['lambda_min'] > 0}, zeros off the network {ch['zeros_off_graph']}")

    shape = (len(NETS), len(METHODS), len(N_LIST), R_RUNS, len(R_GRID))
    res = {f"{e}_{f}": np.full(shape, np.nan) for e in ESTIMATORS for f in FIELDS}
    fails = np.zeros((len(NETS), len(METHODS), len(N_LIST)), dtype=int)
    jobs = [(a, b, c, r) for a in range(len(NETS)) for b in range(len(METHODS))
            for c in range(len(N_LIST)) for r in range(R_RUNS)]
    print(f"{len(jobs)} runs on {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(one_run, a, b, c, r, Qs[a, b], adjs[a], N_LIST[c])
                for a, b, c, r in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            (a, b, c, r), out, nf = fut.result()
            fails[a, b, c] += nf
            for est in ESTIMATORS:
                for f in FIELDS:
                    res[f"{est}_{f}"][a, b, c, r] = out[est][f]
            if k % max(1, len(jobs) // 10) == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} runs done, {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, NETS=NETS, METHODS=METHODS, ESTIMATORS=ESTIMATORS, N_LIST=N_LIST,
             R_GRID=R_GRID, P=P, SEED=SEED, adjs=adjs, Qs=Qs, fails=fails, **res)
    print("glasso solver failures:", int(fails.sum()), "of", len(jobs) * len(R_GRID), "fits")
    print("saved", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
