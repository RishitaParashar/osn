"""Part A experiment: ROC curves, and lambda chosen by 5-fold CV and by BIC.

Design (project handout, both versions):
  networks   Erdos-Renyi (pi = 0.1) and hub (3 hubs x 12 neighbours), p = 40
  Q methods  diagonal dominance (DD) and partial orthogonalization (PO), Oct 30 lecture
  N          20, 100, 400  (one with N < p)
  runs       35 per setting; Q fixed, fresh data in every run
  estimators graphical lasso and neighbourhood selection with the OR rule
  lambda     common relative grid lambda = r * lambda_max, r from 1 to 0.01

Every Q is shifted to lambda_min(Q) = 0.1 (Oct 30 lecture, slide 25). This keeps the two
generators comparable and avoids the near-singular Q that made glasso fail before; the
v2 handout no longer asks to report lambda_min.

For both estimators lambda_max = max_{j != k} |S_jk| (glasso: Check 1 of the glasso
slides; lasso of x_j on X_{-j}: max_k |x_k^T x_j| / N, the same number).

The 420 runs (12 settings x 35) are independent and run in parallel. Run (a, b, c, run)
uses its own seed [SEED, a, b, c, run], so results do not depend on the number of cores.

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
from sklearn.linear_model import lasso_path
from partA_generate import er_adj, hub_adj, make_Q, sample_gaussian, checks

# At the smallest lambda glasso can stop at max_iter; that is expected (the graph there is
# almost complete) and is not a failure. Set at import so every worker process inherits it.
warnings.filterwarnings("ignore")

P = 40
PI_ER = 0.1
HUBS, HUB_DEG = 3, 12
LAM_MIN = 0.1
N_LIST = [20, 100, 400]                    # N/p = 0.5, 2.5, 10
R_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 35
N_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
R_GRID = np.logspace(0, -2, 25)            # decreasing lambda
K_FOLDS = 5
GLASSO_MAX_ITER = 300
NETS = ["ER", "hub"]
METHODS = ["DD", "PO"]
ESTIMATORS = ["glasso", "neigh_OR"]
FIELDS = ["TPR", "FPR", "F1", "NEDGES", "BIC", "CV"]
SEED = 2026
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "partA_roc.npz")


# ---------- scoring ----------

def scores(E, truth):
    """TPR, FPR, F1 and number of edges over unordered pairs i<j (handout definitions)."""
    iu = np.triu_indices(truth.shape[0], k=1)
    e, t = E[iu], truth[iu]
    tp = np.sum(e & t); fp = np.sum(e & ~t); fn = np.sum(~e & t)
    return tp / t.sum(), fp / (t.size - t.sum()), 2 * tp / (2 * tp + fp + fn), e.sum()


# ---------- graphical lasso ----------

def glasso_fit(S, lam):
    """Precision matrix from sklearn's glasso, or None if the solver fails."""
    try:
        _, prec = graphical_lasso(S, alpha=lam, mode="cd", max_iter=GLASSO_MAX_ITER)
        return prec
    except Exception:
        return None


def edges_of(prec):
    E = np.abs(prec) > 1e-8
    np.fill_diagonal(E, False)
    return E


def neg_loglik(prec, S):
    """Gaussian negative log-likelihood per sample (up to constants): -log det Q + tr(SQ)."""
    _, logdet = np.linalg.slogdet(prec)
    return -logdet + np.trace(S @ prec)


def glasso_all(X, lams, folds):
    """Edges and BIC on all the data, and the 5-fold CV score, at each lambda."""
    N = X.shape[0]
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / N
    edges, bic = [], np.full(len(lams), np.nan)
    for i, lam in enumerate(lams):
        prec = glasso_fit(S, lam)
        if prec is None:
            edges.append(None)
            continue
        E = edges_of(prec)
        edges.append(E)
        bic[i] = N * neg_loglik(prec, S) + np.log(N) * (E.sum() // 2)
    cv = np.zeros(len(lams))
    for test in folds:
        train = np.setdiff1d(np.arange(N), test)
        mu = X[train].mean(0)
        Xtr, Xte = X[train] - mu, X[test] - mu
        Str, Ste = Xtr.T @ Xtr / len(train), Xte.T @ Xte / len(test)
        for i, lam in enumerate(lams):
            prec = glasso_fit(Str, lam)
            cv[i] += np.inf if prec is None else neg_loglik(prec, Ste) / len(folds)
    return edges, bic, cv


# ---------- neighbourhood selection ----------

def lasso_coefs(Xtr, j, lams):
    """Lasso path of column j on the others (same objective as our lasso_cd)."""
    others = np.delete(np.arange(Xtr.shape[1]), j)
    alphas, coefs, _ = lasso_path(Xtr[:, others], Xtr[:, j], alphas=lams, max_iter=5000)
    assert np.allclose(alphas, lams)
    return others, coefs                   # coefs: (p-1) x n_lams


def neigh_all(X, lams, folds):
    """OR-rule edges and BIC on all the data, and the 5-fold CV prediction error."""
    N, p = X.shape
    Xc = X - X.mean(0)
    B = np.zeros((len(lams), p, p), dtype=bool)
    bic = np.zeros(len(lams))
    for j in range(p):
        others, coefs = lasso_coefs(Xc, j, lams)
        nz = coefs != 0
        B[:, j, others] = nz.T
        rss = ((Xc[:, [j]] - Xc[:, others] @ coefs) ** 2).sum(0)
        bic += N * np.log(np.maximum(rss / N, 1e-12)) + np.log(N) * nz.sum(0)
    E = B | B.transpose(0, 2, 1)
    for i in range(len(lams)):
        np.fill_diagonal(E[i], False)
    cv = np.zeros(len(lams))
    for test in folds:
        train = np.setdiff1d(np.arange(N), test)
        mu = X[train].mean(0)
        Xtr, Xte = X[train] - mu, X[test] - mu
        for j in range(p):
            others, coefs = lasso_coefs(Xtr, j, lams)
            cv += ((Xte[:, [j]] - Xte[:, others] @ coefs) ** 2).sum(0) / N
    return list(E), bic, cv


# ---------- one run ----------

def one_run(a, b, c, run, Q, adj, N):
    rng = np.random.default_rng([SEED, a, b, c, run])
    X = sample_gaussian(Q, N, rng)
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / N
    lams = R_GRID * np.abs(S - np.diag(np.diag(S))).max()
    folds = np.array_split(rng.permutation(N), K_FOLDS)
    truth = adj.astype(bool)
    out, fails = {}, 0
    for est, fn in [("glasso", glasso_all), ("neigh_OR", neigh_all)]:
        E_list, bic, cv = fn(X, lams, folds)
        arr = {f: np.full(len(lams), np.nan) for f in ["TPR", "FPR", "F1", "NEDGES"]}
        for g, E in enumerate(E_list):
            if E is None:
                fails += 1
                continue
            arr["TPR"][g], arr["FPR"][g], arr["F1"][g], arr["NEDGES"][g] = scores(E, truth)
        arr["BIC"], arr["CV"] = bic, cv
        out[est] = arr
    return (a, b, c, run), out, fails


def main():
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(SEED)
    adjs = np.zeros((len(NETS), P, P), dtype=int)
    Qs = np.zeros((len(NETS), len(METHODS), P, P))
    for a, net in enumerate(NETS):
        adjs[a] = er_adj(P, PI_ER, rng) if net == "ER" else hub_adj(P, HUBS, HUB_DEG, rng)
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
             R_GRID=R_GRID, P=P, K_FOLDS=K_FOLDS, SEED=SEED, adjs=adjs, Qs=Qs,
             fails=fails, **res)
    print("glasso solver failures:", int(fails.sum()), "of",
          len(jobs) * len(R_GRID), "fits on all the data")
    print("saved", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
