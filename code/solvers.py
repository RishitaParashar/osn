"""Own implementations of the two estimators used in the project.

1. lasso_cd:               coordinate-descent lasso for one regression (our own solver).
2. neighbourhood_selection: one lasso per node, combined with the AND rule (default).
3. neighbourhood_path:      the same, for a whole grid of lambda values (used by Parts A and B).
4. glasso_support:          graphical lasso from sklearn (existing software) -> edge support.

The AND rule is the one used in every result of the project. The OR rule is kept only so that
the comparison in the report can be reproduced.
"""
import numpy as np
from sklearn.linear_model import lasso_path


def soft(z, t):
    """Soft-thresholding operator S(z, t) = sign(z) * max(|z| - t, 0)."""
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lasso_cd(X, y, lam, w=None, n_iter=500, tol=1e-8):
    """Coordinate descent for the lasso with optional per-coefficient weights w.

    Each coordinate update has a closed form: fix all other b_k, then
    b_j <- S(x_j^T r / n, lam * w_j) / (x_j^T x_j / n),
    where r is the current residual y - X b.
    """
    n, p = X.shape
    w = np.ones(p) if w is None else np.asarray(w, dtype=float)
    b = np.zeros(p)
    col_sq = (X ** 2).sum(axis=0) / n          # x_j^T x_j / n
    r = y.astype(float).copy()                  # residual, starts at y since b = 0
    for _ in range(n_iter):
        b_old = b.copy()
        for j in range(p):
            if col_sq[j] == 0:
                continue
            # partial residual without coordinate j
            rho = X[:, j] @ r / n + col_sq[j] * b[j]
            new = soft(rho, lam * w[j]) / col_sq[j]
            if new != b[j]:
                r += X[:, j] * (b[j] - new)     # keep r = y - X b up to date
                b[j] = new
        if np.max(np.abs(b - b_old)) < tol:
            break
    return b


def _combine(nz, rule):
    """Symmetric adjacency from the directed selection matrix nz (nz[j, k] = k picked by j)."""
    adj = (nz & nz.T) if rule == "and" else (nz | nz.T)
    np.fill_diagonal(adj, False)
    return adj


def neighbourhood_selection(X, lam, rule="and"):
    """Estimate the support of Q by regressing each variable on all others (one lasso per node).

    Node j regresses X_j on X_{-j}; a nonzero coefficient beta_jk means k is a neighbour of j.
    The rule decides when the two directions disagree:
      'and' (default): edge (j,k) only if beta_jk != 0 AND beta_kj != 0
      'or'           : edge (j,k) if beta_jk != 0 OR  beta_kj != 0
    Uses our lasso_cd. Returns a boolean p x p symmetric matrix with zero diagonal.
    """
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    nz = np.zeros((p, p), dtype=bool)
    for j in range(p):
        idx = [k for k in range(p) if k != j]
        nz[j, idx] = lasso_cd(Xc[:, idx], Xc[:, j], lam) != 0
    return _combine(nz, rule)


def neighbourhood_path(X, lams, rule="and"):
    """The same estimator for a list of lambdas, using sklearn's lasso_path for speed.

    sklearn's lasso_path minimises the same objective as lasso_cd (see check_solver.py, which
    shows the two agree). Returns a list of boolean p x p matrices, one per lambda.
    """
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    lams = np.asarray(lams, dtype=float)
    B = np.zeros((len(lams), p, p), dtype=bool)
    for j in range(p):
        others = np.delete(np.arange(p), j)
        alphas, coefs, _ = lasso_path(Xc[:, others], Xc[:, j], alphas=lams, max_iter=5000)
        B[:, j, others] = (coefs != 0).T
    return [_combine(B[g], rule) for g in range(len(lams))]


def glasso_support(X, lam):
    """Graphical lasso from sklearn (existing software) -> boolean support."""
    from sklearn.covariance import graphical_lasso
    S = np.cov(X, rowvar=False)
    _, prec = graphical_lasso(S, alpha=lam, max_iter=500)
    adj = np.abs(prec) > 1e-8
    np.fill_diagonal(adj, False)
    return adj, prec
