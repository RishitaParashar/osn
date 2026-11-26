"""Own implementations of the two estimators (Part A/B).

1. lasso_cd: coordinate-descent lasso for one regression.
   Objective:  (1/(2n)) ||y - X b||^2 + lam * sum_j w_j |b_j|
2. neighbourhood_selection: one lasso per node; combine with OR / AND rule.
3. glasso_support: wrapper around sklearn's graphical lasso (existing software).
"""
import numpy as np


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


def neighbourhood_selection(X, lam, rule="or"):
    """Estimate the support of Q by regressing each variable on all others.

    For node j, regress X_j on X_{-j}. Coefficient beta_jk != 0 means k is a
    neighbour of j. The two rules disagree when only one direction is selected:
      'or'  : edge (j,k) if beta_jk != 0 OR  beta_kj != 0
      'and' : edge (j,k) only if beta_jk != 0 AND beta_kj != 0
    Returns a boolean p x p symmetric adjacency matrix with zero diagonal.
    """
    n, p = X.shape
    Xc = X - X.mean(axis=0)                     # centre each variable
    B = np.zeros((p, p))
    for j in range(p):
        idx = [k for k in range(p) if k != j]
        beta = lasso_cd(Xc[:, idx], Xc[:, j], lam)
        B[j, idx] = beta
    nz = B != 0
    adj = (nz | nz.T) if rule == "or" else (nz & nz.T)
    np.fill_diagonal(adj, False)
    return adj


def glasso_support(X, lam):
    """Graphical lasso from sklearn (existing software) -> boolean support."""
    from sklearn.covariance import graphical_lasso
    S = np.cov(X, rowvar=False)
    _, prec = graphical_lasso(S, alpha=lam, max_iter=500)
    adj = np.abs(prec) > 1e-8
    np.fill_diagonal(adj, False)
    return adj, prec
