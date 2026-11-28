"""Part C2: reweighted graphical lasso by majorization-minimization (MM).

Objective (log-sum penalty):
    F(Q) = -log det Q + tr(S Q) + lam * sum_{i != j} log(|q_ij| + eps),   Q > 0.

MM step: log(|q| + eps) is concave in |q|, so it is majorized by its tangent at the
current point q_t, which gives the weighted graphical lasso
    Q_{t+1} = argmin  -log det Q + tr(S Q) + lam * sum_{i != j} w_ij |q_ij|,
    w_ij = 1 / (|q_ij^(t)| + eps).

The weighted graphical lasso is solved by block coordinate descent (Friedman, Hastie,
Tibshirani 2008). Each column update is a lasso with a weight per coefficient, solved
here by a column rescaling into sklearn's Lasso (see quad_lasso).
"""
import numpy as np
from sklearn.linear_model import Lasso


def quad_lasso(A, b, c):
    """argmin_beta 0.5 beta' A beta - b' beta + sum_k c_k |beta_k|,   A positive definite, c > 0.

    Write A = L L', y = L^{-1} b. Then the objective is 0.5||y - L' beta||^2 + sum c_k|beta_k|
    up to a constant. Put beta = gamma / c; then it is 0.5||y - X gamma||^2 + sum |gamma_k|
    with X = L' diag(1/c), which is sklearn's Lasso with alpha = 1/n (n = dimension).
    """
    n = A.shape[0]
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)
    X = L.T / c[None, :]
    model = Lasso(alpha=1.0 / n, fit_intercept=False, max_iter=10000, tol=1e-10)
    model.fit(X, y)
    return model.coef_ / c


def weighted_glasso(S, lam, Wt, tol=1e-6, max_iter=300):
    """Precision matrix minimising -log det Q + tr(SQ) + lam * sum_{i!=j} Wt_ij |q_ij|.

    Wt: symmetric p x p matrix of nonnegative weights (its diagonal is not used).
    With Wt = 1 this is the usual graphical lasso (checked against sklearn in
    code/check_weighted_glasso.py).
    """
    p = S.shape[0]
    W = S.copy()                              # working covariance; diagonal stays S_jj
    betas = [None] * p
    for _ in range(max_iter):
        W_old = W.copy()
        for j in range(p):
            idx = np.delete(np.arange(p), j)
            A = W[np.ix_(idx, idx)]
            beta = quad_lasso(A, S[idx, j], lam * Wt[idx, j])
            W[idx, j] = W[j, idx] = A @ beta
            betas[j] = beta
        if np.max(np.abs(W - W_old)) < tol:
            break
    # recover the precision matrix from the last column solutions
    Theta = np.zeros((p, p))
    for j in range(p):
        idx = np.delete(np.arange(p), j)
        theta_jj = 1.0 / (W[j, j] - W[idx, j] @ betas[j])
        Theta[j, j] = theta_jj
        Theta[idx, j] = -betas[j] * theta_jj
    return 0.5 * (Theta + Theta.T)


def logsum_objective(Q, S, lam, eps):
    """F(Q) of the log-sum penalised problem (the quantity MM decreases)."""
    p = Q.shape[0]
    off = ~np.eye(p, dtype=bool)
    _, logdet = np.linalg.slogdet(Q)
    return -logdet + np.trace(S @ Q) + lam * np.sum(np.log(np.abs(Q[off]) + eps))


def support(Q, thr=1e-8):
    E = np.abs(Q) > thr
    np.fill_diagonal(E, False)
    return E


def reweighted_glasso(S, lam, eps=0.01, n_steps=5, Q0=None):
    """MM for the log-sum penalty, started from the plain graphical lasso (or from Q0).

    Returns the final Q and the list of F(Q) values after each MM step (must not increase).
    """
    p = S.shape[0]
    off = ~np.eye(p, dtype=bool)
    Q = weighted_glasso(S, lam, np.ones((p, p))) if Q0 is None else Q0
    hist = [logsum_objective(Q, S, lam, eps)]
    for _ in range(n_steps):
        Wt = np.where(off, 1.0 / (np.abs(Q) + eps), 0.0)
        Q_new = weighted_glasso(S, lam, Wt)
        hist.append(logsum_objective(Q_new, S, lam, eps))
        same = np.array_equal(support(Q_new), support(Q))
        Q = Q_new
        if same:
            break
    return Q, hist
