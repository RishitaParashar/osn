"""Checks for code/weighted_glasso.py.
1. With all weights 1, our solver matches sklearn's graphical_lasso.
2. On a Part A style problem, the MM objective never increases.
3. Timing of one weighted fit at p = 60.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.covariance import graphical_lasso
from weighted_glasso import weighted_glasso, reweighted_glasso, logsum_objective, support

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

# 1. agreement with sklearn on a small problem
p = 10
A = rng.standard_normal((p, p)); Sig = A @ A.T / p + np.eye(p)
X = rng.multivariate_normal(np.zeros(p), Sig, size=300)
S = np.cov(X, rowvar=False, bias=True)
for lam in [0.02, 0.1, 0.3]:
    _, prec_sk = graphical_lasso(S, alpha=lam, mode="cd", tol=1e-10, max_iter=5000)
    prec_ours = weighted_glasso(S, lam, np.ones((p, p)))
    print(f"lam={lam}: max |ours - sklearn| = {np.abs(prec_ours - prec_sk).max():.2e}, "
          f"same support: {np.array_equal(support(prec_ours), support(prec_sk))}")

# 2. MM objective on a p = 60 chain-like problem (Part A style, N = 100)
p = 60
Q_true = np.eye(p)
for i in range(p - 1):
    Q_true[i, i + 1] = Q_true[i + 1, i] = -0.4
Q_true = Q_true / np.sqrt(np.outer(np.diag(Q_true), np.diag(Q_true)))
Xs = rng.multivariate_normal(np.zeros(p), np.linalg.inv(Q_true), size=100)
Xc = Xs - Xs.mean(0); S = Xc.T @ Xc / 100
lam = 0.05 * np.abs(S - np.diag(np.diag(S))).max() * 4
t0 = time.time()
Q, hist = reweighted_glasso(S, lam, eps=0.01, n_steps=5)
print(f"p=60 timing (plain + MM): {time.time() - t0:.1f}s")
print("F after each MM step:", np.round(hist, 6))
print("monotone non-increasing:", bool(np.all(np.diff(hist) <= 1e-8)))
