"""Check our lasso_cd against sklearn's Lasso on the same problem.

sklearn's Lasso minimises (1/(2n))||y - Xb||^2 + alpha * ||b||_1, the same
objective as ours with lam = alpha, so the coefficients should match closely.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.linear_model import Lasso
from solvers import lasso_cd, neighbourhood_selection, glasso_support

rng = np.random.default_rng(0)
n, p = 200, 30
X = rng.standard_normal((n, p))
beta_true = np.zeros(p); beta_true[:5] = [1.5, -1, 0.8, 0.5, -0.3]
y = X @ beta_true + 0.5 * rng.standard_normal(n)

for lam in [0.05, 0.2, 0.5]:
    mine = lasso_cd(X, y, lam, n_iter=2000, tol=1e-10)
    ref = Lasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X, y).coef_
    print(f"lam={lam}: max|mine-sklearn| = {np.max(np.abs(mine-ref)):.2e}, "
          f"nonzeros mine={np.sum(mine!=0)} sklearn={np.sum(ref!=0)}")
