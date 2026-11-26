"""Checks for the Part A generators (run once; prints a short report)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from partA_generate import (er_adj, hub_adj, make_Q, set_lambda_min, sample_gaussian,
                            checks, unit_diag_strength)

rng = np.random.default_rng(1)
p = 40
for net_name, make_adj in [("ER", lambda: er_adj(p, 0.1, rng)),
                           ("hub", lambda: hub_adj(p, 3, 12, rng))]:
    for method in ["DD", "PO"]:
        adj = make_adj()
        Q_raw = make_Q(adj, method, rng, lam_min=None)
        Q = set_lambda_min(Q_raw, 0.1)
        c = checks(Q, adj)
        s = unit_diag_strength(Q, adj)
        print(f"{net_name:3s} {method}: edges={c['n_edges']:3d} "
              f"lambda_min raw={np.linalg.eigvalsh(Q_raw).min():.3f} -> {c['lambda_min']:.3f} "
              f"unit_diag={c['unit_diag']} zeros_ok={c['zeros_off_graph']} edges_nonzero={c['nonzero_on_edges']} "
              f"strength median={np.median(s):.2f} range=({s.min():.2f},{s.max():.2f})")

# covariance check: large-N sample covariance should approach Q^{-1}
adj = er_adj(p, 0.1, rng)
Q = make_Q(adj, "DD", rng)
X = sample_gaussian(Q, 400000, rng)
S = X.T @ X / X.shape[0]
err = np.max(np.abs(S - np.linalg.inv(Q)))
print(f"sampler check: max |S - Q^-1| with N=400000 = {err:.4f} (Q^-1 entries ~ {np.max(np.abs(np.linalg.inv(Q))):.2f})")
