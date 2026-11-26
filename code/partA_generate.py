"""Part A: networks, precision matrices Q, and samples.

Two networks:   Erdos-Renyi (ER) and hub.
Two Q methods (Oct 30 lecture): diagonal dominance (DD) and partial orthogonalization (PO).
"""
import numpy as np
from scipy.linalg import solve_triangular


# ---------- networks: symmetric 0/1 adjacency matrices with zero diagonal ----------

def er_adj(p, pi, rng):
    """Each pair (i<j) is an edge independently with probability pi."""
    upper = np.triu(rng.random((p, p)) < pi, k=1)
    return (upper | upper.T).astype(int)


def hub_adj(p, n_hubs=3, deg=12, rng=None):
    """Nodes 0..n_hubs-1 are hubs; each hub is joined to `deg` random other nodes."""
    A = np.zeros((p, p), dtype=int)
    for h in range(n_hubs):
        others = [v for v in range(p) if v != h]
        for v in rng.choice(others, size=deg, replace=False):
            A[h, v] = A[v, h] = 1
    return A


# ---------- precision matrices ----------

def dd_Q(adj, rng, theta=0.1):
    """Diagonal dominance.
    1) m_ij ~ U(-1,1) on edges (symmetric), 0 elsewhere.
    2) m_ii = sum_j |m_ij| + theta  (Gershgorin => M is positive definite).
    3) Q = D^{-1/2} M D^{-1/2}, D = diag(M)  => unit diagonal.
    """
    p = adj.shape[0]
    U = np.triu(rng.uniform(-1, 1, (p, p)), k=1)
    M = (U + U.T) * adj
    np.fill_diagonal(M, np.abs(M).sum(axis=1) + theta)
    d = np.sqrt(np.diag(M))
    return M / np.outer(d, d)


def po_Q(adj, rng):
    """Partial orthogonalization.
    Rows m_1..m_p of a Gaussian matrix. For i in order, remove from m_i its part in
    span{m_j : j < i, (i,j) not an edge}, then normalise. Q = M M^T.
    Earlier rows are final when used, so earlier zeros stay zero.
    """
    p = adj.shape[0]
    M = rng.standard_normal((p, p))
    for i in range(p):
        J = [j for j in range(i) if adj[i, j] == 0]
        if J:
            Qb, _ = np.linalg.qr(M[J].T)          # orthonormal basis of span{m_j, j in J}
            M[i] -= Qb @ (Qb.T @ M[i])
        M[i] /= np.linalg.norm(M[i])
    Q = M @ M.T
    Q[adj == 0] = 0.0                              # remove round-off on non-edges
    np.fill_diagonal(Q, 1.0)
    return Q


def set_lambda_min(Q, target=0.1):
    """Shift a unit-diagonal Q so that lambda_min(Q) = target, keeping unit diagonal.

    Q_new = (Q + delta I) / (1 + delta): each eigenvalue mu becomes (mu + delta)/(1 + delta),
    the diagonal stays 1 and the zeros stay zero. Solving (lmin + delta)/(1 + delta) = target
    gives delta = (target - lmin)/(1 - target). As in the Oct 30 lecture (slide 25), every
    generator is set to the same lambda_min, so the generators are compared at equal
    conditioning. Edge values are scaled by 1/(1 + delta).
    """
    lmin = np.linalg.eigvalsh(Q).min()
    delta = (target - lmin) / (1.0 - target)
    p = Q.shape[0]
    return (Q + delta * np.eye(p)) / (1.0 + delta)


def make_Q(adj, method, rng, lam_min=0.1):
    """Build Q on the network `adj` with method 'DD' or 'PO', then set lambda_min(Q).
    Pass lam_min=None to keep the raw generator output."""
    if method == "DD":
        Q = dd_Q(adj, rng)
    elif method == "PO":
        Q = po_Q(adj, rng)
    else:
        raise ValueError(method)
    return Q if lam_min is None else set_lambda_min(Q, lam_min)


# ---------- sampling ----------

def sample_gaussian(Q, N, rng):
    """N samples from N(0, Q^{-1}). Q = L L^T, x = L^{-T} z  =>  Cov(x) = Q^{-1}."""
    L = np.linalg.cholesky(Q)
    Z = rng.standard_normal((Q.shape[0], N))
    X = solve_triangular(L.T, Z, lower=False)      # p x N
    return X.T                                     # N x p


# ---------- summaries ----------

def unit_diag_strength(Q, adj):
    """Edge strengths |q~_ij| on true edges (i<j), with Q rescaled to unit diagonal."""
    d = np.sqrt(np.diag(Q))
    R = Q / np.outer(d, d)
    iu = np.triu_indices(Q.shape[0], k=1)
    mask = adj[iu] == 1
    return np.abs(R[iu][mask])


def checks(Q, adj):
    """Return a dict of sanity checks for a generated Q."""
    p = Q.shape[0]
    off = ~np.eye(p, dtype=bool)
    return {
        "lambda_min": np.linalg.eigvalsh(Q).min(),
        "unit_diag": np.allclose(np.diag(Q), 1.0),
        "zeros_off_graph": np.all(Q[(adj == 0) & off] == 0),
        "nonzero_on_edges": np.all(np.abs(Q[(adj == 1)]) > 0),
        "n_edges": int(adj.sum() // 2),
    }
