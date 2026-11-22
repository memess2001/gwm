"""Steelman check: noise covariance NOT Fisher (K = 3, Ne = 10).  Model A: fluctuating selection on top of WF drift;
model B: type-dependent offspring variance v (Gillespie 1974), D = P diag(v/x) P / (2 Ne), b gets -P v / Ne.

x = softmax(Y) with dY = f dt + dW_f, Cov(dW_f) = S dt  (environmental fitness noise), plus WF sampling noise:
  D = P / (2 Ne) + P S P / 2,      b = P f + u x Q + P (diag(S)/2 - S x)   (last term: Ito noise-induced drift, a
  replicator field with symmetric payoff -S, hence a Shahshahani gradient; see review_c2_derivations.py).
D has zero row sums, so D = sum_{p<q} (-D_pq) (e_p - e_q)(e_p - e_q)' and the lattice chain of review_c2_fp.py applies
with kappa_pq = -D_pq (non-negative here because S_11 x_1^2 <= 1/Ne for S_11 <= 0.1).
Reports: exact chain EP (= D^{-1}-Hodge residual = 'thermodynamic' non-gradient readout) vs the rho-weighted
Fisher/Shahshahani-Hodge residual of b (the mechanism readout of cell 2), both in units where the Fisher residual is
multiplied by 2 Ne (so that the two coincide when S = 0).
"""
import sys, time
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
sys.path.insert(0, "/home/user/gwm_repo/cells/2/exploratory")
from review_c2_fp import F_SEL, Q_UNI, Q_CYC, B, lattice, potential_basis_grad

K, NE = 3, 10.0

def Pm(x):
    return np.einsum("ni,ij->nij", x, np.eye(K)) - np.einsum("ni,nj->nij", x, x)

def Dmat(x, s2):
    """s2 = float: fluctuating selection on type 1 (S_11 = s2).  s2 = tuple v: type-dependent offspring variance
    (Gillespie): Cov per generation = P diag(v / x) P / N  ->  D = P diag(v/x) P / (2 Ne)."""
    P = Pm(x)
    if isinstance(s2, tuple):
        v = np.asarray(s2); xv = x * v; vb = xv.sum(-1)[:, None, None]
        PWP = (np.einsum("ni,ij->nij", xv, np.eye(K)) - np.einsum("ni,nj->nij", xv, x)
               - np.einsum("ni,nj->nij", x, xv) + vb * np.einsum("ni,nj->nij", x, x))      # = P diag(v/x) P
        return PWP / (2 * NE)
    S = np.diag([s2, 0.0, 0.0])
    return P / (2 * NE) + 0.5 * P @ S @ P

def drift(x, uU, uC, s2):
    sel = x * (F_SEL - (x * F_SEL).sum(-1, keepdims=True))
    if isinstance(s2, tuple):          # Gillespie: E[dx] gets -x (v - x.v) / Ne  (selection against variance)
        nd = -np.einsum("nij,j->ni", Pm(x), np.asarray(s2)) / NE
    else:
        S = np.diag([s2, 0.0, 0.0])
        nd = np.einsum("nij,nj->ni", Pm(x), np.diag(S)[None, :] / 2 - x @ S)
    return sel + x @ (uU * Q_UNI + uC * Q_CYC) + nd

def divD(x, s2, eps=1e-6):
    """(div D)_i = sum_{j<K} d D_ij / d x_j in the affine chart (x_1, x_2); returned as an ambient tangent vector."""
    out = np.zeros_like(x)
    for j in range(K - 1):
        e = np.zeros(K); e[j] = 1.0; e[K - 1] = -1.0
        dD = (Dmat(x + eps * e, s2) - Dmat(x - eps * e, s2)) / (2 * eps)
        out[:, : K - 1] += dD[:, : K - 1, j]
    out[:, K - 1] = -out[:, : K - 1].sum(1)
    return out

def tpinv(D):
    """Pseudo-inverse of a symmetric D with D 1 = 0 (relative cut 1e-10: removes the round-off eigenvalue along 1,
    and the exact zero normal direction at face midpoints)."""
    return np.linalg.pinv(D, rcond=1e-10, hermitian=True)

def force(x, v, s2):
    """F with D F = v on the tangent space, F orthogonal to 1 (pseudo-inverse)."""
    return np.einsum("nij,nj->ni", tpinv(Dmat(x, s2)), v)

def chain(M, uU, uC, s2):
    pts, idx = lattice(M)
    h = 1.0 / M
    rows, cols, vals, edges = [], [], [], []
    for p, q in ((0, 1), (1, 2), (2, 0)):
        d = np.zeros(K, int); d[p] += 1; d[q] -= 1
        nb = pts + d
        ok = (nb >= 0).all(1)
        a = np.nonzero(ok)[0]
        b_ = np.array([idx[tuple(v)] for v in nb[ok]])
        m = (pts[a] + nb[ok]) / (2.0 * M)
        Dm = Dmat(m, s2)
        kap = -Dm[:, p, q]
        assert (kap > -1e-15).all()
        F = force(m, drift(m, uU, uC, s2) - divD(m, s2), s2)
        Pe = h * (F[:, p] - F[:, q])
        rf, rb = kap / h ** 2 * B(-Pe), kap / h ** 2 * B(Pe)
        rows += [a, b_]; cols += [b_, a]; vals += [rf, rb]
        edges.append((a, b_, rf, rb))
    n = len(pts)
    W = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    W = W - sps.diags(np.asarray(W.sum(1)).ravel())
    A = W.T.tolil(); A[0, :] = 1.0
    rhs = np.zeros(n); rhs[0] = 1.0
    pi = np.clip(spla.spsolve(A.tocsc(), rhs), 0, None); pi /= pi.sum()
    ep = 0.0
    for a, b_, rf, rb in edges:
        fa, fb = pi[a] * rf, pi[b_] * rb
        g = (fa > 0) & (fb > 0)
        ep += ((fa - fb)[g] * np.log(fa[g] / fb[g])).sum()
    return pts / M, pi, ep

def residual(x, w, v, Ginv, dp=8, dl=3):
    """min_phi sum w ||v - Ginv grad phi||^2_{Ginv^+}, the norm taken on the tangent space."""
    Gm = tpinv(Ginv)
    L = np.linalg.cholesky(Gm + np.ones((K, K)))        # + 11' does not change the norm of tangent vectors
    Wb = np.einsum("nij,nbj->nbi", Ginv, potential_basis_grad(x, dp, dl))
    A = np.einsum("nji,nbj->nbi", L, Wb) * np.sqrt(w)[:, None, None]
    A = A.transpose(0, 2, 1).reshape(-1, Wb.shape[1])
    y = (np.einsum("nji,nj->ni", L, v) * np.sqrt(w)[:, None]).reshape(-1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(((y - A @ c) ** 2).sum())

if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    print(f"M = {M}, Ne = {NE:g}; columns: EP_chain | D^-1-Hodge(b - div D) [= EP by (c)] | 2Ne * Fisher-Hodge(b) | 2Ne * Fisher-Hodge(b - div D)")
    for name, uU, uC in (("uniform u=1 + sel (gradient mechanisms)", 1.0, 0.0), ("uniform 0.5 + cyclic 1 + sel", 0.5, 1.0)):
        for s2 in (0.0, 0.1, (0.6, 1.0, 1.4), (0.5, 1.0, 1.5)):
            t0 = time.time()
            x, pi, ep = chain(M, uU, uC, s2)
            inner = (x > 0.5 / M).all(1)
            xi, wi = x[inner], pi[inner]
            b = drift(xi, uU, uC, s2); bt = b - divD(xi, s2)
            r_ep = residual(xi, wi, bt, Dmat(xi, s2))
            r_f = 2 * NE * residual(xi, wi, b, Pm(xi))
            r_f2 = 2 * NE * residual(xi, wi, bt, Pm(xi))
            print(f"{name:40} noise {str(s2):16}: EP_chain {ep:8.4f} | D-Hodge {r_ep:8.4f} | Fisher(b) {r_f:8.4f} | Fisher(b-divD) {r_f2:8.4f}   ({time.time() - t0:.1f}s)")
