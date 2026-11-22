"""Steelman, strong environmental noise: EP vs Fisher-Hodge when the noise is NOT Fisher (K = 3).

Chart: alr theta_i = log(x_i / x_3), square lattice of spacing h on [-L, L]^2, nearest-neighbour edges along
a = (1,0), b = (0,1), c = (1,1).  Noise per unit time: WF sampling P/Ne plus isotropic fitness noise P (s2 I) P, i.e.
in theta  D = (diag(1/x1, 1/x2) + 11'/x3)/(2Ne) + (s2/2)[[2,1],[1,2]]
          = (1/(2Ne x1) + s2/2) aa' + (1/(2Ne x2) + s2/2) bb' + (1/(2Ne x3) + s2/2) cc'   (all weights > 0).
The x-chart Ito drift b_x = selection + mutation - s2 P x (noise-induced, a Shahshahani gradient) is mapped to the
theta-chart Ito drift; Scharfetter-Gummel chain as in review_c2_fp.py (reflecting box).  Exact chain EP vs the
rho-weighted Hodge residuals of b_x computed in x at the lattice nodes:  2Ne * Fisher-Hodge(b_x)  and  D_x^{-1}-Hodge
(b_x - div D_x), D_x = P/(2Ne) + s2 P^2 / 2.  Check: s2 = 0, Ne = 10 must reproduce review_c2_fp.py (0.7211).
"""
import sys, time
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
sys.path.insert(0, "/home/user/gwm_repo/cells/2/exploratory")
from review_c2_fp import F_SEL, Q_UNI, Q_CYC, B, potential_basis_grad

K = 3

def x_of(th):
    z = np.c_[th, np.zeros(len(th))]
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)

def Pm(x):
    return np.einsum("ni,ij->nij", x, np.eye(K)) - np.einsum("ni,nj->nij", x, x)

def Dx(x, Ne, s2):
    P = Pm(x)
    return P / (2 * Ne) + 0.5 * s2 * P @ P

def bx(x, uU, uC, s2):
    sel = x * (F_SEL - (x * F_SEL).sum(-1, keepdims=True))
    return sel + x @ (uU * Q_UNI + uC * Q_CYC) - s2 * np.einsum("nij,nj->ni", Pm(x), x)

def kap(x, Ne, s2):          # weights of a, b, c
    return np.c_[1 / (2 * Ne * x[:, 0]) + s2 / 2, 1 / (2 * Ne * x[:, 1]) + s2 / 2, 1 / (2 * Ne * x[:, 2]) + s2 / 2]

def Dth(th, Ne, s2):
    k = kap(x_of(th), Ne, s2)
    D = np.zeros((len(th), 2, 2))
    D[:, 0, 0] = k[:, 0] + k[:, 2]; D[:, 1, 1] = k[:, 1] + k[:, 2]; D[:, 0, 1] = D[:, 1, 0] = k[:, 2]
    return D

def bth(th, uU, uC, Ne, s2):
    """Ito drift of theta_i = log x_i - log x_3:  b_i/x_i - b_3/x_3 - D_ii/x_i^2 + D_33/x_3^2  (D = D_x)."""
    x = x_of(th); b = bx(x, uU, uC, s2); D = Dx(x, Ne, s2)
    out = np.zeros((len(th), 2))
    for i in range(2):
        out[:, i] = b[:, i] / x[:, i] - b[:, 2] / x[:, 2] - D[:, i, i] / x[:, i] ** 2 + D[:, 2, 2] / x[:, 2] ** 2
    return out

def div_th(th, Ne, s2, eps=1e-5):
    out = np.zeros((len(th), 2))
    for j in range(2):
        e = np.zeros(2); e[j] = eps
        out += (Dth(th + e, Ne, s2)[:, :, j] - Dth(th - e, Ne, s2)[:, :, j]) / (2 * eps)
    return out

def div_x(x, Ne, s2, eps=1e-6):
    out = np.zeros_like(x)
    for j in range(K - 1):
        e = np.zeros(K); e[j] = 1.0; e[K - 1] = -1.0
        dD = (Dx(x + eps * e, Ne, s2) - Dx(x - eps * e, Ne, s2)) / (2 * eps)
        out[:, : K - 1] += dD[:, : K - 1, j]
    out[:, K - 1] = -out[:, : K - 1].sum(1)
    return out

def chain(L, h, uU, uC, Ne, s2):
    g = np.arange(-L, L + h / 2, h); n1 = len(g)
    T1, T2 = np.meshgrid(g, g, indexing="ij")
    th = np.c_[T1.ravel(), T2.ravel()]
    I1, I2 = np.meshgrid(np.arange(n1), np.arange(n1), indexing="ij")
    I1, I2 = I1.ravel(), I2.ravel()
    rows, cols, vals, edges = [], [], [], []
    for k, d in enumerate(((1, 0), (0, 1), (1, 1))):
        ok = (I1 + d[0] < n1) & (I2 + d[1] < n1)
        a = np.nonzero(ok)[0]
        b_ = (I1[ok] + d[0]) * n1 + (I2[ok] + d[1])
        m = (th[a] + th[b_]) / 2
        F = np.linalg.solve(Dth(m, Ne, s2), (bth(m, uU, uC, Ne, s2) - div_th(m, Ne, s2))[:, :, None])[:, :, 0]
        Pe = h * (F @ np.array(d, float))
        kk = kap(x_of(m), Ne, s2)[:, k] / h ** 2
        rf, rb = kk * B(-Pe), kk * B(Pe)
        rows += [a, b_]; cols += [b_, a]; vals += [rf, rb]
        edges.append((a, b_, rf, rb))
    n = len(th)
    W = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    W = W - sps.diags(np.asarray(W.sum(1)).ravel())
    A = W.T.tolil(); A[n // 2, :] = 1.0
    rhs = np.zeros(n); rhs[n // 2] = 1.0
    pi = np.clip(spla.spsolve(A.tocsc(), rhs), 0, None); pi /= pi.sum()
    ep = 0.0
    for a, b_, rf, rb in edges:
        fa, fb = pi[a] * rf, pi[b_] * rb
        ok = (fa > 1e-300) & (fb > 1e-300)
        ep += ((fa - fb)[ok] * np.log(fa[ok] / fb[ok])).sum()
    edge_mass = pi.reshape(n1, n1); edge_mass = edge_mass[[0, -1], :].sum() + edge_mass[:, [0, -1]].sum()
    return x_of(th), pi, ep, edge_mass

def residual(x, w, v, Ginv, dp=8, dl=3):
    Gm = np.linalg.pinv(Ginv, rcond=1e-10, hermitian=True)
    Lc = np.linalg.cholesky(Gm + np.ones((K, K)))
    Wb = np.einsum("nij,nbj->nbi", Ginv, potential_basis_grad(x, dp, dl))
    A = (np.einsum("nji,nbj->nbi", Lc, Wb) * np.sqrt(w)[:, None, None]).transpose(0, 2, 1).reshape(-1, Wb.shape[1])
    y = (np.einsum("nji,nj->ni", Lc, v) * np.sqrt(w)[:, None]).reshape(-1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(((y - A @ c) ** 2).sum())

WEAK = (("gradient: weak PIM u=0.1 + sel", 0.1, 0.0, 1e4, 1.0),
        ("non-gradient: unif 0.1 + cyc 0.05 + sel", 0.1, 0.05, 1e4, 1.0),
        ("non-gradient: unif 0.05 + cyc 0.1 + sel", 0.05, 0.1, 1e4, 1.0))
STRONG = (("uniform u=1 + sel (gradient)", 1.0, 0.0, 100.0, 1.0),
          ("uniform u=1 + sel (gradient)", 1.0, 0.0, 1e4, 1.0),
          ("unif 0.5 + cyc 1 + sel (non-gradient)", 0.5, 1.0, 100.0, 0.0),
          ("unif 0.5 + cyc 1 + sel (non-gradient)", 0.5, 1.0, 100.0, 1.0),
          ("unif 0.5 + cyc 1 + sel (non-gradient)", 0.5, 1.0, 1e4, 1.0))

if __name__ == "__main__":
    # usage: review_c2_noise_theta.py [L h] [--strong | --weak]   (default: the h = 0.1 table incl. the check case)
    mode = [a for a in sys.argv[1:] if a.startswith("--")]
    nums = [a for a in sys.argv[1:] if not a.startswith("--")]
    L, h = (float(nums[0]), float(nums[1])) if len(nums) > 1 else (10.0, 0.1)
    # sanity: pure fitness noise + selection, no WF, no mutation -> theta drift must be f_i - f_3
    th0 = np.random.default_rng(0).normal(size=(5, 2))
    print("sanity (s2 only): theta Ito drift - (f_i - f_3) =", np.abs(bth(th0, 0, 0, 1e12, 1.0) - (F_SEL[:2] - F_SEL[2])).max())
    print(f"box [-{L:g},{L:g}]^2, h = {h:g}")
    print(f"{'case':44} {'Ne':>4} {'s2':>4} {'EP_chain':>9} {'D-Hodge':>8} {'2Ne*Fisher(b)':>13} {'box-edge mass':>13} {'sec':>5}")
    cases = STRONG if "--strong" in mode else WEAK if "--weak" in mode else None
    for name, uU, uC, Ne, s2 in cases or (("check vs x-lattice: unif 0.5 + cyc 1 + sel", 0.5, 1.0, 10.0, 0.0),
                                 ("uniform u=1 + sel (gradient)", 1.0, 0.0, 100.0, 0.0),
                                 ("uniform u=1 + sel (gradient)", 1.0, 0.0, 100.0, 0.2),
                                 ("uniform u=1 + sel (gradient)", 1.0, 0.0, 100.0, 1.0),
                                 ("unif 0.5 + cyc 1 + sel (non-gradient)", 0.5, 1.0, 100.0, 0.0),
                                 ("unif 0.5 + cyc 1 + sel (non-gradient)", 0.5, 1.0, 100.0, 0.2),
                                 ("unif 0.5 + cyc 1 + sel (non-gradient)", 0.5, 1.0, 100.0, 1.0)):
        t0 = time.time()
        x, pi, ep, em = chain(L, h, uU, uC, Ne, s2)
        keep = pi > 1e-14 * pi.max()
        xk, wk = x[keep], pi[keep] / pi[keep].sum()
        b = bx(xk, uU, uC, s2)
        r_d = residual(xk, wk, b - div_x(xk, Ne, s2), Dx(xk, Ne, s2))
        r_f = 2 * Ne * residual(xk, wk, b, Pm(xk))
        print(f"{name:44} {Ne:4g} {s2:4g} {ep:9.4f} {r_d:8.4f} {r_f:13.4f} {em:13.1e} {time.time() - t0:5.1f}"
              f"   Fisher-Hodge unscaled {r_f / (2 * Ne):.2e}")
