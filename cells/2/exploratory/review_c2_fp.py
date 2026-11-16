"""Independent numeric check of claim (c): stationary entropy production (EP) vs rho-weighted Fisher-Hodge residual.

World: K = 3 Wright-Fisher diffusion, Ito  dx = b dt + sqrt(2D) dW,  D = (diag x - x x') / (2 Ne),
b = selection x(f - x.f) + mutation u x Q.  Ne = N * dt = 100 * 0.1 = 10 matches the zoo default (pop_size 100, dt 0.1).

EP (independent of any Hodge projection): discretise the generator on the lattice x in (1/M) Z^3 with nearest-neighbour
jumps along e_p - e_q.  D = sum_{p<q} x_p x_q (e_p - e_q)(e_p - e_q)' / (2 Ne), so each edge gets diffusivity
kappa = m_p m_q / (2 Ne) at its midpoint m and a Scharfetter-Gummel tilt Pe = h (e_p - e_q).F(m), F = D^+ (b - div D).
First and second moments of this jump generator match (b, 2D) to O(h); the stationary pi is exact (sparse solve) and the
chain's EP  sum_edges (pi_a r_ab - pi_b r_ba) log(pi_a r_ab / (pi_b r_ba))  is exact for the chain and -> the diffusion EP
as M -> infinity.  The chain uses only rates, never a potential basis.

Hodge residual (the cell-2 readout with mu = rho): weighted least squares of b (or of b - div D) on gradient fields
D grad(phi), phi in a basis (monomials of degree <= dp, and monomials of degree <= dl times log x_i), norm
||v||^2_{D^{-1}} = 2 Ne sum_i v_i^2 / x_i, weights pi at lattice nodes with min_i x_i >= h (boundary nodes excluded:
there b' D^{-1} b = inf).  Claim (c): R_Fisher(b) = min_phi E_rho ||b - D grad phi||^2_{D^{-1}} = EP  (constant 1 in this
normalisation; = 2 Ne * the Shahshahani-metric residual).
"""
import itertools, sys, time
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

K, NE = 3, 10.0
F_SEL = np.array([0.8, -0.3, -0.5])
Q_UNI = np.ones((K, K)) / K - np.eye(K)
Q_CYC = np.roll(np.eye(K), 1, axis=1) - np.eye(K)                    # zoo 'cyclic': (xQ)_i = x_{i-1} - x_i

def drift(x, uU, uC, f=F_SEL):
    sel = x * (f - (x * f).sum(-1, keepdims=True))
    return sel + x @ (uU * Q_UNI + uC * Q_CYC)

def divD(x, Ne=NE):
    return (1.0 - K * x) / (2 * Ne)

def B(z):
    z = np.asarray(z, float)
    out = np.ones_like(z)
    nz = np.abs(z) > 1e-12
    out[nz] = z[nz] / np.expm1(z[nz])
    return out

def lattice(M):
    pts = np.array([(i, j, M - i - j) for i in range(M + 1) for j in range(M + 1 - i)])
    idx = {tuple(p): k for k, p in enumerate(pts)}
    return pts, idx

def chain(M, uU, uC, Ne=NE):
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
        bt = drift(m, uU, uC) - divD(m, Ne)
        Pe = h * ((2 * Ne * (bt[:, p]) ) / m[:, p] - (2 * Ne * (bt[:, q])) / m[:, q])
        kap = m[:, p] * m[:, q] / (2 * Ne) / h ** 2
        rf, rb = kap * B(-Pe), kap * B(Pe)
        rows += [a, b_]; cols += [b_, a]; vals += [rf, rb]
        edges.append((a, b_, rf, rb))
    n = len(pts)
    W = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    W = W - sps.diags(np.asarray(W.sum(1)).ravel())
    A = W.T.tolil(); A[0, :] = 1.0
    rhs = np.zeros(n); rhs[0] = 1.0
    pi = spla.spsolve(A.tocsc(), rhs)
    pi = np.clip(pi, 0, None); pi /= pi.sum()
    ep = 0.0
    for a, b_, rf, rb in edges:
        fa, fb = pi[a] * rf, pi[b_] * rb
        good = (fa > 0) & (fb > 0)
        ep += ((fa - fb)[good] * np.log(fa[good] / fb[good])).sum()
    return pts / M, pi, ep

def potential_basis_grad(x, dp, dl):
    """Ambient gradients dphi/dx (n, B, K) of: monomials x1^a x2^b x3^c, 1 <= a+b+c <= dp; and x^m log x_i, deg m <= dl."""
    G = []
    mons = [e for e in itertools.product(range(dp + 1), repeat=K) if 1 <= sum(e) <= dp]
    for e in mons:
        g = np.zeros_like(x)
        for i in range(K):
            if e[i]:
                ei = list(e); ei[i] -= 1
                g[:, i] = e[i] * np.prod(x ** np.array(ei), 1)
        G.append(g)
    lmons = [e for e in itertools.product(range(dl + 1), repeat=K) if sum(e) <= dl]
    for i in range(K):
        for e in lmons:
            mon = np.prod(x ** np.array(e), 1)
            g = np.zeros_like(x)
            g[:, i] += mon / x[:, i]
            for j in range(K):
                if e[j]:
                    ej = list(e); ej[j] -= 1
                    g[:, j] += e[j] * np.prod(x ** np.array(ej), 1) * np.log(x[:, i])
            G.append(g)
    return np.stack(G, 1)

def hodge_residual(x, w, v, dp, dl, Ne=NE):
    """min_phi sum_n w_n ||v_n - D grad phi||^2_{D^{-1}} over the basis; returns (residual, total)."""
    P = np.einsum("ni,ij->nij", x, np.eye(K)) - np.einsum("ni,nj->nij", x, x)
    Gb = np.einsum("nij,nbj->nbi", P, potential_basis_grad(x, dp, dl)) / (2 * Ne)   # D grad phi_b
    s = np.sqrt(2 * Ne * w[:, None] / x)                                             # D^{-1} norm on the tangent space
    A = (Gb * s[:, None, :]).transpose(0, 2, 1).reshape(-1, Gb.shape[1])
    y = (v * s).reshape(-1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(((y - A @ c) ** 2).sum()), float((y ** 2).sum())

def direct_firr(x, pi, M, uU, uC, Ne=NE):
    """f_irr = b - div D - D grad log rho by central differences of log pi on nodes with min x >= 2h; EP_direct."""
    h = 1.0 / M
    pts = np.rint(x * M).astype(int)
    lut = {tuple(p): k for k, p in enumerate(pts)}
    sel = np.nonzero((pts >= 2).all(1))[0]
    lp = np.log(np.maximum(pi, 1e-300))
    g = np.zeros((len(sel), K))
    # gradient of log rho in the chart (x1, x2): derivative along e1 - e3 and e2 - e3
    for c, d in enumerate(((1, 0, -1), (0, 1, -1))):
        d = np.array(d)
        fwd = np.array([lut[tuple(p)] for p in pts[sel] + d])
        bwd = np.array([lut[tuple(p)] for p in pts[sel] - d])
        g[:, c] = (lp[fwd] - lp[bwd]) / (2 * h)
    xs = x[sel]
    P2 = np.einsum("ni,ij->nij", xs[:, :2], np.eye(2)) - np.einsum("ni,nj->nij", xs[:, :2], xs[:, :2])
    Dg = np.einsum("nij,nj->ni", P2, g[:, :2]) / (2 * Ne)
    f = drift(xs, uU, uC)[:, :2] - divD(xs, Ne)[:, :2] - Dg
    f3 = np.c_[f, -f.sum(1)]
    return float((pi[sel] * 2 * Ne * (f3 ** 2 / xs).sum(1)).sum()), float(pi[sel].sum())

CASES = {"null: uniform mut (u=1) + sel": (1.0, 0.0),
         "uniform 0.5 + cyclic 1.0 + sel": (0.5, 1.0),
         "cyclic u=1 + sel (zoo 'cyclic')": (0.0, 1.0),
         "cyclic u=0.2 + sel": (0.0, 0.2),
         "null weak: uniform u=0.1 + sel": (0.1, 0.0)}

if __name__ == "__main__":
    # usage: review_c2_fp.py [M ...] [--cases i,j]   (default M = 40 80 160 320, cases 0-3)
    args = sys.argv[1:]
    sel_cases = [0, 1, 2, 3]
    if "--cases" in args:
        k = args.index("--cases"); sel_cases = [int(c) for c in args[k + 1].split(",")]; args = args[:k] + args[k + 2:]
    Ms = [int(a) for a in args] or [40, 80, 160, 320]
    for ci, (name, (uU, uC)) in enumerate(CASES.items()):
        if ci not in sel_cases:
            continue
        print(f"\n=== {name}   (Ne = {NE:g}, f = {F_SEL.tolist()}; boundary exponent alpha_min = "
              f"{2 * NE * uU / K:.2f} + cyclic part 2 Ne u x_(i-1))")
        print(f"{'M':>5} {'EP_chain':>10} {'EP_direct':>10} {'R(b) p6,l2':>11} {'R(b) p8,l3':>11} {'R(b-divD)':>10}"
              f" {'R(b) poly only':>14} {'R(b-divD) poly':>14} {'mass used':>9} {'ratio R/|b|^2':>13} {'sec':>5}")
        for M in Ms:
            t0 = time.time()
            x, pi, ep = chain(M, uU, uC)
            epd, md = direct_firr(x, pi, M, uU, uC)
            inner = (x > 0.5 / M).all(1)
            xi, wi = x[inner], pi[inner]
            b = drift(xi, uU, uC); bt = b - divD(xi)
            r1, tot = hodge_residual(xi, wi, b, 6, 2)
            r2, _ = hodge_residual(xi, wi, b, 8, 3)
            r3, _ = hodge_residual(xi, wi, bt, 8, 3)
            r4, _ = hodge_residual(xi, wi, b, 8, -1)
            r5, _ = hodge_residual(xi, wi, bt, 8, -1)
            print(f"{M:5d} {ep:10.4f} {epd:10.4f} {r1:11.4f} {r2:11.4f} {r3:10.4f} {r4:14.4f} {r5:14.4f}"
                  f" {wi.sum():9.4f} {r2 / tot:13.2e} {time.time() - t0:5.1f}")
