"""Instantaneous WFR minimal-action split, 1-D finite volume.

Convention (Chizat et al.): action A = int rho (|v|^2 + delta^2 g^2), constraint
  d_t rho + div(rho v) = rho g.
Lagrange: v = -grad phi, g = -phi/delta^2,  -div(rho grad phi) + rho phi/delta^2 = -rho_dot.
Transport share = T/(T+R),  T = int rho |v|^2,  R = delta^2 int rho g^2.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def split_1d(rho, rhodot, h, delta, periodic=True):
    N = len(rho)
    if periodic:
        rho_f = 0.5 * (rho + np.roll(rho, -1))
        D = sp.lil_matrix((N, N))
        for i in range(N):
            D[i, i] = -1.0
            D[i, (i + 1) % N] = 1.0
        D = D.tocsr()
    else:
        rho_f = 0.5 * (rho[:-1] + rho[1:])
        D = sp.diags([-np.ones(N - 1), np.ones(N - 1)], [0, 1], shape=(N - 1, N)).tocsr()
    L = (D.T @ sp.diags(rho_f) @ D) / h**2
    K = (L + sp.diags(rho) / delta**2).tocsc()
    phi = -spla.spsolve(K, rhodot)
    T = h * phi @ (L @ phi)
    R = h * np.sum(rho * phi**2) / delta**2
    # amplitude decomposition of rhodot: transport part = -div(rho v) = -L phi ; reweight part = rho g
    tr_part = -(L @ phi)
    rw_part = -rho * phi / delta**2
    return T, R, tr_part, rw_part


def share(T, R):
    return T / (T + R)


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    deltas = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])

    # ---------- A. uniform / near-uniform periodic background ----------
    N = 512
    Lx = 2 * np.pi
    h = Lx / N
    x = np.arange(N) * h
    print("A. near-uniform density: share vs k^2 d^2/(1+k^2 d^2)  (continuum k; discrete k_d in parens)")
    for eps_bg in [0.0, 0.01]:
        rho = 1.0 + eps_bg * np.cos(3 * x)
        for k in [1, 2, 4, 8]:
            rd = np.cos(k * x)
            row = []
            maxerr = 0
            for d in deltas:
                T, R, trp, rwp = split_1d(rho, rd, h, d)
                s = share(T, R)
                kd2 = (2 / h * np.sin(k * h / 2)) ** 2
                pred = kd2 * d**2 / (1 + kd2 * d**2)
                predc = k**2 * d**2 / (1 + k**2 * d**2)
                # amplitude fraction of rhodot explained by transport
                amp = np.dot(trp, rd) / np.dot(rd, rd)
                maxerr = max(maxerr, abs(s - pred), abs(amp - pred))
                row.append((s, predc))
            print(f"  eps_bg={eps_bg:4.2f} k={k}: " + "  ".join(f"{s:.4f}/{p:.4f}" for s, p in row)
                  + f"   max|share-formula_discrete|={maxerr:.1e}")

    # ---------- B. envelope identity: transport share = 1 - dlnA/dln(delta^2) ----------
    print("\nB. envelope identity on strongly non-uniform rho (rho=1+0.9cos x, rhodot = cos5x + 0.3 sin2x + 0.1)")
    rho = 1.0 + 0.9 * np.cos(x)
    rd = np.cos(5 * x) + 0.3 * np.sin(2 * x) + 0.1
    for d in [0.1, 0.3, 1.0]:
        T, R, *_ = split_1d(rho, rd, h, d)
        e = 1e-4
        Tp, Rp, *_ = split_1d(rho, rd, h, d * np.exp(e))
        Tm, Rm, *_ = split_1d(rho, rd, h, d * np.exp(-e))
        dlnA = (np.log(Tp + Rp) - np.log(Tm + Rm)) / (2 * 2 * e)  # d ln A / d ln delta^2
        print(f"  delta={d}: share={share(T, R):.6f}  1-dlnA/dln(d^2)={1 - dlnA:.6f}")

    # ---------- C. strongly non-uniform: departure from Fourier filter ----------
    print("\nC. non-uniform background, same rhodot=cos(5x): share depends on rho (Fourier filter only for near-uniform)")
    for eps_bg in [0.0, 0.3, 0.9]:
        rho = 1.0 + eps_bg * np.cos(x)
        rd = np.cos(5 * x)
        out = [share(*split_1d(rho, rd, h, d)[:2]) for d in [0.1, 0.2, 0.5]]
        print(f"  eps_bg={eps_bg}: shares at delta=0.1,0.2,0.5 -> {np.round(out, 4)}")

    # ---------- D. Gaussian bump on zero background (far from near-uniform) ----------
    print("\nD. Gaussian bump (std w=1) on (almost) zero background, Neumann box [-10,10]")
    w = 1.0
    N = 2001
    xb = np.linspace(-10, 10, N)
    hb = xb[1] - xb[0]
    g = np.exp(-xb**2 / (2 * w**2)) / np.sqrt(2 * np.pi) / w
    rho = g + 1e-14
    z = xb / w
    cases = {
        "growth   (He0)": g * 1.0,
        "translate(He1)": g * z,               # = -c d_x rho with c = w
        "spread   (He2)": g * (z**2 - 1),      # mass-conserving widening / breathing
        "He3 mode      ": g * (z**3 - 3 * z),
    }
    nmode = {"growth   (He0)": 0, "translate(He1)": 1, "spread   (He2)": 2, "He3 mode      ": 3}
    for name, rd in cases.items():
        n = nmode[name]
        rows = []
        for d in [0.25, 0.5, 1.0, 2.0, 4.0]:
            T, R, *_ = split_1d(rho, rd, hb, d, periodic=False)
            pred = n * d**2 / (n * d**2 + w**2)
            rows.append(f"{share(T, R):.4f}/{pred:.4f}")
        print(f"  {name}: share/[n d^2/(n d^2+w^2)] at delta=.25,.5,1,2,4: " + "  ".join(rows))

    # translation of Gaussian: minimal-action velocity is c*delta^2/(delta^2+w^2), not c
    print("\n  translating Gaussian, true speed c=1: minimal-action v at x=0 vs c*d^2/(d^2+w^2)")
    c = 1.0
    rd = -c * np.gradient(g, hb)
    for d in [0.5, 1.0, 2.0]:
        N_ = len(rho)
        rho_f = 0.5 * (rho[:-1] + rho[1:])
        D = sp.diags([-np.ones(N_ - 1), np.ones(N_ - 1)], [0, 1], shape=(N_ - 1, N_)).tocsr()
        L = (D.T @ sp.diags(rho_f) @ D) / hb**2
        K = (L + sp.diags(rho) / d**2).tocsc()
        phi = -spla.spsolve(K, rd)
        v = -(D @ phi) / hb
        i0 = N_ // 2
        print(f"   delta={d}: v(0)={v[i0]:.4f}  predicted={c * d**2 / (d**2 + w**2):.4f}")

    # ---------- E. compact, skewed 'glider-like' profile: shape dependence ----------
    print("\nE. compact skewed profile (Lenia-like, support |x|<3), translation: share vs Gaussian-of-same-variance formula")
    a = 3.0
    prof = np.where(np.abs(xb) < a, (1 - (xb / a) ** 2) ** 3 * (1 + 0.6 * np.tanh(xb)), 0.0)
    prof = np.clip(prof, 0, None)
    m = prof.sum() * hb
    mu = (xb * prof).sum() * hb / m
    var = ((xb - mu) ** 2 * prof).sum() * hb / m
    rho = prof + 1e-14
    rd = -np.gradient(prof, hb)
    rows = []
    for d in [0.25, 0.5, 1.0, 2.0, 4.0]:
        T, R, *_ = split_1d(rho, rd, hb, d, periodic=False)
        rows.append(f"{share(T, R):.4f}/{d**2 / (d**2 + var):.4f}")
    print(f"  (std={np.sqrt(var):.3f}) " + "  ".join(rows))
