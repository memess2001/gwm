import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from estimators import *
from estimators import _prep

rng = np.random.default_rng(5)


def geo_rhs(t, y, c):
    x, v = y[:3], y[3:]
    q = np.sum(v * v / x)
    return np.concatenate([v, c * (v * v / x - x * q)])


print("== recovery on exact alpha-geodesics (K=3) ==")
for alpha in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
    out = []
    for k in range(3):
        while True:
            x0 = rng.dirichlet(np.ones(3) * 2)
            v0 = rng.normal(size=3); v0 -= v0.mean(); v0 *= 0.25 / np.linalg.norm(v0)
            sol = solve_ivp(geo_rhs, (0, 1), np.r_[x0, v0], args=((1 + alpha) / 2,), rtol=1e-11, atol=1e-13,
                            dense_output=True)
            X = sol.sol(np.linspace(0, 1, 40))[:3].T
            if X.min() > 1e-3:
                break
        X /= X.sum(1, keepdims=True)
        out.append((est_svd(X)[0], est_fisher(X)[0], est_anchor(X)[0]))
    print(f"alpha={alpha:+.1f}: svd/fisher/anchor = {np.round(out, 3).tolist()}")

print("== pure selection / migration ==")
x0 = rng.dirichlet(np.ones(3) * 2); f = rng.normal(size=3); m = rng.dirichlet(np.ones(3) * 2); mu = 0.6
for w in (0.0, 1.0):
    X = simulate(x0, f, m, mu, w)
    print(f"w={w}: svd {est_svd(X)[0]:+.3f} fisher {est_fisher(X)[0]:+.3f} anchor {est_anchor(X)[0]:+.3f} "
          f"osc0 {est_osc(X, f, m, mu, w) if w == 1 else float('nan'):+.3f} curvreg {est_curvreg(X, f, m, mu, w) if w == 1 else float('nan'):+.3f}")
# migration: N = 0 exactly, so curvreg = -1, osc = -1
X = simulate(x0, f, m, mu, 0.0)
print("migration osc0/curvreg:", est_osc(X, f, m, mu, 0.0), est_curvreg(X, f, m, mu, 0.0))

print("== FNS vs least_squares (direct 2-parameter Sampson minimisation) on mixed segments ==")


def ls_J(X, lam, a0):
    eta, Ch, B = _prep(X, np.array([lam]))
    eta, Ch = eta[0], Ch[0]

    def res(p):
        a = np.array([np.cos(p[0]) * np.cos(p[1]), np.cos(p[0]) * np.sin(p[1]), np.sin(p[0])])
        s = eta @ a
        q = np.einsum('i,nij,j->n', a, Ch, a)
        return s / np.sqrt(q)

    best = np.inf
    for st in range(8):
        if st == 0:
            a = a0 / np.linalg.norm(a0)
        else:
            a = rng.normal(size=3); a /= np.linalg.norm(a)
        p0 = [np.arcsin(np.clip(a[2], -1, 1)), np.arctan2(a[1], a[0])]
        r = least_squares(res, p0, xtol=1e-15, ftol=1e-15, gtol=1e-15)
        best = min(best, np.sum(r.fun ** 2))
    return best


for k in range(4):
    x0 = rng.dirichlet(np.ones(3) * 2); f = rng.normal(size=3); m = rng.dirichlet(np.ones(3) * 2)
    mu = rng.uniform(0.2, 1); w = rng.uniform(0.2, 0.8)
    X = simulate(x0, f, m, mu, w)
    t0 = time.time(); J, A = sampson_profile(X); t1 = time.time()
    idx = [0, 100, 150, 200, 250, 300, 400, int(np.argmin(J))]
    rel = []
    for i in idx:
        Jl = ls_J(X, LAMS1[i], A[i])
        rel.append((LAMS1[i], J[i], Jl))
    print(f"seg {k}: fns time {t1 - t0:.2f}s  (lam, J_fns, J_ls): " +
          ", ".join(f"({l:+.2f},{a:.2e},{b:.2e})" for l, a, b in rel))
