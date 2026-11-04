# Exploratory (disclosed) check: is the alpha-meter a monotone function of the selection share
# for mixed selection+migration dynamics? Global segment fit and pointwise osculating alpha (K=3).
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

rng = np.random.default_rng(1)
lams = np.round(np.linspace(-1.5, 2.5, 401), 6)


def ratio(X, lam):
    if abs(lam) < 1e-12:
        Th = np.log(X[:, :-1] / X[:, -1:])
        Th = Th - Th.mean(0)
        s = np.linalg.svd(Th, compute_uv=False)
        return s[1] / s[0]
    s = np.linalg.svd(X ** lam, compute_uv=False)
    return np.sqrt((s[2:] ** 2).sum()) / s[1]


def F_sel(x, f):
    return x * (f - x @ f)


def F_mig(x, m, mu):
    return mu * (m - x)


def field(x, f, m, mu, w):
    return w * F_sel(x, f) + (1 - w) * F_mig(x, m, mu)


def jac(x, f, m, mu, w, eps=1e-7):
    K = len(x)
    J = np.zeros((K, K))
    for j in range(K):
        e = np.zeros(K)
        e[j] = eps
        J[:, j] = (field(x + e, f, m, mu, w) - field(x - e, f, m, mu, w)) / (2 * eps)
    return J


def fisher_norm(v, x):
    return np.sqrt(np.sum(v * v / x))


def osculating_alpha(x, v, a):
    # K=3: tangent plane {sum=0}, Fisher metric diag(1/x); normal n in tangent plane, g-orthogonal to v.
    K = len(x)
    basis = np.array([[1, -1, 0], [0, 1, -1]], float).T if K == 3 else None
    G = basis.T @ np.diag(1 / x) @ basis
    vc = np.linalg.lstsq(basis, v, rcond=None)[0]
    nc = np.array([-(G @ vc)[1], (G @ vc)[0]])  # g-orthogonal to v in coordinates
    n = basis @ nc
    gx = lambda u, w_: np.sum(u * w_ / x)
    q = np.sum(v * v / x)
    wv = v * v / x - x * q
    den = gx(wv, n)
    if abs(den) < 1e-12 * (fisher_norm(wv, x) * fisher_norm(n, x) + 1e-300):
        return np.nan
    c = gx(a, n) / den
    return 2 * c - 1


glob_rows, loc_rows = [], []
for draw in range(40):
    x0 = rng.dirichlet(np.ones(3) * 2)
    f = rng.normal(size=3)
    m = rng.dirichlet(np.ones(3) * 2)
    mu = rng.uniform(0.2, 1.0)
    for w in np.linspace(0, 1, 11):
        sol = solve_ivp(lambda t, x: field(x, f, m, mu, w), (0, 2.0), x0, rtol=1e-11, atol=1e-13, dense_output=True)
        ts = np.linspace(0, 2, 40)
        X = sol.sol(ts).T
        X /= X.sum(1, keepdims=True)
        # selection share rho along the segment (Fisher norms of the two weighted fields)
        ns = np.array([fisher_norm(w * F_sel(x, f), x) for x in X])
        nm = np.array([fisher_norm((1 - w) * F_mig(x, m, mu), x) for x in X])
        rho = ns.sum() / (ns.sum() + nm.sum() + 1e-300)
        r = np.array([ratio(X, l) for l in lams])
        glob_rows.append((draw, w, rho, 1 - 2 * lams[int(np.argmin(r))], r.min()))
        for x in X[5:-5:5]:
            v = field(x, f, m, mu, w)
            a = jac(x, f, m, mu, w) @ v
            rl = fisher_norm(w * F_sel(x, f), x) / (fisher_norm(w * F_sel(x, f), x) + fisher_norm((1 - w) * F_mig(x, m, mu), x) + 1e-300)
            loc_rows.append((draw, w, rl, osculating_alpha(x, v, a)))

G = np.array(glob_rows)
L = np.array(loc_rows)
mixed = (G[:, 1] > 0) & (G[:, 1] < 1)
print("GLOBAL segment fit, mixed segments only:")
print(f"  Spearman(alpha_hat, rho) pooled = {spearmanr(G[mixed, 3], G[mixed, 2]).correlation:.3f}")
print(f"  fraction with alpha_hat in [-1.2, 1.2] = {np.mean(np.abs(G[mixed, 3]) <= 1.2):.2f}")
per = []
for d in range(40):
    s = G[(G[:, 0] == d)]
    per.append(spearmanr(s[:, 3], s[:, 2]).correlation)
per = np.array(per)
print(f"  per-draw Spearman over w (incl. endpoints): median {np.nanmedian(per):.2f}, frac >= 0.8: {np.nanmean(per >= 0.8):.2f}")
print(f"  pure endpoints recovered: w=0 -> {np.allclose(G[G[:,1]==0,3], -1)}, w=1 -> {np.allclose(G[G[:,1]==1,3], 1)}")
ok = ~np.isnan(L[:, 3])
mixedL = ok & (L[:, 1] > 0) & (L[:, 1] < 1)
print("LOCAL osculating alpha (K=3), mixed only:")
print(f"  Spearman(alpha(t), rho(t)) pooled = {spearmanr(L[mixedL, 3], L[mixedL, 2]).correlation:.3f}")
print(f"  fraction alpha(t) in [-1.2, 1.2] = {np.mean(np.abs(L[mixedL, 3]) <= 1.2):.2f}; undefined points: {np.mean(~ok):.3f}")
qs = np.quantile(L[mixedL, 3], [0.05, 0.25, 0.5, 0.75, 0.95])
print(f"  alpha(t) quantiles 5/25/50/75/95%: {np.round(qs, 2).tolist()}")
for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0)]:
    sel = mixedL & (L[:, 2] >= lo) & (L[:, 2] < hi)
    if sel.sum():
        print(f"  rho in [{lo:.1f},{hi:.1f}): n={sel.sum():4d}  median alpha(t)={np.median(L[sel,3]):+.2f}  IQR=({np.quantile(L[sel,3],.25):+.2f},{np.quantile(L[sel,3],.75):+.2f})")
