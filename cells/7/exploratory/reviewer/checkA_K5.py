import numpy as np
from estimators import ratio, LAMS0
rng = np.random.default_rng(11)
# (A) converse: alpha-mixture path x(s) ∝ [(1-s) p^lam + s q^lam]^(1/lam) has osculating alpha == alpha everywhere (K=3)
def path(s, p, q, lam):
    if abs(lam) < 1e-12:
        y = np.exp((1 - s)[:, None] * np.log(p) + s[:, None] * np.log(q))
    else:
        y = ((1 - s)[:, None] * p ** lam + s[:, None] * q ** lam) ** (1 / lam)
    return y / y.sum(1, keepdims=True)
def osc_from_path(X, h):
    v = (X[2:] - X[:-2]) / (2 * h); a = (X[2:] - 2 * X[1:-1] + X[:-2]) / h ** 2; x = X[1:-1]
    n = np.cross(np.ones_like(v), v / x)
    q = np.sum(v * v / x, 1); Wv = v * v / x - x * q[:, None]
    return 2 * np.sum(a * n / x, 1) / np.sum(Wv * n / x, 1) - 1
worst = 0
for alpha in [-3, -2, -1.5, -0.5, 0, 0.3, 1, 2, 3]:
    lam = (1 - alpha) / 2
    for _ in range(5):
        p, q = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3))
        h = 1e-4; s = np.arange(0.05, 0.95, h)
        A = osc_from_path(path(s, p, q, lam), h)
        A = A[np.isfinite(A)]
        worst = max(worst, np.max(np.abs(A[::50] - alpha)))
print(f"(A) converse check: max |osc_alpha(path) - alpha| over 45 alpha-mixture paths = {worst:.1e}")

# K=5: first-order oddness under f -> -f for the unweighted SVD reading, and the dip fraction
K = 5
from scipy.integrate import solve_ivp
def Fw(x, f, m, mu, w): return w * x * (f - x @ f) + (1 - w) * mu * (m - x)
opp, dip, n = 0, 0, 0
for i in range(20):
    x0 = rng.dirichlet(np.ones(K) * 2); f = rng.normal(size=K); m = rng.dirichlet(np.ones(K) * 2); mu = rng.uniform(.2, 1)
    devs = []
    for sg in (1, -1):
        sol = solve_ivp(lambda t, x: Fw(x, sg * f, m, mu, 0.005), (0, 2), x0, rtol=1e-11, atol=1e-13, dense_output=True)
        X = sol.sol(np.linspace(0, 2, 40)).T; X /= X.sum(1, keepdims=True)
        lams = np.round(np.linspace(0.5, 1.5, 1001), 6)       # fine grid near alpha = -1
        r = np.array([ratio(X, l) for l in lams])
        devs.append(1 - 2 * lams[int(np.argmin(r))] + 1)
    if abs(devs[0]) > 1e-9 and abs(devs[1]) > 1e-9:
        n += 1; opp += np.sign(devs[0]) != np.sign(devs[1])
    dip += (devs[0] < -1e-9) + (devs[1] < -1e-9)
print(f"K=5, w=0.005, svd reading: opposite-sign pairs {opp}/{n}; segments with alpha_hat < -1: {dip}/40")
