# Numerical check of the alpha-meter proposition on the probability simplex.
# Amari convention: alpha=+1 e-connection, alpha=-1 m-connection.
# alpha-geodesic equation in full mixture coordinates x (sum x = 1), derived from
# Gamma^(alpha) = (1+alpha)/2 * Gamma^(e) in eta-coords, Gamma^(m)=0:
#     x_k'' = c * ( v_k^2 / x_k - x_k * sum_j v_j^2 / x_j ),   c = (1+alpha)/2
# Claim: points of an alpha-geodesic satisfy rank([x^lam]) = 2 with lam = (1-alpha)/2
#        (lam = 0: centred theta = log(x_i/x_K) collinear).
import numpy as np
from scipy.integrate import solve_ivp


def rhs(t, y, c, K):
    x, v = y[:K], y[K:]
    q = np.sum(v * v / x)
    return np.concatenate([v, c * (v * v / x - x * q)])


def hit_boundary(t, y, c, K):
    return y[:K].min() - 1e-4


hit_boundary.terminal = True


def ratio(X, lam):
    if abs(lam) < 1e-12:
        Th = np.log(X[:, :-1] / X[:, -1:])
        Th = Th - Th.mean(0)
        s = np.linalg.svd(Th, compute_uv=False)
        return s[1] / s[0]
    s = np.linalg.svd(X ** lam, compute_uv=False)
    return np.sqrt((s[2:] ** 2).sum()) / s[1]


rng = np.random.default_rng(0)
lams = np.round(np.linspace(-1.0, 2.0, 301), 6)
for K in (3, 4, 6):
    for alpha in [1.0, 0.5, 0.0, -0.5, -1.0, 2.0, -2.0]:
        c = (1 + alpha) / 2
        res_true, res_med, est = [], [], []
        trials = 0
        while len(est) < 6 and trials < 60:
            trials += 1
            x0 = rng.dirichlet(np.ones(K) * 2)
            v0 = rng.normal(size=K)
            v0 -= v0.mean()
            v0 *= 0.25 / np.linalg.norm(v0)
            sol = solve_ivp(rhs, (0, 1.0), np.concatenate([x0, v0]), args=(c, K), rtol=1e-11, atol=1e-13,
                            dense_output=True, events=hit_boundary)
            if sol.status != 0:
                continue
            X = sol.sol(np.linspace(0, 1, 40))[:K].T
            X = X / X.sum(1, keepdims=True)
            lam_true = (1 - alpha) / 2
            res_true.append(ratio(X, lam_true))
            r = np.array([ratio(X, l) for l in lams])
            est.append(1 - 2 * lams[int(np.argmin(r))])
            res_med.append(np.median(r))
        print(f"K={K} alpha={alpha:+.1f}: n={len(est)}  r(lam_true) max={max(res_true):.1e}  "
              f"median r over grid={np.median(res_med):.1e}  alpha_hat={np.round(est, 2).tolist()}", flush=True)

# Parametrization invariance: same geodesic, strongly non-uniform time sampling
K, alpha = 4, 0.5
c = (1 + alpha) / 2
x0 = rng.dirichlet(np.ones(K) * 2)
v0 = rng.normal(size=K)
v0 -= v0.mean()
v0 *= 0.25 / np.linalg.norm(v0)
sol = solve_ivp(rhs, (0, 1.0), np.concatenate([x0, v0]), args=(c, K), rtol=1e-11, atol=1e-13, dense_output=True)
ts = np.sort(rng.uniform(0, 1, 40)) ** 3
X = sol.sol(ts)[:K].T
X /= X.sum(1, keepdims=True)
r = np.array([ratio(X, l) for l in lams])
print(f"param-invariance (K=4, alpha=0.5, t^3 sampling): alpha_hat={1 - 2 * lams[int(np.argmin(r))]:.3f}", flush=True)

# Closed-form checks: selection (e-line) and migration (m-line), and a mixed field
K = 3
x0 = rng.dirichlet(np.ones(K))
f = rng.normal(size=K)
m = rng.dirichlet(np.ones(K))
t = np.linspace(0, 2, 30)
Xs = x0 * np.exp(np.outer(t, f))
Xs /= Xs.sum(1, keepdims=True)
Xm = m + np.exp(-0.7 * t)[:, None] * (x0 - m)
for name, X in (("selection", Xs), ("migration", Xm)):
    r = np.array([ratio(X, l) for l in lams])
    print(f"{name}: alpha_hat={1 - 2 * lams[int(np.argmin(r))]:.3f}  r_min={r.min():.1e}", flush=True)


def mixed_rhs(t, x, f, m, mu, w):
    return w * x * (f - x @ f) + (1 - w) * mu * (m - x)


print("mixed (selection weight w): alpha_hat, r_min", flush=True)
for w in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    sol = solve_ivp(mixed_rhs, (0, 2.0), x0, args=(f, m, 0.7, w), rtol=1e-11, atol=1e-13, dense_output=True)
    X = sol.sol(np.linspace(0, 2, 40)).T
    X /= X.sum(1, keepdims=True)
    r = np.array([ratio(X, l) for l in lams])
    print(f"  w={w:.2f}: alpha_hat={1 - 2 * lams[int(np.argmin(r))]:+.3f}  r_min={r.min():.1e}", flush=True)
