"""Cell 9 review, check 1-3 (analytic / toy).

1. Fisher information of N(mu, sigma^2) in (mu, sigma): numerical Hessian of KL -> expect diag(1/s^2, 2/s^2).
   Also check the multivariate formula ds^2 = dmu^T S^-1 dmu + 1/2 tr((S^-1 dS)^2) on a diagonal S.
2. Discretisation: for two equal-sigma Gaussians, compare per-step "Fisher distances" used in practice
   (line element |dmu|/s, sqrt(2 KL), Fisher-Rao geodesic distance, Hellinger, Bhattacharyya angle)
   as a function of |dmu|/s.  Only the first two are linear in |dmu|.
3. Toy: constant but anisotropic sigma (active dims at floor 0.1, a nuisance dim at sigma=1 whose mean wanders):
   CV of event spacing under raw Euclidean, whitened Euclidean, Fisher line element, Fisher geodesic sum.
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1
def kl_1d(m1, s1, m2, s2):
    return np.log(s2 / s1) + (s1**2 + (m1 - m2) ** 2) / (2 * s2**2) - 0.5

m0, s0, h = 0.3, 0.7, 1e-4
def f(p):
    return kl_1d(m0, s0, m0 + p[0], s0 + p[1])
H = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        e_i, e_j = np.eye(2)[i] * h, np.eye(2)[j] * h
        H[i, j] = (f(e_i + e_j) - f(e_i - e_j) - f(-e_i + e_j) + f(-e_i - e_j)) / (4 * h * h)
print("1) Hessian of KL at (mu,sigma)=(0.3,0.7):\n", np.round(H, 6))
print("   expected diag(1/s^2, 2/s^2) =", np.round([1 / s0**2, 2 / s0**2], 6))

# Monte-Carlo score check
x = rng.normal(m0, s0, 2_000_000)
score = np.stack([(x - m0) / s0**2, -1 / s0 + (x - m0) ** 2 / s0**3])
print("   MC Fisher:\n", np.round(score @ score.T / x.size, 4))

# multivariate formula with diagonal covariance
s = np.array([0.1, 0.5, 2.0]); ds = np.array([0.01, -0.02, 0.03]); dm = np.array([0.1, 0.2, -0.3])
S, dS = np.diag(s**2), np.diag(2 * s * ds)
full = dm @ np.linalg.inv(S) @ dm + 0.5 * np.trace(np.linalg.inv(S) @ dS @ np.linalg.inv(S) @ dS)
diag_formula = np.sum((dm**2 + 2 * ds**2) / s**2)
print(f"   full-Gaussian formula {full:.6f} vs sum_i (dmu^2+2dsig^2)/sig^2 {diag_formula:.6f}")

# ---------------------------------------------------------------- 2
def fr_geodesic_1d(m1, s1, m2, s2):
    # Fisher-Rao distance on univariate normals (metric (dm^2+2ds^2)/s^2): hyperbolic plane after m -> m/sqrt2
    a = ((m1 - m2) ** 2 / 2 + (s1 - s2) ** 2)
    b = ((m1 - m2) ** 2 / 2 + (s1 + s2) ** 2)
    return 2 * np.sqrt(2) * np.arctanh(np.sqrt(a / b))

print("\n2) per-step distances between N(0,s^2) and N(d,s^2), r = d/s")
print("   r     line=|d|/s  sqrt(2KL)  FR-geodesic  ratio_geo/line  Hellinger*sqrt8  2*BhattAngle*sqrt2")
for r in (0.1, 0.5, 1, 2, 3, 5, 10, 20, 50):
    geo = fr_geodesic_1d(0, 1, r, 1)
    kl = kl_1d(0, 1, r, 1)
    bc = np.exp(-r**2 / 8)                      # Bhattacharyya coefficient for equal variances
    hel = np.sqrt(1 - bc)                       # Hellinger (one convention); ~ r/sqrt8 for small r
    ang = np.arccos(bc)                         # Bhattacharyya angle; ~ r/(2 sqrt2) for small r
    print(f"   {r:5.1f} {r:10.3f} {np.sqrt(2*kl):10.3f} {geo:11.3f} {geo/r:14.3f} {hel*np.sqrt(8):15.3f} {2*np.sqrt(2)*ang:18.3f}")

# ---------------------------------------------------------------- 3
# pendulum-like latent: active dims (cos th, sin th, omega/2) at floor sigma=0.1; nuisance dim sigma=1 with wandering mean
from scipy.integrate import solve_ivp
def pend(t, y):
    return [y[1], -np.sin(y[0])]
T, dt = 60.0, 0.1
tt = np.arange(0, T + dt / 2, dt)
cvs = {k: [] for k in ("wall", "eucl_raw", "eucl_whitened", "fisher_line", "fisher_geo_sum")}
ratios = []
for seed in range(40):
    r_ = np.random.default_rng(seed)
    E = r_.uniform(-0.9, 0.8)                      # librations only
    th0 = np.arccos(-E)                            # start at turning point
    sol = solve_ivp(pend, (0, T), [th0, 0.0], t_eval=tt, rtol=1e-10, atol=1e-10)
    th, om = sol.y
    nuis = np.cumsum(r_.normal(0, 0.3, tt.size)) * 0.2     # nuisance mean wanders (posterior==prior dim)
    mu = np.stack([2 * np.cos(th), 2 * np.sin(th), om, nuis], 1)
    sig = np.array([0.1, 0.1, 0.1, 1.0])
    dmu = np.diff(mu, axis=0)
    speeds = {
        "wall": np.full(len(dmu), dt),
        "eucl_raw": np.linalg.norm(dmu, axis=1),
        "eucl_whitened": np.linalg.norm(dmu / sig, axis=1),
        "fisher_line": np.sqrt(np.sum(dmu**2 / sig**2, 1)),
        "fisher_geo_sum": np.sqrt(np.sum(fr_geodesic_1d(0, sig, dmu, sig) ** 2, 1)),  # product-manifold distance
    }
    ratios.append(np.median(np.abs(dmu[:, :3]) / 0.1))
    # events: bottom passages (sign change of theta)
    idx = np.where(np.sign(th[1:]) != np.sign(th[:-1]))[0]
    for k, v in speeds.items():
        tau = np.concatenate([[0], np.cumsum(v)])
        ev = tau[idx]
        gaps = np.diff(ev)
        cvs[k].append(gaps)
print("\n3) toy deterministic pendulum latent, sigma constant in time but anisotropic (0.1,0.1,0.1,1.0)")
print(f"   median per-step |dmu_i|/sigma on active dims: {np.median(ratios):.2f}")
for k, g in cvs.items():
    g = np.concatenate(g)
    print(f"   {k:15s}: CV of inter-event spacing = {g.std() / g.mean():.4f}")
