# f -> -f symmetry test of the first-order response of alpha-readings to a small selection admixture (w -> 0).
# Claim under test: for any smooth reparametrisation-invariant reading, d alpha_hat / dw at w=0 is odd in f,
# because the normal deviation of the trajectory from the m-line is linear in F_sel (odd in f) at first order.
import numpy as np
from estimators import *

rng = np.random.default_rng(7)
NP = 20
WS = [0.005, 0.02, 0.05]
EST = {
    "svd_global": lambda X, *p: est_svd(X)[0],
    "fisher_global": lambda X, *p: est_fisher(X)[0],
    "fisher_early25": lambda X, *p: est_fisher(X[:10])[0],
    "osc_t0": lambda X, *p: est_osc(X, *p, 0),
    "curvreg_global": lambda X, *p: est_curvreg(X, *p),
    "anchor_endpoints": lambda X, *p: est_anchor(X)[0],
}
res = {k: np.zeros((NP, 2, len(WS))) for k in EST}
slope_check = []
for i in range(NP):
    x0 = rng.dirichlet(np.ones(3) * 2); f = rng.normal(size=3); m = rng.dirichlet(np.ones(3) * 2)
    mu = rng.uniform(0.2, 1.0)
    for sgn_i, sg in enumerate((1.0, -1.0)):
        for j, w in enumerate(WS):
            X = simulate(x0, sg * f, m, mu, w)
            for k, fn in EST.items():
                res[k][i, sgn_i, j] = fn(X, sg * f, m, mu, w) + 1.0  # deviation from the migration value -1
    # analytic slope of the osculating alpha at x0, w = 0:  d alpha/dw = 2 g(DFs Fm, n) / g(W(Fm), n)
    x = x0[None]
    vm = Fm(x, m, mu)
    fb = x @ f
    dFs_vm = vm * (f - fb[..., None]) - x * (vm @ f)[..., None]
    n = np.cross(np.ones((1, 3)), vm / x)
    Wv = vm * vm / x - x * np.sum(vm * vm / x, 1)[:, None]
    slope = 2 * np.sum(dFs_vm * n / x) / np.sum(Wv * n / x)
    num = (est_osc(simulate(x0, f, m, mu, 1e-5), f, m, mu, 1e-5, 0) + 1) / 1e-5
    slope_check.append((slope, num))

sc = np.array(slope_check)
print("analytic vs numerical slope of osc alpha at x0 (first 6 pairs):", np.round(sc[:6], 4).tolist())
print(f"max rel err of analytic slope: {np.max(np.abs(sc[:, 0] - sc[:, 1]) / np.abs(sc[:, 1])):.1e}")
print("fraction of draws with NEGATIVE analytic slope (alpha dips below -1):", np.mean(sc[:, 0] < 0))
print()
print("For each estimator and w: fraction of pairs whose deviations (alpha_hat+1) have OPPOSITE sign under f->-f,")
print("median ratio dev(-f)/dev(f), and fraction of the 2*NP segments with alpha_hat < -1 (dip):")
for k in EST:
    A = res[k]
    line = []
    for j, w in enumerate(WS):
        a, b = A[:, 0, j], A[:, 1, j]
        nz = (np.abs(a) > 1e-9) & (np.abs(b) > 1e-9)
        opp = np.mean(np.sign(a[nz]) != np.sign(b[nz])) if nz.any() else np.nan
        ratio = np.median(b[nz] / a[nz]) if nz.any() else np.nan
        dip = np.mean(A[:, :, j] < -1e-9)
        line.append(f"w={w}: opp={opp:.2f} ratio={ratio:+.2f} dip={dip:.2f}")
    print(f"  {k:18s} " + " | ".join(line))
