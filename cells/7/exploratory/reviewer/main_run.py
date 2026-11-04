# Compare alpha-readings on mixed selection+migration segments (same draw protocol / RNG as cells/7/exploratory/alpha_mixed.py)
import sys
import time
import numpy as np
from estimators import *

T = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
NDRAW = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
OUT = sys.argv[4] if len(sys.argv) > 4 else f"run_T{T}_s{SEED}.npz"
NPTS = 40
WS = np.linspace(0, 1, 11)
NAMES = ["svd_global", "fisher_global", "fisher_early25", "fisher_win8_median", "osc_t0", "osc_median",
         "curvreg_global", "curvreg_early25", "anchor_endpoints"]

rng = np.random.default_rng(SEED)
R = np.full((NDRAW, len(WS), len(NAMES)), np.nan)
RHO = np.full((NDRAW, len(WS)), np.nan)
PAR = []
t0 = time.time()
for d in range(NDRAW):
    x0 = rng.dirichlet(np.ones(3) * 2)
    f = rng.normal(size=3)
    m = rng.dirichlet(np.ones(3) * 2)
    mu = rng.uniform(0.2, 1.0)
    PAR.append(np.r_[x0, f, m, mu])
    for j, w in enumerate(WS):
        X = simulate(x0, f, m, mu, w, T=T, n=NPTS)
        RHO[d, j] = rho_share(X, f, m, mu, w)
        e = NPTS // 4
        wins = [est_fisher(X[s:s + 8])[0] for s in range(0, NPTS - 7, 4)]
        R[d, j] = [est_svd(X)[0], est_fisher(X)[0], est_fisher(X[:e])[0], np.median(wins),
                   est_osc(X, f, m, mu, w, 0), est_osc_median(X, f, m, mu, w),
                   est_curvreg(X, f, m, mu, w), est_curvreg(X, f, m, mu, w, slice(0, e)), est_anchor(X)[0]]
    if d % 10 == 9:
        print(f"draw {d + 1}/{NDRAW}  {time.time() - t0:.0f}s", flush=True)
np.savez(OUT, R=R, RHO=RHO, WS=WS, NAMES=np.array(NAMES), PAR=np.array(PAR), T=T)
print("saved", OUT)
