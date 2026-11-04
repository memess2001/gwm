# Store full Fisher (Sampson) and unweighted-SVD profiles over lam for each segment, several segment lengths T.
import sys
import time
import numpy as np
from estimators import *

Ts = [float(t) for t in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.5, 1.0, 2.0, 4.0]
NDRAW = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
NPTS = 40
WS = np.linspace(0, 1, 11)
for T in Ts:
    rng = np.random.default_rng(SEED)
    JF = np.full((NDRAW, len(WS), len(LAMS1)), np.nan)
    JFe = np.full_like(JF, np.nan)
    RS = np.full((NDRAW, len(WS), len(LAMS0)), np.nan)
    RHO = np.full((NDRAW, len(WS)), np.nan)
    LFR = np.full((NDRAW, len(WS)), np.nan)
    t0 = time.time()
    for d in range(NDRAW):
        x0 = rng.dirichlet(np.ones(3) * 2); f = rng.normal(size=3); m = rng.dirichlet(np.ones(3) * 2)
        mu = rng.uniform(0.2, 1.0)
        for j, w in enumerate(WS):
            X = simulate(x0, f, m, mu, w, T=T, n=NPTS)
            RHO[d, j] = rho_share(X, f, m, mu, w)
            # Fisher-Rao arc length (Shahshahani metric = 4 x Hellinger^2 metric): 2 * sum |d sqrt x|
            LFR[d, j] = 2 * np.sum(np.linalg.norm(np.diff(np.sqrt(X), axis=0), axis=1))
            JF[d, j] = sampson_profile(X)[0]
            JFe[d, j] = sampson_profile(X[:NPTS // 4])[0]
            RS[d, j] = est_svd(X)[1]
    np.savez(f"prof_T{T}_s{SEED}.npz", JF=JF, JFe=JFe, RS=RS, RHO=RHO, LFR=LFR, WS=WS, T=T)
    print(f"T={T} done in {time.time() - t0:.0f}s", flush=True)
