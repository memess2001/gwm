# e/m residual-contrast readings (bounded in [-1, 1]); same draw protocol / RNG as alpha_mixed.py
#   C_fisher = (sqrt Jm - sqrt Je) / (sqrt Jm + sqrt Je),  J = Fisher-weighted (Sampson) residual of the best m-line
#              (lam = 1) and best e-line (lam = 0) - the large-n likelihood-ratio of "m-straight" vs "e-straight"
#   C_svd    = same with the original unweighted ratios r(lam=1), r(lam=0)
import sys
import numpy as np
from scipy.stats import spearmanr
from estimators import *

T = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
NDRAW = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
NPTS = 40
WS = np.linspace(0, 1, 11)
LE = np.array([0.0, 1.0])


def cfisher(X):
    J, _ = sampson_profile(X, LE)
    se, sm = np.sqrt(J[0]), np.sqrt(J[1])
    return (sm - se) / (sm + se + 1e-300)


def csvd(X):
    re, rm = ratio(X, 0.0), ratio(X, 1.0)
    return (rm - re) / (rm + re + 1e-300)


NAMES = ["Cfisher_global", "Cfisher_early25", "Cfisher_win8_median", "Csvd_global"]
rng = np.random.default_rng(SEED)
R = np.full((NDRAW, len(WS), len(NAMES)), np.nan)
RHO = np.full((NDRAW, len(WS)), np.nan)
for d in range(NDRAW):
    x0 = rng.dirichlet(np.ones(3) * 2); f = rng.normal(size=3); m = rng.dirichlet(np.ones(3) * 2)
    mu = rng.uniform(0.2, 1.0)
    for j, w in enumerate(WS):
        X = simulate(x0, f, m, mu, w, T=T, n=NPTS)
        RHO[d, j] = rho_share(X, f, m, mu, w)
        wins = [cfisher(X[s:s + 8]) for s in range(0, NPTS - 7, 4)]
        R[d, j] = [cfisher(X), cfisher(X[:NPTS // 4]), np.median(wins), csvd(X)]
np.savez(f"contrast_T{T}_s{SEED}.npz", R=R, RHO=RHO, WS=WS, NAMES=np.array(NAMES), T=T)
mixed = (WS > 0) & (WS < 1)
print(f"T={T} draws={NDRAW} seed={SEED}")
print(f"{'reading':22s} {'endpts':>6s} {'mono':>5s} {'medSpW':>6s} {'Sp>=.8':>6s} {'Sp(rho)':>7s} {'Sp(w)':>6s}  "
      f"median C at w=0.1..0.9")
for k, nm in enumerate(NAMES):
    A = R[:, :, k]
    ok = np.mean((np.abs(A[:, 0] + 1) < 0.03) & (np.abs(A[:, -1] - 1) < 0.03))
    mono = np.mean(np.all(np.diff(A, axis=1) >= -1e-9, axis=1))
    per = np.array([spearmanr(A[i], WS).correlation for i in range(NDRAW)])
    am, rm = A[:, mixed].ravel(), RHO[:, mixed].ravel()
    wm = np.broadcast_to(WS[mixed], A[:, mixed].shape).ravel()
    print(f"{nm:22s} {ok:6.2f} {mono:5.2f} {np.median(per):6.2f} {np.mean(per >= 0.8):6.2f} "
          f"{spearmanr(am, rm).correlation:7.2f} {spearmanr(am, wm).correlation:6.2f}  "
          f"{np.round(np.median(A[:, mixed], 0), 2).tolist()}")
