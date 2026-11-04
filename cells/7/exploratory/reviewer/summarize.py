import sys
import numpy as np
from scipy.stats import spearmanr

for fn in sys.argv[1:]:
    Z = np.load(fn)
    R, RHO, WS, NAMES = Z["R"], Z["RHO"], Z["WS"], list(Z["NAMES"])
    nd = R.shape[0]
    mixed = (WS > 0) & (WS < 1)
    rho_mono = np.mean(np.all(np.diff(RHO, axis=1) >= -1e-12, axis=1))
    print(f"\n=== {fn}: T={float(Z['T'])}, draws={nd}; rho monotone in w in {rho_mono:.0%} of draws ===")
    print(f"{'estimator':20s} {'endpts':>6s} {'mono':>5s} {'dip<-1':>6s} {'ovr>1':>6s} {'medSpW':>6s} "
          f"{'Sp>=.8':>6s} {'Sp(rho)':>7s} {'Sp(w)':>6s} {'in1.2':>6s}  quantiles 5/25/50/75/95 (mixed)")
    for k, nm in enumerate(NAMES):
        A = R[:, :, k]
        ok_end = np.mean((np.abs(A[:, 0] + 1) < 0.03) & (np.abs(A[:, -1] - 1) < 0.03))
        mono = np.mean(np.all(np.diff(A, axis=1) >= -1e-9, axis=1))
        dip = np.mean(A[:, 1] < -1.02)        # w = 0.1 below the migration value
        ovr = np.mean(A[:, -2] > 1.02)        # w = 0.9 above the selection value
        per = np.array([spearmanr(A[d], WS).correlation for d in range(nd)])
        am, rm = A[:, mixed].ravel(), RHO[:, mixed].ravel()
        wm = np.broadcast_to(WS[mixed], A[:, mixed].shape).ravel()
        g = np.isfinite(am)
        sp_rho = spearmanr(am[g], rm[g]).correlation
        sp_w = spearmanr(am[g], wm[g]).correlation
        inr = np.mean(np.abs(am[g]) <= 1.2)
        qs = np.quantile(am[g], [.05, .25, .5, .75, .95])
        print(f"{nm:20s} {ok_end:6.2f} {mono:5.2f} {dip:6.2f} {ovr:6.2f} {np.nanmedian(per):6.2f} "
              f"{np.nanmean(per >= 0.8):6.2f} {sp_rho:7.2f} {sp_w:6.2f} {inr:6.2f}  {np.round(qs, 2).tolist()}")
