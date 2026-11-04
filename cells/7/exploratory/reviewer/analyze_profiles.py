import sys
import numpy as np
from scipy.stats import spearmanr
from estimators import LAMS0, LAMS1

AL1 = 1 - 2 * LAMS1
AL0 = 1 - 2 * LAMS0
inside = (AL1 >= -1 - 1e-9) & (AL1 <= 1 + 1e-9)
i_e, i_m = int(np.argmin(np.abs(LAMS1 - 0.0))), int(np.argmin(np.abs(LAMS1 - 1.0)))


def readings(Z):
    JF, JFe, RS = Z["JF"], Z["JFe"], Z["RS"]
    out = {}
    out["svd_argmin"] = AL0[np.argmin(RS, -1)]
    out["fisher_argmin"] = AL1[np.argmin(JF, -1)]
    out["fisher_argmin_in[-1,1]"] = AL1[inside][np.argmin(JF[..., inside], -1)]
    out["fisher_early_argmin"] = AL1[np.argmin(JFe, -1)]
    out["fisher_early_in[-1,1]"] = AL1[inside][np.argmin(JFe[..., inside], -1)]
    for n in (1e3, 1e4):
        L = -0.5 * n * (JF[..., inside] - JF[..., inside].min(-1, keepdims=True))
        P = np.exp(L); P /= P.sum(-1, keepdims=True)
        out[f"post_mean_n{int(n)}_in[-1,1]"] = (P * AL1[inside]).sum(-1)
        L = -0.5 * n * (JF - JF.min(-1, keepdims=True))
        P = np.exp(L); P /= P.sum(-1, keepdims=True)
        out[f"post_mean_n{int(n)}_full"] = (P * AL1).sum(-1)
    for nm, J in (("contrast_global", JF), ("contrast_early", JFe)):
        se, sm = np.sqrt(J[..., i_e]), np.sqrt(J[..., i_m])
        out[nm] = (sm - se) / (sm + se + 1e-300)
    return out


for fn in sys.argv[1:]:
    Z = np.load(fn)
    WS, RHO, LFR = Z["WS"], Z["RHO"], Z["LFR"]
    nd = RHO.shape[0]
    mixed = (WS > 0) & (WS < 1)
    lmed = np.median(LFR[:, mixed])
    print(f"\n=== {fn}  T={float(Z['T'])}  draws={nd}  median Fisher-Rao length of mixed segments {lmed:.2f} ===")
    print(f"{'reading':26s} {'mono':>5s} {'medSpW':>6s} {'Sp>=.8':>6s} {'Sp(rho)':>7s} {'in1.2':>6s} {'wDrop':>6s} "
          f"{'Sp(rho)|l>=.5':>13s}")
    for nm, A in readings(Z).items():
        mono = np.mean(np.all(np.diff(A, axis=1) >= -1e-9, axis=1))
        per = np.array([spearmanr(A[d], WS).correlation for d in range(nd)])
        am, rm, lm = A[:, mixed].ravel(), RHO[:, mixed].ravel(), LFR[:, mixed].ravel()
        sp = spearmanr(am, rm).correlation
        long_ = lm >= 0.5
        spl = spearmanr(am[long_], rm[long_]).correlation if long_.sum() > 10 else np.nan
        drop = np.median(np.min(np.diff(A, axis=1), axis=1))
        print(f"{nm:26s} {mono:5.2f} {np.nanmedian(per):6.2f} {np.nanmean(per >= 0.8):6.2f} {sp:7.2f} "
              f"{np.mean(np.abs(am) <= 1.2):6.2f} {drop:+6.2f} {spl:9.2f} (n={long_.sum()})")
