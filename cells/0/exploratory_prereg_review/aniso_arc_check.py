"""Reviewer check of the cell-0 prereg (ground-truth embeddings ONLY; no model latents anywhere).

Question: how do omega-anisotropy kappa (below the seam flag kappa > 10) and a partial gap at the cut move the two
angle_raw primary readouts (H1 rule of prereg section 3.3 and the seam ratio of section 3.8)?
Cloud: emb_arc(alpha) (theta isometric on an arc of angle 2 pi alpha, gap g = 2 sin(pi alpha)/alpha at the cut)
with omega multiplied by kappa. States: zoo pendulum, prereg world, 500 trajectories x t = 1..64, fresh seeds
split_seed("analysis", 990_000 + r) (outside every calibration range, the tests and every model split).
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/user/gwm")
import zoo  # noqa: E402
from analysis import controls as C  # noqa: E402
from analysis import seam as SM  # noqa: E402
from analysis import topology as T  # noqa: E402

TAU, N_FPS, R_STAB = 2.8, 600, 4


def h1_rule(Z, theta, seed):
    """Prereg 3.3: #bars with pi >= 2.8 on FPS n = 600 equals b1 = 1, and the same in all R = 4 FPS-on-random-half
    subsamples; plus the winding condition |rho| >= 0.5 on the top class (Z_47 cocycle)."""
    rng = np.random.default_rng(seed)
    idx = T.fps_subsample(Z, N_FPS, int(rng.integers(2 ** 31)))
    dg = T.h1_diagram(Z[idx], pi_need=6.0, coeff=47, do_cocycles=True)
    top = T.top_pi(dg, 4)
    b_main = int((top >= TAU).sum())
    b_stab = []
    for _ in range(R_STAB):
        half = T.random_subsample(Z, len(Z) // 2, int(rng.integers(2 ** 31)))
        sub = half[T.fps_subsample(Z[half], N_FPS, int(rng.integers(2 ** 31)))]
        b_stab.append(int((T.top_pi(T.h1_diagram(Z[sub], pi_need=6.0), 4) >= TAU).sum()))
    rho = np.nan
    if len(dg["pi"]):
        rho = T.winding_check(dg, theta[idx])["abs_rho"]
    h1_eq1 = b_main == 1 and all(b == 1 for b in b_stab)
    return dict(top_pi=[round(float(x), 2) for x in top[:2]], b_main=b_main, b_stab=b_stab,
                abs_rho=None if not np.isfinite(rho) else round(float(rho), 3),
                H1_is_1_wrapping=bool(h1_eq1 and np.isfinite(rho) and rho >= 0.5))


def main():
    alphas = [float(a) for a in sys.argv[1].split(",")]
    kappas = [float(k) for k in sys.argv[2].split(",")]
    reps = [int(r) for r in sys.argv[3].split(",")]
    for r in reps:
        seed = zoo.split_seed("analysis", 990_000 + r)
        S, info = C.pendulum_states(seed)
        th, om = S[:, 0], S[:, 1]
        groups = np.repeat(np.arange(500), 64)
        for a in alphas:
            x, y = C._arc_xy(th, a)
            for k in kappas:
                t0 = time.perf_counter()
                Z = np.stack([x, y, k * om], 1)
                sr = SM.seam_ratio(Z, th, om, groups=groups, n_boot=100, seed=r)
                srf = SM.seam_ratio(Z, th, om, groups=groups, n_boot=100, seed=r, omega_abs_min=1.6)
                fl = SM.seam_flags(sr)
                h = h1_rule(Z, th, seed=1000 * r + 7)
                print(json.dumps(dict(r=r, alpha=a, gap=round(C.arc_gap(a), 3), kappa=k,
                                      SR_all=round(float(sr["ratio"]), 2), SR_fast=round(float(srf["ratio"]), 2),
                                      kappa_est=round(fl["omega_aspect"], 2), fold=round(fl["fold_index"], 2),
                                      seam_in_range=fl["in_range"], **h,
                                      sec=round(time.perf_counter() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
