"""Check 4: does the data-density baseline already reveal a THIN invisible wall (where the binary support map is blind)?
Random isotropic actions, first-order maze, |d| = 0.25, walkers started uniformly (uniform is stationary for any
symmetric kernel).  Liouville link: contact rules that compress phase space (slide/stop) make density pile up at
walls; measure-preserving rules (reject/reflect) keep it uniform -- exactly where the a.e. C_k is blind (check 2)."""
import numpy as np
from maze import walls, step1

rng = np.random.default_rng(0)
W = walls(thick=False)
N, T, BURN = 20000, 1500, 500
bins = np.linspace(0, 10, 101)
for mode in ("slide", "stop", "reject", "reflect"):
    P = rng.uniform(0, 10, size=(N, 2))
    Hc = np.zeros((100, 100))
    for t in range(T):
        a = rng.uniform(-np.pi, np.pi, N)
        D = 0.25 * np.stack([np.cos(a), np.sin(a)], 1)
        P = step1(P, D, mode, W)
        if t >= BURN and t % 5 == 0:
            Hc += np.histogram2d(P[:, 1], P[:, 0], bins=[bins, bins])[0]
    Hc /= Hc.mean()
    xc = 0.5 * (bins[1:] + bins[:-1])
    face_rows = (xc > 1) & (xc < 5)                         # y in (1,5): along the thin wall, away from ends
    col_l, col_r = 49, 50                                   # bins [4.9,5.0) and [5.0,5.1)
    far_cols = (xc > 2) & (xc < 3)
    print(f"{mode:>7}: density (mean=1) in bins touching the thin wall: left {Hc[face_rows, col_l].mean():.2f}, "
          f"right {Hc[face_rows, col_r].mean():.2f}; next bins {Hc[face_rows, col_l-1].mean():.2f}/{Hc[face_rows, col_r+1].mean():.2f}; "
          f"far {Hc[face_rows][:, far_cols].mean():.2f} (sd {Hc[face_rows][:, far_cols].std():.2f})")
