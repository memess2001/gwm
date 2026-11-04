"""Ground-truth-only test of two candidate fixes for the seam ratio (no model latents):
(A) kappa-adaptive omega matching: omega_tol = 0.08 / max(1, kappa_est);
(B) drop straddle windows (frames on both sides of the cut) and widen the window to w = 0.35 rad > omega_max*dt.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/gwm")
import zoo
from analysis import controls as C, seam as SM

def wrap(x): return np.mod(x + np.pi, 2 * np.pi) - np.pi
w = zoo.make("pendulum")
for r in (0, 1):
    tr = w.sample(seed=zoo.split_seed("analysis", 992_000 + r), horizon=64, n_traj=500, obs_mode="state", omega_max=2.5)
    th_all, om_all = tr.state[..., 0], tr.state[..., 1]
    th, om, thp = th_all[:, 1:].reshape(-1), om_all[:, 1:].reshape(-1), th_all[:, :-1].reshape(-1)
    groups = np.repeat(np.arange(500), 64)
    strad = np.abs(th - thp) > np.pi
    # (A) arcs x kappa
    for a in (1.0, 0.9, 0.85):
        x, y = C._arc_xy(th, a)
        for k in (1.0, 5.0, 9.5):
            Z = np.stack([x, y, k * om], 1)
            base = SM.seam_ratio(Z, th, om, groups=groups, n_boot=0, seed=r)
            kap = max(1.0, base["omega_aspect"])
            adj = SM.seam_ratio(Z, th, om, groups=groups, n_boot=0, seed=r, omega_tol=0.08 / kap)
            print(json.dumps(dict(fix="A", r=r, alpha=a, kappa=k, SR=round(base["ratio"], 2), SR_adapt=round(adj["ratio"], 2),
                                  pairs=base["n_cut_pairs"], pairs_adapt=adj["n_cut_pairs"], status=adj["status"])))
    # (B) straddle exclusion with a wider window, on glued / torn / displaced-tear charts
    lift = thp + wrap(th - thp)
    charts = {"glued_ref": np.stack([np.cos(th), np.sin(th), om], 1), "torn": np.stack([th, om], 1),
              "displaced_tear": np.stack([np.where(strad, lift, th), om], 1)}
    keep = ~strad
    for name, Z in charts.items():
        out = {}
        for wdt in (0.12, 0.35):
            full = SM.seam_ratio(Z, th, om, groups=groups, n_boot=0, seed=r, half_width=wdt)
            ns = SM.seam_ratio(Z[keep], th[keep], om[keep], groups=groups[keep], n_boot=0, seed=r, half_width=wdt)
            out[f"w{wdt}"] = dict(SR_all_windows=round(full["ratio"], 2), SR_no_straddle=round(ns["ratio"], 2), status_ns=ns["status"])
        print(json.dumps(dict(fix="B", r=r, chart=name, **out)))
