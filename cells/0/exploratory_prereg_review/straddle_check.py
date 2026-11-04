"""Reviewer check of the cell-0 prereg (ground-truth charts ONLY; no model latents anywhere).

angle_raw with history 2: the encoder sees the window (theta_{t-1}, theta_t). Windows whose two frames lie on
opposite sides of the cut ("straddle" windows) form a separate input cluster. The calibrated seam ratio and H1
rule were validated on charts of the CURRENT state only. Here:
  (1) fraction of cut-window points (|wrap(theta_t - pi)| < 0.12) that are straddle windows, by |omega|;
  (2) a "displaced tear" chart: every window is placed on the branch of its OLDER frame,
      theta_tilde = theta_{t-1} + wrap(theta_t - theta_{t-1}) (continuous lift from theta_{t-1}); it is torn
      (a strip), but the tear sits between straddle and post-straddle windows instead of at theta_t = +-pi;
  (3) charts that put straddle windows a fraction lam of the way from the arrival branch (lam = 0, the ordinary
      torn chart of theta_t) to the departure branch (lam = 1, the displaced tear).
Readouts: seam ratio (all omega = prereg primary; |omega| >= 1.6 = secondary) with flags, and the prereg H1 rule.
States: zoo pendulum, prereg world, 500 trajectories, windows t = 1..64, fresh seeds split_seed("analysis", 991_000 + r).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/user/gwm")
import zoo  # noqa: E402
from analysis import seam as SM  # noqa: E402
from analysis import topology as T  # noqa: E402

sys.path.insert(0, "/tmp/claude-0/-home-user/8bc92419-2f30-547d-8b18-8324b70c6f75/scratchpad/rev0")
from aniso_arc_check import h1_rule  # noqa: E402


def wrap(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def main():
    reps = [int(r) for r in sys.argv[1].split(",")]
    w = zoo.make("pendulum")
    for r in reps:
        tr = w.sample(seed=zoo.split_seed("analysis", 991_000 + r), horizon=64, n_traj=500, obs_mode="state",
                      omega_max=2.5)
        th_all, om_all = tr.state[..., 0], tr.state[..., 1]
        th, om = th_all[:, 1:].reshape(-1), om_all[:, 1:].reshape(-1)       # current frame t = 1..64
        thp = th_all[:, :-1].reshape(-1)                                       # older frame t - 1
        groups = np.repeat(np.arange(500), 64)
        straddle = np.abs(th - thp) > np.pi
        cutwin = np.abs(wrap(th - np.pi)) < 0.12
        prof = {}
        for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 1.6), (1.6, 9)):
            m = cutwin & (np.abs(om) >= lo) & (np.abs(om) < hi)
            prof[f"|om| {lo}-{hi}"] = dict(n=int(m.sum()), straddle_frac=round(float(straddle[m].mean()), 3) if m.any() else None)
        print(json.dumps(dict(r=r, straddle_frac_all=round(float(straddle.mean()), 4),
                              straddle_frac_in_cut_window=round(float(straddle[cutwin].mean()), 3), by_omega=prof)))
        lift = thp + wrap(th - thp)                                            # branch of the older frame
        for lam in (0.0, 0.5, 1.0):
            tt = np.where(straddle, th + lam * (lift - th), th)
            Z = np.stack([tt, om], 1)
            sr = SM.seam_ratio(Z, th, om, groups=groups, n_boot=100, seed=r)
            srf = SM.seam_ratio(Z, th, om, groups=groups, n_boot=100, seed=r, omega_abs_min=1.6)
            fl = SM.seam_flags(sr)
            h = h1_rule(Z, th, seed=1000 * r + 11)
            print(json.dumps(dict(r=r, lam=lam, SR_all=round(float(sr["ratio"]), 2),
                                  SR_all_ci=[round(float(x), 2) for x in sr["ci95"]],
                                  SR_fast=round(float(srf["ratio"]), 2), verdict_all=fl["verdict"],
                                  kappa_est=round(fl["omega_aspect"], 2), fold=round(fl["fold_index"], 2),
                                  in_range=fl["in_range"], **h)), flush=True)


if __name__ == "__main__":
    main()
