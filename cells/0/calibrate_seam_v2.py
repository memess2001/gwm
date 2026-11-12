#!/usr/bin/env python
"""Cell 0, prereg v2 item 3: recalibrate the seam ratio v2 (analysis.seam.seam_ratio_v2) on ground truth.

Ground-truth charts ONLY (no model latents). States: zoo pendulum in the prereg world (omega_max = 2.5, box init),
500 trajectories, history-2 windows t = 1..64 (theta_{t-1}, theta_t), fresh seeds
split_seed("analysis", 994_000 + r), r = 0..R-1 (outside every earlier calibration / review range).

Chart families (Z built from theta_t, the window's older frame theta_{t-1} and omega; kappa multiplies omega):
  glued      (cos theta, sin theta, kappa omega)                                   -> truth: glued
  arc(a)     isometric arc of angle 2 pi a (gap 2 sin(pi a) / a at the cut), kappa omega   (a < 1: torn by g)
  tear       (theta, kappa omega)                                                   -> truth: torn
  displaced(l) straddle windows moved a fraction l of the way to the older frame's branch     -> truth: torn
  fold       (|theta|, omega)  first-order fold                                     -> truth: folded
Rules declared BEFORE the first repeat was drawn (2026-11-12 JST):
  R1 glued (every kappa in KAPPAS): SR_v2 <= 1.5 in >= 95 % of (chart, repeat);
  R2 tear and displaced(0.5), displaced(1.0) (every kappa): SR_v2 > 5 in >= 95 %;
  R3 the PH x SR consistency (analysis.seam.seam_ph_consistency) never returns the wrong side on the core charts:
     glued charts never 'torn', torn / displaced charts never 'glued' (0 errors allowed); the inconclusive rate is
     reported, not judged;
  R4 fold: fold flag raised (fold_index < 0.7) in >= 95 %, and consistency never 'glued';
  arcs: reported as a lookup table (median SR_v2 per (gap, kappa)), the gap where SR_v2 crosses 1.5 and 5 per kappa
  ("equivalent gap", prereg v2 item 14); no pass / fail.
Output: cells/0/calib_seam_v2.json (raw per repeat + summary).

    python cells/0/calibrate_seam_v2.py --repeats 20 --ph-repeats 10
"""
from __future__ import annotations

import argparse
import zlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import zoo                                        # noqa: E402
from analysis import controls as C                # noqa: E402
from analysis import seam as SM                   # noqa: E402
from analysis import topology as T                # noqa: E402

KAPPAS = (1.0, 2.0, 5.0, 9.5)
ARCS = (0.97, 0.95, 0.9, 0.85, 0.8)
DISPLACED = (0.5, 1.0)
TAU, N_FPS, R_STAB = 2.8, 600, 4
OUT = ROOT / "cells" / "0" / "calib_seam_v2.json"


def wrap(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def h1_state(Z, theta, seed):
    """Prereg section 3.3 H1 rule -> 'ring' (H1 = 1 in the main and all R = 4 stability subsamples, and the top class
    wraps, |rho| >= 0.5), 'ring_not_wrapping', 'none' (0 bars in the main sample and all subsamples), 'unstable'."""
    rng = np.random.default_rng(seed)
    idx = T.fps_subsample(Z, N_FPS, int(rng.integers(2 ** 31)))
    dg = T.h1_diagram(Z[idx], pi_need=6.0, coeff=47, do_cocycles=True)
    b_main = int((T.top_pi(dg, 4) >= TAU).sum())
    b_stab = []
    for _ in range(R_STAB):
        half = T.random_subsample(Z, len(Z) // 2, int(rng.integers(2 ** 31)))
        sub = half[T.fps_subsample(Z[half], N_FPS, int(rng.integers(2 ** 31)))]
        b_stab.append(int((T.top_pi(T.h1_diagram(Z[sub], pi_need=6.0), 4) >= TAU).sum()))
    if b_main == 1 and all(b == 1 for b in b_stab):
        rho = T.winding_check(dg, theta[idx])["abs_rho"] if len(dg["pi"]) else np.nan
        return "ring" if (np.isfinite(rho) and rho >= 0.5) else "ring_not_wrapping"
    if b_main == 0 and all(b == 0 for b in b_stab):
        return "none"
    return "unstable"


def charts(th, thp, om):
    strad = SM.straddle_mask(th, thp)
    lift = thp + wrap(th - thp)
    out = {}
    for k in KAPPAS:
        out[f"glued_k{k}"] = ("glued", np.stack([np.cos(th), np.sin(th), k * om], 1))
        out[f"tear_k{k}"] = ("torn", np.stack([th, k * om], 1))
        for lam in DISPLACED:
            tt = np.where(strad, th + lam * (lift - th), th)
            out[f"displaced{lam}_k{k}"] = ("torn", np.stack([tt, k * om], 1))
        for a in ARCS:
            x, y = C._arc_xy(th, a)
            out[f"arc{a}_k{k}"] = ("arc", np.stack([x, y, k * om], 1))
    out["fold"] = ("folded", np.stack([np.abs(th), om], 1))
    return out


def one_repeat(r: int, do_ph: bool) -> dict:
    w = zoo.make("pendulum")
    tr = w.sample(seed=zoo.split_seed("analysis", 994_000 + r), horizon=64, n_traj=500, obs_mode="state",
                  omega_max=2.5)
    th_all, om_all = tr.state[..., 0], tr.state[..., 1]
    th, om, thp = th_all[:, 1:].reshape(-1), om_all[:, 1:].reshape(-1), th_all[:, :-1].reshape(-1)
    groups = np.repeat(np.arange(500), 64)
    rec = {"r": r, "charts": {}}
    for name, (truth, Z) in charts(th, thp, om).items():
        t0 = time.perf_counter()
        res = SM.seam_ratio_v2(Z, th, om, thp, groups=groups, n_boot=0, seed=r)
        v1 = SM.seam_ratio(Z, th, om, groups=groups, n_boot=0, seed=r, diagnostics=False)
        fl = SM.seam_flags_v2(res)
        row = dict(truth=truth, sr_v2=res["ratio"], sr_v1=v1["ratio"], verdict3=fl["verdict3"],
                   kappa=res["kappa_first_pass"], fold_index=fl["fold_index"], folded=fl["folded_at_cut"],
                   status=res["status"], n_cut_pairs=res["n_cut_pairs"])
        core = name.startswith(("glued", "tear", "displaced")) or name == "fold"
        if do_ph and core:
            ph = h1_state(Z, th, seed=1000 * r + zlib.crc32(name.encode()) % 997)
            row["ph"] = ph
            row["consistency"] = SM.seam_ph_consistency(ph, fl["verdict3"], res["kappa_first_pass"], fl["folded_at_cut"])
        row["sec"] = round(time.perf_counter() - t0, 2)
        rec["charts"][name] = row
    return rec


def summarize(recs: list) -> dict:
    def rows(pred):
        return [c for rec in recs for n, c in rec["charts"].items() if pred(n)]
    glued = rows(lambda n: n.startswith("glued"))
    torn = rows(lambda n: n.startswith(("tear", "displaced")))
    fold = rows(lambda n: n == "fold")
    s = {"R1_glued_le_1.5": float(np.mean([c["sr_v2"] <= 1.5 for c in glued])),
         "R2_torn_gt_5": float(np.mean([c["sr_v2"] > 5 for c in torn])),
         "R4_fold_flag": float(np.mean([c["folded"] for c in fold]))}
    ph_g = [c for c in glued if "consistency" in c]
    ph_t = [c for c in torn if "consistency" in c]
    ph_f = [c for c in fold if "consistency" in c]
    s["R3_errors"] = int(sum(c["consistency"] == "torn" for c in ph_g) + sum(c["consistency"] == "glued" for c in ph_t))
    s["R4_fold_consistency_glued"] = int(sum(c["consistency"] == "glued" for c in ph_f))
    s["consistency_rates"] = {
        grp: {v: float(np.mean([c["consistency"] == v for c in cs])) for v in ("glued", "torn", "inconclusive")}
        for grp, cs in (("glued", ph_g), ("torn", ph_t), ("fold", ph_f)) if cs}
    s["pass"] = {"R1": s["R1_glued_le_1.5"] >= 0.95, "R2": s["R2_torn_gt_5"] >= 0.95, "R3": s["R3_errors"] == 0,
                 "R4": s["R4_fold_flag"] >= 0.95 and s["R4_fold_consistency_glued"] == 0}
    names = sorted(recs[0]["charts"])
    s["median_sr_v2"] = {n: float(np.nanmedian([rec["charts"][n]["sr_v2"] for rec in recs])) for n in names}
    s["median_sr_v1"] = {n: float(np.nanmedian([rec["charts"][n]["sr_v1"] for rec in recs])) for n in names}
    lookup = {}
    for k in KAPPAS:
        gaps = [C.arc_gap(a) for a in ARCS]
        med = [s["median_sr_v2"][f"arc{a}_k{k}"] for a in ARCS]
        cross = {}
        for thr in (1.5, 5.0):
            g = None
            for (g0, m0), (g1, m1) in zip(zip(gaps, med), zip(gaps[1:], med[1:])):
                if (m0 - thr) * (m1 - thr) <= 0 and m1 != m0:
                    g = g0 + (thr - m0) * (g1 - g0) / (m1 - m0)
                    break
            cross[str(thr)] = g
        lookup[str(k)] = {"gaps": gaps, "median_sr_v2": med, "gap_at_sr": cross}
    s["equivalent_gap_lookup"] = lookup
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--ph-repeats", type=int, default=10)
    a = ap.parse_args(argv)
    recs = []
    for r in range(a.repeats):
        t = time.perf_counter()
        recs.append(one_repeat(r, do_ph=r < a.ph_repeats))
        print(f"[seam v2] repeat {r} done in {time.perf_counter() - t:.0f}s", flush=True)
        OUT.write_text(json.dumps({"repeats": recs, "summary": summarize(recs)}, indent=1, default=float))
    print(json.dumps(summarize(recs)["pass"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
