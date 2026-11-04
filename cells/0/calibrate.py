#!/usr/bin/env python
"""Cell 0 instrument calibration on ground-truth pendulum embeddings (NO trained-model latents).

A. Topology (analysis/topology.py). Controls with known b1 in {0, 1, 2} are built from zoo pendulum states
   (analysis/controls.py). For n in {300, 600, 1000} and farthest-point vs random subsampling (plus PCA
   whitening with FPS) every diagram's top H1 bars (pi = death/birth) and the Bobrowski-Skraba test are
   recorded. Count rules are fitted on repeats r < R/2 (calibration half) of the CORE controls and evaluated on
   repeats r >= R/2 (held-out half), with Clopper-Pearson 95% intervals. Also: subsample stability over
   STAB_R subsamples, the LGumbel null under FPS vs random subsampling, and the winding check (circular
   coordinate of the top class vs. the true angle) separating the cylinder loop from the contractible
   libration ring.
B. Energy (analysis/smoothness.py). Probes and smoothness scores on smooth / non-smooth / graded controls;
   thresholds = 5th percentile (95th for the Lipschitz tail ratio) of the smooth controls on the calibration
   half; held-out TPR / FPR.
C. Confirmation. The candidate instruments in CONFIRM_CANDIDATES (fixed before any confirmation repeat was
   drawn) are re-evaluated with their calibration-half parameters on fresh repeats R..2R-1 of the core controls.
D. Seam ratio (analysis/seam.py) and theta-decodability gate (analysis/gates.py), prereg section 0b, on their own
   seeds SEAM_BASE + r: glued / torn / stressed / partially glued embeddings (SEAM_CONTROLS; threshold fitted on
   r < R/2 of the core controls, held-out r >= R/2, the preregistered 5 evaluated alongside; the H1 rule on the
   partial-gluing arcs for comparison) and theta-injective vs folded / conserved-quantity latents (GATE_CONTROLS).
E. Second-round review (calibration.md "审稿意见（第二轮）"), on its own seeds REVIEW_BASE + r: the frozen seam
   thresholds re-checked on fresh repeats of the core controls; the review diagnostics of analysis/seam.py (omega
   aspect, fold index, trajectories at the cut) on adversarial latents (omega-stretched torn charts, folds that are
   continuous across the cut); the fold-index rule fitted on r < R/2 and held out on r >= R/2; three declared
   theta-gate variants and a partial-fold dose-response. From the stored JSON (no recomputation): the joint energy
   rule of prereg 3.4 (kNN AND quadratic) and repeat-level intervals for the part-D rates.

Usage (from the repo root; nothing depends on the working directory):
    python cells/0/calibrate.py                        # all parts, writes cells/0/calib_results.json
    python cells/0/calibrate.py --parts topo,energy    # fit / held-out only (~25 min wall on 4 shared cores)
    python cells/0/calibrate.py --parts confirm        # confirmation only, re-uses the fits stored in the JSON
    python cells/0/calibrate.py --parts seam --jobs 1  # part D only (~9 min, one core); other sections kept
    python cells/0/calibrate.py --parts review --jobs 1  # part E only (~4 min, one core); other sections kept
    python cells/0/calibrate.py --quick                # smoke test: 2 repeats, a few controls, writes nothing
Options: --jobs N (default min(4, cpu_count)), --repeats R (default 20), --parts, --out PATH,
         --cache PATH (pickle raw per-task records) and --reanalyze (re-run only the analysis from --cache).
Pure numpy / scipy / scikit-learn / ripser (no torch, no GPU), so it runs unchanged on Linux, macOS, Windows.
Seeds: zoo states for repeat r use zoo.split_seed("analysis", 900_000 + r) (an index range no model uses; part D
uses 950_000 + r and part E 960_000 + r instead, disjoint from parts A-C, from each other and from the tests);
every other draw is seeded from (repeat, control name, purpose) via SeedSequence, so results are reproducible
and independent of the worker count / scheduling.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker process; parallelism is across tasks

import argparse
import json
import pickle
import platform
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                     # noqa: E402
from scipy import stats                                # noqa: E402

import zoo                                             # noqa: E402
from analysis import controls as C                     # noqa: E402
from analysis import gates as GT                       # noqa: E402
from analysis import seam as SE                        # noqa: E402
from analysis import smoothness as SM                  # noqa: E402
from analysis import topology as T                     # noqa: E402

OUT_DEFAULT = ROOT / "cells" / "0" / "calib_results.json"
CALIB_BASE = zoo.split_seed("analysis", 900_000)       # = 3_900_000; repeat r uses CALIB_BASE + r

# ----------------------------------------------------------------------------------------------- settings
NS = (300, 600, 1000)
PI_NEED = 6.0                                          # diagrams exact for pi <= 6 (censored bars: pi >= 6)
TAU_GRID = np.round(np.arange(1.5, 6.0 + 1e-9, 0.1), 2)
GAP_GRID = np.round(np.arange(1.1, 3.0 + 1e-9, 0.1), 2)
TAU_S_GRID = (0.6, 0.8, 1.0)
STAB_R = 4
STAB_PLAN = (("random", 300), ("fps_half", 300), ("fps_half", 600))   # (method, n); random/600 dropped for cost
WHITE_NS = (300, 600)
WIND_N, WIND_COEFF = 600, 47
RHO_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
# Confirmation on fresh seeds (repeats R..2R-1). Candidate list fixed on 2026-11-03 after the calibration /
# held-out analysis and BEFORE any confirmation repeat was drawn: the pre-specified pick (FPS n = 1000), the two
# cheaper FPS sizes and the two FPS-half stability rules; parameters are the calibration-half fits.
CONFIRM_CANDIDATES = (("main:fps/1000", "thr"), ("main:fps/600", "thr"), ("main:fps/300", "thr"),
                      ("stab:fps_half/600", "thr"), ("stab:fps_half/300", "thr"))
CONFIRM_NS = (300, 600, 1000)
CONFIRM_STAB = (("fps_half", 300), ("fps_half", 600))
ENERGY_NS, ENERGY_KS, KRR_N = (1000, 2000, 4000), (5, 10, 20), 600
ALPHA_BS = 0.05
# D. Seam ratio + theta-decodability gate (prereg section 0b; added 2026-11-04). Own seed range, disjoint from
# CALIB_BASE + 0..39 (parts A-C), the tests (3_999_xxx) and every model split (split_seed index < 900_000).
SEAM_BASE = zoo.split_seed("analysis", 950_000)        # = 3_950_000; repeat r uses SEAM_BASE + r
SEAM_KW = dict(half_width=0.12, omega_tol=0.08, omega_bin=0.25, n_dtheta_bins=4, min_pairs=10, min_ctrl_angles=3,
               min_strata=3, min_cut_pairs=50, max_per_window=2000)   # = analysis.seam.seam_ratio defaults
SEAM_NBOOT = 200
SEAM_FAST = 1.6                                        # secondary readout: fast rotations only (the pilot's restriction)
SEAM_THRESHOLD_PREREG = 5.0                            # P0.4' (post-pilot prediction)
SEAM_PH_N, SEAM_PH_TAU = 600, 2.8                      # the preregistered H1 rule, for the cross-check on partial gluing
ARC_ALPHAS = (0.995, 0.99, 0.98, 0.95, 0.93, 0.9, 0.85, 0.8, 0.6, 0.3)
GATE_N, GATE_K = 2000, 10
GATE_THRESHOLD_PREREG = 0.99
# E. Second-round review (2026-11-04; calibration.md "审稿意见（第二轮）"). Own seeds REVIEW_BASE + r, disjoint from
# CALIB_BASE + 0..39, SEAM_BASE + 0..19, the tests (3_999_xxx) and the reviewer's scratch probes (3_970_xxx).
# Every rule evaluated in part E is declared here, before the first review repeat was drawn: the frozen seam
# thresholds (prereg 5, part-D fit 5.7544), the fast-only readout, the omega-aspect range, the fold-index rule
# (fitted on the review calibration half, held out on the other half) and three theta-gate variants.
REVIEW_BASE = zoo.split_seed("analysis", 960_000)      # = 3_960_000; repeat r uses REVIEW_BASE + r
REVIEW_SEAM_THRESHOLDS = (5.0, 5.7544)
REVIEW_GATE_RULES = (("n2000_t0.99", 2000, 0.99), ("n2000_t0.9", 2000, 0.9), ("n8000_t0.99", 8000, 0.99))
REVIEW_FOLD_GRID = np.round(10.0 ** (np.arange(-100, 21) / 100.0), 4)   # fold-index thresholds 0.1 .. 1.58
REVIEW_PARTIAL_FOLD_Q = (0.02, 0.05, 0.10, 0.20)

# ----------------------------------------------------------------------------------------------- controls
# map: (kind, params); effect: "embed" keeps the zoo topology, "kill" -> b1 = 0 (torn / folded), "blob" -> b1 = 0.
MED = {"level": "medium"}
TOPO_CONTROLS = [
    # --- core, b1 = 1 (wrapping unless noted)
    dict(name="ref_s1", map=("ref", {"s": 1.0}), core=True, desc="(cos, sin, omega), trajectories"),
    dict(name="ref_s0.3", map=("ref", {"s": 0.3}), core=True, desc="(cos, sin, omega/0.3)"),
    dict(name="ref_s3", map=("ref", {"s": 3.0}), core=True, desc="(cos, sin, omega/3)"),
    dict(name="ref_box", sampling="box", map=("ref", {"s": 1.0}), core=True, desc="i.i.d. uniform box states"),
    dict(name="emb8_mild", map=("emb8", {"level": "mild"}), core=True, desc="random smooth embedding into R^8, mild"),
    dict(name="emb8_medium", map=("emb8", MED), core=True, desc="random smooth embedding into R^8, medium"),
    dict(name="noise2", map=("ref", {}), noise=0.02, core=True, desc="ref + isotropic noise, 2% of RMS radius"),
    dict(name="noise5", map=("ref", {}), noise=0.05, core=True, desc="ref + 5% noise"),
    dict(name="emb8_medium_noise5", map=("emb8", MED), noise=0.05, core=True, desc="emb8 medium + 5% noise"),
    dict(name="helix", map=("helix", {"f": 1.0}), core=True,
         desc="(cos, sin, omega, 2 cos omega, 2 sin omega): loop outside the top-3 PCs"),
    dict(name="ring_ref", window=dict(E_lo=-0.5, E_hi=0.5), map=("ref", {}), core=True,
         desc="libration ring E in [-0.5, 0.5]: b1 = 1 but contractible in the cylinder (wrap 0)"),
    dict(name="ring_emb8", window=dict(E_lo=-0.5, E_hi=0.5), map=("emb8", MED), core=True, desc="ring, emb8 medium"),
    # --- core, b1 = 2
    dict(name="rot2_ref", window=dict(E_lo=1.5), map=("ref", {}), core=True,
         desc="rotations only E >= 1.5: two annuli"),
    dict(name="rot2_emb8", window=dict(E_lo=1.5), map=("emb8", MED), core=True, desc="two annuli, emb8 medium"),
    # --- core, b1 = 0
    dict(name="torn_r2", map=("torn", {}), effect="kill", core=True, desc="(theta_wrapped, omega) in R^2"),
    dict(name="torn_emb8", map=("torn_emb8", MED), effect="kill", core=True, desc="torn chart, emb8 medium"),
    dict(name="disk_ref", window=dict(E_hi=0.8), map=("ref", {}), core=True, desc="libration only E <= 0.8: disk"),
    dict(name="disk_emb8", window=dict(E_hi=0.8), map=("emb8", MED), core=True, desc="disk, emb8 medium"),
    dict(name="disk_noise5", window=dict(E_hi=0.8), map=("ref", {}), noise=0.05, core=True, desc="disk + 5% noise"),
    dict(name="folded", map=("folded", {}), effect="kill", core=True, desc="(cos theta, omega): non-injective fold"),
    dict(name="blob8", map=("blob", {"dim": 8}), effect="blob", core=True, desc="Gaussian blob in R^8"),
    # --- stress (detection limits; never used for the primary fit)
    dict(name="emb8_strong", map=("emb8", {"level": "strong"}), core=False, desc="random smooth embedding, strong"),
    dict(name="noise10", map=("ref", {}), noise=0.10, core=False, desc="ref + 10% noise"),
    *[dict(name=f"thin_{lab}", map=("thin", {"f": f}), core=False, desc=f"(f cos, f sin, omega), f = {lab}")
      for lab, f in (("1/2", 0.5), ("1/3", 1 / 3), ("1/5", 0.2), ("1/10", 0.1), ("1/20", 0.05), ("1/30", 1 / 30))],
    *[dict(name=f"aniso_{lab}", map=("aniso", {"f": f}), core=False, desc=f"(cos, f sin, omega), f = {lab}")
      for lab, f in (("1/3", 1 / 3), ("1/10", 0.1))],
    dict(name="torn_emb8_strong", map=("torn_emb8", {"level": "strong"}), effect="kill", core=False,
         desc="torn chart, emb8 strong"),
    dict(name="helix_thin", map=("helix", {"f": 1 / 3}), core=False, desc="helix with the loop shrunk to 1/3"),
]
# Energy controls: decodable / smooth labels (1 / 0 / None = graded, reported but not used for thresholds).
ENERGY_CONTROLS = [
    dict(name="ref", dec=1, smooth=1), dict(name="omega_x3", dec=1, smooth=1), dict(name="omega_x1/3", dec=1, smooth=1),
    dict(name="emb8_mild", dec=1, smooth=1), dict(name="emb8_medium", dec=1, smooth=1),
    dict(name="emb8_strong", dec=1, smooth=1),
    dict(name="fold_theta", dec=1, smooth=1), dict(name="fold_omega", dec=1, smooth=1),
    dict(name="torn", dec=1, smooth=1), dict(name="folded", dec=1, smooth=1),
    dict(name="sin3", dec=0, smooth=0), dict(name="sin10", dec=0, smooth=0), dict(name="sin30", dec=0, smooth=0),
    dict(name="perm", dec=0, smooth=0),
    dict(name="patch4", dec=None, smooth=0), dict(name="patch8", dec=None, smooth=0),
    dict(name="patch16", dec=None, smooth=0),
    dict(name="noise2", dec=None, smooth=None), dict(name="noise5", dec=None, smooth=None),
    dict(name="noise10", dec=None, smooth=None), dict(name="noise20", dec=None, smooth=None),
    dict(name="wiggle3", dec=None, smooth=None), dict(name="wiggle10", dec=None, smooth=None),
    dict(name="wiggle30", dec=None, smooth=None),
    dict(name="omega_x10", dec=None, smooth=None), dict(name="omega_x1/10", dec=None, smooth=None),
]
# Seam controls. group: "glued" (expect SR ~ 1), "torn" (expect SR >> 1), "graded" (partial gluing: dose-response,
# never used for the fit), "none" (data never reach theta = pi: expect status "insufficient").
# core = used to fit the threshold; stress = detection limits.
SEAM_CONTROLS = [
    # --- glued, core
    dict(name="g_ref", group="glued", core=True, map=("ref", {"s": 1.0}), desc="(cos, sin, omega)"),
    dict(name="g_ref_s0.3", group="glued", core=True, map=("ref", {"s": 0.3}), desc="(cos, sin, omega/0.3)"),
    dict(name="g_ref_s3", group="glued", core=True, map=("ref", {"s": 3.0}), desc="(cos, sin, omega/3)"),
    dict(name="g_box", group="glued", core=True, sampling="box", map=("ref", {"s": 1.0}), desc="i.i.d. box states"),
    dict(name="g_emb8_mild", group="glued", core=True, map=("emb8", {"level": "mild"}), desc="ref -> R^8, mild"),
    dict(name="g_emb8_medium", group="glued", core=True, map=("emb8", MED), desc="ref -> R^8, medium"),
    dict(name="g_noise5", group="glued", core=True, map=("ref", {"s": 1.0}), noise=0.05, desc="ref + 5% noise"),
    dict(name="g_emb8_medium_noise5", group="glued", core=True, map=("emb8", MED), noise=0.05,
         desc="ref -> R^8 medium, + 5% noise"),
    dict(name="g_helix", group="glued", core=True, map=("helix", {"f": 1.0}), desc="(cos, sin, omega, 2cos omega, 2sin omega)"),
    # --- torn, core
    dict(name="t_r2", group="torn", core=True, map=("torn", {"s": 1.0}), desc="(theta_raw, omega)"),
    dict(name="t_s0.3", group="torn", core=True, map=("torn", {"s": 0.3}), desc="(theta_raw, omega/0.3)"),
    dict(name="t_s3", group="torn", core=True, map=("torn", {"s": 3.0}), desc="(theta_raw, omega/3)"),
    dict(name="t_box", group="torn", core=True, sampling="box", map=("torn", {"s": 1.0}), desc="torn, i.i.d. box states"),
    dict(name="t_emb8_mild", group="torn", core=True, map=("torn_emb8", {"level": "mild"}), desc="torn -> R^8, mild"),
    dict(name="t_emb8_medium", group="torn", core=True, map=("torn_emb8", MED), desc="torn -> R^8, medium"),
    dict(name="t_noise5", group="torn", core=True, map=("torn", {"s": 1.0}), noise=0.05, desc="torn + 5% noise"),
    dict(name="t_emb8_medium_noise5", group="torn", core=True, map=("torn_emb8", MED), noise=0.05,
         desc="torn -> R^8 medium, + 5% noise"),
    # --- glued, stress
    dict(name="g_emb8_strong", group="glued", core=False, map=("emb8", {"level": "strong"}), desc="ref -> R^8, strong"),
    *[dict(name=f"g_aniso_sin_{lab}", group="glued", core=False, map=("aniso", {"f": f}), desc=f"(cos, {lab} sin, omega)")
      for lab, f in (("1/3", 1 / 3), ("1/10", 0.1))],
    *[dict(name=f"g_aniso_cos_{lab}", group="glued", core=False, map=("aniso_cos", {"f": f}),
           desc=f"({lab} cos, sin, omega)") for lab, f in (("1/3", 1 / 3), ("1/10", 0.1))],
    *[dict(name=f"g_thin_{lab}", group="glued", core=False, map=("thin", {"f": f}), desc=f"({lab} cos, {lab} sin, omega)")
      for lab, f in (("1/10", 0.1), ("1/30", 1 / 30))],
    *[dict(name=f"g_noise{p}", group="glued", core=False, map=("ref", {"s": 1.0}), noise=p / 100, desc=f"ref + {p}% noise")
      for p in (10, 20)],
    # --- torn, stress
    dict(name="t_emb8_strong", group="torn", core=False, map=("torn_emb8", {"level": "strong"}), desc="torn -> R^8, strong"),
    *[dict(name=f"t_omega_x{c}", group="torn", core=False, map=("torn", {"s": 1.0 / c}), desc=f"(theta_raw, {c} omega)")
      for c in (10, 30)],
    *[dict(name=f"t_noise{p}", group="torn", core=False, map=("torn", {"s": 1.0}), noise=p / 100, desc=f"torn + {p}% noise")
      for p in (10, 20)],
    # --- graded: an arc of angle 2 pi alpha (isometric in theta), gap 2 sin(pi alpha) / alpha at the cut
    *[dict(name=f"arc_{a}", group="graded", core=False, map=("arc", {"alpha": a}), ph=True,
           desc=f"(sin(a theta)/a, (1-cos(a theta))/a, omega), a = {a}") for a in ARC_ALPHAS],
    dict(name="arc_rot_glued", group="graded", core=False, ph=True, profile=True,
         map=("arc_omega", {"alpha_lo": 0.3, "w_lo": 1.0, "w_hi": 1.5, "glue_fast": True}),
         desc="glued for |omega| >= 1.5 (fast rotations), torn (a = 0.3) for |omega| <= 1, smooth in between"),
    dict(name="arc_rot_torn", group="graded", core=False, ph=True, profile=True,
         map=("arc_omega", {"alpha_lo": 0.3, "w_lo": 1.0, "w_hi": 1.5, "glue_fast": False}),
         desc="the reverse: torn for |omega| >= 1.5, glued for |omega| <= 1"),
    # --- no data at the cut
    dict(name="none_libration", group="none", core=False, window=dict(E_hi=0.5), map=("ref", {"s": 1.0}),
         desc="libration only E <= 0.5: never reaches theta = pi"),
]
# Theta-decodability gate controls: label 1 = theta-injective (must pass), 0 = theta folded / conserved
# quantities only (must fail), None = graded (reported only).
GATE_CONTROLS = [
    dict(name="ref", label=1, core=True, map=("ref", {"s": 1.0})),
    dict(name="ref_s0.3", label=1, core=True, map=("ref", {"s": 0.3})),
    dict(name="ref_s3", label=1, core=True, map=("ref", {"s": 3.0})),
    dict(name="box", label=1, core=True, sampling="box", map=("ref", {"s": 1.0})),
    dict(name="emb8_mild", label=1, core=True, map=("emb8", {"level": "mild"})),
    dict(name="emb8_medium", label=1, core=True, map=("emb8", MED)),
    dict(name="emb8_strong", label=1, core=True, map=("emb8", {"level": "strong"})),
    dict(name="helix", label=1, core=True, map=("helix", {"f": 1.0})),
    dict(name="torn", label=1, core=True, map=("torn", {"s": 1.0})),
    dict(name="torn_emb8", label=1, core=True, map=("torn_emb8", MED)),
    dict(name="fold_omega", label=1, core=True, map=("fold_omega", {}), desc="(cos, sin, |omega|): theta kept"),
    dict(name="noise2", label=1, core=True, map=("ref", {"s": 1.0}), noise=0.02),
    dict(name="noise5", label=1, core=True, map=("ref", {"s": 1.0}), noise=0.05),
    dict(name="cos_absomega", label=0, core=True, map=("cos_absomega", {}), desc="(cos theta, |omega|)"),
    dict(name="folded", label=0, core=True, map=("folded", {}), desc="(cos theta, omega)"),
    dict(name="fold_theta", label=0, core=True, map=("fold_theta", {}), desc="(cos theta, |sin theta|, omega)"),
    dict(name="energy_sign", label=0, core=True, map=("energy_sign", {}), desc="(E, sign omega)"),
    dict(name="energy_only", label=0, core=True, map=("energy_only", {}), desc="(E,)"),
    dict(name="energy_absomega", label=0, core=True, map=("energy_absomega", {}), desc="(E, |omega|)"),
    dict(name="energy_sign_emb8", label=0, core=True, map=("energy_sign_emb8", MED), desc="(E, sign omega) -> R^8"),
    dict(name="noise10", label=None, core=False, map=("ref", {"s": 1.0}), noise=0.10),
    dict(name="noise20", label=None, core=False, map=("ref", {"s": 1.0}), noise=0.20),
    dict(name="thin_1/3", label=None, core=False, map=("thin", {"f": 1 / 3})),
    dict(name="thin_1/10", label=None, core=False, map=("thin", {"f": 0.1})),
    dict(name="aniso_1/10", label=None, core=False, map=("aniso", {"f": 0.1})),
]
# Review (part E) seam controls: the core of SEAM_CONTROLS again (confirmation on fresh seeds), plus adversarial
# latents. fold = True: continuous across the cut but folded there (theta ~ -theta), the fold index must flag them.
REVIEW_SEAM_CONTROLS = [
    *[dict(s, fold=False) for s in SEAM_CONTROLS if s["core"]],
    *[dict(name=f"t_omega_x{c}", group="torn", core=False, fold=False, map=("torn", {"s": 1.0 / c}),
           desc=f"(theta_raw, {c} omega)") for c in (10, 20, 30, 50, 100)],
    *[dict(name=f"g_omega_x{c}", group="glued", core=False, fold=False, map=("ref", {"s": 1.0 / c}),
           desc=f"(cos, sin, {c} omega)") for c in (10, 30, 100)],
    dict(name="g_aniso_sin_1/10", group="glued", core=False, fold=False, map=("aniso", {"f": 0.1}),
         desc="(cos, 0.1 sin, omega): glued, squeezed along the cut"),
    dict(name="g_emb8_strong", group="glued", core=False, fold=False, map=("emb8", {"level": "strong"}),
         desc="ref -> R^8, strong"),
    dict(name="g_noise20", group="glued", core=False, fold=False, map=("ref", {"s": 1.0}), noise=0.20,
         desc="ref + 20% noise"),
    dict(name="t_noise20", group="torn", core=False, fold=False, map=("torn", {"s": 1.0}), noise=0.20,
         desc="torn + 20% noise"),
    dict(name="arc_0.9", group="graded", core=False, fold=False, map=("arc", {"alpha": 0.9}), desc="arc, gap 0.69"),
    dict(name="f_cos", group="folded", core=False, fold=True, map=("folded", {}), desc="(cos theta, omega)"),
    dict(name="f_abs", group="folded", core=False, fold=True, map=("fold_abs", {}), desc="(|theta|, omega)"),
    dict(name="f_iso", group="folded", core=False, fold=True, map=("fold_theta", {}),
         desc="(cos theta, |sin theta|, omega)"),
    dict(name="f_abs_emb8", group="folded", core=False, fold=True, map=("fold_abs_emb8", MED),
         desc="(|theta|, omega) -> R^8 medium"),
]
# Review theta-gate controls: the core of GATE_CONTROLS on fresh seeds, graded partial folds, three stress latents.
REVIEW_GATE_CONTROLS = [
    *[s for s in GATE_CONTROLS if s["core"]],
    *[dict(name=f"pfold_{q}", label=None, core=False, map=("partial_fold", {"q": q}),
           desc=f"theta -> -theta folded for the lowest-energy fraction {q} of the states") for q in REVIEW_PARTIAL_FOLD_Q],
    dict(name="ref_s0.1", label=None, core=False, map=("ref", {"s": 0.1}), desc="(cos, sin, 10 omega)"),
    dict(name="thin_1/10", label=None, core=False, map=("thin", {"f": 0.1})),
    dict(name="noise10", label=None, core=False, map=("ref", {"s": 1.0}), noise=0.10),
]


# ----------------------------------------------------------------------------------------------- helpers
def _seed(*parts) -> int:
    ints = [int(p) if isinstance(p, (int, np.integer)) else zlib.crc32(str(p).encode()) for p in parts]
    return int(np.random.SeedSequence(ints).generate_state(1)[0])


def _py(x):
    """JSON-safe conversion (numpy -> python, nan/inf -> None)."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return _py(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not np.isfinite(v) else round(v, 6)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def clopper_pearson(k: int, n: int, level: float = 0.95):
    if n == 0:
        return (None, None)
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return (round(lo, 4), round(hi, 4))


def _rate(k, n):
    return dict(k=int(k), n=int(n), rate=(round(k / n, 4) if n else None), ci95=clopper_pearson(int(k), int(n)))


# ----------------------------------------------------------------------------------------------- clouds
def build_topo_cloud(spec: dict, r: int):
    S, info = C.pendulum_states(CALIB_BASE + r, sampling=spec.get("sampling", "traj"), **spec.get("window", {}))
    kind, kw = spec["map"]
    ms = _seed(r, spec["name"], "map")
    if kind == "ref":
        Z = C.emb_ref(S, kw.get("s", 1.0))
    elif kind == "thin":
        Z = C.emb_thin(S, kw["f"])
    elif kind == "aniso":
        Z = C.emb_aniso(S, kw["f"])
    elif kind == "helix":
        Z = C.emb_helix(S, kw["f"])
    elif kind == "emb8":
        Z = C.RandomSmoothEmbedding(3, 8, kw["level"], seed=ms)(C.emb_ref(S))
    elif kind == "torn":
        Z = C.emb_torn(S)
    elif kind == "torn_emb8":
        Z = C.RandomSmoothEmbedding(2, 8, kw["level"], seed=ms)(C.emb_torn(S))
    elif kind == "folded":
        Z = C.emb_folded(S)
    elif kind == "blob":
        Z = C.gaussian_blob(len(S), kw.get("dim", 8), seed=ms)
    else:
        raise ValueError(kind)
    if spec.get("noise"):
        Z = C.add_noise(Z, spec["noise"], _seed(r, spec["name"], "noise"))
    effect = spec.get("effect", "embed")
    b1 = info["b1"] if effect == "embed" else 0
    wrap = info["wrap"] if effect == "embed" else 0
    return Z, S[:, 0], dict(b1=b1, wrap=wrap, zoo_betti=info["betti"], dim=int(Z.shape[1]))


def topo_truth(spec: dict) -> dict:
    """Ground truth of a topology control from the zoo (no sampling needed beyond a tiny probe)."""
    w = zoo.make("pendulum")
    params = {**C.PRE_REG_WORLD, **spec.get("window", {})}
    gt = w.ground_truth(**params)
    box = spec.get("sampling", "traj") == "box"
    betti = gt["init_region_betti"] if box else gt["region_betti"]
    wrap = gt["init_region_b1_wrapping"] if box else gt["region_b1_wrapping"]
    effect = spec.get("effect", "embed")
    return dict(b1=int(betti[1]) if effect == "embed" else 0, wrap=int(wrap) if effect == "embed" else 0,
                zoo_betti=list(betti), effect=effect)


def build_energy_cloud(name: str, S, Zref, r: int):
    ms = _seed(r, "energy", name)
    if name == "ref":
        return Zref.copy()
    if name.startswith("omega_x"):
        c = {"omega_x3": 3.0, "omega_x1/3": 1 / 3, "omega_x10": 10.0, "omega_x1/10": 0.1}[name]
        return C.emb_ref(S, 1.0 / c)
    if name.startswith("emb8_"):
        return C.RandomSmoothEmbedding(3, 8, name.split("_")[1], seed=ms)(Zref)
    if name == "fold_theta":
        return C.emb_fold_theta(S)
    if name == "fold_omega":
        return C.emb_fold_omega(S)
    if name == "torn":
        return C.emb_torn(S)
    if name == "folded":
        return C.emb_folded(S)
    if name.startswith("sin"):
        return C.emb_sin_omega(S, float(name[3:]))
    if name == "perm":
        return Zref[np.random.default_rng(ms).permutation(len(Zref))]
    if name.startswith("patch"):
        return C.patch_shuffle(S, Zref, int(name[5:]), amp=0.5, seed=ms)
    if name.startswith("noise"):
        return C.add_noise(Zref, float(name[5:]) / 100.0, ms)
    if name.startswith("wiggle"):
        return C.emb_wiggle(S, float(name[6:]), a=0.9)
    raise ValueError(name)


# ----------------------------------------------------------------------------------------------- workers
def _diag_record(dg: dict, b1_true: int, sub_seconds: float, want_ell: bool = False, bs_exact: bool = True) -> dict:
    bs = T.bs_test(dg, alpha=ALPHA_BS)
    order = np.argsort(-np.asarray(dg["pi"]))
    top = T.top_pi(dg, 4)
    cens = np.zeros(4, bool)
    cens[:min(4, len(order))] = np.asarray(dg["censored"])[order[:4]]
    rec = dict(top=np.round(top, 4), cens=cens, bs=int(bs["n_sig"]) if bs_exact else None,
               pistar=bs["pi_star"], m=int(bs["m"]), sec=round(dg["seconds"] + sub_seconds, 4),
               passes=int(dg["passes"]), T=dg["thresh"])
    noise = order[b1_true:]
    ell = np.asarray(bs["ell"])[noise]
    fin = np.isfinite(ell)
    if fin.sum() >= 10:
        p = np.asarray(bs["p"])[noise][fin]
        rec["null"] = dict(ks=float(stats.kstest(ell[fin], T.lgumbel_cdf).statistic),
                           rej05=float(np.mean(p < 0.05)), rej01=float(np.mean(p < 0.01)), n=int(fin.sum()))
        if want_ell:
            rec["ell"] = ell[fin].astype(np.float32)
    return rec


def _subsample(Z, n, method, seed):
    t = time.perf_counter()
    if method == "fps":
        idx = T.fps_subsample(Z, n, seed)
    elif method == "random":
        idx = T.random_subsample(Z, n, seed)
    elif method == "fps_half":                              # FPS on a random half: subsample variability + FPS
        half = T.random_subsample(Z, len(Z) // 2, _seed(seed, "half"))
        idx = half[T.fps_subsample(Z[half], n, seed)]
    else:
        raise ValueError(method)
    return idx, time.perf_counter() - t


def topo_task(spec: dict, r: int, cfg: dict) -> dict:
    t_task = time.perf_counter()
    Z, theta, truth = build_topo_cloud(spec, r)
    b1 = truth["b1"]
    name = spec["name"]
    out = dict(name=name, r=r, truth=truth, main={}, white={}, stab={}, wind=None)
    want_ell = b1 == 0 or spec.get("core", False)
    # A. main grid: n x {fps, random}
    for n in cfg["ns"]:
        for method in ("fps", "random"):
            idx, ts = _subsample(Z, n, method, _seed(r, name, n, method, 0))
            bs_exact = method == "random" or n <= 600
            dg = T.h1_diagram(Z[idx], pi_need=PI_NEED, bs_alpha=ALPHA_BS if bs_exact else None)
            out["main"][f"{method}/{n}"] = _diag_record(dg, b1, ts, want_ell=want_ell and n == 600, bs_exact=bs_exact)
    # B. whitening (FPS only): pca3 = top-3 PCs, full = all PCs (identical when dim <= 3)
    if cfg["white"]:
        variants = {"full": None} if Z.shape[1] <= 3 else {"pca3": 3, "full": None}
        for vname, k in variants.items():
            Zw = T.pca_whiten(Z, k)[0]
            for n in cfg["white_ns"]:
                idx, ts = _subsample(Zw, n, "fps", _seed(r, name, n, "fps", 0))
                dg = T.h1_diagram(Zw[idx], pi_need=PI_NEED)
                out["white"][f"{vname}/{n}"] = _diag_record(dg, b1, ts, bs_exact=False)
        if "pca3" not in variants:
            for n in cfg["white_ns"]:
                out["white"][f"pca3/{n}"] = out["white"][f"full/{n}"]
    # C. stability: STAB_R subsamples, core controls only
    if cfg["stab"] and spec.get("core", False):
        for method, n in cfg["stab_plan"]:
            recs = []
            for j in range(cfg["stab_r"]):
                if method == "random" and j == 0:
                    recs.append(out["main"][f"random/{n}"])
                    continue
                idx, ts = _subsample(Z, n, method, _seed(r, name, n, method, j))
                dg = T.h1_diagram(Z[idx], pi_need=PI_NEED)
                recs.append(_diag_record(dg, b1, ts, bs_exact=False))
            out["stab"][f"{method}/{n}"] = [dict(top=x["top"], cens=x["cens"], sec=x["sec"]) for x in recs]
    # D. winding of the top class (b1 = 1 controls)
    if cfg["wind"] and b1 == 1:
        idx, ts = _subsample(Z, WIND_N, "fps", _seed(r, name, WIND_N, "fps", 0))
        t = time.perf_counter()
        dg = T.h1_diagram(Z[idx], pi_need=PI_NEED, coeff=WIND_COEFF, do_cocycles=True)
        wc = T.winding_check(dg, theta[idx])
        out["wind"] = dict(abs_rho=wc["abs_rho"], degree=wc["degree"], coverage=wc["coverage"],
                           min_resultant=wc["min_resultant"], top=T.top_pi(dg, 2),
                           sec=round(time.perf_counter() - t + ts, 4))
    out["seconds"] = time.perf_counter() - t_task
    return out


def energy_task(spec: dict, r: int, cfg: dict) -> dict:
    t_task = time.perf_counter()
    S, info = C.pendulum_states(CALIB_BASE + r)
    E = info["energy"]
    perm = np.random.default_rng(_seed(r, "energy", "subsample")).permutation(len(S))[:max(cfg["energy_ns"])]
    S, E = S[perm], E[perm]                                # random order: every prefix is a random subset
    Zref = C.emb_ref(S)
    Z = build_energy_cloud(spec["name"], S, Zref, r)
    out = dict(name=spec["name"], r=r, scores={})
    seed = _seed(r, "energy", "cv")
    for n in cfg["energy_ns"]:
        rep = SM.energy_report(Z[:n], E[:n], Zref[:n], ks=cfg["energy_ks"], seed=seed, krr=False)
        Zw = T.pca_whiten(Z[:n])[0]
        for k in cfg["energy_ks"]:
            rep[f"white_knn{k}_r2"] = SM.knn_probe_r2(Zw, E[:n], k, seed=seed)
            sm = SM.smoothness_score(Zw, E[:n], Zref[:n], k=k)
            rep[f"white_s_k{k}"], rep[f"white_lip_tail_k{k}"] = sm["s"], sm["lip_tail"]
        out["scores"][str(n)] = rep
    kr = SM.krr_probe_r2(Z[:KRR_N], E[:KRR_N], seed=seed, max_n=KRR_N)
    out["scores"]["krr"] = dict(krr_r2=kr["r2"], krr_gamma=kr["gamma"], krr_alpha=kr["alpha"])
    out["seconds"] = time.perf_counter() - t_task
    return out


def void_radius(spec: dict, r: int) -> float:
    """Radius sqrt(2 (min E + omega0^2)) of the innermost sampled orbit around the stable equilibrium: the
    finite trajectory sample leaves a hole of about this radius around (theta, omega) = (0, 0)."""
    _, info = C.pendulum_states(CALIB_BASE + r, sampling=spec.get("sampling", "traj"), **spec.get("window", {}))
    return float(np.sqrt(2.0 * (info["energy"].min() + 1.0)))


def confirm_task(spec: dict, r: int, cfg: dict) -> dict:
    """Fresh-seed confirmation (repeats R..2R-1) of the candidate instruments fixed in CONFIRM_CANDIDATES."""
    t_task = time.perf_counter()
    Z, theta, truth = build_topo_cloud(spec, r)
    b1, name = truth["b1"], spec["name"]
    out = dict(name=name, r=r, truth=truth, main={}, stab={}, wind=None,
               void_radius=void_radius(spec, r) if spec["map"][0] != "blob" else None)
    for n in cfg["confirm_ns"]:
        idx, ts = _subsample(Z, n, "fps", _seed(r, name, n, "fps", 0))
        dg = T.h1_diagram(Z[idx], pi_need=PI_NEED)
        out["main"][f"fps/{n}"] = dict(top=T.top_pi(dg, 4), sec=round(dg["seconds"] + ts, 4))
    for method, n in cfg["confirm_stab"]:
        recs = []
        for j in range(cfg["stab_r"]):
            idx, ts = _subsample(Z, n, method, _seed(r, name, n, method, j))
            recs.append(T.top_pi(T.h1_diagram(Z[idx], pi_need=PI_NEED), 4))
        out["stab"][f"{method}/{n}"] = recs
    if b1 == 1:
        idx, ts = _subsample(Z, WIND_N, "fps", _seed(r, name, WIND_N, "fps", 0))
        dg = T.h1_diagram(Z[idx], pi_need=PI_NEED, coeff=WIND_COEFF, do_cocycles=True)
        wc = T.winding_check(dg, theta[idx])
        out["wind"] = dict(abs_rho=wc["abs_rho"], degree=wc["degree"])
    out["seconds"] = time.perf_counter() - t_task
    return out


def build_gt_cloud(kind: str, kw: dict, S, E, ms: int) -> np.ndarray:
    """Ground-truth embeddings for the seam / gate calibration (states S, energy E, map seed ms)."""
    th, om = S[:, 0], S[:, 1]
    if kind == "ref":
        return C.emb_ref(S, kw.get("s", 1.0))
    if kind == "torn":
        return np.stack([th, om / kw.get("s", 1.0)], 1)
    if kind == "emb8":
        return C.RandomSmoothEmbedding(3, 8, kw["level"], seed=ms)(C.emb_ref(S))
    if kind == "torn_emb8":
        return C.RandomSmoothEmbedding(2, 8, kw["level"], seed=ms)(C.emb_torn(S))
    if kind == "helix":
        return C.emb_helix(S, kw["f"])
    if kind == "aniso":
        return C.emb_aniso(S, kw["f"])
    if kind == "aniso_cos":
        return np.stack([kw["f"] * np.cos(th), np.sin(th), om], 1)
    if kind == "thin":
        return C.emb_thin(S, kw["f"])
    if kind == "arc":
        return C.emb_arc(S, kw["alpha"])
    if kind == "arc_omega":
        return C.emb_arc_omega(S, **kw)
    if kind == "fold_omega":
        return C.emb_fold_omega(S)
    if kind == "cos_absomega":
        return np.stack([np.cos(th), np.abs(om)], 1)
    if kind == "folded":
        return C.emb_folded(S)
    if kind == "fold_theta":
        return C.emb_fold_theta(S)
    if kind == "energy_sign":
        return np.stack([E, np.sign(om)], 1)
    if kind == "energy_only":
        return np.asarray(E, dtype=np.float64)[:, None]
    if kind == "energy_absomega":
        return np.stack([E, np.abs(om)], 1)
    if kind == "energy_sign_emb8":
        return C.RandomSmoothEmbedding(2, 8, kw["level"], seed=ms)(np.stack([E, np.sign(om)], 1))
    if kind == "fold_abs":
        return C.emb_fold_abs(S)
    if kind == "fold_abs_emb8":
        return C.RandomSmoothEmbedding(2, 8, kw["level"], seed=ms)(C.emb_fold_abs(S))
    if kind == "partial_fold":
        return C.emb_partial_fold(S, E, float(np.quantile(E, kw["q"])))
    raise ValueError(kind)


def pilot_seam_ratio(Z, th, om) -> float:
    """The reviewer pilot's seam formula (cells/0/exploratory_reviewer_pilot/wm_experiment.py), re-implemented
    for comparison only: rotations |omega| > 1.6 of each sign, window 0.12 on each side, <= 300 points per side,
    |d omega| < 0.08, median pair distance; the cut at pi over the mean of the same at -pi/2, 0, pi/2."""
    def side_median(th0, width=0.12):
        d = np.angle(np.exp(1j * (th - th0)))
        a, b = (d > -width) & (d < 0), (d >= 0) & (d < width)
        ds = []
        for sgn in (1, -1):
            A, B = np.flatnonzero(a & (sgn * om > 1.6))[:300], np.flatnonzero(b & (sgn * om > 1.6))[:300]
            if len(A) == 0 or len(B) == 0:
                continue
            m = np.abs(om[A][:, None] - om[B][None]) < 0.08
            dz = np.linalg.norm(Z[A][:, None] - Z[B][None], axis=-1)
            if m.any():
                ds.append(np.median(dz[m]))
        return float(np.mean(ds)) if ds else np.nan

    ctrl = [side_median(c) for c in (-np.pi / 2, 0.0, np.pi / 2)]
    return side_median(np.pi) / np.nanmean(ctrl) if np.isfinite(ctrl).any() else np.nan


def seam_task(spec: dict, r: int, cfg: dict) -> dict:
    """Part D, one repeat: every seam control (seam ratio + bootstrap; H1 rule on the graded arcs) and every
    theta-gate control, all on states drawn with SEAM_BASE + r."""
    t_task = time.perf_counter()
    states = {}

    def get_states(sp):
        key = (sp.get("sampling", "traj"), tuple(sorted(sp.get("window", {}).items())))
        if key not in states:
            S, info = C.pendulum_states(SEAM_BASE + r, sampling=key[0], **dict(key[1]))
            # trajectory ids for the cluster bootstrap (pendulum_states keeps t = 1..64 of each trajectory, in order)
            grp = np.arange(len(S)) // 64 if key[0] == "traj" else None
            states[key] = (S, info["energy"], grp)
        return states[key]

    out = dict(name="seam", r=r, seam={}, gate={})
    for sp in cfg["seam_specs"]:
        nm = sp["name"]
        S, E, grp = get_states(sp)
        Z = build_gt_cloud(sp["map"][0], sp["map"][1], S, E, _seed(r, "seam", nm, "map"))
        if sp.get("noise"):
            Z = C.add_noise(Z, sp["noise"], _seed(r, "seam", nm, "noise"))
        t = time.perf_counter()
        res = SE.seam_ratio(Z, S[:, 0], S[:, 1], groups=grp, n_boot=cfg["seam_nboot"],
                            seed=_seed(r, "seam", nm, "boot"), **SEAM_KW)
        rec = dict(ratio=res["ratio"], ci95=res["ci95"], log_sd=res["log_sd"], status=res["status"],
                   n_cut_pairs=res["n_cut_pairs"], n_strata=res["n_strata"], n_boot_ok=res["n_boot_ok"],
                   n_cut_points=res["window_points"]["cut"], placebo=res["placebo"], placebo_max=res["placebo_max"],
                   sec=round(time.perf_counter() - t, 4), pilot=pilot_seam_ratio(Z, S[:, 0], S[:, 1]),
                   ratio_fast=SE.seam_ratio(Z, S[:, 0], S[:, 1], n_boot=0, diagnostics=False,
                                            omega_abs_min=SEAM_FAST, **SEAM_KW)["ratio"])
        if sp.get("profile"):
            rec["by_omega"] = res["by_omega"]
        if sp.get("ph"):
            idx = T.fps_subsample(Z, SEAM_PH_N, _seed(r, "seam", nm, "fps"))
            top = T.top_pi(T.h1_diagram(Z[idx], pi_need=PI_NEED), 4)
            rec.update(ph_top=top[:2], ph_b1=T.count_pi_threshold(top, SEAM_PH_TAU))
        out["seam"][nm] = rec
    for sp in cfg["gate_specs"]:
        nm = sp["name"]
        S, E, _ = get_states(sp)
        Z = build_gt_cloud(sp["map"][0], sp["map"][1], S, E, _seed(r, "gate", nm, "map"))
        if sp.get("noise"):
            Z = C.add_noise(Z, sp["noise"], _seed(r, "gate", nm, "noise"))
        g = GT.theta_decodability(Z, S[:, 0], k=GATE_K, n=GATE_N, seed=_seed(r, "gate", nm, "probe"),
                                  threshold=GATE_THRESHOLD_PREREG)
        out["gate"][nm] = dict(r2_cos=g["r2_cos"], r2_sin=g["r2_sin"], r2_min=g["r2_min"])
    out["seconds"] = time.perf_counter() - t_task
    return out


def review_task(spec: dict, r: int, cfg: dict) -> dict:
    """Part E, one repeat on REVIEW_BASE + r: seam ratio (bootstrap only for the core controls) with the review
    diagnostics (omega aspect, fold index, number of trajectories at the cut) and the fast-only readout; the
    theta gate at n = 2000 and n = 8000."""
    t_task = time.perf_counter()
    states = {}

    def get_states(sp):
        key = (sp.get("sampling", "traj"), tuple(sorted(sp.get("window", {}).items())))
        if key not in states:
            S, info = C.pendulum_states(REVIEW_BASE + r, sampling=key[0], **dict(key[1]))
            states[key] = (S, info["energy"], np.arange(len(S)) // 64 if key[0] == "traj" else None)
        return states[key]

    out = dict(name="review", r=r, seam={}, gate={})
    for sp in cfg["review_seam_specs"]:
        nm = sp["name"]
        S, E, grp = get_states(sp)
        Z = build_gt_cloud(sp["map"][0], sp["map"][1], S, E, _seed(r, "review", nm, "map"))
        if sp.get("noise"):
            Z = C.add_noise(Z, sp["noise"], _seed(r, "review", nm, "noise"))
        nb = cfg["seam_nboot"] if sp["core"] else 0
        res = SE.seam_ratio(Z, S[:, 0], S[:, 1], groups=grp, n_boot=nb, seed=_seed(r, "review", nm, "boot"), **SEAM_KW)
        out["seam"][nm] = dict(ratio=res["ratio"], ci95=res["ci95"], status=res["status"],
                               n_cut_pairs=res["n_cut_pairs"], n_cut_groups=res["n_cut_groups"],
                               omega_aspect=res["omega_aspect"], fold_index=res["fold_index"],
                               fold_index_ctrl_min=(float(np.nanmin(res["fold_index_ctrl"]))
                                                    if np.isfinite(res["fold_index_ctrl"]).any() else np.nan),
                               placebo_max=res["placebo_max"],
                               ratio_fast=SE.seam_ratio(Z, S[:, 0], S[:, 1], n_boot=0, diagnostics=False,
                                                        omega_abs_min=SEAM_FAST, **SEAM_KW)["ratio"])
    for sp in cfg["review_gate_specs"]:
        nm = sp["name"]
        S, E, _ = get_states(sp)
        Z = build_gt_cloud(sp["map"][0], sp["map"][1], S, E, _seed(r, "review", "gate", nm, "map"))
        if sp.get("noise"):
            Z = C.add_noise(Z, sp["noise"], _seed(r, "review", "gate", nm, "noise"))
        rec = {}
        for n in sorted({n for _, n, _ in REVIEW_GATE_RULES}):
            g = GT.theta_decodability(Z, S[:, 0], k=GATE_K, n=n, seed=_seed(r, "review", "gate", nm, n))
            rec[f"r2_min_n{n}"] = g["r2_min"]
        out["gate"][nm] = rec
    out["seconds"] = time.perf_counter() - t_task
    return out


def _run(task):
    kind, spec, r, cfg = task
    fn = {"topo": topo_task, "energy": energy_task, "confirm": confirm_task, "seam": seam_task,
          "review": review_task}[kind]
    return kind, fn(spec, r, cfg)


# ----------------------------------------------------------------------------------------------- topology analysis
def rule_counts(top: np.ndarray, family: str, param) -> np.ndarray:
    """b1_hat for every diagram (top: [..., 4] descending pi lower bounds)."""
    if family == "thr":
        return (top >= param).sum(-1)
    ratios = top[..., :3] / np.maximum(top[..., 1:4], 1e-300)
    if family == "gap":
        m = ratios.argmax(-1)
        return np.where(np.take_along_axis(ratios, m[..., None], -1)[..., 0] >= param, m + 1, 0)
    if family == "thrgap":
        tau, g = param
        m = (top >= tau).sum(-1)
        prev = np.take_along_axis(top, np.clip(m - 1, 0, 3)[..., None], -1)[..., 0]
        nxt = np.where(m < 4, np.take_along_axis(top, np.clip(m, 0, 3)[..., None], -1)[..., 0], 1.0)
        return np.where(m == 0, 0, np.where(prev / np.maximum(nxt, 1e-300) >= g, m, -1))
    raise ValueError(family)


def stable_decision(b: np.ndarray, tau_s: float) -> np.ndarray:
    """b: [..., R] per-subsample counts -> mode if its frequency >= tau_s (and >= 0), else -1."""
    vals = np.arange(0, 5)
    freq = (b[..., None] == vals).mean(-2)                      # [..., 5]
    m = freq.argmax(-1)
    return np.where(freq.max(-1) >= tau_s - 1e-12, m, -1)


FAMILY_PARAMS = {"bs": [None], "thr": list(TAU_GRID), "gap": list(GAP_GRID),
                 "thrgap": [(t, g) for t in TAU_GRID for g in GAP_GRID]}
FAMILY_NPARAM = {"bs": 0, "thr": 1, "gap": 1, "thrgap": 2}


def _class_scores(dec: np.ndarray, b1: np.ndarray, sel_c: np.ndarray, reps: np.ndarray):
    """Mean accuracy over positive (b1 >= 1) and negative controls in sel_c, using repeats reps."""
    acc = (dec[:, reps] == b1[:, None]).mean(1)
    pos, neg = sel_c & (b1 >= 1), sel_c & (b1 == 0)
    tpr = acc[pos].mean() if pos.any() else np.nan
    tnr = acc[neg].mean() if neg.any() else np.nan
    return tpr, tnr, acc


def _flat(p) -> tuple:
    if p is None:
        return ()
    if isinstance(p, (tuple, list)):
        return tuple(x for q in p for x in _flat(q))
    return (float(p),)


def plateau_center(params, opt_idx) -> int:
    """Index of the 'centre' of the optimal parameter set: dimension by dimension, the optimal value closest
    (in log scale) to the geometric mean of the smallest and largest optimal values. Keeps thresholds away from
    the edge of the calibration data (the noise ceiling / the weakest signal)."""
    cand = [int(i) for i in opt_idx]
    flat = [_flat(params[i]) for i in cand]
    for d in range(len(flat[0]) if flat and flat[0] else 0):
        vals = sorted(set(f[d] for f in flat))
        target = np.sqrt(vals[0] * vals[-1])
        v = min(vals, key=lambda x: abs(np.log(x / target)))
        keep = [k for k, f in enumerate(flat) if f[d] == v]
        cand, flat = [cand[k] for k in keep], [flat[k] for k in keep]
    return cand[0]


def fit_and_eval(decide, params, b1, core, R):
    """Fit params on the calibration half of the core controls (maximise min(TPR, TNR) = mean accuracy of the
    worse class; among the optimal params take the plateau centre) and evaluate on the held-out half.
    The stress-control score of the chosen params (calibration half) is reported and used only to rank
    configurations against each other. ``decide(param) -> dec [C, R]``."""
    cal, hold = np.arange(R) < R // 2, np.arange(R) >= R // 2
    stress = ~core
    scores = []
    for p in params:
        dec = decide(p)
        tpr, tnr, _ = _class_scores(dec, b1, core, cal)
        st_p, st_n, _ = _class_scores(dec, b1, stress, cal)
        sec = np.nanmean([st_p, st_n]) if stress.any() else 0.0
        scores.append((float(np.nanmin([tpr, tnr])), sec))
    obj = np.array([s[0] for s in scores])
    sec = np.array([s[1] for s in scores])
    best = np.flatnonzero(obj >= obj.max() - 1e-12)
    i = plateau_center(params, best)
    p = params[i]
    plateau = [_flat(params[j]) for j in best]
    dec = decide(p)
    tpr_c, tnr_c, _ = _class_scores(dec, b1, core, cal)
    per = {}
    tot = dict(pos=[0, 0], neg=[0, 0], fp1=[0, 0], fpany=[0, 0])
    for c in range(len(b1)):
        d = dec[c, hold]
        k = int((d == b1[c]).sum())
        per[c] = dict(correct=_rate(k, len(d)), counts={str(v): int((d == v).sum()) for v in np.unique(d)})
        if core[c]:
            key = "pos" if b1[c] >= 1 else "neg"
            tot[key][0] += k
            tot[key][1] += len(d)
            if b1[c] == 0:
                tot["fp1"][0] += int((d == 1).sum())
                tot["fp1"][1] += len(d)
                tot["fpany"][0] += int((d >= 1).sum())
                tot["fpany"][1] += len(d)
    lo = [min(f[d] for f in plateau) for d in range(len(plateau[0]))] if plateau and plateau[0] else []
    hi = [max(f[d] for f in plateau) for d in range(len(plateau[0]))] if plateau and plateau[0] else []
    return dict(param=p, calib=dict(objective=float(obj[i]), tpr_core=float(tpr_c), tnr_core=float(tnr_c),
                                    stress_score=float(sec[i]), plateau_size=int(len(best)),
                                    plateau_lo=lo, plateau_hi=hi),
                heldout=dict(tpr_core=_rate(*tot["pos"]), tnr_core=_rate(*tot["neg"]),
                             fpr_claims_b1_eq_1=_rate(*tot["fp1"]), fpr_claims_b1_ge_1=_rate(*tot["fpany"])),
                per_control=per)


def analyze_topology(recs: dict, specs: list, R: int, cfg: dict) -> dict:
    names = [s["name"] for s in specs]
    b1 = np.array([recs[nm][0]["truth"]["b1"] for nm in names])
    core = np.array([s.get("core", False) for s in specs])
    res = dict(configs={}, controls=[dict(name=s["name"], desc=s.get("desc", ""), core=bool(s.get("core")),
                                          **recs[s["name"]][0]["truth"]) for s in specs])

    def arr(section, key, field="top"):
        return np.array([[recs[nm][r][section][key][field] for r in range(R)] for nm in names], dtype=float)

    configs = [("main", f"{m}/{n}") for n in cfg["ns"] for m in ("fps", "random")]
    if cfg["white"]:
        configs += [("white", f"{v}/{n}") for n in cfg["white_ns"] for v in ("pca3", "full")]
    for section, key in configs:
        top = arr(section, key)
        fam_res = {}
        for fam, params in FAMILY_PARAMS.items():
            if fam == "bs":
                vals = [[recs[nm][r][section][key]["bs"] for r in range(R)] for nm in names]
                if any(v is None for row in vals for v in row):
                    continue
                bsarr = np.array(vals)
                fam_res[fam] = fit_and_eval(lambda p: bsarr, params, b1, core, R)
            else:
                fam_res[fam] = fit_and_eval(lambda p, f=fam: rule_counts(top, f, p), params, b1, core, R)
        res["configs"][f"{section}:{key}"] = fam_res
    if cfg["stab"]:
        core_idx = np.flatnonzero(core)
        for method, n in cfg["stab_plan"]:
            key = f"{method}/{n}"
            top = np.array([[[x["top"] for x in recs[names[c]][r]["stab"][key]] for r in range(R)]
                            for c in core_idx], dtype=float)                      # [Cc, R, Rs, 4]
            fam_res = {}
            for fam in ("thr", "gap", "thrgap"):
                params = [(ts, p) for ts in TAU_S_GRID for p in FAMILY_PARAMS[fam]]       # (tau_s, rule param)
                fam_res[fam] = fit_and_eval(
                    lambda pp, f=fam: stable_decision(rule_counts(top, f, pp[1]), pp[0]),
                    params, b1[core_idx], np.ones(len(core_idx), bool), R)
            res["configs"][f"stab:{key}"] = fam_res
        res["stab_controls"] = [names[c] for c in core_idx]
    # Selection of the recommended instrument, calibration half only. Order (fixed before the full run):
    # core objective min(TPR, TNR) > stress-control score (not available for stability, which runs on core
    # controls only) > smaller n > fewer moving parts (#rule parameters + whitening + stability) > FPS.
    cands = []
    for ck, fams in res["configs"].items():
        n = int(ck.split("/")[-1])
        complexity = (1 if ck.startswith("white") else 0) + (1 if ck.startswith("stab") else 0)
        fps_pref = 1 if ("fps" in ck or ck.startswith("white")) else 0
        for fam, fr in fams.items():
            cands.append((fr["calib"]["objective"], fr["calib"]["stress_score"] if not ck.startswith("stab") else -1,
                          -n, -(FAMILY_NPARAM[fam] + complexity), fps_pref, ck, fam))
    cands.sort(key=lambda c: c[:5], reverse=True)
    res["ranking_calib"] = [dict(config=c[5], family=c[6], objective=c[0], stress=c[1]) for c in cands[:25]]
    ck_best, fam_best = cands[0][5], cands[0][6]
    res["recommended"] = dict(config=ck_best, family=fam_best, param=res["configs"][ck_best][fam_best]["param"])
    # per-control names in per_control dicts
    for ck, fams in res["configs"].items():
        nm_list = res.get("stab_controls") if ck.startswith("stab") else names
        for fam, fr in fams.items():
            fr["per_control"] = {nm_list[c]: v for c, v in fr["per_control"].items()}
    return res


def analyze_null(recs: dict, specs: list, R: int, cfg: dict) -> dict:
    """LGumbel null check: per (method, n) for negatives and noise bars of positives."""
    out = {}
    for n in cfg["ns"]:
        for method in ("fps", "random"):
            key = f"{method}/{n}"
            for grp in ("neg", "pos"):
                ks, rej05, rej01, fp, pistar, pooled = [], [], [], [], [], []
                for s in specs:
                    if not s.get("core"):
                        continue
                    for r in range(R):
                        rc = recs[s["name"]][r]
                        b1 = rc["truth"]["b1"]
                        if (grp == "neg") != (b1 == 0):
                            continue
                        d = rc["main"][key]
                        if "null" in d:
                            ks.append(d["null"]["ks"])
                            rej05.append(d["null"]["rej05"])
                            rej01.append(d["null"]["rej01"])
                        if d["bs"] is not None and b1 == 0:
                            fp.append(d["bs"] >= 1)
                        pistar.append(d["pistar"])
                        if "ell" in d and grp == "neg":
                            pooled.append(np.asarray(d["ell"], float))
                entry = dict(mean_ks_D=float(np.mean(ks)) if ks else None,
                             bar_rejection_at_0p05=float(np.mean(rej05)) if rej05 else None,
                             bar_rejection_at_0p01=float(np.mean(rej01)) if rej01 else None,
                             median_pi_star=float(np.median(pistar)) if pistar else None)
                if fp:
                    entry["bonferroni_fpr_per_diagram"] = _rate(int(np.sum(fp)), len(fp))
                if pooled:
                    allv = np.concatenate(pooled)
                    k = stats.kstest(allv, T.lgumbel_cdf)
                    entry.update(pooled_ks_D=float(k.statistic), pooled_ks_p=float(k.pvalue), pooled_n=int(len(allv)),
                                 pooled_sd=float(allv.std()), lgumbel_sd=float(np.pi / np.sqrt(6)),
                                 pooled_q99=float(np.quantile(allv, 0.99)),
                                 lgumbel_q99=float(np.log(-np.log(0.01))))
                out[f"{key}/{grp}"] = entry
    return out


def analyze_runtime(recs: dict, specs: list, R: int, cfg: dict) -> dict:
    out = {}
    for n in cfg["ns"]:
        for method in ("fps", "random"):
            for grp in ("b1>=1", "b1=0"):
                v = [recs[s["name"]][r]["main"][f"{method}/{n}"]["sec"] for s in specs for r in range(R)
                     if (recs[s["name"]][r]["truth"]["b1"] >= 1) == (grp == "b1>=1")]
                if v:
                    out[f"{method}/{n}/{grp}"] = dict(median_s=float(np.median(v)), q90_s=float(np.quantile(v, 0.9)),
                                                      max_s=float(np.max(v)), n=len(v))
    if cfg["stab"]:
        for method, n in cfg["stab_plan"]:
            v = [sum(x["sec"] for x in recs[s["name"]][r]["stab"][f"{method}/{n}"]) for s in specs
                 if s.get("core") for r in range(R)]
            out[f"stability_R{cfg['stab_r']}/{method}/{n}"] = dict(median_s=float(np.median(v)),
                                                                  q90_s=float(np.quantile(v, 0.9)))
    w = [recs[s["name"]][r]["wind"]["sec"] for s in specs for r in range(R) if recs[s["name"]][r]["wind"]]
    if w:
        out[f"winding/fps/{WIND_N}"] = dict(median_s=float(np.median(w)), q90_s=float(np.quantile(w, 0.9)))
    return out


def analyze_winding(recs: dict, specs: list, R: int) -> dict:
    rows = [(s, r) for s in specs for r in range(R) if recs[s["name"]][r]["wind"] is not None]
    if not rows:
        return {}
    cal = lambda r: r < R // 2
    pos_core = [(s, r) for s, r in rows if s.get("core") and recs[s["name"]][r]["truth"]["wrap"] == 1]
    neg = [(s, r) for s, r in rows if recs[s["name"]][r]["truth"]["wrap"] == 0]
    rho = lambda s, r: recs[s["name"]][r]["wind"]["abs_rho"]
    best, bi = None, None
    sc = []
    for t in RHO_GRID:
        tp = np.mean([rho(s, r) >= t for s, r in pos_core if cal(r)])
        tn = np.mean([rho(s, r) < t for s, r in neg if cal(r)])
        sc.append(min(tp, tn))
    sc = np.array(sc)
    opt = np.flatnonzero(sc >= sc.max() - 1e-12)
    thr = float(RHO_GRID[opt[len(opt) // 2]])
    hold_pos = [rho(s, r) >= thr for s, r in pos_core if not cal(r)]
    hold_neg = [rho(s, r) < thr for s, r in neg if not cal(r)]
    deg_pos = [abs(recs[s["name"]][r]["wind"]["degree"]) == 1 for s, r in pos_core if not cal(r)]
    deg_neg = [abs(recs[s["name"]][r]["wind"]["degree"]) != 1 for s, r in neg if not cal(r)]
    per = {}
    for s in specs:
        v = [recs[s["name"]][r]["wind"] for r in range(R) if recs[s["name"]][r]["wind"] is not None]
        if v:
            a = np.array([x["abs_rho"] for x in v])
            per[s["name"]] = dict(wrap=recs[s["name"]][0]["truth"]["wrap"], core=bool(s.get("core")),
                                  abs_rho_median=float(np.median(a)), abs_rho_min=float(a.min()),
                                  abs_rho_max=float(a.max()),
                                  frac_degree_pm1=float(np.mean([abs(x["degree"]) == 1 for x in v])),
                                  frac_pass_rho_thr=float(np.mean(a >= thr)))
    return dict(rho_threshold=thr, calib_minrate=float(sc.max()),
                heldout_tpr_rho=_rate(int(np.sum(hold_pos)), len(hold_pos)),
                heldout_tnr_rho=_rate(int(np.sum(hold_neg)), len(hold_neg)),
                heldout_tpr_degree=_rate(int(np.sum(deg_pos)), len(deg_pos)),
                heldout_tnr_degree=_rate(int(np.sum(deg_neg)), len(deg_neg)), per_control=per)


def analyze_confirm(crecs: dict, specs: list, topo: dict, rho_thr: float, R: int) -> dict:
    """Evaluate the CONFIRM_CANDIDATES (parameters as fitted on the calibration half) on fresh repeats."""
    names = [s["name"] for s in specs]
    b1 = np.array([crecs[nm][0]["truth"]["b1"] for nm in names])
    out = dict(repeats=f"{R}..{2 * R - 1} (fresh seeds, never used for fitting or selection)",
               candidates_fixed_before_run=[f"{c}:{f}" for c, f in CONFIRM_CANDIDATES], instruments={})
    for ck, fam in CONFIRM_CANDIDATES:
        if ck not in topo["configs"] or fam not in topo["configs"][ck]:
            out["instruments"][f"{ck}:{fam}"] = "not fitted in this run (e.g. --quick)"
            continue
        param = topo["configs"][ck][fam]["param"]
        section, key = ck.split(":")
        if section == "main":
            top = np.array([[crecs[nm][j]["main"][key]["top"] for j in range(R)] for nm in names], float)
            dec = rule_counts(top, fam, param)
        else:
            top = np.array([[crecs[nm][j]["stab"][key] for j in range(R)] for nm in names], float)
            dec = stable_decision(rule_counts(top, fam, param[1]), param[0])
        per, tot = {}, dict(pos=[0, 0], neg=[0, 0], fp1=[0, 0])
        for c, nm in enumerate(names):
            k = int((dec[c] == b1[c]).sum())
            per[nm] = dict(correct=_rate(k, R), counts={str(v): int((dec[c] == v).sum()) for v in np.unique(dec[c])})
            grp = "pos" if b1[c] >= 1 else "neg"
            tot[grp][0] += k
            tot[grp][1] += R
            if b1[c] == 0:
                tot["fp1"][0] += int((dec[c] == 1).sum())
                tot["fp1"][1] += R
        out["instruments"][f"{ck}:{fam}"] = dict(param=param, tpr_core=_rate(*tot["pos"]),
                                                 tnr_core=_rate(*tot["neg"]),
                                                 fpr_claims_b1_eq_1=_rate(*tot["fp1"]), per_control=per)
    wpos = [crecs[nm][j]["wind"]["abs_rho"] >= rho_thr for nm in names for j in range(R)
            if crecs[nm][j]["wind"] is not None and crecs[nm][j]["truth"]["wrap"] == 1]
    wneg = [crecs[nm][j]["wind"]["abs_rho"] < rho_thr for nm in names for j in range(R)
            if crecs[nm][j]["wind"] is not None and crecs[nm][j]["truth"]["wrap"] == 0]
    out["winding"] = dict(rho_threshold=rho_thr, tpr=_rate(int(np.sum(wpos)), len(wpos)),
                          tnr=_rate(int(np.sum(wneg)), len(wneg)))
    # sampling voids around the stable equilibrium vs the largest bar of the libration-only (disk) controls
    voids = {}
    for nm in names:
        if not nm.startswith("disk"):
            continue
        voids[nm] = [dict(r=crecs[nm][j]["r"], void_radius=crecs[nm][j]["void_radius"],
                          **{f"top_pi_{k}": float(crecs[nm][j]["main"][k]["top"][0]) for k in crecs[nm][j]["main"]})
                     for j in range(R)]
    out["disk_voids"] = voids
    # the same for every core b1 = 0 control built from trajectory states (the torn chart and the fold share the
    # states, hence the void, of the full annulus): correlation of the void radius with the largest bar, and the
    # repeat with the largest void
    vc = {}
    for nm in names:
        recs = crecs[nm]
        if recs[0]["truth"]["b1"] != 0 or recs[0]["void_radius"] is None:
            continue
        vr = np.array([x["void_radius"] for x in recs])
        jmax = int(np.argmax(vr))
        row = dict(max_void=dict(r=int(recs[jmax]["r"]), void_radius=float(vr[jmax])))
        for k in recs[0]["main"]:
            top0 = np.array([float(x["main"][k]["top"][0]) for x in recs])
            row[k] = dict(corr_void_top_pi=float(np.corrcoef(vr, top0)[0, 1]), top_pi_max=float(top0.max()),
                          top_pi_at_max_void=float(top0[jmax]))
        vc[nm] = row
    out["void_vs_top_bar"] = vc
    return out


# ----------------------------------------------------------------------------------------------- energy analysis
def analyze_energy(erecs: dict, especs: list, R: int, cfg: dict) -> dict:
    names = [s["name"] for s in especs]
    cal, hold = np.arange(R) < R // 2, np.arange(R) >= R // 2

    def get(nm, r, n, key):
        sc = erecs[nm][r]["scores"]
        return sc["krr"][key] if key == "krr_r2" else sc[str(n)][key]

    dec_keys = ["linear_r2", "quad_r2", *[f"knn{k}_r2" for k in cfg["energy_ks"]],
                *[f"white_knn{k}_r2" for k in cfg["energy_ks"]]]
    smooth_keys = [*[f"s_k{k}" for k in cfg["energy_ks"]], *[f"white_s_k{k}" for k in cfg["energy_ks"]]]
    tail_keys = [*[f"lip_tail_k{k}" for k in cfg["energy_ks"]], *[f"white_lip_tail_k{k}" for k in cfg["energy_ks"]]]
    out = dict(scores={}, distributions={})
    lab = {s["name"]: s for s in especs}

    def evaluate(key, n, higher=True, kind="dec"):
        vals = np.array([[get(nm, r, n, key) for r in range(R)] for nm in names], dtype=float)
        labels = np.array([lab[nm][kind] if lab[nm][kind] is not None else -1 for nm in names])
        pos = labels == 1
        q = 5 if higher else 95
        thr = float(np.percentile(vals[pos][:, cal].ravel(), q))
        passed = vals >= thr if higher else vals <= thr
        neg = labels == 0
        cal_fpr = float(passed[neg][:, cal].mean()) if neg.any() else np.nan
        res = dict(threshold=thr, higher_is_better=higher, calib_fpr=cal_fpr,
                   heldout_tpr=_rate(int(passed[pos][:, hold].sum()), int(pos.sum() * hold.sum())),
                   heldout_fpr=_rate(int(passed[neg][:, hold].sum()), int(neg.sum() * hold.sum())),
                   per_control_pass_heldout={nm: float(passed[i, hold].mean()) for i, nm in enumerate(names)})
        return res, vals

    for n in cfg["energy_ns"]:
        for key in dec_keys:
            res, vals = evaluate(key, n, True, "dec")
            out["scores"][f"{key}@{n}"] = res
            out["distributions"][f"{key}@{n}"] = {nm: [float(np.median(vals[i])), float(np.percentile(vals[i], 5)),
                                                       float(np.percentile(vals[i], 95))] for i, nm in enumerate(names)}
        for key in smooth_keys:
            res, vals = evaluate(key, n, True, "smooth")
            out["scores"][f"{key}@{n}"] = res
            out["distributions"][f"{key}@{n}"] = {nm: [float(np.median(vals[i])), float(np.percentile(vals[i], 5)),
                                                       float(np.percentile(vals[i], 95))] for i, nm in enumerate(names)}
        for key in tail_keys:
            res, vals = evaluate(key, n, False, "smooth")
            out["scores"][f"{key}@{n}"] = res
            out["distributions"][f"{key}@{n}"] = {nm: [float(np.median(vals[i])), float(np.percentile(vals[i], 5)),
                                                       float(np.percentile(vals[i], 95))] for i, nm in enumerate(names)}
        # combined smoothness rule: s >= thr_s AND tail <= thr_tail (each threshold fitted separately)
        for k in cfg["energy_ks"]:
            for pre in ("", "white_"):
                rs, vs = evaluate(f"{pre}s_k{k}", n, True, "smooth")
                rt, vt = evaluate(f"{pre}lip_tail_k{k}", n, False, "smooth")
                passed = (vs >= rs["threshold"]) & (vt <= rt["threshold"])
                labels = np.array([lab[nm]["smooth"] if lab[nm]["smooth"] is not None else -1 for nm in names])
                pos, neg = labels == 1, labels == 0
                out["scores"][f"{pre}s_and_tail_k{k}@{n}"] = dict(
                    thresholds=[rs["threshold"], rt["threshold"]], calib_fpr=float(passed[neg][:, cal].mean()),
                    calib_tpr=float(passed[pos][:, cal].mean()),
                    heldout_tpr=_rate(int(passed[pos][:, hold].sum()), int(pos.sum() * hold.sum())),
                    heldout_fpr=_rate(int(passed[neg][:, hold].sum()), int(neg.sum() * hold.sum())),
                    per_control_pass_heldout={nm: float(passed[i, hold].mean()) for i, nm in enumerate(names)})
    res, vals = evaluate("krr_r2", KRR_N, True, "dec")
    out["scores"][f"krr_r2@{KRR_N}"] = res
    out["distributions"][f"krr_r2@{KRR_N}"] = {nm: [float(np.median(vals[i])), float(np.percentile(vals[i], 5)),
                                                   float(np.percentile(vals[i], 95))] for i, nm in enumerate(names)}
    out["krr_selected_gamma"] = {nm: float(np.median([erecs[nm][r]["scores"]["krr"]["krr_gamma"] for r in range(R)]))
                                 for nm in names}
    out["labels"] = {nm: dict(dec=lab[nm]["dec"], smooth=lab[nm]["smooth"]) for nm in names}
    return out


# ----------------------------------------------------------------------------------------------- seam / gate analysis
SEAM_GRID = np.round(10.0 ** (np.arange(0, 201) / 100.0), 4)          # thresholds 1 .. 100, log-spaced


def _q(v, q):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, q)) if v.size else None


def _interp_cross(x, y, level):
    """First x at which y (ordered along x) crosses ``level``, by linear interpolation in log(y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for k in range(len(x) - 1):
        a, b = y[k], y[k + 1]
        if np.isfinite(a) and np.isfinite(b) and (a - level) * (b - level) <= 0 and a != b:
            la, lb, ll = np.log(a), np.log(b), np.log(level)
            return float(x[k] + (x[k + 1] - x[k]) * (ll - la) / (lb - la))
    return None


def analyze_seam(srecs: list, R: int, cfg: dict) -> dict:
    """Seam ratio: separation glued / torn, threshold fit (calibration half) and held-out error rates, stress,
    partial gluing vs the H1 rule, internal placebo null, bootstrap honesty. Theta gate: distributions, the
    5th-percentile threshold of the theta-injective controls and held-out error rates at it and at 0.99."""
    cal, hold = np.arange(R) < R // 2, np.arange(R) >= R // 2
    specs = cfg["seam_specs"]
    names = [s["name"] for s in specs]
    ratio = {nm: np.array([srecs[j]["seam"][nm]["ratio"] for j in range(R)], dtype=float) for nm in names}
    spec = {s["name"]: s for s in specs}
    controls = {}
    for nm in names:
        recs = [srecs[j]["seam"][nm] for j in range(R)]
        v = ratio[nm]
        row = dict(group=spec[nm]["group"], core=spec[nm]["core"], desc=spec[nm].get("desc", ""),
                   n_ok=int(sum(x["status"] == "ok" for x in recs)), median=_q(v, 0.5), min=_q(v, 0.0),
                   q05=_q(v, 0.05), q95=_q(v, 0.95), max=_q(v, 1.0),
                   median_ci95=[_q([x["ci95"][0] for x in recs], 0.5), _q([x["ci95"][1] for x in recs], 0.5)],
                   median_boot_log_sd=_q([x["log_sd"] for x in recs], 0.5),
                   across_repeat_log_sd=float(np.std(np.log(v[np.isfinite(v)]), ddof=1)) if np.isfinite(v).sum() > 2
                   else None,
                   median_cut_pairs=_q([x["n_cut_pairs"] for x in recs], 0.5),
                   median_cut_points=_q([x["n_cut_points"] for x in recs], 0.5),
                   placebo_max_median=_q([x["placebo_max"] for x in recs], 0.5),
                   pilot_formula_median=_q([x["pilot"] for x in recs], 0.5),
                   fast_median=_q([x["ratio_fast"] for x in recs], 0.5),
                   fast_min=_q([x["ratio_fast"] for x in recs], 0.0), fast_max=_q([x["ratio_fast"] for x in recs], 1.0),
                   pilot_formula_min=_q([x["pilot"] for x in recs], 0.0), pilot_formula_max=_q([x["pilot"] for x in recs], 1.0),
                   placebo_max_max=_q([x["placebo_max"] for x in recs], 1.0),
                   frac_gt_prereg_threshold=float(np.mean(v > SEAM_THRESHOLD_PREREG)) if np.isfinite(v).all() else None)
        if spec[nm].get("ph"):
            row.update(ph_frac_b1_eq_1=float(np.mean([x["ph_b1"] == 1 for x in recs])),
                       ph_top_pi_median=_q([x["ph_top"][0] for x in recs], 0.5))
        if spec[nm]["map"][0] == "arc":
            row["gap"] = C.arc_gap(spec[nm]["map"][1]["alpha"])
        if spec[nm].get("profile"):
            keys = sorted({k for x in recs for k in x["by_omega"]}, key=lambda s: float(s.split("..")[0]))
            row["by_omega_median"] = {k: _q([x["by_omega"][k]["ratio"] for x in recs if k in x["by_omega"]], 0.5)
                                      for k in keys}
        controls[nm] = row
    out = dict(settings=dict(seed_base=SEAM_BASE, repeats=R, **SEAM_KW, n_boot=cfg["seam_nboot"], fast_omega_abs_min=SEAM_FAST,
                             control_angles_deg=[-135, -90, -45, 0, 45, 90, 135], bootstrap="trajectories (box: points)",
                             ph_rule=dict(n=SEAM_PH_N, tau=SEAM_PH_TAU, subsample="fps")),
               controls=controls)
    # ---- threshold fit on the calibration half of the core controls
    core_g = [nm for nm in names if spec[nm]["core"] and spec[nm]["group"] == "glued"]
    core_t = [nm for nm in names if spec[nm]["core"] and spec[nm]["group"] == "torn"]
    if core_g and core_t:
        G_cal = np.concatenate([ratio[nm][cal] for nm in core_g])
        T_cal = np.concatenate([ratio[nm][cal] for nm in core_t])
        obj = np.array([min(np.mean(G_cal <= t), np.mean(T_cal > t)) for t in SEAM_GRID])
        best = np.flatnonzero(obj >= obj.max() - 1e-12)
        lo, hi = float(SEAM_GRID[best].min()), float(SEAM_GRID[best].max())
        thr = float(SEAM_GRID[best][np.argmin(np.abs(np.log(SEAM_GRID[best] / np.sqrt(lo * hi))))])

        def rates(t, sel):
            Gv = np.concatenate([ratio[nm][sel] for nm in core_g])
            Tv = np.concatenate([ratio[nm][sel] for nm in core_t])
            return dict(threshold=t, tpr_torn=_rate(int((Tv > t).sum()), Tv.size),
                        tnr_glued=_rate(int((Gv <= t).sum()), Gv.size))

        G_all = np.concatenate([ratio[nm] for nm in core_g])
        T_all = np.concatenate([ratio[nm] for nm in core_t])
        out["fit"] = dict(rule="maximise min(TNR glued, TPR torn) on the calibration half of the core controls; "
                               "geometric centre of the optimal plateau (grid 10^(k/100))",
                          objective=float(obj.max()), plateau=[lo, hi], threshold=thr,
                          glued_core_max_calib=float(G_cal.max()), torn_core_min_calib=float(T_cal.min()),
                          glued_core_max_all=float(G_all.max()), torn_core_min_all=float(T_all.min()),
                          log10_margin_prereg_to_glued=float(np.log10(SEAM_THRESHOLD_PREREG / G_all.max())),
                          log10_margin_prereg_to_torn=float(np.log10(T_all.min() / SEAM_THRESHOLD_PREREG)))
        out["heldout"] = dict(fitted=rates(thr, hold), prereg=rates(SEAM_THRESHOLD_PREREG, hold))
        out["all_repeats_prereg"] = rates(SEAM_THRESHOLD_PREREG, np.ones(R, bool))
        fast = {nm: np.array([srecs[j]["seam"][nm]["ratio_fast"] for j in range(R)], float) for nm in names}
        out["fast_only"] = dict(omega_abs_min=SEAM_FAST,
                                glued_core_max=float(np.nanmax(np.concatenate([fast[nm] for nm in core_g]))),
                                torn_core_min=float(np.nanmin(np.concatenate([fast[nm] for nm in core_t]))),
                                tnr_glued_at_prereg=_rate(int(sum((fast[nm] <= SEAM_THRESHOLD_PREREG).sum()
                                                                  for nm in core_g)), len(core_g) * R),
                                tpr_torn_at_prereg=_rate(int(sum((fast[nm] > SEAM_THRESHOLD_PREREG).sum()
                                                                 for nm in core_t)), len(core_t) * R))
        stress = {}
        for nm in names:
            if spec[nm]["core"] or spec[nm]["group"] not in ("glued", "torn"):
                continue
            v = ratio[nm]
            want_torn = spec[nm]["group"] == "torn"
            stress[nm] = dict(group=spec[nm]["group"],
                              correct_at_fitted=float(np.mean((v > thr) == want_torn)),
                              correct_at_prereg=float(np.mean((v > SEAM_THRESHOLD_PREREG) == want_torn)))
        out["stress"] = stress
        # internal null: placebo cuts at the control angles of the core controls
        pm = np.array([srecs[j]["seam"][nm]["placebo_max"] for nm in core_g + core_t for j in range(R)], float)
        out["placebo_core"] = dict(median=_q(pm, 0.5), q99=_q(pm, 0.99), max=_q(pm, 1.0), n=int(np.isfinite(pm).sum()))
        # bootstrap honesty: bootstrap log-sd vs the spread over independent repeats; coverage of the 20-repeat median.
        # Only controls with a fixed map: the random R^8 embeddings are redrawn every repeat, a variation that a
        # bootstrap conditional on the latent cannot (and should not) see.
        cov, sdr = [], {}
        fixed = [nm for nm in core_g + core_t if spec[nm]["map"][0] not in ("emb8", "torn_emb8")]
        for nm in fixed:
            med = np.nanmedian(ratio[nm])
            recs = [srecs[j]["seam"][nm] for j in range(R)]
            cov += [x["ci95"][0] <= med <= x["ci95"][1] for x in recs if x["ci95"][0] is not None
                    and np.isfinite(x["ci95"][0])]
            c = controls[nm]
            if c["across_repeat_log_sd"] and c["median_boot_log_sd"]:
                sdr[nm] = c["median_boot_log_sd"] / c["across_repeat_log_sd"]
        out["bootstrap_check"] = dict(controls=fixed, coverage_of_repeat_median=_rate(int(np.sum(cov)), len(cov)),
                                      boot_sd_over_repeat_sd=dict(median=_q(list(sdr.values()), 0.5),
                                                                  min=_q(list(sdr.values()), 0.0),
                                                                  max=_q(list(sdr.values()), 1.0)),
                                      per_control=sdr)
    # ---- partial gluing: SR and the H1 rule along the arc family
    arcs = [nm for nm in names if spec[nm]["map"][0] == "arc"]
    if arcs:
        arcs.sort(key=lambda nm: -spec[nm]["map"][1]["alpha"])
        gaps = [C.arc_gap(spec[nm]["map"][1]["alpha"]) for nm in arcs]
        med = [controls[nm]["median"] for nm in arcs]
        ph1 = [controls[nm]["ph_frac_b1_eq_1"] for nm in arcs]
        cross = dict(seam_prereg=_interp_cross(gaps, med, SEAM_THRESHOLD_PREREG))
        if "fit" in out:
            cross["seam_fitted"] = _interp_cross(gaps, med, out["fit"]["threshold"])
        k = next((i for i in range(len(ph1) - 1) if ph1[i] >= 0.5 > ph1[i + 1]), None)
        cross["ph_half"] = (float(gaps[k] + (gaps[k + 1] - gaps[k]) * (ph1[k] - 0.5) / (ph1[k] - ph1[k + 1]))
                            if k is not None else None)
        medf = [controls[nm]["fast_median"] for nm in arcs]
        cross["seam_prereg_fast_only"] = _interp_cross(gaps, medf, SEAM_THRESHOLD_PREREG)
        out["arc"] = dict(alphas=[spec[nm]["map"][1]["alpha"] for nm in arcs], gaps=gaps, seam_median=med,
                          seam_fast_median=medf, ph_frac_b1_eq_1=ph1, gap_at_crossing=cross)
    nonec = [nm for nm in names if spec[nm]["group"] == "none"]
    out["no_cut_data"] = {nm: dict(insufficient=int(sum(srecs[j]["seam"][nm]["status"] != "ok" for j in range(R))),
                                   n=R) for nm in nonec}
    out["seconds_per_cloud"] = dict(median=_q([srecs[j]["seam"][nm]["sec"] for nm in names for j in range(R)], 0.5),
                                    max=_q([srecs[j]["seam"][nm]["sec"] for nm in names for j in range(R)], 1.0))
    # ---- theta gate
    gspecs = cfg["gate_specs"]
    gnames = [s["name"] for s in gspecs]
    glab = {s["name"]: s for s in gspecs}
    gate = dict(settings=dict(k=GATE_K, n=GATE_N, n_splits=5, statistic="min(R2 cos theta, R2 sin theta), kNN"),
                controls={})
    v = {nm: np.array([srecs[j]["gate"][nm]["r2_min"] for j in range(R)]) for nm in gnames}
    for nm in gnames:
        gate["controls"][nm] = dict(label=glab[nm]["label"], core=glab[nm]["core"], desc=glab[nm].get("desc", ""),
                                    r2_min_median=_q(v[nm], 0.5), r2_min_min=_q(v[nm], 0.0), r2_min_max=_q(v[nm], 1.0),
                                    r2_cos_median=_q([srecs[j]["gate"][nm]["r2_cos"] for j in range(R)], 0.5),
                                    r2_sin_median=_q([srecs[j]["gate"][nm]["r2_sin"] for j in range(R)], 0.5),
                                    pass_prereg=float(np.mean(v[nm] >= GATE_THRESHOLD_PREREG)))
    pos = [nm for nm in gnames if glab[nm]["label"] == 1 and glab[nm]["core"]]
    neg = [nm for nm in gnames if glab[nm]["label"] == 0 and glab[nm]["core"]]
    if pos and neg:
        thr_g = float(np.percentile(np.concatenate([v[nm][cal] for nm in pos]), 5))

        def grates(t, sel):
            P = np.concatenate([v[nm][sel] for nm in pos])
            Nn = np.concatenate([v[nm][sel] for nm in neg])
            return dict(threshold=t, tpr_injective=_rate(int((P >= t).sum()), P.size),
                        fpr_folded=_rate(int((Nn >= t).sum()), Nn.size))

        gate["fit"] = dict(rule="5th percentile of the theta-injective core controls, calibration half",
                           threshold=thr_g,
                           injective_core_min_all=float(min(v[nm].min() for nm in pos)),
                           folded_core_max_all=float(max(v[nm].max() for nm in neg)))
        gate["heldout"] = dict(fitted=grates(thr_g, hold), prereg=grates(GATE_THRESHOLD_PREREG, hold))
    out["theta_gate"] = gate
    return out


# ----------------------------------------------------------------------------------------------- review analysis
def _repeat_level(ok: np.ndarray) -> dict:
    """ok [C, R] (bool): a repeat counts as correct only if every control in it is correct. Conservative interval
    for the pooled rate when the controls of one repeat share states (worst case: they fail together)."""
    ok = np.asarray(ok, bool)
    return _rate(int(ok.all(0).sum()), ok.shape[1])


def _p_at_least(k: int, n: int, p: float) -> float:
    return float(stats.binom.sf(k - 1, n, p))


def analyze_review(rrecs: list, R: int, cfg: dict, stored: dict) -> dict:
    """Part E. Seam: confirmation of the frozen thresholds on fresh seeds, omega aspect, fold-index rule (fit on
    r < R/2, held out r >= R/2), adversarial latents. Theta gate: the three declared rules on fresh seeds and the
    partial-fold dose-response. From the stored results (no recomputation): the joint energy rule of prereg 3.4
    and repeat-level (clustered) intervals for the part-D rates."""
    cal, hold = np.arange(R) < R // 2, np.arange(R) >= R // 2
    allr = np.ones(R, bool)
    specs = cfg["review_seam_specs"]
    spec = {s["name"]: s for s in specs}
    names = [s["name"] for s in specs]
    get = lambda nm, key: np.array([np.nan if rrecs[j]["seam"][nm][key] is None else rrecs[j]["seam"][nm][key]
                                    for j in range(R)], dtype=float)
    ratio = {nm: get(nm, "ratio") for nm in names}
    kappa = {nm: get(nm, "omega_aspect") for nm in names}
    fold = {nm: get(nm, "fold_index") for nm in names}
    fast = {nm: get(nm, "ratio_fast") for nm in names}
    core_g = [nm for nm in names if spec[nm]["core"] and spec[nm]["group"] == "glued"]
    core_t = [nm for nm in names if spec[nm]["core"] and spec[nm]["group"] == "torn"]
    out = dict(settings=dict(seed_base=REVIEW_BASE, repeats=R, seam_thresholds=REVIEW_SEAM_THRESHOLDS,
                             omega_aspect_max=SE.OMEGA_ASPECT_MAX, fold_index_min_code=SE.FOLD_INDEX_MIN,
                             gate_rules=REVIEW_GATE_RULES, partial_fold_q=REVIEW_PARTIAL_FOLD_Q,
                             n_boot_core=cfg["seam_nboot"], **SEAM_KW))
    # ---- seam: frozen thresholds on fresh seeds (core controls)
    conf = {}
    for t in REVIEW_SEAM_THRESHOLDS:
        okg = np.array([ratio[nm] <= t for nm in core_g])
        okt = np.array([ratio[nm] > t for nm in core_t])
        conf[f"{t:g}"] = dict(tnr_glued=_rate(int(okg.sum()), okg.size), tpr_torn=_rate(int(okt.sum()), okt.size),
                              repeat_level_glued=_repeat_level(okg), repeat_level_torn=_repeat_level(okt))
    straddle = [rrecs[j]["seam"][nm]["ci95"][0] is not None and rrecs[j]["seam"][nm]["ci95"][0] <= 5.0
                <= rrecs[j]["seam"][nm]["ci95"][1] for nm in core_g + core_t for j in range(R)]
    okgf = np.array([fast[nm] <= 5.0 for nm in core_g])
    oktf = np.array([fast[nm] > 5.0 for nm in core_t])
    grp_counts = [rrecs[j]["seam"][nm]["n_cut_groups"] for nm in core_g + core_t for j in range(R)
                  if rrecs[j]["seam"][nm]["n_cut_groups"] is not None]
    out["seam_confirm"] = dict(
        thresholds=conf, glued_core_max=float(np.nanmax([ratio[nm] for nm in core_g])),
        torn_core_min=float(np.nanmin([ratio[nm] for nm in core_t])),
        ci_straddles_5=_rate(int(np.sum(straddle)), len(straddle)),
        fast_only_at_5=dict(tnr_glued=_rate(int(okgf.sum()), okgf.size), tpr_torn=_rate(int(oktf.sum()), oktf.size)),
        n_cut_groups_min=int(min(grp_counts)) if grp_counts else None,
        n_cut_groups_median=float(np.median(grp_counts)) if grp_counts else None)
    # ---- fold-index rule: folded if fold_index < t. Fit on the calibration half (core = not folded vs folds).
    folds = [nm for nm in names if spec[nm]["fold"]]
    core = core_g + core_t
    stress_nf = [nm for nm in names if not spec[nm]["core"] and not spec[nm]["fold"]]
    Fc = np.concatenate([fold[nm][cal] for nm in folds])
    Nc = np.concatenate([fold[nm][cal] for nm in core])
    obj = np.array([min(np.mean(Fc < t), np.mean(Nc >= t)) for t in REVIEW_FOLD_GRID])
    best = np.flatnonzero(obj >= obj.max() - 1e-12)
    lo, hi = float(REVIEW_FOLD_GRID[best].min()), float(REVIEW_FOLD_GRID[best].max())
    thr = float(REVIEW_FOLD_GRID[best][np.argmin(np.abs(np.log(REVIEW_FOLD_GRID[best] / np.sqrt(lo * hi))))])

    def fold_rates(t, sel):
        F = np.array([fold[nm][sel] < t for nm in folds])
        N = np.array([fold[nm][sel] >= t for nm in core])
        Sn = np.array([fold[nm][sel] >= t for nm in stress_nf])
        return dict(threshold=t, tpr_folds=_rate(int(F.sum()), F.size), tnr_core=_rate(int(N.sum()), N.size),
                    tnr_stress=_rate(int(Sn.sum()), Sn.size), repeat_level_core=_repeat_level(N))

    out["fold_rule"] = dict(rule="folded if fold_index < t; maximise min(TPR folds, TNR core) on r < R/2; "
                                 "geometric centre of the optimal plateau",
                            objective=float(obj.max()), plateau=[lo, hi], threshold=thr,
                            folds_max_all=float(np.nanmax([fold[nm] for nm in folds])),
                            core_min_all=float(np.nanmin([fold[nm] for nm in core])),
                            stress_min_all=float(np.nanmin([fold[nm] for nm in stress_nf])),
                            heldout_fitted=fold_rates(thr, hold), heldout_code=fold_rates(SE.FOLD_INDEX_MIN, hold),
                            all_repeats_code=fold_rates(SE.FOLD_INDEX_MIN, allr))
    # POST HOC (added after the run above): the declared rule counts the second-order fold (cos theta, omega)
    # as a target, which the fold index cannot see by construction. The same fit restricted to the first-order
    # folds, reported only to show where SE.FOLD_INDEX_MIN (fixed before the run) sits in that plateau.
    first = [nm for nm in folds if spec[nm]["map"][0] != "folded"]
    F1 = np.concatenate([fold[nm][cal] for nm in first])
    obj1 = np.array([min(np.mean(F1 < t), np.mean(Nc >= t)) for t in REVIEW_FOLD_GRID])
    best1 = np.flatnonzero(obj1 >= obj1.max() - 1e-12)
    lo1, hi1 = float(REVIEW_FOLD_GRID[best1].min()), float(REVIEW_FOLD_GRID[best1].max())
    out["fold_rule"]["post_hoc_first_order_only"] = dict(
        folds=first, objective=float(obj1.max()), plateau=[lo1, hi1],
        centre=float(REVIEW_FOLD_GRID[best1][np.argmin(np.abs(np.log(REVIEW_FOLD_GRID[best1] / np.sqrt(lo1 * hi1))))]),
        first_order_max_all=float(np.nanmax([fold[nm] for nm in first])))
    # ---- omega aspect
    out["omega_aspect"] = dict(core_max=float(np.nanmax([kappa[nm] for nm in core])),
                               core_min=float(np.nanmin([kappa[nm] for nm in core])),
                               core_flagged=_rate(int(sum((kappa[nm] > SE.OMEGA_ASPECT_MAX).sum() for nm in core)),
                                                  len(core) * R))
    # ---- per control
    ctr = {}
    for nm in names:
        flags = [SE.seam_flags(dict(ratio=ratio[nm][j], status=rrecs[j]["seam"][nm]["status"],
                                    ci95=rrecs[j]["seam"][nm]["ci95"], omega_aspect=kappa[nm][j],
                                    fold_index=fold[nm][j])) for j in range(R)]
        ctr[nm] = dict(group=spec[nm]["group"], core=spec[nm]["core"], fold=spec[nm]["fold"],
                       desc=spec[nm].get("desc", ""), sr_median=_q(ratio[nm], 0.5), sr_min=_q(ratio[nm], 0.0),
                       sr_max=_q(ratio[nm], 1.0), frac_torn_at_5=float(np.mean(ratio[nm] > 5.0)),
                       fast_median=_q(fast[nm], 0.5),
                       kappa_median=_q(kappa[nm], 0.5), kappa_min=_q(kappa[nm], 0.0), kappa_max=_q(kappa[nm], 1.0),
                       fold_median=_q(fold[nm], 0.5), fold_min=_q(fold[nm], 0.0), fold_max=_q(fold[nm], 1.0),
                       fold_ctrl_min_median=_q(get(nm, "fold_index_ctrl_min"), 0.5),
                       placebo_max_median=_q(get(nm, "placebo_max"), 0.5),
                       frac_omega_dominated=float(np.mean([f["omega_dominated"] for f in flags])),
                       frac_folded_at_cut=float(np.mean([f["folded_at_cut"] for f in flags])),
                       frac_in_range=float(np.mean([f["in_range"] for f in flags])))
    out["seam_controls"] = ctr
    # ---- theta gate on fresh seeds
    gspecs = cfg["review_gate_specs"]
    glab = {s["name"]: s for s in gspecs}
    gn = [s["name"] for s in gspecs]
    pos = [nm for nm in gn if glab[nm]["label"] == 1 and glab[nm]["core"]]
    neg = [nm for nm in gn if glab[nm]["label"] == 0 and glab[nm]["core"]]
    gv = {(nm, n): np.array([rrecs[j]["gate"][nm][f"r2_min_n{n}"] for j in range(R)], float)
          for nm in gn for n in sorted({n for _, n, _ in REVIEW_GATE_RULES})}
    rules = {}
    for rn, n, t in REVIEW_GATE_RULES:
        P = np.array([gv[(nm, n)] >= t for nm in pos])
        Nn = np.array([gv[(nm, n)] >= t for nm in neg])
        rules[rn] = dict(n=n, threshold=t, tpr_injective=_rate(int(P.sum()), P.size),
                         fpr_folded=_rate(int(Nn.sum()), Nn.size), repeat_level_injective=_repeat_level(P),
                         per_control_pass={nm: float(np.mean(gv[(nm, n)] >= t)) for nm in gn})
    out["theta_gate"] = dict(rules=rules, r2_min_median={nm: {f"n{n}": _q(gv[(nm, n)], 0.5) for (m, n) in gv if m == nm}
                                                         for nm in gn},
                             r2_min_min={nm: {f"n{n}": _q(gv[(nm, n)], 0.0) for (m, n) in gv if m == nm} for nm in gn})
    # ---- joint energy rule (prereg 3.4: kNN R^2 AND quadratic R^2 for "decodable"; plus the tail ratio for
    # "smooth"), from the stored raw energy records and the stored thresholds; n = 2000, k = 10, held-out half
    need = ("knn10_r2@2000", "quad_r2@2000", "lip_tail_k10@2000")
    if stored.get("raw_energy") and stored.get("energy") and all(k in stored["energy"]["scores"] for k in need):
        re_, sc, lab = stored["raw_energy"], stored["energy"]["scores"], stored["energy"]["labels"]
        tk, tq, tt = (sc["knn10_r2@2000"]["threshold"], sc["quad_r2@2000"]["threshold"],
                      sc["lip_tail_k10@2000"]["threshold"])
        Re = len(next(iter(re_.values())))
        hold_e = range(Re // 2, Re)

        def ok(nm, r, rule):
            x = re_[nm][r]["2000"]
            dec = x["knn10_r2"] >= tk and x["quad_r2"] >= tq
            return dec and x["lip_tail_k10"] <= tt if rule == "dec_and_smooth" else dec

        ej = dict(thresholds=dict(knn10_r2=tk, quad_r2=tq, lip_tail_k10=tt), heldout={})
        for rule, key in (("decodable_knn_and_quad", "dec"), ("dec_and_smooth", "smooth")):
            P = [nm for nm in lab if lab[nm][key] == 1]
            Nn = [nm for nm in lab if lab[nm][key] == 0]
            kp = sum(ok(nm, r, rule) for nm in P for r in hold_e)
            kn = sum(ok(nm, r, rule) for nm in Nn for r in hold_e)
            ej["heldout"][rule] = dict(tpr=_rate(kp, len(P) * len(hold_e)), fpr=_rate(kn, len(Nn) * len(hold_e)),
                                       per_control_pass={nm: float(np.mean([ok(nm, r, rule) for r in hold_e]))
                                                         for nm in P})
        p_joint = ej["heldout"]["decodable_knn_and_quad"]["tpr"]["rate"]
        ej["p_prediction_met_if_all_runs_decodable"] = {
            f"p_pass={p:.2f}": {"2_of_3": _p_at_least(2, 3, p), "11_of_12": _p_at_least(11, 12, p),
                                "6_of_6": _p_at_least(6, 6, p)} for p in sorted({p_joint, 0.93, 0.98})}
        out["energy_joint"] = ej
    # ---- repeat-level (clustered) intervals for the part-D held-out rates, from the stored raw seam records
    rs = stored.get("raw_seam")
    if rs:
        Rs = len(rs)
        hs = [j for j in range(Rs) if j >= Rs // 2]
        sspec = {s["name"]: s for s in SEAM_CONTROLS}
        have_s, have_g = set(rs[0]["seam"]), set(rs[0]["gate"])          # (--quick runs hold a subset)
        cg = [nm for nm in sspec if sspec[nm]["core"] and sspec[nm]["group"] == "glued" and nm in have_s]
        ct = [nm for nm in sspec if sspec[nm]["core"] and sspec[nm]["group"] == "torn" and nm in have_s]
        fv = lambda x: np.nan if x is None else float(x)
        okg = np.array([[fv(rs[j]["seam"][nm]["ratio"]) <= 5.0 for j in hs] for nm in cg])
        okt = np.array([[fv(rs[j]["seam"][nm]["ratio"]) > 5.0 for j in hs] for nm in ct])
        gpos = [s["name"] for s in GATE_CONTROLS if s["label"] == 1 and s["core"] and s["name"] in have_g]
        gneg = [s["name"] for s in GATE_CONTROLS if s["label"] == 0 and s["core"] and s["name"] in have_g]
        okp = np.array([[fv(rs[j]["gate"][nm]["r2_min"]) >= GATE_THRESHOLD_PREREG for j in hs] for nm in gpos])
        okn = np.array([[fv(rs[j]["gate"][nm]["r2_min"]) < GATE_THRESHOLD_PREREG for j in hs] for nm in gneg])
        out["part_d_repeat_level_heldout"] = dict(
            seam_glued_at_5=dict(pooled=_rate(int(okg.sum()), okg.size), repeat_level=_repeat_level(okg)),
            seam_torn_at_5=dict(pooled=_rate(int(okt.sum()), okt.size), repeat_level=_repeat_level(okt)),
            gate_injective_pass_at_0_99=dict(pooled=_rate(int(okp.sum()), okp.size), repeat_level=_repeat_level(okp)),
            gate_folded_reject_at_0_99=dict(pooled=_rate(int(okn.sum()), okn.size), repeat_level=_repeat_level(okn)))
    return out


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--parts", default="topo,energy,confirm,seam",
                    help="any of topo, energy, confirm, seam, review; confirm / seam / review without topo and "
                         "energy update only their sections of the stored --out (confirm re-uses the fits stored "
                         "there, review reads the stored energy and part-D records)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 repeats, few controls, no whitening/stability")
    ap.add_argument("--out", default=None, help=f"output JSON (default {OUT_DEFAULT.relative_to(ROOT)}; "
                                                "--quick writes nothing unless --out is given)")
    ap.add_argument("--cache", default=None, help="pickle the raw per-task records here (or read them with "
                                                  "--reanalyze)")
    ap.add_argument("--reanalyze", action="store_true", help="skip the computation, re-run the analysis on --cache")
    args = ap.parse_args(argv)
    if args.reanalyze and not args.cache:
        raise SystemExit("--reanalyze needs --cache")
    parts = set(args.parts.split(","))
    R = 2 if args.quick else args.repeats
    if R % 2:
        raise SystemExit("--repeats must be even (calibration / held-out halves)")
    cfg = dict(ns=NS, white=True, white_ns=WHITE_NS, stab=True, stab_plan=STAB_PLAN, stab_r=STAB_R, wind=True,
               energy_ns=ENERGY_NS, energy_ks=ENERGY_KS, confirm_ns=CONFIRM_NS, confirm_stab=CONFIRM_STAB)
    tspecs, especs = TOPO_CONTROLS, ENERGY_CONTROLS
    if args.quick:
        keep_t = {"ref_s1", "ring_ref", "rot2_ref", "torn_r2", "disk_ref", "blob8", "thin_1/5", "emb8_medium"}
        tspecs = [s for s in TOPO_CONTROLS if s["name"] in keep_t]
        especs = [s for s in ENERGY_CONTROLS if s["name"] in {"ref", "sin10", "patch8", "noise5", "emb8_medium",
                                                               "perm"}]
        cfg.update(ns=(300, 600), white_ns=(300,), stab_plan=(("random", 300), ("fps_half", 300)), stab_r=3,
                   energy_ns=(1000,), energy_ks=(10,), confirm_ns=(300, 600), confirm_stab=(("fps_half", 300),))
        if "confirm" in parts and "topo" not in parts:
            raise SystemExit("--quick --parts confirm is not supported (needs the quick topology fits)")
    cfg.update(seam_specs=SEAM_CONTROLS, gate_specs=GATE_CONTROLS, seam_nboot=SEAM_NBOOT)
    if args.quick:
        cfg.update(seam_specs=[s for s in SEAM_CONTROLS if s["name"] in {"g_ref", "g_emb8_medium", "t_r2",
                                                                          "t_emb8_medium", "arc_0.9", "arc_rot_glued",
                                                                          "none_libration"}],
                   gate_specs=[s for s in GATE_CONTROLS if s["name"] in {"ref", "torn", "cos_absomega",
                                                                          "energy_sign"}], seam_nboot=50)
    cfg.update(review_seam_specs=REVIEW_SEAM_CONTROLS, review_gate_specs=REVIEW_GATE_CONTROLS)
    if args.quick:
        cfg.update(review_seam_specs=[s for s in REVIEW_SEAM_CONTROLS if s["name"] in {
                       "g_ref", "g_emb8_medium", "t_r2", "t_emb8_medium", "t_omega_x100", "f_abs", "f_iso"}],
                   review_gate_specs=[s for s in REVIEW_GATE_CONTROLS if s["name"] in {
                       "ref", "ref_s0.3", "folded", "energy_sign", "pfold_0.05"}])
    tasks = []
    if "topo" in parts:
        tasks += [("topo", s, r, cfg) for s in tspecs for r in range(R)]
    if "energy" in parts:
        tasks += [("energy", s, r, cfg) for s in especs for r in range(R)]
    cspecs = [s for s in tspecs if s.get("core")]
    if "confirm" in parts:
        tasks += [("confirm", s, r, cfg) for s in cspecs for r in range(R, 2 * R)]
    if "seam" in parts:
        tasks += [("seam", dict(name="seam"), r, cfg) for r in range(R)]
    if "review" in parts:
        tasks += [("review", dict(name="review"), r, cfg) for r in range(R)]
    # heavy tasks first (b1 >= 1 topology controls dominate the cost)
    heavy = lambda t: (t[0] == "topo", t[0] == "topo" and topo_truth(t[1])["b1"] >= 1, t[1].get("core", False))
    tasks.sort(key=heavy, reverse=True)
    print(f"[calibrate] {len(tasks)} tasks, {args.jobs} workers, repeats = {R}, parts = {sorted(parts)}", flush=True)
    t0 = time.time()
    if args.reanalyze:
        with open(args.cache, "rb") as fh:
            cached = pickle.load(fh)
        trecs, erecs, crecs, wall = cached["trecs"], cached["erecs"], cached.get("crecs"), cached["wall"]
        srecs, rrecs = cached.get("srecs"), cached.get("rrecs")
        if "seam" in parts and srecs is None:
            raise SystemExit("--reanalyze --parts seam: the cache holds no seam records")
        if "review" in parts and rrecs is None:
            raise SystemExit("--reanalyze --parts review: the cache holds no review records")
    else:
        trecs = {s["name"]: [None] * R for s in tspecs}
        erecs = {s["name"]: [None] * R for s in especs}
        crecs = {s["name"]: [None] * R for s in cspecs}
        srecs = [None] * R
        rrecs = [None] * R
        done = 0
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_run, t) for t in tasks]
            for f in as_completed(futs):
                kind, res = f.result()
                if kind == "confirm":
                    crecs[res["name"]][res["r"] - R] = res
                elif kind == "seam":
                    srecs[res["r"]] = res
                elif kind == "review":
                    rrecs[res["r"]] = res
                else:
                    (trecs if kind == "topo" else erecs)[res["name"]][res["r"]] = res
                done += 1
                if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                    print(f"[calibrate] {done}/{len(tasks)} tasks, {time.time() - t0:.0f} s", flush=True)
        wall = time.time() - t0
        if args.cache:
            with open(args.cache, "wb") as fh:
                pickle.dump(dict(trecs=trecs, erecs=erecs, crecs=crecs, srecs=srecs, rrecs=rrecs, wall=wall), fh)
    results = dict(
        meta=dict(script="cells/0/calibrate.py", date=time.strftime("%Y-%m-%d %H:%M:%S"), wall_seconds=round(wall, 1),
                  jobs=args.jobs, repeats=R, quick=args.quick, python=platform.python_version(),
                  platform=platform.platform(), numpy=np.__version__,
                  ripser=__import__("ripser").__version__ if hasattr(__import__("ripser"), "__version__") else None,
                  calib_seed_base=CALIB_BASE, calibration_half="repeats r < R/2", heldout_half="repeats r >= R/2",
                  settings=dict(ns=cfg["ns"], pi_need=PI_NEED, tau_grid=[float(TAU_GRID[0]), float(TAU_GRID[-1]), 0.1],
                                gap_grid=[float(GAP_GRID[0]), float(GAP_GRID[-1]), 0.1], tau_s_grid=TAU_S_GRID,
                                stab_r=cfg["stab_r"], stab_plan=cfg["stab_plan"], white_ns=cfg["white_ns"],
                                wind_n=WIND_N, wind_coeff=WIND_COEFF, alpha_bs=ALPHA_BS, bs_A_rips=T.BS_A["rips"],
                                energy_ns=cfg["energy_ns"], energy_ks=cfg["energy_ks"], krr_n=KRR_N,
                                world=dict(**C.PRE_REG_WORLD, n_traj=500, horizon=64, keep="t = 1..64"))),
        note=("Instrument calibration on ground-truth pendulum embeddings only (zoo states -> known maps). "
              "No trained-model latents were used."))
    if "topo" in parts:
        print("[calibrate] analysing topology ...", flush=True)
        results["topology"] = analyze_topology(trecs, tspecs, R, cfg)
        results["null_lgumbel"] = analyze_null(trecs, tspecs, R, cfg)
        results["runtime"] = analyze_runtime(trecs, tspecs, R, cfg)
        results["winding"] = analyze_winding(trecs, tspecs, R)
        results["raw_topology"] = {nm: [dict(r=x["r"], main={k: dict(top=v["top"], cens=v["cens"], bs=v["bs"],
                                                                      pistar=v["pistar"], sec=v["sec"])
                                                             for k, v in x["main"].items()},
                                             white={k: v["top"] for k, v in x["white"].items()},
                                             wind=x["wind"]) for x in recs]
                                     for nm, recs in trecs.items()}
    if "energy" in parts:
        print("[calibrate] analysing energy ...", flush=True)
        results["energy"] = analyze_energy(erecs, especs, R, cfg)
        results["raw_energy"] = {nm: [x["scores"] for x in recs] for nm, recs in erecs.items()}
    out = Path(args.out) if args.out else (None if args.quick else OUT_DEFAULT)
    if parts & {"confirm", "seam", "review"} and "topo" not in parts and "energy" not in parts:
        # confirm / seam / review only: re-use the stored fits and keep every other section of the stored results
        src = out or OUT_DEFAULT
        if src.exists():
            stored = json.loads(src.read_text())
            for part in ("confirm", "seam", "review"):
                if part in parts:
                    stored["meta"][f"{part}_wall_seconds"] = round(wall, 1)
            results = stored
        elif "confirm" in parts:
            raise SystemExit(f"--parts confirm needs the stored topology fits in {src}")
    if "confirm" in parts:
        if "topology" not in results:                   # e.g. --parts energy,confirm: fits from the stored results
            prev = json.loads((out or OUT_DEFAULT).read_text())
            results["topology"], results["winding"] = prev["topology"], prev["winding"]
        print("[calibrate] analysing confirmation ...", flush=True)
        results["confirmation"] = analyze_confirm(crecs, cspecs, results["topology"],
                                                  results["winding"]["rho_threshold"], R)
    if "seam" in parts:
        print("[calibrate] analysing seam ratio and theta gate ...", flush=True)
        seam = analyze_seam(srecs, R, cfg)
        seam["run"] = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), jobs=args.jobs, repeats=R, quick=args.quick,
                           wall_seconds=round(wall, 1), python=platform.python_version(), numpy=np.__version__,
                           platform=platform.platform(),
                           note="ground-truth pendulum embeddings only; no trained-model latents")
        results["seam"] = seam
        results["raw_seam"] = [dict(r=x["r"], seam={nm: {k: v for k, v in rec.items() if k != "placebo"}
                                                    for nm, rec in x["seam"].items()}, gate=x["gate"])
                               for x in srecs]
    if "review" in parts:
        print("[calibrate] analysing the second-round review checks ...", flush=True)
        stored = results if "raw_energy" in results else (json.loads((out or OUT_DEFAULT).read_text())
                                                           if (out or OUT_DEFAULT).exists() else {})
        stored = json.loads(json.dumps(_py(stored)))          # JSON view (None for nan) of the stored sections
        review = analyze_review(rrecs, R, cfg, stored)
        review["run"] = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), jobs=args.jobs, repeats=R, quick=args.quick,
                             wall_seconds=round(wall, 1), note="ground-truth pendulum embeddings only; no "
                                                               "trained-model latents")
        results["review2"] = review
        results["raw_review2"] = [dict(r=x["r"], seam=x["seam"], gate=x["gate"]) for x in rrecs]
    if out is not None:
        out.write_text(json.dumps(_py(results), indent=1))
        print(f"[calibrate] wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {wall:.0f} s", flush=True)
    summarize(results)
    return results


def summarize(res: dict) -> None:
    if "review2" in res:
        rv = res["review2"]
        for t, v in rv["seam_confirm"]["thresholds"].items():
            print(f"[review] seam @ {t} on fresh seeds: TNR glued {v['tnr_glued']} TPR torn {v['tpr_torn']}")
        fr = rv["fold_rule"]
        print(f"[review] fold rule: plateau {fr['plateau']}, fitted {fr['threshold']}; held-out (code value) "
              f"{fr['heldout_code']}")
        for nm, v in rv["seam_controls"].items():
            if not v["core"]:
                print(f"[review] {nm:18s} SR {v['sr_median']} kappa {v['kappa_median']} fold {v['fold_median']} "
                      f"in_range {v['frac_in_range']}")
        for rn, v in rv["theta_gate"]["rules"].items():
            print(f"[review] gate {rn}: TPR {v['tpr_injective']} FPR {v['fpr_folded']}")
        if "energy_joint" in rv:
            for k, v in rv["energy_joint"]["heldout"].items():
                print(f"[review] energy {k}: TPR {v['tpr']} FPR {v['fpr']}")
    if "seam" in res:
        sm = res["seam"]
        if "fit" in sm:
            f = sm["fit"]
            print(f"\n[seam] glued core max {f['glued_core_max_all']:.3g}, torn core min {f['torn_core_min_all']:.3g}; "
                  f"plateau {f['plateau']}, fitted threshold {f['threshold']}")
            for k, v in sm["heldout"].items():
                print(f"[seam] held-out @ {v['threshold']:.3g} ({k}): TPR torn {v['tpr_torn']} TNR glued {v['tnr_glued']}")
        if "arc" in sm:
            print(f"[seam] arc gap at crossing: {sm['arc']['gap_at_crossing']}")
        g = sm["theta_gate"]
        if "fit" in g:
            print(f"[gate] injective core min {g['fit']['injective_core_min_all']:.4f}, folded core max "
                  f"{g['fit']['folded_core_max_all']:.4f}, 5th-percentile threshold {g['fit']['threshold']:.4f}")
            for k, v in g["heldout"].items():
                print(f"[gate] held-out @ {v['threshold']:.4f} ({k}): TPR {v['tpr_injective']} FPR {v['fpr_folded']}")
    if "topology" in res:
        tp = res["topology"]
        rec = tp["recommended"]
        fr = tp["configs"][rec["config"]][rec["family"]]
        print(f"\n[topology] recommended: {rec['config']} rule={rec['family']} param={rec['param']}")
        print(f"  calib objective {fr['calib']['objective']:.3f}; held-out TPR(core) {fr['heldout']['tpr_core']}; "
              f"TNR(core) {fr['heldout']['tnr_core']}; FPR(b1=1 claim) {fr['heldout']['fpr_claims_b1_eq_1']}")
        for row in tp["ranking_calib"][:10]:
            print("  ", row)
    if "winding" in res and res["winding"]:
        w = res["winding"]
        print(f"[winding] |rho| threshold {w['rho_threshold']}: held-out TPR {w['heldout_tpr_rho']}, "
              f"TNR {w['heldout_tnr_rho']}")
    if "confirmation" in res:
        for k, v in res["confirmation"]["instruments"].items():
            if isinstance(v, dict):
                print(f"[confirm] {k:24s} param={v['param']} TPR {v['tpr_core']} TNR {v['tnr_core']}")
        w = res["confirmation"]["winding"]
        print(f"[confirm] winding |rho| >= {w['rho_threshold']}: TPR {w['tpr']} TNR {w['tnr']}")
    if "energy" in res:
        sc = res["energy"]["scores"]
        for key in [k for k in sc if k.startswith(("quad_r2@", "knn10_r2@", "s_k10@", "lip_tail_k10@",
                                                   "s_and_tail_k10@", "krr_r2@"))]:
            v = sc[key]
            print(f"[energy] {key:22s} thr={v.get('threshold', v.get('thresholds'))} TPR={v['heldout_tpr']['rate']} "
                  f"FPR={v['heldout_fpr']['rate']}")


if __name__ == "__main__":
    main()
