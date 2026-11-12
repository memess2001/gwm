"""Seam ratio for cell 0, angle_raw condition: is the latent glued or torn across the input cut theta = +-pi?

Definition (cells/0/prereg.md section 0b, item 2)
------------------------------------------------
    SR = median latent distance of pairs just ACROSS the cut (theta near +pi vs theta near -pi, matched omega)
         / median latent distance of pairs with the SAME angular separation and omega across other angles.
SR ~ 1: glued (the cut is a place like any other); SR >> 1: torn (the two sides of the cut are far apart).

Construction
------------
* Reference angles: the cut a = pi and ``control_angles`` (default k pi / 4, k = -3..3, i.e. every 45 degrees
  except pi). Around each angle a, the window |wrap(theta - a)| < w is split into a left side (just below a) and a
  right side (just above a). For a = pi the left side is theta in (pi - w, pi), the right side theta in [-pi, -pi + w).
* Crossing pairs: one point from each side with |omega_i - omega_j| <= ``omega_tol`` (matched omega). Each pair
  gets an angular separation dtheta = d_right - d_left in (0, 2w) and a mid velocity (omega_i + omega_j) / 2.
* Strata = (bin of mid omega, width ``omega_bin``) x (bin of dtheta, ``n_dtheta_bins`` equal bins on (0, 2w)), so
  "same angular separation, matched omega" is enforced bin by bin.
* Per stratum s: den(s) = median over control angles of that angle's median pair distance (an angle counts if it
  has >= ``min_pairs`` pairs in s; s is usable if >= ``min_ctrl_angles`` angles count and the cut has >= ``min_pairs``
  pairs in s). The median over angles (not the pooled median) keeps one anisotropic direction from setting the
  scale.
* SR = exp( sum_s n_s log(m_cut(s) / den(s)) / sum_s n_s ): the geometric mean over usable strata of the ratio
  of medians (cut median m_cut(s) over den(s)), weighted by the number n_s of the cut's pairs in s, so the cut's
  own (dtheta, omega) mix sets the weights. (The median over all cut pairs of d_pair / den(s) was tried first and
  dropped: on torn clouds the normalised distances form one cluster per dtheta bin and that median jumps between
  clusters, by up to 40% between repeats; the stratum average is continuous in the gap.)
Properties: invariant under translation, rotation and global rescaling of Z (scale free); NOT invariant under
anisotropic maps (calibration.md section 9 reports how much that matters). The scale of SR is tied to the
window: for a gap g across the cut, SR ~ g / (typical dtheta in latent units), so ``half_width`` must be fixed
together with any threshold (defaults w = 0.12 rad, omega_tol = 0.08 follow the reviewer pilot of section 0b).

Uncertainty: bootstrap over points, or over ``groups`` (trajectory ids; recommended, consecutive states of one
trajectory are not independent). A resample with multiplicities m gives pair weights m_i m_j (with groups:
m_g m_h for pairs from two groups, m_g for a pair inside one group, so both have expectation ~1), so pairs are built
once and every replicate is a set of weighted block medians (lower weighted median, i.e. for equal weights and an
even count the lower of the two middle values) and weighted stratum counts. Percentile 95% interval.

Diagnostics (no bootstrap): ``placebo`` = SR computed with a control angle in the role of the cut (the other
control angles as denominator); on a smooth latent all placebos ~ 1, so they are an internal null of the same
latent. ``by_omega`` = SR restricted to each mid-omega bin (shows partial gluing, e.g. glued only for fast
rotations). ``omega_abs_min`` > 0 restricts the whole computation to |omega| >= omega_abs_min (secondary readout).

Review additions (calibration.md, section "审稿意见（第二轮）"); none of them changes ``ratio``:
* ``omega_aspect`` (kappa): least squares |dz|^2 ~ a dtheta^2 + b domega^2 + c dtheta domega over the control-angle
  pairs, kappa = sqrt(b / a) = latent length per unit omega over latent length per unit theta (1 for the reference
  chart (cos, sin, omega); c for (theta, c omega)). Scale free. Matched pairs still differ by up to ``omega_tol``
  in omega, so when kappa is large the omega part of every pair distance swamps the theta part and SR is pulled
  towards 1 even for a torn latent ((theta, 100 omega) gives SR ~ 1.8, "glued"). ``seam_flags`` marks kappa >
  ``OMEGA_ASPECT_MAX`` as outside the instrument's range.
* ``fold_index``: within each stratum of the cut, median distance of the "mirror" pairs (offsets a, b from the
  cut with |a - b| < (a + b) / 3) over that of the lopsided pairs (|a - b| > 2 (a + b) / 3), geometric mean
  weighted by stratum size. A latent that is continuous across the cut but FOLDED there (theta and -theta
  identified, e.g. (|theta|, omega) or (cos theta, |sin theta|, omega)) maps mirror pairs onto (almost) the same
  point: index ~ 0.5, while glued, torn, stretched or noisy latents give ~ 1. SR alone reads such a fold as glued
  (SR ~ 0.5). ``fold_index_ctrl`` = the same at every control angle (internal null). Only first-order folds are
  seen: in (cos theta, omega) the two sides differ by O(dtheta^2), the omega part of the pair distances dominates
  and the index stays ~ 0.95 (SR ~ 0.36); that fold is excluded by the theta-decodability gate instead.
* ``n_cut_groups``: number of distinct ``groups`` (trajectories) among the cut's usable pairs; with ``groups``
  given, fewer than ``min_cut_groups`` makes the status "insufficient" (an SR from a handful of trajectories has a
  meaningless trajectory bootstrap).

The true theta and omega used here are analysis-only ground truth. Nothing here may be applied to trained-model
latents before the cell-0 preregistration is approved.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

DEFAULT_CONTROL_ANGLES = tuple(k * np.pi / 4 for k in range(-3, 4))      # every 45 degrees except pi
# Flags of ``seam_flags`` (calibration.md, "审稿意见（第二轮）"; ground-truth controls only, fresh seeds 960_000 + r).
OMEGA_ASPECT_MAX = 10.0       # torn chart: SR ~ 15 at kappa = 10, ~ 5.1 at kappa = 30 (no margin), ~ 1.8 at 100
FOLD_INDEX_MIN = 0.7          # first-order folds <= 0.58; core >= 0.97, not-folded stress >= 0.86 (20 review
                              # repeats). Fixed before the review run. Does NOT see second-order folds such as
                              # (cos theta, omega) (0.85-1.08): those are left to the theta gate.


def wrap_angle(x) -> np.ndarray:
    """Wrap to [-pi, pi)."""
    return np.mod(np.asarray(x, dtype=np.float64) + np.pi, 2 * np.pi) - np.pi


def _crossing_pairs(d, om, idx_win, omega_tol):
    """Pairs (left, right) inside one window with |omega_l - omega_r| <= omega_tol.

    d = wrapped offset from the reference angle (left: d < 0, right: d >= 0). Returns global indices (i, j)."""
    left = idx_win[d[idx_win] < 0]
    right = idx_win[d[idx_win] >= 0]
    if len(left) == 0 or len(right) == 0:
        e = np.zeros(0, dtype=np.int64)
        return e, e
    right = right[np.argsort(om[right], kind="stable")]
    omr = om[right]
    lo = np.searchsorted(omr, om[left] - omega_tol, side="left")
    hi = np.searchsorted(omr, om[left] + omega_tol, side="right")
    cnt = hi - lo
    tot = int(cnt.sum())
    if tot == 0:
        e = np.zeros(0, dtype=np.int64)
        return e, e
    li = np.repeat(left, cnt)
    offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    ri = right[np.repeat(lo, cnt) + offs]
    return li, ri


def _group_wmedian(vals, starts, ends, w):
    """Lower weighted median of vals[starts[g]:ends[g]] (each block sorted by value) with weights w (>= 0).

    Returns (medians, total weight per block); a block with total weight 0 gets an arbitrary median."""
    cs = np.cumsum(w)
    before = np.where(starts > 0, cs[np.maximum(starts - 1, 0)], 0.0)
    tot = cs[ends - 1] - before
    idx = np.searchsorted(cs, before + 0.5 * tot, side="left")
    idx = np.clip(idx, starts, ends - 1)
    return vals[idx], tot


class _PairTable:
    """All crossing pairs of all reference angles, sorted by (angle, stratum, distance) for weighted medians."""

    def __init__(self, Z, theta, omega, angles, half_width, omega_tol, omega_bin, n_dtheta_bins,
                 max_per_window, rng):
        N = len(theta)
        self.n_angles = len(angles)
        om_lo = np.floor(omega.min() / omega_bin) * omega_bin
        self.omega_edges = np.arange(om_lo, omega.max() + omega_bin + 1e-12, omega_bin)
        self.n_om = len(self.omega_edges) - 1
        self.n_dt = int(n_dtheta_bins)
        self.n_strata = self.n_om * self.n_dt
        I, J, A, S, DT, AS = [], [], [], [], [], []
        self.window_points = []
        for a_i, a in enumerate(angles):
            d = wrap_angle(theta - a)
            win = np.flatnonzero(np.abs(d) < half_width)
            if max_per_window is not None and len(win) > max_per_window:
                win = np.sort(rng.choice(win, size=max_per_window, replace=False))
            self.window_points.append(len(win))
            i, j = _crossing_pairs(d, omega, win, omega_tol)
            dth = d[j] - d[i]                                          # in (0, 2w)
            tb = np.minimum((dth / (2 * half_width) * self.n_dt).astype(np.int64), self.n_dt - 1)
            mid = 0.5 * (omega[i] + omega[j])
            ob = np.clip(((mid - om_lo) / omega_bin).astype(np.int64), 0, self.n_om - 1)
            I.append(i), J.append(j), A.append(np.full(len(i), a_i, dtype=np.int64)), S.append(ob * self.n_dt + tb)
            # offsets of the two points from the reference angle: left a = -d_i > 0, right b = d_j >= 0;
            # asymmetry |a - b| / (a + b) in [0, 1] (0 = mirror images of each other across the angle)
            DT.append(dth), AS.append(np.abs(-d[i] - d[j]) / np.maximum(dth, 1e-300))
        I, J, A, S = (np.concatenate(v) if len(v) else np.zeros(0, np.int64) for v in (I, J, A, S))
        DT, AS = (np.concatenate(v) if len(v) else np.zeros(0) for v in (DT, AS))
        dist = np.linalg.norm(Z[I] - Z[J], axis=1) if len(I) else np.zeros(0)
        gid = A * self.n_strata + S
        order = np.lexsort((dist, gid))
        self.i, self.j, self.angle, self.stratum = I[order], J[order], A[order], S[order]
        self.dist, self.gid = dist[order], gid[order]
        self.dth, self.asym = DT[order], AS[order]
        self.dom = omega[self.j] - omega[self.i] if len(self.i) else np.zeros(0)
        self.ug, self.starts = np.unique(self.gid, return_index=True)
        self.ends = np.append(self.starts[1:], len(self.gid)).astype(np.int64)
        self.g_angle, self.g_stratum = self.ug // self.n_strata, self.ug % self.n_strata
        self.N = N

    def group_medians(self, w):
        """Weighted medians and weights of every (angle, stratum) block -> dense [n_angles, n_strata] arrays."""
        med = np.full((self.n_angles, self.n_strata), np.nan)
        tot = np.zeros((self.n_angles, self.n_strata))
        if len(self.ug):
            m, t = _group_wmedian(self.dist, self.starts, self.ends, w)
            med[self.g_angle, self.g_stratum] = m
            tot[self.g_angle, self.g_stratum] = t
        return med, tot


def _ratio_for(table, med, tot, cut: int, ctrl: Sequence[int], min_pairs, min_ctrl_angles,
               strata: Optional[np.ndarray] = None, return_mask: bool = False):
    """SR with angle ``cut`` as the cut and angles ``ctrl`` as controls, from the block medians / weights.

    SR = exp( sum_s n_s log(m_cut(s) / den(s)) / sum_s n_s ) over the usable strata s, n_s = (weighted) number of
    the cut's pairs in s. ``strata`` (bool per stratum) restricts the strata that enter the average.
    Returns (ratio, total weight of the cut's pairs used, number of usable strata), plus the bool mask of the
    strata that entered the average if ``return_mask``."""
    ctrl = np.asarray(ctrl, dtype=np.int64)
    okc = tot[ctrl] >= min_pairs                                          # [n_ctrl, n_strata]
    n_ok = okc.sum(0)
    den = np.full(med.shape[1], np.nan)
    enough = n_ok >= min_ctrl_angles
    if enough.any():
        mc = np.where(okc[:, enough], med[ctrl][:, enough], np.nan)
        den[enough] = np.nanmedian(mc, axis=0)                           # every column has >= 1 finite value
    usable = np.isfinite(den) & (tot[cut] >= min_pairs) & (den > 0) & (med[cut] > 0)
    n_usable = int(usable.sum())
    if strata is not None:
        usable &= strata
    if not usable.any():
        return (np.nan, 0.0, n_usable, usable) if return_mask else (np.nan, 0.0, n_usable)
    wt = tot[cut, usable]
    lr = np.log(med[cut, usable] / den[usable])
    out = float(np.exp(np.sum(wt * lr) / wt.sum())), float(wt.sum()), n_usable
    return (*out, usable) if return_mask else out


def _omega_aspect(tab) -> float:
    """kappa = sqrt(b / a) from the least-squares fit |dz|^2 ~ a dtheta^2 + b domega^2 + c dtheta domega over the
    pairs of the control angles (never the cut). inf if the latent does not resolve theta locally (a <= 0)."""
    sel = tab.angle != 0
    if sel.sum() < 10:
        return np.nan
    dth, dom = tab.dth[sel], tab.dom[sel]
    X = np.stack([dth ** 2, dom ** 2, dth * dom], 1)
    a, b, _ = np.linalg.lstsq(X, tab.dist[sel] ** 2, rcond=None)[0]
    if a <= 0:
        return np.inf
    return float(np.sqrt(max(b, 0.0) / a))


def _fold_index(tab, angle: int, min_each: int = 5) -> float:
    """Mirror pairs (asymmetry < 1/3) vs lopsided pairs (> 2/3) at one reference angle, stratum by stratum:
    exp of the stratum-size-weighted mean of log(median d_mirror / median d_lopsided). ~1 if the latent is smooth
    (or torn) at that angle, ~0.5 if it is folded there (theta -> -theta about the angle)."""
    sel = np.flatnonzero(tab.angle == angle)
    if len(sel) == 0:
        return np.nan
    st, dist, asym = tab.stratum[sel], tab.dist[sel], tab.asym[sel]
    lr, wt = [], []
    for s in np.unique(st):
        m = st == s
        dm, dl = dist[m & (asym < 1 / 3)], dist[m & (asym > 2 / 3)]
        if len(dm) >= min_each and len(dl) >= min_each:
            a, b = np.median(dm), np.median(dl)
            if a > 0 and b > 0:
                lr.append(np.log(a / b)), wt.append(m.sum())
    if not lr:
        return np.nan
    lr, wt = np.asarray(lr), np.asarray(wt, dtype=np.float64)
    return float(np.exp(np.sum(wt * lr) / wt.sum()))


def seam_ratio(Z, theta, omega, *, half_width: float = 0.12, omega_tol: float = 0.08, omega_bin: float = 0.25,
               n_dtheta_bins: int = 4, control_angles: Sequence[float] = DEFAULT_CONTROL_ANGLES, cut: float = np.pi,
               min_pairs: int = 10, min_ctrl_angles: int = 3, min_strata: int = 3, min_cut_pairs: int = 50,
               max_per_window: Optional[int] = 2000, n_boot: int = 200, groups=None, seed=0,
               diagnostics: bool = True, omega_abs_min: float = 0.0, min_cut_groups: int = 10) -> dict[str, Any]:
    """Seam ratio of a latent cloud Z [N, D] with ground-truth angle ``theta`` [N] and velocity ``omega`` [N].

    ``groups`` [N] (e.g. trajectory ids) switches the bootstrap to resampling whole groups. Status "insufficient"
    (ratio nan) if fewer than ``min_strata`` usable strata or fewer than ``min_cut_pairs`` usable pairs cross the
    cut (e.g. libration-only data never reaches theta = pi), or, with ``groups``, if the usable cut pairs come
    from fewer than ``min_cut_groups`` distinct groups. ``omega_abs_min`` > 0 keeps only points with
    |omega| >= omega_abs_min (e.g. 1.6: fast rotations only, the restriction of the reviewer pilot); the default 0
    is the preregistered definition (all omega).
    Returns dict(ratio, ci95, log_sd, status, n_cut_pairs, n_cut_groups, n_strata, n_boot_ok, window_points,
    omega_aspect, fold_index, placebo, placebo_max, fold_index_ctrl, by_omega, settings); the last four only with
    ``diagnostics`` (omega_aspect and fold_index are always computed: they are cheap and feed ``seam_flags``).
    """
    Z = np.asarray(Z, dtype=np.float64)
    theta = wrap_angle(theta)
    omega = np.asarray(omega, dtype=np.float64)
    if Z.ndim != 2 or len(Z) != len(theta) or len(theta) != len(omega):
        raise ValueError("Z must be [N, D] and theta, omega [N]")
    if groups is not None and len(groups) != len(theta):
        raise ValueError("groups must have one entry per point")
    if omega_abs_min > 0:
        keep = np.abs(omega) >= omega_abs_min
        Z, theta, omega = Z[keep], theta[keep], omega[keep]
        groups = None if groups is None else np.asarray(groups)[keep]
    ctrl_angles = [float(a) for a in control_angles]
    if any(abs(wrap_angle(a - cut)) < 2 * half_width for a in ctrl_angles):
        raise ValueError("control angles must be at least 2 * half_width away from the cut")
    angles = [float(cut)] + ctrl_angles
    rng = np.random.default_rng(seed)
    tab = _PairTable(Z, theta, omega, angles, half_width, omega_tol, omega_bin, n_dtheta_bins, max_per_window, rng)
    ctrl = list(range(1, len(angles)))
    med, tot = tab.group_medians(np.ones(len(tab.dist)))
    ratio, n_used, n_strata, used_mask = _ratio_for(tab, med, tot, 0, ctrl, min_pairs, min_ctrl_angles,
                                                    return_mask=True)
    if groups is not None:
        g_all = np.asarray(groups)
        cut_used = (tab.angle == 0) & used_mask[tab.stratum]
        n_groups = int(len(np.union1d(g_all[tab.i[cut_used]], g_all[tab.j[cut_used]])))
    else:
        n_groups = None
    settings = dict(half_width=half_width, omega_tol=omega_tol, omega_bin=omega_bin, n_dtheta_bins=n_dtheta_bins,
                    control_angles=ctrl_angles, cut=float(cut), min_pairs=min_pairs, min_ctrl_angles=min_ctrl_angles,
                    min_strata=min_strata, min_cut_pairs=min_cut_pairs, max_per_window=max_per_window,
                    n_boot=n_boot, bootstrap="groups" if groups is not None else "points",
                    omega_abs_min=float(omega_abs_min), min_cut_groups=min_cut_groups)
    out: dict[str, Any] = dict(ratio=np.nan, ci95=(np.nan, np.nan), log_sd=np.nan, status="insufficient",
                               n_cut_pairs=int(n_used), n_cut_groups=n_groups, n_strata=int(n_strata), n_boot_ok=0,
                               window_points=dict(zip(["cut"] + [f"{a:.4f}" for a in ctrl_angles], tab.window_points)),
                               omega_aspect=_omega_aspect(tab), fold_index=_fold_index(tab, 0),
                               placebo=[], placebo_max=np.nan, fold_index_ctrl=[], by_omega={}, settings=settings)
    if not (np.isfinite(ratio) and n_strata >= min_strata and n_used >= min_cut_pairs
            and (n_groups is None or n_groups >= min_cut_groups)):
        return out
    out.update(ratio=ratio, status="ok")
    # ------------------------------------------------------------------ bootstrap
    if n_boot > 0:
        if groups is not None:
            g = np.asarray(groups)
            _, ginv = np.unique(g, return_inverse=True)
            G = int(ginv.max()) + 1
            gi, gj = ginv[tab.i], ginv[tab.j]
            same = gi == gj
        reps = []
        for _ in range(n_boot):
            if groups is not None:
                # a group drawn m times contributes its own pairs m times (not m^2: the pairs between two copies of
                # one trajectory are not pairs of the original data), pairs between groups m_g * m_h times; both
                # have expectation ~1, so within-trajectory pairs are not up-weighted
                mg = rng.multinomial(G, np.full(G, 1.0 / G)).astype(np.float64)
                w = np.where(same, mg[gi], mg[gi] * mg[gj])
            else:
                m = rng.multinomial(len(theta), np.full(len(theta), 1.0 / len(theta))).astype(np.float64)
                w = m[tab.i] * m[tab.j]                                  # i != j always (opposite sides)
            mb, tb = tab.group_medians(w)
            rb, nb, sb = _ratio_for(tab, mb, tb, 0, ctrl, min_pairs, min_ctrl_angles)
            if np.isfinite(rb) and rb > 0:
                reps.append(rb)
        if len(reps) >= max(10, n_boot // 2):
            lr = np.log(reps)
            out.update(ci95=(float(np.exp(np.quantile(lr, 0.025))), float(np.exp(np.quantile(lr, 0.975)))),
                       log_sd=float(lr.std(ddof=1)), n_boot_ok=len(reps))
    # ------------------------------------------------------------------ diagnostics
    if diagnostics:
        plac = []
        for c in ctrl:
            others = [k for k in ctrl if k != c]
            rc, nc, _ = _ratio_for(tab, med, tot, c, others, min_pairs, min_ctrl_angles)
            plac.append(rc if nc >= min_cut_pairs else np.nan)
        out["placebo"] = plac
        out["placebo_max"] = float(np.nanmax(plac)) if np.isfinite(plac).any() else np.nan
        out["fold_index_ctrl"] = [_fold_index(tab, c) for c in ctrl]
        om_of_stratum = np.arange(tab.n_strata) // tab.n_dt
        prof = {}
        for b in range(tab.n_om):
            rb, nb, _ = _ratio_for(tab, med, tot, 0, ctrl, min_pairs, min_ctrl_angles, strata=om_of_stratum == b)
            if nb >= min_pairs:
                lo, hi = tab.omega_edges[b], tab.omega_edges[b + 1]
                prof[f"{lo:+.2f}..{hi:+.2f}"] = dict(ratio=rb, n_pairs=int(nb))
        out["by_omega"] = prof
    return out


def classify_seam(res: dict, threshold: float = 5.0) -> str:
    """'torn' if ratio > threshold, 'glued' if ratio <= threshold, 'insufficient' if no ratio (point estimate,
    as in prediction P0.4'); see res['ci95'] for whether the interval straddles the threshold."""
    r = res.get("ratio", np.nan)
    if res.get("status") != "ok" or not np.isfinite(r):
        return "insufficient"
    return "torn" if r > threshold else "glued"


def seam_flags(res: dict, threshold: float = 5.0, omega_aspect_max: float = OMEGA_ASPECT_MAX,
               fold_index_min: float = FOLD_INDEX_MIN) -> dict[str, Any]:
    """Verdict plus the review flags (calibration.md, "审稿意见（第二轮）"):

    * ``ci_straddles``: the bootstrap 95% interval contains ``threshold`` (prereg suggestion: "seam uncertain");
    * ``omega_dominated``: omega_aspect > omega_aspect_max, SR diluted towards 1 (a torn latent can read as glued);
    * ``folded_at_cut``: fold_index < fold_index_min, the two sides are identified by a fold rather than glued
      (SR reads ~0.5, "glued");
    * ``in_range``: verdict is "glued" / "torn", not folded_at_cut, and not (omega_dominated and verdict "glued").
      The dilution only pulls SR towards 1, so a "torn" verdict stays valid when omega dominates, a "glued" one
      does not (omega-stretched glued latents read ~0.84, torn ones 14.6 / 5.2 / 1.8 at kappa = 10 / 30 / 100).
    The verdict itself is ``classify_seam`` (point estimate vs threshold), unchanged by the flags."""
    verdict = classify_seam(res, threshold)
    lo, hi = res.get("ci95", (np.nan, np.nan))
    lo = np.nan if lo is None else lo
    hi = np.nan if hi is None else hi
    kappa = res.get("omega_aspect", np.nan)
    fi = res.get("fold_index", np.nan)
    kappa = np.nan if kappa is None else float(kappa)
    fi = np.nan if fi is None else float(fi)
    omega_dom = bool(kappa > omega_aspect_max)              # inf (theta not resolved) counts; nan does not
    folded = bool(fi < fold_index_min)                      # nan (too few pairs) does not count
    return dict(verdict=verdict, ci_straddles=bool(np.isfinite(lo) and np.isfinite(hi) and lo <= threshold <= hi),
                omega_dominated=omega_dom, folded_at_cut=folded,
                in_range=bool(verdict in ("glued", "torn") and not folded and not (omega_dom and verdict == "glued")),
                omega_aspect=float(kappa), fold_index=float(fi), threshold=float(threshold))


# ================================================================================================ v2 (cell-0 prereg v2)
# Changes required by the cell-0 preregistration review (cells/0/prereg.md, v2 list items 2, 3, 6, 14), checked
# on ground truth by the reviewer (cells/0/exploratory_prereg_review/fixes_check.py, fixes.jsonl):
#   (B) drop "straddle" windows: with history 2 the encoder sees (theta_{t-1}, theta_t); 33-36 % of the windows in
#       the cut window have their two frames on opposite sides of the cut. A "displaced tear" (the tear sits between
#       straddle and post-straddle windows) read SR 2.8-4.9 with every flag silent; without straddle windows it
#       reads 44-46 (torn), a glued chart 1.01;
#   (A) kappa-adaptive omega matching, omega_tol = 0.08 / max(1, kappa_hat): with omega anisotropy kappa < 10 a torn
#       arc read SR 2-5 ("in range"); with the adaptive tolerance the same arcs read 6.4-10.
# The window half width (0.12 rad), bins and thresholds are unchanged. Verdicts are three-way (glued / partial /
# torn); the PH x SR consistency table below decides "glued" / "torn" / "inconclusive" for a run.
SR_GLUED_MAX, SR_TORN_MIN = 1.5, 5.0
KAPPA_PH_MAX = 5.0             # prereg v2 item 6: H1 = 0 with kappa > 5 is inconclusive (PH false negatives)


def straddle_mask(theta, theta_prev) -> np.ndarray:
    """Windows (theta_{t-1}, theta_t) whose two frames lie on opposite sides of the cut theta = +-pi (raw values
    in [-pi, pi): the raw input jumps by more than pi between the two frames)."""
    return np.abs(wrap_angle(theta) - wrap_angle(theta_prev)) > np.pi


def seam_ratio_v2(Z, theta, omega, theta_prev, *, omega_tol_base: float = 0.08, **kw) -> dict[str, Any]:
    """Seam ratio v2: straddle windows dropped, omega tolerance = omega_tol_base / max(1, kappa_hat), where
    kappa_hat is ``omega_aspect`` of a first pass (default tolerance, no bootstrap) on the kept points.
    All other keyword arguments go to ``seam_ratio`` (groups, n_boot, seed, omega_abs_min, ...). Adds
    ``kappa_first_pass``, ``n_straddle_dropped``, ``omega_tol_used`` and ``version = 2``."""
    Z = np.asarray(Z, dtype=np.float64)
    keep = ~straddle_mask(theta, theta_prev)
    groups = kw.pop("groups", None)
    g = None if groups is None else np.asarray(groups)[keep]
    first = seam_ratio(Z[keep], np.asarray(theta)[keep], np.asarray(omega)[keep], groups=g, n_boot=0,
                       diagnostics=False, seed=kw.get("seed", 0),
                       omega_abs_min=kw.get("omega_abs_min", 0.0))
    kap = first.get("omega_aspect", np.nan)
    kap = 1.0 if (kap is None or not np.isfinite(kap)) else float(kap)
    tol = omega_tol_base / max(1.0, kap)
    out = seam_ratio(Z[keep], np.asarray(theta)[keep], np.asarray(omega)[keep], groups=g, omega_tol=tol, **kw)
    out.update(kappa_first_pass=kap, n_straddle_dropped=int((~keep).sum()), omega_tol_used=tol, version=2)
    return out


def classify_seam3(res: dict, glued_max: float = SR_GLUED_MAX, torn_min: float = SR_TORN_MIN) -> str:
    """Three-way verdict on the point estimate: 'glued' (SR <= 1.5), 'partial' (1.5 < SR <= 5), 'torn' (SR > 5),
    or 'insufficient'."""
    r = res.get("ratio", np.nan)
    if res.get("status") != "ok" or r is None or not np.isfinite(r):
        return "insufficient"
    if r <= glued_max:
        return "glued"
    return "torn" if r > torn_min else "partial"


def seam_flags_v2(res: dict) -> dict[str, Any]:
    """Three-way verdict plus the review flags of ``seam_flags`` (fold, omega dominance) and the v2 fold-suspect
    flag (fold_index > 1.3 reported as 'suspected misalignment', prereg v2 item 14)."""
    f = seam_flags(res)
    v = classify_seam3(res)
    fi = f["fold_index"]
    f.update(verdict3=v, misalignment_suspect=bool(np.isfinite(fi) and fi > 1.3),
             in_range3=bool(v in ("glued", "partial", "torn") and not f["folded_at_cut"]
                            and not (f["omega_dominated"] and v != "torn")))
    return f


def seam_ph_consistency(ph_state: str, sr_verdict3: str, kappa: float, folded: bool = False) -> str:
    """Cell-0 prereg v2 item 2: one verdict per run from the H1 rule and the seam ratio.

    ph_state: 'ring' (H1 = 1 and wrapping), 'none' (H1 = 0), 'unstable', 'ring_not_wrapping'.
    sr_verdict3: classify_seam3 output. Returns 'glued', 'torn' or 'inconclusive':
      glued  <=> ring and SR glued (and not folded at the cut);
      torn   <=> none and SR torn (and kappa <= 5: H1 = 0 under strong omega anisotropy is a known PH false negative);
      everything else (disagreement, 'partial', 'unstable', insufficient data, folds) -> inconclusive."""
    if folded or sr_verdict3 == "insufficient" or ph_state not in ("ring", "none"):
        return "inconclusive"
    if ph_state == "ring" and sr_verdict3 == "glued":
        return "glued"
    if ph_state == "none" and sr_verdict3 == "torn":
        return "torn" if (np.isfinite(kappa) and kappa <= KAPPA_PH_MAX) else "inconclusive"
    return "inconclusive"
