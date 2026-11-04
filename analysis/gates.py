"""Gates and pointwise checks for cell 0 (cells/0/prereg.md section 0b, items 3 and 4).

* ``theta_decodability``: JEPA gate. With gamma = 0 a latent that keeps only conserved quantities (energy E, the
  sign of omega) is a zero-loss solution of latent prediction; the variance / covariance terms of VICReg and the
  original gate (per-dimension std, effective rank) do not exclude it. The gate requires the true angle to be
  locally decodable: kNN R^2 of cos(theta) AND of sin(theta) from the latent >= threshold (prereg proposes 0.99;
  calibration.md section 9 gives the measured distributions). Same probe settings as the energy probe
  (k = 10, n = 2000 random points, 5-fold CV, Euclidean metric of the latent).
  What the threshold means (calibration.md, "审稿意见（第二轮）", fresh seeds): folding theta -> -theta for the
  lowest-energy 5% of the states gives r2_min ~ 0.986, 10% ~ 0.95, 20% ~ 0.83, so 0.99 rejects latents that fold
  >~ 5% of the states and 0.9 only those that fold >~ 15%. At n = 2000 the kNN neighbourhoods are coarse, and
  injective latents whose metric is dominated by omega fall below 0.99 ((cos, sin, omega / 0.3): 30% rejected);
  n = 8000 at 0.99 passed all injective core controls (260/260) with the same partial-fold sensitivity. The
  default stays n = 2000 (the calibrated setting); n is a preregistration decision.
* ``pointwise_angle_check``: P0.1 theorem condition. The b1 >= 1 argument needs a continuous map latent -> S^1 whose
  pointwise error is < pi/2 (then it is homotopic to the true angle). Given decoded angles (e.g. the angle of the
  nearest rendered frame to each decoded frame) and the true angles of the same analysis points, report the
  fraction with circular error < pi/2; the condition holds iff that fraction is 1.
* ``angle_from_cs`` and ``nearest_frame_angle``: two ways to get decoded angles (angle_cs outputs; pixel outputs
  matched to a bank of rendered frames with known angles).

The true angles are analysis-only ground truth. Nothing here may be applied to trained-model latents before the
cell-0 preregistration is approved.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .smoothness import knn_probe_r2

THETA_GATE_THRESHOLD = 0.99          # prereg section 0b item 3 (proposed); see calibration.md section 9


# ----------------------------------------------------------------------------------------------- theta gate
def theta_decodability(Z, theta, k: int = 10, n: int = 2000, n_splits: int = 5, seed=0,
                       threshold: float = THETA_GATE_THRESHOLD) -> dict[str, Any]:
    """kNN R^2 of cos(theta) and sin(theta) from Z on ``n`` random points; ``passed`` iff both >= threshold."""
    Z = np.asarray(Z, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    if len(Z) != len(theta):
        raise ValueError("Z and theta must have the same length")
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(Z), size=n, replace=False)) if len(Z) > n else np.arange(len(Z))
    cv_seed = int(rng.integers(2 ** 31))
    r2c = knn_probe_r2(Z[sel], np.cos(theta[sel]), k=k, n_splits=n_splits, seed=cv_seed)
    r2s = knn_probe_r2(Z[sel], np.sin(theta[sel]), k=k, n_splits=n_splits, seed=cv_seed)
    r2min = float(min(r2c, r2s))
    return dict(r2_cos=float(r2c), r2_sin=float(r2s), r2_min=r2min, passed=bool(r2min >= threshold),
                threshold=float(threshold), n=int(len(sel)), k=int(k))


# ----------------------------------------------------------------------------------------------- P0.1 pointwise
def circular_error(a, b) -> np.ndarray:
    """|a - b| on the circle, in [0, pi]."""
    d = np.mod(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64) + np.pi, 2 * np.pi) - np.pi
    return np.abs(d)


def pointwise_angle_check(theta_hat, theta, tol: float = np.pi / 2) -> dict[str, Any]:
    """Fraction of points whose decoded angle is within ``tol`` (strictly) of the true angle on the circle.

    Non-finite decoded angles count as failures. ``all_ok`` is the P0.1 condition (every analysis point passes).
    """
    theta_hat = np.asarray(theta_hat, dtype=np.float64).reshape(-1)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta_hat.shape != theta.shape:
        raise ValueError("theta_hat and theta must have the same length")
    n = len(theta)
    if n == 0:
        raise ValueError("no points")
    fin = np.isfinite(theta_hat)
    err = np.full(n, np.inf)
    err[fin] = circular_error(theta_hat[fin], theta[fin])
    ok = err < tol
    fe = err[fin]
    return dict(frac_ok=float(ok.mean()), all_ok=bool(ok.all()), n=int(n), n_bad=int((~ok).sum()),
                n_nonfinite=int((~fin).sum()), tol=float(tol),
                max_err=float(err.max()), q99_err=float(np.quantile(fe, 0.99)) if fe.size else np.inf,
                median_err=float(np.median(fe)) if fe.size else np.inf)


def angle_from_cs(cs) -> np.ndarray:
    """Decoded angle atan2(sin, cos) from [N, 2] (cos, sin) outputs (angle_cs observations)."""
    cs = np.asarray(cs, dtype=np.float64)
    return np.arctan2(cs[..., 1], cs[..., 0])


def nearest_frame_angle(frames, bank, bank_angles, batch: int = 1024) -> np.ndarray:
    """Angle of the nearest (Euclidean) frame of ``bank`` [M, ...] to each frame of ``frames`` [N, ...].

    ``bank`` holds rendered frames at known angles ``bank_angles`` [M] (e.g. a fine theta grid rendered with the
    world's renderer). Works in batches so memory stays O(batch * M)."""
    F = np.asarray(frames, dtype=np.float64).reshape(len(frames), -1)
    B = np.asarray(bank, dtype=np.float64).reshape(len(bank), -1)
    ang = np.asarray(bank_angles, dtype=np.float64)
    if F.shape[1] != B.shape[1] or len(ang) != len(B):
        raise ValueError("frames and bank must have the same frame size, bank_angles one entry per bank frame")
    bsq = np.einsum("ij,ij->i", B, B)
    out = np.empty(len(F))
    for s in range(0, len(F), batch):
        f = F[s:s + batch]
        d2 = np.einsum("ij,ij->i", f, f)[:, None] - 2.0 * f @ B.T + bsq[None]
        out[s:s + batch] = ang[np.argmin(d2, axis=1)]
    return out
