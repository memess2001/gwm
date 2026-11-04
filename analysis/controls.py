"""Control point clouds with known answers for calibrating the cell-0 instruments.

Everything here starts from ground-truth pendulum STATES drawn from ``zoo.make("pendulum")`` and maps them
through embeddings whose topology / energy-smoothness is known by construction. These are instrument
calibration data only: no trained-model latents are involved anywhere.

Known answers
-------------
* Topology of the sampled region comes from the zoo (``ground_truth()``: ``region_betti`` for trajectory
  samples, ``init_region_betti`` for i.i.d. box samples, plus the number of classes that wrap the cylinder).
* Embeddings (injective, continuous) keep it: ``emb_ref``, ``emb_thin``, ``emb_aniso``, ``RandomSmoothEmbedding``.
* Non-injective maps change it in a known way: ``emb_torn`` (theta cut open at +-pi: b1 -> 0 for the full
  annulus), ``emb_folded`` ((cos theta, omega): the circle is folded onto [-1, 1], b1 -> 0).
* Partial gluing (seam-ratio calibration): ``emb_arc`` lays theta on an arc of angle 2 pi alpha, leaving a gap
  ``arc_gap(alpha)`` across theta = +-pi (alpha = 1 glued, alpha -> 0 the torn chart); ``emb_arc_omega`` makes
  alpha depend on |omega| (glued only for fast or only for slow rotations).
* Energy E = omega^2/2 - omega0^2 cos(theta) is invariant under theta -> -theta and omega -> -omega, so folds
  along those symmetries (``emb_fold_theta``, ``emb_fold_omega``, ``emb_folded``) keep E a smooth function of
  the embedding even though they destroy injectivity (and, for the first and third, the loop).
* Review controls: ``emb_fold_abs`` ((|theta|, omega): glued across the cut but folded, b1 = 0) and
  ``emb_partial_fold`` (theta -> -theta folded only for the low-energy states; graded theta-gate control).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

import zoo
from zoo.pendulum import energy as pendulum_energy

PRE_REG_WORLD = dict(omega_max=2.5)       # cell-0 world: omega0 = 1, gamma = 0, dt = 0.1, init = "box" (defaults)


# ----------------------------------------------------------------------------------------------- states
def pendulum_states(seed: int, sampling: str = "traj", n_traj: int = 500, horizon: int = 64,
                    **window) -> tuple[np.ndarray, dict[str, Any]]:
    """Ground-truth states [N, 2] = (theta wrapped, omega) and the known answers for their support.

    sampling = "traj": n_traj conservative trajectories of ``horizon`` steps from i.i.d. box initial states,
    keeping t = 1..horizon (the cell-0 analysis set uses z_t for t = 1..64, 500 trajectories -> 32,000 points).
    sampling = "box": n_traj * horizon i.i.d. initial states (uniform in the box intersected with the energy
    window). ``window`` = pendulum parameters (E_lo, E_hi, omega_max, ...).
    """
    w = zoo.make("pendulum")
    params = {**PRE_REG_WORLD, **window}
    if sampling == "traj":
        tr = w.sample(seed=seed, horizon=horizon, n_traj=n_traj, obs_mode="state", **params)
        S = tr.state[:, 1:, :].reshape(-1, 2).astype(np.float64)
        gt = w.ground_truth(**tr.meta["params"])
        betti, wrap = gt["region_betti"], gt["region_b1_wrapping"]
    elif sampling == "box":
        tr = w.sample(seed=seed, horizon=0, n_traj=n_traj * horizon, obs_mode="state", **params)
        S = tr.state[:, 0, :].astype(np.float64)
        gt = w.ground_truth(**tr.meta["params"])
        betti, wrap = gt["init_region_betti"], gt["init_region_b1_wrapping"]
    else:
        raise ValueError("sampling must be 'traj' or 'box'")
    E = pendulum_energy(S, tr.meta["params"]["omega0"])
    return S, dict(betti=list(betti), b1=int(betti[1]), wrap=int(wrap), energy=E, params=tr.meta["params"])


def energy_of(S, omega0: float = 1.0) -> np.ndarray:
    return pendulum_energy(S, omega0)


# ----------------------------------------------------------------------------------------------- elementary maps
def emb_ref(S, s: float = 1.0) -> np.ndarray:
    """Reference embedding (cos theta, sin theta, omega / s) of S^1 x R into R^3."""
    return np.stack([np.cos(S[:, 0]), np.sin(S[:, 0]), S[:, 1] / s], 1)


def emb_thin(S, f: float) -> np.ndarray:
    """Thin loop: the circle directions shrunk by f relative to omega: (f cos, f sin, omega)."""
    return np.stack([f * np.cos(S[:, 0]), f * np.sin(S[:, 0]), S[:, 1]], 1)


def emb_aniso(S, f: float) -> np.ndarray:
    """Elliptic cross-section: (cos theta, f sin theta, omega)."""
    return np.stack([np.cos(S[:, 0]), f * np.sin(S[:, 0]), S[:, 1]], 1)


def emb_helix(S, f: float = 1.0, A: float = 2.0, kappa: float = 1.0) -> np.ndarray:
    """(f cos theta, f sin theta, omega, A cos(kappa omega), A sin(kappa omega)) in R^5: the cylinder bent along a
    helix. Injective (omega is a coordinate). The helix carries most of the variance, so for A = 2 the loop is
    NOT among the top-3 principal components (a smooth embedding on which top-k PCA whitening fails)."""
    w = S[:, 1]
    return np.stack([f * np.cos(S[:, 0]), f * np.sin(S[:, 0]), w, A * np.cos(kappa * w), A * np.sin(kappa * w)], 1)


def emb_torn(S) -> np.ndarray:
    """Cut chart (theta in [-pi, pi), omega): the loop is torn open at theta = +-pi."""
    return np.asarray(S, dtype=np.float64).copy()


def emb_folded(S) -> np.ndarray:
    """(cos theta, omega): theta and -theta coincide, the circle is folded onto [-1, 1] (kills the loop)."""
    return np.stack([np.cos(S[:, 0]), S[:, 1]], 1)


def emb_fold_theta(S) -> np.ndarray:
    """(cos theta, |sin theta|, omega): fold along the energy symmetry theta -> -theta (kills the loop)."""
    return np.stack([np.cos(S[:, 0]), np.abs(np.sin(S[:, 0])), S[:, 1]], 1)


def emb_fold_omega(S) -> np.ndarray:
    """(cos theta, sin theta, |omega|): fold along the energy symmetry omega -> -omega (keeps the loop)."""
    return np.stack([np.cos(S[:, 0]), np.sin(S[:, 0]), np.abs(S[:, 1])], 1)


def emb_fold_abs(S) -> np.ndarray:
    """(|theta|, omega): the torn chart folded at theta = 0, so theta and -theta coincide. Continuous across the
    cut at +-pi (both sides map to |theta| ~ pi) but non-injective: a latent that is "glued but folded" (b1 = 0)."""
    return np.stack([np.abs(S[:, 0]), S[:, 1]], 1)


def emb_partial_fold(S, E, E_fold: float, width: float = 0.1) -> np.ndarray:
    """(cos theta, s sin theta + (1 - s) |sin theta|, omega) with s = smoothstep of E over [E_fold - width,
    E_fold + width]: theta -> -theta is folded (s = 0) for the states below E_fold - width (the low-energy
    librations) and kept (s = 1) above E_fold + width; injective wherever s > 0. Choosing E_fold as the q-quantile
    of E folds a fraction ~ q of the states (graded control for the theta-decodability gate)."""
    th, om = S[:, 0], S[:, 1]
    u = np.clip((np.asarray(E, dtype=np.float64) - (E_fold - width)) / (2.0 * width), 0.0, 1.0)
    s = u * u * (3.0 - 2.0 * u)
    return np.stack([np.cos(th), s * np.sin(th) + (1.0 - s) * np.abs(np.sin(th)), om], 1)


def emb_sin_omega(S, k: float) -> np.ndarray:
    """(cos theta, sin theta, sin(k omega)): omega is replaced by a non-injective oscillation; E is no longer a
    function of the embedding (|omega| is ambiguous)."""
    return np.stack([np.cos(S[:, 0]), np.sin(S[:, 0]), np.sin(k * S[:, 1])], 1)


def emb_wiggle(S, k: float, a: float = 0.9) -> np.ndarray:
    """(cos, sin, omega + a sin(k omega)/k): injective for a < 1 and smooth, but E has a wildly varying gradient."""
    w = S[:, 1]
    return np.stack([np.cos(S[:, 0]), np.sin(S[:, 0]), w + a * np.sin(k * w) / k], 1)


def _arc_xy(theta, alpha):
    """(sin(a theta)/a, (1 - cos(a theta))/a), written with sinc so that a -> 0 is the straight line (theta, 0)."""
    x = theta * np.sinc(alpha * theta / np.pi)
    y = 0.5 * alpha * theta ** 2 * np.sinc(alpha * theta / (2 * np.pi)) ** 2
    return x, y


def emb_arc(S, alpha: float) -> np.ndarray:
    """Partially glued chart: theta in [-pi, pi) is laid on a circular arc of angle 2 pi alpha (radius 1/alpha,
    arc length = theta, so the map is isometric in theta), times omega. alpha = 1 is a full circle (glued, a rigid
    motion of the reference chart), alpha -> 0 is the torn chart (theta, 0, omega). The two sides of the cut are
    2 sin(pi alpha) / alpha apart (``arc_gap``); for 0 < alpha < 1 the map is injective and the support is torn."""
    x, y = _arc_xy(np.asarray(S[:, 0], dtype=np.float64), float(alpha))
    return np.stack([x, y, S[:, 1]], 1)


def arc_gap(alpha: float) -> float:
    """Distance between the images of theta = pi and theta = -pi (same omega) under ``emb_arc``."""
    return float(2.0 * np.pi * np.sinc(alpha))          # = 2 sin(pi alpha) / alpha, = 2 pi at alpha = 0


def emb_arc_omega(S, alpha_lo: float = 0.3, w_lo: float = 1.0, w_hi: float = 1.5, glue_fast: bool = True):
    """``emb_arc`` with an omega-dependent arc angle: alpha = 1 (glued) for |omega| >= w_hi and alpha = alpha_lo
    (torn) for |omega| <= w_lo if ``glue_fast``, the reverse otherwise; smoothstep in between. Injective (omega is
    a coordinate), continuous away from the cut: partial gluing that depends on the velocity."""
    om = np.asarray(S[:, 1], dtype=np.float64)
    u = np.clip((np.abs(om) - w_lo) / (w_hi - w_lo), 0.0, 1.0)
    s = u * u * (3.0 - 2.0 * u)
    alpha = alpha_lo + (1.0 - alpha_lo) * s if glue_fast else 1.0 - (1.0 - alpha_lo) * s
    x, y = _arc_xy(np.asarray(S[:, 0], dtype=np.float64), alpha)
    return np.stack([x, y, om], 1)


def rms_radius(Z) -> float:
    """sqrt(mean |z - mean z|^2) (= sqrt of the total variance)."""
    Z = np.asarray(Z, dtype=np.float64)
    return float(np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean()))


def add_noise(Z, frac: float, seed) -> np.ndarray:
    """Isotropic Gaussian noise whose RMS norm is ``frac`` times the cloud's RMS radius (per-coordinate
    sigma = frac * rms_radius / sqrt(D))."""
    Z = np.asarray(Z, dtype=np.float64)
    if frac <= 0:
        return Z.copy()
    sig = frac * rms_radius(Z) / np.sqrt(Z.shape[1])
    return Z + sig * np.random.default_rng(seed).standard_normal(Z.shape)


def patch_shuffle(S, Z, K: int, amp: float = 0.5, seed=0) -> np.ndarray:
    """Partition the state space (theta, omega) into K x K cells and translate each cell of the cloud Z by an
    independent uniform vector in [-amp, amp]^D * rms_radius(Z). Locally isometric inside cells, torn across
    cell boundaries, cells overlap after the shuffle."""
    Z = np.asarray(Z, dtype=np.float64)
    th, om = S[:, 0], S[:, 1]
    it = np.minimum(((th + np.pi) / (2 * np.pi) * K).astype(int), K - 1)
    lo, hi = om.min(), om.max()
    io = np.minimum(((om - lo) / max(hi - lo, 1e-12) * K).astype(int), K - 1)
    cell = it * K + io
    shift = np.random.default_rng(seed).uniform(-amp, amp, (K * K, Z.shape[1])) * rms_radius(Z)
    return Z + shift[cell]


def gaussian_blob(N: int, dim: int = 8, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((N, dim))


# ----------------------------------------------------------------------------------------------- random embeddings
class RandomSmoothEmbedding:
    """Random smooth embedding R^d_in -> R^d_out (a random tanh network that is provably injective).

    x -> standardise (centre, divide by the RMS radius of the cloud it is applied to) -> random isometric lift
    Q (d_out x d_in, orthonormal columns) -> ``n_blocks`` residual blocks y <- y + W2 tanh(W1 y + b) with
    ||W2|| ||W1|| = lip < 1, so each block is a bi-Lipschitz homeomorphism of R^d_out (i-ResNet argument,
    Behrmann et al. 2019) -> random linear map with singular values log-spaced in [1/cond, 1].
    The composite is an injective immersion, hence the Betti numbers (and wrapping) of the image equal those of
    the input support. ``gain`` sets the curvature scale of the tanh features (larger = more wiggly).
    """

    LEVELS = {"mild": dict(n_blocks=1, gain=1.0, lip=0.5, cond=1.0),
              "medium": dict(n_blocks=3, gain=2.0, lip=0.8, cond=3.0),
              "strong": dict(n_blocks=10, gain=1.0, lip=0.95, cond=30.0)}

    def __init__(self, d_in: int, d_out: int = 8, level: Optional[str] = "medium", width: int = 64, seed=0,
                 **override):
        cfg = dict(self.LEVELS[level]) if level is not None else {}
        cfg.update(override)
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.d_in, self.d_out = d_in, d_out
        self.Q = np.linalg.qr(rng.standard_normal((d_out, d_in)))[0]
        self.blocks = []
        for _ in range(int(cfg["n_blocks"])):
            W1 = cfg["gain"] * rng.standard_normal((width, d_out))
            b = cfg["gain"] * rng.standard_normal(width) * 0.7
            W2 = rng.standard_normal((d_out, width))
            W2 *= cfg["lip"] / (np.linalg.norm(W2, 2) * np.linalg.norm(W1, 2))
            self.blocks.append((W1, b, W2))
        U = np.linalg.qr(rng.standard_normal((d_out, d_out)))[0]
        V = np.linalg.qr(rng.standard_normal((d_out, d_out)))[0]
        sv = np.exp(np.linspace(0.0, -np.log(cfg["cond"]), d_out))
        self.A = U @ np.diag(rng.permutation(sv)) @ V.T

    def __call__(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        y = (X - X.mean(0)) / max(rms_radius(X), 1e-12)
        y = y @ self.Q.T
        for W1, b, W2 in self.blocks:
            y = y + np.tanh(y @ W1.T + b) @ W2.T
        return y @ self.A.T


def local_distortion(X, Z, k: int = 10, max_points: int = 4000, seed=0) -> dict[str, float]:
    """Spread of the local stretch |dz| / |dx| over k-NN edges of X (after normalising both clouds' RMS radius):
    q01, q50, q99 and q99/q01. 1 for an isometry (up to scale)."""
    from scipy.spatial import cKDTree
    X, Z = np.asarray(X, dtype=np.float64), np.asarray(Z, dtype=np.float64)
    if len(X) > max_points:
        sel = np.random.default_rng(seed).choice(len(X), max_points, replace=False)
        X, Z = X[sel], Z[sel]
    X, Z = X / rms_radius(X), Z / rms_radius(Z)
    _, nb = cKDTree(X).query(X, k=k + 1)
    i = np.repeat(np.arange(len(X)), k)
    j = nb[:, 1:].reshape(-1)
    dx = np.linalg.norm(X[i] - X[j], axis=1)
    dz = np.linalg.norm(Z[i] - Z[j], axis=1)
    ok = dx > 0
    r = dz[ok] / dx[ok]
    q01, q50, q99 = np.quantile(r, [0.01, 0.5, 0.99])
    return dict(q01=float(q01), q50=float(q50), q99=float(q99), spread=float(q99 / max(q01, 1e-300)))
