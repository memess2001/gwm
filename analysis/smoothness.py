"""Energy instruments for cell 0: is the true energy E decodable from a cloud Z, and is it a SMOOTH function of Z?

Probes (``n_splits``-fold CV on shuffled folds; R^2 = 1 - SSE/SST of the pooled out-of-fold predictions):
  * linear ridge (standardised features, ridge strength by internal efficient LOO, sklearn RidgeCV);
  * quadratic ridge (degree-2 polynomial features, same) - still a low-capacity, frozen readout;
  * kNN regression, k in {5, 10, 20}, Euclidean metric of Z as given;
  * RBF kernel ridge; bandwidth and ridge chosen per training fold by exact leave-one-out (eigendecomposition),
    bandwidth grid relative to the median squared pairwise distance of the training fold.
Smoothness on the directed kNN graph of Z (edges i -> j for the k nearest neighbours j of i):
  * D(Z, E) = mean_edges (E_i - E_j)^2 / Var(E);
  * D_perm  = permutation null: for a uniformly random relabelling of E, E[D] = 2 N / (N - 1) exactly
    (any graph without self-loops); ``n_perm`` > 0 adds an empirical permutation mean and sd;
  * D_ref   = D on the reference embedding (cos theta, sin theta, omega) of the SAME states, same k;
  * s = (D_perm - D_Z) / (D_perm - D_ref): 1 = as smooth as the reference chart, 0 = no better than shuffled;
  * Lipschitz tail ratio q99 / q50 of |E_i - E_j| / |z_i - z_j| over the same edges (scale free).
All scores use the metric of Z as given (``prep="whiten"`` first PCA-whitens Z), so they are NOT invariant
under anisotropic rescaling of the latent; calibration.md reports how much that matters.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.linalg import eigh
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

RIDGE_ALPHAS = np.logspace(-6, 4, 21)
KRR_GAMMAS = (0.03, 0.125, 0.5, 2.0, 8.0, 32.0)   # K = exp(-g * d^2 / median(d^2)); smooth E picks the wide end
KRR_ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)


def _prep(Z, prep: str = "raw") -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    if prep == "raw":
        return Z
    if prep == "whiten":
        from .topology import pca_whiten
        return pca_whiten(Z)[0]
    raise ValueError("prep must be 'raw' or 'whiten'")


def r2(y, yhat) -> float:
    y, yhat = np.asarray(y, dtype=np.float64), np.asarray(yhat, dtype=np.float64)
    sst = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yhat) ** 2).sum() / sst) if sst > 0 else np.nan


def _folds(n: int, n_splits: int, seed):
    return KFold(n_splits=n_splits, shuffle=True, random_state=int(np.random.default_rng(seed).integers(2 ** 31)))\
        .split(np.arange(n))


# ----------------------------------------------------------------------------------------------- probes
def ridge_probe_r2(Z, E, degree: int = 1, n_splits: int = 5, seed=0) -> float:
    """Out-of-fold R^2 of a ridge readout on standardised (degree-``degree`` polynomial) features of Z."""
    Z, E = np.asarray(Z, dtype=np.float64), np.asarray(E, dtype=np.float64)
    pred = np.empty_like(E)
    for tr, te in _folds(len(E), n_splits, seed):
        steps = ([PolynomialFeatures(degree, include_bias=False)] if degree > 1 else []) + \
                [StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS)]
        m = make_pipeline(*steps).fit(Z[tr], E[tr])
        pred[te] = m.predict(Z[te])
    return r2(E, pred)


def knn_probe_r2(Z, E, k: int = 10, n_splits: int = 5, seed=0) -> float:
    """Out-of-fold R^2 of uniform-weight kNN regression in the metric of Z."""
    Z, E = np.asarray(Z, dtype=np.float64), np.asarray(E, dtype=np.float64)
    pred = np.empty_like(E)
    for tr, te in _folds(len(E), n_splits, seed):
        pred[te] = KNeighborsRegressor(n_neighbors=k).fit(Z[tr], E[tr]).predict(Z[te])
    return r2(E, pred)


def _krr_select(D2, y, gammas: Sequence[float], alphas: Sequence[float]):
    """Exact LOO error of kernel ridge for every (gamma, alpha); returns best (g, a, eigvals, eigvecs, mean)."""
    med = float(np.median(D2[np.triu_indices_from(D2, 1)]))
    med = med if med > 0 else 1.0
    ym = y.mean()
    yc = y - ym
    best = (np.inf, None)
    for g in gammas:
        w, U = eigh(np.exp(-g * D2 / med))
        w = np.maximum(w, 0.0)
        Uy = U.T @ yc
        U2 = U * U
        for a in alphas:
            h = w / (w + a)
            yhat = U @ (h * Uy)
            hdiag = U2 @ h
            loo = (yc - yhat) / np.maximum(1.0 - hdiag, 1e-12)
            err = float(np.mean(loo ** 2))
            if err < best[0]:
                best = (err, (g, a, w, U, Uy, ym, med))
    return best[1]


def krr_probe_r2(Z, E, n_splits: int = 5, seed=0, max_n: int = 800, gammas: Sequence[float] = KRR_GAMMAS,
                 alphas: Sequence[float] = KRR_ALPHAS) -> dict[str, Any]:
    """Out-of-fold R^2 of RBF kernel ridge; (bandwidth, ridge) chosen on each training fold by exact LOO.

    Uses a random subset of ``max_n`` points (cost O(len(gammas) * n^3)). Returns dict(r2, gamma, alpha) where
    gamma/alpha are the most frequently selected values over folds.
    """
    Z, E = np.asarray(Z, dtype=np.float64), np.asarray(E, dtype=np.float64)
    if len(E) > max_n:
        sel = np.sort(np.random.default_rng(seed).choice(len(E), max_n, replace=False))
        Z, E = Z[sel], E[sel]
    pred = np.empty_like(E)
    chosen = []
    for tr, te in _folds(len(E), n_splits, seed):
        D2 = cdist(Z[tr], Z[tr], "sqeuclidean")
        g, a, w, U, Uy, ym, med = _krr_select(D2, E[tr], gammas, alphas)
        coef = U @ (Uy / (w + a))
        pred[te] = np.exp(-g * cdist(Z[te], Z[tr], "sqeuclidean") / med) @ coef + ym
        chosen.append((g, a))
    gs, as_ = zip(*chosen)
    mode = lambda v: max(set(v), key=v.count)
    return dict(r2=r2(E, pred), gamma=float(mode(list(gs))), alpha=float(mode(list(as_))))


# ----------------------------------------------------------------------------------------------- smoothness
def knn_edges(Z, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Directed kNN edges (i, j): j is one of the k nearest other points of i in the metric of Z."""
    Z = np.asarray(Z, dtype=np.float64)
    _, nb = cKDTree(Z).query(Z, k=k + 1)
    ii = np.broadcast_to(np.arange(len(Z))[:, None], nb.shape)
    keep = nb != ii                                    # drop self (robust to duplicates: then drop the last)
    keep &= np.cumsum(keep, axis=1) <= k
    return ii[keep], nb[keep]


def dirichlet_energy(Z, E, k: int = 10, edges=None) -> float:
    """Normalised kNN-graph Dirichlet energy mean_edges (E_i - E_j)^2 / Var(E)."""
    E = np.asarray(E, dtype=np.float64)
    i, j = knn_edges(Z, k) if edges is None else edges
    return float(np.mean((E[i] - E[j]) ** 2) / E.var())


def lipschitz_tail(Z, E, k: int = 10, q_hi: float = 0.99, q_lo: float = 0.5, edges=None) -> float:
    """q_hi / q_lo quantile ratio of the local slopes |E_i - E_j| / |z_i - z_j| over kNN edges."""
    Z, E = np.asarray(Z, dtype=np.float64), np.asarray(E, dtype=np.float64)
    i, j = knn_edges(Z, k) if edges is None else edges
    dz = np.linalg.norm(Z[i] - Z[j], axis=1)
    ok = dz > 0
    L = np.abs(E[i] - E[j])[ok] / dz[ok]
    lo, hi = np.quantile(L, [q_lo, q_hi])
    return float(hi / lo) if lo > 0 else np.inf


def smoothness_score(Z, E, Zref, k: int = 10, n_perm: int = 0, seed=0) -> dict[str, float]:
    """s = (D_perm - D_Z) / (D_perm - D_ref) plus the Lipschitz tail ratios of Z and of the reference."""
    E = np.asarray(E, dtype=np.float64)
    N = len(E)
    eZ, eR = knn_edges(Z, k), knn_edges(Zref, k)
    DZ, DR = dirichlet_energy(Z, E, edges=eZ), dirichlet_energy(Zref, E, edges=eR)
    Dp = 2.0 * N / (N - 1)
    out = dict(D_Z=DZ, D_ref=DR, D_perm=Dp, s=float((Dp - DZ) / (Dp - DR)),
               lip_tail=lipschitz_tail(Z, E, edges=eZ), lip_tail_ref=lipschitz_tail(Zref, E, edges=eR))
    if n_perm > 0:
        rng = np.random.default_rng(seed)
        vals = [dirichlet_energy(Z, E[rng.permutation(N)], edges=eZ) for _ in range(n_perm)]
        out.update(D_perm_emp=float(np.mean(vals)), D_perm_sd=float(np.std(vals)))
    return out


# ----------------------------------------------------------------------------------------------- one call
def energy_report(Z, E, Zref, ks: Sequence[int] = (5, 10, 20), n_splits: int = 5, seed=0, krr: bool = True,
                  krr_max_n: int = 800, prep: str = "raw", quadratic: bool = True) -> dict[str, Any]:
    """All energy scores for one cloud (Z, E) with reference chart Zref of the same states."""
    Zp = _prep(Z, prep)
    out: dict[str, Any] = dict(n=int(len(E)), prep=prep)
    out["linear_r2"] = ridge_probe_r2(Zp, E, 1, n_splits, seed)
    if quadratic:
        out["quad_r2"] = ridge_probe_r2(Zp, E, 2, n_splits, seed)
    for k in ks:
        out[f"knn{k}_r2"] = knn_probe_r2(Zp, E, k, n_splits, seed)
        sm = smoothness_score(Zp, E, Zref, k=k)
        out[f"s_k{k}"] = sm["s"]
        out[f"lip_tail_k{k}"] = sm["lip_tail"]
        out[f"lip_tail_ref_k{k}"] = sm["lip_tail_ref"]
        out[f"D_Z_k{k}"], out[f"D_ref_k{k}"] = sm["D_Z"], sm["D_ref"]
    if krr:
        kr = krr_probe_r2(Zp, E, n_splits, seed, max_n=krr_max_n)
        out.update(krr_r2=kr["r2"], krr_gamma=kr["gamma"], krr_alpha=kr["alpha"])
    return out
