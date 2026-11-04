"""H1 instruments for cell 0: does a point cloud (latents, or a ground-truth embedding) carry the S^1 x R loop?

Pipeline (the calibrated choices are in ``cells/0/calibration.md``):

    Z [N, D]  --(optional) pca_whiten (fit on the full cloud)
              --> subsample n points: fps_subsample | random_subsample
              --> h1_diagram: Vietoris-Rips persistence, H1 (ripser), adaptive filtration threshold
              --> criteria: bs_test (Bobrowski & Skraba 2023), count rules on pi = death/birth, stability over
                  R subsamples
              --> winding: circular coordinate of the top class (persistent cohomology) vs. the true angle.

Conventions
-----------
* Rips filtration value of an edge = Euclidean length (ripser's convention; Bobrowski & Skraba 2023 use the
  same "all points within distance r"). Every criterion here is a function of the ratios pi = death / birth,
  which are invariant under global rescaling, so ``normalize_scale`` does not change any decision; it only
  makes absolute numbers (thresholds, additive persistence) comparable between clouds.
* ``h1_diagram`` does not always compute the whole filtration (that costs 5-20 s at n = 1000). It starts at
  T0 = t0_factor * q99(10-NN distance), above which no noise bar of the calibration controls died, and raises
  the threshold until every bar is either exact or certified to have pi >= ``pi_need`` (censored bars carry
  the lower bound pi >= T / birth). Decisions that only compare pi with values <= pi_need are therefore exact.
* Bobrowski & Skraba (Sci. Rep. 13, 12274, 2023; https://www.nature.com/articles/s41598-023-37842-2), Eqs. 1-4:
  l(p) = A log log(pi(p)) + B, A = 1 (Rips) or 1/2 (Cech), B = -euler_gamma - A * mean_dgm(log log pi);
  under the null l ~ LGumbel, CDF 1 - exp(-exp(x)), so p-value = exp(-exp(l)); a bar is signal if
  p < alpha / |dgm| (Bonferroni). The paper states the law for i.i.d. samples and notes (Remark after
  Conjecture 2) that repulsive processes / perturbed lattices do NOT follow it; farthest-point subsamples are
  such a process (see calibration.md for the measured breakdown).

Nothing here may be applied to trained-model latents before the cell-0 preregistration is approved.
"""
from __future__ import annotations

import time
from typing import Any, Optional, Sequence

import numpy as np
from ripser import ripser
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

EULER_GAMMA = float(np.euler_gamma)
BS_A = {"rips": 1.0, "cech": 0.5}       # Bobrowski & Skraba 2023, Eq. 2
_MIN_LOGPI = 1e-6                       # bars with log(pi) below this are numerical ties (pi ~ 1); ignored


# ----------------------------------------------------------------------------------------------- subsampling
def random_subsample(X, n: int, seed) -> np.ndarray:
    """Indices of n points drawn uniformly without replacement (sorted). All points if n >= N."""
    N = len(X)
    if n >= N:
        return np.arange(N)
    return np.sort(np.random.default_rng(seed).choice(N, size=n, replace=False))


def fps_subsample(X, n: int, seed, return_radius: bool = False):
    """Farthest-point (greedy max-min) subsample of n points; the first point is drawn with ``seed``.

    Returns indices in selection order (and the covering radius max_x min_i |x - x_i| of the selected net if
    ``return_radius``). Stops early if fewer than n distinct points exist. O(n N D) time, O(N) memory.
    """
    X = np.asarray(X, dtype=np.float64)
    N = len(X)
    if n >= N:
        idx = np.arange(N)
        return (idx, 0.0) if return_radius else idx
    sq = np.einsum("ij,ij->i", X, X)
    idx = np.empty(n, dtype=np.int64)
    idx[0] = np.random.default_rng(seed).integers(N)
    d2 = np.maximum(sq - 2.0 * (X @ X[idx[0]]) + sq[idx[0]], 0.0)
    m = n
    for i in range(1, n):
        j = int(np.argmax(d2))
        if d2[j] <= 0.0:                      # only duplicates left
            m = i
            break
        idx[i] = j
        np.minimum(d2, np.maximum(sq - 2.0 * (X @ X[j]) + sq[j], 0.0), out=d2)
    idx = idx[:m]
    return (idx, float(np.sqrt(d2.max()))) if return_radius else idx


# ----------------------------------------------------------------------------------------------- geometry
def knn_distances(X, k: int = 10) -> np.ndarray:
    """Distance of every point to its k-th nearest other point."""
    X = np.asarray(X, dtype=np.float64)
    k = min(k, len(X) - 1)
    return cKDTree(X).query(X, k=k + 1)[0][:, k]


def median_knn_distance(X, k: int = 10, max_points: int = 4000, seed=0) -> float:
    X = np.asarray(X, dtype=np.float64)
    if len(X) > max_points:
        X = X[random_subsample(X, max_points, seed)]
    return float(np.median(knn_distances(X, k)))


def normalize_scale(X, k: int = 10, seed=0) -> np.ndarray:
    """Divide by the median k-NN distance (scale-free units). No effect on any pi-based decision."""
    return np.asarray(X, dtype=np.float64) / median_knn_distance(X, k, seed=seed)


def pca_whiten(X, k: Optional[int] = None, eps: float = 1e-12):
    """Project on the top-k principal components and scale each to unit variance (k = None: all).

    Fit on the full cloud before subsampling. Whitening all components makes every criterion invariant under
    affine maps of the cloud (so it undoes linear thinning), but it also inflates low-variance noise directions.
    Returns (Xw [N, k], info) with info = {"var": variances of the kept PCs, "frac": explained fraction}.
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2 / max(len(X) - 1, 1)
    k = len(var) if k is None else int(min(k, len(var)))
    Xw = Xc @ Vt[:k].T / np.sqrt(np.maximum(var[:k], eps))
    return Xw, {"var": var[:k], "frac": float(var[:k].sum() / max(var.sum(), eps))}


def unique_rows(Y) -> np.ndarray:
    """Indices of the first occurrence of every distinct row, in the original order."""
    _, first = np.unique(np.asarray(Y), axis=0, return_index=True)
    return np.sort(first)


def _enclosing_radius(Y) -> float:
    """min_i max_j |y_i - y_j|: beyond it the Rips complex is a cone, so every finite bar has died."""
    return float(cdist(Y, Y).max(1).min())


# ----------------------------------------------------------------------------------------------- B&S null
def lgumbel_cdf(x):
    """Left-skewed Gumbel CDF 1 - exp(-exp(x)) (Bobrowski & Skraba 2023, Eq. 3); its mean is -euler_gamma."""
    return -np.expm1(-np.exp(np.asarray(x, dtype=np.float64)))


def bs_ell(pi, A: float = 1.0):
    """l-values A loglog(pi) + B with B = -gamma - A mean(loglog pi) over all given bars (ties pi ~ 1 dropped).

    Returns (ell (-inf for dropped ties), B, m = number of bars used).
    """
    pi = np.asarray(pi, dtype=np.float64)
    lp = np.log(pi)
    keep = lp > _MIN_LOGPI
    L = np.log(lp[keep])
    B = -EULER_GAMMA - A * L.mean() if L.size else np.nan
    ell = np.full(pi.shape, -np.inf)
    ell[keep] = A * L + B
    return ell, B, int(keep.sum())


def bs_pi_star(pi, alpha: float = 0.05, A: float = 1.0) -> float:
    """Smallest pi that is significant at level alpha (Bonferroni over the bars of this diagram)."""
    _, B, m = bs_ell(pi, A)
    if m == 0:
        return np.inf
    xstar = np.log(np.log(m / alpha))
    return float(np.exp(np.exp((xstar - B) / A)))


# ----------------------------------------------------------------------------------------------- diagram
def h1_diagram(Y, pi_need: float = 6.0, bs_alpha: Optional[float] = None, t0_factor: float = 2.0,
               t0_k: int = 10, max_passes: int = 6, coeff: int = 2, do_cocycles: bool = False) -> dict[str, Any]:
    """H1 Vietoris-Rips diagram of a point cloud with an adaptive filtration threshold.

    Every returned bar is exact unless ``censored``; a censored bar (alive at the final threshold T) has
    pi >= T / birth >= pi_need, where pi_need = max(pi_need, B&S Bonferroni cut-off) if ``bs_alpha`` is set.
    Bars born after T are not computed (no noise bar of the calibration controls was born after T0).
    Exact duplicate points are removed first (``keep`` = indices of the retained rows of Y, original order;
    ``points`` = Y[keep], the vertex set the cocycles refer to).

    Returns dict(birth, death (inf if censored), censored, pi (lower bound if censored), thresh, passes,
    seconds, n, cocycles (list, only with do_cocycles), coeff, points, keep).
    """
    t_start = time.perf_counter()
    Y = np.asarray(Y, dtype=np.float64)
    keep = unique_rows(Y)
    Y = Y[keep]
    n = len(Y)
    if n < 4:
        e = np.zeros(0)
        return dict(birth=e, death=e, censored=e.astype(bool), pi=e, thresh=np.inf, passes=0,
                    seconds=time.perf_counter() - t_start, n=n, cocycles=[], coeff=coeff, points=Y, keep=keep)
    T = float(t0_factor * np.quantile(knn_distances(Y, t0_k), 0.99))
    T_enc = None
    passes = 0
    while True:
        passes += 1
        res = ripser(Y, maxdim=1, thresh=T, coeff=coeff, do_cocycles=do_cocycles)
        dg = res["dgms"][1]
        b, d = dg[:, 0].astype(np.float64), dg[:, 1].astype(np.float64)
        cens = ~np.isfinite(d)
        if not cens.any() or not np.isfinite(T) or passes >= max_passes:
            break
        pi_lb = np.where(cens, T / np.maximum(b, 1e-300), d / np.maximum(b, 1e-300))
        need = pi_need
        if bs_alpha is not None:
            need = max(need, bs_pi_star(pi_lb, bs_alpha))
        und = cens & (pi_lb < need)
        if not und.any():
            break
        T_new = 1.05 * need * float(b[und].max())
        if T_enc is None:
            T_enc = _enclosing_radius(Y)
        T = np.inf if T_new >= T_enc else max(T_new, 1.05 * T)
    pi = np.where(cens, T / np.maximum(b, 1e-300), d / np.maximum(b, 1e-300))
    out = dict(birth=b, death=d, censored=cens, pi=pi, thresh=T, passes=passes,
               seconds=time.perf_counter() - t_start, n=n, coeff=coeff, points=Y, keep=keep)
    out["cocycles"] = res["cocycles"][1] if do_cocycles else []
    return out


def top_pi(dgm: dict, m: int = 4) -> np.ndarray:
    """The m largest pi values (descending), padded with 1.0."""
    p = np.sort(np.asarray(dgm["pi"], dtype=np.float64))[::-1][:m]
    return np.concatenate([p, np.ones(m - len(p))])


# ----------------------------------------------------------------------------------------------- criteria
def bs_test(dgm: dict, alpha: float = 0.05, A: float = BS_A["rips"]) -> dict[str, Any]:
    """Bobrowski & Skraba (2023) per-bar p-values with Bonferroni correction.

    Censored bars use their lower bound pi >= T/birth, i.e. an upper bound on their p-value; with
    ``h1_diagram(..., bs_alpha=alpha)`` the bound already decides significance.
    Returns dict(n_sig, p (per bar), ell, B, m (#bars used), pi_star (cut-off pi)).
    """
    pi = np.asarray(dgm["pi"], dtype=np.float64)
    ell, B, m = bs_ell(pi, A)
    p = np.exp(-np.exp(ell))                     # = 1 for dropped ties (ell = -inf)
    if m == 0:
        return dict(n_sig=0, p=p, ell=ell, B=B, m=0, pi_star=np.inf)
    sig = p < alpha / m
    xstar = np.log(np.log(m / alpha))
    return dict(n_sig=int(sig.sum()), p=p, ell=ell, B=float(B), m=m,
                pi_star=float(np.exp(np.exp((xstar - B) / A))))


def count_pi_threshold(pis_desc, tau: float) -> int:
    """b1_hat = #{bars with pi >= tau}."""
    return int((np.asarray(pis_desc) >= tau).sum())


def count_pi_gap(pis_desc, gap: float, max_classes: int = 3, floor: float = 1.0) -> int:
    """b1_hat = argmax_m pi_(m) / pi_(m+1) over m <= max_classes if that ratio >= gap and pi_(m) >= floor, else 0.

    Scale-free: compares bars only with each other (pis_desc padded with 1.0 by ``top_pi``).
    """
    p = np.asarray(pis_desc, dtype=np.float64)
    p = np.concatenate([p, np.ones(max(0, max_classes + 1 - len(p)))])
    ratios = p[:max_classes] / np.maximum(p[1:max_classes + 1], 1e-300)
    m = int(np.argmax(ratios))
    return m + 1 if (ratios[m] >= gap and p[m] >= floor) else 0


def count_pi_threshold_gap(pis_desc, tau: float, gap: float) -> int:
    """b1_hat = m = #{pi >= tau}; a positive claim (m >= 1) also needs pi_(m) / pi_(m+1) >= gap, else -1 (ambiguous)."""
    p = np.asarray(pis_desc, dtype=np.float64)
    m = int((p >= tau).sum())
    if m == 0:
        return 0
    nxt = p[m] if m < len(p) else 1.0
    return m if p[m - 1] / max(nxt, 1e-300) >= gap else -1


def stability(b1_hats: Sequence[int], target: int) -> float:
    """Fraction of subsamples whose criterion reports exactly ``target`` classes."""
    b = np.asarray(b1_hats)
    return float(np.mean(b == target)) if b.size else np.nan


def stable_count(b1_hats: Sequence[int], tau_s: float) -> int:
    """Report m if at least a fraction tau_s (> 0.5) of the subsamples report m, else -1 (unstable)."""
    b = np.asarray(b1_hats)
    if b.size == 0:
        return -1
    vals, cnt = np.unique(b, return_counts=True)
    j = int(np.argmax(cnt))
    return int(vals[j]) if (cnt[j] / b.size >= tau_s and vals[j] >= 0) else -1


# ----------------------------------------------------------------------------------------------- winding
def circular_coordinate(dgm: dict, which: int = 0, r_frac: float = 0.5) -> dict[str, Any]:
    """Circle-valued coordinate of an H1 class (de Silva, Morozov & Vejdemo-Johansson 2011).

    ``dgm`` must come from ``h1_diagram(..., do_cocycles=True)`` with a prime ``coeff`` >= 3 (the integer lift of
    a Z_2 cocycle is ambiguous). ``which`` indexes the bars in descending pi. The cocycle is lifted to integers,
    restricted to the Rips complex at r = birth + r_frac * (min(death, T) - birth), and smoothed by least squares
    (alpha - delta f -> harmonic); phi = 2 pi (f mod 1). Returns dict(phi [n], r, n_edges, ...) on the vertices
    of the diagram (``dgm["points"]``).
    """
    p = int(dgm["coeff"])
    if p < 3:
        raise ValueError("circular coordinates need a prime coefficient field p >= 3")
    order = np.argsort(-np.asarray(dgm["pi"]))
    i_bar = int(order[which])
    b, d = float(dgm["birth"][i_bar]), float(dgm["death"][i_bar])
    top = d if np.isfinite(d) else float(dgm["thresh"]) if np.isfinite(dgm["thresh"]) else 2.0 * b
    r = b + r_frac * (top - b)
    Y = dgm["points"]
    n = len(Y)
    pairs = np.sort(cKDTree(Y).query_pairs(r, output_type="ndarray").astype(np.int64), axis=1)  # i < j, |.| <= r
    pkey = pairs[:, 0] * n + pairs[:, 1]
    srt = np.argsort(pkey)
    pairs, pkey = pairs[srt], pkey[srt]
    coc = np.asarray(dgm["cocycles"][i_bar])
    val = coc[:, 2].astype(np.int64) % p
    val[val > (p - 1) // 2] -= p
    ii, jj = coc[:, 0].astype(np.int64), coc[:, 1].astype(np.int64)
    # ripser lists every edge with its vertices in decreasing order, so reading the value on the edge oriented
    # from the smaller to the larger index is one consistent orientation (a global flip only reverses phi).
    key = np.minimum(ii, jj) * n + np.maximum(ii, jj)
    pos = np.clip(np.searchsorted(pkey, key), 0, max(len(pkey) - 1, 0))
    ok = (pkey[pos] == key) if len(pkey) else np.zeros(len(key), bool)
    alpha = np.zeros(len(pairs))
    alpha[pos[ok]] = val[ok]
    E = len(pairs)
    rows = np.repeat(np.arange(E), 2)
    cols = pairs.reshape(-1)
    vals = np.tile([-1.0, 1.0], E)
    delta = sparse.csr_matrix((vals, (rows, cols)), shape=(E, n))
    f = lsqr(delta, alpha, atol=1e-10, btol=1e-10, iter_lim=20 * n)[0]
    phi = 2.0 * np.pi * np.mod(f, 1.0)
    return dict(phi=phi, r=float(r), n_edges=int(E), cocycle_edges_found=int(ok.sum()), cocycle_size=int(len(coc)))


def fisher_lee(a, b) -> float:
    """Fisher & Lee (1983) circular-circular correlation in [-1, 1]; rotation invariant, +-1 iff b = +-a + c."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = len(a)
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    num = 4.0 * ((sa * sb).sum() * (ca * cb).sum() - (sa * cb).sum() * (ca * sb).sum())
    den = ((n * n - np.cos(2 * a).sum() ** 2 - np.sin(2 * a).sum() ** 2)
           * (n * n - np.cos(2 * b).sum() ** 2 - np.sin(2 * b).sum() ** 2))
    return float(num / np.sqrt(den)) if den > 0 else 0.0


def winding_degree(theta, phi, n_bins: int = 12, min_per_bin: int = 3) -> dict[str, Any]:
    """Degree of the map theta -> phi estimated from circular means of phi in theta bins.

    Returns dict(degree (int; 0 if some bin has < min_per_bin points, i.e. theta does not cover the circle),
    coverage (fraction of non-empty bins), min_resultant (smallest within-bin mean resultant length)).
    """
    theta = np.mod(np.asarray(theta, dtype=np.float64), 2 * np.pi)
    phi = np.asarray(phi, dtype=np.float64)
    k = np.minimum((theta / (2 * np.pi) * n_bins).astype(int), n_bins - 1)
    z = np.zeros(n_bins, dtype=complex)
    cnt = np.bincount(k, minlength=n_bins)
    np.add.at(z, k, np.exp(1j * phi))
    ok = cnt >= min_per_bin
    cover = float(np.mean(ok))
    R = np.abs(z) / np.maximum(cnt, 1)
    if cover < 1.0:
        return dict(degree=0, coverage=cover, min_resultant=float(R[ok].min()) if ok.any() else 0.0)
    ang = np.angle(z)
    steps = np.angle(np.exp(1j * np.diff(np.concatenate([ang, ang[:1]]))))
    return dict(degree=int(np.rint(steps.sum() / (2 * np.pi))), coverage=cover, min_resultant=float(R.min()))


def winding_check(dgm: dict, theta_of_points, which: int = 0, r_frac: float = 0.5) -> dict[str, Any]:
    """Circular coordinate of a class vs. the true angle of the same points (analysis-only ground truth).

    ``theta_of_points`` are the angles of the rows of the cloud passed to ``h1_diagram`` (aligned through
    ``dgm["keep"]`` here, so duplicates removed there are handled).
    """
    cc = circular_coordinate(dgm, which=which, r_frac=r_frac)
    th = np.asarray(theta_of_points, dtype=np.float64)[dgm["keep"]]
    rho = fisher_lee(cc["phi"], th)
    deg = winding_degree(th, cc["phi"])
    return dict(rho=rho, abs_rho=abs(rho), degree=deg["degree"], coverage=deg["coverage"],
                min_resultant=deg["min_resultant"], r=cc["r"], n_edges=cc["n_edges"],
                cocycle_edges_found=cc["cocycle_edges_found"], cocycle_size=cc["cocycle_size"])


# ----------------------------------------------------------------------------------------------- one-call API
def assess_h1(Z, n: int = 600, method: str = "fps", seed=0, whiten: Optional[int] = None, pi_need: float = 6.0,
              bs_alpha: Optional[float] = 0.05, coeff: int = 2) -> dict[str, Any]:
    """Convenience: (optional whitening; whiten=0 means all PCs) -> subsample -> diagram -> top pi + B&S result."""
    t0 = time.perf_counter()
    X = np.asarray(Z, dtype=np.float64)
    if whiten is not None:
        X = pca_whiten(X, None if whiten <= 0 else whiten)[0]
    idx = fps_subsample(X, n, seed) if method == "fps" else random_subsample(X, n, seed)
    dg = h1_diagram(X[idx], pi_need=pi_need, bs_alpha=bs_alpha, coeff=coeff)
    bs = bs_test(dg, alpha=0.05 if bs_alpha is None else bs_alpha)
    return dict(top_pi=top_pi(dg), bs_n_sig=bs["n_sig"], bs_pi_star=bs["pi_star"], diagram=dg, index=idx,
                seconds=time.perf_counter() - t0)
