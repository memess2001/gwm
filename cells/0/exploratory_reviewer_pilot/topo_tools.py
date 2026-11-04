"""Small topology instruments: FPS, VR H1 via ripser, circular coordinate of the top H1 class, degree vs true theta."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from ripser import ripser


def fps(X, m, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = [int(rng.integers(n))]
    d = np.linalg.norm(X - X[idx[0]], axis=1)
    for _ in range(m - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return np.array(idx), float(d.max())          # indices, covering radius


def h1(X, p=47):
    r = ripser(X, maxdim=1, coeff=p, do_cocycles=True)
    dgm = r["dgms"][1]
    order = np.argsort(-(dgm[:, 1] - dgm[:, 0])) if len(dgm) else np.array([], int)
    return dgm[order], [r["cocycles"][1][i] for i in order], r["dgms"][0]


def circular_coordinate(X, cocycle, birth, death, p=47, frac=0.5):
    """Perea-style circular coordinate from a ripser cocycle; returns angle per point in [0, 2pi)."""
    r = birth + frac * (death - birth)
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    n = len(X)
    I, J = np.where(np.triu(D < r, 1))
    # integer lift of the cocycle
    alpha = {}
    for a, b, v in cocycle:
        v = int(v)
        if v > (p - 1) // 2:
            v -= p
        a, b = int(a), int(b)
        if a < b:
            alpha[(a, b)] = alpha.get((a, b), 0) + v
        else:
            alpha[(b, a)] = alpha.get((b, a), 0) - v
    al = np.array([alpha.get((i, j), 0) for i, j in zip(I, J)], float)
    m = len(I)
    rows = np.concatenate([np.arange(m), np.arange(m)])
    cols = np.concatenate([J, I])
    vals = np.concatenate([np.ones(m), -np.ones(m)])
    delta = sp.csr_matrix((vals, (rows, cols)), shape=(m, n))
    tau = lsqr(delta, al, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    # connected-component check of the r-graph
    ncomp = sp.csgraph.connected_components(sp.csr_matrix((np.ones(m), (I, J)), shape=(n, n)), directed=False)[0]
    return np.mod(-tau, 1.0) * 2 * np.pi, ncomp


def degree_vs_theta(ang, theta, nbins=36):
    """Winding number of the circular coordinate as true theta goes once around; plus phase-lock |mean e^{i(ang - s theta)}|."""
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    k = np.clip(np.digitize(theta, bins) - 1, 0, nbins - 1)
    means = []
    for b in range(nbins):
        sel = k == b
        means.append(np.angle(np.exp(1j * ang[sel]).mean()) if sel.any() else np.nan)
    means = np.array(means)
    ok = ~np.isnan(means)
    mm = means[ok]
    if len(mm) < nbins * 0.8:
        deg = np.nan
    else:
        steps = np.angle(np.exp(1j * (np.roll(mm, -1) - mm)))
        deg = steps.sum() / (2 * np.pi)
    lock = max(abs(np.exp(1j * (ang - s * theta)).mean()) for s in (1, -1))
    return deg, lock


def theta_winding_along(ang, theta, nbins=24):
    """Winding of TRUE theta along the loop parametrised by the circular coordinate (wrapping test)."""
    bins = np.linspace(0, 2 * np.pi, nbins + 1)
    k = np.clip(np.digitize(ang, bins) - 1, 0, nbins - 1)
    means = []
    for b in range(nbins):
        sel = k == b
        if sel.any():
            means.append(np.angle(np.exp(1j * theta[sel]).mean()))
    if len(means) < 0.8 * nbins:
        return np.nan
    mm = np.array(means)
    return np.angle(np.exp(1j * (np.roll(mm, -1) - mm))).sum() / (2 * np.pi)


def analyse_cloud(Z, theta, m=400, seed=0):
    idx, cover = fps(Z, m, seed)
    X = Z[idx]
    dgm, cocs, dgm0 = h1(X)
    out = dict(cover=cover, bars=dgm[:3].tolist())
    if len(dgm):
        b, d = dgm[0]
        ang, ncomp = circular_coordinate(X, cocs[0], b, d)
        deg, lock = degree_vs_theta(ang, theta[idx])
        out.update(deg=deg, lock=lock, ncomp=ncomp, wind=theta_winding_along(ang, theta[idx]))
    else:
        out.update(deg=np.nan, lock=np.nan, ncomp=np.nan, wind=np.nan)
    return out
