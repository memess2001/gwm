# Alternative alpha-readings for cell 7 review (K = 3, Amari convention: alpha=+1 e, alpha=-1 m).
# lam = (1 - alpha)/2.  All estimators are point-set (shape) based unless noted.
import numpy as np
from scipy.integrate import solve_ivp

ONES = np.ones(3)


# ------------------------------------------------------------------ dynamics
def Fs(X, f):
    return X * (f - (X @ f)[..., None])


def Fm(X, m, mu):
    return mu * (m - X)


def Fw(X, f, m, mu, w):
    return w * Fs(X, f) + (1 - w) * Fm(X, m, mu)


def accel(X, f, m, mu, w):
    V = Fw(X, f, m, mu, w)
    fb = X @ f
    dFsV = V * (f - fb[..., None]) - X * (V @ f)[..., None]
    return w * dFsV - (1 - w) * mu * V, V


def simulate(x0, f, m, mu, w, T=2.0, n=40):
    sol = solve_ivp(lambda t, x: Fw(x[None], f, m, mu, w)[0], (0, T), x0, rtol=1e-11, atol=1e-13,
                    dense_output=True)
    X = sol.sol(np.linspace(0, T, n)).T
    return X / X.sum(1, keepdims=True)


def fnorm(V, X):
    return np.sqrt(np.sum(V * V / X, -1))


def rho_share(X, f, m, mu, w):
    ns = fnorm(w * Fs(X, f), X)
    nm = fnorm((1 - w) * Fm(X, m, mu), X)
    return ns.sum() / (ns.sum() + nm.sum() + 1e-300)


# ------------------------------------------------------------------ E0: original unweighted SVD ratio
LAMS0 = np.round(np.linspace(-1.5, 2.5, 401), 6)


def ratio(X, lam):
    if abs(lam) < 1e-12:
        Th = np.log(X[:, :-1] / X[:, -1:])
        Th = Th - Th.mean(0)
        s = np.linalg.svd(Th, compute_uv=False)
        return s[1] / s[0]
    s = np.linalg.svd(X ** lam, compute_uv=False)
    return np.sqrt((s[2:] ** 2).sum()) / s[1]


def est_svd(X, lams=LAMS0):
    r = np.array([ratio(X, l) for l in lams])
    i = int(np.argmin(r))
    return 1 - 2 * lams[i], r


# ------------------------------------------------------------------ E1: Fisher-weighted (Sampson) orthogonal fit
# alpha-geodesic for lam: {x : u . BoxCox_lam(x) = b} with u.1 + lam b = 0 (equivalent to u . x^lam = 0 for lam != 0,
# to a theta-line for lam = 0).  First-order Fisher (Shahshahani) distance of x_t to that curve:
#   d_t = |h(x_t)| / sqrt(Var_{x_t}(u o x_t^(lam-1))).  J(lam) = min_{u,b} sum_t d_t^2  (= large-n multinomial
#   profile likelihood up to the factor n/2).  FNS iteration, vectorised over the lam grid.
LAMS1 = np.round(np.linspace(-1.5, 2.5, 401), 6)


def _prep(X, lams):
    L = len(lams)
    lam = lams[:, None, None]
    logX = np.log(X)[None]
    Xl = np.exp(lam * logX)
    small = np.abs(lam) < 1e-12
    Y = np.where(small, logX, (Xl - 1) / np.where(small, 1.0, lam))
    Yt = np.concatenate([Y, -np.ones(Y.shape[:-1] + (1,))], -1)
    c = np.concatenate([np.ones((L, 3)), lams[:, None]], 1)
    _, _, Vh = np.linalg.svd(c[:, None, :])
    B = np.transpose(Vh[:, 1:, :], (0, 2, 1))  # (L,4,3) basis of c-perp
    eta = Yt @ B
    B3 = B[:, :3, :]
    Z = Xl / X[None]
    D1 = X[None] * Z ** 2
    Ch = np.einsum('lki,lnk,lkj->lnij', B3, D1, B3, optimize=True)
    bxz = np.einsum('lki,lnk->lni', B3, Xl, optimize=True)
    Ch = Ch - bxz[..., :, None] * bxz[..., None, :]
    return eta, Ch, B


def sampson_profile(X, lams=LAMS1, iters=60, tol=1e-12):
    eta, Ch, B = _prep(X, lams)
    L = len(lams)
    M0 = np.einsum('lni,lnj->lij', eta, eta)
    _, V0 = np.linalg.eigh(M0)
    a = V0[:, :, 0]

    def cost(a):
        s = np.einsum('lni,li->ln', eta, a)
        q = np.einsum('li,lnij,lj->ln', a, Ch, a)
        with np.errstate(divide="ignore", invalid="ignore"):
            J = np.sum(s ** 2 / q, 1)
        J[~np.isfinite(J) | (np.min(q, 1) <= 0)] = np.inf
        return J, s, q

    J, s, q = cost(a)
    bestJ, bestA = J.copy(), a.copy()
    for _ in range(iters):
        qq = np.maximum(q, 1e-300)
        M = np.einsum('ln,lni,lnj->lij', 1 / qq, eta, eta)
        Lm = np.einsum('ln,lnij->lij', s ** 2 / qq ** 2, Ch)
        Xm = M - Lm
        bad = ~np.all(np.isfinite(Xm.reshape(L, -1)), 1)
        Xm[bad] = np.eye(3)
        ev, EV = np.linalg.eigh(Xm)
        idx = np.argmin(np.abs(ev), 1)
        an = EV[np.arange(L), :, idx]
        an[bad] = a[bad]
        sg = np.sign(np.einsum('li,li->l', an, a))
        sg[sg == 0] = 1
        an = an * sg[:, None]
        done = np.max(np.abs(an - a)) < tol
        a = an
        J, s, q = cost(a)
        better = J < bestJ
        bestJ[better], bestA[better] = J[better], a[better]
        if done:
            break
    return bestJ, bestA


def est_fisher(X, lams=LAMS1):
    J, _ = sampson_profile(X, lams)
    i = int(np.nanargmin(J))
    return 1 - 2 * lams[i], J


# ------------------------------------------------------------------ E5: endpoint-anchored alpha-mixture fit
def boxcox(X, lam):
    return np.log(X) if abs(lam) < 1e-12 else (X ** lam - 1) / lam


def est_anchor(X, lams=LAMS1):
    p, q_ = X[0], X[-1]
    J = np.empty(len(lams))
    for i, lam in enumerate(lams):
        yp, yq = boxcox(p, lam), boxcox(q_, lam)
        u = np.cross(yp - yq, p ** lam)
        b = u @ yp
        s = boxcox(X, lam) @ u - b
        Z = X ** (lam - 1)
        om = u * Z
        var = np.sum(X * om ** 2, 1) - np.sum(X * om, 1) ** 2
        J[i] = np.sum(s[1:-1] ** 2 / var[1:-1])
    i = int(np.argmin(J))
    return 1 - 2 * lams[i], J


# ------------------------------------------------------------------ local curvature quantities (exact, from the field)
def curv_terms(X, f, m, mu, w):
    """Per point: N = g(a, n)/|v|^2 (normal m-curvature of the trajectory), D = g(W(v), n)/|v|^2 (normal
    m-curvature of the e-geodesic with the same tangent), ds = |v|_g (arc-length weight per unit time)."""
    A, V = accel(X, f, m, mu, w)
    n = np.cross(np.broadcast_to(ONES, V.shape), V / X)
    n = n / fnorm(n, X)[:, None]
    q = np.sum(V * V / X, 1)
    Wv = V * V / X - X * q[:, None]
    N = np.sum(A * n / X, 1) / q
    D = np.sum(Wv * n / X, 1) / q
    return N, D, np.sqrt(q)


def est_osc(X, f, m, mu, w, i=0):
    N, D, _ = curv_terms(X, f, m, mu, w)
    return 2 * N[i] / D[i] - 1


def est_osc_median(X, f, m, mu, w):
    N, D, _ = curv_terms(X, f, m, mu, w)
    return np.median(2 * N / D - 1)


def est_curvreg(X, f, m, mu, w, sl=slice(None)):
    """Pooled least squares of trajectory curvature on e-curvature, arc-length weighted: c* = int N D ds / int D^2 ds."""
    N, D, ds = curv_terms(X, f, m, mu, w)
    N, D, ds = N[sl], D[sl], ds[sl]
    c = np.sum(ds * N * D) / np.sum(ds * D * D)
    return 2 * c - 1
