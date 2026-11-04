"""Tiny point-maze simulator with exact segment collisions (no learning).

Arena [0,10]^2 with outer walls; interior THIN wall x = 5, y in [0, WALL_TOP] (one convex end at (5, WALL_TOP)).
Optional THICK interior wall [4.8, 5.2] x [0, WALL_TOP].
Contact models for one step of displacement D from P:
  slide   : move to contact, drop normal component of the remaining displacement, continue (frictionless clip)
  stop    : move to contact, discard the rest of the step
  reject  : if the step would cross a wall, stay put
  reflect : move to contact, flip normal component of the remaining displacement (elastic), continue
Second-order (ballistic) variant: state (pos, vel); reflect -> v_n -> -v_n ; inelastic -> v_n -> 0 (slide).
"""
import numpy as np

WALL_TOP = 6.0
EPS = 1e-9


def walls(thick=False):
    V = [(0.0, 0.0, 10.0), (10.0, 0.0, 10.0)]          # vertical: x = c, y in [a, b]
    Hh = [(0.0, 0.0, 10.0), (10.0, 0.0, 10.0)]         # horizontal: y = c, x in [a, b]
    if thick:
        V += [(4.8, 0.0, WALL_TOP), (5.2, 0.0, WALL_TOP)]
        Hh += [(WALL_TOP, 4.8, 5.2)]
    else:
        V += [(5.0, 0.0, WALL_TOP)]
    return V, Hh


def _first_hit(pos, rem, V, Hh):
    n = pos.shape[0]
    smin = np.full(n, np.inf); axis = np.full(n, -1); cval = np.zeros(n)
    for ax, segs in ((0, V), (1, Hh)):
        o = 1 - ax
        d = rem[:, ax]
        with np.errstate(divide="ignore", invalid="ignore"):
            for c, a, b in segs:
                s = (c - pos[:, ax]) / d
                q = pos[:, o] + s * rem[:, o]
                ok = (d != 0) & (s > EPS) & (s <= 1.0) & (q >= a) & (q <= b)
                better = ok & (s < smin)
                smin = np.where(better, s, smin); axis = np.where(better, ax, axis); cval = np.where(better, c, cval)
    return smin, axis, cval


def step1(P, D, mode, W):
    """First-order step. P, D: [n,2]."""
    V, Hh = W
    pos = P.astype(float).copy(); rem = D.astype(float).copy()
    for _ in range(4):
        smin, axis, cval = _first_hit(pos, rem, V, Hh)
        hit = np.isfinite(smin)
        free = ~hit
        pos[free] += rem[free]; rem[free] = 0
        if not hit.any():
            break
        idx = np.where(hit)[0]
        s = smin[idx][:, None]; ax = axis[idx]
        contact = pos[idx] + s * rem[idx]
        sign = np.sign(rem[idx, ax])
        contact[np.arange(len(idx)), ax] = cval[idx] - sign * 1e-7       # stay on the incoming side
        r = (1 - s) * rem[idx]
        if mode == "slide":
            r[np.arange(len(idx)), ax] = 0.0
        elif mode == "reflect":
            r[np.arange(len(idx)), ax] *= -1.0
        elif mode == "stop":
            r[:] = 0.0
        elif mode == "reject":
            contact = P[idx].astype(float); r[:] = 0.0
        pos[idx] = contact; rem[idx] = r
    return pos


def step2(S, dt, mode, W):
    """Second-order ballistic step (no action). S: [n,4] = (x, y, vx, vy). mode: 'reflect' | 'inelastic'."""
    V, Hh = W
    pos = S[:, :2].astype(float).copy(); vel = S[:, 2:].astype(float).copy()
    rem = vel * dt
    for _ in range(4):
        smin, axis, cval = _first_hit(pos, rem, V, Hh)
        hit = np.isfinite(smin)
        free = ~hit
        pos[free] += rem[free]; rem[free] = 0
        if not hit.any():
            break
        idx = np.where(hit)[0]
        s = smin[idx][:, None]; ax = axis[idx]; ar = np.arange(len(idx))
        contact = pos[idx] + s * rem[idx]
        sign = np.sign(rem[idx, ax])
        contact[ar, ax] = cval[idx] - sign * 1e-7
        r = (1 - s) * rem[idx]
        v = vel[idx]
        if mode == "reflect":
            r[ar, ax] *= -1.0; v[ar, ax] *= -1.0
        elif mode == "inelastic":
            r[ar, ax] = 0.0; v[ar, ax] = 0.0
        vel[idx] = v; pos[idx] = contact; rem[idx] = r
    return np.concatenate([pos, vel], 1)


def gh_nodes(m):
    x, w = np.polynomial.hermite_e.hermegauss(m)       # probabilists' Hermite: weight exp(-x^2/2)
    w = w / w.sum()
    X, Y = np.meshgrid(x, x); WW = np.outer(w, w)
    return np.stack([X.ravel(), Y.ravel()], 1), WW.ravel()


def step1_smooth(P, D, mode, W, eps, nodes):
    """Per-step Gaussian smoothing of the one-step map (proxy for a smooth learned 1-step model)."""
    xi, wq = nodes
    n = P.shape[0]; q = len(wq)
    Pq = (P[:, None, :] + eps * xi[None]).reshape(-1, 2)
    Pq = np.clip(Pq, 1e-6, 10 - 1e-6)
    Dq = np.repeat(D, q, axis=0)
    out = step1(Pq, Dq, mode, W).reshape(n, q, 2)
    return (out * wq[None, :, None]).sum(1)
