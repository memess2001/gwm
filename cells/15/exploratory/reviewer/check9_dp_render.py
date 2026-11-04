# Double pendulum rendered by a RIGID camera as two Gaussian blobs: image pull-back metric g(q) vs mass matrix M(q).
import numpy as np
yy, xx = np.mgrid[0:256, 0:256] / 4.0 - 31.875           # 4x supersampled 64px canvas, pivot at centre
def frame(q, s1=2.5, s2=2.5, l1=12.0, l2=12.0):
    p1 = l1 * np.array([np.sin(q[0]), np.cos(q[0])]); p2 = p1 + l2 * np.array([np.sin(q[1]), np.cos(q[1])])
    b = lambda p, s: np.exp(-((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (2 * s * s))
    return 1 - (1 - b(p1, s1)) * (1 - b(p2, s2))
def g(q, **kw):
    h = 1e-4; J = []
    for i in range(2):
        e = np.zeros(2); e[i] = h
        J.append(((frame(q + e, **kw) - frame(q - e, **kw)) / (2 * h)).ravel())
    J = np.array(J); return J @ J.T
rng = np.random.default_rng(0)
Q = rng.uniform(-np.pi, np.pi, (300, 2))
for name, kw in (("equal blobs (E1=E2) <-> m1=m2", {}), ("bob 2 drawn larger (s2=4)", {"s2": 4.0})):
    detg = np.array([np.linalg.det(g(q, **kw)) for q in Q])
    detM = 2 - np.cos(Q[:, 0] - Q[:, 1]) ** 2              # m1=m2=l1=l2=1
    r = detg / detM
    print(f"{name:<34s}: det g / det M  max/min = {r.max()/r.min():.4f}   (det g alone max/min = {detg.max()/detg.min():.3f})")
