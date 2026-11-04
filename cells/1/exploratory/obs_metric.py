"""Check 1 (first half of the claim): fixed-sigma decoder Fisher pullback g = J^T J / sigma^2 for a
top-down 64x64 rendering of a point agent drawn as a Gaussian blob, with walls rendered in several ways.

Exact rendering derivative (perfect decoder): o(u; x) = composite(blob(u - x), W(u)).
g_ij(x) = sum_u d_i o(u;x) d_j o(u;x)  (sigma = 1).  Gaussian curvature K via the Brioschi formula.
"""
import numpy as np

H = W_ = 64
s = 2.0                      # blob width in px (zoo pendulum bob_sigma = 2.5)
step = 0.5                   # agent-position grid step (px)
u_r, u_c = np.mgrid[0:H, 0:W_].astype(float) + 0.5   # pixel centres (row = y, col = x)

# interior wall: thick vertical bar x in [30,34], y in [0,40]  (attached to the bottom image edge)
wall = ((u_c >= 30) & (u_c < 34) & (u_r < 40)).astype(float)

def blob_grads(x, y):
    b = np.exp(-((u_c - x) ** 2 + (u_r - y) ** 2) / (2 * s * s))
    return b, b * (u_c - x) / s ** 2, b * (u_r - y) / s ** 2

# derivative multiplier m(u) per compositing rule:  d o = m(u) * d blob
modes = {
    "invisible walls":            np.ones_like(wall),
    "additive (o=blob+0.5W)":     np.ones_like(wall),
    "agent over wall (alpha)":    1 - 0.5 * wall,          # o = blob + (1-blob)*0.5W
    "wall over agent (opaque)":   1 - wall,                # o = blob*(1-W) + 0.5W ; also min(1, blob+W)
}

xs = np.arange(0.25, 64, step)
X, Y = np.meshgrid(xs, xs)            # X = col (x), Y = row (y)

def metric_field(m):
    E = np.zeros_like(X); F = np.zeros_like(X); G = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            _, gx, gy = blob_grads(X[i, j], Y[i, j])
            gx, gy = gx * m, gy * m
            E[i, j] = (gx * gx).sum(); F[i, j] = (gx * gy).sum(); G[i, j] = (gy * gy).sum()
    return E, F, G

def gauss_curv(E, F, G, h):
    # coordinates: u = x (axis 1), v = y (axis 0)
    du = lambda A: np.gradient(A, h, axis=1)
    dv = lambda A: np.gradient(A, h, axis=0)
    Eu, Ev, Fu, Fv, Gu, Gv = du(E), dv(E), du(F), dv(F), du(G), dv(G)
    Evv, Guu, Fuv = dv(Ev), du(Gu), dv(Fu)
    M1 = np.stack([np.stack([-Evv / 2 + Fuv - Guu / 2, Eu / 2, Fu - Ev / 2], -1),
                   np.stack([Fv - Gu / 2, E, F], -1),
                   np.stack([Gv / 2, F, G], -1)], -2)
    M2 = np.stack([np.stack([np.zeros_like(E), Ev / 2, Gu / 2], -1),
                   np.stack([Ev / 2, E, F], -1),
                   np.stack([Gu / 2, F, G], -1)], -2)
    return (np.linalg.det(M1) - np.linalg.det(M2)) / (E * G - F * F) ** 2

# geometry masks on the agent grid
inside_wall = (X >= 30) & (X < 34) & (Y < 40)
dx = np.maximum(np.maximum(30 - X, X - 34), 0); dy = np.maximum(Y - 40, 0)
d_wall = np.hypot(dx, dy)                                  # distance to interior wall
d_border = np.minimum.reduce([X, 64 - X, Y, 64 - Y])        # distance to image border
free = ~inside_wall
near_wall = free & (d_wall < 2 * s) & (d_border > 4 * s)
near_border = free & (d_border < 2 * s) & (d_wall > 4 * s)
far = free & (d_wall > 4 * s) & (d_border > 4 * s)

g0 = None
print(f"blob s={s}px, grid step {step}px; g normalised by its far-field value; K in units of 1/(g_far * px^2)")
for name, m in modes.items():
    E, F, G = metric_field(m)
    if g0 is None:
        g0 = np.median(E[far])
    E, F, G = E / g0, F / g0, G / g0
    K = gauss_curv(E, F, G, step)
    anom = np.sqrt((E - 1) ** 2 + 2 * F ** 2 + (G - 1) ** 2)       # ||g - g_far||_F / g_far
    lit = free & (anom > 0.05)
    band = free & (d_wall < 2 * s)
    iou = (lit & band).sum() / max((lit | band).sum(), 1)
    lit_int = lit & (d_border > 4 * s)
    iou_int = (lit_int & band).sum() / max((lit_int | band).sum(), 1)
    print(f"\n[{name}]")
    print(f"  metric anomaly ||g-g_far||/g_far: far max {anom[far].max():.2e} | near interior wall mean {anom[near_wall].mean():.3f} "
          f"max {anom[near_wall].max():.3f} | near image border mean {anom[near_border].mean():.3f} max {anom[near_border].max():.3f}")
    Kf = np.abs(K)
    print(f"  |K|: far max {Kf[far].max():.2e} | near interior wall max {Kf[near_wall].max():.2e} | near image border max {Kf[near_border].max():.2e}")
    print(f"  IoU(lit = anomaly>5%, band = free & d_wall<2s): all lit {iou:.2f}; lit excluding image-border zone {iou_int:.2f}")
