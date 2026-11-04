"""Check 2: predictive Cauchy-Green C_k = sum_{j=0..k} J_{F^j}^T g_obs J_{F^j} for a FIRST-ORDER point maze
(state = position, action = commanded displacement, |d| = 0.25 per step, k = 4 -> reach B = 1.0).
g_obs = I (translating-blob rendering is flat, see check 1).  Free-space value: C_k = (k+1) I.

Readouts per position, per action:
  FTLE  = 0.5 log(lam_max / (k+1))     (> 0: more stretching than free space; repelling-LCS ridges)
  COMP  = 0.5 log((k+1) / lam_min)     (> 0: compression; rank loss at contact)
  K     = Gaussian curvature of C_k viewed as a metric (Brioschi)
  RES   = || F^k(x) - (x + k d) ||    (non-geometric baseline: deviation from free-flight prediction)
Models: exact simulator (FD h=1e-4 => a.e. Jacobian; FD h=grid/2 => discontinuity 'ridge detector'),
        and per-step Gaussian-smoothed 1-step map (eps=0.05; proxy for a smooth learned 1-step WM).
"""
import sys
import numpy as np
from maze import walls, step1, step1_smooth, gh_nodes, WALL_TOP

K_STEPS, STEP, B = 4, 0.25, 1.0
W = walls(thick=False)
ACTS = {name: STEP * np.array([np.cos(a), np.sin(a)]) for name, a in
        [("E", 0), ("NE", np.pi / 4), ("N", np.pi / 2), ("NW", 3 * np.pi / 4), ("W", np.pi),
         ("SW", -3 * np.pi / 4), ("S", -np.pi / 2), ("SE", -np.pi / 4)]}


def kmap(P, d, mode, smooth=None):
    out = [P]
    x = P
    for _ in range(K_STEPS):
        D = np.repeat(d[None], len(x), 0)
        x = step1(x, D, mode, W) if smooth is None else step1_smooth(x, D, mode, W, *smooth)
        out.append(x)
    return out                                        # list of [n,2], j = 0..k


def cauchy_green(P, d, mode, h, smooth=None):
    e = np.eye(2)
    plus = [kmap(P + h * e[i], d, mode, smooth) for i in range(2)]
    minus = [kmap(P - h * e[i], d, mode, smooth) for i in range(2)]
    C = np.zeros((len(P), 2, 2))
    for j in range(K_STEPS + 1):
        J = np.stack([(plus[i][j] - minus[i][j]) / (2 * h) for i in range(2)], -1)   # [n, out, in]
        C += np.einsum("noi,noj->nij", J, J)
    Fk = kmap(P, d, mode, smooth)[-1]
    res = np.linalg.norm(Fk - (P + K_STEPS * d), axis=1)
    return C, res


def gauss_curv(E, F, G, h):
    du = lambda A: np.gradient(A, h, axis=1); dv = lambda A: np.gradient(A, h, axis=0)
    Eu, Ev, Fu, Fv, Gu, Gv = du(E), dv(E), du(F), dv(F), du(G), dv(G)
    Evv, Guu, Fuv = dv(Ev), du(Gu), dv(Fu)
    M1 = np.stack([np.stack([-Evv / 2 + Fuv - Guu / 2, Eu / 2, Fu - Ev / 2], -1),
                   np.stack([Fv - Gu / 2, E, F], -1), np.stack([Gv / 2, F, G], -1)], -2)
    M2 = np.stack([np.stack([np.zeros_like(E), Ev / 2, Gu / 2], -1),
                   np.stack([Ev / 2, E, F], -1), np.stack([Gu / 2, F, G], -1)], -2)
    return (np.linalg.det(M1) - np.linalg.det(M2)) / (E * G - F * F) ** 2


def zones(X, Y):
    dseg = np.hypot(np.abs(X - 5.0), np.maximum(Y - WALL_TOP, 0))
    dout = np.minimum.reduce([X, 10 - X, Y, 10 - Y])
    dcorner = np.hypot(X - 5.0, Y - WALL_TOP)
    face = (np.abs(X - 5.0) < B) & (Y > B) & (Y < WALL_TOP - B)
    corner = dcorner < B
    outer = (dout < B) & (dseg > B)
    far = (dout > B) & (dseg > B)
    band = (dout < B) | (dseg < B)
    return dict(face=face, corner=corner, outer=outer, far=far), band


def run(grid_step, h, smooth, label, actions, modes=("slide", "stop", "reject", "reflect")):
    xs = np.arange(grid_step / 2 + 1e-3, 10, grid_step)
    X, Y = np.meshgrid(xs, xs)                      # axis 1 = x, axis 0 = y
    P = np.stack([X.ravel(), Y.ravel()], 1)
    Z, band = zones(X, Y)
    print(f"\n===== {label}: grid {grid_step}, FD h={h}, k={K_STEPS}, |d|={STEP} (band B={B}) =====")
    for mode in modes:
        agg = {r: np.full(X.shape, -np.inf) for r in ("FTLE", "COMP", "|K|", "RES")}
        per_act = {}
        for an in actions:
            C, res = cauchy_green(P, ACTS[an], mode, h, smooth)
            C = C.reshape(X.shape + (2, 2))
            ev = np.linalg.eigvalsh(C)
            ftle = 0.5 * np.log(ev[..., 1] / (K_STEPS + 1))
            comp = 0.5 * np.log((K_STEPS + 1) / np.maximum(ev[..., 0], 1e-12))
            Kc = np.abs(gauss_curv(C[..., 0, 0], C[..., 0, 1], C[..., 1, 1], grid_step))
            r = dict(FTLE=ftle, COMP=comp, **{"|K|": Kc}, RES=res.reshape(X.shape))
            per_act[an] = r
            for k_ in agg:
                agg[k_] = np.maximum(agg[k_], r[k_])
        thr = {"FTLE": 0.05, "COMP": 0.05, "|K|": 1e-2, "RES": 0.01}
        print(f"\n-- contact model: {mode} --")
        for an in actions[:2]:
            r = per_act[an]
            s = "  ".join(f"{k_}: face {np.mean(r[k_][Z['face']]):.3f}/{np.max(r[k_][Z['face']]):.2g}, "
                          f"corner max {np.max(r[k_][Z['corner']]):.2g}, far max {np.max(r[k_][Z['far']]):.1g}"
                          for k_ in ("FTLE", "COMP", "|K|"))
            print(f"  action {an:>2}: {s}")
        print(f"  max over {len(actions)} actions -> zone coverage (fraction of zone lit) and IoU(lit, all-wall band):")
        for k_, A in agg.items():
            lit = A > thr[k_]
            cov = "  ".join(f"{z} {lit[m].mean():.2f}" for z, m in Z.items())
            iou = (lit & band).sum() / max((lit | band).sum(), 1)
            print(f"    {k_:>4} (thr {thr[k_]}): {cov}   IoU {iou:.2f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "exact"
    if which == "exact":
        run(0.05, 1e-4, None, "EXACT simulator, a.e. Jacobian", list(ACTS))
        run(0.05, 0.025, None, "EXACT simulator, coarse FD (discontinuity detector)", list(ACTS))
    else:
        run(0.1, 1e-3, (0.05, gh_nodes(5)), "SMOOTHED 1-step model (eps=0.05)", list(ACTS))
