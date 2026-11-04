# (1) analytic renderer == zoo renderer at k=0; (2) full finite-difference Jacobian Gram det of the 2-frame map
#     vs dt*a(th)*a(th-om*dt), on the script's own warped frames, incl. the TRUE previous state from zoo's integrator.
import numpy as np, zoo
from zoo.pendulum import render
from common import *

th = np.linspace(-np.pi, np.pi, 37, endpoint=False)
dx, dy, shp = out_grid(0.0)
mine = scene(dx, dy, th).reshape(len(th), *shp)
ref = render(th, quantize=False).astype(np.float64)
print("analytic scene vs zoo float renderer, k=0: max |diff| =", np.abs(mine - ref).max())
for k in (0.3, 0.6):
    dx, dy, shp = out_grid(k)
    an = scene(dx, dy, th).reshape(len(th), *shp)
    sc = script_frames(th, k)
    print(f"k={k}: analytic warp vs script bilinear warp: max|diff| = {np.abs(an-sc).max():.3f}, "
          f"rel L2 = {np.linalg.norm(an-sc)/np.linalg.norm(an):.3f}")

rng = np.random.default_rng(0)
w = zoo.make("pendulum")
for k in (0.0, 0.3, 0.6):
    f = lambda t: script_frames(np.atleast_1d(t), k).reshape(len(np.atleast_1d(t)), -1)
    errs, errs_true = [], []
    for _ in range(6):
        t0, o0 = rng.uniform(-np.pi, np.pi), rng.uniform(-2.5, 2.5)
        h = 1e-3
        R = lambda t, o: np.concatenate([f(t)[0], f(t - o * DT)[0]])
        Jt = (R(t0 + h, o0) - R(t0 - h, o0)) / (2 * h)
        Jo = (R(t0, o0 + h) - R(t0, o0 - h)) / (2 * h)
        J = np.stack([Jt, Jo], 1)
        nu_num = np.sqrt(np.linalg.det(J.T @ J))
        a = lambda t: np.linalg.norm((f(t + h)[0] - f(t - h)[0]) / (2 * h))
        nu_formula = DT * a(t0) * a(t0 - o0 * DT)
        cosang = Jt[:4096] @ Jt[4096:] / (np.linalg.norm(Jt[:4096]) * np.linalg.norm(Jt[4096:]))
        errs.append((nu_num / nu_formula - 1, cosang))
        # true previous state via time reversal: Phi_{-dt}(th, om) = T Phi_{dt}(th, -om)
        prev = lambda t, o: w.rollout(np.array([[t, -o]]), 1)[0, 1, 0]
        Rt = lambda t, o: np.concatenate([f(t)[0], f(prev(t, o))[0]])
        Jt = (Rt(t0 + h, o0) - Rt(t0 - h, o0)) / (2 * h)
        Jo = (Rt(t0, o0 + h) - Rt(t0, o0 - h)) / (2 * h)
        J = np.stack([Jt, Jo], 1)
        errs_true.append(np.sqrt(np.linalg.det(J.T @ J)) / nu_formula - 1)
    e = np.array(errs)
    print(f"k={k}: Gram-det / (dt a a) - 1 (linearised prev): max {np.abs(e[:,0]).max():.1e}; "
          f"cos(angle f'(th), f'(th-om dt)) in [{e[:,1].min():+.2f},{e[:,1].max():+.2f}] (non-orthogonal); "
          f"true-flow prev: max rel dev {np.abs(errs_true).max():.1e}")
