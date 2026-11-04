"""Reach of the pendulum image manifold (pixels64, float renderer) vs the error allowed by the prereg gate.

Reach (Federer): tau = inf_{p != q in M} ||q - p||^2 / (2 dist(q - p, T_p M)).
Gate: normalized recon MSE < 0.05 (MSE / observation variance); per-frame L2 error ~ sqrt(0.05 * sum_pixel var).
Also: same quantities for angle_cs (unit circle: tau = 1).
"""
import sys
import numpy as np
sys.path.insert(0, "/home/user/gwm")
import zoo
from zoo.pendulum import render

n = 1440
th = np.linspace(-np.pi, np.pi, n, endpoint=False)
F = render(th, quantize=False).reshape(n, -1).astype(np.float64)
h = 2 * np.pi / n
# tangent by central differences on the periodic grid
dF = (np.roll(F, -1, 0) - np.roll(F, 1, 0)) / (2 * h)
d2F = (np.roll(F, -1, 0) - 2 * F + np.roll(F, 1, 0)) / h ** 2
sp = np.linalg.norm(dF, axis=1)
T = dF / sp[:, None]
# curvature kappa = ||d2F_perp|| / ||dF||^2
d2perp = d2F - (d2F * T).sum(1)[:, None] * T
kappa = np.linalg.norm(d2perp, axis=1) / sp ** 2
print(f"pixels: |dframe/dtheta| min/max = {sp.min():.2f}/{sp.max():.2f}; curvature radius min = {1/kappa.max():.3f}")

# Federer reach over all pairs
G = F @ F.T
sq = np.diag(G)
D2 = sq[:, None] + sq[None, :] - 2 * G                       # ||q - p||^2
Tq = T @ F.T                                                  # Tq[p, q] = t_p . f(q)
Tp = (T * F).sum(1)                                           # t_p . f(p)
tang = Tq - Tp[:, None]                                       # (q - p) . t_p
perp2 = np.maximum(D2 - tang ** 2, 1e-300)
R = D2 / (2 * np.sqrt(perp2))
np.fill_diagonal(R, np.inf)
# exclude immediate neighbours where the finite-difference tangent is too coarse
for k in range(1, 4):
    idx = np.arange(n)
    R[idx, (idx + k) % n] = np.inf
    R[idx, (idx - k) % n] = np.inf
tau = R.min()
i, j = np.unravel_index(np.argmin(R), R.shape)
dth = np.angle(np.exp(1j * (th[j] - th[i])))
print(f"pixels: Federer reach tau ~ {tau:.3f} (attained at dtheta = {dth:.3f} rad)")
dist = np.sqrt(np.maximum(D2, 0))
print(f"pixels: frame norm ~ {np.sqrt(sq).mean():.2f}; distance theta vs theta+pi ~ {dist[0, n//2]:.2f}; "
      f"theta vs theta+0.1 ~ {dist[0, int(0.1/h)]:.2f}; vs +0.3 ~ {dist[0, int(0.3/h)]:.2f}")

# data distribution of theta under the prereg's main data (box init, omega_max 2.5, T=64)
W = zoo.make("pendulum")
tr = W.sample(seed=123, horizon=64, n_traj=500, obs_mode="angle_cs")
thd = tr.state[:, 1:, 0].reshape(-1)
Fd = render(thd[::7], quantize=False).reshape(len(thd[::7]), -1).astype(np.float64)
totvar = Fd.var(0).sum()
for nm in (0.05, 0.01):
    err = np.sqrt(nm * totvar)
    print(f"pixels: normalized MSE {nm}: per-frame L2 error ~ {err:.3f}  ->  err / tau = {err / tau:.2f}")
# angle_cs
cs = np.stack([np.cos(thd), np.sin(thd)], 1)
tv = cs.var(0).sum()
for nm in (0.05,):
    err = np.sqrt(nm * tv)
    print(f"angle_cs: total var {tv:.3f}; normalized MSE {nm}: L2 error ~ {err:.3f}; tau = 1 -> err/tau = {err:.2f}")
