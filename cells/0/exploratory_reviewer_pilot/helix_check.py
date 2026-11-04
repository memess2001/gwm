"""'theta-projection covers S^1' is not the right hypothesis: a damped helical strip covers S^1 but has no wrapping loop.
The latent of a perfect recon WM E = (cos theta_t, sin theta_t, omega_hat) (decoder = first two coords, zero error) on this
data has b1 = 0, although the theta-projection of the data covers the circle."""
import sys
import numpy as np
sys.path.insert(0, "/home/user/gwm")
sys.path.insert(0, "/tmp/claude-0/-home-user/8bc92419-2f30-547d-8b18-8324b70c6f75/scratchpad/review")
import zoo
from topo_tools import analyse_cloud

rng = np.random.default_rng(0)
n = 300
init = np.stack([rng.uniform(-0.15, 0.15, n), rng.uniform(2.75, 2.85, n)], 1)
W = zoo.make("pendulum", gamma=0.06, init=init)
tr = W.sample(seed=5, horizon=40, n_traj=n, obs_mode="angle_cs")
th = tr.state[:, 1:, 0].reshape(-1); om = tr.state[:, 1:, 1].reshape(-1)
unw = tr.info["theta_unwrapped"]
print("unwrapped theta advance per trajectory: min %.2f  max %.2f  (2pi = 6.28)" % ((unw[:, -1] - unw[:, 0]).min(), (unw[:, -1] - unw[:, 0]).max()))
hist = np.histogram(th, bins=36, range=(-np.pi, np.pi))[0]
print("theta bins empty:", int((hist == 0).sum()), "of 36  -> theta-projection covers S^1" if (hist > 0).all() else "")
# omega at theta ~ 0 on first vs second pass
first = om[(np.abs(th) < 0.1)]
print("omega values near theta = 0: %s" % np.round(np.percentile(first, [0, 25, 50, 75, 100]), 2))
Z = np.stack([np.cos(th), np.sin(th), om], 1)
r = analyse_cloud(Z, th)
print("perfect-recon latent (cos, sin, omega): bars", np.round(r["bars"], 3).tolist(), "wind", r["wind"])
