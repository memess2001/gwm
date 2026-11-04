"""Perfect-recon latent (cos theta, sin theta, lam * omega): the theorem holds for every lam (decoder = first two coords,
zero error), but the finite-sample VR H1 bar (n = 400 FPS points) shrinks as the omega axis is stretched."""
import sys
import numpy as np
sys.path.insert(0, "/home/user/gwm")
sys.path.insert(0, "/tmp/claude-0/-home-user/8bc92419-2f30-547d-8b18-8324b70c6f75/scratchpad/review")
import zoo
from topo_tools import analyse_cloud

W = zoo.make("pendulum")
tr = W.sample(seed=777, horizon=64, n_traj=200, obs_mode="angle_cs")
th, om = tr.state[:, 1:, 0].reshape(-1), tr.state[:, 1:, 1].reshape(-1)
for lam in (1, 3, 6, 10, 20):
    Z = np.stack([np.cos(th), np.sin(th), lam * om], 1)
    for m in (400, 1000):
        r = analyse_cloud(Z, th, m=m)
        b = np.array(r["bars"])
        top = b[0] if len(b) else [np.nan, np.nan]
        sec = b[1] if len(b) > 1 else [np.nan, np.nan]
        print(f"lam={lam:3d} n={m:5d} cover={r['cover']:.2f} top bar=({top[0]:.2f},{top[1]:.2f}) ratio={top[1]/top[0]:.2f} "
              f"second=({sec[0]:.2f},{sec[1]:.2f}) wind={r['wind']}")
