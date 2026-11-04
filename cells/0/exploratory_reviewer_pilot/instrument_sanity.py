import sys, time
import numpy as np
sys.path.insert(0, "/home/user/gwm")
sys.path.insert(0, "/tmp/claude-0/-home-user/8bc92419-2f30-547d-8b18-8324b70c6f75/scratchpad/review")
import zoo
from topo_tools import analyse_cloud

W = zoo.make("pendulum")
def emb(tr):
    th, om = tr.state[:, 1:, 0].reshape(-1), tr.state[:, 1:, 1].reshape(-1)
    return np.stack([np.cos(th), np.sin(th), om], 1), th

t0 = time.time()
for name, kw in [("main annulus", {}), ("libration disk E<=0.5", dict(E_hi=0.5)),
                 ("libration ring E in [-0.3,0.5]", dict(E_lo=-0.3, E_hi=0.5))]:
    tr = W.sample(seed=777, horizon=64, n_traj=200, obs_mode="angle_cs", **kw)
    Z, th = emb(tr)
    r = analyse_cloud(Z, th)
    print(f"{name:32s} cover={r['cover']:.3f} bars={np.round(r['bars'], 3).tolist()} deg={r["deg"]:.2f} wind={r["wind"]:.2f} lock={r['lock']:.2f}")
print(f"{time.time()-t0:.1f}s")
