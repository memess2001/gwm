# Ground-truth-only check: distinguishing power of GENERIC random probes (no knowledge of door positions).
# For a state pair that differs only in the door bit (same position), what fraction of random 20-step action
# sequences produce different observation sequences in the true world? And with 16 such probes?
import sys
import numpy as np
sys.path.insert(0, "/home/user/gwm")
import zoo

w = zoo.make("toggle_grid", n_doors=1)
rng = np.random.default_rng(3)
cells = np.argwhere(w.room == 0)[:, ::-1]               # (x, y) of R0 interior cells
P = rng.integers(0, 4, (4000, 20))
ps = []
for x, y in cells:
    a = np.array([x, y, 0]); b = np.array([x, y, 1])
    oa = w.observe(w.trajectory(a, P), "pos"); ob = w.observe(w.trajectory(b, P), "pos")
    ps.append((oa != ob).any(-1).any(-1).mean())
ps = np.array(ps)
p16 = 1 - (1 - ps) ** 16
print(f"random 20-step probe distinguishes bit 0 vs 1: per-cell p min {ps.min():.3f} median {np.median(ps):.3f} max {ps.max():.3f}; "
      f"P(at least one of 16 random probes distinguishes) min {p16.min():.3f} median {np.median(p16):.3f}")

# no-switch control as specified (switches_enabled=False, init_doors closed): where can the agent ever be?
for init in ("closed", (1,)):
    w0 = zoo.make("toggle_grid", n_doors=1, switches_enabled=False, init_doors=init)
    tr = w0.sample(seed=5, horizon=63, n_traj=2000, obs_mode="pos")
    print(f"no-switch world, init_doors={init}: fraction of frames outside R0 {(tr.info['room'] > 0).mean():.3f}, "
          f"door-cell visits {(tr.info['room'] < 0).mean():.4f}")

# plaquette bookkeeping for P6.2 (k = 1, 2, 3)
for k in (1, 2, 3):
    wk = zoo.make("toggle_grid", n_doors=k)
    cm = wk.curvature_map()
    P_ = cm["corners"]; par = cm["parity"].any(1)
    rooms = wk.room[P_[:, 1], P_[:, 0]]
    print(f"k={k}: executable plaquettes {len(P_)} (in R0: {(rooms == 0).sum()}), switch plaquettes {par.sum()}, "
          f"random precision@4 expectation {par.sum() / len(P_):.3f}")
