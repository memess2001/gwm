# Ground-truth-only check: are the 2nd / 3rd door bits ever observable in 63-step training episodes?
import sys
import numpy as np
sys.path.insert(0, "/home/user/gwm")
import zoo
from zoo.toggle_grid import MOVES

for k in (2, 3):
    for pol, kw in (("random", {}), ("loops(2,8)", {"policy": "loops"})):
        w = zoo.make("toggle_grid", n_doors=k)
        tr = w.sample(seed=4242, horizon=63, n_traj=2000, obs_mode="pos", **kw)
        st, acts = tr.state, tr.actions
        res = []
        for i in range(1, k):                 # door D_{i+1} (0-based i), switch S_{i+1} lies in room R_i
            att_open = att = tog = 0
            for t in range(63):
                x, y = st[:, t, 0], st[:, t, 1]
                tx = np.clip(x + MOVES[acts[:, t], 0], 0, w.W - 1); ty = np.clip(y + MOVES[acts[:, t], 1], 0, w.H - 1)
                hit = w.door_id[ty, tx] == i
                att += hit.sum(); att_open += (hit & (st[:, t, 2 + i] == 1)).sum()
            tog = (tr.info["switch_entered"] == i).any(1).sum()
            res.append(f"D{i+1}: episodes toggling S{i+1} {tog}, attempts {att}, attempts with D{i+1} open {att_open}")
        print(f"k={k} {pol}: frames in room>=1 {(tr.info['room']>=1).mean():.4f}, in room>=2 {(tr.info['room']>=2).mean():.5f}; " + "; ".join(res))

# gate sample size on a 200-episode held-out split (uniform random and loops(2,8)), k = 1
w = zoo.make("toggle_grid", n_doors=1)
for pol, kw in (("random", {}), ("loops(2,8)", {"policy": "loops"})):
    n_att, n_open = [], []
    for seed in range(20):
        tr = w.sample(seed=10_000 + seed, horizon=63, n_traj=200, obs_mode="pos", **kw)
        st, acts = tr.state, tr.actions
        a = o = 0
        for t in range(63):
            x, y = st[:, t, 0], st[:, t, 1]
            tx = np.clip(x + MOVES[acts[:, t], 0], 0, w.W - 1); ty = np.clip(y + MOVES[acts[:, t], 1], 0, w.H - 1)
            hit = (w.door_id[ty, tx] == 0) & (w.room[y, x] == 0)
            a += hit.sum(); o += (hit & (st[:, t, 2] == 1)).sum()
        n_att.append(a); n_open.append(o)
    print(f"k=1 {pol}: 200 held-out episodes -> switch-side D1 attempts {np.mean(n_att):.0f} (range {min(n_att)}-{max(n_att)}), "
          f"with the door open {np.mean(n_open):.0f} (range {min(n_open)}-{max(n_open)})")
