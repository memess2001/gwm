# Ground-truth-only check (zoo dynamics, no models): base rates relevant to the cell-6 gate.
# "Door attempt" = the agent stands on a cell 4-adjacent to door D_i and moves into it.
import sys
import numpy as np
sys.path.insert(0, "/home/user/gwm")
import zoo
from zoo.toggle_grid import MOVES

def attempts(w, st, acts):
    """Per door attempt: (side 0 = from R_{i-1} / 1 = from R_i, door open?, switch entries so far, any entry so far)."""
    rows = []
    N, T = acts.shape
    ent = np.zeros((N, w.k), np.int64)
    for t in range(T):
        x, y = st[:, t, 0], st[:, t, 1]
        tx, ty = x + MOVES[acts[:, t], 0], y + MOVES[acts[:, t], 1]
        tx, ty = np.clip(tx, 0, w.W - 1), np.clip(ty, 0, w.H - 1)
        di = w.door_id[ty, tx]
        for n in np.flatnonzero(di >= 0):
            i = di[n]
            side = 0 if w.room[y[n], x[n]] == i else 1
            rows.append((side, st[n, t, 2 + i], ent[n, i], ent[n, i] > 0, i, t))
        # count switch entries at this step
        moved = (st[:, t + 1, :2] != st[:, t, :2]).any(-1)
        sid = np.where(moved, w.switch_id[st[:, t + 1, 1], st[:, t + 1, 0]], -1)
        for n in np.flatnonzero(sid >= 0):
            ent[n, sid[n]] += 1
    return np.array(rows)

for k in (1, 2):
    for pol, kw in (("random", {}), ("loops(2,8)", {"policy": "loops"}), ("loops(4,20)", {"policy": "loops", "loop_len": (4, 20)})):
        w = zoo.make("toggle_grid", n_doors=k)
        tr = w.sample(seed=12345, horizon=63, n_traj=2000, obs_mode="pos", **kw)
        r = attempts(w, tr.state, tr.actions)
        out_R0 = (tr.info["room"] > 0).mean()
        ever_pass = (tr.info["room"].max(1) > 0).mean()
        s = r[r[:, 0] == 0]          # attempts from the switch side
        d1 = s[s[:, 4] == 0]
        open_frac = d1[:, 1].mean() if len(d1) else float("nan")
        # heuristics a non-parity model might learn for D1 attempts from the switch side
        always_closed = 1 - open_frac
        any_entry = (d1[:, 3] == d1[:, 1]).mean()        # predicts open iff the switch was ever entered
        even_ge2 = ((d1[:, 2] >= 2) & (d1[:, 2] % 2 == 0)).mean()
        print(f"k={k} {pol:12s}: frac frames outside R0 {out_R0:.3f}, episodes ever leaving R0 {ever_pass:.3f}; "
              f"D1 attempts from switch side per episode {len(d1)/2000:.2f}, door open in {open_frac:.3f}; "
              f"acc 'door is a wall' {always_closed:.3f}, acc 'open iff ever entered' {any_entry:.3f}; "
              f"attempts after an even >=2 number of entries {even_ge2:.3f}; far-side attempts/ep {np.sum(r[:,0]==1)/2000:.2f} "
              f"(open {r[r[:,0]==1][:,1].mean() if np.any(r[:,0]==1) else float('nan'):.3f})")
