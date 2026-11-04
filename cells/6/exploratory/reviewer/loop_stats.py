# Ground-truth-only check: what non-parity features of the analysis loops already separate odd from even loops?
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/home/user/gwm")
import zoo

def entries(w, s0, acts):
    st = w.trajectory(s0, acts)
    moved = (st[1:, :2] != st[:-1, :2]).any(-1)
    sid = np.where(moved, w.switch_id[st[1:, 1], st[1:, 0]], -1)
    blocked = (~moved).sum()
    return np.array([(sid == i).sum() for i in range(w.k)]), blocked, st

rng = np.random.default_rng(7)
for k in (1, 2, 3):
    for pol in ("random", "loops"):
        w = zoo.make("toggle_grid", n_doors=k)
        tr = w.sample(seed=999, horizon=63, n_traj=100, obs_mode="pos", policy=pol)
        pref = rng.integers(0, 31, 100)
        z_states = tr.state[np.arange(100), pref]                     # true states at the end of the prefix
        L, odd, ent_any, n_ent, ge2even, room0, tot_len, n_cls, blk, dsw = [], [], [], [], [], [], [], [], [], []
        for i in range(100):
            s0 = z_states[i]
            flips_seen = set()
            for j in range(200):
                a = w.random_loop(s0, rng=rng, length=(4, 20))
                e, b, st = entries(w, s0, a)
                h = w.holonomy(s0, a)
                assert h["position_closed"]
                f = tuple(h["flips"].tolist())
                flips_seen.add(f)
                L.append(len(a)); odd.append(int(h["flips"][0])); ent_any.append(int(e[0] > 0)); n_ent.append(e[0])
                ge2even.append(int(e[0] >= 2 and e[0] % 2 == 0)); room0.append(int(w.room[s0[1], s0[0]] == 0))
                tot_len.append(pref[i] + len(a) + 20); blk.append(b)
                sx, sy = w.switches[0]
                dsw.append(np.abs(st[:, 0] - sx).min() + 0 if True else 0)
            n_cls.append(len(flips_seen))
        L, odd, ent_any, n_ent, ge2even, room0, tot_len, blk = map(np.array, (L, odd, ent_any, n_ent, ge2even, room0, tot_len, blk))
        ev = odd == 0
        print(f"k={k} prefix-policy={pol:6s}: odd frac {odd.mean():.3f}; loop length mean odd {L[odd==1].mean():.1f} / even {L[ev].mean():.1f}, "
              f"max {L.max()}; AUC(length) {roc_auc_score(odd, L):.3f}; AUC(any switch entry) {roc_auc_score(odd, ent_any):.3f}; "
              f"AUC(#entries) {roc_auc_score(odd, n_ent):.3f}; even loops with >=2 entries {ge2even[ev].mean():.3f} "
              f"(of switch-entering even+odd: {ge2even.sum()/max(1,ent_any.sum()):.3f}); start in R0 {room0.mean():.2f}; "
              f"loops with a blocked move {np.mean(blk>0):.2f}; prefix+loop+probe20 > 63: {np.mean(tot_len>63):.3f}, max {tot_len.max()}; "
              f"distinct flip classes per z: median {int(np.median(n_cls))}, max {max(n_cls)} (2^k = {2**k})")
