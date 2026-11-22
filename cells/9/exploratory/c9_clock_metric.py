"""Cell 9 pre-prereg check (ground truth only, no learning): does the Fisher (Shahshahani) arc length make event
times more regular across random seeds than other arc lengths, on exact Wright-Fisher paths?
Clocks: wall time; Euclidean arc length in x; Fisher/Shahshahani arc length; logit (clr / Aitchison) arc length.
Every arc length removes the speed (all are reparametrisation-invariant along a fixed path); they differ only in how
they weight different parts of the path, so which one is most regular depends on the mechanism and the event."""
import sys
import numpy as np
sys.path.insert(0, "/home/user/gwm_repo")
from zoo.wright_fisher import WrightFisher

def clocks(X, dt):
    d = np.diff(X, axis=1); xm = 0.5 * (X[:, 1:] + X[:, :-1])
    clr = np.log(X) - np.log(X).mean(-1, keepdims=True)
    inc = dict(wall=np.full(d.shape[:2], dt),
               euclid=np.linalg.norm(d, axis=-1),
               fisher=np.sqrt((d ** 2 / xm).sum(-1)),
               logit=np.linalg.norm(np.diff(clr, axis=1), axis=-1))
    return {k: np.concatenate([np.zeros((len(X), 1)), np.cumsum(v, 1)], 1) for k, v in inc.items()}

def event_cv(X, dt, event):
    C, hit = clocks(X, dt), event(X)
    first = np.argmax(hit, 1); ok = hit.any(1)
    out = {}
    for k, c in C.items():
        v = c[np.arange(len(X)), first][ok]
        out[k] = v.std() / v.mean()
    return out, ok.mean()

T, n = 600, 400
rng_seed = 1
out = []
for mech, sched in (("selection", "selection_only"), ("migration", "migration_only"), ("mixed", "mixed_only")):
    for start in ("dirichlet", "rare_invader"):
        w = WrightFisher(K=3, schedule=sched, seg_steps=T)
        kw = {}
        if start == "rare_invader":
            kw["x0"] = np.array([0.02, 0.49, 0.49])
        tr = w.sample(seed=rng_seed, horizon=T, n_traj=n, obs_mode="freq", **kw)
        X = tr.state
        X = X[np.isfinite(X).all((1, 2)) & (X >= 0).all((1, 2))]     # mixed_only: RK4 gives a few non-finite paths
        final = X[:, -1]
        win = final.argmax(-1)
        for thr in (0.5, 0.9):
            # exclude trajectories whose winner already starts above the threshold
            keep = X[np.arange(len(X)), 0, win] < thr
            res, frac = event_cv(X[keep], 0.1, lambda Y, w=win[keep], t=thr: np.take_along_axis(Y, w[:, None, None].repeat(Y.shape[1], 1), 2)[..., 0] > t)
            out.append((mech, start, thr, int(keep.sum()), frac, res))
print(f"{'mechanism':10} {'start':13} {'event':12} {'n':>4} {'hit':>5}  " + "  ".join(f"{k:>7}" for k in ("wall", "euclid", "fisher", "logit")) + "   best")
for mech, start, thr, nk, frac, r in out:
    if frac < 0.5:
        continue                                   # too few trajectories reach the event within the horizon
    best = min(("euclid", "fisher", "logit"), key=lambda k: r[k])
    print(f"{mech:10} {start:13} {'win > ' + str(thr):12} {nk:4d} {frac:5.2f}  " + "  ".join(f"{r[k]:7.3f}" for k in ("wall", "euclid", "fisher", "logit")) + f"   {best}")
