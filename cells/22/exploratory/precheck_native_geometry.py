"""Cell 22, check before preregistration (ground truth only; nothing is learned).

"Native geometry = the metric whose geodesics fit the trajectories best", on the Wright-Fisher world (the cell
predicts "population world = Fisher"). Reparametrisation-free straightness tests of each path:
   alpha = +1 (e-flat):   straight line in theta = logit coordinates   residual sigma_2 / sigma_1 of centred theta(t)
   alpha =  0 (Fisher, Levi-Civita): great circle of sqrt(x) on the sphere   residual sigma_3 / sigma_1 of sqrt(x(t))
   alpha = -1 (m-flat):   straight line in x                            residual sigma_2 / sigma_1 of centred x(t)
(alpha-geodesic <=> x^((1-alpha)/2) spans a 2-plane through the origin; cells/7, proposition checked to 1e-11.)
"Euclidean in x" and "Euclidean in theta" are the alpha = -1 and alpha = +1 tests: which chart is called Euclidean
decides the answer. Runs in a few seconds.
"""
import numpy as np

from zoo.wright_fisher import WrightFisher


def residuals(X):
    th = np.log(X[:, :-1]) - np.log(X[:, -1:])
    s = lambda M: np.linalg.svd(M, compute_uv=False)
    st = s(th - th.mean(0))
    sx = s(X - X.mean(0))
    sq = s(np.sqrt(X))
    return st[1] / st[0], sq[2] / sq[0], sx[1] / sx[0]


def main():
    print("median (max) straightness residual over 100 paths of 40 steps; 0 = exactly a geodesic")
    print(f"{'mechanism':12} {'alpha=+1 (theta)':>18} {'alpha=0 (Fisher LC)':>22} {'alpha=-1 (x)':>16}")
    for sched in ("selection_only", "migration_only", "mixed_only"):
        w = WrightFisher(K=3, schedule=sched, seg_steps=40)
        X = w.sample(seed=2, horizon=40, n_traj=100, obs_mode="freq").state
        R = np.array([residuals(X[i]) for i in range(len(X))])
        med, mx = np.median(R, 0), R.max(0)
        print(f"{sched.replace('_only', ''):12} " + " ".join(f"{m:9.1e} ({x:7.1e})" for m, x in zip(med, mx)))


if __name__ == "__main__":
    main()
