"""Exact entropy production of the simulator's own discrete-time chain (zoo/wright_fisher.py, pop_size = N), K = 3.

One generation = deterministic flow Phi_dt of the zoo (RK4, 'mutation_selection') followed by Multinomial(N, Phi_dt(x)) / N.
On the (N+1)(N+2)/2 lattice states the transition matrix is dense and exact; pi solves pi P = pi; the chain EP per
generation is  1/2 sum_ab (pi_a P_ab - pi_b P_ba) log(pi_a P_ab / (pi_b P_ba))  (Kullback-Leibler rate between the
stationary chain and its time reversal; model-free, no drift, no D, no Hodge).  Divided by dt it is compared with the
diffusion-limit EP from review_c2_fp.py (Ne = N dt = 10): 0 (uniform), 0.7211 (uniform 0.5 + cyclic 1), 0.9472 (cyclic 1).
Also reports the drift-based (diffusion-formula) EP computed from the chain's own pi with the Hodge projection of the
exact per-generation drift (Phi_dt(x) - x)/dt in the metric (N dt) * Shahshahani -- i.e. what a drift-based estimator
(SFI-type) would converge to on infinite simulator data.
"""
import sys, time
import numpy as np
from scipy.special import gammaln
from scipy.linalg import solve
sys.path.insert(0, "/home/user/gwm_repo")
sys.path.insert(0, "/home/user/gwm_repo/cells/2/exploratory")
from zoo.wright_fisher import WrightFisher
from review_c2_fp import F_SEL, Q_UNI, Q_CYC, hodge_residual

K = 3

def run(N, dt, uU, uC):
    Q = uU * Q_UNI + uC * Q_CYC
    wf = WrightFisher(K=3, mutation=Q, dt=dt, n_sub=20)
    cnt = np.array([(i, j, N - i - j) for i in range(N + 1) for j in range(N + 1 - i)])
    x = cnt / N
    n = len(x)
    p = wf._flow(x.copy(), "mutation_selection", dict(f=np.tile(F_SEL, (n, 1)), u=np.ones(n)), wf.cfg, wf._Q(wf.cfg), None)
    p = np.clip(p, 1e-300, None); p /= p.sum(1, keepdims=True)
    logP = gammaln(N + 1) - gammaln(cnt + 1).sum(1)[None, :] + np.log(p) @ cnt.T.astype(float)   # [from, to]
    P = np.exp(logP)
    P /= P.sum(1, keepdims=True)
    A = P.T - np.eye(n); A[0] = 1.0
    rhs = np.zeros(n); rhs[0] = 1.0
    pi = np.clip(solve(A, rhs), 1e-300, None); pi /= pi.sum()
    L = np.log(pi)[:, None] + np.log(np.clip(P, 1e-300, None))
    del P, logP
    ep_step = 0.5 * float(((np.exp(L) - np.exp(L.T)) * (L - L.T)).sum())
    del L
    # drift-based EP: Hodge residual (D^{-1} metric, D = P/(2 N dt)) of the exact mean drift, weights pi, interior states
    inner = (cnt > 0).all(1)
    v = (p[inner] - x[inner]) / dt
    r, tot = hodge_residual(x[inner], pi[inner], v, 8, 3, Ne=N * dt)
    return ep_step / dt, r, pi[inner].sum()

def k2_binomial_ep(N, dt, u=1.0, s=1.0):
    """K = 2 analogue (exact flow: symmetric mutation then selection, then Binomial(N)); every 1-D diffusion is
    reversible, so any EP of this chain is a discrete-generation effect."""
    i = np.arange(N + 1); x = i / N
    p = 0.5 + (x - 0.5) * np.exp(-u * dt)
    p = p * np.exp(s * dt) / (p * np.exp(s * dt) + 1 - p)
    lp = (gammaln(N + 1) - gammaln(i + 1)[None, :] - gammaln(N - i + 1)[None, :]
          + np.outer(np.log(p), i) + np.outer(np.log1p(-p), N - i))
    P = np.exp(lp); P /= P.sum(1, keepdims=True)
    A = P.T - np.eye(N + 1); A[0] = 1.0
    rhs = np.zeros(N + 1); rhs[0] = 1.0
    pi = np.clip(solve(A, rhs), 1e-300, None); pi /= pi.sum()
    L = np.log(pi)[:, None] + np.log(np.clip(P, 1e-300, None))
    return 0.5 * float(((np.exp(L) - np.exp(L.T)) * (L - L.T)).sum())

if __name__ == "__main__":
    if "--scan" in sys.argv:      # discrete-generation EP floor of gradient worlds at fixed Ne = N dt = 10
        for N in (25, 50, 100, 141):
            ep, r, m = run(N, 10.0 / N, 1.0, 0.0)
            print(f"K=3 uniform u=1 + sel  N={N:4d} dt={10 / N:.4f}: EP_chain/dt = {ep:.4f}  (per generation {ep * 10 / N:.5f}); drift-Hodge {r:.1e}")
        for N in (50, 100, 200, 400, 800, 1600):
            e = k2_binomial_ep(N, 10.0 / N)
            print(f"K=2 mutation + sel     N={N:4d} dt={10 / N:.4f}: EP_chain/dt = {e / (10 / N):.4f}  (per generation {e:.2e})")
        sys.exit()
    cases = {"null: uniform u=1 + sel": (1.0, 0.0), "uniform 0.5 + cyclic 1 + sel": (0.5, 1.0), "cyclic u=1 + sel": (0.0, 1.0)}
    diff_limit = {"null: uniform u=1 + sel": 0.0, "uniform 0.5 + cyclic 1 + sel": 0.7211, "cyclic u=1 + sel": 0.9472}
    print(f"{'case':30} {'N':>4} {'dt':>5} {'EP_chain/dt':>12} {'Hodge(drift)':>12} {'diff. limit':>11} {'interior mass':>13} {'sec':>5}")
    for name, (uU, uC) in cases.items():
        for N, dt in ((50, 0.2), (100, 0.1), (141, 10 / 141)):
            t0 = time.time()
            ep, r, m = run(N, dt, uU, uC)
            print(f"{name:30} {N:4d} {dt:5.2f} {ep:12.4f} {r:12.4f} {diff_limit[name]:11.4f} {m:13.4f} {time.time() - t0:5.1f}")
