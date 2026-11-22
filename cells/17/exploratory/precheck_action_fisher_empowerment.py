"""Cell 17, check before preregistration (ground truth only; nothing is learned).

Local action channel s' = f(s, a) + noise, linearised at (s, a): y = J a + e, e ~ N(0, Sigma). The cell's readout is
the action Fisher matrix G_A = J^T Sigma^-1 J and its numerical rank ("effective action dimension"). Local
(Gaussian, power-constrained) empowerment is the capacity of the same channel (Salge, Glackin, Polani 2014, MIMO
section): water-filling over the eigenvalues of G_A.
A. capacity = water-filling on eig(G_A): checked against random feasible input covariances (none exceeds it);
B. two channels with different J but the same eig(G_A) give the same capacity and the same effective dimension;
C. effective dimension at threshold tau = number of active water-filling channels at water level 1/tau (exact);
D. low power: C(P) / P -> lambda_max(G_A) / 2;
E. the zoo's only continuous-action world (pendulum, one torque): G_A is 1 x 1 and the same at every state.
Runs in a few seconds.
"""
import numpy as np

import zoo

rng = np.random.default_rng(0)


def waterfill(lam, P):
    lam = np.sort(lam)[::-1]
    inv = 1.0 / lam
    for k in range(len(lam), 0, -1):
        nu = (P + inv[:k].sum()) / k
        if nu > inv[k - 1]:
            p = np.maximum(nu - inv, 0.0)
            return 0.5 * np.sum(np.log1p(p * lam)), int((p > 0).sum()), nu
    raise RuntimeError


def capacity_logdet(J, Sig, Q):
    W = np.linalg.inv(np.linalg.cholesky(Sig))
    M = W @ J
    return 0.5 * np.linalg.slogdet(np.eye(len(M)) + M @ Q @ M.T)[1]


def main():
    d_obs, d_act, P = 4, 3, 2.0
    J = rng.standard_normal((d_obs, d_act))
    B = rng.standard_normal((d_obs, d_obs))
    Sig = B @ B.T / d_obs + 0.1 * np.eye(d_obs)
    G = J.T @ np.linalg.solve(Sig, J)
    lam, V = np.linalg.eigh(G)
    C, k, nu = waterfill(lam, P)
    p = np.maximum(nu - 1 / lam, 0)
    Q = V @ np.diag(p) @ V.T
    best_random = 0.0
    for _ in range(20000):
        R = rng.standard_normal((d_act, d_act))
        Qr = R @ R.T
        Qr *= P / np.trace(Qr)
        best_random = max(best_random, capacity_logdet(J, Sig, Qr))
    print(f"A. water-filling on eig(G_A): C = {C:.6f}; log-det at the water-filling input: {capacity_logdet(J, Sig, Q):.6f};"
          f" best of 20000 random inputs with the same power: {best_random:.6f}")
    U = np.linalg.qr(rng.standard_normal((d_obs, d_obs)))[0]
    R = np.linalg.qr(rng.standard_normal((d_act, d_act)))[0]
    L = np.linalg.cholesky(Sig)
    J2 = L @ U @ np.linalg.solve(L, J) @ R                   # different channel, same spectrum of G_A
    G2 = J2.T @ np.linalg.solve(Sig, J2)
    print(f"B. a different J with the same spectrum: |J2 - J| = {np.abs(J2 - J).max():.2f}; capacity {waterfill(np.linalg.eigvalsh(G2), P)[0]:.6f}"
          f" vs {C:.6f}; eigenvalues equal to {np.abs(np.sort(np.linalg.eigvalsh(G2)) - np.sort(lam)).max():.1e}")
    ok = True
    for tau in np.geomspace(lam.min() / 2, lam.max() * 2, 50):
        eff = int((lam > tau).sum())
        Ptau = np.maximum(1 / tau - 1 / lam, 0).sum()        # the power at which the water level is 1 / tau
        active = waterfill(lam, Ptau)[1] if Ptau > 0 else 0
        ok &= eff == active
    print(f"C. effective dimension at threshold tau == active water-filling channels at water level 1/tau, 50 thresholds: {ok}")
    for P_small in (1e-2, 1e-3, 1e-4):
        print(f"D. P = {P_small:.0e}: C / P = {waterfill(lam, P_small)[0] / P_small:.5f}, lambda_max / 2 = {lam.max() / 2:.5f}")
    w = zoo.make("pendulum")
    s = np.stack([rng.uniform(-np.pi, np.pi, 500), rng.uniform(-2.5, 2.5, 500)], 1)
    h = 1e-4
    up = w.rollout(s, 1, actions=np.full((500, 1, 1), h), gamma=0.1)[:, 1]
    dn = w.rollout(s, 1, actions=np.full((500, 1, 1), -h), gamma=0.1)[:, 1]
    Ja = (up - dn) / (2 * h)                                  # d(theta', omega') / du, one step dt = 0.1
    g = (Ja ** 2).sum(1)                                      # G_A for isotropic unit noise, 1 x 1
    print(f"E. pendulum, one torque, 500 random states: d(theta')/du in [{Ja[:, 0].min():.5f}, {Ja[:, 0].max():.5f}],"
          f" d(omega')/du in [{Ja[:, 1].min():.5f}, {Ja[:, 1].max():.5f}]; G_A in [{g.min():.6f}, {g.max():.6f}]: rank 1 everywhere")


if __name__ == "__main__":
    main()
