"""Cell 27, check before preregistration (ground truth only; nothing is learned).

"Holonomy around a time loop (one period)" has two readings.
Reading A (transport = the world's own state evolution over one period T): the holonomy is the monodromy
  (period map) M = Phi(T). It is not the identity for a stationary world unless T is a multiple of the world's own
  period, so "periodic or stationary world -> holonomy 0" fails.
Reading B (transport = alignment of the representation at t with the one at t + T): time is one-dimensional, every
  connection on a line is flat, and the composite over one period is the net change of the representation over that
  period: the drift. Its growth with the drift rate is the definition of drift.
Here: a linear oscillator x'' + omega(t)^2 x = 0, period of observation T = 2 pi.
  1. stationary world (omega constant, not an integer): |M - I| > 0;
  2. drifting world (omega = omega0 + kappa t): the change of the period map between consecutive periods divided by
     kappa T is the same for every kappa (first-order: dM/d omega), i.e. "holonomy change" = drift x a fixed factor.
Runs in a second.
"""
import numpy as np
from scipy.integrate import solve_ivp

T = 2 * np.pi


def period_map(omega_of_t, t0):
    cols = []
    for e in np.eye(2):
        sol = solve_ivp(lambda t, z: [z[1], -omega_of_t(t) ** 2 * z[0]], (t0, t0 + T), e, rtol=1e-11, atol=1e-12)
        cols.append(sol.y[:, -1])
    return np.array(cols).T


def main():
    for om in (1.0, 1.3, 2.0, 2.5):
        M = period_map(lambda t: om, 0.0)
        print(f"1. stationary oscillator, omega = {om}: |M - I| = {np.linalg.norm(M - np.eye(2)):.4f}")
    for kappa in (1e-4, 1e-3, 1e-2):
        om = lambda t: 1.3 + kappa * t
        M0, M1 = period_map(om, 0.0), period_map(om, T)
        print(f"2. drift rate {kappa:.0e}: |M(next period) - M(this period)| / (kappa T) = {np.linalg.norm(M1 - M0) / (kappa * T):.4f}")


if __name__ == "__main__":
    main()
