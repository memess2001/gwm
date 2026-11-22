"""Cell 26, check before preregistration (ground truth only; nothing is learned).

Linear-Gaussian case of "mutual-information-preserving temporal coarse-graining": a reversible Ornstein-Uhlenbeck
process with relaxation rates lambda_i, observed through an arbitrary invertible linear map M (the observer's chart).
A. MI(x_t ; x_{t+tau}) from the joint covariance equals -1/2 sum log(1 - s_i(tau)^2), s_i = singular values of the
   whitened lagged covariance (the Koopman operator on linear features, VAMP; Wu & Noe), equals
   -1/2 sum log(1 - exp(-2 lambda_i tau)); it does not depend on M.
B. The lag tau* at which keeping only the slowest component retains 99 % of the MI is a function of the rates alone:
   the "natural event scale" of the coarse-graining is set by the fast relaxation time, i.e. the spectral gap.
Runs in a second.
"""
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov, sqrtm

rng = np.random.default_rng(0)


def gauss_mi(C0, Ct):
    J = np.block([[C0, Ct.T], [Ct, C0]])
    return 0.5 * (2 * np.linalg.slogdet(C0)[1] - np.linalg.slogdet(J)[1])


def main():
    lam = np.array([1.0, 12.0])
    A = np.diag(lam)
    C0x = solve_continuous_lyapunov(A, -np.eye(2)) * -1          # A C + C A^T = I  ->  C = A^-1 / 2 (reversible)
    for trial in range(3):
        M = rng.standard_normal((2, 2))                           # observer's chart y = M x
        C0 = M @ C0x @ M.T
        rows = []
        for tau in (0.01, 0.1, 0.5, 2.0):
            Ct = M @ expm(-A * tau) @ C0x @ M.T                     # Cov(y_{t+tau}, y_t)
            W = np.linalg.inv(np.real(sqrtm(C0)))
            s = np.linalg.svd(W @ Ct @ W, compute_uv=False)
            rows.append((gauss_mi(C0, Ct), -0.5 * np.sum(np.log1p(-s ** 2)), -0.5 * np.sum(np.log1p(-np.exp(-2 * lam * tau)))))
        dev = max(abs(a - b) + abs(a - c) for a, b, c in rows)
        print(f"A. chart {trial}: MI at tau = 0.01, 0.1, 0.5, 2: " + ", ".join(f"{r[0]:.4f}" for r in rows)
              + f"; max deviation of the singular-value and rate formulas: {dev:.1e}")

    print("B. lag at which the slowest component alone keeps 99 % of the MI")
    for fast in (3.0, 12.0, 48.0):
        l = np.array([1.0, fast])
        taus = np.geomspace(1e-3, 10, 20000)
        mi = -0.5 * np.log1p(-np.exp(-2 * l[None, :] * taus[:, None]))
        frac = mi[:, 0] / mi.sum(1)
        t_star = taus[np.argmax(frac >= 0.99)]
        print(f"   rates (1, {fast:4.0f}): tau* = {t_star:.4f};  tau* x fast rate = {t_star * fast:.3f}")


if __name__ == "__main__":
    main()
