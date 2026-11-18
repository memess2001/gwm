"""Cell 13, check before preregistration (ground truth only; nothing is learned).

IGCI for a deterministic, invertible relation Y = f(X) with both variables rescaled to [0, 1] (uniform reference):
    C(X -> Y) = E[log f'(X)] = H(Y) - H(X)          (differential entropies; infer X -> Y iff C < 0)
A. The slope estimator and the entropy difference agree (two estimators of one number).
B. A monotone recoding X' = g(X) of the cause leaves the causal structure and every prediction unchanged (the
   mechanism becomes f o g^-1), but changes H(X') by E[log g'(X)]. In a world model's latent space such per-
   coordinate recodings are free (the encoder can apply them, the transition and decoder absorb them), so the sign
   of an IGCI readout between latent coordinates is a choice of coordinates. Count how often a random recoding
   flips a correct IGCI decision.
C. On a stationary process the marginals of z_t and z_{t+1} are equal, so C(z_t -> z_{t+1}) = H(z_{t+1}) - H(z_t)
   = 0 whatever the dynamics: the forward/backward "orthogonality difference" carries no arrow of time there.
Runs in a few seconds.
"""
import numpy as np
from scipy.stats import beta as Beta

rng = np.random.default_rng(0)
N = 4000


def unit(v):
    return (v - v.min()) / (v.max() - v.min())


def slope_C(x, y):
    """IGCI slope estimator (Daniusis et al. 2010) on [0, 1]-rescaled data."""
    x, y = unit(x), unit(y)
    o = np.argsort(x)
    dx, dy = np.diff(x[o]), np.diff(y[o])
    ok = (dx != 0) & (dy != 0)
    return float(np.mean(np.log(np.abs(dy[ok] / dx[ok]))))


def entropy_m(v, m=None):
    """Vasicek m-spacing estimator of the differential entropy of [0, 1]-rescaled data."""
    v = np.sort(unit(v))
    n = len(v)
    m = m or int(round(np.sqrt(n)))
    d = v[m:] - v[:-m]
    d = d[d > 0]
    return float(np.mean(np.log(n / m * d)))


def random_monotone():
    a, b = np.exp(rng.uniform(np.log(0.3), np.log(4.0), 2))
    return lambda u: Beta.cdf(u, a, b)


def main():
    print("A. identity: slope estimator vs entropy difference, Y = f(X), X ~ U[0,1]")
    for name, f in (("x^3", lambda u: u ** 3), ("sqrt", np.sqrt), ("Beta(0.5,2) cdf", lambda u: Beta.cdf(u, 0.5, 2))):
        x = rng.uniform(0, 1, N)
        y = f(x)
        print(f"   f = {name:16} slope C = {slope_C(x, y):+.3f}   H(Y) - H(X) = {entropy_m(y) - entropy_m(x):+.3f}"
              f"   -> {'X -> Y (correct)' if slope_C(x, y) < 0 else 'Y -> X (wrong)'}")

    print("\nB. recoding the cause, X' = g(X), g monotone: the causal direction X' -> Y is unchanged")
    x = rng.uniform(0, 1, N)
    y = x ** 3
    for k in (1, 2, 4, 10):
        xp = x ** k
        c = slope_C(xp, y)
        print(f"   g(x) = x^{k:<2d}: C(X' -> Y) = {c:+.3f} -> {'X -> Y (correct)' if c < 0 else 'Y -> X (wrong)'}")
    flips, total = 0, 0
    for _ in range(300):
        f, g = random_monotone(), random_monotone()
        x = rng.uniform(0, 1, N)                      # IGCI's postulate holds exactly on the native coordinates
        y = f(x)
        if slope_C(x, y) >= 0:
            continue                                  # keep worlds where IGCI is right on the native coordinates
        total += 1
        flips += slope_C(g(x), y) >= 0
    print(f"   random Beta-cdf mechanisms f and recodings g: IGCI right on the native coordinates in {total} of 300 worlds;"
          f" a random recoding of the cause flips it in {flips} of them ({flips / total:.0%})")

    print("\nC. stationary process: C(z_t -> z_{t+1}) and C(z_{t+1} -> z_t) on the same pairs")
    # circle rotation by an irrational angle, observed through a monotone chart of [0, 1): invertible, stationary
    z = (0.137 + np.arange(N + 1) * (np.sqrt(5) - 1) / 2) % 1.0
    for name, h in (("identity chart", lambda u: u), ("chart u^2", lambda u: u ** 2), ("chart Beta(3,1) cdf", lambda u: Beta.cdf(u, 3, 1))):
        a, b = h(z[:-1]), h(z[1:])
        print(f"   {name:20} H(z_t+1) - H(z_t) = {entropy_m(b) - entropy_m(a):+.4f}   (the rotation is piecewise-linear, so the"
              f" slope estimator is not used here)")


if __name__ == "__main__":
    main()
