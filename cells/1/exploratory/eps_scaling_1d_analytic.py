"""Check 2b (clean): same 1-D wall as eps_scaling_1d.py, but the one-step map is convolved with N(0, eps^2) in
closed form (no quadrature staircase).  m := x + d - w (signed overshoot), Phi/phi = standard normal cdf/pdf.
  slide   F(y) = min(y + d, w)            -> E = w + m Phi(-m/eps) - eps phi(m/eps)
  reject  F(y) = y + d 1[y + d <= w]      -> E = x + d Phi(-m/eps)
  reflect F(y) = w - |w - y - d|          -> E = w - [ (-m)(1 - 2 Phi(m/eps)) + 2 eps phi(m/eps) ]
C_k = sum_{j=0..k} (dF^j/dx)^2 by the chain rule (analytic derivatives); free value k+1."""
import numpy as np
from scipy.stats import norm

w, d, k = 5.0, 0.25, 4


def F_and_dF(name, x, eps):
    m = x + d - w
    if eps == 0:
        if name == "slide":
            return np.minimum(x + d, w), (m < 0).astype(float)
        if name == "reject":
            return np.where(m > 0, x, x + d), np.ones_like(x)          # a.e. derivative; jump not counted
        return w - np.abs(m), -np.sign(m) * 1.0 + (m == 0)
    z = m / eps
    if name == "slide":
        return w + m * norm.cdf(-z) - eps * norm.pdf(z), norm.cdf(-z)
    if name == "reject":
        return x + d * norm.cdf(-z), 1 - d / eps * norm.pdf(z)
    # reflect: E|w - y - d| with w - y - d = -(m + eps xi):  E|m + eps xi| = m(1 - 2Phi(-z)) + 2 eps phi(z)
    Eabs = m * (1 - 2 * norm.cdf(-z)) + 2 * eps * norm.pdf(z)
    dEabs = 1 - 2 * norm.cdf(-z)
    return w - Eabs, -dEabs


xs = np.linspace(4.0 + 1e-6, 5.0 - 1e-6, 200001)       # the band, very fine grid
print(f"{'model':>8} {'eps':>6} | frac compressed | frac stretched | min / max of 0.5 log(C_k/(k+1))")
for name in ("slide", "reject", "reflect"):
    for eps in (0.1, 0.05, 0.02, 0.01, 0.005, 0.0):
        x, D, C = xs.copy(), np.ones_like(xs), np.ones_like(xs)
        for _ in range(k):
            x, dF = F_and_dF(name, x, eps)
            D = D * dF; C = C + D ** 2
        r = 0.5 * np.log(np.maximum(C, 1e-300) / (k + 1))
        print(f"{name:>8} {eps:6.3f} | {(r < -0.05).mean():15.4f} | {(r > 0.05).mean():14.4f} | {r.min():7.2f} / {r.max():7.2f}")
