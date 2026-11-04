"""Check 2b: 1-D wall at x = 5 (free side x < 5), action d = +0.25 into the wall, k = 4 (band (4,5)).
One-step maps: slide/clip  x' = min(x+d, 5);  reject  x' = x if x+d > 5 else x+d;  reflect (fold)  x' = 5 - |5 - (x+d)|.
A smooth learned 1-step model is proxied by Gaussian input smoothing at scale eps (eps -> 0 = exact simulator).
C_k = sum_{j=0..k} (dF^j/dx)^2 ; free value k+1.  Report the fraction of the band where |0.5 log(C_k/(k+1))| > 0.05,
split into compression (C_k < k+1) and stretching (C_k > k+1), and the peak stretching."""
import numpy as np

w, d, k = 5.0, 0.25, 4
maps = {"slide": lambda x: np.minimum(x + d, w),
        "reject": lambda x: np.where(x + d > w, x, x + d),
        "reflect": lambda x: w - np.abs(w - (x + d))}
xi, wq = np.polynomial.hermite_e.hermegauss(41); wq = wq / wq.sum()
xs = np.linspace(4.0 + 1e-4, 5.0 - 1e-4, 20001)          # the band; grid much finer than any eps
h = 1e-5

def kstep(f, x, eps):
    out = [x]
    for _ in range(k):
        x = f(x) if eps == 0 else (f(x[:, None] + eps * xi[None]) * wq).sum(1)
        out.append(x)
    return out

print(f"{'model':>8} {'eps':>6} | frac compressed | frac stretched | peak 0.5log(C/(k+1))")
for name, f in maps.items():
    for eps in (0.1, 0.05, 0.02, 0.01, 0.005, 0.0):
        A, Bm = kstep(f, xs + h, eps), kstep(f, xs - h, eps)
        C = sum(((a - b) / (2 * h)) ** 2 for a, b in zip(A, Bm))
        r = 0.5 * np.log(C / (k + 1))
        comp, stre = (r < -0.05).mean(), (r > 0.05).mean()
        print(f"{name:>8} {eps:6.3f} | {comp:15.3f} | {stre:14.3f} | {r.max():8.2f}")
