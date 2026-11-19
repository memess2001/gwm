"""Cell 20, 2026-11-19 (Antigravity review round). Two-wavenumber shape loop (1/w and 3/w quadrature components), no
noise: where does death/birth (longest consecutive step vs shortest antipodal distance, as in the review snippet) drop
below 2.8, and how does that depend on the number n of samples around the loop? Plus a block average whose sinc zero
hits only one quadrature component.  Run from the repo root:  python cells/20/exploratory/precheck_anisotropic_blur.py
"""
import numpy as np
L = 256
x = np.arange(L) - L / 2
blur_hat = lambda s: np.exp(-(2 * np.pi * np.fft.fftfreq(L)) ** 2 * s ** 2 / 2)
def ratio(n, w, sig):
    phis = np.arange(n) * 2 * np.pi / n
    g = np.exp(-x ** 2 / (2 * w ** 2))
    U = g[None] * (1 + 0.3 * np.cos(x[None] / w) * np.cos(phis[:, None]) + 0.3 * np.cos(3 * x[None] / w) * np.sin(phis[:, None]))
    Ub = np.real(np.fft.ifft(np.fft.fft(U, axis=1) * blur_hat(sig)[None], axis=1))
    step = np.linalg.norm(Ub - np.roll(Ub, 1, 0), axis=1).max()
    anti = np.linalg.norm(Ub - np.roll(Ub, n // 2, 0), axis=1).min()
    return anti / step
w = 8
for n in (64, 256, 1024):
    grid = np.linspace(0.5, 2.0, 151) * w
    r = np.array([ratio(n, w, s) for s in grid])
    cross = grid[np.argmax(r < 2.8)] / w if (r < 2.8).any() else np.nan
    print(f"n = {n:4d}: death/birth < 2.8 first at sigma = {cross:.2f} w")
# box filter of width 8: zero at wavelength 8; loop = cos(2 pi x / 8) cos(phi) + cos(2 pi x / 16) sin(phi)
n = 64; phis = np.arange(n) * 2 * np.pi / n
U = np.cos(2 * np.pi * x[None] / 8) * np.cos(phis[:, None]) + np.cos(2 * np.pi * x[None] / 16) * np.sin(phis[:, None])
ker = np.zeros(L); ker[:8] = 1 / 8
Ub = np.real(np.fft.ifft(np.fft.fft(U, axis=1) * np.fft.fft(ker)[None], axis=1))
sv = np.linalg.svd(Ub - Ub.mean(0), compute_uv=False)
print(f"box B=8 on (wavelength 8, wavelength 16) loop: singular values of the image cloud = {sv[:3].round(3)} (rank 1 = a segment)")
