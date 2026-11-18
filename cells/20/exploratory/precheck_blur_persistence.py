"""Cell 20, check before preregistration (ground truth only; nothing is learned).

The cell coarse-grains the world state by a spatial scale and asks at which scale a loop (S^1) of the attractor
disappears, predicting "vanishing scale proportional to the kernel radius". A 1-D periodic field shows what sets
that scale. Coarse-graining = Gaussian blur of width sigma (a linear filter, Fourier factor exp(-k^2 sigma^2 / 2)).
Two loops:
  A. translation loop: a bump of width w carried once around a periodic domain of length L;
  B. internal loop: a bump of width w whose shape oscillates, u_phi = g_w (1 + d cos(c x / w - phi)), in the
     co-moving frame (the loop is an ellipse in the plane of the two deformation modes).
Readout: the longest H1 bar of the Vietoris-Rips complex of the orbit (ripser), judged by the project's rule
death / birth >= 2.8 (cells/0 calibration), with observation noise of sd eta added to every grid value after the
blur. A blur never removes a loop topologically (a linear injective-on-the-loop map keeps it a loop); it shrinks
the loop's size, and the rule fails where the shrunk loop meets the noise floor.
Runs in about a minute on one free CPU thread.
"""
import warnings

import numpy as np
from ripser import ripser

warnings.filterwarnings("ignore", message="The input point cloud has more columns than rows")
warnings.filterwarnings("ignore", message="The input matrix is square")       # point clouds, not distance matrices

rng = np.random.default_rng(0)


def circ_bump(L, w):
    x = np.arange(L)
    d = np.minimum(x, L - x)
    return np.exp(-d ** 2 / (2 * w ** 2))


def blur_hat(L, sigma):
    k = 2 * np.pi * np.fft.fftfreq(L)
    return np.exp(-k ** 2 * sigma ** 2 / 2)


def h1(points):
    dg = ripser(points, maxdim=1)["dgms"][1]
    if len(dg) == 0:
        return 0.0, 0.0
    i = np.argmax(dg[:, 1] / np.maximum(dg[:, 0], 1e-12))
    return float(dg[i, 0]), float(dg[i, 1])


def translation_orbit(L, w, sigma, eta, n=128):
    u = np.fft.fft(circ_bump(L, w)) * blur_hat(L, sigma)
    k = np.fft.fftfreq(L) * L
    shifts = np.arange(n) * L / n
    U = np.real(np.fft.ifft(u[None, :] * np.exp(-2j * np.pi * k[None, :] * shifts[:, None] / L), axis=1))
    return U + eta * rng.standard_normal(U.shape)


def internal_orbit(L, w, sigma, eta, c=3.0, d=0.3, n=64):
    x = np.arange(L) - L / 2
    g = np.exp(-x ** 2 / (2 * w ** 2))
    phis = np.arange(n) * 2 * np.pi / n
    U = g[None, :] * (1 + d * np.cos(c * x[None, :] / w - phis[:, None]))
    U = np.real(np.fft.ifft(np.fft.fft(U, axis=1) * blur_hat(L, sigma)[None, :], axis=1))
    return U + eta * rng.standard_normal(U.shape)


def vanishing(make, sigmas):
    """First sigma on the grid at which the longest bar fails death / birth >= 2.8."""
    for s in sigmas:
        b, dth = h1(make(s))
        if b == 0 or dth / b < 2.8:
            return s
    return None


def main():
    print("A. translation loop (bump carried once around the domain), noise sd 0.01")
    print("   death/birth of the longest H1 bar, no noise, L = 128:")
    for w in (3, 6, 12):
        row = []
        for s in (0, 4, 16, 32):
            b, dth = h1(translation_orbit(128, w, s, 0.0))
            row.append(f"sigma {s:2d}: {dth / b:6.1f}")
        print(f"     w = {w:2d}: " + "   ".join(row))
    sig = np.round(np.geomspace(1, 160, 30), 2)
    for L in (128, 256):
        out = [f"w = {w:2d}: {vanishing(lambda s: translation_orbit(L, w, s, 0.01), sig)}" for w in (3, 6, 12)]
        print(f"   vanishing sigma, L = {L}: " + "   ".join(out))

    print("\nB. internal loop (shape oscillation of a bump of width w, deformation wavenumber 3 / w), L = 256")
    sig = np.round(np.geomspace(0.25, 40, 40), 2)
    for eta in (0.003, 0.01):
        out = []
        for w in (4, 8, 16):
            v = vanishing(lambda s: internal_orbit(256, w, s, eta), sig)
            out.append(f"w = {w:2d}: {v} ({v / w:.2f} w)" if v else f"w = {w:2d}: none")
        print(f"   noise sd {eta}: vanishing sigma  " + "   ".join(out))
    x = np.arange(256) - 128
    for w in (4, 8, 16):
        A = np.exp(-x ** 2 / (2 * w ** 2)) * np.cos(3 * x / w)
        Ah = np.fft.fft(A)
        f = lambda s: np.linalg.norm(np.fft.ifft(Ah * blur_hat(256, s))) / np.linalg.norm(A)
        print(f"   w = {w:2d}: size of the blurred deformation mode, |blur(A)| / |A|, at sigma = 0.25 w, 0.5 w, 1 w: "
              f"{f(0.25 * w):.3f}, {f(0.5 * w):.3f}, {f(1.0 * w):.4f}")


if __name__ == "__main__":
    main()
