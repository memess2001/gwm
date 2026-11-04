"""Exploratory (disclosed) check for cell 15: fake dissipation from the observation map.

For an undamped pendulum (true phase-space divergence 0), the divergence measured w.r.t. the
volume form pulled back from a frame-pair observation map R(theta, omega) =
(frame(theta), frame(theta - omega*dt)) is
    div_nu F = div F + F . grad log nu,   nu = sqrt(det(J_R^T J_R)) = dt * a(theta) * a(theta - omega*dt),
with a(theta) = ||d frame / d theta||.  Rotation-equivariant renderer -> a ~ const -> no fake term.
Oblique / projective camera -> a(theta) varies -> fake divergence ~ 2*omega*(log a)'(theta).
We compute nu directly from finite-difference Jacobians of the actual rendered frame pairs.
"""
import numpy as np
from scipy.ndimage import map_coordinates

import zoo
from zoo.pendulum import render

w = zoo.make("pendulum")
dt = 0.1


def frames(theta):
    # float renderer (no uint8 quantisation: quantisation noise would dominate finite differences)
    return render(theta, quantize=False).astype(np.float64)  # [n, 64, 64]


def projective_warp(img, k):
    """Camera tilted about the horizontal axis: rows far from the camera are compressed.
    Output pixel (r, c) samples the source at a homography-mapped location; k=0 is identity."""
    H, W = img.shape[-2:]
    r, c = np.mgrid[0:H, 0:W].astype(np.float64)
    yc, xc = (r - 31.5) / 32.0, (c - 31.5) / 32.0
    denom = 1.0 - k * yc
    ys, xs = yc / denom, xc / denom
    rs, cs = ys * 32.0 + 31.5, xs * 32.0 + 31.5
    out = np.stack([map_coordinates(im, [rs, cs], order=1, mode="constant") for im in img.reshape(-1, H, W)])
    return out.reshape(img.shape)


def a_of_theta(thetas, k, h=1e-3):
    f = lambda th: projective_warp(frames(th), k) if k else frames(th)
    d = (f(thetas + h) - f(thetas - h)) / (2 * h)
    return np.sqrt((d.reshape(len(thetas), -1) ** 2).sum(1))


thetas = np.linspace(-np.pi, np.pi, 721, endpoint=False)
omegas = np.linspace(-2.5, 2.5, 51)
TH, OM = np.meshgrid(thetas, omegas, indexing="ij")


def fake_div(dloga):
    interp = lambda x: np.interp(np.mod(x + np.pi, 2 * np.pi) - np.pi, thetas, dloga, period=2 * np.pi)
    return OM * (interp(TH) + interp(TH - OM * dt)) + (-np.sin(TH)) * (-dt) * interp(TH - OM * dt)


print("true divergence of the undamped pendulum = 0; for scale, damping gamma=0.05 gives -2*gamma = -0.100")
for k in (0.0, 0.3, 0.6):
    a = a_of_theta(thetas, k)
    loga = np.log(a)
    # split log a into low-frequency (camera geometry; a smooth learned decoder inherits it) and
    # high-frequency (pixel-grid aliasing; a smooth decoder would not reproduce it) parts
    F = np.fft.rfft(loga)
    n = np.arange(len(F))
    for name, keep in (("all", n >= 0), ("low (harmonics<=4)", n <= 4)):
        Fk = np.where(keep, F, 0)
        dloga = np.fft.irfft(1j * n * Fk, len(loga))  # analytic derivative on the circle
        fd = fake_div(dloga)
        print(f"warp k={k:.1f} [{name:>18s}]: a max/min = {a.max()/a.min():.2f}; |(log a)'| max = {np.abs(dloga).max():.3f}; "
              f"fake div max |.| = {np.abs(fd).max():.3f}, RMS = {np.sqrt((fd**2).mean()):.3f}")
