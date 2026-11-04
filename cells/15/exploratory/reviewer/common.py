"""Independent re-implementation helpers for the cell-15 fake-divergence review (float64 throughout)."""
import numpy as np
from scipy.ndimage import map_coordinates

R0, ROD_S, BOB_S = 22.0, 0.8, 2.5
DT = 0.1


def scene(dx, dy, th):
    """Continuous zoo pendulum scene (same formula as zoo.pendulum.render) at plane points (dx, dy) in source-pixel
    units relative to the pivot; th [n] -> [n, P]. float64."""
    s, c = np.sin(th)[:, None], np.cos(th)[:, None]
    proj = s * dx[None] + c * dy[None]
    t = np.clip(proj, 0.0, R0)
    r2 = (dx * dx + dy * dy)[None]
    rod = np.exp(-(r2 - 2 * t * proj + t * t) * (0.5 / ROD_S ** 2))
    bob = np.exp(-(r2 - 2 * R0 * proj + R0 * R0) * (0.5 / BOB_S ** 2))
    return 1.0 - (1.0 - rod) * (1.0 - bob)


def out_grid(k, size=64, ss=1, half=1.0, kind="proj", phi=0.0):
    """Plane sample points for an output canvas. Output normalized coords (yc, xc) in [-half, half] (half=1 is the
    zoo 64x64 field of view); size*ss samples per side. kind='proj': the script's homography ys = yc/(1-k yc),
    xs = xc/(1-k yc). kind='affine': ys = yc / cos(phi) (vertical foreshortening). Returns source-pixel offsets."""
    n = size * ss
    half_px = half * 32.0
    # pixel centres of an n x n canvas spanning [-half, half] with the zoo convention (r - 31.5)/32 at size 64
    g = (np.arange(n) + 0.5) / n * 2 * half - half
    yc, xc = np.meshgrid(g, g, indexing="ij")
    if kind == "proj":
        den = 1.0 - k * yc
        ys, xs = yc / den, xc / den
    elif kind == "affine":
        ys, xs = yc / np.cos(phi), xc
    else:
        raise ValueError(kind)
    return (xs * 32.0).ravel(), (ys * 32.0).ravel(), (n, n)


def a_analytic(thetas, k, size=64, ss=1, half=1.0, kind="proj", phi=0.0, h=1e-5, chunk=64):
    """||d frame/d theta|| of the analytically warped scene (point sampling at output pixel centres, no
    interpolation, no source-image boundary). ss>1: ss x ss supersampled canvas (higher-resolution camera)."""
    dx, dy, _ = out_grid(k, size, ss, half, kind, phi)
    out = np.empty(len(thetas))
    for i in range(0, len(thetas), chunk):
        th = thetas[i:i + chunk]
        d = (scene(dx, dy, th + h) - scene(dx, dy, th - h)) / (2 * h)
        out[i:i + chunk] = np.sqrt((d ** 2).sum(1))
    return out


# ---- the script's pipeline (zoo float32 renderer + bilinear projective warp), for comparison
def script_frames(theta, k):
    from zoo.pendulum import render
    f = render(theta, quantize=False).astype(np.float64)
    if not k:
        return f
    H, W = f.shape[-2:]
    r, c = np.mgrid[0:H, 0:W].astype(np.float64)
    yc, xc = (r - 31.5) / 32.0, (c - 31.5) / 32.0
    den = 1.0 - k * yc
    rs, cs = yc / den * 32.0 + 31.5, xc / den * 32.0 + 31.5
    return np.stack([map_coordinates(im, [rs, cs], order=1, mode="constant") for im in f.reshape(-1, H, W)]).reshape(f.shape)


def a_script(thetas, k, h=1e-3):
    d = (script_frames(thetas + h, k) - script_frames(thetas - h, k)) / (2 * h)
    return np.sqrt((d.reshape(len(thetas), -1) ** 2).sum(1))


# ---- spectral tools on the circle
def dlog_lowpass(loga, nmax=None):
    F = np.fft.rfft(loga)
    n = np.arange(len(F))
    if nmax is not None:
        F = np.where(n <= nmax, F, 0)
    return np.fft.irfft(1j * n * F, len(loga))


def harmonics(loga, upto=12):
    F = np.fft.rfft(loga) / len(loga) * 2
    return np.abs(F[:upto + 1])


def interp_circ(thetas, vals):
    return lambda x: np.interp(np.mod(x + np.pi, 2 * np.pi) - np.pi, thetas, vals, period=2 * np.pi)


def fake_div_2frame(thetas, dloga, TH, OM, dt=DT):
    """F . grad log nu, nu = dt a(th) a(th - om dt), F = (om, -sin th). Same formula as the script."""
    I = interp_circ(thetas, dloga)
    return OM * (I(TH) + I(TH - OM * dt)) + (-np.sin(TH)) * (-dt) * I(TH - OM * dt)
