# Is the "Klein level of the camera" the right classifier for fake divergence?
# a_out(th)^2 = int |d_th I_src(s; th)|^2 |det DH(s)| ds  (change of variables, continuum / fine sampling).
import numpy as np
from scipy.ndimage import map_coordinates
from common import *

thetas = np.linspace(-np.pi, np.pi, 721, endpoint=False)
omegas = np.linspace(-2.5, 2.5, 51)
TH, OM = np.meshgrid(thetas, omegas, indexing="ij")

def grid(size=64, ss=4, half=1.0):
    n = size * ss
    g = (np.arange(n) + 0.5) / n * 2 * half - half
    yc, xc = np.meshgrid(g, g, indexing="ij")
    return yc.ravel(), xc.ravel()

def a_from_inverse_map(inv, bg=None, ss=4, h=1e-5, chunk=32):
    """inv: (yc, xc) -> (ys, xs) normalized plane coords seen by each output sample. bg: optional background
    intensity per output sample (soft union with the pendulum, like the zoo renderer)."""
    yc, xc = grid(ss=ss)
    ys, xs = inv(yc, xc)
    dx, dy = xs * 32.0, ys * 32.0
    out = np.empty(len(thetas))
    for i in range(0, len(thetas), chunk):
        th = thetas[i:i + chunk]
        Ip, Im = scene(dx, dy, th + h), scene(dx, dy, th - h)
        if bg is not None:
            Ip, Im = 1 - (1 - Ip) * (1 - bg), 1 - (1 - Im) * (1 - bg)
        out[i:i + chunk] = np.sqrt((((Ip - Im) / (2 * h)) ** 2).sum(1))
    return out

def rep(name, a):
    fd_low = fake_div_2frame(thetas, dlog_lowpass(np.log(a), 4), TH, OM)
    fd_all = fake_div_2frame(thetas, dlog_lowpass(np.log(a)), TH, OM)
    print(f"  {name:<62s} a max/min {a.max()/a.min():.3f} | fake RMS (<=4) {np.sqrt((fd_low**2).mean()):.3f} "
          f"| fake RMS (all) {np.sqrt((fd_all**2).mean()):.3f}")

print("supersampled 4x, 64px field of view:")
rng = np.random.default_rng(1)
A = np.array([[0.8, 0.35], [-0.15, 1.1]])            # generic affine (shear + anisotropic scale), det != 1
Ainv = np.linalg.inv(A)
rep("generic affine [[.8,.35],[-.15,1.1]] (Klein: affine)", a_from_inverse_map(lambda y, x: (Ainv[0,0]*y + Ainv[0,1]*x, Ainv[1,0]*y + Ainv[1,1]*x)))
rep("affine + translation (pivot off-centre by (0.1,-0.08))", a_from_inverse_map(lambda y, x: (Ainv[0,0]*(y-.1) + Ainv[0,1]*(x+.08), Ainv[1,0]*(y-.1) + Ainv[1,1]*(x+.08))))
rep("projective k=0.3 (script homography)", a_from_inverse_map(lambda y, x: (y/(1-0.3*y), x/(1-0.3*y))))
# radial (barrel) distortion centred on the pivot: Homeo-level, NOT projective, yet radially symmetric area Jacobian
rd = lambda y, x, c=0.35: (y * (1 + c*(x*x+y*y)), x * (1 + c*(x*x+y*y)))
rep("radial lens distortion centred on pivot (Homeo, non-projective)", a_from_inverse_map(rd))
# same distortion centred off the pivot -> area Jacobian not symmetric about the pivot
rep("radial lens distortion centred 0.3 below pivot", a_from_inverse_map(lambda y, x: (lambda yy, xx: (yy+0.3, xx))(*rd(y-0.3, x))))
# Euclidean (identity) camera but non-uniform background: vertical brightness gradient 0 (top) -> 0.4 (bottom)
yc, xc = grid()
rep("identity camera + background gradient 0->0.4 (Klein: Euclidean)", a_from_inverse_map(lambda y, x: (y, x), bg=0.2*(1+yc)))
rep("identity camera + uniform background 0.3", a_from_inverse_map(lambda y, x: (y, x), bg=0.3*np.ones_like(yc)))

# script-style pipeline for an affine camera: zoo f32 render + bilinear resampling, 64px, low-pass part
from zoo.pendulum import render
def a_bilinear_affine(Ainv, h=1e-3):
    r, c = np.mgrid[0:64, 0:64].astype(np.float64)
    yc, xc = (r - 31.5) / 32.0, (c - 31.5) / 32.0
    ys, xs = Ainv[0,0]*yc + Ainv[0,1]*xc, Ainv[1,0]*yc + Ainv[1,1]*xc
    rs, cs = ys*32 + 31.5, xs*32 + 31.5
    warp = lambda f: np.stack([map_coordinates(im, [rs, cs], order=1, mode="constant") for im in f])
    d = (warp(render(thetas + h, quantize=False).astype(np.float64)) - warp(render(thetas - h, quantize=False).astype(np.float64))) / (2*h)
    return np.sqrt((d.reshape(len(thetas), -1)**2).sum(1))
print("script-style pipeline (zoo float32 render + bilinear warp, 64px):")
rep("generic affine, bilinear (as the script would implement it)", a_bilinear_affine(Ainv))
