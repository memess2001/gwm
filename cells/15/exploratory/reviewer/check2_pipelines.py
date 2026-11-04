# Robustness of the low-frequency fake divergence to: bilinear interpolation, pixel aliasing (supersampling),
# and cropping (bob leaving the 64x64 frame), plus the harmonic cut-off.
import numpy as np
from common import *

thetas = np.linspace(-np.pi, np.pi, 721, endpoint=False)
omegas = np.linspace(-2.5, 2.5, 51)
TH, OM = np.meshgrid(thetas, omegas, indexing="ij")

# where does the bob centre land? forward homography yo = ys/(1+k ys), xo = xs/(1+k ys) (normalized units)
for k in (0.3, 0.6):
    ys, xs = (R0 / 32) * np.cos(thetas), (R0 / 32) * np.sin(thetas)
    yo, xo = ys / (1 + k * ys), xs / (1 + k * ys)
    outside = (np.abs(yo) > 1) | (np.abs(xo) > 1)
    edge = 1 - np.maximum(np.abs(yo), np.abs(xo))          # distance of bob centre to frame edge (normalized)
    print(f"k={k}: bob centre at theta=pi -> yo={yo[0]:+.3f} (frame edge at -1); fraction of circle with bob centre "
          f"outside frame = {outside.mean():.3f}; min margin = {edge.min()*32:+.1f} px")

def report(name, a, extra=""):
    loga = np.log(a)
    out = [f"{name:<44s} a max/min={a.max()/a.min():.2f}"]
    for nm in (4, 8, 16, None):
        fd = fake_div_2frame(thetas, dlog_lowpass(loga, nm), TH, OM)
        out.append(f"<= {str(nm):>4s}: RMS {np.sqrt((fd**2).mean()):.3f} max {np.abs(fd).max():.2f}")
    print("  " + " | ".join(out) + extra)

for k in (0.0, 0.3, 0.6):
    print(f"--- k = {k}")
    report("script (zoo f32 + bilinear warp, 64px)", a_script(thetas, k))
    report("analytic warp, point-sampled, 64px", a_analytic(thetas, k))
    report("analytic warp, 4x supersampled (256px)", a_analytic(thetas, k, ss=4))
    report("analytic, no crop (128px canvas, same pitch)", a_analytic(thetas, k, size=128, half=2.0))
    report("analytic, no crop, 4x supersampled", a_analytic(thetas, k, size=128, half=2.0, ss=4, chunk=16))
