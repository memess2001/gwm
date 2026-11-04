import numpy as np
from common import *
thetas = np.linspace(-np.pi, np.pi, 721, endpoint=False)
omegas = np.linspace(-2.5, 2.5, 51)
TH, OM = np.meshgrid(thetas, omegas, indexing="ij")
np.set_printoptions(precision=4, suppress=True, linewidth=200)
for k in (0.3, 0.6):
    s = harmonics(np.log(a_script(thetas, k)), 16)
    c = harmonics(np.log(a_analytic(thetas, k, ss=4)), 16)
    print(f"k={k} |harmonic n| of log a, n=1..16\n  script  {s[1:]}\n  clean   {c[1:]}")
    # the analytic leading-order prediction for the clean camera geometry: fake ~ 2 om (log a)'(th)
    loga = np.log(a_analytic(thetas, k, ss=4))
    d = dlog_lowpass(loga)
    lead = 2 * OM * interp_circ(thetas, d)(TH)
    full = fake_div_2frame(thetas, d, TH, OM)
    print(f"  clean: |(log a)'|max={np.abs(d).max():.3f}; leading-order 2 om (log a)' RMS={np.sqrt((lead**2).mean()):.3f} vs exact-formula RMS={np.sqrt((full**2).mean()):.3f}")

# small-k scaling and the affine (foreshortening) camera
print("--- small-k scaling (clean, 4x supersampled, 64px canvas)")
for k in (0.01, 0.03, 0.1, 0.2):
    fd = fake_div_2frame(thetas, dlog_lowpass(np.log(a_analytic(thetas, k, ss=4))), TH, OM)
    print(f"  k={k:<5}: fake RMS {np.sqrt((fd**2).mean()):.4f}  max {np.abs(fd).max():.3f}   (RMS/k = {np.sqrt((fd**2).mean())/k:.2f})")
print("--- affine camera: vertical foreshortening ys = yc/cos(phi) (pure Klein 'affine' level), clean")
for deg in (10, 20, 30, 45):
    a = a_analytic(thetas, 0, ss=4, kind="affine", phi=np.deg2rad(deg), size=64, half=1.0)
    fd = fake_div_2frame(thetas, dlog_lowpass(np.log(a)), TH, OM)
    print(f"  tilt {deg:>2d} deg (scale cos={np.cos(np.deg2rad(deg)):.3f}): a max/min {a.max()/a.min():.3f}; fake RMS {np.sqrt((fd**2).mean()):.3f} max {np.abs(fd).max():.2f}")
