"""Cell 15, ground-truth check for the preregistration draft (no model is trained).

Can a cheap observation reproduce the camera artefact? Instead of 64x64 frames, observe the bob's image position
through the same tilted (projective) camera as the cell 15 review (cells/15/exploratory/reviewer/common.py:
forward homography yo = ys / (1 + k ys), xo = xs / (1 + k ys) in normalised units, bob at (R0/32)(sin th, cos th)).
For a two-frame window the pulled-back volume is nu = dt a(th) a(th - om dt) with a(th) = |d obs / d th|, and the
fake divergence is F . grad log nu (same formula as the review). Compared with the pixel version (a = |d frame/d th|,
the review's a_analytic) at k = 0, 0.3, 0.6, keeping harmonics <= 4 of log a as the review did. Also the affine
camera (vertical foreshortening by cos phi), which the review found to give no fake term for pixels.
Run from the repository root with PYTHONPATH=. (it imports the review's common.py); about a minute.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "reviewer"))
from common import R0, a_analytic, dlog_lowpass, fake_div_2frame      # noqa: E402

thetas = np.linspace(-np.pi, np.pi, 721, endpoint=False)
omegas = np.linspace(-2.5, 2.5, 51)
TH, OM = np.meshgrid(thetas, omegas, indexing="ij")
rms = lambda x: float(np.sqrt((x ** 2).mean()))


def a_point(k, kind="proj", phi=0.0, h=1e-6):
    def obs(th):
        ys, xs = (R0 / 32) * np.cos(th), (R0 / 32) * np.sin(th)
        if kind == "proj":
            return np.stack([xs / (1 + k * ys), ys / (1 + k * ys)], -1)
        return np.stack([xs, ys * np.cos(phi)], -1)                 # affine: vertical foreshortening
    d = (obs(thetas + h) - obs(thetas - h)) / (2 * h)
    return np.sqrt((d ** 2).sum(-1))


def fake(a, nmax=4):
    return rms(fake_div_2frame(thetas, dlog_lowpass(np.log(a), nmax), TH, OM))


print("fake divergence RMS (harmonics <= 4 of log a), undamped pendulum; true divergence 0, signal 2 gamma = 0.1")
for k in (0.0, 0.3, 0.6):
    ap, ax = a_point(k), a_analytic(thetas, k)
    print(f"   projective k = {k}: point observation {fake(ap):.3f} (a max/min {ap.max() / ap.min():.2f})"
          f" | pixels {fake(ax):.3f} (a max/min {ax.max() / ax.min():.2f})"
          f" | correlation of log a: {np.corrcoef(np.log(ap), np.log(ax))[0, 1] if k else float('nan'):.3f}")
for phi in (0.5, 1.0):
    ap = a_point(0.0, "affine", phi)
    print(f"   affine, foreshortening cos({phi}) = {np.cos(phi):.2f}: point observation {fake(ap):.3f} (a max/min {ap.max() / ap.min():.2f})")
