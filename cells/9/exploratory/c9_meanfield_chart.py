"""Cell 9 review: sigma_t can vary along a DETERMINISTIC trajectory purely as a chart artefact of the diagonal
(mean-field) posterior.

Latent z in R^2 embeds the pendulum angle as a circle z = R (cos th, sin th).  Suppose the decoder is sensitive
along the tangent t(th) with precision lam_t (uniform in th: rotation-equivariant renderer, the zoo default)
and only weakly along the radial direction with precision lam_r (decoder ~ reads the angle only).
Exact Gaussian posterior precision: Lam = lam_t t t^T + lam_r rhat rhat^T  (same for every th up to rotation,
so the TRUE Fisher speed along the orbit, |dz/dt|^2 lam_t, is constant at constant angular speed).
Mean-field optimum (reverse KL, diagonal q): sigma_i^2 = 1/Lam_ii, with a floor sigma_i >= s_min.
"""
import numpy as np

th = np.linspace(0, 2 * np.pi, 2001)
R = 1.5
for lam_t, lam_r, s_min in ((1e4, 4.0, 0.1), (1e4, 4.0, 0.0), (400.0, 4.0, 0.1), (1e4, 1e3, 0.1)):
    t = np.stack([-np.sin(th), np.cos(th)], 1); r = np.stack([np.cos(th), np.sin(th)], 1)
    Lii = lam_t * t**2 + lam_r * r**2                       # diagonal of the precision, [n, 2]
    sig = np.maximum(1 / np.sqrt(Lii), s_min)
    dz = R * t                                               # dz/dth (unit angular speed)
    dsig = np.gradient(sig, th, axis=0)
    v_mu = np.sqrt((dz**2 / sig**2).sum(1))                  # mean part of the diagonal-Fisher speed
    v_full = np.sqrt((dz**2 / sig**2 + 2 * dsig**2 / sig**2).sum(1))
    v_true = np.sqrt(lam_t) * R * np.ones_like(th)           # full-covariance Fisher speed of the mean
    cv_sig = (sig.std(0) / sig.mean(0)).mean()
    print(f"lam_t={lam_t:g} lam_r={lam_r:g} floor={s_min}: sigma_1 range [{sig[:,0].min():.3f}, {sig[:,0].max():.3f}], "
          f"temporal CV(sigma) = {cv_sig:.2f}; diag-Fisher speed max/min = {v_full.max()/v_full.min():.2f} "
          f"(mu-part only {v_mu.max()/v_mu.min():.2f}); share of speed^2 from 2dsig^2 term = "
          f"{(2*dsig**2/sig**2).sum()/ (dz**2/sig**2 + 2*dsig**2/sig**2).sum():.3f}; whitened-Euclidean max/min = "
          f"{(np.sqrt((dz**2/sig.mean(0)**2).sum(1))).max()/(np.sqrt((dz**2/sig.mean(0)**2).sum(1))).min():.2f}")
