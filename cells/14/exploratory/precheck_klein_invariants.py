"""Cell 14, check before preregistration (ground truth only; nothing is learned).

The pendulum scene seen through four classes of camera (the cell's camera_ladder): similarity (Euclidean up to
scale), affine, projective, and a smooth non-projective warp. The image of the bob's circle and of the pivot decide
the class with three classical invariants, no world model:
  1. conic test: the bob's image points lie on a conic iff the camera is projective (or a sub-class). Statistic: RMS
     Sampson distance to the fitted conic divided by the (known) noise sd, about 1 for a conic;
  2. centre test: the image of the circle's centre (the pivot) is the conic's centre iff the camera is affine (the
     conic centre is the pole of the line at infinity, and a projective map moves that line unless it is affine);
  3. axis ratio: the image conic is a circle iff the camera is a similarity.
Thresholds: the 99.5th percentile of each statistic under its null classes on a first draw of cameras (false-alarm
rate about 0.5 %), then accuracy on a fresh draw. Errors concentrate where a class is close to the one below it
(weak perspective, weak warp); there the classes are not distinguishable at this noise by any method.
Noise: Gaussian, sd 0.01 per coordinate in units of the bob radius (about 0.2 pixel for the zoo renderer, radius
22 px); 300 bob positions per camera (a few hundred frames). A first version used the singular-value ratio of the
conic design matrix, 60 points and midpoint thresholds; it could not separate the warp class (accuracy 0.86).
Runs in under a minute.
"""
import numpy as np

rng = np.random.default_rng(0)
N_PTS, NOISE, N_CAM = 300, 0.01, 300
LEVELS = ("similarity", "affine", "projective", "warp")


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def camera(level):
    s = np.exp(rng.uniform(np.log(0.6), np.log(1.6)))
    t = rng.uniform(-0.5, 0.5, 2)
    A = s * rot(rng.uniform(0, 2 * np.pi))
    if level != "similarity":
        r = np.exp(rng.uniform(np.log(1.3), np.log(2.5)))                  # axis ratio of the linear part
        A = A @ rot(rng.uniform(0, np.pi)) @ np.diag([1.0, 1.0 / r]) @ rot(rng.uniform(0, np.pi))
    vn = rng.uniform(0.1, 0.5) if level in ("projective", "warp") else 0.0  # perspective strength
    a = rng.uniform(0, 2 * np.pi)
    v = vn * np.array([np.cos(a), np.sin(a)])
    eps = rng.uniform(0.01, 0.08) if level == "warp" else 0.0              # warp amplitude
    k, ph = rng.uniform(1.5, 3.0), rng.uniform(0, 2 * np.pi)

    def H(p):
        p = np.atleast_2d(p)
        q = (p @ A.T + t) / (1.0 + p @ v)[:, None]
        if eps:
            q = q + eps * np.stack([np.sin(k * q[:, 1] + ph), np.sin(k * q[:, 0] - ph)], 1)
        return q
    return H, dict(v=vn, eps=eps)


def stats(H):
    th = rng.uniform(-np.pi, np.pi, N_PTS)
    bob = H(np.stack([np.sin(th), np.cos(th)], 1)) + NOISE * rng.standard_normal((N_PTS, 2))
    piv = H(np.zeros(2))[0] + NOISE * rng.standard_normal(2)
    x, y = bob[:, 0], bob[:, 1]
    D = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], 1)
    sc = np.linalg.norm(D, axis=0)
    c = np.linalg.svd(D / sc, full_matrices=False)[2][-1] / sc
    a, b, cc, d, e, f = c
    F = D @ c
    gx, gy = 2 * a * x + b * y + d, b * x + 2 * cc * y + e
    resid = np.sqrt(np.mean(F ** 2 / (gx ** 2 + gy ** 2))) / NOISE         # RMS Sampson distance / noise sd
    M = np.array([[a, b / 2], [b / 2, cc]])
    centre = np.linalg.solve(M, -0.5 * np.array([d, e]))
    w = np.linalg.eigvalsh(M)
    ratio = np.sqrt(abs(w).max() / abs(w).min())
    size = np.sqrt(((bob - bob.mean(0)) ** 2).sum(1).mean())
    return np.array([resid, np.linalg.norm(centre - piv) / size, ratio])


def draw():
    out = {}
    for L in LEVELS:
        rows = []
        for _ in range(N_CAM):
            H, par = camera(L)
            rows.append((stats(H), par))
        out[L] = rows
    return out


def main():
    first = draw()
    S = {L: np.array([s for s, _ in first[L]]) for L in LEVELS}
    print(f"{N_CAM} random cameras per level, {N_PTS} bob positions, noise sd {NOISE} (bob radius = 1); first draw:")
    print(f"{'level':11} {'Sampson RMS / noise':>22} {'centre offset / size':>24} {'axis ratio':>18}")
    for L in LEVELS:
        r, o, q = S[L].T
        print(f"{L:11} {r.min():9.3f} - {r.max():7.3f}   {o.min():10.4f} - {o.max():7.4f}   {q.min():8.3f} - {q.max():7.3f}")
    t_res = np.quantile(np.concatenate([S[L][:, 0] for L in LEVELS[:3]]), 0.995)
    t_off = np.quantile(np.concatenate([S[L][:, 1] for L in LEVELS[:2]]), 0.995)
    t_rat = np.quantile(S["similarity"][:, 2], 0.995)
    print(f"thresholds = 99.5th percentile under the null classes: residual {t_res:.3f}, offset {t_off:.4f}, ratio {t_rat:.3f}")

    def classify(s):
        if s[0] > t_res:
            return "warp"
        if s[1] > t_off:
            return "projective"
        return "affine" if s[2] > t_rat else "similarity"
    fresh = draw()
    pred = {L: [classify(s) for s, _ in fresh[L]] for L in LEVELS}
    acc = np.mean([p == L for L in LEVELS for p in pred[L]])
    print(f"fresh draw: accuracy {acc:.3f}")
    for L in LEVELS:
        print(f"   {L:11} classified as: " + ", ".join(f"{m} {pred[L].count(m)}" for m in LEVELS if pred[L].count(m)))
    print("where the errors are:")
    v = np.array([p["v"] for _, p in fresh["projective"]])
    okp = np.array([q == "projective" for q in pred["projective"]])
    for lo, hi in ((0.1, 0.2), (0.2, 0.3), (0.3, 0.5)):
        m = (v >= lo) & (v < hi)
        print(f"   projective, perspective strength {lo:.1f}-{hi:.1f}: {okp[m].mean():.2f} correct ({m.sum()} cameras)")
    e = np.array([p["eps"] for _, p in fresh["warp"]])
    okw = np.array([q == "warp" for q in pred["warp"]])
    for lo, hi in ((0.01, 0.02), (0.02, 0.04), (0.04, 0.08)):
        m = (e >= lo) & (e < hi)
        print(f"   warp, amplitude {lo:.2f}-{hi:.2f} (noise sd {NOISE}): {okw[m].mean():.2f} detected ({m.sum()} cameras)")


if __name__ == "__main__":
    main()
