"""Cell 4 pre-prereg check (exact maps only, no learning): how large is the topological obstruction of a
Euclidean latent with l = m = 1 on an angle-only S^1 world, as a function of smoothness (Lipschitz) and horizon K?

Worlds (state = angle only, closed manifold S^1, m = 1):
  rot    rigid rotation theta -> theta + w,                         w = 0.5 rad per step
  adler  time-dt map of theta' = nu - a sin(theta) (no fixed point), nu = 1, a = 0.8, dt = 0.5; invariant density
         rho(theta) = sqrt(nu^2 - a^2) / (2 pi (nu - a sin theta)), so rho_max / rho_min = (nu + a) / (nu - a) = 9.
Hand-built continuous models (encoder E, latent map G, decoder D), error = chord |D(G^K(E(theta0))) - e^{i Phi^K(theta0)}|
over theta0 drawn from the invariant density (= what a long data set covers):
  S1     latent S^1, E = id, G = Phi, D = e^{i.}: exact (error 0, every Lipschitz constant ~1).            [not computed]
  R2     latent R^2 (l = m + 1), E = e^{i.}, G = |x| e^{i Phi(arg x)}, D = id: exact on the data circle.  [not computed]
  cover  latent R (l = m), the Kvalheim-Sontag 2025 Thm 5 construction: E = cut chart with a linear ramp of width
         delta = 2 pi / L back across the cut (Lip(E) ~ L), G = the lift of Phi to R (Lip ~ max Phi'), D = e^{i.} on all
         of R (a covering map, Lip 1). The tear lives only in E, so errors come only from theta0 in the ramp.
  chart  latent R (l = m), what a generic decoder trained on a bounded latent range forces: E as above, D = e^{i.}
         used only on the chart [-pi, pi], so G must map the chart into itself and needs its own ramp of width delta
         (Lip(G) ~ L) on the preimage of the cut. Every revolution the trajectory crosses that ramp once.
Theory to check: sup error >= 1 for every L and K (degree argument: D o G^K o E factors through R, so it is
null-homotopic, while the truth has degree 1); mean error >= 1/(4 pi Lambda) (2 pi rho_min) with Lambda the
Lipschitz constant of the error map; cover: mean ~ delta * rho(cut) * c, flat in K; chart: mean grows ~ K * delta * rho(cut)
until it saturates. Moving the cut from the density maximum to the minimum divides the mean error by rho_max/rho_min.
"""
import numpy as np

TWO_PI = 2 * np.pi


def adler_exact(nu, a, dt):
    """Closed form: tan(theta/2) = a/nu + (Om/nu) tan(psi/2), Om = sqrt(nu^2 - a^2), and psi rotates at rate Om.
    Lifted to R branch by branch, so phi(th + 2 pi) = phi(th) + 2 pi."""
    Om = np.sqrt(nu ** 2 - a ** 2)

    def lift(f, x):
        n = np.floor((x + np.pi) / TWO_PI); r = x - TWO_PI * n
        return TWO_PI * n + f(r)
    h = lambda r: 2 * np.arctan2(nu * np.sin(r / 2) - a * np.cos(r / 2), Om * np.cos(r / 2))
    hinv = lambda r: 2 * np.arctan2(a * np.cos(r / 2) + Om * np.sin(r / 2), nu * np.cos(r / 2))
    return lambda th: lift(hinv, lift(h, np.asarray(th, dtype=np.float64)) + Om * dt)


def adler_flow(nu, a, dt, nsub=400):
    h = dt / nsub
    f = lambda th: nu - a * np.sin(th)

    def phi(th):                                    # lift: acts on R, phi(th + 2 pi) = phi(th) + 2 pi
        th = np.array(th, dtype=np.float64)
        for _ in range(nsub):
            k1 = f(th); k2 = f(th + 0.5 * h * k1); k3 = f(th + 0.5 * h * k2); k4 = f(th + h * k3)
            th = th + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return th
    return phi


def invert_monotone(phi, y, lo, hi, it=80):
    """phi increasing on [lo, hi]: x with phi(x) = y by bisection (vectorised)."""
    lo = np.full_like(y, lo, dtype=np.float64); hi = np.full_like(y, hi, dtype=np.float64)
    for _ in range(it):
        mid = 0.5 * (lo + hi); up = phi(mid) > y
        hi = np.where(up, mid, hi); lo = np.where(up, lo, mid)
    return 0.5 * (lo + hi)


def make_models(phi, cut, delta):
    """Chart coordinate z in [-pi, pi]: theta = cut + pi + z (mod 2 pi), the cut sits at z = +-pi (theta = cut)."""
    def E(th):                                      # continuous S^1 -> [-pi, pi]: identity, then a ramp of width delta
        z = np.mod(th - cut - np.pi + np.pi, TWO_PI) - np.pi      # in [-pi, pi)
        r = z > np.pi - delta
        return np.where(r, (np.pi - delta) - (z - (np.pi - delta)) * (TWO_PI - delta) / delta, z)

    D = lambda z: np.exp(1j * (z + cut + np.pi))
    psi = lambda z: phi(z + cut + np.pi) - (cut + np.pi)          # Phi in chart coordinates, on R

    zc = float(invert_monotone(psi, np.array([np.pi]), -np.pi, np.pi)[0])   # psi(zc) = pi: steps from z > zc cross the cut

    def G_chart(z):                                 # chart-preserving latent map with a ramp of width delta before zc
        lo = zc - delta
        y_lo, y_hi = psi(np.array(lo)), psi(np.array(zc)) - TWO_PI         # ~ pi - small,  -pi
        out = np.where(z <= lo, psi(z), psi(z) - TWO_PI)
        ramp = (z > lo) & (z < zc)
        return np.where(ramp, y_lo + (z - lo) * (y_hi - y_lo) / delta, out)
    return E, D, psi, G_chart


def errors(phi, cut, delta, theta0, Ks):
    E, D, psi, G_chart = make_models(phi, cut, delta)
    truth = theta0.copy(); zc = E(theta0); zh = E(theta0)
    res, k = {}, 0
    for K in range(max(Ks) + 1):
        if K in Ks:
            res[K] = dict(cover=np.abs(D(zc) - np.exp(1j * truth)), chart=np.abs(D(zh) - np.exp(1j * truth)))
        truth = phi(truth); zc = psi(zc); zh = G_chart(zh)
    return res


def sample_invariant(rho, n, rng):
    g = np.linspace(-np.pi, np.pi, 200001); c = np.cumsum(rho(g)); c /= c[-1]
    return np.interp(rng.random(n), c, g)


def main():
    rng = np.random.default_rng(0)
    n = 200_000
    Ks = [0, 1, 10, 50, 200, 1000]
    worlds = {}
    w = 0.5
    worlds["rot"] = (lambda th: np.asarray(th, dtype=np.float64) + w, lambda th: np.full_like(th, 1 / TWO_PI), 1.0)
    nu, a, dt = 1.0, 0.8, 0.5
    phiA = adler_exact(nu, a, dt)
    rhoA = lambda th: np.sqrt(nu ** 2 - a ** 2) / (TWO_PI * (nu - a * np.sin(th)))
    # Lipschitz of the Adler step map on S^1 (for the lower bound)
    g = np.linspace(-np.pi, np.pi, 20001); lipA = np.max(np.diff(phiA(g)) / np.diff(g))
    worlds["adler"] = (phiA, rhoA, lipA)
    # self-check of the Adler map: step doubling
    t0 = np.linspace(-np.pi, np.pi, 1001)
    t1 = np.linspace(-3 * np.pi, 3 * np.pi, 3001)
    print(f"adler map check: |closed form - RK4(400 substeps)| max = {np.max(np.abs(phiA(t1) - adler_flow(nu, a, dt)(t1))):.1e}; "
          f"Lip(Phi) = {lipA:.3f}; rho_max/rho_min = {(nu + a) / (nu - a):.1f}")
    print("mean / max chord error over theta0 ~ invariant density (n = 200000); L = Lip of the ramp ~ 2 pi / delta\n")
    print(f"{'world':6} {'cut at':9} {'L':>5} {'model':6} " + " ".join(f"{'K=' + str(K):>14}" for K in Ks)
          + f"  {'lower bnd K=0':>13}")
    for name, (phi, rho, lip) in worlds.items():
        cuts = [("any", 0.0)] if name == "rot" else [("rho max", np.pi / 2), ("rho min", -np.pi / 2)]
        # adler: nu - a sin(theta) is slowest at theta = pi/2 (density max) and fastest at -pi/2 (density min)
        for cname, cut in cuts:
            theta0 = sample_invariant(rho, n, rng)
            rho_min = float(np.min(rho(np.linspace(-np.pi, np.pi, 1001))))
            for L in (10, 100, 1000):
                delta = TWO_PI / L
                res = errors(phi, cut, delta, theta0, Ks)
                lam0 = L + 1.0                     # Lip(D o E) + Lip(id) at K = 0
                bound = 1 / (4 * np.pi * lam0) * TWO_PI * rho_min
                for model in ("cover", "chart"):
                    cells = " ".join(f"{res[K][model].mean():7.4f}/{res[K][model].max():5.2f}" for K in Ks)
                    print(f"{name:6} {cname:9} {L:5d} {model:6} {cells}  {bound:13.5f}")
    print("\nPredicted chart-model hit fraction by step K (K >= 1): ~ K * delta * rho(cut) (one ramp crossing per revolution);")
    for name, (phi, rho, lip) in worlds.items():
        cuts = [("any", 0.0)] if name == "rot" else [("rho max", np.pi / 2), ("rho min", -np.pi / 2)]
        for cname, cut in cuts:
            theta0 = sample_invariant(rho, n, rng)
            for L in (100,):
                delta = TWO_PI / L
                res = errors(phi, cut, delta, theta0, [1, 10, 50])
                for K in (1, 10, 50):
                    hit = (res[K]["chart"] > 1e-6).mean()
                    hitc = (res[K]["cover"] > 1e-6).mean()
                    print(f"  {name:6} {cname:8} L={L} K={K:3d}: chart hit {hit:.4f}  (K+1)*delta*rho(cut) = "
                          f"{(K + 1) * delta * float(rho(np.array(cut))):.4f}   cover hit {hitc:.4f}  delta*rho(cut) = "
                          f"{delta * float(rho(np.array(cut))):.4f}")


if __name__ == "__main__":
    main()
