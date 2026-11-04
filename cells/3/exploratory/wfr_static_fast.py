"""Finite-time WFR (static LET formulation, Chizat delta convention), stabilised unbalanced Sinkhorn.
WFR_delta^2 = min <C,g> + tau KL(g1|mu) + tau KL(g2|nu),  C = 4d^2 * (-log cos^2(min(|x-y|/2d, pi/2))), tau = 4 d^2.
(eps-entropic, eps -> 0 by annealing + Richardson.)  Transport share via envelope identity:
  share(delta) = 1 - d ln WFR^2 / d ln delta^2.
"""
import numpy as np, time

x = np.linspace(-6, 6, 241); w = 1.0

def gauss(mu=0.0, s=w, m=1.0):
    g = np.exp(-(x - mu) ** 2 / (2 * s**2)); return m * g / g.sum()

def kl(p, q):
    m = p > 1e-300
    return np.sum(p[m] * np.log(p[m] / q[m])) - p.sum() + q.sum()

def cost(delta):
    tau = 4 * delta**2
    r = np.abs(x[:, None] - x[None, :]) / (2 * delta)
    with np.errstate(divide="ignore"):
        return np.where(r < np.pi / 2, -tau * np.log(np.cos(np.minimum(r, np.pi / 2)) ** 2), np.inf), tau

def uot(mu, nu, delta, eps, F=None, G=None, n_iter=30000, tol=1e-10):
    C, tau = cost(delta)
    lam = tau / (tau + eps)
    lmu, lnu = np.log(mu), np.log(nu)
    F = np.zeros_like(x) if F is None else F.copy()
    G = np.zeros_like(x) if G is None else G.copy()
    def kern():
        return np.exp((F[:, None] + G[None, :] - C) / eps + lmu[:, None] + lnu[None, :])
    Kt = kern(); u = np.ones_like(x); v = np.ones_like(x)
    for it in range(n_iter):
        u_new = np.exp((lam - 1) * F / eps) * (mu / (Kt @ v)) ** lam
        v = np.exp((lam - 1) * G / eps) * (nu / (Kt.T @ u_new)) ** lam
        conv = it % 50 == 0 and np.max(np.abs(np.log(u_new) - np.log(u))) < tol
        u = u_new
        if np.max(np.abs(np.log(u))) > 30 or np.max(np.abs(np.log(v))) > 30 or conv:
            F += eps * np.log(u); G += eps * np.log(v)
            Kt = kern(); u = np.ones_like(x); v = np.ones_like(x)
        if conv:
            break
    gam = Kt * u[:, None] * v[None, :]
    a, b = gam.sum(1), gam.sum(0)
    P = np.sum(np.where(np.isfinite(C), C, 0) * gam) + tau * kl(a, mu) + tau * kl(b, nu)
    return P, F, G, it

def wfr2(mu, nu, delta, epss=(0.02, 0.008, 0.004, 0.002)):
    F = G = None; vals = []
    for e in epss:
        P, F, G, it = uot(mu, nu, delta, e, F, G); vals.append(P)
    return 2 * vals[-1] - vals[-2], vals, it

if __name__ == "__main__":
    t0 = time.time()
    deltas = np.exp(np.linspace(np.log(0.25), np.log(4.0), 7))
    mu0 = gauss()
    cases = {"translate d=0.5": gauss(mu=0.5), "grow x1.21": gauss(m=1.21), "translate d=3.0": gauss(mu=3.0)}
    for name, nu in cases.items():
        W = np.array([wfr2(mu0, nu, d)[0] for d in deltas])
        share = 1 - np.gradient(np.log(W), np.log(deltas**2))
        print(f"\n{name}   ({time.time()-t0:.0f}s)", flush=True)
        for d, Wd, s in zip(deltas, W, share):
            if name.startswith("translate d=0.5"):
                ex = f"   small-d prediction: WFR^2={0.25*d**2/(d**2+1):.4f}, share={d**2/(d**2+1):.3f}"
            elif name.startswith("grow"):
                ex = f"   exact: WFR^2={4*d**2*(np.sqrt(1.21)-1)**2:.4f}, share=0"
            else:
                ex = f"   pi*delta={np.pi*d:.2f} (displacement 3.0)"
            print(f"  delta={d:6.3f}  WFR^2={Wd:.4f}  share={s:6.3f}{ex}", flush=True)
