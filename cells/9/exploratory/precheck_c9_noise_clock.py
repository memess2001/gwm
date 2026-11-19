"""Cell 9 precheck (2026-11-16): ground truth only, no model is trained.

Question: in Wright-Fisher (WF) worlds, does the Fisher clock of the latent posterior differ from cheap
same-source clocks, and how do the clocks behave near fixed points? Three parts:

A. Noise identity, from zoo/wright_fisher.py samples (K = 3, finite pop_size N).
   One multinomial resampling has Cov(dx) = (diag x - x x') / N, the inverse of the Fisher (Shahshahani) metric
   G = diag(1/x) up to the factor N. So E[dx' G dx | x] = (K - 1)/N + drift'G drift, and the noise part does
   not depend on the state. The Euclidean noise part is H(x)/N, where H = 1 - sum x_i^2 is the heterozygosity.
   Check: per-step squared increments binned by H.

B. Exact Bayes filter (K = 2, grid on {0, 1/N, ..., 1}, binomial drift, binomial counts, selection + symmetric
   mutation). The posterior is exact, with no Gaussian or model assumption. The "Fisher clock" is the Gaussian line
   element of the posterior moments, ds^2 = (dmu^2 + 2 dsigma^2) / sigma^2, as for a diagonal-Gaussian latent.
   Compared with: wall; frozen sigma (|dmu| / mean sigma); Hellinger arc length of the posterior mean
   (2 |d sqrt(mu)| on the sphere = Fisher arc length of the mean); Hellinger and Euclidean arc length of the raw
   observed frequencies; surprise sqrt(2 KL(post || prior)).
   Two regimes: noise-dominated (N = 100) and drift-dominated (N = 2000).

C. Deterministic exact zoo paths (K = 3, no noise): rate of each arc-length clock near the fixed point.
   Slope of log(rate) against log(distance to the fixed point), and the fraction of the wall horizon during which the
   clock runs at < 1 % of its maximum rate. Evaluating a tau-trained model in wall time needs dt/dtau there.
"""
import sys
import numpy as np
from scipy.stats import binom

sys.path.insert(0, "/home/user/gwm_repo")
from zoo.wright_fisher import WrightFisher

np.set_printoptions(precision=4, suppress=True)

# ------------------------------------------------------------------------------------------------ A
print("A. per-step squared increment, binned by heterozygosity H (zoo WF, K=3, selection_only, dt=0.1)")
for N in (100, 1000):
    w = WrightFisher(K=3, schedule="selection_only", seg_steps=400, pop_size=N)
    tr = w.sample(seed=7, horizon=400, n_traj=400, obs_mode="freq")
    X, F = tr.state, np.asarray(tr.info["fitness"])
    x0, x1, f = X[:, :-1].reshape(-1, 3), X[:, 1:].reshape(-1, 3), F.reshape(-1, 3)
    alive = (x0 > 0).sum(1)
    keep = alive == 3                                   # identity (K-1)/N needs all K types present
    x0, x1, f = x0[keep], x1[keep], f[keep]
    d = x1 - x0
    fis = (d ** 2 / x0).sum(1)                          # metric at the start point of the step
    euc = (d ** 2).sum(1)
    H = 1 - (x0 ** 2).sum(1)
    phi = x0 * np.exp(0.1 * (f - f.max(1, keepdims=True))); phi /= phi.sum(1, keepdims=True)   # zoo selection flow
    nz = d - (phi - x0)                                 # the multinomial noise of the step
    fis_noise, euc_noise = (nz ** 2 / x0).sum(1), (nz ** 2).sum(1)
    bins = np.quantile(H, np.linspace(0, 1, 6))
    idx = np.clip(np.digitize(H, bins[1:-1]), 0, 4)
    print(f"  N={N:5d}  (K-1)/N = {2 / N:.5f}")
    print("   H-bin(mean)  E[Fisher^2] total / noise only   E[Euclid^2] noise only * N   n")
    for b in range(5):
        m = idx == b
        print(f"   {H[m].mean():10.3f}  {fis[m].mean():11.5f} / {fis_noise[m].mean():.5f}         {(euc_noise[m] * N).mean():8.3f}"
              f"   (H = {H[m].mean():.3f})  {m.sum():6d}")

# ------------------------------------------------------------------------------------------------ B
def run_filter(N, n_obs, s, u, dt=0.1, T=300, n_traj=200, seed=0, burn=20, seg=None):
    rng = np.random.default_rng(seed)
    grid = np.arange(N + 1) / N

    def flow(x, sgn):
        y = x * np.exp(sgn * s * dt) / (x * np.exp(sgn * s * dt) + 1 - x)   # exact selection flow (K = 2)
        return 0.5 + (y - 0.5) * np.exp(-2 * u * dt)                        # exact symmetric mutation flow

    sgn0 = rng.choice([-1.0, 1.0], n_traj)
    sg = np.ones(T + 1) if seg is None else np.where((np.arange(T + 1) // seg) % 2 == 0, 1.0, -1.0)
    SG = sgn0[:, None] * sg[None, :]                                        # sign in force from step t to t+1
    trans = {g: binom.pmf(np.arange(N + 1)[None, :], N, flow(grid, g)[:, None]) for g in (-1.0, 1.0)}
    x = rng.uniform(0.05, 0.95, n_traj)
    xs, cs = [x], [rng.binomial(n_obs, x)]
    for t in range(T):
        x = rng.binomial(N, flow(x, SG[:, t])) / N
        xs.append(x); cs.append(rng.binomial(n_obs, x))
    xs, cs = np.array(xs).T, np.array(cs).T                                 # [n, T+1]
    post = np.ones((n_traj, N + 1)) / (N + 1)
    mu, sd, kl = [], [], []
    for t in range(T + 1):
        if t > 0:
            prior = np.empty_like(post)
            for g in (-1.0, 1.0):                                           # the filter knows the dynamics
                m = SG[:, t - 1] == g
                prior[m] = post[m] @ trans[g]
        else:
            prior = post
        lik = binom.pmf(cs[:, t:t + 1], n_obs, grid[None, :])
        post = prior * lik
        post /= post.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl.append(np.where(post > 0, post * (np.log(post) - np.log(prior)), 0).sum(1))
        m1 = post @ grid
        mu.append(m1); sd.append(np.sqrt(np.maximum(post @ grid ** 2 - m1 ** 2, 1e-300)))
    mu, sd, kl = np.array(mu).T, np.array(sd).T, np.array(kl).T
    sl = slice(burn, None)
    mu, sd, kl, xs, cs = mu[:, sl], sd[:, sl], kl[:, sl], xs[:, sl], cs[:, sl]
    netdrift = flow(mu[:, :-1], SG[:, burn:-1]) - mu[:, :-1]                # net (selection + mutation) drift per step
    y = cs / n_obs
    dmu, dsd = np.diff(mu, axis=1), np.diff(sd, axis=1)
    smid = 0.5 * (sd[:, 1:] + sd[:, :-1])
    hell = lambda a: 2 * np.sqrt(np.diff(np.sqrt(a), axis=1) ** 2 + np.diff(np.sqrt(1 - a), axis=1) ** 2)
    inc = dict(
        wall=np.ones_like(dmu),
        fisher=np.sqrt(dmu ** 2 + 2 * dsd ** 2) / smid,
        frozen=np.abs(dmu) / sd.mean(),
        hell_post=hell(mu),
        hell_obs=hell(y),
        eucl_obs=np.abs(np.diff(y, axis=1)),
        surprise=np.sqrt(2 * kl[:, 1:]),
    )
    return dict(mu=mu, sd=sd, xs=xs, inc=inc, N=N, n_obs=n_obs, s=s, u=u, dt=dt, netdrift=netdrift, seg=seg)


def report(r):
    mu, sd, inc, N, s, dt = r["mu"], r["sd"], r["inc"], r["N"], r["s"], r["dt"]
    xm = 0.5 * (mu[:, 1:] + mu[:, :-1])
    c = sd ** 2 / (mu * (1 - mu))
    m0 = mu[:, :-1]
    drift2 = r["netdrift"] ** 2 / (m0 * (1 - m0))                           # per-step net drift^2, Shahshahani metric
    print(f"\n  N={N}, n_obs={r['n_obs']}, s={s}, u={r['u']}, sign {'constant' if r['seg'] is None else 'alternates every %d steps' % r['seg']}:"
          f"  noise (K-1)/N = {1 / N:.5f}, mean net drift^2 = {drift2.mean():.5f};"
          f" share of steps with drift^2 > noise: {(drift2 > 1 / N).mean():.2f}")
    print(f"   posterior var / (mu(1-mu)) = c: mean {c.mean():.2e}, CV across all steps {c.std() / c.mean():.3f}, "
          f"CV of per-step mean over the 5 mu-bins: ", end="")
    bins = np.quantile(xm, [0.2, 0.4, 0.6, 0.8])
    idx = np.digitize(xm, bins)
    cm = 0.5 * (c[:, 1:] + c[:, :-1])
    cb = np.array([cm[idx == b].mean() for b in range(5)])
    print(f"{cb.std() / cb.mean():.3f}")
    # rate profile across mu-bins: E[inc^2 | bin] / overall mean (flat = a constant-rate clock = wall time)
    print("   E[increment^2 | mu-bin] / overall E[increment^2]   (mu-bins = quintiles of the posterior mean)")
    print("   bin mean mu  " + "  ".join(f"{k:>9}" for k in inc))
    for b in range(5):
        m = idx == b
        print(f"   {xm[m].mean():11.3f}  " + "  ".join(f"{(v[m] ** 2).mean() / (v ** 2).mean():9.3f}" for v in inc.values()))
    pred = (1 / N + drift2) / cm                                             # predicted E[ds_F^2]
    print(f"   predicted vs measured E[ds_F^2]: bins " + ", ".join(
        f"{pred[idx == b].mean():.2f}/{(inc['fisher'][idx == b] ** 2).mean():.2f}" for b in range(5)))
    # clock-to-clock agreement: R^2 of per-trajectory cumulative clocks (detrended per trajectory)
    C = {k: np.cumsum(v, 1) for k, v in inc.items()}
    def r2(a, b):
        out = []
        for i in range(len(a)):
            A = np.c_[b[i], np.ones_like(b[i])]
            res = a[i] - A @ np.linalg.lstsq(A, a[i], rcond=None)[0]
            out.append(1 - res.var() / a[i].var())
        return np.median(out)
    print("   median over trajectories of R^2(fisher clock ~ affine in other clock):  " + ", ".join(
        f"{k} {r2(C['fisher'], C[k]):.4f}" for k in inc if k != "fisher"))
    # per-step increment agreement (fluctuations, not just the trend)
    corr = {k: np.corrcoef(inc["fisher"].ravel(), v.ravel())[0, 1] for k, v in inc.items() if k not in ("fisher", "wall")}
    print("   per-step correlation of fisher increments with:  " + ", ".join(f"{k} {v:.3f}" for k, v in corr.items()))
    cv = {k: v.std() / v.mean() for k, v in inc.items() if k != "wall"}
    print("   CV of per-step increments (0 = wall clock):  " + ", ".join(f"{k} {v:.2f}" for k, v in cv.items()))


print("\nB. exact grid Bayes filter, K = 2 (selection +-s with symmetric mutation u; dt = 0.1; burn-in 20 steps removed)")
report(run_filter(N=100, n_obs=100, s=0.5, u=0.1))
report(run_filter(N=2000, n_obs=2000, s=3.0, u=0.1))
report(run_filter(N=2000, n_obs=2000, s=3.0, u=0.1, seg=40))

# ------------------------------------------------------------------------------------------------ C
print("\nC. deterministic exact paths (zoo, K = 3): clock rate near the fixed point")
def rates(X, dt=0.1):
    d = np.diff(X, axis=1); xm = 0.5 * (X[:, 1:] + X[:, :-1])
    clr = np.log(X) - np.log(X).mean(-1, keepdims=True)
    return dict(euclid=np.linalg.norm(d, axis=-1) / dt, fisher=np.sqrt((d ** 2 / xm).sum(-1)) / dt,
                logit=np.linalg.norm(np.diff(clr, axis=1), axis=-1) / dt)
for mech, sched, fp in (("selection", "selection_only", "vertex"), ("migration", "migration_only", "m (interior)")):
    w = WrightFisher(K=3, schedule=sched, seg_steps=600)
    tr = w.sample(seed=3, horizon=600, n_traj=200, obs_mode="freq")
    X = tr.state
    if mech == "selection":
        star = np.eye(3)[np.asarray(tr.info["fitness"][:, 0]).argmax(-1)][:, None, :]
    else:
        star = np.asarray(tr.info["pool"][:, 0])[:, None, :]
    dist = np.linalg.norm(0.5 * (X[:, 1:] + X[:, :-1]) - star, axis=-1)
    R = rates(X)
    out = []
    for k, v in R.items():
        m = (dist > 1e-6) & (dist < 1e-2) & (v > 0)
        slope = np.polyfit(np.log(dist[m]), np.log(v[m]), 1)[0]
        stopped = (v < 0.01 * v.max(1, keepdims=True)).mean()
        out.append(f"{k}: slope {slope:+.2f}, stopped {stopped:.2f}")
    print(f"  {mech:9} -> {fp:13}  " + ";  ".join(out))
