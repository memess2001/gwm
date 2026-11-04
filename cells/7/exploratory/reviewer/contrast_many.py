# e/m residual contrast C = (sqrt Jm - sqrt Je)/(sqrt Jm + sqrt Je) (Fisher/Sampson residuals) on many draws;
# pooled Spearman with rho (mixed segments only) with a draw-clustered bootstrap CI; per-draw monotonicity.
import numpy as np
from scipy.stats import spearmanr
from estimators import simulate, rho_share, sampson_profile

LE = np.array([0.0, 1.0])
WS = np.linspace(0, 1, 11)
mixed = (WS > 0) & (WS < 1)


def C_of(X):
    J, _ = sampson_profile(X, LE)
    se, sm = np.sqrt(J[0]), np.sqrt(J[1])
    return (sm - se) / (sm + se + 1e-300)


for T in (1.0, 2.0):
    allC, allR, per_seed = [], [], []
    for seed in range(1, 6):
        rng = np.random.default_rng(seed)
        Cs, Rs = np.zeros((40, 11, 2)), np.zeros((40, 11))
        for d in range(40):
            x0 = rng.dirichlet(np.ones(3) * 2); f = rng.normal(size=3); m = rng.dirichlet(np.ones(3) * 2)
            mu = rng.uniform(0.2, 1.0)
            for j, w in enumerate(WS):
                X = simulate(x0, f, m, mu, w, T=T)
                Rs[d, j] = rho_share(X, f, m, mu, w)
                Cs[d, j] = [C_of(X), C_of(X[:10])]
        per_seed.append([spearmanr(Cs[:, mixed, k].ravel(), Rs[:, mixed].ravel()).correlation for k in (0, 1)])
        allC.append(Cs); allR.append(Rs)
    C = np.concatenate(allC); Rr = np.concatenate(allR)
    nd = len(C)
    bs = np.random.default_rng(0)
    print(f"T={T}: per-seed pooled Spearman(C, rho) [global, early25]: {np.round(per_seed, 2).tolist()}")
    for k, nm in enumerate(("global", "early25")):
        sp = spearmanr(C[:, mixed, k].ravel(), Rr[:, mixed].ravel()).correlation
        boot = []
        for _ in range(300):
            idx = bs.integers(0, nd, nd)
            boot.append(spearmanr(C[idx][:, mixed, k].ravel(), Rr[idx][:, mixed].ravel()).correlation)
        mono = np.mean(np.all(np.diff(C[:, :, k], axis=1) >= -1e-9, axis=1))
        per = np.array([spearmanr(C[d, :, k], WS).correlation for d in range(nd)])
        worst = np.min(np.diff(C[:, :, k], axis=1), axis=1)
        print(f"   C_{nm:8s} 200 draws: pooled Spearman(C,rho)={sp:.3f}  95% cluster-bootstrap CI "
              f"[{np.quantile(boot, .025):.3f}, {np.quantile(boot, .975):.3f}]  strictly monotone draws {mono:.2f}  "
              f"per-draw Sp median {np.median(per):.2f}, frac>=0.8 {np.mean(per >= 0.8):.2f}  "
              f"worst reversal median {np.median(worst):+.2f}, frac reversal<-0.2 {np.mean(worst < -0.2):.2f}")
