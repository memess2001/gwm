"""Cell 9 review: is 'deterministic vs stochastic WORLD' the right proxy for 'posterior sigma varies'?

Ideal single-observation posterior std of the type frequencies (flat Dirichlet prior):
  obs 'counts' (Multinomial(n_obs, x)/n_obs): posterior Dirichlet(c+1) -> sd_i = sqrt(a_i(A-a_i)/(A^2(A+1)))
  obs 'freq'   (exact x): posterior is a point mass -> sd = 0 (an RSSM would sit at its floor).
A: deterministic ODE (infinite population) + counts;  B: finite-population drift (stochastic) + exact freq.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/user/gwm")
import zoo

w = zoo.make("wright_fisher")
def post_sd(counts):
    a = counts + 1.0; A = a.sum(-1, keepdims=True)
    return np.sqrt(a * (A - a) / (A**2 * (A + 1)))

for label, kw, mode in (("A deterministic ODE, obs=counts", dict(), "counts"),
                        ("B drift pop_size=200, obs=freq ", dict(pop_size=200), "freq")):
    tr = w.sample(seed=zoo.split_seed("analysis", 9), horizon=200, n_traj=64, obs_mode=mode, **kw)
    if mode == "counts":
        n = int(tr.meta["params"]["n_obs"])
        sd = post_sd(np.rint(tr.obs.astype(np.float64) * n))
    else:
        sd = np.zeros_like(tr.obs, dtype=np.float64)
    cv = sd.std(1) / np.maximum(sd.mean(1), 1e-300)          # temporal CV per trajectory, per type
    stoch = np.abs(np.diff(tr.state, axis=1)).mean()
    print(f"{label}: mean |dx| per step {stoch:.4f}; posterior sd range [{sd.min():.4f}, {sd.max():.4f}]; "
          f"median temporal CV(sd) over traj x type = {np.median(cv):.3f}")
