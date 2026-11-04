"""Post-hoc on saved RSSM arrays: paired cluster-bootstrap (over trajectories) of event-spacing CV differences
between clocks, plus how much of the Fisher-vs-frozen discrepancy a CV(sigma) gate would predict.
usage: c9_posthoc.py ARRAYS.npz
"""
import sys
import numpy as np

d = np.load(sys.argv[1])
mq, sq, h, th = d["mq"], d["sq"], d["h"], d["state"][..., 0]
BURN = 10

def fr_geo(m1, s1, m2, s2):
    a = (m1 - m2) ** 2 / 2 + (s1 - s2) ** 2; b = (m1 - m2) ** 2 / 2 + (s1 + s2) ** 2
    return 2 * np.sqrt(2) * np.arctanh(np.sqrt(np.clip(a / b, 0, 1 - 1e-15)))

dmu, dsg = np.diff(mq, axis=1), np.diff(sq, axis=1)
smid = 0.5 * (sq[:, 1:] + sq[:, :-1]); sbar = sq[:, BURN:].mean(1, keepdims=True)
speeds = {
    "fisher_line": np.sqrt(((dmu**2 + 2 * dsg**2) / smid**2).sum(-1)),
    "whitened_frozen": np.sqrt((dmu**2 / sbar**2).sum(-1)),
    "eucl_mu_raw": np.sqrt((dmu**2).sum(-1)),
    "eucl_h_mu": np.sqrt((np.diff(h, axis=1) ** 2).sum(-1) + (dmu**2).sum(-1)),
    "fisher_geo_sum": np.sqrt((fr_geo(mq[:, :-1], sq[:, :-1], mq[:, 1:], sq[:, 1:]) ** 2).sum(-1)),
}
# per-trajectory event gaps for each clock (same events for all clocks -> paired)
gaps = {k: [] for k in speeds}
for n in range(th.shape[0]):
    idx = np.where((th[n, 1:] * th[n, :-1] < 0) & (np.abs(th[n, :-1]) < np.pi / 2))[0]
    idx = idx[idx >= BURN]
    for k, v in speeds.items():
        tau = np.concatenate([[0], np.cumsum(v[n])])
        gaps[k].append(np.diff(tau[idx]))
def cv(gl):
    g = np.concatenate(gl) if len(gl) else np.array([np.nan])
    return g.std() / g.mean()
rng = np.random.default_rng(0)
N = th.shape[0]
boots = rng.integers(0, N, (2000, N))
print(f"{sys.argv[1]}: n_traj={N}, n_gaps={sum(len(g) for g in gaps['fisher_line'])}")
for a, b in (("fisher_line", "whitened_frozen"), ("fisher_line", "eucl_mu_raw"), ("fisher_line", "eucl_h_mu"),
             ("fisher_geo_sum", "whitened_frozen")):
    diff = cv(gaps[a]) - cv(gaps[b])
    bs = [cv([gaps[a][i] for i in bi]) - cv([gaps[b][i] for i in bi]) for bi in boots]
    lo, hi = np.quantile(bs, [0.025, 0.975])
    print(f"  CV[{a}] - CV[{b}] = {diff:+.4f}  95% cluster-bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
