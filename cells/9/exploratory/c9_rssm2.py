"""Cell 9 review: does a Dreamer-style Gaussian RSSM on the DETERMINISTIC zoo pendulum put sigma at its floor,
constant along trajectories?  And how do the clocks compare?

Minimal DreamerV1-style RSSM (no actions): GRU deterministic path h_t, stochastic z_t with
prior p(z_t|h_t), posterior q(z_t|h_t,e_t), std = softplus(.) + min_std; unit-variance Gaussian decoder on
16x16 frames (4x4 average-pooled pixels64, values in [0,1], minus 0.5); loss = recon + max(KL, free_nats).
usage: c9_rssm2.py MIN_STD STEPS [FREE_NATS] [SEED] [DECODER_STD]
"""
import sys, time, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn

torch.set_num_threads(2)
sys.path.insert(0, "/home/user/gwm")
import zoo

MIN_STD = float(sys.argv[1]); STEPS = int(sys.argv[2])
FREE = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
SOBS = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
torch.manual_seed(SEED); np.random.seed(SEED)
DZ, DH, W = 16, 128, 128

w = zoo.make("pendulum")
def frames16(tr):
    x = tr.obs.astype(np.float32) / 255.0                      # [N, T+1, 64, 64]
    N, T1 = x.shape[:2]
    x = x.reshape(N, T1, 16, 4, 16, 4).mean((3, 5)).reshape(N, T1, 256)
    return x - 0.5
tr_train = w.sample(seed=zoo.split_seed("train", 9), horizon=64, n_traj=384, obs_mode="pixels64")
tr_test = w.sample(seed=zoo.split_seed("test", 9), horizon=120, n_traj=48, obs_mode="pixels64")
Xtr = torch.tensor(frames16(tr_train)); Xte = torch.tensor(frames16(tr_test))


class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(256, W), nn.ELU(), nn.Linear(W, W), nn.ELU())
        self.inp = nn.Sequential(nn.Linear(DZ, DH), nn.ELU())
        self.gru = nn.GRUCell(DH, DH)
        self.prior = nn.Sequential(nn.Linear(DH, W), nn.ELU(), nn.Linear(W, 2 * DZ))
        self.post = nn.Sequential(nn.Linear(DH + W, W), nn.ELU(), nn.Linear(W, 2 * DZ))
        self.dec = nn.Sequential(nn.Linear(DH + DZ, W), nn.ELU(), nn.Linear(W, W), nn.ELU(), nn.Linear(W, 256))

    @staticmethod
    def stats(x):
        m, s = x.chunk(2, -1)
        return m, Fn.softplus(s) + MIN_STD

    def filter(self, X, sample=True):
        B, T = X.shape[:2]
        e = self.enc(X)
        h = X.new_zeros(B, DH); z = X.new_zeros(B, DZ)
        out = {k: [] for k in ("h", "mq", "sq", "mp", "sp", "z")}
        for t in range(T):
            h = self.gru(self.inp(z), h)
            mp, sp = self.stats(self.prior(h))
            mq, sq = self.stats(self.post(torch.cat([h, e[:, t]], -1)))
            z = mq + sq * torch.randn_like(sq) if sample else mq
            for k, v in zip(out, (h, mq, sq, mp, sp, z)):
                out[k].append(v)
        return {k: torch.stack(v, 1) for k, v in out.items()}

    def loss(self, X):
        o = self.filter(X)
        xh = self.dec(torch.cat([o["h"], o["z"]], -1))
        rec = 0.5 * (((xh - X) / SOBS) ** 2).sum(-1).mean()
        kl = (torch.log(o["sp"] / o["sq"]) + (o["sq"] ** 2 + (o["mq"] - o["mp"]) ** 2) / (2 * o["sp"] ** 2) - 0.5).sum(-1)
        klm = kl.mean()
        return rec + torch.clamp(klm, min=FREE), rec.detach(), klm.detach()


m = RSSM()
opt = torch.optim.Adam(m.parameters(), lr=6e-4, eps=1e-4)
t0 = time.time(); B, L = 16, 50
for it in range(STEPS):
    i = torch.randint(0, Xtr.shape[0], (B,)); s = torch.randint(0, Xtr.shape[1] - L + 1, (1,)).item()
    loss, rec, kl = m.loss(Xtr[i, s:s + L])
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 100.0); opt.step()
    if it % 250 == 0 or it == STEPS - 1:
        print(f"it {it:5d} rec {rec.item():8.3f} kl {kl.item():7.3f}  ({time.time() - t0:5.1f}s)", flush=True)

# ---------------------------------------------------------------- analysis on held-out trajectories
with torch.no_grad():
    o = m.filter(Xte, sample=False)          # posterior means fed back (deterministic filtering)
    xh = m.dec(torch.cat([o["h"], o["mq"]], -1))
    print(f"test recon (sum sq / 2 per frame): {0.5 * ((xh - Xte) ** 2).sum(-1).mean():.3f}  "
          f"(predicting the mean frame: {0.5 * ((Xte - Xte.mean((0, 1))) ** 2).sum(-1).mean():.3f})")
mq, sq, h = o["mq"].numpy().astype(np.float64), o["sq"].numpy().astype(np.float64), o["h"].numpy().astype(np.float64)
mp, sp = o["mp"].numpy().astype(np.float64), o["sp"].numpy().astype(np.float64)
BURN = 10
kl_dim = (np.log(sp / sq) + (sq ** 2 + (mq - mp) ** 2) / (2 * sp ** 2) - 0.5)[:, BURN:].mean((0, 1))
active = kl_dim > 0.05
sig_mean = sq[:, BURN:].mean((0, 1))
at_floor = (sq[:, BURN:] < MIN_STD * 1.02).mean((0, 1))
cv_time = (sq[:, BURN:].std(1) / sq[:, BURN:].mean(1)).mean(0)          # temporal CV per traj, averaged
print(f"\nmin_std = {MIN_STD}, free nats = {FREE}; active dims (KL>0.05): {active.sum()} / {DZ}")
print(" dim  KL/dim  mean_sigma  frac_at_floor  temporal_CV(sigma)  std_t(mu)")
mu_std_t = mq[:, BURN:].std(1).mean(0)
for d in np.argsort(-kl_dim):
    print(f" {d:3d} {kl_dim[d]:7.3f} {sig_mean[d]:10.4f} {at_floor[d]:13.3f} {cv_time[d]:18.4f} {mu_std_t[d]:10.4f}")
print(f"burn-in: mean sigma over active dims at t=0,1,2,5,10,50: "
      + ", ".join(f"{sq[:, t][:, active].mean():.4f}" for t in (0, 1, 2, 5, 10, 50)))


def fr_geo(m1, s1, m2, s2):
    a = (m1 - m2) ** 2 / 2 + (s1 - s2) ** 2; b = (m1 - m2) ** 2 / 2 + (s1 + s2) ** 2
    return 2 * np.sqrt(2) * np.arctanh(np.sqrt(np.clip(a / b, 0, 1 - 1e-15)))

dmu, dsg = np.diff(mq, axis=1), np.diff(sq, axis=1)
smid = 0.5 * (sq[:, 1:] + sq[:, :-1])
sbar = sq[:, BURN:].mean(1, keepdims=True)                                # frozen sigma (per traj, per dim)
X = Xte.numpy().astype(np.float64)
speeds = {
    "fisher_line": np.sqrt(((dmu ** 2 + 2 * dsg ** 2) / smid ** 2).sum(-1)),
    "fisher_mu_only": np.sqrt((dmu ** 2 / smid ** 2).sum(-1)),
    "fisher_geo_sum": np.sqrt((fr_geo(mq[:, :-1], sq[:, :-1], mq[:, 1:], sq[:, 1:]) ** 2).sum(-1)),
    "whitened_frozen": np.sqrt((dmu ** 2 / sbar ** 2).sum(-1)),
    "eucl_mu_raw": np.sqrt((dmu ** 2).sum(-1)),
    "eucl_mu_active": np.sqrt((dmu[..., active] ** 2).sum(-1)),
    "eucl_h_mu": np.sqrt((np.diff(h, axis=1) ** 2).sum(-1) + (dmu ** 2).sum(-1)),
    "obs_space": np.sqrt((np.diff(X, axis=1) ** 2).sum(-1)),
    "wall": np.ones_like(dmu[..., 0]),
}
r = np.abs(dmu[:, BURN:][..., active]) / smid[:, BURN:][..., active]
print(f"\nper-step |dmu_i|/sigma_i on active dims: median {np.median(r):.2f}, 90% {np.quantile(r, .9):.2f}, max {r.max():.1f}")
sig_term = (2 * dsg ** 2 / smid ** 2).sum(-1)[:, BURN:]; mu_term = (dmu ** 2 / smid ** 2).sum(-1)[:, BURN:]
print(f"share of Fisher speed^2 from the 2 dsigma^2/sigma^2 term: {sig_term.sum() / (sig_term + mu_term).sum():.4f}")

# events: bottom passages theta: crossing 0 with |theta|<pi/2 (ground truth, analysis only)
th = tr_test.state[..., 0]
ref = speeds["fisher_line"][:, BURN:]
print("\nclock            corr(log speed, fisher_line)  sd(log(speed/fisher)) [rescaled]   CV(inter-event spacing)")
for k, v in speeds.items():
    vv = v[:, BURN:]
    lr = np.log(vv + 1e-12) - np.log(ref + 1e-12)
    lr = lr - lr.mean()
    c = np.corrcoef(np.log(vv + 1e-12).ravel(), np.log(ref + 1e-12).ravel())[0, 1] if k != "wall" else np.nan
    gaps = []
    for n in range(th.shape[0]):
        tau = np.concatenate([[0], np.cumsum(v[n])])
        idx = np.where((th[n, 1:] * th[n, :-1] < 0) & (np.abs(th[n, :-1]) < np.pi / 2))[0]
        idx = idx[idx >= BURN]
        gaps.extend(np.diff(tau[idx]))
    gaps = np.array(gaps)
    print(f"{k:16s} {c:28.4f} {lr.std():22.4f} {gaps.std() / gaps.mean():24.4f}   (n={len(gaps)})")
