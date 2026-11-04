"""Small recon-vs-JEPA world models on the zoo pendulum (angle_raw / angle_cs); latent H1 + wrapping + probes.

Variants
  recon_lc   : D(z_t) ~ o_t  (k = 0 recon) + latent consistency sum_k ||F^k(z_t) - sg E(o_{t+k-1}, o_{t+k})||^2   ("Dreamer-like")
  recon_ms   : sum_{k=0..K} ||D(F^k(z_t)) - o_{t+k}||^2                                                         (multi-step decoding)
  recon_ms_lc: recon_ms + latent consistency
  recon_circ : like recon_lc but loss 1 - cos(theta_hat - theta)  (angle_raw only)
  jepa       : sum_k ||F^k(z_t) - sg E(...)||^2 + VICReg variance / covariance on z_t
usage: python wm_experiment.py OBS VARIANT SEED [STEPS] [K]
"""
import sys, time, json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/user/gwm")
sys.path.insert(0, "/tmp/claude-0/-home-user/8bc92419-2f30-547d-8b18-8324b70c6f75/scratchpad/review")
import zoo
from topo_tools import analyse_cloud
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

torch.set_num_threads(1)
OBS, VAR, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3])
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
K = int(sys.argv[5]) if len(sys.argv) > 5 else 8
ACT = sys.argv[6] if len(sys.argv) > 6 else "silu"
D_LAT, H = 8, 128
torch.manual_seed(SEED); np.random.seed(SEED)

W = zoo.make("pendulum")
tr_train = W.sample(seed=1000 + SEED, horizon=64, n_traj=1000, obs_mode=OBS)
tr_ana = W.sample(seed=777, horizon=64, n_traj=200, obs_mode=OBS)


def pairs(obs):
    o = torch.tensor(obs, dtype=torch.float32)                       # [N, 65, d]
    return torch.cat([o[:, :-1], o[:, 1:]], -1), o[:, 1:]             # P[:, i] = (o_i, o_{i+1}); target o_{i+1}


P, O = pairs(tr_train.obs)
Pa, Oa = pairs(tr_ana.obs)
var_o = O.reshape(-1, O.shape[-1]).var(0).sum().item()
act = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}[ACT]


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, H), act(), nn.Linear(H, H), act(), nn.Linear(H, o))


din, dout = P.shape[-1], O.shape[-1]
E, F = mlp(din, D_LAT), mlp(D_LAT, D_LAT)
Dec = mlp(D_LAT, dout) if VAR.startswith("recon") else None
params = list(E.parameters()) + list(F.parameters()) + ([] if Dec is None else list(Dec.parameters()))
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
N, T = P.shape[:2]


def recon_loss(pred, tgt):
    if VAR == "recon_circ":
        return (1 - torch.cos(pred[..., 0] - tgt[..., 0])).mean()
    return ((pred - tgt) ** 2).sum(-1).mean() / var_o


def vicreg(z):
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    c = (z.T @ z) / (len(z) - 1)
    off = c - torch.diag(torch.diag(c))
    return torch.relu(1 - std).mean(), (off ** 2).sum() / D_LAT


t0 = time.time()
for step in range(STEPS):
    n = torch.randint(0, N, (256,))
    i = torch.randint(0, T - K, (256,))
    idx = i[:, None] + torch.arange(K + 1)[None]
    Pin = P[n[:, None], idx]                     # [B, K+1, din]
    Ot = O[n[:, None], idx]                      # [B, K+1, dout]
    z0 = E(Pin[:, 0])
    zs = [z0]
    for k in range(K):
        zs.append(zs[-1] + F(zs[-1]))
    zs = torch.stack(zs, 1)                      # [B, K+1, d]
    loss = 0.0
    if VAR in ("recon_lc", "recon_circ", "recon_ms_lc", "jepa"):
        with torch.no_grad():
            tgt = E(Pin[:, 1:].reshape(-1, din)).reshape(len(n), K, D_LAT)
        lc = ((zs[:, 1:] - tgt) ** 2).sum(-1).mean()
        loss = loss + lc
    if VAR in ("recon_lc", "recon_circ"):
        loss = loss + recon_loss(Dec(z0), Ot[:, 0])
    if VAR in ("recon_ms", "recon_ms_lc"):
        loss = loss + recon_loss(Dec(zs), Ot)
    if VAR == "jepa":
        v, c = vicreg(z0)
        loss = loss + 25.0 * v + 1.0 * c       # VICReg-style weights (pred term ~ invariance)
    opt.zero_grad(); loss.backward(); opt.step()
train_s = time.time() - t0

with torch.no_grad():
    Za = E(Pa.reshape(-1, din)).numpy()
    st = tr_ana.state[:, 1:].reshape(-1, 2)
    th, om = st[:, 0], st[:, 1]
    en = 0.5 * om ** 2 - np.cos(th)
    out = dict(obs=OBS, var=VAR, seed=SEED, steps=STEPS, K=K, act=ACT, train_s=round(train_s, 1))
    if Dec is not None:
        rec = Dec(torch.tensor(Za)).numpy()
        if VAR == "recon_circ":
            out["rec_circ_err"] = float(np.mean(1 - np.cos(rec[:, 0] - Oa.reshape(-1, dout).numpy()[:, 0])))
        else:
            err = ((rec - Oa.reshape(-1, dout).numpy()) ** 2).sum(1)
            out["nmse0"] = float(err.mean() / var_o)
            out["sup_err"] = float(np.sqrt(err.max()))
        # multi-step decoded error at k = K from the analysis set (open loop)
    zc = Za - Za.mean(0)
    s = np.linalg.svd(zc, compute_uv=False)
    pr = s ** 2 / (s ** 2).sum()
    out["eff_rank"] = float(np.exp(-(pr * np.log(pr + 1e-12)).sum()))
    out["min_std"] = float(Za.std(0).min())

# probes: kNN R^2 (k = 10, 5-fold) on a 4000-point subsample
rng = np.random.default_rng(0)
sub = rng.choice(len(Za), 4000, replace=False)
for name, y in [("cos", np.cos(th)), ("sin", np.sin(th)), ("E", en), ("omega", om)]:
    out["r2_" + name] = float(np.mean(cross_val_score(KNeighborsRegressor(10), Za[sub], y[sub], cv=5, scoring="r2")))

# seam ratio: latent distance across theta = +-pi vs across theta = 0 / +-pi/2 (same omega band, rotations)
def seam(th0, width=0.12):
    a = (np.angle(np.exp(1j * (th - th0))) > -width) & (np.angle(np.exp(1j * (th - th0))) < 0)
    b = (np.angle(np.exp(1j * (th - th0))) >= 0) & (np.angle(np.exp(1j * (th - th0))) < width)
    ds = []
    for sgn in (1, -1):
        A = np.where(a & (sgn * om > 1.6))[0]; B = np.where(b & (sgn * om > 1.6))[0]
        if len(A) == 0 or len(B) == 0:
            continue
        A, B = A[:300], B[:300]
        dom = np.abs(om[A][:, None] - om[B][None])
        dz = np.linalg.norm(Za[A][:, None] - Za[B][None], axis=-1)
        m = dom < 0.08
        if m.any():
            ds.append(np.median(dz[m]))
    return float(np.mean(ds)) if ds else np.nan

ctrl = np.nanmean([seam(c) for c in (-np.pi / 2, 0.0, np.pi / 2)])
out["seam_ratio"] = seam(np.pi) / ctrl
out["latent_diam"] = float(np.linalg.norm(Za - Za.mean(0), axis=1).max())

r = analyse_cloud(Za, th)
out.update({k: (v if not isinstance(v, float) else round(v, 3)) for k, v in r.items()})
out["bars"] = np.round(np.array(r["bars"]), 3).tolist()
print(json.dumps(out))
