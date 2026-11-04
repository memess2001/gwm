# Does a SMOOTH learned decoder inherit the low-frequency part of log a(theta)? And does a learned decoder on a
# rotation-equivariant renderer (k=0) create its OWN low-frequency fake divergence?
# Decoder g: (cos th, sin th) -> 64x64 frame (MLP, SiLU), MSE. a_g(th) = ||d g / d th|| by forward-mode AD.
# (The decoder pulled-back volume is coordinate-free, so a fixed latent chart is WLOG for this question; the two-frame
#  nu factorises as dt a(th) a(th - om dt) for a block decoder (g(th), g(th - om dt)).)
import sys, time, numpy as np, torch
from common import *
torch.set_num_threads(1); torch.manual_seed(0)
k, width, steps, ntrain = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
torch.manual_seed(seed)
thetas = np.linspace(-np.pi, np.pi, 721, endpoint=False)
omegas = np.linspace(-2.5, 2.5, 51)
TH, OM = np.meshgrid(thetas, omegas, indexing="ij")
dx, dy, _ = out_grid(k)                                # clean analytic warp, point-sampled 64px camera (= what a camera records)
rng = np.random.default_rng(seed)
tr = rng.uniform(-np.pi, np.pi, ntrain)
Y = torch.tensor(scene(dx, dy, tr), dtype=torch.float32)
X = torch.tensor(np.stack([np.cos(tr), np.sin(tr)], 1), dtype=torch.float32)
net = torch.nn.Sequential(torch.nn.Linear(2, width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU(),
                          torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, 4096))
opt = torch.optim.Adam(net.parameters(), 2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
t0 = time.time()
for it in range(steps):
    idx = torch.randint(0, ntrain, (256,))
    loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
th = torch.tensor(thetas, dtype=torch.float32, requires_grad=False)
def g_of_theta(t):
    return net(torch.stack([torch.cos(t), torch.sin(t)], 1))
with torch.no_grad():
    rec = g_of_theta(th).numpy()
truth = scene(dx, dy, thetas)
rel = np.linalg.norm(rec - truth) / np.linalg.norm(truth)
_, jv = torch.func.jvp(g_of_theta, (th,), (torch.ones_like(th),))
a_g = jv.detach().numpy().astype(np.float64)
a_g = np.sqrt((a_g ** 2).sum(1))
a_true = a_analytic(thetas, k, ss=1)
a_cont = a_analytic(thetas, k, ss=4)
def fd(a, nm):
    f = fake_div_2frame(thetas, dlog_lowpass(np.log(a), nm), TH, OM); return np.sqrt((f ** 2).mean()), np.abs(f).max()
h_g, h_t = harmonics(np.log(a_g), 6), harmonics(np.log(a_cont), 6)
print(f"k={k} width={width} steps={steps} ntrain={ntrain} seed={seed}: train loss {loss.item():.2e}, test rel L2 recon err {rel:.3f}, "
      f"time {time.time()-t0:.0f}s")
print(f"  a_g/a_true mean {np.mean(a_g/a_true):.3f}  | log a harmonics n=1..6: decoder {np.round(h_g[1:],4)}  camera(cont.) {np.round(h_t[1:],4)}")
for nm in (4, None):
    (r1, m1), (r2, m2) = fd(a_g, nm), fd(a_cont, nm)
    print(f"  fake div RMS/max [harmonics <= {nm}]: decoder {r1:.3f}/{m1:.2f}   vs continuum camera {r2:.3f}/{m2:.2f}")
