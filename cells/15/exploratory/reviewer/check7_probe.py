# "Decode to physical coordinates and use Lebesgue volume": with a PERFECT world model (latent field f = phi_* F) and
# a learned MSE probe psi_hat: z -> (cos th, sin th, om), the readout is div F + F . grad log|det D(psi_hat o phi)|.
# How big is the probe-induced term for a typical MLP probe? (2-D latent; phi = smooth chart of the cylinder.)
import sys, numpy as np, torch
torch.set_num_threads(1)
ntrain, steps, width, noise = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
torch.manual_seed(0); rng = np.random.default_rng(0)
def phi(th, om):            # latent chart: cylinder -> punctured plane, plus a smooth nonlinear warp
    r = torch.exp(0.35 * om) * (1 + 0.15 * torch.sin(2 * th + 0.5 * om))
    z1, z2 = r * torch.cos(th), r * torch.sin(th)
    return torch.stack([z1 + 0.2 * torch.tanh(z2), z2 + 0.1 * z1 ** 2], -1)
th = torch.tensor(rng.uniform(-np.pi, np.pi, ntrain)); om = torch.tensor(rng.uniform(-2.5, 2.5, ntrain))
Z = phi(th, om).float()
Yl = torch.stack([torch.cos(th), torch.sin(th), om], -1).float()
Yl = Yl + noise * torch.randn_like(Yl)                     # label noise (e.g. labels from a noisy tracker)
mu, sd = Z.mean(0), Z.std(0)
net = torch.nn.Sequential(torch.nn.Linear(2, width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU(),
                          torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, 3))
opt = torch.optim.Adam(net.parameters(), 2e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
for it in range(steps):
    i = torch.randint(0, ntrain, (512,))
    loss = ((net((Z[i] - mu) / sd) - Yl[i]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
net = net.double(); mu, sd = mu.double(), sd.double()
def chart(x):               # x = (th, om) -> decoded physical coords (th_hat, om_hat)
    y = net((phi(x[..., 0], x[..., 1]) - mu) / sd)
    return torch.stack([torch.atan2(y[..., 1], y[..., 0]), y[..., 2]], -1)
TH, OM = np.meshgrid(np.linspace(-np.pi, np.pi, 181, endpoint=False), np.linspace(-2.3, 2.3, 41), indexing="ij")
X = torch.tensor(np.stack([TH.ravel(), OM.ravel()], 1), requires_grad=True)
def logdet(x):
    J = torch.func.vmap(torch.func.jacrev(chart))(x)
    return torch.log(torch.abs(torch.linalg.det(J)))
ld = logdet(X)
grad = torch.autograd.grad(ld.sum(), X)[0].detach().numpy()
F = np.stack([OM.ravel(), -np.sin(TH.ravel())], 1)
fake = (F * grad).sum(1)
with torch.no_grad():
    err = chart(X) - X
    err[:, 0] = torch.remainder(err[:, 0] + np.pi, 2 * np.pi) - np.pi
print(f"ntrain={ntrain} steps={steps} width={width} label_noise={noise}: probe RMSE th {err[:,0].pow(2).mean().sqrt():.4f} rad, "
      f"om {err[:,1].pow(2).mean().sqrt():.4f}; |det D(psi o phi)| in [{ld.exp().min():.3f},{ld.exp().max():.3f}]; "
      f"probe-induced fake div RMS {np.sqrt((fake**2).mean()):.3f}, max {np.abs(fake).max():.2f}  (signal 0.10)")
