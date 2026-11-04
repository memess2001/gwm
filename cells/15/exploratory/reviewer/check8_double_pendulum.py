# Lebesgue volume in decoded physical coords (q, qdot) is NOT the Liouville volume when the mass matrix depends on q.
# Conservative double pendulum (m1=m2=l1=l2=g=1): div_Leb F = -d/dt log det M(q), det M = 2 - cos^2(th1-th2).
import numpy as np, torch
torch.set_default_dtype(torch.float64)
def F(s):
    t1, t2, w1, w2 = s.unbind(-1); D = t1 - t2
    M11, M12, M22 = 2.0 * torch.ones_like(D), torch.cos(D), torch.ones_like(D)
    b1 = -w2 ** 2 * torch.sin(D) - 2 * torch.sin(t1)
    b2 = w1 ** 2 * torch.sin(D) - torch.sin(t2)
    det = M11 * M22 - M12 ** 2
    a1 = (M22 * b1 - M12 * b2) / det; a2 = (-M12 * b1 + M11 * b2) / det
    return torch.stack([w1, w2, a1, a2], -1)
# sample states along a chaotic conservative trajectory (RK4), E well above the chaos threshold
s = torch.tensor([2.0, -1.0, 0.0, 0.5]); h = 0.002; traj = []
for i in range(200000):
    k1 = F(s); k2 = F(s + h/2*k1); k3 = F(s + h/2*k2); k4 = F(s + h*k3); s = s + h/6*(k1 + 2*k2 + 2*k3 + k4)
    if i % 50 == 0: traj.append(s.clone())
X = torch.stack(traj).requires_grad_(True)
J = torch.func.vmap(torch.func.jacrev(F))(X)
div = torch.diagonal(J, dim1=-2, dim2=-1).sum(-1).detach()
D = (X[:, 0] - X[:, 1]).detach(); Dd = (X[:, 2] - X[:, 3]).detach()
pred = -torch.sin(2 * D) * Dd / (2 - torch.cos(D) ** 2)
print(f"double pendulum, {len(X)} states on a chaotic orbit: Lebesgue div in (q,qdot): RMS {div.pow(2).mean().sqrt():.3f}, "
      f"max {div.abs().max():.2f}; matches -d/dt log det M: max|diff| {(div - pred).abs().max():.1e}; orbit mean {div.mean():+.4f}")
# divergence w.r.t. det M(q) dq dqdot (the Liouville volume in velocity coordinates): should be 0
logdetM = lambda x: torch.log(2 - torch.cos(x[..., 0] - x[..., 1]) ** 2)
g = torch.autograd.grad(logdetM(X).sum(), X)[0]
divL = div + (F(X).detach() * g).sum(-1)
print(f"  divergence w.r.t. det M(q) dq dqdot: max |.| = {divL.abs().max():.1e}")
