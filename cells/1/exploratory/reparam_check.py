"""Check 5: which readouts of C_k survive a latent reparametrisation z = psi(x)?
Model in x-coords: first-order slide maze, decoder mu = id (flat blob metric).  Same model in z-coords:
F~ = psi F psi^-1, mu~ = mu psi^-1  =>  mu~ F~^j = F^j psi^-1.  psi = (exp-stretch) o shear2 o shear1 (volume-changing).
Compare raw eigenvalues / raw log det C_k (coordinate-dependent) with eig(C_0^-1 C_k) (relative, should be invariant)."""
import numpy as np
from maze import walls, step1

W = walls(thick=False)
k, dvec = 4, 0.25 * np.array([np.cos(0.3), np.sin(0.3)])
al = 0.3
psi = lambda x: np.stack([np.exp(0.2 * (x[:, 0] + al * np.sin(x[:, 1]))),
                          x[:, 1] + al * np.sin(x[:, 0] + al * np.sin(x[:, 1]))], 1)
def psi_inv(z):
    a = np.log(z[:, 0]) / 0.2                     # undo exp-stretch: a = x0 + al sin(x1)
    x1 = z[:, 1] - al * np.sin(a)                 # undo shear2
    return np.stack([a - al * np.sin(x1), x1], 1) # undo shear1

def Fk_list(x):
    out = [x]
    for _ in range(k):
        x = step1(x, np.repeat(dvec[None], len(x), 0), "slide", W); out.append(x)
    return out

def CG(fun, p, h=1e-5):
    e = np.eye(2); P = [fun(p + h * e[i]) for i in range(2)]; M = [fun(p - h * e[i]) for i in range(2)]
    Cs = []
    for j in range(k + 1):
        J = np.stack([(P[i][j] - M[i][j]) / (2 * h) for i in range(2)], -1)
        Cs.append(np.einsum("noi,noj->nij", J, J))
    return Cs[0], sum(Cs)

rng = np.random.default_rng(1)
x = np.concatenate([np.stack([rng.uniform(4.0, 4.95, 300), rng.uniform(1.2, 4.8, 300)], 1),   # wall band (left side)
                    np.stack([rng.uniform(2.0, 3.0, 300), rng.uniform(2.0, 8.0, 300)], 1)])  # far field
assert np.allclose(psi_inv(psi(x)), x)
C0x, Ckx = CG(Fk_list, x)
C0z, Ckz = CG(lambda z: Fk_list(psi_inv(z)), psi(x))
rel = lambda C0, Ck: np.linalg.eigvals(np.linalg.solve(C0, Ck)).real
ex, ez = np.sort(np.linalg.eigvalsh(Ckx), 1), np.sort(np.linalg.eigvalsh(Ckz), 1)
rx, rz = np.sort(rel(C0x, Ckx), 1), np.sort(rel(C0z, Ckz), 1)
ldx, ldz = np.log(np.linalg.det(Ckx)), np.log(np.linalg.det(Ckz))
print(f"raw eigenvalues of C_k: max |x-coords - z-coords| / value = {np.max(np.abs(ex - ez) / ex):.2f}")
print(f"raw log det C_k: max |difference| = {np.max(np.abs(ldx - ldz)):.2f}  (= -2 log|det Dpsi|, a gauge term)")
print(f"relative eig(C_0^-1 C_k): max |x - z| / value = {np.max(np.abs(rx - rz) / rx):.1e}")
print(f"relative log det(C_0^-1 C_k): max |difference| = {np.max(np.abs(np.log(rx).sum(1) - np.log(rz).sum(1))):.1e}")
band, far = slice(0, 300), slice(300, 600)
print(f"(sanity) relative lam_min in band: {rx[band, 0].min():.2f}..{rx[band, 0].max():.2f}, far: {rx[far, 0].min():.2f}..{rx[far, 0].max():.2f} (free = {k+1})")
