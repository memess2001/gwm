"""Cell 2 ground-truth checks (exact fields only, no learning): Hodge residual of known Wright-Fisher fields under
three metrics on the 2-simplex, with one fixed potential basis. Also checks that the WF diffusion drift correction
(div D) is itself a Shahshahani gradient, so 'Fisher curl = 0' <=> detailed balance for finite-N WF."""
import itertools, sys
import numpy as np
sys.path.insert(0, "/home/user/gwm_repo")
from zoo.wright_fisher import WrightFisher, replicator_hodge

K = 3
rng = np.random.default_rng(0)

def interior(n, lo=0.02):
    x = rng.dirichlet(np.ones(K), 4 * n)
    return x[(x > lo).all(1)][:n]

def basis_grads(x):
    """Gradients d phi / dx (in R^K, ambient) of the fixed basis: x_i, x_i x_j, x_i x_j x_k, log x_i."""
    n, G = len(x), []
    for i in range(K):
        g = np.zeros((n, K)); g[:, i] = 1; G.append(g)
    for i, j in itertools.combinations_with_replacement(range(K), 2):
        g = np.zeros((n, K)); g[:, i] += x[:, j]; g[:, j] += x[:, i]; G.append(g)
    for i, j, k in itertools.combinations_with_replacement(range(K), 3):
        g = np.zeros((n, K))
        for a, (b, c) in ((i, (j, k)), (j, (i, k)), (k, (i, j))):
            g[:, a] += x[:, b] * x[:, c]
        G.append(g)
    for i in range(K):
        g = np.zeros((n, K)); g[:, i] = 1 / x[:, i]; G.append(g)
    return np.stack(G, 1)                       # [n, B, K]

def Ginv(x, metric):
    P = np.einsum("ni,ij->nij", x, np.eye(K)) - np.einsum("ni,nj->nij", x, x)    # diag x - x x'
    if metric == "fisher":
        return P
    if metric == "euclid_freq":
        return np.broadcast_to(np.eye(K) - 1 / K, (len(x), K, K))
    if metric == "aitchison_clr":
        return P @ P
    raise ValueError(metric)

def residual(x, v, metric):
    Gi = Ginv(x, metric)
    Gm = np.linalg.pinv(Gi, hermitian=True)          # metric on the tangent plane
    L = np.linalg.cholesky(Gm + 1e-12 * np.eye(K))   # ||w||_G^2 = |L' w|^2
    W = np.einsum("nij,nbj->nbi", Gi, basis_grads(x))    # gradient fields of the basis
    A = np.einsum("nji,nbj->nbi", L, W).transpose(0, 2, 1).reshape(-1, W.shape[1])
    y = np.einsum("nji,nj->ni", L, v).reshape(-1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(((y - A @ c) ** 2).sum() / (y ** 2).sum())

wf = WrightFisher(K=3)
x = interior(4000)
f = np.array([0.8, -0.3, -0.5]); m = np.array([0.5, 0.3, 0.2])
Asym = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], float)
fields = {
    "selection (linear f)": wf.vector_field(x, "selection", f=f),
    "sym payoff replicator": WrightFisher(K=3, rps_A=Asym).vector_field(x, "rps"),
    "migration": wf.vector_field(x, "migration", mu=0.5, m=m),
    "uniform mutation+sel": WrightFisher(K=3, mutation="uniform").vector_field(x, "mutation_selection", f=f, u=0.5),
    "mixed (sel+mig)": wf.vector_field(x, "mixed", f=f, mu=0.5, m=m),
}
for u in (0.05, 0.1, 0.2, 0.4, 0.8):
    fields[f"cyclic mut+sel u={u}"] = WrightFisher(K=3, mutation="cyclic").vector_field(x, "mutation_selection", f=f, u=u)
fields["zero-sum RPS"] = wf.vector_field(x, "rps")
print(f"{'field':26} {'label':10} " + " ".join(f"{m:>13}" for m in ("fisher", "euclid_freq", "aitchison_clr")))
for name, v in fields.items():
    lab = ("gradient" if not any(s in name for s in ("cyclic", "RPS")) else "rot/mixed")
    print(f"{name:26} {lab:10} " + " ".join(f"{residual(x, v, mt):13.2e}" for mt in ("fisher", "euclid_freq", "aitchison_clr")))

# div D for WF diffusion D = (diag x - x x') / (2N) in K-1 coordinates: (div D)_i = (1 - K x_i) / (2N)
N = 100.0
h = 1e-6
def D(z):     # z = (x1, x2), x3 = 1 - x1 - x2
    xx = np.r_[z, 1 - z.sum()][:2]
    return (np.diag(xx) - np.outer(xx, xx)) / (2 * N)
z = np.array([0.3, 0.45])
divD = np.array([sum((D(z + h * np.eye(2)[j])[i, j] - D(z - h * np.eye(2)[j])[i, j]) / (2 * h) for j in range(2)) for i in range(2)])
xfull = np.r_[z, 1 - z.sum()]
print("div D numeric", divD, " formula (1 - K x_i)/(2N)", (1 - K * xfull[:2]) / (2 * N))
xs = interior(4000)
corr = (1 - K * xs) / (2 * N)          # = (K/(2N)) (1/K - x): migration toward uniform -> Shahshahani gradient
print("Fisher residual of div D field:", f"{residual(xs, corr, 'fisher'):.2e}")
