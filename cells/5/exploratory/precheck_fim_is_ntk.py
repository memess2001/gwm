"""Cells 5 and 23, check before preregistration (ground truth and untrained networks only; nothing is trained).

A. Identity: the prediction-space Fisher information of a network with Gaussian output noise is J J^T / sigma^2,
   J = d(predictions)/d(parameters). J J^T is the empirical neural tangent kernel (NTK) Gram matrix, so the "hard
   directions in prediction space" are its top eigenvectors.
B. Cell 5's criterion ("top-k hard directions of two architectures within 30 degrees, well above random subspaces")
   evaluated on UNTRAINED networks of different architectures.
C. How much of the untrained hard subspace is spanned by low-order functions of the inputs (the data side).
D. Cell 23: the sample size at which the conservative term and the damping term of the pendulum become detectable,
   as a function of the damping gamma (ordinary least squares on ground-truth features, no network).

Run from the repository root with PYTHONPATH=. ; a few minutes on one free CPU thread (about 10 minutes next to
three training jobs).
"""
import numpy as np
import torch

import zoo

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
N_PTS, K_LIST = 300, (3, 6, 12)


def data(n, gamma=0.0, seed=0):
    w = zoo.make("pendulum")
    tr = w.sample(seed=seed, horizon=1, n_traj=n, obs_mode="angle_cs", gamma=gamma)
    th, om = tr.state[:, 0, 0], tr.state[:, 0, 1]
    th1, om1 = tr.state[:, 1, 0], tr.state[:, 1, 1]
    X = np.stack([np.cos(th), np.sin(th), om], 1)
    Y = np.stack([np.cos(th1), np.sin(th1), om1], 1)
    return X, Y, th, om, om1


def mlp(act, width, depth, norm=False, residual=False, seed=0):
    torch.manual_seed(seed)
    layers, d = [], 3
    for _ in range(depth):
        layers.append(torch.nn.Linear(d, width))
        if norm:
            layers.append(torch.nn.LayerNorm(width))
        layers.append(act())
        d = width
    layers.append(torch.nn.Linear(d, 3))
    net = torch.nn.Sequential(*layers)
    return (lambda x: x + net(x), net) if residual else (net, net)


ARCHS = {
    "tanh 2x64": lambda s: mlp(torch.nn.Tanh, 64, 2, seed=s),
    "relu 3x128": lambda s: mlp(torch.nn.ReLU, 128, 3, seed=s),
    "silu+LN 2x96": lambda s: mlp(torch.nn.SiLU, 96, 2, norm=True, seed=s),
    "residual silu 2x64": lambda s: mlp(torch.nn.SiLU, 64, 2, residual=True, seed=s),
}


def jacobian(fn, net, X):
    """J [n * 3, P]: rows ordered (point, output channel)."""
    params = [p for p in net.parameters()]
    x = torch.tensor(X)
    y = fn(x).reshape(-1)
    rows = []
    for i in range(y.numel()):
        g = torch.autograd.grad(y[i], params, retain_graph=True)
        rows.append(torch.cat([a.reshape(-1) for a in g]))
    return torch.stack(rows).numpy()


def top_subspace(J, k):
    U, s, _ = np.linalg.svd(J, full_matrices=False)
    return U[:, :k], s


def angles(U, V):
    c = np.clip(np.linalg.svd(U.T @ V, compute_uv=False), 0, 1)
    return np.degrees(np.arccos(c.min())), float((c ** 2).mean())      # largest principal angle, mean cos^2


def main():
    X, Y, th, om, _ = data(N_PTS)
    n = len(X)
    print(f"A. identity on {n} pendulum states (3 inputs, 3 outputs; prediction space has {3 * n} dimensions)")
    J = {}
    for name, make in ARCHS.items():
        for seed in (0, 1, 2):
            fn, net = make(seed)
            J[name, seed] = jacobian(fn, net, X)
    Ja = J["tanh 2x64", 0]
    ev_fim = np.sort(np.linalg.eigvalsh(Ja.T @ Ja))[::-1][:20]          # parameter-space Fisher (x sigma^2)
    ev_ntk = np.sort(np.linalg.eigvalsh(Ja @ Ja.T))[::-1][:20]          # NTK Gram matrix
    print(f"   top-20 eigenvalues of J^T J (Fisher, {Ja.shape[1]} parameters) vs J J^T (NTK Gram): "
          f"max relative difference {np.abs(ev_fim / ev_ntk - 1).max():.1e}")
    U, _ = top_subspace(Ja, 6)
    w, V = np.linalg.eigh(Ja @ Ja.T)
    print(f"   top-6 left singular subspace of J vs top-6 eigenvectors of the NTK Gram: largest principal angle "
          f"{angles(U, V[:, ::-1][:, :6])[0]:.1e} degrees")

    print("\nB. cell 5 criterion on untrained networks: largest principal angle (degrees) / mean cos^2 between the")
    print("   top-k prediction-space hard subspaces; pairs of different architectures, 3 initialisation seeds each")
    rng = np.random.default_rng(0)
    names = list(ARCHS)
    for k in K_LIST:
        sub = {key: top_subspace(v, k)[0] for key, v in J.items()}
        cross = [angles(sub[a, s], sub[b, t]) for i, a in enumerate(names) for b in names[i + 1:]
                 for s in range(3) for t in range(3)]
        same = [angles(sub[a, s], sub[a, t]) for a in names for s in range(3) for t in range(s + 1, 3)]
        null = []
        for _ in range(200):
            A = np.linalg.qr(rng.standard_normal((3 * n, k)))[0]
            B = np.linalg.qr(rng.standard_normal((3 * n, k)))[0]
            null.append(angles(A, B))
        f = lambda v: (np.median([x[0] for x in v]), np.max([x[0] for x in v]), np.median([x[1] for x in v]))
        print(f"   k = {k:2d}: different architectures  angle median {f(cross)[0]:5.1f} max {f(cross)[1]:5.1f}  cos^2 {f(cross)[2]:.3f}"
              f" | same architecture, other seed  angle median {f(same)[0]:5.1f}  cos^2 {f(same)[2]:.3f}"
              f" | random subspaces  angle median {f(null)[0]:5.1f} min {np.min([x[0] for x in null]):5.1f}  cos^2 {f(null)[2]:.3f}")

    print("\nC. share of the untrained top-k hard subspace inside low-order functions of the inputs")
    feats1 = np.stack([np.ones(n), X[:, 0], X[:, 1], X[:, 2]], 1)                      # 1, cos, sin, omega
    quad = [X[:, i] * X[:, j] for i in range(3) for j in range(i, 3)]
    feats2 = np.concatenate([feats1, np.stack(quad, 1)], 1)                            # + all quadratic monomials
    for label, F in (("affine in (cos, sin, omega), per output", feats1), ("quadratic, per output", feats2)):
        D = np.linalg.qr(np.kron(F, np.eye(3)))[0]                                     # rows (point, channel)
        out = []
        for k in K_LIST:
            share = [np.linalg.norm(D.T @ top_subspace(v, k)[0]) ** 2 / k for v in J.values()]
            out.append(f"k = {k}: {np.min(share):.3f}-{np.max(share):.3f}")
        print(f"   {label:42} ({D.shape[1]:2d} dims of {3 * n}): " + "   ".join(out))
    print("   (the Jacobian of an untrained network does not involve the targets, so none of this depends on the dynamics)")

    print("\nD. cell 23: sample size at which each term of  d(omega)/dt = -sin(theta) - 2 gamma omega  is detected")
    print("   OLS of the one-step change of omega on (sin theta, omega), added noise sd 0.5 on the change per unit time;")
    print("   detected = |t| > 3 in at least 90 % of 200 replicates; formula: N_damping / N_conservative = Var(sin) / (4 gamma^2 Var(omega))")
    rng = np.random.default_rng(1)
    grid = [int(round(v)) for v in np.geomspace(8, 60000, 40)]
    for gamma in (0.05, 0.1, 0.173, 0.5):
        Xg, _, thg, omg, om1 = data(60000, gamma=gamma, seed=3)
        dw = (om1 - omg) / 0.1
        F = np.stack([np.sin(thg), omg], 1)
        thr = []
        for col in (0, 1):
            found = None
            for N in grid:
                hits = 0
                for _ in range(200):
                    idx = rng.integers(0, len(F), N)
                    y = dw[idx] + 0.5 * rng.standard_normal(N)
                    A = F[idx]
                    beta, res = np.linalg.lstsq(A, y, rcond=None)[:2]
                    s2 = ((y - A @ beta) ** 2).sum() / (N - 2)
                    se = np.sqrt(s2 * np.linalg.inv(A.T @ A)[col, col])
                    hits += abs(beta[col] / se) > 3
                if hits >= 180:
                    found = N
                    break
            thr.append(found)
        pred = np.sin(thg).var() / (4 * gamma ** 2 * omg.var())
        ratio = (thr[1] / thr[0]) if all(thr) else float("nan")
        print(f"   gamma = {gamma:5.3f}: N(conservative) = {thr[0]}, N(damping) = {thr[1]}, ratio {ratio:6.2f}, formula {pred:6.2f}"
              f"   -> {'conservative first' if ratio > 1 else 'damping first'}")


if __name__ == "__main__":
    main()
