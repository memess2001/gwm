"""Check 3: SECOND-ORDER (ballistic) point mass, state (x, y, vx, vy), no action, speed 1, dt = 0.25, k = 4.
Observation = position (flat blob rendering => g_obs = P^T P, rank 2 in the 4-D state space).
C_k = sum_{j=0..k} J_j^T J_j, J_j = d pos_j / d state.  Reference: the free-flight C_k (constant in these coords).
Readouts: generalized eigenvalues of (C_k, C_free): FTLE = 0.5 log lam_max, COMP = 0.5 log(1/lam_min); RES baseline.
Contact: 'reflect' (elastic, v_n -> -v_n) and 'inelastic' (v_n -> 0, frictionless slide).
"""
import numpy as np
from scipy.linalg import eigh
from maze import walls, step2, WALL_TOP

K_STEPS, DT, SPEED, B = 4, 0.25, 1.0, 1.0
W = walls(thick=False)


def traj(S, mode):
    out = [S[:, :2]]
    for _ in range(K_STEPS):
        S = step2(S, DT, mode, W)
        out.append(S[:, :2])
    return out


def cg(S, mode, h):
    e = np.eye(4)
    plus = [traj(S + h * e[i], mode) for i in range(4)]
    minus = [traj(S - h * e[i], mode) for i in range(4)]
    C = np.zeros((len(S), 4, 4))
    for j in range(K_STEPS + 1):
        J = np.stack([(plus[i][j] - minus[i][j]) / (2 * h) for i in range(4)], -1)   # [n, 2, 4]
        C += np.einsum("noi,noj->nij", J, J)
    free_end = S[:, :2] + K_STEPS * DT * S[:, 2:]
    res = np.linalg.norm(traj(S, mode)[-1] - free_end, axis=1)
    return C, res


# free-flight reference
Cf = np.zeros((4, 4))
for j in range(K_STEPS + 1):
    Jf = np.hstack([np.eye(2), j * DT * np.eye(2)])
    Cf += Jf.T @ Jf

gs = 0.05
xs = np.arange(gs / 2 + 1e-3, 10, gs)
X, Y = np.meshgrid(xs, xs)
dseg = np.hypot(np.abs(X - 5.0), np.maximum(Y - WALL_TOP, 0)); dout = np.minimum.reduce([X, 10 - X, Y, 10 - Y])
Z = dict(face=(np.abs(X - 5.0) < B) & (Y > B) & (Y < WALL_TOP - B),
         corner=np.hypot(X - 5.0, Y - WALL_TOP) < B, outer=(dout < B) & (dseg > B), far=(dout > B) & (dseg > B))
band = (dout < B) | (dseg < B)
angles = np.arange(8) * np.pi / 4

for h, lab in ((1e-4, "a.e. Jacobian (FD h=1e-4)"), (0.025, "coarse FD h=0.025 (discontinuity detector)")):
    print(f"\n===== second-order point mass, {lab}; k={K_STEPS}, step={SPEED*DT}, band B={B} =====")
    for mode in ("reflect", "inelastic"):
        agg = {r: np.full(X.shape, -np.inf) for r in ("FTLE", "COMP", "RES")}
        for a in angles:
            V = SPEED * np.array([np.cos(a), np.sin(a)])
            S = np.concatenate([np.stack([X.ravel(), Y.ravel()], 1), np.repeat(V[None], X.size, 0)], 1)
            C, res = cg(S, mode, h)
            Li = np.linalg.inv(np.linalg.cholesky(Cf))
            lam = np.linalg.eigvalsh(Li @ C @ Li.T)
            r = dict(FTLE=0.5 * np.log(lam[:, -1]).reshape(X.shape),
                     COMP=0.5 * np.log(1 / np.maximum(lam[:, 0], 1e-12)).reshape(X.shape), RES=res.reshape(X.shape))
            if a in (0.0, np.pi / 4):
                print(f"  [{mode}] v angle {np.degrees(a):.0f}: " + "  ".join(
                    f"{k}: face mean {r[k][Z['face']].mean():.3f} max {r[k][Z['face']].max():.2g}, corner max {r[k][Z['corner']].max():.2g}"
                    for k in r))
            for k in agg:
                agg[k] = np.maximum(agg[k], r[k])
        print(f"  [{mode}] max over 8 velocity directions -> coverage / IoU with all-wall band:")
        for k, A in agg.items():
            lit = A > (0.01 if k == "RES" else 0.05)
            iou = (lit & band).sum() / max((lit | band).sum(), 1)
            print(f"     {k:>4}: " + "  ".join(f"{z} {lit[m].mean():.2f}" for z, m in Z.items()) + f"   IoU {iou:.2f}")
