"""Pendulum ground truth, each claim checked by an independent numerical method."""
import gudhi
import numpy as np
import pytest
from ripser import ripser
from scipy.integrate import cumulative_simpson, solve_ivp

import zoo
from zoo.base import SPLIT_SEEDS
from zoo.pendulum import Pendulum, _region_betti, render


def emb(state):
    """Ground-truth embedding (cos theta, sin theta, omega) used for instrument calibration."""
    s = np.asarray(state).reshape(-1, 2)
    return np.stack([np.cos(s[:, 0]), np.sin(s[:, 0]), s[:, 1]], 1)


def ripser_betti(X, thresh=0.75, n_perm=500):
    dgms = ripser(X, maxdim=1, n_perm=n_perm)["dgms"]
    return [int(((d[:, 1] - d[:, 0]) > thresh).sum()) for d in dgms]


def grid_betti(lo, hi, W, w0, n=360):
    """Betti numbers of the region from a periodic cubical complex on a (theta, omega) grid (gudhi)."""
    lo, hi = (-np.inf if lo is None else lo), (np.inf if hi is None else hi)
    ext = 1.05 * min(W, np.sqrt(max(2 * (min(hi, 0.5 * W * W + w0 ** 2) + w0 ** 2), 1e-6)))
    th = (np.arange(n) + 0.5) * 2 * np.pi / n - np.pi
    om = np.linspace(-ext, ext, n)
    T, O = np.meshgrid(th, om, indexing="ij")
    E = 0.5 * O ** 2 - w0 ** 2 * np.cos(T)
    inside = (np.abs(O) <= W) & (E >= lo) & (E <= hi)
    cc = gudhi.PeriodicCubicalComplex(top_dimensional_cells=np.where(inside, 0.0, 1.0), periodic_dimensions=[True, False])
    cc.compute_persistence()
    return list(cc.persistent_betti_numbers(0.0, 0.0)[:2])


# ------------------------------------------------------------------ conservation, damping, divergence
def test_energy_drift_conservative():
    w = Pendulum()
    tr = w.sample(seed=0, horizon=1000, n_traj=200, obs_mode="state")
    th, om = tr.state[..., 0], tr.state[..., 1]
    E = 0.5 * om ** 2 - np.cos(th)                          # recomputed independently of info["energy"]
    rel = np.abs(E - E[:, :1]).max(1) / (E[:, 0] + 1.0)    # relative to the energy above the minimum -omega0^2
    assert rel.max() < 1e-5
    assert np.allclose(E, tr.info["energy"])
    y4 = Pendulum(integrator="yoshida4").sample(seed=0, horizon=300, n_traj=50, obs_mode="state").info["energy"]
    assert (np.abs(y4 - y4[:, :1]).max(1) / (y4[:, 0] + 1.0)).max() < 1e-9      # optional 4th-order symplectic
    gt = w.ground_truth()
    assert gt["conserved"] == ["energy"] and gt["symplectic"] and gt["divergence"] == 0


def test_energy_not_conserved_when_damped_or_actuated():
    assert Pendulum(gamma=0.1).ground_truth()["conserved"] == []
    gt = Pendulum(policy="random").ground_truth()
    assert gt["conserved"] == [] and gt["symplectic"] and gt["region_betti"] is None
    tr = Pendulum(policy="random", u_max=1.0).sample(1, 300, 16, "state")
    E = tr.info["energy"]
    assert np.abs(E - E[:, :1]).max() > 0.1


def test_verlet_and_rk4_match_independent_integrator():
    for gamma, tol in ((0.0, 1e-3), (0.1, 1e-7)):           # Verlet: O(h^2) phase error; RK4: O(h^4)
        w = Pendulum(gamma=gamma)
        tr = w.sample(1, 200, 6, "state")
        for i in range(6):
            y0 = [tr.info["theta_unwrapped"][i, 0], tr.state[i, 0, 1]]
            sol = solve_ivp(lambda t, y: [y[1], -np.sin(y[0]) - 2 * gamma * y[1]], (0, tr.t[-1]), y0,
                            t_eval=tr.t, method="DOP853", rtol=1e-12, atol=1e-12)
            assert np.abs(sol.y[0] - tr.info["theta_unwrapped"][i]).max() < tol
            assert np.abs(sol.y[1] - tr.state[i, :, 1]).max() < tol


def test_torque_is_zero_order_hold():
    w = Pendulum(policy="random", gamma=0.05, hold=7)
    tr = w.sample(2, 60, 3, "state")
    assert tr.actions.shape == (3, 60, 1) and tr.actions.dtype == np.float32
    assert np.abs(tr.actions).max() <= 1.0 and (np.diff(tr.actions[:, :7, 0], axis=1) == 0).all()
    for i in range(3):
        y = np.array([tr.info["theta_unwrapped"][i, 0], tr.state[i, 0, 1]])
        for k in range(60):
            u = float(tr.actions[i, k, 0])
            y = solve_ivp(lambda t, z: [z[1], -np.sin(z[0]) - 0.1 * z[1] + u], (0, 0.1), y,
                          method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
        assert abs(y[0] - tr.info["theta_unwrapped"][i, -1]) < 1e-7 and abs(y[1] - tr.state[i, -1, 1]) < 1e-7


def test_damped_energy_decay_matches_analytic_rate():
    gamma = 0.15
    tr = Pendulum(gamma=gamma, dt=0.02).sample(3, 1000, 32, "state")
    th, om = tr.info["theta_unwrapped"], tr.state[..., 1]
    E = 0.5 * om ** 2 - np.cos(th)
    lost = cumulative_simpson(-2 * gamma * om ** 2, x=tr.t, axis=1, initial=0.0)   # integral of dE/dt
    assert np.abs(E - E[:, :1] - lost).max() < 1e-6 * (E[:, 0] + 1).max()
    assert (np.diff(E, axis=1) <= 1e-12).all()                                      # monotone decay


def test_divergence_by_finite_differences():
    rng = np.random.default_rng(0)
    pts = rng.uniform([-np.pi, -3.0], [np.pi, 3.0], (400, 2))
    h = 1e-5
    for gamma in (0.0, 0.1, 0.37):
        gt = Pendulum(gamma=gamma).ground_truth()
        f = gt["vector_field"]
        div = ((f(pts + [h, 0])[:, 0] - f(pts - [h, 0])[:, 0]) + (f(pts + [0, h])[:, 1] - f(pts - [0, h])[:, 1])) / (2 * h)
        assert gt["divergence"] == -2 * gamma
        assert np.abs(div - gt["divergence"]).max() < 1e-8


def test_phase_volume_contracts_at_rate_two_gamma():
    """Liouville: det d(flow)/d(x0) = exp(div * t), via finite differences of the simulator itself."""
    rng = np.random.default_rng(1)
    for gamma in (0.0, 0.2):
        w, T, h = Pendulum(gamma=gamma), 50, 1e-6
        x0 = rng.uniform([-2.0, -1.5], [2.0, 1.5], (10, 2))
        J = np.stack([(w.rollout(x0 + h * e, T)[:, -1] - w.rollout(x0 - h * e, T)[:, -1]) / (2 * h)
                      for e in np.eye(2)], -1)
        assert np.allclose(np.linalg.det(J), np.exp(-2 * gamma * T * w.cfg["dt"]), rtol=1e-4)


# ------------------------------------------------------------------ topology of the sampled region
BRANCHES = [  # (E_lo, E_hi, omega_max, expected [b0, b1]) - one generic case per branch of _region_betti
    (None, None, 2.5, [1, 1]), (None, 0.0, 2.5, [1, 0]), (1.5, None, 2.5, [2, 2]), (2.5, None, 2.5, [2, 0]),
    (-0.5, None, 2.5, [1, 2]), (-0.5, 0.5, 2.5, [1, 1]), (-0.5, None, 0.8, [1, 0]), (-0.5, 0.5, 0.8, [2, 0]),
    (None, -1.5, 2.5, [0, 0]),
]


@pytest.mark.parametrize("lo,hi,W,expected", BRANCHES)
def test_region_betti_branches_against_cubical_homology(lo, hi, W, expected):
    got = _region_betti(-np.inf if lo is None else lo, np.inf if hi is None else hi, W, 1.0)
    assert list(got[:2]) == expected == grid_betti(lo, hi, W, 1.0)


def test_region_betti_random_configs_against_cubical_homology():
    rng = np.random.default_rng(7)
    checked = 0
    while checked < 10:
        w0, W = rng.uniform(0.6, 1.6), rng.uniform(0.5, 3.5)
        w2, Emax = w0 * w0, 0.5 * W * W + w0 * w0
        lo = None if rng.random() < 0.3 else rng.uniform(-w2, Emax)
        hi = None if rng.random() < 0.3 else rng.uniform(-w2 if lo is None else lo, Emax + 0.5)
        L, Hh = (-np.inf if lo is None else lo), (np.inf if hi is None else hi)
        cs = np.array([-Hh / w2, -L / w2, (0.5 * W * W - L) / w2])
        if np.isfinite(cs).any() and (np.abs(np.abs(cs[np.isfinite(cs)]) - 1) < 0.08).any():
            continue                                              # skip near-degenerate (non-generic) cases
        if Hh - L < 0.25 * w2:
            continue                                              # too thin to resolve on the grid
        for W_ in (W, np.inf):                                    # initial region and invariant energy band
            a, b = (L, Hh) if W_ == W else (max(L, -w2), min(Hh, Emax))
            ref = grid_betti(a, b, W_ if np.isfinite(W_) else 1e9, w0)
            assert list(_region_betti(a, b, W_, w2)[:2]) == ref, (lo, hi, W, w0, W_)
        checked += 1


@pytest.mark.parametrize("cfg", [dict(), dict(E_hi=0.0), dict(E_lo=1.5), dict(E_lo=-0.5, E_hi=0.5),
                                 dict(E_lo=-0.5), dict(E_lo=2.5)])
def test_region_betti_matches_ripser_on_true_state_embeddings(cfg):
    w = Pendulum(**cfg)
    gt = w.ground_truth()
    traj = w.sample(5, 120, 120, "state").state                   # visited states (orbit hull)
    assert ripser_betti(emb(traj)) == gt["region_betti"]
    init = w.sample(6, 0, 2500, "state").state                   # i.i.d. initial states
    assert ripser_betti(emb(init)) == gt["init_region_betti"]


def test_region_regimes_and_wrapping():
    gt = lambda **c: Pendulum(**c).ground_truth()
    assert gt()["region_betti"] == [1, 1] and gt()["region_b1_wrapping"] == 1
    assert gt(E_hi=0.5)["region_betti"] == [1, 0]                # libration only: disk
    assert gt(E_lo=1.5)["region_betti"] == [2, 2] and gt(E_lo=1.5)["region_b1_wrapping"] == 2
    ring = gt(E_lo=-0.5, E_hi=0.5)                               # libration ring: H1 = 1 but contractible
    assert ring["region_betti"] == [1, 1] and ring["region_b1_wrapping"] == 0
    assert gt(E_lo=1.0)["region_betti"] == [1, 2]                # E_lo = E_sep: bands touch at the saddle
    assert gt(gamma=0.1)["region_betti"] is None and gt(gamma=0.1)["init_region_betti"] == [1, 1]
    with pytest.raises(ValueError):
        Pendulum(E_hi=-1.5).sample(0, 1, 1, "state")


def test_energy_window_rejection_sampling():
    tr = Pendulum(E_lo=1.2, E_hi=2.0).sample(0, 0, 500, "state")
    E = tr.info["energy"][:, 0]
    assert (E >= 1.2).all() and (E <= 2.0).all() and tr.info["is_rotation"].all()
    assert (np.abs(tr.state[:, 0, 1]) <= 2.5).all()


# ------------------------------------------------------------------ renderer
def test_renderer_layout():
    img = render(np.array([0.0, np.pi / 2]), quantize=False)            # bob centre at (31.5, 31.5) + 22 * dir
    for frame, (row, col) in zip(img, [(53.5, 31.5), (31.5, 53.5)]):  # theta = 0 down, theta = pi/2 right
        r, c = np.unravel_index(np.argmax(frame), frame.shape)
        assert abs(r - row) == 0.5 and abs(c - col) == 0.5
    assert img.min() >= 0 and img.max() <= 1 and img[0, 0, 0] == 0 and img.max() > 0.9


def test_renderer_smooth_in_theta_including_wrap():
    n = 2048
    th = np.linspace(-np.pi, np.pi, n + 1)                       # both ends are the same physical angle
    d = 2 * np.pi / n
    I = render(th, quantize=False).reshape(n + 1, -1).astype(np.float64)
    assert np.abs(I[0] - I[-1]).max() < 1e-5                     # periodic: no jump at +-pi
    steps = np.linalg.norm(np.diff(I, axis=0), axis=1)
    mid = 0.5 * (th[1:] + th[:-1])
    h = 1e-4
    speed = np.linalg.norm(render(mid + h, quantize=False).reshape(n, -1).astype(np.float64)
                           - render(mid - h, quantize=False).reshape(n, -1), axis=1) / (2 * h)
    assert np.abs(steps / (speed * d) - 1).max() < 0.02          # distance ~ linear in dtheta everywhere
    for t0 in (0.3, np.pi - 1e-3, -np.pi):                       # linearity for small dtheta, also across the wrap
        e = [np.linalg.norm(render(t0 + s, quantize=False).astype(np.float64) - render(t0, quantize=False))
             for s in (1e-3, 2e-3, 4e-3)]
        assert abs(e[1] / e[0] - 2) < 0.02 and abs(e[2] / e[1] - 2) < 0.02
    q = render(th).reshape(n + 1, -1).astype(int)                # uint8: no jump beyond quantisation
    assert np.abs(np.diff(q, axis=0)).max() <= np.ceil(255 * np.abs(np.diff(I, axis=0)).max()) + 1


def test_renderer_injective():
    n = 2048
    I = render(np.linspace(-np.pi, np.pi, n, endpoint=False)).reshape(n, -1).astype(np.float64)
    sq = (I * I).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2 * I @ I.T                 # exact: integer arithmetic in float64
    np.fill_diagonal(D2, np.inf)
    assert D2.min() > 0


def test_pixel_noise_and_obs_modes():
    w = Pendulum(pixel_noise=0.05)
    a, b = w.sample(0, 5, 3, "pixels64"), w.sample(0, 5, 3, "pixels64")
    assert a.obs.dtype == np.uint8 and a.obs.shape == (3, 6, 64, 64) and np.array_equal(a.obs, b.obs)
    clean = render(a.state[..., 0]).astype(float)
    mid = (clean > 60) & (clean < 195)                                 # away from clipping at 0 and 255
    assert 0.045 < np.std(a.obs[mid] - clean[mid]) / 255 < 0.055
    raw = Pendulum().sample(1, 50, 4, "angle_raw").obs
    assert raw.shape == (4, 51, 1) and raw.min() >= -np.pi and raw.max() < np.pi
    cs = Pendulum().sample(1, 50, 4, "angle_cs").obs
    assert np.allclose((cs ** 2).sum(-1), 1, atol=1e-6)
    for m in w.obs_modes:
        assert w.sample(0, 2, 2, m).obs.shape[2:] == w.spec(m)["obs_shape"]


# ------------------------------------------------------------------ determinism, splits, registry
def test_seed_determinism_and_disjoint_splits():
    w = zoo.make("pendulum", policy="random", gamma=0.05)
    a, b = w.sample(11, 20, 8, "angle_cs"), w.sample(11, 20, 8, "angle_cs")
    assert np.array_equal(a.obs, b.obs) and np.array_equal(a.actions, b.actions)
    init = {s: w.sample(zoo.split_seed(s, 0), 0, 64, "state").state[:, 0] for s in SPLIT_SEEDS}
    names = list(init)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (init[names[i]][:, None, :] == init[names[j]][None, :, :]).all(-1).any()
    bases = sorted(SPLIT_SEEDS.values())
    assert all(b2 - b1 >= 1_000_000 for b1, b2 in zip(bases, bases[1:]))
    with pytest.raises(ValueError):
        zoo.split_seed("train", 1_000_000)


def test_registry():
    assert {"pendulum", "toggle_grid", "wright_fisher"} <= set(zoo.list_worlds())
    w = zoo.make("pendulum", gamma=0.2, omega0=1.5)
    assert isinstance(w, Pendulum) and w.ground_truth()["divergence"] == -0.4
    assert w.ground_truth()["separatrix_energy"] == pytest.approx(2.25)
    with pytest.raises(KeyError):
        zoo.make("no_such_world")
    with pytest.raises(TypeError):
        zoo.make("pendulum", gama=0.1)
    tr = w.sample(0, 3, 2, "state")
    assert w.ground_truth(**tr.meta["params"])["divergence"] == -0.4
