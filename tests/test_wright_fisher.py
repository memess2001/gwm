"""Wright-Fisher ground truth: geodesic labels, Hodge character, invariants and noise, each checked independently."""
import itertools

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import zoo
from zoo.base import SPLIT_SEEDS
from zoo.wright_fisher import RPS_A, WrightFisher, theta_coords, theta_to_x

F = np.array([1.0, -0.3, -0.9])
M, MU = np.array([0.6, 0.1, 0.3]), 0.7
X0 = np.array([0.2, 0.5, 0.3])
CYC = np.roll(np.eye(3), 1, axis=1) - np.eye(3)


def ratio(path):
    """Relative residual of a rank-1 fit to the increments of a path [T+1, d] (sigma_2 / sigma_1)."""
    s = np.linalg.svd(np.diff(path, axis=0), compute_uv=False)
    return s[1] / s[0]


def ode(rhs, x0, t):
    return solve_ivp(lambda _, x: rhs(x), (0, t[-1]), x0, t_eval=t, method="DOP853", rtol=1e-13, atol=1e-15).y.T


FIELDS = {  # independent re-implementations of the vector fields
    "selection": lambda x: x * (F - x @ F),
    "migration": lambda x: MU * (M - x),
    "mixed": lambda x: x * (F - x @ F) + MU * (M - x),
    "mutation": lambda x: 0.5 * (1 / 3 - x),                       # uniform mutation, rate 0.5
    "rps": lambda x: x * (RPS_A @ x - x @ RPS_A @ x),
}
PARAMS = {"selection": {"f": F}, "migration": {"mu": MU, "m": M}, "mixed": {"f": F, "mu": MU, "m": M},
          "mutation": {"u": 0.5}, "rps": {}}


@pytest.mark.parametrize("mech", list(FIELDS))
def test_flows_match_independent_ode_solver(mech):
    w = WrightFisher(x0=X0, schedule=[(mech, 5.0, PARAMS[mech])])
    tr = w.sample(0, 50, 1, "freq")
    assert np.abs(tr.state[0] - ode(FIELDS[mech], X0, tr.t)).max() < 1e-9
    v = w.vector_field(tr.state[0], mech, **PARAMS[mech])
    assert np.allclose(v, np.array([FIELDS[mech](x) for x in tr.state[0]]), atol=1e-14)
    tr = WrightFisher(x0=X0, mutation="cyclic", schedule=[("mutation", 5.0, {"u": 0.5})]).sample(0, 50, 1, "freq")
    assert np.abs(tr.state[0] - ode(lambda x: 0.5 * x @ CYC, X0, tr.t)).max() < 1e-9


def test_selection_is_e_geodesic_and_migration_is_m_geodesic():
    sel = WrightFisher(x0=X0, schedule=[("selection", 3.0, {"f": F})]).sample(0, 30, 1, "freq")
    mig = WrightFisher(x0=X0, schedule=[("migration", 3.0, {"mu": MU, "m": M})]).sample(0, 30, 1, "freq")
    assert ratio(theta_coords(sel.state[0])) < 1e-9 < 1e-4 < ratio(sel.state[0])     # straight in theta only
    assert ratio(mig.state[0]) < 1e-9 < 1e-4 < ratio(theta_coords(mig.state[0]))      # straight in eta only
    d = np.diff(theta_coords(sel.state[0]), axis=0)
    assert np.allclose(d, 0.1 * (F[:2] - F[2]), atol=1e-12)                           # theta(t) = theta0 + t (f_i - f_K)
    assert np.allclose(np.cross(np.diff(mig.state[0], axis=0), M - X0), 0, atol=1e-14)  # heading to the pool m
    # uniform mutation = migration toward 1/K (m-geodesic); cyclic (parent-dependent) mutation = neither
    uni = WrightFisher(schedule=[("mutation", 3.0, {})]).sample(1, 30, 50, "freq").state
    cyc = WrightFisher(schedule=[("mutation", 3.0, {})], mutation="cyclic").sample(1, 30, 50, "freq").state
    assert max(ratio(p) for p in uni) < 1e-9 and min(ratio(p) for p in cyc) > 1e-4
    assert min(ratio(theta_coords(p)) for p in cyc) > 1e-4


def test_random_segments_geodesic_residuals():
    w = WrightFisher()
    for sch, straight_theta, straight_eta in (("selection_only", True, False), ("migration_only", False, True),
                                              ("mixed_only", False, False), ("mutation_selection", False, False)):
        tr = w.sample(2, 20, 200, "freq", schedule=sch)
        rt = np.array([ratio(p) for p in tr.info["theta"]])
        re = np.array([ratio(p) for p in tr.info["eta"]])
        assert (rt < 1e-9).all() if straight_theta else (rt > 1e-6).all()
        assert (re < 1e-9).all() if straight_eta else (re > 1e-6).all()


def test_alternating_preset_labels_per_step():
    w = WrightFisher(seg_steps=15)
    tr = w.sample(3, 150, 40, "freq")
    mech, alpha, seg = tr.info["mechanism"], tr.info["alpha"], tr.info["segment_id"]
    assert set(np.unique(mech)) == {"selection", "migration"}
    assert np.array_equal(alpha, np.where(mech == "selection", 1.0, -1.0))
    for i in range(40):
        ids = np.unique(seg[i])
        assert len(ids) == 10
        for j in ids:
            steps = np.flatnonzero(seg[i] == j)
            assert len(steps) == 15 and len(set(mech[i, steps])) == 1
            path = slice(steps[0], steps[-1] + 2)                    # x_t .. x_{t+L}
            coords = tr.info["theta"][i, path] if mech[i, steps[0]] == "selection" else tr.state[i, path]
            assert ratio(coords) < 1e-9
            if j:
                assert mech[i, steps[0]] != mech[i, steps[0] - 1]   # mechanisms alternate
            assert (np.diff(tr.info["potential"][i, steps]) > -1e-12).all()   # gradient ascent of V


def chart_curl_div(v, x, h=1e-5):
    """Curl of the Fisher-dual 1-form and Shahshahani divergence of v at simplex points x [n, 3] (chart x1, x2)."""
    e = [np.array([1.0, 0.0, -1.0]), np.array([0.0, 1.0, -1.0])]

    def omega(z):                     # omega_a = g(v, e_a) = sum_i v_i (e_a)_i / x_i
        vz = v(z)
        return np.stack([vz[:, 0] / z[:, 0] - vz[:, 2] / z[:, 2], vz[:, 1] / z[:, 1] - vz[:, 2] / z[:, 2]], 1)

    def flux(z):                      # sqrt(det g) v^a, det g = 1 / (x1 x2 x3)
        return v(z)[:, :2] / np.sqrt(z.prod(1))[:, None]

    d = lambda fn, a, c: (fn(x + h * e[a])[:, c] - fn(x - h * e[a])[:, c]) / (2 * h)
    return d(omega, 0, 1) - d(omega, 1, 0), (d(flux, 0, 0) + d(flux, 1, 1)) * np.sqrt(x.prod(1))


def test_hodge_character_in_fisher_metric():
    w = WrightFisher()
    x = np.random.default_rng(0).dirichlet(np.ones(3) * 4, 300)
    gt = w.ground_truth()
    fields = {m: (lambda m: lambda z: w.vector_field(z, m, **PARAMS[m]))(m) for m in PARAMS}
    for m, vf in fields.items():
        curl, div = chart_curl_div(vf, x)
        if gt["hodge"][m] == "gradient":
            assert np.abs(curl).max() < 1e-6, m                       # closed dual 1-form
        else:
            assert gt["hodge"][m] == "rotational" == {"rps": "rotational"}.get(m)
            assert np.abs(curl).min() > 1.0 and np.abs(div).max() < 1e-6   # pure rotation, Fisher-div-free
    assert gt["hodge"]["migration"] == gt["hodge"]["mixed"] == "gradient"
    cyc = WrightFisher(mutation="cyclic")
    curl, _ = chart_curl_div(lambda z: cyc.vector_field(z, "mutation", u=0.5), x)
    assert np.abs(curl).min() > 0.1 and cyc.ground_truth()["hodge"]["mutation"] == "mixed"
    assert cyc.ground_truth()["alpha"]["mutation"] is None and gt["alpha"]["mutation"] == -1


def test_potentials_are_fisher_gradient_potentials():
    """dV(w) = g(v, w) for all tangent w <=> v = grad_g V: checked by finite differences of V."""
    rng = np.random.default_rng(1)
    w = WrightFisher()
    x = rng.dirichlet(np.ones(3) * 3, 200)
    t = rng.standard_normal((200, 3))
    t -= t.mean(1, keepdims=True)                                  # tangent vectors (sum 0)
    eps = 1e-6
    for m in ("selection", "migration", "mixed", "mutation", "mutation_selection"):
        p = {**PARAMS.get(m, {}), **({"f": F, "u": 0.5} if m == "mutation_selection" else {})}
        V = lambda z: w.potential(z, m, **p)
        dV = (V(x + eps * t) - V(x - eps * t)) / (2 * eps)
        g = (w.vector_field(x, m, **p) * t / x).sum(1)
        assert np.allclose(dV, g, atol=1e-6), m
    assert np.isnan(w.potential(x, "rps")).all()
    assert np.allclose(w.potential(x, "selection", f=F), x @ F)    # mean fitness


def test_rps_invariant_and_closed_orbits():
    w = WrightFisher(schedule="rps")
    tr = w.sample(4, 1000, 64, "freq")
    H = (np.log(tr.state) / 3).sum(-1)                            # sum x*_i log x_i with x* = 1/3 (independent)
    assert np.abs(H - H[:, :1]).max() < 1e-6
    assert np.allclose(H, tr.info["rps_invariant"])
    assert np.allclose(w.ground_truth()["rps"]["x_star"], 1 / 3)
    ang = np.unwrap(np.arctan2(*(theta_coords(tr.state) - 0.0).T[::-1]), axis=0)   # winds around x* (theta* = 0)
    assert (np.abs(ang[-1] - ang[0]) > 2 * np.pi).all()
    assert (tr.info["alpha"] != tr.info["alpha"]).all() and (tr.info["hodge"] == "rotational").all()


def test_finite_population_drift_variance():
    N, n = 200, 40000
    w = WrightFisher(pop_size=N, x0=X0, schedule=[("selection", 0.1, {"f": F})])
    x1 = w.sample(0, 1, n, "freq").state[:, 1]
    p = X0 * np.exp(0.1 * F)
    p /= p.sum()                                                  # independent closed form of Phi_dt(x0)
    assert np.allclose(x1.mean(0), p, atol=5 * np.sqrt(p * (1 - p) / N / n).max())
    assert np.allclose(np.cov(x1.T), (np.diag(p) - np.outer(p, p)) / N, rtol=0.05, atol=2e-6)
    assert np.allclose(x1 * N, np.round(x1 * N))                    # integer counts


def test_count_observation_noise():
    w = WrightFisher(x0=X0, n_obs=500, schedule=[("selection", 10.0, {"f": np.zeros(3)})])
    tr = w.sample(1, 100, 200, "counts")
    o = tr.obs.reshape(-1, 3).astype(np.float64)
    assert np.allclose(tr.state, X0) and np.allclose(o.mean(0), X0, atol=2e-3)
    assert np.allclose(o.var(0), X0 * (1 - X0) / 500, rtol=0.05)


def test_coordinates_conventions_and_schedules():
    gt = WrightFisher().ground_truth()
    assert "Koide" in gt["convention"] and gt["alpha"] == {"selection": 1, "migration": -1, "mutation": -1,
                                                           "mixed": None, "mutation_selection": None, "rps": None}
    x = np.random.default_rng(2).dirichlet(np.ones(3), 10)
    assert np.allclose(theta_to_x(theta_coords(x)), x) and np.allclose(gt["coords"]["eta"](x), x)
    tr = WrightFisher(x0=X0).sample(0, 30, 2, "logratio", schedule=[("selection", 1.0, {"f": F}), ("migration", 2.0)])
    assert np.allclose(tr.obs, theta_coords(tr.state), atol=1e-6) and tr.obs.shape == (2, 31, 2)
    assert np.allclose(tr.info["fitness"][:, :10], F) and np.isnan(tr.info["fitness"][:, 10:]).all()
    assert not np.allclose(tr.info["pool"][0, 10:], tr.info["pool"][1, 10:])   # missing params drawn per trajectory
    w = WrightFisher()
    with pytest.raises(ValueError):
        w.sample(0, 10, 1, "freq", schedule=[("selection", 0.55)])          # not a multiple of dt
    with pytest.raises(ValueError):
        w.sample(0, 10, 1, "freq", schedule=[("selection", 0.5)])           # shorter than the horizon
    with pytest.raises(ValueError):
        w.sample(0, 10, 1, "freq", schedule=[("selection", 1.0, {"mu": 1.0})])
    with pytest.raises(ValueError):
        WrightFisher(K=4).sample(0, 10, 1, "freq", schedule="rps")          # RPS needs K = 3 or rps_A
    assert WrightFisher(K=5).sample(0, 10, 3, "freq").obs.shape == (3, 11, 5)


def test_seed_determinism_and_disjoint_splits():
    w = zoo.make("wright_fisher", pop_size=100)
    a, b = w.sample(5, 40, 8, "counts"), w.sample(5, 40, 8, "counts")
    assert np.array_equal(a.obs, b.obs) and np.array_equal(a.info["mechanism"], b.info["mechanism"])
    x0 = {s: w.sample(zoo.split_seed(s, 3), 0, 64, "freq").state[:, 0] for s in SPLIT_SEEDS}
    for s1, s2 in itertools.combinations(x0, 2):
        assert not (x0[s1][:, None] == x0[s2][None]).all(-1).any()
    for m in w.obs_modes:
        assert w.sample(0, 2, 2, m).obs.shape[2:] == w.spec(m)["obs_shape"]
