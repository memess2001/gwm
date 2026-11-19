"""Independent (adversarial) checks of the zoo ground truths.

Written by a second author from hand derivations, deliberately avoiding the builder's own helpers and methods:
topology by Euler characteristic + double cover (not gudhi / ripser), Hodge character by line integrals and
flow-map Jacobians (not the chart curl), toggle-grid dynamics by a simulator parsed from the ASCII layout, one-step
re-simulation for the obs / action / label alignment, and cross-world determinism, split and leakage checks.
Persistent homology / probes are only ever computed on ground-truth states here (instrument hygiene).
"""
import itertools

import numpy as np
import pytest
from scipy import ndimage
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import zoo
from zoo.base import SPLIT_SEEDS
from zoo.pendulum import Pendulum, _region_betti, render, wrap
from zoo.toggle_grid import ToggleGrid
from zoo.wright_fisher import RPS_A, WrightFisher, replicator_hodge, theta_coords

F32_PI = np.float32(np.pi)


# ================================================================== pendulum
def test_pendulum_obs_depend_on_theta_only_and_match_observe():
    w = Pendulum()
    rng = np.random.default_rng(0)
    s = rng.uniform([-np.pi, -3.0], [np.pi, 3.0], (64, 2))
    s2 = np.stack([s[:, 0] + 2 * np.pi * rng.integers(-2, 3, 64), rng.uniform(-3, 3, 64)], 1)   # other omega, winding
    for mode in ("angle_cs", "angle_raw", "pixels64"):
        assert np.array_equal(w.observe(s, mode), w.observe(s2, mode)), mode          # no omega / winding leak
    tr = w.sample(4, 25, 6, "pixels64")
    assert np.array_equal(tr.obs, render(tr.state[..., 0]))                           # nothing but the frame
    for mode in ("angle_cs", "angle_raw", "state"):
        assert np.array_equal(w.sample(4, 25, 6, mode).obs, w.observe(tr.state, mode))
    cs = w.sample(4, 25, 6, "angle_cs").obs
    assert np.allclose(cs, np.stack([np.cos(tr.state[..., 0]), np.sin(tr.state[..., 0])], -1), atol=1e-7)


def test_pendulum_action_t_takes_state_t_to_state_t_plus_1():
    w = Pendulum(policy="random", gamma=0.07, hold=3, u_max=1.0)
    tr = w.sample(8, 30, 3, "state")
    th, om, u = tr.info["theta_unwrapped"], tr.state[..., 1], tr.actions[..., 0].astype(np.float64)
    worst_ok, worst_shift = 0.0, 0.0
    for i in range(3):
        for t in range(30):
            for shift in (0, 1):
                uu = u[i, t - shift] if t >= shift else u[i, t + 1]
                y = solve_ivp(lambda _, z: [z[1], -np.sin(z[0]) - 0.14 * z[1] + uu], (0, 0.1), [th[i, t], om[i, t]],
                              method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
                e = max(abs(y[0] - th[i, t + 1]), abs(y[1] - om[i, t + 1]))
                worst_ok, worst_shift = (max(worst_ok, e), worst_shift) if shift == 0 else (worst_ok, max(worst_shift, e))
    assert worst_ok < 1e-8 and worst_shift > 1e-3          # aligned actions reproduce the step, shifted ones do not
    assert np.array_equal(w.rollout(tr.state[:, 0], 30, tr.actions), tr.state)           # rollout == sample


def test_pendulum_dt_units_and_energy_balance():
    a = Pendulum(dt=0.1).sample(5, 40, 8, "state")
    b = Pendulum(dt=0.05, n_sub=20).sample(5, 80, 8, "state")                # different substep h
    assert np.array_equal(a.state[:, 0], b.state[:, 0]) and np.allclose(a.t, b.t[::2])
    assert np.abs(a.info["theta_unwrapped"] - b.info["theta_unwrapped"][:, ::2]).max() < 2e-3    # O(h^2) Verlet
    c = Pendulum(dt=0.1, gamma=0.2).sample(5, 40, 8, "state")
    d = Pendulum(dt=0.05, gamma=0.2, n_sub=7).sample(5, 80, 8, "state")
    assert np.abs(c.info["theta_unwrapped"] - d.info["theta_unwrapped"][:, ::2]).max() < 1e-7   # RK4
    # energy balance with torque: E(t_{k+1}) - E(t_k) = int (u omega - 2 gamma omega^2) dt, from a fine reference
    g, w0 = 0.1, 1.3
    tr = Pendulum(policy="random", gamma=g, omega0=w0, hold=4, dt=0.05).sample(2, 40, 3, "state")
    for i in range(3):
        for k in range(40):
            uk = float(tr.actions[i, k, 0])
            sol = solve_ivp(lambda _, z: [z[1], -w0 ** 2 * np.sin(z[0]) - 2 * g * z[1] + uk, uk * z[1] - 2 * g * z[1] ** 2],
                            (0, 0.05), [tr.info["theta_unwrapped"][i, k], tr.state[i, k, 1], 0.0],
                            method="DOP853", rtol=1e-12, atol=1e-13).y[:, -1]
            dE = tr.info["energy"][i, k + 1] - tr.info["energy"][i, k]
            assert abs(dE - sol[2]) < 1e-8
    gt = Pendulum(gamma=g, omega0=w0).ground_truth()
    x = np.random.default_rng(1).uniform([-np.pi, -3], [np.pi, 3], (200, 2))
    grad_E = np.stack([w0 ** 2 * np.sin(x[:, 0]), x[:, 1]], 1)                # dE/d(theta, omega), by hand
    assert np.allclose((grad_E * gt["vector_field"](x)).sum(1), -2 * g * x[:, 1] ** 2)
    assert np.allclose(gt["energy_fn"](x), 0.5 * x[:, 1] ** 2 - w0 ** 2 * np.cos(x[:, 0]))


def test_pendulum_wrap_continuity():
    w = Pendulum(E_lo=1.6)                                                     # rotations: cross +-pi often
    tr = w.sample(2, 200, 16, "state")
    th, un = tr.state[..., 0], tr.info["theta_unwrapped"]
    assert (th >= -np.pi).all() and (th < np.pi).all() and np.allclose(wrap(un), th, atol=1e-12)
    assert np.abs(np.diff(un, axis=1)).max() <= np.abs(tr.state[..., 1]).max() * 0.1 * 1.01   # no jumps in the lift
    crossings = np.abs(np.diff(th, axis=1)) > np.pi
    assert crossings.sum() > 20
    raw = w.observe(tr.state, "angle_raw")[..., 0]
    assert raw.dtype == np.float32 and raw.min() >= -F32_PI and raw.max() < F32_PI
    assert np.array_equal(np.abs(np.diff(raw, axis=1)) > np.pi, crossings)          # raw jumps exactly at the wrap
    for mode in ("angle_cs", "pixels64"):                                            # embeddings do not jump there
        o = w.observe(tr.state, mode).astype(np.float64).reshape(16, 201, -1)
        step = np.linalg.norm(np.diff(o, axis=1), axis=-1)
        assert step[crossings].max() <= 1.05 * step[~crossings].max()
    edge = w.observe(np.array([[np.pi - 1e-9, 0.0], [-np.pi, 0.0], [np.pi, 0.0]]), "angle_raw").ravel()
    assert np.array_equal(edge, [-F32_PI] * 3)                                       # pi is reported as -pi


def test_renderer_geometry_mirror_symmetry_and_no_folding():
    rng = np.random.default_rng(2)
    th = rng.uniform(-np.pi, np.pi, 200)
    img = render(th, quantize=False).astype(np.float64)
    c = 31.5
    yy, xx = np.mgrid[0:64, 0:64]
    m = img.sum((1, 2))
    cx, cy = (img * (xx - c)).sum((1, 2)) / m, (img * (yy - c)).sum((1, 2)) / m
    dec = np.arctan2(cx, cy)                                    # bob = pivot + r (sin, cos) in (col, row): 0 = down
    assert np.abs(np.angle(np.exp(1j * (dec - th)))).max() < 0.02
    q = render(th)
    assert np.array_equal(render(-th), q[:, :, ::-1])            # theta -> -theta is exactly the left-right mirror
    flip = render(np.pi - th).astype(int) - q[:, ::-1, :].astype(int)
    assert np.abs(flip).max() <= 1                               # theta -> pi - theta: up-down mirror (float rounding)
    # pixel distance depends only on the angular separation (rotation equivariance up to the pixel grid) and is
    # bounded away from 0 for separated angles: the circle is embedded without folding (theta vs -theta included)
    for sep in (0.02, 0.3, 1.0, 2.0, np.pi):
        dist = np.linalg.norm((render(th + sep, quantize=False) - render(th, quantize=False)).reshape(200, -1), axis=1)
        assert dist.max() / dist.min() < 1.1, sep
    grid = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    I = render(grid, quantize=False).reshape(720, -1).astype(np.float64)
    sq = (I * I).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * I @ I.T, 0.0))
    ang = np.abs(np.angle(np.exp(1j * (grid[:, None] - grid[None]))))
    near = np.linalg.norm((render(0.05, quantize=False) - render(0.0, quantize=False)).ravel())
    assert D[ang >= 0.05 - 1e-9].min() >= 0.95 * near


# ---------------------------------------------------------------- region topology, by an independent method
def _components_periodic(mask):
    lab, n = ndimage.label(mask)
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in zip(lab[0], lab[-1]):                       # glue theta = -pi to theta = pi (axis 0 is periodic)
        if a and b:
            parent[find(a)] = find(b)
    return len({find(i) for i in range(1, n + 1)})


def _euler_char_periodic(m):
    """V - E + F of the union of closed pixels of a mask, periodic along axis 0."""
    pad = np.zeros((m.shape[0], m.shape[1] + 2), bool)
    pad[:, 1:-1] = m
    a, b = pad[:, :-1], pad[:, 1:]                           # pixels (i, j-1), (i, j) around vertex / edge (i, j)
    V = (a | b | np.roll(a, 1, 0) | np.roll(b, 1, 0)).sum()
    E = (a | b).sum() + (m | np.roll(m, 1, 0)).sum()
    return int(V - E + m.sum())


def _grid_topology(lo, hi, W, w2, n=720):
    """(b0, b1, b1_wrapping) of {|omega| <= W, lo <= E <= hi} on a periodic grid: b1 = b0 - chi (b2 = 0 for a
    2-d subcomplex of the cylinder); wrapping components lift to ONE component of the double cover, others to two."""
    if np.isfinite(W):
        ext = 1.05 * min(W, np.sqrt(max(2 * (min(hi, 0.5 * W * W + w2) + w2), 1e-9)))
    elif np.isfinite(hi):
        ext = 1.05 * np.sqrt(max(2 * (hi + w2), 1e-9))
    else:                                                    # unbounded band: cut far beyond every feature
        ext = 1.5 * np.sqrt(2 * (max(lo, -w2) + 2 * w2)) + 0.5
    T, O = np.meshgrid((np.arange(n) + 0.5) * 2 * np.pi / n - np.pi, np.linspace(-ext, ext, n), indexing="ij")
    E = 0.5 * O ** 2 - w2 * np.cos(T)
    inside = (np.abs(O) <= W) & (E >= lo) & (E <= hi)
    b0 = _components_periodic(inside)
    return b0, b0 - _euler_char_periodic(inside), 2 * b0 - _components_periodic(np.concatenate([inside, inside]))


INF = np.inf
EVERY_BRANCH = [  # (E_lo, E_hi, omega_max) -> one generic case per branch of _region_betti, omega0 = 1
    (-INF, INF, 2.5), (-INF, 0.0, 2.5), (-INF, -1.5, 2.5),                  # no lower cut: annulus, disk, empty
    (1.5, INF, 2.5), (2.5, INF, 2.5), (3.5, INF, 2.5), (5.0, INF, 2.5),     # rotations: 2 annuli, 2 disks x2, empty
    (-0.5, INF, 2.5), (-0.5, 0.5, 2.5), (-0.5, INF, 0.8), (-0.5, 0.5, 0.8),  # hole: pants, ring, tree, 2 pieces
    (1.5, 3.0, INF), (-0.9, 3.0, INF), (-0.5, 0.5, INF), (0.3, 0.9, 1.2), (1.05, 1.1, 2.0)]


@pytest.mark.parametrize("lo,hi,W", EVERY_BRANCH)
def test_region_betti_every_branch_by_euler_characteristic(lo, hi, W):
    assert tuple(_region_betti(lo, hi, W, 1.0)) == _grid_topology(lo, hi, W, 1.0)


def test_region_betti_by_euler_characteristic_and_double_cover():
    rng = np.random.default_rng(11)
    checked = 0
    while checked < 24:
        w0, W = rng.uniform(0.6, 1.6), rng.choice([rng.uniform(0.5, 3.5), np.inf])
        w2 = w0 * w0
        Emax = (0.5 * W * W if np.isfinite(W) else 6.0) + w2
        lo = -np.inf if rng.random() < 0.3 else rng.uniform(-w2, Emax)
        hi = np.inf if rng.random() < 0.3 else rng.uniform(max(lo, -w2), Emax + 0.5)
        cs = np.array([-hi / w2, -lo / w2, ((0.5 * W * W if np.isfinite(W) else np.inf) - lo) / w2])
        fin = np.isfinite(cs)
        if (np.abs(np.abs(cs[fin]) - 1) < 0.08).any() or min(hi, Emax) - max(lo, -w2) < 0.25 * w2:
            continue                                         # non-generic or too thin for the grid
        assert tuple(_region_betti(lo, hi, W, w2)) == _grid_topology(lo, hi, W, w2), (lo, hi, W, w0)
        checked += 1


def test_region_betti_at_thresholds_is_the_closed_set_limit():
    """At a degenerate threshold the closed region is the intersection of slightly larger regions, so its Betti
    numbers must equal those of {E_lo - eps <= E <= E_hi + eps} (and box W + eps) for small eps."""
    eps, w2 = 1e-7, 1.0
    for lo, hi, W in [(-np.inf, 1.0, 2.5), (1.0, np.inf, 2.5), (1.0, np.inf, 0.8), (1.0, 1.5, np.inf),
                      (-1.0, np.inf, 2.5), (-1.0, -1.0, 2.5), (2.125, np.inf, 2.5), (4.125, np.inf, 2.5),
                      (-0.5, 1.0, 2.5), (-0.5, 1.0, 0.8), (0.3, 0.3, 2.5), (2.0, 2.0, 2.5)]:
        assert _region_betti(lo, hi, W, w2) == _region_betti(lo - eps, hi + eps, W + eps, w2), (lo, hi, W)


def test_init_states_fill_exactly_the_region():
    for cfg in (dict(), dict(E_lo=-0.5, E_hi=0.5), dict(E_lo=2.5), dict(E_lo=-0.5, omega_max=0.8)):
        w = Pendulum(**cfg)
        s = w.sample(3, 0, 40000, "state").state[:, 0]
        E, W = 0.5 * s[:, 1] ** 2 - np.cos(s[:, 0]), w.cfg["omega_max"]
        lo, hi = w._window(w.cfg)
        assert (np.abs(s[:, 1]) <= W).all() and (E >= lo).all() and (E <= hi).all()
        # every coarse cell lying well inside R0 is populated (support = R0, uniform density)
        th_e, om_e = np.linspace(-np.pi, np.pi, 25), np.linspace(-W, W, 21)
        Hc, _, _ = np.histogram2d(s[:, 0], s[:, 1], [th_e, om_e])
        T, O = np.meshgrid(th_e, om_e, indexing="ij")
        Ec = 0.5 * O ** 2 - np.cos(T)
        inner = ((Ec >= lo) & (Ec <= hi))
        inner = inner[:-1, :-1] & inner[1:, :-1] & inner[:-1, 1:] & inner[1:, 1:]
        assert (Hc[inner] > 0).all() and inner.any()
        assert Hc[inner].std() / Hc[inner].mean() < 0.25            # uniform density on R0 (equal-area cells)


def test_phase_area_contracts_like_exp_minus_two_gamma_t_by_shoelace():
    for gamma in (0.0, 0.25):
        w = Pendulum(gamma=gamma, dt=0.05)
        s = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        x0 = np.stack([0.4 + 0.05 * np.cos(s), 0.9 + 0.05 * np.sin(s)], 1)
        T = 40
        x = w.sample(0, T, len(x0), "state", init=x0)
        th, om = x.info["theta_unwrapped"], x.state[..., 1]
        area = lambda a, b: 0.5 * abs((a * np.roll(b, -1) - np.roll(a, -1) * b).sum())
        A0, AT = area(th[:, 0], om[:, 0]), area(th[:, -1], om[:, -1])
        assert abs(AT / A0 - np.exp(-2 * gamma * T * 0.05)) < 1e-4


# ================================================================== toggle grid
def _ascii_sim(w):
    """Simulator built only from the ASCII layout and the documented rules (door i <-> switch i in list order)."""
    rows = w.ascii().split("\n")
    assert sorted((x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch == "D") == sorted(w.doors)
    assert {(x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch == "S"} == set(w.switches)
    dmap, smap = {d: i for i, d in enumerate(w.doors)}, {s: i for i, s in enumerate(w.switches)}

    def step(s, a):
        x, y, *b = s
        nx, ny = x + (0, 0, -1, 1)[a], y + (-1, 1, 0, 0)[a]
        ch = rows[ny][nx]
        if not (ch in ".S" or (ch == "D" and b[dmap[(nx, ny)]] == 1)):
            return (x, y, *b), None
        e = smap.get((nx, ny))
        if e is not None:
            b[e] ^= 1
        return (nx, ny, *b), e
    return step


LAYOUTS = [dict(), dict(n_doors=2), dict(n_doors=3, room_w=3, room_h=4), dict(n_doors=2, room_w=4, room_h=6),
           dict(n_doors=1, room_w=7, room_h=3)]


@pytest.mark.parametrize("cfg", LAYOUTS)
def test_toggle_grid_against_ascii_simulator_and_alignment(cfg):
    w = ToggleGrid(**cfg)
    step = _ascii_sim(w)
    for policy in ("random", "loops"):
        tr = w.sample(13, 150, 12, "pos", policy=policy, init_doors="random", start="any")
        for i in range(12):
            for t in range(150):
                nxt, e = step(tuple(int(v) for v in tr.state[i, t]), int(tr.actions[i, t]))
                assert nxt == tuple(tr.state[i, t + 1])                                  # a_t: s_t -> s_{t+1}
                assert tr.info["switch_entered"][i, t] == (-1 if e is None else e)
                assert tr.info["blocked"][i, t] == (nxt[:2] == tuple(tr.state[i, t, :2]))


@pytest.mark.parametrize("cfg", LAYOUTS[:3])
def test_toggle_grid_hidden_obs_identity_and_injectivity_via_sample(cfg):
    w = ToggleGrid(**cfg)
    tr = w.sample(21, 300, 24, "grid", init_doors="random", start="any")
    st, ob = tr.state.reshape(-1, 2 + w.k), tr.obs.reshape(len(tr.state.reshape(-1, 2 + w.k)), -1)
    pos = st[:, 1] * w.W + st[:, 0]
    varied = 0
    for p in np.unique(pos):
        sel = pos == p
        assert (ob[sel] == ob[sel][:1]).all()                    # same position, any door bits -> identical obs
        varied += len(np.unique(st[sel, 2:], axis=0)) > 1
    assert varied > 0.8 * len(np.unique(pos))                    # the identity check was not vacuous
    first = {p: ob[np.flatnonzero(pos == p)[0]] for p in np.unique(pos)}
    assert len(np.unique(np.stack(list(first.values())), axis=0)) == len(first)   # obs determine the position
    vis = w._with(door_visible=True).observe(tr.state, "grid").reshape(len(st), -1)
    assert len(np.unique(vis, axis=0)) == len(np.unique(st, axis=0))                # visible: obs <-> full state
    assert np.array_equal(w.observe(tr.state, "grid"), tr.obs)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_toggle_grid_reachable_bits_from_the_cell6_start(k):
    """BFS from start='room0', init_doors='closed' (the configuration cell 6 uses) with the ASCII simulator."""
    w = ToggleGrid(n_doors=k)
    step = _ascii_sim(w)
    start = {(int(x), int(y), *([0] * k)) for y, x in np.argwhere(w.room == 0)}
    seen, frontier = set(start), set(start)
    while frontier:
        frontier = {step(s, a)[0] for s in frontier for a in range(4)} - seen
        seen |= frontier
    for j, nbits in enumerate(w.ground_truth()["hidden_bits_by_room"]):
        configs = {s[2:] for s in seen if w.room[s[1], s[0]] == j}
        assert len(configs) == 2 ** nbits and all(all(c[:j]) for c in configs)


def test_toggle_grid_loop_holonomy_and_obs_closure():
    rng = np.random.default_rng(5)
    for k in (1, 2):
        hid, vis = ToggleGrid(n_doors=k), ToggleGrid(n_doors=k, door_visible=True)
        step = _ascii_sim(hid)
        odd = 0
        for trial in range(120):
            s0 = hid._initial(rng, 1, dict(hid.cfg, init_doors="random"))[0]
            sw = (None, True, False)[trial % 3]
            if sw is False and hid.switch_id[s0[1], s0[0]] >= 0:
                sw = None
            acts = hid.random_loop(s0, rng, length=(1, 15), switch=sw)
            s, n_in = tuple(int(v) for v in s0), np.zeros(k, int)
            for a in acts:
                s, e = step(s, int(a))
                if e is not None:
                    n_in[e] += 1
            assert s[:2] == tuple(s0[:2])                                            # position-closed
            st, ob = hid.rollout(s0, acts)
            assert np.array_equal(ob[0], ob[-1])                                     # hidden: obs-closed
            assert np.array_equal(np.array(s[2:]) ^ s0[2:], n_in % 2)               # holonomy = entry parity
            assert np.array_equal(hid.holonomy(s0, acts)["flips"], n_in % 2)
            _, ov = vis.rollout(s0, acts)
            if np.array_equal(ov[0], ov[-1]):
                assert not (n_in % 2).any()                                          # visible + obs-closed => 0
            assert vis.holonomy(s0, acts)["obs_closed"] == np.array_equal(ov[0], ov[-1])
            odd += bool((n_in % 2).any())
        assert odd > 15


@pytest.mark.parametrize("cfg", [dict(n_doors=2), dict(n_doors=1, room_w=3, room_h=3), dict(switches=[(1, 1)])])
def test_toggle_grid_plaquette_parity_by_independent_simulation(cfg):
    w = ToggleGrid(**cfg)
    step = _ascii_sim(w)
    cm = w.curvature_map()
    rng = np.random.default_rng(6)
    loops = [(cm["loop"], ((0, 0), (1, 0), (1, 1), (0, 1))), (cm["reverse"], ((0, 0), (0, 1), (1, 1), (1, 0)))]
    for (x, y), par in zip(cm["corners"], cm["parity"]):
        for acts, offs in loops:
            for r in range(4):                                                      # every start corner
                bits = tuple(int(b) for b in rng.integers(0, 2, w.k))
                s = (x + offs[r][0], y + offs[r][1], *bits)
                s1 = s
                for a in np.roll(acts, -r):
                    s1, _ = step(s1, int(a))
                assert s1[:2] == s[:2] and np.array_equal(np.array(s1[2:]) ^ bits, par)
        touches = [((sx - x) in (0, 1)) and ((sy - y) in (0, 1)) for sx, sy in w.switches]
        assert np.array_equal(par.astype(bool), touches)
        assert np.array_equal(cm["map"][y, x], par)


# ================================================================== Wright-Fisher
def _phi(x, mech, f, mu, pool, u, K, dt, Q):
    """One observation step by closed forms (selection, migration, mutation) or a high-accuracy ODE solve."""
    if mech == "selection":
        y = x * np.exp(dt * f)
        return y / y.sum()
    if mech == "migration":
        return pool + (x - pool) * np.exp(-mu * dt)
    if mech == "mutation":
        return x @ expm(u * dt * Q)

    def rhs(_, z):
        v = np.zeros(K)
        if mech in ("mixed", "mutation_selection"):
            v += z * (f - z @ f)
        if mech == "mixed":
            v += mu * (pool - z)
        if mech == "mutation_selection":
            v += u * (z @ Q)
        return v
    return solve_ivp(rhs, (0, dt), x, method="DOP853", rtol=1e-13, atol=1e-15).y[:, -1]


@pytest.mark.parametrize("K,schedule", [(3, "alternating"), (4, "mixed_only"), (3, "mutation_selection"),
                                        (5, [("migration", 0.7), ("mutation", 0.5), ("selection", 1.0)])])
def test_wf_labels_at_step_t_reproduce_step_t(K, schedule):
    w = WrightFisher(K=K, seg_steps=(2, 6))
    tr = w.sample(17, 22, 4, "freq", schedule=schedule)
    Q, I = np.ones((K, K)) / K - np.eye(K), tr.info
    for i in range(4):
        for t in range(22):
            y = _phi(tr.state[i, t], I["mechanism"][i, t], I["fitness"][i, t], I["mu"][i, t], I["pool"][i, t],
                     I["mut_rate"][i, t], K, 0.1, Q)
            assert np.abs(y - tr.state[i, t + 1]).max() < 1e-10
    assert I["mechanism"].shape == I["alpha"].shape == I["segment_id"].shape == (4, 22)
    assert np.array_equal(tr.obs, tr.state.astype(np.float32))                      # freq obs = state only


@pytest.mark.parametrize("mech", ["mixed", "mutation_selection"])
def test_wf_long_horizon_against_log_coordinate_solver(mech):
    """600 steps (60 time units) with every fitness negative, so the mean fitness plus mu is below zero on the whole
    path: off the simplex the field is then transversally unstable (d(sum x - 1)/dt = -(x.f + mu)(sum x - 1)), and an
    integrator working in x drifts away. The reference works in log coordinates y (x = softmax(y)), which cannot
    leave the simplex: dy_i/dt = f_i + mu m_i / x_i (mixed), f_i + u / (K x_i) (selection + uniform mutation)."""
    K, f, x0 = 3, np.array([-1.0, -1.6, -2.3]), np.array([0.3, 0.3, 0.4])
    mu, m, u = 0.2, np.array([0.2, 0.3, 0.5]), 0.5
    par = dict(f=f, mu=mu, m=m) if mech == "mixed" else dict(f=f, u=u)
    tr = WrightFisher(K=K, x0=x0, schedule=[(mech, 60.0, par)]).sample(0, 600, 1, "freq")

    def soft(y):
        z = np.exp(y - y.max())
        return z / z.sum()

    def rhs(_, y):
        x = soft(y)
        return f + (mu * m / x if mech == "mixed" else u / (K * x))
    sol = solve_ivp(rhs, (0, 60.0), np.log(x0), t_eval=tr.t, method="DOP853", rtol=1e-12, atol=1e-13)
    ref = np.array([soft(y) for y in sol.y.T])
    assert (ref @ f + (mu if mech == "mixed" else 0.0)).max() < -0.5            # unstable off the simplex throughout
    assert np.abs(tr.state[0] - ref).max() < 1e-8


def _linear_closed_form(x0, t, f, rate, R):
    """x(t) [len(t), K] of dx/dt = x (f - x.f) + rate x R for a rate matrix R (rows sum to 0), by hand:
    y' = y (diag f + rate R) is LINEAR, and x = y / sum(y) obeys x' = x diag f + rate x R - (x.f) x because R 1 = 0.
    Hence x(t) = x0 expm(t (diag f + rate R)) / sum. Subtracting max f from the diagonal only rescales y."""
    y = np.einsum("i,tij->tj", x0, expm(t[:, None, None] * (np.diag(f - f.max()) + rate * R)))
    return y / y.sum(-1, keepdims=True)


@pytest.mark.parametrize("K,schedule,mutation", [(3, "mixed_only", "uniform"), (3, "mutation_selection", "uniform"),
                                                 (4, "mutation_selection", "cyclic")])
def test_wf_long_deterministic_paths_match_the_linear_closed_form(K, schedule, mutation):
    """600 steps, 400 paths, one segment each. Selection + mutation is linear before normalisation, and migration
    toward a pool m is mutation with R = 1 m' - I, so the exact path is a matrix exponential from t = 0: no time
    stepping and no vector field of the builder. The zoo integrates both mechanisms by RK4, where s = sum(x) obeys
    d(s - 1)/dt = -r (s - 1) with r = x.f (+ mu): r < 0 on many paths, and round-off in s would reach order 1.
    (Added together with the simplex projection of 2026-11-17, by its author: independent in method, not in person.)"""
    T = 600
    tr = WrightFisher(K=K, schedule=schedule, mutation=mutation, seg_steps=T).sample(1, T, 400, "freq")
    X, I = tr.state, tr.info
    assert np.isfinite(X).all() and X.min() >= 0 and np.abs(X.sum(-1) - 1).max() < 1e-12
    assert (I["segment_id"] == 0).all()                                              # step-0 parameters hold throughout
    shift = np.zeros((K, K))
    shift[np.arange(K), (np.arange(K) + 1) % K] = 1.0                                # cyclic: type i -> type i + 1
    R_mut = (shift if mutation == "cyclic" else np.ones((K, K)) / K) - np.eye(K)
    worst = 0.0
    for i in range(len(X)):
        if schedule == "mixed_only":
            R, rate = np.outer(np.ones(K), I["pool"][i, 0]) - np.eye(K), I["mu"][i, 0]
        else:
            R, rate = R_mut, I["mut_rate"][i, 0]
        worst = max(worst, np.abs(X[i] - _linear_closed_form(X[i, 0], tr.t, I["fitness"][i, 0], rate, R)).max())
    assert worst < 1e-9
    r = (X[:, :-1] * I["fitness"]).sum(-1) + (I["mu"] if schedule == "mixed_only" else 0.0)
    assert (r[:, -1] < 0).sum() >= 10                                                # not vacuous: r < 0 is sampled


def _circulation(v, center, r, n=3000):
    """Line integral of the Fisher-dual 1-form sum_i v_i / x_i dx_i around a circle in the simplex."""
    K = len(center)
    e = np.linalg.qr(np.eye(K) - 1.0 / K)[0][:, :2]                                 # orthonormal tangent pair
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = center + r * (np.cos(s)[:, None] * e[:, 0] + np.sin(s)[:, None] * e[:, 1])
    dX = r * (-np.sin(s)[:, None] * e[:, 0] + np.cos(s)[:, None] * e[:, 1]) * (2 * np.pi / n)
    return float((v(X) / X * dX).sum())


def test_wf_hodge_labels_by_circulation():
    F, M = np.array([1.0, -0.3, -0.9, 0.4]), np.array([0.4, 0.1, 0.3, 0.2])
    rng = np.random.default_rng(8)
    for K, mutation in ((3, "uniform"), (3, "cyclic"), (4, "uniform"), (4, "cyclic")):
        w = WrightFisher(K=K, mutation=mutation)
        f, m = F[:K], M[:K] / M[:K].sum()
        par = {"selection": dict(f=f), "migration": dict(mu=0.7, m=m), "mixed": dict(f=f, mu=0.7, m=m),
               "mutation": dict(u=0.5), "mutation_selection": dict(f=f, u=0.5), "rps": {}}
        hodge = w.ground_truth()["hodge"]
        for mech, p in par.items():
            if hodge[mech] == "undefined":
                continue
            v = lambda X: w.vector_field(X, mech, **p)
            circ = max(abs(_circulation(v, rng.dirichlet(np.ones(K) * 8), 0.07)) for _ in range(12))
            assert (circ < 1e-12) == (hodge[mech] == "gradient"), (K, mutation, mech, circ)
    assert WrightFisher(K=4).ground_truth()["hodge"]["rps"] == "undefined"


def _num_curl_div(A, x, h=1e-6):
    K = x.shape[1]
    E = [np.eye(K)[a] - np.eye(K)[K - 1] for a in range(K - 1)]
    fld = lambda z: z * (z @ A.T - np.einsum("ni,ij,nj->n", z, A, z)[:, None])
    om = lambda z: np.stack([fld(z)[:, a] / z[:, a] - fld(z)[:, -1] / z[:, -1] for a in range(K - 1)], 1)
    flux = lambda z: fld(z)[:, :K - 1] / np.sqrt(z.prod(1))[:, None]
    d = lambda fn, a, c: (fn(x + h * E[a])[:, c] - fn(x - h * E[a])[:, c]) / (2 * h)
    curl = max(np.abs(d(om, a, b) - d(om, b, a)).max() for a in range(K - 1) for b in range(a + 1, K - 1))
    return curl, np.abs(sum(d(flux, a, a) for a in range(K - 1)) * np.sqrt(x.prod(1))).max()


def test_wf_replicator_hodge_exact_criteria_against_finite_differences():
    rng = np.random.default_rng(9)
    for K in (3, 4, 5):
        x = rng.dirichlet(np.ones(K) * 3, 150)
        R = np.roll(np.eye(K), 1, 1) - np.roll(np.eye(K), -1, 1)                    # cyclic zero-sum RPS
        c, u, G = rng.standard_normal(K), rng.standard_normal(K), rng.standard_normal((K, K))
        one = np.ones(K)
        cases = {"rps": R, "rps + column consts": R + np.outer(one, c), "rps + row consts": R + np.outer(c, one),
                 "symmetric": G + G.T, "antisym u1'-1u'": np.outer(u, one) - np.outer(one, u),
                 "antisym generic": G - G.T, "generic": G}
        for name, A in cases.items():
            h = replicator_hodge(A)
            curl, div = _num_curl_div(A, x)
            assert h["gradient"] == (curl < 1e-5) and h["divergence_free"] == (div < 1e-5), (K, name)
            wf = WrightFisher(K=K, rps_A=A)
            assert wf.labels()["rps"][1] == h["label"]
            if h["gradient"]:                                       # dV(t) = g(v, t) for tangent t
                t = rng.standard_normal((150, K))
                t -= t.mean(1, keepdims=True)
                V = lambda z: wf.potential(z, "rps")
                dV = (V(x + 1e-6 * t) - V(x - 1e-6 * t)) / 2e-6
                assert np.allclose(dV, (wf.vector_field(x, "rps") * t / x).sum(1), atol=1e-6)
    assert WrightFisher().labels()["rps"][1] == "rotational"


@pytest.mark.parametrize("A,conservative", [(None, True), ([[0, -1, 2], [1, 0, -1], [-2, 1, 0]], True),
                                            ([[0, -1, 2], [2, 0, -1], [-1, 2, 0]], False),
                                            ([[1, -1, 3], [2, 0, 1], [0, 1, 2]], True),    # RPS + column consts
                                            ([[-1, -2, 0], [0, -1, -2], [-2, 0, -1]], True)])   # RPS - 1: x.Ax = -1 < 0
def test_wf_rps_invariant_claim_matches_simulation(A, conservative):
    w = WrightFisher(schedule="rps", rps_A=A)
    gt = w.ground_truth()["rps"]
    tr = w.sample(1, 600, 6, "freq")
    xs = gt["x_star"]
    H = (xs * np.log(tr.state)).sum(-1)
    drift = np.abs(H - H[:, :1]).max()
    assert gt["conservative"] == conservative and (drift < 1e-8) == conservative
    assert (gt["invariant"] is not None) == conservative and gt["orbits"].startswith("closed") == conservative
    # Shahshahani volume is preserved iff divergence-free: det DPhi_t = sqrt(prod x_t / prod x_0) in the chart
    x0 = np.array([0.3, 0.45, 0.25])
    flow = lambda z: WrightFisher(schedule="rps", rps_A=A, x0=z).sample(0, 30, 1, "freq").state[0, -1]
    J = np.stack([(flow(x0 + 1e-6 * e) - flow(x0 - 1e-6 * e))[:2] / 2e-6 for e in (np.array([1, 0, -1.0]), np.array([0, 1, -1.0]))], 1)
    ratio = np.linalg.det(J) / np.sqrt(flow(x0).prod() / x0.prod())
    assert (abs(ratio - 1) < 1e-6) == gt["divergence_free"]


def test_wf_potential_grows_at_rate_g_v_v_along_simulated_paths():
    h = 0.01
    w = WrightFisher(dt=h, n_sub=4, x0=[0.3, 0.45, 0.25])
    for sch in ("selection_only", "migration_only", "mixed_only", "mutation_selection"):
        tr = w.sample(3, 40, 6, "freq", schedule=sch, seg_steps=40)
        V = tr.info["potential"]
        dVdt = (V[:, :-4] - 8 * V[:, 1:-3] + 8 * V[:, 3:-1] - V[:, 4:]) / (12 * h)  # 5-point stencil in time
        I = tr.info
        for i in range(6):
            p = {k: I[n][i, 0] for k, n in (("f", "fitness"), ("mu", "mu"), ("m", "pool"), ("u", "mut_rate"))
                 if not np.isnan(np.atleast_1d(I[n][i, 0])).any()}
            x = tr.state[i, 2:-2]
            v = w.vector_field(x, I["mechanism"][i, 0], **p)
            assert np.allclose(dVdt[i], (v * v / x).sum(-1), rtol=1e-5, atol=1e-8), sch    # dV/dt = g(v, v)


def test_wf_geodesics_closed_form_K5():
    rng = np.random.default_rng(4)
    K = 5
    f, x0, m = rng.standard_normal(K), rng.dirichlet(np.ones(K)), rng.dirichlet(np.ones(K))
    t = np.arange(31) * 0.1
    sel = WrightFisher(K=K, x0=x0).sample(0, 30, 1, "freq", schedule=[("selection", 3.0, {"f": f})]).state[0]
    ref = x0 * np.exp(t[:, None] * f)
    assert np.allclose(sel, ref / ref.sum(1, keepdims=True), atol=1e-13)
    assert np.allclose(np.diff(theta_coords(sel), axis=0), 0.1 * (f[:-1] - f[-1]), atol=1e-12)   # affine in theta
    mig = WrightFisher(K=K, x0=x0).sample(0, 30, 1, "freq", schedule=[("migration", 3.0, {"mu": 0.6, "m": m})]).state[0]
    assert np.allclose(mig, m + (x0 - m) * np.exp(-0.6 * t)[:, None], atol=1e-13)                # straight in x
    gt = WrightFisher(K=K).ground_truth()
    assert gt["alpha"]["selection"] == 1 and gt["alpha"]["migration"] == -1 and "Amari" in gt["convention"]


def test_wf_drift_variance_per_unit_time_scales_with_generation_time():
    X0, N, out = np.array([0.2, 0.5, 0.3]), 400, {}
    for dt in (0.1, 0.05):
        w = WrightFisher(pop_size=N, x0=X0, dt=dt, schedule=[("selection", 1.0, {"f": np.zeros(3)})])
        T = int(round(1.0 / dt))
        x = w.sample(2, T, 20000, "freq").state[:, -1]
        out[dt] = x[:, 0].var()
        assert abs(out[dt] / (0.2 * 0.8 * (1 - (1 - 1 / N) ** T)) - 1) < 0.05          # neutral WF, T generations
        assert w.ground_truth()["generation_time"] == dt
    assert abs(out[0.05] / out[0.1] - 2) < 0.15
    assert WrightFisher().ground_truth()["generation_time"] is None


def test_wf_count_obs_are_multinomial_samples_of_the_state():
    w = WrightFisher(n_obs=200)
    tr = w.sample(6, 50, 400, "counts", schedule="rps")
    o, x = tr.obs.astype(np.float64), tr.state
    assert np.allclose(o * 200, np.round(o * 200), atol=1e-4) and np.allclose(o.sum(-1), 1, atol=1e-6)
    z = (o - x) / np.sqrt(x * (1 - x) / 200)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1) < 0.02


# ================================================================== cross-world
WORLDS = [("pendulum", dict(policy="random", gamma=0.05), "pixels64"), ("pendulum", dict(), "angle_cs"),
          ("toggle_grid", dict(n_doors=2, policy="loops"), "grid"), ("wright_fisher", dict(pop_size=300), "counts")]


@pytest.mark.parametrize("name,params,mode", WORLDS)
def test_determinism_no_hidden_state_and_meta_roundtrip(name, params, mode):
    w, w2 = zoo.make(name, **params), zoo.make(name, **params)
    cfg0 = dict(w.cfg)
    a = w.sample(123, 12, 5, mode)
    w.sample(7, 9, 3, mode)                                       # interleaved calls must not matter
    zoo.make("wright_fisher").sample(1, 3, 2, "freq")
    other = {"pendulum": dict(gamma=0.3), "toggle_grid": dict(door_visible=True), "wright_fisher": dict(K=4)}[name]
    w.sample(5, 4, 2, mode, **other)                              # per-call overrides must not stick
    assert w.cfg == cfg0
    for b in (w.sample(123, 12, 5, mode), w2.sample(123, 12, 5, mode)):
        assert np.array_equal(a.obs, b.obs) and np.array_equal(a.state, b.state)
        assert (a.actions is None and b.actions is None) or np.array_equal(a.actions, b.actions)
    assert a.obs.shape[2:] == tuple(w.spec(mode)["obs_shape"]) and str(a.obs.dtype) == w.spec(mode)["obs_dtype"]
    assert a.obs.shape[:2] == (5, 13) and a.state.shape[:2] == (5, 13) and len(a.t) == 13
    w.ground_truth(**a.meta["params"])                           # ground truth reconstructible from meta


@pytest.mark.parametrize("name,params,mode", WORLDS)
def test_splits_never_share_data(name, params, mode):
    w = zoo.make(name, **params)
    data = {(s, i): w.sample(zoo.split_seed(s, i), 6, 16, mode) for s in SPLIT_SEEDS for i in (0, 1)}
    keys = list(data)
    for k1, k2 in itertools.combinations(keys, 2):
        A, B = (data[k].state.reshape(16, -1) for k in (k1, k2))
        assert not (A[:, None] == B[None]).all(-1).any(), (k1, k2)              # no shared trajectory
        if name != "toggle_grid":                                                # continuous: no shared start
            assert not (data[k1].state[:, 0][:, None] == data[k2].state[:, 0][None]).all(-1).any()
    seeds = [zoo.split_seed(s, i) for s in SPLIT_SEEDS for i in (0, 999_999)]
    assert len(set(seeds)) == len(seeds)
    ranges = sorted((zoo.split_seed(s, 0), zoo.split_seed(s, 999_999)) for s in SPLIT_SEEDS)
    assert all(r1[1] < r2[0] for r1, r2 in zip(ranges, ranges[1:]))


def test_truths_stated_in_cells_yaml_worlds_block():
    """cells.yaml worlds: pendulum 'S^1 x R; symplectic; energy conserved; div = 0'; damped_pendulum
    'div = -2*gamma (uniform)'; toggle_grid 'hidden door state; non-commuting actions; holonomy around switch
    loops'; wright_fisher 'selection = e-geodesic, migration = m-geodesic; RPS variant is rotational'."""
    p = zoo.make("pendulum").ground_truth()
    assert p["state_space"] == "S1 x R" and p["phase_space_betti"] == [1, 1] and p["symplectic"]
    assert p["conserved"] == ["energy"] and p["divergence"] == 0
    for g in (0.05, 0.3):
        assert zoo.make("pendulum", gamma=g).ground_truth()["divergence"] == -2 * g
    t = zoo.make("toggle_grid").ground_truth()
    assert t["hidden_state_bits"] >= 1 and any(e["same_position"] and e["hidden_differs"] for e in t["commutator_examples"])
    assert (t["curvature_map"]["parity"].sum(1) > 0).sum() == 4
    wf = zoo.make("wright_fisher").ground_truth()
    assert wf["alpha"]["selection"] == 1 and wf["alpha"]["migration"] == -1 and wf["hodge"]["rps"] == "rotational"
