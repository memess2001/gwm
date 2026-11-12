"""Ground-truth tests of the cell-6 instruments (analysis/holonomy.py) and of cells/6/c6lib.py (rules.R4)."""
import sys
from pathlib import Path

import numpy as np
import pytest

import zoo
from analysis import holonomy as ho

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells" / "6"))
import c6lib as c                                    # noqa: E402


@pytest.fixture(scope="module")
def w():
    return zoo.make("toggle_grid")


def _starts(w, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s = w._initial(rng, 1, w.cfg)[0]
        if w.switch_id[s[1], s[0]] < 0:
            out.append(s)
    return out


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_constructive_loops_exact(w, n):
    rng = np.random.default_rng(n)
    for s0 in _starts(w, 30, 10 + n):
        p = ho.constructive_path(w, s0, n, rng, max_len=28)
        assert p is not None and len(p.actions) <= 28
        st, ent = ho.simulate(w, s0, p.actions)
        assert ent == n
        assert (st[-1, :2] == s0[:2]).all()
        assert int(w.holonomy(s0, p.actions)["flips"][0]) == n % 2


def test_constructive_path_to_end_cell(w):
    rng = np.random.default_rng(3)
    for s0 in _starts(w, 20, 4):
        p = ho.constructive_path(w, s0, 2, rng, end_cell=(5, 3), max_len=40)
        assert p is not None
        st, ent = ho.simulate(w, s0, p.actions)
        assert ent == 2 and tuple(st[-1, :2]) == (5, 3)


def test_null_reference_is_time_matched_and_trivial(w):
    rng = np.random.default_rng(5)
    for s0 in _starts(w, 20, 6):
        for L in (6, 13, 21, 28):
            ref = ho.null_reference(w, s0, L, rng)
            assert ref is not None and len(ref) == L
            st, ent = ho.simulate(w, s0, ref)
            assert ent == 0 and (st[-1] == s0).all()


def test_pad_with_bumps_keeps_end_state(w):
    rng = np.random.default_rng(7)
    s0 = np.array([1, 1, 0])
    a = np.array([3, 1, 2, 0])
    p = ho.pad_with_bumps(w, s0, a, 11, rng)
    assert len(p) == 11
    assert (w.trajectory(s0, p)[-1] == w.trajectory(s0, a)[-1]).all()


def test_plaquette_probe_holonomy_iff_switch(w):
    rng = np.random.default_rng(8)
    corners = ho.plaquette_corners(w, 0)
    assert len(corners) == 16
    sw = ho.switch_plaquette_mask(w, corners)
    assert sw.sum() == 4
    for s0 in _starts(w, 10, 9):
        for k, cn in enumerate(corners):
            g, r = ho.plaquette_probe(w, s0, cn, rng)
            assert len(g) == len(r)
            hg = int(w.holonomy(s0, g)["flips"][0])
            hr = int(w.holonomy(s0, r)["flips"][0])
            assert hr == 0 and hg == int(sw[k])
            assert (w.trajectory(s0, g)[-1, :2] == w.trajectory(s0, r)[-1, :2]).all()


def test_door_adjacent_plaquettes(w):
    corners = ho.plaquette_corners(w, 0)
    m = ho.door_adjacent_plaquette_mask(w, corners)
    assert sorted(map(tuple, corners[m])) == [(4, 2), (4, 3)]


def test_true_predictor_holonomy_matches_truth(w):
    rng = np.random.default_rng(11)
    tp = ho.TruePredictor(w)
    pr = ho.probe_set()
    for s0 in _starts(w, 8, 12):
        Z0 = tp.start(s0[None])
        for n in (0, 1, 2, 3):
            p = ho.constructive_path(w, s0, n, rng)
            ref = ho.null_reference(w, s0, len(p.actions), rng)
            h = ho.h_obs(tp, tp.run(Z0, p.actions[None]), tp.run(Z0, ref[None]), pr)["max"][0]
            if n % 2 == 0:
                assert h == 0.0
            else:
                assert h > 0.0          # 64 random probes catch the bit from every room-0 cell (probe_power)


def test_memoryless_has_zero_holonomy(w):
    rng = np.random.default_rng(13)
    mp = ho.MemorylessPredictor(w)
    pr = ho.probe_set()
    for s0 in _starts(w, 5, 14):
        Z0 = mp.start(s0[None])
        p = ho.constructive_path(w, s0, 1, rng)
        ref = ho.null_reference(w, s0, len(p.actions), rng)
        assert ho.h_obs(mp, mp.run(Z0, p.actions[None]), mp.run(Z0, ref[None]), pr)["max"][0] == 0.0


def test_any_entry_loophole_shows_on_even_loops(w):
    rng = np.random.default_rng(15)
    ap = ho.AnyEntryPredictor(w)
    pr = ho.probe_set()
    s0 = np.array([4, 4, 0])
    Z0 = ap.start(s0[None])
    p = ho.constructive_path(w, s0, 2, rng)
    ref = ho.null_reference(w, s0, len(p.actions), rng)
    assert ho.h_obs(ap, ap.run(Z0, p.actions[None]), ap.run(Z0, ref[None]), pr)["max"][0] > 0.0


def test_abelian_control_zero_on_balanced_loops(w):
    """P6.5b as a unit test: antisymmetric increments give exactly zero holonomy on any loop whose action counts
    balance (all plaquette loops), for any random parameters; an unbalanced pair differs."""
    pr = ho.probe_set()
    for seed in range(5):
        ab = ho.AbelianPredictor(w, seed=seed)
        rng = np.random.default_rng(seed)
        for s0 in _starts(w, 4, 20 + seed):
            Z0 = ab.start(s0[None])
            for cn in ho.plaquette_corners(w, 0):
                g = ho.plaquette_loop_from(cn, tuple(cn))
                ZA, ZB = ab.run(Z0, g[None]), Z0
                assert np.allclose(ZA, ZB)
                assert ho.h_obs(ab, ZA, ZB, pr)["max"][0] == 0.0
        ZA = ab.run(ab.start(np.array([[3, 3, 0]])), np.array([[3, 3]]))
        assert not np.allclose(ZA, ab.start(np.array([[3, 3, 0]])))



def test_auc_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.integers(0, 4, rng.integers(1, 12)).astype(float)
        n = rng.integers(0, 4, rng.integers(1, 12)).astype(float)
        bf = np.mean([(a > b) + 0.5 * (a == b) for a in p for b in n])
        assert ho.auc(p, n) == pytest.approx(bf)


def test_stratified_auc_removes_length_confound():
    # score = length; label correlated with length across strata but not within: stratified AUC = 0.5
    length = np.array([5, 5, 5, 5, 20, 20, 20, 20])
    label = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    score = length.astype(float)
    assert ho.auc(score[label == 1], score[label == 0]) > 0.5
    assert ho.stratified_auc(score, label, length) == pytest.approx(0.5)


def test_precision_at_k():
    s = np.array([5, 4, 3, 2, 1, 0])
    t = np.array([1, 0, 1, 0, 0, 1], bool)
    assert ho.precision_at_k(s, t, 2) == 0.5
    assert list(ho.ranks_desc(s)) == [1, 2, 3, 4, 5, 6]


def test_probe_set_pinned():
    a, b = ho.probe_set(), ho.probe_set()
    assert a.shape == (64, 20) and (a == b).all() and a.dtype == np.int64


def test_passive_nocover_never_attempts_door():
    w = c.make_world("H-pn")
    st, a = c.sample_policy(w, "passive_nocover", 1, 300)
    assert c.coverage(w, st, a)["door_attempts"] == 0


def test_policies_are_deterministic_and_valid():
    w = c.make_world("H-act")
    for pol, ph in (("passive_cover", 0.0), ("active", 0.01)):
        st1, a1 = c.sample_policy(w, pol, 5, 50, p_home=ph)
        st2, a2 = c.sample_policy(w, pol, 5, 50, p_home=ph)
        assert (st1 == st2).all() and (a1 == a2).all()
        assert (w.trajectory(st1[:, 0], a1) == st1).all()


def test_gate_set_balanced_and_exact():
    w = c.make_world("H-act")
    g = c.gate_set(w, 1, n_per_cell=25)
    ax, ay = w.doors[0][0] - 1, w.doors[0][1]
    by_n = {}
    for e in g:
        by_n[e["n"]] = by_n.get(e["n"], 0) + 1
        assert len(e["actions"]) <= c.HORIZON and e["actions"][-1] == 3
        st = e["states"]
        assert tuple(st[-2, :2]) == (ax, ay)
        _, ent = ho.simulate(w, e["s0"], e["actions"])
        assert ent == e["n"]
        assert st[-1, 2] == e["n"] % 2 and (tuple(st[-1, :2]) == (ax + 1, ay)) == (e["n"] % 2 == 1)
    assert by_n == {0: 25, 1: 25, 2: 25, 3: 25}


def test_gate_eval_on_oracles():
    w = c.make_world("H-act")
    g = c.gate_set(w, 2, n_per_cell=20)
    r = c.gate_eval(ho.TruePredictor(w), g)
    assert c.gate_pass(r) and r["position_acc"] == 1.0
    r = c.gate_eval(ho.MemorylessPredictor(w), g)
    assert r["door_acc"][1] == 0.0 and not c.gate_pass(r)
    r = c.gate_eval(ho.AnyEntryPredictor(w), g)
    assert r["door_acc"][2] == 0.0 and not c.gate_pass(r)       # parity, not 'ever entered'
