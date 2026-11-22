"""Ground-truth tests of the seam ratio v2 and the PH x SR consistency table (cell-0 prereg v2 items 2, 3, 6)."""
import numpy as np
import pytest

import zoo
from analysis import controls as C
from analysis import seam as SM


def _wrap(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


@pytest.fixture(scope="module")
def states():
    w = zoo.make("pendulum")
    tr = w.sample(seed=zoo.split_seed("analysis", 995_000), horizon=64, n_traj=500, obs_mode="state", omega_max=2.5)
    th_all, om_all = tr.state[..., 0], tr.state[..., 1]
    th, om, thp = th_all[:, 1:].reshape(-1), om_all[:, 1:].reshape(-1), th_all[:, :-1].reshape(-1)
    return th, om, thp, np.repeat(np.arange(500), 64)


def test_straddle_mask():
    th = np.array([3.1, -3.1, 0.5, -3.0])
    thp = np.array([-3.1, 3.1, 0.4, -2.9])
    assert list(SM.straddle_mask(th, thp)) == [True, True, False, False]


def test_v2_fixes_displaced_tear_and_kappa(states):
    th, om, thp, g = states
    strad = SM.straddle_mask(th, thp)
    lift = thp + _wrap(th - thp)
    displaced = np.stack([np.where(strad, lift, th), om], 1)
    assert SM.seam_ratio(displaced, th, om, groups=g, n_boot=0)["ratio"] < 10         # v1: 2.8-5 (reviewer), ~45 in v2
    assert SM.seam_ratio_v2(displaced, th, om, thp, groups=g, n_boot=0)["ratio"] > 20
    glued = np.stack([np.cos(th), np.sin(th), 9.5 * om], 1)
    r = SM.seam_ratio_v2(glued, th, om, thp, groups=g, n_boot=0)
    assert r["ratio"] <= 1.5 and r["kappa_first_pass"] > 8 and r["omega_tol_used"] < 0.01
    x, y = C._arc_xy(th, 0.9)
    arc = np.stack([x, y, 5.0 * om], 1)
    assert SM.seam_ratio_v2(arc, th, om, thp, groups=g, n_boot=0)["ratio"] > 5             # v1: ~3.4 (partial)


def test_classify_seam3():
    assert SM.classify_seam3({"status": "ok", "ratio": 1.2}) == "glued"
    assert SM.classify_seam3({"status": "ok", "ratio": 3.0}) == "partial"
    assert SM.classify_seam3({"status": "ok", "ratio": 6.0}) == "torn"
    assert SM.classify_seam3({"status": "insufficient", "ratio": np.nan}) == "insufficient"


@pytest.mark.parametrize("ph,sr,kappa,folded,want", [
    ("ring", "glued", 1.0, False, "glued"),
    ("none", "torn", 1.0, False, "torn"),
    ("none", "torn", 7.0, False, "inconclusive"),        # PH false negatives under omega anisotropy (item 6)
    ("ring", "torn", 1.0, False, "inconclusive"),
    ("none", "glued", 1.0, False, "inconclusive"),
    ("ring", "partial", 1.0, False, "inconclusive"),
    ("none", "partial", 1.0, False, "inconclusive"),
    ("unstable", "glued", 1.0, False, "inconclusive"),
    ("ring_not_wrapping", "glued", 1.0, False, "inconclusive"),
    ("ring", "glued", 1.0, True, "inconclusive"),
    ("ring", "insufficient", 1.0, False, "inconclusive"),
])
def test_consistency_table(ph, sr, kappa, folded, want):
    assert SM.seam_ph_consistency(ph, sr, kappa, folded) == want
