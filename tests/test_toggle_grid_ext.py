"""Tests for the cell-6 control options added to toggle_grid on 2026-11-03 (defaults unchanged)."""
import numpy as np

import zoo


def test_defaults_unchanged():
    w = zoo.make("toggle_grid")
    assert w.cfg["switches_enabled"] is True and w.cfg["switch_visible"] is True
    gt = w.ground_truth()
    assert gt["hidden_state_bits"] == 1 and gt["switches_enabled"] and gt["switch_visible"]
    tr = w.sample(seed=0, horizon=200, n_traj=50, obs_mode="grid")
    assert (tr.state[:, :, 2] != tr.state[:, :1, 2]).any(), "default world must toggle doors"


def test_switches_disabled_is_a_zero_holonomy_control():
    for k in (1, 2, 3):
        w = zoo.make("toggle_grid", n_doors=k, switches_enabled=False)
        tr = w.sample(seed=1, horizon=300, n_traj=100, obs_mode="grid", policy="loops")
        assert (tr.state[:, :, 2:] == tr.state[:, :1, 2:]).all(), "door bits must be constant"
        gt = w.ground_truth()
        assert gt["hidden_state_bits"] == 0 and gt["hidden_bits_by_room"] == [0] * (k + 1)
        assert (gt["curvature_map"]["parity"] == 0).all()
        # a loop through the switch flips nothing
        sx, sy = w.switches[0]
        s0 = np.array([sx - 1, sy] + [0] * k)
        h = w.holonomy(s0, [3, 1, 2, 0])     # right, down, left, up from the cell left of S_1
        assert h["position_closed"] and not h["flips"].any()


def test_switch_invisible_only_blanks_marker_channel():
    a = zoo.make("toggle_grid")
    b = zoo.make("toggle_grid", switch_visible=False)
    st = a.sample(seed=2, horizon=50, n_traj=10, obs_mode="grid").state
    oa, ob = a.observe(st, "grid"), b.observe(st, "grid")
    assert oa.shape == ob.shape
    assert (ob[..., 2, :, :] == 0).all() and (oa[..., 2, :, :].sum() > 0)
    keep = [0, 1, 3]
    assert np.array_equal(oa[..., keep, :, :], ob[..., keep, :, :])
    # dynamics identical: switches still toggle when invisible
    ta = a.sample(seed=3, horizon=100, n_traj=20, obs_mode="pos").state
    tb = b.sample(seed=3, horizon=100, n_traj=20, obs_mode="pos").state
    assert np.array_equal(ta, tb)
