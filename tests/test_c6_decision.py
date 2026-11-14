"""The cell-6 decision table (cells/6/c6_score.aggregate, v1.4: M only) on fabricated run directories (no models)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells" / "6"))
import c6_score as cs                                   # noqa: E402

STEPS = 20000
SIZES = ("M",)


def _arrays(h_even=0.0, h_null=0.0, h_odd=1.0, n_z=20, obs_closed_odd=True):
    cls, h, z, oc = [], [], [], []
    for i in range(n_z):
        for cc, hv in (("odd", h_odd), ("even2", h_even), ("null", h_null)):
            for _ in range(4):
                cls.append(cc)
                h.append(hv)
                z.append(i)
                oc.append(obs_closed_odd if cc == "odd" else True)
    return {"loop_cls": np.array(cls), "loop_h": np.array(h, float), "loop_z": np.array(z),
            "loop_obs_closed": np.array(oc)}


def _write(root, cond, size, seed, n=2000, gate=True, s=None, arrays=None, steps=STEPS, pilot=False):
    d = root / (f"{cond}_{size}_s{seed}_n{n}" + (f"_pilot{steps}" if pilot else ""))
    d.mkdir(parents=True)
    (d / "c6_run.json").write_text(json.dumps({"cond": cond, "size": size, "seed": seed, "n_train": n,
                                               "pilot": pilot, "steps": steps}))
    (d / "gate.json").write_text(json.dumps({"pass": gate, "door_acc": {"0": 1.0}, "position_acc": 1.0}))
    if pilot:
        return
    base = {"auc_strat": 1.0, "prec4_R0": 1.0, "tau": 0.0, "model_check": 1.0,
            "base_auc_strat": 1.0, "base_prec4_R0": 1.0, "base_auc_strat_mixed": 1.0}
    base.update(s or {})
    (d / "scores.json").write_text(json.dumps(base))
    np.savez(d / "scores.npz", **(arrays or _arrays()))


def _world(root, primary=None, controls=None, ctl_s=None, ctl_gate=True, eff=None, gates=None, skip=(),
           pilot_pass=True, pilot_size="M", pilot_steps=STEPS):
    _write(root, "H-act", pilot_size, 99, steps=pilot_steps, pilot=True, gate=pilot_pass)
    eff = eff or {n: {"auc_strat": 0.5, "base_auc_strat": 0.5, "prec4_R0": 0.25, "base_prec4_R0": 0.25}
                  for n in (250, 500)}
    for size in SIZES:
        for sd in (0, 1, 2):
            g = True if gates is None else gates[size][sd]
            _write(root, "H-act", size, sd, gate=g, s=(primary or {}))
            for ctl in ("C-open", "C-vis"):
                if (ctl, size, sd) in skip:
                    continue
                _write(root, ctl, size, sd, gate=ctl_gate, s=(ctl_s or {}), arrays=(controls or _arrays(h_odd=0.0)))
            for cond in ("H-pc", "H-pn"):
                _write(root, cond, size, sd, s={"auc_strat": 1.0 if cond == "H-pc" else 0.5})
            for n, s in eff.items():
                _write(root, "H-act", size, sd, n=n, s=s)


def test_readout_works_but_ties_baseline_merges(tmp_path):
    _world(tmp_path)
    r = cs.aggregate(tmp_path)
    assert r["missing_runs"] == [] and r["protocol"]["ok"]
    assert r["readout_verdict"] == "READOUT_WORKS" and r["geometric_claim_survives"] is False
    assert r["status"].startswith("merged_into:HCDP")


def test_margin_win_on_auc_gives_alive(tmp_path):
    _world(tmp_path, primary={"base_auc_strat": 0.92})
    r = cs.aggregate(tmp_path)
    assert r["readout_verdict"] == "READOUT_WORKS" and r["status"] == "alive"


def test_data_efficiency_win(tmp_path):
    _world(tmp_path, eff={250: {"auc_strat": 0.95, "base_auc_strat": 0.6, "prec4_R0": 0.25, "base_prec4_R0": 0.25},
                          500: {"auc_strat": 0.95, "base_auc_strat": 0.95, "prec4_R0": 0.25, "base_prec4_R0": 0.25}})
    r = cs.aggregate(tmp_path)
    assert r["P66"]["M"]["win_data_det"] and r["status"] == "alive"


def test_nonspecific_readout_is_dead_when_controls_learned(tmp_path):
    _world(tmp_path, controls=_arrays(h_even=1.0, h_null=1.0, h_odd=1.0))
    r = cs.aggregate(tmp_path)
    assert r["readout_verdict"] == "READOUT_DEAD" and r["status"] == "dead"
    assert r["per_size"]["M"]["not_specific"]


def test_control_that_did_not_learn_is_unchecked(tmp_path):            # review C
    _world(tmp_path, controls=_arrays(h_even=1.0, h_null=1.0, h_odd=1.0), ctl_s={"model_check": 0.3})
    assert cs.aggregate(tmp_path)["readout_verdict"] == "INCONCLUSIVE"


def test_control_gate_failure_is_unchecked(tmp_path):                  # review C
    _world(tmp_path, ctl_gate=False)
    assert cs.aggregate(tmp_path)["readout_verdict"] == "INCONCLUSIVE"


def test_tau_ceiling_fails_p63(tmp_path):                              # review E
    _world(tmp_path, primary={"tau": 1.0})
    r = cs.aggregate(tmp_path)
    assert not r["per_size"]["M"]["P63_size"] and r["per_size"]["M"]["exceedance"][0]["tau_ceiling"]


def test_failure_with_failed_model_check_is_inconclusive(tmp_path):
    _world(tmp_path, primary={"auc_strat": 0.6, "model_check": 0.5})
    assert cs.aggregate(tmp_path)["readout_verdict"] == "INCONCLUSIVE"


def test_gate_failures_make_size_not_evaluable(tmp_path):
    _world(tmp_path, gates={"M": [True, False, False]})
    r = cs.aggregate(tmp_path)
    assert r["per_size"]["M"]["state"] == "NE" and r["readout_verdict"] == "INCONCLUSIVE"


def test_one_checked_failure_plus_gate_failure_is_unchecked(tmp_path):  # review N
    _world(tmp_path, gates={"M": [True, True, False]})
    for size in SIZES:
        d = tmp_path / f"H-act_{size}_s0_n2000"
        s = json.loads((d / "scores.json").read_text())
        s["auc_strat"] = 0.6
        (d / "scores.json").write_text(json.dumps(s))
    r = cs.aggregate(tmp_path)
    assert r["per_size"]["M"]["state"] == "FAIL_UNCHECKED" and r["readout_verdict"] == "INCONCLUSIVE"


def test_checked_failure_on_m_is_dead(tmp_path):                      # v1.4: M's state alone decides
    _world(tmp_path)
    for sd in (0, 1, 2):
        d = tmp_path / f"H-act_M_s{sd}_n2000"
        s = json.loads((d / "scores.json").read_text())
        s["auc_strat"] = 0.6
        (d / "scores.json").write_text(json.dumps(s))
    r = cs.aggregate(tmp_path)
    assert r["per_size"]["M"]["state"] == "FAIL_CHECKED" and r["readout_verdict"] == "READOUT_DEAD"
    assert list(r["per_size"]) == ["M"]


def test_v14_needs_21_runs_and_no_s_runs():
    req = cs.required_runs()
    assert len(req) == 21 and {k[1] for k in req} == {"M"}
    assert cs.c.SIZES == ("M",) and cs.c.MAIN_STEPS == 20000


def test_s_pilot_does_not_set_the_step_count(tmp_path):                # v1.4 step rule
    _world(tmp_path, pilot_size="S")                                    # only an S pilot, even if it passed
    r = cs.aggregate(tmp_path)
    assert r["readout_verdict"] == "INCOMPLETE" and r["protocol"]["steps_by_rule"] is None


def test_runs_at_10k_steps_are_incomplete(tmp_path):                    # v1.4: main runs must use 20k steps
    _world(tmp_path)
    for d in tmp_path.glob("*_n2000"):
        m = json.loads((d / "c6_run.json").read_text())
        m["steps"] = 10000
        (d / "c6_run.json").write_text(json.dumps(m))
    r = cs.aggregate(tmp_path)
    assert r["readout_verdict"] == "INCOMPLETE" and not r["protocol"]["ok"]


def test_missing_control_run_is_incomplete(tmp_path):                  # review B
    _world(tmp_path, skip={("C-vis", "M", 0), ("C-vis", "M", 1), ("C-vis", "M", 2)})
    r = cs.aggregate(tmp_path)
    assert r["readout_verdict"] == "INCOMPLETE" and len(r["missing_runs"]) == 3


def test_missing_data_efficiency_run_is_incomplete(tmp_path):          # review B
    _world(tmp_path, eff={250: {"auc_strat": 0.5, "base_auc_strat": 0.5, "prec4_R0": 0.25, "base_prec4_R0": 0.25}})
    assert cs.aggregate(tmp_path)["readout_verdict"] == "INCOMPLETE"


def test_step_rule_is_enforced(tmp_path):                               # review J
    _world(tmp_path, pilot_pass=False)                                  # the M 20k pilot failed its gate
    r = cs.aggregate(tmp_path)
    assert r["readout_verdict"] == "INCOMPLETE" and not r["protocol"]["ok"]


def test_pilot_refused(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    (d / "c6_run.json").write_text(json.dumps({"cond": "H-act", "size": "S", "seed": 99, "pilot": True}))
    with pytest.raises(SystemExit):
        cs.score_run(d)


def test_model_check_is_min_over_branches():
    """v1.2 (second review): the model check takes the minimum over the loop and reference branches, never the
    pooled frames (the reference branch is almost always right and would dilute the loop branch's errors)."""
    import c6lib as c
    from analysis import holonomy as ho
    w = c.make_world("H-act")
    rng = np.random.default_rng(0)
    z = c.sample_zs(w, 7, 3)
    loops = [lp for zz in z for lp in c.loops_for_z(w, zz, rng) if lp["cls"] == "odd"]
    sz = [zz["sz"] for zz in z for lp in c.loops_for_z(w, zz, np.random.default_rng(0)) if lp["cls"] == "odd"]
    sz = np.stack(sz[:len(loops)])
    pred = ho.MemorylessPredictor(w)
    ZA = np.stack([pred.run(pred.start(s[None]), lp["gamma"][None])[0] for s, lp in zip(sz, loops)])
    ZB = np.stack([pred.run(pred.start(s[None]), lp["ref"][None])[0] for s, lp in zip(sz, loops)])
    tA = np.stack([w.trajectory(s, lp["gamma"])[-1] for s, lp in zip(sz, loops)])
    tB = np.stack([w.trajectory(s, lp["ref"])[-1] for s, lp in zip(sz, loops)])
    r = cs.compare(pred, w, ZA, ZB, tA, tB)
    assert r["model_check_bit_sensitive_B"] == 1.0 and r["model_check_bit_sensitive_A"] < 0.5
    assert r["model_check_bit_sensitive"] == r["model_check_bit_sensitive_A"]
    assert r["model_check"] == min(r["model_check_all_A"], r["model_check_all_B"],
                                   r["model_check_bit_sensitive_A"], r["model_check_bit_sensitive_B"])
