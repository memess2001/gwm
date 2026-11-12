"""The cell-6 decision table (cells/6/c6_score.aggregate) on fabricated run directories (no models)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells" / "6"))
import c6_score as cs                                   # noqa: E402


def _arrays(h_even=0.0, h_null=0.0, h_odd=1.0, n_z=20, obs_closed_odd=True):
    cls, h, z, oc = [], [], [], []
    for i in range(n_z):
        for k, (cc, hv) in enumerate((("odd", h_odd), ("even2", h_even), ("null", h_null))):
            for _ in range(4):
                cls.append(cc)
                h.append(hv)
                z.append(i)
                oc.append(obs_closed_odd if cc == "odd" else True)
    return {"loop_cls": np.array(cls), "loop_h": np.array(h, float), "loop_z": np.array(z),
            "loop_obs_closed": np.array(oc)}


def _write(root, cond, size, seed, n=2000, gate=True, s=None, arrays=None):
    d = root / f"{cond}_{size}_s{seed}_n{n}"
    d.mkdir(parents=True)
    (d / "c6_run.json").write_text(json.dumps({"cond": cond, "size": size, "seed": seed, "n_train": n, "pilot": False}))
    (d / "gate.json").write_text(json.dumps({"pass": gate}))
    base = {"auc_strat": 1.0, "prec4_R0": 1.0, "tau": 0.0, "model_check": 1.0,
            "base_auc_strat": 1.0, "base_prec4_R0": 1.0}
    base.update(s or {})
    (d / "scores.json").write_text(json.dumps(base))
    np.savez(d / "scores.npz", **(arrays or _arrays()))


def _world(root, primary=None, controls=None, eff=None, gates=None):
    for size in ("S", "M"):
        for sd in (0, 1, 2):
            g = True if gates is None else gates[size][sd]
            _write(root, "H-act", size, sd, gate=g, s=(primary or {}))
            for ctl in ("C-open", "C-vis"):
                _write(root, ctl, size, sd, s={}, arrays=(controls or _arrays(h_odd=0.0)))
            for cond in ("H-pc", "H-pn"):
                _write(root, cond, size, sd, s={"auc_strat": 1.0 if cond == "H-pc" else 0.5})
            for n, s in (eff or {}).items():
                _write(root, "H-act", size, sd, n=n, s=s)


def test_readout_works_but_ties_baseline_merges(tmp_path):
    _world(tmp_path, eff={250: {"auc_strat": 0.5, "base_auc_strat": 0.5, "prec4_R0": 0.25, "base_prec4_R0": 0.25},
                          500: {"auc_strat": 0.5, "base_auc_strat": 0.5, "prec4_R0": 0.25, "base_prec4_R0": 0.25}})
    r = cs.aggregate(tmp_path)
    assert r["readout_verdict"] == "READOUT_WORKS"
    assert r["geometric_claim_survives"] is False
    assert r["status"].startswith("merged_into:HCDP")


def test_margin_win_on_auc_gives_alive(tmp_path):
    _world(tmp_path, primary={"base_auc_strat": 0.92},
           eff={250: {"auc_strat": 0.5, "base_auc_strat": 0.5, "prec4_R0": 0.25, "base_prec4_R0": 0.25},
                500: {"auc_strat": 0.5, "base_auc_strat": 0.5, "prec4_R0": 0.25, "base_prec4_R0": 0.25}})
    r = cs.aggregate(tmp_path)
    assert r["readout_verdict"] == "READOUT_WORKS" and r["geometric_claim_survives"] is True
    assert r["status"] == "alive"


def test_data_efficiency_win(tmp_path):
    _world(tmp_path, eff={250: {"auc_strat": 0.95, "base_auc_strat": 0.6, "prec4_R0": 0.25, "base_prec4_R0": 0.25},
                          500: {"auc_strat": 0.95, "base_auc_strat": 0.95, "prec4_R0": 0.25, "base_prec4_R0": 0.25}})
    r = cs.aggregate(tmp_path)
    assert r["P66"]["S"]["win_data_det"] and r["status"] == "alive"


def test_nonspecific_readout_is_dead(tmp_path):
    _world(tmp_path, controls=_arrays(h_even=1.0, h_null=1.0, h_odd=1.0))
    r = cs.aggregate(tmp_path)
    assert r["readout_verdict"] == "READOUT_DEAD" and r["status"] == "dead"
    assert r["per_size"]["S"]["P61_size"] and not r["per_size"]["S"]["P63_size"]


def test_failure_with_failed_model_check_is_inconclusive(tmp_path):
    _world(tmp_path, primary={"auc_strat": 0.6, "model_check": 0.5})
    r = cs.aggregate(tmp_path)
    assert r["readout_verdict"] == "INCONCLUSIVE"


def test_gate_failures_make_size_not_evaluable(tmp_path):
    gates = {"S": [True, False, False], "M": [True, False, False]}
    _world(tmp_path, gates=gates)
    r = cs.aggregate(tmp_path)
    assert r["per_size"]["S"]["state"] == "NE" and r["readout_verdict"] == "INCONCLUSIVE"


def test_size_disagreement_is_inconclusive(tmp_path):
    _world(tmp_path)
    for sd in (0, 1, 2):                          # M fails P6.1 with a good model check
        d = tmp_path / f"H-act_M_s{sd}_n2000"
        s = json.loads((d / "scores.json").read_text())
        s["auc_strat"] = 0.6
        (d / "scores.json").write_text(json.dumps(s))
    r = cs.aggregate(tmp_path)
    assert r["per_size"]["S"]["state"] == "PASS" and r["per_size"]["M"]["state"] == "FAIL_CHECKED"
    assert r["readout_verdict"] == "INCONCLUSIVE"


def test_missing_runs_incomplete(tmp_path):
    _write(tmp_path, "H-act", "S", 0)
    assert cs.aggregate(tmp_path)["readout_verdict"] == "INCOMPLETE"


def test_pilot_refused(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    (d / "c6_run.json").write_text(json.dumps({"cond": "H-act", "size": "S", "seed": 99, "pilot": True}))
    with pytest.raises(SystemExit):
        cs.score_run(d)
