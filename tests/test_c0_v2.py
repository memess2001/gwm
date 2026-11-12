"""Cell 0 v2 decision table (cells/0/c0_aggregate.py) on fabricated measures (no models)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells" / "0"))
import c0_aggregate as ag                               # noqa: E402
import c0lib as c                                       # noqa: E402


def _m(h1="ring", consistency="glued", theta_ok=True, excluded=False, kappa=1.0, seam=1.0, pointwise=True,
       decodable=True, smooth=True):
    return {"h1": {"state": h1}, "consistency": consistency, "theta_gate_n8000": {"passed": theta_ok},
            "exclusion": {"excluded": excluded}, "h1_none_high_kappa": h1 == "none" and kappa > 5,
            "seam": {"ratio": seam}, "pointwise": {"holds": pointwise},
            "energy": {"decodable": decodable, "smooth": smooth}}


def _world(tmp_path, monkeypatch, arm_cls, neg="none", cs_h1="ring"):
    monkeypatch.setattr(ag, "RUNS", tmp_path)
    for r in c.run_table():
        if r["phase"] != 1:
            continue
        if r["world"] == "neg":
            m = _m(h1=neg)
        elif r["obs"] == "angle_raw":
            cls = arm_cls.get(r["arm"], "torn")
            m = _m(h1="ring" if cls == "glued" else "none", consistency=cls,
                   seam=1.0 if cls == "glued" else 40.0)
        else:
            m = _m(h1=cs_h1)
        d = tmp_path / r["name"]
        d.mkdir()
        (d / "measures.json").write_text(json.dumps(m))


def test_pilot_pattern(tmp_path, monkeypatch):
    """sg tears, sym glues, recon-mse tears, recon-circ glues -> P0.3 T, P0.4-sg F, P0.4-sym T."""
    _world(tmp_path, monkeypatch, {"raw-recon-mse": "torn", "raw-recon-circ": "glued", "raw-jepa-sg": "torn",
                                   "raw-jepa-sym": "glued"})
    r = ag.aggregate()
    assert r["P03"] == "T" and r["P04"] == {"sg": "F", "sym": "T"}
    assert r["verdicts"]["decision"]["state"] == "DECIDED"
    assert "对称" in r["verdicts"]["decision"]["updates"]["cell0.notes.mechanism"]


def test_everything_inherits(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch, {})                          # every angle_raw arm torn
    r = ag.aggregate()
    assert r["P03"] == "F" and r["P04"] == {"sg": "F", "sym": "F"}
    assert r["verdicts"]["decision"]["updates"]["cell0.status"] == "dead"


def test_instrument_failure_stops(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch, {"raw-recon-circ": "glued"}, neg="ring")    # negative control shows a ring
    assert ag.aggregate()["verdicts"]["decision"]["state"] == "STOP"


def test_exclusions_make_arm_not_evaluable(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch, {"raw-recon-circ": "glued", "raw-jepa-sym": "glued"})
    for r in c.run_table():
        if r["arm"] == "raw-jepa-sym" and r["phase"] == 1 and r["world"] == "main" and r["K"] == 8 and r["seed"] in (0, 1):
            f = tmp_path / r["name"] / "measures.json"
            m = json.loads(f.read_text())
            m["exclusion"]["excluded"] = True
            f.write_text(json.dumps(m))
    r = ag.aggregate()
    assert r["angle_raw_arms"]["raw-jepa-sym"]["n_not_excluded"] == 3 and r["P04"]["sym"] == "I"


def test_theta_gate_failure_counts_as_not_glued(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch, {"raw-recon-circ": "glued", "raw-jepa-sym": "glued"})
    for r in c.run_table():
        if r["arm"] == "raw-jepa-sym" and r["phase"] == 1 and r["world"] == "main" and r["K"] == 8 and r["seed"] in (0, 1):
            f = tmp_path / r["name"] / "measures.json"
            m = json.loads(f.read_text())
            m["theta_gate_n8000"]["passed"] = False
            f.write_text(json.dumps(m))
    r = ag.aggregate()
    assert r["angle_raw_arms"]["raw-jepa-sym"]["glued_rate"] == pytest.approx(3 / 9) and r["P04"]["sym"] == "F"


def test_missing_runs_inconclusive(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch, {"raw-recon-circ": "glued"})
    import shutil
    shutil.rmtree(tmp_path / "raw-recon-mse_S_s0")
    assert ag.aggregate()["P03"] == "I"


def test_pixel_p05_deferred_in_phase_1(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch, {"raw-recon-circ": "glued"})
    r = ag.aggregate()
    assert r["P05"]["pixels64-sg"]["status"] == "missing"
    assert r["verdicts"]["decision"]["updates"]["P05"].startswith("deferred")
