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
            "theta_gate_n2000": {"passed": theta_ok},
            "exclusion": {"excluded": excluded}, "h1_none_high_kappa": h1 == "none" and kappa > 5,
            "seam": {"ratio": seam, "kappa": kappa}, "pointwise": {"holds": pointwise},
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
    assert r["verdicts"]["decision"]["updates"]["cell0.notes.mechanism"] == {"sg": "继承切口（撕开率 >= 2/3）",
                                                                              "sym": "粘合"}


def test_everything_inherits(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch, {})                          # every angle_raw arm torn
    r = ag.aggregate()
    assert r["P03"] == "F" and r["P04"] == {"sg": "F", "sym": "F"}
    u = r["verdicts"]["decision"]["updates"]
    assert "cell0.status" not in u and u["cell0.confidence.exists"].startswith("high -> mid")
    assert u["cell0.notes.mechanism"] == {"sg": "继承切口（撕开率 >= 2/3）", "sym": "继承切口（撕开率 >= 2/3）"}
    assert u["cell4.confidence.yields"].startswith("up")


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
    assert r["angle_raw_arms"]["raw-jepa-sym"]["n_decided"] == 3 and r["P04"]["sym"] == "I"


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


def test_all_inconclusive_is_I_not_F(tmp_path, monkeypatch):            # v2.1 review B1
    _world(tmp_path, monkeypatch, {a: "inconclusive" for a in ("raw-recon-mse", "raw-recon-circ", "raw-jepa-sg",
                                                                "raw-jepa-sym")})
    r = ag.aggregate()
    assert r["P03"] == "I" and r["P04"] == {"sg": "I", "sym": "I"}
    assert r["verdicts"]["decision"]["state"] == "PARTIAL" and "cell0.status" not in r["verdicts"]["decision"]["updates"]


def test_theta_gate_failure_with_high_kappa_is_inconclusive(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch, {"raw-recon-circ": "glued"})
    for r in c.run_table():
        if r["arm"] == "raw-jepa-sym" and r["phase"] == 1 and r["world"] == "main" and r["K"] == 8:
            f = tmp_path / r["name"] / "measures.json"
            m = json.loads(f.read_text())
            m["theta_gate_n8000"]["passed"] = False
            m["theta_gate_n8000"]["r2_min"] = 0.93                 # omega-stretched, not folded (v2.2)
            m["seam"]["kappa"] = 8.0
            f.write_text(json.dumps(m))
    assert ag.aggregate()["angle_raw_arms"]["raw-jepa-sym"]["n_decided"] == 0


def test_missing_negative_control_gives_no_verdict(tmp_path, monkeypatch):   # v2.1 review B2
    _world(tmp_path, monkeypatch, {"raw-recon-circ": "glued"})
    import shutil
    for d in tmp_path.glob("*_neg"):
        shutil.rmtree(d)
    assert ag.aggregate()["verdicts"]["decision"]["state"] == "INSTRUMENT_INSUFFICIENT"


def test_p05_ignores_excluded_runs(tmp_path, monkeypatch):               # v2.1 review B5
    _world(tmp_path, monkeypatch, {"raw-recon-circ": "glued"})
    rs = [r for r in c.run_table() if r["arm"] == "cs-jepa-sg" and r["size"] in ("M", "L") and r["world"] == "main"
          and r["K"] == 8]
    for r in rs[:2]:
        f = tmp_path / r["name"] / "measures.json"
        m = json.loads(f.read_text())
        m["exclusion"]["excluded"] = True
        f.write_text(json.dumps(m))
    v = ag.aggregate()["P05"]["angle_cs-sg"]
    assert v["status"] == "I" and v["decided"] == 4


def test_fold_with_high_kappa_stays_decided(tmp_path, monkeypatch):     # v2.2 (second review R1)
    """A latent folded onto conserved quantities has a large kappa-hat AND a very low theta R^2: it must stay in
    the denominator as not_carrying_theta; only an omega-stretched latent (r2_min >= 0.8) is inconclusive."""
    m = {"exclusion": {"excluded": False}, "theta_gate_n8000": {"passed": False, "r2_min": -0.1},
         "seam": {"kappa": 5.7}, "consistency": "inconclusive"}
    assert ag.run_class(m) == "not_carrying_theta"
    m["theta_gate_n8000"]["r2_min"] = 0.93
    assert ag.run_class(m) == "inconclusive"
    m["seam"]["kappa"] = float("nan")
    assert ag.run_class(m) == "inconclusive"


def test_diverged_stub_does_not_crash(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch, {"raw-recon-circ": "glued"})
    f = tmp_path / "raw-jepa-sym_L_s2" / "measures.json"
    f.write_text(json.dumps({"exclusion": {"excluded": True, "learned": False, "reason": "diverged"}}))
    r = ag.aggregate()
    assert "excluded" in r["P07"]["raw-jepa-sym"]["L"]
