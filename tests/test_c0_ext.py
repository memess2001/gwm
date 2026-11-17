"""Cell 0 extension E1 (convergence check): the run table, the training arguments and the pre-registered reading.
No model is trained here."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cells" / "0"))
import c0_ext as E      # noqa: E402
import c0lib as c       # noqa: E402


def test_table_is_the_main_angle_raw_rows():
    rows = E.ext_table()
    assert len(rows) == 4 * 2 * 3
    assert {r["obs"] for r in rows} == {"angle_raw"}
    assert {r["K"] for r in rows} == {c.K_MAIN} and {r["role"] for r in rows} == {"main"}
    assert all(r["lambda_lat"] == (0.1 if r["variant"] == "recon" else 0.0) for r in rows)


def test_only_the_step_count_differs_from_phase_1():
    for r in E.ext_table():
        a, b = c.train_args(r, "out", 1, "cpu"), E.ext_train_args(r, "out")
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        assert len(a) == len(b) and diff == [(str(c.STEPS), str(E.EXT_STEPS))]
    assert E.EXT_STEPS == 5 * c.STEPS


def test_waves_cover_every_run_once():
    cells = [x for w in E.WAVES.values() for x in w]
    assert sorted(cells) == sorted((arm, size) for arm in E.ARMS for size in E.SIZES)


def test_cell_reading():
    assert E.cell_verdict("torn", ["torn", "torn", "glued"]) == "kept"
    assert E.cell_verdict("torn", ["glued", "glued", "torn"]) == "flipped"
    assert E.cell_verdict("glued", ["torn", "torn", "inconclusive"]) == "flipped"
    assert E.cell_verdict("glued", ["glued", "inconclusive", "torn"]) == "unclear"
    assert E.cell_verdict("torn", ["inconclusive", "excluded", "not_carrying_theta"]) == "unclear"
    assert E.cell_verdict("torn", ["torn", "torn"]) == "incomplete"


def test_overall_reading():
    assert E.overall_verdict({"a": "kept", "b": "kept", "c": "incomplete"}) == "stable"
    assert E.overall_verdict({"a": "kept", "b": "flipped"}) == "transient"
    assert E.overall_verdict({"a": "kept", "b": "unclear"}) == "partly_unclear"
    assert E.overall_verdict({"a": "incomplete"}) == "none"
