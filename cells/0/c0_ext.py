#!/usr/bin/env python
"""Cell 0 extension E1, the convergence check (cells/0/prereg_ext_convergence.md).

The four angle_raw arms of the main experiment, trained with the same recipe at 5x the training length
(100000 steps instead of 20000), then scored with the frozen cell 0 scorer.

    python cells/0/c0_ext.py --list                          # the run table of the extension
    python cells/0/c0_ext.py --envcheck                      # repeat one phase 1 run at 20000 steps (environment)
    python cells/0/c0_ext.py --wave S-jepa --jobs 3          # train + score one wave (restartable)
    python cells/0/c0_ext.py --summary                       # classes vs phase 1 -> result_ext_convergence.json

Orchestration only. No frozen file is modified: the training arguments come from ``c0lib.train_args`` with the
value of ``--steps`` replaced, scoring is ``c0_score.py --run`` unchanged, and the class of a run is
``c0_aggregate.run_class`` unchanged. Outputs go to runs/c0_ext100k/<name> (not under runs/c0, so the phase 1
aggregate never sees them).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c0lib as c                                   # noqa: E402

PY = sys.executable
EXT_STEPS = 100000
ARMS = ("raw-recon-mse", "raw-recon-circ", "raw-jepa-sg", "raw-jepa-sym")
SIZES = ("S", "M")                                  # S first (primary), M if compute allows
OUT = c.ROOT / "runs" / "c0_ext100k"
BASELINE = HERE / "ext_convergence_baseline.json"
RESULT = HERE / "result_ext_convergence.json"
OPPOSITE = {"torn": "glued", "glued": "torn"}
ENVCHECK_RUN = "raw-jepa-sym_S_s0"                  # one phase 1 run repeated unchanged (20000 steps)
ENVCHECK_OUT = c.ROOT / "runs" / "c0_envcheck"
# waves are sized so that one wave with 3 single-thread jobs stays well below 2 hours of wall-clock
WAVES = {
    "S-jepa": [("raw-jepa-sg", "S"), ("raw-jepa-sym", "S")],
    "S-recon": [("raw-recon-mse", "S"), ("raw-recon-circ", "S")],
    "M-sg": [("raw-jepa-sg", "M")],
    "M-sym": [("raw-jepa-sym", "M")],
    "M-mse": [("raw-recon-mse", "M")],
    "M-circ": [("raw-recon-circ", "M")],
}


def ext_table() -> list[dict]:
    """Rows of the frozen run table that the extension repeats: main role, K = 8, lambda_lat = 0.1."""
    want = {f"{arm}_{size}_s{sd}" for arm in ARMS for size in SIZES for sd in c.SEEDS}
    rows = [r for r in c.run_table() if r["name"] in want and r["role"] == "main"]
    assert len(rows) == len(want), "the frozen run table no longer contains the main angle_raw rows"
    return rows


def ext_train_args(row: dict, out: str) -> list[str]:
    """``c0lib.train_args`` with --steps replaced by EXT_STEPS; everything else is the phase 1 recipe."""
    a = c.train_args(row, out, 1, "cpu")
    i = a.index("--steps")
    assert a[i + 1] == str(c.STEPS)
    a[i + 1] = str(EXT_STEPS)
    return a


def cell_verdict(baseline_cls: str, classes: list[str]) -> str:
    """Pre-registered reading for one (arm, size) cell from the classes of its completed seeds.

    kept     >= 2 seeds have the phase 1 class
    flipped  >= 2 seeds have the opposite decided class (torn <-> glued)
    unclear  anything else with all 3 seeds present
    incomplete  fewer than 3 seeds finished
    """
    if len(classes) < len(c.SEEDS):
        return "incomplete"
    if sum(x == baseline_cls for x in classes) >= 2:
        return "kept"
    if sum(x == OPPOSITE.get(baseline_cls) for x in classes) >= 2:
        return "flipped"
    return "unclear"


def overall_verdict(cells: dict) -> str:
    """stable: every completed cell kept; transient: any cell flipped; partly_unclear otherwise; none if nothing
    is complete."""
    done = [v for v in cells.values() if v != "incomplete"]
    if not done:
        return "none"
    if any(v == "flipped" for v in done):
        return "transient"
    return "stable" if all(v == "kept" for v in done) else "partly_unclear"


def summary() -> dict:
    import c0_aggregate as A
    import c0_score as SC
    base = json.loads(BASELINE.read_text())["runs"]
    runs, cells = {}, {}
    for r in ext_table():
        d = OUT / r["name"]
        if not (d / "measures.json").exists():
            continue
        m = json.loads((d / "measures.json").read_text())
        mt = json.loads((d / "metrics.json").read_text())
        key = base[r["name"]]["val_loss_key"]
        val = [x for x in mt.get("curves", {}).get("val", []) if key in x]
        runs[r["name"]] = dict(
            cls=A.run_class(m), cls_phase1=base[r["name"]]["cls"], steps=mt.get("steps_done"),
            seam_ratio=m.get("seam", {}).get("ratio"), seam_ratio_phase1=base[r["name"]]["seam_ratio"],
            kappa=m.get("seam", {}).get("kappa"), h1_state=m.get("h1", {}).get("state"),
            theta_gate_passed=m.get("theta_gate_n8000", {}).get("passed"),
            excluded=m.get("exclusion", {}).get("excluded"),
            convergence=SC.convergence(mt, r["variant"]), conv_rel_change_phase1=base[r["name"]]["conv_rel_change"],
            val_loss_key=key, val_loss_final=(val[-1][key] if val else None),
            val_loss_final_phase1=base[r["name"]]["val_loss_final"],
            val_loss_at={str(x["step"]): x[key] for x in val if x["step"] % 20000 == 0},
            cross_cut=m.get("cross_cut"))
    for arm in ARMS:
        for size in SIZES:
            names = [f"{arm}_{size}_s{sd}" for sd in c.SEEDS]
            b = {base[n]["cls"] for n in names}
            assert len(b) == 1, "phase 1 classes of one cell are not unanimous"
            cells[f"{arm}_{size}"] = cell_verdict(b.pop(), [runs[n]["cls"] for n in names if n in runs])
    res = dict(steps=EXT_STEPS, runs=runs, cells=cells, overall=overall_verdict(cells),
               n_runs_scored=len(runs), n_runs_planned=len(ext_table()))
    RESULT.write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False))
    return res


def wave(name: str, jobs: int) -> int:
    if not (c.ROOT / "runs" / "c0" / "data_gate.json").exists():
        raise SystemExit("run `python cells/0/c0_score.py --data-gate` first")
    logs = OUT / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    rows = {r["name"]: r for r in ext_table()}
    todo = []
    for sd in c.SEEDS:                                   # seed-major order: three arms never share a seed slot
        for arm, size in WAVES[name]:
            r = rows[f"{arm}_{size}_s{sd}"]
            d = OUT / r["name"]
            if (d / "measures.json").exists():
                continue
            mt = (d / "metrics.json").read_text() if (d / "metrics.json").exists() else ""
            trained = '"final"' in mt or '"diverged": true' in mt
            parts = []
            if not trained:
                cmd = [PY, "-m", "baselines.train"] + ext_train_args(r, str(d)) + ["--overwrite"]
                parts.append(" ".join(f"'{x}'" if " " in x or "{" in x else x for x in cmd))
            # score only a finished or diverged run, as in c0_queue.py
            parts.append(f"grep -qE '\"final\"|\"diverged\": true' '{d / 'metrics.json'}' || exit 3")
            parts.append(f"{PY} {HERE / 'c0_score.py'} --run {d}")
            todo.append((r["name"], " ; ".join(parts)))
    print(f"[c0 ext] wave {name}: {len(todo)} runs", flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    running, failed = [], 0
    while todo or running:
        while todo and len(running) < jobs:
            nm, script = todo.pop(0)
            p = subprocess.Popen(["bash", "-c", script], stdout=open(logs / f"{nm}.log", "w"),
                                 stderr=subprocess.STDOUT, cwd=c.ROOT, env=env)
            running.append((nm, p, time.time()))
        for item in list(running):
            nm, p, t0 = item
            if p.poll() is not None:
                running.remove(item)
                failed += int(p.returncode != 0)
                print(f"[c0 ext] {'done' if p.returncode == 0 else 'FAILED'} {nm} {(time.time() - t0) / 60:.1f} min",
                      flush=True)
        time.sleep(5)
    print(f"[c0 ext] wave {name} finished, {failed} failed", flush=True)
    return 1 if failed else 0


def envcheck() -> int:
    """Repeat one phase 1 run with the unchanged phase 1 arguments and compare its class and seam ratio with the
    backed-up phase 1 measures. Tells whether this environment reproduces phase 1; exit code 4 if the class differs."""
    import c0_aggregate as A
    row = next(r for r in ext_table() if r["name"] == ENVCHECK_RUN)
    d = ENVCHECK_OUT / row["name"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    if not (d / "measures.json").exists():
        subprocess.run([PY, "-m", "baselines.train"] + c.train_args(row, str(d), 1, "cpu") + ["--overwrite"],
                       cwd=c.ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
        subprocess.run([PY, str(HERE / "c0_score.py"), "--run", str(d)], cwd=c.ROOT, env=env, check=True,
                       stdout=subprocess.DEVNULL)
    m = json.loads((d / "measures.json").read_text())
    mt = json.loads((d / "metrics.json").read_text())
    b = json.loads(BASELINE.read_text())["runs"][row["name"]]
    val = [x[b["val_loss_key"]] for x in mt["curves"]["val"] if b["val_loss_key"] in x]
    out = dict(run=row["name"], steps=mt.get("steps_done"), cls=A.run_class(m), cls_phase1=b["cls"],
               seam_ratio=m["seam"]["ratio"], seam_ratio_phase1=b["seam_ratio"],
               val_loss_final=val[-1], val_loss_final_phase1=b["val_loss_final"])
    out["same_class"] = out["cls"] == out["cls_phase1"]
    (ENVCHECK_OUT / "envcheck.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return 0 if out["same_class"] else 4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--envcheck", action="store_true")
    ap.add_argument("--wave", choices=list(WAVES))
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for r in ext_table():
            print(r["name"], " ".join(ext_train_args(r, str(OUT / r["name"]))))
        return 0
    if a.envcheck:
        return envcheck()
    if a.wave:
        return wave(a.wave, a.jobs)
    if a.summary:
        res = summary()
        print(json.dumps(dict(cells=res["cells"], overall=res["overall"], n_runs_scored=res["n_runs_scored"]), indent=1))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
