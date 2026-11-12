#!/usr/bin/env python
"""Run queue for cell 0 v2 (orchestration only).

    python cells/0/c0_queue.py --phase 1 --jobs 3                  # cloud: 3 single-thread jobs
    python cells/0/c0_queue.py --phase 1 --jobs 6 --device mps     # Mac mini (MPS for training; scoring on CPU)
    python cells/0/c0_queue.py --phase 2 --jobs 2 --device mps
    python cells/0/c0_queue.py --phase 1 --dry-run

Phase 1 = angle S/M/L, pixels S, angle_cs negative control (M), K = 1 (M), lambda_lat = 0 (M), untrained-encoder
nulls; phase 2 = XL and pixels M/L/XL incl. the pixel negative control. Requires runs/c0/data_gate.json
(``python cells/0/c0_score.py --data-gate``). Restartable: skips runs whose measures.json exists.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c0lib as c                                   # noqa: E402

PY = sys.executable


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2], required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    runs = c.ROOT / "runs" / "c0"
    if not (runs / "data_gate.json").exists():
        raise SystemExit("run `python cells/0/c0_score.py --data-gate` first")
    logs = runs / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    todo = []
    for r in c.run_table():
        if r["phase"] != a.phase:
            continue
        d = runs / r["name"]
        if (d / "measures.json").exists():
            continue
        cmds = []
        mt = (d / "metrics.json").read_text() if (d / "metrics.json").exists() else ""
        trained = '"final"' in mt or '"diverged": true' in mt          # a diverged run is not retrained
        dev = a.device
        if r["role"] == "device_control":                               # the same config on the other device
            dev = "cpu" if a.device != "cpu" else "mps"
        if not trained:
            cmds.append([PY, "-m", "baselines.train"] + c.train_args(r, str(d), 1, dev) + ["--overwrite"])
        cmds.append([PY, str(HERE / "c0_score.py"), "--run", str(d)])
        todo.append((r["name"], cmds))
    if a.phase == 1:
        for n in c.null_table():
            if not (runs / n["name"] / "measures.json").exists():
                todo.append((n["name"], [[PY, str(HERE / "c0_score.py"), "--null", n["obs"], "--seed", str(n["seed"])]]))
    print(f"[c0 queue] phase {a.phase}: {len(todo)} jobs", flush=True)
    if a.dry_run:
        for name, _ in todo:
            print(name)
        return 0
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    running = []
    while todo or running:
        while todo and len(running) < a.jobs:
            name, cmds = todo.pop(0)
            script = " ; ".join(" ".join(f"'{x}'" if " " in x or "{" in x else x for x in cmd) for cmd in cmds)
            p = subprocess.Popen(["bash", "-c", script], stdout=open(logs / f"{name}.log", "w"),
                                 stderr=subprocess.STDOUT, cwd=c.ROOT, env=env)
            running.append((name, p, time.time()))
        for item in list(running):
            name, p, t0 = item
            if p.poll() is not None:
                running.remove(item)
                print(f"[c0 queue] {'done' if p.returncode == 0 else 'FAILED'} {name} {(time.time() - t0) / 60:.1f} min",
                      flush=True)
        time.sleep(5)
    print("[c0 queue] all finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
