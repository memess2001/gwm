#!/usr/bin/env python
"""Run queue for cell 6 (orchestration only; not part of the frozen scoring code).

    python cells/6/c6_queue.py --steps 20000 --jobs 4            # all 21 runs (v1.4, M only): train + gate, then score
    python cells/6/c6_queue.py --steps 20000 --jobs 4 --dry-run

Order: main conditions before data-efficiency runs. Each job is single-threaded (OpenMP
oversubscription slows small models 10-100x here). Skips runs whose gate.json / scores.json already exist, so it
can be restarted. Logs go to runs/c6/_logs/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c6lib as c                                  # noqa: E402
from c6_train import run_dir                       # noqa: E402

PY = sys.executable


def run_list():
    out = []
    for size in c.SIZES:
        for cond in ("H-act", "C-open", "C-vis", "H-pc", "H-pn"):
            for seed in (0, 1, 2):
                out.append((cond, size, seed, c.N_TRAIN_MAIN))
        for n in c.DATA_EFF_N:
            for seed in (0, 1, 2):
                out.append(("H-act", size, seed, n))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=c.MAIN_STEPS)        # v1.4 review: must equal MAIN_STEPS
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--jobs-file", help="if this file exists, its integer overrides --jobs (re-read every 5 s)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.steps != c.MAIN_STEPS:
        raise SystemExit(f"--steps {a.steps} != MAIN_STEPS {c.MAIN_STEPS} (prereg section 3); the runs would be INCOMPLETE")
    logs = c.ROOT / "runs" / "c6" / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    todo = []
    for cond, size, seed, n in run_list():
        d = run_dir(cond, size, seed, n, False)
        cmd = []
        if not (d / "gate.json").exists():
            cmd.append([PY, str(HERE / "c6_train.py"), "--cond", cond, "--size", size, "--seed", str(seed),
                        "--steps", str(a.steps), "--n-train", str(n), "--threads", "1", "--overwrite"])
        if not (d / "scores.json").exists():
            cmd.append([PY, str(HERE / "c6_score.py"), "--run", str(d)])
        if cmd:
            todo.append((d.name, cmd))
    print(f"[queue] {len(todo)} runs to do", flush=True)
    if a.dry_run:
        for name, cmd in todo:
            print(name, [" ".join(x[1:3]) for x in cmd])
        return 0
    running = []
    env = {**__import__("os").environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    jobs = a.jobs
    while todo or running:
        if a.jobs_file and Path(a.jobs_file).exists():
            try:
                new = int(Path(a.jobs_file).read_text().strip())
                if new != jobs:
                    print(f"[queue] jobs {jobs} -> {new}", flush=True)
                    jobs = new
            except ValueError:
                pass
        while todo and len(running) < jobs:
            name, cmds = todo.pop(0)
            script = " && ".join(" ".join(x) for x in cmds)
            log = open(logs / f"{name}.log", "w")
            p = subprocess.Popen(["bash", "-c", script], stdout=log, stderr=subprocess.STDOUT, cwd=c.ROOT, env=env)
            running.append((name, p, time.time()))
            print(f"[queue] start {name}", flush=True)
        for item in list(running):
            name, p, t0 = item
            if p.poll() is not None:
                running.remove(item)
                print(f"[queue] {'done' if p.returncode == 0 else 'FAILED'} {name} rc={p.returncode} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
        time.sleep(5)
    print("[queue] all finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
