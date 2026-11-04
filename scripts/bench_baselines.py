"""CPU benchmark of the baseline world models -> results/bench_cpu.md (report in Simplified Chinese).

    .venv/bin/python scripts/bench_baselines.py                                   # full benchmark, ~1 h on 4 cores
    .venv/bin/python scripts/bench_baselines.py --quick --out /tmp/b.md           # small sweep, tests the script
    .venv/bin/python scripts/bench_baselines.py --from-json results/bench_cpu.md  # re-render the report only
The report embeds all raw results as one JSON block, so ``--from-json`` accepts the report itself (or a --json file);
``--notes FILE`` adds hand-written observations (Simplified Chinese) that are stored in that block and survive
re-rendering.

Measures seconds per training step at batch 128 - one step = batch gather + K-step loss + backward + grad clip +
AdamW step, exactly the loop body of ``baselines.train`` - for the configurations of ``BENCHES``:
    pendulum     angle_cs / angle_raw / pixels64 x recon / jepa-sym / jepa-sg x S / M / L / XL  (history 2, K 8)
    pendulum     angle_cs / angle_raw, K = 1 (cell 0 side factor), size M, the same three variants
    toggle_grid  grid obs, recon, whole-episode open-loop window (anchors="episode", history 1, K = 63), S / M / L / XL
each at 4 threads and at 1 thread, plus 4 CONCURRENT single-thread jobs for representative configurations (the
"many single-thread runs in parallel" plan: measures the slowdown from sharing caches / memory bandwidth).
angle_raw recon-circular costs the same as recon-mse (one cos per element) and is not timed separately.

Quiet-machine protocol (other agents share this machine): before the sweep the script polls every ``--poll``
seconds (default 60, at most ``--max-wait-quiet`` = 30 min) until the 1-minute load average is below
``--max-load`` (1.5) and no other python process uses >= 0.5 core (/proc on Linux, ``ps`` elsewhere). Before
every configuration it checks again for heavy python processes and, if one is found, runs the same polling wait
(this process is idle while it waits, so the load criterion is fair). All waits together are capped by
``--max-total-wait`` (90 min); after that the sweep measures without waiting and the report flags every
measurement during which other processes had runnable threads (column 他人线程). Every measurement records the 1-minute load
average just before it and during it (mean / max; it INCLUDES this process's own threads) and the number of
runnable threads of OTHER processes (the contamination). The sweep runs ``--passes`` times; each configuration
reports its fastest pass (median step time within a pass). Projections for 10k / 20k steps and a 12-hour budget
plan for the pre-registered runs of cells 0 and 6 are written to the report.

Only synthetic training steps on freshly sampled data are run; no model is trained to convergence and no latent is
analysed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.data import WindowDataset, make_splits          # noqa: E402
from baselines.mlp_wm import SIZES, build_wm, describe, train_flops_per_sample   # noqa: E402
from baselines.train import make_optimizer, setup_runtime, use_channels_last     # noqa: E402

VARIANTS = ("recon", "jepa-sym", "jepa-sg")
ALL_SIZES = tuple(SIZES)
FLAG_HOURS = 12.0
# name -> data / model settings, variants and sizes measured.  "required" marks what the task asked for; the rest
# (toggle_grid L / XL, K = 1) is measured because the budget plan needs it.
BENCHES = {
    "angle_cs": dict(world="pendulum", obs="angle_cs", horizon=64, history=2, K=8, anchors="all",
                     variants=VARIANTS, sizes=ALL_SIZES, label="pendulum angle_cs"),
    "angle_raw": dict(world="pendulum", obs="angle_raw", horizon=64, history=2, K=8, anchors="all",
                      variants=VARIANTS, sizes=ALL_SIZES, label="pendulum angle_raw"),
    "pixels64": dict(world="pendulum", obs="pixels64", horizon=64, history=2, K=8, anchors="all",
                     variants=VARIANTS, sizes=ALL_SIZES, label="pendulum pixels64"),
    "toggle_ep63": dict(world="toggle_grid", obs="grid", horizon=63, history=1, K=None, anchors="episode",
                        variants=("recon",), sizes=ALL_SIZES, label="toggle_grid grid，整段开环 K = 63"),
    "angle_cs_K1": dict(world="pendulum", obs="angle_cs", horizon=64, history=2, K=1, anchors="all",
                        variants=VARIANTS, sizes=("M",), label="pendulum angle_cs，K = 1"),
    "angle_raw_K1": dict(world="pendulum", obs="angle_raw", horizon=64, history=2, K=1, anchors="all",
                         variants=VARIANTS, sizes=("M",), label="pendulum angle_raw，K = 1"),
}
# 4 concurrent single-thread jobs of the same configuration (representative of each family and size range)
GUARD_TRIGGERS = [0]          # how often a heavy python process forced a quiet-machine wait mid-sweep
WAITED = [0.0]                # seconds spent in quiet-machine waits so far (capped by --max-total-wait)
DISTURBED = 0.5               # report flag: other processes' mean runnable threads during a measurement above this
PARALLEL = (("angle_cs", "recon", "S"), ("angle_cs", "recon", "M"), ("angle_cs", "recon", "L"),
            ("angle_cs", "recon", "XL"), ("angle_raw", "jepa-sym", "M"), ("angle_cs", "jepa-sg", "M"),
            ("pixels64", "recon", "S"), ("pixels64", "jepa-sym", "S"), ("pixels64", "recon", "M"),
            ("toggle_ep63", "recon", "S"), ("toggle_ep63", "recon", "M"))


# ---------------------------------------------------------------------- load monitoring
def other_running_threads(own_pid: int) -> Optional[int]:
    """Runnable threads (state R) of processes other than this one; None where /proc is unavailable."""
    if not os.path.isdir("/proc"):
        return None
    n = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == own_pid:
            continue
        try:
            for tid in os.listdir(f"/proc/{pid}/task"):
                with open(f"/proc/{pid}/task/{tid}/stat", "rb") as f:
                    if f.read().rsplit(b")", 1)[1].split()[0] == b"R":
                        n += 1
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, OSError):
            continue
    return n


def loadavg() -> Optional[float]:
    return os.getloadavg()[0] if hasattr(os, "getloadavg") else None


def _proc_cpu_times() -> dict:
    """pid -> (utime + stime in clock ticks, command line) for every readable process (Linux /proc)."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                fields = f.read().rsplit(b")", 1)[1].split()
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            out[int(pid)] = (int(fields[11]) + int(fields[12]), cmd)      # utime, stime (fields 14, 15 of stat)
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError, OSError):
            continue
    return out


def heavy_processes(min_cpu: float = 0.5, interval: float = 1.0, exclude: tuple = ()) -> list[dict]:
    """Processes other than this one (and ``exclude``) using >= ``min_cpu`` cores, averaged over ``interval``
    seconds. Linux: /proc; elsewhere ``ps`` (instantaneous %CPU; Windows: empty list, use the load average)."""
    own = {os.getpid(), *exclude}
    if os.path.isdir("/proc"):
        tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        a = _proc_cpu_times()
        t0 = time.monotonic()
        time.sleep(interval)
        b = _proc_cpu_times()
        dt_s = max(time.monotonic() - t0, 1e-3)
        res = []
        for pid, (cpu1, cmd) in b.items():
            if pid in own or pid not in a:
                continue
            c = (cpu1 - a[pid][0]) / tick / dt_s
            if c >= min_cpu:
                res.append({"pid": pid, "cpu": round(c, 2), "cmd": cmd[:120], "python": "python" in cmd.lower()})
        return sorted(res, key=lambda r: -r["cpu"])
    try:
        txt = subprocess.run(["ps", "-A", "-o", "pid=,%cpu=,command="], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    res = []
    for line in txt.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, c = int(parts[0]), float(parts[1]) / 100.0
        except ValueError:
            continue
        if pid not in own and c >= min_cpu:
            res.append({"pid": pid, "cpu": round(c, 2), "cmd": parts[2][:120], "python": "python" in parts[2].lower()})
    return sorted(res, key=lambda r: -r["cpu"])


def wait_for_quiet_machine(max_load: float = 1.5, poll: float = 60.0, max_wait: float = 1800.0,
                           log: Callable = print, load_fn: Callable = loadavg, heavy_fn: Callable = heavy_processes,
                           sleep_fn: Callable = time.sleep, clock: Callable = time.monotonic) -> dict:
    """Poll every ``poll`` s, at most ``max_wait`` s, until the 1-minute load < ``max_load`` and no other python
    process uses >= 0.5 core. Returns {"quiet", "waited_sec", "load_start", "load_end", "polls", "heavy_seen"}.
    ``quiet`` is False if the limit was hit (the caller measures anyway and the report flags it)."""
    t0, polls, seen = clock(), 0, {}
    load_start = load_fn()
    while True:
        load, heavy = load_fn(), heavy_fn()
        py = [h for h in heavy if h["python"]]
        for h in heavy:
            seen.setdefault(h["pid"], h)
        ok = (load is None or load < max_load) and not py
        waited = clock() - t0
        if ok or waited >= max_wait:
            if not ok:
                log(f"[quiet] still busy after {waited:.0f}s (load {load}, heavy python {len(py)}): measuring anyway")
            return {"quiet": bool(ok), "waited_sec": float(waited), "load_start": load_start, "load_end": load,
                    "polls": polls, "heavy_seen": list(seen.values())}
        polls += 1
        names = ", ".join(f"pid {h['pid']} {h['cpu']:.1f} core {h['cmd'][:50]}" for h in py) or "none"
        log(f"[quiet] load {load if load is None else round(load, 2)} (need < {max_load}); heavy python: {names}; "
            f"waiting {poll:.0f}s ({waited:.0f}s / {max_wait:.0f}s)")
        sleep_fn(min(poll, max(1.0, max_wait - waited)))


class Sampler:
    """Samples load average and other processes' runnable threads in a background thread."""

    def __init__(self, period: float = 0.5):
        self.period, self.pid, self.load, self.other = period, os.getpid(), [], []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.load.append(loadavg())
            self.other.append(other_running_threads(self.pid))
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self) -> dict:
        ld = [x for x in self.load if x is not None]
        ot = [x for x in self.other if x is not None]
        return {"load_mean": float(np.mean(ld)) if ld else None, "load_max": float(np.max(ld)) if ld else None,
                "other_running_mean": float(np.mean(ot)) if ot else None}


def settle(max_threads: int, max_wait: float) -> float:
    """Short settle before a measurement: other processes' runnable threads <= ``max_threads`` on 5 consecutive
    checks 0.2 s apart (at most ``max_wait`` s); returns seconds waited."""
    t0, pid, ok = time.monotonic(), os.getpid(), 0
    while time.monotonic() - t0 < max_wait:
        n = other_running_threads(pid)
        if n is None:
            return 0.0
        ok = ok + 1 if n <= max_threads else 0
        if ok >= 5:
            break
        time.sleep(0.2)
    return time.monotonic() - t0


# ---------------------------------------------------------------------- measurement
def make_data(bench: str, n_traj: int = 64):
    b = BENCHES[bench]
    spec, sp = make_splits(b["world"], b["obs"], b["horizon"], {"train": n_traj}, b.get("world_kwargs"))
    return spec, WindowDataset(sp["train"], b["history"], b["K"], spec["action"], anchors=b["anchors"])


def build(spec, bench: str, variant: str, size: str, K: int, device=torch.device("cpu")):
    torch.manual_seed(0)
    m = build_wm(spec, variant, size, history=BENCHES[bench]["history"], K=K)
    m.to(device)
    m.set_channels_last(use_channels_last("auto", device, m))
    return m


def time_steps(model, ds, batch, min_time, min_steps, max_steps, warmup, sync: Optional[Callable] = None):
    opt = make_optimizer(model, 1e-3, 1e-4)
    rng = np.random.default_rng(0)

    def step():
        b = ds.sample_batch(rng, batch)
        loss, _ = model.loss(b["frames"], b["actions"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(warmup):
        step()
    if sync is not None:
        sync()
    times, t0 = [], time.perf_counter()
    while len(times) < max_steps and (len(times) < min_steps or time.perf_counter() - t0 < min_time):
        t = time.perf_counter()
        step()
        times.append(time.perf_counter() - t)
    return times


def configs(benches, sizes=None):
    for bench in benches:
        b = BENCHES[bench]
        for v in b["variants"]:
            for s in b["sizes"]:
                if sizes is None or s in sizes:
                    yield bench, v, s


def guard(args, log=print) -> Optional[dict]:
    """Per-configuration check: if another python process is heavy, wait with the full protocol."""
    heavy = [h for h in heavy_processes(interval=0.5) if h["python"]]
    if not heavy:
        return None
    GUARD_TRIGGERS[0] += 1
    budget_left = max(0.0, args.max_total_wait - WAITED[0])
    log(f"[quiet] heavy python process(es) appeared: {[(h['pid'], h['cpu']) for h in heavy]}; "
        f"wait budget left {budget_left:.0f}s")
    r = wait_for_quiet_machine(args.max_load, args.poll, min(args.max_wait_quiet, budget_left), log=log)
    WAITED[0] += r["waited_sec"]
    return r


def _disturbed(r: dict) -> bool:
    return (r.get("other_running_mean") or 0.0) > DISTURBED


def sweep(threads, cfgs, data, args, tag):
    """Time every configuration; a measurement during which other processes had runnable threads is repeated
    (after the quiet-machine check) up to ``args.retries`` times while the wait budget lasts; the fastest
    attempt is kept and the number of attempts recorded."""
    torch.set_num_threads(threads)
    res = {}
    for bench, variant, size in cfgs:
        spec, ds = data[bench]
        model = build(spec, bench, variant, size, ds.K)
        best = None
        for attempt in range(1 + getattr(args, "retries", 0)):
            q = guard(args)
            waited = settle(args.settle_threads, args.settle_wait)
            load_before = loadavg()
            with Sampler() as smp:
                ts = time_steps(model, ds, args.batch, args.min_time, args.min_steps, args.max_steps, args.warmup)
            r = {"sec_per_step": float(np.median(ts)), "sec_mean": float(np.mean(ts)), "n_steps": len(ts),
                 "settle_sec": waited, "load_before": load_before, **smp.summary(),
                 "quiet_wait": None if q is None else {k: q[k] for k in ("quiet", "waited_sec", "load_end")},
                 "attempts": attempt + 1}
            print(f"[{tag} t={threads}] {bench:12s} {variant:8s} {size:2s} {r['sec_per_step']:.4f} s/step "
                  f"(n={len(ts)}, load {fmt_opt(load_before)} -> {fmt_opt(r['load_mean'])}, "
                  f"other runnable {fmt_opt(r['other_running_mean'], '.1f')})"
                  + (" DISTURBED" if _disturbed(r) else ""), flush=True)
            if best is None or r["sec_per_step"] < best["sec_per_step"]:
                best = r
            best["attempts"] = attempt + 1
            if not _disturbed(r) or WAITED[0] >= args.max_total_wait:
                break
        res[(bench, variant, size)] = best
        del model
    return res


def _parallel_worker(key, a: dict, barrier, queue):
    """One of N concurrent single-thread jobs (spawned process): warm up, wait for all, time, report."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    bench, variant, size = key
    spec, ds = make_data(bench)
    model = build(spec, bench, variant, size, ds.K)
    try:
        ts = time_steps(model, ds, a["batch"], a["min_time"], a["min_steps"], a["max_steps"], a["warmup"],
                        sync=lambda: barrier.wait(timeout=1200))
        queue.put((os.getpid(), ts, None))
    except Exception as e:                          # report instead of hanging the parent
        queue.put((os.getpid(), [], repr(e)))


def parallel_sweep(keys, n_jobs, args, tag):
    ctx = mp.get_context("spawn")                   # fork after OpenMP init is unsafe; spawn is portable
    a = {k: getattr(args, k) for k in ("batch", "min_time", "min_steps", "max_steps", "warmup")}
    a["min_time"] = max(a["min_time"], 2 * args.min_time)
    res = {}
    for key in keys:
        q = guard(args)
        barrier, queue = ctx.Barrier(n_jobs), ctx.Queue()
        procs = [ctx.Process(target=_parallel_worker, args=(key, a, barrier, queue)) for _ in range(n_jobs)]
        load_before = loadavg()
        with Sampler() as smp:
            for p in procs:
                p.start()
            out = [queue.get(timeout=3600) for _ in procs]
            for p in procs:
                p.join()
        errs = [e for _, _, e in out if e]
        if errs:
            print(f"[{tag}] {key}: worker error {errs[0]}", flush=True)
            continue
        med = [float(np.median(ts)) for _, ts, _ in out]
        r = {"sec_per_step": float(np.mean(med)), "sec_per_step_jobs": med, "n_jobs": n_jobs,
             "load_before": load_before, **smp.summary(),
             "quiet_wait": None if q is None else {k: q[k] for k in ("quiet", "waited_sec", "load_end")}}
        res[key] = r
        print(f"[{tag} {n_jobs}x1] {key[0]:12s} {key[1]:8s} {key[2]:2s} {r['sec_per_step']:.4f} s/step per job "
              f"(jobs {[round(m, 4) for m in med]}, load {fmt_opt(r['load_mean'])})", flush=True)
    return res


def data_gen_seconds(benches) -> dict:
    """Wall time to sample one run's data (train 2000 + val 200 trajectories) per (world, obs, horizon)."""
    out = {}
    for bench in benches:
        b = BENCHES[bench]
        key = f"{b['world']}/{b['obs']}/T{b['horizon']}"
        if key in out:
            continue
        t0 = time.perf_counter()
        _, sp = make_splits(b["world"], b["obs"], b["horizon"], {"train": 2000, "val": 200})
        out[key] = {"sec": time.perf_counter() - t0, "mb": sum(t.obs.nbytes for t in sp.values()) / 2 ** 20}
        del sp
    return out


# ---------------------------------------------------------------------- budget plan (cells 0 and 6)
def plan_groups() -> list[dict]:
    """Pre-registered runs (cells/0/prereg.md sections 2 and 6, cells/6/prereg.md section 7), counted with the jepa
    split of cell 0 section 0b: every jepa run of the drafts becomes one jepa-sg AND one jepa-sym run.
    Each group: cell, name, size, list of (bench, variant) per seed, seeds."""
    g = []
    cs = [("angle_cs", v) for v in VARIANTS]
    raw = [("angle_raw", "recon"), ("angle_raw", "recon"), ("angle_raw", "jepa-sg"), ("angle_raw", "jepa-sym")]
    pix = [("pixels64", v) for v in VARIANTS]
    for s in ALL_SIZES:
        g.append(dict(cell="0", name="angle 主实验（angle_cs 3 变体 + angle_raw 4 变体）", size=s, runs=cs + raw))
    g.append(dict(cell="0", name="angle_cs 负对照", size="M", runs=cs))
    g.append(dict(cell="0", name="副因子 K = 1（angle_cs / angle_raw × 3 变体）", size="M",
                  runs=[("angle_cs_K1", v) for v in VARIANTS] + [("angle_raw_K1", v) for v in VARIANTS]))
    for s in ALL_SIZES:
        g.append(dict(cell="0", name="像素主实验（3 变体）", size=s, runs=pix))
    g.append(dict(cell="0", name="像素负对照", size="M", runs=pix))
    for s in ALL_SIZES:
        n = 3 + 1 + 2 + (1 + 2 if s in ("S", "M") else 0)      # tiers, switch-invisible, 2 neg. controls, abelian, bits
        g.append(dict(cell="6", name="格 6（toggle_grid，整段开环）", size=s, runs=[("toggle_ep63", "recon")] * n))
    for x in g:
        x["seeds"] = 3
    return g


def parallel_factor(par: dict, best1: dict, bench: str, size: str) -> tuple[float, str]:
    """Slowdown of one single-thread job when 4 run concurrently (measured, else nearest measured size of the same
    data family, else the median of all measurements)."""
    fam = lambda b: "pixels64" if b.startswith("pixels") else ("toggle" if b.startswith("toggle") else "angle")
    ratios = {}
    for k, r in par.items():
        if k in best1:
            ratios[k] = r["sec_per_step"] / best1[k]["sec_per_step"]
    if not ratios:
        return 1.0, "未测"
    exact = [v for k, v in ratios.items() if k[0] == bench and k[2] == size]
    if exact:
        return float(np.mean(exact)), "实测"
    same = [(abs(ALL_SIZES.index(k[2]) - ALL_SIZES.index(size)), v) for k, v in ratios.items() if fam(k[0]) == fam(bench)]
    if same:
        dmin = min(d for d, _ in same)
        return float(np.mean([v for d, v in same if d == dmin])), "同族最近尺寸"
    return float(np.median(list(ratios.values()))), "全体中位数"


def run_cost(best4, best1, par, bench, variant, size, steps) -> dict:
    """Wall-clock hours of one run and amortised machine-hours in the better of the two modes:
    'par' = single-thread job, 4 jobs side by side (amortised cost = job time / 4); 'seq' = 4 threads, one job."""
    k = (bench, variant, size)
    out = {}
    if k in best4:
        h = best4[k]["sec_per_step"] * steps / 3600
        out["seq"] = {"job_h": h, "machine_h": h}
    if k in best1:
        f, src = parallel_factor(par, best1, bench, size)
        h = best1[k]["sec_per_step"] * f * steps / 3600
        out["par"] = {"job_h": h, "machine_h": h / 4, "factor": f, "factor_src": src}
    if not out:
        return {}
    mode = min(out, key=lambda m: out[m]["machine_h"])
    return {"mode": mode, **out[mode], "alt": out}


def budget(best4, best1, par, steps, hours=FLAG_HOURS, cons: Optional[tuple] = None) -> list[dict]:
    """Group costs, then fill the local budget with the cheapest groups first (most runs done here; the largest
    models go to the Mac). A group with a single job longer than ``hours`` never stays local. ``cons`` =
    (slow4, slow1): conservative step times (slowest undisturbed pass); when given, the fill uses them, so noise
    in the measurements cannot push the local plan over ``hours``. Each run keeps the mode chosen on best times."""
    slow4, slow1 = cons if cons is not None else (best4, best1)
    rows = []
    for g in plan_groups():
        costs = [run_cost(best4, best1, par, b, v, g["size"], steps) for b, v in g["runs"]]
        if any(not c for c in costs):
            continue
        slow = [run_cost(slow4, slow1, par, b, v, g["size"], steps) for b, v in g["runs"]]
        slow = [s_["alt"].get(c["mode"], s_) for s_, c in zip(slow, costs)]          # same mode as the best case
        n = len(g["runs"]) * g["seeds"]
        rows.append({**g, "n": n, "machine_h": sum(c["machine_h"] for c in costs) * g["seeds"],
                     "job_max_h": max(c["job_h"] for c in costs), "modes": sorted({c["mode"] for c in costs}),
                     "machine_h_cons": sum(c["machine_h"] for c in slow) * g["seeds"],
                     "job_max_h_cons": max(c["job_h"] for c in slow)})
    used = 0.0
    for r in sorted(rows, key=lambda r: r["machine_h_cons"]):
        if r["job_max_h_cons"] <= hours and used + r["machine_h_cons"] <= hours:
            used += r["machine_h_cons"]
            r["local"] = True
        else:
            r["local"] = False
        r["cum_local_h"] = used if r["local"] else None
    return rows


def worst_of(passes: list[dict]) -> dict:
    """Per configuration, the SLOWEST undisturbed measurement (conservative planning; the fastest measurement if
    every one was disturbed)."""
    keys = dict.fromkeys(k for p in passes for k in p)
    out = {}
    for k in keys:
        rs = [p[k] for p in passes if k in p]
        clean = [r for r in rs if not _disturbed(r)]
        out[k] = max(clean, key=lambda r: r["sec_per_step"]) if clean else min(rs, key=lambda r: r["sec_per_step"])
    return out


def passes_from_raw(meta: dict, threads: int) -> list[dict]:
    import re
    out = []
    for name, res in (meta.get("raw") or {}).items():
        m = re.fullmatch(r"(?:file\d+_)?(\d+)threads_pass\d+", name)
        if m and int(m.group(1)) == threads:
            out.append({_key(k): v for k, v in res.items()})
    return out


def recommendation_text(best4, best1, par, cons=None) -> str:
    """Section 6.3: the plan in words, generated from the same numbers as the tables."""
    L = ["### 6.3 建议（由上两表自动生成，本机/ Mac 的划分按保守估计）\n"]
    for steps in (20000, 10000):
        rows = budget(best4, best1, par, steps, cons=cons)
        loc = [r for r in rows if r["local"]]
        mac = [r for r in rows if not r["local"]]
        def name(r):
            n = r["name"].split("（")[0]
            return f"{n if n.startswith('格') else '格 ' + r['cell'] + ' ' + n} {r['size']}"
        L.append(f"- **每次 {steps // 1000}k 步**：本机 12 小时内放得下 {', '.join(name(r) for r in loc) or '（无）'}"
                 f"（{sum(r['n'] for r in loc)} 次运行，约 {sum(r['machine_h'] for r in loc):.1f}–"
                 f"{sum(r['machine_h_cons'] for r in loc):.1f} 小时）。"
                 f"放到 Mac：{', '.join(name(r) for r in mac) or '（无）'}"
                 f"（{sum(r['n'] for r in mac)} 次，按本机速度约 {sum(r['machine_h'] for r in mac):.0f}–"
                 f"{sum(r['machine_h_cons'] for r in mac):.0f} 机时）。")
    parts = (("格 0 angle（主实验 + 负对照 + K = 1）", lambda r: r["cell"] == "0" and "像素" not in r["name"]),
             ("格 0 像素（主实验 + 负对照）", lambda r: r["cell"] == "0" and "像素" in r["name"]),
             ("格 6", lambda r: r["cell"] == "6"))
    for steps in (20000, 10000):
        rows = budget(best4, best1, par, steps, cons=cons)
        sub = "；".join(f"{name} {sum(r['n'] for r in rows if f(r))} 次 ≈ {sum(r['machine_h'] for r in rows if f(r)):.1f}–"
                       f"{sum(r['machine_h_cons'] for r in rows if f(r)):.1f} h" for name, f in parts)
        L.append(f"- 各部分按本机速度的均摊机时（每次 {steps // 1000}k 步，最优方式）：{sub}。")
    big = sorted({(k[0], k[2]) for k, r in best4.items() if r["sec_per_step"] * 2e4 / 3600 > FLAG_HOURS})
    if big:
        L.append(f"- 单次 20k 步在本机 4 线程下就超过 {FLAG_HOURS:.0f} 小时的配置："
                 f"{', '.join(f'{b} {s}' for b, s in big)}。这些必须去 Mac（或 CUDA 笔记本）。")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------- report
def fmt_h(x: float) -> str:
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def fmt_opt(x, spec: str = ".2f") -> str:
    """Format a possibly missing number (load average / /proc are not available on every OS)."""
    return "–" if x is None else format(x, spec)


def best_of(passes: list[dict]) -> dict:
    """Per configuration, the fastest UNDISTURBED measurement over all passes that measured it (the fastest of all
    of them if every measurement was disturbed, which the report then flags)."""
    keys = dict.fromkeys(k for p in passes for k in p)
    out = {}
    for k in keys:
        rs = [p[k] for p in passes if k in p]
        out[k] = min([r for r in rs if not _disturbed(r)] or rs, key=lambda r: r["sec_per_step"])
    return out


def spread(meta: dict, threads: int, key: tuple) -> Optional[float]:
    """Slowest / fastest undisturbed measurement of one configuration over all passes (and merged runs)."""
    import re
    vals = []
    for name, res in (meta.get("raw") or {}).items():
        m = re.fullmatch(r"(?:file\d+_)?(\d+)threads_pass\d+", name)
        r = res.get("/".join(key)) if m and int(m.group(1)) == threads else None
        if r and not _disturbed(r):
            vals.append(r["sec_per_step"])
    return max(vals) / min(vals) if len(vals) >= 2 else None


def _load_cell(r) -> str:
    return f"{fmt_opt(r.get('load_before'))} / {fmt_opt(r['load_mean'])} / {fmt_opt(r['load_max'])}"


def write_report(path: Path, best4: dict, best1: dict, par: dict, info: dict, flops: dict, meta: dict,
                 benches: list, n_passes: int):
    L = []
    q = meta.get("quiet_start") or {}
    L.append("# CPU 基准：baseline 世界模型（recon / jepa-sym / jepa-sg × 4 个尺寸，含格 6 整段开环）\n")
    L.append(f"- 生成：`scripts/bench_baselines.py`，{meta['date']}；机器：{meta['cpu']}，{meta['ncpu']} 个逻辑核，"
             f"{meta['mem_gb']:.0f} GB 内存，无 GPU。软件：Python {meta['python']}，torch {meta['torch']}，"
             f"numpy {meta['numpy']}。整套测量用时 {meta.get('total_sec', 0) / 60:.0f} 分钟。")
    L.append(f"- 一步 = 取一个 batch（{meta['batch']} 个窗口）+ K 步开环损失 + 反向 + 梯度裁剪 + AdamW，与 "
             "`baselines.train` 每步做的事完全相同。单摆配置用 history 2、K 8（K = 1 的两组除外）；toggle_grid 用 "
             "`anchors=\"episode\"`、history 1、K = 63（64 帧的整段 episode 作为一个开环窗口，动作 one-hot）。"
             "像素模型在 CPU 上用 channels-last。")
    if q:
        L.append(f"- **安静机器协议**：开测前每 {meta['poll']:.0f} 秒查一次，最多等 {meta['max_wait_quiet'] / 60:.0f} 分钟，"
                 f"直到 1 分钟 load < {meta['max_load']} 且没有其他占 ≥ 0.5 核的 python 进程。本次开测前 load = "
                 f"{fmt_opt(q.get('load_start'))}，等待 {q.get('waited_sec', 0):.0f} 秒后 load = "
                 f"{fmt_opt(q.get('load_end'))}，{'满足条件' if q.get('quiet') else '**超时仍未满足，结果偏慢**'}；"
                 f"开测前看到的其他重进程：{_heavy_text(q.get('heavy_seen'))}。每个配置测量前再查一次重 python 进程，"
                 f"出现就重新按上面的规则等待（本次触发 {meta.get('guard_triggers', 0)} 次，全部等待合计 "
                 f"{meta.get('waited_total_sec', 0) / 60:.0f} 分钟，上限 {meta.get('max_total_wait', 0) / 60:.0f} 分钟）。"
                 f"测量期间他人线程均值 > {DISTURBED} 的配置在“标记”列注明“受干扰”。")
    L.append("- 表里的 load 三个数是：测量前 / 测量中均值 / 测量中最大的 1 分钟 `os.getloadavg()`。它**包含本进程"
             "自己的线程**，所以 4 线程测量时会逐渐升到 4 左右，这是正常的；判断有没有被别人干扰要看“他人线程”，"
             "即测量期间其他进程处于运行态的线程数均值（/proc）。")
    passes_txt = meta.get("passes_note") or f"整套跑了 {n_passes} 遍"
    L.append(f"- {passes_txt}，每个配置取最快的一次未受干扰的测量（每遍内取每步耗时的中位数）。外推只算训练步；"
             "每 1000 步一次的 val 和最终指标按 FLOPs 估计不到 1%，数据生成见第 5 节。\n")

    def table(best, title, note="", threads=None):
        L.append(title)
        if note:
            L.append(note)
        L.append("| 配置 | variant | size | params | train GFLOPs/样本 | s/step | 各遍最慢/最快 | 10k 步 (h) | 20k 步 (h) | "
                 "load 前/中/峰 | 他人线程 | 标记 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k in configs(benches):
            if k not in best:
                continue
            r = best[k]
            h10, h20 = r["sec_per_step"] * 1e4 / 3600, r["sec_per_step"] * 2e4 / 3600
            flag = "⚠ > 12 h" if h20 > FLAG_HOURS else ""
            if (r.get("other_running_mean") or 0) > DISTURBED:
                flag = (flag + " " if flag else "") + "受干扰"
            sp = spread(meta, threads, k)
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {info[k]['params']:,} | "
                     f"{flops[k]['train_flops_per_sample'] / 1e9:.3f} | {r['sec_per_step']:.4f} | "
                     f"{'–' if sp is None else f'{sp:.2f}×'} | {fmt_h(h10)} | "
                     f"{fmt_h(h20)} | {_load_cell(r)} | {fmt_opt(r['other_running_mean'], '.1f')} | {flag} |")
        L.append("")

    table(best4, f"## 1. {meta.get('threads', 4)} 线程（batch 128），一次只跑一个任务\n",
          "“各遍最慢/最快”是同一配置在各遍（含补测）中未受干扰的测量之间的比值，反映本机（虚拟机，宿主上的"
          "争用在 /proc 里看不到）的测量噪声；s/step 列取最快的一遍。\n", threads=meta.get("threads", 4))
    if best1:
        table(best1, "## 2. 1 线程（batch 128），机器上只有这一个任务\n",
              "这是单线程任务**单独**跑时的速度。4 个单线程任务同时跑时每个任务会慢一些，实测见第 3 节。"
              "并行时线程总数绝不能超过核数：之前在其他进程占满 3 个核时实测，angle_cs S 用 4 线程 2.6 s/step，"
              "用 1 线程 0.017 s/step，差 150 倍（OpenMP 线程互相等待）。\n", threads=1)
    if par:
        L.append("## 3. 4 个单线程任务同时跑（同一配置 × 4，`spawn` 子进程）\n")
        L.append("| 配置 | variant | size | 每个任务 s/step（4 个均值） | 单独跑 1 线程 | 变慢倍数 | 4 并行吞吐 / 4 线程单任务吞吐 | load 前/中/峰 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k, r in par.items():
            t1 = best1.get(k, {}).get("sec_per_step")
            t4 = best4.get(k, {}).get("sec_per_step")
            slow = f"{r['sec_per_step'] / t1:.2f}×" if t1 else "–"
            thr = f"{4 * t4 / r['sec_per_step']:.2f}×" if t4 else "–"
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {r['sec_per_step']:.4f} | {fmt_opt(t1, '.4f')} | {slow} | {thr} | "
                     f"{_load_cell(r)} |")
        L.append("")
        L.append("最后一列 > 1 表示“4 个单线程任务并行”比“4 线程一个一个跑”的总吞吐更高。"
                 "没有实测的配置在第 6 节里借用同一数据族最近尺寸的变慢倍数。\n")

    L.append("## 4. 尺寸阶梯（d = 8）\n")
    L.append("目标是 recon 模型的总参数量（±15%）。两个 jepa 变体结构完全相同，用同一组宽度，只是没有 decoder。"
             "向量观测：编码器 / 转移 / 解码器都是 2 个隐藏层、宽 w 的 MLP（LayerNorm + SiLU）；toggle_grid 的 history 是 1，"
             "所以输入更窄、同尺寸下 w 更大。像素：卷积通道 c, 2c, 4c, 8c（64→32→16→8→4），MLP 宽 w ≈ 4c，"
             "decoder 输出层为线性（`pixel_out=\"linear\"`，见 `baselines/mlp_wm.py` 文档）。\n")
    L.append("| 配置 | size | w | c | recon params | jepa params | 相对目标 | encoder | transition | decoder | "
             "train GFLOPs/样本 recon / jepa-sym / jepa-sg |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for bench in benches:
        if bench.endswith("_K1"):
            continue
        for s in BENCHES[bench]["sizes"]:
            ir = info.get((bench, "recon", s))
            if ir is None:
                continue
            fl = " / ".join(f"{flops[(bench, v, s)]['train_flops_per_sample'] / 1e9:.3g}" if (bench, v, s) in flops
                            else "–" for v in VARIANTS)
            L.append(f"| {bench} | {s} | {ir['width']} | {ir['channels'] or '–'} | {ir['params_recon']:,} | "
                     f"{ir['params_jepa']:,} | {ir['params_recon'] / SIZES[s] - 1:+.1%} | {ir['params_encoder']:,} | "
                     f"{ir['params_transition']:,} | {ir['params_decoder']:,} | {fl} |")
    L.append("")
    L.append("FLOPs 用 `torch.utils.flop_counter.FlopCounterMode` 数，只计矩阵乘和卷积；train 一项是一个样本的完整 "
             "K 步损失（前向 + 反向，含不求梯度的目标编码）。jepa-sg 的目标编码不求梯度，所以比 jepa-sym 省一截。\n")

    dg = meta.get("data_gen") or {}
    if dg:
        L.append("## 5. 每次运行的数据生成（train 2000 + val 200 条，单进程）\n")
        L.append("| 世界 / 观测 / 步数 | 秒 | 内存 (MB) |")
        L.append("|---|---|---|")
        for k, v in dg.items():
            L.append(f"| {k} | {v['sec']:.1f} | {v['mb']:.0f} |")
        L.append("")
        L.append("像素数据以 uint8 常驻内存，4 个像素任务并行约需上表内存 × 4，15 GB 内存足够。\n")

    if best1 and best4 and all(b in benches for b in ("angle_cs", "angle_raw", "pixels64", "toggle_ep63")):
        L.append("## 6. 预算：格 0 与格 6 的预注册运行在本机 12 小时内放得下哪些\n")
        L.append("运行清单按预注册计（格 0 §2、§6；格 6 §7），jepa 已按格 0 §0b 拆成 jepa-sg 和 jepa-sym，"
                 "所以草案里每个 jepa 运行都变成两个。每组 3 个种子。格 6 每个尺寸的运行数：3 档数据 + 开关不可见 + "
                 "2 个负对照世界，S、M 另加阿贝尔对照和 2、3 位扫描（共 90 次）。阿贝尔增量模型按 recon 的速度估算；"
                 "2、3 扇门的网格更宽，但阶梯按参数量对齐，FLOPs 基本不变。SIGReg 副因子（6 次）尚未实现，不在表内。"
                 "副因子 K = 1 的 jepa 运行同样按两个变体计。\n")
        L.append("每次运行取两种方式中更省机时的一种：**并行** = 单线程、4 个任务同时跑（速度取第 2 节 × 第 3 节的"
                 "变慢倍数，均摊机时 = 单任务时间 / 4）；**串行** = 4 线程、一次一个（第 1 节）。"
                 "本机预算按“最便宜的组先放”填满 12 小时，单个任务超过 12 小时的组一律放到 Z 的 Mac（MPS）。\n")
        cons = (worst_of(passes_from_raw(meta, meta.get("threads", 4))) or best4, worst_of(passes_from_raw(meta, 1)) or best1)
        L.append("“最快”用每个配置最快的一遍，“保守”用最慢的一遍未受干扰的测量（见第 1、2 节的“各遍最慢/最快”）；"
                 "本机 / Mac 的划分按保守估计做，免得测量噪声把本机方案推过 12 小时。\n")
        for steps in (20000, 10000):
            rows = budget(best4, best1, par, steps, cons=cons)
            L.append(f"### 6.{1 if steps == 20000 else 2} 每次 {steps // 1000}k 步\n")
            L.append("| 格 | 部分 | size | 运行数 | 方式 | 单任务墙钟最长 (h) 最快 / 保守 | 均摊机时 (h) 最快 / 保守 | "
                     "本机累计 (h，保守，按从便宜到贵的填入顺序) | 建议 |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for r in rows:
                mode = "/".join({"par": "并行", "seq": "串行"}[m] for m in r["modes"])
                L.append(f"| {r['cell']} | {r['name']} | {r['size']} | {r['n']} | {mode} | {fmt_h(r['job_max_h'])} / "
                         f"{fmt_h(r['job_max_h_cons'])} | {fmt_h(r['machine_h'])} / {fmt_h(r['machine_h_cons'])} | "
                         f"{fmt_opt(r['cum_local_h'], '.1f')} | {'本机' if r['local'] else '**Mac**'} |")
            tot = lambda key, loc: sum(r[key] for r in rows if r["local"] == loc)
            n_loc = sum(r["n"] for r in rows if r["local"])
            n_all = sum(r["n"] for r in rows)
            L.append("")
            L.append(f"合计：本机 {n_loc} / {n_all} 次运行，约 {tot('machine_h', True):.1f}–{tot('machine_h_cons', True):.1f} "
                     f"小时；其余 {n_all - n_loc} 次按本机速度需要约 {tot('machine_h', False):.0f}–"
                     f"{tot('machine_h_cons', False):.0f} 机时（4 核），放到 Mac。\n")
        L.append(recommendation_text(best4, best1, par, cons))
        if meta.get("notes"):
            L.append("### 6.4 补充观察（人工撰写）\n")
            L.append(meta["notes"].strip() + "\n")

    L.append("## 7. 说明与局限\n")
    L.append("- MPS 与 CUDA 没有在这里测。Mac 上的实际速度要在那台机器上跑一次 `baselines.train`，看 `metrics.json` "
             "里的 `timing.sec_per_step`；代码不分设备（`--device auto`：cuda > mps > cpu）。\n"
             "- 第 6 节的均摊机时假设任务能整齐地 4 个一组排满；最后不满 4 个的那批会多占一点墙钟。\n"
             "- 4 线程测量时 load 包含本进程自己；“他人线程”一列接近 0 才说明测量没被别的 agent 干扰。\n")

    L.append("## 原始数据\n")
    L.append("全部原始结果（每遍的逐配置中位数 s/step、均值、步数、load、他人线程、等待秒数，参数量，FLOPs，元数据）"
             "在下面的 JSON 里；`--from-json results/bench_cpu.md` 可以据此重新生成本报告。\n")
    L.append("```json")
    L.append(json.dumps(payload(meta, info, flops), ensure_ascii=False, indent=None))
    L.append("```")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _heavy_text(heavy) -> str:
    if not heavy:
        return "无"
    return "；".join(f"pid {h['pid']}（{h['cpu']:.1f} 核，`{h['cmd'][:40]}`）" for h in heavy[:5])


def _key(k: str) -> tuple:
    return tuple(k.split("/"))


def _raw_dict(name_to_res: dict) -> dict:
    return {name: {"/".join(k): v for k, v in res.items()} for name, res in name_to_res.items()}


def payload(meta: dict, info: dict, flops: dict) -> dict:
    return {"meta": meta, "info": {"/".join(k): v for k, v in info.items()},
            "flops": {"/".join(k): v for k, v in flops.items()}}


def read_payload(src: Path) -> dict:
    """The raw results: a --json file, or the JSON block embedded at the end of a report."""
    txt = src.read_text(encoding="utf-8")
    if src.suffix == ".md":
        txt = txt[txt.rindex("```json") + len("```json"):]
        txt = txt[:txt.index("```")]
    return json.loads(txt)


def render_from_json(srcs: list[Path], out: Path, notes: Optional[str] = None) -> int:
    """Rebuild the markdown report from the raw results of earlier runs (``--json`` files or reports); with several
    files every configuration reports the fastest of ALL their passes (e.g. an extra pass taken later in a quieter
    window). ``notes`` replaces the stored hand-written notes."""
    import re

    runs, par, info, flops, meta, raw_all = {}, {}, {}, {}, None, {}
    for j, src in enumerate(srcs):
        d = read_payload(src)
        meta = meta or d["meta"]
        info.update({_key(k): v for k, v in d["info"].items()})
        flops.update({_key(k): v for k, v in d["flops"].items()})
        for name, res in d["meta"]["raw"].items():
            raw_all[name if len(srcs) == 1 else f"file{j + 1}_{name}"] = res
            m = re.fullmatch(r"(?:file\d+_)?(\d+)threads_pass(\d+)", name)      # prefixed if already merged
            if m:
                runs.setdefault(int(m.group(1)), []).append({_key(k): v for k, v in res.items()})
            elif re.fullmatch(r"(?:file\d+_)?parallel.*", name):
                for k, v in res.items():
                    if _key(k) not in par or v["sec_per_step"] < par[_key(k)]["sec_per_step"]:
                        par[_key(k)] = v
    merged = "" if len(srcs) == 1 or "合并" in meta["date"] else f"（合并 {len(srcs)} 次运行）"
    meta = {**meta, "raw": raw_all, "date": meta["date"] + merged}
    if len(srcs) > 1:
        extra = [read_payload(src)["meta"] for src in srcs[1:]]
        meta["passes_note"] = (f"主测完整跑了 {meta.get('passes')} 遍（{meta.get('total_sec', 0) / 60:.0f} 分钟）；另有 "
                               + "、".join(f"{len(e.get('raw', {}).get('4threads_pass1', {}))} 个配置的补测 "
                                           f"{e.get('passes')} 遍（{e.get('total_sec', 0) / 60:.0f} 分钟）" for e in extra))
    if notes is not None:
        meta["notes"] = notes
    threads = meta.get("threads") or max(runs)
    best4 = best_of(runs[threads])
    best1 = best_of(runs[1]) if 1 in runs and threads != 1 else {}
    write_report(out, best4, best1, par, info, flops, meta, meta.get("benches") or list(BENCHES), len(runs[threads]))
    print(f"re-rendered {out} from {', '.join(map(str, srcs))}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--benches", nargs="+", default=list(BENCHES), choices=list(BENCHES))
    p.add_argument("--sizes", nargs="+", default=None, help="restrict sizes (default: each bench's own list)")
    p.add_argument("--no-single-thread", action="store_true", help="skip the 1-thread sweep")
    p.add_argument("--no-parallel", action="store_true", help="skip the 4 concurrent single-thread jobs")
    p.add_argument("--parallel-jobs", type=int, default=4)
    p.add_argument("--min-time", type=float, default=4.0, help="seconds of timed steps per configuration")
    p.add_argument("--min-steps", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--max-load", type=float, default=1.5, help="quiet machine: 1-minute load average below this")
    p.add_argument("--poll", type=float, default=60.0, help="seconds between quiet-machine checks")
    p.add_argument("--max-wait-quiet", type=float, default=1800.0, help="give up waiting for quiet after this")
    p.add_argument("--max-total-wait", type=float, default=5400.0, help="cap on all quiet-machine waits together")
    p.add_argument("--retries", type=int, default=2, help="re-measure a disturbed configuration up to this often")
    p.add_argument("--only", nargs="+", default=None, metavar="BENCH/VARIANT/SIZE",
                   help="measure only these configurations (e.g. a supplementary pass for disturbed ones; merge "
                        "the reports with --from-json old.md new.md)")
    p.add_argument("--settle-threads", type=int, default=0, help="short settle: max runnable threads of others")
    p.add_argument("--settle-wait", type=float, default=30.0, help="max seconds of the short settle")
    p.add_argument("--no-data-gen", action="store_true", help="skip timing the full-size data generation")
    p.add_argument("--out", default=str(ROOT / "results" / "bench_cpu.md"))
    p.add_argument("--json", default=None, help="optional path for the raw results as JSON")
    p.add_argument("--notes", default=None, help="markdown file with hand-written observations for section 6")
    p.add_argument("--quick", action="store_true", help="angle_cs + toggle_ep63, sizes S M, 1 pass, short timing")
    p.add_argument("--from-json", nargs="+", default=None,
                   help="re-render the report from previous --json file(s), best of all their passes (no timing)")
    a = p.parse_args(argv)
    if a.from_json:
        notes = Path(a.notes).read_text(encoding="utf-8") if a.notes else None
        return render_from_json([Path(f) for f in a.from_json], Path(a.out), notes)
    if a.quick:
        a.benches, a.sizes, a.passes, a.min_time, a.max_wait_quiet = ["angle_cs", "toggle_ep63"], ["S", "M"], 1, 0.5, 0.0
        a.no_data_gen, a.settle_wait = True, 2.0

    t_all = time.perf_counter()
    quiet_start = wait_for_quiet_machine(a.max_load, a.poll, min(a.max_wait_quiet, a.max_total_wait))
    WAITED[0] += quiet_start["waited_sec"]
    print(f"[quiet] start: {quiet_start}", flush=True)
    setup_runtime(0, a.threads, deterministic=True)
    cfgs = list(configs(a.benches, a.sizes))
    if a.only:
        want = {tuple(k.split("/")) for k in a.only}
        bad = want - set(configs(list(BENCHES)))
        if bad:
            raise SystemExit(f"unknown configurations: {sorted(bad)}")
        cfgs = [k for k in cfgs if k in want]
        a.benches = [b for b in a.benches if any(k[0] == b for k in cfgs)]
    data = {b: make_data(b) for b in a.benches}
    info, flops = {}, {}
    for bench, v, s in cfgs:
        m = build(data[bench][0], bench, v, s, data[bench][1].K)
        info[(bench, v, s)] = describe(m)
        flops[(bench, v, s)] = train_flops_per_sample(m.cfg, batch=2)
    passes4, passes1 = [], []
    for i in range(a.passes):
        passes4.append(sweep(a.threads, cfgs, data, a, f"pass {i + 1}"))
        if not a.no_single_thread:
            passes1.append(sweep(1, cfgs, data, a, f"pass {i + 1}"))
    torch.set_num_threads(a.threads)
    par = {}
    if not a.no_parallel and not a.no_single_thread:
        keys = [k for k in PARALLEL if k in set(cfgs)]
        par = parallel_sweep(keys, a.parallel_jobs, a, "parallel")
    dgen = {} if a.no_data_gen else data_gen_seconds(a.benches)
    best4 = best_of(passes4)
    best1 = best_of(passes1) if passes1 else {}
    try:
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        mem_gb = float("nan")
    cpu = platform.processor() or platform.machine()
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="ignore"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    raw = _raw_dict({f"{t}threads_pass{i + 1}": ps for t, plist in ((a.threads, passes4), (1, passes1))
                     for i, ps in enumerate(plist)})
    if par:
        raw.update(_raw_dict({f"parallel{a.parallel_jobs}x1": par}))
    meta = {"date": dt.date.today().isoformat(), "cpu": cpu, "ncpu": os.cpu_count(), "mem_gb": mem_gb,
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "batch": a.batch, "max_load": a.max_load, "poll": a.poll, "max_wait_quiet": a.max_wait_quiet,
            "quiet_start": quiet_start, "guard_triggers": GUARD_TRIGGERS[0], "waited_total_sec": WAITED[0],
            "max_total_wait": a.max_total_wait, "raw": raw, "data_gen": dgen,
            "threads": a.threads, "passes": a.passes, "benches": a.benches, "sizes": a.sizes,
            "total_sec": time.perf_counter() - t_all,
            "notes": Path(a.notes).read_text(encoding="utf-8") if a.notes else ""}
    out = Path(a.out)
    write_report(out, best4, best1, par, info, flops, meta, a.benches, a.passes)
    if a.json:
        Path(a.json).write_text(json.dumps(payload(meta, info, flops), indent=1, ensure_ascii=False),
                                encoding="utf-8")
    print(f"wrote {out} in {time.perf_counter() - t_all:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
