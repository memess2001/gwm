#!/usr/bin/env python
"""Train one cell-6 world model (recon, history 1, whole-episode open loop, lambda_lat = 0) and score its gate.

    python cells/6/c6_train.py --cond H-act --size S --seed 0 --steps 10000            # main run
    python cells/6/c6_train.py --cond H-act --size S --seed 99 --steps 10000 --pilot   # pilot (gate only)
    python cells/6/c6_train.py --cond H-act --size M --seed 1 --n-train 500 ...        # data-efficiency run

Writes runs/c6/<cond>_<size>_s<seed>[_n<n_train>][_pilot]/ : config.json, model.pt, metrics.json (baselines),
gate.json, c6_run.json. Only the gate (a behavioural check of the model) is computed here; the readouts are
computed by c6_score.py, which refuses pilot runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c6lib as c                                                    # noqa: E402

from zoo.base import SPLIT_SEEDS                                      # noqa: E402
from baselines.train import TrainConfig, run                          # noqa: E402
from analysis.holonomy import WMPredictor                             # noqa: E402

ROOT = c.ROOT


def p_home() -> float:
    mc = json.loads((HERE / "manipulation_check.json").read_text())
    return float(mc["p_home"]["p_home"])


def run_dir(cond, size, seed, n_train, pilot) -> Path:
    tag = f"{cond}_{size}_s{seed}" + ("" if n_train == c.N_TRAIN_MAIN else f"_n{n_train}")
    return ROOT / "runs" / "c6" / tag


def pilot_dir(cond, size, seed, steps) -> Path:
    """v1.2: one directory per pilot step count (the 10k and the 20k pilot must not overwrite each other)."""
    return ROOT / "runs" / "c6" / f"{cond}_{size}_s{seed}_pilot{steps}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=list(c.CONDITIONS))
    ap.add_argument("--size", required=True, choices=["S", "M"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--n-train", type=int, default=c.N_TRAIN_MAIN)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if a.pilot and a.seed in (0, 1, 2):
        raise SystemExit("pilot runs use a seed outside the main seeds {0, 1, 2}")
    if a.n_train != c.N_TRAIN_MAIN and a.cond != "H-act":
        raise SystemExit("data-efficiency runs exist only for H-act")

    cond = c.CONDITIONS[a.cond]
    w = c.make_world(a.cond)
    ph = p_home() if cond["policy"] == "active" else 0.0
    splits = {}
    for split, n in (("train", a.n_train), ("val", c.N_VAL)):
        seed = SPLIT_SEEDS[split]                                      # data seed independent of the model seed
        st, acts = c.sample_policy(w, cond["policy"], seed, n, p_home=ph)
        splits[split] = c.to_traj(w, st, acts, seed, cond["policy"], {"p_home": ph})
    spec = w.spec("grid")
    out = pilot_dir(a.cond, a.size, a.seed, a.steps) if a.pilot else run_dir(a.cond, a.size, a.seed, a.n_train, False)
    cfg = TrainConfig(world="toggle_grid", obs="grid", world_kwargs=dict(cond["world"]), horizon=c.HORIZON,
                      n_train=a.n_train, n_val=c.N_VAL, anchors="episode", variant="recon", size=a.size,
                      latent_dim=8, history=1, lambda_lat=0.0, seed=a.seed, steps=a.steps, threads=a.threads,
                      device="cpu", out=str(out), overwrite=a.overwrite, quiet=False)
    res = run(cfg, data=(spec, splits))
    model = res["model"].cpu().eval()
    pred = WMPredictor(model, res["normalizer"], w)
    gate = c.gate_eval(pred, c.gate_set(w, c.GATE_SEED))
    door_gate = a.cond != "H-pn"
    gate["door_gate_applies"] = door_gate
    gate["pass"] = c.gate_pass(gate, door_gate)
    (out / "gate.json").write_text(json.dumps(gate, indent=1))
    (out / "c6_run.json").write_text(json.dumps({"cond": a.cond, "size": a.size, "seed": a.seed, "steps": a.steps,
                                                 "n_train": a.n_train, "pilot": a.pilot, "p_home": ph}, indent=1))
    print("[c6] gate", json.dumps(gate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
