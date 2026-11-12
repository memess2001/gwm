"""Cell 0 v2: the unified run table, training configurations and data (cells/0/prereg.md v2).

Import with ``sys.path.insert(0, "cells/0")`` from the repo root. Nothing here touches a trained model.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORLD = dict(omega_max=2.5)                        # pendulum defaults: omega0 = 1, gamma = 0, dt = 0.1, init "box"
NEG_WORLD = dict(omega_max=2.5, E_lo=-1.0, E_hi=0.5)   # libration only (true b1 = 0); E_hi = 0.5 (calibration)
HORIZON, N_TRAIN, N_VAL, N_ANALYSIS = 64, 2000, 200, 500
STEPS = 20000
HISTORY, LATENT, K_MAIN = 2, 8, 8
SEEDS = (0, 1, 2)
VICREG = (25.0, 25.0, 1.0)                         # frozen (prereg v2 item 11)

# arm name -> (obs, baseline variant, loss)
ARMS = {
    "cs-recon": ("angle_cs", "recon", "mse"),
    "cs-jepa-sg": ("angle_cs", "jepa-sg", "mse"),
    "cs-jepa-sym": ("angle_cs", "jepa-sym", "mse"),
    "raw-recon-mse": ("angle_raw", "recon", "mse"),
    "raw-recon-circ": ("angle_raw", "recon", "circular"),
    "raw-jepa-sg": ("angle_raw", "jepa-sg", "mse"),
    "raw-jepa-sym": ("angle_raw", "jepa-sym", "mse"),
    "pix-recon": ("pixels64", "recon", "mse"),
    "pix-jepa-sg": ("pixels64", "jepa-sg", "mse"),
    "pix-jepa-sym": ("pixels64", "jepa-sym", "mse"),
}
SCORED_SIZES = ("S", "M", "L")                     # P0.3 / P0.4 / P0.5 scoring set; XL is descriptive only


def run_table() -> list[dict]:
    """Every training run of the v2 design (prereg v2 item 13: one table, used as the scoring set).

    keys: name, arm, obs, variant, loss, size, seed, world ('main' | 'neg'), K, lambda_lat, phase (1 | 2), role.
    """
    rows = []

    def add(arm, size, seed, world="main", K=K_MAIN, lam=0.1, role="main"):
        obs, var, loss = ARMS[arm]
        phase = 1 if (obs != "pixels64" or size == "S") and size != "XL" else 2
        tag = f"{arm}_{size}_s{seed}" + ("" if world == "main" else "_neg") + ("" if K == K_MAIN else f"_K{K}") \
            + ("" if (lam == 0.1 or var != "recon") else f"_lat{lam:g}")
        rows.append(dict(name=tag, arm=arm, obs=obs, variant=var, loss=loss, size=size, seed=seed, world=world,
                         K=K, lambda_lat=(lam if var == "recon" else 0.0), phase=phase, role=role))

    for arm in ARMS:
        for size in ("S", "M", "L", "XL"):
            for sd in SEEDS:
                add(arm, size, sd, role="main" if size != "XL" else "descriptive_xl")
    for arm in ("cs-recon", "cs-jepa-sg", "cs-jepa-sym", "pix-recon", "pix-jepa-sg", "pix-jepa-sym"):
        for sd in SEEDS:
            add(arm, "M", sd, world="neg", role="negative_control")
    for arm in ("cs-recon", "cs-jepa-sg", "cs-jepa-sym", "raw-recon-mse", "raw-jepa-sg", "raw-jepa-sym"):
        for sd in SEEDS:
            add(arm, "M", sd, K=1, role="horizon_K1")
    for arm in ("raw-recon-mse", "raw-recon-circ"):
        for sd in SEEDS:
            add(arm, "M", sd, lam=0.0, role="lambda_lat0")
    return rows


def null_table() -> list[dict]:
    """Initialised-but-untrained encoders (Z's choice 3, prereg v2 item 9): one per observation x seed, size M,
    recon architecture (the encoder is identical across variants of one size)."""
    return [dict(name=f"null-{obs}_M_s{sd}", obs=obs, size="M", seed=sd, role="null_encoder")
            for obs in ("angle_cs", "angle_raw", "pixels64") for sd in SEEDS]


def train_args(row: dict, out: str, threads: int = 1, device: str = "auto") -> list[str]:
    """Command-line arguments of ``python -m baselines.train`` for one row of the run table."""
    import json
    world = WORLD if row["world"] == "main" else NEG_WORLD
    a = ["--world", "pendulum", "--obs", row["obs"], "--variant", row["variant"], "--size", row["size"],
         "--seed", str(row["seed"]), "--steps", str(STEPS), "--history", str(HISTORY), "--latent-dim", str(LATENT),
         "--K", str(row["K"]), "--lambda-lat", str(row["lambda_lat"]), "--loss", row["loss"],
         "--vicreg", ",".join(str(v) for v in VICREG), "--horizon", str(HORIZON), "--n-train", str(N_TRAIN),
         "--n-val", str(N_VAL), "--world-kwargs", json.dumps(world), "--out", out, "--threads", str(threads),
         "--device", device]
    if row["obs"] == "pixels64":
        a += ["--pixel-out", "linear"]
    return a
