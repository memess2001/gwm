"""Train a baseline world model (variant recon | jepa-sym | jepa-sg, ladder size S | M | L | XL) on a zoo world.

    python -m baselines.train --world pendulum --obs pixels64 --variant recon --size M --seed 0 \
        --steps 20000 --out runs/c0_pix_recon_M_s0 [--world-kwargs '{"omega_max": 2.5}'] [--threads 4]

Writes to ``--out``: config.json (all settings, widths, parameter counts, FLOPs, environment), model.pt
(state dict + model config + normaliser; reload with ``baselines.encode.load``), metrics.json (train / val loss
curves and final metrics). Paths are relative to the current directory (run from the repo root).

Optimisation: AdamW (lr 1e-3, weight decay 1e-4 on weight matrices only), linear warmup then cosine decay to 0,
batch 128, global grad-norm clip 1.0, batches drawn i.i.d. with a numpy Generator seeded by ``--seed``.
Data: train / val trajectories from disjoint seeds ``SPLIT_SEEDS[split] + data_index`` (independent of the model
seed, so all model seeds see the same data).

Final metrics (val split, never the analysis split):
  recon  val reconstruction error of the current frame D(E(w_t)) vs o_t (MSE in raw observation units, and
         nmse = sum_f mse_f / sum_f var_f with the TRAIN per-feature variance); open-loop rollout error at
         k in {1, 8, 32} from the FIRST window of every val trajectory (z = E(o_0..o_{h-1}), k latent steps,
         decode, compare with o_{h-1+k}); angle_raw also gets circ = mean(1 - cos dtheta) and the wrapped-difference
         errors mse_wrapped / nmse_wrapped (the meaningful reconstruction error for loss="circular").
  jepa   (both jepa-sym and jepa-sg; "jepa" is an alias of jepa-sym) variance-normalised latent prediction error
         at k in {1, 8}: E||F^k(E(w_t)) - E(w_{t+k})||^2 / sum_d Var(z_d), plus the same for the copy predictor
         z_{t+k} ~ z_t; per-dimension latent std and effective rank of z_t = E(w_t) (collapse diagnostics, the
         cell 0 jepa gate; also logged during training: "latent_std" [d] = batch std per dim averaged over the
         log interval, "effective_rank" on the current batch).

Pre-registration hygiene: this file computes losses and prediction errors only. FORBIDDEN here: persistent
homology, energy / state probes, any use of ground-truth state beyond data generation (``Trajectories.state`` and
``.info`` are never read). The pre-registered measurements live in ``analysis/`` and run on latents from
``baselines.encode`` only after the pre-registration is approved. ``--obs state`` (the pendulum's true (theta,
omega) as the observation) is refused unless ``--allow-state-obs`` is given: such a model is an oracle, never a
baseline. ``--loss circular`` is accepted only for ``--obs angle_raw`` (every feature an angle).

Determinism: with ``--deterministic`` (default on) results are bitwise reproducible on CPU for the same seed,
thread count, torch build and machine. CUDA: we set CUBLAS_WORKSPACE_CONFIG, cudnn.deterministic and
``torch.use_deterministic_algorithms(True, warn_only=True)``; TF32 stays at the torch defaults. MPS has no
deterministic mode: seeds fix the initial weights (always built on the CPU, then moved) and the batch order,
but MPS kernels may differ run to run in the last bits, so MPS runs are reproducible only statistically.
Different devices never agree bitwise. Pixel models use channels-last conv stacks on CPU (``--channels-last auto``),
which changes the last bits relative to the default layout; keep one setting per experiment.

Hidden-state worlds (cell 6): ``--anchors episode --history 1`` trains on ONE open-loop window per episode,
o_0 .. o_T, with K = horizon - history + 1 set automatically (K = 64 for --horizon 64; pass --horizon 63 for a
64-frame episode with K = 63); discrete actions are one-hot. The final rollout errors then also include k = K.
The recon latent-consistency target sg(E(w_{t+j})) comes from a memoryless encoder, so lambda_lat > 0 pulls the
open-loop latent towards "no memory" (it penalises carrying door bits that the current frame does not show);
consider ``--lambda-lat 0`` there. The default stays 0.1 everywhere: the choice belongs to the pre-registration.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .data import ANCHORS, Normalizer, WindowDataset, episode_K, make_splits
from .encode import CHECKPOINT_FORMAT
from .mlp_wm import (SIZES, VARIANT_ALIASES, VARIANTS, WorldModel, build_wm, canonical_variant, describe,
                     effective_rank, is_jepa, latent_stats, train_flops_per_sample)


STATE_OBS = ("state",)            # obs modes that ARE the ground-truth state (pendulum "state" = (theta, omega))
CIRCULAR_OBS = ("angle_raw",)     # obs modes whose every feature is an angle: the only ones for loss="circular"


@dataclass
class TrainConfig:
    # data
    world: str = "pendulum"
    obs: str = "angle_cs"
    allow_state_obs: bool = False        # obs in STATE_OBS is an oracle (model sees the true state), never a baseline
    world_kwargs: dict = field(default_factory=dict)
    horizon: int = 64
    n_train: int = 2000
    n_val: int = 200
    data_index: int = 0
    standardize: bool = False            # vector obs only; statistics of the train split
    anchors: str = "all"                 # "all" | "start" | "episode" (whole episode as one open-loop window)
    # model
    variant: str = "recon"               # recon | jepa-sym | jepa-sg  ("jepa" -> jepa-sym)
    size: str = "S"
    latent_dim: int = 8
    history: int = 2
    K: Optional[int] = None              # None: 8, or horizon - history + 1 for anchors="episode"
    loss: str = "mse"                    # recon only: mse | circular (angle_raw)
    pixel_out: str = "linear"            # recon on pixels: decoder output linear | sigmoid (see baselines.mlp_wm)
    lambda_lat: float = 0.1
    vicreg: tuple = (25.0, 25.0, 1.0)
    norm: str = "layer"
    width: Optional[int] = None          # override the ladder widths
    channels: Optional[int] = None
    # optimisation
    seed: int = 0
    steps: int = 20000
    batch: int = 128
    lr: float = 1e-3
    wd: float = 1e-4
    warmup: Optional[int] = None         # default min(1000, steps // 20)
    clip: float = 1.0
    # runtime
    device: str = "auto"
    channels_last: str = "auto"          # image models: "auto" (= on for CPU only) | "on" | "off"
    threads: Optional[int] = None
    deterministic: bool = True
    log_every: int = 100
    eval_every: int = 1000
    n_eval: int = 1024                   # fixed val windows for the periodic val loss
    n_eval_final: int = 4096             # fixed val windows for the final metrics
    rollout_k: tuple = (1, 8, 32)
    latent_k: tuple = (1, 8)
    out: Optional[str] = None
    overwrite: bool = False
    quiet: bool = False

    def __post_init__(self):
        self.variant = canonical_variant(self.variant)
        if self.obs in STATE_OBS and not self.allow_state_obs:
            raise ValueError(f"obs={self.obs!r} feeds the ground-truth state to the model: an oracle, not a baseline "
                             "(set allow_state_obs=True / --allow-state-obs for a deliberate oracle run)")
        if self.loss == "circular" and self.obs not in CIRCULAR_OBS:
            raise ValueError(f"loss='circular' treats every observed feature as an angle: only for obs in "
                             f"{CIRCULAR_OBS} (got obs={self.obs!r})")
        if self.anchors not in ANCHORS:
            raise ValueError(f"anchors must be one of {ANCHORS}")
        if self.K is None:
            self.K = episode_K(self.horizon, self.history) if self.anchors == "episode" else 8
        self.K = int(self.K)
        if self.anchors == "episode" and self.K != episode_K(self.horizon, self.history):
            raise ValueError(f"anchors='episode' uses K = horizon - history + 1 = "
                             f"{episode_K(self.horizon, self.history)}; got K={self.K}")
        self.vicreg = tuple(float(v) for v in self.vicreg)
        self.rollout_k = tuple(int(k) for k in self.rollout_k)
        self.latent_k = tuple(int(k) for k in self.latent_k)
        if self.size not in SIZES and self.width is None:
            raise ValueError(f"size must be one of {list(SIZES)} (or give --width)")
        if self.steps < 1 or self.batch < 2:
            raise ValueError("need steps >= 1 and batch >= 2")


# ---------------------------------------------------------------------- runtime
def pick_device(name: str = "auto") -> torch.device:
    """'auto' -> cuda > mps > cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_runtime(seed: int, threads: Optional[int] = None, deterministic: bool = True) -> None:
    if threads:
        torch.set_num_threads(int(threads))
        try:
            torch.set_num_interop_threads(int(threads))
        except RuntimeError:                        # can only be set once per process
            pass
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def use_channels_last(mode: str, device: torch.device, model: WorldModel) -> bool:
    """'auto': channels-last conv stacks on CPU only (measured ~30 % faster there; MPS / CUDA fp32 gain little)."""
    if mode not in ("auto", "on", "off"):
        raise ValueError("channels_last must be auto | on | off")
    if model.cfg.obs_kind != "image" or mode == "off":
        return False
    return mode == "on" or device.type == "cpu"


def obs_moments(obs: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and variance of all frames in raw units (pixels scaled to [0, 1]), chunked."""
    flat = obs.reshape(-1, int(np.prod(obs.shape[2:])))
    scale = 1.0 / 255.0 if obs.dtype == np.uint8 else 1.0
    s, s2 = np.zeros(flat.shape[1]), np.zeros(flat.shape[1])
    for i in range(0, len(flat), chunk):
        x = flat[i:i + chunk].astype(np.float64) * scale
        s += x.sum(0)
        s2 += (x * x).sum(0)
    m = s / len(flat)
    return m, np.maximum(s2 / len(flat) - m * m, 0.0)


def lr_factor(step: int, steps: int, warmup: int) -> float:
    """Linear warmup over ``warmup`` steps, then cosine decay to 0 at ``steps``."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, steps - warmup))))


def make_optimizer(model: torch.nn.Module, lr: float, wd: float) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, torch.Tensor):
        return _jsonable(x.detach().cpu().tolist())
    return x


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def save_checkpoint(path, model: WorldModel, normalizer: Optional[Normalizer] = None,
                    train_config: Optional[dict] = None, steps_done: int = 0, final: bool = True) -> None:
    """model.pt in the format read by ``baselines.encode.load`` (CPU tensors, plain-python metadata)."""
    path = Path(path)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"format": CHECKPOINT_FORMAT, "model_config": model.cfg.to_dict(), "state_dict": state,
                "normalizer": None if normalizer is None else normalizer.to_dict(),
                "train_config": _jsonable(train_config or {}), "steps_done": int(steps_done), "final": bool(final)},
               tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------- evaluation (losses / errors only)
def _fixed_subset(n_total: int, n: int) -> np.ndarray:
    """The same val windows for every run (seed 0, independent of the model seed)."""
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(n_total, size=min(n, n_total), replace=False))


@torch.no_grad()
def eval_loss(model: WorldModel, ds: WindowDataset, idx: np.ndarray, device, batch: int) -> dict:
    """Mean training-loss components over fixed windows, in chunks of the training batch size."""
    model.eval()
    tot, n = {}, 0
    for i in range(0, len(idx), batch):
        chunk = idx[i:i + batch]
        if len(chunk) < 2:
            break
        b = ds.get_batch(chunk, device)
        _, logs = model.loss(b["frames"], b["actions"])
        for k, v in logs.items():
            if v.numel() == 1:
                tot[k] = tot.get(k, 0.0) + float(v) * len(chunk)
        n += len(chunk)
    model.train()
    return {k: v / max(n, 1) for k, v in tot.items()}


def _wrap(x):
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def _error_summary(pred_raw, tgt_raw, total_var: float, angular: bool) -> dict:
    """MSE / nmse in raw units (+ circular and wrapped errors for angle observations)."""
    d = pred_raw - tgt_raw
    mse = float((d ** 2).mean())
    out = {"mse": mse, "nmse": mse * d[0].numel() / total_var if total_var > 0 else None}
    if angular:
        out["circ"] = float((1.0 - torch.cos(d)).mean())
        out["mse_wrapped"] = float((_wrap(d) ** 2).mean())
        out["nmse_wrapped"] = out["mse_wrapped"] * d[0].numel() / total_var if total_var > 0 else None
    return out


@torch.no_grad()
def final_metrics(model: WorldModel, val, cfg: TrainConfig, normalizer: Optional[Normalizer], total_var: float,
                  device, batch: int = 256) -> dict:
    model.eval()
    h, T = cfg.history, val.horizon
    angular = cfg.obs == "angle_raw"

    def raw(x):
        if normalizer is None:
            return x
        return x * torch.as_tensor(normalizer.std, device=x.device) + torch.as_tensor(normalizer.mean, device=x.device)

    out: dict[str, Any] = {}
    if model.cfg.variant == "recon":
        # (1) reconstruction of the current frame on fixed val windows
        ds = WindowDataset(val, h, 1, model_action_spec(val, model), normalizer)
        idx = _fixed_subset(len(ds), cfg.n_eval_final)
        preds, tgts = [], []
        for i in range(0, len(idx), batch):
            b = ds.get_batch(idx[i:i + batch], device)
            preds.append(raw(model.decode(model.encode(b["hist"]))).cpu())
            tgts.append(raw(b["hist"][:, -1]).cpu())
        out["val_recon"] = _error_summary(torch.cat(preds), torch.cat(tgts), total_var, angular)
        # (2) open-loop rollouts from the first window of every val trajectory (+ the full training window K for
        # whole-episode training, where k = K is the last frame of the episode)
        ks = [k for k in cfg.rollout_k if h - 1 + k <= T]
        if cfg.anchors == "episode" and h - 1 + cfg.K <= T and cfg.K not in ks:
            ks.append(cfg.K)
        if ks:
            ds = WindowDataset(val, h, max(ks), model_action_spec(val, model), normalizer, anchors="start")
            res = {k: ([], []) for k in ks}
            for i in range(0, len(ds), batch):
                b = ds.get_batch(np.arange(i, min(i + batch, len(ds))), device)
                Z = model.rollout(model.encode(b["hist"]), b["actions"], max(ks))
                for k in ks:
                    res[k][0].append(raw(model.decode(Z[:, k])).cpu())
                    res[k][1].append(raw(b["future"][:, k - 1]).cpu())
            out["rollout"] = {str(k): _error_summary(torch.cat(p), torch.cat(t), total_var, angular)
                              for k, (p, t) in res.items()}
        out["rollout_k_skipped"] = [k for k in cfg.rollout_k if k not in ks]
    else:
        ks = [k for k in cfg.latent_k if h - 1 + k <= T]
        kmax = max(ks) if ks else 1
        ds = WindowDataset(val, h, kmax, model_action_spec(val, model), normalizer)
        idx = _fixed_subset(len(ds), cfg.n_eval_final)
        Z0, err, copy = [], {k: [] for k in ks}, {k: [] for k in ks}
        for i in range(0, len(idx), batch):
            b = ds.get_batch(idx[i:i + batch], device)
            B = b["frames"].shape[0]
            Zenc = model.encode(model.windows(b["frames"]).flatten(0, 1)).view(B, kmax + 1, -1)
            Zpred = model.rollout(Zenc[:, 0], b["actions"], kmax)
            Z0.append(Zenc[:, 0].cpu())
            for k in ks:
                err[k].append(((Zpred[:, k] - Zenc[:, k]) ** 2).sum(-1).cpu())
                copy[k].append(((Zenc[:, 0] - Zenc[:, k]) ** 2).sum(-1).cpu())
        Z0 = torch.cat(Z0)
        tv = float(Z0.double().var(0).sum())
        out["latent_pred_err"] = {str(k): float(torch.cat(err[k]).mean()) / tv for k in ks}
        out["latent_copy_err"] = {str(k): float(torch.cat(copy[k]).mean()) / tv for k in ks}
        out["latent_k_skipped"] = [k for k in cfg.latent_k if k not in ks]
        out.update(latent_stats(Z0))
        out["latent_total_var"] = tv
    model.train()
    return out


def model_action_spec(traj, model: WorldModel) -> dict:
    """Action spec consistent with the model's action width (one-hot width for discrete actions)."""
    if traj.actions is None:
        return {"type": "none"}
    if np.issubdtype(traj.actions.dtype, np.integer):
        return {"type": "discrete", "n": model.cfg.action_dim}
    return {"type": "continuous", "dim": model.cfg.action_dim}


# ---------------------------------------------------------------------- training
def run(cfg: TrainConfig, data: Optional[tuple] = None) -> dict:
    """Train one model. ``data`` = (spec, {"train": Trajectories, "val": Trajectories}) skips sampling.

    Returns {"model", "metrics", "config", "normalizer"}; writes files when ``cfg.out`` is set.
    """
    out_dir = Path(cfg.out) if cfg.out else None
    if out_dir is not None:
        if (out_dir / "metrics.json").exists() and not cfg.overwrite:
            raise FileExistsError(f"{out_dir} already holds a run (use --overwrite)")
        out_dir.mkdir(parents=True, exist_ok=True)
    setup_runtime(cfg.seed, cfg.threads, cfg.deterministic)
    device = pick_device(cfg.device)
    say = (lambda *a: None) if cfg.quiet else (lambda *a: print(*a, flush=True))
    if device.type == "cpu" and hasattr(os, "getloadavg"):
        load, ncpu = os.getloadavg()[0], os.cpu_count() or 1
        if load + torch.get_num_threads() > ncpu + 0.5:
            say(f"[train] warning: load {load:.1f} + {torch.get_num_threads()} threads > {ncpu} cores; OpenMP "
                "oversubscription can slow small models 10-100x (measured here) - use --threads 1 per parallel job")

    t_data = time.perf_counter()
    if data is None:
        data = make_splits(cfg.world, cfg.obs, cfg.horizon, {"train": cfg.n_train, "val": cfg.n_val},
                           cfg.world_kwargs, cfg.data_index)
    spec, splits = data
    train_traj, val_traj = splits["train"], splits["val"]
    if cfg.anchors == "episode":
        T_data = {s: int(t.obs.shape[1]) - 1 for s, t in (("train", train_traj), ("val", val_traj))}
        if set(T_data.values()) != {cfg.horizon}:
            raise ValueError(f"anchors='episode': the data horizons {T_data} must equal cfg.horizon={cfg.horizon} "
                             f"(K = horizon - history + 1 = {cfg.K} is derived from it)")
    normalizer = Normalizer.fit(train_traj) if (cfg.standardize and train_traj.obs.dtype != np.uint8) else None
    _, train_var = obs_moments(train_traj.obs)
    total_var = float(train_var.sum())
    t_data = time.perf_counter() - t_data

    angle_scale = float(normalizer.std.reshape(-1)[0]) if (normalizer is not None and cfg.loss == "circular") else 1.0
    torch.manual_seed(cfg.seed)                       # initial weights depend on the seed only (built on CPU)
    model = build_wm(spec, cfg.variant, cfg.size, cfg.latent_dim, cfg.history, cfg.K, width=cfg.width,
                     channels=cfg.channels, loss=cfg.loss, angle_scale=angle_scale, lambda_lat=cfg.lambda_lat,
                     vicreg=cfg.vicreg, norm=cfg.norm, pixel_out=cfg.pixel_out)
    info = describe(model)
    flops = train_flops_per_sample(model.cfg, batch=2)
    model.to(device)
    use_cl = use_channels_last(cfg.channels_last, device, model)
    model.set_channels_last(use_cl)
    act_spec = model_action_spec(train_traj, model)
    train_ds = WindowDataset(train_traj, cfg.history, cfg.K, act_spec, normalizer, cfg.anchors)
    val_ds = WindowDataset(val_traj, cfg.history, cfg.K, act_spec, normalizer, cfg.anchors)
    val_idx = _fixed_subset(len(val_ds), cfg.n_eval)

    opt = make_optimizer(model, cfg.lr, cfg.wd)
    warmup = cfg.warmup if cfg.warmup is not None else max(1, min(1000, cfg.steps // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, cfg.steps, warmup))
    rng = np.random.default_rng([cfg.seed, 1])        # batch order

    full_config = {
        "train": asdict(cfg), "model": model.cfg.to_dict(), "params": info, "flops": flops, "warmup": warmup,
        "data": {"spec": spec, "seeds": {s: int(t.meta["seed"]) for s, t in splits.items()},
                 "n_traj": {s: t.n for s, t in splits.items()}, "horizon": train_traj.horizon,
                 "windows": {"train": len(train_ds), "val": len(val_ds)},
                 "world_params": train_traj.meta.get("params"), "obs_total_var_train": total_var,
                 "normalizer": None if normalizer is None else normalizer.to_dict()},
        "env": {"device": str(device), "threads": torch.get_num_threads(), "channels_last": use_cl,
                "torch": torch.__version__,
                "numpy": np.__version__, "python": sys.version.split()[0], "platform": platform.platform()},
    }
    if out_dir is not None:
        _write_json(out_dir / "config.json", full_config)
    say(f"[train] {cfg.world}/{cfg.obs} {cfg.variant}-{cfg.size} seed={cfg.seed} params={info['params']} "
        f"device={device} threads={torch.get_num_threads()} windows={len(train_ds)} data={t_data:.1f}s")

    curves = {"train": [], "val": []}
    metrics: dict[str, Any] = {"curves": curves, "diverged": False}
    acc, n_acc, t0, t_last = {}, 0, time.perf_counter(), time.perf_counter()
    step = 0

    def save(final: bool):
        if out_dir is None:
            return
        save_checkpoint(out_dir / "model.pt", model, normalizer, asdict(cfg), step, final)
        _write_json(out_dir / "metrics.json", metrics)

    model.train()
    for step in range(1, cfg.steps + 1):
        diag = step % cfg.log_every == 0 or step == cfg.steps
        b = train_ds.sample_batch(rng, cfg.batch, device)
        loss, logs = model.loss(b["frames"], b["actions"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        sched.step()
        logs["grad_norm"] = gnorm.detach()
        for k, v in logs.items():
            acc[k] = acc.get(k, 0.0) + v
        n_acc += 1
        if diag:
            row = {"step": step, "lr": opt.param_groups[0]["lr"]}
            for k, v in acc.items():
                v = (v / n_acc).detach().cpu()
                row[k] = float(v) if v.numel() == 1 else [float(x) for x in v]
            if is_jepa(cfg.variant):                   # collapse diagnostics on the current batch
                with torch.no_grad():
                    z = model.encode(b["hist"])
                row["effective_rank"] = effective_rank(z)
            now = time.perf_counter()
            row["sec_per_step"] = (now - t_last) / n_acc
            row["elapsed"] = now - t0
            curves["train"].append(row)
            acc, n_acc, t_last = {}, 0, now
            say(f"[train] step {step:6d}  loss {row['loss']:.5g}  lr {row['lr']:.2e}  {row['sec_per_step']:.3f}s/step")
            if not math.isfinite(row["loss"]):
                metrics["diverged"] = True
                say("[train] non-finite loss: stopping")
                break
        if step % cfg.eval_every == 0 or step == cfg.steps:
            v = eval_loss(model, val_ds, val_idx, device, cfg.batch)
            curves["val"].append({"step": step, **v})
            say(f"[val]   step {step:6d}  loss {v.get('loss', float('nan')):.5g}")
            metrics["steps_done"] = step
            save(final=False)
    t_train = time.perf_counter() - t0
    metrics["steps_done"] = step
    if not metrics["diverged"]:
        t1 = time.perf_counter()
        metrics["final"] = final_metrics(model, val_traj, cfg, normalizer, total_var, device)
        metrics["final"]["val_loss"] = curves["val"][-1] if curves["val"] else None
        metrics["timing"] = {"data_sec": t_data, "train_sec": t_train, "final_eval_sec": time.perf_counter() - t1,
                             "sec_per_step": t_train / max(step, 1)}
    save(final=True)
    say(f"[train] done in {t_train:.1f}s")
    return {"model": model, "metrics": metrics, "config": full_config, "normalizer": normalizer}


# ---------------------------------------------------------------------- CLI
def _parse_args(argv=None) -> TrainConfig:
    d = TrainConfig()
    p = argparse.ArgumentParser(prog="python -m baselines.train", description=__doc__.split("\n\n")[0])
    p.add_argument("--world", default=d.world)
    p.add_argument("--obs", default=d.obs)
    p.add_argument("--allow-state-obs", action="store_true",
                   help="permit --obs state (the model then sees the ground-truth state: an oracle, not a baseline)")
    p.add_argument("--world-kwargs", default="{}", help="JSON dict passed to zoo.make (e.g. '{\"gamma\": 0.1}')")
    p.add_argument("--variant", default=d.variant, choices=list(VARIANTS) + sorted(VARIANT_ALIASES),
                   help="recon | jepa-sym (symmetric VICReg) | jepa-sg (stop-grad targets); jepa = jepa-sym")
    p.add_argument("--size", default=d.size, choices=list(SIZES))
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--steps", type=int, default=d.steps)
    p.add_argument("--out", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--loss", default=d.loss, choices=["mse", "circular"])
    p.add_argument("--pixel-out", default=d.pixel_out, choices=["linear", "sigmoid"])
    p.add_argument("--latent-dim", type=int, default=d.latent_dim)
    p.add_argument("--history", type=int, default=d.history)
    p.add_argument("--K", type=int, default=None, help="open-loop training steps (default 8; with --anchors "
                   "episode: horizon - history + 1)")
    p.add_argument("--lambda-lat", type=float, default=d.lambda_lat)
    p.add_argument("--vicreg", default=",".join(str(v) for v in d.vicreg), help="w_inv,w_var,w_cov")
    p.add_argument("--norm", default=d.norm, choices=["layer", "none"])
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--batch", type=int, default=d.batch)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--wd", type=float, default=d.wd)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--clip", type=float, default=d.clip)
    p.add_argument("--horizon", type=int, default=d.horizon)
    p.add_argument("--n-train", type=int, default=d.n_train)
    p.add_argument("--n-val", type=int, default=d.n_val)
    p.add_argument("--data-index", type=int, default=d.data_index)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--anchors", default=d.anchors, choices=list(ANCHORS),
                   help="all | start | episode (one open-loop window o_0..o_T per episode, K automatic)")
    p.add_argument("--device", default=d.device, help="auto (cuda > mps > cpu) | cpu | cuda | mps | cuda:1 ...")
    p.add_argument("--channels-last", default=d.channels_last, choices=["auto", "on", "off"])
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--n-eval", type=int, default=d.n_eval)
    p.add_argument("--n-eval-final", type=int, default=d.n_eval_final)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    kw = json.loads(a.world_kwargs)
    if not isinstance(kw, dict):
        raise SystemExit("--world-kwargs must be a JSON object")
    return TrainConfig(world=a.world, obs=a.obs, allow_state_obs=a.allow_state_obs, world_kwargs=kw,
                       horizon=a.horizon, n_train=a.n_train,
                       n_val=a.n_val, data_index=a.data_index, standardize=a.standardize, anchors=a.anchors,
                       variant=a.variant, size=a.size, latent_dim=a.latent_dim, history=a.history, K=a.K,
                       loss=a.loss, pixel_out=a.pixel_out, lambda_lat=a.lambda_lat,
                       vicreg=tuple(float(v) for v in a.vicreg.split(",")),
                       norm=a.norm, width=a.width, channels=a.channels, seed=a.seed, steps=a.steps, batch=a.batch,
                       lr=a.lr, wd=a.wd, warmup=a.warmup, clip=a.clip, device=a.device,
                       channels_last=a.channels_last, threads=a.threads,
                       deterministic=a.deterministic, log_every=a.log_every, eval_every=a.eval_every,
                       n_eval=a.n_eval, n_eval_final=a.n_eval_final, out=a.out, overwrite=a.overwrite,
                       quiet=a.quiet)


def main(argv=None) -> int:
    cfg = _parse_args(argv)
    res = run(cfg)
    return 1 if res["metrics"]["diverged"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
