"""Load a baseline checkpoint and compute latents z_t for a zoo ``Trajectories`` object.

    from baselines.encode import load, encode
    model, normalizer, ckpt = load("runs/c0_pix_recon_M_s0")          # run directory or path to model.pt
    z, t = encode(model, traj, normalizer)                             # z [N, T-h+2, d], t = h-1 .. T

This is the hand-off point to the pre-registered analysis (``analysis/``). It only reads ``traj.obs``. Do not run
it on trained models before the corresponding pre-registration has been approved.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from zoo.base import Trajectories

from .data import Normalizer, frames_to_tensor
from .mlp_wm import WMConfig, WorldModel

__all__ = ["load", "encode"]

CHECKPOINT_FORMAT = "gwm-baseline-v1"


def load(path: Union[str, Path], device: Union[str, torch.device] = "cpu"):
    """(model in eval mode on ``device``, Normalizer or None, raw checkpoint dict) from a run dir or model.pt."""
    p = Path(path)
    if p.is_dir():
        p = p / "model.pt"
    ckpt = torch.load(p, map_location="cpu", weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{p} is not a {CHECKPOINT_FORMAT} checkpoint")
    mc = dict(ckpt["model_config"])
    mc.setdefault("pixel_out", "sigmoid")              # checkpoints from before the option always used a sigmoid
    with torch.random.fork_rng(devices=[]):            # building the module draws init weights that are overwritten
        model = WorldModel(WMConfig.from_dict(mc))     # below: keep the caller's torch RNG stream untouched
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, Normalizer.from_dict(ckpt.get("normalizer")), ckpt


@torch.no_grad()
def encode(model: WorldModel, traj: Trajectories, normalizer: Optional[Normalizer] = None,
           batch_size: int = 2048, device: Optional[Union[str, torch.device]] = None):
    """Latents z_t = E(o_{t-h+1}, ..., o_t) for every trajectory and every t = h-1 .. T.

    Returns (z float32 [N, T-h+2, d], t int64 [T-h+2]). Pass the checkpoint's normaliser (from ``load``) so the
    inputs are scaled exactly as in training. Windows are processed a few trajectories at a time, so pixel data
    are converted to float only chunk by chunk.
    """
    device = next(model.parameters()).device if device is None else torch.device(device)
    was_training = model.training
    model.eval()
    h, d = model.cfg.history, model.cfg.latent_dim
    N, T1 = traj.obs.shape[:2]
    if T1 < h:
        raise ValueError("trajectories shorter than the model's history window")
    times = np.arange(h - 1, T1)
    out = np.empty((N, len(times), d), np.float32)
    per = max(1, batch_size // len(times))                       # trajectories per chunk
    for i in range(0, N, per):
        x = frames_to_tensor(traj.obs[i:i + per], device, normalizer)                  # [n, T+1, *obs]
        win = model.windows(x)                                                          # [n, T-h+2, h, *obs]
        z = model.encode(win.flatten(0, 1)).view(x.shape[0], len(times), d)
        out[i:i + per] = z.cpu().numpy()
    model.train(was_training)
    return out, times
