"""Windowed training data for world models, built from zoo ``Trajectories``.

Only ``Trajectories.obs`` and ``Trajectories.actions`` are ever read here; ``state`` / ``info`` are ground
truth for analysis only (see ``zoo/base.py``) and never reach a model.

Item at anchor time t (history h, horizon K), all indices into the trajectory's observation clock:
    hist     o_{t-h+1} .. o_t            [h, *obs_shape]
    future   o_{t+1}   .. o_{t+K}        [K, *obs_shape]
    actions  a_t       .. a_{t+K-1}      [K, act_dim]   (absent for autonomous worlds)
Alignment: a_t maps state_t -> state_{t+1} (zoo convention: ``actions[:, k]`` acts between obs k and k+1).
Discrete actions are one-hot encoded, continuous actions are passed raw. Anchors (``anchors=``):
    "all"      every h-1 <= t <= T-K (default)
    "start"    only t = h-1: one window per episode whose history starts at o_0, any K <= T-h+1
    "episode"  only t = h-1 AND K = T-h+1 (the default when K is None): the whole episode as ONE open-loop window,
               o_0 .. o_T, so a model must carry any hidden state open-loop from the first observation (cell 6;
               with history 1 the window is o_0 plus all T actions and T future frames, e.g. K = 64 for the zoo's
               horizon 64, which has 65 frames). T is the number of transitions (zoo ``horizon``).

Internally a batch is gathered as one contiguous block ``frames`` = o_{t-h+1} .. o_{t+K} [B, h+K, *obs_shape];
``hist`` / ``future`` are views of it. Pixel observations stay uint8 in memory and are converted to float32 / 255
per batch, on the target device. Low-dimensional observations can be standardised with statistics of the TRAIN
split only (``Normalizer.fit`` on the train trajectories, then passed to every split).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

import zoo
from zoo.base import SPLIT_SEEDS, Trajectories

__all__ = ["ANCHORS", "Normalizer", "WindowDataset", "episode_K", "obs_kind", "action_dim", "make_splits",
           "SPLIT_SEEDS"]

ANCHORS = ("all", "start", "episode")


def episode_K(T: int, history: int) -> int:
    """Open-loop steps of the whole-episode window (anchors='episode'): o_0..o_{h-1} as history, then o_h..o_T."""
    return int(T) - int(history) + 1


def obs_kind(spec: dict) -> str:
    """'image' for uint8 observations of shape (H, W) or (C, H, W), else 'vector' (flattened by the model)."""
    shape = tuple(spec["obs_shape"])
    return "image" if spec["obs_dtype"] == "uint8" and len(shape) in (2, 3) else "vector"


def action_dim(spec: dict) -> int:
    """Width of the action vector fed to the transition: n for discrete (one-hot), dim for continuous, 0 if none."""
    act = spec.get("action", {"type": "none"})
    return {"none": 0, "discrete": int(act.get("n", 0)), "continuous": int(act.get("dim", 0))}[act["type"]]


@dataclass
class Normalizer:
    """Per-feature affine standardisation x -> (x - mean) / std (vector observations only)."""

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self):                     # float32 always: these arrays are moved to the device (MPS: no f64)
        self.mean = np.asarray(self.mean, np.float32)
        self.std = np.asarray(self.std, np.float32)

    @classmethod
    def fit(cls, traj: Trajectories, eps: float = 1e-6) -> "Normalizer":
        """Fit on ALL frames of one split's observations (call it on the train split only)."""
        if traj.obs.dtype == np.uint8:
            raise ValueError("pixel observations are scaled by 1/255, not standardised")
        x = traj.obs.reshape((-1,) + traj.obs.shape[2:]).astype(np.float64)
        return cls(mean=x.mean(0).astype(np.float32), std=np.maximum(x.std(0), eps).astype(np.float32))

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["Normalizer"]:
        if d is None:
            return None
        return cls(mean=np.asarray(d["mean"], np.float32), std=np.asarray(d["std"], np.float32))


def frames_to_tensor(x: np.ndarray, device, normalizer: Optional[Normalizer] = None) -> torch.Tensor:
    """numpy observations -> float32 model input on ``device``.

    uint8 is moved first (4x fewer bytes), then scaled by 1/255 on the device. Every other dtype is cast to float32
    on the HOST before the move: MPS has no float64, so moving e.g. a float64 array first would fail there.
    """
    x = np.ascontiguousarray(x)
    if x.dtype != np.uint8 and x.dtype != np.float32:
        x = x.astype(np.float32)
    t = torch.from_numpy(x).to(device)
    if t.dtype == torch.uint8:
        return t.float().div_(255.0)
    if normalizer is not None:
        mean = torch.as_tensor(normalizer.mean, device=device)
        std = torch.as_tensor(normalizer.std, device=device)
        t = (t - mean) / std
    return t


class WindowDataset:
    """All (trajectory, anchor) windows of one split. Indexable (``__getitem__``) and batchable (``get_batch``)."""

    def __init__(self, traj: Trajectories, history: int = 2, K: Optional[int] = 8,
                 action_spec: Optional[dict] = None, normalizer: Optional[Normalizer] = None, anchors: str = "all"):
        self.obs, self.raw_actions = traj.obs, traj.actions          # no copies; state / info are never touched
        self.N, self.T = int(self.obs.shape[0]), int(self.obs.shape[1]) - 1
        K = episode_K(self.T, history) if (K is None and anchors == "episode") else K
        if K is None:
            raise ValueError("K is required unless anchors='episode'")
        if history < 1 or K < 1:
            raise ValueError("need history >= 1 and K >= 1")
        if anchors == "episode" and K != episode_K(self.T, history):
            raise ValueError(f"anchors='episode' needs K = T - history + 1 = {episode_K(self.T, history)} "
                             f"(got K={K}); use anchors='start' for a shorter window from the episode start")
        self.h, self.K, self.normalizer = int(history), int(K), normalizer
        self.obs_shape = tuple(self.obs.shape[2:])
        if self.normalizer is not None and self.obs.dtype == np.uint8:
            raise ValueError("a Normalizer is only for vector observations")
        self.action_spec = action_spec if action_spec is not None else self._infer_action_spec(traj.actions)
        self.act_type, self.act_dim = self.action_spec["type"], action_dim({"action": self.action_spec})
        if (self.act_type == "none") != (traj.actions is None):
            raise ValueError("action spec does not match the trajectories (actions present iff type != 'none')")
        if self.T - self.K < self.h - 1:
            raise ValueError(f"horizon T={self.T} too short for history={self.h}, K={self.K}")
        if anchors == "all":
            self.anchors = np.arange(self.h - 1, self.T - self.K + 1)
        elif anchors in ("start", "episode"):
            self.anchors = np.array([self.h - 1])
        else:
            raise ValueError(f"anchors must be one of {ANCHORS}")
        self.anchor_mode = anchors
        self.offsets = np.arange(-(self.h - 1), self.K + 1)          # frame offsets relative to the anchor t
        self.act_offsets = np.arange(self.K)

    @staticmethod
    def _infer_action_spec(actions) -> dict:
        if actions is None:
            return {"type": "none"}
        if np.issubdtype(actions.dtype, np.integer):
            return {"type": "discrete", "n": int(actions.max()) + 1}
        return {"type": "continuous", "dim": int(actions.shape[-1])}

    def __len__(self) -> int:
        return self.N * len(self.anchors)

    def locate(self, idx) -> tuple[np.ndarray, np.ndarray]:
        """Flat window index -> (trajectory index n, anchor time t)."""
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError("window index out of range")
        return idx // len(self.anchors), self.anchors[idx % len(self.anchors)]

    def _gather(self, idx):
        n, t = self.locate(idx)
        frames = self.obs[n[:, None], t[:, None] + self.offsets[None, :]]               # [B, h+K, *obs]
        acts = None
        if self.raw_actions is not None:
            acts = self.raw_actions[n[:, None], t[:, None] + self.act_offsets[None, :]]  # [B, K] or [B, K, dim]
        return n, t, frames, acts

    def _encode_actions(self, acts: Optional[np.ndarray], device) -> Optional[torch.Tensor]:
        if acts is None:
            return None
        a = torch.from_numpy(np.ascontiguousarray(acts))
        if self.act_type == "discrete":                      # one-hot on the CPU (small), then move
            return torch.nn.functional.one_hot(a.long(), self.act_dim).float().to(device)
        return a.float().reshape(a.shape[0], self.K, self.act_dim).to(device)

    def __getitem__(self, i: int) -> dict[str, Any]:
        """One raw item (numpy; pixels stay uint8, not normalised). Training uses ``get_batch`` instead."""
        n, t, frames, acts = self._gather(np.array([i]))
        item = {"hist": frames[0, :self.h], "future": frames[0, self.h:], "index": (int(n[0]), int(t[0]))}
        if acts is not None:
            a = acts[0]
            item["actions"] = np.eye(self.act_dim, dtype=np.float32)[a] if self.act_type == "discrete" \
                else a.astype(np.float32).reshape(self.K, self.act_dim)
        return item

    def get_batch(self, idx, device="cpu") -> dict[str, Any]:
        """Batch of windows as float32 tensors on ``device``.

        Keys: frames [B, h+K, *obs] (o_{t-h+1}..o_{t+K}), hist / future (views of frames), actions [B, K, A] or
        None, index (n, t) int64 numpy arrays [B] (bookkeeping only; never fed to a model).
        """
        n, t, frames, acts = self._gather(idx)
        x = frames_to_tensor(frames, device, self.normalizer)
        return {"frames": x, "hist": x[:, :self.h], "future": x[:, self.h:],
                "actions": self._encode_actions(acts, device), "index": (n, t)}

    def sample_batch(self, rng: np.random.Generator, batch_size: int, device="cpu") -> dict[str, Any]:
        """Uniformly random windows (with replacement) drawn with a numpy Generator (deterministic given rng)."""
        return self.get_batch(rng.integers(0, len(self), batch_size), device)


def make_splits(world: str, obs_mode: str, horizon: int = 64, n_traj: Optional[dict] = None,
                world_kwargs: Optional[dict] = None, index: int = 0) -> tuple[dict, dict[str, Trajectories]]:
    """Sample the requested splits from disjoint seeds ``SPLIT_SEEDS[split] + index``.

    Returns (spec, {split: Trajectories}); ``n_traj`` maps split name -> number of trajectories
    (default {"train": 2000, "val": 200}).
    """
    n_traj = {"train": 2000, "val": 200} if n_traj is None else n_traj
    w = zoo.make(world, **(world_kwargs or {}))
    spec = w.spec(obs_mode)
    out = {s: w.sample(seed=zoo.split_seed(s, index), horizon=horizon, n_traj=int(n), obs_mode=obs_mode)
           for s, n in n_traj.items() if int(n) > 0}
    return spec, out
