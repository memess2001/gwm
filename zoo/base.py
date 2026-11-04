"""World API contract for the GWM zoo ("答案是定理的" worlds).

Every world is a deterministic simulator whose ground truth is known analytically.

Rules (all worlds):
- ``sample(seed, ...)`` is a pure function of its arguments: all randomness comes from
  ``np.random.default_rng(seed)``. Splits use disjoint seeds (see ``SPLIT_SEEDS``).
- ``Trajectories.state`` and ``Trajectories.info`` are ground truth for ANALYSIS ONLY.
  Models must only ever see ``obs`` and ``actions``.
- ``ground_truth(**cfg)`` returns world-level truths (topology, conserved quantities,
  divergence, holonomy, geodesic labels, ...). Where a truth depends on the sampling
  distribution (e.g. the Betti numbers of the *sampled region* of phase space), it must be
  computed from ``cfg`` rather than hard-coded.
- No torch inside ``zoo/`` (numpy / scipy only), so worlds run anywhere.

Shapes: N trajectories, T steps (T+1 observations per trajectory).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# Disjoint seed bases for data splits. A split's i-th dataset uses seed = base + i.
SPLIT_SEEDS = {"train": 0, "val": 1_000_000, "test": 2_000_000, "analysis": 3_000_000}


@dataclass
class Trajectories:
    obs: np.ndarray                  # [N, T+1, *obs_shape]; float32, or uint8 for pixel modes
    state: np.ndarray                # [N, T+1, state_dim] ground-truth state (analysis only)
    actions: Optional[np.ndarray]    # [N, T] int64 (discrete) | [N, T, act_dim] float32 | None
    t: np.ndarray                    # [T+1] float64 observation times
    info: dict = field(default_factory=dict)  # per-step truths, leading dims [N, T+1] or [N, T] or [N]
    meta: dict = field(default_factory=dict)  # echo of the sampling config (seed, horizon, obs_mode, params)

    @property
    def n(self) -> int:
        return int(self.obs.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.obs.shape[1] - 1)


class World:
    """Base class. Subclasses set ``name`` and ``obs_modes`` and implement the methods."""

    name: str = "base"
    obs_modes: tuple = ()

    def sample(self, seed: int, horizon: int, n_traj: int, obs_mode: str, **kwargs) -> Trajectories:
        raise NotImplementedError

    def observe(self, state: np.ndarray, obs_mode: str, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Map ground-truth states [..., state_dim] to observations [..., *obs_shape]."""
        raise NotImplementedError

    def ground_truth(self, **cfg) -> dict[str, Any]:
        raise NotImplementedError

    def spec(self, obs_mode: str) -> dict[str, Any]:
        """{"obs_shape": tuple, "obs_dtype": str,
            "action": {"type": "none"} | {"type": "discrete", "n": int} | {"type": "continuous", "dim": int}}"""
        raise NotImplementedError
