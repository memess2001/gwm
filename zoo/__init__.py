"""GWM world zoo: worlds whose ground truth is a theorem. See ``zoo/base.py`` for the API contract.

    import zoo
    w = zoo.make("pendulum", gamma=0.0)
    tr = w.sample(seed=zoo.split_seed("train", 0), horizon=100, n_traj=256, obs_mode="pixels64")
    gt = w.ground_truth(**tr.meta["params"])
"""
from __future__ import annotations

from .base import SPLIT_SEEDS, Trajectories, World
from .pendulum import Pendulum
from .toggle_grid import ToggleGrid
from .wright_fisher import WrightFisher

_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Register a World subclass under ``cls.name`` (usable as a decorator)."""
    if not (isinstance(cls, type) and issubclass(cls, World)) or cls.name in _REGISTRY:
        raise ValueError(f"cannot register {cls!r}")
    _REGISTRY[cls.name] = cls
    return cls


for _cls in (Pendulum, ToggleGrid, WrightFisher):
    register(_cls)


def make(name: str, **params) -> World:
    """Instantiate a registered world with parameters (unknown parameters raise TypeError)."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown world {name!r}; available: {list_worlds()}")
    return _REGISTRY[name](**params)


def list_worlds() -> list[str]:
    return sorted(_REGISTRY)


def split_seed(split: str, i: int = 0) -> int:
    """Seed of the i-th dataset of a split (train / val / test / analysis); splits never share seeds."""
    if split not in SPLIT_SEEDS or not 0 <= i < 1_000_000:
        raise ValueError(f"split must be one of {sorted(SPLIT_SEEDS)} and 0 <= i < 1e6")
    return SPLIT_SEEDS[split] + i


__all__ = ["World", "Trajectories", "SPLIT_SEEDS", "Pendulum", "ToggleGrid", "WrightFisher",
           "make", "list_worlds", "register", "split_seed"]
