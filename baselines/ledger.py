"""Append rows to the experiment ledger ``results/ledger.md`` (HANDOFF section 5, step 4).

    from baselines.ledger import append_entry
    append_entry("2026-11-05", 0, "pendulum", "angle_cs", "recon", "M", 499965, 0,
                 {"h1_pass": True, "knn_r2": 0.97}, "P0.1-pass", "gate ok")

The ledger is a markdown table with the columns of ``COLUMNS``; the row is inserted directly after the last row
of that table (so the table stays contiguous even if text follows it), cells are escaped (``|`` -> ``\\|``,
newlines -> spaces) and the file is rewritten atomically under a lock file, so parallel runs can append safely.
Ledger rules (see the file header): one row per finished configuration, only quantities listed in the
pre-registration, verdicts only from categories the pre-registration defines. Never write test rows to the real
ledger: pass ``path=`` (e.g. a temporary copy).
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

__all__ = ["COLUMNS", "DEFAULT_LEDGER", "append_entry", "format_metrics", "read_rows"]

COLUMNS = ("date", "cell", "world", "obs", "variant", "size", "params", "seed", "metrics", "verdict", "notes")
DEFAULT_LEDGER = Path(__file__).resolve().parents[1] / "results" / "ledger.md"
_PIPE = re.compile(r"(?<!\\)\|")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _split_row(line: str) -> Optional[list[str]]:
    """Cells of a markdown table row (unescaped pipes split cells), or None if the line is not a row."""
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")) or len(s) < 2:
        return None
    return [c.strip() for c in _PIPE.split(s[1:-1])]


def _cell(x: Any) -> str:
    s = " ".join(str(x).split())            # collapse newlines / runs of whitespace
    return s.replace("\\|", "|").replace("|", "\\|")


def _fmt(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return "NA" if math.isnan(v) else f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt(u) for u in v) + "]"
    try:                                     # numpy scalars and 0-d arrays
        return _fmt(v.item())
    except (AttributeError, ValueError):
        return str(v)


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """{'a': 1.23456, 'r': {'1': {'mse': 0.1}}} -> 'a=1.235; r.1.mse=0.1' (nested dicts flattened with dots)."""
    parts = []
    for k, v in metrics.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            sub = format_metrics(v, key + ".")
            if sub:
                parts.append(sub)
        else:
            parts.append(f"{key}={_fmt(v)}")
    return "; ".join(parts)


@contextmanager
def _locked(path: Path, timeout: float = 30.0, stale: float = 120.0):
    lock = path.with_name(path.name + ".lock")
    t0 = time.monotonic()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale:     # left behind by a crashed writer
                    lock.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"ledger lock {lock} held for more than {timeout}s")
            time.sleep(0.05)
    try:
        os.close(fd)
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _table_bounds(lines: list[str]) -> tuple[int, int]:
    """(header line index, index one past the last table row)."""
    for i, line in enumerate(lines):
        if _split_row(line) == list(COLUMNS):
            sep = _split_row(lines[i + 1]) if i + 1 < len(lines) else None
            if sep is None or len(sep) != len(COLUMNS) or not all(re.fullmatch(r":?-{3,}:?", c) for c in sep):
                raise ValueError("ledger header is not followed by a markdown separator row")
            end = i + 2
            while end < len(lines) and _split_row(lines[end]) is not None:
                end += 1
            return i, end
    raise ValueError(f"no ledger table with columns {COLUMNS} found")


def read_rows(path: Union[str, Path, None] = None) -> list[dict]:
    """The ledger rows as dicts (cells unescaped)."""
    lines = Path(path or DEFAULT_LEDGER).read_text(encoding="utf-8").splitlines()
    h, end = _table_bounds(lines)
    return [dict(zip(COLUMNS, (c.replace("\\|", "|") for c in _split_row(l)))) for l in lines[h + 2:end]]


def append_entry(date, cell, world: str, obs: str, variant: str, size: str, params: int, seed: int,
                 metrics: dict, verdict: str, notes: str = "", path: Union[str, Path, None] = None) -> str:
    """Append one row to the ledger table and return it. ``date`` is a date or 'YYYY-MM-DD' (actual run date)."""
    if isinstance(date, (_dt.date, _dt.datetime)):
        date = date.isoformat()[:10]
    if not isinstance(date, str) or not _DATE.match(date):
        raise ValueError("date must be a datetime.date or 'YYYY-MM-DD'")
    if not str(verdict).strip():
        raise ValueError("verdict must be a non-empty category from the pre-registration")
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")
    cells = [date, cell, world, obs, variant, size, int(params), int(seed), format_metrics(metrics), verdict, notes]
    row = "| " + " | ".join(_cell(c) for c in cells) + " |"
    if len(_split_row(row)) != len(COLUMNS):
        raise AssertionError("internal error: malformed ledger row")
    path = Path(path or DEFAULT_LEDGER)
    with _locked(path):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        _, end = _table_bounds(lines)
        lines.insert(end, row)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    return row
