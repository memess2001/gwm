"""Cell 6 v1: conditions, data-tier policies, manipulation check, scripted gate set, analysis sets.

Pinned by cells/6/prereg.md (v1). Import with ``sys.path.insert(0, "cells/6")`` from the repo root.
Nothing in this file touches a trained model except ``gate_eval`` / ``model_check`` (which take a Predictor).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import zoo                                                     # noqa: E402
from zoo.base import SPLIT_SEEDS, Trajectories                  # noqa: E402
from analysis import holonomy as ho                              # noqa: E402

HORIZON = 63                    # transitions per episode (64 frames)
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# ---------------------------------------------------------------- conditions (k = 1 only; prereg item 7)
HIDDEN = dict(n_doors=1)
CONDITIONS = {
    "H-act": dict(world=dict(HIDDEN), policy="active"),            # primary tier (P6.1, P6.2, P6.6)
    "H-pc": dict(world=dict(HIDDEN), policy="passive_cover"),
    "H-pn": dict(world=dict(HIDDEN), policy="passive_nocover"),
    "C-open": dict(world=dict(n_doors=1, switches_enabled=False, init_doors=(1,)), policy="active"),
    "C-vis": dict(world=dict(n_doors=1, door_visible=True), policy="active"),
}
N_TRAIN_MAIN, N_VAL = 2000, 200
DATA_EFF_N = (250, 500)                 # extra H-act runs (P6.6 data efficiency; 2000 = the main run)
P_HOME_GRID = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1)
LOOP_LEN = (2, 8)                       # random-walk part of each active-tier loop (zoo default)
CAL_SEED_OFFSET = 777                   # coverage-matching seed: SPLIT_SEEDS["train"] + 777 (never trained on)


def make_world(cond: str):
    return zoo.make("toggle_grid", **CONDITIONS[cond]["world"])


# ---------------------------------------------------------------- policies (prereg item 9)
def _door_adjacent(w):
    dx, dy = w.doors[0]
    return dx - 1, dy


def sample_policy(w, policy: str, seed: int, n_traj: int, horizon: int = HORIZON, p_home: float = 0.0):
    """(states [n, T+1, 3], actions [n, T]) from the room-0 start distribution of ``w``.

    passive_cover   : uniform random actions (= zoo policy 'random').
    passive_nocover : uniform random, except that on the cell left of the door 'right' is replaced by a uniform
                      choice of up / down / left: the door is never attempted, the hidden bit is never observable.
    active          : concatenated loops (random walk of LOOP_LEN steps, then greedy shortest return home, as the
                      zoo 'loops' policy) and, after each closed loop with probability p_home, a homing probe:
                      shortest path to the cell left of the door, then one 'right' (a door attempt).
    """
    rng = np.random.default_rng(seed)
    s0 = w._initial(rng, n_traj, w.cfg)
    if policy == "passive_cover":
        acts = rng.integers(0, 4, (n_traj, horizon)).astype(np.int64)
        return w.trajectory(s0, acts), acts
    if policy == "passive_nocover":
        ax, ay = _door_adjacent(w)
        st = np.empty((n_traj, horizon + 1, 3), np.int64)
        acts = np.empty((n_traj, horizon), np.int64)
        st[:, 0], s = s0, s0
        for t in range(horizon):
            a = rng.integers(0, 4, n_traj)
            alt = rng.choice(np.array([UP, DOWN, LEFT]), n_traj)
            at_door = (s[:, 0] == ax) & (s[:, 1] == ay)
            a = np.where(at_door & (a == RIGHT), alt, a)
            s = w.step(s, a)
            st[:, t + 1], acts[:, t] = s, a
        return st, acts
    if policy != "active":
        raise ValueError(policy)
    ax, ay = _door_adjacent(w)
    arow = w.cell_row[ay * w.W + ax]
    st = np.empty((n_traj, horizon + 1, 3), np.int64)
    acts = np.empty((n_traj, horizon), np.int64)
    for i in range(n_traj):
        s = s0[i].copy()
        st[i, 0] = s
        t = 0
        mode, home, rem = "walk", w._row(s), int(rng.integers(LOOP_LEN[0], LOOP_LEN[1] + 1))
        while t < horizon:
            sid = w.encode(s)
            if mode == "walk":
                a = int(rng.integers(0, 4))
                rem -= 1
                if rem <= 0:
                    mode = "return"
            elif mode == "return":
                d = w.D[home, w.NS[sid]] + 0.5 * rng.random(4)
                a = int(np.argmin(d))
            elif mode == "home_to_door":
                if w._row(s) == arow:
                    a, mode = RIGHT, "attempted"
                else:
                    d = w.D[arow, w.NS[sid]] + 0.5 * rng.random(4)
                    a = int(np.argmin(d))
            s = w.step(s, a)
            acts[i, t] = a
            st[i, t + 1] = s
            t += 1
            if mode == "return" and w._row(s) == home:
                mode = "home_to_door" if rng.random() < p_home else "next"
            if mode == "attempted":
                mode = "next"
            if mode == "next":
                mode, home, rem = "walk", w._row(s), int(rng.integers(LOOP_LEN[0], LOOP_LEN[1] + 1))
    return st, acts


def to_traj(w, st, acts, seed, policy, extra=None) -> Trajectories:
    T = acts.shape[1]
    meta = dict(world="toggle_grid", seed=int(seed), horizon=T, n_traj=len(st), obs_mode="grid",
                params=dict(w.cfg), policy=policy, **(extra or {}))
    return Trajectories(obs=w.observe(st, "grid"), state=st, actions=acts,
                        t=np.arange(T + 1, dtype=np.float64), info={}, meta=meta)


def coverage(w, st, acts) -> dict:
    """Manipulation check: door attempts (by switch entries so far and by door state) and room-1 visits."""
    ax, ay = _door_adjacent(w)
    pos, nxt = st[:, :-1, :2], st[:, 1:, :2]
    att = (pos[..., 0] == ax) & (pos[..., 1] == ay) & (acts == RIGHT)
    bits = st[:, :-1, 2]
    moved = (nxt != pos).any(-1)
    ent = moved & (w.switch_id[nxt[..., 1], nxt[..., 0]] >= 0)
    nent = np.concatenate([np.zeros((len(st), 1), np.int64), np.cumsum(ent, 1)[:, :-1]], 1)
    room = w.room[st[..., 1], st[..., 0]]
    out = {"episodes": int(len(st)), "door_attempts": int(att.sum()), "open_door_attempts": int((att & (bits == 1)).sum()),
           "frames_outside_R0": float((room != 0).mean()), "episodes_leaving_R0": float((room != 0).any(1).mean())}
    for n in range(4):
        out[f"attempts_after_{n}_entries"] = int((att & (nent == n)).sum())
    out["attempts_after_4plus_entries"] = int((att & (nent >= 4)).sum())
    return out


def choose_p_home(n: int = 2000) -> dict:
    """Coverage matching (pinned): the p_home in P_HOME_GRID whose open-door attempts on the matching seed are
    closest to passive_cover's on the same seed (hidden world). Ties -> the smaller p_home."""
    w = make_world("H-act")
    seed = SPLIT_SEEDS["train"] + CAL_SEED_OFFSET
    st, a = sample_policy(w, "passive_cover", seed, n)
    target = coverage(w, st, a)["open_door_attempts"]
    rows = []
    for p in P_HOME_GRID:
        st, a = sample_policy(w, "active", seed, n, p_home=p)
        rows.append((p, coverage(w, st, a)["open_door_attempts"]))
    best = min(rows, key=lambda r: (abs(r[1] - target), r[0]))
    return {"p_home": best[0], "target_open_attempts": target, "grid": rows, "seed": seed}


# ---------------------------------------------------------------- gate set (prereg item 3)
def gate_set(w, seed: int, n_per_cell: int = 200, n_values=(0, 1, 2, 3), max_prefix: int = 30):
    """Scripted balanced door gate: each episode starts in room 0, walks a switch-avoiding random prefix of
    U{0..max_prefix} steps, enters the switch exactly n times, walks to the cell left of the door and attempts
    'right'. Returns a list of dicts {s0, actions, n, outcome_pos}; total length <= HORIZON."""
    rng = np.random.default_rng(seed)
    ax, ay = _door_adjacent(w)
    out = []
    for n in n_values:
        k = 0
        while k < n_per_cell:
            s0 = w._initial(rng, 1, w.cfg)[0]
            if w.switch_id[s0[1], s0[0]] >= 0:
                continue
            P = int(rng.integers(0, max_prefix + 1))
            pre = []
            s = s0.copy()
            for _ in range(P):
                opts = [a for a in range(4) if not ho._entered(w, s, a)]
                a = int(rng.choice(opts))
                pre.append(a)
                s = w.step(s, a)
            if w.switch_id[s[1], s[0]] >= 0 or (int(s[0]), int(s[1])) == (ax, ay):
                continue
            p = ho.constructive_path(w, s, n, rng, end_cell=(ax, ay), max_len=HORIZON - 1 - P, tries=20)
            if p is None:
                continue
            acts = np.array(pre + list(p.actions) + [RIGHT], dtype=np.int64)
            if len(acts) > HORIZON:
                continue
            stt = w.trajectory(s0, acts)
            out.append({"s0": s0, "actions": acts, "n": n, "states": stt})
            k += 1
    return out


def rollout_positions(pred, s0s: np.ndarray, act_list: list) -> list:
    """Predicted positions (decode of every latent after each action) along each action sequence."""
    L = np.array([len(a) for a in act_list])
    A = np.zeros((len(act_list), L.max()), np.int64)
    for i, a in enumerate(act_list):
        A[i, :len(a)] = a
    Z = pred.start(s0s)
    out = np.zeros((len(act_list), L.max(), 2), np.int64)
    for t in range(L.max()):
        m = t < L
        Z = pred.where(m, pred.step(Z, A[:, t]), Z)
        out[:, t] = pred.decode(Z)[:, :2]
    return [out[i, :L[i]] for i in range(len(act_list))]


def gate_eval(pred, gate: list) -> dict:
    """Door gate per n (agent position after the final attempt) and position gate (all earlier steps)."""
    s0s = np.stack([g["s0"] for g in gate])
    preds = rollout_positions(pred, s0s, [g["actions"] for g in gate])
    door, pos_ok, pos_n = {}, 0, 0
    for g, p in zip(gate, preds):
        true = g["states"][1:, :2]
        ok = (p == true).all(1)
        door.setdefault(g["n"], []).append(bool(ok[-1]))
        pos_ok += int(ok[:-1].sum())
        pos_n += len(ok) - 1
    res = {"door_acc": {int(n): float(np.mean(v)) for n, v in sorted(door.items())},
           "door_n": {int(n): len(v) for n, v in sorted(door.items())},
           "position_acc": pos_ok / max(pos_n, 1), "position_n": pos_n}
    return res


GATE_THRESHOLD = 0.95


def gate_pass(res: dict, door_gate: bool = True) -> bool:
    ok = res["position_acc"] >= GATE_THRESHOLD
    if door_gate:
        ok = ok and all(v >= GATE_THRESHOLD for v in res["door_acc"].values())
    return bool(ok)


# ---------------------------------------------------------------- analysis sets (prereg items 2, 4, 6, 8, 13)
N_EVAL_Z, N_CAL_Z = 100, 50
PREFIX_MAX = 15                                   # prefix + |gamma| (<= 28) + |probe| (20) <= 63
MAX_LOOP = 28
N_LOOPS = {"odd": 16, "even2": 16, "null": 8}     # per z
ODD_N = (1, 3)
N_GENERIC_PAIRS = 16                              # = number of R0 plaquettes (equal query budget)
PAIR_CLASSES = {"odd": [(1, 0), (0, 1), (1, 2), (2, 1)], "even2": [(1, 1), (2, 0), (0, 2), (2, 2)], "null": [(0, 0)]}


def sample_zs(w, seed: int, n: int):
    """Episode start states, uniform-random prefixes of U{0..PREFIX_MAX} steps, and the true state at z
    (never on the switch cell)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s0 = w._initial(rng, 1, w.cfg)[0]
        P = int(rng.integers(0, PREFIX_MAX + 1))
        pre = rng.integers(0, 4, P).astype(np.int64)
        sz = w.trajectory(s0, pre)[-1]
        if w.switch_id[sz[1], sz[0]] >= 0:
            continue
        out.append({"s0": s0, "prefix": pre, "sz": sz})
    return out


def loops_for_z(w, z: dict, rng) -> list:
    """Per z: N_LOOPS odd / even(>= 2 entries) / null position-closed loops, each with a time-matched switch-free
    reference of the same length. obs_closed: false only for odd loops in the visible-door world."""
    vis = bool(w.cfg["door_visible"])
    out = []
    for cls, m in N_LOOPS.items():
        k = 0
        while k < m:
            n = {"odd": int(rng.choice(ODD_N)), "even2": 2, "null": 0}[cls]
            p = ho.constructive_path(w, z["sz"], n, rng, max_len=MAX_LOOP)
            if p is None:
                continue
            ref = ho.null_reference(w, z["sz"], len(p.actions), rng)
            if ref is None:
                continue
            flips = int(w.holonomy(z["sz"], p.actions)["flips"].any())
            out.append({"cls": cls, "n": n, "gamma": p.actions, "ref": ref, "length": len(p.actions),
                        "true_holonomy": flips, "obs_closed": not (vis and n % 2 == 1)})
            k += 1
    return out


def plaquettes_for_z(w, z: dict, rng, room: int) -> list:
    corners = ho.plaquette_corners(w, room)
    out = []
    for c in corners:
        r = ho.plaquette_probe(w, z["sz"], c, rng, avoid_switch_transport=(room == 0), max_transport=24)
        out.append(None if r is None else {"corner": c, "gamma": r[0], "ref": r[1]})
    return out


def detection_pairs_for_z(w, z: dict, rng) -> list:
    """Baseline detection pairs: two constructive paths from z with (n1, n2) entries ending on the same cell
    (a random room-0 cell, not the switch, not z's own cell), equal length via wall-bump padding."""
    vis = bool(w.cfg["door_visible"])
    room0 = [(int(x), int(y)) for y, x in np.argwhere(w.room == 0) if w.switch_id[y, x] < 0
             and (int(x), int(y)) != (int(z["sz"][0]), int(z["sz"][1]))]
    out = []
    for cls in ("odd", "even2", "null"):
        k = 0
        while k < N_LOOPS[cls]:
            n1, n2 = PAIR_CLASSES[cls][int(rng.integers(len(PAIR_CLASSES[cls])))]
            c = room0[int(rng.integers(len(room0)))]
            p1 = ho.constructive_path(w, z["sz"], n1, rng, end_cell=c, max_len=MAX_LOOP, tries=30)
            p2 = ho.constructive_path(w, z["sz"], n2, rng, end_cell=c, max_len=MAX_LOOP, tries=30)
            if p1 is None or p2 is None:
                continue
            L = max(len(p1.actions), len(p2.actions))
            a1 = ho.pad_with_bumps(w, z["sz"], p1.actions, L, rng)
            a2 = ho.pad_with_bumps(w, z["sz"], p2.actions, L, rng)
            if a1 is None or a2 is None:
                continue
            e1, e2 = w.trajectory(z["sz"], a1)[-1], w.trajectory(z["sz"], a2)[-1]
            out.append({"cls": cls, "n": (n1, n2), "h1": a1, "h2": a2, "length": L,
                        "true_differs": int((e1[2:] != e2[2:]).any()),
                        "obs_closed": not (vis and (n1 - n2) % 2 == 1)})
            k += 1
    return out


def generic_pairs_for_z(w, z: dict, rng) -> list:
    out = []
    while len(out) < N_GENERIC_PAIRS:
        r = ho.generic_pair(w, z["sz"], rng, max_len=MAX_LOOP)
        if r is None:
            continue
        h1, h2 = r
        out.append({"h1": h1, "h2": h2, "length": len(h1),
                    "parity_diff": ho.cell_entry_parity(w, z["sz"], h1) ^ ho.cell_entry_parity(w, z["sz"], h2)})
    return out


def analysis_set(w, seed: int, n_z: int, full: bool = True) -> dict:
    """All ground-truth queries for one world (deterministic given the seed). full=False: loops only
    (calibration z's)."""
    zs = sample_zs(w, seed, n_z)
    rng = np.random.default_rng([seed, 1])
    for z in zs:
        z["loops"] = loops_for_z(w, z, rng)
        if full:
            z["plaq_R0"] = plaquettes_for_z(w, z, rng, 0)
            z["plaq_R1"] = plaquettes_for_z(w, z, rng, 1)
            z["det_pairs"] = detection_pairs_for_z(w, z, rng)
            z["gen_pairs"] = generic_pairs_for_z(w, z, rng)
    return {"seed": seed, "zs": zs}


EVAL_SEED = SPLIT_SEEDS["analysis"] + 600        # evaluation z's
CAL_SEED = SPLIT_SEEDS["analysis"] + 601         # calibration z's (P6.3 threshold)
GATE_SEED = SPLIT_SEEDS["analysis"] + 602        # scripted gate set
