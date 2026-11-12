"""Cell 6 instruments: observation-level holonomy, plaquette curvature map, and the non-geometric baseline.

Everything here reads a world model only through a ``Predictor`` (start latent from the episode's first frame,
one latent step per action, decoded agent position), so the same code runs on trained world models and on
ground-truth oracles. The oracles are how the instruments are checked (rules.R4: ground truth only):

    TruePredictor        latent = true state (x, y, bits): exact holonomy
    MemorylessPredictor  latent = position only, bits frozen at the start value: zero holonomy everywhere
    AnyEntryPredictor    bit = "the switch was ever entered": the any-entry loophole of the v0 design
    AbelianPredictor     z' = z + delta_a with delta_L = -delta_R, delta_D = -delta_U (cells/6 P6.5b unit test)

Definitions (cells/6/prereg.md v1):
  * every latent comes from rolling out the start latent E(o_0) along the episode's actions (no E(o_t), t > 0);
  * H_obs(branch A, branch B) = max over the probe set Pi of the fraction of the 20 probe frames whose argmax-
    decoded agent positions differ (visible-door world: position or decoded door bit differs). The two branches
    always have the same number of steps (time-matched reference);
  * probe set Pi = 64 seeded uniform random 20-step action sequences (``probe_set()``), position independent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
MOVES = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
INVERSE = np.array([DOWN, UP, RIGHT, LEFT])
PLAQ_CW = np.array([RIGHT, DOWN, LEFT, UP])        # clockwise from the top-left corner
PROBE_SEED, N_PROBES, PROBE_LEN = 60_064, 64, 20


# ============================================================================ predictors
class Predictor:
    """Interface: ``start(states0)`` -> latents, ``step(Z, a)`` -> latents, ``decode(Z)`` -> int [B, 2] (+ bit)."""
    door_visible: bool = False

    def start(self, states0: np.ndarray):
        raise NotImplementedError

    def step(self, Z, a: np.ndarray):
        raise NotImplementedError

    def decode(self, Z) -> np.ndarray:
        """[B, 2] int positions (x, y), or [B, 3] (x, y, door bit) when ``door_visible``."""
        raise NotImplementedError

    # generic helpers ---------------------------------------------------------------------------
    def take(self, Z, idx):
        return Z[idx]

    def where(self, mask: np.ndarray, Znew, Zold):
        m = mask.reshape((-1,) + (1,) * (np.ndim(Zold) - 1))
        return np.where(m, Znew, Zold)

    def run(self, Z, actions: np.ndarray, lengths: Optional[np.ndarray] = None):
        """Apply action rows [B, L] (row i only up to lengths[i]) and return the final latents."""
        actions = np.asarray(actions, dtype=np.int64)
        B, L = actions.shape
        lengths = np.full(B, L) if lengths is None else np.asarray(lengths)
        for t in range(L):
            m = t < lengths
            if not m.any():
                break
            Zn = self.step(Z, np.where(m, actions[:, t], 0))
            Z = self.where(m, Zn, Z)
        return Z


class TruePredictor(Predictor):
    def __init__(self, world):
        self.w = world
        self.door_visible = bool(world.cfg["door_visible"])

    def start(self, states0):
        return np.asarray(states0, dtype=np.int64).copy()

    def step(self, Z, a):
        return self.w.step(Z, a)

    def decode(self, Z):
        return Z[:, :3].copy() if self.door_visible else Z[:, :2].copy()


class MemorylessPredictor(TruePredictor):
    """Tracks the position under the TRUE dynamics but never lets the door bit change: no memory of entries."""

    def step(self, Z, a):
        nz = self.w.step(Z, a)
        nz[:, 2:] = Z[:, 2:]
        return nz


class AnyEntryPredictor(TruePredictor):
    """Believes 'door open iff the switch was ever entered' (the v0 loophole); dynamics otherwise exact."""

    def start(self, states0):
        s = np.asarray(states0, dtype=np.int64)                       # columns: x, y, believed bit, b0, flag
        return np.concatenate([s, s[:, 2:3], np.zeros((len(s), 1), np.int64)], 1)

    def step(self, Z, a):
        s, b0, flag = Z[:, :3], Z[:, 3], Z[:, 4]
        nz = self.w.step(s, a)                                        # moves under the BELIEVED door bit
        moved = (nz[:, :2] != s[:, :2]).any(1)
        entered = moved & (self.w.switch_id[nz[:, 1], nz[:, 0]] >= 0)
        flag = flag | entered.astype(np.int64)
        nz[:, 2] = np.where(flag == 1, 1 - b0, b0)                    # 'toggled once' as soon as ever entered
        return np.concatenate([nz, b0[:, None], flag[:, None]], 1)


class AbelianPredictor(Predictor):
    """Untrained Abelian latent model z' = z + delta_a with antisymmetric increments and a random decoder."""

    def __init__(self, world, d: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w, self.d = world, d
        dr, du = rng.normal(size=d), rng.normal(size=d)
        self.delta = np.stack([du, -du, -dr, dr])          # UP, DOWN, LEFT, RIGHT: delta_L = -delta_R etc.
        self.W = rng.normal(size=(d, world.H * world.W))
        self.b = rng.normal(size=world.H * world.W)

    def start(self, states0):
        s = np.asarray(states0, dtype=np.float64)
        return np.concatenate([s[:, :2], np.zeros((len(s), self.d - 2))], 1)

    def step(self, Z, a):
        return Z + self.delta[np.asarray(a)]

    def decode(self, Z):
        idx = np.argmax(Z @ self.W + self.b, 1)
        return np.stack([idx % self.w.W, idx // self.w.W], 1)


class WMPredictor(Predictor):
    """A trained baselines.WorldModel (history 1, discrete actions, 'grid' observations)."""

    def __init__(self, model, normalizer, world, device="cpu", batch: int = 16384):
        import torch
        self.torch, self.model, self.norm, self.w, self.device, self.batch = torch, model, normalizer, world, device, batch
        self.door_visible = bool(world.cfg["door_visible"])
        if model.cfg.history != 1:
            raise ValueError("cell 6 models use history = 1")
        self.A = model.cfg.action_dim

    def _t(self, x):
        return self.torch.as_tensor(x, device=self.device)

    def start(self, states0):
        from baselines.data import frames_to_tensor
        obs = self.w.observe(np.asarray(states0, dtype=np.int64), "grid")
        with self.torch.no_grad():
            x = frames_to_tensor(obs[:, None], self.device, self.norm)          # [B, 1, C, H, W]
            return self.model.encode(x).float()

    def step(self, Z, a):
        oh = self.torch.nn.functional.one_hot(self._t(np.asarray(a, dtype=np.int64)), self.A).float()
        with self.torch.no_grad():
            out = [self.model.step(Z[i:i + self.batch], oh[i:i + self.batch]) for i in range(0, len(Z), self.batch)]
        return self.torch.cat(out, 0)

    def take(self, Z, idx):
        return Z[self._t(np.asarray(idx, dtype=np.int64))]

    def where(self, mask, Znew, Zold):
        return self.torch.where(self._t(np.asarray(mask))[:, None], Znew, Zold)

    def decode(self, Z):
        H, W = self.w.H, self.w.W
        outs = []
        with self.torch.no_grad():
            for i in range(0, len(Z), self.batch):
                f = self.model.decode(Z[i:i + self.batch])                        # [b, C, H, W] (model units)
                if self.norm is not None:
                    f = f * self._t(self.norm.std) + self._t(self.norm.mean)
                idx = f[:, 1].reshape(len(f), -1).argmax(1).cpu().numpy()
                pos = np.stack([idx % W, idx // W], 1)
                if self.door_visible:
                    dx, dy = self.w.doors[0]
                    bit = (f[:, 4, dy, dx] > 0.5).cpu().numpy().astype(np.int64)[:, None]
                    pos = np.concatenate([pos, bit], 1)
                outs.append(pos)
        return np.concatenate(outs, 0)


# ============================================================================ probes and readouts
def probe_set(n: int = N_PROBES, length: int = PROBE_LEN, seed: int = PROBE_SEED) -> np.ndarray:
    """The pre-registered probe set Pi: ``n`` uniform random action sequences of ``length`` steps."""
    return np.random.default_rng(seed).integers(0, 4, (n, length)).astype(np.int64)


def probe_positions(pred: Predictor, Z, probes: np.ndarray) -> np.ndarray:
    """Decoded positions along every probe from every latent: int [B, P, L, c]."""
    B, (P, L) = len(Z), probes.shape
    rep = pred.take(Z, np.repeat(np.arange(B), P))
    acts = np.tile(probes, (B, 1))
    out = []
    for t in range(L):
        rep = pred.step(rep, acts[:, t])
        out.append(pred.decode(rep))
    return np.stack(out, 1).reshape(B, P, L, -1)


def h_obs(pred: Predictor, ZA, ZB, probes: np.ndarray, chunk: int = 256) -> dict:
    """{'max', 'mean'}: per-pair holonomy scores from two time-matched branch latents (same number of steps)."""
    n = len(ZA)
    hmax, hmean = np.empty(n), np.empty(n)
    for i in range(0, n, chunk):
        idx = np.arange(i, min(n, i + chunk))
        pa = probe_positions(pred, pred.take(ZA, idx), probes)
        pb = probe_positions(pred, pred.take(ZB, idx), probes)
        diff = (pa != pb).any(-1).mean(-1)                        # [b, P] fraction of differing frames
        hmax[idx], hmean[idx] = diff.max(1), diff.mean(1)
    return {"max": hmax, "mean": hmean}


# ============================================================================ ground-truth path construction
def _entered(w, s, a) -> bool:
    return bool(w.ENTER[w.encode(s), a] >= 0)


def _bump_actions(w, s) -> list:
    """Actions that walk into a true wall cell from state s (the agent stays; doors excluded)."""
    x, y = int(s[0]), int(s[1])
    out = []
    for a in range(4):
        nx, ny = x + MOVES[a, 0], y + MOVES[a, 1]
        if 0 <= nx < w.W and 0 <= ny < w.H and w.walls[ny, nx] > 0:
            out.append(a)
    return out


@dataclass
class Path:
    actions: np.ndarray        # int64 [L]
    entries: int               # switch entries along the path (true dynamics)
    end_state: np.ndarray      # true end state


def simulate(w, s0, actions) -> tuple[np.ndarray, int]:
    st = w.trajectory(np.asarray(s0, dtype=np.int64), np.asarray(actions, dtype=np.int64))
    moved = (st[1:, :2] != st[:-1, :2]).any(-1)
    ent = int((moved & (w.switch_id[st[1:, 1], st[1:, 0]] >= 0)).sum())
    return st, ent


def constructive_path(w, s0, n_entries: int, rng, end_cell: Optional[tuple] = None, walk=(2, 8),
                      gap=(0, 2), max_len: int = 28, tries: int = 200) -> Optional[Path]:
    """A path from s0 entering the (single) switch exactly ``n_entries`` times and ending on ``end_cell``
    (default: the start cell, i.e. a position-closed loop). Random walk (switch-avoiding) -> n x [walk to the
    switch, enter, step off, short switch-avoiding walk] -> switch-avoiding shortest path to the end cell.
    Returns None if no path of length <= max_len was found in ``tries`` attempts."""
    s0 = np.asarray(s0, dtype=np.int64)
    if w.switch_id[s0[1], s0[0]] >= 0:
        raise ValueError("start on the switch cell is not used")
    end = (int(s0[0]), int(s0[1])) if end_cell is None else tuple(int(v) for v in end_cell)
    if w.switch_id[end[1], end[0]] >= 0:
        raise ValueError("end cell must not be the switch")
    erow = w.cell_row[end[1] * w.W + end[0]]
    srow = w.cell_row[w.switches[0][1] * w.W + w.switches[0][0]]
    for _ in range(tries):
        s, acts, ok = s0.copy(), [], True

        def go(a):
            nonlocal s
            acts.append(int(a))
            s = w.step(s, a)

        def avoid_walk(L):
            for _ in range(L):
                opts = [a for a in range(4) if not _entered(w, s, a)]
                go(rng.choice(opts))

        avoid_walk(int(rng.integers(walk[0], walk[1] + 1)))
        for _i in range(n_entries):
            guard = 0
            while not (w.switch_id[s[1], s[0]] >= 0):           # walk to the switch along D (enters on arrival)
                sid = w.encode(s)
                d = w.D[srow, w.NS[sid]].astype(float)
                go(rng.choice(np.flatnonzero(d == d.min())))
                guard += 1
                if guard > 60:
                    ok = False
                    break
            if not ok:
                break
            offs = [a for a in range(4) if (w.NS[w.encode(s), a] % (w.H * w.W)) != (w.encode(s) % (w.H * w.W))]
            go(rng.choice(offs))                                   # step off (not an entry)
            avoid_walk(int(rng.integers(gap[0], gap[1] + 1)))
        if not ok:
            continue
        guard = 0
        while (int(s[0]), int(s[1])) != end:                       # switch-avoiding shortest path to the end
            sid = w.encode(s)
            d = w.D_avoid[erow, w.NS[sid]].astype(float)
            d[w.ENTER[sid] >= 0] = np.inf
            if not np.isfinite(d.min()) or d.min() >= (1 << 20):
                ok = False
                break
            go(rng.choice(np.flatnonzero(d == d.min())))
            guard += 1
            if guard > 60:
                ok = False
                break
        if not ok or len(acts) > max_len:
            continue
        a = np.array(acts, dtype=np.int64)
        st, ent = simulate(w, s0, a)
        if ent == n_entries and (int(st[-1, 0]), int(st[-1, 1])) == end:
            return Path(a, ent, st[-1])
    return None


def pad_with_bumps(w, s0, actions: np.ndarray, target_len: int, rng) -> Optional[np.ndarray]:
    """Insert wall-bump actions (state unchanged) until the sequence has ``target_len`` steps."""
    acts = list(np.asarray(actions, dtype=np.int64))
    while len(acts) < target_len:
        st = w.trajectory(np.asarray(s0, dtype=np.int64), np.array(acts, dtype=np.int64))
        cand = [(t, b) for t in range(len(st)) for b in _bump_actions(w, st[t])]
        if not cand:
            return None
        t, b = cand[int(rng.integers(len(cand)))]
        acts.insert(t, b)
    return np.array(acts, dtype=np.int64)


def null_reference(w, s0, target_len: int, rng, tries: int = 200) -> Optional[np.ndarray]:
    """A switch-free position-closed loop of exactly ``target_len`` steps (null loop + wall-bump padding)."""
    for _ in range(tries):
        # several switch-free candidates; keep the longest one that fits, so that little padding is needed
        cands = [constructive_path(w, s0, 0, rng, walk=(1, max(1, target_len - 1)), max_len=target_len, tries=5)
                 for _ in range(12)]
        cands = [c for c in cands if c is not None]
        if not cands:
            continue
        top = max(len(c.actions) for c in cands)
        best = [c for c in cands if len(c.actions) == top]
        p = best[int(rng.integers(len(best)))]
        a = pad_with_bumps(w, s0, p.actions, target_len, rng)
        if a is None:
            continue
        st, ent = simulate(w, s0, a)
        if ent == 0 and (st[-1, :2] == np.asarray(s0)[:2]).all() and (st[-1, 2:] == np.asarray(s0)[2:]).all():
            return a
    return None


def plaquette_corners(w, room: int = 0) -> np.ndarray:
    """Top-left corners (x, y) of the executable plaquettes inside ``room`` (row-major order)."""
    pl = w.plaquette_loops()["corners"]
    keep = [c for c in pl if w.room[c[1], c[0]] == room and w.room[c[1] + 1, c[0] + 1] == room]
    return np.array(sorted(keep, key=lambda c: (c[1], c[0])), dtype=np.int64)


def plaquette_loop_from(corner_tl, start_corner) -> np.ndarray:
    """The clockwise plaquette loop starting at one of the four corners of the plaquette with top-left corner_tl."""
    x, y = corner_tl
    order = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]                 # TL, TR, BR, BL (clockwise)
    k = order.index(tuple(int(v) for v in start_corner))
    return np.roll(PLAQ_CW, -k).astype(np.int64)


def switch_plaquette_mask(w, corners: np.ndarray) -> np.ndarray:
    sx, sy = w.switches[0]
    return ((corners[:, 0] == sx) | (corners[:, 0] == sx - 1)) & ((corners[:, 1] == sy) | (corners[:, 1] == sy - 1))


def door_adjacent_plaquette_mask(w, corners: np.ndarray) -> np.ndarray:
    """Plaquettes containing a cell 4-adjacent to the door (named failure mode: 'consequence, not source')."""
    dx, dy = w.doors[0]
    cells = [(dx - 1, dy), (dx + 1, dy)]
    m = np.zeros(len(corners), bool)
    for cx, cy in cells:
        m |= ((corners[:, 0] == cx) | (corners[:, 0] == cx - 1)) & ((corners[:, 1] == cy) | (corners[:, 1] == cy - 1))
    return m


def plaquette_probe(w, s0, corner_tl, rng, avoid_switch_transport: bool = True, max_transport: int = 12):
    """(gamma, reference) action sequences of equal length for one plaquette from state s0:
    gamma = transport to a non-switch corner + clockwise plaquette loop; reference = the same transport +
    an out-and-back (a, a^-1, a, a^-1) to a free non-switch, non-door neighbour. None if not constructible."""
    x, y = int(corner_tl[0]), int(corner_tl[1])
    corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
    cand = [c for c in corners if w.switch_id[c[1], c[0]] < 0]
    D = w.D_avoid if avoid_switch_transport else w.D
    sid0 = w.encode(np.asarray(s0, dtype=np.int64))
    dist = [D[w.cell_row[c[1] * w.W + c[0]], sid0] for c in cand]
    best = [c for c, d in zip(cand, dist) if d == min(dist)]
    c = best[int(rng.integers(len(best)))]
    crow = w.cell_row[c[1] * w.W + c[0]]
    s, tau = np.asarray(s0, dtype=np.int64).copy(), []
    while (int(s[0]), int(s[1])) != c:
        sid = w.encode(s)
        d = D[crow, w.NS[sid]].astype(float)
        if avoid_switch_transport:
            d[w.ENTER[sid] >= 0] = np.inf
        if not np.isfinite(d.min()) or d.min() >= (1 << 20) or len(tau) > max_transport:
            return None
        a = int(rng.choice(np.flatnonzero(d == d.min())))
        tau.append(a)
        s = w.step(s, a)
    loop = plaquette_loop_from((x, y), c)
    outs = []
    for a in range(4):
        nx, ny = c[0] + MOVES[a, 0], c[1] + MOVES[a, 1]
        if w.free[ny, nx] and w.switch_id[ny, nx] < 0 and w.door_id[ny, nx] < 0:
            outs.append(a)
    if not outs:
        return None
    a = outs[int(rng.integers(len(outs)))]
    ref = np.array([a, INVERSE[a], a, INVERSE[a]], dtype=np.int64)
    tau = np.array(tau, dtype=np.int64)
    return np.concatenate([tau, loop]), np.concatenate([tau, ref])


def random_walk(w, s0, L: int, rng) -> np.ndarray:
    return rng.integers(0, 4, L).astype(np.int64)


def generic_pair(w, s0, rng, walk=(4, 12), max_len: int = 28, tries: int = 100):
    """Label-free history pair for the non-geometric localiser: h1 = uniform random walk; h2 = an independent
    uniform random walk followed by a shortest path (full dynamics) to h1's end cell; the shorter one padded
    with wall bumps. Returns (h1, h2) of equal length or None."""
    s0 = np.asarray(s0, dtype=np.int64)
    for _ in range(tries):
        h1 = random_walk(w, s0, int(rng.integers(walk[0], walk[1] + 1)), rng)
        e1 = w.trajectory(s0, h1)[-1]
        h2 = list(random_walk(w, s0, int(rng.integers(walk[0], walk[1] + 1)), rng))
        s = w.trajectory(s0, np.array(h2, dtype=np.int64))[-1]
        erow = w.cell_row[e1[1] * w.W + e1[0]]
        ok = True
        while (s[:2] != e1[:2]).any():
            sid = w.encode(s)
            d = w.D[erow, w.NS[sid]].astype(float)
            if d.min() >= (1 << 20) or len(h2) > max_len:
                ok = False
                break
            a = int(rng.choice(np.flatnonzero(d == d.min())))
            h2.append(a)
            s = w.step(s, a)
        if not ok:
            continue
        h2 = np.array(h2, dtype=np.int64)
        L = max(len(h1), len(h2))
        if L > max_len:
            continue
        p1 = pad_with_bumps(w, s0, h1, L, rng)
        p2 = pad_with_bumps(w, s0, h2, L, rng)
        if p1 is None or p2 is None:
            continue
        return p1, p2
    return None


def cell_entry_parity(w, s0, actions) -> np.ndarray:
    """Per walkable cell (w.cells order): parity of the number of entries (arrivals from another cell)."""
    st = w.trajectory(np.asarray(s0, dtype=np.int64), np.asarray(actions, dtype=np.int64))
    moved = (st[1:, :2] != st[:-1, :2]).any(-1)
    par = np.zeros(len(w.cells), np.int64)
    rows = w.cell_row[st[1:, 1] * w.W + st[1:, 0]]
    for r in rows[moved]:
        par[r] ^= 1
    return par


# ============================================================================ statistics
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted 1/2; nan if a class is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


LENGTH_BINS = (4, 11, 17, 23, 29)      # strata [4,10], [11,16], [17,22], [23,28]


def stratified_auc(score, label, length, bins: Sequence[int] = LENGTH_BINS) -> float:
    """Length-stratified AUC: within-stratum AUCs combined with weights n_pos * n_neg (strata lacking a class skipped)."""
    score, label, length = map(np.asarray, (score, label, length))
    num = den = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (length >= lo) & (length < hi)
        p, n = score[m & (label == 1)], score[m & (label == 0)]
        if len(p) and len(n):
            wgt = len(p) * len(n)
            num += wgt * auc(p, n)
            den += wgt
    return float(num / den) if den else float("nan")


def cluster_bootstrap(stat: Callable[[np.ndarray], float], clusters: np.ndarray, n_boot: int = 2000,
                      seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of ``stat(index array)`` resampling whole clusters (z's) with replacement."""
    clusters = np.asarray(clusters)
    ids = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in ids}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        out[b] = stat(np.concatenate([members[c] for c in pick]))
    return out


def precision_at_k(scores: np.ndarray, truth: np.ndarray, k: int = 4, seed: int = 0) -> float:
    """Fraction of the top-k scored items that are true; ties broken at random with a fixed seed."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(scores)), -np.asarray(scores, float)))
    return float(np.asarray(truth, bool)[order[:k]].mean())


def ranks_desc(scores: np.ndarray, seed: int = 0) -> np.ndarray:
    """1-based ranks (1 = highest score), ties broken at random with a fixed seed."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(scores)), -np.asarray(scores, float)))
    r = np.empty(len(scores), np.int64)
    r[order] = np.arange(1, len(scores) + 1)
    return r
