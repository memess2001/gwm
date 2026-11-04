"""Toggle-door grid world (cells 6, 8, 12): hidden door bits, non-commuting actions, holonomy of switch loops.

Layout (n_doors = k in {1, 2, 3}; defaults room_w = room_h = 5), e.g. k = 1:

    #############    x = column (0 .. W-1), y = row (0 .. H-1, growing DOWNWARD); W = (k+1)(room_w+1)+1, H = room_h+2
    #.....#.....#    rooms R0..Rk in a row; R_j has interior columns j(room_w+1)+1 .. j(room_w+1)+room_w, rows 1..room_h;
    #.S...#.....#    1-cell walls between rooms and around.
    #.....D.....#    door D_i (i = 1..k): the wall cell (i(room_w+1), 1 + room_h//2) between R_{i-1} and R_i.
    #.....#.....#    switch S_i (default): cell (x0+1, 2) of R_{i-1}, x0 its first interior column (2nd column,
    #.....#.....#    2nd row): never 4-adjacent to D_i and a corner of four plaquettes. Override with ``switches``.
    #############

State    (x, y, d_1..d_k), d_i in {0 closed, 1 open}. Actions 0 up (y-1), 1 down (y+1), 2 left (x-1), 3 right (x+1).
Rules    a move into a wall or a closed door fails (the agent stays); an open door cell is walkable. ENTERING a
         switch cell (arriving from a different cell) toggles its door; a failed move that leaves the agent on a
         switch does not toggle. S_i lies in R_{i-1}, so the door an agent came through cannot be toggled from
         the far side: every state can always walk back to any position it came from.
Obs      "grid": float32 [C, H, W] = walls (door cells excluded), agent, switches, door cells (static marker, the
         same open or closed) [+ door_open if door_visible]. Hidden variant: the observation is a function of the
         agent position only, exactly identical for all door bits. "pos": float32 (x/(W-1), y/(H-1))
         [+ door bits if door_visible], for low-dim debugging. NOTE: switch and door cells are visibly MARKED in
         "grid" (static channels 2, 3), so "where is the hidden mechanism" can be read off the observation; a
         localisation claim (curvature map -> switches) must be compared with that trivial baseline.
Policies "random" = uniform random actions (passive logs); "loops" = concatenated random position-closed loops
         (random walk + shortest return under the true dynamics) = active probing.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

import numpy as np

from .base import Trajectories, World

DEFAULTS: dict[str, Any] = dict(n_doors=1, room_w=5, room_h=5, switches=None, door_visible=False,
                                init_doors="closed", start="room0", policy="random", loop_len=(2, 8),
                                switches_enabled=True, switch_visible=True)
# switches_enabled=False: switch cells never toggle (door bits are constant) -> the "no-switch" control of cell 6.
# switch_visible=False: the static switch-marker channel of "grid" is all zeros -> removes the trivial
#   "read the marker" baseline for localisation claims (cell 6, P6.2 / P6.6). Both default to the original world.
LAYOUT = ("n_doors", "room_w", "room_h", "switches", "switches_enabled")
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
MOVES = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
INVERSE = np.array([DOWN, UP, RIGHT, LEFT])
PLAQUETTE = np.array([RIGHT, DOWN, LEFT, UP])     # from the top-left corner, clockwise on screen
REVERSE = INVERSE[PLAQUETTE[::-1]]                # [DOWN, RIGHT, UP, LEFT]: same square, opposite orientation
INF = 1 << 20


class ToggleGrid(World):
    name = "toggle_grid"
    obs_modes = ("grid", "pos")

    def __init__(self, **params):
        self.cfg = self._check({**DEFAULTS, **params}, params)
        self._build()

    # ---------------------------------------------------------------- config / layout
    @staticmethod
    def _check(c, over) -> dict:
        bad = set(over) - set(DEFAULTS)
        if bad:
            raise TypeError(f"unknown toggle_grid parameters: {sorted(bad)}")
        if c["switches"] is not None:
            c["switches"] = tuple(tuple(int(v) for v in s) for s in c["switches"])
        if int(c["n_doors"]) not in (1, 2, 3) or int(c["room_w"]) < 3 or int(c["room_h"]) < 3:
            raise ValueError("need n_doors in {1, 2, 3} and room_w, room_h >= 3")
        if c["policy"] not in ("random", "loops"):
            raise ValueError("policy must be 'random' or 'loops'")
        lo, hi = c["loop_len"]
        if not 1 <= lo <= hi:
            raise ValueError("loop_len = (lo, hi) with 1 <= lo <= hi")
        return c

    def _with(self, **over) -> "ToggleGrid":
        """This world with overrides applied (a rebuilt world if the layout changes)."""
        c = self._check({**self.cfg, **over}, over)
        if any(c[k] != self.cfg[k] for k in LAYOUT):
            return ToggleGrid(**c)
        w = copy.copy(self)
        w.cfg = c
        return w

    def _build(self):
        c = self.cfg
        k, rw, rh = int(c["n_doors"]), int(c["room_w"]), int(c["room_h"])
        W, H = (k + 1) * (rw + 1) + 1, rh + 2
        free, room = np.zeros((H, W), bool), np.full((H, W), -1)
        for j in range(k + 1):
            x0 = j * (rw + 1) + 1
            free[1:rh + 1, x0:x0 + rw], room[1:rh + 1, x0:x0 + rw] = True, j
        doors = [(i * (rw + 1), 1 + rh // 2) for i in range(1, k + 1)]
        sw = list(c["switches"]) if c["switches"] is not None else [((i - 1) * (rw + 1) + 2, 2) for i in range(1, k + 1)]
        if len(sw) != k:
            raise ValueError("need one switch per door")
        for i, ((sx, sy), (dx, dy)) in enumerate(zip(sw, doors)):
            if not (0 <= sx < W and 0 <= sy < H) or room[sy, sx] != i or abs(sx - dx) + abs(sy - dy) <= 1:
                raise ValueError(f"switch S_{i + 1} must be an interior cell of R_{i} not adjacent to D_{i + 1}")
        self.k, self.W, self.H, self.free, self.room, self.doors, self.switches = k, W, H, free, room, doors, sw
        self.door_id, self.switch_id = np.full((H, W), -1), np.full((H, W), -1)
        for i, ((dx, dy), (sx, sy)) in enumerate(zip(doors, sw)):
            self.door_id[dy, dx], self.switch_id[sy, sx] = i, i
        self.switches_on = bool(c["switches_enabled"])
        self.walls = (~free & (self.door_id < 0)).astype(np.float32)
        # transition tables over every (x, y, bits) combination; state id = bits * H * W + y * W + x
        S = (1 << k) * H * W
        sid = np.arange(S)
        pos = sid % (H * W)
        allst = np.concatenate([(pos % W)[:, None], (pos // W)[:, None], ((sid // (H * W))[:, None] >> np.arange(k)) & 1], 1)
        self.NS = np.stack([self.encode(self.step(allst, np.full(S, a))) for a in range(4)], 1)       # [S, 4]
        npos = self.NS % (H * W)
        self.ENTER = np.where(npos != pos[:, None], self.switch_id.ravel()[npos], -1)                 # [S, 4]
        self.cells = np.flatnonzero((free | (self.door_id >= 0)).ravel())   # walkable position ids y*W + x
        self.cell_row = np.full(H * W, -1)
        self.cell_row[self.cells] = np.arange(len(self.cells))
        self.D = self._dist(None)                  # D[row(p), s] = fewest actions from s to stand on p
        self.D_avoid = self._dist(self.ENTER < 0)  # same, never entering a switch

    def _dist(self, allowed):
        S, P = self.NS.shape[0], len(self.cells)
        D = np.where(np.arange(S)[None, :] % (self.H * self.W) == self.cells[:, None], 0, INF).astype(np.int32)
        nxt = self.NS if allowed is None else np.where(allowed, self.NS, S)    # S = sentinel column (INF)
        while True:
            Dn = np.minimum(D, np.concatenate([D, np.full((P, 1), INF, np.int32)], 1)[:, nxt].min(-1) + 1)
            if np.array_equal(Dn, D):
                return D
            D = Dn

    def encode(self, state) -> np.ndarray:
        s = np.asarray(state, dtype=np.int64)
        bits = (s[..., 2:] << np.arange(self.k)).sum(-1)
        return (bits * self.H + s[..., 1]) * self.W + s[..., 0]

    def _row(self, state):
        s = np.asarray(state, dtype=np.int64)
        return self.cell_row[s[..., 1] * self.W + s[..., 0]]

    def ascii(self) -> str:
        g = np.where(self.free, ".", "#").astype("<U1")
        for (dx, dy), (sx, sy) in zip(self.doors, self.switches):
            g[dy, dx], g[sy, sx] = "D", "S"
        return "\n".join("".join(r) for r in g)

    # ---------------------------------------------------------------- pure dynamics
    def step(self, state, action) -> np.ndarray:
        """Vectorised true dynamics: states [..., 2+k], actions [...] -> next states [..., 2+k]."""
        s, a = np.asarray(state, dtype=np.int64), np.asarray(action, dtype=np.int64)
        x, y, bits = s[..., 0], s[..., 1], s[..., 2:]
        nx, ny = np.clip(x + MOVES[a, 0], 0, self.W - 1), np.clip(y + MOVES[a, 1], 0, self.H - 1)
        di = self.door_id[ny, nx]
        is_open = np.take_along_axis(bits, np.maximum(di, 0)[..., None], -1)[..., 0] == 1
        ok = self.free[ny, nx] | ((di >= 0) & is_open)
        nx, ny = np.where(ok, nx, x), np.where(ok, ny, y)
        si = np.where(ok & self.switches_on, self.switch_id[ny, nx], -1)   # only a successful move enters a switch
        nb = bits ^ (si[..., None] == np.arange(self.k))
        return np.concatenate([nx[..., None], ny[..., None], nb], -1)

    def trajectory(self, state, actions) -> np.ndarray:
        """States [..., L+1, 2+k] visited from state(s) [..., 2+k] under actions [..., L]."""
        a = np.asarray(actions, dtype=np.int64)
        s = np.broadcast_to(np.asarray(state, dtype=np.int64), a.shape[:-1] + (2 + self.k,))
        out = np.empty(a.shape[:-1] + (a.shape[-1] + 1, 2 + self.k), np.int64)
        out[..., 0, :] = s
        for t in range(a.shape[-1]):
            out[..., t + 1, :] = s = self.step(s, a[..., t])
        return out

    def rollout(self, state, actions, obs_mode: str = "grid"):
        """Resettable rollout for active diagnostics: (states [..., L+1, 2+k], obs [..., L+1, *obs_shape])."""
        st = self.trajectory(state, actions)
        return st, self.observe(st, obs_mode)

    # ---------------------------------------------------------------- loops
    def random_loop(self, start_state, rng=None, length: Optional[tuple] = None, switch: Optional[bool] = None):
        """Random position-closed loop (int64 actions) from ``start_state`` under the true dynamics.

        A random walk of L ~ U{lo..hi} actions (failed moves included) followed by a shortest return to the
        start position (shortest over the full state graph, so door changes on the way are accounted for).
        switch=None: unconstrained; True: the loop enters at least one switch (if the walk did not, it first
        walks to the nearest switch); False: no switch is ever entered (walk and return avoid switches).
        Its holonomy is ``holonomy(start_state, actions)``. Pass a seed or Generator as ``rng`` for reproducible
        loops (rng=None draws fresh OS entropy).
        """
        rng = np.random.default_rng(rng)
        lo, hi = self.cfg["loop_len"] if length is None else length
        s = np.asarray(start_state, dtype=np.int64).copy()
        home = self._row(s)
        if switch is False and self.switch_id[s[1], s[0]] >= 0:
            raise ValueError("a loop that starts on a switch must enter it to close")
        D, acts, entered = (self.D_avoid if switch is False else self.D), [], False

        def go(a):
            nonlocal s, entered
            entered |= bool(self.ENTER[self.encode(s), a] >= 0)
            s = self.step(s, a)
            acts.append(int(a))

        def toward(row):
            sid = self.encode(s)
            d = D[row, self.NS[sid]].astype(float)
            if switch is False:
                d[self.ENTER[sid] >= 0] = np.inf
            if not np.isfinite(d.min()) or d.min() >= INF:
                raise ValueError("target position unreachable from this state")
            return rng.choice(np.flatnonzero(d == d.min()))

        for _ in range(int(rng.integers(lo, hi + 1))):
            go(rng.choice(np.arange(4) if switch is not False else np.flatnonzero(self.ENTER[self.encode(s)] < 0)))
        if switch is True and not entered:
            if self.switch_id[s[1], s[0]] >= 0:      # standing on a switch: step off, then come back
                go(rng.choice(np.flatnonzero(self.NS[self.encode(s)] != self.encode(s))))
            rows = np.array([self.cell_row[y * self.W + x] for x, y in self.switches])
            target = rows[np.argmin(self.D[rows, self.encode(s)])]
            while not entered:
                go(toward(target))
        while self._row(s) != home:
            go(toward(home))
        return np.array(acts, dtype=np.int64)

    def plaquette_loops(self) -> dict:
        """All executable plaquette loops: top-left corners [P, 2] (x, y), loop R,D,L,U and its reverse D,R,U,L."""
        f = self.free
        ys, xs = np.nonzero(f[:-1, :-1] & f[:-1, 1:] & f[1:, :-1] & f[1:, 1:])
        return {"corners": np.stack([xs, ys], 1), "loop": PLAQUETTE.copy(), "reverse": REVERSE.copy()}

    def curvature_map(self) -> dict:
        """Z2 holonomy (door-flip parity vector) of every executable plaquette loop.

        A plaquette loop enters each of its four corner cells exactly once (the start corner on the last move)
        and never touches a door cell (door cells sit in wall columns between two wall cells), so all four
        moves succeed whatever the door bits, and its flip parity for door i is 1 iff S_i is one of the corners:
        independent of the start corner, the orientation and the door bits. ``map`` is [H-1, W-1, k] indexed by
        the top-left corner (y, x), -1 where the plaquette is not executable.
        """
        pl = self.plaquette_loops()
        P = pl["corners"]
        par = np.stack([((sx - P[:, 0]) // 2 == 0) & ((sy - P[:, 1]) // 2 == 0) for sx, sy in self.switches], 1)
        par = par & self.switches_on                       # disabled switches never toggle: zero holonomy
        mp = np.full((self.H - 1, self.W - 1, self.k), -1, np.int64)
        mp[P[:, 1], P[:, 0]] = par
        return {**pl, "parity": par.astype(np.int64), "map": mp}

    def holonomy(self, start_state, actions) -> dict:
        """{position_closed, obs_closed, flips}: flips = door bits(end) XOR door bits(start) (parity vector).

        obs_closed: the end observation equals the start observation (hidden doors: same position; visible
        doors: same position and same door bits). Broadcasts over leading dims of start_state / actions.
        """
        st = self.trajectory(start_state, actions)
        s0, s1 = st[..., 0, :], st[..., -1, :]
        pos = (s0[..., :2] == s1[..., :2]).all(-1)
        flips = s0[..., 2:] ^ s1[..., 2:]
        return {"position_closed": pos, "obs_closed": pos & ~flips.any(-1) if self.cfg["door_visible"] else pos,
                "flips": flips}

    def _commutators(self) -> list:
        z, ex = [0] * self.k, []
        cases = [((sx - 1, sy), [RIGHT, DOWN], [DOWN, RIGHT]) for sx, sy in self.switches]
        cases += [((sx, sy - 1), [DOWN, RIGHT], [RIGHT, DOWN]) for sx, sy in self.switches]
        cases += [((self.W - 3, self.H - 3), [RIGHT, DOWN], [DOWN, RIGHT])]          # control: far from switches
        for start, a, b in cases:
            if not (0 <= start[0] < self.W and 0 <= start[1] < self.H and self.free[start[1], start[0]]):
                continue
            s0 = np.array([*start, *z])
            ta, tb = self.trajectory(s0, a), self.trajectory(s0, b)
            if any((np.diff(t[:, :2], axis=0) == 0).all(-1).any() for t in (ta, tb)):
                continue                                   # a move would fail: not a clean example
            ea, eb = ta[-1], tb[-1]
            ex.append(dict(start=s0, seq_a=a, seq_b=b, end_a=ea, end_b=eb,
                           same_position=bool((ea[:2] == eb[:2]).all()), hidden_differs=bool((ea[2:] != eb[2:]).any())))
        return ex

    # ---------------------------------------------------------------- sampling / observation
    def _initial(self, rng, n, c):
        st = c["start"]
        if isinstance(st, str):
            if st not in ("room0", "any"):
                raise ValueError("start must be 'room0', 'any' or a cell (x, y)")
            yx = np.argwhere(self.room == 0) if st == "room0" else np.argwhere(self.free)
            yx = yx[rng.integers(0, len(yx), n)]
            xy = yx[:, ::-1]
        else:
            if not self.free[st[1], st[0]]:
                raise ValueError("start cell must be a free interior cell")
            xy = np.tile(np.asarray(st, dtype=np.int64), (n, 1))
        d = c["init_doors"]
        if isinstance(d, str):
            if d not in ("closed", "random"):
                raise ValueError("init_doors must be 'closed', 'random' or bits [k]")
            bits = np.zeros((n, self.k), np.int64) if d == "closed" else rng.integers(0, 2, (n, self.k))
        else:
            bits = np.broadcast_to(np.asarray(d, dtype=np.int64), (n, self.k))
        return np.concatenate([xy, bits], 1).astype(np.int64)

    def _loops(self, rng, s0, horizon, c):
        n, (lo, hi) = len(s0), c["loop_len"]
        st = np.empty((n, horizon + 1, 2 + self.k), np.int64)
        acts, loop_id = np.empty((n, horizon), np.int64), np.zeros((n, horizon), np.int64)
        closed = np.zeros((n, horizon + 1), bool)
        st[:, 0], closed[:, 0], s = s0, True, s0
        home, rem, lid = self._row(s0), rng.integers(lo, hi + 1, n), np.zeros(n, np.int64)
        for t in range(horizon):
            greedy = np.argmin(self.D[home[:, None], self.NS[self.encode(s)]] + 0.5 * rng.random((n, 4)), 1)
            a = np.where(rem > 0, rng.integers(0, 4, n), greedy)       # random walk, then shortest return
            s = self.step(s, a)
            st[:, t + 1], acts[:, t], loop_id[:, t] = s, a, lid
            rem = np.maximum(rem - 1, 0)
            done = (rem == 0) & (self._row(s) == home)
            closed[:, t + 1] = done
            lid = lid + done
            rem = np.where(done, rng.integers(lo, hi + 1, n), rem)
        return st, acts, {"loop_id": loop_id, "loop_closed": closed}

    def sample(self, seed: int, horizon: int, n_traj: int, obs_mode: str, **kwargs) -> Trajectories:
        w = self._with(**kwargs)
        c, rng = w.cfg, np.random.default_rng(seed)
        s0 = w._initial(rng, n_traj, c)
        if c["policy"] == "random":
            acts = rng.integers(0, 4, (n_traj, horizon))
            st, extra = w.trajectory(s0, acts), {}
        else:
            st, acts, extra = w._loops(rng, s0, horizon, c)
        moved = (st[:, 1:, :2] != st[:, :-1, :2]).any(-1)
        info = {"switch_entered": np.where(moved, w.switch_id[st[:, 1:, 1], st[:, 1:, 0]], -1),
                "blocked": ~moved, "room": w.room[st[..., 1], st[..., 0]], **extra}
        meta = dict(world=self.name, seed=seed, horizon=horizon, n_traj=n_traj, obs_mode=obs_mode, params=dict(c))
        return Trajectories(obs=w.observe(st, obs_mode), state=st, actions=acts,
                            t=np.arange(horizon + 1, dtype=np.float64), info=info, meta=meta)

    def observe(self, state, obs_mode, rng=None) -> np.ndarray:
        s = np.asarray(state, dtype=np.int64)
        vis = bool(self.cfg["door_visible"])
        if obs_mode == "pos":
            xy = s[..., :2] / np.array([self.W - 1, self.H - 1], dtype=np.float64)
            return (np.concatenate([xy, s[..., 2:]], -1) if vis else xy).astype(np.float32)
        if obs_mode != "grid":
            raise ValueError(f"unknown obs_mode {obs_mode!r}; choose from {self.obs_modes}")
        flat = s.reshape(-1, 2 + self.k)
        out = np.zeros((len(flat), 5 if vis else 4, self.H, self.W), np.float32)
        out[:, 0] = self.walls
        out[:, 2] = (self.switch_id >= 0) & bool(self.cfg["switch_visible"])
        out[:, 3] = self.door_id >= 0
        out[np.arange(len(flat)), 1, flat[:, 1], flat[:, 0]] = 1.0
        if vis:
            for i, (dx, dy) in enumerate(self.doors):
                out[:, 4, dy, dx] = flat[:, 2 + i]
        return out.reshape(s.shape[:-1] + out.shape[1:])

    def spec(self, obs_mode: str, **over) -> dict:
        w = self._with(**over)
        vis = bool(w.cfg["door_visible"])
        if obs_mode == "grid":
            shape = (5 if vis else 4, w.H, w.W)
        elif obs_mode == "pos":
            shape = (2 + w.k,) if vis else (2,)
        else:
            raise ValueError(f"unknown obs_mode {obs_mode!r}")
        return {"obs_shape": shape, "obs_dtype": "float32", "action": {"type": "discrete", "n": 4}}

    # ---------------------------------------------------------------- ground truth
    def ground_truth(self, **cfg) -> dict[str, Any]:
        w = self._with(**cfg)
        vis = bool(w.cfg["door_visible"])
        on = bool(w.cfg["switches_enabled"])
        return dict(
            state_space=f"agent cell x {{0,1}}^{w.k}", state_dim=2 + w.k, n_doors=w.k, door_visible=vis,
            switches_enabled=on, switch_visible=bool(w.cfg["switch_visible"]),
            # with switches disabled the door bits are constant: nothing is hidden that ever changes
            hidden_state_bits=0 if (vis or not on) else w.k,
            # an agent that started in R0 and stands in R_j knows D_1..D_j are open (they cannot be toggled from
            # beyond): only k - j bits are uncertain there (hidden variant; start="room0")
            hidden_bits_by_room=[0 if (vis or not on) else w.k - j for j in range(w.k + 1)],
            switch_cells=list(w.switches), door_cells=list(w.doors), bottleneck_cells=list(w.doors),
            holonomy=w.holonomy, curvature_map=w.curvature_map(), commutator_examples=w._commutators(),
            actions=("up", "down", "left", "right"), layout=w.ascii(),
            rule="entering S_i (arriving from another cell) toggles d_i; walls / closed doors block; flips = parity of switch entries",
        )
