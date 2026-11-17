"""Cell 24 pre-prereg check (exact transition tables only, no learning): what does the topology of the k-step
reachable set add to BFS distance?

Identity being checked: for any deterministic transition table T (the true world, or a learned model queried on
every (state, action) the ball touches), R_{<=k}(s) = {s' : d_T(s, s') <= k}, so the Betti curve beta_i(k) is the
sublevel-set homology of the BFS distance field d_T(s, .) and every topology-change horizon k* is a function of d_T.

Complex: the reached cells as a union of CLOSED unit squares in the plane (corner contact connects), so
b0 = 8-connected components and b1 = b0 - chi with chi = V - E + F (b2 = 0 in the plane).
Variants on toggle_grid (hidden doors, start in room 0, doors closed, policy-free BFS over the full state (x, y, bits)):
  V1 positions, <= k, door cells included          (the natural reading)
  V2 positions, exactly k actions                   (no "<=": parity of the bipartite grid)
  V3 full state, <= k, one plane per door-bit pattern (disjoint union of layers)
  V4 positions, <= k, door cells left out of the complex
Plus a loop maze (two rooms joined by the door AND a second opening, so the free space has b1 = 1), and a corrupted
"model" (random 10 % of (state, action) entries redirected) to show k* follows the MODEL's BFS, right or wrong.
"""
import sys
import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

sys.path.insert(0, "/home/user/gwm_repo")
from zoo.toggle_grid import ToggleGrid

KMAX = 80
S8 = np.ones((3, 3), int)


def betti(mask):
    """(b0, b1) of the union of closed unit squares at the True cells of a 2-D mask."""
    M = np.asarray(mask, bool)
    if not M.any():
        return (0, 0)
    P = np.pad(M, 1)
    F = int(M.sum())
    V = int((P[:-1, :-1] | P[:-1, 1:] | P[1:, :-1] | P[1:, 1:]).sum())    # corners touched
    Eh = int((P[:-1, 1:-1] | P[1:, 1:-1]).sum())                          # horizontal unit edges
    Ev = int((P[1:-1, :-1] | P[1:-1, 1:]).sum())                          # vertical unit edges
    chi = V - (Eh + Ev) + F
    b0 = ndimage.label(M, structure=S8)[1]
    return (b0, b0 - chi)


def bfs(NS, s0):
    d = np.full(NS.shape[0], -1, np.int64); d[s0] = 0; fr = np.array([s0]); k = 0
    while len(fr):
        k += 1
        nx = np.unique(NS[fr].ravel()); nx = nx[d[nx] < 0]; d[nx] = k; fr = nx
    return d


def exact_k_sets(NS, s0, K):
    out, cur = [np.array([s0])], np.array([s0])
    for _ in range(K):
        cur = np.unique(NS[cur].ravel()); out.append(cur)
    return out


def curves(w, NS, s0, door_mask):
    """Betti curves for V1-V4 under transition table NS from state id s0."""
    HW = w.H * w.W
    d = bfs(NS, s0)
    ex = exact_k_sets(NS, s0, KMAX)
    cv = {v: [] for v in ("V1", "V2", "V3", "V4")}
    for k in range(KMAX + 1):
        ball = np.flatnonzero((d >= 0) & (d <= k))
        pos = np.zeros(HW, bool); pos[ball % HW] = True; pos = pos.reshape(w.H, w.W)
        cv["V1"].append(betti(pos))
        pe = np.zeros(HW, bool); pe[ex[k] % HW] = True
        cv["V2"].append(betti(pe.reshape(w.H, w.W)))
        b = (0, 0)
        for bits in np.unique(ball // HW):
            lay = np.zeros(HW, bool); lay[ball[ball // HW == bits] % HW] = True
            bb = betti(lay.reshape(w.H, w.W)); b = (b[0] + bb[0], b[1] + bb[1])
        cv["V3"].append(b)
        cv["V4"].append(betti(pos & ~door_mask))
    return cv, d


def kstar(curve):
    for k in range(1, len(curve)):
        if curve[k] != curve[0]:
            return k
    return None


def n_changes(curve):
    return sum(curve[k] != curve[k - 1] for k in range(1, len(curve)))


def report_toggle(n_doors, corrupt=0.0, seed=0):
    w = ToggleGrid(n_doors=n_doors)
    HW = w.H * w.W
    NS = w.NS.copy()
    if corrupt > 0:      # a wrong "model": redirect a fraction of entries to a random 4-neighbour / stay state
        rng = np.random.default_rng(seed)
        m = rng.random(NS.shape) < corrupt
        alt = w.NS[np.arange(NS.shape[0])[:, None].repeat(4, 1), rng.integers(0, 4, NS.shape)]
        NS[m] = alt[m]
    door_mask = w.door_id >= 0
    starts = [w.encode(np.array([x, y] + [0] * n_doors)) for y, x in np.argwhere(w.room == 0)]
    rows = []
    for s0 in starts:
        cv, d = curves(w, NS, s0, door_mask)
        dtrue = bfs(w.NS, s0)
        st_pos, st_bits = np.arange(len(d)) % HW, np.arange(len(d)) // HW
        on_door1 = door_mask.ravel()[st_pos] & (w.door_id.ravel()[st_pos] == 0)
        room1 = w.room.ravel()[st_pos] == 1
        first = lambda dd, sel: int(dd[sel & (dd >= 0)].min()) if (sel & (dd >= 0)).any() else None
        rows.append(dict(
            k={v: kstar(c) for v, c in cv.items()}, nch={v: n_changes(c) for v, c in cv.items()},
            V1_vals=set(cv["V1"]),
            d_switch=first(d, st_bits > 0), d_door=first(d, on_door1), d_room1=first(d, room1),
            t_door=first(dtrue, on_door1)))
    tag = f"toggle_grid n_doors={n_doors}" + (f", corrupted model ({corrupt:.0%} of entries)" if corrupt else "")
    print(f"\n== {tag}: {len(rows)} starts (room 0, doors closed), K <= {KMAX}")
    print(f"   V1 (positions, <=k, doors in): Betti values seen = {sorted(set().union(*[r['V1_vals'] for r in rows]))};"
          f" starts with any change = {sum(r['k']['V1'] is not None for r in rows)}")
    v2 = [r['nch']['V2'] for r in rows]
    print(f"   V2 (exactly k): topology changes per start, median {int(np.median(v2))} (min {min(v2)}, max {max(v2)});"
          f" first change at k = {sorted(set(r['k']['V2'] for r in rows), key=lambda v: -1 if v is None else v)}")
    ok3 = all(r['k']['V3'] == r['d_switch'] for r in rows)
    ok4 = all(r['k']['V4'] == r['d_room1'] for r in rows)
    print(f"   V3 (full state, layers): k* == model-BFS steps to first switch entry for every start: {ok3}")
    print(f"   V4 (positions, door cells left out): k* == model-BFS steps to first cell of room 1 for every start: {ok4}")
    for v in ("V3", "V4"):
        ks = np.array([r['k'][v] for r in rows], float); td = np.array([r['t_door'] for r in rows], float)
        md = np.array([r['d_door'] if r['d_door'] is not None else np.nan for r in rows], float)
        okm = np.isfinite(ks) & np.isfinite(md)
        rho_t = spearmanr(ks[np.isfinite(ks)], td[np.isfinite(ks)])[0]
        rho_m = spearmanr(ks[okm], md[okm])[0] if okm.sum() > 2 else np.nan
        off = (ks - md)[okm]
        print(f"   {v}: Spearman(k*, true BFS steps to door) = {rho_t:.3f}; Spearman(k*, model-BFS steps to door) = "
              f"{rho_m:.3f}; k* - model door steps in {sorted(set(off.astype(int).tolist()))}")
    return rows


def loop_maze():
    """Two 5x5 rooms; the wall between them has the door (middle row) AND a second opening in the top row, so the
    wall cell between them is a free-standing obstacle and the free space has b1 = 1. Doors static (open)."""
    rw = rh = 5
    W, H = 2 * (rw + 1) + 1, rh + 2
    free = np.zeros((H, W), bool); free[1:rh + 1, 1:rw + 1] = True; free[1:rh + 1, rw + 2:2 * rw + 2] = True
    wx = rw + 1; door = (wx, 1 + rh // 2); gap = (wx, 1)
    free[door[1], door[0]] = free[gap[1], gap[0]] = True
    ids = -np.ones((H, W), int); cells = np.argwhere(free); ids[free] = np.arange(len(cells))
    NS = np.zeros((len(cells), 4), int)
    for i, (y, x) in enumerate(cells):
        for a, (dx, dy) in enumerate(((0, -1), (0, 1), (-1, 0), (1, 0))):
            ny, nx = y + dy, x + dx
            NS[i, a] = ids[ny, nx] if (0 <= ny < H and 0 <= nx < W and free[ny, nx]) else i
    print(f"\n== loop maze (door at {door}, second opening at {gap}); free space Betti = {betti(free)}")
    ks, dd, dg, far = [], [], [], []
    for s0 in [ids[y, x] for y, x in cells if x < wx]:
        d = bfs(NS, s0)
        curve = []
        for k in range(KMAX + 1):
            m = np.zeros((H, W), bool); sel = (d >= 0) & (d <= k); m[cells[sel, 0], cells[sel, 1]] = True
            curve.append(betti(m))
        ks.append(kstar(curve)); dd.append(d[ids[door[1], door[0]]]); dg.append(d[ids[gap[1], gap[0]]])
        # BFS-only prediction of the loop's birth: the smallest k whose ball contains a cycle around the obstacle;
        # with corner contact that is max(d) over the obstacle's 4 edge neighbours (they touch each other's corners)
        oy, ox = 2, wx                                       # the free-standing wall cell
        far.append(max(d[ids[oy + dy, ox + dx]] for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))))
    ks, dd, dg, far = map(np.array, (ks, dd, dg, far))
    print(f"   k* (b1: 0 -> 1) == max BFS distance over the obstacle's 4 neighbours, every start: {bool((ks == far).all())}")
    print(f"   Spearman(k*, BFS steps to door) = {spearmanr(ks, dd)[0]:.3f}; to second opening = {spearmanr(ks, dg)[0]:.3f};"
          f" to min(door, opening) = {spearmanr(ks, np.minimum(dd, dg))[0]:.3f}; to max(door, opening) = {spearmanr(ks, np.maximum(dd, dg))[0]:.3f}")


def articulation_baseline():
    """The cheap graph baseline: cut vertices of the walkable position graph (Tarjan, O(V + E)), no horizon needed."""
    import networkx as nx
    for nd in (1, 2, 3):
        w = ToggleGrid(n_doors=nd)
        walk = w.free | (w.door_id >= 0)
        G = nx.grid_2d_graph(w.H, w.W).subgraph([tuple(v) for v in np.argwhere(walk)])
        ap = sorted((int(x), int(y)) for y, x in nx.articulation_points(G))
        expect = sorted((dx + o, dy) for dx, dy in w.doors for o in (-1, 0, 1))
        print(f"   n_doors={nd}: articulation points (x, y) = {ap}; == door cells and their two neighbours: {ap == expect}")


if __name__ == "__main__":
    print("== articulation-point baseline on toggle_grid")
    articulation_baseline()
    for nd in (1, 2, 3):
        report_toggle(nd)
    report_toggle(1, corrupt=0.10, seed=0)
    loop_maze()
