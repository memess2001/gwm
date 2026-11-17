"""Cell 12 pre-prereg check (exact graphs only, no learning, no runs/): does Ollivier-Ricci curvature (ORC) find
bottleneck edges that cheap same-source non-geometric statistics miss?

Graphs: undirected 4-neighbour position graphs of grid worlds (doors open; toggle_grid layouts from zoo/ plus
hand-drawn ones) and one non-grid control (two random 3-regular graphs joined by two edges).
Truth = cut edges: edges with exactly one endpoint on a cell marked 'D' (2w edges for a door of width w), or the
two cross edges of the expander pair.  Score per edge (higher = more bottleneck):
  orc0 / orc05     -ORC with the 1-step walk measure, laziness alpha = 0 / 0.5 (Ollivier 2009)
  orc_t            -dynamical ORC at scale t (Gosztolai & Arnaudon 2021): measures m_x^t = row x of exp(-t(I-P))
  sq               -(number of 4-cycles through the edge): a raw local cycle count, same 3-hop neighbourhood as orc0
  c4               -(Radicchi 2004 square edge-clustering (q+1)/((d_x-1)(d_y-1))): the same count, normalised
  tv_t, dd_t       TV and diffusion distance (L2(1/pi), Coifman-Lafon) between the SAME m_x^t, m_y^t as orc_t
  fiedler          |phi_2(x) - phi_2(y)| of the random-walk Laplacian
  speck            ||Phi(x) - Phi(y)||, Phi = random-walk eigenvectors 2..k+1, k by the largest RATIO lambda_{j+1}/lambda_j
                   (j = 1..7; multi-way spectral clustering embedding)
  speco            the same with the oracle k = number of doors (the strongest spectral baseline)
  btw              edge betweenness (Simsek & Barto 2008)
Sections: (1) exact graphs, all scores, plus a check that 1-step ORC is a function of the local signature
(min degree, max degree, #4-cycles through the edge); (2) short horizons: W1 vs TV / Hellinger on the same measures,
with separation margins; (3) count-estimated kernels (n observed transitions per state, no learning).
Runtime about 6 minutes on one CPU (section 3 dominates). Output: precheck_orc_vs_spectral.out.
W1 on a graph metric = min-cost flow on the graph (Beckmann form), solved as a small LP.
Precision@m (m = #truth edges) counts ties by their expected value under random tie-breaking."""
import sys
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.linalg import expm, eigh
from scipy.optimize import linprog
from scipy.stats import spearmanr
sys.path.insert(0, "/home/user/gwm_repo")
from zoo.toggle_grid import ToggleGrid

T_SCALES = (1.0, 4.0, 16.0, 64.0)


def grid_graph(rows):
    """rows: list of strings, '#' wall, '.' free, 'D' door (free, marks truth)."""
    G, door = nx.Graph(), set()
    H, W = len(rows), len(rows[0])
    for y in range(H):
        for x in range(W):
            if rows[y][x] == "#":
                continue
            G.add_node((x, y))
            if rows[y][x] == "D":
                door.add((x, y))
            for dx, dy in ((1, 0), (0, 1)):
                if x + dx < W and y + dy < H and rows[y + dy][x + dx] != "#":
                    G.add_edge((x, y), (x + dx, y + dy))
    truth = {e for e in G.edges if (e[0] in door) != (e[1] in door)}
    return G, truth


def toggle_rows(k, rw=5, rh=5):
    w = ToggleGrid(n_doors=k, room_w=rw, room_h=rh)
    return [r.replace("S", ".") for r in w.ascii().split("\n")]


def rooms(widths, rw=7, rh=7, sizes=None):
    """Rooms in a row; door i has width widths[i] (centred); sizes = optional per-room (w, h) overrides."""
    sizes = sizes or [(rw, rh)] * (len(widths) + 1)
    H = max(h for _, h in sizes) + 2
    cols = [["#"] * H]
    for j, (w, h) in enumerate(sizes):
        for _ in range(w):
            cols.append(["#"] + ["." if 1 <= y <= h else "#" for y in range(1, H - 1)] + ["#"])
        wall = ["#"] * H
        if j < len(widths):
            hmin = min(h, sizes[j + 1][1])
            c0 = 1 + (hmin - widths[j]) // 2
            for y in range(c0, c0 + widths[j]):
                wall[y] = "D"
        cols.append(wall)
    return ["".join(cols[x][y] for x in range(len(cols))) for y in range(H)]


def with_deadend(rows, x, y0, length):
    """Carve a 1-wide dead-end corridor going up from (x, y0) through the top wall (adds rows)."""
    W = len(rows[0])
    new = ["#" * W for _ in range(length)] + list(rows)
    new = [list(r) for r in new]
    for y in range(0, length + y0 + 1):
        if y >= 1:
            new[y][x] = "." if new[y][x] == "#" else new[y][x]
    new[0][x] = "#"
    return ["".join(r) for r in new]


def expander_pair(n=100, seed=0):
    rng = np.random.default_rng(seed)
    A = nx.random_regular_graph(3, n, seed=int(rng.integers(1 << 30)))
    B = nx.relabel_nodes(nx.random_regular_graph(3, n, seed=int(rng.integers(1 << 30))), lambda v: v + n)
    G = nx.union(A, B)
    (a, b), (c, d) = list(A.edges)[0], list(B.edges)[0]
    G.remove_edges_from([(a, b), (c, d)])
    G.add_edges_from([(a, c), (b, d)])                      # degree-preserving: every node keeps degree 3
    return G, {(a, c), (b, d)}


class Graph:
    def __init__(self, G):
        self.nodes = list(G.nodes)
        self.idx = {v: i for i, v in enumerate(self.nodes)}
        self.edges = [(self.idx[u], self.idx[v]) for u, v in G.edges]
        n, m = len(self.nodes), len(self.edges)
        A = nx.to_numpy_array(G, nodelist=self.nodes)
        self.A, self.deg = A, A.sum(1)
        self.P = A / self.deg[:, None]
        self.pi = self.deg / self.deg.sum()
        r = np.repeat(np.arange(m), 2)
        c = np.array(self.edges).ravel()
        v = np.tile([1.0, -1.0], m)
        B = sp.csr_matrix((v, (c, r)), shape=(n, m))          # incidence matrix, node x edge
        self.Aeq = sp.hstack([B, -B]).tocsc()
        self.G = G

    def w1(self, mu, nu):
        diff = mu - nu
        diff = diff - diff.mean()                               # exact zero total (HiGHS rejects 1e-16 imbalances)
        s = np.abs(diff).sum()
        if s < 1e-300:
            return 0.0
        res = linprog(np.ones(self.Aeq.shape[1]), A_eq=self.Aeq, b_eq=diff / s, bounds=(0, None), method="highs")
        assert res.status == 0, res.message
        return res.fun * s

    def step_measure(self, x, alpha):
        m = (1 - alpha) * self.P[x].copy()
        m[x] += alpha
        return m


def scores(g, k_oracle):
    E = g.edges
    out = {}
    for a in (0.0, 0.5):
        out[f"orc{'0' if a == 0 else '05'}"] = np.array([-(1 - g.w1(g.step_measure(x, a), g.step_measure(y, a))) for x, y in E])
    nb = [set(np.flatnonzero(g.A[i])) for i in range(len(g.nodes))]
    c4 = []
    for x, y in E:
        q = sum(1 for a_ in nb[x] - {y} for b_ in nb[y] - {x} if a_ != b_ and g.A[a_, b_] > 0)
        c4.append((q + 1) / max(1.0, (g.deg[x] - 1) * (g.deg[y] - 1)))
    out["c4"] = -np.array(c4)
    out["sq"] = -np.array([(cc * max(1.0, (g.deg[x] - 1) * (g.deg[y] - 1)) - 1) for cc, (x, y) in zip(c4, E)])
    L = np.eye(len(g.nodes)) - g.P
    for t in T_SCALES:
        Ht = expm(-t * L)
        out[f"orc_t{t:g}"] = np.array([g.w1(Ht[x], Ht[y]) for x, y in E])          # -(1 - W1) up to a constant
        out[f"tv_t{t:g}"] = np.array([0.5 * np.abs(Ht[x] - Ht[y]).sum() for x, y in E])
        out[f"dd_t{t:g}"] = np.array([np.sqrt(((Ht[x] - Ht[y]) ** 2 / g.pi).sum()) for x, y in E])
    Dh = np.diag(g.deg ** -0.5)
    w, V = eigh(np.eye(len(g.nodes)) - Dh @ g.A @ Dh)
    phi2 = Dh @ V[:, 1]
    out["fiedler"] = np.array([abs(phi2[x] - phi2[y]) for x, y in E])
    k = int(np.argmax(w[2:9] / w[1:8])) + 1                     # ratio lambda_{j+1}/lambda_j, j = 1..7
    for name, kk in (("speck", k), ("speco", k_oracle)):
        Phi = Dh @ V[:, 1:kk + 1]
        out[name] = np.array([np.linalg.norm(Phi[x] - Phi[y]) for x, y in E])
    out["_k"] = k
    bc = nx.edge_betweenness_centrality(g.G, normalized=True)
    out["btw"] = np.array([bc.get((g.nodes[x], g.nodes[y]), bc.get((g.nodes[y], g.nodes[x]))) for x, y in E])
    out["_sig"] = [(min(g.deg[x], g.deg[y]), max(g.deg[x], g.deg[y]), -out["sq"][i]) for i, (x, y) in enumerate(E)]
    return out, w[1]


def precision(score, is_true):
    m = int(is_true.sum())
    s = np.round(score, 10)
    thr = np.sort(s)[::-1][m - 1]
    above, tie = s > thr, s == thr
    need = m - above.sum()
    return (is_true[above].sum() + need * is_true[tie].mean()) / m


CASES = {   # name: (constructor, number of doors = oracle k)
    "A1 toggle_grid k=1 (5x5, door w=1)": (lambda: grid_graph(toggle_rows(1)), 1),
    "A3 toggle_grid k=3 (4 rooms, w=1)": (lambda: grid_graph(toggle_rows(3)), 3),
    "B2 two 7x7 rooms, door w=2": (lambda: grid_graph(rooms([2])), 1),
    "B3 two 7x7 rooms, door w=3": (lambda: grid_graph(rooms([3])), 1),
    "C  9x9 | 9x9 | 3x3 closet, doors w=1": (lambda: grid_graph(rooms([1, 1], sizes=[(9, 9), (9, 9), (3, 3)])), 2),
    "D  two 7x7 rooms door w=2 + dead-end corridor": (lambda: grid_graph(with_deadend(rooms([2]), 3, 1, 4)), 1),
    "E  two random 3-regular (n=100) joined by 2 edges": (lambda: expander_pair(), 1),
}

if __name__ == "__main__":
    sig_map = {}
    for name, (make, k_or) in CASES.items():
        G, truth = make()
        g = Graph(G)
        is_true = np.array([(g.nodes[x], g.nodes[y]) in truth or (g.nodes[y], g.nodes[x]) in truth for x, y in g.edges])
        sc, lam2 = scores(g, k_or)
        sig, kgap = sc.pop("_sig"), sc.pop("_k")
        if not name.startswith("E"):
            for sg, val in zip(sig, np.round(-sc["orc0"], 6)):
                sig_map.setdefault(sg, set()).add(float(val))
        print(f"\n## {name}: n={len(g.nodes)} |E|={len(g.edges)} truth={is_true.sum()} lambda2={lam2:.4f} k_gap={kgap}")
        print("precision@m: " + "  ".join(f"{k}={precision(v, is_true):.2f}" for k, v in sc.items()))
        rho = lambda a, b: spearmanr(sc[a], sc[b]).statistic
        print("spearman over edges: orc0~c4 %.2f  orc05~c4 %.2f" % (rho("orc0", "c4"), rho("orc05", "c4")))
        print("  " + "  ".join(f"t={t:g}: orc~dd {rho(f'orc_t{t:g}', f'dd_t{t:g}'):.2f} orc~tv {rho(f'orc_t{t:g}', f'tv_t{t:g}'):.2f}"
                               f" orc~fiedler {rho(f'orc_t{t:g}', 'fiedler'):.2f}" for t in T_SCALES))
        v = np.round(-sc["orc0"], 4)
        print("  orc0 (non-lazy): values on truth edges", sorted(set(v[is_true].tolist())), "| non-truth edges at or below"
              " the most negative truth value:", int((v[~is_true] <= v[is_true].min()).sum()), "| most negative non-truth",
              v[~is_true].min())

    print("\n## 1-step ORC (non-lazy) by local signature (min deg, max deg, #4-cycles through the edge), grid graphs A-D:")
    for sg in sorted(sig_map):
        print("  ", tuple(int(v) for v in sg), "->", sorted(sig_map[sg]))
    print("  ORC0 is a function of the signature on these graphs:", all(len(v) == 1 for v in sig_map.values()))

    # ---- section 2: short horizons. Does the OT ground metric beat metric-free divergences on the SAME measures,
    # and by how much? margin = (smallest truth score - largest non-truth score) / largest non-truth score (> 0: separated)
    print("\n## short horizons: W1 vs metric-free divergences on the same m_x^t, m_y^t (precision@m; margin in brackets)")
    for name, (make, _) in CASES.items():
        G, truth = make()
        g = Graph(G)
        is_true = np.array([(g.nodes[x], g.nodes[y]) in truth or (g.nodes[y], g.nodes[x]) in truth for x, y in g.edges])
        L = np.eye(len(g.nodes)) - g.P
        row = []
        for t in (0.5, 1.0, 2.0, 4.0, 16.0):
            Ht = expm(-t * L)
            sc = dict(w1=np.array([g.w1(Ht[x], Ht[y]) for x, y in g.edges]),
                      tv=np.array([0.5 * np.abs(Ht[x] - Ht[y]).sum() for x, y in g.edges]),
                      hel=np.array([1 - np.sqrt(Ht[x] * Ht[y]).sum() for x, y in g.edges]))
            cell = []
            for k, v in sc.items():
                mg = (v[is_true].min() - v[~is_true].max()) / v[~is_true].max()
                cell.append(f"{k} {precision(v, is_true):.2f} ({mg:+.3f})")
            row.append(f"t={t:g}: " + ", ".join(cell))
        print(name)
        for r in row:
            print("   ", r)

    # ---- section 3: count-estimated kernels (no learning). P_hat(x, .) = multinomial(n, P(x, .)) / n, i.e. the
    # empirical kernel from n observed transitions per state. The same P_hat feeds every score (spectral scores use
    # S = (P_hat + P_hat^T)/2 as a weighted adjacency). 10 draws per n. Statistics on the unweighted SUPPORT graph
    # (1-step ORC, 4-cycle counts, betweenness, unweighted spectral) are exact here as soon as every edge has been seen
    # once (n = 20: P(miss an edge) <= 0.75^20 = 0.3%), so they keep their section-1 values.
    print("\n## count-estimated kernels: mean precision@m over 10 draws (n = transitions observed per state)")
    rng = np.random.default_rng(0)
    for name in ("A1 toggle_grid k=1 (5x5, door w=1)", "B2 two 7x7 rooms, door w=2", "C  9x9 | 9x9 | 3x3 closet, doors w=1",
                 "D  two 7x7 rooms door w=2 + dead-end corridor", "E  two random 3-regular (n=100) joined by 2 edges"):
        make, k_or = CASES[name]
        G, truth = make()
        g = Graph(G)
        is_true = np.array([(g.nodes[x], g.nodes[y]) in truth or (g.nodes[y], g.nodes[x]) in truth for x, y in g.edges])
        print(name)
        for n_obs in (20, 100):
            acc = {}
            for _ in range(10):
                Ph = np.stack([rng.multinomial(n_obs, p) / n_obs for p in g.P])
                L = np.eye(len(g.nodes)) - Ph
                for t in (4.0, 16.0):
                    Ht = expm(-t * L)
                    for k, v in (("w1", [g.w1(Ht[x], Ht[y]) for x, y in g.edges]),
                                 ("tv", [0.5 * np.abs(Ht[x] - Ht[y]).sum() for x, y in g.edges]),
                                 ("hel", [1 - np.sqrt(Ht[x] * Ht[y]).sum() for x, y in g.edges])):
                        acc.setdefault(f"{k}_t{t:g}", []).append(precision(np.array(v), is_true))
                S = 0.5 * (Ph + Ph.T)
                Dh = np.diag(S.sum(1) ** -0.5)
                w, V = eigh(np.eye(len(g.nodes)) - Dh @ S @ Dh)
                kg = int(np.argmax(w[2:9] / w[1:8])) + 1
                for nm, kk in (("fiedler", 1), ("speck", kg), ("speco", k_or)):
                    Phi = Dh @ V[:, 1:kk + 1]
                    acc.setdefault(nm, []).append(precision(np.array([np.linalg.norm(Phi[x] - Phi[y]) for x, y in g.edges]), is_true))
            print(f"   n={n_obs:3d}: " + "  ".join(f"{k}={np.mean(v):.2f}" for k, v in acc.items()))
