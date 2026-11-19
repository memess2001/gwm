"""Cell 12, 2026-11-19 (Antigravity review round): symmetric multi-room graphs, where lambda_2 is degenerate.
A single Fiedler vector is an arbitrary vector of the 2-D (or 3-D) eigenspace; does dynamic ORC beat the basis-invariant
same-source baselines that the precheck already lists (k-way spectral embedding, diffusion distance, TV, betweenness)?
Run from the repo root:  PYTHONPATH=. python cells/12/exploratory/precheck_degenerate_rooms.py
"""
import importlib.util, numpy as np, networkx as nx
from scipy.linalg import expm, eigh
from scipy.stats import spearmanr
spec = importlib.util.spec_from_file_location("m12", "cells/12/exploratory/precheck_orc_vs_spectral.py")
m12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m12)

def star(n_rooms=3, rw=5):
    G = nx.Graph([((100, i), (100, (i + 1) % n_rooms)) for i in range(n_rooms)])
    truth = set()
    for r in range(n_rooms):
        for xx in range(rw):
            for yy in range(rw):
                if xx + 1 < rw: G.add_edge((r, xx * rw + yy), (r, (xx + 1) * rw + yy))
                if yy + 1 < rw: G.add_edge((r, xx * rw + yy), (r, xx * rw + yy + 1))
        e = ((100, r), (r, 0)); G.add_edge(*e); truth.add(e)
    return G, truth

for n_rooms in (3, 4):
    G, truth = star(n_rooms)
    g = m12.Graph(G)
    is_tr = np.array([(g.nodes[x], g.nodes[y]) in truth or (g.nodes[y], g.nodes[x]) in truth for x, y in g.edges])
    sc, lam2 = m12.scores(g, n_rooms)
    L = np.eye(len(g.nodes)) - g.P
    for t in (256.0,):
        Ht = expm(-t * L)
        sc[f"orc_t{t:g}"] = np.array([g.w1(Ht[x], Ht[y]) for x, y in g.edges])
        sc[f"tv_t{t:g}"] = np.array([0.5 * np.abs(Ht[x] - Ht[y]).sum() for x, y in g.edges])
        sc[f"dd_t{t:g}"] = np.array([np.sqrt(((Ht[x] - Ht[y]) ** 2 / g.pi).sum()) for x, y in g.edges])
    Dh = np.diag(g.deg ** -0.5)
    w, V = eigh(np.eye(len(g.nodes)) - Dh @ g.A @ Dh)
    print(f"== {n_rooms}-room star: lambda_2..5 = {np.round(w[1:5], 5)}, auto k = {sc['_k']}")
    keys = [k for k in sc if not k.startswith("_")]
    print("  P@doors:", ", ".join(f"{k}={m12.precision(sc[k], is_tr):.3f}" for k in keys))
    sub = Dh @ V[:, 1:n_rooms]                       # the whole degenerate lambda_2 eigenspace (n_rooms - 1 dims)
    es = np.array([np.linalg.norm(sub[x] - sub[y]) for x, y in g.edges])
    print(f"  lambda_2 eigenspace (dim {n_rooms - 1}) P@doors = {m12.precision(es, is_tr):.3f}; "
          f"Spearman(orc_t256, eigenspace) = {spearmanr(sc['orc_t256'], es).statistic:.3f}; "
          f"Spearman(orc_t256, fiedler) = {spearmanr(sc['orc_t256'], sc['fiedler']).statistic:.3f}")
