"""Binomial arithmetic for the cell-0 decision rules (no data)."""
import itertools
from math import comb

def pmf(n, p, k): return comb(n, k) * p**k * (1 - p)**(n - k)
def p_atleast(n, p, k): return sum(pmf(n, p, j) for j in range(k, n + 1))

print("P0.2 phase 1: 9 neg-control runs, '>= 90%' needs 9/9. P(STOP by instrument alone):")
for fp in (0.01, 0.02, 0.05, 0.10):
    print(f"  per-run false wrap rate {fp:.2f}: {1 - (1 - fp)**9:.3f}")
print("P0.1 phase 1: 12 recon runs (angle_cs S/M/L x3 + pixels S x3), '>= 90%' needs >= 11/12. P(STOP):")
for fn in (0.02, 0.05, 0.10, 0.20):
    print(f"  per-run false negative {fn:.2f}: {1 - p_atleast(12, 1 - fn, 11):.3f}")

def p_monotone(p, sizes, seeds=3):
    tot = 0.0
    for ks in itertools.product(range(seeds + 1), repeat=sizes):
        if all(ks[i] <= ks[i + 1] for i in range(sizes - 1)):
            pr = 1.0
            for k in ks: pr *= pmf(seeds, p, k)
            tot += pr
    return tot
print("P0.7 'pass rate non-decreasing in size' with NO size effect (same p at every size), 3 seeds/size:")
for p in (0.5, 0.7, 0.9, 0.97, 1.0):
    print(f"  p = {p:.2f}: 3 sizes {p_monotone(p, 3):.3f}, 4 sizes {p_monotone(p, 4):.3f}")

print("P0.4 false-positive probability when jepa and recon-mse glue with the SAME prob p,")
print("  criterion: glued fraction among GATED jepa >= 2/3 and minus recon-mse fraction >= 0.5; recon-mse 9 gated")
for p in (0.1, 0.2, 0.33, 0.5):
    row = []
    for nj in (2, 3, 5, 9):
        s = 0.0
        for xj in range(nj + 1):
            if 3 * xj < 2 * nj: continue
            for xr in range(10):
                if xj / nj - xr / 9 >= 0.5 - 1e-12:
                    s += pmf(nj, p, xj) * pmf(9, p, xr)
        row.append(f"n_gated_jepa={nj}: {s:.3f}")
    print(f"  p = {p:.2f}: " + ", ".join(row))

print("P0.5 per (obs, variant, size) cell with 3 seeds: original '1-2 of 3' vs author '>= 2 of 3'")
for p in (0.3, 0.5, 0.7, 0.9, 0.97):
    p0, p1, p2, p3 = (pmf(3, p, k) for k in range(4))
    print(f"  p = {p:.2f}: both 'confirmed' (2/3) {p2:.3f}; only original (1/3) {p1:.3f}; only author (3/3) {p3:.3f}; neither (0/3) {p0:.3f}")

print("P0.6-style 'all gated runs decodable' with per-run pass prob q (calibrated conjunction 0.88; torn charts 0.5):")
for q in (0.93, 0.88, 0.5):
    print(f"  q = {q:.2f}: P(all of 9) {q**9:.3f}, P(all of 12) {q**12:.3f}, P(>= 2/3 of 9) {p_atleast(9, q, 6):.3f}")
