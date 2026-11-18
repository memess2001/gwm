"""Cell 10, check before preregistration (ground truth only, exhaustive enumeration; nothing is learned).

Israeli & Goldenfeld's coarse-graining of elementary cellular automata (ECA), block size N = 2:
a rule A can be coarse-grained to a rule B with the block projection P : {0,1}^2 -> {0,1} iff

    P( A^2 (x) ) = B( P(x) )      for every configuration x        (two fine steps = one coarse step).

For a given (A, P) the coarse rule B, if it exists, is fixed by a finite consistency check over 6 fine cells
(3 blocks), so "the rule at the next scale" is obtained by enumeration, not by learning.

Counts printed: rules with a non-trivial coarse-graining at N = 2, self-similar rules (B = A), and the fixed
points / flows of the map A -> B. Runs in well under a second.
"""
from itertools import product


def step(rule, cells):
    """One ECA step on a tuple; the output is two cells shorter (no boundary assumption)."""
    return tuple((rule >> (4 * cells[i - 1] + 2 * cells[i] + cells[i + 1])) & 1 for i in range(1, len(cells) - 1))


def coarse_rule(A, P):
    """The ECA number B with P(A^2(x)) = B(P(x)) for all x, or None if no such rule exists.
    P is a 4-bit table: P[(a, b)] = (P >> (2 * a + b)) & 1."""
    proj = lambda a, b: (P >> (2 * a + b)) & 1
    table = {}
    for x in product((0, 1), repeat=10):                     # 10 fine cells -> 2 steps -> 6 cells = 3 blocks
        y = step(A, step(A, x))                              # y = cells 2..7 of x after two steps
        # blocks of x aligned with y: (x2,x3) (x4,x5) (x6,x7); their neighbours (x0,x1) and (x8,x9) feed the update
        key = (proj(x[2], x[3]), proj(x[4], x[5]), proj(x[6], x[7]))
        val = proj(y[2], y[3])                               # the centre block after two steps
        if table.setdefault(key, val) != val:
            return None
    if len(table) < 8:
        return None                                          # projection not onto: no ECA on the coarse level
    return sum(table[(l, c, r)] << (4 * l + 2 * c + r) for l, c, r in table)


def main():
    nontrivial = [P for P in range(16) if P not in (0, 15)]  # drop the two constant projections
    found = {}                                               # A -> list of (P, B)
    for A in range(256):
        for P in nontrivial:
            B = coarse_rule(A, P)
            if B is not None:
                found.setdefault(A, []).append((P, B))
    n_pairs = sum(len(v) for v in found.values())
    print(f"block size 2, 14 non-constant projections, 256 rules: {len(found)} rules have at least one coarse-graining "
          f"({n_pairs} (rule, projection) pairs)")
    selfsim = sorted(A for A, v in found.items() if any(B == A for _, B in v))
    print(f"self-similar at N = 2 (coarse rule = fine rule for some projection): {len(selfsim)} rules: {selfsim}")
    targets = sorted({B for v in found.values() for _, B in v})
    print(f"distinct coarse rules reached: {len(targets)}: {targets}")
    for A in (90, 105, 150, 146, 110, 30, 54):
        v = found.get(A, [])
        print(f"   rule {A:3d}: " + (", ".join(f"P={P:04b} -> rule {B}" for P, B in v) if v else "no coarse-graining at N = 2"))
    # the "flow of rules along scale": iterate A -> B (all projections) and list the closed sets
    reach = {A: sorted({B for _, B in v}) for A, v in found.items()}
    fixed = sorted(A for A, bs in reach.items() if A in bs)
    print(f"fixed points of the scale map A -> B: {len(fixed)} (same list as the self-similar rules: {fixed == selfsim})")
    chains = [(A, bs) for A, bs in sorted(reach.items()) if A not in bs]
    print(f"rules that flow to a different rule: {len(chains)}; examples: " +
          "; ".join(f"{A} -> {bs}" for A, bs in chains[:8]))
    print("a (rule, projection) pair is decided by one pass over 2^10 fine configurations: enumeration, no learning")


if __name__ == "__main__":
    main()
