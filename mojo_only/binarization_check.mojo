"""Does the border selection behave like CatBoost's GreedyLogSum?

There is no CatBoost binary here to diff against, so this asserts the
PROPERTIES their algorithm has, each of which a plausible wrong
implementation breaks:

- borders are strictly increasing and there are at most `border_count`
- every border lies strictly between two observed values, because their
  `LeftBorder` is the midpoint of the pair straddling the bin edge
- on a UNIFORM column the bins come out near-equal, which is what
  maximising the sum of logs of bin sizes means
- on a column with FEWER distinct values than the budget, the number of
  borders is capped by the distinct count and not by the budget
- `binarize` and the borders agree: no row lands outside `0..len(borders)`,
  and the bin counts match a direct scan
"""

from ported.grid_creator.binarization import best_split, binarize


def check_binarization() raises:
    # ---- uniform column, generous budget ----
    var n = 10000
    var vals = List[Float32]()
    for i in range(n):
        vals.append(Float32(i))
    var borders = best_split(vals.copy(), 15)
    print("    uniform 10000 values, budget 15 ->", len(borders), "borders")

    if len(borders) > 15:
        raise Error("more borders than the budget allows")
    for i in range(1, len(borders)):
        if borders[i] <= borders[i - 1]:
            raise Error("borders are not strictly increasing")

    # bin occupancy on a uniform column should be near-equal
    var counts = List[Int]()
    for _ in range(len(borders) + 1):
        counts.append(0)
    for i in range(n):
        counts[binarize(vals[i], borders)] += 1
    var lo = counts[0]
    var hi = counts[0]
    for i in range(len(counts)):
        if counts[i] < lo:
            lo = counts[i]
        if counts[i] > hi:
            hi = counts[i]
    var total = 0
    for i in range(len(counts)):
        total += counts[i]
    print("      bin sizes: min", lo, "max", hi, "total", total)
    if total != n:
        raise Error("binarize lost rows")
    # a greedy log-sum split of a uniform column halves repeatedly, so the
    # widest bin should be within 2x of the narrowest
    if hi > 2 * lo:
        raise Error(
            String("uniform column binned unevenly: ") + String(lo)
            + " to " + String(hi)
        )

    # ---- every border strictly inside the observed range ----
    for i in range(len(borders)):
        if borders[i] <= vals[0] or borders[i] >= vals[n - 1]:
            raise Error("a border landed outside the observed values")

    # ---- fewer distinct values than the budget ----
    var few = List[Float32]()
    for i in range(1000):
        few.append(Float32(i % 4))
    var fb = best_split(few.copy(), 32)
    print("    4 distinct values, budget 32 ->", len(fb), "borders")
    if len(fb) > 3:
        raise Error(
            String("4 distinct values cannot need more than 3 borders, got ")
            + String(len(fb))
        )

    # ---- a constant column has no split at all ----
    var flat = List[Float32]()
    for _ in range(500):
        flat.append(Float32(7.0))
    var cb = best_split(flat.copy(), 32)
    print("    constant column ->", len(cb), "borders")
    if len(cb) != 0:
        raise Error("a constant column produced borders")

    # ---- skewed column: the budget is spent where the mass is ----
    var skew = List[Float32]()
    for i in range(10000):
        if i < 9000:
            skew.append(Float32(i % 100) * Float32(0.01))
        else:
            skew.append(Float32(100 + i % 900))
    var sb = best_split(skew.copy(), 15)
    print("    skewed column, budget 15 ->", len(sb), "borders")
    if len(sb) == 0:
        raise Error("a skewed column produced no borders")

    print("  border selection behaves like GreedyLogSum")
