"""CatBoost's OWN MinEntropy borders, dumped as a fixture the port is gated on.

    pixi run -e bench python tools/minentropy_oracle.py > bench/minentropy_oracle.txt

WHY THIS EXISTS. `MinEntropy` is the border selection CatBoost's GPU uses for
FeatureFreq simple CTRs (`CreateDefaultCounter`,
`catboost_options.cpp:392-415`), and it is NOT the `GreedyLogSum` we ported for
numeric features. `gbdt/grid_creator/binarization.best_split_min_entropy` is a
transliteration of their exact dynamic program, and this file is the only thing
that can tell us it is right: everything else we could compare it against is a
second reading of the same source by the same reader.

THE COLUMNS ARE BUILT TIE-HEAVY ON PURPOSE. FeatureFreq values are
`count / (n + 1)`, so two categories with the same count get the SAME value
exactly, and the DP's equally-optimal cut sets are the common case rather than
an edge case. Consecutive integer category sizes make equal-sum subranges
abundant. A column of distinct well-spread floats would verify the optimum and
nothing about which optimum their tie-breaks reach, which is the failure this
repository already had once, in `best_split`'s heap order.

BUDGETS ARE SWEPT, AND THAT IS NOT DECORATION. The GreedyLogSum bug passed at
budget 15 for months because 15 = 1+2+4+8 lands on a complete tier and is
order-invariant. Any single budget can sit on a symmetry point.

FILE FORMAT. Columns are emitted once and cases reference them by id, because
the same column is gated at several budgets and repeating 27,000 floats per
budget would make the fixture larger than the rest of bench/ combined.

    columns <n>
    column <id> <n_values>
      <value> x n_values
    cases <k>
    case <column_id> <budget> <n_borders>
      <border> x n_borders
"""
import numpy as np
import catboost

#: (family, parameter, budget). "ramp" is consecutive category sizes;
#: "equal" is the tie generator described below.
#: (family, parameter) -> the columns. "ramp" is consecutive category sizes.
#: "equal" is the tie generator, and is the only family that ties.
COLUMNS = [
    ("ramp", 40), ("ramp", 60), ("ramp", 80), ("ramp", 120), ("ramp", 200),
    ("equal", 840),
]

#: (column index, budget). A column with `u` unique values needs
#: `budget + 1 < u` or their short-circuit fires and the DP never runs.
CASES = [
    (0, 7), (0, 13), (1, 23), (2, 31), (3, 47), (4, 100),
    (5, 7), (5, 13), (5, 23), (5, 30),
]


def sizes_ramp(ndist: int):
    return list(range(1, ndist + 1))


def sizes_equal(W: int):
    """Distinct category COUNTS whose per-value WEIGHTS are all identical.

    THIS IS THE FAMILY THAT ACTUALLY TIES, and the ramp family does not. A
    unique ctr value's weight is (how many categories share that count) x
    (the count). Take every divisor c of W and give it multiplicity W/c:
    every unique value then weighs exactly W, the prefix sums form an
    arithmetic progression, and `Penalty(sweights[j] - sweights[i])`
    depends only on `j - i`. Equal-cost cut sets stop being incidental and
    become the structure of the problem, which is what a tie-break test
    needs. The unique count is d(W), the number of divisors, so W sets the
    largest budget that still reaches the DP: 840 has 32.

    The ramp family was committed first and a deliberate sabotage of the
    last-match tie-break did not move a single border, which is how we
    learned it verifies the optimum and nothing about tie order. Same
    lesson as `best_split`'s heap: a fixture that cannot distinguish two
    orders reports both correct.
    """
    out = []
    for c in range(1, W + 1):
        if W % c == 0:
            out.extend([c] * (W // c))
    return out


def build(family: str, param: int, seed: int = 11):
    rng = np.random.default_rng(seed)
    sizes = sizes_ramp(param) if family == "ramp" else sizes_equal(param)
    codes = np.concatenate(
        [np.full(s, i, dtype=np.int64) for i, s in enumerate(sizes)])
    rng.shuffle(codes)
    n = len(codes)
    counts = np.bincount(codes)
    ctr = (counts[codes].astype(np.float64) / (n + 1)).astype(np.float32)
    return ctr, rng.normal(size=n).astype(np.float32)


def borders_for(ctr, y, budget):
    import os
    import tempfile
    pool = catboost.Pool(ctr.reshape(-1, 1), y)
    pool.quantize(border_count=budget, feature_border_type="MinEntropy")
    with tempfile.TemporaryDirectory() as td:
        bp = os.path.join(td, "b.tsv")
        pool.save_quantization_borders(bp)
        out = [float(l.strip().split("\t")[1]) for l in open(bp) if l.strip()]
    return sorted(out)


def main() -> int:
    cols = [build(f, p) for f, p in COLUMNS]
    print("columns %d" % len(cols))
    for i, (ctr, _) in enumerate(cols):
        print("column %d %d" % (i, len(ctr)))
        for v in ctr:
            print("%.9g" % v)
    print("cases %d" % len(CASES))
    for ci, budget in CASES:
        ctr, y = cols[ci]
        bs = borders_for(ctr, y, budget)
        print("case %d %d %d" % (ci, budget, len(bs)))
        for b in bs:
            print("%.9g" % b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
