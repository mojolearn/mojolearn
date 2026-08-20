"""CatBoost's OWN target borders for the CTR target grid, as a fixture.

    pixi run -e bench python tools/ctr_target_oracle.py > bench/ctr_target_oracle.txt

WHY THIS IS A SEPARATE FIXTURE FROM `minentropy_oracle.txt`.

Both gate the same dynamic program, but at opposite ends of it and on
different data. `minentropy_oracle.txt` sweeps budgets 7 to 100 over CTR
VALUE columns -- `count / (n + 1)`, which mints exact ties by construction.
The CTR TARGET grid is `TBinarizationOptions(EBorderSelectionType::MinEntropy,
1)` (`cat_feature_options.cpp:230`), built by
`featuresManager.SetTargetBorders(TBordersBuilder(...)(...))`
(`cuda/train_lib/train.cpp:370-375`) over the TARGET, at budget **one**.

Budget 1 is not a small case of budget 15, it is a different code path:
`bins = 2`, so `bins - 2 == 0` and the DP's per-level loop never runs at
all. Everything is decided by the "Last match" scan at
`binarization.cpp:629-645`, whose tie-break is `<` (first index wins),
OPPOSITE to the `<=` the per-level loops use. A fixture that never runs at
budget 1 gates the loops and leaves the only surviving comparison
unchecked.

So the columns here are TARGET-shaped, not CTR-shaped: continuous, binary,
heavily tied, bimodal, and one that is nearly constant. Budgets 1, 2 and 3
are swept, because `ctr_target_border_count` is a user option and 1 is only
its default.

WHAT THIS IS AN ORACLE FOR, EXACTLY. `pool.quantize(feature_border_type=
"MinEntropy")` runs `NSplitSelection::MakeBinarizer(MinEntropy)->BestSplit`,
which is the same binarizer `TGridBuilderFactory::Create(MinEntropy)`
returns for the target (`grid_creator.cpp:60-78`). Below 100000 rows their
`BuildBorders` does not subsample (`GetSampleSizeForBorderSelectionType`,
`quantization/utils.h:131-135`), and the target path disables subsampling
outright (`TargetBinarization.Get().DisableMaxSubsetSizeForBuildBordersOption()`,
`cat_feature_options.cpp:239`), so the two agree on these column sizes.

FILE FORMAT, the same shape as `minentropy_oracle.txt`:

    columns <n>
    column <id> <n_values>
      <value> x n_values
    cases <k>
    case <column_id> <budget> <n_borders>
      <border> x n_borders
"""
import os
import sys
import tempfile

import numpy as np
import catboost

#: (name, n) -- the shapes a TARGET takes, not the shapes a CTR value takes.
COLUMNS = [
    ("continuous", 4001),
    ("binary", 4001),
    ("tied", 4001),
    ("bimodal", 4001),
    ("near_constant", 4001),
]

#: (column index, budget). Budget 1 is the GPU default and the only one
#: that skips the DP's per-level loops entirely.
CASES = [(c, b) for c in range(len(COLUMNS)) for b in (1, 2, 3)]


def build(name: str, n: int, seed: int = 29):
    rng = np.random.default_rng(seed + hash(name) % 1000)
    if name == "continuous":
        y = rng.normal(size=n)
    elif name == "binary":
        y = (rng.random(size=n) < 0.37).astype(np.float64)
    elif name == "tied":
        # eleven distinct values, wildly unequal counts: the last-match
        # scan has plateaus to settle on
        levels = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                           4.5, 5.0])
        weights = np.array([50, 1, 30, 1, 1, 40, 1, 1, 20, 1, 10], float)
        weights /= weights.sum()
        y = rng.choice(levels, size=n, p=weights)
    elif name == "bimodal":
        y = np.where(rng.random(size=n) < 0.5,
                     rng.normal(-3.0, 0.4, size=n),
                     rng.normal(+3.0, 0.4, size=n))
    else:
        y = np.full(n, 2.25)
        y[: n // 200] = 9.75
    return y.astype(np.float32)


def borders_for(y, budget):
    """CatBoost's own MinEntropy borders for `y` at `budget`."""
    pool = catboost.Pool(y.reshape(-1, 1), y)
    pool.quantize(border_count=budget, feature_border_type="MinEntropy")
    with tempfile.TemporaryDirectory() as td:
        bp = os.path.join(td, "b.tsv")
        pool.save_quantization_borders(bp)
        out = [float(l.strip().split("\t")[1]) for l in open(bp) if l.strip()]
    return sorted(out)


def main() -> int:
    cols = [build(name, n) for name, n in COLUMNS]
    print("columns %d" % len(cols))
    for i, y in enumerate(cols):
        print("column %d %d" % (i, len(y)))
        for v in y:
            print("%.9g" % v)
    print("cases %d" % len(CASES))
    for ci, budget in CASES:
        bs = borders_for(cols[ci], budget)
        print("case %d %d %d" % (ci, budget, len(bs)))
        for b in bs:
            print("%.9g" % b)
    print("names %d" % len(COLUMNS), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
