"""`permutation_count`: four CTR column sets, and which one the model gets.

    pixi run check-permutation-count

CatBoost builds `permutation_count` datasets that differ ONLY in their
permutation-dependent CTR columns (`doc_parallel_dataset_builder.cpp:
104-124`, `:249-260`); the float, one-hot and `FeatureFreq` columns are
shared. The exported model comes from the ESTIMATION permutation,
`PermutationsCount() - 1` (`doc_parallel_boosting.h:101-103`).

Three facts follow, and each one is a gate here. None of them is observable
from a single fit, which is why this check exists at all: everything the
permutation machinery does happens inside `train()` and shows up only as a
different model.

1. **THE IDENTITY.** `permutation_count=1` and
   `permutation_count=4, ctr_estimation_permutation_id=0` must produce the
   SAME MODEL, bit for bit. Both train on permutation 0's ordered
   statistics and both binarize against permutation 0's borders, so the
   data the exported model sees is identical and nothing else about the
   fit depends on how many sets were built beside it. This is the gate
   that catches a mis-indexed permutation: taking the wrong set, or
   computing the borders from the estimation permutation instead of from
   permutation 0, moves this comparison and nothing else would.

2. **AND IT IS NOT VACUOUS.** `est=0` against `est=1` must DIFFER, or gate
   1 would pass with an implementation that ignored the index entirely.
   The four sets are four different orderings of the same ordered target
   statistic; if they collapsed to one value, permutations would be a
   no-op and CatBoost would not build them.

3. **`permutation_count` IS OVERRIDDEN, NOT DEFAULTED, WITHOUT CTRs.**
   `UpdateGpuSpecificDefaults` (`cuda/train_lib/train.cpp:99-108`) assigns
   1 when no categorical feature feeds a CTR -- it is an assignment, so an
   explicit 4 is discarded too, because four identical permutations of a
   dataset with no permutation-dependent column are four identical
   datasets. On numeric-only data the two fits must therefore be bit
   identical.

SABOTAGE:

    P1  gate 1's two arms compared against a THIRD          that the
        (est=1) which must come out different                comparison
                                                             resolves the
                                                             permutation
                                                             index at all
"""

from max.gpu.host import DeviceContext

from gbdt.train import predict_floats, train

comptime PC_CATEGORIES = 50
comptime PC_ROWS = 4000


def _hashed(x: Int, salt: Int) -> Int:
    var v = ((x + 1) * 2654435761 + salt * 40503) % 1000003
    if v < 0:
        v += 1000003
    return v


def _fixture() raises -> Tuple[List[Float32], List[Float32]]:
    """One categorical column above `one_hot_max_size` so it takes CTRs,
    one numeric column, and a target that varies WITHIN a category.

    The within-category variation is what makes the ordered statistic
    depend on the order: if `y` were a pure function of the code, every
    prefix would see the same mean and the permutations would differ only
    by their prefix LENGTHS. They would still differ, but weakly, and a
    gate that can only just see the difference is a gate that will stop
    seeing it.
    """
    var x = List[Float32]()
    for r in range(PC_ROWS):
        x.append(Float32(_hashed(r, 3) % PC_CATEGORIES))
    for r in range(PC_ROWS):
        x.append(Float32(_hashed(r, 17) % 1000) / Float32(1000.0))

    var y = List[Float32]()
    for r in range(PC_ROWS):
        var code = _hashed(r, 3) % PC_CATEGORIES
        y.append(
            Float32(code) / Float32(PC_CATEGORIES)
            + Float32(_hashed(r, 29) % 100) / Float32(400.0)
        )
    return (x^, y^)


def _predict(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Float32],
    n_features: Int,
    cat: List[Bool],
    perm_count: Int,
    est: Int,
) raises -> List[Float32]:
    var tm = train(
        ctx, x, y, PC_ROWS, n_features,
        border_count=32, n_estimators=6, max_depth=4,
        loss="RMSE", learning_rate=Float32(0.3),
        cat_features=cat,
        permutation_count=perm_count,
        ctr_estimation_permutation_id=est,
    )
    return predict_floats(ctx, tm, x, PC_ROWS)


def _differing_rows(a: List[Float32], b: List[Float32]) -> Int:
    var n = 0
    for r in range(len(a)):
        if a[r] != b[r]:
            n += 1
    return n


def check_permutation_count() raises:
    var ctx = DeviceContext()
    var failures = 0

    var xy = _fixture()
    var x = xy[0].copy()
    var y = xy[1].copy()
    var cat: List[Bool] = [True, False]

    print("-- gate 1: one permutation == four, read at permutation 0 --")
    var one = _predict(ctx, x, y, 2, cat, 1, -1)
    var four_at_0 = _predict(ctx, x, y, 2, cat, 4, 0)
    var d10 = _differing_rows(one, four_at_0)
    if d10 != 0:
        print(
            "  FAIL", d10, "of", PC_ROWS,
            "rows differ between permutation_count=1 and"
            " permutation_count=4 read at 0",
        )
        failures += 1
    else:
        print(
            "  ok   all", PC_ROWS,
            "predictions bit-identical, so the extra three sets change"
            " nothing about the one the model takes",
        )

    print()
    print("-- gate 2: the estimation permutation is really read --")
    var four_at_3 = _predict(ctx, x, y, 2, cat, 4, -1)
    var d13 = _differing_rows(one, four_at_3)
    if d13 == 0:
        print(
            "  FAIL permutation 3's model matches permutation 0's on every"
            " row; the permutations are not distinct",
        )
        failures += 1
    else:
        print(
            "  ok  ", d13, "of", PC_ROWS,
            "rows move when the model comes from permutation 3 instead",
        )

    print()
    print("-- gate 3: no CTRs means permutation_count is OVERRIDDEN --")
    var numeric: List[Bool] = [False, False]
    var num_one = _predict(ctx, x, y, 2, numeric, 1, -1)
    var num_four = _predict(ctx, x, y, 2, numeric, 4, -1)
    var dnum = _differing_rows(num_one, num_four)
    if dnum != 0:
        print(
            "  FAIL", dnum,
            "rows differ on numeric-only data; permutation_count=4 was not"
            " collapsed to 1",
        )
        failures += 1
    else:
        print(
            "  ok   numeric-only: 4 and 1 agree on every row, and"
            " est would be 3 vs 0 if it had not collapsed",
        )

    print()
    print("-- gate 5: the STRUCTURE comes off a different permutation --")
    # Same estimation permutation, same data for it, same everything the
    # exported model is FITTED on. The only difference is which
    # permutation the tree SHAPE was searched on, and their draw
    # (`doc_parallel_boosting.h:349-351`) makes that depend on
    # `permutation_count`:
    #
    #   count 3, est 2 -> learnPermutationCount 2 -> modulus 1 -> always 0
    #   count 4, est 2 -> learnPermutationCount 3 -> modulus 2 -> 0 or 1
    #
    # so the two fits estimate identical leaves on identical rows and can
    # only differ through the structure. A loop that searched on the
    # estimation permutation -- which is what this port did before the
    # permutation loop existed -- gives the SAME model for both.
    var three_at_2 = _predict(ctx, x, y, 2, cat, 3, 2)
    var four_at_2 = _predict(ctx, x, y, 2, cat, 4, 2)
    var dstruct = _differing_rows(three_at_2, four_at_2)
    if dstruct == 0:
        print(
            "  FAIL the structure permutation changes nothing; the loop is"
            " searching on the estimation permutation",
        )
        failures += 1
    else:
        print(
            "  ok  ", dstruct, "of", PC_ROWS,
            "rows move when the structure is searched on permutation 0 or"
            " 1 instead of 0 alone",
        )

    print()
    print("-- gate 4: permutation_count must be positive --")
    var refused = False
    try:
        var _z = _predict(ctx, x, y, 2, cat, 0, -1)
    except e:
        refused = True
    if not refused:
        print("  FAIL permutation_count=0 was accepted")
        failures += 1
    else:
        print("  ok   refused")

    print()
    print("-- sabotages --")
    # P1: gate 1 asserts an EQUALITY, and an equality gate is worthless
    # unless the same comparison can come out unequal. Permutation 1 is
    # the control: same fit, same borders, one index apart.
    var four_at_1 = _predict(ctx, x, y, 2, cat, 4, 1)
    var d01 = _differing_rows(four_at_0, four_at_1)
    if d01 == 0:
        print(
            "  FAIL P1: est=0 and est=1 produce the same model, so gate 1"
            " would pass with the index ignored",
        )
        failures += 1
    else:
        print(
            "  ok   P1", d01, "of", PC_ROWS,
            "rows move between est=0 and est=1, so gate 1's equality is"
            " a real constraint",
        )

    print()
    if failures != 0:
        raise Error(
            "permutation_count check: " + String(failures) + " failures"
        )
    print("permutation_count: PASS")


def main() raises:
    check_permutation_count()
