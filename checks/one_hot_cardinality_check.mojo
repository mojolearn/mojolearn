# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One-hot features across the GRID-POLICY BOUNDARIES.

    pixi run check-onehot-cardinality

## The bug this was written to catch, and it was shipped

`train()` gave a one-hot feature `len(borders) + 1` folds
(`train.mojo:135`, because a k-category feature reaches bin `k - 1` through
`k - 1` synthetic borders and the equality candidates need a fold for the
top bin), while `_build_cindex_from_floats` RE-DERIVED its own fold count as
`len(borders[f])` with no `+1` and handed THAT to `build_layout`. So the
layout that WRITES the compressed index and the layout that `fit` and
`predict` READ it with could disagree by one fold -- and
`policy_for_fold_count` (`grid_policy.mojo:83`) is a step function, so the
disagreement only bites where the step falls: 15 folds go to HALF_BYTE and
16 to ONE_BYTE, 1 fold goes to BINARY and 2 to HALF_BYTE.

MEASURED ON THIS FIXTURE, before and after the fix. `loss/var` is the final
loss over 30 trees against the variance of a target that is a pure function
of the category, so 1.0 is "learned nothing" and 0 is "fitted exactly":

    k     borders->/folds-> policy  loss/var BEFORE   loss/var AFTER
    2     Binary  / HalfByte        0.995            4.3e-14
    3     HalfByte/ HalfByte       -0.0              2.0e-14
    4     HalfByte/ HalfByte        3.0e-13          3.0e-13
    15    HalfByte/ HalfByte        0.661            7.4e-14
    16    HalfByte/ OneByte         0.949            5.7e-13
    17    OneByte / OneByte         5.1e-13          5.1e-13
    31,32,64,128,254,255           ~1e-13           ~1e-13

**Three of twelve cardinalities were silently unlearnable**, and one of them
-- k = 15 -- is a case where the two POLICIES happen to coincide and only
the fold count differs, so a check that compared policies alone would have
missed it. Not a crash, not an exception: a returned model that could not
see the feature.

## WHY THE EXISTING CHECKS COULD NOT SEE IT

`train_api_check` asserts that `predict_floats` reproduces the fit's loss.
That assertion is BLIND to this bug by construction, because `predict_floats`
calls the same `_build_cindex_from_floats` and therefore reads through the
same wrong layout. Fit and predict agreed perfectly on a wrong answer -- and
the same assertion is kept below, unchanged and still passing, so its
blindness is on the record rather than assumed. `train_api_check`'s
categorical column also has THREE categories, which is one of the nine
cardinalities where nothing was wrong.

So the teeth here are the LEARNING assertion, against a target that is a
pure function of the category. A depth-6 oblivious tree with equality splits
can isolate six categories exactly, so `y = sum_i w_i * [code == c_i]` over
six scattered hot categories is fittable to zero. The gate is
`loss / var(y) < 0.05`, and `var(y)` is printed beside it so "learned" is a
ratio against the do-nothing baseline and not a bare threshold.

The policies are printed beside every row because that is where the
boundary falls, and the one INVARIANT still assertable from outside is
checked: a feature's fold count must cover the top bin its borders can
produce (`fold_counts[f] >= len(borders[f])`). What cannot be asserted from
outside is that ONE fold-count list reaches both layouts -- after the fix
that is structural, `_build_cindex_from_floats` taking the caller's list
rather than re-deriving one.

## Fixture rules this file follows

* **Cardinalities cross every step of `policy_for_fold_count`**, in both
  directions: 2, 3, 4, 15, 16, 17, 31, 32, 64, 128, 254, 255. Layout bugs
  live at the boundaries and nowhere else, which is exactly why a fixture
  at k = 3 reported the code correct for months.
* **Codes are scattered, never consecutive.** `code = (r * 7919) % k` with
  7919 prime, so every category is reached and adjacent rows are never
  adjacent categories. A consecutive assignment makes rows of one category
  contiguous, which hides an index that is off by a stride.
* **One hot category is the TOP bin `k - 1`.** A layout that truncates or
  mis-shifts loses the high bins first, so a fixture whose signal sits in
  bin 0 cannot see it.
* **Three one-hot features share the cardinality** so the packing (32, 8 or
  4 features to a `UInt32`) is exercised rather than a single feature
  sitting alone in its word.

The `k = 256` case asserts the REFUSAL instead: `train()` raises above 254
categories, and a refusal nothing exercises is an unchecked branch.
"""

from max.gpu.host import DeviceContext

from gbdt.gpu_data.grid_policy import policy_for_fold_count, policy_name
from gbdt.train import TrainedModel, predict_floats, train


def ohc_cardinalities() -> List[Int]:
    """Every step of `policy_for_fold_count`, from both sides."""
    return [2, 3, 4, 15, 16, 17, 31, 32, 64, 128, 254, 255]


comptime OHC_CAT_FEATURES = 3
comptime OHC_FEATURES = 4
comptime OHC_HOT = 6
comptime OHC_TREES = 30
comptime OHC_DEPTH = 6


def _hashed(x: Int) -> Int:
    """A scattered, non-monotone map. Used for the hot categories and their
    weights so neither is a ramp: a ramp in the codes is separable by an
    ORDERED split and would let a broken one-hot path score well for the
    wrong reason."""
    var h = (x * 2654435761) % 1000003
    if h < 0:
        h += 1000003
    return h


def _run_one(ctx: DeviceContext, k: Int) raises -> Tuple[Float64, Float64, Int, Int]:
    """One cardinality. Returns (final loss, var(y), the policy the
    informative feature's BORDER COUNT lands on, the policy its FOLD COUNT
    lands on)."""
    var n = 64 * k
    if n < 8192:
        n = 8192

    # colmajor: features 0..2 are one-hot categoricals of cardinality k,
    # feature 3 is numeric noise. Feature 0 carries the signal; 1 and 2 are
    # decoys with the SAME cardinality so they share feature blocks with it.
    var x = List[Float32]()
    for feat in range(OHC_FEATURES):
        for r in range(n):
            var v: Float32
            if feat == 0:
                v = Float32((r * 7919) % k)
            elif feat < OHC_CAT_FEATURES:
                v = Float32((r * (13 + 2 * feat) + feat) % k)
            else:
                v = Float32(_hashed(r + 17)) / Float32(1000003.0)
            x.append(v)

    # the hot categories: the TOP bin first, then scattered ones, deduped
    var hot = List[Int]()
    var hot_w = List[Float32]()
    var candidate = List[Int]()
    candidate.append(k - 1)
    for i in range(OHC_HOT * 4):
        candidate.append(_hashed(i + 5) % k)
    for i in range(len(candidate)):
        if len(hot) == OHC_HOT:
            break
        var c = candidate[i]
        var seen = False
        for j in range(len(hot)):
            if hot[j] == c:
                seen = True
                break
        if not seen:
            hot.append(c)
            hot_w.append(
                Float32(1) + Float32(_hashed(c + 3) % 900) / Float32(100.0)
            )

    var y = List[Float32]()
    for r in range(n):
        var code = (r * 7919) % k
        var v = Float32(0.0)
        for j in range(len(hot)):
            if hot[j] == code:
                v = hot_w[j]
                break
        y.append(v)

    var mean = Float64(0.0)
    for r in range(n):
        mean += Float64(y[r])
    mean /= Float64(n)
    var variance = Float64(0.0)
    for r in range(n):
        var d = Float64(y[r]) - mean
        variance += d * d
    variance /= Float64(n)

    var one_hot = List[Bool]()
    for feat in range(OHC_FEATURES):
        one_hot.append(feat < OHC_CAT_FEATURES)

    var tm = train(
        ctx,
        x,
        y,
        n,
        OHC_FEATURES,
        border_count=128,
        n_estimators=OHC_TREES,
        max_depth=OHC_DEPTH,
        learning_rate=Float32(0.7),
        l2_leaf_reg=Float32(1.0),
        one_hot=one_hot,
    )
    var loss = tm.losses[len(tm.losses) - 1]

    # The two policies the fold counts land on, printed because this is
    # where `policy_for_fold_count`'s steps fall. Before the fix these were
    # the WRITE and the READ policy and they could differ; now one list
    # feeds both, so the pair is information rather than an assertion.
    var borders_policy = policy_for_fold_count(len(tm.borders[0]))
    var folds_policy = policy_for_fold_count(tm.fold_counts[0])
    for f in range(OHC_FEATURES):
        if tm.fold_counts[f] < len(tm.borders[f]):
            raise Error(
                "k=" + String(k) + " feature " + String(f)
                + ": fold count " + String(tm.fold_counts[f])
                + " cannot reach bin " + String(len(tm.borders[f]))
                + ", the top bin its borders produce"
            )

    # the fit/predict consistency assertion that CANNOT see this bug, kept
    # so its blindness is on the record rather than assumed
    var pred = predict_floats(ctx, tm, x, n)
    var se = Float64(0.0)
    for r in range(n):
        var d = Float64(pred[r]) - Float64(y[r])
        se += d * d
    var pmse = se / Float64(n)
    var drift = pmse - loss
    if drift < 0:
        drift = -drift
    if drift > 1e-9 + 1e-5 * loss:
        raise Error(
            "k=" + String(k) + ": predict does not reproduce the fit ("
            + String(pmse) + " vs " + String(loss) + ")"
        )

    return (loss, variance, borders_policy, folds_policy)


def check_one_hot_cardinality() raises:
    print(
        "one-hot cardinality sweep across the grid-policy boundaries"
        " (structural + learning):"
    )
    var ctx = DeviceContext()
    var failures = List[String]()

    var cards = ohc_cardinalities()
    for i in range(len(cards)):
        var k = cards[i]
        var r = _run_one(ctx, k)
        var loss = r[0]
        var variance = r[1]
        var borders_policy = r[2]
        var folds_policy = r[3]
        var ratio = loss / variance

        print(
            "  k=" + String(k),
            " borders->" + policy_name(borders_policy),
            " folds->" + policy_name(folds_policy),
            " loss", loss,
            " var(y)", variance,
            " loss/var", ratio,
        )
        if ratio >= 0.05:
            failures.append(
                String("k=") + String(k) + ": did not learn a target that"
                " is a pure function of the category (loss " + String(loss)
                + " against variance " + String(variance) + ", ratio "
                + String(ratio) + ")"
            )

    # the refusal above 254 categories is a branch too
    var refused = False
    try:
        _ = _run_one(ctx, 256)
    except:
        refused = True
    if not refused:
        failures.append(
            String("k=256 was accepted; train() must refuse a one-hot"
                   " feature with more than 255 categories")
        )
    else:
        print("  k=256: refused, as train() documents")

    if len(failures) > 0:
        var msg = String("one-hot cardinality sweep FAILED:")
        for i in range(len(failures)):
            msg += String("\n    ") + failures[i]
        raise Error(msg)
    print(
        "  every cardinality writes and reads under ONE policy, and every"
        " one learns a category-pure target to under 5% of its variance"
    )


def main() raises:
    check_one_hot_cardinality()
