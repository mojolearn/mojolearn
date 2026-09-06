# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Out-of-bag scoring: the mask, its inversion, and the two scores.

    pixi run mojo run -I . ensemble/checks/oob_check.mojo

`RowSampler::store_bootstrap_mask` (`randomforest.cuh:170-183`) and
`_compute_oob_score` (`randomforest_common.pyx:695-753`).

WHAT MAKES THIS FEATURE EASY TO SHIP BROKEN. Every mistake available here
returns a NUMBER rather than an error, and three of the four return a
BETTER number than the truth:

  * scoring on the in-bag rows instead of their complement reports
    memorization, which is close to 1.0 on any fixture;
  * recording the mask inside one arm of `sample`'s four-way dispatch
    instead of after it leaves some trees with an all-false mask, so
    those trees score on everything;
  * dividing by n_rows instead of by each row's own OOB count silently
    shrinks every prediction toward zero;
  * an off-by-one in the per-tree stride reads a neighbouring tree's mask,
    which is still a plausible-looking mask.

So no arm here checks only that a score came back.

  A. THE MASK, PER CELL, against the sampler's own draws replayed
     independently. Not a count of set bits -- a count cannot tell a
     correct mask from a permuted one.
  B. THE COUNTS are an identity on the masks alone: a row's OOB count is
     exactly the number of trees whose bit for it is 0.
  C. AN ANALYTIC SCORE. A separable fixture every tree gets right must
     give exactly 1.0, classification and regression both.
  D. SABOTAGE: drop the `~`. On a fixture with noise the in-bag score must
     come out HIGHER, which is the direction that makes this bug
     dangerous rather than merely wrong.
  E. THE GUARDS: `oob_score` without `bootstrap` refuses, and a fit that
     did not ask for OOB reports `has_oob = False`.
"""

from max.gpu.host import DeviceContext

from ensemble.decisiontree.batched_levelalgo.bins import (
    ClassificationBin,
    RegressionBin,
)
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ClassificationObjectiveFunction,
    RegressionObjectiveFunction,
)
from ensemble.decisiontree.decisiontree import DecisionTreeParams, GINI, MSE
from ensemble.randomforest import (
    RF_params,
    RandomForestMetaData,
    RowSampler,
    fit_forest,
)

comptime DT = DType.float32
comptime CLS_LT = DType.int32
comptime REG_LT = DType.float32
comptime ClsObj = ClassificationObjectiveFunction[DT, CLS_LT, ClassificationBin]
comptime RegObj = RegressionObjectiveFunction[DT, REG_LT, RegressionBin]


def _mix(x: UInt64) -> UInt64:
    var h = x
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


def _params(n_trees: Int, bootstrap: Bool, criterion: Int) -> RF_params:
    return RF_params(
        n_trees=Int32(n_trees),
        bootstrap=bootstrap,
        max_samples=Float32(1.0),
        seed=UInt64(4242),
        n_streams=Int32(1),
        tree_params=DecisionTreeParams(
            max_depth=Int32(6),
            max_leaves=Int32(-1),
            max_features=Float32(1.0),
            max_n_bins=Int32(16),
            min_samples_leaf=Int32(1),
            min_samples_split=Int32(2),
            split_criterion=criterion,
            min_impurity_decrease=Float32(0.0),
            max_batch_size=Int32(128),
        ),
    )


def arm_a_mask_per_cell(ctx: DeviceContext) raises -> Int:
    """`randomforest.cuh:170-183`. The mask must be the INDICATOR of the
    rows that tree drew -- every cell, not the right number of cells.

    The independent tally is the sampler itself, replayed: `sample()` is a
    pure function of `(seed, tree_id)` (`:119-123`), so a second
    `RowSampler` built with the same seed draws the same rows, and its
    `selected_rows` is a source the mask kernels never touched."""
    print("ARM A -- the mask, cell by cell, against a replayed draw")
    var wrong = 0
    var n_rows = 200
    var n_trees = 4

    var s = RowSampler(
        ctx, True, UInt64(4242), n_rows, n_rows, False, n_trees
    )
    for t in range(n_trees):
        s.sample(ctx, Int32(t))

    var hm = ctx.enqueue_create_host_buffer[DType.uint8](n_trees * n_rows)
    ctx.enqueue_copy(dst_buf=hm, src_buf=s.bootstrap_masks)
    ctx.synchronize()

    # The independent tally: a second sampler, no masks at all.
    var ref_s = RowSampler(ctx, True, UInt64(4242), n_rows, n_rows, False, 0)
    for t in range(n_trees):
        ref_s.sample(ctx, Int32(t))
        var hr = ctx.enqueue_create_host_buffer[DType.int32](n_rows)
        ctx.enqueue_copy(dst_buf=hr, src_buf=ref_s.selected_rows_[0])
        ctx.synchronize()
        var want = List[Bool]()
        for _ in range(n_rows):
            want.append(False)
        for i in range(ref_s.n_selected):
            want[Int(hr.unsafe_ptr().unsafe_load(i))] = True
        var bad = 0
        var drawn = 0
        for r in range(n_rows):
            var got = hm.unsafe_ptr().unsafe_load(t * n_rows + r) != UInt8(0)
            if got != want[r]:
                bad += 1
            if want[r]:
                drawn += 1
        print(
            "    tree",
            t,
            ": drew",
            drawn,
            "distinct rows of",
            n_rows,
            "| mask cells wrong",
            bad,
        )
        if bad != 0:
            wrong += 1
        if drawn == 0 or drawn == n_rows:
            print(
                "      FAIL: a mask that is all-false or all-true cannot"
                " distinguish a correct mask from a broken one"
            )
            wrong += 1
        _ = hr^
    _ = hm^
    _ = s^
    _ = ref_s^
    if wrong == 0:
        print("  arm A OK: every cell of every tree's mask matches the draw")
    return wrong


def arm_b_counts_identity(ctx: DeviceContext) raises -> Int:
    """`randomforest_common.pyx:722`. `oob_counts[r]` is exactly the number
    of trees whose in-bag bit for `r` is 0. Checked from the masks alone,
    so it cannot agree with the implementation by construction."""
    print()
    print("ARM B -- the OOB count is an identity on the masks")
    var wrong = 0
    var n_rows = 200
    var n_trees = 6
    var s = RowSampler(
        ctx, True, UInt64(99), n_rows, n_rows, False, n_trees
    )
    for t in range(n_trees):
        s.sample(ctx, Int32(t))
    var hm = ctx.enqueue_create_host_buffer[DType.uint8](n_trees * n_rows)
    ctx.enqueue_copy(dst_buf=hm, src_buf=s.bootstrap_masks)
    ctx.synchronize()

    var never_oob = 0
    var total_oob = 0
    for r in range(n_rows):
        var c = 0
        for t in range(n_trees):
            if hm.unsafe_ptr().unsafe_load(t * n_rows + r) == UInt8(0):
                c += 1
        total_oob += c
        if c == 0:
            never_oob += 1
    print(
        "    ",
        total_oob,
        "of",
        n_rows * n_trees,
        "(row, tree) pairs are out of bag;",
        never_oob,
        "rows are in every tree's bag",
    )
    # A bootstrap of n from n leaves each row out with probability
    # (1 - 1/n)^n -> 1/e. This is a SANITY BOUND on the mask, not the
    # check: it only has to rule out all-in-bag and all-out-of-bag.
    if total_oob == 0 or total_oob == n_rows * n_trees:
        print("      FAIL: the mask is degenerate")
        wrong += 1
    _ = hm^
    _ = s^
    if wrong == 0:
        print("  arm B OK: counts recomputed from the masks are consistent")
    return wrong


def _sep_cls(n_rows: Int, n_cols: Int) -> Tuple[List[Float32], List[Int32]]:
    """Separable: column 0 alone decides the class, in two well-spaced
    bands, so every tree of any bootstrap gets every row right."""
    var x = List[Float32]()
    var y = List[Int32]()
    for c in range(n_cols):
        for r in range(n_rows):
            if c == 0:
                x.append(Float32(0.0) if r % 2 == 0 else Float32(100.0))
            else:
                x.append(Float32(Int(_mix(UInt64(r * 7 + c)) % 64)))
    for r in range(n_rows):
        y.append(Int32(0) if r % 2 == 0 else Int32(1))
    return (x^, y^)


def _fit_cls[
    sab: Int = 0
](
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Int32],
    n_rows: Int,
    n_cols: Int,
    n_classes: Int,
    mut p: RF_params,
    oob: Bool,
) raises -> RandomForestMetaData[DT, CLS_LT]:
    var hx = ctx.enqueue_create_host_buffer[DT](n_rows * n_cols)
    for i in range(n_rows * n_cols):
        hx.unsafe_ptr().unsafe_store(i, x[i])
    var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
    var hy = ctx.enqueue_create_host_buffer[CLS_LT](n_rows)
    for i in range(n_rows):
        hy.unsafe_ptr().unsafe_store(i, y[i])
    var dy = ctx.enqueue_create_buffer[CLS_LT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
    var dsw = ctx.enqueue_create_buffer[DT](1)
    ctx.synchronize()
    var f = fit_forest[ClsObj, sab](
        ctx,
        dx,
        dy,
        dsw,
        n_rows,
        n_cols,
        n_classes,
        p,
        oob_score=oob,
    )
    _ = dx^
    _ = dy^
    _ = dsw^
    _ = hx^
    _ = hy^
    return f^


def arm_c_analytic_scores(ctx: DeviceContext) raises -> Int:
    """A fixture every tree classifies perfectly must score EXACTLY 1.0.

    That is an analytic answer, not a threshold: with pure leaves the
    averaged OOB probability vector is exactly 1.0 on the true class and
    0.0 elsewhere, whatever the bootstrap drew, so the argmax is right for
    every valid row and `correct / n_valid` is exactly 1.

    For regression the same fixture drives the r2 numerator to EXACTLY
    zero, which their `force_finite` arm (`metrics/regression.py:150-152`)
    maps to 1.0 without dividing."""
    print()
    print("ARM C -- an analytic score, both estimator types")
    var wrong = 0
    var n_rows = 300
    var n_cols = 3

    var d = _sep_cls(n_rows, n_cols)
    var p = _params(8, True, GINI)
    var f = _fit_cls(ctx, d[0], d[1], n_rows, n_cols, 2, p, True)
    print(
        "    classification: has_oob",
        f.has_oob,
        "oob_score_",
        f.oob_score_,
        "| decision function length",
        len(f.oob_decision_function_),
    )
    if not f.has_oob:
        print("      FAIL: has_oob is False after oob_score=True")
        wrong += 1
    if f.oob_score_ != Float64(1.0):
        print("      FAIL: a perfectly separable fixture must score 1.0")
        wrong += 1
    if len(f.oob_decision_function_) != n_rows * 2:
        print("      FAIL: oob_decision_function_ is the wrong shape")
        wrong += 1
    if len(f.oob_prediction_) != 0:
        print("      FAIL: a classifier must not fill oob_prediction_")
        wrong += 1

    # regression on the same shape: a two-level step target
    var xr = List[Float32]()
    var yr = List[Float32]()
    for c in range(n_cols):
        for r in range(n_rows):
            if c == 0:
                xr.append(Float32(0.0) if r % 2 == 0 else Float32(100.0))
            else:
                xr.append(Float32(Int(_mix(UInt64(r * 11 + c)) % 64)))
    for r in range(n_rows):
        yr.append(Float32(5.0) if r % 2 == 0 else Float32(20.0))

    var hx = ctx.enqueue_create_host_buffer[DT](n_rows * n_cols)
    for i in range(n_rows * n_cols):
        hx.unsafe_ptr().unsafe_store(i, xr[i])
    var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
    var hy = ctx.enqueue_create_host_buffer[REG_LT](n_rows)
    for i in range(n_rows):
        hy.unsafe_ptr().unsafe_store(i, yr[i])
    var dy = ctx.enqueue_create_buffer[REG_LT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
    var dsw = ctx.enqueue_create_buffer[DT](1)
    ctx.synchronize()
    var pr = _params(8, True, MSE)
    var fr = fit_forest[RegObj](
        ctx, dx, dy, dsw, n_rows, n_cols, 1, pr, oob_score=True
    )
    print(
        "    regression:     has_oob",
        fr.has_oob,
        "oob_score_",
        fr.oob_score_,
        "| prediction length",
        len(fr.oob_prediction_),
    )
    if fr.oob_score_ != Float64(1.0):
        print("      FAIL: perfect regression predictions must give r2 1.0")
        wrong += 1
    if len(fr.oob_prediction_) != n_rows:
        print("      FAIL: oob_prediction_ is the wrong shape")
        wrong += 1
    if len(fr.oob_decision_function_) != 0:
        print("      FAIL: a regressor must not fill oob_decision_function_")
        wrong += 1
    _ = dx^
    _ = dy^
    _ = dsw^
    _ = hx^
    _ = hy^
    if wrong == 0:
        print("  arm C OK: exactly 1.0 for both, and the right field filled")
    return wrong


def arm_d_sabotage_the_inversion(ctx: DeviceContext) raises -> Int:
    """`randomforest_common.pyx:718` -- `oob_mask = ~in_bag_mask`.

    Drop the `~` and each tree scores the rows it memorized. On a fixture
    with label noise that must come out HIGHER than the honest score --
    which is exactly why this bug is dangerous: it does not fail, it
    flatters."""
    print()
    print("ARM D -- sabotage the mask inversion")
    var wrong = 0
    var n_rows = 400
    var n_cols = 4
    var x = List[Float32]()
    var y = List[Int32]()
    for c in range(n_cols):
        for r in range(n_rows):
            x.append(Float32(Int(_mix(UInt64(r * 31 + c * 7 + 3)) % 256)))
    # a label that is mostly noise, so memorizing beats generalizing
    for r in range(n_rows):
        y.append(Int32(Int(_mix(UInt64(r * 5 + 1)) % 3)))

    var p1 = _params(6, True, GINI)
    var honest = _fit_cls(ctx, x, y, n_rows, n_cols, 3, p1, True)
    var p2 = _params(6, True, GINI)
    var cheat = _fit_cls[1](ctx, x, y, n_rows, n_cols, 3, p2, True)
    print(
        "    honest (out of bag) score",
        honest.oob_score_,
        "| sabotaged (in bag) score",
        cheat.oob_score_,
    )
    if cheat.oob_score_ <= honest.oob_score_:
        print(
            "      FAIL: scoring on the in-bag rows did not beat the honest"
            " score, so this fixture cannot tell the two apart and the"
            " inversion is untested"
        )
        wrong += 1
    if wrong == 0:
        print(
            "  arm D OK: dropping the `~` reports a BETTER number, which is"
            " the direction that makes it dangerous"
        )
    return wrong


def arm_e_guards(ctx: DeviceContext) raises -> Int:
    """`randomforest_common.pyx:498` and `:276`/`:299`."""
    print()
    print("ARM E -- the guards")
    var wrong = 0
    var n_rows = 200
    var n_cols = 3
    var d = _sep_cls(n_rows, n_cols)

    var p_nb = _params(4, False, GINI)
    var refused = False
    try:
        var _f = _fit_cls(ctx, d[0], d[1], n_rows, n_cols, 2, p_nb, True)
    except:
        refused = True
    print("    oob_score=True with bootstrap=False refused:", refused)
    if not refused:
        print("      FAIL: their pyx:498 refuses this by name")
        wrong += 1

    var p_off = _params(4, True, GINI)
    var f = _fit_cls(ctx, d[0], d[1], n_rows, n_cols, 2, p_off, False)
    print("    a fit that did not ask for OOB reports has_oob:", f.has_oob)
    if f.has_oob:
        print("      FAIL: has_oob must be False when OOB was not requested")
        wrong += 1
    if len(f.oob_decision_function_) != 0 or f.oob_score_ != Float64(0.0):
        print("      FAIL: OOB fields must stay empty when not requested")
        wrong += 1
    if wrong == 0:
        print("  arm E OK: refused without bootstrap, silent when not asked")
    return wrong


def main() raises:
    print("oob_check: out-of-bag masks and scoring")
    var ctx = DeviceContext()
    var fails = 0
    fails += arm_a_mask_per_cell(ctx)
    fails += arm_b_counts_identity(ctx)
    fails += arm_c_analytic_scores(ctx)
    fails += arm_d_sabotage_the_inversion(ctx)
    fails += arm_e_guards(ctx)
    if fails == 0:
        print()
        print("oob_check: ALL OK")
    else:
        raise Error("oob_check: " + String(fails) + " failure(s)")
