# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The HELD-OUT arm of the Plain doc-parallel fit on the host, a second
spelling of `fit_with_test`'s test cursor, held-out curve, overfitting
detector and `ShrinkToBestIteration` (lane/close-no-cpu-path-gbdt,
2026-09-20).

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu`, a `DeviceContext` or a
module that defines a kernel. The only library import is the shared
overfitting detector (`gbdt/overfitting_detector/overfitting_detector.mojo`),
which is GPU-free host code the device fit itself runs on the host, and which
`gbdt/host/gbdt_oracle_ordered.mojo` and `gbdt/host/gbdt_oracle_pointwise.mojo`
already call for the same purpose.

WHY THIS FILE EXISTS. Before it, `gbdt_fit` refused `eval_set` BY NAME on
every Plain arm (`bindings/_mojolearn_gbdt_host.mojo`, `_refuse("eval_set")`),
so a CPU-only install could not check any early-stopping fit against a GPU
column. The Ordered arm (`gbdt_oracle_ordered.mojo`) and the pointwise
searcher's own arm (`gbdt_oracle_pointwise.mojo`) each grew a held-out cursor
of their own; this is the same four pieces factored out, for the Plain
SymmetricTree Logloss arm of `gbdt/host/gbdt_oracle.mojo::gbdt_host_fit`.

AN EVAL SET DOES NOT CHANGE THE FIT. On the Plain doc-parallel path the
learn cursor, the borders, the splits and the leaves are what they were
without a held-out set (`doc_parallel_boosting.mojo::fit_with_test`: the
test arm is read only by `_apply_last_tree_to_test`, `_test_loss` and the
detector, and only the querywise losses refuse it outright). So the model
text a CPU column produces for a fit WITH an eval set is the model text it
produces without one, and what this file adds to the comparison is the
held-out curve, `best_iteration_` and `stopped_early_`.

WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT

  1. `CreateCursors`' test seed (`doc_parallel_boosting.mojo:1531-1537`):
     the held-out cursor starts at 0, or at the model's own bias when
     `boost_from_average` resolved True -- the CB_ENSURE at their `:174-182`
     names TestDataProvider precisely because the seed reaches it.
  2. `_apply_last_tree_to_test` (`:272-330`), their `AddObliviousTree`:
     the LAST weak model only, one tree's `depth` level records and
     `2^depth` values, the per-row leaf walked from the test rows'
     compressed index (`gbdt_eval_test_bins`), then `cursor += values[bin]`
     (`gbdt_eval_add_tree`). THE STORED VALUES ALREADY CARRY THE LEARNING
     RATE, so nothing reapplies it; a depth-0 tree adds nothing.
  3. `_test_loss` (`:368-412`): the held-out `functionValue` through the
     SAME target kernel the learn loss uses, negated and divided by the
     held-out row count, so the two curves are the same quantity and the
     detector's comparison means something. The caller supplies it, because
     the loss kernel belongs to the arm's own oracle.
  4. `DetectOverfitting` (`overfitting_detector.h:25-34`): each held-out
     error into the shared detector, and `IsNeedStop()` breaks the boosting
     loop. `best_iteration` is then the DETECTOR's.
  5. `ShrinkToBestIteration` (`boosting_progress_tracker.h:113-125`, called
     from `gbdt/train.mojo:2241-2254` after the loop): a SECOND best-iteration
     tracker, fed only the iterations at or past `best_model_min_trees`, its
     strict `<` so the first of a tie wins, and the ensemble truncated to it
     (`gbdt_eval_shrink_point`). Their two trackers disagree by design, and
     it is this one the shrink reads.

THE NEGATIVE CONTROL. `-D MOJOLEARN_GBDT_EVAL_SABOTAGE=1`
(`GBDT_EVAL_HOST_SABOTAGE`) adds one ULP to every value this arm puts on the
held-out cursor. The MODEL is untouched, so the train column's model text and
the learn curve do not move and only the new held-out cells do -- which is the
point: the family's own arm (`-D MOJOLEARN_HOST_SABOTAGE=1`, the Newton
Hessian regularizer) moves the leaves and would read DIVERGENT on this arm
whether or not the held-out restatement is right, and cannot tell a broken
test cursor from a broken fit. This one can.

IDENTITY IS NOT CORRECTNESS. What a matching column shows is that this
spelling and the device's compute the same bits, not that either computes the
held-out curve a reader would call right.
"""
from std.sys.compile import is_defined
from std.memory import bitcast

from gbdt.overfitting_detector.overfitting_detector import (
    OD_NONE,
    OverfittingDetector,
    make_overfitting_detector,
)


#: This arm's own negative control (see THE NEGATIVE CONTROL above). It is
#: NOT the family's `MOJOLEARN_HOST_SABOTAGE`: that one moves the leaves, so
#: it cannot distinguish a wrong held-out cursor from a wrong fit.
comptime GBDT_EVAL_HOST_SABOTAGE = is_defined["MOJOLEARN_GBDT_EVAL_SABOTAGE"]()


@fieldwise_init
struct GbdtHostEval(Movable):
    """The held-out pool and the stopping options, as `gbdt_fit` resolves
    them. `n_rows == 0` means NO TEST SET and every other field is unread,
    the contract `TestArm` keeps (`doc_parallel_boosting.mojo:181-200`)."""

    var x_colmajor: List[Float32]
    var y: List[Float32]
    var n_rows: Int
    #: `OD_NONE`, `OD_ITER` or `OD_INC_TO_DEC`, already through their `Load`
    #: (`load_overfitting_detector_options`), as the Ordered arm does it.
    var od_type: Int
    var od_pvalue: Float64
    var od_wait: Int
    #: 1, 0 or -1 (unset); resolve -1 with `gbdt_eval_want_best_model`.
    var want_best_model: Int
    var best_model_min_trees: Int

    @staticmethod
    def none() -> GbdtHostEval:
        """No held-out set."""
        return GbdtHostEval(
            List[Float32](), List[Float32](), 0, OD_NONE, Float64(-1.0), -1,
            0, 1,
        )

    def has_test(self) -> Bool:
        return self.n_rows > 0


@fieldwise_init
struct GbdtHostEvalFit(Movable):
    """What the held-out arm reports beyond the model: their
    `FitResult.test_losses`, `best_iteration` and `stopped_early`."""

    var test_losses: List[Float64]
    var best_iteration: Int
    var stopped_early: Bool


def gbdt_eval_want_best_model(want: Int, y: List[Float32], n_rows: Int) -> Int:
    """`UpdateUseBestModel` (`options_helper.cpp:100-113`): unset is True
    when there IS a held-out set and its target is not constant, and their
    reason for the constant test is that a flat curve has no best iteration
    to shrink to. Restated here exactly as `_gbdt_fit_ordered_arm` restates
    it, so the two arms resolve the same flag from the same rows."""
    if want != -1:
        return want
    if n_rows <= 0:
        return 0
    for r in range(1, n_rows):
        if y[r] != y[0]:
            return 1
    return 0


def gbdt_eval_detector(eval: GbdtHostEval) raises -> OverfittingDetector:
    """`CreateOverfittingDetector(options, maxIsOptimal, hasTest)`
    (`overfitting_detector.cpp:205-207`). EVERY loss this implementation
    trains is minimized, so `maxIsOptimal` is False; a detector built with
    no held-out set is inert by construction (`:122-124`)."""
    return make_overfitting_detector(
        eval.od_type, False, eval.od_pvalue, eval.od_wait, eval.has_test()
    )


def gbdt_eval_test_bins(
    test_cindex: List[UInt32],
    n_eval: Int,
    split_features: List[Int],
    split_bins: List[Int],
    feat_offset: List[UInt32],
    feat_mask: List[UInt32],
    feat_shift: List[UInt32],
    feat_one_hot: List[Bool],
) -> List[Int]:
    """`_apply_last_tree_to_test`'s packed level records and its per-row
    leaf (`doc_parallel_boosting.mojo:306-330`, the walk
    `core/gbdt_host_predict.mojo:400-424` runs over a saved model): the
    masked feature word compared to the split's value, EQUAL on a one-hot
    column and GREATER on an ordered one, the level's bit OR-ed in."""
    var bins = List[Int](length=n_eval if n_eval > 0 else 1, fill=0)
    var depth = len(split_features)
    for r in range(n_eval):
        var leaf = 0
        for level in range(depth):
            var fid = split_features[level]
            var mask = feat_mask[fid] << feat_shift[fid]
            var value = UInt32(split_bins[level]) << feat_shift[fid]
            var feature_val = (
                test_cindex[Int(feat_offset[fid]) * n_eval + r] & mask
            )
            var goes_right: Bool
            if feat_one_hot[fid]:
                goes_right = feature_val == value
            else:
                goes_right = feature_val > value
            if goes_right:
                leaf += 1 << level
        bins[r] = leaf
    return bins^


@no_inline
def gbdt_eval_add_tree(
    mut cursor: List[Float32], bins: List[Int], values: List[Float32]
):
    """`compute_bins_and_add_kernel`'s add (`add_bin_values.mojo`):
    `cursor += values[bin]`, the STORED model values, which already carry
    the learning rate.

    `@no_inline` IS LOAD BEARING, for the reason
    `gbdt_oracle_ordered.mojo::_add_tree_values` records: inlined, the host
    compiler contracted `cursor + leaf * rate` into one fma, one rounding
    where the device kernel rounds the stored product first -- measured on
    the M4, 258 of 800 held-out rows one ulp apart after the first tree."""
    for r in range(len(bins)):
        var v = values[bins[r]]
        comptime if GBDT_EVAL_HOST_SABOTAGE:
            v = bitcast[DType.float32](bitcast[DType.uint32](v) + UInt32(1))
        cursor[r] = cursor[r] + v


def gbdt_eval_shrink_point(
    test_losses: List[Float64], want_best_model: Int, best_model_min_trees: Int
) -> Int:
    """`ShrinkToBestIteration`'s own tracker (`gbdt/train.mojo:2241-2254`,
    their `boosting_progress_tracker.h:113-125`): the tree count to keep, or
    -1 for no shrink.

    It is fed ONLY the iterations at or past `best_model_min_trees`, so its
    best iteration can differ from the detector's, and their strict `<`
    (`error_tracker.h:58-64`) means the FIRST of a tie wins and a plateau
    does not walk the cut rightwards."""
    if want_best_model != 1 or len(test_losses) == 0:
        return -1
    var min_trees_best = -1
    var min_trees_err = Float64(0.0)
    for i in range(len(test_losses)):
        if i + 1 < best_model_min_trees:
            continue
        if min_trees_best < 0 or test_losses[i] < min_trees_err:
            min_trees_err = test_losses[i]
            min_trees_best = i
    return min_trees_best + 1
