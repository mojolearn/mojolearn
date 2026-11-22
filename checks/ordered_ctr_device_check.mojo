# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`train(boosting_type='Ordered')` with categorical columns that build CTRs.

    pixi run check-native-gpu ordered_ctr_device_check

An Ordered fit gives every permutation its own CTR columns (a row's CTR
reads the rows before it in that permutation) and its own compressed
index; `fit_ordered` searches a tree on the learn permutation's index and
applies it to each permutation's folds through that permutation's index.

THE FIXTURE CAN FAIL. Two categorical columns of twelve categories, every
pair equally often, and a 0/1 target that is a function of column `a`
alone (1 for a third of its categories). Column `b` carries nothing. A fit
whose CTR columns reach the trees fits the target; a fit that read the
wrong permutation's index for a fold, or no CTR column at all, does not.

Checked:

  * the Ordered fit ends below a quarter of the loss of its first tree
    and predicts the learn rows as well as that (the final tables and
    the estimation permutation's index);
  * the model text round-trips to the same text and the same predictions;
  * one permutation (`permutation_count=1`: learn and estimation are the
    same order, one index) fits too;
  * with the learn rows as the eval set the held-out loss follows;
  * feature combinations (complexity 2) fit under Ordered and round-trip;
  * what is refused by name: an estimation permutation that is not the
    last.
"""

from max.gpu.host import DeviceContext

from gbdt.models.model_text import load_model_text, model_text
from gbdt.options.catboost_options import TCatFeatureParams
from gbdt.train import TrainedModel, predict_floats, train

comptime OC_CATS = 12
comptime OC_REPEATS = 8
comptime OC_ROWS = OC_CATS * OC_CATS * OC_REPEATS
comptime OC_TREES = 30
comptime OC_DEPTH = 3


def _need(ok: Bool, what: String) raises:
    if not ok:
        raise Error("ordered CTR check: " + what)


def _fixture() -> Tuple[List[Float32], List[Float32]]:
    """Column-major x (a, then b) and y."""
    var x = List[Float32](capacity=2 * OC_ROWS)
    var y = List[Float32](capacity=OC_ROWS)
    for r in range(OC_ROWS):
        x.append(Float32((r * 7) % OC_CATS))
    for r in range(OC_ROWS):
        x.append(Float32((r // OC_CATS) % OC_CATS))
    for r in range(OC_ROWS):
        var a = (r * 7) % OC_CATS
        y.append(Float32(1.0) if a % 3 == 0 else Float32(0.0))
    return (x^, y^)


def _mse(preds: List[Float32], y: List[Float32]) -> Float64:
    var total = Float64(0.0)
    for r in range(len(y)):
        var d = Float64(preds[r]) - Float64(y[r])
        total += d * d
    return total / Float64(len(y))


def _fit(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Float32],
    permutations: Int,
    estimation_permutation: Int,
    complexity: Int,
    with_eval: Bool,
) raises -> TrainedModel:
    var p = TCatFeatureParams.default()
    p.max_tensor_complexity = complexity
    var params = List[TCatFeatureParams]()
    params.append(p^)
    var flags = List[Bool](length=2, fill=True)
    var eval_x = List[Float32]()
    var eval_y = List[Float32]()
    if with_eval:
        eval_x = x.copy()
        eval_y = y.copy()
    return train(
        ctx, x, y, OC_ROWS, 2,
        border_count=32,
        n_estimators=OC_TREES,
        max_depth=OC_DEPTH,
        learning_rate=Float32(0.3),
        l2_leaf_reg=Float32(1.0),
        cat_features=flags,
        cat_feature_params=params,
        ctr_estimation_permutation_id=estimation_permutation,
        permutation_count=permutations,
        eval_x_colmajor=eval_x,
        eval_y=eval_y,
        use_best_model=0,
        boosting_type=String("Ordered"),
    )


def _expect_refusal(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Float32],
    estimation_permutation: Int,
    complexity: Int,
    needle: String,
    what: String,
) raises:
    var message = String("")
    try:
        _ = _fit(ctx, x, y, 4, estimation_permutation, complexity, False)
    except error:
        message = String(error)
    _need(message != "", what + " was not refused")
    _need(
        message.find(needle) >= 0,
        what + " was refused in other words: " + message,
    )


def check_ordered_ctr_device() raises:
    print("train(boosting_type='Ordered') with CTR categorical columns:")
    var ctx = DeviceContext()
    var data = _fixture()
    var x = data[0].copy()
    var y = data[1].copy()

    var fit4 = _fit(ctx, x, y, 4, -1, 1, False)
    _need(len(fit4.losses) == OC_TREES, "one learn loss per tree")
    var first = fit4.losses[0]
    var last = fit4.losses[len(fit4.losses) - 1]
    print("  four permutations, loss after tree 1:", first, " after the last:", last)
    _need(last == last, "the final loss is NaN")
    _need(
        last < 0.25 * first,
        "the CTR columns did not carry the signal (loss " + String(last)
        + " after " + String(first) + ")",
    )
    _need(fit4.ctr_column_count > 0, "the model holds no CTR column")
    var preds = predict_floats(ctx, fit4, x, OC_ROWS)
    var pred_mse = _mse(preds, y)
    print("  predict on the learn rows, mse:", pred_mse)
    _need(
        pred_mse < 0.25 * first,
        "predict does not reproduce the fit (mse " + String(pred_mse) + ")",
    )
    var text = model_text(fit4)
    var back = load_model_text(text)
    _need(model_text(back) == text, "the model text changed on a round trip")
    var again = predict_floats(ctx, back, x, OC_ROWS)
    for r in range(OC_ROWS):
        if again[r] != preds[r]:
            raise Error(
                "ordered CTR check: row " + String(r)
                + " predicts differently after the text round trip"
            )

    var fit1 = _fit(ctx, x, y, 1, -1, 1, False)
    var last1 = fit1.losses[len(fit1.losses) - 1]
    print("  one permutation, final loss:", last1)
    _need(
        last1 < 0.25 * fit1.losses[0],
        "the one-permutation fit did not learn (loss " + String(last1) + ")",
    )

    var held = _fit(ctx, x, y, 4, -1, 1, True)
    _need(len(held.test_losses) == len(held.losses), "one held-out loss per tree")
    var held_last = held.test_losses[len(held.test_losses) - 1]
    print("  held-out loss on the learn rows:", held_last)
    _need(
        held_last < 0.25 * held.test_losses[0],
        "the held-out rows did not receive the CTR columns (loss "
        + String(held_last) + ")",
    )

    _expect_refusal(
        ctx, x, y, 0, 1, String("estimates on its last permutation"),
        "an estimation permutation that is not the last",
    )
    # Feature combinations under Ordered fit since 3f6d5496f4 (cat-ctr step 4
    # unit 3, the fold searcher's level driver); this check still expected
    # the refusal that commit removed (lane sr-gbdt, 2026-11-22).
    var combo = _fit(ctx, x, y, 4, -1, 2, False)
    _need(len(combo.losses) == OC_TREES, "one learn loss per tree (combinations)")
    var combo_last = combo.losses[len(combo.losses) - 1]
    print("  combinations (complexity 2), final loss:", combo_last)
    _need(combo_last == combo_last, "the combinations fit's final loss is NaN")
    _need(
        combo_last < 0.25 * combo.losses[0],
        "the combinations fit did not learn (loss " + String(combo_last) + ")",
    )
    var combo_text = model_text(combo)
    _need(model_text(load_model_text(combo_text)) == combo_text,
          "the combinations model text changed on a round trip")
    print("  Ordered CTRs fit (combinations too), predict, eval, round-trip and refuse by name")


def run_checks() raises:
    check_ordered_ctr_device()


# Device column only: the suite needs a GPU, so the CPU build never
# instantiates it.
from checks.native_tests.runner import Check, Column, Report, CPU_COLUMN_BUILD


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("ordered_ctr_device_check", cpu=False, gpu=True, serial="", run=run_cases)
