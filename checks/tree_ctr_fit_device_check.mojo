# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`train(max_ctr_complexity=2)`: feature combinations through the real fit.

    pixi run check-native-gpu tree_ctr_fit_device_check

THE FIXTURE CAN FAIL. Two categorical columns of ten categories each, every
pair equally often, and a target that is a function of the PAIR:

    y = 1 when a == 0 or (2a + 3b) % 5 == 0, else 0

Column `a` alone separates only its category 0; over the other nine
categories the target is 1 for two `b` in ten whatever `a` is, so a fit
limited to single-column CTRs keeps a loss near 0.9 * 0.2 * 0.8 = 0.144.
The combination a x b determines the target, so a fit that mints it (the
first split puts one column into the tree, the next level offers its
combination with the other) drives the loss toward zero.

Checked:

  * the complexity-2 fit ends below a quarter of the complexity-1 fit's
    loss on the same data and settings;
  * its model holds a tree-CTR section (maps, tables, one column each after
    the base columns) and every split names a model column;
  * `predict_floats` on the learn rows (the map lookup, the table and the
    borders, no slot) is as good as the fit says;
  * the model text round-trips: the same text again, the same predictions
    bit for bit;
  * with the learn rows as the eval set, the held-out loss follows the
    learn loss (the persist slots of the test index);
  * the FeatureFreq-only configuration (one permutation, no Borders CTR)
    fits, predicts and round-trips.
"""

from max.gpu.host import DeviceContext

from gbdt.models.model_text import load_model_text, model_text
from gbdt.options.catboost_options import TCatFeatureParams
from gbdt.train import TrainedModel, predict_floats, train

comptime FIT_CATS = 10
comptime FIT_REPEATS = 60
comptime FIT_ROWS = FIT_CATS * FIT_CATS * FIT_REPEATS
comptime FIT_TREES = 20
comptime FIT_DEPTH = 4


def _need(ok: Bool, what: String) raises:
    if not ok:
        raise Error("tree CTR fit check: " + what)


def _fixture() -> Tuple[List[Float32], List[Float32]]:
    """Column-major x (a, then b) and y."""
    var x = List[Float32](capacity=2 * FIT_ROWS)
    var y = List[Float32](capacity=FIT_ROWS)
    for r in range(FIT_ROWS):
        x.append(Float32(r % FIT_CATS))
    for r in range(FIT_ROWS):
        x.append(Float32((r // FIT_CATS) % FIT_CATS))
    for r in range(FIT_ROWS):
        var a = r % FIT_CATS
        var b = (r // FIT_CATS) % FIT_CATS
        var one = a == 0 or (2 * a + 3 * b) % 5 == 0
        y.append(Float32(1.0) if one else Float32(0.0))
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
    complexity: Int,
    freq_only: Bool,
    with_eval: Bool,
) raises -> TrainedModel:
    var p = (
        TCatFeatureParams.feature_freq_only() if freq_only
        else TCatFeatureParams.default()
    )
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
        ctx, x, y, FIT_ROWS, 2,
        border_count=32,
        n_estimators=FIT_TREES,
        max_depth=FIT_DEPTH,
        learning_rate=Float32(0.5),
        l2_leaf_reg=Float32(1.0),
        cat_features=flags,
        cat_feature_params=params,
        eval_x_colmajor=eval_x,
        eval_y=eval_y,
        use_best_model=0,
    )


def _round_trip(
    ctx: DeviceContext, tm: TrainedModel, x: List[Float32], what: String
) raises:
    var text = model_text(tm)
    var back = load_model_text(text)
    _need(model_text(back) == text, what + ": the model text changed on a round trip")
    var p0 = predict_floats(ctx, tm, x, FIT_ROWS)
    var p1 = predict_floats(ctx, back, x, FIT_ROWS)
    _need(len(p0) == FIT_ROWS and len(p1) == FIT_ROWS, what + ": prediction count")
    for r in range(FIT_ROWS):
        if p0[r] != p1[r]:
            raise Error(
                "tree CTR fit check: " + what + ": row " + String(r)
                + " predicts differently after the text round trip"
            )


def _section_is_whole(tm: TrainedModel, what: String) raises:
    var n_columns = len(tm.fold_counts)
    var n_tree = len(tm.combination_ctrs.tables)
    _need(n_tree > 0, what + ": no combination was chosen")
    _need(len(tm.combination_ctrs.maps) > 0, what + ": no combination map")
    _need(len(tm.borders) == n_columns, what + ": one grid per model column")
    for t in range(n_tree):
        _need(
            tm.combination_ctrs.tables[t].column == n_columns - n_tree + t,
            what + ": the tree-CTR tables do not fill the last columns",
        )
    var on_tree_ctr = 0
    for t in range(len(tm.model.weak_models)):
        for i in range(len(tm.model.weak_models[t].structure.splits)):
            var f = Int(tm.model.weak_models[t].structure.splits[i].feature_id)
            _need(f >= 0 and f < n_columns, what + ": a split names no model column")
            if f >= n_columns - n_tree:
                on_tree_ctr += 1
    _need(on_tree_ctr > 0, what + ": no split uses a combination")


def check_tree_ctr_fit_device() raises:
    print("train(max_ctr_complexity=2) end to end:")
    var ctx = DeviceContext()
    var data = _fixture()
    var x = data[0].copy()
    var y = data[1].copy()

    var single = _fit(ctx, x, y, 1, False, False)
    var single_loss = single.losses[len(single.losses) - 1]
    _need(single.combination_ctrs.is_empty(), "complexity 1 built a combination")

    var pair = _fit(ctx, x, y, 2, False, False)
    var pair_loss = pair.losses[len(pair.losses) - 1]
    print("  loss, complexity 1:", single_loss, " complexity 2:", pair_loss)
    _need(pair_loss == pair_loss, "the complexity-2 loss is NaN")
    _need(
        pair_loss < 0.25 * single_loss,
        "the combination did not carry the pair's signal (complexity 2 loss "
        + String(pair_loss) + " against " + String(single_loss) + ")",
    )
    _section_is_whole(pair, "default CTRs")
    var preds = predict_floats(ctx, pair, x, FIT_ROWS)
    var pred_mse = _mse(preds, y)
    print("  predict on the learn rows, mse:", pred_mse)
    _need(
        pred_mse < 0.25 * single_loss,
        "predict does not reproduce the fit (mse " + String(pred_mse) + ")",
    )
    _round_trip(ctx, pair, x, "default CTRs")

    var held = _fit(ctx, x, y, 2, False, True)
    _need(len(held.test_losses) == len(held.losses), "one held-out loss per tree")
    var held_loss = held.test_losses[len(held.test_losses) - 1]
    print("  held-out loss on the learn rows:", held_loss)
    _need(
        held_loss < 0.25 * single_loss,
        "the held-out rows did not receive the combination columns (loss "
        + String(held_loss) + ")",
    )

    var freq = _fit(ctx, x, y, 2, True, False)
    var freq_loss = freq.losses[len(freq.losses) - 1]
    _need(freq_loss == freq_loss, "the FeatureFreq-only loss is NaN")
    var freq_preds = predict_floats(ctx, freq, x, FIT_ROWS)
    _need(len(freq_preds) == FIT_ROWS, "FeatureFreq-only prediction count")
    if not freq.combination_ctrs.is_empty():
        _round_trip(ctx, freq, x, "FeatureFreq only")
    print("  feature combinations fit, predict, eval and round-trip")


def run_checks() raises:
    check_tree_ctr_fit_device()


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
    return Check("tree_ctr_fit_device_check", cpu=False, gpu=True, serial="", run=run_cases)
