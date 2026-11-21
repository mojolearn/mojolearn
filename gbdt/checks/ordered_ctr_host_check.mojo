# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column of an Ordered fit with CTR categorical columns.

    pixi run check-native-gpu gbdt_ordered_ctr_host_check

`gbdt/host/gbdt_oracle_ordered.mojo::gbdt_ordered_host_fit` with categorical
flags is the host spelling of `train(boosting_type='Ordered')` with
categorical columns above `one_hot_max_size`: the CTR estimation orders are
the fold permutations, one compressed index per permutation, each tree
searched on its learn permutation's index and every permutation's leaves
read off its own index. Under `identical` the two give the same model, so
the check is the model text, byte for byte, and the loss curves, bit for
bit.

THE FIXTURE CAN FAIL. Two categorical columns of ten categories and one
numeric column; the 0/1 target depends on the first categorical column and
on the numeric one. Neither categorical column is one-hot (ten categories
are above `one_hot_max_size`), so each builds its Borders and FeatureFreq
columns and the Borders ones differ per permutation: a host fit that read
one index for every permutation, or took the Plain CTR orders, gives
another text.

Checked, host against device:

  * Logloss, four permutations: the text and the learn losses;
  * RMSE (a `bias` record under `boost_from_average`), four permutations;
  * Logloss with the learn rows as the eval set: the text, the held-out
    losses;
  * one permutation (one index, learn and estimation the same order);
  * a one-hot column (two categories) under Ordered: the same text and losses;
  * FEATURE COMBINATIONS (`max_ctr_complexity` 2) on a second fixture whose
    target is a function of the PAIR (a, b), so no single column explains
    it: four permutations (Logloss, with and without an eval set; RMSE)
    and one permutation. The device fit must have chosen a combination
    (else the fixture no longer exercises tree CTRs), the texts are the
    same bytes (the `combination` records included) and the loss curves
    the same bits. A level driver that kept the parent's histogram cells,
    left a won slot's stale column in the fold-order index, or took
    another permutation's ordered statistic gives another tree.

The suite needs a GPU for the device arm; the host arm runs on the calling
host. Serial by nature: pairs of fits, each pair compared whole.
"""

from max.gpu.host import DeviceContext

from gbdt.data.ordered_plan import (
    ORDERED_MIN_FOLD_SIZE,
    ordered_permutation_block_size,
)
from gbdt.host.gbdt_oracle import (
    GBDT_LOGLOSS_NEWTON_ITERATIONS,
    GbdtHostParams,
)
from gbdt.host.gbdt_oracle_losses import (
    GBDT_LEAF_NEWTON,
    GBDT_OBJ_LOGLOSS,
    GBDT_OBJ_RMSE,
    GbdtHostLoss,
)
from gbdt.host.gbdt_oracle_ordered import (
    GbdtOrderedHostEval,
    GbdtOrderedHostFit,
    GbdtOrderedHostOptions,
    gbdt_ordered_host_fit,
)
from gbdt.models.model_text import model_text
from gbdt.options.catboost_options import TCatFeatureParams
from gbdt.options.data_processing_options import NAN_MODE_MIN
from gbdt.overfitting_detector.overfitting_detector import OD_NONE
from gbdt.train import TrainedModel, train

comptime OH_CATS = 10
comptime OH_REPEATS = 12
comptime OH_ROWS = OH_CATS * OH_CATS * OH_REPEATS
comptime OH_FEATURES = 3
comptime OH_TREES = 12
comptime OH_DEPTH = 4
comptime OH_BORDERS = 32
comptime OH_BORDER_SAMPLES = 200_000


def _need(ok: Bool, what: String) raises:
    if not ok:
        raise Error("ordered CTR host check: " + what)


def _fixture(binary_second: Bool) -> Tuple[List[Float32], List[Float32]]:
    """Column-major x (categorical a, categorical b, numeric c) and y.
    `binary_second` gives column b two categories (a one-hot column)."""
    var x = List[Float32](capacity=OH_FEATURES * OH_ROWS)
    var y = List[Float32](capacity=OH_ROWS)
    for r in range(OH_ROWS):
        x.append(Float32((r * 7) % OH_CATS))
    for r in range(OH_ROWS):
        var b = (r // OH_CATS) % OH_CATS
        x.append(Float32(b % 2 if binary_second else b))
    for r in range(OH_ROWS):
        x.append(Float32((r * 37) % 101) * Float32(0.25))
    for r in range(OH_ROWS):
        var a = (r * 7) % OH_CATS
        var c = (r * 37) % 101
        var one = a % 3 == 0 or c > 80
        y.append(Float32(1.0) if one else Float32(0.0))
    return (x^, y^)


def _fixture_pair() -> Tuple[List[Float32], List[Float32]]:
    """The combination fixture: categorical a and b (ten categories each,
    every pair present `OH_REPEATS` times), numeric c, and a 0/1 target
    that is a function of the pair (a, b) with a numeric exception."""
    var x = List[Float32](capacity=OH_FEATURES * OH_ROWS)
    var y = List[Float32](capacity=OH_ROWS)
    for r in range(OH_ROWS):
        x.append(Float32(r % OH_CATS))
    for r in range(OH_ROWS):
        x.append(Float32((r // OH_CATS) % OH_CATS))
    for r in range(OH_ROWS):
        x.append(Float32((r * 37) % 101) * Float32(0.25))
    for r in range(OH_ROWS):
        var a = r % OH_CATS
        var b = (r // OH_CATS) % OH_CATS
        var c = (r * 37) % 101
        var one = (2 * a + 3 * b) % 5 == 0 or c > 90
        y.append(Float32(1.0) if one else Float32(0.0))
    return (x^, y^)


def _device_fit(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Float32],
    loss: String,
    permutations: Int,
    with_eval: Bool,
    complexity: Int,
) raises -> TrainedModel:
    var params = List[TCatFeatureParams]()
    if complexity > 1:
        var cp = TCatFeatureParams.default()
        cp.max_tensor_complexity = complexity
        params.append(cp^)
    var flags = List[Bool](length=OH_FEATURES, fill=True)
    flags[2] = False
    var eval_x = List[Float32]()
    var eval_y = List[Float32]()
    if with_eval:
        eval_x = x.copy()
        eval_y = y.copy()
    return train(
        ctx, x, y, OH_ROWS, OH_FEATURES,
        border_count=OH_BORDERS,
        border_build_max_samples=OH_BORDER_SAMPLES,
        n_estimators=OH_TREES,
        max_depth=OH_DEPTH,
        learning_rate=Float32(0.5),
        l2_leaf_reg=Float32(1.0),
        bootstrap_type=String("No"),
        random_seed=UInt64(0),
        cat_features=flags,
        cat_feature_params=params,
        loss=loss,
        permutation_count=permutations,
        eval_x_colmajor=eval_x,
        eval_y=eval_y,
        use_best_model=0,
        boosting_type=String("Ordered"),
    )


def _host_fit(
    x: List[Float32],
    y: List[Float32],
    logloss: Bool,
    permutations: Int,
    with_eval: Bool,
    complexity: Int,
) raises -> GbdtOrderedHostFit:
    # the slots as `_gbdt_fit_ordered_arm` resolves them for this fit
    # (`gbdt/mojolearn_gbdt_host_run.mojo`): the border 0.5, Newton leaves
    # (one iteration for RMSE, Logloss's count), no bootstrap, Cosine, no
    # score noise, `boost_from_average` on for RMSE only, GreedyLogSum (0)
    var border = Float32(0.5)
    var objective = GBDT_OBJ_LOGLOSS if logloss else GBDT_OBJ_RMSE
    var iterations = GBDT_LOGLOSS_NEWTON_ITERATIONS if logloss else 1
    var hloss = GbdtHostLoss(
        objective, border if logloss else Float32(0.0), Float32(0.5),
        GBDT_LEAF_NEWTON, iterations, -1, Float32(0.0), border,
    )
    var p = GbdtHostParams(
        OH_BORDERS, OH_BORDER_SAMPLES, OH_TREES, OH_DEPTH,
        Float32(0.5), Float32(1.0), UInt64(0), NAN_MODE_MIN, border,
        iterations, 0,
    )
    var opts = GbdtOrderedHostOptions(
        False, Float32(0.0), -1, Float32(0.0), permutations, Float64(2.0),
        ordered_permutation_block_size(OH_ROWS, 0), ORDERED_MIN_FOLD_SIZE,
        not logloss,
    )
    var ex = List[Float32]()
    var ey = List[Float32]()
    var n_eval = 0
    if with_eval:
        ex = x.copy()
        ey = y.copy()
        n_eval = OH_ROWS
    var ev = GbdtOrderedHostEval(
        ex^, ey^, n_eval, OD_NONE, Float64(-1.0), -1, 0, 1,
    )
    # bit 0: categorical, on a and b; c is numeric
    var flags = List[UInt32](length=OH_FEATURES, fill=UInt32(1))
    flags[2] = UInt32(0)
    return gbdt_ordered_host_fit(
        x, y, OH_ROWS, OH_FEATURES, p, hloss, opts, ev, flags, complexity
    )


def _line_at(text: String, start: Int) -> String:
    """The line of `text` holding byte `start`, at most 200 bytes of it."""
    var bytes = text.as_bytes()
    var n = len(bytes)
    var lo = start if start < n else n
    while lo > 0 and bytes[lo - 1] != UInt8(10):
        lo -= 1
    var hi = lo
    while hi < n and bytes[hi] != UInt8(10) and hi - lo < 200:
        hi += 1
    return String(StringSlice(unsafe_from_utf8=bytes[lo:hi]))


def _same_text(device: String, host: String, what: String) raises:
    if device == host:
        return
    var a = device.as_bytes()
    var b = host.as_bytes()
    var n = len(a) if len(a) < len(b) else len(b)
    var at = n
    for i in range(n):
        if a[i] != b[i]:
            at = i
            break
    var line_no = 1
    for i in range(at):
        if a[i] == UInt8(10):
            line_no += 1
    raise Error(
        "ordered CTR host check: " + what + ": the host model text is not"
        " the device's (" + String(len(b)) + " against " + String(len(a))
        + " bytes; first difference at byte " + String(at) + ", line "
        + String(line_no) + ")\n  device: " + _line_at(device, at)
        + "\n  host:   " + _line_at(host, at)
    )


def _same_losses(
    device: List[Float64], host: List[Float64], what: String
) raises:
    _need(
        len(device) == len(host),
        what + ": " + String(len(host)) + " host losses against "
        + String(len(device)),
    )
    for t in range(len(device)):
        if device[t] != host[t]:
            raise Error(
                "ordered CTR host check: " + what + " of tree " + String(t)
                + ": host " + String(host[t]) + ", device " + String(device[t])
            )


def check_ordered_ctr_host() raises:
    print("gbdt_ordered_host_fit with CTR columns against train, byte for byte:")
    var ctx = DeviceContext()
    var data = _fixture(False)
    var x = data[0].copy()
    var y = data[1].copy()

    var dev_ll = _device_fit(ctx, x, y, String("Logloss"), 4, False, 1)
    _need(dev_ll.ctr_column_count > 0, "the device model holds no CTR column")
    var host_ll = _host_fit(x, y, True, 4, False, 1)
    _same_text(model_text(dev_ll), host_ll.text, "Logloss, four permutations")
    _same_losses(dev_ll.losses, host_ll.losses, "Logloss learn loss")
    print("  Logloss, four permutations: same text, same learn losses")

    var dev_rmse = _device_fit(ctx, x, y, String("RMSE"), 4, False, 1)
    var host_rmse = _host_fit(x, y, False, 4, False, 1)
    _need(
        host_rmse.text.find("\nbias ") >= 0,
        "the RMSE host text carries no bias record",
    )
    _same_text(model_text(dev_rmse), host_rmse.text, "RMSE, four permutations")
    _same_losses(dev_rmse.losses, host_rmse.losses, "RMSE learn loss")
    print("  RMSE, four permutations: same text (bias included)")

    var dev_ev = _device_fit(ctx, x, y, String("Logloss"), 4, True, 1)
    var host_ev = _host_fit(x, y, True, 4, True, 1)
    _same_text(model_text(dev_ev), host_ev.text, "Logloss with an eval set")
    _same_losses(dev_ev.test_losses, host_ev.test_losses, "held-out loss")
    print("  Logloss with an eval set: same text, same held-out losses")

    var dev_one = _device_fit(ctx, x, y, String("Logloss"), 1, False, 1)
    var host_one = _host_fit(x, y, True, 1, False, 1)
    _same_text(model_text(dev_one), host_one.text, "one permutation")
    print("  one permutation: same text")

    # a one-hot column (b has two categories): the host fold searcher's
    # take-bin arm against the device's, the same text and losses
    var binary = _fixture(True)
    var bx = binary[0].copy()
    var by = binary[1].copy()
    var dev_oh = _device_fit(ctx, bx, by, String("Logloss"), 4, True, 1)
    var host_oh = _host_fit(bx, by, True, 4, True, 1)
    _need(
        host_oh.text.find(" one_hot 1") >= 0,
        "the host text of the one-hot fit names no one-hot feature",
    )
    _same_text(model_text(dev_oh), host_oh.text, "a one-hot column, Logloss")
    _same_losses(dev_oh.losses, host_oh.losses, "one-hot learn loss")
    _same_losses(dev_oh.test_losses, host_oh.test_losses, "one-hot held-out loss")
    var dev_oh_rmse = _device_fit(ctx, bx, by, String("RMSE"), 4, False, 1)
    var host_oh_rmse = _host_fit(bx, by, False, 4, False, 1)
    _same_text(model_text(dev_oh_rmse), host_oh_rmse.text, "a one-hot column, RMSE")
    print("  a one-hot column under Ordered: same text, same losses")

    # FEATURE COMBINATIONS under Ordered: the fold searcher's level driver
    # on the device against the host's
    var pair = _fixture_pair()
    var px = pair[0].copy()
    var py = pair[1].copy()
    var dev_tc = _device_fit(ctx, px, py, String("Logloss"), 4, False, 2)
    _need(
        not dev_tc.combination_ctrs.is_empty(),
        "the device Ordered fit at complexity 2 chose no combination (the"
        " fixture no longer exercises tree CTRs)",
    )
    var host_tc = _host_fit(px, py, True, 4, False, 2)
    _need(
        host_tc.text.find("combination") >= 0,
        "the host text at complexity 2 carries no combination record",
    )
    _same_text(model_text(dev_tc), host_tc.text, "combinations, Logloss")
    _same_losses(dev_tc.losses, host_tc.losses, "combinations learn loss")
    var dev_tc1 = _device_fit(ctx, px, py, String("Logloss"), 4, False, 1)
    _need(
        model_text(dev_tc) != model_text(dev_tc1),
        "complexity 2 gave the complexity 1 model",
    )
    print("  combinations, Logloss, four permutations: same text, same losses")

    var dev_tc_ev = _device_fit(ctx, px, py, String("Logloss"), 4, True, 2)
    var host_tc_ev = _host_fit(px, py, True, 4, True, 2)
    _same_text(
        model_text(dev_tc_ev), host_tc_ev.text, "combinations with an eval set"
    )
    _same_losses(
        dev_tc_ev.test_losses, host_tc_ev.test_losses,
        "combinations held-out loss",
    )
    print("  combinations with an eval set: same text, same held-out losses")

    var dev_tc_rmse = _device_fit(ctx, px, py, String("RMSE"), 4, False, 2)
    var host_tc_rmse = _host_fit(px, py, False, 4, False, 2)
    _same_text(model_text(dev_tc_rmse), host_tc_rmse.text, "combinations, RMSE")
    _same_losses(
        dev_tc_rmse.losses, host_tc_rmse.losses, "combinations RMSE learn loss"
    )
    print("  combinations, RMSE: same text, same losses")

    var dev_tc_one = _device_fit(ctx, px, py, String("Logloss"), 1, False, 2)
    var host_tc_one = _host_fit(px, py, True, 1, False, 2)
    _same_text(
        model_text(dev_tc_one), host_tc_one.text,
        "combinations, one permutation",
    )
    print("  combinations, one permutation: same text")


def run_checks() raises:
    check_ordered_ctr_host()
