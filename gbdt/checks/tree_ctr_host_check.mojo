# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column of the tree-CTR fit against the device, byte for byte.

    pixi run check-native-gpu gbdt_tree_ctr_host_check

`gbdt_ctr_host_fit(..., max_ctr_complexity=2)` (`gbdt/host/
gbdt_oracle_ctr.mojo`) and `train` with `max_tensor_complexity = 2`
(`gbdt/train.mojo`) are two spellings of one algorithm with one fold order,
so their model texts must be the same bytes: the grids, every split (the
combination columns included), every leaf, the simple CTR tables, the
combination maps and the combination tables.

THE FIXTURE CAN FAIL, and is the one of `checks/tree_ctr_fit_device_check.
mojo`: two categorical columns of ten categories, every pair equally often,
a 0/1 target that is a function of the PAIR, so the trees must mint the
combination to fit it. The fit is the configuration the host categorical
arm covers: Logloss, SymmetricTree, the greedy searcher, no bootstrap,
`random_strength` 0, the default CTRs.

Checked:

  * complexity 1: the host text equals the device text (the fit without
    tree CTRs; if this fails the difference is not in the combinations);
  * complexity 2: the device model holds combinations, and the host text
    equals the device text;
  * complexity 2 with the learn rows as the eval set: the texts are equal
    and the held-out losses are the same bits (the persist slots of the
    test index on both columns).

A difference is reported as the first differing line of the two texts.
The suite needs a GPU for the device arm; the host arm runs on the calling
host. Serial by nature: three pairs of fits, each pair compared whole.
"""

from max.gpu.host import DeviceContext

from gbdt.host.gbdt_oracle import (
    GBDT_LOGLOSS_NEWTON_ITERATIONS,
    GbdtHostParams,
)
from gbdt.host.gbdt_oracle_ctr import (
    GbdtCtrHostFit,
    gbdt_ctr_host_fit,
    gbdt_ctr_host_model_text,
)
from gbdt.host.gbdt_oracle_eval import GbdtHostEval
from gbdt.models.model_text import model_text
from gbdt.options.catboost_options import TCatFeatureParams
from gbdt.options.data_processing_options import NAN_MODE_MIN
from gbdt.overfitting_detector.overfitting_detector import OD_NONE
from gbdt.train import TrainedModel, train

comptime HOST_CATS = 10
comptime HOST_REPEATS = 60
comptime HOST_ROWS = HOST_CATS * HOST_CATS * HOST_REPEATS
comptime HOST_TREES = 20
comptime HOST_DEPTH = 4
comptime HOST_BORDERS = 32
comptime HOST_BORDER_SAMPLES = 200_000


def _need(ok: Bool, what: String) raises:
    if not ok:
        raise Error("tree CTR host check: " + what)


def _fixture() -> Tuple[List[Float32], List[Float32]]:
    """Column-major x (a, then b) and y."""
    var x = List[Float32](capacity=2 * HOST_ROWS)
    var y = List[Float32](capacity=HOST_ROWS)
    for r in range(HOST_ROWS):
        x.append(Float32(r % HOST_CATS))
    for r in range(HOST_ROWS):
        x.append(Float32((r // HOST_CATS) % HOST_CATS))
    for r in range(HOST_ROWS):
        var a = r % HOST_CATS
        var b = (r // HOST_CATS) % HOST_CATS
        var one = a == 0 or (2 * a + 3 * b) % 5 == 0
        y.append(Float32(1.0) if one else Float32(0.0))
    return (x^, y^)


def _device_fit(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Float32],
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
        ctx, x, y, HOST_ROWS, 2,
        border_count=HOST_BORDERS,
        border_build_max_samples=HOST_BORDER_SAMPLES,
        n_estimators=HOST_TREES,
        max_depth=HOST_DEPTH,
        learning_rate=Float32(0.5),
        l2_leaf_reg=Float32(1.0),
        bootstrap_type=String("No"),
        random_seed=UInt64(0),
        cat_features=flags,
        cat_feature_params=params,
        loss=String("Logloss"),
        eval_x_colmajor=eval_x,
        eval_y=eval_y,
        use_best_model=0,
    )


def _host_fit(
    x: List[Float32], y: List[Float32], complexity: Int, with_eval: Bool
) raises -> GbdtCtrHostFit:
    # the slots as `gbdt_fit_native_run` resolves them for this fit
    # (`gbdt/mojolearn_gbdt_host_run.mojo`): the Logloss border 0.5, the
    # Newton iteration count of Logloss, GreedyLogSum (0)
    var p = GbdtHostParams(
        HOST_BORDERS, HOST_BORDER_SAMPLES, HOST_TREES, HOST_DEPTH,
        Float32(0.5), Float32(1.0), UInt64(0), NAN_MODE_MIN, Float32(0.5),
        GBDT_LOGLOSS_NEWTON_ITERATIONS, 0,
    )
    # bit 0: categorical; bit 1 (one-hot) clear on both columns
    var flags = List[UInt32](length=2, fill=UInt32(1))
    var ev = GbdtHostEval.none()
    if with_eval:
        ev = GbdtHostEval(
            x.copy(), y.copy(), HOST_ROWS, OD_NONE, Float64(-1.0), -1, 0, 1,
        )
    return gbdt_ctr_host_fit(
        x, y, HOST_ROWS, 2, p, flags, -1, -1, ev^, complexity,
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
        "tree CTR host check: " + what + ": the host model text is not the"
        " device's (" + String(len(b)) + " against " + String(len(a))
        + " bytes; first difference at byte " + String(at) + ", line "
        + String(line_no) + ")\n  device: " + _line_at(device, at)
        + "\n  host:   " + _line_at(host, at)
    )


def check_tree_ctr_host() raises:
    print("gbdt_ctr_host_fit against train, model text byte for byte:")
    var ctx = DeviceContext()
    var data = _fixture()
    var x = data[0].copy()
    var y = data[1].copy()

    var dev1 = _device_fit(ctx, x, y, 1, False)
    var host1 = _host_fit(x, y, 1, False)
    _need(dev1.combination_ctrs.is_empty(), "complexity 1 built a combination")
    _same_text(model_text(dev1), gbdt_ctr_host_model_text(host1), "complexity 1")
    print("  complexity 1: same text")

    var dev2 = _device_fit(ctx, x, y, 2, False)
    var host2 = _host_fit(x, y, 2, False)
    _need(
        not dev2.combination_ctrs.is_empty(),
        "the device fit at complexity 2 chose no combination (the fixture"
        " no longer exercises tree CTRs)",
    )
    _need(
        len(host2.combination_ctrs.tables) == len(dev2.combination_ctrs.tables),
        "the host fit kept " + String(len(host2.combination_ctrs.tables))
        + " combination tables, the device "
        + String(len(dev2.combination_ctrs.tables)),
    )
    _same_text(model_text(dev2), gbdt_ctr_host_model_text(host2), "complexity 2")
    print(
        "  complexity 2: same text,", len(dev2.combination_ctrs.tables),
        "combination tables",
    )

    var dev3 = _device_fit(ctx, x, y, 2, True)
    var host3 = _host_fit(x, y, 2, True)
    _same_text(
        model_text(dev3), gbdt_ctr_host_model_text(host3),
        "complexity 2 with an eval set",
    )
    _need(
        len(host3.fit.eval.test_losses) == len(dev3.test_losses),
        "held-out loss count: host "
        + String(len(host3.fit.eval.test_losses)) + ", device "
        + String(len(dev3.test_losses)),
    )
    for t in range(len(dev3.test_losses)):
        if host3.fit.eval.test_losses[t] != dev3.test_losses[t]:
            raise Error(
                "tree CTR host check: held-out loss of tree " + String(t)
                + ": host " + String(host3.fit.eval.test_losses[t])
                + ", device " + String(dev3.test_losses[t])
            )
    print("  complexity 2 with an eval set: same text, same held-out losses")


def run_checks() raises:
    check_tree_ctr_host()
