# SPDX-License-Identifier: Apache-2.0
"""MultiClassOneVsAll: dimensions, the sigmoid link, argmax classes and the
effect of sample weights through the native fit (ported from the native part
of `python/mojolearn/tests/test_multiclass_ova_surface.py`
`test_ova_dimensions_link_weights_and_saved_mode`; its save/load under a
patched default mode and the refusal tests stay in Python).

For 2 and 3 classes: 30 rows of 2 uniform [0, 1) features, labels `i % k`,
class_weights `[1, ..., k]`, sample_weight `(1..30) / 2`, n_estimators 3,
max_depth 2, bootstrap_type 'No', random_strength 0, `identical`:

    the model's approx dimension is k, and predict is (30, k) float32
    predict_proba is the float64 sigmoid of each class's own raw score
        (never a softmax across classes) within atol 1e-6
    predict_classes (the argmax of predict_proba, `argmax_rows_f32`) is
        the first maximum of each probability row
    the unweighted fit predicts other bytes than the weighted one

DEVICE COLUMN ONLY: the host fit (`gbdt/mojolearn_gbdt_host_run.mojo`
`gbdt_fit_native_run`) refuses sample_weight outside Plain Logloss, as the
Python test's CPU-only skip says, so the weighted OVA fit exists only on the
device image (`gbdt.mojolearn_gbdt_run.gbdt_fit_native_run`). The two class
counts run in order: each device fit owns the one device for its graph.
Predictions read the device-fitted model text with the host reader, as
`gbdt_public_fits` does. The features are a SplitMix64 uniform stream
regenerated here instead of NumPy's PCG64 (`default_rng(0).random`): every
assertion is a relation that holds for any such data.
"""
from std.math import exp
from std.memory import bitcast
from core.host_helpers_impl import argmax_rows_f32
from gbdt.fit_request import GbdtFitArgs, GbdtFitOutcome
from gbdt.gbdt_entry import GbdtImage
from gbdt.mojolearn_gbdt_run import gbdt_fit_native_run as gbdt_fit_native_run_gpu
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.native_tests.gbdt_public_fits import GbdtOpts, GbdtInputs, fit, predict, predict_proba, model_dim, same_f32
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime OVA_ROWS = 30
comptime OVA_UNITS = 2

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


def _next(mut state: UInt64) -> UInt64:
    """SplitMix64."""
    state += UInt64(0x9E3779B97F4A7C15)
    var z = state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


struct _DeviceFit(GbdtImage):
    """The GPU image's column-major fit (`bindings/_mojolearn_gbdt.mojo` `_GbdtGpuImage`)."""
    @staticmethod
    def fit(row_major: Bool, args: GbdtFitArgs) raises -> GbdtFitOutcome:
        require(not row_major, "the check fits column-major")
        return gbdt_fit_native_run_gpu(False, args)

    @staticmethod
    def row_major() -> Bool:
        return False

    @staticmethod
    def model_dim(text: String) raises -> Int:
        return model_dim(text)

    # The predict, resident and ordered/feature-freq operations of GbdtImage
    # (lane dw-trees-boost, 2026-11-21): this check image fits only, so each
    # refuses in words.
    @staticmethod
    def predict(text: String, x: Int, n_rows: Int, dst: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_predict")

    @staticmethod
    def predict_multi(text: String, x: Int, n_rows: Int, dst: Int, mode: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_predict_multi")

    @staticmethod
    def binary(probabilities: Bool, raw: Int, dst: Int, n: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no binary prediction")

    @staticmethod
    def sigmoid_pair(raw: Int, dst: Int, n: Int) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_sigmoid_pair")

    @staticmethod
    def resident() -> Bool:
        return False

    @staticmethod
    def resident_prepare(text: String) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_resident_prepare")

    @staticmethod
    def resident_predict(handle: Int, x: Int, n_rows: Int, dst: Int, mode: Int, row_major: Bool) raises -> Int:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_resident_predict")

    @staticmethod
    def ordered_fit(x: Int, y: Int, weights: Int, permutation: Int, n_rows: Int, n_features: Int, n_weights: Int,
                    n_permutation: Int, n_estimators: Int, max_depth: Int, border_count: Int,
                    learning_rate: Float32, l2_leaf_reg: Float32) raises -> String:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_fit_ordered_rmse")

    @staticmethod
    def feature_freq_fit(x: Int, y: Int, weights: Int, sources: Int, n_rows: Int, n_features: Int, n_weights: Int,
                         n_sources: Int, learning_rate: Float32, l2_leaf_reg: Float32,
                         random_seed: Int) raises -> String:
        raise Error("AttributeError: mojolearn: this check image has no gbdt_fit_two_level_feature_freq")


def _inputs(classes: Int, weighted: Bool) -> GbdtInputs:
    """`_ova(classes)`'s X (column-major, 2 features) and y; the weights when asked."""
    var state = UInt64(0x2545F4914F6CDD1D) + UInt64(1)
    var x = List[Float32](length=OVA_ROWS * 2, fill=0)
    for i in range(OVA_ROWS * 2):  # small-loop(one fixture's sequential draw stream)
        x[i] = Float32(Float64(Int(_next(state) >> 40)) * (1.0 / 16777216.0))
    var y = List[Float32](length=OVA_ROWS, fill=0)
    for i in range(OVA_ROWS):  # small-loop(rows)
        y[i] = Float32(i % classes)
    var inp = GbdtInputs(x^, OVA_ROWS, 2, y^)
    if weighted:
        inp.weights = List[Float32](length=OVA_ROWS, fill=0)
        for i in range(OVA_ROWS):  # small-loop(rows): np.arange(1, 31) / 2
            inp.weights[i] = Float32(i + 1) / 2
    return inp^


def _options(classes: Int) -> GbdtOpts:
    var o = GbdtOpts("MultiClassOneVsAll", 3, 2)
    o.deterministic()
    o.class_weights = List[Float64]()
    for k in range(classes):  # small-loop(classes)
        o.class_weights.append(Float64(k + 1))
    return o^


def ova_unit[I: GbdtImage](classes: Int) raises:
    var tag = "OVA " + String(classes) + " classes: "
    var o = _options(classes)
    var inp = _inputs(classes, True)
    var model = fit[I](o, inp)
    require(model_dim(model.text) == classes, tag + "approx_dim_ is not n_classes_")
    var raw = predict(model.text, inp.x, OVA_ROWS)
    var probability = predict_proba(model.text, "MultiClassOneVsAll", inp.x, OVA_ROWS)
    require(len(raw) == OVA_ROWS * classes and len(probability) == OVA_ROWS * classes,
            tag + "predict / predict_proba are not (30, classes)")
    var far = 0
    for i in range(OVA_ROWS * classes):  # small-loop(one prediction block)
        var sigmoid = 1.0 / (1.0 + exp(-Float64(raw[i])))
        if abs(probability[i] - sigmoid) > 1e-6:
            far += 1
    require(far == 0, tag + String(far) + " probabilities are not the sigmoid of their own raw score")
    var p32 = List[Float32](length=OVA_ROWS * classes, fill=0)
    for i in range(OVA_ROWS * classes):  # small-loop(one proba block): float32 carried exactly
        p32[i] = Float32(probability[i])
    var predicted = List[Int64](length=OVA_ROWS, fill=-1)
    argmax_rows_f32(_F32P(unsafe_from_address=Int(p32.unsafe_ptr())), OVA_ROWS, classes,
                    _I64P(unsafe_from_address=Int(predicted.unsafe_ptr())))
    var moved = 0
    for r in range(OVA_ROWS):  # small-loop(rows)
        var best = 0
        for k in range(1, classes):  # small-loop(classes)
            if p32[r * classes + k] > p32[r * classes + best]:
                best = k
        if predicted[r] != Int64(best):
            moved += 1
    require(moved == 0, tag + String(moved) + " predict_classes rows are not the argmax of predict_proba")
    var plain = _inputs(classes, False)
    var unweighted = fit[I](o, plain)
    require(not same_f32(predict(unweighted.text, plain.x, OVA_ROWS), raw),
            tag + "sample_weight did not change the predictions")
    _ = inp^
    _ = plain^
    _ = p32^
    _ = predicted^


# ---- The device column only (see the docstring) ----
def _run_gpu(mut report: Report) raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var cases = report.cases(OVA_UNITS)
    for unit in range(OVA_UNITS):  # driver-loop: over the fits, one device graph at a time
        try:
            ova_unit[_DeviceFit](unit + 2)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("gbdt_ova_link", cpu=False, gpu=True, serial="", run=run_cases)
