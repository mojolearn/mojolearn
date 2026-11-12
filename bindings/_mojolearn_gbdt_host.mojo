# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_gbdt` family, GradientBoosting on the
gbdt-symmetric, gbdt-rmse, gbdt-depthwise and gbdt-lossguide lanes
(workstream E batch 3, 2026-09-14) and the gbdt-nan-modes, gbdt-adapter-clf,
gbdt-adapter-reg, gbdt-parametric-losses, gbdt-exact-mae,
gbdt-lossguide-newtoncosine, gbdt-multiclass and gbdt-onevsall lanes
(lane/cpu-training-gbdt-losses, 2026-09-15: the pointwise losses, the Exact
leaves and the row bootstraps through `gbdt/host/gbdt_oracle_losses.mojo`,
the Lossguide searcher options through `gbdt/host/gbdt_oracle_depthwise.mojo`,
MultiClass and MultiClassOneVsAll through
`gbdt/host/gbdt_oracle_multiclass.mojo`. `loss="RMSE"` fits through
`gbdt/host/gbdt_oracle_rmse.mojo::gbdt_rmse_host_fit`.

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The fit is
`gbdt/host/gbdt_oracle.mojo::gbdt_host_fit` for SymmetricTree and
`gbdt/host/gbdt_oracle_depthwise.mojo::gbdt_host_fit_non_symmetric` for
Depthwise and Lossguide (the `ntree` / `node` model shape), the device trainer
(`gbdt/train.mojo::train`, `gbdt/methods/doc_parallel_boosting.mojo::
fit_with_test`, the greedy symmetric searcher and the Newton leaf estimator)
restated on the host, so the model text is meant to be the GPU columns'
bytes. The predict entry is `core/gbdt_host_predict.mojo::gbdt_host_predict`,
the walk the forest host binding runs and the forest host gate holds to the
recorded GPU predictions, over the model text parsed here.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES for what this covers, so
`python/mojolearn/ensemble.py` runs unchanged on a CPU-only install through
`_backend._HOST_MODULES` (`"_mojolearn_gbdt": "_mojolearn_gbdt_host"`):
`gbdt_fit` (the same eight arguments, the same 35-slot params list with its
counted class-weight tail and optional float tail, the same four strings,
the same five-element return), `gbdt_predict` and `gbdt_model_dim` (the
model text in, as `bindings/_mojolearn_gbdt.mojo:410-450`), `gbdt_sigmoid_pair`
(its body), `gbdt_binary_probabilities` and `gbdt_binary_classes` (the
classifier adapter's transforms, `bindings/_mojolearn_gbdt.mojo:107-130`
over `gbdt/binary_prediction.mojo`'s kernel, restated per element),
`gbdt_predict_multi` (a multi-dimensional model's raw, softmax and sigmoid
columns), `gbdt_vendor` answering "cpu" and `gbdt_numeric_mode`.

`gbdt_fit_ordered_rmse` (the gbdt-ordered-rmse lane, through
`gbdt/host/gbdt_oracle_ordered.mojo`) and `gbdt_fit_two_level_feature_freq`
(the gbdt-feature-freq lane, through `gbdt/host/gbdt_oracle_feature_freq.mojo`)
take the GPU binding's params and return its model text
(lane/cpu-training-gbdt-ordered, 2026-09-15); both refuse `sample_weight` by
name. `gbdt_fit` with `use_pointwise_searcher=True` trains the
gbdt-pointwise-l2-bayesian-eval lane (`_gbdt_fit_pointwise_arm`,
`gbdt/host/gbdt_oracle_pointwise.mojo`) and one-hot categorical columns train
under SymmetricTree with Logloss (gbdt-categorical-ctr,
`gbdt/host/gbdt_oracle_onehot.mojo`).

AN EVAL SET, THE OVERFITTING DETECTOR AND `use_best_model` on the PLAIN
SymmetricTree Logloss fit (lane/close-no-cpu-path-gbdt, 2026-09-20, the
gbdt-symmetric-eval lane), through `gbdt/host/gbdt_oracle_eval.mojo`:
their `testCursor` (`_apply_last_tree_to_test`), the held-out curve
(`_test_loss` through the same Logloss kernel the learn curve uses), the
shared detector and `ShrinkToBestIteration`. An eval set does not reach the
learn cursor, the borders, the splits or the leaves on this path, so the
model text of a fit WITH one is the model text of the same fit without one
and what the arm adds is `test_loss_curve_`, `best_iteration_` and
`stopped_early_`. Every OTHER Plain arm (RMSE, the pointwise losses,
MultiClass, Depthwise, Lossguide) still refuses an eval set BY NAME: their
test cursor runs the same four pieces over a different loss kernel or a
different model shape, and neither is restated.

ABSENT, and so refused BY NAME through `_HostBinding`: `gbdt_per_round_paths`
and the two `*_parallel_available` probes.

NaN IN X. The oracle's grid places the NaN border (`nan_mode` Min or Max,
`_calc_quantization_phase_b`) and `_binarize_columns` substitutes the value
before binning, which is the device fit's order, so an X carrying NaN trains
through the same entries on SymmetricTree with Logloss (the gbdt-nan-modes
lane). Under RMSE or a non-symmetric policy NaN is unmeasured and still
refused by name.

REFUSED BY NAME INSIDE `gbdt_fit`, with the sentence
`cpu_identity_gate_check.py` requires ("no CPU implementation of"), every
parameter value the oracle does not restate: see `_refuse` and the oracle's
module docstring. Every other GBDT lane of tools/identity_break.py therefore
reads REFUSED on a CPU-only install, as it did before this binding existed.

The sabotage arm (`gbdt_host_sabotage`) is
`gbdt/host/gbdt_oracle.mojo::GBDT_ORACLE_HOST_SABOTAGE`: the Newton walker's
Hessian regularizer is one larger, so every leaf of every tree moves; on
the RMSE arm, which runs no walker, the searcher's leaf regularizer
(`gbdt/host/gbdt_oracle_rmse.mojo::_rmse_leaf_value`) is one larger; the
ordered arm's Newton Hessian regularizer and the FeatureFreq leaf value's l2
are one larger.
"""
from core.experiment_guards_trees import TREES_GUARDS_OK
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from core.hostptr import f32_ptr, f64_ptr, u32_ptr
from bindings.gbdt_fit_boundary_unpack import gbdt_fit_args_py
from bindings.gbdt_fit_boundary_pack import gbdt_fit_outcome_py
from gbdt.fit_request import GbdtFitArgs, GbdtFitOutcome
from gbdt.mojolearn_gbdt_host_run import (
    gbdt_fit_native_run,
    gbdt_fit_ordered_rmse_run,
    gbdt_fit_two_level_feature_freq_run,
    gbdt_predict_multi_run,
)
from gbdt.gbdt_entry import gbdt_entry_publish, gbdt_entry_dispatch, GbdtImage
from core.native_descriptor_abi import NativeDescriptorError
# The typed bodies (lane thin-trees-metrics, 2026-10-21): the model text
# reader, the predict walks and links, the binary transforms, and the scans
# and reads of the fit's tails, moved word for word out of this shell.
from gbdt.host.gbdt_host_entries_impl import (
    _parse_model,
    gbdt_host_binary_margins_check,
    gbdt_host_binary_prediction_into,
    gbdt_host_predict_into,
    gbdt_host_sigmoid_pair_into,
)
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from gbdt.host.gbdt_oracle import GBDT_ORACLE_HOST_SABOTAGE


def gbdt_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def gbdt_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def gbdt_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives in
    a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "gbdt host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_gbdt_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `gbdt_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build. The comptime assert
# above is the check.


def gbdt_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary adds 1.0 to the Newton walker's Hessian
    regularizer and to the RMSE arm's leaf regularizer on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control;
    `gbdt/host/gbdt_oracle.mojo::GBDT_ORACLE_HOST_SABOTAGE`)."""
    return PythonObject(GBDT_ORACLE_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def gbdt_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def gbdt_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def gbdt_fit_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    weights_addr: PythonObject,
    cat_flags_addr: PythonObject,
    eval_x_addr: PythonObject,
    eval_y_addr: PythonObject,
    params: PythonObject,
    strs: PythonObject,
) raises -> PythonObject:
    """`gbdt_fit`: UNPACK and PACK only (bindings/gbdt_fit_boundary.mojo)
    around `gbdt_fit_native`; the GBDT doors call the fit natively through
    this image's table entry."""
    var args = gbdt_fit_args_py(x_addr, y_addr, weights_addr, cat_flags_addr, eval_x_addr, eval_y_addr, params, strs)
    return gbdt_fit_outcome_py(gbdt_fit_native(args))


# ===========================================================================
# THE MODEL TEXT, READ (the records `gbdt_host_model_text` and the GPU
# binding's `model_text` write; `gbdt/models/model_text.mojo:706` is the
# reader this follows, for the oblivious float-only shape)
# ===========================================================================


def gbdt_fit_native(params: GbdtFitArgs) raises -> GbdtFitOutcome:
    """`gbdt_fit_native_run` (gbdt/mojolearn_gbdt_host_run.mojo: the slot
    checks, refusals and host fit) with the GIL released around the whole
    typed call."""
    var outcome: GbdtFitOutcome
    with GILReleased(Python()):
        outcome = gbdt_fit_native_run(params)
    return outcome^


def gbdt_predict_binding(
    model: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`gbdt_predict_binding` (`bindings/_mojolearn_gbdt.mojo:410-439`): RAW
    approxes for a one-dimensional model, `params` is `[n_rows]`, the
    feature count comes from the model, returns rows written."""
    if len(params) != 1:
        raise Error(
            "gbdt_predict: params must hold [n_rows], got "
            + String(len(params))
        )
    var text = String(py=model)
    var x_address = Int(py=x_addr)
    var op = f32_ptr(Int(py=out_addr))
    _ = f32_ptr(x_address)
    var n_rows = Int(py=params[0])
    if n_rows <= 0:
        raise Error("gbdt_predict: n_rows must be positive")
    var m = _parse_model(text)
    if m.dim != 1:
        raise Error(
            "predict_floats is one-dimensional; this model has"
            " approx_dim " + String(m.dim)
            + ". Use predict_multi_floats, which returns"
            " [row * approx_dim + dim]."
        )
    var n_features = len(m.fold_counts)
    var n_trees = len(m.tree_offsets) - 1
    with GILReleased(Python()):
        gbdt_host_predict_into(m^, x_address, n_rows, n_features, n_trees, op)
    return PythonObject(n_rows)


def gbdt_predict_multi_binding(
    model: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`gbdt_predict_multi_binding` (`bindings/_mojolearn_gbdt.mojo:453-485`)
    over `gbdt/estimator.mojo::gbdt_predict_multi`: `params` is
    `[n_rows, mode]`, mode 0 RAW (`dim` columns), 1 SOFTMAX (`dim + 1`,
    `gbdt/train.mojo::multiclass_probabilities`), 2 SIGMOID (`dim`,
    `one_vs_all_probabilities`), both transforms in double through
    `identical_exp64`, row-major out. Returns the width written."""
    if len(params) != 2:
        raise Error(
            "gbdt_predict_multi: params must hold [n_rows,"
            " as_probabilities], got " + String(len(params))
        )
    var text = String(py=model)
    var x_address = Int(py=x_addr)
    var op = f32_ptr(Int(py=out_addr))
    _ = f32_ptr(x_address)
    var n_rows = Int(py=params[0])
    var mode = Int(py=params[1])
    var width: Int
    with GILReleased(Python()):
        width = gbdt_predict_multi_run(text, x_address, n_rows, mode, op)
    return PythonObject(width)


def gbdt_model_dim_binding(model: PythonObject) raises -> PythonObject:
    """`gbdt_model_dim` (`bindings/_mojolearn_gbdt.mojo:442-450`): 1 for an
    empty ensemble, else the trees' common dim."""
    return PythonObject(_parse_model(String(py=model)).dim)


def gbdt_fit_two_level_feature_freq_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    weights_addr: PythonObject,
    sources_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`gbdt_fit_two_level_feature_freq_binding`
    (`bindings/_mojolearn_gbdt.mojo:499-532`): the same seven params in the
    same order, `[n_rows, n_features, n_weights, n_sources, learning_rate,
    l2_leaf_reg, random_seed]`, the model text back, through
    `gbdt/host/gbdt_oracle_feature_freq.mojo::gbdt_feature_freq_host_fit`
    (the gbdt-feature-freq lane). `sample_weight` refuses by name. The seed
    is read and, as on the device (`gbdt/estimator.mojo:220-226` hands it to
    a searcher that draws no noise at `score_std_dev` 0), reaches no bit."""
    if len(params) != 7:
        raise Error("two-level FeatureFreq params must have seven values")
    var x_address = Int(py=x_addr)
    var y_address = Int(py=y_addr)
    _ = f32_ptr(x_address)
    _ = f32_ptr(y_address)
    _ = f32_ptr(Int(py=weights_addr))
    var sp = u32_ptr(Int(py=sources_addr))
    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_weights = Int(py=params[2])
    var n_sources = Int(py=params[3])
    var learning_rate = Float32(Float64(py=params[4]))
    var l2_leaf_reg = Float32(Float64(py=params[5]))
    _ = UInt64(Int(py=params[6]))
    var text = String("")
    with GILReleased(Python()):
        text = gbdt_fit_two_level_feature_freq_run(
            x_address, y_address, sp, n_rows, n_features, n_weights,
            n_sources, learning_rate, l2_leaf_reg,
        )
    return PythonObject(text)


def gbdt_fit_ordered_rmse_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    weights_addr: PythonObject, permutation_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`gbdt_fit_ordered_rmse_binding` (`bindings/_mojolearn_gbdt.mojo:
    535-591`): the same nine params in the same order, `[n_rows,
    n_features, n_weights, n_permutation, n_estimators, max_depth,
    border_count, learning_rate, l2_leaf_reg]`, the model text back, through
    `gbdt/host/gbdt_oracle_ordered.mojo::gbdt_ordered_rmse_host_fit` (the
    gbdt-ordered-rmse lane). `sample_weight` refuses by name."""
    if len(params) != 9:
        raise Error("ordered RMSE params must have nine values")
    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_weights = Int(py=params[2])
    var n_permutation = Int(py=params[3])
    var n_estimators = Int(py=params[4])
    var max_depth = Int(py=params[5])
    var border_count = Int(py=params[6])
    var learning_rate = Float32(Float64(py=params[7]))
    var l2_leaf_reg = Float32(Float64(py=params[8]))
    var x_address = Int(py=x_addr)
    var y_address = Int(py=y_addr)
    var weights_address = Int(py=weights_addr)
    var permutation_address = Int(py=permutation_addr)
    var text = String("")
    with GILReleased(Python()):
        text = gbdt_fit_ordered_rmse_run(
            x_address, y_address, weights_address, permutation_address,
            n_rows, n_features, n_weights, n_permutation, n_estimators,
            max_depth, border_count, learning_rate, l2_leaf_reg,
        )
    return PythonObject(text)


# TOMBSTONE: gbdt_sigmoid (dead native entry: no live caller) deleted 2026-10-26 by 7c1201602; code recoverable at 485ba9cc1.
# Restore: git apply experiments/removed/gbdt_sigmoid.patch


def gbdt_sigmoid_pair_binding(
    raw_addr: PythonObject, out_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """`gbdt_sigmoid_pair_binding` (`bindings/_mojolearn_gbdt.mojo`, DEVIATION
    2902), its body: `out[2 * i] = 1 - p`, `out[2 * i + 1] = p` with `p`
    exactly `1 / (1 + identical_exp64(-raw))`, `n` rows, both buffers float64. The
    `1.0 - p` is the one IEEE double subtraction the Python layer computed
    per row. Under the forest walk's sabotage define
    (`MOJOLEARN_FOREST_HOST_SABOTAGE`, the lane's negative control) the two
    columns are written swapped."""
    var rp = f64_ptr(Int(py=raw_addr))
    var op = f64_ptr(Int(py=out_addr))
    var count = Int(py=n)
    if count < 0:
        raise Error("gbdt_sigmoid_pair: n must be non-negative")
    gbdt_host_sigmoid_pair_into(rp, op, count)
    return PythonObject(count)


def gbdt_binary_prediction_binding[probabilities: Bool, dtype: DType](
    raw_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`gbdt_binary_prediction_binding` (`bindings/_mojolearn_gbdt.mojo:
    107-130`), the GradientBoostingClassifier adapter's two transforms, with
    the same checks in the same words. The body is
    `gbdt/binary_prediction.mojo::binary_prediction_kernel` restated per
    element on the host: the probability pair
    `[ftz(1 - p), p]` with `p = ftz(identical_sigmoid(ftz(margin)))`, and the
    class code from the margin's sign and magnitude bits (strict raw > 0,
    so a positive subnormal margin is class one on every column)."""
    if len(params) != 1:
        raise Error("binary prediction: params must contain n")
    var n = Int(py=params[0])
    if n <= 0 or n > 2147483647:
        raise Error("binary prediction: positive n<=Int32.max required")
    var rp = f32_ptr(Int(py=raw_addr))
    var address = Int(py=out_addr)
    if address == 0:
        raise Error("binary prediction: null output")
    var op = MutPointer[Scalar[dtype], MutUntrackedOrigin](unsafe_from_address=address)
    gbdt_host_binary_margins_check(rp, n)
    var count = 2 * n if probabilities else n
    with GILReleased(Python()):
        gbdt_host_binary_prediction_into[probabilities, dtype](rp, op, n)
    return PythonObject(count)


def gbdt_binary_probabilities_binding(
    raw_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`[1 - p, p]` per margin, Float32 (see `gbdt_binary_prediction_binding`)."""
    return gbdt_binary_prediction_binding[True, DType.float32](raw_addr, out_addr, params)


def gbdt_binary_classes_binding(
    raw_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """The Int32 class code per margin (see `gbdt_binary_prediction_binding`)."""
    return gbdt_binary_prediction_binding[False, DType.int32](raw_addr, out_addr, params)


# ---- The image's native gbdt entry (lane calltable-gbdt-forest-dart,
# 2026-11-11; ABI gbdt/fit_request.mojo, table core/gbdt_entry_table.mojo):
# the GBDT doors fit and read the model dimension through it instead of the
# Python entries above, which remain for the unconverted Python callers and
# only UNPACK and PACK. The host image serves no row-major fit.
struct _GbdtHostImage(GbdtImage):
    @staticmethod
    def fit(row_major: Bool, args: GbdtFitArgs) raises -> GbdtFitOutcome:
        if row_major:
            raise Error("AttributeError: mojolearn: this gbdt binding has no gbdt_fit_rowmajor")
        return gbdt_fit_native(args)

    @staticmethod
    def row_major() -> Bool:
        return False

    @staticmethod
    def model_dim(text: String) raises -> Int:
        return _parse_model(text).dim


def gbdt_host_entry_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """This image's gbdt entry (`GBDT_ENTRY_ABI`, gbdt/fit_request.mojo)."""
    return gbdt_entry_dispatch[_GbdtHostImage](operation, aa, na, pp, np, ff, nf, result, context, sink)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_gbdt_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_gbdt_host() -> PythonObject:
    comptime assert TREES_GUARDS_OK, "invalid integrated experiment configuration (trees guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_gbdt_host")
        # The native gbdt entry (core/gbdt_entry_table.mojo) under the cpu key.
        gbdt_entry_publish["MojolearnGbdtEntryOwnerHostV1"]("cpu", gbdt_host_entry_v1)
        module.def_function[gbdt_host_numeric_mode_binding]("gbdt_host_numeric_mode")
        module.def_function[gbdt_host_vendor_binding]("gbdt_host_vendor")
        module.def_function[gbdt_host_column_binding]("gbdt_host_column")
        module.def_function[gbdt_host_sabotage_binding]("gbdt_host_sabotage")
        module.def_function[gbdt_vendor_binding]("gbdt_vendor")
        module.def_function[gbdt_numeric_mode_binding]("gbdt_numeric_mode")
        module.def_function[gbdt_fit_binding]("gbdt_fit")
        module.def_function[gbdt_predict_binding]("gbdt_predict")
        module.def_function[gbdt_predict_multi_binding]("gbdt_predict_multi")
        module.def_function[gbdt_model_dim_binding]("gbdt_model_dim")
        # TOMBSTONE: gbdt_sigmoid (dead native entry: no live caller) deleted 2026-10-26 by 7c1201602; code recoverable at 485ba9cc1.
        # Restore: git apply experiments/removed/gbdt_sigmoid.patch
        module.def_function[gbdt_sigmoid_pair_binding]("gbdt_sigmoid_pair")
        module.def_function[gbdt_binary_probabilities_binding]("gbdt_binary_probabilities")
        module.def_function[gbdt_binary_classes_binding]("gbdt_binary_classes")
        module.def_function[gbdt_fit_two_level_feature_freq_binding]("gbdt_fit_two_level_feature_freq")
        module.def_function[gbdt_fit_ordered_rmse_binding]("gbdt_fit_ordered_rmse")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_gbdt_host: ", error))
