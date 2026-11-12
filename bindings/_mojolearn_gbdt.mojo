# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the gradient-boosted oblivious trees.

Built by `bindings/build_gbdt.sh` into `python/mojolearn/_mojolearn_gbdt.so`.
The public Python surface is `python/mojolearn/ensemble.py`.

WHY GBDT HAS ITS OWN EXTENSION
-------------------------------
Mojo 1.0.0 (ed45d567) decides how many kernels it compiles ahead of time from
THE BASENAME OF THE ENTRY FILE -- see the long comment in
`bindings/build.sh`. That is an upstream defect with no fix here, only a
workaround: compile a copy under a measured basename and CHECK the artifact.

The workaround was failing. Measured across this repository's history, with
the same stem list, the GBDT blob count in the COMBINED extension is not
being knocked out by one commit; it is DECLINING AS THE MODULE GROWS:

    shipped .so (older source, works)   gbdt 85   total 113
    at 2cb82ac~1                        gbdt 73   total 101
    at 9ab10bc                          gbdt 58   total  86
    at HEAD 2026-08-21                  gbdt 56   total  84

A continuous decline is not fixed by adding stems and not fixed by removing
one kernel family. It is fixed by making the entry module smaller. So GBDT
gets its own, exactly as `bindings/_mojolearn_estimators.mojo` gave DBSCAN,
PCA, tSVD and OLS theirs -- and for the second reason that file states too:
an independently changing binding stops being a merge point.

The two extensions are independent artifacts. `_mojolearn.so` keeps k-means
and k-NN; nothing here imports `cluster.` or `neighbors.`, and that is the
whole point of the split. Do not add an import from either.

EVERYTHING BELOW MIRRORS `bindings/_mojolearn.mojo`'s GBDT half
----------------------------------------------------------------
Data crosses as raw buffer addresses plus lengths; the model crosses as TEXT
rather than as a handle, because an integer handle into a table of live models
would put object lifetime across the CPython boundary where nothing here can
check it. A `DeviceContext` is constructed per call and that cost is real, not
hidden. Scalars arrive as ONE LIST because `PythonModuleBuilder.def_function`
infers its signature from arity and stops being able to above roughly nine
arguments -- `gbdt_fit` needs far more than nine. The order is written out
beside the unpacking and mirrored in the wrapper, in the same words, because a
silent reordering here is a wrong answer rather than a failure. The GIL is
released around the device work: nothing inside touches a Python object and
the wrapper holds the caller's arrays for the length of the call.
"""

from core.experiment_guards_trees import TREES_GUARDS_OK
from bindings.device_plan_binding import device_plan_enter_binding, device_plan_exit_binding, device_plan_log_binding
from bindings.py_lists import py_list_int
from bindings.gbdt_fit_boundary import gbdt_fit_args_py, gbdt_fit_outcome_py
from gbdt.gbdt_entry import gbdt_entry_publish, gbdt_entry_dispatch, GbdtImage
from core.native_descriptor_abi import NativeDescriptorError
from gbdt.fit_request import GbdtFitArgs, GbdtFitOutcome
from gbdt.methods.ordered_fast_switches import ordered_doc_count
from std.os import abort
from std.python import Python, PythonObject
from std.sys.compile import is_defined
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.vendor import COMPILED_VENDOR
from gbdt.gpu_data.sym_feat_switches import GBDT_QUANT_DEVICE
from gbdt.binary_prediction import binary_prediction_device, sigmoid_f64_device

from max.gpu.host import DeviceContext
from core.neural_context import process_ctx

from gbdt.estimator import (
    gbdt_fit_row_major_available,
    gbdt_model_dim,
    gbdt_per_round_paths,
    gbdt_predict,
    gbdt_predict_multi,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    identical_exp64,
)
from gbdt.resident_model import (
    ResidentGbdtModel,
    gbdt_resident_info,
    gbdt_resident_release,
)
from gbdt.mojolearn_gbdt_run import (
    _DEVCTX_SLOT,
    _f32_ptr,
    _f64_ptr,
    _gbdt_device_plan,
    _gbdt_lock,
    _gbdt_lock_word,
    _gbdt_locked_insert_run,
    _gbdt_locked_predict,
    _gbdt_unlock,
    _i64_ptr,
    _u32_ptr,
    gbdt_fit_native_run,
    gbdt_fit_ordered_rmse_run,
    gbdt_fit_two_level_feature_freq_run,
)


def gbdt_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST,
    1 IDENTICAL."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def gbdt_per_round_paths_binding() raises -> PythonObject:
    """Which side of each gbdt speed switch this binary compiled
    (DEVIATIONS 2550, 2551, 2580, 2581), printed by the benchmark beside its timing
    (CONTRIBUTING.md (Non-default paths))."""
    return PythonObject(gbdt_per_round_paths())


def gbdt_binary_prediction_binding[probabilities: Bool, dtype: DType](
    raw_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    # params=[n]; input Float32 margins. Output probabilities n*2 Float32,
    # or class codes n Int32. Both perform the arithmetic/selection on GPU.
    if len(params) != 1:
        raise Error("binary prediction: params must contain n")
    var n = Int(py=params[0])
    if n <= 0 or n > 2147483647:
        raise Error("binary prediction: positive n<=Int32.max required")
    var raw_address = Int(py=raw_addr)
    var address = Int(py=out_addr)
    if raw_address == 0:
        raise Error("mojolearn: null buffer address")
    if address == 0:
        raise Error("binary prediction: null output")
    var count = 2*n if probabilities else n
    with GILReleased(Python()):
        # cpu2-l6-bindings: upload from the caller's pointer, device
        # finiteness scan, output copied straight into the caller's buffer
        binary_prediction_device[probabilities,dtype](raw_address,address,n)
    return PythonObject(count)


# TOMBSTONE: gbdt_sigmoid (dead native entry: no live caller; every tier's link runs gbdt_sigmoid_pair) deleted 2026-10-26 by 7c1201602; code recoverable at 485ba9cc1.
# Restore: git apply experiments/removed/gbdt_sigmoid.patch


#: The identity gate's negative control for `gbdt_sigmoid_pair`: the same
#: define the forest and GBDT host walks take (`core/forest_host_predict.mojo`),
#: passed to this build through MOJOLEARN_EXTRA_DEFINES. A build with it
#: writes the two probability columns swapped.
comptime GBDT_PAIR_SABOTAGE = is_defined["MOJOLEARN_FOREST_HOST_SABOTAGE"]()


def gbdt_sigmoid_pair_binding(
    raw_addr: PythonObject, out_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """DEVIATION 2902 (lane/infer-speed-trees, 2026-09-17): both columns of
    a Logloss or CrossEntropy `predict_proba` in one pass, `out[2 * i] =
    1 - p` and `out[2 * i + 1] = p` with `p = 1 / (1 + exp(-raw[i]))`
    in double through `identical_exp64`, `n` rows, both buffers float64 and the caller's. `1.0 - p` is
    the one IEEE double subtraction the Python layer computed per row
    (DEVIATION 2333, retired where the binary carries this entry point);
    a lone subtraction has no fusion partner and no association, so the
    column's bits are the Python column's on every host."""
    var count = Int(py=n)
    if count < 0:
        raise Error("gbdt_sigmoid_pair: n must be non-negative")
    var raw_address = Int(py=raw_addr)
    var out_address = Int(py=out_addr)
    if count > 0 and (raw_address == 0 or out_address == 0):
        raise Error("mojolearn: null buffer address")
    with GILReleased(Python()):
        # cpu2-l6-bindings: `resident_link_kernel`'s pair, soft binary64,
        # on the device (`sigmoid_f64_kernel`); same words as the host loop
        sigmoid_f64_device(
            raw_address, out_address, count, True, GBDT_PAIR_SABOTAGE
        )
    return PythonObject(count)


def _gbdt_locked_insert(var entry: ResidentGbdtModel) raises -> Int:
    """`gbdt_resident_prepare`'s registry step, under the lock word: the
    lock is taken with the GIL released, the insert is RUN."""
    var word = _gbdt_lock_word()
    with GILReleased(Python()):
        _gbdt_lock(word)
    return _gbdt_locked_insert_run(word, entry^)


def gbdt_fit_native(row_major: Bool, params: GbdtFitArgs) raises -> GbdtFitOutcome:
    """Fit a gradient-boosted ensemble: `gbdt_fit_native_run`
    (gbdt/mojolearn_gbdt_run.mojo, which documents the slots) with the GIL
    released around the whole typed call."""
    var outcome: GbdtFitOutcome
    with GILReleased(Python()):
        outcome = gbdt_fit_native_run(row_major, params)
    return outcome^


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
    """`gbdt_fit`: X column-major, as documented on `gbdt_fit_native`.
    UNPACK and PACK only (bindings/gbdt_fit_boundary.mojo); the GBDT doors
    call the fit natively through this image's table entry."""
    var args = gbdt_fit_args_py(x_addr, y_addr, weights_addr, cat_flags_addr, eval_x_addr, eval_y_addr, params, strs)
    return gbdt_fit_outcome_py(gbdt_fit_native(False, args))


def gbdt_fit_rowmajor_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    weights_addr: PythonObject,
    cat_flags_addr: PythonObject,
    eval_x_addr: PythonObject,
    eval_y_addr: PythonObject,
    params: PythonObject,
    strs: PythonObject,
) raises -> PythonObject:
    """lane/apple-fast-sym-feat: `gbdt_fit` with X ROW-MAJOR (a C-order
    float32 array handed over as it is; the device transposes it under
    `GBDT_QUANT_DEVICE`). Every other argument is `gbdt_fit`'s; the eval
    set stays column-major. Served only when
    `gbdt_fit_row_major_available()` says so (raw float columns only, so
    `params[3]` must be 0); the Python layer asks first."""
    var args = gbdt_fit_args_py(x_addr, y_addr, weights_addr, cat_flags_addr, eval_x_addr, eval_y_addr, params, strs)
    return gbdt_fit_outcome_py(gbdt_fit_rowmajor_native(args))


def gbdt_fit_rowmajor_native(args: GbdtFitArgs) raises -> GbdtFitOutcome:
    """The row-major fit's own refusal, then `gbdt_fit_native`."""
    if args.i(3) != 0:
        raise Error(
            "gbdt_fit_rowmajor takes raw float columns only; pass categorical"
            " or one-hot features through gbdt_fit"
        )
    return gbdt_fit_native(True, args)


def gbdt_fit_row_major_available_binding() raises -> PythonObject:
    """lane/apple-fast-sym-feat: 1 when this binary serves
    `gbdt_fit_rowmajor` (FAST + Apple with `-D MOJOLEARN_GBDT_QUANT_DEVICE`
    or a switch that implies it), else 0."""
    return PythonObject(1 if gbdt_fit_row_major_available() else 0)


def gbdt_predict_binding(
    model: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Apply a model returned by `gbdt_fit`. Returns rows written.

    `params` is `[n_rows]`. The feature count is not passed: it comes from
    the MODEL, which is the only place that cannot disagree with the
    quantization grid the model was fitted on.

    PREDICTIONS ARE RAW APPROXES for every loss, exactly as their `predict`
    without a `prediction_type` is. A Logloss caller applies the sigmoid.
    """
    if len(params) != 1:
        raise Error(
            "gbdt_predict: params must hold [n_rows], got "
            + String(len(params))
        )
    var text = String(py=model)
    var xp = _f32_ptr(Int(py=x_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var n_rows = Int(py=params[0])

    var wrote: Int
    with GILReleased(Python()):
        var ctx = process_ctx[_DEVCTX_SLOT]()
        wrote = gbdt_predict(ctx, text, xp, n_rows, op)
        ctx.synchronize()
    return PythonObject(wrote)


def gbdt_model_dim_binding(model: PythonObject) raises -> PythonObject:
    """The model's approx dimension: 1, or `numClasses - 1` for MultiClass.

    The wrapper calls this once after `fit` to size its output arrays and
    to recover `n_classes` as `dim + 1`. It parses the model text, which
    is the price of the handle-free boundary `gbdt/estimator.mojo`
    argues for.
    """
    return PythonObject(gbdt_model_dim(String(py=model)))


def gbdt_predict_multi_binding(
    model: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Apply a multi-dimensional model. Returns the width written.

    `params` is `[n_rows, mode]`, where mode is 0 RAW, 1 SOFTMAX, 2
    SIGMOID -- their `RawFormulaVal`, `Probability` and `MultiProbability`
    (`libs/model/eval_processing.h:186-226`).

    RAW and SIGMOID write `n_rows * dim`; SOFTMAX writes
    `n_rows * (dim + 1)`, because MultiClass's pinned class is a real
    class whose approx is zero. The caller must size `out_addr` for the
    widest case it may ask for; the return says which it got.
    """
    if len(params) != 2:
        raise Error(
            "gbdt_predict_multi: params must hold [n_rows,"
            " as_probabilities], got " + String(len(params))
        )
    var text = String(py=model)
    var xp = _f32_ptr(Int(py=x_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var n_rows = Int(py=params[0])
    var mode = Int(py=params[1])

    var width: Int
    with GILReleased(Python()):
        var ctx = process_ctx[_DEVCTX_SLOT]()
        width = gbdt_predict_multi(ctx, text, xp, n_rows, op, mode)
        ctx.synchronize()
    return PythonObject(width)


def gbdt_resident_prepare_binding(model: PythonObject) raises -> PythonObject:
    """DEVIATION 2980 (lane/gbdt-resident-predict, 2026-09-17): parse the
    model text once, pack and upload it once, and return the integer
    handle of the process-wide registry entry (`gbdt/resident_model.mojo`).
    The wrapper keeps the handle on the instance beside the sha256 of the
    text it was prepared from, releases it on refit and never pickles it.
    The GIL is held around the registry and released for the parse and
    the uploads, as `gbdt_predict` releases it for its work. Every
    registry step takes the registry lock word (see `_GbdtRegistryLock`)."""
    var text = String(py=model)
    # Parse, pack and upload with the GIL released and outside the registry
    # lock; only the insert is locked (hazard B note above).
    var entry: ResidentGbdtModel
    with GILReleased(Python()):
        entry = ResidentGbdtModel(text)
    return PythonObject(_gbdt_locked_insert(entry^))


def gbdt_resident_release_binding(handle: PythonObject) raises -> PythonObject:
    """Drop a prepared model. A released or unknown handle raises."""
    var h = Int(py=handle)
    var word = _gbdt_lock_word()
    with GILReleased(Python()):
        _gbdt_lock(word)
    try:
        gbdt_resident_release(h)
    except error:
        _gbdt_unlock(word)
        raise error^
    _gbdt_unlock(word)
    return PythonObject(None)


def gbdt_resident_info_binding(handle: PythonObject) raises -> PythonObject:
    """`[approx_dim, n_input_features, workspace_rows, oblivious, n_trees]`
    of a live handle."""
    var h = Int(py=handle)
    var word = _gbdt_lock_word()
    with GILReleased(Python()):
        _gbdt_lock(word)
    var info: List[Int]
    try:
        info = gbdt_resident_info(h)
    except error:
        _gbdt_unlock(word)
        raise error^
    _gbdt_unlock(word)
    return py_list_int(info)


def gbdt_resident_predict_binding(
    handle: PythonObject,
    x_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Apply a prepared model. Returns the width written per row.

    `params` is `[n_rows, mode, row_major]` with mode 0 RAW, 1 SOFTMAX, 2
    SIGMOID as `gbdt_predict_multi` takes them, plus 3 SIGMOID_PAIR: the
    Logloss and CrossEntropy `predict_proba` columns `[1 - p, p]` written
    as FLOAT64 to `out_addr` (`gbdt_sigmoid_pair`'s two statements over
    the exact widening of the raw float32 value). Modes 4..6 write int64
    class codes; every other mode writes float32 to `out_addr`, `n_rows *
    width` values row-major, exactly as `gbdt_predict` and
    `gbdt_predict_multi` write them. `row_major` (an
    optional third entry, 0 when absent) says `x` is the C-order
    `[row * n_features + f]` block rather than the column-major one; the
    staging pass transposes it. The feature count comes from the prepared
    model. The caller holds `x` and `out` for the length of the call."""
    if len(params) != 2 and len(params) != 3:
        raise Error(
            "gbdt_resident_predict: params must hold [n_rows, mode] or"
            " [n_rows, mode, row_major], got " + String(len(params))
        )
    var h = Int(py=handle)
    var addr = Int(py=out_addr)
    var xp = _f32_ptr(Int(py=x_addr))
    var n_rows = Int(py=params[0])
    var mode = Int(py=params[1])
    var row_major = False
    if len(params) == 3:
        row_major = Int(py=params[2]) != 0
    var op32 = _f32_ptr(addr)
    var op64 = _f64_ptr(addr)
    var op64i = _i64_ptr(addr)
    var width: Int
    with GILReleased(Python()):
        width = _gbdt_locked_predict(h, xp, n_rows, op32, op64, op64i, mode, row_major)
    return PythonObject(width)


def gbdt_resident_binary_classes_binding(
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """FAST resident Logloss/CrossEntropy codes into caller-owned Int64."""
    if len(params) != 2:
        raise Error("gbdt_resident_binary_classes requires [n_rows, row_major]")
    var h = Int(py=handle)
    var n_rows = Int(py=params[0])
    var row_major = Int(py=params[1]) != 0
    if n_rows <= 0:
        raise Error("invalid resident binary class prediction row count")
    var xp = _f32_ptr(Int(py=x_addr))
    var out_address = Int(py=out_addr)
    var op32 = _f32_ptr(out_address)
    var op64 = _f64_ptr(out_address)
    var op64i = _i64_ptr(out_address)
    var width: Int
    with GILReleased(Python()):
        width = _gbdt_locked_predict(
            h, xp, n_rows, op32, op64, op64i, 4, row_major
        )
    if width != 1:
        raise Error("resident binary class prediction returned an invalid width")
    return PythonObject(n_rows)


def gbdt_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


def gbdt_fit_two_level_feature_freq_binding(
    x_addr: PythonObject,
    y_addr: PythonObject,
    weights_addr: PythonObject,
    sources_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Explicit mixed numeric/categorical two-level FeatureFreq fit.

    `params` is `[n_rows, n_features, n_weights, n_sources, learning_rate,
    l2_leaf_reg, random_seed]`; UInt32 source indices identify the dense
    categorical columns and every other column is numeric.
    """
    if len(params) != 7:
        raise Error("two-level FeatureFreq params must have seven values")
    var xp = _f32_ptr(Int(py=x_addr))
    var yp = _f32_ptr(Int(py=y_addr))
    var wp = _f32_ptr(Int(py=weights_addr))
    var sp = _u32_ptr(Int(py=sources_addr))
    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_weights = Int(py=params[2])
    var n_sources = Int(py=params[3])
    var learning_rate = Float32(Float64(py=params[4]))
    var l2_leaf_reg = Float32(Float64(py=params[5]))
    var random_seed = UInt64(Int(py=params[6]))
    var text: String
    var device_plan = _gbdt_device_plan()
    with GILReleased(Python()):
        text = gbdt_fit_two_level_feature_freq_run(
            xp, n_rows, n_features, yp, wp, n_weights, sp, n_sources,
            learning_rate, l2_leaf_reg, random_seed, device_plan,
        )
    return PythonObject(text)


def gbdt_fit_ordered_rmse_binding(
    x_addr: PythonObject, y_addr: PythonObject,
    weights_addr: PythonObject, permutation_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Numeric, single-permutation ordered RMSE; return normal model text.

    params: [n_rows, n_features, n_weights, n_permutation, n_estimators,
             max_depth, border_count, learning_rate, l2_leaf_reg].
    The permutation buffer holds UInt32 original row ids. All Python
    conversions finish before releasing the GIL; buffers remain borrowed
    from live arrays held by the wrapper for the duration of this call.
    """
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
    if n_rows < 4 or n_features < 1 or n_permutation != n_rows:
        raise Error("ordered RMSE requires >=4 rows, features and a full permutation")
    if n_weights != 0 and n_weights != n_rows:
        raise Error("ordered RMSE sample weight shape mismatch")
    _ = _f32_ptr(Int(py=x_addr))
    var yp = _f32_ptr(Int(py=y_addr))
    var wp = _f32_ptr(Int(py=weights_addr))
    var pp = _u32_ptr(Int(py=permutation_addr))
    var text: String
    var x_address = Int(py=x_addr)
    var device_plan = _gbdt_device_plan()
    with GILReleased(Python()):
        text = gbdt_fit_ordered_rmse_run(
            x_address, yp, wp, pp, n_rows, n_features, n_weights,
            n_estimators, max_depth, border_count, learning_rate,
            l2_leaf_reg, device_plan,
        )
    return PythonObject(text)


def ordered_doc_count_binding() raises -> PythonObject:
    return PythonObject(ordered_doc_count())


# ---- The image's native gbdt entry (lane calltable-gbdt-forest-dart,
# 2026-11-11; ABI gbdt/fit_request.mojo, table core/gbdt_entry_table.mojo):
# the GBDT doors fit, probe and read the model dimension through it instead
# of the Python entries above, which remain for the unconverted Python
# callers (python/mojolearn/ensemble*.py) and only UNPACK and PACK.
struct _GbdtGpuImage(GbdtImage):
    @staticmethod
    def fit(row_major: Bool, args: GbdtFitArgs) raises -> GbdtFitOutcome:
        if row_major:
            comptime if GBDT_QUANT_DEVICE:
                return gbdt_fit_rowmajor_native(args)
            raise Error("AttributeError: mojolearn: this gbdt binding has no gbdt_fit_rowmajor")
        return gbdt_fit_native(False, args)

    @staticmethod
    def row_major() -> Bool:
        comptime if GBDT_QUANT_DEVICE:
            return gbdt_fit_row_major_available()
        return False

    @staticmethod
    def model_dim(text: String) raises -> Int:
        return gbdt_model_dim(text)


def gbdt_gpu_entry_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """This image's gbdt entry (`GBDT_ENTRY_ABI`, gbdt/fit_request.mojo)."""
    return gbdt_entry_dispatch[_GbdtGpuImage](operation, aa, na, pp, np, ff, nf, result, context, sink)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_gbdt.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_gbdt() -> PythonObject:
    comptime assert TREES_GUARDS_OK, "invalid integrated experiment configuration (trees guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_gbdt")
        # The native gbdt entry (core/gbdt_entry_table.mojo) under this image's key.
        gbdt_entry_publish["MojolearnGbdtEntryOwnerGpuV1"](COMPILED_VENDOR, gbdt_gpu_entry_v1)
        m.def_function[ordered_doc_count_binding]("gbdt_ordered_doc_count")
        m.def_function[device_plan_enter_binding]("device_plan_enter")
        m.def_function[device_plan_exit_binding]("device_plan_exit")
        m.def_function[device_plan_log_binding]("device_plan_log")
        # TOMBSTONE: gbdt_parallel_available, pointwise_parallel_available (dead native entries: constant probes no driver reads) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        m.def_function[gbdt_vendor_binding]("gbdt_vendor")
        m.def_function[gbdt_fit_binding]("gbdt_fit")
        # lane/apple-fast-sym-feat: the row-major door and its probe, only
        # in a FAST Apple build with the device quantizer; every other
        # binary registers main's functions (Python treats a missing probe
        # as unavailable)
        comptime if GBDT_QUANT_DEVICE:
            m.def_function[gbdt_fit_rowmajor_binding]("gbdt_fit_rowmajor")
            m.def_function[gbdt_fit_row_major_available_binding](
                "gbdt_fit_row_major_available"
            )
        m.def_function[gbdt_fit_ordered_rmse_binding]("gbdt_fit_ordered_rmse")
        m.def_function[gbdt_fit_two_level_feature_freq_binding](
            "gbdt_fit_two_level_feature_freq"
        )
        m.def_function[gbdt_predict_binding]("gbdt_predict")
        m.def_function[gbdt_model_dim_binding]("gbdt_model_dim")
        m.def_function[gbdt_predict_multi_binding]("gbdt_predict_multi")
        m.def_function[gbdt_numeric_mode_binding]("gbdt_numeric_mode")
        m.def_function[gbdt_per_round_paths_binding]("gbdt_per_round_paths")
        m.def_function[gbdt_binary_prediction_binding[True,DType.float32]]("gbdt_binary_probabilities")
        m.def_function[gbdt_binary_prediction_binding[False,DType.int32]]("gbdt_binary_classes")
        # TOMBSTONE: gbdt_sigmoid (dead native entry: no live caller) deleted 2026-10-26 by 7c1201602; code recoverable at 485ba9cc1.
        # Restore: git apply experiments/removed/gbdt_sigmoid.patch
        m.def_function[gbdt_sigmoid_pair_binding]("gbdt_sigmoid_pair")
        # DEVIATION 2980: the device-resident parsed model
        m.def_function[gbdt_resident_prepare_binding]("gbdt_resident_prepare")
        m.def_function[gbdt_resident_predict_binding]("gbdt_resident_predict")
        m.def_function[gbdt_resident_binary_classes_binding]("gbdt_resident_binary_classes")
        m.def_function[gbdt_resident_release_binding]("gbdt_resident_release")
        m.def_function[gbdt_resident_info_binding]("gbdt_resident_info")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_gbdt module: ", e))


# TOMBSTONE: gbdt_parallel_available, pointwise_parallel_available (dead native entries: constant probes no driver reads) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_trees.patch

