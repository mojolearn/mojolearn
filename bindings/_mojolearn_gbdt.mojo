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
from bindings.device_plan_binding_pack import device_plan_enter_binding, device_plan_exit_binding, device_plan_log_binding
from bindings.py_lists import py_list_int
from bindings.gbdt_fit_boundary_unpack import gbdt_fit_args_py
from bindings.gbdt_fit_boundary_pack import gbdt_fit_outcome_py
from gbdt.gbdt_entry import gbdt_entry_publish, gbdt_entry_dispatch, gbdt_entry_releases, GbdtImage
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_abi import NativeDescriptorError
from gbdt.fit_request import GbdtFitArgs, GbdtFitOutcome
from gbdt.methods.ordered_fast_switches import ordered_doc_count
from std.os import abort
from std.python import Python, PythonObject
from std.sys.compile import is_defined
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.vendor import COMPILED_VENDOR
from core.gbdt_member import GbdtMemberImage, gbdt_member_publish_image
from core.fold_x_scan import fold_x_first_nan
from gbdt.gpu_data.sym_feat_switches import GBDT_QUANT_DEVICE
from gbdt.binary_prediction import binary_prediction_device, sigmoid_f64_device
from gbdt.gpu_importance import gbdt_feature_importances_device

from max.gpu.host import DeviceContext
from core.neural_context import process_ctx

from gbdt.estimator import (
    gbdt_fit_row_major_available,
    gbdt_model_dim,
    gbdt_per_round_paths,
    gbdt_predict,
    gbdt_predict_multi,
)
from checks.mode_policy import mode_policy_mask
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


def gbdt_mode_policy_mask_binding() raises -> PythonObject:
    """The mode policy rows this binary was compiled with
    (`checks/mode_policy.mojo`): bit i set = row i relaxed. 0 in every
    identical build. Read by `python -m mojolearn verify`, which refuses an
    identical set that reports anything else."""
    return PythonObject(Int(mode_policy_mask()))


def gbdt_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST,
    1 IDENTICAL."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def gbdt_per_round_paths_binding() raises -> PythonObject:
    """Which side of each gbdt speed switch this binary compiled
    (DEVIATIONS 2550, 2551, 2580, 2581), printed by the benchmark beside its timing
    (CONTRIBUTING.md (Non-default paths))."""
    return PythonObject(gbdt_per_round_paths())


def gbdt_binary_native(probabilities: Bool, raw_address: Int, address: Int, n: Int) raises -> Int:
    """The adapters' binary transforms (XG_BINARY): n Float32 margins to the
    Float32 pair `[1 - p, p]` (2 n values) or the Int32 class code (n).
    Both perform the arithmetic/selection on GPU. GIL-free (see
    `gbdt_gpu_entry_v1`): the caller released it or holds none."""
    if n <= 0 or n > 2147483647:
        raise Error("binary prediction: positive n<=Int32.max required")
    if raw_address == 0:
        raise Error("mojolearn: null buffer address")
    if address == 0:
        raise Error("binary prediction: null output")
    # cpu2-l6-bindings: upload from the caller's pointer, device
    # finiteness scan, output copied straight into the caller's buffer
    if probabilities:
        binary_prediction_device[True, DType.float32](raw_address, address, n)
    else:
        binary_prediction_device[False, DType.int32](raw_address, address, n)
    return 2 * n if probabilities else n


def gbdt_binary_prediction_binding[probabilities: Bool](
    raw_addr: PythonObject, out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    # The Python form of XG_BINARY, kept for the direct kernel check
    # checks/gbdt_binary_prediction_smoke.py (tools/check_gbdt_adapters.sh),
    # which loads this image alone: params=[n]; UNPACK only. No product
    # operation calls it.
    if len(params) != 1:
        raise Error("binary prediction: params must contain n")
    var n = Int(py=params[0])
    var raw = Int(py=raw_addr)
    var dst = Int(py=out_addr)
    var wrote = 0
    with GILReleased(Python()):
        wrote = gbdt_binary_native(probabilities, raw, dst, n)
    return PythonObject(wrote)


# TOMBSTONE: gbdt_sigmoid_pair, gbdt_predict, gbdt_predict_multi, gbdt_resident_predict, gbdt_resident_binary_classes (Python entries of this image: no caller left, the GBDT door calls the typed operations XG_BINARY, XG_SIGMOID_PAIR, XG_PREDICT, XG_PREDICT_MULTI, XG_RESIDENT_PREDICT of this image's native entry; kernels unchanged) deleted 2026-11-21 by lane/dw-trees-boost; code recoverable at 5544c66aae.
# Restore: git apply experiments/removed/dead_entries_gbdt_predict_calls.patch


# TOMBSTONE: gbdt_sigmoid (dead native entry: no live caller; every tier's link runs gbdt_sigmoid_pair) deleted 2026-10-26 by 7c1201602; code recoverable at 485ba9cc1.
# Restore: git apply experiments/removed/gbdt_sigmoid.patch


#: The identity gate's negative control for `gbdt_sigmoid_pair`: the same
#: define the forest and GBDT host walks take (`core/forest_host_predict.mojo`),
#: passed to this build through MOJOLEARN_EXTRA_DEFINES. A build with it
#: writes the two probability columns swapped.
comptime GBDT_PAIR_SABOTAGE = is_defined["MOJOLEARN_FOREST_HOST_SABOTAGE"]()


def gbdt_sigmoid_pair_native(raw_address: Int, out_address: Int, count: Int) raises -> Int:
    """DEVIATION 2902 (lane/infer-speed-trees, 2026-09-17): both columns of
    a Logloss or CrossEntropy `predict_proba` in one pass, `dst[2 * i] =
    1 - p` and `dst[2 * i + 1] = p` with `p = 1 / (1 + exp(-raw[i]))`
    in double through `identical_exp64`, `n` rows, both buffers float64 and the caller's. `1.0 - p` is
    the one IEEE double subtraction the Python layer computed per row
    (DEVIATION 2333, retired where the binary carries this entry point);
    a lone subtraction has no fusion partner and no association, so the
    column's bits are the Python column's on every host. XG_SIGMOID_PAIR."""
    if count < 0:
        raise Error("gbdt_sigmoid_pair: n must be non-negative")
    if count > 0 and (raw_address == 0 or out_address == 0):
        raise Error("mojolearn: null buffer address")
    # cpu2-l6-bindings: `resident_link_kernel`'s pair, soft binary64,
    # on the device (`sigmoid_f64_kernel`); same words as the host loop
    sigmoid_f64_device(
        raw_address, out_address, count, True, GBDT_PAIR_SABOTAGE
    )
    return count


def _gbdt_locked_insert(var entry: ResidentGbdtModel) raises -> Int:
    """`gbdt_resident_prepare`'s registry step, under the lock word. The
    caller holds no GIL here (the entry released it, or a member's worker
    never had it), so the lock is never waited on with the GIL held."""
    var word = _gbdt_lock_word()
    _gbdt_lock(word)
    return _gbdt_locked_insert_run(word, entry^)


def gbdt_fit_native(row_major: Bool, params: GbdtFitArgs) raises -> GbdtFitOutcome:
    """Fit a gradient-boosted ensemble: `gbdt_fit_native_run`
    (gbdt/mojolearn_gbdt_run.mojo, which documents the slots). GIL-free:
    the image's entry (`gbdt_gpu_entry_v1`) or the Python entry below
    releases it around the whole typed call; a member's worker holds none."""
    return gbdt_fit_native_run(row_major, params)


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
    var outcome: GbdtFitOutcome
    with GILReleased(Python()):
        outcome = gbdt_fit_native(False, args)
    return gbdt_fit_outcome_py(outcome^)


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
    var outcome: GbdtFitOutcome
    with GILReleased(Python()):
        outcome = gbdt_fit_rowmajor_native(args)
    return gbdt_fit_outcome_py(outcome^)


def gbdt_fit_rowmajor_native(args: GbdtFitArgs) raises -> GbdtFitOutcome:
    """The row-major fit's own refusal, then `gbdt_fit_native`."""
    if args.i(3) != 0:
        raise Error(
            "gbdt_fit_rowmajor takes raw float columns only; pass categorical"
            " or one-hot features through gbdt_fit"
        )
    return gbdt_fit_native(True, args)


# TOMBSTONE: MOJOLEARN_GBDT_QUANT_DEVICE (EXP:1303 NEUTRAL; prerequisite of a never-run Apple FAST card) deleted 2026-11-15 by lane/expdel2-38-trees; code recoverable at 52922acecb.
# Restore: git apply experiments/removed/MOJOLEARN_GBDT_QUANT_DEVICE.expdel2-38-trees.patch; record in docs/TOMBSTONES.md.
def gbdt_fit_row_major_available_binding() raises -> PythonObject:
    """lane/apple-fast-sym-feat: 1 when this binary serves
    `gbdt_fit_rowmajor` (FAST + Apple with a switch that implies
    `GBDT_QUANT_DEVICE`), else 0."""
    return PythonObject(1 if gbdt_fit_row_major_available() else 0)


def gbdt_predict_native(text: String, x_address: Int, n_rows: Int, out_address: Int) raises -> Int:
    """Apply a model returned by `gbdt_fit` (XG_PREDICT). Returns rows written.

    The feature count is not passed: it comes from the MODEL, which is the
    only place that cannot disagree with the quantization grid the model
    was fitted on.

    PREDICTIONS ARE RAW APPROXES for every loss, exactly as their `predict`
    without a `prediction_type` is. A Logloss caller applies the sigmoid.
    """
    var xp = _f32_ptr(x_address)
    var op = _f32_ptr(out_address)
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var wrote = gbdt_predict(ctx, text, xp, n_rows, op)
    ctx.synchronize()
    return wrote


def gbdt_model_dim_binding(model: PythonObject) raises -> PythonObject:
    """The model's approx dimension: 1, or `numClasses - 1` for MultiClass.

    The wrapper calls this once after `fit` to size its output arrays and
    to recover `n_classes` as `dim + 1`. It parses the model text, which
    is the price of the handle-free boundary `gbdt/estimator.mojo`
    argues for.
    """
    return PythonObject(gbdt_model_dim(String(py=model)))


def gbdt_predict_multi_native(text: String, x_address: Int, n_rows: Int, out_address: Int, mode: Int) raises -> Int:
    """Apply a multi-dimensional model (XG_PREDICT_MULTI). Returns the width
    written.

    `mode` is 0 RAW, 1 SOFTMAX, 2 SIGMOID -- their `RawFormulaVal`,
    `Probability` and `MultiProbability`
    (`libs/model/eval_processing.h:186-226`).

    RAW and SIGMOID write `n_rows * dim`; SOFTMAX writes
    `n_rows * (dim + 1)`, because MultiClass's pinned class is a real
    class whose approx is zero. The caller must size the output for the
    widest case it may ask for; the return says which it got.
    """
    var xp = _f32_ptr(x_address)
    var op = _f32_ptr(out_address)
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var width = gbdt_predict_multi(ctx, text, xp, n_rows, op, mode)
    ctx.synchronize()
    return width


def gbdt_resident_prepare_native(text: String) raises -> Int:
    """XG_RESIDENT_PREPARE: parse, pack and upload outside the registry
    lock; only the insert is locked. GIL-free (see `gbdt_gpu_entry_v1`)."""
    var entry = ResidentGbdtModel(text)
    return _gbdt_locked_insert(entry^)


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
    var handle = 0
    with GILReleased(Python()):
        handle = gbdt_resident_prepare_native(text)
    return PythonObject(handle)


def _gbdt_registry_release(handle: Int) raises:
    """The registry's release under its lock word. GIL-free."""
    var word = _gbdt_lock_word()
    _gbdt_lock(word)
    try:
        gbdt_resident_release(handle)
    except error:
        _gbdt_unlock(word)
        raise error^
    _gbdt_unlock(word)


def _gbdt_registry_info(handle: Int) raises -> List[Int]:
    """The registry's info words under its lock word. GIL-free."""
    var word = _gbdt_lock_word()
    _gbdt_lock(word)
    var info: List[Int]
    try:
        info = gbdt_resident_info(handle)
    except error:
        _gbdt_unlock(word)
        raise error^
    _gbdt_unlock(word)
    return info^


def gbdt_resident_release_binding(handle: PythonObject) raises -> PythonObject:
    """Drop a prepared model. A released or unknown handle raises."""
    var h = Int(py=handle)
    # The lock is taken AND dropped with the GIL released (lane fx-gil): a
    # member's release may run on a thread that holds the GIL
    # (`_GbdtMemberGpuImage.release` at PACK), so no thread may wait for the
    # GIL while it holds the registry lock.
    with GILReleased(Python()):
        _gbdt_registry_release(h)
    return PythonObject(None)


def gbdt_resident_info_binding(handle: PythonObject) raises -> PythonObject:
    """`[approx_dim, n_input_features, workspace_rows, oblivious, n_trees]`
    of a live handle."""
    var h = Int(py=handle)
    var info: List[Int]
    # Locked and unlocked with the GIL released, as the release above.
    with GILReleased(Python()):
        info = _gbdt_registry_info(h)
    return py_list_int(info)


def gbdt_resident_predict_native(h: Int, x_address: Int, n_rows: Int, addr: Int, mode: Int,
                                 row_major: Bool) raises -> Int:
    """Apply a prepared model (XG_RESIDENT_PREDICT). Returns the width
    written per row.

    `mode` is 0 RAW, 1 SOFTMAX, 2 SIGMOID as `gbdt_predict_multi` takes
    them, plus 3 SIGMOID_PAIR: the Logloss and CrossEntropy `predict_proba`
    columns `[1 - p, p]` written as FLOAT64 (`gbdt_sigmoid_pair`'s two
    statements over the exact widening of the raw float32 value). Modes
    4..6 write int64 class codes (4: the FAST Logloss/CrossEntropy codes);
    every other mode writes float32, `n_rows * width` values row-major,
    exactly as `gbdt_predict` and `gbdt_predict_multi` write them.
    `row_major` says `x` is the C-order `[row * n_features + f]` block
    rather than the column-major one; the staging pass transposes it. The
    feature count comes from the prepared model. The caller holds `x` and
    the output for the length of the call."""
    var xp = _f32_ptr(x_address)
    var op32 = _f32_ptr(addr)
    var op64 = _f64_ptr(addr)
    var op64i = _i64_ptr(addr)
    return _gbdt_locked_predict(h, xp, n_rows, op32, op64, op64i, mode, row_major)


def gbdt_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


def gbdt_fit_two_level_feature_freq_native(
    x_address: Int, y_address: Int, weights_address: Int, sources_address: Int, n_rows: Int, n_features: Int,
    n_weights: Int, n_sources: Int, learning_rate: Float32, l2_leaf_reg: Float32, seed: Int,
) raises -> String:
    """Explicit mixed numeric/categorical two-level FeatureFreq fit
    (XG_FEATURE_FREQ_FIT): UInt32 source indices identify the dense
    categorical columns and every other column is numeric. Returns the
    model text. Buffers are borrowed from the caller for the call."""
    var xp = _f32_ptr(x_address)
    var yp = _f32_ptr(y_address)
    var wp = _f32_ptr(weights_address)
    var sp = _u32_ptr(sources_address)
    var random_seed = UInt64(seed)
    var device_plan = _gbdt_device_plan()
    return gbdt_fit_two_level_feature_freq_run(
        xp, n_rows, n_features, yp, wp, n_weights, sp, n_sources,
        learning_rate, l2_leaf_reg, random_seed, device_plan,
    )


def gbdt_fit_ordered_rmse_native(
    x_address: Int, y_address: Int, weights_address: Int, permutation_address: Int, n_rows: Int, n_features: Int,
    n_weights: Int, n_permutation: Int, n_estimators: Int, max_depth: Int, border_count: Int,
    learning_rate: Float32, l2_leaf_reg: Float32,
) raises -> String:
    """Numeric, single-permutation ordered RMSE (XG_ORDERED_FIT); returns
    normal model text. The permutation buffer holds UInt32 original row
    ids. Buffers remain borrowed from live arrays held by the caller for
    the duration of this call."""
    if n_rows < 4 or n_features < 1 or n_permutation != n_rows:
        raise Error("ordered RMSE requires >=4 rows, features and a full permutation")
    if n_weights != 0 and n_weights != n_rows:
        raise Error("ordered RMSE sample weight shape mismatch")
    _ = _f32_ptr(x_address)
    var yp = _f32_ptr(y_address)
    var wp = _f32_ptr(weights_address)
    var pp = _u32_ptr(permutation_address)
    var device_plan = _gbdt_device_plan()
    return gbdt_fit_ordered_rmse_run(
        x_address, yp, wp, pp, n_rows, n_features, n_weights,
        n_estimators, max_depth, border_count, learning_rate,
        l2_leaf_reg, device_plan,
    )


# TOMBSTONE: gbdt_fit_ordered_rmse, gbdt_fit_two_level_feature_freq (Python entries of this image: no caller left, the GBDT door calls the typed operations XG_ORDERED_FIT, XG_FEATURE_FREQ_FIT of this image's native entry; fits unchanged) deleted 2026-11-21 by lane/dw-trees-boost; code recoverable at 9c65b76827.
# Restore: git apply experiments/removed/dead_entries_gbdt_text_fit_calls.patch


# ---- The image's native gbdt entry (lane calltable-gbdt-forest-dart,
# 2026-11-11; ABI gbdt/fit_request.mojo, table core/gbdt_entry_table.mojo):
# the GBDT doors fit, probe and read the model dimension through it instead

struct _GbdtMemberGpuImage(GbdtMemberImage):
    """The Gbdt member's resident-model release in this image (the registry
    step of `gbdt_resident_release_binding`, under the same lock word, with
    no interpreter: a member call runs without the GIL)."""

    @staticmethod
    def release(handle: Int) raises:
        _gbdt_registry_release(handle)

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

    @staticmethod
    def predict(text: String, x: Int, n_rows: Int, dst: Int) raises -> Int:
        return gbdt_predict_native(text, x, n_rows, dst)

    @staticmethod
    def predict_multi(text: String, x: Int, n_rows: Int, dst: Int, mode: Int) raises -> Int:
        return gbdt_predict_multi_native(text, x, n_rows, dst, mode)

    @staticmethod
    def binary(probabilities: Bool, raw: Int, dst: Int, n: Int) raises -> Int:
        return gbdt_binary_native(probabilities, raw, dst, n)

    @staticmethod
    def sigmoid_pair(raw: Int, dst: Int, n: Int) raises -> Int:
        return gbdt_sigmoid_pair_native(raw, dst, n)

    @staticmethod
    def resident() -> Bool:
        return True

    @staticmethod
    def resident_prepare(text: String) raises -> Int:
        return gbdt_resident_prepare_native(text)

    @staticmethod
    def resident_predict(handle: Int, x: Int, n_rows: Int, dst: Int, mode: Int, row_major: Bool) raises -> Int:
        return gbdt_resident_predict_native(handle, x, n_rows, dst, mode, row_major)

    @staticmethod
    def ordered_fit(x: Int, y: Int, weights: Int, permutation: Int, n_rows: Int, n_features: Int, n_weights: Int,
                    n_permutation: Int, n_estimators: Int, max_depth: Int, border_count: Int,
                    learning_rate: Float32, l2_leaf_reg: Float32) raises -> String:
        return gbdt_fit_ordered_rmse_native(x, y, weights, permutation, n_rows, n_features, n_weights,
            n_permutation, n_estimators, max_depth, border_count, learning_rate, l2_leaf_reg)

    @staticmethod
    def feature_freq_fit(x: Int, y: Int, weights: Int, sources: Int, n_rows: Int, n_features: Int, n_weights: Int,
                         n_sources: Int, learning_rate: Float32, l2_leaf_reg: Float32,
                         random_seed: Int) raises -> String:
        return gbdt_fit_two_level_feature_freq_native(x, y, weights, sources, n_rows, n_features, n_weights,
            n_sources, learning_rate, l2_leaf_reg, random_seed)

    @staticmethod
    def importances(text: String, x: Int, n_rows: Int, n_input: Int, row_major: Bool, weights: Int,
                    n_weights: Int, dst: Int) raises -> Int:
        # XG_IMPORTANCES (lane mf-imp6): the device column on this image's
        # process context; it synchronizes before it returns.
        var ctx = process_ctx[_DEVCTX_SLOT]()
        return gbdt_feature_importances_device(ctx, text, x, n_rows, n_input, row_major, weights, n_weights, dst)

    @staticmethod
    def slot_nan(x_word: Int, count: Int) raises -> Int:
        # XG_SLOT_NAN (lane vdf2-tail2): the member's nan_mode='Forbidden'
        # refusal over the fold slot, a device reduction on this image's
        # context; one word comes home.
        return fold_x_first_nan(process_ctx[_DEVCTX_SLOT](), x_word, count)


def gbdt_gpu_entry_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """This image's gbdt entry (`GBDT_ENTRY_ABI`, gbdt/fit_request.mojo).

    ONE BODY, TWO ENTRIES (lane fx-gil, 2026-11-21). The operations of
    `_GbdtGpuImage` never touch the GIL. A door calls on the Python thread
    with the GIL held: the long operations (`gbdt_entry_releases`) run with
    it released around the whole dispatch, as each released it inside
    before. A member's worker (core/gbdt_member.mojo) holds no GIL and says
    so with `ENTRY_OP_GIL_FREE`: the same dispatch runs with no release and
    no CPython call."""
    if not gbdt_entry_releases(operation):
        return gbdt_entry_dispatch[_GbdtGpuImage](operation, aa, na, pp, np, ff, nf, result, context, sink)
    var status = 1
    try:
        with GILReleased(Python()):
            status = gbdt_entry_dispatch[_GbdtGpuImage](operation, aa, na, pp, np, ff, nf, result, context, sink)
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
    return status


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_gbdt.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_gbdt() -> PythonObject:
    comptime assert TREES_GUARDS_OK, "invalid integrated experiment configuration (trees guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_gbdt")
        # The native gbdt entry (core/gbdt_entry_table.mojo) under this image's key.
        gbdt_entry_publish["MojolearnGbdtEntryOwnerGpuV1"](COMPILED_VENDOR, gbdt_gpu_entry_v1)
        # The Gbdt member (core/gbdt_member.mojo), after the gbdt entry.
        gbdt_member_publish_image[_GbdtMemberGpuImage](COMPILED_VENDOR)
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
        m.def_function[gbdt_model_dim_binding]("gbdt_model_dim")
        m.def_function[gbdt_numeric_mode_binding]("gbdt_numeric_mode")
        m.def_function[gbdt_mode_policy_mask_binding]("gbdt_mode_policy_mask")
        m.def_function[gbdt_per_round_paths_binding]("gbdt_per_round_paths")
        # the direct kernel check's two entries (checks/gbdt_binary_prediction_smoke.py)
        m.def_function[gbdt_binary_prediction_binding[True]]("gbdt_binary_probabilities")
        m.def_function[gbdt_binary_prediction_binding[False]]("gbdt_binary_classes")
        # TOMBSTONE: gbdt_sigmoid (dead native entry: no live caller) deleted 2026-10-26 by 7c1201602; code recoverable at 485ba9cc1.
        # Restore: git apply experiments/removed/gbdt_sigmoid.patch
        # DEVIATION 2980: the device-resident parsed model (predict is the
        # typed XG_RESIDENT_PREDICT; release stays a Python entry because
        # the lifetime owner `TreesResident` holds a Python callable)
        m.def_function[gbdt_resident_prepare_binding]("gbdt_resident_prepare")
        m.def_function[gbdt_resident_release_binding]("gbdt_resident_release")
        m.def_function[gbdt_resident_info_binding]("gbdt_resident_info")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_gbdt module: ", e))


# TOMBSTONE: gbdt_parallel_available, pointwise_parallel_available (dead native entries: constant probes no driver reads) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_trees.patch

