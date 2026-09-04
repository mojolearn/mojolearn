# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the gradient-boosted oblivious trees.

Built by `bindings/build_gbdt.sh` into `python/mojolearn/_mojolearn_gbdt.so`.
The public Python surface is `python/mojolearn/ensemble.py`.

WHY GBDT HAS ITS OWN EXTENSION
-------------------------------
Mojo 1.0.0 (ed45d567) decides how many kernels it compiles ahead of time from
THE BASENAME OF THE ENTRY FILE -- see archive/reference/PORTING.md 70 and the long comment in
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

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.vendor import COMPILED_VENDOR

from max.gpu.host import DeviceContext

from gbdt.estimator import (
    GbdtFitParams,
    GbdtFitResult,
    gbdt_fit,
    gbdt_model_dim,
    gbdt_predict,
    gbdt_predict_multi,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    identical_exp64,
)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)


def gbdt_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST,
    1 IDENTICAL, 2 DETERMINISTIC.

    It returned a BOOLEAN (1 for identical, else 0) until
    2026-08-29, and a boolean stopped being able to tell the truth
    the moment the middle tier existed: a DETERMINISTIC binary
    answered 0 and every reader printed it as "fast". That is the
    mislabelled measurement this read-back exists to make
    impossible. Widening it is backward compatible because the two
    old answers are the two codes they already were.

    A caller gating the CROSS-VENDOR guarantee still tests `== 1`,
    and should: 2 promises reproducibility on one device and says
    nothing about a second."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def gbdt_sigmoid_binding(
    raw_addr: PythonObject, out_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """`out[i] = 1 / (1 + exp(-raw[i]))` in double, `n` values, through
    `identical_exp64` -- DEVIATION 258: under IDENTICAL the Logloss /
    CrossEntropy probability is the same bits on every host (the wrapper
    used numpy's exp, whose last bit is the host libm's); under FAST this
    is the host stdlib and the wrapper keeps numpy. Both buffers are
    float64, the caller's."""
    var rp = _f64_ptr(Int(py=raw_addr))
    var op = _f64_ptr(Int(py=out_addr))
    var count = Int(py=n)
    for i in range(count):
        var r = rp.unsafe_load(i)
        op.unsafe_store(i, 1.0 / (1.0 + identical_exp64(-r)))
    return PythonObject(count)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    """A caller's float32 buffer, borrowed, never owned.

    The origin is untracked because the owner is a NumPy array on the other
    side of the boundary and Mojo cannot see it. The wrapper holds that array
    for the length of the call, which is the whole contract.
    """
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _u32_ptr(addr: Int) raises -> MutPointer[UInt32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=addr)


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
    """Fit a gradient-boosted ensemble.

    Returns `[model_text, best_iteration, stopped_early, learn_losses,
    test_losses]`. It used to return the text alone; the four that follow
    it are what a caller needs to tell a model that STOPPED from one that
    ran out of iterations, and to plot the curve that made it stop.

    `params` is, in this exact order -- and the wrapper names the same
    order in the same words, because a silent reordering here is a wrong
    answer rather than a failure:

         0  n_rows
         1  n_features
         2  n_weights        (0 means unit weights; weights_addr unread)
         3  n_flags          (0 means all-numeric; cat_flags_addr unread)
         4  border_count
         5  n_estimators
         6  max_depth
         7  learning_rate    (float)
         8  l2_leaf_reg      (float)
         9  random_seed
        10  score_function
        11  loss_alpha       (float, -1 = unset)
        12  loss_q           (float, -1 = unset)
        13  loss_delta       (float, -1 = unset)
        14  loss_variance_power (float, -1 = unset)
        15  loss_border      (float, -1 = unset)
        16  leaf_estimation_iterations  (-1 = unset, the loss decides)
        17  leaf_estimation_method      (-1 = unset, the loss decides)
        18  bagging_temperature (float)
        19  subsample        (float, -1 = unset)
        20  n_eval_rows      (0 means no held-out set; the eval addresses
                              are unread)
        21  od_pvalue        (float, -1 = unset)
        22  od_wait          (-1 = unset)
        23  use_best_model   (-1 = unset, 0 off, 1 on)
        24  best_model_min_trees
        25  random_strength  (float)
        26  use_pointwise_searcher  (0 or 1)
        27  border_build_max_samples  (0 = every row)
        28  permutation_count            (-1 = unset)
        29  ctr_estimation_permutation_id (-1 = unset)
        30  boost_from_average (-1 = unset, 0 off, 1 on)
        31  grow_policy      (0 SymmetricTree, 1 Depthwise, 2 Lossguide --
                              their EGrowPolicy order)
        32  max_leaves       (-1 = unset: 1 << depth, or 31 under Lossguide)
        33  min_data_in_leaf
        34  n_class_weights  (0 means none)

    AND THEN `n_class_weights` MORE VALUES, the class weights themselves,
    at `params[35 .. 35 + n_class_weights)`. They ride in this list rather
    than at a seventh buffer address for two reasons. The arity: this
    function already takes eight arguments and
    `PythonModuleBuilder.def_function` stops inferring a signature at
    around nine. And the ROUND TRIP: a Python float reaches `Float64(py=)`
    exactly, where a float written into a string and parsed back does not
    -- `String(Float32)` on this toolchain returns a one-ULP-wrong value
    for 0.46% of float32 values, and a class weight is a user's number,
    not ours to round.

    `strs` is `[loss, bootstrap_type, od_type, nan_mode]`, their
    `ELossFunction`, `EBootstrapType`, `EOverfittingDetectorType` and
    `ENanMode` spellings. An empty `bootstrap_type` means none, and an
    empty `od_type` means UNSET -- which is not the same as `None`: their
    `Load` (`overfitting_detector_options.cpp:24-32`) picks the type from
    whichever of the other two was given, so an unset type with a wait is
    `Iter`. `nan_mode` is "Min", "Max" or "Forbidden"; an empty string is
    read as Min by `nan_mode_from_name`, which is their default
    (`data_processing_options.cpp:26`).

    EVERY `-1` ABOVE IS THEIR `TOption::NotSet()`, not a magic number: the
    loss picks the leaf estimator and its iteration count through
    `set_leaves_estimation_default`, which is the port of
    `catboost_options.cpp:273-360`. A caller that passes explicit values
    is overriding CatBoost's own defaults and should know it.
    """
    # 35 FIXED SLOTS PLUS ONE PER CLASS WEIGHT (slot 30 grew
    # boost_from_average 2026-08-22 and the count moved to 31; slots 31-33
    # grew grow_policy / max_leaves / min_data_in_leaf 2026-08-23,
    # DEVIATION 259, and the count moved to 34 -- the count is ALWAYS the
    # last fixed slot, so a new option goes before it and bumps these
    # three numbers and the wrapper's `_params`). The count is checked
    # against its slot rather than assumed, because a wrapper that
    # appended the wrong number of weights would otherwise read whatever
    # followed them -- and for a weight that is a wrong answer, not a
    # failure.
    if len(params) < 35:
        raise Error(
            "gbdt_fit: params must hold at least 35 values, got "
            + String(len(params))
        )
    var n_class_weights = Int(py=params[34])
    if n_class_weights < 0:
        raise Error(
            "gbdt_fit: n_class_weights must not be negative, got "
            + String(n_class_weights)
        )
    if len(params) != 35 + n_class_weights:
        raise Error(
            "gbdt_fit: params must hold 35 + n_class_weights ("
            + String(35 + n_class_weights)
            + ") values, got "
            + String(len(params))
        )
    if len(strs) != 4:
        raise Error(
            "gbdt_fit: strs must hold [loss, bootstrap_type, od_type,"
            " nan_mode], got " + String(len(strs))
        )
    var xp = _f32_ptr(Int(py=x_addr))
    var yp = _f32_ptr(Int(py=y_addr))
    # unread when n_weights / n_flags is 0; the wrapper passes the X
    # address rather than allocating a throwaway, as `kmeans_fit` does,
    # and `_f32_ptr` still refuses a null.
    var wp = _f32_ptr(Int(py=weights_addr))
    var cp = _u32_ptr(Int(py=cat_flags_addr))
    # unread when params[20] is 0, the same contract the weights have;
    # the wrapper passes the X address rather than allocating a throwaway
    var ep = _f32_ptr(Int(py=eval_x_addr))
    var eyp = _f32_ptr(Int(py=eval_y_addr))

    var n_rows = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_weights = Int(py=params[2])
    var n_flags = Int(py=params[3])

    # Read BEFORE the GIL is released, like every other Python value here:
    # `params` is a Python list and touching it without the GIL is a data
    # race, not a slow path.
    var class_weights = List[Float32]()
    for i in range(n_class_weights):
        class_weights.append(Float32(Float64(py=params[35 + i])))
    # their `EGrowPolicy` by ORDINAL (enums.h order: SymmetricTree,
    # Depthwise, Lossguide), resolved to the spelling `train` takes
    var grow_code = Int(py=params[31])
    var grow_name = String("SymmetricTree")
    if grow_code == 1:
        grow_name = String("Depthwise")
    elif grow_code == 2:
        grow_name = String("Lossguide")
    elif grow_code != 0:
        raise Error(
            "gbdt_fit: grow_policy code must be 0 (SymmetricTree), 1"
            " (Depthwise) or 2 (Lossguide), got " + String(grow_code)
        )

    var fp = GbdtFitParams(
        Int(py=params[4]),
        Int(py=params[5]),
        Int(py=params[6]),
        Float32(Float64(py=params[7])),
        Float32(Float64(py=params[8])),
        UInt64(Int(py=params[9])),
        Int(py=params[10]),
        String(py=strs[0]),
        Float32(Float64(py=params[11])),
        Float32(Float64(py=params[12])),
        Float32(Float64(py=params[13])),
        Float32(Float64(py=params[14])),
        Float32(Float64(py=params[15])),
        Int(py=params[16]),
        Int(py=params[17]),
        String(py=strs[1]),
        Float32(Float64(py=params[18])),
        Float32(Float64(py=params[19])),
        String(py=strs[2]),
        Float64(py=params[21]),
        Int(py=params[22]),
        Int(py=params[23]),
        Int(py=params[24]),
        String(py=strs[3]),
        Float32(Float64(py=params[25])),
        Int(py=params[26]) != 0,
        Int(py=params[27]),
        Int(py=params[28]),
        Int(py=params[29]),
        Int(py=params[30]),
        class_weights^,
        grow_name,
        Int(py=params[32]),
        Int(py=params[33]),
    )
    var n_eval_rows = Int(py=params[20])

    var result: GbdtFitResult
    with GILReleased(Python()):
        var ctx = DeviceContext()
        result = gbdt_fit(
            ctx, xp, n_rows, n_features, yp, wp, n_weights,
            cp, n_flags, ep, eyp, n_eval_rows, fp,
        )

    var learn = Python.list()
    for i in range(len(result.learn_losses)):
        learn.append(PythonObject(result.learn_losses[i]))
    var test = Python.list()
    for i in range(len(result.test_losses)):
        test.append(PythonObject(result.test_losses[i]))

    var out = Python.list()
    out.append(PythonObject(result.text))
    out.append(PythonObject(result.best_iteration))
    out.append(PythonObject(result.stopped_early))
    out.append(learn)
    out.append(test)
    return out


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
        var ctx = DeviceContext()
        wrote = gbdt_predict(ctx, text, xp, n_rows, op)
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
        var ctx = DeviceContext()
        width = gbdt_predict_multi(ctx, text, xp, n_rows, op, mode)
    return PythonObject(width)


def gbdt_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_gbdt() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_gbdt")
        m.def_function[gbdt_vendor_binding]("gbdt_vendor")
        m.def_function[gbdt_fit_binding]("gbdt_fit")
        m.def_function[gbdt_predict_binding]("gbdt_predict")
        m.def_function[gbdt_model_dim_binding]("gbdt_model_dim")
        m.def_function[gbdt_predict_multi_binding]("gbdt_predict_multi")
        m.def_function[gbdt_numeric_mode_binding]("gbdt_numeric_mode")
        m.def_function[gbdt_sigmoid_binding]("gbdt_sigmoid")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_gbdt module: ", e))
