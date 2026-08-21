"""CPython extension module for mojolearn.

Built by `bindings/build.sh` into `python/mojolearn/_mojolearn.so`. The public
Python surface is the scikit-learn-style wrapper in `python/mojolearn/`.

**Data crosses as raw buffer addresses plus lengths**, which is the convention
`neighbors/estimator.mojo` and `cluster/estimator.mojo` were written to. The
Python wrapper passes float32 C-contiguous arrays and keeps them alive for the
duration of the call; nothing here retains a Python buffer after it returns.
That contract is the wrapper's to honor and it is stated in
`python/mojolearn/_arrays.py`.

WHAT IS EXPOSED, AND WHAT IS NOT
---------------------------------
`knn_search`, `kmeans_fit`, `gbdt_fit` and `gbdt_predict`. Those are the
algorithms in this repository with a caller-facing surface: they take host
pointers, own their device work, and have checks covering the policy they add.
DBSCAN, PCA and OLS have verified kernels and no such surface yet, so binding
them would mean inventing one here, at the boundary, where no check can see
it. They are named in the wrapper's `__all__` as absent rather than silently
missing.

THE GBDT MODEL CROSSES AS TEXT, NOT AS A HANDLE, and `gbdt/estimator.mojo`
argues why: an integer handle into a table of live models on this side would
put object lifetime across the CPython boundary where nothing here can check
it. The text format is the one `check-model-io` already gates bit-for-bit.

THE DEVICE CONTEXT IS CREATED PER CALL, AND THAT IS A REAL COST
----------------------------------------------------------------
Each entry point below constructs its own `DeviceContext`. That is not free
and it is not hidden: a caller fitting in a loop pays it every iteration. It
is done this way because a module-global context would have to outlive the
GIL-released regions below and be safe against a caller using mojolearn from
two threads, and neither of those has been checked. **When someone measures
the per-call cost and wants it gone, the fix is a cached context with an
explicit thread contract, not a global slipped in quietly.**

SCALARS ARRIVE AS ONE LIST, WHICH IS NOT A STYLE CHOICE
--------------------------------------------------------
`PythonModuleBuilder.def_function` infers its signature from the function's
arity and stops being able to above roughly nine arguments; mojotrees' widest
binding takes nine and that is not a coincidence. `knn_search` needs ten and
`kmeans_fit` fourteen. So each entry point takes its BUFFER ADDRESSES
positionally, where a mistake is a crash rather than a wrong answer, and its
scalars in one list whose order is written out beside the unpacking below and
mirrored in the wrapper. Both sides name the order in the same words on
purpose: a silent reordering here would be a wrong answer, not a failure.

THE GIL IS RELEASED AROUND THE DEVICE WORK
-------------------------------------------
Both calls hand a buffer address to the GPU and wait. Holding the GIL across
that would block every other Python thread for the whole fit for no reason:
nothing inside touches a Python object, and the caller's arrays are kept alive
by the wrapper on the Python side. The pattern matches mojotrees'
`buffer_has_infinite`.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceContext

from cluster.estimator import kmeans_fit
from gbdt.estimator import (
    GbdtFitParams,
    GbdtFitResult,
    gbdt_fit,
    gbdt_model_dim,
    gbdt_predict,
    gbdt_predict_multi,
)
from neighbors.estimator import knn_search


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


def knn_search_binding(
    index_addr: PythonObject,
    queries_addr: PythonObject,
    out_dist_addr: PythonObject,
    out_idx_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Exact k-NN. Returns the query tile that actually ran.

    `params` is, in this exact order:

        0  n_index
        1  n_queries
        2  n_features
        3  k
        4  return_sqrt   (0 or 1)
        5  query_tile

    The return value is not decoration. `plan_query_tile` may lower the tile
    below what was asked for when the workspace cap fires, and a caller
    recording a benchmark number needs to know which configuration produced
    it. The wrapper surfaces it as `used_query_tile`.
    """
    if len(params) != 6:
        raise Error(
            "knn_search: params must hold 6 values, got "
            + String(len(params))
        )
    var ip = _f32_ptr(Int(py=index_addr))
    var qp = _f32_ptr(Int(py=queries_addr))
    var dp = _f32_ptr(Int(py=out_dist_addr))
    var xp = _u32_ptr(Int(py=out_idx_addr))
    var ni = Int(py=params[0])
    var nq = Int(py=params[1])
    var nf = Int(py=params[2])
    var kk = Int(py=params[3])
    var sq = Int(py=params[4]) != 0
    var qt = Int(py=params[5])

    var used: Int
    with GILReleased(Python()):
        var ctx = DeviceContext()
        used = knn_search(ctx, ip, ni, qp, nq, nf, kk, dp, xp, sq, qt)
    return PythonObject(used)


def kmeans_fit_binding(
    x_addr: PythonObject,
    out_centroids_addr: PythonObject,
    out_labels_addr: PythonObject,
    weights_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Fit k-means. Returns [inertia, n_iter, sum_scale, weight_scale].

    `params` is, in this exact order:

        0  n_samples
        1  n_features
        2  n_clusters
        3  n_weights   (0 means unit weights; weights_addr is then unread)
        4  max_iter
        5  tol         (float)
        6  seed
        7  n_init
        8  init
        9  metric

    All four returns are given because a wrong answer here comes from the two
    scales, and a caller reproducing a result needs them. **`inertia` is 0.0
    when it was NEVER COMPUTED**: `inertia_check` is False by default, per
    cuVS, and with it off the Lloyd loop never forms the cluster cost. Do not
    read a 0.0 as a perfect clustering. The wrapper repeats this warning.
    """
    if len(params) != 10:
        raise Error(
            "kmeans_fit: params must hold 10 values, got "
            + String(len(params))
        )
    var xp = _f32_ptr(Int(py=x_addr))
    var cp = _f32_ptr(Int(py=out_centroids_addr))
    var lp = _u32_ptr(Int(py=out_labels_addr))
    # When n_weights is 0 the estimator never reads this pointer, so the
    # wrapper passes the X address rather than allocating a throwaway array
    # of ones. `_f32_ptr` still refuses a null.
    var wp = _f32_ptr(Int(py=weights_addr))

    var ns = Int(py=params[0])
    var nf = Int(py=params[1])
    var nc = Int(py=params[2])
    var nw = Int(py=params[3])
    var mi = Int(py=params[4])
    var tl = Float64(py=params[5])
    var sd = UInt64(Int(py=params[6]))
    var ninit = Int(py=params[7])
    var ii = Int(py=params[8])
    var mm = Int(py=params[9])

    var inertia = Float64(0.0)
    var n_iter = 0
    var sum_scale = Float64(0.0)
    var weight_scale = Float64(0.0)
    with GILReleased(Python()):
        var ctx = DeviceContext()
        var r = kmeans_fit(
            ctx, xp, ns, nf, nc, cp, lp, wp, nw, mi, tl, sd, ninit, ii, mm
        )
        inertia = r.inertia
        n_iter = r.n_iter
        sum_scale = r.sum_scale
        weight_scale = r.weight_scale

    var out = Python.list()
    out.append(PythonObject(inertia))
    out.append(PythonObject(n_iter))
    out.append(PythonObject(sum_scale))
    out.append(PythonObject(weight_scale))
    return out


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

    `strs` is `[loss, bootstrap_type, od_type]`, their `ELossFunction`,
    `EBootstrapType` and `EOverfittingDetectorType` spellings. An empty
    `bootstrap_type` means none, and an empty `od_type` means UNSET --
    which is not the same as `None`: their `Load`
    (`overfitting_detector_options.cpp:24-32`) picks the type from
    whichever of the other two was given, so an unset type with a wait is
    `Iter`.

    EVERY `-1` ABOVE IS THEIR `TOption::NotSet()`, not a magic number: the
    loss picks the leaf estimator and its iteration count through
    `set_leaves_estimation_default`, which is the port of
    `catboost_options.cpp:273-360`. A caller that passes explicit values
    is overriding CatBoost's own defaults and should know it.
    """
    if len(params) != 25:
        raise Error(
            "gbdt_fit: params must hold 25 values, got "
            + String(len(params))
        )
    if len(strs) != 3:
        raise Error(
            "gbdt_fit: strs must hold [loss, bootstrap_type, od_type],"
            " got " + String(len(strs))
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


@export
def PyInit__mojolearn() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn")
        m.def_function[knn_search_binding]("knn_search")
        m.def_function[kmeans_fit_binding]("kmeans_fit")
        m.def_function[gbdt_fit_binding]("gbdt_fit")
        m.def_function[gbdt_predict_binding]("gbdt_predict")
        m.def_function[gbdt_model_dim_binding]("gbdt_model_dim")
        m.def_function[gbdt_predict_multi_binding]("gbdt_predict_multi")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn module: ", e))
