# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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
`knn_search`, `knn_classify`, `knn_regress` and `kmeans_fit`. Those are the
algorithms with a caller-facing surface in THIS extension: they take host
pointers, own their device work, and have checks covering the policy they
add. `knn_classify` / `knn_regress` (2026-08-23) are the k-NN classifier and
regressor over `knn_search`: one call does the search AND the vote (or the
mean), for the identity-trace reason `neighbors/estimator.mojo`'s
`knn_search` docstring gives.

**GBDT MOVED OUT.** It lives in `bindings/_mojolearn_gbdt.mojo`, built by
`bindings/build_gbdt.sh` into a second extension, for the reason
`bindings/_mojolearn_estimators.mojo` gives for the third: an independently
changing binding should not be a merge point. GBDT is the fastest-moving
surface in this repository and every parameter added to `GbdtFitParams` used
to have to be unpacked in TWO files that could silently disagree about the
order of a flat list -- which is a wrong answer, not a failure. Now there is
one. Do not re-add a gbdt import here.

DBSCAN, PCA, truncated SVD and OLS are bound in the THIRD extension,
`bindings/_mojolearn_estimators.mojo`, and exported from `mojolearn` since
2026-08-23 (`mojolearn.DBSCAN`, `.PCA`, `.TruncatedSVD`,
`.LinearRegression`). This paragraph used to say they had no surface; that
stopped being true when their host-pointer surfaces landed and the sentence
outlived the fact.

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
binding takes nine and that is not a coincidence. `knn_search` needs ten,
`knn_classify` fourteen plus one per output and `kmeans_fit` fourteen. So each entry point takes its BUFFER ADDRESSES
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

FIVE HOST HELPERS WITH NO DEVICE CONTEXT (DEVIATION 2303 + 2440, 2026-09-07)
-----------------------------------------------------------------------------
`all_finite_f32`, `all_finite_f64`, `column_mean_f64` (DEVIATION 2303) and
`center_columns_f32`, `scale_rows_f32` (DEVIATION 2440) at the bottom of
this file run on the CPU over the caller's buffer and never construct a
`DeviceContext`. They exist so the NumPy-free Python layer
(`python/mojolearn/NUMPY_FREE_CONTRACT.md`) has somewhere other than a
Python loop to put a per-element pass. `column_mean_f64` is the DEFINITION
of the centering order OLS/ridge now use, and its docstring states that
order because callers rely on it; the two elementwise helpers reproduce
`linear_model.py`'s `_center` / `_scale_rows` operation for operation.
"""

from std.os import abort
from std.math import isfinite
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.numerics import GLOBAL_NUMERIC_MODE

from checks.vendor import COMPILED_VENDOR

from max.gpu.host import DeviceContext

from cluster.estimator import kmeans_fit
from neighbors.impl.detail.knn_brute_force import KNN_METHOD_AUTO
from neighbors.estimator import (
    knn_classifier_predict,
    knn_regressor_predict,
    knn_search,
    radius_neighbors_count,
    radius_neighbors_fill,
    rbc_knn_search,
)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    """A caller's float32 buffer, borrowed, never owned.

    The origin is untracked because the owner is a NumPy array on the other
    side of the boundary and Mojo cannot see it. The wrapper holds that array
    for the length of the call, which is the whole contract.
    """
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    """A caller's float64 buffer, borrowed, never owned; `_f32_ptr`'s twin
    (DEVIATION 2321). The same helper, spelled the same way, sits in
    `bindings/_mojolearn_gbdt.mojo`, `_mojolearn_estimators.mojo`,
    `_mojolearn_svm.mojo` and `_mojolearn_gp.mojo`."""
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)


def _u32_ptr(addr: Int) raises -> MutPointer[UInt32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


# ===========================================================================
# THE METRIC / WEIGHTING TRIPLE, ADDED 2026-09-01.
#
# The three k-NN bindings below each grew ONE trailing argument,
# `dist_params`, rather than three more slots in `params`. Two reasons, and
# the first is not style: `knn_classify_binding`'s `params` is variable
# length -- `7 + n_outputs`, with the class counts at the TAIL (`:196-206`)
# -- so anything appended there would collide with `n_classes` and the
# arity check would have to guess. A separate list cannot. The second is
# that `bindings/_mojolearn_estimators.mojo::kde_score_samples_binding`
# already passes `kernel` and `metric` as their own arguments beside
# `params`, so this is the shape this file's sibling already uses.
#
# `dist_params` is, in this exact order (mirrored in
# `python/mojolearn/neighbors.py::NearestNeighbors._dist_params`):
#
#     0  metric      a cuVS DistanceType value, or METRIC_FROM_IS_SQRT (-1)
#     1  metric_arg  Minkowski's p (float); discarded by every other metric
#     2  weights     WEIGHTS_UNIFORM (0) or WEIGHTS_DISTANCE (1)
#
# `weights` is read only by the classifier and the regressor;
# `knn_search_binding` takes the triple anyway so the three signatures
# stay parallel and the wrapper builds ONE list for all three.
# ===========================================================================


def _dist_triple(dist_params: PythonObject) raises -> Tuple[Int, Float32, Int]:
    """`dist_params` -> `(metric, metric_arg, weights)`, length-checked."""
    if len(dist_params) != 3:
        raise Error(
            "knn: dist_params must hold 3 values (metric, metric_arg,"
            " weights), got " + String(len(dist_params))
        )
    return (
        Int(py=dist_params[0]),
        Float32(Float64(py=dist_params[1])),
        Int(py=dist_params[2]),
    )


def knn_search_binding(
    index_addr: PythonObject,
    queries_addr: PythonObject,
    out_dist_addr: PythonObject,
    out_idx_addr: PythonObject,
    params: PythonObject,
    dist_params: PythonObject,
) raises -> PythonObject:
    """Exact k-NN. Returns the query tile that actually ran.

    `dist_params` is `(metric, metric_arg, weights)`; see the block above.
    `weights` is unread here (a search returns distances, it does not
    vote) and is present so the wrapper can pass one list to all three.

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
    var dt = _dist_triple(dist_params)

    var used: Int
    with GILReleased(Python()):
        var ctx = DeviceContext()
        used = knn_search(
            ctx, ip, ni, qp, nq, nf, kk, dp, xp, sq, qt, KNN_METHOD_AUTO,
            dt[0], dt[1],
        )
    return PythonObject(used)


def knn_classify_binding(
    index_addr: PythonObject,
    queries_addr: PythonObject,
    y_addr: PythonObject,
    out_labels_addr: PythonObject,
    out_proba_addr: PythonObject,
    out_uniq_addr: PythonObject,
    params: PythonObject,
    dist_params: PythonObject,
) raises -> PythonObject:
    """The k-NN classifier: search, then vote or tally. Returns the query tile
    that ran (the same number `knn_search` returns, for the same reason).

    `params` is, in this exact order (mirrored in
    `python/mojolearn/neighbors.py::KNeighborsClassifier`):

        0  n_index
        1  n_queries
        2  n_features
        3  k
        4  query_tile
        5  n_outputs
        6  want_proba      (0: write out_labels; 1: write out_proba)
        7.. n_classes per output, `n_outputs` of them

    `y_addr` is `n_outputs` CONTIGUOUS int32 columns of `n_index`
    (`neighbors/estimator.mojo` policy 6). `out_labels_addr` is
    `n_queries x n_outputs` int32 row-major; `out_proba_addr` is the
    per-output `n_queries x n_classes[i]` float32 blocks concatenated;
    `out_uniq_addr` is `sum(n_classes)` int32 and is always written (policy
    7: the wrapper asserts it against `classes_`). Whichever of the two
    outputs is not selected by `want_proba` is unread, and the wrapper
    passes a one-element array for it rather than a null.
    """
    if len(params) < 7:
        raise Error(
            "knn_classify: params must hold at least 7 values, got "
            + String(len(params))
        )
    var ni = Int(py=params[0])
    var nq = Int(py=params[1])
    var nf = Int(py=params[2])
    var kk = Int(py=params[3])
    var qt = Int(py=params[4])
    var no = Int(py=params[5])
    var want_proba = Int(py=params[6]) != 0
    if len(params) != 7 + no:
        raise Error(
            "knn_classify: params must hold 7 + n_outputs ("
            + String(7 + no)
            + ") values, got "
            + String(len(params))
        )
    var n_classes = List[Int]()
    for i in range(no):
        n_classes.append(Int(py=params[7 + i]))
    var ip = _f32_ptr(Int(py=index_addr))
    var qp = _f32_ptr(Int(py=queries_addr))
    var yp = _i32_ptr(Int(py=y_addr))
    var lp = _i32_ptr(Int(py=out_labels_addr))
    var pp = _f32_ptr(Int(py=out_proba_addr))
    var up = _i32_ptr(Int(py=out_uniq_addr))
    var dt = _dist_triple(dist_params)

    var used: Int
    with GILReleased(Python()):
        var ctx = DeviceContext()
        used = knn_classifier_predict(
            ctx, ip, ni, qp, nq, nf, kk, yp, no, n_classes, lp, pp, up,
            want_proba, qt, dt[0], dt[1], dt[2],
        )
    return PythonObject(used)


def knn_regress_binding(
    index_addr: PythonObject,
    queries_addr: PythonObject,
    y_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    dist_params: PythonObject,
) raises -> PythonObject:
    """The k-NN regressor: search, then the mean (uniform) or the
    distance-weighted mean (DEVIATION 556) of the neighbours' targets.
    Returns the query tile that ran.

    `params` is, in this exact order (mirrored in
    `python/mojolearn/neighbors.py::KNeighborsRegressor`):

        0  n_index
        1  n_queries
        2  n_features
        3  k
        4  query_tile
        5  n_outputs

    `y_addr` is `n_outputs` CONTIGUOUS float32 columns of `n_index`;
    `out_addr` is `n_queries x n_outputs` float32 row-major.
    """
    if len(params) != 6:
        raise Error(
            "knn_regress: params must hold 6 values, got "
            + String(len(params))
        )
    var ip = _f32_ptr(Int(py=index_addr))
    var qp = _f32_ptr(Int(py=queries_addr))
    var yp = _f32_ptr(Int(py=y_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var ni = Int(py=params[0])
    var nq = Int(py=params[1])
    var nf = Int(py=params[2])
    var kk = Int(py=params[3])
    var qt = Int(py=params[4])
    var no = Int(py=params[5])
    var dt = _dist_triple(dist_params)

    var used: Int
    with GILReleased(Python()):
        var ctx = DeviceContext()
        used = knn_regressor_predict(
            ctx, ip, ni, qp, nq, nf, kk, yp, no, op, qt, dt[0], dt[1], dt[2]
        )
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
    scales, and a caller reproducing a result needs them. `inertia` is the
    weighted cost against the FINAL centroids from cuVS's post-loop
    assignment (`detail/kmeans.cuh:516-535`) and is always formed;
    `inertia_check` (False by default, per cuVS) governs only the IN-LOOP
    cost. This docstring used to say "0.0 when it was NEVER COMPUTED",
    which was false (corrected 2026-08-23).
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


def mojolearn_numeric_mode_binding() raises -> PythonObject:
    """Read the compiled numeric mode of the reached kNN/kmeans binary."""
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def mojolearn_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back
    (`gbdt_numeric_mode`): the answer comes from the binary that actually
    loaded, never from the directory it sat in or from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


def radius_neighbors_count_binding(
    index_addr: PythonObject,
    queries_addr: PythonObject,
    out_indptr_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Pass one of the radius query. Returns the edge count.

    `params` is, in this exact order:

        0  n_index
        1  n_queries
        2  n_features
        3  radius        (float)
        4  metric        cuVS `DistanceType`, and only the four the ball
                         cover admits (DEVIATION 564); a non-metric is
                         refused BY NAME on the Mojo side with the triangle
                         inequality as the reason
        5  metric_arg    Minkowski's `p`, read only when metric is
                         LpUnexpanded, and refused below p = 1

    A radius query's output size is not a function of its inputs, so the
    caller cannot allocate before this call tells it how much to allocate.
    That is why there are two of these and not one; the reasoning is in
    `neighbors/estimator.mojo`'s RADIUS NEIGHBOURS banner.
    """
    if len(params) != 6:
        raise Error(
            "radius_neighbors_count: params must hold 6 values, got "
            + String(len(params))
        )
    var ip = _f32_ptr(Int(py=index_addr))
    var qp = _f32_ptr(Int(py=queries_addr))
    var ap = _i32_ptr(Int(py=out_indptr_addr))
    var ni = Int(py=params[0])
    var nq = Int(py=params[1])
    var nf = Int(py=params[2])
    var rad = Float32(Float64(py=params[3]))
    var mtr = Int(py=params[4])
    var marg = Float32(Float64(py=params[5]))

    var nnz: Int
    with GILReleased(Python()):
        var ctx = DeviceContext()
        nnz = radius_neighbors_count(
            ctx, ip, ni, qp, nq, nf, rad, ap, mtr, marg
        )
    return PythonObject(nnz)


def radius_neighbors_fill_binding(
    index_addr: PythonObject,
    queries_addr: PythonObject,
    out_indptr_addr: PythonObject,
    out_idx_addr: PythonObject,
    out_dist_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Pass two. Returns the edge count it actually found.

    `params` is, in this exact order:

        0  n_index
        1  n_queries
        2  n_features
        3  radius          (float)
        4  nnz_capacity    what pass one returned and the caller allocated
        5  return_sqrt     (0 or 1); a EUCLIDEAN policy, a no-op on the
                           metrics that never took a root
        6  metric          as pass one, and it MUST be the same value: the
                           index is built inside each call, so a metric that
                           differed between the two passes would count under
                           one and fill under another
        7  metric_arg      as pass one

    The return value is checked against `nnz_capacity` on the Mojo side and
    refused rather than truncated; it is returned as well so the wrapper can
    assert the same thing rather than trust it.
    """
    if len(params) != 8:
        raise Error(
            "radius_neighbors_fill: params must hold 8 values, got "
            + String(len(params))
        )
    var ip = _f32_ptr(Int(py=index_addr))
    var qp = _f32_ptr(Int(py=queries_addr))
    var ap = _i32_ptr(Int(py=out_indptr_addr))
    var xp = _i32_ptr(Int(py=out_idx_addr))
    var dp = _f32_ptr(Int(py=out_dist_addr))
    var ni = Int(py=params[0])
    var nq = Int(py=params[1])
    var nf = Int(py=params[2])
    var rad = Float32(Float64(py=params[3]))
    var cap = Int(py=params[4])
    var sq = Int(py=params[5]) != 0
    var mtr = Int(py=params[6])
    var marg = Float32(Float64(py=params[7]))

    var nnz: Int
    with GILReleased(Python()):
        var ctx = DeviceContext()
        nnz = radius_neighbors_fill(
            ctx, ip, ni, qp, nq, nf, rad, ap, xp, dp, cap, sq, mtr, marg
        )
    return PythonObject(nnz)


def rbc_knn_search_binding(
    index_addr: PythonObject,
    queries_addr: PythonObject,
    out_idx_addr: PythonObject,
    out_dist_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """EXACT k-NN over a random ball cover. Returns the number of candidate
    distances the query computed, which is the pruning it achieved.

    `params` is, in this exact order:

        0  n_index
        1  n_queries
        2  n_features
        3  k
        4  metric        cuVS `DistanceType`, and only the four the ball
                         cover admits (DEVIATION 564)
        5  metric_arg    Minkowski's `p`, refused below 1

    ONE call, not two, because a k-NN query's output size is
    `n_queries * k` and is known before the call. `out_idx` is int32 and
    holds `-1` in an unfilled slot; `out_dist` is float32 and holds TRUE
    distances in `metric`.

    The return value is the CANDIDATE COUNT and not a tile size: brute force
    over the same shapes would compute `n_index * n_queries`, so the caller
    can divide and see how much the index pruned instead of assuming it
    pruned anything. `neighbors/impl/ball_cover/knn.mojo` explains
    why that number is the one worth returning.
    """
    if len(params) != 6:
        raise Error(
            "rbc_knn_search: params must hold 6 values, got "
            + String(len(params))
        )
    var ip = _f32_ptr(Int(py=index_addr))
    var qp = _f32_ptr(Int(py=queries_addr))
    var xp = _i32_ptr(Int(py=out_idx_addr))
    var dp = _f32_ptr(Int(py=out_dist_addr))
    var ni = Int(py=params[0])
    var nq = Int(py=params[1])
    var nf = Int(py=params[2])
    var kk = Int(py=params[3])
    var mtr = Int(py=params[4])
    var marg = Float32(Float64(py=params[5]))

    var n_dists: Int
    with GILReleased(Python()):
        var ctx = DeviceContext()
        n_dists = rbc_knn_search(
            ctx, ip, ni, qp, nq, nf, kk, xp, dp, mtr, marg
        )
    return PythonObject(n_dists)

# ===========================================================================
# HOST CONVERTERS, DEVIATION 2470, 2471 and 2472 (2026-09-10).
#
# `python/mojolearn/_buffer.py` turns whatever a caller hands an estimator
# into the float32 block the kernels read. When NumPy left the Python layer
# that cast became a pure-Python `array.array` loop and the column-major
# reorder a per-column memoryview slice; measured at 2,000,000 x 20 float64
# on the M4 they cost 1,235 ms and 1,424 ms against NumPy's 5.7 ms and
# 119.4 ms. These two helpers put the work back in compiled code.
#
# Neither touches a device. Each is one pass over host memory with the GIL
# released, the `all_finite_f64_binding` shape from DEVIATION 2303: refuse a
# negative count by name, do nothing for an empty one, otherwise loop.
#
# THE CAST IS THE DEFINITION. Every float64 becomes float32 through exactly
# one IEEE-754 round-to-nearest-even narrowing, `SIMD.cast[DType.float32]`,
# which is the hardware `fcvt`/`cvtsd2ss` and the same operation NumPy's
# `astype(float32)` and CPython's `array.array('f')` item setter perform.
# Overflow goes to the signed infinity, NaN stays NaN (payload not
# promised), subnormal float64 rounds like any other value. No flush to
# zero, no other rounding mode. The bytes therefore equal
# `numpy.ascontiguousarray(x, dtype=float32)`'s and
# `numpy.asfortranarray(x, dtype=float32)`'s, and
# `python/mojolearn/tests/test_native_convert.py` holds them to that on
# ties, overflows, subnormals and non-finite input.
# ===========================================================================

comptime _CAST_WIDTH = 8
"""SIMD lanes per step of the flat cast: 8 float64 in, 8 float32 out. A
tail shorter than this is finished one element at a time through the same
`cast`, so the width never changes a bit of the answer."""

comptime _TILE_ROWS = 128
comptime _TILE_COLS = 64
"""The fused cast-and-transpose walks the source in `_TILE_ROWS` x
`_TILE_COLS` tiles: 64 KB of float64 in, 32 KB of float32 out, sized to
stay in a performance core's L1 so the strided side of the transpose hits
cache instead of memory. That is DEVIATION 1887's row-tile arm, which the
pure-Python converter does not carry, moved to where it can be fast. The
tile shape changes only the order in which independent elements are
written; each element's value is the same single cast."""


def cast_f64_to_f32_binding(
    src_addr: PythonObject, dst_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """Write `Float32(src[i])` to `dst[i]` for `i` in `[0, n)` (DEVIATION
    2470). Returns 0. `n == 0` writes nothing and does not read either
    address; a negative `n` is refused rather than read. `src` and `dst`
    must not overlap.
    """
    var count = Int(py=n)
    if count < 0:
        raise Error(
            "cast_f64_to_f32: n must be non-negative, got " + String(count)
        )
    if count == 0:
        return PythonObject(0)
    var sp = _f64_ptr(Int(py=src_addr))
    var dp = _f32_ptr(Int(py=dst_addr))
    with GILReleased(Python()):
        var i = 0
        var body = count - (count % _CAST_WIDTH)
        while i < body:
            var v = sp.unsafe_load[width=_CAST_WIDTH](i)
            dp.unsafe_store[width=_CAST_WIDTH](i, v.cast[DType.float32]())
            i += _CAST_WIDTH
        while i < count:
            dp.unsafe_store(i, sp.unsafe_load(i).cast[DType.float32]())
            i += 1
    return PythonObject(0)


def cast_colmajor_f64_to_f32_binding(
    src_addr: PythonObject,
    dst_addr: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
) raises -> PythonObject:
    """Read a C-contiguous float64 `[rows, cols]` matrix at `src`, write the
    COLUMN-MAJOR float32 matrix at `dst`, `dst[c * rows + r] ==
    Float32(src[r * cols + c])` (DEVIATION 2471). Returns 0. ONE fused
    pass: every element is read once, narrowed once and written once to
    its transposed position; there is no float32 intermediate in the
    source layout. An empty matrix writes nothing and reads neither
    address; a negative dimension is refused rather than read. `src` and
    `dst` must not overlap.
    """
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    if nr < 0:
        raise Error(
            "cast_colmajor_f64_to_f32: rows must be non-negative, got "
            + String(nr)
        )
    if nc < 0:
        raise Error(
            "cast_colmajor_f64_to_f32: cols must be non-negative, got "
            + String(nc)
        )
    if nr == 0 or nc == 0:
        return PythonObject(0)
    var sp = _f64_ptr(Int(py=src_addr))
    var dp = _f32_ptr(Int(py=dst_addr))
    with GILReleased(Python()):
        _tiled_transpose_to_f32(sp, dp, nr, nc)
    return PythonObject(0)


def _tiled_transpose_to_f32[
    S: DType
](
    sp: MutPointer[Scalar[S], MutUntrackedOrigin],
    dp: MutPointer[Float32, MutUntrackedOrigin],
    nr: Int,
    nc: Int,
):
    """`dp[c * nr + r] = Float32(sp[r * nc + c])` over `_TILE_ROWS` x
    `_TILE_COLS` tiles. The one loop behind DEVIATION 2471 (float64 in)
    and 2472 (float32 in, where the cast is the identity and compiles
    away). Caller holds valid, non-overlapping, non-empty buffers and has
    released the GIL."""
    var r0 = 0
    while r0 < nr:
        var r1 = min(r0 + _TILE_ROWS, nr)
        var c0 = 0
        while c0 < nc:
            var c1 = min(c0 + _TILE_COLS, nc)
            # Inside the tile: one column at a time, so the writes are
            # contiguous and the strided reads stay within the tile's
            # rows, which the previous column just pulled into cache.
            for c in range(c0, c1):
                var dbase = c * nr
                for r in range(r0, r1):
                    dp.unsafe_store(
                        dbase + r,
                        sp.unsafe_load(r * nc + c).cast[DType.float32](),
                    )
            c0 = c1
        r0 = r1


def transpose_f32_binding(
    src_addr: PythonObject,
    dst_addr: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
) raises -> PythonObject:
    """Read a C-contiguous float32 `[rows, cols]` matrix at `src`, write its
    COLUMN-MAJOR layout at `dst`, `dst[c * rows + r] == src[r * cols + c]`
    (DEVIATION 2472). Returns 0. A pure move, no arithmetic: every float32
    bit pattern, NaN payloads included, arrives unchanged. The same call
    turns a column-major `[rows, cols]` block into C order, because that
    block IS a C-contiguous `[cols, rows]` matrix: pass `rows=cols,
    cols=rows`. An empty matrix writes nothing and reads neither address;
    a negative dimension is refused rather than read. `src` and `dst`
    must not overlap.
    """
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    if nr < 0:
        raise Error(
            "transpose_f32: rows must be non-negative, got " + String(nr)
        )
    if nc < 0:
        raise Error(
            "transpose_f32: cols must be non-negative, got " + String(nc)
        )
    if nr == 0 or nc == 0:
        return PythonObject(0)
    var sp = _f32_ptr(Int(py=src_addr))
    var dp = _f32_ptr(Int(py=dst_addr))
    with GILReleased(Python()):
        _tiled_transpose_to_f32(sp, dp, nr, nc)
    return PythonObject(0)


# ===========================================================================
# HOST HELPERS FOR THE NUMPY-FREE PYTHON LAYER (DEVIATION 2303).
#
# DEVIATION 2320: these three take their scalars POSITIONALLY rather than
# in a `params` list. The list convention above exists because
# `def_function` cannot infer an arity past roughly nine; two and four
# arguments are well inside that, and a positional `Int(py=...)` is the
# shape `bindings/_mojolearn_gbdt.mojo::gbdt_sigmoid_binding` already uses
# for exactly this kind of host loop.
#
# None of them constructs a `DeviceContext`: the work is a single pass over
# a host buffer. The GIL is still released around the pass (the
# `var result: Int` / `with GILReleased(Python())` shape of
# `knn_search_binding`) because nothing inside touches a Python object and
# a caller scanning a million rows should not stall its other threads.
# ===========================================================================


def all_finite_f32_binding(
    addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """1 if every one of the `n` float32 values at `addr` is finite, else 0
    (DEVIATION 2322). `n == 0` is 1: an empty buffer has no non-finite
    element. A negative `n` is refused rather than read.

    NaN, +inf and -inf all fail; subnormals PASS, because they are finite
    numbers and `isfinite` says so. This is the test
    `bindings/_mojolearn_estimators.mojo::_pca_whiten_finite` applies,
    one element at a time, with no SIMD width and no early-exit trick a
    different backend could reorder: the FIRST non-finite element ends the
    scan and the answer is the same whichever element it was.
    """
    var p = _f32_ptr(Int(py=addr))
    var count = Int(py=n)
    if count < 0:
        raise Error(
            "all_finite_f32: n must be non-negative, got " + String(count)
        )
    var ok: Int = 1
    with GILReleased(Python()):
        for i in range(count):
            if not isfinite(p.unsafe_load(i)):
                ok = 0
                break
    return PythonObject(ok)


def all_finite_f64_binding(
    addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """`all_finite_f32_binding` over float64 values (DEVIATION 2323); the
    same rules, including subnormals passing and `n == 0` answering 1."""
    var p = _f64_ptr(Int(py=addr))
    var count = Int(py=n)
    if count < 0:
        raise Error(
            "all_finite_f64: n must be non-negative, got " + String(count)
        )
    var ok: Int = 1
    with GILReleased(Python()):
        for i in range(count):
            if not isfinite(p.unsafe_load(i)):
                ok = 0
                break
    return PythonObject(ok)


def column_mean_f64_binding(
    x_addr: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Per-column mean of a C-contiguous float32 `[rows, cols]` matrix,
    accumulated in float64 IN THIS EXACT ORDER, written to the caller's
    float64 `out[cols]` (DEVIATION 2324). Returns 0.

    THE ORDER IS THE DEFINITION. `python/mojolearn/linear_model.py` centers
    OLS/ridge with this helper where it used NumPy's `mean(axis=0,
    dtype=float64)`, whose blocked pairwise reduction has no defined order
    a second implementation could reproduce. This one does, and a caller
    checking a centering bit-for-bit reproduces it with:

        acc = [0.0] * cols                      # float64 zeros
        for r in range(rows):                   # row-major, row by row
            for c in range(cols):               # column by column
                acc[c] = acc[c] + float64(x[r * cols + c])
        for c in range(cols):
            out[c] = acc[c] / float64(rows)

    Every `+` is one IEEE-754 binary64 round-to-nearest-even addition of
    the widened float32 element onto the running column total, in row
    order. There is NO pairwise tree, NO SIMD lane split, NO Kahan term and
    NO fused multiply-add (there is no multiply to fuse). The float32 to
    float64 widening is exact. A `math.fsum`-style correctly rounded sum is
    a DIFFERENT number and is deliberately not what this computes:
    `python/mojolearn/tests/test_native_helpers.py` plants a column whose
    sequential total and correctly rounded total disagree, to keep anyone
    from "fixing" this into fsum and moving every IDENTICAL OLS bit.

    `rows` must be positive (a mean over zero rows is not a number this
    helper will invent) and `cols` must be positive.
    """
    var xp = _f32_ptr(Int(py=x_addr))
    var op = _f64_ptr(Int(py=out_addr))
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    if nr <= 0:
        raise Error(
            "column_mean_f64: rows must be positive, got " + String(nr)
        )
    if nc <= 0:
        raise Error(
            "column_mean_f64: cols must be positive, got " + String(nc)
        )
    with GILReleased(Python()):
        var acc = List[Float64](length=nc, fill=Float64(0.0))
        for r in range(nr):
            for c in range(nc):
                acc[c] += Float64(xp.unsafe_load(r * nc + c))
        for c in range(nc):
            op.unsafe_store(c, acc[c] / Float64(nr))
    return PythonObject(0)


# ---------------------------------------------------------------------------
# The two elementwise helpers of DEVIATION 2440, replacing the Python loops
# `python/mojolearn/linear_model.py` flagged as DEFECTS under DEVIATION
# 2362 (`_center`, `_scale_rows`). Each reproduces its Python loop
# OPERATION FOR OPERATION: widen both float32 operands to float64, one
# binary64 operation, ONE narrowing to float32 through `Float32(...)`, which
# is the same round-to-nearest-even the Python's `array.array('f')` item
# setter (a C `(float)` cast) performs. That is the definition; it is not
# "the float32 op" and it is not an approximation of it (for `-` and `*` on
# two float32 values the two coincide, but the helper is written as the
# Python is written so the equality is by construction, not by theorem).
# ---------------------------------------------------------------------------


def center_columns_f32_binding(
    x_addr: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
    mean_addr: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """`out[r, c] = fl32(float64(x[r, c]) - mean[c])` over a C-contiguous
    float32 `[rows, cols]` matrix, `mean` a float64 `[cols]` buffer, `out`
    float32 `[rows, cols]` (DEVIATION 2441). Returns 0.

    THIS IS `linear_model.py::_center` (DEVIATION 2362) IN THE SAME ORDER:
    row by row, column by column, the element widened to binary64 (exact),
    the float64 `mean[c]` subtracted in binary64 (one round-to-nearest-even),
    the difference narrowed to binary32 (one more). The caller passes the
    means it already narrowed to float32 values (`_column_means` ends in
    `_round_f32`), stored in a float64 buffer; this helper does NOT narrow
    the mean itself, because the Python does not. Nothing here depends on
    any other element, so `out` may alias `x`.

    `rows == 0` or `cols == 0` writes nothing; negatives are refused.
    """
    var xp = _f32_ptr(Int(py=x_addr))
    var mp = _f64_ptr(Int(py=mean_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    if nr < 0 or nc < 0:
        raise Error(
            "center_columns_f32: rows and cols must be non-negative, got "
            + String(nr) + " x " + String(nc)
        )
    with GILReleased(Python()):
        for r in range(nr):
            for c in range(nc):
                var d = Float64(xp.unsafe_load(r * nc + c)) - mp.unsafe_load(c)
                op.unsafe_store(r * nc + c, Float32(d))
    return PythonObject(0)


def scale_rows_f32_binding(
    x_addr: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
    w_addr: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """`out[r, c] = fl32(float64(x[r, c]) * float64(w[r]))` over a
    C-contiguous float32 `[rows, cols]` matrix, `w` a float32 `[rows]`
    buffer, `out` float32 `[rows, cols]` (DEVIATION 2442). Returns 0.

    THIS IS `linear_model.py::_scale_rows` (DEVIATION 2362) IN THE SAME
    ORDER: both float32 operands widened to binary64 (exact), one binary64
    multiply (exact too, since two 24-bit significands fit in 53), and ONE
    narrowing to binary32. The caller's `w` is `fl32(sqrt(sample_weight))`,
    already a float32 value; this helper takes no root and applies no
    weight semantics, it multiplies. No FMA is possible: there is no add.
    Nothing here depends on any other element, so `out` may alias `x`.

    `rows == 0` or `cols == 0` writes nothing; negatives are refused.
    """
    var xp = _f32_ptr(Int(py=x_addr))
    var wp = _f32_ptr(Int(py=w_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    if nr < 0 or nc < 0:
        raise Error(
            "scale_rows_f32: rows and cols must be non-negative, got "
            + String(nr) + " x " + String(nc)
        )
    with GILReleased(Python()):
        for r in range(nr):
            var w = Float64(wp.unsafe_load(r))
            for c in range(nc):
                var p = Float64(xp.unsafe_load(r * nc + c)) * w
                op.unsafe_store(r * nc + c, Float32(p))
    return PythonObject(0)


# NumPy-free host input validation and byte gathering. These do no learning:
# metric arithmetic and estimator work remain in the GPU bindings.
def probability_rows_f32_binding(
    src_addr: PythonObject, dst_addr: PythonObject, rows: PythonObject,
    cols: PythonObject, binary: PythonObject,
) raises -> PythonObject:
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    var bin = Int(py=binary)
    if nr < 0 or nc <= 0 or (bin != 0 and bin != 1):
        raise Error("probability_rows_f32: invalid dimensions or binary flag")
    if bin == 1 and nc != 1:
        raise Error("probability_rows_f32: binary input must have one column")
    if nr == 0:
        return PythonObject(0)
    var src = _f32_ptr(Int(py=src_addr))
    # Multiclass validation never reads or writes dst.
    var dst = src
    if bin == 1:
        dst = _f32_ptr(Int(py=dst_addr))
    var nonfinite = False
    var outside = False
    var bad_sum = False
    with GILReleased(Python()):
        for r in range(nr):
            var total = Float64(0)
            for c in range(nc):
                var p = src.unsafe_load(r * nc + c)
                nonfinite = nonfinite or not isfinite(p)
                outside = outside or p < 0 or p > 1
                total += Float64(p)
                if bin == 1:
                    dst.unsafe_store(2 * r, Float32(1) - p)
                    dst.unsafe_store(2 * r + 1, p)
            # Existing NumPy validation used sqrt(Float32 epsilon), itself
            # rounded to Float32, then widened for the Float64 comparison.
            if bin == 0:
                var error = total - Float64(1)
                bad_sum = bad_sum or abs(error) > Float64(0.00034526697709225118)
    if nonfinite:
        return PythonObject(1)
    if outside:
        return PythonObject(2)
    if bad_sum:
        return PythonObject(3)
    return PythonObject(0)


def gather_rows_bytes_binding(
    src_addr: PythonObject, dst_addr: PythonObject, indices_addr: PythonObject,
    source_rows: PythonObject, output_rows: PythonObject, row_bytes: PythonObject,
) raises -> PythonObject:
    var ns = Int(py=source_rows)
    var no = Int(py=output_rows)
    var width = Int(py=row_bytes)
    if ns < 0 or no < 0 or width < 0:
        raise Error("gather_rows_bytes: dimensions must be non-negative")
    if no == 0 or width == 0:
        return PythonObject(0)
    if Int(py=src_addr) == 0 or Int(py=dst_addr) == 0 or Int(py=indices_addr) == 0:
        raise Error("gather_rows_bytes: null buffer address")
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=src_addr))
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=dst_addr))
    var idx = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=indices_addr))
    var invalid = False
    with GILReleased(Python()):
        # Validate all indices before any output mutation.
        for r in range(no):
            var index = Int(idx.unsafe_load(r))
            if index < 0 or index >= ns:
                invalid = True
                break
        if not invalid:
            for r in range(no):
                var index = Int(idx.unsafe_load(r))
                memcpy(dest=dst + r * width, src=src + index * width, count=width)
    if invalid:
        raise Error("gather_rows_bytes: row index out of bounds")
    return PythonObject(0)


@export
def PyInit__mojolearn() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn")
        m.def_function[mojolearn_vendor_binding]("mojolearn_vendor")
        m.def_function[mojolearn_numeric_mode_binding]("mojolearn_numeric_mode")
        m.def_function[knn_search_binding]("knn_search")
        m.def_function[knn_classify_binding]("knn_classify")
        m.def_function[knn_regress_binding]("knn_regress")
        m.def_function[kmeans_fit_binding]("kmeans_fit")
        m.def_function[radius_neighbors_count_binding](
            "radius_neighbors_count"
        )
        m.def_function[radius_neighbors_fill_binding]("radius_neighbors_fill")
        m.def_function[rbc_knn_search_binding]("rbc_knn_search")
        m.def_function[cast_f64_to_f32_binding]("cast_f64_to_f32")
        m.def_function[cast_colmajor_f64_to_f32_binding](
            "cast_colmajor_f64_to_f32"
        )
        m.def_function[transpose_f32_binding]("transpose_f32")
        # DEVIATION 2325: the three host helpers of DEVIATION 2303.
        m.def_function[all_finite_f32_binding]("all_finite_f32")
        m.def_function[all_finite_f64_binding]("all_finite_f64")
        m.def_function[column_mean_f64_binding]("column_mean_f64")
        # DEVIATION 2443: the two elementwise helpers of DEVIATION 2440.
        m.def_function[center_columns_f32_binding]("center_columns_f32")
        m.def_function[scale_rows_f32_binding]("scale_rows_f32")
        m.def_function[probability_rows_f32_binding]("probability_rows_f32")
        m.def_function[gather_rows_bytes_binding]("gather_rows_bytes")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn module: ", e))
