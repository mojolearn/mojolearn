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
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.vendor import COMPILED_VENDOR

from max.gpu.host import DeviceContext

from cluster.estimator import kmeans_fit
from neighbors.impl.neighbors.detail.knn_brute_force import KNN_METHOD_AUTO
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
    pruned anything. `neighbors/impl/neighbors/ball_cover/knn.mojo` explains
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


@export
def PyInit__mojolearn() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn")
        m.def_function[mojolearn_vendor_binding]("mojolearn_vendor")
        m.def_function[knn_search_binding]("knn_search")
        m.def_function[knn_classify_binding]("knn_classify")
        m.def_function[knn_regress_binding]("knn_regress")
        m.def_function[kmeans_fit_binding]("kmeans_fit")
        m.def_function[radius_neighbors_count_binding](
            "radius_neighbors_count"
        )
        m.def_function[radius_neighbors_fill_binding]("radius_neighbors_fill")
        m.def_function[rbc_knn_search_binding]("rbc_knn_search")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn module: ", e))
