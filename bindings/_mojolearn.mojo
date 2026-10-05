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

TWO HOST HELPERS WITH NO DEVICE CONTEXT (DEVIATION 2303, 2026-09-07)
---------------------------------------------------------------------
`all_finite_f32` and `all_finite_f64` at the bottom of this file keep
their CPU loop as the host column and fallback only: since lane
cpu2-l1-input (2026-10-04) this GPU binding scans on the device first
(`hpdev_try_all_finite`, `core/input_device.mojo`), as it does the float64
cast, both transposes and the strided and ragged layout copies. The centering helpers `column_mean_f64`,
`center_columns_f32` and `scale_rows_f32` were deleted by lane
hr-small-passes (2026-10-02): LinearRegression and Ridge center on the
device through the estimators binding (glm/impl/center_device.mojo).
"""

from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, u32_ptr
from core.dense_coo import (
    nonzero_f32_count as dense_nonzero_f32_count,
    nonzero_f32_fill as dense_nonzero_f32_fill,
)
from core.dense_coo_device import knn_affinity_f32_device
from core.label_encode_device import device_unique_inverse
# lane cpu2-l2-labels: the label helpers below run their device twins
# directly, on every tier (no host loop left in this GPU binding).
from core.hotpath_device import (
    HPD_F32,
    HPD_F64,
    HPD_I32,
    HPD_I64,
    HPD_MAX_N,
    HPD_U32,
    HPD_U8,
    device_encode_labels,
    device_gather_u64,
)
from core.label_rows_device import LRD_MAX_N, device_argmax_rows, device_argmax_last_f32
from bindings.array_helpers import (
    nsum_f64_binding,
    class_ratio_f64_binding,
)
from core.shard_merge_device import device_shard_topk_merge_f32
from core.rows_bytes_device import device_gather_rows_bytes, device_scatter_rows_bytes
from core.ic_min_device import device_ic_running_min
from bindings.hotpath_helpers import (
    next_combination_i64_binding,
    scale_shift_ftz_f32_binding,
    compact_notnan_f32_binding,
    gather_keep_neg_i32_binding,
    dot_rows_f32_binding,
    assign_fold_i64_binding,
    count_fold_hits_i64_binding,
    split_table_i32_binding,
    epoch_order_i32_binding,
    adam_hyper_f64_binding,
    mean_std_f32_binding,
    strat_alloc_i64_binding,
    ocsvm_alpha_init_f32_binding,
    weighted_pick_i32_binding,
    draw_rows_without_replacement_i32_binding,
    weighted_draw_rows_i32_binding,
    group_fold_assign_i32_binding,
    strat_group_assign_i32_binding,
    strat_group_plan_i32_binding,
)
# lane fam2-shared (2026-10-04): these eighteen helpers run on the device in
# this binding (bindings/hotpath_device.mojo, same names and signatures; each
# falls back to its host helper of bindings/hotpath_helpers.mojo when its
# -D MOJOLEARN_IDN_HPDEV_*_OFF switch or -D MOJOLEARN_IDN_ALL_OFF is given).
from bindings.hotpath_device import (
    check_indices_i64_binding,
    equal_elements_binding,
    fold_ids_binding,
    gather_i32_binding,
    indices_overlap_i64_binding,
    select_fold_i64_binding,
    arange_i64_binding,
    leave_range_i64_binding,
    mask_from_indices_u8_binding,
    select_mask_u8_i64_binding,
    count_mask_u8_binding,
    fold_pair_f32_binding,
    threshold_labels_i64_binding,
    bincount_i64_binding,
    bincount2_i32_binding,
    first_seen_i32_binding,
    strat_fold_assign_i32_binding,
    hpdev_try_cast_f64_to_f32,
    # lane cpu2-l1-input: the input helpers on the device (no _OFF arm; the
    # host loops below and in bindings/array_helpers.mojo are the host column
    # and the fallback for what a twin does not cover)
    hpdev_try_all_finite,
    hpdev_try_transpose_to_f32,
    strided_copy_bytes_binding,
    check_lengths_i64_binding,
    ragged_rows_bytes_binding,
    reduce_stat_binding,
    uniform_init_f32_binding,
    normal_init_f32_binding,
    cast_elements_binding,
)
# lane cpu2-l4-modelsel (2026-10-04): model selection's resident fold store,
# fold-row gather, cross_val_predict scatter, binary proba column and the
# parallel forest's offset merge (bindings/msel_device.mojo).
from bindings.msel_device import (
    msel_put_binding,
    msel_alloc_binding,
    msel_read_binding,
    msel_free_binding,
    msel_live_binding,
    msel_take_rows_binding,
    msel_scatter_rows_binding,
    msel_proba_column_binding,
    msel_rebase_offsets_i32_binding,
    msel_split_table_i32_binding,
    msel_group_fold_perm_i32_binding,
)
from std.os import abort
from std.math import isfinite
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.numerics import GLOBAL_NUMERIC_MODE

from checks.vendor import COMPILED_VENDOR

from max.gpu.host import DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojoCoreContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoCoreContextFast"


from cluster.estimator import kmeans_fit, kmeans_predict, kmeans_transform
from cluster.impl.detail.kmeans import KMEANS_FAST_LAZY_SHIFT
from neighbors.impl.detail.knn_brute_force import KNN_METHOD_AUTO
from neighbors.resident_index import (
    knn_index_classify,
    knn_index_prepare,
    knn_index_regress,
    knn_index_release,
    knn_index_search,
)
from neighbors.estimator import (
    knn_classifier_predict,
    knn_classifier_from_neighbors,
    knn_regressor_from_neighbors,
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
    return f32_ptr(addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    """A caller's float64 buffer, borrowed, never owned; `_f32_ptr`'s twin
    (DEVIATION 2321). The same helper, spelled the same way, sits in
    `bindings/_mojolearn_gbdt.mojo`, `_mojolearn_estimators.mojo`,
    `_mojolearn_svm.mojo` and `_mojolearn_gp.mojo`."""
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return f64_ptr(addr)


def _u32_ptr(addr: Int) raises -> MutPointer[UInt32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return u32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return i32_ptr(addr)


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
        var ctx = process_ctx[_DEVCTX_SLOT]()
        used = knn_search(
            ctx, ip, ni, qp, nq, nf, kk, dp, xp, sq, qt, KNN_METHOD_AUTO,
            dt[0], dt[1], True,
        )
        ctx.synchronize()
    return PythonObject(used)


def knn_index_prepare_binding(
    index_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """DEVIATION 2921: upload a fitted index ONCE and return the handle
    `knn_search_resident` searches through. `params` is `[n_index,
    n_features]`; the bytes at `index_addr` are copied to the device
    inside this call and never read again by a search except for the
    host-side refusals. Release with `knn_index_release`."""
    if len(params) != 2:
        raise Error("knn_index_prepare: params must hold n_index, n_features")
    var ip = _f32_ptr(Int(py=index_addr))
    var ni = Int(py=params[0])
    var nf = Int(py=params[1])
    if ni <= 0 or nf <= 0:
        raise Error("knn_index_prepare: n_index and n_features must be positive")
    var handle: Int
    with GILReleased(Python()):
        handle = knn_index_prepare(ip, ni, nf)
    return PythonObject(handle)


def knn_index_release_binding(handle: PythonObject) raises -> PythonObject:
    """Drop a resident index (DEVIATION 2921)."""
    knn_index_release(Int(py=handle))
    return PythonObject(None)


def knn_search_resident_binding(
    handle: PythonObject,
    index_addr: PythonObject,
    queries_addr: PythonObject,
    out_dist_addr: PythonObject,
    out_idx_addr: PythonObject,
    params: PythonObject,
    dist_params: PythonObject,
) raises -> PythonObject:
    """`knn_search` over a resident index (DEVIATION 2921): the same
    `params` and `dist_params` as `knn_search_binding`, plus the handle
    `knn_index_prepare` returned first, and `index_addr` still the host
    bytes (the cosine zero-row refusal reads them). Returns the query tile
    that ran, as `knn_search` does."""
    if len(params) != 6:
        raise Error(
            "knn_search_resident: params must hold 6 values, got "
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
    var h = Int(py=handle)
    var used: Int
    with GILReleased(Python()):
        used = knn_index_search(
            h, ip, ni, qp, nq, nf, kk, dp, xp, sq, qt, KNN_METHOD_AUTO,
            dt[0], dt[1], True,
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
    for i in range(no):  # small-loop(no: model outputs, one class count each): parameter list, not data
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
        var ctx = process_ctx[_DEVCTX_SLOT]()
        used = knn_classifier_predict(
            ctx, ip, ni, qp, nq, nf, kk, yp, no, n_classes, lp, pp, up,
            want_proba, qt, dt[0], dt[1], dt[2],
        )
        ctx.synchronize()
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
        var ctx = process_ctx[_DEVCTX_SLOT]()
        used = knn_regressor_predict(
            ctx, ip, ni, qp, nq, nf, kk, yp, no, op, qt, dt[0], dt[1], dt[2]
        )
        ctx.synchronize()
    return PythonObject(used)


def knn_classify_resident_binding(
    index_addr: PythonObject,
    queries_addr: PythonObject,
    y_addr: PythonObject,
    out_labels_addr: PythonObject,
    out_proba_addr: PythonObject,
    out_uniq_addr: PythonObject,
    params: PythonObject,
    dist_params: PythonObject,
) raises -> PythonObject:
    """`knn_classify` over a resident index (DEVIATION 3002): the same
    arguments as `knn_classify_binding`, with the handle `knn_index_prepare`
    returned PREPENDED to `params` (a Python binding takes at most eight
    arguments, and the classifier already uses them all): params[0] is the
    handle, params[1..] the classifier's own list. `index_addr` is still
    the host bytes, read for the host-side refusals only."""
    if len(params) < 8:
        raise Error(
            "knn_classify_resident: params must hold at least 8 values, got "
            + String(len(params))
        )
    var h = Int(py=params[0])
    var ni = Int(py=params[1])
    var nq = Int(py=params[2])
    var nf = Int(py=params[3])
    var kk = Int(py=params[4])
    var qt = Int(py=params[5])
    var no = Int(py=params[6])
    var want_proba = Int(py=params[7]) != 0
    if len(params) != 8 + no:
        raise Error(
            "knn_classify_resident: params must hold 8 + n_outputs ("
            + String(8 + no)
            + ") values, got "
            + String(len(params))
        )
    var n_classes = List[Int]()
    for i in range(no):  # small-loop(no: model outputs, one class count each): parameter list, not data
        n_classes.append(Int(py=params[8 + i]))
    var ip = _f32_ptr(Int(py=index_addr))
    var qp = _f32_ptr(Int(py=queries_addr))
    var yp = _i32_ptr(Int(py=y_addr))
    var lp = _i32_ptr(Int(py=out_labels_addr))
    var pp = _f32_ptr(Int(py=out_proba_addr))
    var up = _i32_ptr(Int(py=out_uniq_addr))
    var dt = _dist_triple(dist_params)
    var used: Int
    with GILReleased(Python()):
        used = knn_index_classify(
            h, ip, ni, qp, nq, nf, kk, yp, no, n_classes, lp, pp, up,
            want_proba, qt, dt[0], dt[1], dt[2],
        )
    return PythonObject(used)


def knn_regress_resident_binding(
    handle: PythonObject,
    index_addr: PythonObject,
    queries_addr: PythonObject,
    y_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    dist_params: PythonObject,
) raises -> PythonObject:
    """`knn_regress` over a resident index (DEVIATION 3002): the same
    `params` and `dist_params` as `knn_regress_binding`, plus the handle."""
    if len(params) != 6:
        raise Error(
            "knn_regress: params must hold 6 values, got "
            + String(len(params))
        )
    var h = Int(py=handle)
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
        used = knn_index_regress(
            h, ip, ni, qp, nq, nf, kk, yp, no, op, qt, dt[0], dt[1], dt[2]
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
       10  oversampling_factor   (float; OPTIONAL, cuVS's 2.0 when the list
                                  holds ten values. 0.0 selects the classic
                                  sequential k-means++ seeding, anything
                                  positive the scalable k-means|| one;
                                  routed from python/mojolearn/cluster.py
                                  since 2026-09-14, workstream D)

    All four returns are given because a wrong answer here comes from the two
    scales, and a caller reproducing a result needs them. `inertia` is the
    weighted cost against the FINAL centroids from cuVS's post-loop
    assignment (`detail/kmeans.cuh:516-535`) and is always formed;
    `inertia_check` (False by default, per cuVS) governs only the IN-LOOP
    cost. This docstring used to say "0.0 when it was NEVER COMPUTED",
    which was false (corrected 2026-08-23).
    """
    if len(params) != 10 and len(params) != 11:
        raise Error(
            "kmeans_fit: params must hold 10 or 11 values, got "
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
    var ovs = Float64(2.0)
    if len(params) == 11:
        ovs = Float64(py=params[10])

    var inertia = Float64(0.0)
    var n_iter = 0
    var sum_scale = Float64(0.0)
    var weight_scale = Float64(0.0)
    with GILReleased(Python()):
        var ctx = process_ctx[_DEVCTX_SLOT]()
        var r = kmeans_fit(
            ctx, xp, ns, nf, nc, cp, lp, wp, nw, mi, tl, sd, ninit, ii, mm,
            0.0, ovs,
            # OPT-IN, unmeasured (2026-10-04): the lazy convergence read for the
            # KMeans estimator under -D MOJOLEARN_KMEANS_FAST_LAZY_SHIFT
            lazy_shift=KMEANS_FAST_LAZY_SHIFT,
        )
        inertia = r.inertia
        n_iter = r.n_iter
        sum_scale = r.sum_scale
        weight_scale = r.weight_scale
        ctx.synchronize()

    var out = Python.list()
    out.append(PythonObject(inertia))
    out.append(PythonObject(n_iter))
    out.append(PythonObject(sum_scale))
    out.append(PythonObject(weight_scale))
    return out


def kmeans_predict_binding(
    x_addr: PythonObject,
    centroids_addr: PythonObject,
    out_labels_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The nearest centroid of every row (cuML's `KMeans.predict`). Returns
    n_samples.

    `params` is, in this exact order: 0 n_samples, 1 n_features,
    2 n_clusters, 3 metric. `out_labels_addr` is `n_samples` uint32 (the
    caller's int32 array is the same bytes). The pass is `kmeans_fit`'s
    final assignment (`cluster/estimator.mojo::kmeans_predict`), so predict
    on the training rows is `labels_`."""
    if len(params) != 4:
        raise Error(
            "kmeans_predict: params must hold 4 values, got "
            + String(len(params))
        )
    var xp = _f32_ptr(Int(py=x_addr))
    var cp = _f32_ptr(Int(py=centroids_addr))
    var lp = _u32_ptr(Int(py=out_labels_addr))
    var ns = Int(py=params[0])
    var nf = Int(py=params[1])
    var nc = Int(py=params[2])
    var mm = Int(py=params[3])
    with GILReleased(Python()):
        var ctx = process_ctx[_DEVCTX_SLOT]()
        kmeans_predict(ctx, xp, ns, nf, nc, cp, lp, mm)
        ctx.synchronize()
    return PythonObject(ns)


def kmeans_transform_binding(
    x_addr: PythonObject,
    centroids_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The distance from every row to every centroid (cuML's
    `KMeans.transform`). Returns n_samples.

    `params` is, in this exact order: 0 n_samples, 1 n_features,
    2 n_clusters, 3 metric. `out_addr` is `n_samples x n_clusters` float32,
    row-major (`cluster/estimator.mojo::kmeans_transform`)."""
    if len(params) != 4:
        raise Error(
            "kmeans_transform: params must hold 4 values, got "
            + String(len(params))
        )
    var xp = _f32_ptr(Int(py=x_addr))
    var cp = _f32_ptr(Int(py=centroids_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var ns = Int(py=params[0])
    var nf = Int(py=params[1])
    var nc = Int(py=params[2])
    var mm = Int(py=params[3])
    with GILReleased(Python()):
        var ctx = process_ctx[_DEVCTX_SLOT]()
        kmeans_transform(ctx, xp, ns, nf, nc, cp, op, mm)
        ctx.synchronize()
    return PythonObject(ns)


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
        var ctx = process_ctx[_DEVCTX_SLOT]()
        nnz = radius_neighbors_count(
            ctx, ip, ni, qp, nq, nf, rad, ap, mtr, marg
        )
        ctx.synchronize()
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
        var ctx = process_ctx[_DEVCTX_SLOT]()
        nnz = radius_neighbors_fill(
            ctx, ip, ni, qp, nq, nf, rad, ap, xp, dp, cap, sq, mtr, marg
        )
        ctx.synchronize()
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
        var ctx = process_ctx[_DEVCTX_SLOT]()
        n_dists = rbc_knn_search(
            ctx, ip, ni, qp, nq, nf, kk, xp, dp, mtr, marg
        )
        ctx.synchronize()
    return PythonObject(n_dists)

# ===========================================================================
# HOST CONVERTERS, DEVIATION 2470, 2471, 2472 and 2489 (2026-09-10).
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
    # lane cpu2-l1-input: the device narrowing is the default (it was the
    # fam2-shared candidate arm); False means the host loop below runs.
    if hpdev_try_cast_f64_to_f32(Int(py=src_addr), Int(py=dst_addr), count):
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
    # lane cpu2-l1-input: the fused cast-and-transpose on the device.
    if hpdev_try_transpose_to_f32(Int(py=src_addr), Int(py=dst_addr), nr, nc, True):
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
    # lane cpu2-l1-input: the transpose on the device.
    if hpdev_try_transpose_to_f32(Int(py=src_addr), Int(py=dst_addr), nr, nc, False):
        return PythonObject(0)
    var sp = _f32_ptr(Int(py=src_addr))
    var dp = _f32_ptr(Int(py=dst_addr))
    with GILReleased(Python()):
        _tiled_transpose_to_f32(sp, dp, nr, nc)
    return PythonObject(0)


def nonzero_f64_count_binding(
    src_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """How many of the `n` float64 values at `src` are nonzero under the
    test `v != 0.0` (DEVIATION 2489). That is the Python loop's test, so
    -0.0 is a zero and NaN is NOT: NaN compares unequal to everything,
    `np.nonzero` kept it, the Python loop kept it, this keeps it. The
    count is what `nonzero_f64_fill` needs its three outputs sized to; the
    two-call shape is `radius_neighbors_count`/`_fill` above. An empty
    input reads nothing; a negative count is refused.
    """
    var count = Int(py=n)
    if count < 0:
        raise Error(
            "nonzero_f64_count: n must be non-negative, got " + String(count)
        )
    if count == 0:
        return PythonObject(0)
    var sp = _f64_ptr(Int(py=src_addr))
    var nz = 0
    with GILReleased(Python()):
        for i in range(count):
            if sp.unsafe_load(i) != 0.0:
                nz += 1
    return PythonObject(nz)


def nonzero_f64_fill_binding(
    src_addr: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
    outs: PythonObject,
    capacity: PythonObject,
) raises -> PythonObject:
    """COO triples of a C-contiguous float64 `[rows, cols]` matrix at `src`
    (DEVIATION 2489): for every entry with `v != 0.0`, in row-major scan
    order, write its row to `outs[0]` (int32), its column to `outs[1]`
    (int32) and `Float32(v)` to `outs[2]` (float32), the same
    round-to-nearest-even narrowing `array.array('f')` applies. Returns the
    number written. `capacity` is the length of each output; the fill
    STOPS and raises if the matrix holds more nonzeros than that, so a
    stale count cannot write past the buffers. Byte-for-byte the output of
    `_spectral_impl._coo_triples`'s Python loop, which is the oracle.
    """
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    var cap = Int(py=capacity)
    if nr < 0 or nc < 0:
        raise Error(
            "nonzero_f64_fill: rows and cols must be non-negative, got "
            + String(nr) + " x " + String(nc)
        )
    if cap < 0:
        raise Error(
            "nonzero_f64_fill: capacity must be non-negative, got "
            + String(cap)
        )
    if len(outs) != 3:
        raise Error(
            "nonzero_f64_fill: outs must hold 3 addresses, got "
            + String(len(outs))
        )
    if nr == 0 or nc == 0:
        return PythonObject(0)
    var sp = _f64_ptr(Int(py=src_addr))
    var rp = _i32_ptr(Int(py=outs[0]))
    var cp = _i32_ptr(Int(py=outs[1]))
    var vp = _f32_ptr(Int(py=outs[2]))
    var k = 0
    var overflow = False
    with GILReleased(Python()):
        for r in range(nr):
            var base = r * nc
            for c in range(nc):
                var v = sp.unsafe_load(base + c)
                if v != 0.0:
                    if k >= cap:
                        overflow = True
                        break
                    rp.unsafe_store(k, Int32(r))
                    cp.unsafe_store(k, Int32(c))
                    vp.unsafe_store(k, v.cast[DType.float32]())
                    k += 1
            if overflow:
                break
    if overflow:
        raise Error(
            "nonzero_f64_fill: more than " + String(cap)
            + " nonzero entries; count first with nonzero_f64_count"
        )
    return PythonObject(k)


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
# Their host loops construct no `DeviceContext`: each is a single pass over
# a host buffer, reached only when the device scan (lane cpu2-l1-input,
# `hpdev_try_all_finite`) declines or in a sabotage build. The GIL is still released around the pass (the
# `var result: Int` / `with GILReleased(Python())` shape of
# `knn_search_binding`) because nothing inside touches a Python object and
# a caller scanning a million rows should not stall its other threads.
# ===========================================================================


def _all_finite_host_f32(p: MutPointer[Float32, MutUntrackedOrigin], count: Int) -> Int:
    """Host replay of the device finiteness scan: reached only from a
    HOTPATH_SABOTAGE build (the tests' sabotage switch) or `n == 0`."""
    for i in range(count):
        if not isfinite(p.unsafe_load(i)):
            return 0
    return 1


def _all_finite_host_f64(p: MutPointer[Float64, MutUntrackedOrigin], count: Int) -> Int:
    """`_all_finite_host_f32` over float64 values."""
    for i in range(count):
        if not isfinite(p.unsafe_load(i)):
            return 0
    return 1


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
    # lane cpu2-l1-input: the scan on the device (one status word back).
    var dev = hpdev_try_all_finite(Int(py=addr), count, False)
    if dev >= 0:
        return PythonObject(dev)
    # only a HOTPATH_SABOTAGE build (or n == 0) reaches the host replay
    var ok: Int = 1
    with GILReleased(Python()):
        ok = _all_finite_host_f32(p, count)
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
    var dev = hpdev_try_all_finite(Int(py=addr), count, True)
    if dev >= 0:
        return PythonObject(dev)
    # only a HOTPATH_SABOTAGE build (or n == 0) reaches the host replay
    var ok: Int = 1
    with GILReleased(Python()):
        ok = _all_finite_host_f64(p, count)
    return PythonObject(ok)



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
    if ns > 2147483000 // width or no > 2147483000 // width:
        raise Error("gather_rows_bytes: more bytes than the device gather holds")
    # lane cpu4-python: on the device (core/rows_bytes_device.mojo; indices
    # checked before any write, byte moves); the host walk is the host
    # binding's column (bindings/host_helpers.mojo)
    var sa = Int(py=src_addr)
    var da = Int(py=dst_addr)
    var ia = Int(py=indices_addr)
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var ok = False
    with GILReleased(Python()):
        ok = device_gather_rows_bytes(ctx, sa, da, ia, ns, no, width)
    if not ok:
        raise Error("gather_rows_bytes: row index out of bounds")
    return PythonObject(0)


def scatter_rows_bytes_binding(
    src_addr: PythonObject, rows_addr: PythonObject, m: PythonObject, row_bytes: PythonObject,
    dst_addr: PythonObject, dst_rows: PythonObject,
) raises -> PythonObject:
    """dst row rows[i] = src row i (row_bytes each) for the m int64 rows; a
    row outside [0, dst_rows) raises before any write; a repeated row takes
    the last source row naming it. On the device (lane cpu4-python,
    core/rows_bytes_device.mojo); the host walk is the host binding's
    column (bindings/hotpath_helpers.mojo)."""
    var count = Int(py=m)
    var width = Int(py=row_bytes)
    var nd = Int(py=dst_rows)
    if count < 0 or width < 0 or nd < 0:
        raise Error("scatter_rows_bytes: dimensions must be non-negative")
    if count == 0 or width == 0:
        return PythonObject(0)
    if nd == 0:
        raise Error("scatter_rows_bytes: row index out of bounds")
    if count > 2147483000 // width or nd > 2147483000 // width:
        raise Error("scatter_rows_bytes: more bytes than the device scatter holds")
    var sa = Int(py=src_addr)
    var ra = Int(py=rows_addr)
    var da = Int(py=dst_addr)
    if sa == 0 or ra == 0 or da == 0:
        raise Error("scatter_rows_bytes: null buffer address")
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var ok = False
    with GILReleased(Python()):
        ok = device_scatter_rows_bytes(ctx, sa, ra, count, width, da, nd)
    if not ok:
        raise Error("scatter_rows_bytes: row index out of bounds")
    return PythonObject(0)


# ===========================================================================
# LABEL ENCODING, DEVIATION 2500 (2026-09-10).
#
# `python/mojolearn/_labels.py::sorted_classes` defines the `classes_` ORDER
# RULE (DEVIATION 2340) and encodes `y` with one dict lookup per row. That is
# O(rows) Python: measured at 1,000,000 float32 labels on the M4 it is 60 ms
# to unpack the buffer into Python objects, 210 ms to group and encode and
# 130 ms to pack the codes, about 400 ms of a 2.3 s RandomForest fit, and
# more on a slower host CPU (the H100 leg's RF round was 2,343 ms of which
# 1,406 ms was the Mojo fit). This helper is the SAME rule for the case of
# one contiguous numeric buffer, computed in compiled code with the GIL
# released; the Python routine stays for lists, strings and mixed objects.
#
# The rule, restated for one numeric dtype: classes are the distinct values
# under numeric equality, sorted ascending; the representative kept is the
# FIRST value seen (so `-0.0` and `0.0` are one class and the first spelling
# wins); a NaN label is refused. Lane cpu2-l2-labels (2026-10-04): in this
# GPU binding the encoder is the device's alone, on every tier (the sort,
# flag/scan and gather of `core/hotpath_device.mojo::device_encode_labels`);
# the host insertion-array encoder that ran here (and still ran on FAST and
# on every refusal) is gone. The host column keeps its copy in
# `bindings/hotpath_helpers.mojo`. More distinct values than `max_classes`,
# or a NaN label, returns -1 and the caller takes `unique_inverse` (also the
# device), which has no class cap and raises the NaN refusal.
# ===========================================================================


def _encode_labels_binding[dt: DType](
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    """The ORDER RULE's encoder over `n` values of `dt` at `src_addr`, on the
    device: the sorted distinct values land at `classes_addr` (capacity
    `max_classes`) and one int32 code per row at `codes_addr`. Returns the
    class count, or -1 when the cap was exceeded or a label is NaN (nothing
    is promised about either output then; `_labels._encode_labels_native`
    then runs `unique_inverse`)."""
    var count = Int(py=n)
    var cap = Int(py=max_classes)
    if count < 1:
        raise Error("encode_labels: n must be positive, got " + String(count))
    if cap < 1:
        raise Error("encode_labels: max_classes must be positive")
    if Int(py=src_addr) == 0 or Int(py=classes_addr) == 0 or Int(py=codes_addr) == 0:
        raise Error("encode_labels: null buffer address")
    if count > HPD_MAX_N:
        return PythonObject(-1)
    var code = -1
    comptime if dt == DType.float32:
        code = HPD_F32
    comptime if dt == DType.float64:
        code = HPD_F64
    comptime if dt == DType.int32:
        code = HPD_I32
    comptime if dt == DType.int64:
        code = HPD_I64
    comptime if dt == DType.uint32:
        code = HPD_U32
    comptime if dt == DType.uint8:
        code = HPD_U8
    if code < 0:
        raise Error("encode_labels: dtype without a device encoder")
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var k = -1
    with GILReleased(Python()):
        k = device_encode_labels(
            ctx, Int(py=src_addr), code, count, Int(py=classes_addr), cap, Int(py=codes_addr)
        )
    return PythonObject(k)


def encode_labels_f32_binding(
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    return _encode_labels_binding[DType.float32](src_addr, n, classes_addr, max_classes, codes_addr)


def encode_labels_f64_binding(
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    return _encode_labels_binding[DType.float64](src_addr, n, classes_addr, max_classes, codes_addr)


def encode_labels_i32_binding(
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    return _encode_labels_binding[DType.int32](src_addr, n, classes_addr, max_classes, codes_addr)


def encode_labels_i64_binding(
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    return _encode_labels_binding[DType.int64](src_addr, n, classes_addr, max_classes, codes_addr)


def encode_labels_u32_binding(
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    return _encode_labels_binding[DType.uint32](src_addr, n, classes_addr, max_classes, codes_addr)


def encode_labels_u8_binding(
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    return _encode_labels_binding[DType.uint8](src_addr, n, classes_addr, max_classes, codes_addr)


def _gather_u64_device(
    name: String, table_addr: PythonObject, n_table: PythonObject,
    codes_addr: PythonObject, n: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """`dst[i] = table[codes[i]]` over 64-bit table words and int64 codes
    on the device (`core/hotpath_device.mojo::device_gather_u64`, a move of
    words, so one kernel serves int64 and float64 tables). Lane
    cpu2-l2-labels: the host loop that ran here on FAST and on every refusal
    is gone; the host column keeps it (`bindings/host_helpers.mojo`). A code
    outside `[0, n_table)` raises before any byte of `dst` is written."""
    var count = Int(py=n)
    var nt = Int(py=n_table)
    if count < 0 or nt < 1:
        raise Error(name + ": n must be non-negative and the table non-empty")
    if count == 0:
        return PythonObject(0)
    if Int(py=table_addr) == 0 or Int(py=codes_addr) == 0 or Int(py=dst_addr) == 0:
        raise Error(name + ": null buffer address")
    if count > HPD_MAX_N or nt > HPD_MAX_N:
        raise Error(name + ": more rows than the device gather holds")
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var ok = False
    with GILReleased(Python()):
        ok = device_gather_u64(
            ctx, Int(py=table_addr), nt, Int(py=codes_addr), count, Int(py=dst_addr)
        )
    if not ok:
        raise Error(name + ": code out of range")
    return PythonObject(0)


def gather_i64_binding(
    table_addr: PythonObject, n_table: PythonObject, codes_addr: PythonObject,
    n: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """`dst[i] = table[codes[i]]` over int64 tables (DEVIATION 2500): the
    decode half of label encoding, `classes_[code]` per predicted row, on
    the device. A code outside `[0, n_table)` raises before any write."""
    return _gather_u64_device("gather_i64", table_addr, n_table, codes_addr, n, dst_addr)


def gather_f64_binding(
    table_addr: PythonObject, n_table: PythonObject, codes_addr: PythonObject,
    n: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """`gather_i64_binding` over a float64 table (int64 codes)."""
    return _gather_u64_device("gather_f64", table_addr, n_table, codes_addr, n, dst_addr)


def _argmax_rows_device(
    name: String, wide: Bool, scores_addr: PythonObject, n_rows: PythonObject,
    n_cols: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """Row-wise first-max-wins argmax over a C-order [n_rows, n_cols] block
    into int64 codes (DEVIATION 2500), the rule of `_labels.argmax_rows`:
    strictly greater replaces, so ties keep the lowest column, and a NaN
    never replaces, so a row whose column 0 is NaN answers 0. On the device
    (`core/label_rows_device.mojo`, lane cpu2-l2-labels: integer order keys
    of the IEEE words, the host column's bytes on every vendor); the host
    loop that ran here is gone, the host column keeps it
    (`bindings/host_helpers.mojo`)."""
    var rows = Int(py=n_rows)
    var cols = Int(py=n_cols)
    if rows < 0 or cols < 1:
        raise Error(name + ": n_rows must be non-negative and n_cols positive")
    if rows == 0:
        return PythonObject(0)
    if Int(py=scores_addr) == 0 or Int(py=dst_addr) == 0:
        raise Error(name + ": null buffer address")
    if rows > LRD_MAX_N or cols > LRD_MAX_N:
        raise Error(name + ": more rows or columns than the device argmax holds")
    var ctx = process_ctx[_DEVCTX_SLOT]()
    with GILReleased(Python()):
        device_argmax_rows(ctx, Int(py=scores_addr), wide, rows, cols, Int(py=dst_addr))
    return PythonObject(0)


def argmax_rows_f32_binding(
    scores_addr: PythonObject, n_rows: PythonObject, n_cols: PythonObject,
    dst_addr: PythonObject,
) raises -> PythonObject:
    """`_argmax_rows_device` over float32 scores."""
    return _argmax_rows_device("argmax_rows_f32", False, scores_addr, n_rows, n_cols, dst_addr)


def argmax_rows_f64_binding(
    scores_addr: PythonObject, n_rows: PythonObject, n_cols: PythonObject,
    dst_addr: PythonObject,
) raises -> PythonObject:
    """`_argmax_rows_device` over float64 scores."""
    return _argmax_rows_device("argmax_rows_f64", True, scores_addr, n_rows, n_cols, dst_addr)


def shard_topk_merge_f32_binding(
    table_addr: PythonObject, n_shards: PythonObject, n_queries: PythonObject, k: PythonObject,
    out_dist_addr: PythonObject, out_idx_addr: PythonObject,
) raises -> PythonObject:
    """`shard_topk_merge_f32` (bindings/array_helpers.mojo's contract: 0, 1
    a local id outside its shard, 2 fewer than k candidates) on the device
    (lane cpu4-python, core/shard_merge_device.mojo); the host loop is the
    host binding's column."""
    var s_count = Int(py=n_shards)
    var nq = Int(py=n_queries)
    var kk = Int(py=k)
    if s_count < 1 or nq < 0 or kk < 1:
        raise Error("shard_topk_merge_f32: bad dimensions")
    if nq == 0:
        return PythonObject(0)
    if Int(py=table_addr) == 0 or Int(py=out_dist_addr) == 0 or Int(py=out_idx_addr) == 0:
        raise Error("shard_topk_merge_f32: null buffer address")
    var tp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=table_addr))
    var od = Int(py=out_dist_addr)
    var oi = Int(py=out_idx_addr)
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var rc = 0
    with GILReleased(Python()):
        rc = device_shard_topk_merge_f32(ctx, tp, s_count, nq, kk, od, oi)
    return PythonObject(rc)


def _ic_running_min_device(
    name: String, wide: Bool, llf_addr: PythonObject, n: PythonObject, penalty: PythonObject,
    order: PythonObject, ic_addr: PythonObject, best_ic_addr: PythonObject, best_idx_addr: PythonObject,
) raises -> PythonObject:
    """`ic_running_min_f64` / `_f32` (bindings/hotpath_helpers.mojo's
    contract) on the device (lane cpu4-python, core/ic_min_device.mojo);
    the host walk is the host binding's column."""
    var count = Int(py=n)
    var k = Int(py=order)
    if count < 1 or k < 0:
        raise Error(name + ": n must be positive and order non-negative")
    if count > 2147483000:
        raise Error(name + ": more series than the device holds")
    var la = Int(py=llf_addr)
    var ia = Int(py=ic_addr)
    var ba = Int(py=best_ic_addr)
    var xa = Int(py=best_idx_addr)
    if la == 0 or ia == 0 or ba == 0 or xa == 0:
        raise Error(name + ": null buffer address")
    var pen = Float64(py=penalty)
    var ctx = process_ctx[_DEVCTX_SLOT]()
    with GILReleased(Python()):
        device_ic_running_min(ctx, wide, la, count, pen, k, ia, ba, xa)
    return PythonObject(0)


def ic_running_min_f64_binding(
    llf_addr: PythonObject, n: PythonObject, penalty: PythonObject, order: PythonObject,
    ic_addr: PythonObject, best_ic_addr: PythonObject, best_idx_addr: PythonObject,
) raises -> PythonObject:
    return _ic_running_min_device("ic_running_min_f64", True, llf_addr, n, penalty, order,
                                  ic_addr, best_ic_addr, best_idx_addr)


def ic_running_min_f32_binding(
    llf_addr: PythonObject, n: PythonObject, penalty: PythonObject, order: PythonObject,
    ic_addr: PythonObject, best_ic_addr: PythonObject, best_idx_addr: PythonObject,
) raises -> PythonObject:
    return _ic_running_min_device("ic_running_min_f32", False, llf_addr, n, penalty, order,
                                  ic_addr, best_ic_addr, best_idx_addr)


def argmax_last_rows_f32_binding(
    logits_addr: PythonObject, dims: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """dims = [b, l, v]: the first-max argmax (DEVIATION 2500's rule) of the
    LAST position of each row of float32 logits `[b, l, v]` into int64
    `dst[b]`, on the device (lane cpu4-python: the callers' host row gather
    `gather_rows_bytes` before `argmax_rows_f32` is gone)."""
    if len(dims) != 3:
        raise Error("argmax_last_rows_f32: dims [b, l, v]")
    var b = Int(py=dims[0])
    var l = Int(py=dims[1])
    var v = Int(py=dims[2])
    if b < 0 or l < 1 or v < 1:
        raise Error("argmax_last_rows_f32: b must be non-negative, l and v positive")
    if b == 0:
        return PythonObject(0)
    if Int(py=logits_addr) == 0 or Int(py=dst_addr) == 0:
        raise Error("argmax_last_rows_f32: null buffer address")
    if b > LRD_MAX_N // l or v > LRD_MAX_N:
        raise Error("argmax_last_rows_f32: more rows or columns than the device argmax holds")
    var la = Int(py=logits_addr)
    var da = Int(py=dst_addr)
    var ctx = process_ctx[_DEVCTX_SLOT]()
    with GILReleased(Python()):
        device_argmax_last_f32(ctx, la, b, l, v, da)
    return PythonObject(0)


def kmeans_parallel_available_binding() raises -> PythonObject:
    """Version of the whole-row-tile multi-GPU assignment contract."""
    return PythonObject(1)


# ===========================================================================
# THE FLOAT32 DENSE-TO-COO SCAN AND THE PRECOMPUTED kNN AFFINITY
# (lane/py-dn-kern, 2026-09-28): `core/dense_coo.mojo`, the same bodies in
# the base binding and its CPU route. They replace the Python n^2 loops of
# `_spectral_impl._DenseCOO` and `SpectralEmbedding._precomputed_knn_affinity`
# (compares and the exact values 0, 0.5, 1; no fold).
# ===========================================================================


def nonzero_f32_count_binding(src_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """How many of the `n` float32 values at `src` satisfy `v != 0.0`."""
    var count = Int(py=n)
    if count < 0:
        raise Error("nonzero_f32_count: n must be non-negative, got " + String(count))
    if count == 0:
        return PythonObject(0)
    var sp = f32_ptr(Int(py=src_addr))
    var nz = 0
    with GILReleased(Python()):
        nz = dense_nonzero_f32_count(sp, count)
    return PythonObject(nz)


def nonzero_f32_fill_binding(
    src_addr: PythonObject, rows: PythonObject, cols: PythonObject, outs: PythonObject, capacity: PythonObject,
) raises -> PythonObject:
    """`nonzero_f64_fill` over a C-contiguous float32 `[rows, cols]` matrix:
    row, column and value of every `v != 0.0`, row major. Returns the count."""
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    var cap = Int(py=capacity)
    if nr < 0 or nc < 0 or cap < 0:
        raise Error("nonzero_f32_fill: rows, cols and capacity must be non-negative")
    if len(outs) != 3:
        raise Error("nonzero_f32_fill: outs must hold 3 addresses, got " + String(len(outs)))
    if nr == 0 or nc == 0:
        return PythonObject(0)
    var sp = f32_ptr(Int(py=src_addr))
    var rp = i32_ptr(Int(py=outs[0]))
    var cp = i32_ptr(Int(py=outs[1]))
    var vp = f32_ptr(Int(py=outs[2]))
    var k = 0
    with GILReleased(Python()):
        k = dense_nonzero_f32_fill(sp, nr, nc, rp, cp, vp, cap)
    return PythonObject(k)


def knn_affinity_f32_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """addrs = [dense, rows, cols, vals, affinity, status (3 int32)] (0 where unused);
    params = [n, k, nnz, sparse]. See `core/dense_coo.mojo::knn_affinity_f32`."""
    if len(addrs) != 6 or len(params) != 4:
        raise Error("knn_affinity_f32: needs 6 addresses and 4 parameters")
    var n = Int(py=params[0])
    var k = Int(py=params[1])
    var nnz = Int(py=params[2])
    var sparse = Int(py=params[3]) != 0
    if n < 1 or k < 0 or nnz < 0:
        raise Error("knn_affinity_f32: n must be positive, k and nnz non-negative")
    var aff = Int(f32_ptr(Int(py=addrs[4])))
    var status = Int(i32_ptr(Int(py=addrs[5])))
    var dense = 0
    var rp = 0
    var cp = 0
    var vp = 0
    if sparse:
        if nnz > 0:
            rp = Int(i32_ptr(Int(py=addrs[1])))
            cp = Int(i32_ptr(Int(py=addrs[2])))
            vp = Int(f32_ptr(Int(py=addrs[3])))
    else:
        dense = Int(f32_ptr(Int(py=addrs[0])))
    # the GPU binding's work on the device (core/dense_coo_device.mojo,
    # cpu-gpu-cleanup c-core); the CPU route keeps core/dense_coo.mojo
    var ctx = process_ctx[_DEVCTX_SLOT]()
    with GILReleased(Python()):
        knn_affinity_f32_device(ctx, dense, rp, cp, vp, nnz, sparse, n, k, aff, status)
    return PythonObject(0)


def unique_inverse_binding(
    src_addr: PythonObject, n: PythonObject, kind: PythonObject,
    classes_addr: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    """`np.unique(y, return_inverse=True)` for `n` 64-bit labels at
    `src_addr` (`kind` 0: float64, 1: int64) on the device
    (`core/label_encode_device.mojo`, cpu-gpu-cleanup w2-pyglue): the sorted
    distinct values' 64 bits land at `classes_addr` (n slots), one int32
    code per row at `codes_addr`. Returns the class count. A NaN label
    raises the ORDER RULE's message. The core host binding runs the host
    twin (`core/label_encode.mojo`) under this name."""
    var count = Int(py=n)
    var k_ind = Int(py=kind)
    if count < 1:
        raise Error("unique_inverse: n must be positive, got " + String(count))
    if k_ind != 0 and k_ind != 1:
        raise Error("unique_inverse: kind must be 0 (float64) or 1 (int64)")
    if Int(py=src_addr) == 0 or Int(py=classes_addr) == 0 or Int(py=codes_addr) == 0:
        raise Error("unique_inverse: null buffer address")
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var k: Int
    with GILReleased(Python()):
        k = device_unique_inverse(
            ctx, Int(py=src_addr), count, k_ind, Int(py=classes_addr), Int(py=codes_addr)
        )
    if k == -2:
        raise Error("mojolearn: y contains a NaN label; NaN is not a class")
    return PythonObject(k)


@export
def PyInit__mojolearn() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn")
        m.def_function[mojolearn_vendor_binding]("mojolearn_vendor")
        m.def_function[mojolearn_numeric_mode_binding]("mojolearn_numeric_mode")
        m.def_function[knn_search_binding]("knn_search")
        m.def_function[knn_index_prepare_binding]("knn_index_prepare")
        m.def_function[knn_index_release_binding]("knn_index_release")
        m.def_function[knn_search_resident_binding]("knn_search_resident")
        m.def_function[knn_classify_neighbors_binding]("knn_classify_neighbors")
        m.def_function[knn_regress_neighbors_binding]("knn_regress_neighbors")
        m.def_function[knn_classify_binding]("knn_classify")
        m.def_function[knn_regress_binding]("knn_regress")
        m.def_function[knn_classify_resident_binding]("knn_classify_resident")
        m.def_function[knn_regress_resident_binding]("knn_regress_resident")
        m.def_function[kmeans_fit_binding]("kmeans_fit")
        m.def_function[kmeans_predict_binding]("kmeans_predict")
        m.def_function[kmeans_transform_binding]("kmeans_transform")
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
        m.def_function[nonzero_f64_count_binding]("nonzero_f64_count")
        m.def_function[nonzero_f64_fill_binding]("nonzero_f64_fill")
        m.def_function[nonzero_f32_count_binding]("nonzero_f32_count")
        m.def_function[nonzero_f32_fill_binding]("nonzero_f32_fill")
        m.def_function[knn_affinity_f32_binding]("knn_affinity_f32")
        m.def_function[unique_inverse_binding]("unique_inverse")
        # DEVIATION 2325: the three host helpers of DEVIATION 2303.
        m.def_function[all_finite_f32_binding]("all_finite_f32")
        m.def_function[all_finite_f64_binding]("all_finite_f64")
        m.def_function[probability_rows_f32_binding]("probability_rows_f32")
        m.def_function[gather_rows_bytes_binding]("gather_rows_bytes")
        # lane cpu2-l4-modelsel: model selection on the device (host column on the core host binding)
        m.def_function[msel_put_binding]("msel_put")
        m.def_function[msel_alloc_binding]("msel_alloc")
        m.def_function[msel_read_binding]("msel_read")
        m.def_function[msel_free_binding]("msel_free")
        m.def_function[msel_live_binding]("msel_live")
        m.def_function[msel_take_rows_binding]("msel_take_rows")
        m.def_function[msel_scatter_rows_binding]("msel_scatter_rows")
        m.def_function[msel_proba_column_binding]("msel_proba_column")
        m.def_function[msel_rebase_offsets_i32_binding]("msel_rebase_offsets_i32")
        m.def_function[msel_split_table_i32_binding]("msel_split_table_i32")
        m.def_function[msel_group_fold_perm_i32_binding]("msel_group_fold_perm_i32")
        # DEVIATION 2500: label encode/decode and the class argmax, host side.
        m.def_function[encode_labels_f32_binding]("encode_labels_f32")
        m.def_function[encode_labels_f64_binding]("encode_labels_f64")
        m.def_function[encode_labels_i32_binding]("encode_labels_i32")
        m.def_function[encode_labels_i64_binding]("encode_labels_i64")
        m.def_function[encode_labels_u32_binding]("encode_labels_u32")
        m.def_function[encode_labels_u8_binding]("encode_labels_u8")
        # lane/python-hotpath (2026-09-17, DEVIATIONS 3100-3104): the helpers of
        # bindings/hotpath_helpers.mojo that stand in for per-row Python.
        m.def_function[cast_elements_binding]("cast_elements")
        m.def_function[reduce_stat_binding]("reduce_stat")
        m.def_function[equal_elements_binding]("equal_elements")
        m.def_function[gather_i32_binding]("gather_i32")
        m.def_function[check_indices_i64_binding]("check_indices_i64")
        m.def_function[indices_overlap_i64_binding]("indices_overlap_i64")
        m.def_function[fold_ids_binding]("fold_ids")
        m.def_function[select_fold_i64_binding]("select_fold_i64")
        m.def_function[arange_i64_binding]("arange_i64")
        m.def_function[leave_range_i64_binding]("leave_range_i64")
        m.def_function[mask_from_indices_u8_binding]("mask_from_indices_u8")
        m.def_function[select_mask_u8_i64_binding]("select_mask_u8_i64")
        m.def_function[count_mask_u8_binding]("count_mask_u8")
        m.def_function[next_combination_i64_binding]("next_combination_i64")
        m.def_function[ic_running_min_f64_binding]("ic_running_min_f64")
        m.def_function[ic_running_min_f32_binding]("ic_running_min_f32")
        m.def_function[fold_pair_f32_binding]("fold_pair_f32")
        m.def_function[threshold_labels_i64_binding]("threshold_labels_i64")
        m.def_function[scale_shift_ftz_f32_binding]("scale_shift_ftz_f32")
        m.def_function[bincount_i64_binding]("bincount_i64")
        m.def_function[compact_notnan_f32_binding]("compact_notnan_f32")
        m.def_function[gather_keep_neg_i32_binding]("gather_keep_neg_i32")
        m.def_function[dot_rows_f32_binding]("dot_rows_f32")
        m.def_function[assign_fold_i64_binding]("assign_fold_i64")
        m.def_function[count_fold_hits_i64_binding]("count_fold_hits_i64")
        m.def_function[split_table_i32_binding]("split_table_i32")
        m.def_function[scatter_rows_bytes_binding]("scatter_rows_bytes")
        m.def_function[uniform_init_f32_binding]("uniform_init_f32")
        m.def_function[normal_init_f32_binding]("normal_init_f32")
        m.def_function[epoch_order_i32_binding]("epoch_order_i32")
        m.def_function[adam_hyper_f64_binding]("adam_hyper_f64")
        m.def_function[mean_std_f32_binding]("mean_std_f32")
        m.def_function[first_seen_i32_binding]("first_seen_i32")
        m.def_function[strat_fold_assign_i32_binding]("strat_fold_assign_i32")
        m.def_function[strat_alloc_i64_binding]("strat_alloc_i64")
        m.def_function[ocsvm_alpha_init_f32_binding]("ocsvm_alpha_init_f32")
        m.def_function[weighted_pick_i32_binding]("weighted_pick_i32")
        m.def_function[draw_rows_without_replacement_i32_binding]("draw_rows_without_replacement_i32")
        m.def_function[weighted_draw_rows_i32_binding]("weighted_draw_rows_i32")
        m.def_function[group_fold_assign_i32_binding]("group_fold_assign_i32")
        m.def_function[strat_group_assign_i32_binding]("strat_group_assign_i32")
        m.def_function[strat_group_plan_i32_binding]("strat_group_plan_i32")
        m.def_function[bincount2_i32_binding]("bincount2_i32")
        m.def_function[strided_copy_bytes_binding]("strided_copy_bytes")
        m.def_function[check_lengths_i64_binding]("check_lengths_i64")
        m.def_function[ragged_rows_bytes_binding]("ragged_rows_bytes")
        m.def_function[nsum_f64_binding]("nsum_f64")
        m.def_function[class_ratio_f64_binding]("class_ratio_f64")
        m.def_function[shard_topk_merge_f32_binding]("shard_topk_merge_f32")
        m.def_function[gather_i64_binding]("gather_i64")
        m.def_function[gather_f64_binding]("gather_f64")
        m.def_function[argmax_rows_f32_binding]("argmax_rows_f32")
        m.def_function[argmax_last_rows_f32_binding]("argmax_last_rows_f32")
        m.def_function[argmax_rows_f64_binding]("argmax_rows_f64")
        m.def_function[kmeans_parallel_available_binding]("kmeans_parallel_available")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn module: ", e))


def knn_classify_neighbors_binding(
    dist_addr: PythonObject,
    idx_addr: PythonObject,
    y_addr: PythonObject,
    out_labels_addr: PythonObject,
    out_proba_addr: PythonObject,
    out_uniq_addr: PythonObject,
    params: PythonObject,
    dist_params: PythonObject,
) raises -> PythonObject:
    """Vote on globally merged neighbors. Original parameter layout; feature
    count and query tile slots are unused. Inputs: float32 distances and
    uint32 global indices, both n_queries x k. Returns zero on success."""
    if len(params) < 7:
        raise Error(
            "knn_classify: params must hold at least 7 values, got "
            + String(len(params))
        )
    var ni = Int(py=params[0])
    var nq = Int(py=params[1])
    var kk = Int(py=params[3])
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
    for i in range(no):  # small-loop(no: model outputs, one class count each): parameter list, not data
        n_classes.append(Int(py=params[7 + i]))
    var dp = _f32_ptr(Int(py=dist_addr))
    var xp = _u32_ptr(Int(py=idx_addr))
    var yp = _i32_ptr(Int(py=y_addr))
    var lp = _i32_ptr(Int(py=out_labels_addr))
    var pp = _f32_ptr(Int(py=out_proba_addr))
    var up = _i32_ptr(Int(py=out_uniq_addr))
    var dt = _dist_triple(dist_params)

    with GILReleased(Python()):
        var ctx = process_ctx[_DEVCTX_SLOT]()
        knn_classifier_from_neighbors(
            ctx, dp, xp, ni, nq, kk, yp, no, n_classes, lp, pp, up,
            want_proba, dt[2],
        )
        ctx.synchronize()
    return PythonObject(0)


def knn_regress_neighbors_binding(
    dist_addr: PythonObject,
    idx_addr: PythonObject,
    y_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    dist_params: PythonObject,
) raises -> PythonObject:
    """Regress on globally merged neighbors. Original parameter layout;
    feature count and query tile slots are unused. Inputs: float32 distances
    and uint32 global indices, both n_queries x k. Returns zero on success."""
    if len(params) != 6:
        raise Error(
            "knn_regress: params must hold 6 values, got "
            + String(len(params))
        )
    var dp = _f32_ptr(Int(py=dist_addr))
    var xp = _u32_ptr(Int(py=idx_addr))
    var yp = _f32_ptr(Int(py=y_addr))
    var op = _f32_ptr(Int(py=out_addr))
    var ni = Int(py=params[0])
    var nq = Int(py=params[1])
    var kk = Int(py=params[3])
    var no = Int(py=params[5])
    var dt = _dist_triple(dist_params)

    with GILReleased(Python()):
        var ctx = process_ctx[_DEVCTX_SLOT]()
        knn_regressor_from_neighbors(ctx, dp, xp, ni, nq, kk, yp, no, op, dt[2])
        ctx.synchronize()
    return PythonObject(0)
