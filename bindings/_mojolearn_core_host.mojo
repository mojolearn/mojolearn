# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the base `_mojolearn` family: its HOST HELPERS (the CPU
training lane, phase 1, 2026-09-13) and, since the knn host inference lane (2026-09-14), the three k-NN INFERENCE entries.

HOST ONLY. The helpers carry NO ARITHMETIC. `python/mojolearn/_buffer.py::_native`
resolves the input converters every estimator funnels its arrays through
from the base binding (`bindings/_mojolearn.mojo`): a float64-to-float32
cast, a transpose into column-major order, a finiteness predicate, and the
label helpers of `_labels.py`. On a CPU-only install the base binding is a
by-name stub, so the first Lasso fit refused at `_mojolearn.transpose_f32`
(the Fortran-order move `cdFit` requires) before the solver host binding
was ever called. This module carries those helpers under the base binding's
names, routed by `_backend._HOST_MODULES` (`"_mojolearn":
"_mojolearn_core_host"`), so `_native` finds them through the same
`_backend.binding("_mojolearn")` call it makes on a GPU box. Every one is a
BYTE MOVE (a transpose, a widening or narrowing cast, a predicate, a
gather, an argmax); none carries a fold, so none has a sabotage arm, and
`core_host_sabotage()` reports the define truthfully so a sabotage set
loads as one set.

THE k-NN INFERENCE ENTRIES (the knn host inference lane, 2026-09-14): `knn_search`, `knn_classify` and `knn_regress` are exported
under the GPU binding's names with the GPU binding's address contracts
(`params` and `dist_params` lists, repeated in each docstring below), so
`python/mojolearn/neighbors.py`'s NearestNeighbors.kneighbors,
KNeighborsClassifier.predict / predict_proba and
KNeighborsRegressor.predict run unchanged on a CPU-only install, and the
host subclasses of `python/mojolearn/_classical_host.py` run them beside
the GPU path on a box that has one. Their arithmetic is
`core/knn_host_predict.mojo`, the statement-for-statement restatement of
the pinned distance tile, the halving-tree row norm, the composite-key
selection, the estimator's sort, the vote and the mean; that file's header
names every original by file and line. THESE THREE HAVE A FOLD, so the
sabotage define reaches them (every distance chain walked descending) and
`core_host_sabotage()` reports it. The host computes the L2 expanded pair
(euclidean/l2, sqeuclidean) and, since lane/cpu-training-batch3
(2026-09-14), cosine, manhattan, chebyshev and minkowski through
`metric_distance_kernel`'s cores; the two L2 unexpanded values are refused
BY NAME. The returned "query tile" is 1, the batch the host runs.

THE BALL COVER ENTRIES (lane/cpu-training-batch3, 2026-09-14):
`radius_neighbors_count`, `radius_neighbors_fill` and `rbc_knn_search` keep
the GPU binding's names and params lists and answer by an exhaustive scan
(`core/knn_host_predict.mojo`, "THE BALL COVER'S TWO QUERIES"), the cover's
pruning being exact. Their refusals are `neighbors/estimator.mojo`'s, in its
order and words. The base binding's other helpers stay ABSENT and refuse BY
NAME through `_HostBinding`.

THE k-MEANS TRAINING ENTRY (workstream E batch 2, lane/cpu-training-e2,
2026-09-14): `kmeans_fit` is exported under the GPU binding's name with the
GPU binding's address contract (the ten-value `params` list, repeated in
its docstring), so `python/mojolearn/cluster.py::KMeans.fit` runs unchanged
on a CPU-only install. Its arithmetic is `cluster/host/kmeans_oracle.mojo`,
the statement-for-statement restatement of the k-means|| init, the fused
assignment, the fixed-point centroid update and the shift test; that
file's header names every original by file and line. IT HAS A FOLD AND A
QUANTIZATION, so the sabotage define reaches it (one extra unit in every
quantized centroid-sum cell) and `core_host_sabotage()` reports it.

Workstream E (lane/cpu-training-e, 2026-09-14) added the three centering
helpers of `linear_model.py`; lane hr-small-passes (2026-10-02) deleted
them: the linear models center through the estimators binding on every
tier (`lm_col_sums`, `lm_center`, `lm_scale_rows`).

`transpose_f32` and `cast_colmajor_f64_to_f32` MIRROR
`bindings/_mojolearn.mojo::_tiled_transpose_to_f32` (DEVIATIONS 2471,
2472) element for element, `dst[c * rows + r] = Float32(src[r * cols +
c])`; the tiling there is a cache order, not a value, so a plain loop
writes the same bytes. The seven others are `bindings/host_helpers.mojo`,
the forest host lane's, shared.

The spectral-precomputed lane (2026-09-14) adds `nonzero_f64_count` and
`nonzero_f64_fill`, the base binding's dense-to-COO scan (DEVIATION 2489)
that `_spectral_impl._coo_triples` reaches through `_native` on a dense
precomputed affinity; bodies copied from `bindings/_mojolearn.mojo`, a
comparison and one narrowing per kept value, no fold to sabotage.
"""
# lane cpu2-l4-modelsel (2026-10-04): model selection's resident fold store,
# fold-row gather, cross_val_predict scatter, binary proba column and the
# parallel forest's offset merge (bindings/msel_host.mojo).
from bindings.msel_host import (
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
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call as _balanced_call
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from std.sys.compile import is_defined

from bindings.host_helpers import (
    all_finite_f32_binding,
    all_finite_f64_binding,
    argmax_last_rows_f32_binding,
    argmax_rows_f32_binding,
    argmax_rows_f64_binding,
    cast_f64_to_f32_binding,
    gather_f64_binding,
    gather_i64_binding,
    gather_rows_bytes_binding,
)
from bindings.hotpath_helpers import (
    HOTPATH_SABOTAGE,
    cast_elements_binding,
    check_indices_i64_binding,
    encode_labels_f32_binding,
    encode_labels_f64_binding,
    encode_labels_i32_binding,
    encode_labels_i64_binding,
    encode_labels_u32_binding,
    encode_labels_u8_binding,
    equal_elements_binding,
    fold_ids_binding,
    gather_i32_binding,
    indices_overlap_i64_binding,
    reduce_stat_binding,
    select_fold_i64_binding,
    arange_i64_binding,
    leave_range_i64_binding,
    mask_from_indices_u8_binding,
    select_mask_u8_i64_binding,
    count_mask_u8_binding,
    next_combination_i64_binding,
    ic_running_min_f64_host_binding,
    ic_running_min_f32_host_binding,
    fold_pair_f32_binding,
    threshold_labels_i64_binding,
    bincount_i64_binding,
    assign_fold_i64_binding,
    count_fold_hits_i64_binding,
    mark_split_roles_f32_binding,
    scatter_rows_bytes_host_binding,
    uniform_init_f32_binding,
    normal_init_f32_binding,
    first_seen_i32_binding,
    strat_fold_assign_i32_binding,
    strat_alloc_i64_binding,
    group_fold_assign_i32_binding,
    strat_group_plan_i32_binding,
    bincount2_i32_host_binding,
)
from bindings.normal_dist_helpers import truncnorm_draws_binding
from bindings.lr_table_helpers import (
    lr_schedule_exact_binding,
    lr_schedule_exact_block_binding,
    lr_schedule_at_c, lr_schedule_series_c,
)
from bindings.optimizer_public_door import entry_init
from bindings.array_helpers import (
    strided_copy_bytes_binding,
    check_lengths_i64_binding,
    ragged_rows_bytes_binding,
    nsum_f64_binding,
    row_means_f64_binding,
    fsum_f64_binding,
    row_stds_f64_binding,
    class_ratio_f64_host_binding,
    shard_topk_merge_f32_host_binding,
)
from core.hostptr import f32_ptr, f64_ptr, i32_ptr, u32_ptr
from core.dense_coo import (
    nonzero_f32_count as dense_nonzero_f32_count,
    nonzero_f32_fill as dense_nonzero_f32_fill,
)
from core.dense_coo_host import knn_affinity_f32 as dense_knn_affinity_f32
from core.label_encode import host_unique_inverse
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from cluster.host.kmeans_oracle import KMEANS_ORACLE_HOST_SABOTAGE
from core.knn_host_predict import KNN_HOST_SABOTAGE
from core.core_host_impl import (
    kmeans_fit_host,
    kmeans_predict_host,
    kmeans_transform_host,
    knn_classify_host,
    knn_classify_neighbors_host,
    knn_regress_host,
    knn_regress_neighbors_host,
    knn_search_host,
    nonzero_f64_count,
    nonzero_f64_fill,
    radius_neighbors_count_host,
    radius_neighbors_fill_host,
    rbc_knn_search_host,
    transpose_to_f32,
)


#: Reported by `core_host_sabotage()`. The helpers move bytes and fold
#: nothing, so the define changes none of their answers; the three k-NN
#: entries DO fold (`KNN_HOST_SABOTAGE` in core/knn_host_predict.mojo, the
#: same define) and a sabotage build walks every distance chain descending.
comptime CORE_HOST_SABOTAGE_DEFINE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("core host: integers expected, not booleans")
    var operator_module = native_loaded_module("operator")
    # operator.index through the balanced call: PythonObject.__call__ keeps one
    # reference on each argument (bindings/py_balanced.mojo), which leaked every
    # buffer address passed to a host entry.
    return Int(py=_balanced_call(operator_module.index, value))


def core_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def core_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def core_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_core_host.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "core host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_core_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `core_host_detected_column` read-back, for the reason 8d16ce2f
# removed it from the forest and byte LM host bindings: the detected column
# folds to the GPU of the machine that ran the build, so its name would land
# in the vendor-neutral binary. The comptime assert above is the check.


def core_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary was built with -D MOJOLEARN_HOST_SABOTAGE=1: the
    helpers are untouched by it (byte moves have no fold), the three k-NN
    entries walk every distance chain descending under it and the k-means
    fit adds one unit to every quantized centroid-sum cell (the gate's
    negative control); refused outside the gate as one set."""
    # lane/python-hotpath: -D MOJOLEARN_HOTPATH_SABOTAGE=1 sabotages ONLY the
    # helpers of bindings/hotpath_helpers.mojo (so a divergence under it is
    # theirs and not the k-NN fold's), and is refused outside the gate the
    # same way, by being reported here.
    return PythonObject(
        CORE_HOST_SABOTAGE_DEFINE
        or KNN_HOST_SABOTAGE
        or KMEANS_ORACLE_HOST_SABOTAGE
        or HOTPATH_SABOTAGE
    )


# The base binding's names, same contract.


def mojolearn_vendor_binding() raises -> PythonObject:
    """"cpu". On a CPU-only install `_backend.vendor()` is "cpu" and the
    read-back cross-check expects that string from every host binding."""
    return PythonObject(String("cpu"))


def mojolearn_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def transpose_f32_binding(
    src_addr: PythonObject,
    dst_addr: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
) raises -> PythonObject:
    """Read a C-contiguous float32 `[rows, cols]` matrix at `src`, write its
    COLUMN-MAJOR layout at `dst`, `dst[c * rows + r] == src[r * cols + c]`
    (DEVIATION 2472). Returns 0. A pure move, no arithmetic. An empty
    matrix writes nothing and reads neither address; a negative dimension
    is refused rather than read. `src` and `dst` must not overlap."""
    var nr = _index(rows)
    var nc = _index(cols)
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
    var sp = f32_ptr(_index(src_addr))
    var dp = f32_ptr(_index(dst_addr))
    with GILReleased(Python()):
        transpose_to_f32(sp, dp, nr, nc)
    return PythonObject(0)


def cast_colmajor_f64_to_f32_binding(
    src_addr: PythonObject,
    dst_addr: PythonObject,
    rows: PythonObject,
    cols: PythonObject,
) raises -> PythonObject:
    """Read a C-contiguous float64 `[rows, cols]` matrix at `src`, write the
    COLUMN-MAJOR float32 matrix at `dst`, `dst[c * rows + r] ==
    Float32(src[r * cols + c])` (DEVIATION 2471). Returns 0. Every element
    is read once, narrowed once and written once to its transposed
    position. An empty matrix writes nothing; a negative dimension is
    refused rather than read. `src` and `dst` must not overlap."""
    var nr = _index(rows)
    var nc = _index(cols)
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
    var sp = f64_ptr(_index(src_addr))
    var dp = f32_ptr(_index(dst_addr))
    with GILReleased(Python()):
        transpose_to_f32(sp, dp, nr, nc)
    return PythonObject(0)


# ===========================================================================
# THE DENSE AFFINITY'S COO SCAN (the spectral-precomputed lane, 2026-09-14).
# `python/mojolearn/_spectral_impl.py::_coo_triples` reaches these two
# through `_buffer._native` on a dense precomputed affinity before the
# metrics host binding's `spectral_fit_predict_graph` is called; without
# them the lane refuses at `_mojolearn.nonzero_f64_count`. Both bodies are
# `bindings/_mojolearn.mojo::nonzero_f64_count_binding` and
# `nonzero_f64_fill_binding` (DEVIATION 2489) with this binding's `_index`
# and `core/hostptr.mojo`'s pointers: a comparison, a row-major scan
# and one narrowing per kept value, no fold, so no sabotage arm here (the
# lane's arms are the spectral oracle's).
# ===========================================================================


def nonzero_f64_count_binding(
    src_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """How many of the `n` float64 values at `src` are nonzero under the
    test `v != 0.0` (DEVIATION 2489): -0.0 is a zero and NaN is NOT. An
    empty input reads nothing; a negative count is refused."""
    var count = _index(n)
    if count < 0:
        raise Error(
            "nonzero_f64_count: n must be non-negative, got " + String(count)
        )
    if count == 0:
        return PythonObject(0)
    var sp = f64_ptr(_index(src_addr))
    var nz = 0
    with GILReleased(Python()):
        nz = nonzero_f64_count(sp, count)
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
    order, its row to `outs[0]` (int32), its column to `outs[1]` (int32)
    and `Float32(v)` to `outs[2]` (float32). Returns the number written.
    The fill STOPS and raises if the matrix holds more than `capacity`
    nonzeros, so a stale count cannot write past the buffers."""
    var nr = _index(rows)
    var nc = _index(cols)
    var cap = _index(capacity)
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
    var sp = f64_ptr(_index(src_addr))
    var rp = i32_ptr(_index(outs[0]))
    var cp = i32_ptr(_index(outs[1]))
    var vp = f32_ptr(_index(outs[2]))
    var k = 0
    with GILReleased(Python()):
        k = nonzero_f64_fill(sp, nr, nc, rp, cp, vp, cap)
    return PythonObject(k)


# ===========================================================================
# THE k-NN INFERENCE ENTRIES (2026-09-14). Each keeps the GPU binding's
# name, arity, `params` list and `dist_params` triple
# (`bindings/_mojolearn.mojo:143-370`), so `neighbors.py` needs no edit.
# ===========================================================================


def _dist_triple(dist_params: PythonObject) raises -> Tuple[Int, Float32, Int]:
    """`dist_params` -> `(metric, metric_arg, weights)`, length-checked, the
    GPU binding's `_dist_triple` verbatim."""
    if len(dist_params) != 3:
        raise Error(
            "knn: dist_params must hold 3 values (metric, metric_arg,"
            " weights), got " + String(len(dist_params))
        )
    return (
        _index(dist_params[0]),
        Float32(Float64(py=dist_params[1])),
        _index(dist_params[2]),
    )


#: The "query tile that ran": the host runs one query row per pass.
comptime KNN_HOST_QUERY_TILE = 1


def knn_search_binding(
    index_addr: PythonObject,
    queries_addr: PythonObject,
    out_dist_addr: PythonObject,
    out_idx_addr: PythonObject,
    params: PythonObject,
    dist_params: PythonObject,
) raises -> PythonObject:
    """Exact k-NN on the host (`NearestNeighbors.kneighbors`).

    Returns the query tile that ran, 1 here. `params`, in the GPU
    binding's order:

        0  n_index
        1  n_queries
        2  n_features
        3  k
        4  return_sqrt   (0 or 1; consulted only for the sentinel metric)
        5  query_tile    (unread: the host has no tile)

    `dist_params` is `(metric, metric_arg, weights)`; `weights` is unread
    by a search and `metric_arg` is Minkowski's p, which the L2 pair
    discards. Writes `n_queries x k` float32 distances and uint32 indices,
    ascending by (distance, index) per row."""
    if len(params) != 6:
        raise Error(
            "knn_search: params must hold 6 values, got "
            + String(len(params))
        )
    var index_address = _index(index_addr)
    var queries_address = _index(queries_addr)
    var dp = f32_ptr(_index(out_dist_addr))
    var xp = u32_ptr(_index(out_idx_addr))
    var ni = _index(params[0])
    var nq = _index(params[1])
    var nf = _index(params[2])
    var kk = _index(params[3])
    var sq = _index(params[4]) != 0
    var dt = _dist_triple(dist_params)
    with GILReleased(Python()):
        knn_search_host(
            index_address, queries_address, dp, xp, ni, nq, nf, kk, sq,
            dt[0], dt[1],
        )
    return PythonObject(KNN_HOST_QUERY_TILE)


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
    """The k-NN classifier on the host, a search then a vote or a tally.

    `knn_classifier_predict` (`neighbors/estimator.mojo:743-953`). Returns
    the query tile that ran, 1 here. `params`, in the GPU binding's order:

        0  n_index
        1  n_queries
        2  n_features
        3  k
        4  query_tile     (unread)
        5  n_outputs
        6  want_proba     (0: write out_labels; 1: write out_proba)
        7.. n_classes per output, `n_outputs` of them

    `y_addr` is `n_outputs` CONTIGUOUS int32 columns of `n_index`;
    `out_labels_addr` is `n_queries x n_outputs` int32 row-major;
    `out_proba_addr` is the per-output `n_queries x n_classes[i]` float32
    blocks concatenated; `out_uniq_addr` is `sum(n_classes)` int32 and is
    always written. A unique-label count that differs from `n_classes[i]`
    is refused before anything is written (policy 7). POLICY 8 as on the
    GPU: the weighted arm asks the search for the rooted distance, which is
    inert for an explicit metric."""
    if len(params) < 7:
        raise Error(
            "knn_classify: params must hold at least 7 values, got "
            + String(len(params))
        )
    var ni = _index(params[0])
    var nq = _index(params[1])
    var nf = _index(params[2])
    var kk = _index(params[3])
    var no = _index(params[5])
    var want_proba = _index(params[6]) != 0
    if len(params) != 7 + no:
        raise Error(
            "knn_classify: params must hold 7 + n_outputs ("
            + String(7 + no)
            + ") values, got "
            + String(len(params))
        )
    var n_classes = List[Int]()
    for i in range(no):  # small-loop(n_classes argument words, one per output)
        n_classes.append(_index(params[7 + i]))
    var index_address = _index(index_addr)
    var queries_address = _index(queries_addr)
    var y_address = _index(y_addr)
    var lp = i32_ptr(_index(out_labels_addr))
    var pp = f32_ptr(_index(out_proba_addr))
    var up = i32_ptr(_index(out_uniq_addr))
    var dt = _dist_triple(dist_params)
    with GILReleased(Python()):
        knn_classify_host(
            index_address, queries_address, y_address, lp, pp, up,
            ni, nq, nf, kk, no, want_proba, n_classes, dt[0], dt[1], dt[2],
        )
    return PythonObject(KNN_HOST_QUERY_TILE)


def knn_regress_binding(
    index_addr: PythonObject,
    queries_addr: PythonObject,
    y_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    dist_params: PythonObject,
) raises -> PythonObject:
    """The k-NN regressor on the host, a search then the neighbours' mean.

    Uniform, or distance-weighted (`knn_regressor_predict`,
    `neighbors/estimator.mojo:971-1094`). Returns the query tile that ran,
    1 here. `params`, in the GPU binding's order:

        0  n_index
        1  n_queries
        2  n_features
        3  k
        4  query_tile     (unread)
        5  n_outputs

    `y_addr` is `n_outputs` CONTIGUOUS float32 columns of `n_index`;
    `out_addr` is `n_queries x n_outputs` float32 row-major."""
    if len(params) != 6:
        raise Error(
            "knn_regress: params must hold 6 values, got "
            + String(len(params))
        )
    var index_address = _index(index_addr)
    var queries_address = _index(queries_addr)
    var y_address = _index(y_addr)
    var op = f32_ptr(_index(out_addr))
    var ni = _index(params[0])
    var nq = _index(params[1])
    var nf = _index(params[2])
    var kk = _index(params[3])
    var no = _index(params[5])
    var dt = _dist_triple(dist_params)
    with GILReleased(Python()):
        knn_regress_host(
            index_address, queries_address, y_address, op,
            ni, nq, nf, kk, no, dt[0], dt[1], dt[2],
        )
    return PythonObject(KNN_HOST_QUERY_TILE)


# ===========================================================================
# THE MERGED-NEIGHBOR VOTES (lane/cpu-training-par-wave2, 2026-09-15). The
# GPU binding's `knn_classify_neighbors` / `knn_regress_neighbors`
# (`bindings/_mojolearn.mojo`, over `neighbors/estimator.mojo::
# knn_classifier_from_neighbors` / `knn_regressor_from_neighbors`), which
# `parallel_neighbors_reference._vote` calls on the neighbors the reference
# shards merged in Python. The vote half of `knn_classify_binding` and
# `knn_regress_binding` above, statement for statement, over the given
# distances and global indices instead of a host search.
# ===========================================================================


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
    """Vote on globally merged neighbors, the GPU binding's parameter layout
    (`knn_classify_binding`'s; the feature count and query tile slots are
    unread). float32 distances and uint32 global indices, both
    n_queries x k. Returns zero on success."""
    if len(params) < 7:
        raise Error(
            "knn_classify: params must hold at least 7 values, got "
            + String(len(params))
        )
    var ni = _index(params[0])
    var nq = _index(params[1])
    var kk = _index(params[3])
    var no = _index(params[5])
    var want_proba = _index(params[6]) != 0
    if len(params) != 7 + no:
        raise Error(
            "knn_classify: params must hold 7 + n_outputs ("
            + String(7 + no)
            + ") values, got "
            + String(len(params))
        )
    var n_classes = List[Int]()
    for i in range(no):  # small-loop(n_classes argument words, one per output)
        n_classes.append(_index(params[7 + i]))
    var dist_address = _index(dist_addr)
    var idx_address = _index(idx_addr)
    var y_address = _index(y_addr)
    var lp = i32_ptr(_index(out_labels_addr))
    var pp = f32_ptr(_index(out_proba_addr))
    var up = i32_ptr(_index(out_uniq_addr))
    var dt = _dist_triple(dist_params)
    with GILReleased(Python()):
        knn_classify_neighbors_host(
            dist_address, idx_address, y_address, lp, pp, up,
            ni, nq, kk, no, want_proba, n_classes, dt[2],
        )
    return PythonObject(0)


def knn_regress_neighbors_binding(
    dist_addr: PythonObject,
    idx_addr: PythonObject,
    y_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
    dist_params: PythonObject,
) raises -> PythonObject:
    """Regress on globally merged neighbors, `knn_regress_binding`'s
    parameter layout (the feature count and query tile slots are unread).
    Returns zero on success."""
    if len(params) != 6:
        raise Error(
            "knn_regress: params must hold 6 values, got "
            + String(len(params))
        )
    var dist_address = _index(dist_addr)
    var idx_address = _index(idx_addr)
    var y_address = _index(y_addr)
    var op = f32_ptr(_index(out_addr))
    var ni = _index(params[0])
    var nq = _index(params[1])
    var kk = _index(params[3])
    var no = _index(params[5])
    var dt = _dist_triple(dist_params)
    with GILReleased(Python()):
        knn_regress_neighbors_host(
            dist_address, idx_address, y_address, op, ni, nq, kk, no, dt[2],
        )
    return PythonObject(0)


# ===========================================================================
# THE BALL COVER ENTRIES (lane/cpu-training-batch3, 2026-09-14). The GPU
# binding's names and params lists (`bindings/_mojolearn.mojo:467-627`).
# ===========================================================================


def radius_neighbors_count_binding(
    index_addr: PythonObject,
    queries_addr: PythonObject,
    out_indptr_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Pass one of the radius query on the host. Writes `n_queries + 1`
    int32 row starts and returns the edge count.

    `params`: n_index, n_queries, n_features, radius (float), metric,
    metric_arg, the GPU binding's order."""
    if len(params) != 6:
        raise Error(
            "radius_neighbors_count: params must hold 6 values, got "
            + String(len(params))
        )
    var index_address = _index(index_addr)
    var queries_address = _index(queries_addr)
    var ap = i32_ptr(_index(out_indptr_addr))
    var ni = _index(params[0])
    var nq = _index(params[1])
    var nf = _index(params[2])
    var rad = Float32(Float64(py=params[3]))
    var mtr = _index(params[4])
    var marg = Float32(Float64(py=params[5]))
    var nnz = 0
    with GILReleased(Python()):
        nnz = radius_neighbors_count_host(
            index_address, queries_address, ap, ni, nq, nf, rad, mtr, marg,
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
    """Pass two on the host: the columns ascending within each row and the
    distances, in CSR order. Returns the edge count it found.

    `params`: n_index, n_queries, n_features, radius (float), nnz_capacity,
    return_sqrt, metric, metric_arg, the GPU binding's order."""
    if len(params) != 8:
        raise Error(
            "radius_neighbors_fill: params must hold 8 values, got "
            + String(len(params))
        )
    var index_address = _index(index_addr)
    var queries_address = _index(queries_addr)
    var ap = i32_ptr(_index(out_indptr_addr))
    var ni = _index(params[0])
    var nq = _index(params[1])
    var nf = _index(params[2])
    var rad = Float32(Float64(py=params[3]))
    var cap = _index(params[4])
    var sq = _index(params[5]) != 0
    var mtr = _index(params[6])
    var marg = Float32(Float64(py=params[7]))
    var idx_address = 0
    var dist_address = 0
    if cap > 0:
        idx_address = _index(out_idx_addr)
        dist_address = _index(out_dist_addr)
    var nnz = 0
    with GILReleased(Python()):
        nnz = radius_neighbors_fill_host(
            index_address, queries_address, ap, idx_address, dist_address,
            ni, nq, nf, rad, cap, sq, mtr, marg,
        )
    return PythonObject(nnz)


def rbc_knn_search_binding(
    index_addr: PythonObject,
    queries_addr: PythonObject,
    out_idx_addr: PythonObject,
    out_dist_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """EXACT k-NN on the host under the ball cover's metric table and total
    order. Writes `n_queries * k` int32 indices and float32 TRUE distances
    and returns the candidate distances computed, `n_queries * n_index`.

    `params`: n_index, n_queries, n_features, k, metric, metric_arg, the
    GPU binding's order."""
    if len(params) != 6:
        raise Error(
            "rbc_knn_search: params must hold 6 values, got "
            + String(len(params))
        )
    var index_address = _index(index_addr)
    var queries_address = _index(queries_addr)
    var xp = i32_ptr(_index(out_idx_addr))
    var dp = f32_ptr(_index(out_dist_addr))
    var ni = _index(params[0])
    var nq = _index(params[1])
    var nf = _index(params[2])
    var kk = _index(params[3])
    var mtr = _index(params[4])
    var marg = Float32(Float64(py=params[5]))
    with GILReleased(Python()):
        rbc_knn_search_host(
            index_address, queries_address, xp, dp, ni, nq, nf, kk, mtr, marg,
        )
    return PythonObject(nq * ni)


# ===========================================================================
# THE k-MEANS TRAINING ENTRY (workstream E batch 2, 2026-09-14). The GPU
# binding's name, arity and `params` list (`bindings/_mojolearn.mojo:365`).
# ===========================================================================


def kmeans_fit_binding(
    x_addr: PythonObject,
    out_centroids_addr: PythonObject,
    out_labels_addr: PythonObject,
    weights_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Fit k-means on the host. Returns [inertia, n_iter, sum_scale,
    weight_scale], the GPU binding's four.

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

    `out_centroids_addr` is `n_clusters x n_features` float32 and is
    written; on `init == INIT_ARRAY` it is read first as the start.
    `out_labels_addr` is `n_samples` uint32 (the caller's int32 array is
    the same bytes), the assignment against the FINAL centroids. The shape
    refusals are `kmeans_fit`'s, raised BEFORE any address is read."""
    # 10 or 11, as the GPU binding: workstream D (2026-09-14) appended
    # oversampling_factor as params[10] (cluster.py packs it on every fit),
    # and the 10-only check read REFUSED for every kmeans cell on the CPU gate.
    if len(params) != 10 and len(params) != 11:
        raise Error(
            "kmeans_fit: params must hold 10 or 11 values, got "
            + String(len(params))
        )
    var x_address = _index(x_addr)
    var centroids_address = _index(out_centroids_addr)
    var lp = u32_ptr(_index(out_labels_addr))
    var weights_address = _index(weights_addr)
    var ns = _index(params[0])
    var nf = _index(params[1])
    var nc = _index(params[2])
    var nw = _index(params[3])
    var mi = _index(params[4])
    var tl = Float64(py=params[5])
    var sd = UInt64(_index(params[6]))
    var ninit = _index(params[7])
    var ii = _index(params[8])
    var mm = _index(params[9])
    var ovs = Float64(2.0)
    if len(params) == 11:
        ovs = Float64(py=params[10])
    var inertia = Float64(0.0)
    var n_iter = 0
    var sum_scale = Float64(0.0)
    var weight_scale = Float64(0.0)
    with GILReleased(Python()):
        var r = kmeans_fit_host(
            x_address, centroids_address, lp, weights_address,
            ns, nf, nc, nw, mi, tl, sd, ninit, ii, mm, ovs,
        )
        inertia = r[0]
        n_iter = r[1]
        sum_scale = r[2]
        weight_scale = r[3]
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
    """The nearest centroid of every row on the host, the GPU binding's name,
    arity and `params` (0 n_samples, 1 n_features, 2 n_clusters, 3 metric).
    Public INFERENCE: it
    reads a fitted model's centroids and trains nothing. The arithmetic is
    `host_kmeans_fit`'s final assignment (`host_kmeans_predict`). The shape
    and metric refusals are raised BEFORE any address is read."""
    if len(params) != 4:
        raise Error(
            "kmeans_predict: params must hold 4 values, got "
            + String(len(params))
        )
    var x_address = _index(x_addr)
    var centroids_address = _index(centroids_addr)
    var lp = u32_ptr(_index(out_labels_addr))
    var ns = _index(params[0])
    var nf = _index(params[1])
    var nc = _index(params[2])
    var mm = _index(params[3])
    with GILReleased(Python()):
        kmeans_predict_host(x_address, centroids_address, lp, ns, nf, nc, mm)
    return PythonObject(ns)


def kmeans_transform_binding(
    x_addr: PythonObject,
    centroids_addr: PythonObject,
    out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The distance from every row to every centroid on the host, the GPU
    binding's name, arity and `params` (0 n_samples, 1 n_features,
    2 n_clusters, 3 metric). Public INFERENCE: it reads a fitted model's
    centroids and trains nothing. The arithmetic is `host_kmeans_transform`,
    the GPU kernel's cell statement for statement. The shape and metric
    refusals are raised BEFORE any address is read."""
    if len(params) != 4:
        raise Error(
            "kmeans_transform: params must hold 4 values, got "
            + String(len(params))
        )
    var x_address = _index(x_addr)
    var centroids_address = _index(centroids_addr)
    var op = f32_ptr(_index(out_addr))
    var ns = _index(params[0])
    var nf = _index(params[1])
    var nc = _index(params[2])
    var mm = _index(params[3])
    with GILReleased(Python()):
        kmeans_transform_host(x_address, centroids_address, op, ns, nf, nc, mm)
    return PythonObject(ns)


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
    var aff = f32_ptr(Int(py=addrs[4]))
    var status = i32_ptr(Int(py=addrs[5]))
    var dense = aff
    var rp = status
    var cp = status
    var vp = aff
    if sparse:
        if nnz > 0:
            rp = i32_ptr(Int(py=addrs[1]))
            cp = i32_ptr(Int(py=addrs[2]))
            vp = f32_ptr(Int(py=addrs[3]))
    else:
        dense = f32_ptr(Int(py=addrs[0]))
    with GILReleased(Python()):
        dense_knn_affinity_f32(dense, rp, cp, vp, nnz, sparse, n, k, aff, status)
    return PythonObject(0)


def unique_inverse_binding(
    src_addr: PythonObject, n: PythonObject, kind: PythonObject,
    classes_addr: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    """The GPU binding's `unique_inverse` (its contract and refusals) over
    the host twin `core/label_encode.mojo::host_unique_inverse`, the same
    statements as `core/label_encode_device.mojo` (cpu-gpu-cleanup
    w2-pyglue). Integers and bit moves: the same bytes."""
    var count = Int(py=n)
    var k_ind = Int(py=kind)
    if count < 1:
        raise Error("unique_inverse: n must be positive, got " + String(count))
    if k_ind != 0 and k_ind != 1:
        raise Error("unique_inverse: kind must be 0 (float64) or 1 (int64)")
    if Int(py=src_addr) == 0 or Int(py=classes_addr) == 0 or Int(py=codes_addr) == 0:
        raise Error("unique_inverse: null buffer address")
    var sp = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(py=src_addr))
    var cp = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(py=classes_addr))
    var dp = i32_ptr(Int(py=codes_addr))
    var k: Int
    with GILReleased(Python()):
        k = host_unique_inverse(sp, count, k_ind, cp, dp)
    if k == -2:
        raise Error("mojolearn: y contains a NaN label; NaN is not a class")
    return PythonObject(k)


# ===========================================================================
# THE SHARED ARGUMENT-AND-LABEL ENTRY (public runtime migration, plan section
# 1 item 1, lane migrate-shared-labels). One native call takes a label
# vector and/or a sample-weight vector and returns the ORDER RULE's classes
# and codes and the weight verdict, so `_labels.encode_labels`,
# `_labels.unique_inverse` and `_sample_weight.check_sample_weight` each make
# this ONE call (they made up to three before, plus the Python glue between
# them). It composes the SAME helpers those paths called, in the same order,
# so no output byte moves: `encode_labels_<dt>` (capped encoder), then
# `cast_elements` + `unique_inverse` past the cap, `all_finite_<f>` and
# `reduce_stat` min/max for the weights. The core host binding carries the
# same body over its host twins under the same name.
#
#   labels: `y_addr` != 0 points at `n` values of dtype code `y_code`
#     (0 f32, 1 f64, 2 i32, 3 i64, 4 u32, 5 u8). The k sorted classes land
#     WIDENED at `classes_addr` (n slots of 64 bits: float64 for float
#     labels, int64 for integer labels), one int32 code per row at
#     `codes_addr`. A NaN label raises the ORDER RULE's message.
#   weights: `w_addr` != 0 points at `n` weights of code `w_code` (0 or 1).
#     `w_flags` bit 1 allows negative weights, bit 2 an all-zero vector.
#
# Returns k (0 without labels), or a negative status for refused weights:
# PT_W_NONFINITE, PT_W_NEGATIVE, PT_W_ALL_ZERO (the Python glue raises
# scikit-learn's message for each). Labels are checked before weights.
# ===========================================================================

comptime PT_ENCODE_CAP = 4096
comptime PT_W_NONFINITE = -3
comptime PT_W_NEGATIVE = -4
comptime PT_W_ALL_ZERO = -5
comptime PT_ALLOW_NEGATIVE = 1
comptime PT_ALLOW_ALL_ZERO = 2


def _pt_encode(
    y_addr: PythonObject, y_code: Int, n: PythonObject, classes_addr: Int,
    cap: Int, codes_addr: PythonObject,
) raises -> Int:
    """The capped encoder `encode_labels_<dt>` for dtype code `y_code`."""
    var ca = PythonObject(classes_addr)
    var cp = PythonObject(cap)
    if y_code == 0:
        return Int(py=encode_labels_f32_binding(y_addr, n, ca, cp, codes_addr))
    if y_code == 1:
        return Int(py=encode_labels_f64_binding(y_addr, n, ca, cp, codes_addr))
    if y_code == 2:
        return Int(py=encode_labels_i32_binding(y_addr, n, ca, cp, codes_addr))
    if y_code == 3:
        return Int(py=encode_labels_i64_binding(y_addr, n, ca, cp, codes_addr))
    if y_code == 4:
        return Int(py=encode_labels_u32_binding(y_addr, n, ca, cp, codes_addr))
    if y_code == 5:
        return Int(py=encode_labels_u8_binding(y_addr, n, ca, cp, codes_addr))
    raise Error("prepare_targets: label dtype code must be 0..5, got " + String(y_code))


def prepare_targets_binding(
    y_addr: PythonObject, y_code: PythonObject, n: PythonObject,
    classes_addr: PythonObject, codes_addr: PythonObject,
    w_addr: PythonObject, w_code: PythonObject, w_flags: PythonObject,
) raises -> PythonObject:
    """See the block comment above."""
    var count = Int(py=n)
    if count < 0:
        raise Error("prepare_targets: n must be non-negative, got " + String(count))
    var k = 0
    if Int(py=y_addr) != 0:
        if count < 1:
            raise Error("prepare_targets: labels need n >= 1")
        if Int(py=classes_addr) == 0 or Int(py=codes_addr) == 0:
            raise Error("prepare_targets: null label output address")
        var yc = Int(py=y_code)
        var wide_kind = 0 if (yc == 0 or yc == 1) else 1
        var wide_code = 1 if wide_kind == 0 else 3
        var cap = count if count < PT_ENCODE_CAP else PT_ENCODE_CAP
        if yc == wide_code:
            # 64-bit labels: the encoder writes the classes in place
            k = _pt_encode(y_addr, yc, n, Int(py=classes_addr), cap, codes_addr)
        else:
            var tmp = List[UInt64](length=cap, fill=0)
            var tmp_addr = Int(tmp.unsafe_ptr())
            k = _pt_encode(y_addr, yc, n, tmp_addr, cap, codes_addr)
            if k > 0:
                # widen the k classes (exact: float32 -> float64, int -> int64)
                _ = cast_elements_binding(
                    PythonObject(tmp_addr), y_code, classes_addr,
                    PythonObject(wide_code), PythonObject(k),
                )
            _ = tmp^
        if k < 0:
            # more classes than the cap, or a NaN label: `unique_inverse` over
            # the widened labels (no cap; it raises the NaN refusal)
            if yc == wide_code:
                k = Int(py=unique_inverse_binding(
                    y_addr, n, PythonObject(wide_kind), classes_addr, codes_addr
                ))
            else:
                var wide = List[UInt64](length=count, fill=0)
                var wide_addr = Int(wide.unsafe_ptr())
                var st = Int(py=cast_elements_binding(
                    y_addr, y_code, PythonObject(wide_addr), PythonObject(wide_code), n
                ))
                if st != 0:
                    raise Error("prepare_targets: labels do not widen to 64 bits")
                k = Int(py=unique_inverse_binding(
                    PythonObject(wide_addr), n, PythonObject(wide_kind), classes_addr, codes_addr
                ))
                _ = wide^
    if Int(py=w_addr) != 0 and count > 0:
        var wc = Int(py=w_code)
        var flags = Int(py=w_flags)
        var fin: Int
        if wc == 1:
            fin = Int(py=all_finite_f64_binding(w_addr, n))
        elif wc == 0:
            fin = Int(py=all_finite_f32_binding(w_addr, n))
        else:
            raise Error("prepare_targets: weight dtype code must be 0 or 1, got " + String(wc))
        if fin == 0:
            return PythonObject(PT_W_NONFINITE)
        var lo = Float64(py=reduce_stat_binding(w_addr, w_code, n, PythonObject(0)))
        if (flags & PT_ALLOW_NEGATIVE) == 0 and lo < 0:
            return PythonObject(PT_W_NEGATIVE)
        if (flags & PT_ALLOW_ALL_ZERO) == 0:
            var hi = Float64(py=reduce_stat_binding(w_addr, w_code, n, PythonObject(1)))
            if not (hi > 0 or lo < 0):
                return PythonObject(PT_W_ALL_ZERO)
    return PythonObject(k)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_core_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_core_host() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_core_host")
        module.def_function[core_host_numeric_mode_binding]("core_host_numeric_mode")
        module.def_function[core_host_vendor_binding]("core_host_vendor")
        module.def_function[core_host_column_binding]("core_host_column")
        module.def_function[core_host_sabotage_binding]("core_host_sabotage")
        module.def_function[mojolearn_vendor_binding]("mojolearn_vendor")
        module.def_function[mojolearn_numeric_mode_binding]("mojolearn_numeric_mode")
        module.def_function[knn_search_binding]("knn_search")
        module.def_function[knn_classify_binding]("knn_classify")
        module.def_function[knn_regress_binding]("knn_regress")
        module.def_function[knn_classify_neighbors_binding]("knn_classify_neighbors")
        module.def_function[knn_regress_neighbors_binding]("knn_regress_neighbors")
        module.def_function[kmeans_fit_binding]("kmeans_fit")
        module.def_function[kmeans_predict_binding]("kmeans_predict")
        module.def_function[kmeans_transform_binding]("kmeans_transform")
        module.def_function[radius_neighbors_count_binding]("radius_neighbors_count")
        module.def_function[radius_neighbors_fill_binding]("radius_neighbors_fill")
        module.def_function[rbc_knn_search_binding]("rbc_knn_search")
        module.def_function[transpose_f32_binding]("transpose_f32")
        module.def_function[cast_colmajor_f64_to_f32_binding]("cast_colmajor_f64_to_f32")
        module.def_function[nonzero_f64_count_binding]("nonzero_f64_count")
        module.def_function[nonzero_f64_fill_binding]("nonzero_f64_fill")
        module.def_function[nonzero_f32_count_binding]("nonzero_f32_count")
        module.def_function[nonzero_f32_fill_binding]("nonzero_f32_fill")
        module.def_function[knn_affinity_f32_binding]("knn_affinity_f32")
        module.def_function[unique_inverse_binding]("unique_inverse")
        module.def_function[cast_f64_to_f32_binding]("cast_f64_to_f32")
        module.def_function[all_finite_f32_binding]("all_finite_f32")
        module.def_function[all_finite_f64_binding]("all_finite_f64")
        module.def_function[gather_i64_binding]("gather_i64")
        module.def_function[gather_f64_binding]("gather_f64")
        module.def_function[gather_rows_bytes_binding]("gather_rows_bytes")
        # lane cpu2-l4-modelsel: model selection on the device (host column on the core host binding)
        module.def_function[msel_put_binding]("msel_put")
        module.def_function[msel_alloc_binding]("msel_alloc")
        module.def_function[msel_read_binding]("msel_read")
        module.def_function[msel_free_binding]("msel_free")
        module.def_function[msel_live_binding]("msel_live")
        module.def_function[msel_take_rows_binding]("msel_take_rows")
        module.def_function[msel_scatter_rows_binding]("msel_scatter_rows")
        module.def_function[msel_proba_column_binding]("msel_proba_column")
        module.def_function[msel_rebase_offsets_i32_binding]("msel_rebase_offsets_i32")
        module.def_function[msel_split_table_i32_binding]("msel_split_table_i32")
        module.def_function[msel_group_fold_perm_i32_binding]("msel_group_fold_perm_i32")
        module.def_function[argmax_rows_f32_binding]("argmax_rows_f32")
        module.def_function[argmax_rows_f64_binding]("argmax_rows_f64")
        module.def_function[argmax_last_rows_f32_binding]("argmax_last_rows_f32")
        # lane/python-hotpath (2026-09-17, DEVIATIONS 3100-3104): the helpers
        # of bindings/hotpath_helpers.mojo, and the ORDER RULE's encoder the
        # base binding has carried since DEVIATION 2500, so a CPU-only
        # install stops encoding labels in a Python loop.
        module.def_function[cast_elements_binding]("cast_elements")
        module.def_function[reduce_stat_binding]("reduce_stat")
        module.def_function[equal_elements_binding]("equal_elements")
        module.def_function[encode_labels_f32_binding]("encode_labels_f32")
        module.def_function[encode_labels_f64_binding]("encode_labels_f64")
        module.def_function[encode_labels_i32_binding]("encode_labels_i32")
        module.def_function[encode_labels_i64_binding]("encode_labels_i64")
        module.def_function[encode_labels_u32_binding]("encode_labels_u32")
        module.def_function[encode_labels_u8_binding]("encode_labels_u8")
        # lane migrate-shared-labels: the shared argument-and-label entry
        module.def_function[prepare_targets_binding]("prepare_targets")
        module.def_function[gather_i32_binding]("gather_i32")
        module.def_function[check_indices_i64_binding]("check_indices_i64")
        module.def_function[indices_overlap_i64_binding]("indices_overlap_i64")
        module.def_function[fold_ids_binding]("fold_ids")
        module.def_function[select_fold_i64_binding]("select_fold_i64")
        module.def_function[arange_i64_binding]("arange_i64")
        module.def_function[leave_range_i64_binding]("leave_range_i64")
        module.def_function[mask_from_indices_u8_binding]("mask_from_indices_u8")
        module.def_function[select_mask_u8_i64_binding]("select_mask_u8_i64")
        module.def_function[count_mask_u8_binding]("count_mask_u8")
        module.def_function[next_combination_i64_binding]("next_combination_i64")
        module.def_function[ic_running_min_f64_host_binding]("ic_running_min_f64")
        module.def_function[ic_running_min_f32_host_binding]("ic_running_min_f32")
        module.def_function[fold_pair_f32_binding]("fold_pair_f32")
        module.def_function[threshold_labels_i64_binding]("threshold_labels_i64")
        module.def_function[bincount_i64_binding]("bincount_i64")
        module.def_function[assign_fold_i64_binding]("assign_fold_i64")
        module.def_function[count_fold_hits_i64_binding]("count_fold_hits_i64")
        module.def_function[mark_split_roles_f32_binding]("mark_split_roles_f32")
        module.def_function[scatter_rows_bytes_host_binding]("scatter_rows_bytes")
        module.def_function[uniform_init_f32_binding]("uniform_init_f32")
        module.def_function[normal_init_f32_binding]("normal_init_f32")
        # TOMBSTONE: adam_hyper_f64 (dead native entry: no live caller) deleted 2026-10-26 by @@SHA@@; code recoverable at 485ba9cc1.
        # Restore: git apply experiments/removed/adam_hyper_f64.patch
        module.def_function[first_seen_i32_binding]("first_seen_i32")
        module.def_function[strat_fold_assign_i32_binding]("strat_fold_assign_i32")
        module.def_function[strat_alloc_i64_binding]("strat_alloc_i64")
        module.def_function[group_fold_assign_i32_binding]("group_fold_assign_i32")
        # StratifiedGroupKFold's group x class table and group plan (the GPU
        # base binding's bincount2_i32 / strat_group_plan_i32): a CPU-only
        # install refused StratifiedGroupKFold.split without them (lane
        # run-sabotage-arms, 2026-10-21: x-metrics-splitters on the host column)
        module.def_function[bincount2_i32_host_binding]("bincount2_i32")
        module.def_function[strat_group_plan_i32_binding]("strat_group_plan_i32")
        module.def_function[strided_copy_bytes_binding]("strided_copy_bytes")
        module.def_function[check_lengths_i64_binding]("check_lengths_i64")
        module.def_function[ragged_rows_bytes_binding]("ragged_rows_bytes")
        module.def_function[nsum_f64_binding]("nsum_f64")
        module.def_function[row_means_f64_binding]("row_means_f64")
        module.def_function[truncnorm_draws_binding]("truncnorm_draws")
        module.def_function[fsum_f64_binding]("fsum_f64")
        module.def_function[row_stds_f64_binding]("row_stds_f64")
        module.def_function[lr_schedule_exact_binding]("lr_schedule_exact")
        module.def_function[lr_schedule_exact_block_binding]("lr_schedule_exact_block")
        # TOMBSTONE: lr_schedule_fill (dead native entry: no live caller) deleted 2026-10-26 by @@SHA@@; code recoverable at 485ba9cc1.
        # Restore: git apply experiments/removed/lr_schedule_fill.patch
        module.def_py_c_function(lr_schedule_at_c, "lr_schedule_at")
        module.def_py_c_function(lr_schedule_series_c, "lr_schedule_series")
        module.def_py_c_function(entry_init, "optimizer_door_init")
        module.def_function[class_ratio_f64_host_binding]("class_ratio_f64")
        module.def_function[shard_topk_merge_f32_host_binding]("shard_topk_merge_f32")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_core_host: ", error))
