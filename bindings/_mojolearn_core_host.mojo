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

Workstream E (lane/cpu-training-e, 2026-09-14) adds the three centering
helpers of `linear_model.py` (`column_mean_f64`, `center_columns_f32`,
`scale_rows_f32`, in `bindings/host_helpers.mojo`), which the ols and ridge
host fits reach through `_buffer._native` before the estimators host
binding is called; the seven-runner gate refused both lanes at
`_mojolearn.column_mean_f64` until they were here. Sequential float64
chains and per-cell operations, no fold to sabotage.

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
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from std.sys.compile import is_defined

from bindings.host_helpers import (
    all_finite_f32_binding,
    all_finite_f64_binding,
    argmax_rows_f32_binding,
    argmax_rows_f64_binding,
    cast_f64_to_f32_binding,
    center_columns_f32_binding,
    column_mean_f64_binding,
    gather_f64_binding,
    gather_i64_binding,
    gather_rows_bytes_binding,
    probability_rows_f32_binding,
    scale_rows_f32_binding,
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
)
from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32, read_i32, u32_ptr
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from neighbors.impl.ball_cover.common import rbc_validate_metric
from neighbors.impl.ball_cover.knn import RBC_KNN_MAX_K
from cluster.host.kmeans_oracle import (
    INIT_ARRAY,
    KMEANS_ORACLE_HOST_SABOTAGE,
    host_kmeans_fit,
    host_kmeans_predict,
    host_kmeans_transform,
    host_kmeans_validate,
)
from core.knn_host_predict import (
    KNN_HOST_SABOTAGE,
    host_rbc_radius_counts,
    host_rbc_radius_fill_rows,
    host_rbc_knn_search,
    KNN_HOST_WEIGHTS_DISTANCE,
    KNN_HOST_WEIGHTS_UNIFORM,
    host_class_probs,
    host_class_vote,
    host_distance_weights,
    host_knn_search,
    host_monotonic,
    host_regress_avg,
    host_unique_labels,
    host_weighted_class_probs,
    host_weighted_regress_avg,
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
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


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


def _transpose_to_f32[
    S: DType
](
    sp: MutPointer[Scalar[S], MutUntrackedOrigin],
    dp: MutPointer[Float32, MutUntrackedOrigin],
    nr: Int,
    nc: Int,
):
    """`dp[c * nr + r] = Float32(sp[r * nc + c])`, the value
    `_tiled_transpose_to_f32` writes, in plain column order."""
    for c in range(nc):
        var dbase = c * nr
        for r in range(nr):
            dp.unsafe_store(dbase + r, sp.unsafe_load(r * nc + c).cast[DType.float32]())


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
        _transpose_to_f32(sp, dp, nr, nc)
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
        _transpose_to_f32(sp, dp, nr, nc)
    return PythonObject(0)


# ===========================================================================
# THE DENSE AFFINITY'S COO SCAN (the spectral-precomputed lane, 2026-09-14).
# `python/mojolearn/_spectral_impl.py::_coo_triples` reaches these two
# through `_buffer._native` on a dense precomputed affinity before the
# metrics host binding's `spectral_fit_predict_graph` is called; without
# them the lane refuses at `_mojolearn.nonzero_f64_count`. Both bodies are
# `bindings/_mojolearn.mojo::nonzero_f64_count_binding` and
# `nonzero_f64_fill_binding` (DEVIATION 2489) with this binding's `_index`
# and `bindings/hostptr.mojo`'s pointers: a comparison, a row-major scan
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
        if ni < 0 or nq < 0 or nf < 0 or kk < 0:
            raise Error("knn_search: a negative dimension was passed")
        var index = read_f32(index_address, ni * nf)
        var queries = read_f32(queries_address, nq * nf)
        var out_dist = List[Float32](length=max(0, nq * kk), fill=Float32(0.0))
        var out_idx = List[UInt32](length=max(0, nq * kk), fill=UInt32(0))
        host_knn_search(
            index, ni, queries, nq, nf, kk, dt[0], sq, out_dist, out_idx,
            dt[1],
        )
        for i in range(nq * kk):
            dp[i] = out_dist[i]
            xp[i] = out_idx[i]
    return PythonObject(KNN_HOST_QUERY_TILE)


def _knn_host_weighted(weights: Int, who: String) raises -> Bool:
    """`knn_classifier_predict` / `knn_regressor_predict`'s weights check,
    same message."""
    var weighted = weights == KNN_HOST_WEIGHTS_DISTANCE
    if weights != KNN_HOST_WEIGHTS_UNIFORM and not weighted:
        raise Error(
            who
            + ": weights value "
            + String(weights)
            + " is neither WEIGHTS_UNIFORM nor WEIGHTS_DISTANCE"
        )
    return weighted


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
    for i in range(no):
        n_classes.append(_index(params[7 + i]))
    var index_address = _index(index_addr)
    var queries_address = _index(queries_addr)
    var y_address = _index(y_addr)
    var lp = i32_ptr(_index(out_labels_addr))
    var pp = f32_ptr(_index(out_proba_addr))
    var up = i32_ptr(_index(out_uniq_addr))
    var dt = _dist_triple(dist_params)
    with GILReleased(Python()):
        if no < 1:
            raise Error(
                "knn_classifier_predict: n_outputs must be positive, got "
                + String(no)
            )
        var weighted = _knn_host_weighted(dt[2], "knn_classifier_predict")
        if ni < 0 or nq < 0 or nf < 0 or kk < 0:
            raise Error("knn_classify: a negative dimension was passed")
        var index = read_f32(index_address, ni * nf)
        var queries = read_f32(queries_address, nq * nf)
        var dist = List[Float32](length=max(0, nq * kk), fill=Float32(0.0))
        var idx = List[UInt32](length=max(0, nq * kk), fill=UInt32(0))
        # policy 8: the weighted arm needs the ROOTED distance
        host_knn_search(index, ni, queries, nq, nf, kk, dt[0], weighted, dist, idx, dt[1])
        var w = List[Float32]()
        if weighted:
            w = host_distance_weights(dist, nq, kk)
        # The unique sets first, checked against the caller's counts
        # before a byte is written (`_check_class_counts`).
        var ys = List[List[Int32]]()
        var uniqs = List[List[Int32]]()
        for i in range(no):
            ys.append(read_i32(y_address + i * ni * 4, ni))
            uniqs.append(host_unique_labels(ys[i], ni))
            if len(uniqs[i]) != n_classes[i]:
                raise Error(
                    "knn_classifier_predict: the implemented getUniquelabels found "
                    + String(len(uniqs[i]))
                    + " classes for output "
                    + String(i)
                    + " and the caller sized its buffers for "
                    + String(n_classes[i])
                    + "; one of the two class sets is wrong and nothing is "
                    + "written"
                )
        var labels_out = List[Int32](length=max(0, nq * no), fill=Int32(0))
        var off = 0
        for i in range(no):
            var n_uniq = len(uniqs[i])
            var mono = host_monotonic(ys[i], ni, uniqs[i])
            var proba: List[Float32]
            if weighted:
                proba = host_weighted_class_probs(idx, mono, w, n_uniq, nq, kk)
            else:
                proba = host_class_probs(idx, mono, n_uniq, nq, kk)
            if want_proba:
                for j in range(nq * n_uniq):
                    pp[off + j] = proba[j]
                off += nq * n_uniq
            else:
                host_class_vote(proba, uniqs[i], n_uniq, nq, labels_out, no, i)
        if not want_proba:
            for j in range(nq * no):
                lp[j] = labels_out[j]
        var uoff = 0
        for i in range(no):
            for j in range(len(uniqs[i])):
                up[uoff + j] = uniqs[i][j]
            uoff += len(uniqs[i])
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
        if no < 1:
            raise Error(
                "knn_regressor_predict: n_outputs must be positive, got "
                + String(no)
            )
        var weighted = _knn_host_weighted(dt[2], "knn_regressor_predict")
        if ni < 0 or nq < 0 or nf < 0 or kk < 0:
            raise Error("knn_regress: a negative dimension was passed")
        var index = read_f32(index_address, ni * nf)
        var queries = read_f32(queries_address, nq * nf)
        var dist = List[Float32](length=max(0, nq * kk), fill=Float32(0.0))
        var idx = List[UInt32](length=max(0, nq * kk), fill=UInt32(0))
        host_knn_search(index, ni, queries, nq, nf, kk, dt[0], weighted, dist, idx, dt[1])
        var w = List[Float32]()
        if weighted:
            w = host_distance_weights(dist, nq, kk)
        var out = List[Float32](length=max(0, nq * no), fill=Float32(0.0))
        for i in range(no):
            var y = read_f32(y_address + i * ni * 4, ni)
            if weighted:
                host_weighted_regress_avg(idx, y, w, nq, kk, out, no, i)
            else:
                host_regress_avg(idx, y, nq, kk, out, no, i)
        for j in range(nq * no):
            op[j] = out[j]
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


def _host_validate_neighbors(
    idx: List[UInt32], n_index: Int, n_queries: Int, k: Int,
    n_outputs: Int, weights: Int,
) raises:
    """`neighbors/estimator.mojo::_validate_neighbors`, its words."""
    if n_index < 1 or n_queries < 1 or k < 1 or k > n_index or n_outputs < 1:
        raise Error("precomputed neighbors: invalid shape")
    if weights != KNN_HOST_WEIGHTS_UNIFORM and weights != KNN_HOST_WEIGHTS_DISTANCE:
        raise Error("precomputed neighbors: unsupported weights")
    for i in range(n_queries * k):
        if UInt64(idx[i]) >= UInt64(n_index):
            raise Error("precomputed neighbor index outside reference data")


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
    for i in range(no):
        n_classes.append(_index(params[7 + i]))
    var dist_address = _index(dist_addr)
    var idx_address = _index(idx_addr)
    var y_address = _index(y_addr)
    var lp = i32_ptr(_index(out_labels_addr))
    var pp = f32_ptr(_index(out_proba_addr))
    var up = i32_ptr(_index(out_uniq_addr))
    var dt = _dist_triple(dist_params)
    with GILReleased(Python()):
        if ni < 1 or nq < 1 or kk < 1 or kk > ni or no < 1:
            raise Error("precomputed neighbors: invalid shape")
        var xp = u32_ptr(idx_address)
        var idx = List[UInt32](length=nq * kk, fill=UInt32(0))
        for i in range(nq * kk):
            idx[i] = xp[i]
        _host_validate_neighbors(idx, ni, nq, kk, no, dt[2])
        if len(n_classes) != no:
            raise Error("precomputed neighbors: class-count shape mismatch")
        for count in n_classes:
            if count < 1:
                raise Error("precomputed neighbors: class counts must be positive")
        var weighted = dt[2] == KNN_HOST_WEIGHTS_DISTANCE
        var w = List[Float32]()
        if weighted:
            w = host_distance_weights(read_f32(dist_address, nq * kk), nq, kk)
        var ys = List[List[Int32]]()
        var uniqs = List[List[Int32]]()
        for i in range(no):
            ys.append(read_i32(y_address + i * ni * 4, ni))
            uniqs.append(host_unique_labels(ys[i], ni))
            if len(uniqs[i]) != n_classes[i]:
                raise Error(
                    "knn_classifier_predict: the implemented getUniquelabels found "
                    + String(len(uniqs[i]))
                    + " classes for output "
                    + String(i)
                    + " and the caller sized its buffers for "
                    + String(n_classes[i])
                    + "; one of the two class sets is wrong and nothing is "
                    + "written"
                )
        var labels_out = List[Int32](length=nq * no, fill=Int32(0))
        var off = 0
        for i in range(no):
            var n_uniq = len(uniqs[i])
            var mono = host_monotonic(ys[i], ni, uniqs[i])
            var proba: List[Float32]
            if weighted:
                proba = host_weighted_class_probs(idx, mono, w, n_uniq, nq, kk)
            else:
                proba = host_class_probs(idx, mono, n_uniq, nq, kk)
            if want_proba:
                for j in range(nq * n_uniq):
                    pp[off + j] = proba[j]
                off += nq * n_uniq
            else:
                host_class_vote(proba, uniqs[i], n_uniq, nq, labels_out, no, i)
        if not want_proba:
            for j in range(nq * no):
                lp[j] = labels_out[j]
        var uoff = 0
        for i in range(no):
            for j in range(len(uniqs[i])):
                up[uoff + j] = uniqs[i][j]
            uoff += len(uniqs[i])
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
        if ni < 1 or nq < 1 or kk < 1 or kk > ni or no < 1:
            raise Error("precomputed neighbors: invalid shape")
        var xp = u32_ptr(idx_address)
        var idx = List[UInt32](length=nq * kk, fill=UInt32(0))
        for i in range(nq * kk):
            idx[i] = xp[i]
        _host_validate_neighbors(idx, ni, nq, kk, no, dt[2])
        var weighted = dt[2] == KNN_HOST_WEIGHTS_DISTANCE
        var w = List[Float32]()
        if weighted:
            w = host_distance_weights(read_f32(dist_address, nq * kk), nq, kk)
        var out = List[Float32](length=nq * no, fill=Float32(0.0))
        for i in range(no):
            var y = read_f32(y_address + i * ni * 4, ni)
            if weighted:
                host_weighted_regress_avg(idx, y, w, nq, kk, out, no, i)
            else:
                host_regress_avg(idx, y, nq, kk, out, no, i)
        for j in range(nq * no):
            op[j] = out[j]
    return PythonObject(0)


# ===========================================================================
# THE BALL COVER ENTRIES (lane/cpu-training-batch3, 2026-09-14). The GPU
# binding's names and params lists (`bindings/_mojolearn.mojo:467-627`).
# ===========================================================================


def _radius_check_shapes(
    n_index: Int, n_queries: Int, n_features: Int, radius: Float32, who: String
) raises:
    """`neighbors/estimator.mojo::_radius_check_shapes`, its words."""
    if n_index <= 0:
        raise Error(who + ": n_index must be positive, got " + String(n_index))
    if n_queries <= 0:
        raise Error(
            who + ": n_queries must be positive, got " + String(n_queries)
        )
    if n_features <= 0:
        raise Error(
            who + ": n_features must be positive, got " + String(n_features)
        )
    if not (radius > Float32(0.0)):
        raise Error(
            who
            + ": radius must be positive and finite, got "
            + String(radius)
            + ". A radius of zero returns each query's exact duplicates only,"
            " which the index is not built to answer, and a negative or NaN"
            " radius has no neighbourhood at all."
        )


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
        _radius_check_shapes(ni, nq, nf, rad, "radius_neighbors_count")
        rbc_validate_metric(mtr, marg)
        var index = read_f32(index_address, ni * nf)
        var queries = read_f32(queries_address, nq * nf)
        var counts = List[Int32](length=nq, fill=Int32(0))
        host_rbc_radius_counts(
            index, ni, queries, nq, nf, rad, mtr, marg, counts
        )
        ap[0] = Int32(0)
        for q in range(nq):
            nnz += Int(counts[q])
            ap[q + 1] = Int32(nnz)
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
        _radius_check_shapes(ni, nq, nf, rad, "radius_neighbors_fill")
        rbc_validate_metric(mtr, marg)
        if cap < 0:
            raise Error(
                "radius_neighbors_fill: nnz_capacity must not be negative, got "
                + String(cap)
            )
        var index = read_f32(index_address, ni * nf)
        var queries = read_f32(queries_address, nq * nf)
        var indptr = List[Int32](length=nq + 1, fill=Int32(0))
        if ap[0] != Int32(0) or Int(ap[nq]) != cap:
            raise Error(
                "radius_neighbors_fill: indptr must start at zero and end at"
                " nnz_capacity; call radius_neighbors_count first"
            )
        for q in range(nq):
            if ap[q + 1] < ap[q]:
                raise Error("radius_neighbors_fill: indptr must be nondecreasing")
            indptr[q] = ap[q]
        indptr[nq] = ap[nq]
        var cols = List[Int32](length=cap, fill=Int32(0))
        var dists = List[Float32](length=cap, fill=Float32(0.0))
        var actual_counts = List[Int32](length=nq, fill=Int32(0))
        host_rbc_radius_fill_rows(
            index, ni, queries, nq, nf, rad, sq, mtr, marg,
            indptr, actual_counts, cols, dists,
        )
        for q in range(nq):
            var expected = Int(indptr[q + 1] - indptr[q])
            var actual = Int(actual_counts[q])
            nnz += actual
            if actual != expected:
                raise Error(
                    "radius_neighbors_fill: query row " + String(q)
                    + " changed from " + String(expected) + " to "
                    + String(actual) + " edges between the count and fill calls."
                    " Re-run radius_neighbors_count against the arrays this call"
                    " was given rather than returning a stale CSR layout."
                )
        if nnz > 0:
            var xp = i32_ptr(idx_address)
            var dp = f32_ptr(dist_address)
            for p in range(nnz):
                xp[p] = cols[p]
                dp[p] = dists[p]
        for q in range(nq + 1):
            ap[q] = indptr[q]
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
        _radius_check_shapes(ni, nq, nf, Float32(1.0), "rbc_knn_search")
        rbc_validate_metric(mtr, marg)
        if kk < 1:
            raise Error("rbc_knn_search: k must be at least 1, got " + String(kk))
        if kk > ni:
            raise Error(
                "rbc_knn_search: k = "
                + String(kk)
                + " exceeds the "
                + String(ni)
                + " points in the index. Refused rather than padded: a padded"
                " answer is indistinguishable from a complete one."
            )
        if kk > RBC_KNN_MAX_K:
            raise Error(
                "rbc_knn_search: k = "
                + String(kk)
                + " exceeds RBC_KNN_MAX_K = "
                + String(RBC_KNN_MAX_K)
                + ". Use knn_search, whose selector is sized per launch."
            )
        var index = read_f32(index_address, ni * nf)
        var queries = read_f32(queries_address, nq * nf)
        var out_idx = List[Int32](length=nq * kk, fill=Int32(-1))
        var out_dist = List[Float32](length=nq * kk, fill=Float32(0.0))
        host_rbc_knn_search(
            index, ni, queries, nq, nf, kk, mtr, marg, out_idx, out_dist,
        )
        for i in range(nq * kk):
            xp[i] = out_idx[i]
            dp[i] = out_dist[i]
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
        host_kmeans_validate(ns, nf, nc, nw)
        var x = read_f32(x_address, ns * nf)
        var centroids: List[Float32]
        if ii == INIT_ARRAY:
            centroids = read_f32(centroids_address, nc * nf)
        else:
            centroids = List[Float32](length=nc * nf, fill=Float32(0.0))
        var weights = List[Float32]()
        if nw != 0:
            weights = read_f32(weights_address, nw)
        var labels = List[UInt32](length=ns, fill=UInt32(0))
        var r = host_kmeans_fit(
            x, ns, nf, nc, centroids, labels, weights, nw, mi, tl, sd,
            ninit, ii, mm, ovs,
        )
        var cp = f32_ptr(centroids_address)
        for i in range(nc * nf):
            cp[i] = centroids[i]
        for i in range(ns):
            lp[i] = labels[i]
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
        if ns < 1 or nf < 1 or nc < 1:
            raise Error(
                "kmeans_predict needs n_samples, n_features and n_clusters >= 1: got "
                + String(ns)
                + ", "
                + String(nf)
                + ", "
                + String(nc)
            )
        var x = read_f32(x_address, ns * nf)
        var centroids = read_f32(centroids_address, nc * nf)
        var labels = List[UInt32](length=ns, fill=UInt32(0))
        host_kmeans_predict(x, ns, nf, centroids, nc, mm, labels)
        for i in range(ns):
            lp[i] = labels[i]
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
        if ns < 1 or nf < 1 or nc < 1:
            raise Error(
                "kmeans_transform needs n_samples, n_features and n_clusters >= 1: got "
                + String(ns)
                + ", "
                + String(nf)
                + ", "
                + String(nc)
            )
        var x = read_f32(x_address, ns * nf)
        var centroids = read_f32(centroids_address, nc * nf)
        var out = List[Float32](length=ns * nc, fill=Float32(0.0))
        host_kmeans_transform(x, ns, nf, centroids, nc, mm, out)
        for i in range(ns * nc):
            op[i] = out[i]
    return PythonObject(ns)


@export
def PyInit__mojolearn_core_host() abi("C") -> PythonObject:
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
        module.def_function[cast_f64_to_f32_binding]("cast_f64_to_f32")
        module.def_function[all_finite_f32_binding]("all_finite_f32")
        module.def_function[all_finite_f64_binding]("all_finite_f64")
        module.def_function[gather_i64_binding]("gather_i64")
        module.def_function[gather_f64_binding]("gather_f64")
        module.def_function[gather_rows_bytes_binding]("gather_rows_bytes")
        module.def_function[argmax_rows_f32_binding]("argmax_rows_f32")
        module.def_function[argmax_rows_f64_binding]("argmax_rows_f64")
        module.def_function[column_mean_f64_binding]("column_mean_f64")
        module.def_function[center_columns_f32_binding]("center_columns_f32")
        module.def_function[scale_rows_f32_binding]("scale_rows_f32")
        module.def_function[probability_rows_f32_binding]("probability_rows_f32")
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
        module.def_function[gather_i32_binding]("gather_i32")
        module.def_function[check_indices_i64_binding]("check_indices_i64")
        module.def_function[indices_overlap_i64_binding]("indices_overlap_i64")
        module.def_function[fold_ids_binding]("fold_ids")
        module.def_function[select_fold_i64_binding]("select_fold_i64")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_core_host: ", error))
