# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the base `_mojolearn` family: its HOST HELPERS (the CPU
training lane, phase 1, 2026-09-13; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md section 3.2) and, since the knn
host inference lane (2026-09-14), the three k-NN INFERENCE entries.

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

THE k-NN INFERENCE ENTRIES (the knn host inference lane, 2026-09-14;
brief docs/lanes/BRIEF_forest_host_inference_2026-09-13.md, "Classical
lanes", rank 8): `knn_search`, `knn_classify` and `knn_regress` are exported
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
(euclidean/l2, sqeuclidean) and refuses every other metric BY NAME; the
returned "query tile" is 1, the batch the host runs.

The base binding's OTHER estimator entries (`kmeans_fit`, `rbc_knn_search`,
`radius_neighbors_*`) and its other helpers are deliberately ABSENT here, so
the kmeans, rbc and radius lanes keep refusing BY NAME through
`_HostBinding` until a lane lands them.

`transpose_f32` and `cast_colmajor_f64_to_f32` MIRROR
`bindings/_mojolearn.mojo::_tiled_transpose_to_f32` (DEVIATIONS 2471,
2472) element for element, `dst[c * rows + r] = Float32(src[r * cols +
c])`; the tiling there is a cache order, not a value, so a plain loop
writes the same bytes. The seven others are `bindings/host_helpers.mojo`,
the forest host lane's, shared.
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
    gather_f64_binding,
    gather_i64_binding,
)
from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, read_f32, read_i32, u32_ptr
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.knn_host_predict import (
    KNN_HOST_SABOTAGE,
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
    helpers are untouched by it (byte moves have no fold) and the three
    k-NN entries walk every distance chain descending under it (the gate's
    negative control); refused outside the gate as one set."""
    return PythonObject(CORE_HOST_SABOTAGE_DEFINE or KNN_HOST_SABOTAGE)


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
            index, ni, queries, nq, nf, kk, dt[0], sq, out_dist, out_idx
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
        host_knn_search(index, ni, queries, nq, nf, kk, dt[0], weighted, dist, idx)
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
        host_knn_search(index, ni, queries, nq, nf, kk, dt[0], weighted, dist, idx)
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
        module.def_function[transpose_f32_binding]("transpose_f32")
        module.def_function[cast_colmajor_f64_to_f32_binding]("cast_colmajor_f64_to_f32")
        module.def_function[cast_f64_to_f32_binding]("cast_f64_to_f32")
        module.def_function[all_finite_f32_binding]("all_finite_f32")
        module.def_function[all_finite_f64_binding]("all_finite_f64")
        module.def_function[gather_i64_binding]("gather_i64")
        module.def_function[gather_f64_binding]("gather_f64")
        module.def_function[argmax_rows_f32_binding]("argmax_rows_f32")
        module.def_function[argmax_rows_f64_binding]("argmax_rows_f64")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_core_host: ", error))
