# SPDX-License-Identifier: Apache-2.0
"""The neighbors operations table (lane calltable-cluster-neighbors,
2026-11-11): the k-NN entries of the base binding `_mojolearn` (device,
neighbors/ops_entry_device.mojo) and of its CPU column
`_mojolearn_core_host` (neighbors/ops_entry_host.mojo) as one
`NativeDescriptorCall` per image, dispatching on its operation id. The k-NN
owner (bindings/neighbors_knn_native.mojo) calls it with plain values
instead of the bindings' Python `knn_*` exports.

Modeled on core/solver_entry_table.mojo: the same core/entry_table.mojo
publication and lookup, separate slot names per tier and vendor. The key is
the `(mode, vendor)` an image records as its module attribute
`neighbors_native_key` when it publishes (vendor words 0 cpu, 1 metal,
2 cuda, 3 hip); the owner reads that attribute at UNPACK. Publication and
lookup are metadata operations with no independent numerical work to split.

Request layouts (a: addresses, p: Int words, f: Float64 words). The
classifier's p ends with `n_outputs` class counts.
  NB_OP_INDEX_PREPARE      a [index, status (0: none)]  p [n_index, n_features]       -> handle
  NB_OP_INDEX_RELEASE      a []  p [handle]                                           -> 0
  NB_OP_FOLD_READY         a []  p []                                                 -> 0 / 1
  NB_OP_CLASSIFY           a [index, queries, y, labels, proba, uniq]
                           p [n_index, n_queries, n_features, k, query_tile, n_outputs, want_proba,
                              metric, weights, n_classes...]  f [metric_arg]         -> query tile used
  NB_OP_REGRESS            a [index, queries, y, out]
                           p [n_index, n_queries, n_features, k, query_tile, n_outputs, metric, weights]
                           f [metric_arg]                                             -> query tile used
  NB_OP_CLASSIFY_RESIDENT  a [index, queries, y, labels, proba, uniq, status (0: none)]
                           p [handle, n_index, n_queries, n_features, k, query_tile, n_outputs, want_proba,
                              metric, weights, has_fold, fold_base, fold_rows, fold_index, n_classes...]
                           f [metric_arg]                                             -> query tile used
  NB_OP_REGRESS_RESIDENT   a [index, queries, y, out, status (0: none)]
                           p [handle, n_index, n_queries, n_features, k, query_tile, n_outputs, metric,
                              weights, has_fold, fold_base, fold_rows, fold_index]
                           f [metric_arg]                                             -> query tile used
The float word is rounded to Float32 by the entry, the same
`Float32(Float64(py=...))` rounding the Python exports apply, so the bits
are unchanged. The CPU column answers NB_OP_CLASSIFY, NB_OP_REGRESS, the
two radius passes, the search, the ball-cover search and the two votes on
merged neighbors (the host binding never had the resident entries); every
other operation is refused there.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call

comptime NEIGHBORS_OPS_ABI = "mojolearn.neighbors.ops.v1"
comptime NB_OP_INDEX_PREPARE = 0
comptime NB_OP_INDEX_RELEASE = 1
comptime NB_OP_FOLD_READY = 2
comptime NB_OP_CLASSIFY = 3
comptime NB_OP_REGRESS = 4
comptime NB_OP_CLASSIFY_RESIDENT = 5
comptime NB_OP_REGRESS_RESIDENT = 6
# The two passes of a radius query (lane dw-neighbors, 2026-11-21): the bodies
# of the base bindings' `radius_neighbors_count` / `radius_neighbors_fill`
# exports, so the radius RUN (neighbors/radius_run.mojo) calls them with plain
# words and no Python list.
#   RADIUS_COUNT  a: index, queries, row starts (int32 [n_queries + 1]),
#                    finite status word (int32; 0 = none)
#                 p: n_index, n_queries, n_features, metric
#                 f: radius, metric_arg            -> the edge count
#   RADIUS_FILL   a: index, queries, row starts, ids (int32), distances (float32)
#                 p: n_index, n_queries, n_features, nnz capacity, return_sqrt, metric
#                 f: radius, metric_arg            -> the edge count found
comptime NB_OP_RADIUS_COUNT = 7
comptime NB_OP_RADIUS_FILL = 8
# The k-NN query and the vote on merged neighbors (lane dw-neighbors,
# 2026-11-21): the bodies of the base bindings' `knn_search`,
# `knn_search_resident`, `rbc_knn_search`, `knn_classify_neighbors` and
# `knn_regress_neighbors` exports, so the k-NN RUN
# (neighbors/neighbors_knn_run.mojo) calls them with plain words and no
# Python list.
#   SEARCH             a: index, queries, distances (float32 [nq * k]), ids (uint32 [nq * k])
#                      p: n_index, n_queries, n_features, k, return_sqrt, query_tile, metric
#                      f: metric_arg                -> the query tile that ran
#   SEARCH_RESIDENT    a: index, queries, distances, ids, finite status word (int32; 0 = none)
#                      p: handle, n_index, n_queries, n_features, k, return_sqrt, query_tile, metric
#                      f: metric_arg                -> the query tile that ran
#   RBC_SEARCH         a: index, queries, ids (int32 [nq * k], -1 unfilled), distances (float32)
#                      p: n_index, n_queries, n_features, k, metric
#                      f: metric_arg                -> candidate distances computed
#   CLASSIFY_NEIGHBORS a: distances, ids (uint32), y, labels, proba, uniq
#                      p: n_index, n_queries, k, n_outputs, want_proba, weights, n_classes...
#                      f: (none)                    -> 0
#   REGRESS_NEIGHBORS  a: distances, ids (uint32), y, out
#                      p: n_index, n_queries, k, n_outputs, weights
#                      f: (none)                    -> 0
# The CPU column answers all but SEARCH_RESIDENT (it keeps no resident index).
comptime NB_OP_SEARCH = 9
comptime NB_OP_SEARCH_RESIDENT = 10
comptime NB_OP_RBC_SEARCH = 11
comptime NB_OP_CLASSIFY_NEIGHBORS = 12
comptime NB_OP_REGRESS_NEIGHBORS = 13
#   INDEX_PREPARE_FOLD a: status (two host int32 words: finite, range)
#                      p: fold base id, base rows, fold index id (int64), n_index, n_features
#                      f: (none)                    -> handle (0: a flagged fold, nothing kept)
# Lane vdf-scope-cluster2a (2026-11-24): the resident index built from a
# model-selection fold on the device (the fold's rows of the resident base
# gathered into the index buffer, no host words); the device column only (the
# CPU column keeps no resident index and refuses it).
comptime NB_OP_INDEX_PREPARE_FOLD = 14
comptime NB_OPERATION_COUNT = 15
#: The fixed p words before the classifier's class counts.
comptime NB_CLASSIFY_FIXED = 9
comptime NB_CLASSIFY_RESIDENT_FIXED = 14
#: The fixed p words before the neighbor vote's class counts.
comptime NB_CLASSIFY_NEIGHBORS_FIXED = 6


@fieldwise_init
struct NeighborsEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def neighbors_vendor_word(vendor: StaticString) -> Int:
    """The table's vendor word for the string an image reports ("cpu" for
    the host image, `COMPILED_VENDOR` for the GPU one): 0 cpu, 1 metal,
    2 cuda, 3 hip, else -1 (publishes nothing)."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def nb_ops_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                       ff: Int, nf: Int, result: Int) raises:
    """Refuse a request whose spans do not match the operation's layout
    before any word is read (the classifiers' class-count tail is checked
    against n_outputs by the entry)."""
    if not result:
        raise Error("neighbors ops: null result word")
    if (na > 0 and not aa) or (np > 0 and not pp) or (nf > 0 and not ff) or na < 0 or np < 0 or nf < 0:
        raise Error("neighbors ops: null or invalid request span")
    var want_a: Int
    var want_p: Int
    var want_f: Int
    if operation == NB_OP_INDEX_PREPARE:
        want_a = 2
        want_p = 2
        want_f = 0
    elif operation == NB_OP_INDEX_RELEASE:
        want_a = 0
        want_p = 1
        want_f = 0
    elif operation == NB_OP_FOLD_READY:
        want_a = 0
        want_p = 0
        want_f = 0
    elif operation == NB_OP_INDEX_PREPARE_FOLD:
        want_a = 1
        want_p = 5
        want_f = 0
    elif operation == NB_OP_CLASSIFY:
        want_a = 6
        want_p = -1
        want_f = 1
        if np < NB_CLASSIFY_FIXED:
            raise Error("neighbors ops: classify needs at least " + String(NB_CLASSIFY_FIXED) + " words")
    elif operation == NB_OP_REGRESS:
        want_a = 4
        want_p = 8
        want_f = 1
    elif operation == NB_OP_CLASSIFY_RESIDENT:
        want_a = 7
        want_p = -1
        want_f = 1
        if np < NB_CLASSIFY_RESIDENT_FIXED:
            raise Error("neighbors ops: resident classify needs at least "
                        + String(NB_CLASSIFY_RESIDENT_FIXED) + " words")
    elif operation == NB_OP_REGRESS_RESIDENT:
        want_a = 5
        want_p = 13
        want_f = 1
    elif operation == NB_OP_RADIUS_COUNT:
        want_a = 4
        want_p = 4
        want_f = 2
    elif operation == NB_OP_RADIUS_FILL:
        want_a = 5
        want_p = 6
        want_f = 2
    elif operation == NB_OP_SEARCH:
        want_a = 4
        want_p = 7
        want_f = 1
    elif operation == NB_OP_SEARCH_RESIDENT:
        want_a = 5
        want_p = 8
        want_f = 1
    elif operation == NB_OP_RBC_SEARCH:
        want_a = 4
        want_p = 5
        want_f = 1
    elif operation == NB_OP_CLASSIFY_NEIGHBORS:
        want_a = 6
        want_p = -1
        want_f = 0
        if np < NB_CLASSIFY_NEIGHBORS_FIXED:
            raise Error("neighbors ops: classify neighbors needs at least "
                        + String(NB_CLASSIFY_NEIGHBORS_FIXED) + " words")
    elif operation == NB_OP_REGRESS_NEIGHBORS:
        want_a = 4
        want_p = 5
        want_f = 0
    else:
        raise Error("neighbors ops: unknown operation " + String(operation))
    if na != want_a or (want_p >= 0 and np != want_p) or nf != want_f:
        raise Error("neighbors ops: request spans differ from operation " + String(operation))


def nb_class_counts(p: MutPointer[Int, MutUntrackedOrigin], first: Int, np: Int, n_outputs: Int) raises -> List[Int]:
    """The classifier's `n_outputs` class counts at p[first:], refused when
    the tail length differs (the exports' "7 + n_outputs" refusal)."""
    if n_outputs < 0 or np != first + n_outputs:
        raise Error("knn_classify: params must hold " + String(first) + " + n_outputs ("
                    + String(first + n_outputs) + ") values, got " + String(np))
    var counts = List[Int](capacity=n_outputs)
    for i in range(n_outputs):  # small-loop(no: model outputs, one class count each): parameter list, not data
        counts.append(p[first + i])
    return counts^


def nb_status_word(address: Int) raises -> Int:
    """`_dist_status`'s check: 0, or a positive int32-aligned address."""
    if address < 0 or address % 4 != 0:
        raise Error("knn: the status address must be a positive int32-aligned address")
    return address


def nb_fold_words(has_fold: Int, base: Int, rows: Int, index: Int, status: Int) raises -> Tuple[Int, Int, Int]:
    """`_dist_fold`'s check: `(-1, 0, -1)` when no fold is asked."""
    if has_fold == 0:
        return (-1, 0, -1)
    if base < 0 or rows < 1 or index < 0:
        raise Error("knn: a fold needs a resident base id, its row count and an index id")
    if status == 0:
        raise Error("knn: a fold needs the status words' address")
    return (base, rows, index)


def neighbors_ops_publish(key: NeighborsEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    var schemas: List[String] = [String(NEIGHBORS_OPS_ABI)]
    var operations: List[Int] = [NB_OPERATION_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    if key.mode == 1 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesNeighborsOpsIdenticalCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesNeighborsOpsFastCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesNeighborsOpsIdenticalMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesNeighborsOpsFastMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesNeighborsOpsIdenticalCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesNeighborsOpsFastCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesNeighborsOpsIdenticalHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesNeighborsOpsFastHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    else:
        raise Error("native neighbors ops: unsupported execution key")
    _ = schemas
    _ = operations
    _ = entries


def neighbors_ops_entry(key: NeighborsEntryKey) raises -> NativeDescriptorCall:
    var schema = String(NEIGHBORS_OPS_ABI)
    if key.mode == 1 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesNeighborsOpsIdenticalCPUV1"](key.mode, key.vendor, 0, schema, NB_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesNeighborsOpsFastCPUV1"](key.mode, key.vendor, 0, schema, NB_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesNeighborsOpsIdenticalMetalV1"](key.mode, key.vendor, 0, schema, NB_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesNeighborsOpsFastMetalV1"](key.mode, key.vendor, 0, schema, NB_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesNeighborsOpsIdenticalCUDAV1"](key.mode, key.vendor, 0, schema, NB_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesNeighborsOpsFastCUDAV1"](key.mode, key.vendor, 0, schema, NB_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesNeighborsOpsIdenticalHIPV1"](key.mode, key.vendor, 0, schema, NB_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesNeighborsOpsFastHIPV1"](key.mode, key.vendor, 0, schema, NB_OPERATION_COUNT)
    raise Error("native neighbors ops: unsupported execution key")


# ---------------------------------------------------------------- typed calls
# Each call is synchronous; the caller keeps every addressed buffer alive
# until it returns and releases the GIL around it.


def nb_index_prepare_call(entry: NativeDescriptorCall, index: Int, status: Int, n_index: Int,
                          n_features: Int) raises -> Int:
    var addresses: List[Int] = [index, status]
    var ints: List[Int] = [n_index, n_features]
    var floats = List[Float64]()
    return native_descriptor_call(entry, NB_OP_INDEX_PREPARE, addresses, ints, floats)


def nb_index_prepare_fold_call(entry: NativeDescriptorCall, status: Int, fold: NbFold, n_index: Int,
                               n_features: Int) raises -> Int:
    """NB_OP_INDEX_PREPARE_FOLD: the resident index of fold rows `fold.index`
    of the resident base `fold.base` (`fold.rows` rows), gathered on the
    device; `status` two host int32 words (finite, range), handle 0 when
    either is set."""
    var addresses: List[Int] = [status]
    var ints: List[Int] = [fold.base, fold.rows, fold.index, n_index, n_features]
    var floats = List[Float64]()
    return native_descriptor_call(entry, NB_OP_INDEX_PREPARE_FOLD, addresses, ints, floats)


def nb_index_release_call(entry: NativeDescriptorCall, handle: Int) raises:
    var addresses = List[Int]()
    var ints: List[Int] = [handle]
    var floats = List[Float64]()
    _ = native_descriptor_call(entry, NB_OP_INDEX_RELEASE, addresses, ints, floats)


def nb_fold_ready_call(entry: NativeDescriptorCall) raises -> Bool:
    var addresses = List[Int]()
    var ints = List[Int]()
    var floats = List[Float64]()
    return native_descriptor_call(entry, NB_OP_FOLD_READY, addresses, ints, floats) != 0


def nb_classify_call(entry: NativeDescriptorCall, index: Int, queries: Int, y: Int, labels: Int, proba: Int,
                     uniq: Int, n_index: Int, n_queries: Int, n_features: Int, k: Int, query_tile: Int,
                     want_proba: Bool, n_classes: List[Int], metric: Int, metric_arg: Float64,
                     weights: Int) raises -> Int:
    var addresses: List[Int] = [index, queries, y, labels, proba, uniq]
    var ints: List[Int] = [n_index, n_queries, n_features, k, query_tile, len(n_classes), Int(want_proba),
                           metric, weights]
    for i in range(len(n_classes)):  # small-loop(no: model outputs, one class count each): parameter list, not data
        ints.append(n_classes[i])
    var floats: List[Float64] = [metric_arg]
    return native_descriptor_call(entry, NB_OP_CLASSIFY, addresses, ints, floats)


def nb_regress_call(entry: NativeDescriptorCall, index: Int, queries: Int, y: Int, dst: Int, n_index: Int,
                    n_queries: Int, n_features: Int, k: Int, query_tile: Int, n_outputs: Int, metric: Int,
                    metric_arg: Float64, weights: Int) raises -> Int:
    var addresses: List[Int] = [index, queries, y, dst]
    var ints: List[Int] = [n_index, n_queries, n_features, k, query_tile, n_outputs, metric, weights]
    var floats: List[Float64] = [metric_arg]
    return native_descriptor_call(entry, NB_OP_REGRESS, addresses, ints, floats)


@fieldwise_init
struct NbFold(Copyable, Movable):
    """A model-selection fold the resident entry gathers on the device
    (`has_fold` 0: the queries are host words)."""
    var has_fold: Int
    var base: Int
    var rows: Int
    var index: Int

    @staticmethod
    def none() -> NbFold:
        return NbFold(0, -1, 0, -1)


def nb_classify_resident_call(entry: NativeDescriptorCall, handle: Int, index: Int, queries: Int, y: Int,
                              labels: Int, proba: Int, uniq: Int, status: Int, n_index: Int, n_queries: Int,
                              n_features: Int, k: Int, query_tile: Int, want_proba: Bool, n_classes: List[Int],
                              metric: Int, metric_arg: Float64, weights: Int, fold: NbFold) raises -> Int:
    var addresses: List[Int] = [index, queries, y, labels, proba, uniq, status]
    var ints: List[Int] = [handle, n_index, n_queries, n_features, k, query_tile, len(n_classes), Int(want_proba),
                           metric, weights, fold.has_fold, fold.base, fold.rows, fold.index]
    for i in range(len(n_classes)):  # small-loop(no: model outputs, one class count each): parameter list, not data
        ints.append(n_classes[i])
    var floats: List[Float64] = [metric_arg]
    return native_descriptor_call(entry, NB_OP_CLASSIFY_RESIDENT, addresses, ints, floats)


def nb_regress_resident_call(entry: NativeDescriptorCall, handle: Int, index: Int, queries: Int, y: Int,
                             dst: Int, status: Int, n_index: Int, n_queries: Int, n_features: Int, k: Int,
                             query_tile: Int, n_outputs: Int, metric: Int, metric_arg: Float64, weights: Int,
                             fold: NbFold) raises -> Int:
    var addresses: List[Int] = [index, queries, y, dst, status]
    var ints: List[Int] = [handle, n_index, n_queries, n_features, k, query_tile, n_outputs, metric, weights,
                           fold.has_fold, fold.base, fold.rows, fold.index]
    var floats: List[Float64] = [metric_arg]
    return native_descriptor_call(entry, NB_OP_REGRESS_RESIDENT, addresses, ints, floats)


def nb_radius_count_call(entry: NativeDescriptorCall, index: Int, queries: Int, row_starts: Int, status: Int,
                         n_index: Int, n_queries: Int, n_features: Int, metric: Int,
                         radius: Float64, metric_arg: Float64) raises -> Int:
    var addresses: List[Int] = [index, queries, row_starts, status]
    var ints: List[Int] = [n_index, n_queries, n_features, metric]
    var floats: List[Float64] = [radius, metric_arg]
    return native_descriptor_call(entry, NB_OP_RADIUS_COUNT, addresses, ints, floats)


def nb_radius_fill_call(entry: NativeDescriptorCall, index: Int, queries: Int, row_starts: Int, ids: Int,
                        distances: Int, n_index: Int, n_queries: Int, n_features: Int, capacity: Int,
                        metric: Int, radius: Float64, metric_arg: Float64) raises -> Int:
    var addresses: List[Int] = [index, queries, row_starts, ids, distances]
    var ints: List[Int] = [n_index, n_queries, n_features, capacity, 1, metric]
    var floats: List[Float64] = [radius, metric_arg]
    return native_descriptor_call(entry, NB_OP_RADIUS_FILL, addresses, ints, floats)


def nb_search_call(entry: NativeDescriptorCall, index: Int, queries: Int, distances: Int, ids: Int,
                   n_index: Int, n_queries: Int, n_features: Int, k: Int, return_sqrt: Bool, query_tile: Int,
                   metric: Int, metric_arg: Float64) raises -> Int:
    var addresses: List[Int] = [index, queries, distances, ids]
    var ints: List[Int] = [n_index, n_queries, n_features, k, Int(return_sqrt), query_tile, metric]
    var floats: List[Float64] = [metric_arg]
    return native_descriptor_call(entry, NB_OP_SEARCH, addresses, ints, floats)


def nb_search_resident_call(entry: NativeDescriptorCall, handle: Int, index: Int, queries: Int, distances: Int,
                            ids: Int, status: Int, n_index: Int, n_queries: Int, n_features: Int, k: Int,
                            return_sqrt: Bool, query_tile: Int, metric: Int, metric_arg: Float64) raises -> Int:
    var addresses: List[Int] = [index, queries, distances, ids, status]
    var ints: List[Int] = [handle, n_index, n_queries, n_features, k, Int(return_sqrt), query_tile, metric]
    var floats: List[Float64] = [metric_arg]
    return native_descriptor_call(entry, NB_OP_SEARCH_RESIDENT, addresses, ints, floats)


def nb_rbc_search_call(entry: NativeDescriptorCall, index: Int, queries: Int, ids: Int, distances: Int,
                       n_index: Int, n_queries: Int, n_features: Int, k: Int, metric: Int,
                       metric_arg: Float64) raises -> Int:
    var addresses: List[Int] = [index, queries, ids, distances]
    var ints: List[Int] = [n_index, n_queries, n_features, k, metric]
    var floats: List[Float64] = [metric_arg]
    return native_descriptor_call(entry, NB_OP_RBC_SEARCH, addresses, ints, floats)


def nb_classify_neighbors_call(entry: NativeDescriptorCall, distances: Int, ids: Int, y: Int, labels: Int,
                               proba: Int, uniq: Int, n_index: Int, n_queries: Int, k: Int, want_proba: Bool,
                               n_classes: List[Int], weights: Int) raises:
    var addresses: List[Int] = [distances, ids, y, labels, proba, uniq]
    var ints: List[Int] = [n_index, n_queries, k, len(n_classes), Int(want_proba), weights]
    for i in range(len(n_classes)):  # small-loop(no: model outputs, one class count each): parameter list, not data
        ints.append(n_classes[i])
    var floats = List[Float64]()
    _ = native_descriptor_call(entry, NB_OP_CLASSIFY_NEIGHBORS, addresses, ints, floats)


def nb_regress_neighbors_call(entry: NativeDescriptorCall, distances: Int, ids: Int, y: Int, dst: Int,
                              n_index: Int, n_queries: Int, k: Int, n_outputs: Int, weights: Int) raises:
    var addresses: List[Int] = [distances, ids, y, dst]
    var ints: List[Int] = [n_index, n_queries, k, n_outputs, weights]
    var floats = List[Float64]()
    _ = native_descriptor_call(entry, NB_OP_REGRESS_NEIGHBORS, addresses, ints, floats)
