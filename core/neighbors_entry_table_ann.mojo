# SPDX-License-Identifier: Apache-2.0
"""The ANN operations table (lane calltable-cn-ann, 2026-11-11): the x_ann
entries of the GPU binding `_mojolearn_x_ann` (x_ann/ops_entry_device.mojo)
and of its CPU column `_mojolearn_x_ann_host` (x_ann/ops_entry_host.mojo) as
one `NativeDescriptorCall` per image, dispatching on its operation id. The
ANN search owner (bindings/neighbors_ann_query_native.mojo) and the TSNE
owner (bindings/mani_tsne_native.mojo) call it with plain values instead of
the bindings' Python `x_ann_*` exports.

Modeled on core/neighbors_entry_table.mojo: the same core/entry_table.mojo
publication and lookup, separate slot names per tier and vendor, and the same
key: the `(mode, vendor)` an image records as its module attribute
`x_ann_native_key` when it publishes (vendor words 0 cpu, 1 metal, 2 cuda,
3 hip, `neighbors_vendor_word`). The owners read that attribute at UNPACK.
Publication and lookup are metadata operations with no independent numerical
work to split.

Request layouts (a: addresses, p: Int words, f: Float64 words). Each layout
is the Python export's `(addrs, params)` list, word for word, with the same
optional trailing words (the export's `len(params) > i` tests become
`np > i`); the docstrings in bindings/_mojolearn_x_ann.mojo name the words.
  ANN_OP_INDEX_PREPARE        a [the kind's fitted arrays (2..6)]  p [kind, the kind's prepare words] -> handle
  ANN_OP_INDEX_SEARCH         a [queries, out_d, out_i[, out_n[, filter]]]  p [handle, search words]   -> m
  ANN_OP_INDEX_RELEASE        a []  p [handle]                                                       -> 0
  ANN_OP_INDEX_EXPORT_CODES   a [codes]  p [handle, count]                                           -> 0
  ANN_OP_IVF_PQ_SEARCH        a 10  p n, dim, n_lists, pq_dim, pq_bits, m, k, n_probes[, status]     -> m
  ANN_OP_IVF_SQ_SEARCH        a 11  p n, dim, n_lists, m, k, n_probes[, status]                      -> m
  ANN_OP_IVF_RABITQ_SEARCH    a 11  p n, dim, n_lists, seed, m, k, n_probes[, status]                -> m
  ANN_OP_CAGRA_SEARCH         a 5   p n, d, graph_degree, m, k, itopk, width, max_iter, n_seeds[, rs[, status]] -> m
  ANN_OP_FILTER_TOPK          a 5   p m, n, L, k                                                     -> m
  ANN_OP_TSNE_FIT             a [x, y0, y_out, kl_out]  p n, d, max_iter, exploration[, init mode[, seed low,
                              seed high[, n_components[, finite flag]]]]
                              f perplexity, early_exaggeration, learning_rate, post factor
                              -> X's finite status word (0 when the flag is absent or X is finite)
The t-SNE float words are rounded to Float32 by the body, the same
`Float32(Float64(py=...))` rounding the Python export applies, so the bits
are unchanged. The resident operations exist on the device column only (the
host binding never had them); the CPU column refuses them.

The resident registry (x_ann/resident.mojo) is guarded by the GIL, not by a
lock word: its owners call ANN_OP_INDEX_* with the GIL HELD, as the Python
exports ran. The other operations are called with the GIL released.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.neighbors_entry_table import NeighborsEntryKey

comptime ANN_OPS_ABI = "mojolearn.neighbors.ann.v1"
comptime ANN_OP_INDEX_PREPARE = 0
comptime ANN_OP_INDEX_SEARCH = 1
comptime ANN_OP_INDEX_RELEASE = 2
comptime ANN_OP_INDEX_EXPORT_CODES = 3
comptime ANN_OP_IVF_PQ_SEARCH = 4
comptime ANN_OP_IVF_SQ_SEARCH = 5
comptime ANN_OP_IVF_RABITQ_SEARCH = 6
comptime ANN_OP_CAGRA_SEARCH = 7
comptime ANN_OP_FILTER_TOPK = 8
comptime ANN_OP_TSNE_FIT = 9
comptime ANN_OPERATION_COUNT = 10


def ann_ops_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                        ff: Int, nf: Int, result: Int) raises:
    """Refuse a request whose spans do not match the operation's layout
    before any word is read."""
    if not result:
        raise Error("ann ops: null result word")
    if (na > 0 and not aa) or (np > 0 and not pp) or (nf > 0 and not ff) or na < 0 or np < 0 or nf < 0:
        raise Error("ann ops: null or invalid request span")
    var a_lo: Int
    var a_hi: Int
    var p_lo: Int
    var p_hi: Int
    var want_f = 0
    if operation == ANN_OP_INDEX_PREPARE:
        a_lo = 2
        a_hi = 6
        p_lo = 4
        p_hi = 6
    elif operation == ANN_OP_INDEX_SEARCH:
        a_lo = 3
        a_hi = 5
        p_lo = 4
        p_hi = 9
    elif operation == ANN_OP_INDEX_RELEASE:
        a_lo = 0
        a_hi = 0
        p_lo = 1
        p_hi = 1
    elif operation == ANN_OP_INDEX_EXPORT_CODES:
        a_lo = 1
        a_hi = 1
        p_lo = 2
        p_hi = 2
    elif operation == ANN_OP_IVF_PQ_SEARCH:
        a_lo = 10
        a_hi = 10
        p_lo = 8
        p_hi = 9
    elif operation == ANN_OP_IVF_SQ_SEARCH:
        a_lo = 11
        a_hi = 11
        p_lo = 6
        p_hi = 7
    elif operation == ANN_OP_IVF_RABITQ_SEARCH:
        a_lo = 11
        a_hi = 11
        p_lo = 7
        p_hi = 8
    elif operation == ANN_OP_CAGRA_SEARCH:
        a_lo = 5
        a_hi = 5
        p_lo = 9
        p_hi = 11
    elif operation == ANN_OP_FILTER_TOPK:
        a_lo = 5
        a_hi = 5
        p_lo = 4
        p_hi = 4
    elif operation == ANN_OP_TSNE_FIT:
        a_lo = 4
        a_hi = 4
        p_lo = 4
        p_hi = 9
        want_f = 4
    else:
        raise Error("ann ops: unknown operation " + String(operation))
    if na < a_lo or na > a_hi or np < p_lo or np > p_hi or nf != want_f:
        raise Error("ann ops: request spans differ from operation " + String(operation))


def ann_span_list(p: MutPointer[Int, MutUntrackedOrigin], first: Int, count: Int) -> List[Int]:
    """`count` words at p[first:] as a list (the export's address or
    parameter list)."""
    var words = List[Int](capacity=max(count, 0))
    for i in range(count):  # small-loop(no: the address or parameter words of one call): metadata, not data
        words.append(p[first + i])
    return words^


def ann_ops_publish(key: NeighborsEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    var schemas: List[String] = [String(ANN_OPS_ABI)]
    var operations: List[Int] = [ANN_OPERATION_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    if key.mode == 1 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesNeighborsAnnIdenticalCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesNeighborsAnnFastCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesNeighborsAnnIdenticalMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesNeighborsAnnFastMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesNeighborsAnnIdenticalCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesNeighborsAnnFastCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesNeighborsAnnIdenticalHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesNeighborsAnnFastHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    else:
        raise Error("native ann ops: unsupported execution key")
    _ = schemas
    _ = operations
    _ = entries


def ann_ops_entry(key: NeighborsEntryKey) raises -> NativeDescriptorCall:
    var schema = String(ANN_OPS_ABI)
    if key.mode == 1 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesNeighborsAnnIdenticalCPUV1"](key.mode, key.vendor, 0, schema, ANN_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesNeighborsAnnFastCPUV1"](key.mode, key.vendor, 0, schema, ANN_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesNeighborsAnnIdenticalMetalV1"](key.mode, key.vendor, 0, schema, ANN_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesNeighborsAnnFastMetalV1"](key.mode, key.vendor, 0, schema, ANN_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesNeighborsAnnIdenticalCUDAV1"](key.mode, key.vendor, 0, schema, ANN_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesNeighborsAnnFastCUDAV1"](key.mode, key.vendor, 0, schema, ANN_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesNeighborsAnnIdenticalHIPV1"](key.mode, key.vendor, 0, schema, ANN_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesNeighborsAnnFastHIPV1"](key.mode, key.vendor, 0, schema, ANN_OPERATION_COUNT)
    raise Error("native ann ops: unsupported execution key")


# ---------------------------------------------------------------- typed calls
# Each call is synchronous; the caller keeps every addressed buffer alive
# until it returns. The ann_index_* calls run with the GIL held (see the
# module note); the others with the GIL released by the caller.


def ann_index_prepare_call(entry: NativeDescriptorCall, kind: Int, addresses: List[Int],
                           words: List[Int]) raises -> Int:
    var ints: List[Int] = [kind]
    for i in range(len(words)):  # small-loop(no: one call's prepare words): metadata, not data
        ints.append(words[i])
    var floats = List[Float64]()
    return native_descriptor_call(entry, ANN_OP_INDEX_PREPARE, addresses, ints, floats)


def ann_index_search_call(entry: NativeDescriptorCall, handle: Int, addresses: List[Int],
                          words: List[Int]) raises -> Int:
    var ints: List[Int] = [handle]
    for i in range(len(words)):  # small-loop(no: one call's search words): metadata, not data
        ints.append(words[i])
    var floats = List[Float64]()
    return native_descriptor_call(entry, ANN_OP_INDEX_SEARCH, addresses, ints, floats)


def ann_index_release_call(entry: NativeDescriptorCall, handle: Int) raises:
    var addresses = List[Int]()
    var ints: List[Int] = [handle]
    var floats = List[Float64]()
    _ = native_descriptor_call(entry, ANN_OP_INDEX_RELEASE, addresses, ints, floats)


def ann_index_export_codes_call(entry: NativeDescriptorCall, handle: Int, address: Int, count: Int) raises:
    var addresses: List[Int] = [address]
    var ints: List[Int] = [handle, count]
    var floats = List[Float64]()
    _ = native_descriptor_call(entry, ANN_OP_INDEX_EXPORT_CODES, addresses, ints, floats)


def ann_search_call(entry: NativeDescriptorCall, operation: Int, addresses: List[Int],
                    words: List[Int]) raises -> Int:
    """One of the per-call searches (ANN_OP_IVF_PQ_SEARCH,
    ANN_OP_IVF_SQ_SEARCH, ANN_OP_IVF_RABITQ_SEARCH, ANN_OP_CAGRA_SEARCH) or
    ANN_OP_FILTER_TOPK, with the export's address and parameter lists."""
    if (operation != ANN_OP_IVF_PQ_SEARCH and operation != ANN_OP_IVF_SQ_SEARCH
            and operation != ANN_OP_IVF_RABITQ_SEARCH and operation != ANN_OP_CAGRA_SEARCH
            and operation != ANN_OP_FILTER_TOPK):
        raise Error("ann ops: " + String(operation) + " is not a per-call search")
    var floats = List[Float64]()
    return native_descriptor_call(entry, operation, addresses, words, floats)


def ann_tsne_fit_call(entry: NativeDescriptorCall, x: Int, y0: Int, y_out: Int, kl_out: Int, n: Int, d: Int,
                      max_iter: Int, exploration: Int, perplexity: Float64, early_exaggeration: Float64,
                      learning_rate: Float64, post_factor: Float64, mode: Int, seed_low: Int, seed_high: Int,
                      n_components: Int, finite_flag: Int) raises -> Int:
    """The full t-SNE request (every optional word present); returns X's
    finite status word (0 unless the flag is set and X is not finite)."""
    var addresses: List[Int] = [x, y0, y_out, kl_out]
    var ints: List[Int] = [n, d, max_iter, exploration, mode, seed_low, seed_high, n_components, finite_flag]
    var floats: List[Float64] = [perplexity, early_exaggeration, learning_rate, post_factor]
    return native_descriptor_call(entry, ANN_OP_TSNE_FIT, addresses, ints, floats)
