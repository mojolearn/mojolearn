# SPDX-License-Identifier: Apache-2.0
"""The IVF-Flat operations table (lane dw-neighbors, 2026-11-21): the build,
search, resident and extend entries of the GPU image `_mojolearn_ivf`
(ivf/ops_entry_device.mojo), of its CPU column `_mojolearn_ivf_host`
(ivf/ops_entry_host_build.mojo) and of the shipped search-only CPU image
`_mojolearn_ivf_search_host` (ivf/ops_entry_host.mojo) as one
`NativeDescriptorCall` per image, dispatching on its operation id. The IVF
RUN (ivf/ivf_query_run.mojo) calls it with plain words instead of the
bindings' Python `ivf_flat_*` exports and their two Python lists.

Modeled on core/neighbors_entry_table.mojo: the same core/entry_table.mojo
publication and lookup, one slot name per tier, vendor and image. The key is
the `(mode, vendor, image)` an image records as its module attribute
`ivf_native_key` when it publishes (vendor words 0 cpu, 1 metal, 2 cuda,
3 hip; image 0 the full image, 1 the search-only CPU image, which carries no
build); the door reads that attribute at UNPACK. The two CPU images share one
host handle registry (ivf/ivf_host_search_run.mojo `HOST_IVF_REGISTRY`), so a
handle either minted is read by both. Publication and lookup are metadata
operations with no independent numerical work to split.

Request layouts (a: addresses, p: Int words; no float word). The address and
word orders are the exports' (bindings/ivf_index_arrays_unpack.mojo), with
the optional sample filter as a fixed last address (0: none) and the
partial-storage flag as a fixed last word.
  IVF_OP_BUILD           a [x, centers, center_norms, offsets, indices, list_data]
                         p [n, dim, n_lists, kmeans_n_iters, metric, seed]            -> 0
  IVF_OP_BUILD_RESIDENT  a [x, centers, center_norms, offsets, indices]
                         p [n, dim, n_lists, kmeans_n_iters, metric, seed]            -> handle
  IVF_OP_SEARCH          a [centers, center_norms, offsets, indices, list_data, queries,
                            distances, ids, candidates, filter (0: none)]
                         p [n, dim, n_lists, metric, m, k, n_probes, partial]         -> 0
  IVF_OP_INDEX_PREPARE   a [centers, center_norms, offsets, indices, list_data]
                         p [n, dim, n_lists, metric, partial]                         -> handle
  IVF_OP_INDEX_SEARCH    a [queries, distances, ids, candidates, filter (0: none)]
                         p [handle, n, dim, n_lists, metric, m, k, n_probes, partial] -> 0
  IVF_OP_INDEX_RELEASE   a []  p [handle]                                             -> 0
  IVF_OP_INDEX_EXPORT    a [list_data]  p [handle, count]                             -> 0
  IVF_OP_EXTEND          a [centers, center_norms, offsets, indices, list_data, new_x,
                            offsets_out, indices_out, list_data_out, labels_out]
                         p [n, dim, n_lists, metric, n_new]                           -> 0
A CPU image answers all but IVF_OP_BUILD_RESIDENT and IVF_OP_INDEX_EXPORT (its
prepared index keeps the caller's arrays; nothing lives on a device); the
search-only CPU image also refuses IVF_OP_BUILD, as its absent export did.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call

comptime IVF_OPS_ABI = "mojolearn.ivf.ops.v1"
comptime IVF_OP_BUILD = 0
comptime IVF_OP_BUILD_RESIDENT = 1
comptime IVF_OP_SEARCH = 2
comptime IVF_OP_INDEX_PREPARE = 3
comptime IVF_OP_INDEX_SEARCH = 4
comptime IVF_OP_INDEX_RELEASE = 5
comptime IVF_OP_INDEX_EXPORT = 6
comptime IVF_OP_EXTEND = 7
comptime IVF_OPERATION_COUNT = 8
#: The image word of a key: the full image, or the search-only CPU image.
comptime IVF_IMAGE_FULL = 0
comptime IVF_IMAGE_SEARCH = 1


@fieldwise_init
struct IvfEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int
    var image: Int

    def on_device(self) -> Bool:
        """Whether the image is a GPU image (it keeps a built index on the
        device: the resident build and the list-data export exist)."""
        return self.vendor != 0

    def builds(self) -> Bool:
        return self.image == IVF_IMAGE_FULL

    def same(self, other: IvfEntryKey) -> Bool:
        return self.mode == other.mode and self.vendor == other.vendor and self.image == other.image


def ivf_ops_check_spans(operation: Int, aa: Int, na: Int, pp: Int, np: Int, nf: Int, result: Int) raises:
    """Refuse a request whose spans do not match the operation's layout
    before any word is read."""
    if not result:
        raise Error("ivf ops: null result word")
    if (na > 0 and not aa) or (np > 0 and not pp) or na < 0 or np < 0 or nf != 0:
        raise Error("ivf ops: null or invalid request span")
    var want_a: Int
    var want_p: Int
    if operation == IVF_OP_BUILD:
        want_a = 6
        want_p = 6
    elif operation == IVF_OP_BUILD_RESIDENT:
        want_a = 5
        want_p = 6
    elif operation == IVF_OP_SEARCH:
        want_a = 10
        want_p = 8
    elif operation == IVF_OP_INDEX_PREPARE:
        want_a = 5
        want_p = 5
    elif operation == IVF_OP_INDEX_SEARCH:
        want_a = 5
        want_p = 9
    elif operation == IVF_OP_INDEX_RELEASE:
        want_a = 0
        want_p = 1
    elif operation == IVF_OP_INDEX_EXPORT:
        want_a = 1
        want_p = 2
    elif operation == IVF_OP_EXTEND:
        want_a = 10
        want_p = 5
    else:
        raise Error("ivf ops: unknown operation " + String(operation))
    if na != want_a or np != want_p:
        raise Error("ivf ops: request spans differ from operation " + String(operation))


def ivf_ops_publish(key: IvfEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    var schemas: List[String] = [String(IVF_OPS_ABI)]
    var operations: List[Int] = [IVF_OPERATION_COUNT]
    var entries: List[NativeDescriptorCall] = [entry]
    if key.image == IVF_IMAGE_SEARCH and key.mode == 1 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesIvfOpsIdenticalCPUSearchV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.image != IVF_IMAGE_FULL:
        raise Error("native ivf ops: unsupported execution key")
    elif key.mode == 1 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesIvfOpsIdenticalCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 0:
        entry_table_publish["MojolearnEntriesIvfOpsFastCPUV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesIvfOpsIdenticalMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 1:
        entry_table_publish["MojolearnEntriesIvfOpsFastMetalV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesIvfOpsIdenticalCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 2:
        entry_table_publish["MojolearnEntriesIvfOpsFastCUDAV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 1 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesIvfOpsIdenticalHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    elif key.mode == 0 and key.vendor == 3:
        entry_table_publish["MojolearnEntriesIvfOpsFastHIPV1"](key.mode, key.vendor, publisher, schemas, operations, entries)
    else:
        raise Error("native ivf ops: unsupported execution key")
    _ = schemas
    _ = operations
    _ = entries


def ivf_ops_entry(key: IvfEntryKey) raises -> NativeDescriptorCall:
    var schema = String(IVF_OPS_ABI)
    if key.image == IVF_IMAGE_SEARCH and key.mode == 1 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesIvfOpsIdenticalCPUSearchV1"](key.mode, key.vendor, 0, schema, IVF_OPERATION_COUNT)
    if key.image != IVF_IMAGE_FULL:
        raise Error("native ivf ops: unsupported execution key")
    if key.mode == 1 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesIvfOpsIdenticalCPUV1"](key.mode, key.vendor, 0, schema, IVF_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 0:
        return entry_table_read["MojolearnEntriesIvfOpsFastCPUV1"](key.mode, key.vendor, 0, schema, IVF_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesIvfOpsIdenticalMetalV1"](key.mode, key.vendor, 0, schema, IVF_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 1:
        return entry_table_read["MojolearnEntriesIvfOpsFastMetalV1"](key.mode, key.vendor, 0, schema, IVF_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesIvfOpsIdenticalCUDAV1"](key.mode, key.vendor, 0, schema, IVF_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 2:
        return entry_table_read["MojolearnEntriesIvfOpsFastCUDAV1"](key.mode, key.vendor, 0, schema, IVF_OPERATION_COUNT)
    if key.mode == 1 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesIvfOpsIdenticalHIPV1"](key.mode, key.vendor, 0, schema, IVF_OPERATION_COUNT)
    if key.mode == 0 and key.vendor == 3:
        return entry_table_read["MojolearnEntriesIvfOpsFastHIPV1"](key.mode, key.vendor, 0, schema, IVF_OPERATION_COUNT)
    raise Error("native ivf ops: unsupported execution key")


# ---------------------------------------------------------------- typed calls
# Each call is synchronous; the caller keeps every addressed buffer alive
# until it returns.


@fieldwise_init
struct IvfIndexWords(Copyable, Movable):
    """The four words that name a built index: rows, features, lists and
    the metric code."""
    var n: Int
    var dim: Int
    var n_lists: Int
    var metric: Int


@fieldwise_init
struct IvfIndexSpans(Copyable, Movable):
    """The addresses of a built index's five arrays (the build's outputs)."""
    var centers: Int
    var center_norms: Int
    var offsets: Int
    var indices: Int
    var list_data: Int


def ivf_build_call(entry: NativeDescriptorCall, x: Int, spans: IvfIndexSpans, n: Int, dim: Int, n_lists: Int,
                   iters: Int, metric: Int, seed: Int) raises:
    var addresses: List[Int] = [x, spans.centers, spans.center_norms, spans.offsets, spans.indices, spans.list_data]
    var ints: List[Int] = [n, dim, n_lists, iters, metric, seed]
    var floats = List[Float64]()
    _ = native_descriptor_call(entry, IVF_OP_BUILD, addresses, ints, floats)


def ivf_build_resident_call(entry: NativeDescriptorCall, x: Int, spans: IvfIndexSpans, n: Int, dim: Int,
                            n_lists: Int, iters: Int, metric: Int, seed: Int) raises -> Int:
    """The resident build: the four small arrays written (`spans.list_data`
    is unread: the vectors stay on the device); the handle."""
    var addresses: List[Int] = [x, spans.centers, spans.center_norms, spans.offsets, spans.indices]
    var ints: List[Int] = [n, dim, n_lists, iters, metric, seed]
    var floats = List[Float64]()
    return native_descriptor_call(entry, IVF_OP_BUILD_RESIDENT, addresses, ints, floats)


def ivf_search_call(entry: NativeDescriptorCall, spans: IvfIndexSpans, queries: Int, distances: Int, ids: Int,
                    candidates: Int, keep: Int, index: IvfIndexWords, m: Int, k: Int, n_probes: Int,
                    partial: Bool) raises:
    var addresses: List[Int] = [spans.centers, spans.center_norms, spans.offsets, spans.indices, spans.list_data,
                                queries, distances, ids, candidates, keep]
    var ints: List[Int] = [index.n, index.dim, index.n_lists, index.metric, m, k, n_probes, Int(partial)]
    var floats = List[Float64]()
    _ = native_descriptor_call(entry, IVF_OP_SEARCH, addresses, ints, floats)


def ivf_index_prepare_call(entry: NativeDescriptorCall, spans: IvfIndexSpans, index: IvfIndexWords,
                           partial: Bool) raises -> Int:
    var addresses: List[Int] = [spans.centers, spans.center_norms, spans.offsets, spans.indices, spans.list_data]
    var ints: List[Int] = [index.n, index.dim, index.n_lists, index.metric, Int(partial)]
    var floats = List[Float64]()
    return native_descriptor_call(entry, IVF_OP_INDEX_PREPARE, addresses, ints, floats)


def ivf_index_search_call(entry: NativeDescriptorCall, handle: Int, queries: Int, distances: Int, ids: Int,
                          candidates: Int, keep: Int, index: IvfIndexWords, m: Int, k: Int, n_probes: Int,
                          partial: Bool) raises:
    var addresses: List[Int] = [queries, distances, ids, candidates, keep]
    var ints: List[Int] = [handle, index.n, index.dim, index.n_lists, index.metric, m, k, n_probes, Int(partial)]
    var floats = List[Float64]()
    _ = native_descriptor_call(entry, IVF_OP_INDEX_SEARCH, addresses, ints, floats)


def ivf_index_release_call(entry: NativeDescriptorCall, handle: Int) raises:
    var addresses = List[Int]()
    var ints: List[Int] = [handle]
    var floats = List[Float64]()
    _ = native_descriptor_call(entry, IVF_OP_INDEX_RELEASE, addresses, ints, floats)


def ivf_index_export_call(entry: NativeDescriptorCall, handle: Int, list_data: Int, count: Int) raises:
    var addresses: List[Int] = [list_data]
    var ints: List[Int] = [handle, count]
    var floats = List[Float64]()
    _ = native_descriptor_call(entry, IVF_OP_INDEX_EXPORT, addresses, ints, floats)


def ivf_extend_call(entry: NativeDescriptorCall, spans: IvfIndexSpans, new_x: Int, offsets_out: Int,
                    indices_out: Int, list_data_out: Int, labels_out: Int, index: IvfIndexWords,
                    n_new: Int) raises:
    var addresses: List[Int] = [spans.centers, spans.center_norms, spans.offsets, spans.indices, spans.list_data,
                                new_x, offsets_out, indices_out, list_data_out, labels_out]
    var ints: List[Int] = [index.n, index.dim, index.n_lists, index.metric, n_new]
    var floats = List[Float64]()
    _ = native_descriptor_call(entry, IVF_OP_EXTEND, addresses, ints, floats)
