# SPDX-License-Identifier: Apache-2.0
"""RUN of the ANN fit doors (lane dw-neighbors, 2026-11-21): the typed
requests of `IVFPQIndex.fit`, `IVFSQIndex.fit`, `IVFRaBitQIndex.fit` and
`CagraIndex.fit`, each one call of the image's ANN operations table
(core/neighbors_entry_table_ann.mojo, operations 10 to 15) through the entry
the door resolved once at UNPACK from the image's key, and the release of a
resident handle through its minting image's entry. Typed Mojo only, no Python import: the boundary
(bindings/ann_fit_ops_native_door.mojo) unpacks, calls these and packs, and
holds the interpreter lock rule. No arithmetic lives here; the bodies are
the table entries'.

Every address is the caller's pinned array, kept alive by the door through
the call. The address and word lists of one call are the export's lists,
word for word (a handful of words per fit; nothing per query).
"""
from core.native_descriptor_abi import NativeDescriptorCall
from core.neighbors_entry_table import NeighborsEntryKey
from core.neighbors_entry_table_ann import (
    ANN_OP_IVF_PQ_BUILD, ANN_OP_IVF_PQ_BUILD_RESIDENT, ANN_OP_IVF_SQ_BUILD, ANN_OP_IVF_RABITQ_BUILD,
    ANN_OP_CAGRA_BUILD, ann_build_call, ann_index_release_call, ann_pq_register_built_call,
)

#: `ann_pq_fit_resident_run`'s answer of an image that keeps no resident
#: fit (tickets start at 1): nothing was built.
comptime ANN_NO_TICKET = 0


def ann_on_device(key: NeighborsEntryKey) -> Bool:
    """Whether the image is a GPU image (vendor word 0 is the CPU column)."""
    return key.vendor > 0


def ann_borrowed(minter: NeighborsEntryKey, own: NeighborsEntryKey) -> Bool:
    """Whether a resident handle minted by the image `minter` is BORROWED by
    an estimator whose device image is `own`: the two are different DEVICE
    columns (one the CPU column, the other a GPU image). Two images of one
    device (identical and fast) are never borrowed from each other."""
    return ann_on_device(minter) != ann_on_device(own)


def ann_release_run(entry: NativeDescriptorCall, handle: Int) raises:
    """Release the resident index `handle` in the image that minted it.
    GIL held by the caller (the resident registry)."""
    ann_index_release_call(entry, handle)


@fieldwise_init
struct AnnPqFitRequest(Copyable, Movable):
    """`IVFPQIndex.fit`: the rows at `x`, the output buffers (`codes` is 0
    for the resident build, which keeps the codes on the device) and the
    build words."""
    var x: Int
    var centers: Int
    var offsets: Int
    var indices: Int
    var codebooks: Int
    var codes: Int
    var n: Int
    var dim: Int
    var n_lists: Int
    var iters: Int
    var seed: Int
    var pq_dim: Int
    var pq_bits: Int
    var pq_iters: Int


def _pq_words(request: AnnPqFitRequest) -> List[Int]:
    var words: List[Int] = [request.n, request.dim, request.n_lists, request.iters, request.seed,
                            request.pq_dim, request.pq_bits, request.pq_iters]
    return words^


def ann_pq_fit_run(entry: NativeDescriptorCall, request: AnnPqFitRequest) raises:
    """The build that writes all five arrays, the codes among them. GIL
    released by the caller."""
    var addresses: List[Int] = [request.x, request.centers, request.offsets, request.indices,
                                request.codebooks, request.codes]
    _ = ann_build_call(entry, ANN_OP_IVF_PQ_BUILD, addresses, _pq_words(request))


def ann_pq_fit_resident_run(entry: NativeDescriptorCall, request: AnnPqFitRequest) raises -> Int:
    """The resident build, first phase: the four small arrays written, the
    index left on the device and parked in the image; its ticket for
    `ann_pq_register_run`, or `ANN_NO_TICKET` when the image keeps no
    resident fit (nothing was built). GIL released by the caller."""
    var addresses: List[Int] = [request.x, request.centers, request.offsets, request.indices,
                                request.codebooks]
    return ann_build_call(entry, ANN_OP_IVF_PQ_BUILD_RESIDENT, addresses, _pq_words(request))


def ann_pq_register_run(entry: NativeDescriptorCall, ticket: Int) raises -> Int:
    """The resident build, second phase: the parked index registered; its
    handle. GIL held by the caller (the resident registry)."""
    return ann_pq_register_built_call(entry, ticket)


@fieldwise_init
struct AnnSqFitRequest(Copyable, Movable):
    """`IVFSQIndex.fit`. `train_rows_per_list` 0 keeps the image's default
    training rows."""
    var x: Int
    var centers: Int
    var offsets: Int
    var indices: Int
    var vmin: Int
    var delta: Int
    var codes: Int
    var n: Int
    var dim: Int
    var n_lists: Int
    var iters: Int
    var seed: Int
    var train_rows_per_list: Int


def ann_sq_fit_run(entry: NativeDescriptorCall, request: AnnSqFitRequest) raises:
    var addresses: List[Int] = [request.x, request.centers, request.offsets, request.indices, request.vmin,
                                request.delta, request.codes]
    var words: List[Int] = [request.n, request.dim, request.n_lists, request.iters, request.seed,
                            request.train_rows_per_list]
    _ = ann_build_call(entry, ANN_OP_IVF_SQ_BUILD, addresses, words)


@fieldwise_init
struct AnnRabitqFitRequest(Copyable, Movable):
    """`IVFRaBitQIndex.fit`."""
    var x: Int
    var centers: Int
    var offsets: Int
    var indices: Int
    var codes: Int
    var norms: Int
    var ips: Int
    var n: Int
    var dim: Int
    var n_lists: Int
    var iters: Int
    var seed: Int


def ann_rabitq_fit_run(entry: NativeDescriptorCall, request: AnnRabitqFitRequest) raises:
    var addresses: List[Int] = [request.x, request.centers, request.offsets, request.indices, request.codes,
                                request.norms, request.ips]
    var words: List[Int] = [request.n, request.dim, request.n_lists, request.iters, request.seed]
    _ = ann_build_call(entry, ANN_OP_IVF_RABITQ_BUILD, addresses, words)


@fieldwise_init
struct AnnCagraFitRequest(Copyable, Movable):
    """`CagraIndex.fit`: the rows at `x`, the n x graph_degree graph out."""
    var x: Int
    var graph: Int
    var n: Int
    var d: Int
    var intermediate_degree: Int
    var graph_degree: Int


def ann_cagra_fit_run(entry: NativeDescriptorCall, request: AnnCagraFitRequest) raises:
    var addresses: List[Int] = [request.x, request.graph]
    var words: List[Int] = [request.n, request.d, request.intermediate_degree, request.graph_degree]
    _ = ann_build_call(entry, ANN_OP_CAGRA_BUILD, addresses, words)
