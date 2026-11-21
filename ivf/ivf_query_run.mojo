# SPDX-License-Identifier: Apache-2.0
"""RUN of the IVF-Flat doors (lane dw-neighbors, 2026-11-21): the typed
requests of `IVFIndex.fit`, `search` and `extend` and of the shard worker's
partial search, each one call of the image's IVF-Flat operations table
(core/ivf_entry_table.mojo) under the key the door read once at UNPACK.
Typed Mojo only, no Python import: the door (bindings/ivf_ops_native_door.mojo)
unpacks, calls these and packs. No arithmetic lives here; the bodies are the
table entries'.
"""
from core.ivf_entry_table import (
    IvfEntryKey, IvfIndexSpans, IvfIndexWords, ivf_build_call, ivf_build_resident_call, ivf_extend_call,
    ivf_index_export_call, ivf_index_prepare_call, ivf_index_release_call, ivf_index_search_call,
    ivf_ops_entry, ivf_search_call,
)

#: "No resident handle" (both registries mint handles from 1).
comptime IVF_NO_HANDLE = -1


def ivf_borrowed(minter: IvfEntryKey, own: IvfEntryKey) -> Bool:
    """Whether a resident handle minted by the image `minter` is BORROWED by
    an estimator whose device image is `own`: the two are different DEVICE
    columns (one a CPU image, the other a GPU image). Two images of one
    device (identical and fast; the two CPU images, which share one handle
    registry) are never borrowed from each other."""
    return minter.on_device() != own.on_device()


@fieldwise_init
struct IvfBuildRequest(Copyable, Movable):
    """`IVFIndex.fit`: the rows at `x` and the five output buffers."""
    var x: Int
    var spans: IvfIndexSpans
    var index: IvfIndexWords
    var iters: Int
    var seed: Int


def ivf_build_run(key: IvfEntryKey, request: IvfBuildRequest) raises -> Int:
    """The build. A GPU image keeps the index on the device and writes the
    four small arrays (`spans.list_data` is not written): its handle. A CPU
    image writes all five arrays: `IVF_NO_HANDLE`."""
    var entry = ivf_ops_entry(key)
    if key.on_device():
        return ivf_build_resident_call(entry, request.x, request.spans, request.index.n, request.index.dim,
                                       request.index.n_lists, request.iters, request.index.metric, request.seed)
    ivf_build_call(entry, request.x, request.spans, request.index.n, request.index.dim, request.index.n_lists,
                   request.iters, request.index.metric, request.seed)
    return IVF_NO_HANDLE


@fieldwise_init
struct IvfSearchRequest(Copyable, Movable):
    """One search. `handle` names a resident index, or `IVF_NO_HANDLE`: the
    five arrays at `spans` are read and admitted by this call. `keep` is the
    sample filter's address (0: none; never with partial storage)."""
    var index: IvfIndexWords
    var handle: Int
    var spans: IvfIndexSpans
    var queries: Int
    var distances: Int
    var ids: Int
    var candidates: Int
    var keep: Int
    var m: Int
    var k: Int
    var n_probes: Int
    var partial: Bool


def ivf_search_run(key: IvfEntryKey, request: IvfSearchRequest) raises:
    var entry = ivf_ops_entry(key)
    if request.handle != IVF_NO_HANDLE:
        ivf_index_search_call(entry, request.handle, request.queries, request.distances, request.ids,
                              request.candidates, request.keep, request.index, request.m, request.k,
                              request.n_probes, request.partial)
        return
    ivf_search_call(entry, request.spans, request.queries, request.distances, request.ids, request.candidates,
                    request.keep, request.index, request.m, request.k, request.n_probes, request.partial)


def ivf_prepare_run(key: IvfEntryKey, spans: IvfIndexSpans, index: IvfIndexWords, partial: Bool) raises -> Int:
    """Admit the five arrays once and keep them prepared: the handle."""
    return ivf_index_prepare_call(ivf_ops_entry(key), spans, index, partial)


def ivf_release_run(key: IvfEntryKey, handle: Int) raises:
    ivf_index_release_call(ivf_ops_entry(key), handle)


def ivf_export_run(key: IvfEntryKey, handle: Int, list_data: Int, count: Int) raises:
    """The device index's list vectors written to `count` float32 words."""
    ivf_index_export_call(ivf_ops_entry(key), handle, list_data, count)


@fieldwise_init
struct IvfExtendRequest(Copyable, Movable):
    """`IVFIndex.extend`: the built index, the new rows and the four output
    buffers (the centres and their norms do not change)."""
    var index: IvfIndexWords
    var spans: IvfIndexSpans
    var new_x: Int
    var n_new: Int
    var offsets_out: Int
    var indices_out: Int
    var list_data_out: Int
    var labels_out: Int


def ivf_extend_run(key: IvfEntryKey, request: IvfExtendRequest) raises:
    ivf_extend_call(ivf_ops_entry(key), request.spans, request.new_x, request.offsets_out, request.indices_out,
                    request.list_data_out, request.labels_out, request.index, request.n_new)
