# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident IVF-Flat index (lane/py-dn-ann, 2026-09-28): the
closure of DEVIATION 1804.

`IVFIndex.search` used to hand the five index arrays to the binding on
every call, which copied them into host lists, validated them, uploaded the
centroids and the list data, recomputed the list norms, downloaded them and
rebuilt the host CSR layout, all before the first query was scored (about
0.1 to 0.3 s per call at 1M x 128). cuVS keeps the index on the device from
`build` to the last `search`; this file gives `IVFIndex` the same residency
through a handle: `ivf_flat_index_prepare` admits the arrays ONCE
(`ivf_validate_index_arrays`, the same refusals) and keeps the index and its
`IvfFlatDevice` (ivf_flat_search.mojo), every later search names the handle
and uploads only its queries, and the handle is released when the Python
instance drops it (a refit, an extend, collection).

The registry is `neighbors/resident_index.mojo`'s (FOREST-RESIDENT-1): one
process-wide `_Global` keyed by a monotonically increasing integer handle,
the calling extension holding the GIL across prepare, search and release.
The buffers live on the binding's one process-lifetime context
(`process_ctx`), which outlives every entry.

WHAT DOES NOT CHANGE. `ivf_flat_search_prepared` is the body every search
ran after its own preparation: the same statements over the same bytes, so
a search through a handle returns the bits a one-shot search returns (the
ivf, ivf-euclidean, ivf-filter and par-ivf lanes, base against head, both
columns).
"""
from std.ffi import _Global

from max.gpu.host import DeviceContext
from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE
from std.memory import memcpy
from bindings.hostptr import f32_ptr
from ivf.impl.neighbors.ivf_flat.ivf_flat_build import IvfFlatBuildDevice
from ivf.impl.neighbors.ivf_flat.ivf_flat_index import IvfFlatIndex, IvfFlatSearchParams
from ivf.impl.neighbors.ivf_flat.ivf_flat_search import (
    IvfFlatDevice,
    IvfSearchResult,
    ivf_flat_search_prepared,
)


struct ResidentIvfFlat(Movable):
    """One admitted index: its host arrays (the per-query path gathers from
    them) and its prepared device side."""

    var index: IvfFlatIndex
    var dev: IvfFlatDevice
    var partial_storage: Bool

    def __init__(out self, ctx: DeviceContext, var index: IvfFlatIndex, partial_storage: Bool) raises:
        self.dev = IvfFlatDevice(ctx, index)
        self.index = index^
        self.partial_storage = partial_storage

    def __init__(
        out self, ctx: DeviceContext, var index: IvfFlatIndex, var built: IvfFlatBuildDevice
    ) raises:
        """A RESIDENT build's index (lane gap-ivf, 2026-10-08): the device
        buffers the build left behind, no upload; `index.list_data` empty."""
        self.dev = IvfFlatDevice(ctx, index, built^)
        self.index = index^
        self.partial_storage = False


struct IvfFlatRegistry(Movable):
    var entries: Dict[Int, ResidentIvfFlat]
    var next_id: Int

    def __init__(out self):
        self.entries = Dict[Int, ResidentIvfFlat]()
        self.next_id = 1


comptime IVF_FLAT_REGISTRY = _Global[
    StorageType=IvfFlatRegistry,
    name=(
        "MojoIvfFlatResidentIdentical" if GLOBAL_NUMERIC_MODE == 1 else
        "MojoIvfFlatResidentDeterministic" if GLOBAL_NUMERIC_MODE == 2 else
        "MojoIvfFlatResidentFast"
    ),
    init_fn=IvfFlatRegistry.__init__,
]


def ivf_resident_prepare(ctx: DeviceContext, var index: IvfFlatIndex, partial_storage: Bool) raises -> Int:
    """Upload and prepare the index once; the handle every later search names."""
    var state = IVF_FLAT_REGISTRY.get_or_create_ptr()
    if state[].next_id == 9223372036854775807:
        raise Error("resident IVF-Flat index handle space exhausted")
    var entry = ResidentIvfFlat(ctx, index^, partial_storage)
    var handle = state[].next_id
    state[].next_id += 1
    state[].entries[handle] = entry^
    return handle


def ivf_resident_register_built(
    ctx: DeviceContext, var index: IvfFlatIndex, var built: IvfFlatBuildDevice
) raises -> Int:
    """Register a resident build (`ivf_flat_build_resident`) under a new
    handle: `IVFIndex.fit`'s index stays on the device until its first
    `search` and every later one (lane gap-ivf, plan section 6 item 1)."""
    var state = IVF_FLAT_REGISTRY.get_or_create_ptr()
    if state[].next_id == 9223372036854775807:
        raise Error("resident IVF-Flat index handle space exhausted")
    var entry = ResidentIvfFlat(ctx, index^, built^)
    var handle = state[].next_id
    state[].next_id += 1
    state[].entries[handle] = entry^
    return handle


def ivf_resident_export_list_data(ctx: DeviceContext, handle: Int, dst_addr: Int, count: Int) raises:
    """The handle's n_rows x dim list vectors written to the caller's float32
    buffer at `dst_addr` (`IVFIndex.list_data_`, made on first read: `save`,
    a pickle, `extend`, a clone or a shard split; never inside fit or
    search). One device copy into a staging host buffer, one memcpy."""
    var state = IVF_FLAT_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident IVF-Flat index handle")
    ref e = state[].entries[handle]
    var n = e.index.n_rows * e.index.dim
    if count != n:
        raise Error(
            "ivf_flat_index_export: the buffer holds " + String(count)
            + " words, the index " + String(n)
        )
    if n == 0:
        return
    if dst_addr == 0:
        raise Error("ivf_flat_index_export: null buffer address")
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=e.dev.dlist_data)
    ctx.synchronize()
    memcpy(dest=f32_ptr(dst_addr), src=host.unsafe_ptr(), count=n)
    _ = host^


def ivf_resident_release(handle: Int) raises:
    """Drop the index; a released handle is refused by every later call."""
    var state = IVF_FLAT_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident IVF-Flat index handle")
    var released = state[].entries.pop(handle)
    _ = released^


def ivf_resident_check(handle: Int, n_rows: Int, dim: Int, n_lists: Int, metric: Int, partial_storage: Bool) raises:
    """The handle exists and holds the index the call names."""
    var state = IVF_FLAT_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident IVF-Flat index handle")
    ref e = state[].entries[handle]
    if (
        e.index.n_rows != n_rows or e.index.dim != dim or e.index.n_lists != n_lists
        or e.index.metric != metric or e.partial_storage != partial_storage
    ):
        raise Error(
            "ivf_flat_index_search: the handle holds another index (n="
            + String(e.index.n_rows) + " dim=" + String(e.index.dim) + " n_lists="
            + String(e.index.n_lists) + " metric=" + String(e.index.metric)
            + ") than the call names"
        )


def ivf_resident_n_rows(handle: Int) raises -> Int:
    var state = IVF_FLAT_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident IVF-Flat index handle")
    return state[].entries[handle].index.n_rows


def ivf_resident_search(
    ctx: DeviceContext,
    handle: Int,
    queries: List[Float32],
    n_queries: Int,
    k: Int,
    n_probes: Int,
    keep: List[Int32],
) raises -> IvfSearchResult:
    """`ivf_flat_search_host` over the handle's prepared index."""
    var state = IVF_FLAT_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released resident IVF-Flat index handle")
    ref e = state[].entries[handle]
    var sp = IvfFlatSearchParams(n_probes)
    var trace = IdentityTrace()
    return ivf_flat_search_prepared(
        ctx, trace, e.index, e.dev, sp, queries, n_queries, k,
        partial_storage=e.partial_storage, keep=keep,
    )
