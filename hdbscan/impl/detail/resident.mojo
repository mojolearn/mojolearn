# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HDBSCAN's fitted snapshot ON THE DEVICE (lane vdf-misc4, 2026-11-24).

Before this file every `approximate_predict`, `membership_vector` and
`all_points_membership_vectors` call copied the whole fitted state host to
host (core/hdbscan_query_api.mojo `capture`), uploaded the training rows and
the tree arrays again per call (soft clustering: per BATCH), gathered the
exemplar rows on the host, and ran the k-NN through a host-pointer boundary
(the result went device -> host -> device between two device steps).

`HdbscanResident` holds what one query kind reads, uploaded ONCE from the
fitted arrays' own storage: the training rows (the k-NN index and, for
`all_points_membership_vectors`, the rows themselves, read in place through
a sub-buffer), the core distances and labels, the condensed tree arrays, the
selected clusters and, for the two soft queries, the exemplar rows gathered
ON THE DEVICE from the training rows (`exemplar_gather_kernel`, a copy: the
same words the host gather wrote). The door keeps the handle in the model's
query record and releases it when the record is dropped (a refit, a binding
change, the model's end), so the snapshot is freed deterministically.

The registry follows spectral/impl/spectral_predict_resident.mojo: entries
behind reference-counted pointers, a lock word around every dictionary step;
a query copies the pointer under the lock, so a release during a query drops
only the registry's reference.

WHAT DOES NOT CHANGE: the kernels, their order and their inputs' bytes. The
per-call entries (`approximate_predict`, `membership_vector`,
`all_points_membership_vectors` on host lists) build a temporary snapshot
and run the same device bodies, so both routes are one body."""
from std.atomic import Atomic, Ordering
from std.memory import ArcPointer

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.shared_slot import SharedSlot

comptime _RES_TPB = 256
comptime _F32P = MutPointer[Float32, MutAnyOrigin]
comptime _I32P = MutPointer[Int32, MutAnyOrigin]


def exemplar_gather_kernel(x: _F32P, idx: _I32P, n_ex: Int32, n: Int32, dst: _F32P):
    """`soft_clustering.cuh:65-75` copy_rows: dst[j, f] = x[idx[j], f], one
    thread per cell (was a host loop over the exemplars)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var cells = Int(n_ex) * Int(n)
    if t < cells:
        var j = t // Int(n)
        var f = t % Int(n)
        dst.unsafe_store(t, x.unsafe_load(Int(idx.unsafe_load(j)) * Int(n) + f))


def _up_f32(ctx: DeviceContext, address: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
    """A device buffer of max(n, 1) cells holding the n float32 words at the
    host `address` (one copy; none when n is 0 or the address is 0)."""
    var buf = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    if n > 0 and address != 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=_F32P(unsafe_from_address=address))
    return buf^


def _up_i32(ctx: DeviceContext, address: Int, n: Int) raises -> DeviceBuffer[DType.int32]:
    """`_up_f32` for int32 words."""
    var buf = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    if n > 0 and address != 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=_I32P(unsafe_from_address=address))
    return buf^


struct HdbscanResident(Movable):
    """One fitted model's state for one query kind, on `ctx`'s device.
    Buffers a kind does not read hold one unwritten cell."""

    var ctx: DeviceContext
    var m: Int
    var n: Int
    var n_edges: Int
    var n_clusters: Int
    var n_selected: Int
    var n_exemplars: Int
    var x: DeviceBuffer[DType.float32]
    var core: DeviceBuffer[DType.float32]
    var labels: DeviceBuffer[DType.int32]
    var parents: DeviceBuffer[DType.int32]
    var lambdas: DeviceBuffer[DType.float32]
    var deaths: DeviceBuffer[DType.float32]
    var selected: DeviceBuffer[DType.int32]
    var iic: DeviceBuffer[DType.int32]
    var offsets: DeviceBuffer[DType.int32]
    var ex: DeviceBuffer[DType.float32]

    def __init__(
        out self, ctx: DeviceContext, x: Int, m: Int, n: Int, core: Int, labels: Int,
        parents: Int, lambdas: Int, n_edges: Int, deaths: Int, n_clusters: Int,
        selected: Int, n_selected: Int, iic: Int, exemplar_idx: Int, n_exemplars: Int,
        offsets: Int,
    ) raises:
        """Every array from its host address, uploaded once; an address of 0
        is an array this query kind does not read. `exemplar_idx` names the
        rows of `x` gathered into `ex` on the device."""
        if m < 1 or n < 1:
            raise Error("hdbscan resident snapshot: shape refused by name (m=" + String(m)
                        + ", d=" + String(n) + ")")
        var nx = n_exemplars if exemplar_idx != 0 else 0
        # exact length: the k-NN index is read as exactly m x n
        var xb = ctx.enqueue_create_buffer[DType.float32](m * n)
        ctx.enqueue_copy(dst_buf=xb, src_ptr=_F32P(unsafe_from_address=x))
        var exb = ctx.enqueue_create_buffer[DType.float32](max(nx * n, 1))
        var idx = _up_i32(ctx, exemplar_idx, nx)
        if nx > 0:
            var cells = nx * n
            ctx.enqueue_function[exemplar_gather_kernel](
                xb.unsafe_ptr(), idx.unsafe_ptr(), Int32(nx), Int32(n), exb.unsafe_ptr(),
                grid_dim=((cells + _RES_TPB - 1) // _RES_TPB, 1, 1), block_dim=(_RES_TPB, 1, 1),
            )
        self.core = _up_f32(ctx, core, m if core != 0 else 0)
        self.labels = _up_i32(ctx, labels, m if labels != 0 else 0)
        self.parents = _up_i32(ctx, parents, n_edges if parents != 0 else 0)
        self.lambdas = _up_f32(ctx, lambdas, n_edges)
        self.deaths = _up_f32(ctx, deaths, n_clusters)
        self.selected = _up_i32(ctx, selected, n_selected)
        self.iic = _up_i32(ctx, iic, n_edges + 1)
        self.offsets = _up_i32(ctx, offsets, n_selected + 1 if offsets != 0 else 0)
        # One wait for every upload and the gather; the index upload the
        # gather read lives past it.
        ctx.synchronize()
        _ = idx^
        self.x = xb^
        self.ex = exb^
        self.m = m
        self.n = n
        self.n_edges = n_edges
        self.n_clusters = n_clusters
        self.n_selected = n_selected
        self.n_exemplars = nx
        self.ctx = ctx.copy()

    def __deinit__(deinit self):
        # The buffers before the context they were created on (DEVIATION 1946).
        _ = self.x^
        _ = self.core^
        _ = self.labels^
        _ = self.parents^
        _ = self.lambdas^
        _ = self.deaths^
        _ = self.selected^
        _ = self.iic^
        _ = self.offsets^
        _ = self.ex^
        _ = self.ctx^


struct HdbscanResidentRegistry(Defaultable, Movable):
    var entries: Dict[Int, ArcPointer[HdbscanResident]]
    var next_id: Int
    var lock: List[Int64]

    def __init__(out self):
        self.entries = Dict[Int, ArcPointer[HdbscanResident]]()
        self.next_id = 1
        self.lock = List[Int64](length=1, fill=Int64(0))


comptime HDBSCAN_RESIDENT_REGISTRY = SharedSlot[HdbscanResidentRegistry, (
    "MojolearnHdbscanResidentIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojolearnHdbscanResidentFastV1"
)]


def _registry() raises -> UnsafePointer[HdbscanResidentRegistry, MutAnyOrigin]:
    return HDBSCAN_RESIDENT_REGISTRY.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _lock(state: UnsafePointer[HdbscanResidentRegistry, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the registry's lock word (held for one dictionary step only)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _unlock(state: UnsafePointer[HdbscanResidentRegistry, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


def hdbscan_resident_publish(var snapshot: HdbscanResident) raises -> Int:
    """Register `snapshot`; the handle every later query of the model names."""
    var entry = ArcPointer[HdbscanResident](snapshot^)
    var reg = _registry()
    _lock(reg)
    try:
        if reg[].next_id == 9223372036854775807:
            raise Error("resident hdbscan handle space exhausted")
        var handle = reg[].next_id
        reg[].next_id += 1
        reg[].entries[handle] = entry^
        _unlock(reg)
        return handle
    except error:
        _unlock(reg)
        raise error^


def hdbscan_resident_release(handle: Int) raises:
    """Drop the registry's reference; quiet for a released handle."""
    var reg = _registry()
    _lock(reg)
    try:
        if handle not in reg[].entries:
            _unlock(reg)
            return
        var popped = reg[].entries.pop(handle)
        _unlock(reg)
        # Dropped outside the lock: a running query holds its own reference.
        _ = popped^
    except error:
        _unlock(reg)
        raise error^


def hdbscan_resident_count() raises -> Int:
    """The number of live resident hdbscan handles (tests only)."""
    var reg = _registry()
    _lock(reg)
    var n = len(reg[].entries)
    _unlock(reg)
    return n


def hdbscan_resident_view(handle: Int) raises -> ArcPointer[HdbscanResident]:
    """The handle's snapshot (a new reference; the memory is shared)."""
    var reg = _registry()
    _lock(reg)
    try:
        if handle not in reg[].entries:
            raise Error("unknown or released resident hdbscan handle")
        var view = reg[].entries[handle].copy()
        _unlock(reg)
        return view^
    except error:
        _unlock(reg)
        raise error^
