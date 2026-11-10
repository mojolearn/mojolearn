# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident `cluster_centers_` of the x_cluster estimators
(lane ne1-a-cluster-spectral, owed rows N024 and N021).

MiniBatchKMeans, BisectingKMeans, MeanShift and AffinityPropagation
`predict`, `transform` and `score` used to admit `cluster_centers_` again
and upload it as the aux block of every `x_cluster_call`, and X went
through a host List copy before its upload. The first query after a fit now
uploads the centers ONCE (`xc_centers_prepare`); the handle is kept by the
provider (bindings/cluster_native.mojo, a native side table keyed by the
model's identity and generation word, never the model's `__dict__`), and
every later query names it (`xc_resident_query`): X is uploaded straight
from the caller's array and the result is downloaded straight into the
caller's output array.

The registry follows `cluster/resident_centers.mojo`: one process-wide
`_Global` keyed by a monotonically increasing handle, a lock word around
every dictionary step (the queries run with the GIL released), and a query
takes its own reference-counted copies of the buffer handle under the lock,
so a release during a query drops only the registry's reference. The
buffers live on the shared x_cluster context (`x_cluster_ctx`), the one
every `DeviceOps` of this binding runs on.

WHAT DOES NOT CHANGE. A resident query runs the very kernels of
`entries.nearest_entry` / `entries.distances_entry` (`DeviceOps.nearest`,
`sum_ff` over the distances, `sqdist` then `sqrt`) on the same center bytes
and the same X bytes, in the same order: the bits are the same."""
from std.atomic import Atomic, Ordering

from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_cluster.device_ops import DeviceOps, x_cluster_ctx
from x_cluster.post_bodies import FM_VAL
from core.shared_slot import SharedSlot

comptime XcF32Ptr = MutPointer[Float32, MutUntrackedOrigin]
comptime XcI32Ptr = MutPointer[Int32, MutUntrackedOrigin]

# The query kinds of `xc_resident_query` (the provider's operation words).
comptime XC_RES_PREDICT = 0
"""labels (int32, n) of the nearest center."""
comptime XC_RES_SCORE = 1
"""the float-float fold of the squared distance to the nearest center (returned)."""
comptime XC_RES_TRANSFORM = 2
"""euclidean distances (float32, n x k)."""


struct XcResidentCenters(Movable):
    """One fitted model's `k x d` float32 centers on the device, uploaded once."""

    var ctx: DeviceContext
    var centers: DeviceBuffer[DType.float32]
    var k: Int
    var d: Int

    def __init__(out self, centers_ptr: XcF32Ptr, k: Int, d: Int) raises:
        if k <= 0 or d <= 0:
            raise Error("x_cluster_centers_prepare: k and d must be positive")
        var ctx = x_cluster_ctx()
        var centers = ctx.enqueue_create_buffer[DType.float32](k * d)
        ctx.enqueue_copy(dst_buf=centers, src_ptr=centers_ptr)
        ctx.synchronize()
        self.k = k
        self.d = d
        self.centers = centers^
        self.ctx = ctx^

    def __deinit__(deinit self):
        # The buffer before the context it was created on (DEVIATION 1946).
        _ = self.centers^
        _ = self.ctx^


struct XcCentersRegistry(Defaultable, Movable):
    var entries: Dict[Int, XcResidentCenters]
    var next_id: Int
    # One lock word: queries run with the GIL released (see the module note).
    var lock: List[Int64]

    def __init__(out self):
        self.entries = Dict[Int, XcResidentCenters]()
        self.next_id = 1
        self.lock = List[Int64](length=1, fill=Int64(0))


comptime XC_CENTERS_REGISTRY = SharedSlot[XcCentersRegistry, ( "MojolearnXClusterResidentCentersIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnXClusterResidentCentersFastV1" )]


def _registry() raises -> UnsafePointer[XcCentersRegistry, MutAnyOrigin]:
    return XC_CENTERS_REGISTRY.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _lock(state: UnsafePointer[XcCentersRegistry, MutAnyOrigin]):
    # Non-raising (lesson 6): a lock helper never raises inside a typed try.
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the registry's lock word (held for one dictionary step only)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _unlock(state: UnsafePointer[XcCentersRegistry, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


struct _XcView(Movable):
    """A query's own copies of one entry's context and buffer HANDLES
    (reference counted; the memory is shared) and its shape."""

    var ctx: DeviceContext
    var centers: DeviceBuffer[DType.float32]
    var k: Int
    var d: Int

    def __init__(out self, var ctx: DeviceContext, var centers: DeviceBuffer[DType.float32], k: Int, d: Int):
        self.ctx = ctx^
        self.centers = centers^
        self.k = k
        self.d = d

    def __deinit__(deinit self):
        _ = self.centers^
        _ = self.ctx^


def xc_centers_prepare(centers_ptr: XcF32Ptr, k: Int, d: Int) raises -> Int:
    """Upload the centers once; the handle every later query names."""
    var state = _registry()
    # The upload runs outside the lock; only the dictionary step is locked.
    var entry = XcResidentCenters(centers_ptr, k, d)
    _lock(state)
    try:
        if state[].next_id == 9223372036854775807:
            raise Error("resident x_cluster center handle space exhausted")
        var handle = state[].next_id
        state[].next_id += 1
        state[].entries[handle] = entry^
        _unlock(state)
        return handle
    except error:
        _unlock(state)
        raise error^


def xc_centers_release(handle: Int) raises:
    """Drop the device copy; quiet for a handle already released (a copied
    instance, an eviction that ran first)."""
    var state = _registry()
    _lock(state)
    try:
        if handle not in state[].entries:
            _unlock(state)
            return
        var popped = state[].entries.pop(handle)
        _unlock(state)
        # Destroyed outside the lock: a running query holds its own handle copies.
        _ = popped^
    except error:
        _unlock(state)
        raise error^


def xc_centers_count() raises -> Int:
    """The number of live resident center handles (tests only)."""
    var state = _registry()
    _lock(state)
    var n = len(state[].entries)
    _unlock(state)
    return n


def _view(handle: Int, k: Int, d: Int) raises -> _XcView:
    var state = _registry()
    _lock(state)
    try:
        if handle not in state[].entries:
            raise Error("unknown or released resident x_cluster center handle")
        ref entry = state[].entries[handle]
        if entry.k != k or entry.d != d:
            raise Error(
                "x_cluster resident: the handle holds " + String(entry.k) + " x " + String(entry.d)
                + " centers, the call names " + String(k) + " x " + String(d)
            )
        var view = _XcView(entry.ctx.copy(), entry.centers.copy(), entry.k, entry.d)
        _unlock(state)
        return view^
    except error:
        _unlock(state)
        raise error^


def xc_resident_query(
    handle: Int, kind: Int, x_ptr: XcF32Ptr, n: Int, k: Int, d: Int, out_addr: Int,
) raises -> Float64:
    """One query on the handle's centers. X (n x d float32, C order) is
    uploaded from `x_ptr`; XC_RES_PREDICT writes n int32 labels and
    XC_RES_TRANSFORM n x k float32 distances to `out_addr` (the caller's
    output array, alive for the call); XC_RES_SCORE returns the fold of the
    squared distances (`out_addr` unused). The kernels and their order are
    `entries.nearest_entry` / `entries.distances_entry`'s."""
    if n < 1 or k < 1 or d < 1:
        raise Error(
            "x_cluster resident query needs n, k and d >= 1: got "
            + String(n) + ", " + String(k) + ", " + String(d)
        )
    var view = _view(handle, k, d)
    var ops = DeviceOps()
    # X straight from the caller's array: no host List. The stream orders
    # every kernel behind the copy; the download below synchronizes, and
    # `x_ptr` outlives the call.
    var xbuf = ops.ctx.enqueue_create_buffer[DType.float32](n * d)
    ops.ctx.enqueue_copy(dst_buf=xbuf, src_ptr=x_ptr)
    ops.f.append(xbuf^)
    var xs = len(ops.f) - 1
    ops.f.append(view.centers.copy())
    var cs = len(ops.f) - 1
    var total = Float64(0)
    if kind == XC_RES_TRANSFORM:
        var dist = ops.zeros(n * k)
        ops.sqdist(xs, n, cs, k, d, dist)
        ops.sqrt(dist, n * k)
        var out_f = XcF32Ptr(unsafe_from_address=out_addr)
        var dview = ops.f[dist].create_sub_buffer[DType.float32](0, n * k)
        ops.ctx.enqueue_copy(dst_ptr=out_f, src_buf=dview)
        ops.ctx.synchronize()
    elif kind == XC_RES_PREDICT or kind == XC_RES_SCORE:
        var ls = ops.zeros_i(n)
        var ds = ops.zeros(n)
        ops.nearest(xs, n, cs, k, d, ls, ds)
        if kind == XC_RES_SCORE:
            total = ops.sum_ff(ds, -1, -1, n, FM_VAL)
        else:
            var out_i = XcI32Ptr(unsafe_from_address=out_addr)
            var lview = ops.i[ls].create_sub_buffer[DType.int32](0, n)
            ops.ctx.enqueue_copy(dst_ptr=out_i, src_buf=lview)
        ops.ctx.synchronize()
    else:
        raise Error("x_cluster resident query: unknown kind " + String(kind))
    _ = ops^
    _ = view^
    return total
