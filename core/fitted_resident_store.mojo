# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device-resident fitted arrays, keyed by a lease (lane ne1-a-gp-svm,
workstream A of docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md).

One store struct and its functions, shared by every family that keeps
fitted float32 arrays on the device between predict calls (the Gaussian
processes in `bindings/_mojolearn_gp.mojo`, the kernel SVMs in
`bindings/_mojolearn_svm.mojo`). Each family binding owns ONE `_Global` of
`FittedResidentStore` under its own name (a FAST and an IDENTICAL binding
never share device memory), and passes a pointer to it here. Nothing here
computes on data.

THE KEY. A model's resident copies belong to a LEASE: an integer the
family binding mints (`fitted_lease_open`) and hands back inside a native
lease object (`PythonObject(alloc=...)`, the family's own small struct whose
`__deinit__` calls `fitted_lease_close`). The public door keeps that object
in the model's prepared-owner record (bindings/native_prepared_owner.mojo,
outside the model's `__dict__`), together with the model's generation word
(bindings/native_model_records.mojo) and the identities of the fitted arrays
the copies mirror (bindings/fitted_resident_door.mojo). A refit, a
`set_params`, a `__setstate__`, a replaced fitted array or a new binding
makes the record stale: the door mints a new lease, and replacing the record
drops the old lease object, which frees its copies. The model's death drops
the record the same way (the prepared-owner weakref callback). A lease has
any number of SLOTS (one per fitted machine: the one-vs-one pairs of an SVC,
the binary fits of a one-vs-rest classifier); slot `s` of lease `l` is entry
`l * FITTED_SLOTS_PER_LEASE + s`.

THE READ-ONLY RULE. The door takes a lease only when every mirrored public
array is an exact mojolearn Array it can make read-only
(`bindings/p3nc_fitted_freeze.mojo`), so an in-place write raises NumPy's
read-only error instead of leaving the device copy stale. No content hash.
As a second guard the entry records the host address and float count of
each array it mirrors and is refreshed when either differs.

MEMORY. One byte cap per store: an eighth of the device's total memory, read
once at the first upload (memory reasoning: a handful of fitted models in
one process, their fitted arrays a small share of the device; no model or
dataset shape). Past the cap the least recently used entries are freed; an
evicted model uploads again on its next call. A single model larger than
the cap is never kept: its call uploads into buffers that die with the call,
the per-call route it had before. A lease that is not live (closed, or 0)
gets the same per-call buffers.

THREADS. The predict entries run with the GIL released, so another thread
may close a lease or insert while a predict reads. Every store read and
write takes the store's lock word for one dictionary step only (the pattern
of kernel_methods/krr_resident.mojo); uploads and frees run outside it, and
a predict works on its own copies of the entry's context and buffer HANDLES
(reference counted, the memory is shared), so a close while a predict runs
drops only the store's references. The lock helpers do not raise (lesson 6).

NUMERICS. The device copy is the same bytes the host array holds and the
kernels read it where they read the per-call upload: no bit changes.
"""
from std.atomic import Atomic, Ordering
from max.gpu.host import DeviceBuffer, DeviceContext
from core.ctx_key import ctx_cache_key

# Slots per lease: one per fitted machine of a model. A model with more
# machines than this keeps the per-call upload for the rest (a key-space
# bound, not a tuning knob: 2^20 machines is far past any real model).
comptime FITTED_SLOTS_PER_LEASE = 1 << 20
# The share of the device's total memory one store may keep (1 / 8).
comptime FITTED_CAP_DIVISOR = 8


struct FittedResidentEntry(Copyable, Movable):
    """One slot's fitted arrays on the device."""

    # Copyable because the stdlib Dict iterates (`items()`) only Copyable
    # values (the lease close and the LRU eviction walk `entries`). A copy
    # holds new HANDLES to the same context and buffers (reference counted,
    # the device memory is shared), exactly what `fitted_resident_view`
    # already hands out; no entry is copied on the store's paths.

    var ctx: DeviceContext
    var bufs: List[DeviceBuffer[DType.float32]]
    var srcs: List[Int]
    var counts: List[Int]
    var device: Int
    var bytes: Int
    var used: Int

    def __init__(out self, var ctx: DeviceContext, var bufs: List[DeviceBuffer[DType.float32]],
                 var srcs: List[Int], var counts: List[Int], device: Int, bytes: Int, used: Int):
        self.ctx = ctx^
        self.bufs = bufs^
        self.srcs = srcs^
        self.counts = counts^
        self.device = device
        self.bytes = bytes
        self.used = used

    def __init__(out self, *, copy: Self):
        self.ctx = copy.ctx.copy()
        self.bufs = copy.bufs.copy()
        self.srcs = copy.srcs.copy()
        self.counts = copy.counts.copy()
        self.device = copy.device
        self.bytes = copy.bytes
        self.used = copy.used

    def __deinit__(deinit self):
        # The buffers before the context they were created on (DEVIATION 1946).
        _ = self.bufs^
        _ = self.ctx^


struct FittedResidentView(Movable):
    """A call's own copies of one slot's context and buffer HANDLES (shared
    memory, reference counted). `kept` is False for per-call buffers that
    die with the view; `fresh` is True when this call uploaded them (a
    caller that checks uploaded values checks them then)."""

    var ctx: DeviceContext
    var bufs: List[DeviceBuffer[DType.float32]]
    var kept: Bool
    var fresh: Bool

    def __init__(out self, var ctx: DeviceContext, var bufs: List[DeviceBuffer[DType.float32]], kept: Bool,
                 fresh: Bool):
        self.ctx = ctx^
        self.bufs = bufs^
        self.kept = kept
        self.fresh = fresh

    def __deinit__(deinit self):
        _ = self.bufs^
        _ = self.ctx^


struct FittedResidentStore(Movable):
    var entries: Dict[Int, FittedResidentEntry]
    var live: Dict[Int, Int]
    var next_lease: Int
    var tick: Int
    var bytes: Int
    var cap: Int
    var uploads: Int
    # One lock word (see the module note).
    var lock: List[Int64]

    def __init__(out self):
        self.entries = Dict[Int, FittedResidentEntry]()
        self.live = Dict[Int, Int]()
        self.next_lease = 1
        self.tick = 0
        self.bytes = 0
        self.cap = 0
        self.uploads = 0
        self.lock = List[Int64](length=1, fill=Int64(0))


comptime _StorePtr = UnsafePointer[FittedResidentStore, MutAnyOrigin]


def _store_lock(store: _StorePtr):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the store's lock word (held for one dictionary step only)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _store_unlock(store: _StorePtr):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


def fitted_lease_open(store: _StorePtr) raises -> Int:
    """A new live lease: the integer every later call of this model names."""
    _store_lock(store)
    if store[].next_lease >= (Int.MAX // FITTED_SLOTS_PER_LEASE) - 1:
        _store_unlock(store)
        raise Error("resident fitted-array lease space exhausted")
    var lease = store[].next_lease
    store[].next_lease += 1
    store[].live[lease] = 1
    _store_unlock(store)
    return lease


def _pop_lease_locked(store: _StorePtr, lease: Int) -> List[FittedResidentEntry]:
    """Every entry of `lease`, out of the store (caller holds the lock)."""
    var keys = List[Int]()
    for item in store[].entries.items():  # driver-loop: over resident entries (bounded by the byte cap)
        if item.key // FITTED_SLOTS_PER_LEASE == lease:
            keys.append(item.key)
    var popped = List[FittedResidentEntry]()
    for i in range(len(keys)):  # driver-loop: over this lease's slots
        try:
            var entry = store[].entries.pop(keys[i])
            store[].bytes -= entry.bytes
            popped.append(entry^)
        except:
            pass
    try:
        _ = store[].live.pop(lease)
    except:
        pass
    return popped^


def fitted_lease_close(store: _StorePtr, lease: Int):
    """Free every device copy of `lease`; later calls naming it get per-call
    buffers. Quiet on a lease already closed. Never raises (it runs from a
    native finalizer)."""
    if lease <= 0:
        return
    _store_lock(store)
    var popped = _pop_lease_locked(store, lease)
    _store_unlock(store)
    # Destroyed outside the lock: a predict still running holds its own
    # handle copies of the context and the buffers.
    _ = popped^


def _upload_all(ctx: DeviceContext, srcs: List[Int], counts: List[Int]) raises -> List[DeviceBuffer[DType.float32]]:
    """The same bytes the per-call route uploads, into fresh buffers."""
    var bufs = List[DeviceBuffer[DType.float32]]()
    for i in range(len(srcs)):  # driver-loop: over the model's fitted arrays (a handful)
        var n = counts[i]
        var buf = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        if n > 0:
            if srcs[i] == 0:
                raise Error("resident fitted arrays: null host address for a non-empty array")
            ctx.enqueue_copy(dst_buf=buf, src_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=srcs[i]))
        bufs.append(buf^)
    # The host arrays may change once this returns.
    ctx.synchronize()
    return bufs^


def _same_words(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # driver-loop: over the model's fitted arrays (a handful of address words)
        if a[i] != b[i]:
            return False
    return True


def _slot_state_locked(store: _StorePtr, key: Int, device: Int, srcs: List[Int], counts: List[Int]) -> Int:
    """0: no entry; 1: an entry mirroring exactly these arrays on this
    context; 2: a stale entry (caller holds the lock)."""
    if key not in store[].entries:
        return 0
    try:
        ref entry = store[].entries[key]
        if entry.device == device and _same_words(entry.srcs, srcs) and _same_words(entry.counts, counts):
            return 1
    except:
        return 0
    return 2


def fitted_resident_view(store: _StorePtr, ctx: DeviceContext, lease: Int, slot: Int,
                         srcs: List[Int], counts: List[Int]) raises -> FittedResidentView:
    """The device copies of `srcs` (float32 arrays of `counts` floats) for
    slot `slot` of `lease`: the kept copies when they mirror exactly these
    host arrays on this context, else a fresh upload, kept when the lease is
    live, the slot fits the key space and the bytes fit the cap."""
    if len(srcs) != len(counts):
        raise Error("resident fitted arrays: one count per address")
    var device = ctx_cache_key(ctx)
    var bytes = 0
    for i in range(len(counts)):  # driver-loop: over the model's fitted arrays (a handful of address words)
        if counts[i] < 0:
            raise Error("resident fitted arrays: negative float count")
        bytes += counts[i] * 4
    var keyed = lease > 0 and slot >= 0 and slot < FITTED_SLOTS_PER_LEASE
    var key = lease * FITTED_SLOTS_PER_LEASE + slot if keyed else 0
    var stale = List[FittedResidentEntry]()
    if keyed:
        _store_lock(store)
        store[].tick += 1
        var state = _slot_state_locked(store, key, device, srcs, counts)
        if state == 1:
            try:
                ref entry = store[].entries[key]
                entry.used = store[].tick
                var view = FittedResidentView(entry.ctx.copy(), entry.bufs.copy(), True, False)
                _store_unlock(store)
                return view^
            except:
                pass
        elif state == 2:
            # Another context, or a host array the slot does not mirror:
            # the old copy goes (freed below, outside the lock).
            try:
                var old = store[].entries.pop(key)
                store[].bytes -= old.bytes
                stale.append(old^)
            except:
                pass
        _store_unlock(store)
    _ = stale^
    # The upload runs outside the lock.
    var bufs = _upload_all(ctx, srcs, counts)
    if not keyed:
        return FittedResidentView(ctx.copy(), bufs^, False, True)
    if store[].cap == 0:
        var mem = ctx.get_memory_info()
        var cap = Int(mem[1]) // FITTED_CAP_DIVISOR
        store[].cap = cap if cap > 0 else 1
    var evicted = List[FittedResidentEntry]()
    var kept = False
    _store_lock(store)
    store[].uploads += 1
    if key // FITTED_SLOTS_PER_LEASE in store[].live and bytes <= store[].cap:
        while len(store[].entries) > 0 and store[].bytes + bytes > store[].cap:  # driver-loop: least recently used eviction
            var oldest = -1
            var oldest_used = Int.MAX
            for item in store[].entries.items():  # driver-loop: over resident entries (bounded by the byte cap)
                if item.value.used < oldest_used:
                    oldest_used = item.value.used
                    oldest = item.key
            try:
                var gone = store[].entries.pop(oldest)
                store[].bytes -= gone.bytes
                evicted.append(gone^)
            except:
                break
        if store[].bytes + bytes <= store[].cap and key not in store[].entries:
            store[].entries[key] = FittedResidentEntry(
                ctx.copy(), bufs.copy(), srcs.copy(), counts.copy(), device, bytes, store[].tick
            )
            store[].bytes += bytes
            kept = True
    _store_unlock(store)
    _ = evicted^
    return FittedResidentView(ctx.copy(), bufs^, kept, True)


def fitted_resident_stats(store: _StorePtr) -> List[Int]:
    """[live leases, kept entries, kept bytes, uploads so far, cap bytes]:
    a test-only readout."""
    _store_lock(store)
    var out = List[Int]()
    out.append(len(store[].live))
    out.append(len(store[].entries))
    out.append(store[].bytes)
    out.append(store[].uploads)
    out.append(store[].cap)
    _store_unlock(store)
    return out^
