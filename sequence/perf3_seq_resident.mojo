# SPDX-License-Identifier: Apache-2.0
"""Device-resident fitted parameters of the recurrent and MLP models (lane
perf3-seq-neural, 2026-11-06): the device side. The descriptor words are in
`sequence/perf3_seq_resident_words.mojo`.

Before, every `predict`, `predict_proba` and `hidden_sequence` allocated a
device block and uploaded the whole fitted parameter vector to it, then
dropped it at the end of the call. Here the block of each fitted model is
uploaded once, on its first predict, into a buffer of the shared sequence
context (the context every `DeviceExec` runs on, `sequence_ctx`), and later
predicts launch on it directly. The copy is the same bytes, so no result
changes.

A slot is found by the model's identity word and the device index of the
executor's context, and replaced when the binding names a new token for it
(a refit, a replaced `params_` / `coefs_` / `intercepts_`, or an in-place
edit of their values: the binding keeps a native FNV-1a64 of the fitted
host bytes in its record, `bindings/xseq_neural_native.mojo`
`_xseq_params_stamp`, and mints a new token when it changes), so a stale
device copy is never launched on and a refit frees the old block at the
model's next predict. A slot made on another device is never served (pass 2,
H2 / H8).
The store keeps at most `_SLOTS` blocks and `_KEEP_BYTES` bytes; past either
it frees the least recently used one. Both caps are memory reasoning (a
handful of fitted sequence models in one process, their parameters a small
share of any GPU's memory), not any model or dataset shape; a block larger
than `_KEEP_BYTES` is not kept and is uploaded per call as before.

THREADS AND DEATH (lane ne1-a-cluster-spectral, owed row N172). Every
store read and write takes the store's lock word, so two descriptor entries
on two threads (a GIL released around a device entry) never interleave a
lookup, an upload record and an eviction. A model's death is posted on the
records module's dead-key ring (`records_release`,
bindings/native_model_records.mojo); every call drains it first
(`seq_resident_take_dead`, sequence/perf3_seq_resident_words.mojo) and
frees those models' blocks (every unpinned block when the ring overflowed
since the last drain), after a synchronize of the shared context,
before any lookup, so a block leaves at the model's death instead of
waiting for the least-recently-used eviction, and a new model at a reused
address never meets the old block. `seq_resident_release_owner` frees one
model's block directly (the GPU binding's release door).
A slot whose device address was handed to a predict is PINNED from the
lookup until that predict's launches are queued (`seq_resident_resolve`
returns the pinned owner, the dispatch calls `seq_resident_unpin` after
`neural_model_execute`, sequence/model_device_dispatch.mojo): eviction picks
the least recently used UNPINNED slot (none free: the call takes the
per-call upload), a stale-token replacement of a pinned slot is deferred
(per-call upload for that call), and a posted death of a pinned slot is
posted again for the next drain. Every free synchronizes the shared context
first, so launches queued before the unpin finish before the block goes.
"""
from std.ffi import _Global
from std.atomic import Atomic, Ordering
from max.gpu.host import DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.ops import FP
from sequence.exec_device import DeviceExec, _pcopy
from sequence.perf3_seq_resident_words import SEQ_RESIDENT_ON_EXEC, seq_resident_take_dead, seq_resident_owner_died

comptime _SLOTS = 64
comptime _KEEP_BYTES = 1 << 30


struct _SeqResidentStore(Defaultable, Movable):
    var owners: List[Int]
    var devices: List[Int]
    var tokens: List[Int]
    var counts: List[Int]
    var used: List[Int]
    # Per-slot pin counts: a predict that was handed a slot's device address
    # holds a pin from the lookup until its launches are queued
    # (`seq_resident_unpin`); no eviction, replacement or death drain frees a
    # pinned slot.
    var pins: List[Int]
    var bufs: List[DeviceBuffer[DType.float32]]
    var tick: Int
    var bytes: Int
    # One lock word (see THREADS AND DEATH in the module note).
    var lock: List[Int64]

    def __init__(out self):
        self.owners = List[Int]()
        self.devices = List[Int]()
        self.tokens = List[Int]()
        self.counts = List[Int]()
        self.used = List[Int]()
        self.pins = List[Int]()
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.tick = 0
        self.bytes = 0
        self.lock = List[Int64](length=1, fill=Int64(0))


# One store per numeric tier, next to that tier's sequence context (a FAST
# and an IDENTICAL binding in one process never share device memory).
comptime _STORE_NAME = (
    "MojoXSequenceResidentParamsIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojoXSequenceResidentParamsFastV3"
)
comptime _STORE = _Global[StorageType=_SeqResidentStore, name=_STORE_NAME, init_fn=_SeqResidentStore.__init__]


def _store() raises -> UnsafePointer[_SeqResidentStore, MutAnyOrigin]:
    return _STORE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _store_lock(store: UnsafePointer[_SeqResidentStore, MutAnyOrigin]):
    # Non-raising (lesson 6).
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the store's lock word (held for one lookup or eviction)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _store_unlock(store: UnsafePointer[_SeqResidentStore, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


def _evict_locked(store: UnsafePointer[_SeqResidentStore, MutAnyOrigin], mut ex: DeviceExec, i: Int) raises:
    """Free slot i, after every queued launch that may still read it (the
    caller holds the lock and has checked the slot is not pinned)."""
    if store[].pins[i] > 0:
        raise Error("sequence resident store: a pinned slot cannot be freed")
    ex.ctx.synchronize()
    store[].bytes -= store[].counts[i] * 4
    _ = store[].owners.pop(i)
    _ = store[].devices.pop(i)
    _ = store[].tokens.pop(i)
    _ = store[].counts.pop(i)
    _ = store[].used.pop(i)
    _ = store[].pins.pop(i)
    _ = store[].bufs.pop(i)


def _drop_owner_locked(store: UnsafePointer[_SeqResidentStore, MutAnyOrigin], mut ex: DeviceExec,
                       owner: Int) raises:
    for i in range(len(store[].owners)):  # driver-loop: over resident fitted-model slots (at most _SLOTS), owner lookup
        if store[].owners[i] == owner:
            if store[].pins[i] > 0:
                # A predict is between its lookup and its launch: keep the
                # slot and post the death again for the next drain.
                seq_resident_owner_died(owner)
                return
            _evict_locked(store, ex, i)
            return


def _drain_dead_locked(store: UnsafePointer[_SeqResidentStore, MutAnyOrigin], mut ex: DeviceExec) raises:
    """Free the blocks of every model whose death was posted; every unpinned
    block when the records ring overflowed since the last drain."""
    var forget_all = False
    var dead = seq_resident_take_dead(forget_all)
    if forget_all:
        var i = len(store[].owners) - 1
        while i >= 0:  # driver-loop: over resident fitted-model slots (at most _SLOTS), newest first so pops keep indices
            if store[].pins[i] == 0:
                _evict_locked(store, ex, i)
            i -= 1
    for j in range(len(dead)):  # driver-loop: over posted model deaths
        _drop_owner_locked(store, ex, dead[j])


def seq_resident_release_owner(owner: Int) raises:
    """Free the resident block of model `owner`, if any (and every posted
    death); quiet when it holds none."""
    var store = _store()
    var ex = DeviceExec()
    _store_lock(store)
    try:
        _drain_dead_locked(store, ex)
        _drop_owner_locked(store, ex, owner)
        _store_unlock(store)
    except error:
        _store_unlock(store)
        raise error^


def seq_resident_slots() raises -> Int:
    """The number of resident blocks (tests only)."""
    var store = _store()
    _store_lock(store)
    var n = len(store[].owners)
    _store_unlock(store)
    return n


def _resident_params(mut ex: DeviceExec, owner: Int, token: Int, src: Int, n: Int) raises -> Int:
    """`_resident_params_locked` under the store's lock."""
    var store = _store()
    _store_lock(store)
    try:
        _drain_dead_locked(store, ex)
        var address = _resident_params_locked(store, ex, owner, token, src, n)
        _store_unlock(store)
        return address
    except error:
        _store_unlock(store)
        raise error^


def _resident_params_locked(store: UnsafePointer[_SeqResidentStore, MutAnyOrigin], mut ex: DeviceExec, owner: Int,
                            token: Int, src: Int, n: Int) raises -> Int:
    """The device address of the resident copy of the `n` floats at host
    address `src` named (`owner`, `token`); 0 when the block is not kept."""
    store[].tick += 1
    var device = Int(ex.ctx.id())
    for i in range(len(store[].owners)):  # driver-loop: over resident fitted-model slots (at most _SLOTS), owner lookup
        if store[].owners[i] == owner:
            if store[].tokens[i] == token and store[].counts[i] == n and store[].devices[i] == device:
                store[].used[i] = store[].tick
                store[].pins[i] += 1  # released by seq_resident_unpin after the launch is queued
                return Int(store[].bufs[i].unsafe_ptr())
            if store[].pins[i] > 0:
                # Another thread's predict still reads the old block: this
                # call takes the per-call upload, the slot is replaced later.
                return 0
            _evict_locked(store, ex, i)  # refit, replaced or edited fitted block of this model, or another device
            break
    if token <= 0 or n <= 0 or src == 0 or n * 4 > _KEEP_BYTES:
        return 0
    while len(store[].owners) > 0 and (len(store[].owners) >= _SLOTS or store[].bytes + n * 4 > _KEEP_BYTES):
        var oldest = -1
        for i in range(len(store[].used)):  # driver-loop: over resident fitted-model slots (at most _SLOTS), least recently used unpinned
            if store[].pins[i] == 0 and (oldest < 0 or store[].used[i] < store[].used[oldest]):
                oldest = i
        if oldest < 0:
            # Every slot is pinned by a running predict: keep none for this
            # call (the per-call upload), free nothing.
            return 0
        _evict_locked(store, ex, oldest)
    var buf = ex.ctx.enqueue_create_buffer[DType.float32](n)
    # The executor's own upload route: the caller's floats are copied into a
    # staging buffer now (the host array may change once this returns), the
    # device copy is queued ahead of the launches on the same queue, and the
    # staging buffer lives until the executor's next sync.
    var host = ex.ctx.enqueue_create_host_buffer[DType.float32](n)
    _pcopy(FP(unsafe_from_address=Int(host.unsafe_ptr())), FP(unsafe_from_address=src), n)
    ex.ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ex.staged.append(host^)
    var address = Int(buf.unsafe_ptr())
    store[].owners.append(owner)
    store[].devices.append(device)
    store[].tokens.append(token)
    store[].counts.append(n)
    store[].used.append(store[].tick)
    store[].pins.append(1)  # released by seq_resident_unpin after the launch is queued
    store[].bytes += n * 4
    # `store` carries MutAnyOrigin, which may alias every local, so a deref of
    # it in the same expression as `buf^` (or after it) keeps `buf` alive past
    # its move ("cannot be consumed, because it is used later"). The move goes
    # through an untracked pointer to the list, as the last use of the store.
    var bufs = UnsafePointer(to=store[].bufs).unsafe_origin_cast[MutUntrackedOrigin]()
    bufs[].append(buf^)
    return address


def seq_resident_unpin(owner: Int):
    """Drop the pin `seq_resident_resolve` took for `owner` (0: nothing was
    pinned). Called once the predict's launches are queued, on success and
    on error; a free after this synchronizes the shared context first
    (`_evict_locked`), so the queued launches finish before the block goes.
    Non-raising."""
    if owner == 0:
        return
    try:
        var store = _store()
        _store_lock(store)
        for i in range(len(store[].owners)):  # driver-loop: over resident fitted-model slots (at most _SLOTS), owner lookup
            if store[].owners[i] == owner:
                if store[].pins[i] > 0:
                    store[].pins[i] -= 1
                break
        _store_unlock(store)
    except:
        pass


def seq_resident_resolve(mut ex: DeviceExec, mut addresses: List[Int], base: Int) raises -> Int:
    """A predict's address list (`base` own words, params at index 1): a
    named block (`base + 3` words) becomes `base` words with params replaced
    by its resident device copy plus the marker word; an unnamed one is left
    as it is. A list that already carries the marker is refused: only this
    dispatch may say params is device memory.

    Returns the owner word whose slot is now PINNED (0 when no device
    address was handed out). The caller must call `seq_resident_unpin` with
    it after the launches that read the address are queued, on every path."""
    var n = len(addresses)
    if n == base + 1:
        raise Error("native model: invalid predict address descriptor")
    if n != base + 3:
        return 0
    var owner = addresses[base]
    var token = addresses[base + 1]
    var count = addresses[base + 2]
    var resident = _resident_params(ex, owner, token, addresses[1], count)
    while len(addresses) > base:  # small-loop(three descriptor words)
        _ = addresses.pop()
    if resident != 0:
        addresses[1] = resident
        addresses.append(SEQ_RESIDENT_ON_EXEC)
        return owner
    return 0
