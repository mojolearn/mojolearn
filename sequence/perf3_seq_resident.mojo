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
lookup, an upload record and an eviction. A model's death is posted by its
death callback (`seq_resident_owner_died`,
sequence/perf3_seq_resident_words.mojo); every call drains the posts first
and frees those models' blocks, after a synchronize of the shared context,
before any lookup, so a block leaves at the model's death instead of
waiting for the least-recently-used eviction, and a new model at a reused
address never meets the old block. `seq_resident_release_owner` frees one
model's block directly (the GPU binding's release door).
"""
from std.ffi import _Global
from std.atomic import Atomic, Ordering
from max.gpu.host import DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.ops import FP
from sequence.exec_device import DeviceExec, _pcopy
from sequence.perf3_seq_resident_words import SEQ_RESIDENT_ON_EXEC, seq_resident_take_dead

comptime _SLOTS = 64
comptime _KEEP_BYTES = 1 << 30


struct _SeqResidentStore(Defaultable, Movable):
    var owners: List[Int]
    var devices: List[Int]
    var tokens: List[Int]
    var counts: List[Int]
    var used: List[Int]
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
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.tick = 0
        self.bytes = 0
        self.lock = List[Int64](length=1, fill=Int64(0))


# One store per numeric tier, next to that tier's sequence context (a FAST
# and an IDENTICAL binding in one process never share device memory).
comptime _STORE_NAME = (
    "MojoXSequenceResidentParamsIdenticalV2" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojoXSequenceResidentParamsFastV2"
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
    caller holds the lock)."""
    ex.ctx.synchronize()
    store[].bytes -= store[].counts[i] * 4
    _ = store[].owners.pop(i)
    _ = store[].devices.pop(i)
    _ = store[].tokens.pop(i)
    _ = store[].counts.pop(i)
    _ = store[].used.pop(i)
    _ = store[].bufs.pop(i)


def _drop_owner_locked(store: UnsafePointer[_SeqResidentStore, MutAnyOrigin], mut ex: DeviceExec,
                       owner: Int) raises:
    for i in range(len(store[].owners)):  # driver-loop: over resident fitted-model slots (at most _SLOTS), owner lookup
        if store[].owners[i] == owner:
            _evict_locked(store, ex, i)
            return


def _drain_dead_locked(store: UnsafePointer[_SeqResidentStore, MutAnyOrigin], mut ex: DeviceExec) raises:
    """Free the blocks of every model whose death was posted."""
    var dead = seq_resident_take_dead()
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
                return Int(store[].bufs[i].unsafe_ptr())
            _evict_locked(store, ex, i)  # refit, replaced or edited fitted block of this model, or another device
            break
    if token <= 0 or n <= 0 or src == 0 or n * 4 > _KEEP_BYTES:
        return 0
    while len(store[].owners) > 0 and (len(store[].owners) >= _SLOTS or store[].bytes + n * 4 > _KEEP_BYTES):
        var oldest = 0
        for i in range(1, len(store[].used)):  # driver-loop: over resident fitted-model slots (at most _SLOTS), least recently used
            if store[].used[i] < store[].used[oldest]:
                oldest = i
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
    store[].bufs.append(buf^)
    store[].bytes += n * 4
    return address


def seq_resident_resolve(mut ex: DeviceExec, mut addresses: List[Int], base: Int) raises:
    """A predict's address list (`base` own words, params at index 1): a
    named block (`base + 3` words) becomes `base` words with params replaced
    by its resident device copy plus the marker word; an unnamed one is left
    as it is. A list that already carries the marker is refused: only this
    dispatch may say params is device memory."""
    var n = len(addresses)
    if n == base + 1:
        raise Error("native model: invalid predict address descriptor")
    if n != base + 3:
        return
    var owner = addresses[base]
    var token = addresses[base + 1]
    var count = addresses[base + 2]
    var resident = _resident_params(ex, owner, token, addresses[1], count)
    while len(addresses) > base:  # small-loop(three descriptor words)
        _ = addresses.pop()
    if resident != 0:
        addresses[1] = resident
        addresses.append(SEQ_RESIDENT_ON_EXEC)
