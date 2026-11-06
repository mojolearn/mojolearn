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

A slot is found by the model's identity word and replaced when the binding
names a new token for it (a refit, or a replaced `params_` / `coefs_` /
`intercepts_`), so a refit frees the old block at the model's next predict.
The store keeps at most `_SLOTS` blocks and `_KEEP_BYTES` bytes; past either
it frees the least recently used one. Both caps are memory reasoning (a
handful of fitted sequence models in one process, their parameters a small
share of any GPU's memory), not any model or dataset shape; a block larger
than `_KEEP_BYTES` is not kept and is uploaded per call as before.

Like the executor's buffer pool (`X_SEQUENCE_POOL`), the store has no lock:
descriptor entries run one at a time per process.
"""
from std.ffi import _Global
from max.gpu.host import DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.ops import FP
from sequence.exec_device import DeviceExec, _pcopy
from sequence.perf3_seq_resident_words import SEQ_RESIDENT_ON_EXEC

comptime _SLOTS = 64
comptime _KEEP_BYTES = 1 << 30


struct _SeqResidentStore(Defaultable, Movable):
    var owners: List[Int]
    var tokens: List[Int]
    var counts: List[Int]
    var used: List[Int]
    var bufs: List[DeviceBuffer[DType.float32]]
    var tick: Int
    var bytes: Int

    def __init__(out self):
        self.owners = List[Int]()
        self.tokens = List[Int]()
        self.counts = List[Int]()
        self.used = List[Int]()
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.tick = 0
        self.bytes = 0


# One store per numeric tier, next to that tier's sequence context (a FAST
# and an IDENTICAL binding in one process never share device memory).
comptime _STORE_NAME = (
    "MojoXSequenceResidentParamsIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojoXSequenceResidentParamsFast"
)
comptime _STORE = _Global[StorageType=_SeqResidentStore, name=_STORE_NAME, init_fn=_SeqResidentStore.__init__]


def _evict(mut ex: DeviceExec, i: Int) raises:
    """Free slot i, after every queued launch that may still read it."""
    var store = _STORE.get_or_create_ptr()
    ex.ctx.synchronize()
    store[].bytes -= store[].counts[i] * 4
    _ = store[].owners.pop(i)
    _ = store[].tokens.pop(i)
    _ = store[].counts.pop(i)
    _ = store[].used.pop(i)
    _ = store[].bufs.pop(i)


def _resident_params(mut ex: DeviceExec, owner: Int, token: Int, src: Int, n: Int) raises -> Int:
    """The device address of the resident copy of the `n` floats at host
    address `src` named (`owner`, `token`); 0 when the block is not kept."""
    var store = _STORE.get_or_create_ptr()
    store[].tick += 1
    for i in range(len(store[].owners)):  # small-loop(resident slots, at most _SLOTS): owner lookup, no data
        if store[].owners[i] == owner:
            if store[].tokens[i] == token and store[].counts[i] == n:
                store[].used[i] = store[].tick
                return Int(store[].bufs[i].unsafe_ptr())
            _evict(ex, i)  # refit or replaced fitted block of this model
            break
    if token <= 0 or n <= 0 or src == 0 or n * 4 > _KEEP_BYTES:
        return 0
    while len(store[].owners) > 0 and (len(store[].owners) >= _SLOTS or store[].bytes + n * 4 > _KEEP_BYTES):
        var oldest = 0
        for i in range(1, len(store[].used)):  # small-loop(resident slots, at most _SLOTS): least recently used
            if store[].used[i] < store[].used[oldest]:
                oldest = i
        _evict(ex, oldest)
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
