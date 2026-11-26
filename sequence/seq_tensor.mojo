# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RESIDENT SEQUENCE TENSORS: float32 arrays that live on the shared
sequence context (`sequence_ctx`) between binding calls (lane
gap-neural-io, 2026-10-08; plan plan gaps-2026-10-08 (git history) Section 4;
device binding only).

The board's torch columns for the optimizer and LayerNorm lanes are
kernel-only clocks with every tensor already on the device; ours copied the
caller's host arrays inside every call (an optimizer step moved P and G up
and P down, 3 x 67 MB at 16,777,216 floats, around ~1 ms of kernels; a
LayerNorm fit moved five 67 MB arrays). A Python `SequenceDeviceTensor`
(`python/mojolearn/_x_sequence_device.py`) holds one handle of this pool:
`seq_tensor_alloc` once, `seq_tensor_upload` / `seq_tensor_download` when
the caller moves data, and the device forms of the entries
(`opt_resident_step_dev_py`, `lamb_resident_step_dev_py`,
`adafactor_resident_step_dev_py`, `layer_norm_dev_py`) read and write the
buffers in place. The kernels are the per-call entries' launches on the same
values: no bit moves. Storage is `std.ffi._Global`, one pool per numeric
tier (as `_ResPool`); a freed handle's slot is reused. The Python entries
(`seq_tensor_alloc_py` ...) are `providers/python/sequence_seq_tensor_pyapi.mojo`'s; this
file takes typed handles and addresses only."""
from max.gpu.host import DeviceBuffer

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.exec_device import sequence_ctx
from sequence.ops import FP
from core.shared_slot import SharedSlot
from core.nan_canon import NAN_CANON_ON, nan_canon_f32


struct _SeqTensorPool(Defaultable, Movable):
    #: one buffer per handle (a 1-float placeholder for a free handle)
    var bufs: List[DeviceBuffer[DType.float32]]
    #: floats per handle (0 marks a free handle)
    var n: List[Int]
    #: per handle: a kernel may have written its cells (THE COMPUTED WORD, below)
    var computed: List[Bool]

    def __init__(out self):
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.n = List[Int]()
        self.computed = List[Bool]()


# THE COMPUTED WORD (lane gs-fix-4, 2026-11-26). The download made every NaN word canonical (core/nan_canon.mojo:
# a COMPUTED NaN is a different word on each vendor). It did so for cells no kernel ever wrote as well: a caller's
# array put by `to_device` and read back came home with its NaN payloads and signaling bits rewritten, where the
# CPU column (no resident tensors: `to_device` hands the array back) returns the caller's words. So the same call
# gave different bits on the host column and on a GPU, and every NumPy cast of the read-back array differed
# (test_sequence_{array,object,scalar_cast,string}_* on the L40S). A handle's cells are host words, moved by byte
# copies only, until a kernel is handed its device address: `seq_tensor_ptr` / `seq_tensor_view` are the only two
# ways out of this file, and each sets the word. Upload over the whole tensor clears it, a device copy carries the
# source's. The download canonicalizes exactly the handles whose word is set: computed results are unchanged,
# uploaded words are the same bytes on every column. Per call: one Bool store or load, never a pass over cells.
# (V2: the pool gained a field; an image of another layout must not share the slot.)
comptime _SEQ_TENSOR_NAME = "MojolearnXSequenceTensorsIdenticalV2" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnXSequenceTensorsFastV2"
comptime _SEQ_TENSORS = SharedSlot[_SeqTensorPool, _SEQ_TENSOR_NAME]


def _tensor(h: Int) raises -> Int:
    """Checks handle h is open; returns its float count."""
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    if h < 1 or h > len(pool[].n) or pool[].n[h - 1] == 0:
        raise Error("seq_tensor: handle " + String(h) + " is not open")
    return pool[].n[h - 1]


def seq_tensor_ptr(h: Int, n: Int, what: String) raises -> FP:
    """The device address of open handle `h`, which must hold exactly
    n floats (the entry's shape check)."""
    var have = _tensor(h)
    if have != n:
        raise Error("seq_tensor: " + what + " holds " + String(have) + " floats, the call needs " + String(n))
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    # a kernel is handed the cells: they may be computed from here on (THE COMPUTED WORD)
    pool[].computed[h - 1] = True
    return FP(unsafe_from_address=Int(pool[].bufs[h - 1].unsafe_ptr()))


def seq_tensor_alloc(n: Int, zero: Bool) raises -> Int:
    """A new resident tensor of n >= 1 floats; zero fills it with +0.0
    on the device (else its contents are unspecified until written).
    Returns the handle (>= 1)."""
    if n < 1:
        raise Error("seq_tensor_alloc: n must be >= 1")
    var ctx = sequence_ctx()
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    if zero:
        buf.enqueue_fill(Float32(0.0))
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    var h = -1
    for j in range(len(pool[].n)):  # small-loop(pool: resident tensor handles): free handle slot search, no data
        if pool[].n[j] == 0 and h < 0:
            h = j
    if h < 0:
        pool[].bufs.append(buf^)
        pool[].n.append(n)
        pool[].computed.append(False)
        h = len(pool[].n) - 1
    else:
        pool[].bufs[h] = buf^
        pool[].n[h] = n
        pool[].computed[h] = False
    # the fill is ordered before every later use on the one in-order
    # context, so nothing here waits (lane fb15-sync-counts, 2026-11-18: the
    # wait cost one cuStreamSynchronize per tensor: 3 per device LayerNorm
    # forward (y, mean, rstd), 1 per `to_device`, whose upload waits anyway).
    # An allocation failure raises from `enqueue_create_buffer` itself; a
    # failed fill surfaces at the next wait on the context (every read of a
    # resident tensor waits: download, copy, free).
    return h + 1


def seq_tensor_free(h: Int) raises -> Int:
    """Release a handle (after the queue drains: a queued launch may still
    read its buffer). Returns h."""
    _ = _tensor(h)
    var ctx = sequence_ctx()
    ctx.synchronize()
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    pool[].bufs[h - 1] = ctx.enqueue_create_buffer[DType.float32](1)
    pool[].n[h - 1] = 0
    pool[].computed[h - 1] = False
    return h


def seq_tensor_upload(h: Int, a: Int, n: Int) raises -> Int:
    """The caller's n floats at address `a` into the handle's first n
    cells, then a wait (the caller's array may change once this returns).
    Returns n."""
    var have = _tensor(h)
    if n < 1 or n > have:
        raise Error("seq_tensor_upload: n must lie in [1, " + String(have) + "]")
    if a == 0:
        raise Error("seq_tensor_upload: null host buffer")
    var ctx = sequence_ctx()
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    var view = pool[].bufs[h - 1].create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_buf=view, src_ptr=FP(unsafe_from_address=a))
    ctx.synchronize()
    _ = view^
    if n == have:
        # every cell is the caller's word again (THE COMPUTED WORD)
        pool[].computed[h - 1] = False
    return n


def seq_tensor_download(h: Int, a: Int, n: Int) raises -> Int:
    """The handle's first n cells into the caller's array at address `a`,
    after every queued launch (one wait). Returns n."""
    var have = _tensor(h)
    if n < 1 or n > have:
        raise Error("seq_tensor_download: n must lie in [1, " + String(have) + "]")
    if a == 0:
        raise Error("seq_tensor_download: null host buffer")
    var ctx = sequence_ctx()
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    var view = pool[].bufs[h - 1].create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=a), src_buf=view)
    ctx.synchronize()
    _ = view^
    # A resident tensor holds float32 cells a model computed, and this is
    # where they leave the device: NAN_CANON_ON
    # (default on; core/nan_canon.mojo) makes every NaN word the one
    # canonical word, across the host's cores. Under _OFF, this is nothing.
    # Only where a kernel may have written the cells (THE COMPUTED WORD): words a caller uploaded and no kernel
    # was handed come back as the bytes that went up, as on the CPU column.
    comptime if NAN_CANON_ON:
        if pool[].computed[h - 1]:
            _ = nan_canon_f32(FP(unsafe_from_address=a), n)
    return n


def seq_tensor_copy(hd: Int, hs: Int, n: Int) raises -> Int:
    """Device to device: handle hs's first n cells into hd's (a `copy()`
    of a resident tensor), on the one in-order context, then a wait.
    Returns n."""
    if n < 1 or n > _tensor(hd) or n > _tensor(hs):
        raise Error("seq_tensor_copy: n must fit both tensors")
    var ctx = sequence_ctx()
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    var vd = pool[].bufs[hd - 1].create_sub_buffer[DType.float32](0, n)
    var vs = pool[].bufs[hs - 1].create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_buf=vd, src_buf=vs)
    ctx.synchronize()
    _ = vd^
    _ = vs^
    # a byte move: the destination holds computed cells when either side did (THE COMPUTED WORD)
    pool[].computed[hd - 1] = pool[].computed[hd - 1] or pool[].computed[hs - 1]
    return n


def seq_tensor_view(h: Int, n: Int, what: String) raises -> DeviceBuffer[DType.float32]:
    """A view of open handle `h`'s n floats (exactly its size), for a
    device-to-device copy."""
    _ = seq_tensor_ptr(h, n, what)
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    return pool[].bufs[h - 1].create_sub_buffer[DType.float32](0, n)
