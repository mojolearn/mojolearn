# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RESIDENT SEQUENCE TENSORS: float32 arrays that live on the shared
sequence context (`sequence_ctx`) between binding calls (lane
gap-neural-io, 2026-10-08; plan docs/plans/gaps-2026-10-08.md Section 4;
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
tier (as `_ResPool`); a freed handle's slot is reused."""
from std.ffi import _Global
from std.python import PythonObject
from max.gpu.host import DeviceBuffer

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.exec_device import sequence_ctx
from sequence.ops import FP


struct _SeqTensorPool(Defaultable, Movable):
    #: one buffer per handle (a 1-float placeholder for a free handle)
    var bufs: List[DeviceBuffer[DType.float32]]
    #: floats per handle (0 marks a free handle)
    var n: List[Int]

    def __init__(out self):
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.n = List[Int]()


comptime _SEQ_TENSOR_NAME = "MojoXSequenceTensorsIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXSequenceTensorsFast"
comptime _SEQ_TENSORS = _Global[StorageType=_SeqTensorPool, name=_SEQ_TENSOR_NAME, init_fn=_SeqTensorPool.__init__]


def _tensor(h: Int) raises -> Int:
    """Checks handle h is open; returns its float count."""
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    if h < 1 or h > len(pool[].n) or pool[].n[h - 1] == 0:
        raise Error("seq_tensor: handle " + String(h) + " is not open")
    return pool[].n[h - 1]


def seq_tensor_ptr(handle: PythonObject, n: Int, what: String) raises -> FP:
    """The device address of open handle `handle`, which must hold exactly
    n floats (the entry's shape check)."""
    var h = Int(py=handle)
    var have = _tensor(h)
    if have != n:
        raise Error("seq_tensor: " + what + " holds " + String(have) + " floats, the call needs " + String(n))
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    return FP(unsafe_from_address=Int(pool[].bufs[h - 1].unsafe_ptr()))


def seq_tensor_alloc_py(n_obj: PythonObject, zero_obj: PythonObject) raises -> PythonObject:
    """A new resident tensor of n >= 1 floats; zero = 1 fills it with +0.0
    on the device (else its contents are unspecified until written).
    Returns the handle (>= 1)."""
    var n = Int(py=n_obj)
    if n < 1:
        raise Error("seq_tensor_alloc: n must be >= 1")
    var ctx = sequence_ctx()
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    if Int(py=zero_obj) != 0:
        buf.enqueue_fill(Float32(0.0))
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    var h = -1
    for j in range(len(pool[].n)):  # small-loop(pool: resident tensor handles): free handle slot search, no data
        if pool[].n[j] == 0 and h < 0:
            h = j
    if h < 0:
        pool[].bufs.append(buf^)
        pool[].n.append(n)
        h = len(pool[].n) - 1
    else:
        pool[].bufs[h] = buf^
        pool[].n[h] = n
    # the fill is ordered before every later use on the one in-order
    # context; the wait makes a failed allocation raise here
    ctx.synchronize()
    return PythonObject(h + 1)


def seq_tensor_free_py(handle: PythonObject) raises -> PythonObject:
    """Release a handle (after the queue drains: a queued launch may still
    read its buffer)."""
    var h = Int(py=handle)
    _ = _tensor(h)
    var ctx = sequence_ctx()
    ctx.synchronize()
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    pool[].bufs[h - 1] = ctx.enqueue_create_buffer[DType.float32](1)
    pool[].n[h - 1] = 0
    return PythonObject(h)


def seq_tensor_upload_py(handle: PythonObject, addr: PythonObject, n_obj: PythonObject) raises -> PythonObject:
    """The caller's n floats at `addr` into the handle's first n cells, then
    a wait (the caller's array may change once this returns)."""
    var h = Int(py=handle)
    var have = _tensor(h)
    var n = Int(py=n_obj)
    if n < 1 or n > have:
        raise Error("seq_tensor_upload: n must lie in [1, " + String(have) + "]")
    var a = Int(py=addr)
    if a == 0:
        raise Error("seq_tensor_upload: null host buffer")
    var ctx = sequence_ctx()
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    var view = pool[].bufs[h - 1].create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_buf=view, src_ptr=FP(unsafe_from_address=a))
    ctx.synchronize()
    _ = view^
    return PythonObject(n)


def seq_tensor_download_py(handle: PythonObject, addr: PythonObject, n_obj: PythonObject) raises -> PythonObject:
    """The handle's first n cells into the caller's array at `addr`, after
    every queued launch (one wait)."""
    var h = Int(py=handle)
    var have = _tensor(h)
    var n = Int(py=n_obj)
    if n < 1 or n > have:
        raise Error("seq_tensor_download: n must lie in [1, " + String(have) + "]")
    var a = Int(py=addr)
    if a == 0:
        raise Error("seq_tensor_download: null host buffer")
    var ctx = sequence_ctx()
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    var view = pool[].bufs[h - 1].create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=a), src_buf=view)
    ctx.synchronize()
    _ = view^
    return PythonObject(n)


def seq_tensor_copy_py(dst: PythonObject, src: PythonObject, n_obj: PythonObject) raises -> PythonObject:
    """Device to device: src's first n cells into dst's (a `copy()` of a
    resident tensor), on the one in-order context, then a wait."""
    var hd = Int(py=dst)
    var hs = Int(py=src)
    var n = Int(py=n_obj)
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
    return PythonObject(n)


def seq_tensor_view(handle: PythonObject, n: Int, what: String) raises -> DeviceBuffer[DType.float32]:
    """A view of open handle `handle`'s n floats (exactly its size), for a
    device-to-device copy."""
    _ = seq_tensor_ptr(handle, n, what)
    var pool = _SEQ_TENSORS.get_or_create_ptr()
    return pool[].bufs[Int(py=handle) - 1].create_sub_buffer[DType.float32](0, n)
