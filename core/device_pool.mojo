# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Pooled float32 device buffers kept between calls (lane
gap-neural-overhead2, 2026-10-02).

An entry that allocated its large device buffers fresh on every call (the
cross-entropy host entry: logits, dlogits, shift, expo, weights and the
workspace, 256 MB each at the board's 8,192 x 8,192) paid an allocation and
a free per buffer per call; freeing device memory waits on the device and
cost milliseconds per buffer on CUDA (DEVIATION 5718 measured eighteen
frees at 12.3 ms on the RTX 4090). `pool_take` hands out an owned buffer
of EXACTLY `n` floats, moved out of the pool when an idle one fits, else a
new one; `pool_give` moves it back. A caller that raises between the two
simply frees its buffer, as before (no slot is left marked busy). The pool is a `_Global` per name (one per
binding and tier, as the contexts), capped at POOL_KEEP_BYTES of idle
buffers. A taken buffer's words are whatever the last holder left, as a
fresh allocation's are unspecified: callers write before they read.
Storage only: no bit moves."""
from std.ffi import _Global
from max.gpu.host import DeviceBuffer, DeviceContext

comptime POOL_KEEP_BYTES = 2 * 1024 * 1024 * 1024


struct _DevPool(Defaultable, Movable):
    var bufs: List[DeviceBuffer[DType.float32]]
    var bytes: Int

    def __init__(out self):
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.bytes = 0


def pool_take[name: StaticString](ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
    """An idle pooled buffer of exactly `n` floats (at least one), else a
    fresh allocation. Its words are unspecified, as a fresh one's are."""
    comptime SLOT = _Global[StorageType=_DevPool, name=name, init_fn=_DevPool.__init__]
    var p = SLOT.get_or_create_ptr()
    var need = n if n > 0 else 1
    for i in range(len(p[].bufs)):
        if len(p[].bufs[i]) == need:
            p[].bytes -= need * 4
            return p[].bufs.pop(i)
    return ctx.enqueue_create_buffer[DType.float32](need)


def pool_give[name: StaticString](var b: DeviceBuffer[DType.float32]) raises:
    """Return a taken buffer; the caller has waited on every use of it. Over
    POOL_KEEP_BYTES the oldest idle buffers are freed first."""
    comptime SLOT = _Global[StorageType=_DevPool, name=name, init_fn=_DevPool.__init__]
    var p = SLOT.get_or_create_ptr()
    var nb = len(b) * 4
    if nb > POOL_KEEP_BYTES:
        _ = b^
        return
    while len(p[].bufs) > 0 and p[].bytes + nb > POOL_KEEP_BYTES:
        p[].bytes -= len(p[].bufs[0]) * 4
        _ = p[].bufs.pop(0)
    p[].bytes += nb
    p[].bufs.append(b^)
