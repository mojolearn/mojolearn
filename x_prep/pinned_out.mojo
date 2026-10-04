# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MOJOLEARN_X_PREP_PINNED_OUT (lane/apple-fast-w3-prep, 2026-10-04): FAST +
Apple candidate, opt-in, default OFF. IDENTICAL and other vendors compile
none of it.

Today the program's output region (LabelBinarizer taxi: 1M x 259 int32,
972 MB) reaches the caller through core/staged_download.mojo: per 8 MB chunk
a DMA into a pinned stage, then ONE host thread memcpys the stage into a
FRESH anonymous mapping that Python allocated for the call. That memcpy pays
both the pinned-memory read and the first-touch page faults of the fresh
destination (~3 GB/s each, measured on the M4, metal-transfer-costs-on-apple),
and it is most of the remaining label-binarizer taxi time (270 ms, the device
stages are a few ms).

Here the output region goes device -> pinned HostBuffer in ONE DMA (Apple
unified memory: the pinned buffer IS host memory the CPU reads directly) and
the HostBuffer itself becomes the caller's output: Python wraps its address
(no copy, no fresh mapping) and calls `x_prep_pinned_out_free(id)` when the
last view of it dies. Freed buffers stay in an exact-size idle pool (at most
PINNED_OUT_KEEP_BYTES) so a repeated call of the same shape reuses resident
pages. Copies only: the same words land in the caller's array; no bit moves.

Registry ops run while the binding has released the GIL, as core/
device_pool.mojo's pool does; a binding never runs two x_prep programs at
once from one Python thread.
"""
from std.ffi import _Global
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime X_PREP_PINNED_OUT = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                            and is_defined["MOJOLEARN_X_PREP_PINNED_OUT"]())
#: idle pinned bytes kept for reuse (a 972 MB output fits twice)
comptime PINNED_OUT_KEEP_BYTES = 2 * 1024 * 1024 * 1024


struct _PinnedOutSlots(Defaultable, Movable):
    """Live outputs (owned by a Python view, by id) and idle pooled ones."""
    var live: List[HostBuffer[DType.float32]]
    var ids: List[Int]
    var idle: List[HostBuffer[DType.float32]]
    var idle_bytes: Int
    var next_id: Int

    def __init__(out self):
        self.live = List[HostBuffer[DType.float32]]()
        self.ids = List[Int]()
        self.idle = List[HostBuffer[DType.float32]]()
        self.idle_bytes = 0
        self.next_id = 1


comptime PINNED_OUT = _Global[StorageType=_PinnedOutSlots, name="MojoXPrepPinnedOutFast", init_fn=_PinnedOutSlots.__init__]


def pinned_out_download(ctx: DeviceContext, mut df: DeviceBuffer[DType.float32], out_at: Int, out_n: Int,
                      receipt_addr: Int) raises:
    """`df[out_at, out_at + out_n)` into a pinned HostBuffer of exactly
    out_n floats (an idle pooled one when one fits, else a new one), one DMA,
    waited on here. Writes Int64 [id, host address] at receipt_addr; the
    buffer stays alive until `pinned_out_free(id)`."""
    var p = PINNED_OUT.get_or_create_ptr()
    var need = out_n if out_n > 0 else 1
    var found = -1
    for i in range(len(p[].idle)):
        if len(p[].idle[i]) == need:
            found = i
            break
    var hb: HostBuffer[DType.float32]
    if found >= 0:
        p[].idle_bytes -= need * 4
        hb = p[].idle.pop(found)
    else:
        hb = ctx.enqueue_create_host_buffer[DType.float32](need)
    if out_n > 0:
        var view = df.create_sub_buffer[DType.float32](out_at, out_n)
        ctx.enqueue_copy(dst_buf=hb, src_buf=view)
        ctx.synchronize()
        _ = view^
    else:
        ctx.synchronize()
    var id = p[].next_id
    p[].next_id += 1
    var r = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=receipt_addr)
    r.unsafe_store(0, Int64(id))
    r.unsafe_store(1, Int64(Int(hb.unsafe_ptr())))
    p[].ids.append(id)
    p[].live.append(hb^)


def pinned_out_free(id: Int) raises -> Bool:
    """The caller's last view of output `id` died: back to the idle pool
    (oldest idle freed first above PINNED_OUT_KEEP_BYTES). False when `id` is
    not live (a double free is refused, never applied twice)."""
    var p = PINNED_OUT.get_or_create_ptr()
    var at = -1
    for i in range(len(p[].ids)):
        if p[].ids[i] == id:
            at = i
            break
    if at < 0:
        return False
    _ = p[].ids.pop(at)
    var hb = p[].live.pop(at)
    var nb = len(hb) * 4
    if nb > PINNED_OUT_KEEP_BYTES:
        _ = hb^
        return True
    while len(p[].idle) > 0 and p[].idle_bytes + nb > PINNED_OUT_KEEP_BYTES:
        p[].idle_bytes -= len(p[].idle[0]) * 4
        _ = p[].idle.pop(0)
    p[].idle_bytes += nb
    p[].idle.append(hb^)
    return True


def pinned_out_live() raises -> Int:
    """Live (caller-held) outputs, for checkers."""
    return len(PINNED_OUT.get_or_create_ptr()[].ids)
