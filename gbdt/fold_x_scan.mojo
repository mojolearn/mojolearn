# SPDX-License-Identifier: Apache-2.0
"""Word tests over THE FOLD X WORD's slot on the device (lane vdf2-tail2, 2026-11-24).

A model-selection fold taken on the device (an x_prep store slot, core/fold_x_device.mojo) has no host copy, so a
member's refusal that scans host X words runs here as a device reduction over the slot itself (no copy of X):

  * `fold_x_first_nan`: the first NaN word of the slot, or -1 (the gbdt member's nan_mode='Forbidden' refusal,
    core/gbdt_member.mojo `_has_nan`: a NaN refuses; an infinity does not). One thread per word; an atomic min keeps
    the LOWEST hit, so the answer is the host scan's first NaN in word order whatever order the threads run in.

The slot's words are complete before the member holds its id (`take_rows` synchronized the store's context), and
the scan waits for its own context before it answers. One int32 word comes home per call; X never does. Parallel
over words; nothing serial over data.
"""
from std.atomic import Atomic
from std.math import ceildiv
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from core.device_zero import enqueue_fill
from core.fold_x_device import FOLD_X_STORE
from core.fold_x_word import fold_x_slot

comptime _TPB = 256
comptime _NONE = Int32(2147483647)
comptime _FP = MutPointer[Float32, MutAnyOrigin]
comptime _IP = MutPointer[Int32, MutAnyOrigin]


def fx_nan_word_kernel(x: _FP, first: _IP, count: Int32):
    """first[0] = min(first[0], i) for every word i < count that is NaN."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(count):
        return
    var v = x.unsafe_load(i)
    if v != v:
        _ = Atomic.min(first, Int32(i))


def fold_x_first_nan(ctx: DeviceContext, x_word: Int, count: Int) raises -> Int:
    """The first NaN of the first `count` words of the fold slot the negative `x_word` names, on `ctx`, or -1."""
    if x_word >= 0:
        raise Error("fold_x_first_nan: the X word names no fold slot")
    if count <= 0:
        return -1
    if count > 2147483647:
        raise Error("fold_x_first_nan: the scan exceeds the Int32 index bound")
    var store = FOLD_X_STORE.get_or_create_ptr()
    var x = store[].ptr(fold_x_slot(x_word), count)
    var dfirst = ctx.enqueue_create_buffer[DType.int32](1)
    enqueue_fill(ctx, dfirst, _NONE)
    ctx.enqueue_function[fx_nan_word_kernel](
        x, _IP(unsafe_from_address=Int(dfirst.unsafe_ptr())), Int32(count), grid_dim=ceildiv(count, _TPB),
        block_dim=_TPB,
    )
    var hfirst = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=hfirst, src_buf=dfirst)
    ctx.synchronize()
    var f = hfirst.unsafe_ptr().unsafe_load(0)
    _ = hfirst^
    _ = dfirst^
    return -1 if f == _NONE else Int(f)
