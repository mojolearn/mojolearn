# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fixed-order grid folds for `Results.calc_b` (w2-svm, 2026-10-02).

`cub::DeviceReduce::Sum / Min / Max` at the `CalcB` and `SelectReduce`
sites were one-thread serial chains (`serial_sum_f32_kernel`,
`serial_min/max_f32_kernel`). They are grid folds now:

  * THE SUM (DEVIATION 632's order, new spelling). Level 0 cuts the input
    into consecutive chunks of `FOLD_TPB` cells; each block loads its chunk
    (every cell flushed, cells past `n` are +0.0) and folds it by a halving
    tree (`s[t] = ftz(s[t] + s[t + step])`, step FOLD_TPB/2 .. 1). The
    chunk sums are the next level's input, and the levels repeat until one
    chunk remains. The order is a function of `n` alone, so every vendor
    and the host twin (`svm/host/smo_oracle.mojo::_tree_sum`) add the same
    pairs in the same order.
  * MIN / MAX: the same chunk-and-level shape over (value, index) with
    `_arg_better`'s total order (strict on value, the smaller index on a
    tie), so the winner is the first index in compaction order: the old
    serial scan's answer bit for bit, a +0.0/-0.0 tie included (row 39).
    Padding carries +-inf with the largest key.
"""

from std.gpu import block_idx, thread_idx
from std.math import inf
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz
from svm.checks.pinned_argreduce import _arg_better

#: one chunk of a level: the block width of every fold launch
comptime FOLD_TPB = 256

comptime _KEY_PAD = Int32(2147483647)

comptime _FP = MutPointer[Float32, MutAnyOrigin]
comptime _IP = MutPointer[Int32, MutAnyOrigin]


@always_inline
def fold_blocks(n: Int) -> Int:
    """The chunks of one level over `n` cells."""
    return (n + FOLD_TPB - 1) // FOLD_TPB


def tree_sum_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """One level of the fixed-order sum: block `b` folds
    `src[b * FOLD_TPB, (b + 1) * FOLD_TPB)` into `dst[b]`."""
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var i = blk * FOLD_TPB + tid
    var s = stack_allocation[FOLD_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var v = Float32(0.0)
    if i < Int(n_in):
        v = ftz(src.unsafe_load(i))
    s[tid] = v
    barrier()
    var step = FOLD_TPB // 2
    while step > 0:
        if tid < step:
            s[tid] = ftz(s[tid] + s[tid + step])
        barrier()
        step //= 2
    if tid == 0:
        dst.unsafe_store(blk, s[0])


def tree_arg_f32_kernel[MAX: Bool](
    dst_v: MutPointer[Float32, MutAnyOrigin],
    dst_k: MutPointer[Int32, MutAnyOrigin],
    src_v: MutPointer[Float32, MutAnyOrigin],
    src_k: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    first_level: Int32,
):
    """One level of the keyed min (or max): block `b` folds its chunk of
    (value, key) into `dst_v[b], dst_k[b]`. On the first level the key is
    the cell's index and `src_k` is not read."""
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var i = blk * FOLD_TPB + tid
    var s_v = stack_allocation[FOLD_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var s_k = stack_allocation[FOLD_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var v: Float32
    comptime if MAX:
        v = -inf[DType.float32]()
    else:
        v = inf[DType.float32]()
    var k = _KEY_PAD
    if i < Int(n_in):
        v = src_v.unsafe_load(i)
        k = Int32(i) if first_level != 0 else src_k.unsafe_load(i)
    s_v[tid] = v
    s_k[tid] = k
    barrier()
    var step = FOLD_TPB // 2
    while step > 0:
        if tid < step:
            var ov = s_v[tid + step]
            var ok = s_k[tid + step]
            if _arg_better[MAX](ov, ok, s_v[tid], s_k[tid]):
                s_v[tid] = ov
                s_k[tid] = ok
        barrier()
        step //= 2
    if tid == 0:
        dst_v.unsafe_store(blk, s_v[0])
        dst_k.unsafe_store(blk, s_k[0])


struct FoldScratch(Movable):
    """Two ping-pong levels of partials (values and keys), sized for the
    first level over `capacity` cells."""

    var a: DeviceBuffer[DType.float32]
    var b: DeviceBuffer[DType.float32]
    var ka: DeviceBuffer[DType.int32]
    var kb: DeviceBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        var m = fold_blocks(max(1, capacity)) + 1
        self.a = ctx.enqueue_create_buffer[DType.float32](m)
        self.b = ctx.enqueue_create_buffer[DType.float32](m)
        self.ka = ctx.enqueue_create_buffer[DType.int32](m)
        self.kb = ctx.enqueue_create_buffer[DType.int32](m)


def grid_sum_f32(
    ctx: DeviceContext,
    src: DeviceBuffer[DType.float32],
    n: Int,
    mut scratch: FoldScratch,
    dst: DeviceBuffer[DType.float32],
) raises:
    """The fixed-order sum of `src[0, n)` into `dst[0]`, `n >= 1`: one
    grid launch per level, the last level writing `dst`."""
    var cur = rebind[_FP](src.unsafe_ptr())
    var sa = rebind[_FP](scratch.a.unsafe_ptr())
    var sb = rebind[_FP](scratch.b.unsafe_ptr())
    var n_cur = n
    var level = 0
    while True:
        var nb = fold_blocks(n_cur)
        var out = rebind[_FP](dst.unsafe_ptr())
        if nb > 1:
            out = sa if (level & 1) == 0 else sb
        ctx.enqueue_function[tree_sum_f32_kernel](
            out, cur, Int32(n_cur), grid_dim=nb, block_dim=FOLD_TPB,
        )
        if nb == 1:
            break
        cur = out
        n_cur = nb
        level += 1


def grid_arg_f32[MAX: Bool](
    ctx: DeviceContext,
    src: DeviceBuffer[DType.float32],
    n: Int,
    mut scratch: FoldScratch,
    dst: DeviceBuffer[DType.float32],
) raises:
    """The keyed min (or max) of `src[0, n)` into `dst[0]`, `n >= 1`."""
    var sa = rebind[_FP](scratch.a.unsafe_ptr())
    var sb = rebind[_FP](scratch.b.unsafe_ptr())
    var ska = rebind[_IP](scratch.ka.unsafe_ptr())
    var skb = rebind[_IP](scratch.kb.unsafe_ptr())
    var cur_v = rebind[_FP](src.unsafe_ptr())
    var cur_k = ska
    var n_cur = n
    var level = 0
    while True:
        var nb = fold_blocks(n_cur)
        var out_v = sa if (level & 1) == 0 else sb
        var out_k = ska if (level & 1) == 0 else skb
        if nb == 1:
            out_v = rebind[_FP](dst.unsafe_ptr())
        ctx.enqueue_function[tree_arg_f32_kernel[MAX]](
            out_v, out_k, cur_v, cur_k, Int32(n_cur), Int32(1 if level == 0 else 0),
            grid_dim=nb, block_dim=FOLD_TPB,
        )
        if nb == 1:
            break
        cur_v = out_v
        cur_k = out_k
        n_cur = nb
        level += 1
