# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CalibratedClassifierCV's prediction epilogue as one program
(lane perf-sweep-trees, 2026-11-05).

Before this `predict_proba` ran, per (member, calibrators) pair, a zero fill,
one or two score accumulations (a column gather first for a binary member
with a probability block), one Platt or isotonic apply per class column, the
complement or the row normalization, a float32 narrowing on the host and the
accumulation into the ensemble's mean, then one scale: each glue op its own
upload, launch and readback, about twenty round trips for a three-class
model with one pair. Here every pair's member scores go up once, the same
kernels run on device buffers in the same order with the same operands, and
the mean comes back once (one synchronization).

Bits: every cell is the op sequence it was, through the same kernels. The
one step that used to run on the host, the float32 narrowing of a pair's
probabilities before they are added (`trees_f32` of the float64 block), is
`_narrow_word` here: round-to-nearest-even through `sf64_to_f32` for every
non-NaN value (the host cast's IEEE result), and the host cast's NaN rule
(sign kept, quiet bit set, payload truncated) for a NaN, so the stored bits
are the host cast's. The host binding (`XTREES_DEVICE_OPS` off) runs the
host ops of `xtrees/ops.mojo` in the same order."""
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.soft_f64 import SF64_ONE, sf64_add, sf64_div, sf64_from_f32, sf64_from_int, sf64_mul, sf64_to_f32
from xtrees.ops_device import OPS_TPB, _blocks, _ctx
from xtrees.ops_device_elem import (
    accumulate_cols_kernel, accumulate_kernel, complement_pairs_kernel, normalize_rows_kernel, scale_kernel,
)
from xtrees.ops_device_boost import isotonic_predict_kernel, platt_apply_kernel
from xtrees.ops import (
    accumulate, accumulate_cols, complement_pairs, isotonic_predict_strided, normalize_rows,
    platt_apply_strided, scale_f64,
)


comptime CAL_SIGMOID = 0
comptime CAL_ISOTONIC = 1


@fieldwise_init
struct CalPair(Copyable, Movable):
    """One fitted (member, calibrators) pair of a prediction: the member's
    float32 score block (`blk`, n x bc, row-major) and its c calibrators."""
    var blk: Int
    var bc: Int
    #: k == 2 with bc > 1: the block column of class code 1.
    var at: Int
    #: k > 2: the member's class codes in the ensemble (int32[ks]).
    var cols: Int
    var ks: Int
    var kinds: List[Int]
    var a: List[Float64]
    var b: List[Float64]
    var kx: List[Int]
    var ky: List[Int]
    var m: List[Int]


def _w(x: Float64) -> UInt64:
    return bitcast[DType.uint64](x)


@always_inline
def _narrow_word(w: UInt64) -> Float32:
    """`Float32(Float64)` as the host casts it (see the module note)."""
    if (w & UInt64(0x7FF0000000000000)) == UInt64(0x7FF0000000000000) and (w & UInt64(0x000FFFFFFFFFFFFF)) != 0:
        var bits = ((w >> 63) << 31) | UInt64(0x7FC00000) | ((w >> 29) & UInt64(0x007FFFFF))
        return bitcast[DType.float32](UInt32(bits))
    return sf64_to_f32(w)


def narrow_accumulate_kernel(acc: MutPointer[UInt64, MutAnyOrigin], p: MutPointer[UInt64, MutAnyOrigin], n: Int64,
                             w: UInt64):
    """acc[i] = acc[i] + w * float64(float32(p[i])): `accumulate_kernel` on
    the host-narrowed float32 of p (two roundings, never fused)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        acc.unsafe_store(i, sf64_add(acc.unsafe_load(i), sf64_mul(w, sf64_from_f32(_narrow_word(p.unsafe_load(i))))))
        i += stride


def column_f32_kernel(src: MutPointer[Float32, MutAnyOrigin], n: Int64, c: Int64, j: Int64,
                      dst: MutPointer[Float32, MutAnyOrigin]):
    """dst[r] = src[r, j] (a copy: `_one_col`'s gather of one column)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        dst.unsafe_store(i, src.unsafe_load(i * Int(c) + Int(j)))
        i += stride


def _check_pair(pair: CalPair, k: Int) raises:
    """The refusals the per-op bindings raised, before any launch: the
    knot count of an isotonic calibrator, `accumulate_cols`' column range."""
    for j in range(len(pair.kinds)):  # driver-loop: over the class columns (one calibrator each)
        if pair.kinds[j] != CAL_SIGMOID and pair.m[j] < 1:
            raise Error("x_trees_isotonic_predict_strided: no knots")
    if k == 2:
        return
    var cols = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=pair.cols)
    for c in range(pair.ks):  # driver-loop: over the member's class codes
        var j = Int(cols[unsafe_offset=c])
        if j < 0 or j >= k:
            raise Error("x_trees accumulate_cols: column out of range")


def calibrated_proba_device(pairs: List[CalPair], n: Int, k: Int, acc: MutPointer[Float64, MutUntrackedOrigin]) raises:
    """acc (n x k float64) = the mean over `pairs` of each pair's calibrated
    probabilities, every pair narrowed to float32 before it is added."""
    if n <= 0 or k <= 0:
        return
    var c = 1 if k == 2 else k
    var w = 2 if k == 2 else k
    for p in range(len(pairs)):  # driver-loop: over the (member, calibrators) pairs
        _check_pair(pairs[p], k)
    var ctx = _ctx()
    var keep_f32 = List[DeviceBuffer[DType.float32]]()
    var keep_i32 = List[DeviceBuffer[DType.int32]]()
    var keep_u64 = List[DeviceBuffer[DType.uint64]]()
    var d_acc = ctx.enqueue_create_buffer[DType.uint64](max(1, n * k))
    d_acc.enqueue_fill(UInt64(0))
    var one = _w(1.0)
    try:
        for p in range(len(pairs)):  # driver-loop: over the (member, calibrators) pairs
            ref pair = pairs[p]
            var d_blk = ctx.enqueue_create_buffer[DType.float32](max(1, n * pair.bc))
            ctx.enqueue_copy(dst_buf=d_blk, src_ptr=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=pair.blk))
            # The scores S (n x c): `_cal_scores`.
            var d_s = ctx.enqueue_create_buffer[DType.uint64](max(1, n * c))
            d_s.enqueue_fill(UInt64(0))
            if k == 2:
                if pair.bc == 1:
                    ctx.enqueue_function[accumulate_kernel](
                        d_s.unsafe_ptr(), d_blk.unsafe_ptr(), Int64(n), one, grid_dim=_blocks(n), block_dim=OPS_TPB,
                    )
                else:
                    var d_col = ctx.enqueue_create_buffer[DType.float32](max(1, n))
                    ctx.enqueue_function[column_f32_kernel](
                        d_blk.unsafe_ptr(), Int64(n), Int64(pair.bc), Int64(pair.at), d_col.unsafe_ptr(),
                        grid_dim=_blocks(n), block_dim=OPS_TPB,
                    )
                    ctx.enqueue_function[accumulate_kernel](
                        d_s.unsafe_ptr(), d_col.unsafe_ptr(), Int64(n), one, grid_dim=_blocks(n), block_dim=OPS_TPB,
                    )
                    keep_f32.append(d_col^)
            else:
                var d_cols = ctx.enqueue_create_buffer[DType.int32](max(1, pair.ks))
                if pair.ks > 0:
                    ctx.enqueue_copy(dst_buf=d_cols, src_ptr=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=pair.cols))
                    ctx.enqueue_function[accumulate_cols_kernel](
                        d_s.unsafe_ptr(), d_blk.unsafe_ptr(), d_cols.unsafe_ptr(), Int64(n), Int64(k), Int64(pair.ks), one,
                        grid_dim=_blocks(n), block_dim=OPS_TPB,
                    )
                keep_i32.append(d_cols^)
            # The calibrated probabilities P (n x w): `_calibrated_probs`.
            var d_p = ctx.enqueue_create_buffer[DType.uint64](max(1, n * w))
            d_p.enqueue_fill(UInt64(0))
            for j in range(len(pair.kinds)):  # driver-loop: over the class columns (one calibrator each)
                var col = 1 if k == 2 else j
                if pair.kinds[j] == CAL_SIGMOID:
                    ctx.enqueue_function[platt_apply_kernel](
                        d_s.unsafe_ptr() + j, Int64(c), Int64(n), _w(pair.a[j]), _w(pair.b[j]), d_p.unsafe_ptr() + col,
                        Int64(w), grid_dim=_blocks(n), block_dim=OPS_TPB,
                    )
                else:
                    var m = pair.m[j]
                    var d_kx = ctx.enqueue_create_buffer[DType.uint64](max(1, m))
                    var d_ky = ctx.enqueue_create_buffer[DType.uint64](max(1, m))
                    ctx.enqueue_copy(dst_buf=d_kx, src_ptr=MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=pair.kx[j]))
                    ctx.enqueue_copy(dst_buf=d_ky, src_ptr=MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=pair.ky[j]))
                    ctx.enqueue_function[isotonic_predict_kernel](
                        d_kx.unsafe_ptr(), d_ky.unsafe_ptr(), Int64(m), d_s.unsafe_ptr() + j, Int64(c), Int64(n),
                        d_p.unsafe_ptr() + col, Int64(w), grid_dim=_blocks(n), block_dim=OPS_TPB,
                    )
                    keep_u64.append(d_kx^)
                    keep_u64.append(d_ky^)
            if k == 2:
                ctx.enqueue_function[complement_pairs_kernel](
                    d_p.unsafe_ptr(), Int64(n), grid_dim=_blocks(n), block_dim=OPS_TPB,
                )
            else:
                ctx.enqueue_function[normalize_rows_kernel](
                    d_p.unsafe_ptr(), Int64(n), Int64(k), sf64_div(SF64_ONE, sf64_from_int(k)),
                    grid_dim=_blocks(n), block_dim=OPS_TPB,
                )
            # The ensemble's sum: `_acc_w(acc, float32(P), n * k, 1.0)`.
            ctx.enqueue_function[narrow_accumulate_kernel](
                d_acc.unsafe_ptr(), d_p.unsafe_ptr(), Int64(n * k), one, grid_dim=_blocks(n * k), block_dim=OPS_TPB,
            )
            keep_f32.append(d_blk^)
            keep_u64.append(d_s^)
            keep_u64.append(d_p^)
        # `_scale(acc, len(pairs))`.
        if n * k > 0:
            ctx.enqueue_function[scale_kernel](
                d_acc.unsafe_ptr(), Int64(n * k), _w(Float64(len(pairs))), grid_dim=_blocks(n * k), block_dim=OPS_TPB,
            )
            ctx.enqueue_copy(dst_ptr=acc.bitcast[UInt64](), src_buf=d_acc)
        ctx.synchronize()
    except e:
        ctx.synchronize()
        raise e
    _ = keep_f32^
    _ = keep_i32^
    _ = keep_u64^
    _ = d_acc^


def calibrated_proba_host(pairs: List[CalPair], n: Int, k: Int, acc: MutPointer[Float64, MutUntrackedOrigin]) raises:
    """`calibrated_proba_device` on the host ops (the CPU binding)."""
    if n <= 0 or k <= 0:
        return
    for p in range(len(pairs)):  # driver-loop: over the (member, calibrators) pairs
        _check_pair(pairs[p], k)
    var c = 1 if k == 2 else k
    var w = 2 if k == 2 else k
    for i in range(n * k):  # host-column: the CPU binding's zero fill of the mean
        acc[unsafe_offset=i] = 0.0
    var s = List[Float64](length=max(1, n * c), fill=0.0)
    var prob = List[Float64](length=max(1, n * w), fill=0.0)
    var col32 = List[Float32](length=max(1, n), fill=0.0)
    var p32 = List[Float32](length=max(1, n * k), fill=0.0)
    for p in range(len(pairs)):  # driver-loop: over the (member, calibrators) pairs
        ref pair = pairs[p]
        var blk = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=pair.blk)
        var sp = s.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        var pp = prob.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        for i in range(n * c):  # host-column: the CPU binding's zero fill of the scores
            sp[unsafe_offset=i] = 0.0
        for i in range(n * w):  # host-column: the CPU binding's zero fill of the probabilities
            pp[unsafe_offset=i] = 0.0
        if k == 2:
            if pair.bc == 1:
                accumulate(sp, blk, n, 1.0)
            else:
                var cp = col32.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
                for i in range(n):  # host-column: the CPU binding's column gather
                    cp[unsafe_offset=i] = blk[unsafe_offset=i * pair.bc + pair.at]
                accumulate(sp, cp, n, 1.0)
        else:
            accumulate_cols(sp, blk, MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=pair.cols), n, k, pair.ks,
                            1.0)
        for j in range(len(pair.kinds)):  # driver-loop: over the class columns (one calibrator each)
            var col = 1 if k == 2 else j
            if pair.kinds[j] == CAL_SIGMOID:
                platt_apply_strided(sp + j, c, n, pair.a[j], pair.b[j], pp + col, w)
            else:
                isotonic_predict_strided(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=pair.kx[j]),
                                         MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=pair.ky[j]),
                                         pair.m[j], sp + j, c, n, pp + col, w)
        if k == 2:
            complement_pairs(pp, n)
        else:
            normalize_rows(pp, n, k)
        var qp = p32.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        for i in range(n * k):  # host-column: the CPU binding's float32 narrowing (`trees_f32`)
            qp[unsafe_offset=i] = Float32(pp[unsafe_offset=i])
        accumulate(acc, qp, n * k, 1.0)
    if n * k > 0:
        scale_f64(acc, n * k, Float64(len(pairs)))
