# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The trees lane's element-wise glue on the device (lane cpu2-l5-trees).

`xtrees/api.mojo` ran these entries through the serial host loops of
`xtrees/ops.mojo` on every GPU build: gathers, the vote accumulators,
argmax, the scalings, the margin, softmax / normalize / logit, the weight
checks, the scatter, the RNG draws, the one-hot leaf embedding and the
tree-score accumulator. A GPU install now runs them here: the caller's
arrays go up once, grid-wide kernels do the work, the result comes back
once. The CPU column (`-D MOJOLEARN_COLUMN_CPU`) keeps the serial loops.

BITS. Every binary64 operation is `checks/soft_f64.mojo`'s integer spelling
(the Apple GPU has no float64; every float64 buffer travels as UInt64
words): `sf64_add/sub/mul/div` are correctly rounded IEEE, the host's
hardware double ops (the host's `identical_mul64` is an unfused product);
`sf64_exp/log` are `portable_exp64/log64` statement for statement, the
IEEE host's `identical_exp64/log64` under IDENTICAL. Each kernel performs
the host loop's per-element operations in the host's order (a row's class
loop stays in class order on one thread), so the words agree on every
vendor and the host column. Float compares go through `_gt64` / `_gt32`
(IEEE: a NaN compares false, +0 == -0). The float32 product `mul_f32` is
the exact binary64 product of the widened operands rounded ONCE to binary32
(`sf64_to_f32`): the IEEE float32 product, with no FMA or flush-to-zero any
compiler could introduce. NaN results are the canonical quiet NaN (payloads
are refused by the identity contract, IDENTITY_PATHS row 39).

DATA-SIZED FOLDS. `normalized_weights` sums n weights: host and device use
ONE fixed order (xtrees/fold_order.mojo: rows in `FOLD_CHUNK` chunks summed
in index order, the chunk partials folded by the fixed pairwise tree);
`exact_sum_f32` adds integers (order-free); `sample_indices` without
replacement selects by integer keys (order-free).

REFUSALS. Every index the host loop range-checked is checked by a device
kernel before anything is written; a refusal raises the host loop's message
and leaves the caller's output untouched (the host loop may have written a
prefix before it raised).
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.soft_f64 import (
    SF64_ONE, SF64_ZERO, sf64_add, sf64_div, sf64_exp, sf64_from_f32, sf64_from_int, sf64_is_nan, sf64_log,
    sf64_lt, sf64_mul, sf64_neg, sf64_sub, sf64_to_f32,
)
from xtrees.fold_order import FOLD_CHUNK, fold_chunks, fold_tree_device
from xtrees.exact_sum import ES_FLAGS, ES_LIMBS, ES_NAN, ES_NINF, ES_PINF, ES_TPB, _es_blocks, es_partial_kernel
from xtrees.ops import EXACT_SUM_LIMBS, WS_CHUNK, draw, stream_base
from xtrees.ops_device import OPS_TPB, _blocks, _compact_device, _ctx, cnt_scan_step_kernel

#: 2^-53 as a binary64 word: `ops.unit`'s scale (exact).
comptime _TWO_M53 = UInt64(0x3CA0000000000000)
#: 2.0 as a binary64 word (`margin2`'s halving).
comptime _TWO_W = UInt64(0x4000000000000000)
#: `sample_indices`' radix select: 8-bit digits over the 53-bit keys,
#: shifts 48, 40, ..., 0.
comptime SI_DIGITS = 256
comptime SI_PASSES = 7


# ------------------------------------------------------------ helpers --


@always_inline
def _gt64(a: UInt64, b: UInt64) -> Bool:
    """IEEE `a > b` on binary64 words (a NaN compares false)."""
    if sf64_is_nan(a) or sf64_is_nan(b):
        return False
    return sf64_lt(b, a)


@always_inline
def _gt32(a: UInt32, b: UInt32) -> Bool:
    """IEEE `a > b` on binary32 words (a NaN compares false, +0 == -0)."""
    if (a & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000) or (b & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000):
        return False
    if ((a | b) & UInt32(0x7FFFFFFF)) == 0:
        return False
    var sa = a >> 31
    var sb = b >> 31
    if sa != sb:
        return sb == 1
    if sa == 0:
        return a > b
    return a < b


@always_inline
def _w(x: Float64) -> UInt64:
    return bitcast[DType.uint64](x)


def _up_i32(ctx: DeviceContext, p: MutPointer[Int32, MutUntrackedOrigin], n: Int) raises -> DeviceBuffer[DType.int32]:
    var d = ctx.enqueue_create_buffer[DType.int32](max(1, n))
    if n > 0:
        ctx.enqueue_copy(dst_buf=d, src_ptr=p)
    return d^


def _up_f32(ctx: DeviceContext, p: MutPointer[Float32, MutUntrackedOrigin], n: Int) raises -> DeviceBuffer[DType.float32]:
    var d = ctx.enqueue_create_buffer[DType.float32](max(1, n))
    if n > 0:
        ctx.enqueue_copy(dst_buf=d, src_ptr=p)
    return d^


def _up_f64(ctx: DeviceContext, p: MutPointer[Float64, MutUntrackedOrigin], n: Int) raises -> DeviceBuffer[DType.uint64]:
    """binary64 words up as UInt64 (the Apple GPU has no float64)."""
    var d = ctx.enqueue_create_buffer[DType.uint64](max(1, n))
    if n > 0:
        ctx.enqueue_copy(dst_buf=d, src_ptr=p.bitcast[UInt64]())
    return d^


def _flags(ctx: DeviceContext, k: Int) raises -> DeviceBuffer[DType.int32]:
    var d = ctx.enqueue_create_buffer[DType.int32](k)
    d.enqueue_fill(Int32(0))
    return d^


def _read_i32(ctx: DeviceContext, d: DeviceBuffer[DType.int32], k: Int) raises -> List[Int]:
    """The k words of `d` (exactly k long) back on the host, synchronized."""
    var h = ctx.enqueue_create_host_buffer[DType.int32](k)
    ctx.enqueue_copy(dst_buf=h, src_buf=d)
    ctx.synchronize()
    var res = List[Int](capacity=k)
    for i in range(k):  # small-loop(k: flag words): every caller reads at most ES_FLAGS scalar words
        res.append(Int(h.unsafe_ptr().unsafe_load(i)))
    _ = h^
    return res^


# ------------------------------------------------------- copies, ranges --


def row_check_kernel(
    rows: MutPointer[Int32, MutAnyOrigin], m: Int64, n: Int64, flag: MutPointer[Int32, MutAnyOrigin],
):
    """flag[0] = 1 on an index outside [0, n). Idempotent stores."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < Int(m):
        var i = Int(rows.unsafe_load(r))
        if i < 0 or i >= Int(n):
            flag.unsafe_store(0, Int32(1))
        r += stride


def gather_i32_kernel(
    src: MutPointer[Int32, MutAnyOrigin], rows: MutPointer[Int32, MutAnyOrigin], m: Int64,
    dst: MutPointer[Int32, MutAnyOrigin],
):
    """dst[r] = src[rows[r]], one thread per row (a copy)."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < Int(m):
        dst.unsafe_store(r, src.unsafe_load(Int(rows.unsafe_load(r))))
        r += stride


def gather_i32_device(
    src: MutPointer[Int32, MutUntrackedOrigin], n_src: Int,
    rows: MutPointer[Int32, MutUntrackedOrigin], n_rows: Int,
    dst: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """`ops.gather_i32` on the device: the same words, the same refusal."""
    if n_rows <= 0:
        return
    var ctx = _ctx()
    var d_src = _up_i32(ctx, src, n_src)
    var d_rows = _up_i32(ctx, rows, n_rows)
    var d_flag = _flags(ctx, 1)
    ctx.enqueue_function[row_check_kernel](
        d_rows.unsafe_ptr(), Int64(n_rows), Int64(n_src), d_flag.unsafe_ptr(),
        grid_dim=_blocks(n_rows), block_dim=OPS_TPB,
    )
    var fl = _read_i32(ctx, d_flag, 1)
    if fl[0] != 0:
        raise Error("x_trees gather: row index out of range")
    var d_dst = ctx.enqueue_create_buffer[DType.int32](n_rows)
    ctx.enqueue_function[gather_i32_kernel](
        d_src.unsafe_ptr(), d_rows.unsafe_ptr(), Int64(n_rows), d_dst.unsafe_ptr(),
        grid_dim=_blocks(n_rows), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=dst, src_buf=d_dst)
    ctx.synchronize()
    _ = d_src^
    _ = d_rows^
    _ = d_flag^
    _ = d_dst^


def put_f32_device(
    dst: MutPointer[Float32, MutUntrackedOrigin], offset: Int,
    src: MutPointer[Float32, MutUntrackedOrigin], n: Int,
) raises:
    """`ops.put_f32` on the device: src up once, down into dst[offset:]. A
    copy."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=d, src_ptr=src)
    ctx.enqueue_copy(dst_ptr=dst + offset, src_buf=d)
    ctx.synchronize()
    _ = d^


def iota_kernel(res: MutPointer[Int32, MutAnyOrigin], n: Int64):
    """res[i] = i."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        res.unsafe_store(i, Int32(i))
        i += stride


def iota_i32_device(res: MutPointer[Int32, MutUntrackedOrigin], n: Int) raises:
    """`ops.iota_i32` on the device."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[iota_kernel](d.unsafe_ptr(), Int64(n), grid_dim=_blocks(n), block_dim=OPS_TPB)
    ctx.enqueue_copy(dst_ptr=res, src_buf=d)
    ctx.synchronize()
    _ = d^


def fill_class_major_kernel(
    inits: MutPointer[UInt64, MutAnyOrigin], n: Int64, k: Int64, res: MutPointer[UInt64, MutAnyOrigin],
):
    """res[c * n + i] = inits[c] (a word copy), one thread per cell."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var total = Int(n) * Int(k)
    while t < total:
        res.unsafe_store(t, inits.unsafe_load(t // Int(n)))
        t += stride


def fill_class_major_f64_device(
    inits: MutPointer[Float64, MutUntrackedOrigin], n: Int, k: Int, res: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """`ops.fill_class_major_f64` on the device."""
    if n <= 0 or k <= 0:
        return
    var ctx = _ctx()
    var d_in = _up_f64(ctx, inits, k)
    var d = ctx.enqueue_create_buffer[DType.uint64](n * k)
    ctx.enqueue_function[fill_class_major_kernel](
        d_in.unsafe_ptr(), Int64(n), Int64(k), d.unsafe_ptr(), grid_dim=_blocks(n * k), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=res.bitcast[UInt64](), src_buf=d)
    ctx.synchronize()
    _ = d_in^
    _ = d^


# ------------------------------------------------------ accumulators --


def accumulate_kernel(
    acc: MutPointer[UInt64, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin], n: Int64, w: UInt64,
):
    """acc[i] = acc[i] + w * x[i] (two roundings, never fused)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        acc.unsafe_store(i, sf64_add(acc.unsafe_load(i), sf64_mul(w, sf64_from_f32(x.unsafe_load(i)))))
        i += stride


def accumulate_device(
    acc: MutPointer[Float64, MutUntrackedOrigin], x: MutPointer[Float32, MutUntrackedOrigin], n: Int, weight: Float64,
) raises:
    """`ops.accumulate` on the device."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d_acc = _up_f64(ctx, acc, n)
    var d_x = _up_f32(ctx, x, n)
    ctx.enqueue_function[accumulate_kernel](
        d_acc.unsafe_ptr(), d_x.unsafe_ptr(), Int64(n), _w(weight), grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=acc.bitcast[UInt64](), src_buf=d_acc)
    ctx.synchronize()
    _ = d_acc^
    _ = d_x^


def accumulate_cols_kernel(
    acc: MutPointer[UInt64, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin], cols: MutPointer[Int32, MutAnyOrigin],
    n: Int64, k: Int64, ks: Int64, w: UInt64,
):
    """One thread per row walks its ks columns in order (`cols` may repeat
    a column: the in-order adds are the host's)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        for c in range(Int(ks)):
            var j = i * Int(k) + Int(cols.unsafe_load(c))
            acc.unsafe_store(j, sf64_add(acc.unsafe_load(j), sf64_mul(w, sf64_from_f32(x.unsafe_load(i * Int(ks) + c)))))
        i += stride


def accumulate_cols_device(
    acc: MutPointer[Float64, MutUntrackedOrigin], x: MutPointer[Float32, MutUntrackedOrigin],
    cols: MutPointer[Int32, MutUntrackedOrigin], n: Int, k: Int, ks: Int, weight: Float64,
) raises:
    """`ops.accumulate_cols` on the device: the column check first (the
    host's refusal), then one thread per row."""
    var ctx = _ctx()
    var d_cols = _up_i32(ctx, cols, ks)
    if ks > 0:
        var d_flag = _flags(ctx, 1)
        ctx.enqueue_function[row_check_kernel](
            d_cols.unsafe_ptr(), Int64(ks), Int64(k), d_flag.unsafe_ptr(), grid_dim=_blocks(ks), block_dim=OPS_TPB,
        )
        var fl = _read_i32(ctx, d_flag, 1)
        _ = d_flag^
        if fl[0] != 0:
            raise Error("x_trees accumulate_cols: column out of range")
    if n <= 0 or ks <= 0:
        _ = d_cols^
        return
    var d_acc = _up_f64(ctx, acc, n * k)
    var d_x = _up_f32(ctx, x, n * ks)
    ctx.enqueue_function[accumulate_cols_kernel](
        d_acc.unsafe_ptr(), d_x.unsafe_ptr(), d_cols.unsafe_ptr(), Int64(n), Int64(k), Int64(ks), _w(weight),
        grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=acc.bitcast[UInt64](), src_buf=d_acc)
    ctx.synchronize()
    _ = d_cols^
    _ = d_acc^
    _ = d_x^


def rows_mark_kernel(
    rows: MutPointer[Int32, MutAnyOrigin], m: Int64, n: Int64, cnt: MutPointer[Int32, MutAnyOrigin],
    flag: MutPointer[Int32, MutAnyOrigin],
):
    """flag[0] = 1 on a row outside [0, n); cnt[row] counts its draws by
    integer atomics, flag[1] = 1 when a row is met twice."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < Int(m):
        var i = Int(rows.unsafe_load(r))
        if i < 0 or i >= Int(n):
            flag.unsafe_store(0, Int32(1))
        else:
            var old = Atomic.fetch_add(cnt.unsafe_offset(i), Int32(1))
            if old >= Int32(1):
                flag.unsafe_store(1, Int32(1))
        r += stride


def accumulate_rows_cells_kernel(
    acc: MutPointer[UInt64, MutAnyOrigin], k: Int64, x: MutPointer[UInt64, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin], m: Int64,
):
    """Distinct rows: one thread per (r, c), acc[rows[r], c] += x[r, c]."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var total = Int(m) * Int(k)
    while t < total:
        var r = t // Int(k)
        var c = t - r * Int(k)
        var j = Int(rows.unsafe_load(r)) * Int(k) + c
        acc.unsafe_store(j, sf64_add(acc.unsafe_load(j), x.unsafe_load(t)))
        t += stride


def accumulate_rows_cols_kernel(
    acc: MutPointer[UInt64, MutAnyOrigin], k: Int64, x: MutPointer[UInt64, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin], m: Int64,
):
    """Repeated rows: one thread per column c walks r in order, so a cell
    met twice takes its adds in the host's order."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while c < Int(k):
        for r in range(Int(m)):
            var j = Int(rows.unsafe_load(r)) * Int(k) + c
            acc.unsafe_store(j, sf64_add(acc.unsafe_load(j), x.unsafe_load(r * Int(k) + c)))
        c += stride


def accumulate_rows_device(
    acc: MutPointer[Float64, MutUntrackedOrigin], n: Int, k: Int, x: MutPointer[Float64, MutUntrackedOrigin],
    rows: MutPointer[Int32, MutUntrackedOrigin], m: Int,
) raises:
    """`ops.accumulate_rows` on the device. A mark kernel range-checks the
    rows (the host's refusal) and counts each row's draws with integer
    atomics. Distinct rows (the out-of-bag case): one thread per (r, c).
    A repeated row: one thread per column walking r in order, the host's
    add order for every cell (parallel over the k columns only)."""
    if m <= 0:
        return
    var ctx = _ctx()
    var d_rows = _up_i32(ctx, rows, m)
    var d_cnt = ctx.enqueue_create_buffer[DType.int32](max(1, n))
    d_cnt.enqueue_fill(Int32(0))
    var d_flag = _flags(ctx, 2)
    ctx.enqueue_function[rows_mark_kernel](
        d_rows.unsafe_ptr(), Int64(m), Int64(n), d_cnt.unsafe_ptr(), d_flag.unsafe_ptr(),
        grid_dim=_blocks(m), block_dim=OPS_TPB,
    )
    var fl = _read_i32(ctx, d_flag, 2)
    if fl[0] != 0:
        raise Error("x_trees accumulate_rows: row out of range")
    if k <= 0:
        return
    var d_acc = _up_f64(ctx, acc, n * k)
    var d_x = _up_f64(ctx, x, m * k)
    if fl[1] == 0:
        ctx.enqueue_function[accumulate_rows_cells_kernel](
            d_acc.unsafe_ptr(), Int64(k), d_x.unsafe_ptr(), d_rows.unsafe_ptr(), Int64(m),
            grid_dim=_blocks(m * k), block_dim=OPS_TPB,
        )
    else:
        ctx.enqueue_function[accumulate_rows_cols_kernel](
            d_acc.unsafe_ptr(), Int64(k), d_x.unsafe_ptr(), d_rows.unsafe_ptr(), Int64(m),
            grid_dim=_blocks(k), block_dim=OPS_TPB,
        )
    ctx.enqueue_copy(dst_ptr=acc.bitcast[UInt64](), src_buf=d_acc)
    ctx.synchronize()
    _ = d_rows^
    _ = d_cnt^
    _ = d_flag^
    _ = d_acc^
    _ = d_x^


def accumulate_onehot_kernel(
    acc: MutPointer[UInt64, MutAnyOrigin], codes: MutPointer[Int32, MutAnyOrigin], n: Int64, k: Int64,
    on: UInt64, off: UInt64,
):
    """acc[i, c] += on if codes[i] == c else off, one thread per cell."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var total = Int(n) * Int(k)
    while t < total:
        var i = t // Int(k)
        var c = t - i * Int(k)
        var v = on if c == Int(codes.unsafe_load(i)) else off
        acc.unsafe_store(t, sf64_add(acc.unsafe_load(t), v))
        t += stride


def accumulate_onehot_device(
    acc: MutPointer[Float64, MutUntrackedOrigin], codes: MutPointer[Int32, MutUntrackedOrigin],
    n: Int, k: Int, on: Float64, off: Float64,
) raises:
    """`ops.accumulate_onehot` on the device: the code check, then one
    thread per cell."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d_codes = _up_i32(ctx, codes, n)
    var d_flag = _flags(ctx, 1)
    ctx.enqueue_function[row_check_kernel](
        d_codes.unsafe_ptr(), Int64(n), Int64(k), d_flag.unsafe_ptr(), grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    var fl = _read_i32(ctx, d_flag, 1)
    if fl[0] != 0:
        raise Error("x_trees accumulate_onehot: code out of range")
    var d_acc = _up_f64(ctx, acc, n * k)
    ctx.enqueue_function[accumulate_onehot_kernel](
        d_acc.unsafe_ptr(), d_codes.unsafe_ptr(), Int64(n), Int64(k), _w(on), _w(off),
        grid_dim=_blocks(n * k), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=acc.bitcast[UInt64](), src_buf=d_acc)
    ctx.synchronize()
    _ = d_codes^
    _ = d_flag^
    _ = d_acc^


# ------------------------------------------------------------ argmax --


def argmax_rows_kernel(x: MutPointer[UInt64, MutAnyOrigin], n: Int64, k: Int64, res: MutPointer[Int32, MutAnyOrigin]):
    """First maximum of each row (strict >: a tie keeps the lower index)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        var best = 0
        for c in range(1, Int(k)):
            if _gt64(x.unsafe_load(i * Int(k) + c), x.unsafe_load(i * Int(k) + best)):
                best = c
        res.unsafe_store(i, Int32(best))
        i += stride


def argmax_rows_f32_kernel(x: MutPointer[UInt32, MutAnyOrigin], n: Int64, k: Int64, res: MutPointer[Int32, MutAnyOrigin]):
    """`argmax_rows_kernel` over binary32 words."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        var best = 0
        for c in range(1, Int(k)):
            if _gt32(x.unsafe_load(i * Int(k) + c), x.unsafe_load(i * Int(k) + best)):
                best = c
        res.unsafe_store(i, Int32(best))
        i += stride


def argmax_rows_device(
    x: MutPointer[Float64, MutUntrackedOrigin], n: Int, k: Int, res: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """`ops.argmax_rows` on the device, one thread per row."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d_x = _up_f64(ctx, x, n * k)
    var d_res = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[argmax_rows_kernel](
        d_x.unsafe_ptr(), Int64(n), Int64(k), d_res.unsafe_ptr(), grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=res, src_buf=d_res)
    ctx.synchronize()
    _ = d_x^
    _ = d_res^


def argmax_rows_f32_device(
    x: MutPointer[Float32, MutUntrackedOrigin], n: Int, k: Int, res: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """`ops.argmax_rows_f32` on the device, one thread per row."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d_x = ctx.enqueue_create_buffer[DType.uint32](max(1, n * k))
    if n * k > 0:
        ctx.enqueue_copy(dst_buf=d_x, src_ptr=x.bitcast[UInt32]())
    var d_res = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[argmax_rows_f32_kernel](
        d_x.unsafe_ptr(), Int64(n), Int64(k), d_res.unsafe_ptr(), grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=res, src_buf=d_res)
    ctx.synchronize()
    _ = d_x^
    _ = d_res^


# ------------------------------------------------- binary64 elementwise --


def scale_kernel(x: MutPointer[UInt64, MutAnyOrigin], n: Int64, divisor: UInt64):
    """x[i] = x[i] / divisor."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        x.unsafe_store(i, sf64_div(x.unsafe_load(i), divisor))
        i += stride


def scale_f64_device(x: MutPointer[Float64, MutUntrackedOrigin], n: Int, divisor: Float64) raises:
    """`ops.scale_f64` on the device (the host gate's sabotage stays on the
    host column)."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d_x = _up_f64(ctx, x, n)
    ctx.enqueue_function[scale_kernel](d_x.unsafe_ptr(), Int64(n), _w(divisor), grid_dim=_blocks(n), block_dim=OPS_TPB)
    ctx.enqueue_copy(dst_ptr=x.bitcast[UInt64](), src_buf=d_x)
    ctx.synchronize()
    _ = d_x^


def scale_to_f32_kernel(x: MutPointer[UInt64, MutAnyOrigin], n: Int64, factor: UInt64, res: MutPointer[Float32, MutAnyOrigin]):
    """res[i] = float32(x[i] * factor): one binary64 product, one narrowing."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        res.unsafe_store(i, sf64_to_f32(sf64_mul(x.unsafe_load(i), factor)))
        i += stride


def scale_to_f32_device(
    x: MutPointer[Float64, MutUntrackedOrigin], n: Int, factor: Float64, res: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`ops.scale_to_f32` on the device."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d_x = _up_f64(ctx, x, n)
    var d_res = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_function[scale_to_f32_kernel](
        d_x.unsafe_ptr(), Int64(n), _w(factor), d_res.unsafe_ptr(), grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=res, src_buf=d_res)
    ctx.synchronize()
    _ = d_x^
    _ = d_res^


def margin2_kernel(
    acc: MutPointer[UInt64, MutAnyOrigin], n: Int64, mode: Int64,
    out_f: MutPointer[UInt64, MutAnyOrigin], out_i: MutPointer[Int32, MutAnyOrigin],
):
    """`ops.margin2`'s three bodies, one thread per row."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        var d = sf64_sub(acc.unsafe_load(2 * i + 1), acc.unsafe_load(2 * i))
        if mode == 0:
            out_f.unsafe_store(i, d)
        elif mode == 1:
            out_i.unsafe_store(i, Int32(1) if _gt64(d, SF64_ZERO) else Int32(0))
        else:
            var h = sf64_div(d, _TWO_W)
            out_f.unsafe_store(2 * i, sf64_neg(h))
            out_f.unsafe_store(2 * i + 1, h)
        i += stride


def margin2_device(
    acc: MutPointer[Float64, MutUntrackedOrigin], n: Int, mode: Int,
    dst_f: MutPointer[Float64, MutUntrackedOrigin], dst_i: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """`ops.margin2` on the device: the same mode refusal, the same words."""
    if mode < 0 or mode > 2:
        raise Error("x_trees_margin2: mode must be 0, 1 or 2")
    if n <= 0:
        return
    var ctx = _ctx()
    var d_acc = _up_f64(ctx, acc, 2 * n)
    var d_f = ctx.enqueue_create_buffer[DType.uint64](2 * n if mode == 2 else (n if mode == 0 else 1))
    var d_i = ctx.enqueue_create_buffer[DType.int32](n if mode == 1 else 1)
    ctx.enqueue_function[margin2_kernel](
        d_acc.unsafe_ptr(), Int64(n), Int64(mode), d_f.unsafe_ptr(), d_i.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    if mode == 1:
        ctx.enqueue_copy(dst_ptr=dst_i, src_buf=d_i)
    else:
        ctx.enqueue_copy(dst_ptr=dst_f.bitcast[UInt64](), src_buf=d_f)
    ctx.synchronize()
    _ = d_acc^
    _ = d_f^
    _ = d_i^


def softmax_rows_kernel(x: MutPointer[UInt64, MutAnyOrigin], n: Int64, k: Int64):
    """`ops.softmax_rows`' row body, one thread per row: the max by `>` in
    class order, exp(x - max) in class order, the sum in class order, the
    divisions."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var kk = Int(k)
    while i < Int(n):
        var m = x.unsafe_load(i * kk)
        for c in range(1, kk):
            var v = x.unsafe_load(i * kk + c)
            if _gt64(v, m):
                m = v
        var s = SF64_ZERO
        for c in range(kk):
            var e = sf64_exp(sf64_sub(x.unsafe_load(i * kk + c), m))
            x.unsafe_store(i * kk + c, e)
            s = sf64_add(s, e)
        for c in range(kk):
            x.unsafe_store(i * kk + c, sf64_div(x.unsafe_load(i * kk + c), s))
        i += stride


def softmax_rows_device(x: MutPointer[Float64, MutUntrackedOrigin], n: Int, k: Int) raises:
    """`ops.softmax_rows` on the device."""
    if n <= 0 or k <= 0:
        return
    var ctx = _ctx()
    var d_x = _up_f64(ctx, x, n * k)
    ctx.enqueue_function[softmax_rows_kernel](d_x.unsafe_ptr(), Int64(n), Int64(k), grid_dim=_blocks(n), block_dim=OPS_TPB)
    ctx.enqueue_copy(dst_ptr=x.bitcast[UInt64](), src_buf=d_x)
    ctx.synchronize()
    _ = d_x^


def normalize_rows_kernel(x: MutPointer[UInt64, MutAnyOrigin], n: Int64, k: Int64, uni: UInt64):
    """`ops.normalize_rows`' row body: the sum in class order, x / sum when
    the sum is > 0, else the uniform 1 / k (DEVIATION 5605)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var kk = Int(k)
    while i < Int(n):
        var s = SF64_ZERO
        for c in range(kk):
            s = sf64_add(s, x.unsafe_load(i * kk + c))
        var pos = _gt64(s, SF64_ZERO)
        for c in range(kk):
            if pos:
                x.unsafe_store(i * kk + c, sf64_div(x.unsafe_load(i * kk + c), s))
            else:
                x.unsafe_store(i * kk + c, uni)
        i += stride


def normalize_rows_device(x: MutPointer[Float64, MutUntrackedOrigin], n: Int, k: Int) raises:
    """`ops.normalize_rows` on the device, one thread per row."""
    if n <= 0 or k <= 0:
        return
    var ctx = _ctx()
    var d_x = _up_f64(ctx, x, n * k)
    ctx.enqueue_function[normalize_rows_kernel](
        d_x.unsafe_ptr(), Int64(n), Int64(k), sf64_div(SF64_ONE, sf64_from_int(k)),
        grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=x.bitcast[UInt64](), src_buf=d_x)
    ctx.synchronize()
    _ = d_x^


def logit_kernel(x: MutPointer[UInt64, MutAnyOrigin], n: Int64):
    """x = log(x / (1 - x)): one subtraction, one division, the pinned log."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        var v = x.unsafe_load(i)
        x.unsafe_store(i, sf64_log(sf64_div(v, sf64_sub(SF64_ONE, v))))
        i += stride


def logit_device(x: MutPointer[Float64, MutUntrackedOrigin], n: Int) raises:
    """`ops.logit` on the device."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d_x = _up_f64(ctx, x, n)
    ctx.enqueue_function[logit_kernel](d_x.unsafe_ptr(), Int64(n), grid_dim=_blocks(n), block_dim=OPS_TPB)
    ctx.enqueue_copy(dst_ptr=x.bitcast[UInt64](), src_buf=d_x)
    ctx.synchronize()
    _ = d_x^


def complement_pairs_kernel(x: MutPointer[UInt64, MutAnyOrigin], n: Int64):
    """x[2 i] = 1 - x[2 i + 1]."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        x.unsafe_store(2 * i, sf64_sub(SF64_ONE, x.unsafe_load(2 * i + 1)))
        i += stride


def complement_pairs_device(x: MutPointer[Float64, MutUntrackedOrigin], n: Int) raises:
    """`ops.complement_pairs` on the device."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d_x = _up_f64(ctx, x, 2 * n)
    ctx.enqueue_function[complement_pairs_kernel](d_x.unsafe_ptr(), Int64(n), grid_dim=_blocks(n), block_dim=OPS_TPB)
    ctx.enqueue_copy(dst_ptr=x.bitcast[UInt64](), src_buf=d_x)
    ctx.synchronize()
    _ = d_x^


def uniform_kernel(res: MutPointer[UInt64, MutAnyOrigin], n: Int64, base: UInt64):
    """res[k] = unit(draw(base, k)): the top 53 bits times 2^-53 (exact)."""
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while k < Int(n):
        res.unsafe_store(k, sf64_mul(sf64_from_int(Int(draw(base, k) >> 11)), _TWO_M53))
        k += stride


def uniform_device(res: MutPointer[Float64, MutUntrackedOrigin], n: Int, seed: Int, stream: Int) raises:
    """`ops.uniform` on the device: the draws are made on the device."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d = ctx.enqueue_create_buffer[DType.uint64](n)
    ctx.enqueue_function[uniform_kernel](d.unsafe_ptr(), Int64(n), stream_base(seed, stream), grid_dim=_blocks(n), block_dim=OPS_TPB)
    ctx.enqueue_copy(dst_ptr=res.bitcast[UInt64](), src_buf=d)
    ctx.synchronize()
    _ = d^


def mul_f32_kernel(
    a: MutPointer[Float32, MutAnyOrigin], b: MutPointer[Float32, MutAnyOrigin], n: Int64,
    res: MutPointer[Float32, MutAnyOrigin],
):
    """res[i] = a[i] * b[i] in float32: the exact binary64 product of the
    widened operands (24 x 24 bits fit 53) rounded once to binary32."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        res.unsafe_store(i, sf64_to_f32(sf64_mul(sf64_from_f32(a.unsafe_load(i)), sf64_from_f32(b.unsafe_load(i)))))
        i += stride


def mul_f32_device(
    a: MutPointer[Float32, MutUntrackedOrigin], b: MutPointer[Float32, MutUntrackedOrigin], n: Int,
    res: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`ops.mul_f32` on the device."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d_a = _up_f32(ctx, a, n)
    var d_b = _up_f32(ctx, b, n)
    var d_res = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_function[mul_f32_kernel](
        d_a.unsafe_ptr(), d_b.unsafe_ptr(), Int64(n), d_res.unsafe_ptr(), grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=res, src_buf=d_res)
    ctx.synchronize()
    _ = d_a^
    _ = d_b^
    _ = d_res^


def check_weights_kernel(w: MutPointer[UInt32, MutAnyOrigin], n: Int64, flags: MutPointer[Int32, MutAnyOrigin]):
    """flags[0] = 1 on an entry that is NaN, infinite or negative (-0 is
    fine: `-0 >= 0`), flags[1] = 1 on a positive entry. Order-free flags."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        var bits = w.unsafe_load(i)
        var mag = bits & UInt32(0x7FFFFFFF)
        if mag >= UInt32(0x7F800000) or ((bits >> 31) != 0 and mag != 0):
            flags.unsafe_store(0, Int32(1))
        elif mag != 0:
            flags.unsafe_store(1, Int32(1))
        i += stride


def check_weights_f32_device(w: MutPointer[Float32, MutUntrackedOrigin], n: Int) raises -> Int:
    """`ops.check_weights_f32` on the device: 1 if an entry is bad, 2 if
    none is positive, else 0."""
    if n <= 0:
        return 2
    var ctx = _ctx()
    var d_w = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=w.bitcast[UInt32]())
    var d_flag = _flags(ctx, 2)
    ctx.enqueue_function[check_weights_kernel](d_w.unsafe_ptr(), Int64(n), d_flag.unsafe_ptr(), grid_dim=_blocks(n), block_dim=OPS_TPB)
    var fl = _read_i32(ctx, d_flag, 2)
    _ = d_w^
    _ = d_flag^
    if fl[0] != 0:
        return 1
    return 0 if fl[1] != 0 else 2


# --------------------------------------------------- scatter, leaves --


def scatter_own_kernel(
    rows: MutPointer[Int32, MutAnyOrigin], m: Int64, n_dst: Int64, owner: MutPointer[Int32, MutAnyOrigin],
    flag: MutPointer[Int32, MutAnyOrigin],
):
    """flag[0] = 1 on a row outside [0, n_dst); owner[row] = the LAST r
    that writes it (integer atomic max): the host loop's last write wins."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < Int(m):
        var i = Int(rows.unsafe_load(r))
        if i < 0 or i >= Int(n_dst):
            flag.unsafe_store(0, Int32(1))
        else:
            _ = Atomic.max(owner.unsafe_offset(i), Int32(r))
        r += stride


def scatter_kernel(
    dst: MutPointer[UInt64, MutAnyOrigin], n_cols: Int64, src: MutPointer[Float32, MutAnyOrigin], m: Int64, c: Int64,
    rows: MutPointer[Int32, MutAnyOrigin], col0: Int64, owner: MutPointer[Int32, MutAnyOrigin],
):
    """dst[rows[r], col0 + j] = float64(src[r, j]) for the owning r, one
    thread per (r, j). A widening copy."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var total = Int(m) * Int(c)
    while t < total:
        var r = t // Int(c)
        var j = t - r * Int(c)
        var i = Int(rows.unsafe_load(r))
        if Int(owner.unsafe_load(i)) == r:
            dst.unsafe_store(i * Int(n_cols) + Int(col0) + j, sf64_from_f32(src.unsafe_load(t)))
        t += stride


def scatter_device(
    dst: MutPointer[Float64, MutUntrackedOrigin], n_dst: Int, n_cols: Int,
    src: MutPointer[Float32, MutUntrackedOrigin], m: Int, c: Int,
    rows: MutPointer[Int32, MutUntrackedOrigin], col0: Int,
) raises:
    """The `x_trees_scatter` binding on the device: its row refusal by a
    check kernel, then `ops.scatter`'s cells. A repeated row takes the last
    r's values (the host loop's last write) through the owner pass. The
    binding has already refused col0 + c > n_cols."""
    if m <= 0:
        return
    var ctx = _ctx()
    var d_rows = _up_i32(ctx, rows, m)
    var d_own = ctx.enqueue_create_buffer[DType.int32](max(1, n_dst))
    d_own.enqueue_fill(Int32(-1))
    var d_flag = _flags(ctx, 1)
    ctx.enqueue_function[scatter_own_kernel](
        d_rows.unsafe_ptr(), Int64(m), Int64(n_dst), d_own.unsafe_ptr(), d_flag.unsafe_ptr(),
        grid_dim=_blocks(m), block_dim=OPS_TPB,
    )
    var fl = _read_i32(ctx, d_flag, 1)
    if fl[0] != 0:
        raise Error("x_trees_scatter: row out of range")
    if c <= 0:
        return
    var d_dst = _up_f64(ctx, dst, n_dst * n_cols)
    var d_src = _up_f32(ctx, src, m * c)
    ctx.enqueue_function[scatter_kernel](
        d_dst.unsafe_ptr(), Int64(n_cols), d_src.unsafe_ptr(), Int64(m), Int64(c), d_rows.unsafe_ptr(), Int64(col0),
        d_own.unsafe_ptr(), grid_dim=_blocks(m * c), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=dst.bitcast[UInt64](), src_buf=d_dst)
    ctx.synchronize()
    _ = d_rows^
    _ = d_own^
    _ = d_flag^
    _ = d_dst^
    _ = d_src^


def max_index_kernel(
    idx: MutPointer[Int32, MutAnyOrigin], n: Int64, base: MutPointer[Int32, MutAnyOrigin], nb: Int64,
    flag: MutPointer[Int32, MutAnyOrigin], mx: MutPointer[Int32, MutAnyOrigin],
):
    """For every cell t: j = idx[t] (+ base[t % nb] when nb > 0); flag[0] = 1
    on a negative j, mx[0] = max j (integer atomic). Sizes the table the
    host loop indexes without a length."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while t < Int(n):
        var j = Int(idx.unsafe_load(t))
        if nb > 0:
            j += Int(base.unsafe_load(t % Int(nb)))
        if j < 0:
            flag.unsafe_store(0, Int32(1))
        else:
            _ = Atomic.max(mx, Int32(j))
        t += stride


def _table_len(
    ctx: DeviceContext, d_idx: DeviceBuffer[DType.int32], n: Int, d_base: DeviceBuffer[DType.int32], nb: Int,
) raises -> Int:
    """1 + the largest table index the cells reach, or -1 when one is
    negative."""
    var d_flag = _flags(ctx, 1)
    var d_mx = ctx.enqueue_create_buffer[DType.int32](1)
    d_mx.enqueue_fill(Int32(0))
    ctx.enqueue_function[max_index_kernel](
        rebind[MutPointer[Int32, MutAnyOrigin]](d_idx.unsafe_ptr()), Int64(n),
        rebind[MutPointer[Int32, MutAnyOrigin]](d_base.unsafe_ptr()), Int64(nb),
        rebind[MutPointer[Int32, MutAnyOrigin]](d_flag.unsafe_ptr()),
        rebind[MutPointer[Int32, MutAnyOrigin]](d_mx.unsafe_ptr()),
        grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    var fl = _read_i32(ctx, d_flag, 1)
    var mx = _read_i32(ctx, d_mx, 1)
    _ = d_flag^
    _ = d_mx^
    if fl[0] != 0:
        return -1
    return mx[0] + 1


def onehot_leaves_kernel(
    nodes: MutPointer[Int32, MutAnyOrigin], base: MutPointer[Int32, MutAnyOrigin], node_col: MutPointer[Int32, MutAnyOrigin],
    n: Int64, nt: Int64, n_cols: Int64, res: MutPointer[UInt64, MutAnyOrigin], flag: MutPointer[Int32, MutAnyOrigin],
):
    """res[i, node_col[base[t] + nodes[i, t]]] = 1.0, one thread per (i, t);
    flag[0] = 1 on a column outside [0, n_cols). Idempotent stores."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var total = Int(n) * Int(nt)
    while t < total:
        var i = t // Int(nt)
        var tr = t - i * Int(nt)
        var c = Int(node_col.unsafe_load(Int(base.unsafe_load(tr)) + Int(nodes.unsafe_load(t))))
        if c < 0 or c >= Int(n_cols):
            flag.unsafe_store(0, Int32(1))
        else:
            res.unsafe_store(i * Int(n_cols) + c, SF64_ONE)
        t += stride


def onehot_leaves_device(
    nodes: MutPointer[Int32, MutUntrackedOrigin], tree_base: MutPointer[Int32, MutUntrackedOrigin],
    node_col: MutPointer[Int32, MutUntrackedOrigin], n: Int, n_trees: Int, n_cols: Int,
    res: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """`ops.onehot_leaves` on the device. `node_col` comes without a length:
    a max-index pass sizes the prefix the cells reach (the host loop reads
    exactly those entries); a negative index, which the host loop would
    read out of bounds, is refused with the leaf-column message. `res` goes
    up and back so cells the embedding does not set keep the caller's words."""
    if n <= 0 or n_trees <= 0:
        return
    var ctx = _ctx()
    var d_nodes = _up_i32(ctx, nodes, n * n_trees)
    var d_base = _up_i32(ctx, tree_base, n_trees)
    var tl = _table_len(ctx, d_nodes, n * n_trees, d_base, n_trees)
    if tl < 0:
        raise Error("x_trees onehot_leaves: a row reached a node that is not a leaf column")
    var d_col = _up_i32(ctx, node_col, tl)
    var d_res = _up_f64(ctx, res, n * n_cols)
    var d_flag = _flags(ctx, 1)
    ctx.enqueue_function[onehot_leaves_kernel](
        d_nodes.unsafe_ptr(), d_base.unsafe_ptr(), d_col.unsafe_ptr(), Int64(n), Int64(n_trees), Int64(n_cols),
        d_res.unsafe_ptr(), d_flag.unsafe_ptr(), grid_dim=_blocks(n * n_trees), block_dim=OPS_TPB,
    )
    var fl = _read_i32(ctx, d_flag, 1)
    if fl[0] != 0:
        raise Error("x_trees onehot_leaves: a row reached a node that is not a leaf column")
    if n_cols > 0:
        ctx.enqueue_copy(dst_ptr=res.bitcast[UInt64](), src_buf=d_res)
        ctx.synchronize()
    _ = d_nodes^
    _ = d_base^
    _ = d_col^
    _ = d_res^
    _ = d_flag^


def tree_score_add_kernel(
    nodes: MutPointer[Int32, MutAnyOrigin], values: MutPointer[Float32, MutAnyOrigin], n: Int64, w: UInt64,
    acc: MutPointer[UInt64, MutAnyOrigin],
):
    """acc[i] = acc[i] + w * values[nodes[i]] (two roundings, never fused)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        var v = sf64_from_f32(values.unsafe_load(Int(nodes.unsafe_load(i))))
        acc.unsafe_store(i, sf64_add(acc.unsafe_load(i), sf64_mul(w, v)))
        i += stride


def tree_score_add_device(
    nodes: MutPointer[Int32, MutUntrackedOrigin], values: MutPointer[Float32, MutUntrackedOrigin],
    n: Int, weight: Float64, acc: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """`ops.tree_score_add` on the device. `values` comes without a length:
    a max-index pass sizes the prefix the rows reach (the entries the host
    loop reads). The host loop never range-checks a node; a NEGATIVE node,
    which it would read out of bounds, is refused here."""
    if n <= 0:
        return
    var ctx = _ctx()
    var d_nodes = _up_i32(ctx, nodes, n)
    var d_none = ctx.enqueue_create_buffer[DType.int32](1)
    var tl = _table_len(ctx, d_nodes, n, d_none, 0)
    if tl < 0:
        raise Error("x_trees tree_score_add: node out of range")
    var d_vals = _up_f32(ctx, values, tl)
    var d_acc = _up_f64(ctx, acc, n)
    ctx.enqueue_function[tree_score_add_kernel](
        d_nodes.unsafe_ptr(), d_vals.unsafe_ptr(), Int64(n), _w(weight), d_acc.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=acc.bitcast[UInt64](), src_buf=d_acc)
    ctx.synchronize()
    _ = d_nodes^
    _ = d_none^
    _ = d_vals^
    _ = d_acc^


# --------------------------------------------------------- reductions --


def es_limbs_kernel(partials: MutPointer[Int64, MutAnyOrigin], n_blocks: Int64, limbs: MutPointer[Int64, MutAnyOrigin]):
    """One block: the block partials added exactly into `ES_LIMBS` limbs
    (integer adds, order-free), written unrounded."""
    var tid = Int(thread_idx.x)
    var acc = SIMD[DType.int64, ES_LIMBS](0)
    var b = tid
    while b < Int(n_blocks):
        for l in range(ES_LIMBS):
            acc[l] = acc[l] + partials.unsafe_load(b * ES_LIMBS + l)
        b += ES_TPB
    var sh = stack_allocation[ES_TPB * ES_LIMBS, Int64, address_space=AddressSpace.SHARED]()
    for l in range(ES_LIMBS):
        sh[tid * ES_LIMBS + l] = acc[l]
    barrier()
    var step = ES_TPB // 2
    while step > 0:
        if tid < step:
            for l in range(ES_LIMBS):
                sh[tid * ES_LIMBS + l] = sh[tid * ES_LIMBS + l] + sh[(tid + step) * ES_LIMBS + l]
        barrier()
        step //= 2
    if tid < ES_LIMBS:
        limbs.unsafe_store(tid, sh[tid])


def exact_sum_f32_device(x: MutPointer[Float32, MutUntrackedOrigin], n: Int, mut limbs: List[Int64]) raises -> Bool:
    """`ops.exact_sum_f32` on the device: each block adds its rows' integers
    into Int64 limbs (`exact_sum.es_partial_kernel`, the host's limb map),
    one block adds the block partials; integer adds, so the limbs are the
    host's exactly. The same refusals: False on n > 2^30 or a NaN or
    infinite entry."""
    comptime assert ES_LIMBS >= EXACT_SUM_LIMBS, "exact_sum_f32: device limbs must cover the host limbs"
    limbs = List[Int64](length=EXACT_SUM_LIMBS, fill=0)
    if n > (1 << 30):
        return False
    if n <= 0:
        return True
    var ctx = _ctx()
    var nb = _es_blocks(n)
    var d_x = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_buf=d_x, src_ptr=x.bitcast[UInt32]())
    var d_part = ctx.enqueue_create_buffer[DType.int64](nb * ES_LIMBS)
    var d_flags = _flags(ctx, ES_FLAGS)
    var d_limbs = ctx.enqueue_create_buffer[DType.int64](ES_LIMBS)
    ctx.enqueue_function[es_partial_kernel](
        d_x.unsafe_ptr(), Int64(n), d_part.unsafe_ptr(), d_flags.unsafe_ptr(), grid_dim=nb, block_dim=ES_TPB,
    )
    ctx.enqueue_function[es_limbs_kernel](d_part.unsafe_ptr(), Int64(nb), d_limbs.unsafe_ptr(), grid_dim=1, block_dim=ES_TPB)
    var h_limbs = ctx.enqueue_create_host_buffer[DType.int64](ES_LIMBS)
    ctx.enqueue_copy(dst_buf=h_limbs, src_buf=d_limbs)
    var fl = _read_i32(ctx, d_flags, ES_FLAGS)
    var ok = fl[ES_NAN] == 0 and fl[ES_PINF] == 0 and fl[ES_NINF] == 0
    if ok:
        for l in range(EXACT_SUM_LIMBS):
            limbs[l] = h_limbs.unsafe_ptr().unsafe_load(l)
    _ = d_x^
    _ = d_part^
    _ = d_flags^
    _ = d_limbs^
    _ = h_limbs^
    return ok


def nw_chunk_kernel(
    w: MutPointer[UInt32, MutAnyOrigin], n: Int64, n_chunks: Int64, part: MutPointer[UInt64, MutAnyOrigin],
    flag: MutPointer[Int32, MutAnyOrigin],
):
    """One thread per chunk of `FOLD_CHUNK` rows (xtrees/fold_order.mojo's
    CHUNK): the widened weights summed in index order from +0; flag[0] = 1
    on a NaN, infinite or negative weight."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while c < Int(n_chunks):
        var run = SF64_ZERO
        var i1 = min((c + 1) * FOLD_CHUNK, Int(n))
        for i in range(c * FOLD_CHUNK, i1):
            var bits = w.unsafe_load(i)
            var mag = bits & UInt32(0x7FFFFFFF)
            if mag >= UInt32(0x7F800000) or ((bits >> 31) != 0 and mag != 0):
                flag.unsafe_store(0, Int32(1))
            run = sf64_add(run, sf64_from_f32(bitcast[DType.float32](bits)))
        part.unsafe_store(c, run)
        c += stride


def nw_divide_kernel(
    w: MutPointer[UInt32, MutAnyOrigin], n: Int64, part: MutPointer[UInt64, MutAnyOrigin],
    res: MutPointer[UInt64, MutAnyOrigin],
):
    """res[i] = float64(w[i]) / total (total = part[0]), correctly rounded."""
    var t = part.unsafe_load(0)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        res.unsafe_store(i, sf64_div(sf64_from_f32(bitcast[DType.float32](w.unsafe_load(i))), t))
        i += stride


def normalized_weights_device(
    w: MutPointer[Float32, MutUntrackedOrigin], n: Int, res: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """`ops.normalized_weights` on the device: the chunk sums, the fixed
    pairwise tree over the chunk partials (`fold_order.fold_tree_device`),
    then one division per row. The same status (0; 1 a bad entry; 2 no positive total, `res` untouched)."""
    if n <= 0:
        return 2
    var ctx = _ctx()
    var n_chunks = fold_chunks(n)
    var d_w = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=w.bitcast[UInt32]())
    var d_part = ctx.enqueue_create_buffer[DType.uint64](n_chunks)
    var d_flag = _flags(ctx, 1)
    ctx.enqueue_function[nw_chunk_kernel](
        d_w.unsafe_ptr(), Int64(n), Int64(n_chunks), d_part.unsafe_ptr(), d_flag.unsafe_ptr(),
        grid_dim=_blocks(n_chunks), block_dim=OPS_TPB,
    )
    fold_tree_device(ctx, d_part, n_chunks, 1)
    var h_tot = ctx.enqueue_create_host_buffer[DType.uint64](1)
    var d_first = d_part.create_sub_buffer[DType.uint64](0, 1)
    ctx.enqueue_copy(dst_buf=h_tot, src_buf=d_first)
    var fl = _read_i32(ctx, d_flag, 1)
    var total = h_tot.unsafe_ptr().unsafe_load(0)
    var status = 0
    if fl[0] != 0:
        status = 1
    elif not _gt64(total, SF64_ZERO):
        status = 2
    else:
        var d_res = ctx.enqueue_create_buffer[DType.uint64](n)
        ctx.enqueue_function[nw_divide_kernel](
            d_w.unsafe_ptr(), Int64(n), d_part.unsafe_ptr(), d_res.unsafe_ptr(), grid_dim=_blocks(n), block_dim=OPS_TPB,
        )
        ctx.enqueue_copy(dst_ptr=res.bitcast[UInt64](), src_buf=d_res)
        ctx.synchronize()
        _ = d_res^
    _ = d_w^
    _ = d_part^
    _ = d_flag^
    _ = h_tot^
    _ = d_first^
    return status


# ------------------------------------------------------ sample_indices --


def si_replace_kernel(res: MutPointer[Int32, MutAnyOrigin], n_draw: Int64, base: UInt64, n_pool: UInt64):
    """With replacement: res[k] = draw(base, k) mod n_pool (DEVIATION 5600)."""
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while k < Int(n_draw):
        res.unsafe_store(k, Int32(Int(draw(base, k) % n_pool)))
        k += stride


def si_key_kernel(key: MutPointer[UInt64, MutAnyOrigin], n: Int64, base: UInt64):
    """key[i] = draw(base, i) >> 11 (53 bits)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        key.unsafe_store(i, draw(base, i) >> 11)
        i += stride


def si_hist_kernel(
    key: MutPointer[UInt64, MutAnyOrigin], n: Int64, prefix: MutPointer[UInt64, MutAnyOrigin], pass_: Int64,
    hist: MutPointer[Int32, MutAnyOrigin],
):
    """Radix pass `pass_` (digit shift 48 - 8 pass_): every key whose bits
    above the digit equal the selected prefix's adds 1 to its digit's bin
    (integer atomics, order-free)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var sh = UInt64(48 - 8 * Int(pass_))
    var pre = prefix.unsafe_load(Int(pass_))
    while i < Int(n):
        var k = key.unsafe_load(i)
        if (k >> (sh + 8)) == (pre >> (sh + 8)):
            _ = Atomic.fetch_add(hist.unsafe_offset(Int(pass_) * SI_DIGITS + Int((k >> sh) & UInt64(0xFF))), Int32(1))
        i += stride


def si_pick_kernel(
    hist: MutPointer[Int32, MutAnyOrigin], prefix: MutPointer[UInt64, MutAnyOrigin],
    want: MutPointer[Int32, MutAnyOrigin], pass_: Int64,
):
    """One block of `SI_DIGITS` threads: the bin b whose cumulative count
    first reaches `want` (the rank still sought) extends the prefix by b and
    leaves want - (count below b) for the next pass. State is double
    buffered (pass p reads slot p, the one picking thread writes slot
    p + 1), so no thread reads a word another writes."""
    var b = Int(thread_idx.x)
    if b >= SI_DIGITS:
        return
    var p = Int(pass_)
    var wnt = Int(want.unsafe_load(p))
    var cum = 0
    for q in range(b):
        cum += Int(hist.unsafe_load(p * SI_DIGITS + q))
    var h = Int(hist.unsafe_load(p * SI_DIGITS + b))
    if cum < wnt and wnt <= cum + h:
        prefix.unsafe_store(p + 1, prefix.unsafe_load(p) | (UInt64(b) << UInt64(48 - 8 * p)))
        want.unsafe_store(p + 1, Int32(wnt - cum))


def si_tie_count_kernel(
    key: MutPointer[UInt64, MutAnyOrigin], n: Int64, n_chunks: Int64, prefix: MutPointer[UInt64, MutAnyOrigin],
    cnt: MutPointer[Int32, MutAnyOrigin],
):
    """One thread per chunk: how many keys equal the selected key."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var kstar = prefix.unsafe_load(SI_PASSES)
    while c < Int(n_chunks):
        var k = Int32(0)
        for i in range(c * WS_CHUNK, min((c + 1) * WS_CHUNK, Int(n))):
            if key.unsafe_load(i) == kstar:
                k += 1
        cnt.unsafe_store(c, k)
        c += stride


def si_mark_kernel(
    key: MutPointer[UInt64, MutAnyOrigin], n: Int64, n_chunks: Int64, prefix: MutPointer[UInt64, MutAnyOrigin],
    want: MutPointer[Int32, MutAnyOrigin], scan: MutPointer[Int32, MutAnyOrigin], keep: MutPointer[Int32, MutAnyOrigin],
):
    """One thread per chunk: keep[i] = key < K*, or key == K* and its rank
    among the ties (index order, from the scanned chunk counts) is below the
    ties still wanted."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var kstar = prefix.unsafe_load(SI_PASSES)
    var wnt = Int(want.unsafe_load(SI_PASSES))
    while c < Int(n_chunks):
        var r = 0 if c == 0 else Int(scan.unsafe_load(c - 1))
        for i in range(c * WS_CHUNK, min((c + 1) * WS_CHUNK, Int(n))):
            var k = key.unsafe_load(i)
            var kp = Int32(0)
            if k < kstar:
                kp = Int32(1)
            elif k == kstar:
                if r < wnt:
                    kp = Int32(1)
                r += 1
            keep.unsafe_store(i, kp)
        c += stride


def sample_indices_device(
    res: MutPointer[Int32, MutUntrackedOrigin], n_pool: Int, n_draw: Int, replace: Bool, seed: Int, stream: Int,
) raises:
    """`ops.sample_indices` on the device, the same law and refusal. With
    replacement: one thread per draw. Without: the n_draw smallest
    (key, index) pairs, key_i = draw(base, i) >> 11, in ascending index
    order. An exact radix select (seven 8-bit digit passes over the 53-bit
    keys, integer-atomic histograms, a 256-thread pick per pass) finds the
    n_draw-th smallest key K* and how many of its ties to take; the ties go
    by index through chunk counts and a Hillis-Steele scan; the kept rows
    are compacted in index order (`ops_device._compact_device`)."""
    if n_pool <= 0 or n_draw < 0 or (not replace and n_draw > n_pool):
        raise Error("x_trees sample_indices: need n_pool > 0 and 0 <= n_draw (<= n_pool without replacement)")
    if n_draw == 0:
        return
    var ctx = _ctx()
    var base = stream_base(seed, stream)
    if replace:
        var d_res = ctx.enqueue_create_buffer[DType.int32](n_draw)
        ctx.enqueue_function[si_replace_kernel](
            d_res.unsafe_ptr(), Int64(n_draw), base, UInt64(n_pool), grid_dim=_blocks(n_draw), block_dim=OPS_TPB,
        )
        ctx.enqueue_copy(dst_ptr=res, src_buf=d_res)
        ctx.synchronize()
        _ = d_res^
        return
    var n = n_pool
    var d_key = ctx.enqueue_create_buffer[DType.uint64](n)
    ctx.enqueue_function[si_key_kernel](d_key.unsafe_ptr(), Int64(n), base, grid_dim=_blocks(n), block_dim=OPS_TPB)
    var d_hist = ctx.enqueue_create_buffer[DType.int32](SI_PASSES * SI_DIGITS)
    d_hist.enqueue_fill(Int32(0))
    var d_prefix = ctx.enqueue_create_buffer[DType.uint64](SI_PASSES + 1)
    d_prefix.enqueue_fill(UInt64(0))
    var d_want = ctx.enqueue_create_buffer[DType.int32](SI_PASSES + 1)
    d_want.enqueue_fill(Int32(n_draw))
    for p in range(SI_PASSES):
        ctx.enqueue_function[si_hist_kernel](
            d_key.unsafe_ptr(), Int64(n), d_prefix.unsafe_ptr(), Int64(p), d_hist.unsafe_ptr(),
            grid_dim=_blocks(n), block_dim=OPS_TPB,
        )
        ctx.enqueue_function[si_pick_kernel](
            d_hist.unsafe_ptr(), d_prefix.unsafe_ptr(), d_want.unsafe_ptr(), Int64(p), grid_dim=1, block_dim=SI_DIGITS,
        )
    var n_chunks = ceildiv(n, WS_CHUNK)
    var d_a = ctx.enqueue_create_buffer[DType.int32](n_chunks)
    var d_b = ctx.enqueue_create_buffer[DType.int32](n_chunks)
    ctx.enqueue_function[si_tie_count_kernel](
        d_key.unsafe_ptr(), Int64(n), Int64(n_chunks), d_prefix.unsafe_ptr(), d_a.unsafe_ptr(),
        grid_dim=_blocks(n_chunks), block_dim=OPS_TPB,
    )
    var src_is_a = True
    var s = 1
    while s < n_chunks:
        if src_is_a:
            ctx.enqueue_function[cnt_scan_step_kernel](
                d_a.unsafe_ptr(), d_b.unsafe_ptr(), Int64(n_chunks), Int64(s), grid_dim=_blocks(n_chunks), block_dim=OPS_TPB,
            )
        else:
            ctx.enqueue_function[cnt_scan_step_kernel](
                d_b.unsafe_ptr(), d_a.unsafe_ptr(), Int64(n_chunks), Int64(s), grid_dim=_blocks(n_chunks), block_dim=OPS_TPB,
            )
        src_is_a = not src_is_a
        s *= 2
    var d_keep = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[si_mark_kernel](
        d_key.unsafe_ptr(), Int64(n), Int64(n_chunks), d_prefix.unsafe_ptr(), d_want.unsafe_ptr(),
        d_a.unsafe_ptr() if src_is_a else d_b.unsafe_ptr(), d_keep.unsafe_ptr(),
        grid_dim=_blocks(n_chunks), block_dim=OPS_TPB,
    )
    var got = _compact_device(ctx, d_keep, n, res, d_key, False)
    _ = d_key^
    _ = d_hist^
    _ = d_prefix^
    _ = d_want^
    _ = d_a^
    _ = d_b^
    _ = d_keep^
    if got != n_draw:
        raise Error("x_trees sample_indices: device selection kept " + String(got) + " rows, wanted " + String(n_draw))
