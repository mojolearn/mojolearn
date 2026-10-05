# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The trees lane's ensemble glue on the device (cpu-gpu-cleanup t-gbdt).

`xtrees/ops.mojo`'s row gather, tree apply and transpose ran on the host
thread pool in the GPU binding (DEVIATIONS 5606, 5608). A GPU install now
runs them here: the caller's arrays go up once, one grid-wide kernel does
the work, the result comes back once. `xtrees/api.mojo` picks these on every
GPU build and the serial host loops of `ops.mojo` on the CPU column
(`TARGET_COLUMN == COLUMN_CPU`), so no GPU install reaches a host walk.

BITS. All three are copies or float COMPARES (no arithmetic): a gathered
cell is the source word, a transposed cell is the source word, and a leaf is
decided by `x <= quesval` along the same tree walk (`ops._apply_row`'s
rules: equality left, children `left` and `left + 1`). The device answers
are the serial loop's on every vendor.

REFUSALS. Every index the host loops range-checked is checked by the kernel
that reads it; a refusal raises the serial loop's message. `apply_trees`
reports the FIRST bad (tree, row) in the serial loop's order (tree, then
row) from the per-cell codes it reads back, so the message is the one the
serial loop raised.
"""
from std.atomic import Atomic
from std.ffi import _Global
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.column_stats import CUDA_MAX_GRID_YZ, TRANSPOSE_TILE, transpose_kernel
from checks.soft_f64 import SF64_SIGN, SF64_ZERO, sf64_add, sf64_from_int, sf64_gt, sf64_is_nan, sf64_lt, sf64_mul
from xtrees.ops import WS_CHUNK, draw, stream_base

comptime OPS_TPB = 256
comptime OPS_MAX_BLOCKS = 65535

#: `apply_trees`' per-cell refusal codes (negative so a leaf, >= 0, is never
#: one): the serial loop's three messages and its empty-tree refusal.
comptime APPLY_BAD_COLUMN = Int32(-1)
comptime APPLY_BAD_CHILD = Int32(-2)
comptime APPLY_CYCLE = Int32(-3)
comptime APPLY_NO_REFUSAL = Int32(2147483647)

#: 2^-53 as a binary64 word: `ops.unit`'s scale (exact).
comptime _TWO_M53 = UInt64(0x3CA0000000000000)


struct _OpsContext(Defaultable, Movable):
    """ONE process-lifetime DeviceContext (`xtrees/shap_device.mojo`'s
    pattern: a context per call exhausts Metal's command queues)."""
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime _CTX_NAME = "MojoXTreesOpsContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXTreesOpsContextFast"
comptime X_TREES_OPS_CONTEXT = _Global[StorageType=_OpsContext, name=_CTX_NAME, init_fn=_OpsContext.__init__]


def _ctx() raises -> DeviceContext:
    var slot = X_TREES_OPS_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


def _blocks(units: Int) -> Int:
    return max(1, min(ceildiv(units, OPS_TPB), OPS_MAX_BLOCKS))


def gather_index_check_kernel(
    rows: MutPointer[Int32, MutAnyOrigin], n_rows: Int64, n_src_rows: Int64,
    cols: MutPointer[Int32, MutAnyOrigin], n_cols: Int64, n_src_cols: Int64,
    flags: MutPointer[Int32, MutAnyOrigin],
):
    """flags[0] = 1 on a row index outside [0, n_src_rows), flags[1] = 1 on
    a column index outside [0, n_src_cols), flags[2] = 1 when some column is
    not the identity (`gather_f32`'s `all_cols` test). Idempotent stores."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var k = i
    while k < Int(n_rows):
        var r = Int(rows.unsafe_load(k))
        if r < 0 or r >= Int(n_src_rows):
            flags.unsafe_store(0, Int32(1))
        k += stride
    k = i
    while k < Int(n_cols):
        var c = Int(cols.unsafe_load(k))
        if c < 0 or c >= Int(n_src_cols):
            flags.unsafe_store(1, Int32(1))
        if c != k:
            flags.unsafe_store(2, Int32(1))
        k += stride


def gather_cells_kernel(
    src: MutPointer[Float32, MutAnyOrigin], n_src_cols: Int64,
    rows: MutPointer[Int32, MutAnyOrigin], n_rows: Int64,
    cols: MutPointer[Int32, MutAnyOrigin], n_cols: Int64,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """dst[r, c] = src[rows[r], cols[c]], one thread per destination cell,
    grid-stride. A copy."""
    var total = Int(n_rows) * Int(n_cols)
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var nc = Int(n_cols)
    while k < total:
        var r = k // nc
        var c = k - r * nc
        var i = Int(rows.unsafe_load(r))
        var j = Int(cols.unsafe_load(c))
        dst.unsafe_store(k, src.unsafe_load(i * Int(n_src_cols) + j))
        k += stride


def gather_f32_device(
    src: MutPointer[Float32, MutUntrackedOrigin], n_src_rows: Int, n_src_cols: Int,
    rows: MutPointer[Int32, MutUntrackedOrigin], n_rows: Int,
    cols: MutPointer[Int32, MutUntrackedOrigin], n_cols: Int,
    dst: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`ops.gather_f32` on the device: the same cells, the same refusals
    (a bad row index first, then a bad column index), nothing written to
    `dst` on a refusal."""
    if n_rows <= 0 or n_cols <= 0:
        return
    var ctx = _ctx()
    var n_src = max(1, n_src_rows * n_src_cols)
    var d_src = ctx.enqueue_create_buffer[DType.float32](n_src)
    if n_src_rows * n_src_cols > 0:
        ctx.enqueue_copy(dst_buf=d_src, src_ptr=src)
    var d_rows = ctx.enqueue_create_buffer[DType.int32](n_rows)
    ctx.enqueue_copy(dst_buf=d_rows, src_ptr=rows)
    var d_cols = ctx.enqueue_create_buffer[DType.int32](n_cols)
    ctx.enqueue_copy(dst_buf=d_cols, src_ptr=cols)
    var d_flags = ctx.enqueue_create_buffer[DType.int32](3)
    d_flags.enqueue_fill(Int32(0))
    ctx.enqueue_function[gather_index_check_kernel](
        d_rows.unsafe_ptr(), Int64(n_rows), Int64(n_src_rows),
        d_cols.unsafe_ptr(), Int64(n_cols), Int64(n_src_cols),
        d_flags.unsafe_ptr(),
        grid_dim=_blocks(max(n_rows, n_cols)), block_dim=OPS_TPB,
    )
    var h_flags = ctx.enqueue_create_host_buffer[DType.int32](3)
    ctx.enqueue_copy(dst_buf=h_flags, src_buf=d_flags)
    ctx.synchronize()
    var fp = h_flags.unsafe_ptr()
    if fp.unsafe_load(0) != Int32(0):
        raise Error("x_trees gather: row index out of range")
    if fp.unsafe_load(1) != Int32(0):
        raise Error("x_trees gather: column index out of range")
    var d_dst = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    ctx.enqueue_function[gather_cells_kernel](
        d_src.unsafe_ptr(), Int64(n_src_cols),
        d_rows.unsafe_ptr(), Int64(n_rows),
        d_cols.unsafe_ptr(), Int64(n_cols),
        d_dst.unsafe_ptr(),
        grid_dim=_blocks(n_rows * n_cols), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=dst, src_buf=d_dst)
    ctx.synchronize()
    _ = d_src^
    _ = d_rows^
    _ = d_cols^
    _ = d_flags^
    _ = h_flags^
    _ = d_dst^


def apply_cells_kernel(
    offsets: MutPointer[Int32, MutAnyOrigin], colid: MutPointer[Int32, MutAnyOrigin],
    quesval: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin], n: Int64, d: Int64, t0: Int64, nt: Int64,
    res: MutPointer[Int32, MutAnyOrigin], flag: MutPointer[Int32, MutAnyOrigin],
):
    """res[i * nt + (t - t0)] = the tree-relative leaf row i reaches in tree
    t, one thread per (row, tree), grid-stride; a walk the serial loop
    refuses stores its code (`APPLY_*`) in the cell and lowers `flag` to
    its serial-order key `toff * n + i`."""
    var total = Int(n) * Int(nt)
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while k < total:
        var i = k // Int(nt)
        var toff = k - i * Int(nt)
        var t = Int(t0) + toff
        var lo = Int(offsets.unsafe_load(t))
        var count = Int(offsets.unsafe_load(t + 1)) - lo
        var node = 0
        var steps = 0
        var code = Int32(0)
        while left.unsafe_load(lo + node) != -1:
            var c = Int(colid.unsafe_load(lo + node))
            if c < 0 or c >= Int(d):
                code = APPLY_BAD_COLUMN
                break
            var l = Int(left.unsafe_load(lo + node))
            if l < 1 or l + 1 >= count:
                code = APPLY_BAD_CHILD
                break
            if x.unsafe_load(i * Int(d) + c) <= quesval.unsafe_load(lo + node):
                node = l
            else:
                node = l + 1
            steps += 1
            if steps > count:
                code = APPLY_CYCLE
                break
        if code != Int32(0):
            res.unsafe_store(k, code)
            # the first refusal in the serial order (tree, then row): the
            # smallest toff * n + i (lane/apple-fast-purity2)
            _ = Atomic.min(flag, Int32(toff * Int(n) + i))
        else:
            res.unsafe_store(k, Int32(node))
        k += stride


def apply_trees_device(
    offsets: MutPointer[Int32, MutUntrackedOrigin], colid: MutPointer[Int32, MutUntrackedOrigin],
    quesval: MutPointer[Float32, MutUntrackedOrigin], left: MutPointer[Int32, MutUntrackedOrigin],
    x: MutPointer[Float32, MutUntrackedOrigin], n: Int, d: Int, t0: Int, t1: Int,
    res: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """`ops.apply_trees` on the device: the same leaves, the same refusals in
    the serial loop's order (an empty tree first, by tree; then the first bad
    (tree, row) walk)."""
    var nt = t1 - t0
    if n <= 0 or nt <= 0:
        return
    for t in range(t0, t1):  # small-loop(t1: per-tree node offsets): empty-tree refusal on model metadata, no rows
        if Int(offsets[unsafe_offset=t + 1]) - Int(offsets[unsafe_offset=t]) < 1:
            raise Error("x_trees apply: empty tree")
    var n_nodes = Int(offsets[unsafe_offset=t1])
    var ctx = _ctx()
    var d_off = ctx.enqueue_create_buffer[DType.int32](t1 + 1)
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=offsets)
    var d_col = ctx.enqueue_create_buffer[DType.int32](max(1, n_nodes))
    var d_q = ctx.enqueue_create_buffer[DType.float32](max(1, n_nodes))
    var d_left = ctx.enqueue_create_buffer[DType.int32](max(1, n_nodes))
    if n_nodes > 0:
        ctx.enqueue_copy(dst_buf=d_col, src_ptr=colid)
        ctx.enqueue_copy(dst_buf=d_q, src_ptr=quesval)
        ctx.enqueue_copy(dst_buf=d_left, src_ptr=left)
    var d_x = ctx.enqueue_create_buffer[DType.float32](max(1, n * d))
    if n * d > 0:
        ctx.enqueue_copy(dst_buf=d_x, src_ptr=x)
    var d_res = ctx.enqueue_create_buffer[DType.int32](n * nt)
    var d_flag = ctx.enqueue_create_buffer[DType.int32](1)
    d_flag.enqueue_fill(APPLY_NO_REFUSAL)
    ctx.enqueue_function[apply_cells_kernel](
        d_off.unsafe_ptr(), d_col.unsafe_ptr(), d_q.unsafe_ptr(), d_left.unsafe_ptr(),
        d_x.unsafe_ptr(), Int64(n), Int64(d), Int64(t0), Int64(nt),
        d_res.unsafe_ptr(), d_flag.unsafe_ptr(),
        grid_dim=_blocks(n * nt), block_dim=OPS_TPB,
    )
    var h_flag = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=h_flag, src_buf=d_flag)
    ctx.enqueue_copy(dst_ptr=res, src_buf=d_res)
    ctx.synchronize()
    var first = h_flag.unsafe_ptr().unsafe_load(0)
    _ = d_off^
    _ = d_col^
    _ = d_q^
    _ = d_left^
    _ = d_x^
    _ = d_res^
    _ = d_flag^
    _ = h_flag^
    if first != APPLY_NO_REFUSAL:
        # the serial loop's first refusal, tree then row, found on the
        # device (the kernel's atomic min of toff * n + i): one cell read
        var t_bad = Int(first) // n
        var i_bad = Int(first) - t_bad * n
        var code = res[unsafe_offset=i_bad * nt + t_bad]
        if code == APPLY_BAD_COLUMN:
            raise Error("x_trees apply: split column out of range")
        if code == APPLY_BAD_CHILD:
            raise Error("x_trees apply: child out of range")
        raise Error("x_trees apply: cycle in tree")


def transpose_f32_device(
    src: MutPointer[Float32, MutUntrackedOrigin], n: Int, d: Int,
    dst: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`ops.transpose_f32` on the device: dst (d x n) = src (n x d)^T through
    `core/column_stats.mojo`'s tiled `transpose_kernel`. A copy."""
    if n <= 0 or d <= 0:
        return
    var ctx = _ctx()
    var d_src = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.enqueue_copy(dst_buf=d_src, src_ptr=src)
    var d_dst = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.enqueue_function[transpose_kernel](
        d_dst.unsafe_ptr(), d_src.unsafe_ptr(), Int32(n), Int32(d),
        grid_dim=(
            ceildiv(d, TRANSPOSE_TILE),
            min(ceildiv(n, TRANSPOSE_TILE), CUDA_MAX_GRID_YZ),
            1,
        ),
        block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
    )
    ctx.enqueue_copy(dst_ptr=dst, src_buf=d_dst)
    ctx.synchronize()
    _ = d_src^
    _ = d_dst^


# ---------------------------------------------------------------------------
# weighted_sample on the device (cpu-gpu-cleanup w2-trees). The binary64
# arithmetic is `checks/soft_f64.mojo`'s integer spelling (the Apple GPU has
# no float64), each operation the IEEE result the CPU column's native double
# returns, so the cdf and the draws are `ops.weighted_sample`'s on every
# vendor: the same chunks, the same in-chunk order, the same fixed tree.
# ---------------------------------------------------------------------------


def ws_chunk_kernel(
    w: MutPointer[UInt64, MutAnyOrigin], n: Int64, n_chunks: Int64,
    cdf: MutPointer[UInt64, MutAnyOrigin], tot: MutPointer[UInt64, MutAnyOrigin],
    flag: MutPointer[Int32, MutAnyOrigin],
):
    """One thread per chunk of `WS_CHUNK` rows, grid-stride: the in-chunk
    cumulative sum in index order (cdf), the chunk total (tot), flag[0] = 1
    on a negative or NaN weight."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while c < Int(n_chunks):
        var run = SF64_ZERO
        var i1 = min((c + 1) * WS_CHUNK, Int(n))
        for i in range(c * WS_CHUNK, i1):
            var wi = w.unsafe_load(i)
            if sf64_is_nan(wi) or sf64_lt(wi, SF64_ZERO):
                flag.unsafe_store(0, Int32(1))
            run = sf64_add(run, wi)
            cdf.unsafe_store(i, run)
        tot.unsafe_store(c, run)
        c += stride


def ws_scan_step_kernel(
    src: MutPointer[UInt64, MutAnyOrigin], dst: MutPointer[UInt64, MutAnyOrigin], m: Int64, s: Int64,
):
    """One Hillis-Steele pass of `ops.ws_tree_scan`: dst[j] = src[j] +
    src[j - s] for j >= s, else src[j]. One thread per chunk total."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while j < Int(m):
        if j >= Int(s):
            dst.unsafe_store(j, sf64_add(src.unsafe_load(j), src.unsafe_load(j - Int(s))))
        else:
            dst.unsafe_store(j, src.unsafe_load(j))
        j += stride


def ws_offset_kernel(
    cdf: MutPointer[UInt64, MutAnyOrigin], scan: MutPointer[UInt64, MutAnyOrigin], n: Int64,
):
    """cdf[i] = scan[chunk(i) - 1] + cdf[i] for every row past chunk 0."""
    var i = WS_CHUNK + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        cdf.unsafe_store(i, sf64_add(scan.unsafe_load(i // WS_CHUNK - 1), cdf.unsafe_load(i)))
        i += stride


def ws_draw_kernel(
    w: MutPointer[UInt64, MutAnyOrigin], cdf: MutPointer[UInt64, MutAnyOrigin], n: Int64,
    base: UInt64, n_draw: Int64, res: MutPointer[Int32, MutAnyOrigin],
):
    """One thread per draw: `ops.weighted_sample`'s draw body (u = unit *
    total, the first i with cdf[i] > u, stepped back over trailing zero
    weights)."""
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var total = cdf.unsafe_load(Int(n) - 1)
    while k < Int(n_draw):
        var r = draw(base, k)
        var u = sf64_mul(sf64_mul(sf64_from_int(Int(r >> 11)), _TWO_M53), total)
        var lo = 0
        var hi = Int(n) - 1
        while lo < hi:
            var mid = (lo + hi) // 2
            if sf64_gt(cdf.unsafe_load(mid), u):
                hi = mid
            else:
                lo = mid + 1
        while lo > 0 and (w.unsafe_load(lo) & ~SF64_SIGN) == 0:  # u landed on a flat step at the end
            lo -= 1
        res.unsafe_store(k, Int32(lo))
        k += stride


def weighted_sample_device(
    w: MutPointer[Float64, MutUntrackedOrigin], n: Int,
    res: MutPointer[Int32, MutUntrackedOrigin], n_draw: Int, seed: Int, stream: Int,
) raises:
    """`ops.weighted_sample` on the device: the weights go up once, the cdf
    is built by the chunk kernel, ceil(log2(chunks)) fixed-tree scan passes
    and the offset kernel, one thread per draw searches it, the indices come
    back once. The same refusals; nothing written to `res` on a refusal."""
    if n <= 0:
        raise Error("x_trees weighted_sample: weights must have a positive total")
    var ctx = _ctx()
    var n_chunks = ceildiv(n, WS_CHUNK)
    var d_w = ctx.enqueue_create_buffer[DType.uint64](n)
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=w.bitcast[UInt64]())
    var d_cdf = ctx.enqueue_create_buffer[DType.uint64](n)
    var d_a = ctx.enqueue_create_buffer[DType.uint64](n_chunks)
    var d_b = ctx.enqueue_create_buffer[DType.uint64](n_chunks)
    var d_flag = ctx.enqueue_create_buffer[DType.int32](1)
    d_flag.enqueue_fill(Int32(0))
    ctx.enqueue_function[ws_chunk_kernel](
        d_w.unsafe_ptr(), Int64(n), Int64(n_chunks), d_cdf.unsafe_ptr(), d_a.unsafe_ptr(), d_flag.unsafe_ptr(),
        grid_dim=_blocks(n_chunks), block_dim=OPS_TPB,
    )
    var src_is_a = True
    var s = 1
    while s < n_chunks:
        if src_is_a:
            ctx.enqueue_function[ws_scan_step_kernel](
                d_a.unsafe_ptr(), d_b.unsafe_ptr(), Int64(n_chunks), Int64(s),
                grid_dim=_blocks(n_chunks), block_dim=OPS_TPB,
            )
        else:
            ctx.enqueue_function[ws_scan_step_kernel](
                d_b.unsafe_ptr(), d_a.unsafe_ptr(), Int64(n_chunks), Int64(s),
                grid_dim=_blocks(n_chunks), block_dim=OPS_TPB,
            )
        src_is_a = not src_is_a
        s *= 2
    if n > WS_CHUNK:
        ctx.enqueue_function[ws_offset_kernel](
            d_cdf.unsafe_ptr(), d_a.unsafe_ptr() if src_is_a else d_b.unsafe_ptr(), Int64(n),
            grid_dim=_blocks(n - WS_CHUNK), block_dim=OPS_TPB,
        )
    var h_flag = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=h_flag, src_buf=d_flag)
    var h_total = ctx.enqueue_create_host_buffer[DType.uint64](1)
    var d_last = d_cdf.create_sub_buffer[DType.uint64](n - 1, 1)
    ctx.enqueue_copy(dst_buf=h_total, src_buf=d_last)
    ctx.synchronize()
    if h_flag.unsafe_ptr().unsafe_load(0) != Int32(0):
        raise Error("x_trees weighted_sample: weights must be nonnegative")
    if not sf64_gt(h_total.unsafe_ptr().unsafe_load(0), SF64_ZERO):
        raise Error("x_trees weighted_sample: weights must have a positive total")
    if n_draw > 0:
        var d_res = ctx.enqueue_create_buffer[DType.int32](n_draw)
        ctx.enqueue_function[ws_draw_kernel](
            d_w.unsafe_ptr(), d_cdf.unsafe_ptr(), Int64(n), stream_base(seed, stream), Int64(n_draw),
            d_res.unsafe_ptr(),
            grid_dim=_blocks(n_draw), block_dim=OPS_TPB,
        )
        ctx.enqueue_copy(dst_ptr=res, src_buf=d_res)
        ctx.synchronize()
        _ = d_res^
    _ = d_w^
    _ = d_cdf^
    _ = d_a^
    _ = d_b^
    _ = d_flag^
    _ = h_flag^
    _ = h_total^
    _ = d_last^


# ---------------------------------------------------------------------------
# Row compaction on the device (cpu-gpu-cleanup t-gbdt): the bagged rows of
# a DART / GOSS-free boosting round (`ops.bag_rows`) and a member's
# out-of-bag rows (`ops.unseen_rows`). A keep flag per row, one thread per
# chunk of `WS_CHUNK` rows counts its keeps, a Hillis-Steele scan of the
# chunk counts gives each chunk its offset, and each chunk writes its kept
# rows in index order: the serial loop's list, ascending. Integer work only.
# ---------------------------------------------------------------------------


def unseen_mark_kernel(
    keep: MutPointer[Int32, MutAnyOrigin], rows: MutPointer[Int32, MutAnyOrigin], m: Int64, n: Int64,
    flag: MutPointer[Int32, MutAnyOrigin],
):
    """keep[rows[r]] = 0 for every drawn row (the same word from every
    writer); flag[0] = 1 on a row out of range."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while r < Int(m):
        var i = Int(rows.unsafe_load(r))
        if i < 0 or i >= Int(n):
            flag.unsafe_store(0, Int32(1))
        else:
            keep.unsafe_store(i, Int32(0))
        r += stride


def bag_mark_kernel(
    keep: MutPointer[Int32, MutAnyOrigin], key: MutPointer[UInt64, MutAnyOrigin], n: Int64, base: UInt64,
    frac: UInt64,
):
    """keep[i] = unit(draw(base, i)) < frac (`ops.unit` in binary64 words:
    the top 53 bits times 2^-53, exact); key[i] = the top 53 bits, the
    fallback's order (`unit` is monotone in them)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n):
        var top = draw(base, i) >> 11
        var u = sf64_mul(sf64_from_int(Int(top)), _TWO_M53)
        keep.unsafe_store(i, Int32(1) if sf64_lt(u, frac) else Int32(0))
        key.unsafe_store(i, top)
        i += stride


def cmp_chunk_kernel(
    keep: MutPointer[Int32, MutAnyOrigin], n: Int64, n_chunks: Int64, cnt: MutPointer[Int32, MutAnyOrigin],
    key: MutPointer[UInt64, MutAnyOrigin], has_key: Int32,
    ck: MutPointer[UInt64, MutAnyOrigin], ci: MutPointer[Int32, MutAnyOrigin],
):
    """One thread per chunk: its keep count, and (has_key) its smallest
    (key, row) in index order."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while c < Int(n_chunks):
        var i1 = min((c + 1) * WS_CHUNK, Int(n))
        var k = Int32(0)
        var bk = UInt64.MAX
        var bi = Int32(c * WS_CHUNK)
        for i in range(c * WS_CHUNK, i1):
            k += keep.unsafe_load(i)
            if has_key != 0:
                var ki = key.unsafe_load(i)
                if ki < bk:
                    bk = ki
                    bi = Int32(i)
        cnt.unsafe_store(c, k)
        if has_key != 0:
            ck.unsafe_store(c, bk)
            ci.unsafe_store(c, bi)
        c += stride


def cnt_scan_step_kernel(
    src: MutPointer[Int32, MutAnyOrigin], dst: MutPointer[Int32, MutAnyOrigin], m: Int64, s: Int64,
):
    """One Hillis-Steele pass over the chunk counts (inclusive)."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while j < Int(m):
        if j >= Int(s):
            dst.unsafe_store(j, src.unsafe_load(j) + src.unsafe_load(j - Int(s)))
        else:
            dst.unsafe_store(j, src.unsafe_load(j))
        j += stride


def argmin_scan_step_kernel(
    sk: MutPointer[UInt64, MutAnyOrigin], si: MutPointer[Int32, MutAnyOrigin],
    dk: MutPointer[UInt64, MutAnyOrigin], di: MutPointer[Int32, MutAnyOrigin], m: Int64, s: Int64,
):
    """One Hillis-Steele pass of the prefix (key, row) minimum: the
    lexicographic minimum is unique, so the last cell is the serial scan's."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while j < Int(m):
        var k = sk.unsafe_load(j)
        var i = si.unsafe_load(j)
        if j >= Int(s):
            var k2 = sk.unsafe_load(j - Int(s))
            var i2 = si.unsafe_load(j - Int(s))
            if k2 < k or (k2 == k and i2 < i):
                k = k2
                i = i2
        dk.unsafe_store(j, k)
        di.unsafe_store(j, i)
        j += stride


def cmp_write_kernel(
    keep: MutPointer[Int32, MutAnyOrigin], n: Int64, n_chunks: Int64, scan: MutPointer[Int32, MutAnyOrigin],
    res: MutPointer[Int32, MutAnyOrigin],
):
    """One thread per chunk writes its kept rows from its scanned offset."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while c < Int(n_chunks):
        var off = 0 if c == 0 else Int(scan.unsafe_load(c - 1))
        var i1 = min((c + 1) * WS_CHUNK, Int(n))
        for i in range(c * WS_CHUNK, i1):
            if keep.unsafe_load(i) != 0:
                res.unsafe_store(off, Int32(i))
                off += 1
        c += stride


def _compact_device(
    ctx: DeviceContext, d_keep: DeviceBuffer[DType.int32], n: Int, res: MutPointer[Int32, MutUntrackedOrigin],
    d_key: DeviceBuffer[DType.uint64], has_key: Bool,
) raises -> Int:
    """The kept rows of `d_keep` into `res`, ascending; returns the count.
    With `has_key` and nothing kept, `res[0]` is the smallest (key, row)
    and the count is 1."""
    var n_chunks = ceildiv(n, WS_CHUNK)
    var d_a = ctx.enqueue_create_buffer[DType.int32](n_chunks)
    var d_b = ctx.enqueue_create_buffer[DType.int32](n_chunks)
    var d_ka = ctx.enqueue_create_buffer[DType.uint64](n_chunks)
    var d_kb = ctx.enqueue_create_buffer[DType.uint64](n_chunks)
    var d_ia = ctx.enqueue_create_buffer[DType.int32](n_chunks)
    var d_ib = ctx.enqueue_create_buffer[DType.int32](n_chunks)
    ctx.enqueue_function[cmp_chunk_kernel](
        rebind[MutPointer[Int32, MutAnyOrigin]](d_keep.unsafe_ptr()), Int64(n), Int64(n_chunks), rebind[MutPointer[Int32, MutAnyOrigin]](d_a.unsafe_ptr()),
        rebind[MutPointer[UInt64, MutAnyOrigin]](d_key.unsafe_ptr()), Int32(1 if has_key else 0), rebind[MutPointer[UInt64, MutAnyOrigin]](d_ka.unsafe_ptr()), rebind[MutPointer[Int32, MutAnyOrigin]](d_ia.unsafe_ptr()),
        grid_dim=_blocks(n_chunks), block_dim=OPS_TPB,
    )
    var src_is_a = True
    var s = 1
    while s < n_chunks:
        var cs = rebind[MutPointer[Int32, MutAnyOrigin]](d_a.unsafe_ptr()) if src_is_a else rebind[MutPointer[Int32, MutAnyOrigin]](d_b.unsafe_ptr())
        var cd = rebind[MutPointer[Int32, MutAnyOrigin]](d_b.unsafe_ptr()) if src_is_a else rebind[MutPointer[Int32, MutAnyOrigin]](d_a.unsafe_ptr())
        ctx.enqueue_function[cnt_scan_step_kernel](
            cs, cd, Int64(n_chunks), Int64(s), grid_dim=_blocks(n_chunks), block_dim=OPS_TPB,
        )
        if has_key:
            ctx.enqueue_function[argmin_scan_step_kernel](
                rebind[MutPointer[UInt64, MutAnyOrigin]](d_ka.unsafe_ptr()) if src_is_a else rebind[MutPointer[UInt64, MutAnyOrigin]](d_kb.unsafe_ptr()),
                rebind[MutPointer[Int32, MutAnyOrigin]](d_ia.unsafe_ptr()) if src_is_a else rebind[MutPointer[Int32, MutAnyOrigin]](d_ib.unsafe_ptr()),
                rebind[MutPointer[UInt64, MutAnyOrigin]](d_kb.unsafe_ptr()) if src_is_a else rebind[MutPointer[UInt64, MutAnyOrigin]](d_ka.unsafe_ptr()),
                rebind[MutPointer[Int32, MutAnyOrigin]](d_ib.unsafe_ptr()) if src_is_a else rebind[MutPointer[Int32, MutAnyOrigin]](d_ia.unsafe_ptr()),
                Int64(n_chunks), Int64(s), grid_dim=_blocks(n_chunks), block_dim=OPS_TPB,
            )
        src_is_a = not src_is_a
        s *= 2
    var p_scan = rebind[MutPointer[Int32, MutAnyOrigin]](d_a.unsafe_ptr()) if src_is_a else rebind[MutPointer[Int32, MutAnyOrigin]](d_b.unsafe_ptr())
    var d_res = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[cmp_write_kernel](
        rebind[MutPointer[Int32, MutAnyOrigin]](d_keep.unsafe_ptr()), Int64(n), Int64(n_chunks), p_scan, rebind[MutPointer[Int32, MutAnyOrigin]](d_res.unsafe_ptr()),
        grid_dim=_blocks(n_chunks), block_dim=OPS_TPB,
    )
    # the two scalars: the kept count and the fallback row
    var h_tot = ctx.enqueue_create_host_buffer[DType.int32](1)
    var h_best = ctx.enqueue_create_host_buffer[DType.int32](1)
    var sub_tot = d_a.create_sub_buffer[DType.int32](n_chunks - 1, 1) if src_is_a else d_b.create_sub_buffer[DType.int32](n_chunks - 1, 1)
    var sub_best = d_ia.create_sub_buffer[DType.int32](n_chunks - 1, 1) if src_is_a else d_ib.create_sub_buffer[DType.int32](n_chunks - 1, 1)
    ctx.enqueue_copy(dst_buf=h_tot, src_buf=sub_tot)
    ctx.enqueue_copy(dst_buf=h_best, src_buf=sub_best)
    ctx.synchronize()
    var total = Int(h_tot.unsafe_ptr().unsafe_load(0))
    if total > 0:
        var sub_res = d_res.create_sub_buffer[DType.int32](0, total)
        ctx.enqueue_copy(dst_ptr=res, src_buf=sub_res)
        ctx.synchronize()
        _ = sub_res^
    elif has_key:
        res.unsafe_store(0, h_best.unsafe_ptr().unsafe_load(0))
        total = 1
    _ = d_a^
    _ = d_b^
    _ = d_ka^
    _ = d_kb^
    _ = d_ia^
    _ = d_ib^
    _ = d_res^
    _ = h_tot^
    _ = h_best^
    _ = sub_tot^
    _ = sub_best^
    return total


def bag_rows_device(
    res: MutPointer[Int32, MutUntrackedOrigin], n: Int, seed: Int, stream: Int, frac: Float64,
) raises -> Int:
    """`ops.bag_rows` on the device: the draws are made where they are
    compared (no host draw list), the kept rows come back once."""
    if n <= 0:
        return 0
    var ctx = _ctx()
    var d_keep = ctx.enqueue_create_buffer[DType.int32](n)
    var d_key = ctx.enqueue_create_buffer[DType.uint64](n)
    ctx.enqueue_function[bag_mark_kernel](
        d_keep.unsafe_ptr(), d_key.unsafe_ptr(), Int64(n), stream_base(seed, stream),
        bitcast[DType.uint64](frac), grid_dim=_blocks(n), block_dim=OPS_TPB,
    )
    var k = _compact_device(ctx, d_keep, n, res, d_key, True)
    _ = d_keep^
    _ = d_key^
    return k


def unseen_rows_device(
    rows: MutPointer[Int32, MutUntrackedOrigin], m: Int, n: Int, res: MutPointer[Int32, MutUntrackedOrigin],
) raises -> Int:
    """`ops.unseen_rows` on the device: the drawn rows go up once, every row
    starts kept and each draw clears its row, the kept rows come back once."""
    if n <= 0:
        return 0
    var ctx = _ctx()
    var d_keep = ctx.enqueue_create_buffer[DType.int32](n)
    d_keep.enqueue_fill(Int32(1))
    var d_flag = ctx.enqueue_create_buffer[DType.int32](1)
    d_flag.enqueue_fill(Int32(0))
    var d_key = ctx.enqueue_create_buffer[DType.uint64](1)
    if m > 0:
        var d_rows = ctx.enqueue_create_buffer[DType.int32](m)
        ctx.enqueue_copy(dst_buf=d_rows, src_ptr=rows)
        ctx.enqueue_function[unseen_mark_kernel](
            d_keep.unsafe_ptr(), d_rows.unsafe_ptr(), Int64(m), Int64(n), d_flag.unsafe_ptr(),
            grid_dim=_blocks(m), block_dim=OPS_TPB,
        )
        var h_flag = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_copy(dst_buf=h_flag, src_buf=d_flag)
        ctx.synchronize()
        if h_flag.unsafe_ptr().unsafe_load(0) != Int32(0):
            raise Error("x_trees unseen_rows: row out of range")
        _ = d_rows^
        _ = h_flag^
    var k = _compact_device(ctx, d_keep, n, res, d_key, False)
    _ = d_keep^
    _ = d_flag^
    _ = d_key^
    return k


def transpose_f64_kernel(
    dst: MutPointer[UInt64, MutAnyOrigin], src: MutPointer[UInt64, MutAnyOrigin], n: Int64, d: Int64,
):
    """dst[j * n + i] = src[i * d + j], one thread per cell (a word copy)."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var total = Int(n) * Int(d)
    while c < total:
        var i = c // Int(d)
        var j = c - i * Int(d)
        dst.unsafe_store(j * Int(n) + i, src.unsafe_load(c))
        c += stride


def transpose_f64_device(
    src: MutPointer[Float64, MutUntrackedOrigin], n: Int, d: Int,
    dst: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """`ops.transpose_f64` on the device: binary64 words moved, not computed
    (the Apple GPU has no float64; a word is a word)."""
    if n <= 0 or d <= 0:
        return
    var ctx = _ctx()
    var d_src = ctx.enqueue_create_buffer[DType.uint64](n * d)
    ctx.enqueue_copy(dst_buf=d_src, src_ptr=src.bitcast[UInt64]())
    var d_dst = ctx.enqueue_create_buffer[DType.uint64](n * d)
    ctx.enqueue_function[transpose_f64_kernel](
        d_dst.unsafe_ptr(), d_src.unsafe_ptr(), Int64(n), Int64(d),
        grid_dim=_blocks(n * d), block_dim=OPS_TPB,
    )
    ctx.enqueue_copy(dst_ptr=dst.bitcast[UInt64](), src_buf=d_dst)
    ctx.synchronize()
    _ = d_src^
    _ = d_dst^
