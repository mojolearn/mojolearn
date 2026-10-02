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
from std.ffi import _Global
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.column_stats import CUDA_MAX_GRID_YZ, TRANSPOSE_TILE, transpose_kernel

comptime OPS_TPB = 256
comptime OPS_MAX_BLOCKS = 65535

#: `apply_trees`' per-cell refusal codes (negative so a leaf, >= 0, is never
#: one): the serial loop's three messages and its empty-tree refusal.
comptime APPLY_BAD_COLUMN = Int32(-1)
comptime APPLY_BAD_CHILD = Int32(-2)
comptime APPLY_CYCLE = Int32(-3)


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
    refuses stores its code (`APPLY_*`) in the cell and raises `flag`."""
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
            flag.unsafe_store(0, Int32(1))
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
    for t in range(t0, t1):
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
    d_flag.enqueue_fill(Int32(0))
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
    var bad = h_flag.unsafe_ptr().unsafe_load(0) != Int32(0)
    _ = d_off^
    _ = d_col^
    _ = d_q^
    _ = d_left^
    _ = d_x^
    _ = d_res^
    _ = d_flag^
    _ = h_flag^
    if bad:
        # error path only: the serial loop's first refusal, tree then row
        for t in range(nt):
            for i in range(n):
                var code = res[unsafe_offset=i * nt + t]
                if code == APPLY_BAD_COLUMN:
                    raise Error("x_trees apply: split column out of range")
                if code == APPLY_BAD_CHILD:
                    raise Error("x_trees apply: child out of range")
                if code == APPLY_CYCLE:
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
