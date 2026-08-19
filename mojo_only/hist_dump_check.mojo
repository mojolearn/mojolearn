"""Read a leaf's histogram back at two depths and compare the bytes.

The open bug: the level score is identical from depth 1 onward, which means
the score kernel reads identical histograms. Every inference so far has been
wrong in an interesting way, so this stops inferring and reads the actual
cells.

It rebuilds the level pipeline by hand rather than calling `run_tree`,
because what is needed is the histogram AFTER a known partition, and
`run_tree` does not surrender it.
"""

from max.gpu.host import DeviceContext

from ported.gpu_data.grid_policy import (
    POLICY_BINARY,
    features_per_int,
    policy_mask,
    policy_shift,
)
from ported.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.hist_binary import (
    binary_hist_gather_kernel,
    binary_hist_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
)


def check_hist_depends_on_partition() raises:
    """Does the histogram change when the PARTITION changes?

    The same rows, the same bins, the same stats, built twice: once as ONE
    leaf holding all 1024 rows, then as TWO leaves splitting them in half.
    Leaf 0's histogram must differ between the two, because in the second it
    covers half the rows.

    If it does not, the histogram is ignoring the partition, which would
    explain a score that never changes with depth.
    """
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
    var n_rows = 1024
    var n_features = features_per_int(POLICY_BINARY)
    var stat_count = 1

    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var z = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for i in range(n_rows):
        z.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())
    ctx.synchronize()

    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for f in range(n_features):
        for r in range(n_rows):
            hb.unsafe_ptr().unsafe_store(r, UInt8(((r // (f + 1)) + f) % 2))
        ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(0),
            policy_mask(POLICY_BINARY),
            UInt32(policy_shift(POLICY_BINARY, f)),
            bins.unsafe_ptr(),
            Int32(n_rows),
            cindex.unsafe_ptr(),
            grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    var stats = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    for r in range(n_rows):
        hs.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())

    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())

    var folds = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var fold_off = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var grp_off = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var grp_sz = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var qa = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var qb = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var qc = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var qd = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    for f in range(n_features):
        qa.unsafe_ptr().unsafe_store(f, UInt32(1))
        qb.unsafe_ptr().unsafe_store(f, UInt32(f))
        qc.unsafe_ptr().unsafe_store(f, UInt32(0))
        qd.unsafe_ptr().unsafe_store(f, UInt32(n_features))
    ctx.enqueue_copy(dst_buf=folds, src_ptr=qa.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=fold_off, src_ptr=qb.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=grp_off, src_ptr=qc.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=grp_sz, src_ptr=qd.unsafe_ptr())

    var max_leaves = 4
    var hist = ctx.enqueue_create_buffer[DType.float32](
        max_leaves * stat_count * n_features
    )
    var out = ctx.enqueue_create_host_buffer[DType.float32](
        max_leaves * stat_count * n_features
    )

    var p_off = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
    var ho = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var hz = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    var hid = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
    for i in range(max_leaves):
        hid.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=ids, src_ptr=hid.unsafe_ptr())

    # --- case A: ONE leaf, all 1024 rows ---------------------------------
    for i in range(max_leaves):
        ho.unsafe_ptr().unsafe_store(i, UInt32(0))
        hz.unsafe_ptr().unsafe_store(i, UInt32(0))
    hz.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=ho.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=hz.unsafe_ptr())
    var zf = ctx.enqueue_create_host_buffer[DType.float32](
        max_leaves * stat_count * n_features
    )
    for i in range(max_leaves * stat_count * n_features):
        zf.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=hist, src_ptr=zf.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[binary_hist_gather_kernel](
        folds.unsafe_ptr(), fold_off.unsafe_ptr(), grp_off.unsafe_ptr(),
        grp_sz.unsafe_ptr(), Int32(n_features), cindex.unsafe_ptr(),
        Int32(n_rows), Int32(0), row_index.unsafe_ptr(), stats.unsafe_ptr(),
        Int32(n_rows), p_off.unsafe_ptr(), p_sz.unsafe_ptr(),
        ids.unsafe_ptr(), hist.unsafe_ptr(), acc_scratch.unsafe_ptr(),
        Float32(1.0), Int32(max_leaves), Int32(stat_count),
        grid_dim=(1, 1, 1), block_dim=(BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()
    var whole = out.unsafe_ptr().unsafe_load(0)

    # --- case B: TWO leaves, [0,512) and [512,1024) ----------------------
    hz.unsafe_ptr().unsafe_store(0, UInt32(512))
    ho.unsafe_ptr().unsafe_store(1, UInt32(512))
    hz.unsafe_ptr().unsafe_store(1, UInt32(512))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=ho.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=hz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hist, src_ptr=zf.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[binary_hist_gather_kernel](
        folds.unsafe_ptr(), fold_off.unsafe_ptr(), grp_off.unsafe_ptr(),
        grp_sz.unsafe_ptr(), Int32(n_features), cindex.unsafe_ptr(),
        Int32(n_rows), Int32(0), row_index.unsafe_ptr(), stats.unsafe_ptr(),
        Int32(n_rows), p_off.unsafe_ptr(), p_sz.unsafe_ptr(),
        ids.unsafe_ptr(), hist.unsafe_ptr(), acc_scratch.unsafe_ptr(),
        Float32(1.0), Int32(max_leaves), Int32(stat_count),
        grid_dim=(1, 2, 1), block_dim=(BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()
    var half0 = out.unsafe_ptr().unsafe_load(0)
    var half1 = out.unsafe_ptr().unsafe_load(n_features)

    print("  feature 0, zero-side count")
    print("    1 leaf  of 1024 rows :", whole)
    print("    leaf 0  of  512 rows :", half0)
    print("    leaf 1  of  512 rows :", half1)
    if abs(half0 + half1 - whole) > Float32(1e-3):
        raise Error("the two halves do not sum to the whole")
    if abs(half0 - whole) < Float32(1e-3):
        raise Error(
            "leaf 0's histogram over 512 rows equals the whole 1024-row"
            " histogram: the kernel is IGNORING the partition size"
        )
    print("  the histogram tracks the partition")
