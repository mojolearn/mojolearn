"""How long does one level take at a realistic size?

The first timing number this tree has produced. Everything before it was
correctness.

WHAT THIS IS AND IS NOT COMPARABLE TO
-------------------------------------
CatBoost's recorded figure is ~32 ms per TREE at 800,000 x 100 on an M4, and
mojotrees' symmetric CPU is 130 ms per tree at the same shape. A depth-8 tree
is 8 levels, so a per-level time here multiplied by 8 is the honest
comparison, and it is ROUGH: our level here is one feature group of 32 binary
features, theirs is 100 features across three grouping policies.

So this number answers one question only: **is the packed-load histogram in
the right order of magnitude, or is the widened barrier eating it?** That
barrier, CatBoost's 8-lane tile sync raised to a 512-thread threadgroup
barrier because Mojo has no lane primitives, runs 16 times per row-batch in
the inner loop and has never been measured.
"""

from std.time import perf_counter_ns

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
    binary_hist_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
)
from mojo_only.level_driver import run_one_level


def bench_level(n_rows: Int, repeats: Int) raises:
    var ctx = DeviceContext()
    var n_features = features_per_int(POLICY_BINARY)

    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var z = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for i in range(n_rows):
        z.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())
    ctx.synchronize()

    var host_bin = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for f in range(n_features):
        for r in range(n_rows):
            var v = 1 if (r % 4) == 0 else 0
            if f != 7:
                v = ((r // (f + 2)) + f) % 2
            host_bin.unsafe_ptr().unsafe_store(r, UInt8(v))
        ctx.enqueue_copy(dst_buf=bins, src_ptr=host_bin.unsafe_ptr())
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

    var stats = ctx.enqueue_create_buffer[DType.float32](2 * n_rows)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](2 * n_rows)
    var total_w = Float64(0.0)
    var total_g = Float64(0.0)
    for r in range(n_rows):
        var g = -0.1
        if (r % 4) == 0:
            g = 3.0
        hs.unsafe_ptr().unsafe_store(r, Float32(1.0))
        hs.unsafe_ptr().unsafe_store(n_rows + r, Float32(g))
        total_w += 1.0
        total_g += g
    ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())

    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())
    ctx.synchronize()

    # Warm up: the first launch of each kernel pays Metal pipeline creation,
    # which is not what is being measured.
    var warm = run_one_level(
        ctx, n_rows, n_features, cindex, stats, row_index,
        Float32(total_w), Float32(total_g),
    )
    _ = warm.left_size

    var best_ns = 0
    for i in range(repeats):
        var t0 = perf_counter_ns()
        var r = run_one_level(
            ctx, n_rows, n_features, cindex, stats, row_index,
            Float32(total_w), Float32(total_g),
        )
        var dt = perf_counter_ns() - t0
        _ = r.left_size
        if i == 0 or dt < best_ns:
            best_ns = dt

    var ms = Float64(best_ns) / 1.0e6
    print("  rows", n_rows, " features", n_features, " best of", repeats)
    print("    one level:", ms, "ms")
    print("    x8 levels:", ms * 8.0, "ms per depth-8 tree")
    print("    CatBoost reference: ~32 ms/tree at 800k x 100 (3 policies)")


def bench_histogram_only(n_rows: Int, repeats: Int) raises:
    """The histogram kernel alone, isolated from the rest of the level.

    The aggregate level time cannot say WHERE it goes. This asks the one
    question that decides the next move: is the packed-load histogram fast,
    with the rest of the level slow, or is the histogram itself being eaten
    by the barrier?

    The barrier is the suspect on record. CatBoost syncs an 8-lane tile
    inside `AddPoint`; Mojo has no lane primitives, so ours is a 512-thread
    threadgroup barrier, and `AddPoint` runs it EIGHT times per bin per
    unrolled row, so 16 barriers per row-batch of 2. That is the largest
    known handicap in the port and it has never been measured.
    """
    var ctx = DeviceContext()
    var n_features = features_per_int(POLICY_BINARY)
    var stat_count = 2

    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var z = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for i in range(n_rows):
        z.unsafe_ptr().unsafe_store(i, UInt32(i % 65536))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())

    var stats = ctx.enqueue_create_buffer[DType.float32](2 * n_rows)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](2 * n_rows)
    for r in range(2 * n_rows):
        hs.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())

    var p_off = ctx.enqueue_create_buffer[DType.uint32](1)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](1)
    var p_ids = ctx.enqueue_create_buffer[DType.uint32](1)
    var a = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var b = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var c = ctx.enqueue_create_host_buffer[DType.uint32](1)
    a.unsafe_ptr().unsafe_store(0, UInt32(0))
    b.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    c.unsafe_ptr().unsafe_store(0, UInt32(0))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=a.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=b.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_ids, src_ptr=c.unsafe_ptr())

    var folds = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var fold_off = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var grp_off = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var grp_sz = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var q1 = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var q2 = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var q3 = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var q4 = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    for f in range(n_features):
        q1.unsafe_ptr().unsafe_store(f, UInt32(1))
        q2.unsafe_ptr().unsafe_store(f, UInt32(f))
        q3.unsafe_ptr().unsafe_store(f, UInt32(0))
        q4.unsafe_ptr().unsafe_store(f, UInt32(n_features))
    ctx.enqueue_copy(dst_buf=folds, src_ptr=q1.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=fold_off, src_ptr=q2.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=grp_off, src_ptr=q3.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=grp_sz, src_ptr=q4.unsafe_ptr())

    var hist = ctx.enqueue_create_buffer[DType.float32](
        stat_count * n_features
    )
    ctx.synchronize()

    var best = 0
    for i in range(repeats + 1):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[binary_hist_kernel](
            folds.unsafe_ptr(),
            fold_off.unsafe_ptr(),
            grp_off.unsafe_ptr(),
            grp_sz.unsafe_ptr(),
            Int32(n_features),
            cindex.unsafe_ptr(),
            Int32(n_rows),
            stats.unsafe_ptr(),
            Int32(n_rows),
            p_off.unsafe_ptr(),
            p_sz.unsafe_ptr(),
            p_ids.unsafe_ptr(),
            hist.unsafe_ptr(),
            Int32(1),
            Int32(stat_count),
            grid_dim=(1, 1, stat_count),
            block_dim=(BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()
        var dt = perf_counter_ns() - t0
        if i == 1:
            best = dt
        elif i > 1 and dt < best:
            best = dt

    var ms = Float64(best) / 1.0e6
    var cells = Float64(n_rows) * Float64(n_features)
    print("    histogram alone:", ms, "ms for", n_rows, "rows x", n_features)
    print("    that is", cells / (ms * 1.0e6), "G cell-updates/s")
