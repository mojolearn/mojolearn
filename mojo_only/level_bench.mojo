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

from mojo_only.interleaved import ArmResult, report, summarize

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
from ported.methods.greedy_subsets_searcher.kernel.split_points import (
    PARTITION_BLOCK,
    launch_stable_partition,
)
from ported.methods.greedy_subsets_searcher.kernel.compute_scores import (
    SCORE_BLOCK_SIZE,
    compute_optimal_splits_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    scan_histograms_kernel,
    zero_histograms_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.split_points import (
    SPLIT_BLOCK_SIZE,
    gather_index_in_leaves_kernel,
    split_and_make_sequence_kernel,
    update_partitions_after_split_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.hist_one_byte import (
    ONE_BYTE_BLOCK_SIZE,
    one_byte_hist_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.hist_binary import (
    binary_hist_kernel,
)
from ported.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
)
from ported.methods.greedy_subsets_searcher.kernel.split_points import (
    PARTITION_BLOCK,
    launch_stable_partition,
)
from ported.methods.greedy_subsets_searcher.greedy_search_helper import (
    run_one_level,
    run_tree,
)


def bench_level(n_rows: Int, repeats: Int) raises:
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
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
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
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
            Int32(0),
            stats.unsafe_ptr(),
            Int32(n_rows),
            p_off.unsafe_ptr(),
            p_sz.unsafe_ptr(),
            p_ids.unsafe_ptr(),
            hist.unsafe_ptr(),
            acc_scratch.unsafe_ptr(),
            Float32(1.0),
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


def bench_partition_only(n_rows: Int, repeats: Int) raises:
    """The stable partition alone.

    Named as the prime suspect for the 83 percent of a level that was not the
    histogram, measured at 2.985 ms, rewritten as a three-phase grid-parallel
    scan, and re-measured at 0.248 ms. Kept as a standing check because the
    failure it guards against is a regression to one-block-per-leaf, which is
    correct and 12x slower and would not fail any correctness test.
    """
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)

    var flags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var hf = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    for r in range(n_rows):
        hf.unsafe_ptr().unsafe_store(r, UInt8(1) if (r % 3) == 0 else UInt8(0))
    ctx.enqueue_copy(dst_buf=flags, src_ptr=hf.unsafe_ptr())

    var p_off = ctx.enqueue_create_buffer[DType.uint32](1)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](1)
    var lids = ctx.enqueue_create_buffer[DType.uint32](1)
    var a = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var b = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var c = ctx.enqueue_create_host_buffer[DType.uint32](1)
    a.unsafe_ptr().unsafe_store(0, UInt32(0))
    b.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    c.unsafe_ptr().unsafe_store(0, UInt32(0))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=a.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=b.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=lids, src_ptr=c.unsafe_ptr())

    var gmap = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var sflags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var max_chunks = (n_rows + PARTITION_BLOCK - 1) // PARTITION_BLOCK
    var chunk_zeros = ctx.enqueue_create_buffer[DType.uint32](max_chunks)
    var chunk_offsets = ctx.enqueue_create_buffer[DType.uint32](max_chunks)
    var leaf_zeros = ctx.enqueue_create_buffer[DType.uint32](1)
    ctx.synchronize()

    var best = 0
    for i in range(repeats + 1):
        var t0 = perf_counter_ns()
        launch_stable_partition(
            ctx, 1, n_rows, lids, p_off, p_sz, flags,
            chunk_zeros, chunk_offsets, leaf_zeros, gmap, sflags,
        )
        ctx.synchronize()
        var dt = perf_counter_ns() - t0
        if i == 1:
            best = dt
        elif i > 1 and dt < best:
            best = dt

    var ms = Float64(best) / 1.0e6
    print("    stable partition alone:", ms, "ms for", n_rows, "rows")
    print(
        "    three phases, grid-parallel over",
        (n_rows + PARTITION_BLOCK - 1) // PARTITION_BLOCK,
        "chunks (was one block, 2.985 ms)",
    )


def bench_remaining_phases(n_rows: Int, repeats: Int) raises:
    """The six kernels that are neither the histogram nor the partition.

    The level is 4.9 ms, the histogram 1.3 and the partition 0.25, so 3.4 ms
    is unaccounted for: LARGER THAN EITHER MEASURED PART. Guessing at its
    split is exactly how five conclusions went wrong in the other repository
    this morning, so each kernel gets its own clock.

    Timed one at a time with a `synchronize` after each, so these are not the
    schedule an untimed level gets. They are for attribution, not for a total.
    """
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
    var n_features = features_per_int(POLICY_BINARY)
    var stat_count = 2

    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var stats = ctx.enqueue_create_buffer[DType.float32](2 * n_rows)
    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var new_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var gmap = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var flags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var sflags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var seq = ctx.enqueue_create_buffer[DType.uint32](n_rows)

    var hu = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        hu.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hu.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=gmap, src_ptr=hu.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=hu.unsafe_ptr())

    var p_off = ctx.enqueue_create_buffer[DType.uint32](2)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](2)
    var hp_off = ctx.enqueue_create_buffer[DType.uint32](2)
    var hp_sz = ctx.enqueue_create_buffer[DType.uint32](2)
    var lids = ctx.enqueue_create_buffer[DType.uint32](2)
    var one = ctx.enqueue_create_buffer[DType.uint32](1)
    var two = ctx.enqueue_create_buffer[DType.uint32](1)
    var h2a = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var h2b = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var h2c = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var h1a = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var h1b = ctx.enqueue_create_host_buffer[DType.uint32](1)
    h2a.unsafe_ptr().unsafe_store(0, UInt32(0))
    h2a.unsafe_ptr().unsafe_store(1, UInt32(0))
    h2b.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    h2b.unsafe_ptr().unsafe_store(1, UInt32(0))
    h2c.unsafe_ptr().unsafe_store(0, UInt32(0))
    h2c.unsafe_ptr().unsafe_store(1, UInt32(1))
    h1a.unsafe_ptr().unsafe_store(0, UInt32(0))
    h1b.unsafe_ptr().unsafe_store(0, UInt32(1))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h2a.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h2b.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hp_off, src_ptr=h2a.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hp_sz, src_ptr=h2b.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=lids, src_ptr=h2c.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=one, src_ptr=h1a.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=two, src_ptr=h1b.unsafe_ptr())

    var nf = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var nf2 = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var hnf = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    for f in range(n_features):
        hnf.unsafe_ptr().unsafe_store(f, UInt32(1))
    ctx.enqueue_copy(dst_buf=nf, src_ptr=hnf.unsafe_ptr())
    for f in range(n_features):
        hnf.unsafe_ptr().unsafe_store(f, UInt32(f))
    ctx.enqueue_copy(dst_buf=nf2, src_ptr=hnf.unsafe_ptr())

    var hist = ctx.enqueue_create_buffer[DType.float32](
        2 * stat_count * n_features
    )
    var pstats = ctx.enqueue_create_buffer[DType.float32](2 * stat_count)
    var skip = ctx.enqueue_create_buffer[DType.uint8](n_features)
    var oscore = ctx.enqueue_create_buffer[DType.float32](1)
    var obin = ctx.enqueue_create_buffer[DType.uint32](1)
    var sp1 = ctx.enqueue_create_buffer[DType.uint32](1)
    var sp2 = ctx.enqueue_create_buffer[DType.uint32](1)
    var sp3 = ctx.enqueue_create_buffer[DType.uint32](1)
    var sp5 = ctx.enqueue_create_buffer[DType.uint32](1)
    var sp4 = ctx.enqueue_create_buffer[DType.uint8](1)
    ctx.synchronize()

    var wide = (n_rows + 255) // 256
    if wide > 256:
        wide = 256

    var names = List[String]()
    var times = List[Float64]()

    # zero
    var best = 0
    for i in range(repeats + 1):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[zero_histograms_kernel](
            lids.unsafe_ptr(), Int32(n_features), hist.unsafe_ptr(),
            grid_dim=(1, 2, stat_count), block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        var dt = perf_counter_ns() - t0
        if i == 1 or (i > 1 and dt < best):
            best = dt
    names.append(String("zero")); times.append(Float64(best) / 1.0e6)

    # scan
    best = 0
    for i in range(repeats + 1):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[scan_histograms_kernel](
            nf2.unsafe_ptr(), nf.unsafe_ptr(), Int32(n_features),
            Int32(n_features), hist.unsafe_ptr(),
            grid_dim=(1, 2, stat_count), block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        var dt = perf_counter_ns() - t0
        if i == 1 or (i > 1 and dt < best):
            best = dt
    names.append(String("scan")); times.append(Float64(best) / 1.0e6)

    # score
    best = 0
    for i in range(repeats + 1):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[compute_optimal_splits_kernel](
            skip.unsafe_ptr(), Int32(n_features), hist.unsafe_ptr(),
            pstats.unsafe_ptr(), Int32(stat_count), one.unsafe_ptr(),
            Int32(1), Float32(1.0), oscore.unsafe_ptr(), obin.unsafe_ptr(),
            grid_dim=(1, 1, 1), block_dim=(SCORE_BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()
        var dt = perf_counter_ns() - t0
        if i == 1 or (i > 1 and dt < best):
            best = dt
    names.append(String("score")); times.append(Float64(best) / 1.0e6)

    # split flags
    best = 0
    for i in range(repeats + 1):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[split_and_make_sequence_kernel](
            cindex.unsafe_ptr(), row_index.unsafe_ptr(), p_off.unsafe_ptr(),
            p_sz.unsafe_ptr(), one.unsafe_ptr(), sp1.unsafe_ptr(),
            sp2.unsafe_ptr(), sp3.unsafe_ptr(), sp4.unsafe_ptr(),
            sp5.unsafe_ptr(), flags.unsafe_ptr(), seq.unsafe_ptr(),
            grid_dim=(wide, 1, 1), block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()
        var dt = perf_counter_ns() - t0
        if i == 1 or (i > 1 and dt < best):
            best = dt
    names.append(String("split flags")); times.append(Float64(best) / 1.0e6)

    # gather
    best = 0
    for i in range(repeats + 1):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[gather_index_in_leaves_kernel](
            one.unsafe_ptr(), p_off.unsafe_ptr(), p_sz.unsafe_ptr(),
            row_index.unsafe_ptr(), gmap.unsafe_ptr(), new_index.unsafe_ptr(),
            grid_dim=(wide, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        var dt = perf_counter_ns() - t0
        if i == 1 or (i > 1 and dt < best):
            best = dt
    names.append(String("gather index")); times.append(Float64(best) / 1.0e6)

    # update partitions
    best = 0
    for i in range(repeats + 1):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[update_partitions_after_split_kernel](
            one.unsafe_ptr(), two.unsafe_ptr(), Int32(1), sflags.unsafe_ptr(),
            p_off.unsafe_ptr(), p_sz.unsafe_ptr(), hp_off.unsafe_ptr(),
            hp_sz.unsafe_ptr(),
            grid_dim=(wide, 1, 1), block_dim=(512, 1, 1),
        )
        ctx.synchronize()
        var dt = perf_counter_ns() - t0
        if i == 1 or (i > 1 and dt < best):
            best = dt
    names.append(String("update parts")); times.append(Float64(best) / 1.0e6)

    var total = 0.0
    for i in range(len(names)):
        print("    ", names[i], ":", times[i], "ms")
        total += times[i]
    print("    the other six together:", total, "ms")


def bench_tree(n_rows: Int, max_depth: Int, repeats: Int) raises:
    """A whole depth-`max_depth` tree, measured rather than extrapolated.

    Every number before this one multiplied a depth-0 level by 8, which
    overestimates (sibling subtraction halves the histogram below the root)
    and underestimates (the reorder touches every row at every level) at the
    same time. This is the tree.
    """
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
    var n_features = features_per_int(POLICY_BINARY)

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
            var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            hb.unsafe_ptr().unsafe_store(r, UInt8(Int(x & 1)))
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

    var stats = ctx.enqueue_create_buffer[DType.float32](2 * n_rows)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](2 * n_rows)
    var tw = Float64(0.0)
    var tg = Float64(0.0)
    for r in range(n_rows):
        var g = -0.1
        if (r % 4) == 0:
            g = 3.0
        hs.unsafe_ptr().unsafe_store(r, Float32(1.0))
        hs.unsafe_ptr().unsafe_store(n_rows + r, Float32(g))
        tw += 1.0
        tg += g
    ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())

    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())
    ctx.synchronize()

    var best = 0
    for i in range(repeats + 1):
        # Fresh identity index each repeat: the tree permutes it in place.
        ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var sizes = run_tree(
            ctx, n_rows, n_features, max_depth, cindex, stats, row_index,
            Float32(tw), Float32(tg),
        )
        var dt = perf_counter_ns() - t0
        _ = len(sizes)
        if i == 1 or (i > 1 and dt < best):
            best = dt

    var ms = Float64(best) / 1.0e6
    print(
        "    depth",
        max_depth,
        "tree,",
        n_rows,
        "rows x",
        n_features,
        "binary features:",
        ms,
        "ms",
    )


def bench_replication_interleaved(
    n_rows: Int, max_depth: Int, repeats: Int
) raises:
    """Replication 1 against 32, ARMS INTERLEAVED in one process.

    The first attempt at this compared two separate runs and read 70.3 ms
    against 81.1. That comparison was worthless on a box that has produced
    59.7 and 44.8 ms for identical work an hour apart. This alternates the
    arms so both see the same thermal window.
    """
    var ctx = DeviceContext()
    var n_features = features_per_int(POLICY_BINARY)

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
            var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            hb.unsafe_ptr().unsafe_store(r, UInt8(Int(x & 1)))
        ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(0), policy_mask(POLICY_BINARY),
            UInt32(policy_shift(POLICY_BINARY, f)),
            bins.unsafe_ptr(), Int32(n_rows), cindex.unsafe_ptr(),
            grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    var stats = ctx.enqueue_create_buffer[DType.float32](2 * n_rows)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](2 * n_rows)
    var tw = Float64(0.0)
    var tg = Float64(0.0)
    for r in range(n_rows):
        var g = -0.1
        if (r % 4) == 0:
            g = 3.0
        hs.unsafe_ptr().unsafe_store(r, Float32(1.0))
        hs.unsafe_ptr().unsafe_store(n_rows + r, Float32(g))
        tw += 1.0
        tg += g
    ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())

    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.synchronize()

    var s1 = List[Float64]()
    var s32 = List[Float64]()

    for rep in range(repeats + 1):
        for arm in range(2):
            var reps = 1 if arm == 0 else 32
            ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())
            ctx.synchronize()
            var t0 = perf_counter_ns()
            var sizes = run_tree(
                ctx, n_rows, n_features, max_depth, cindex, stats, row_index,
                Float32(tw), Float32(tg), reps,
            )
            var dt = Float64(perf_counter_ns() - t0) / 1.0e6
            _ = len(sizes)
            if rep == 0:
                continue  # warm-up pass, both arms
            if arm == 0:
                s1.append(dt)
            else:
                s32.append(dt)

    var arms = List[ArmResult]()
    arms.append(summarize(String("replicas=1 "), s1))
    arms.append(summarize(String("replicas=32"), s32))
    print("  depth", max_depth, "tree,", n_rows, "rows, arms interleaved:")
    report(arms)


def bench_wide_histogram_interleaved(n_rows: Int, repeats: Int) raises:
    """Does replication pay when the histogram is thousands of cells?

    The open question from the replication experiment. At 32 binary features
    the histogram is 64 cells and 1 replica is indistinguishable from 32. The
    arithmetic says replication should pay once there are enough output cells
    that atomic contention stops dominating, which is CatBoost's actual shape
    (100 features at up to 256 bins).

    So: the ONE-BYTE kernel at 8 bits, 4 features per word, 256 folds each.
    That is 1,024 cells per feature group against 64, sixteen times the
    output, and it is the kernel CatBoost built the pass loop for.

    Arms interleaved, because the last time this question was answered across
    runs the answer was wrong.
    """
    var ctx = DeviceContext()
    var n_features = 4
    var n_folds = 256
    var group_size = n_features * n_folds

    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var hz = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        var x = UInt32(r * 2654435761 + 0x2545F491)
        x ^= x << 13
        x ^= x >> 17
        x ^= x << 5
        hz.unsafe_ptr().unsafe_store(r, x)
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=hz.unsafe_ptr())

    var stats = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    for r in range(n_rows):
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
        q1.unsafe_ptr().unsafe_store(f, UInt32(n_folds))
        q2.unsafe_ptr().unsafe_store(f, UInt32(f * n_folds))
        q3.unsafe_ptr().unsafe_store(f, UInt32(0))
        q4.unsafe_ptr().unsafe_store(f, UInt32(group_size))
    ctx.enqueue_copy(dst_buf=folds, src_ptr=q1.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=fold_off, src_ptr=q2.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=grp_off, src_ptr=q3.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=grp_sz, src_ptr=q4.unsafe_ptr())

    var sums = ctx.enqueue_create_buffer[DType.float32](group_size)
    ctx.synchronize()

    var s1 = List[Float64]()
    var s16 = List[Float64]()

    for rep in range(repeats + 1):
        for arm in range(2):
            var reps = 1 if arm == 0 else 16
            var t0 = perf_counter_ns()
            ctx.enqueue_function[one_byte_hist_kernel[8]](
                folds.unsafe_ptr(), fold_off.unsafe_ptr(),
                grp_off.unsafe_ptr(), grp_sz.unsafe_ptr(),
                Int32(n_features), cindex.unsafe_ptr(), Int32(n_rows),
                Int32(0),
                stats.unsafe_ptr(), Int32(n_rows),
                p_off.unsafe_ptr(), p_sz.unsafe_ptr(), p_ids.unsafe_ptr(),
                sums.unsafe_ptr(), Int32(1), Int32(1),
                grid_dim=(reps, 1, 1),
                block_dim=(ONE_BYTE_BLOCK_SIZE, 1, 1),
            )
            ctx.synchronize()
            var dt = Float64(perf_counter_ns() - t0) / 1.0e6
            if rep == 0:
                continue
            if arm == 0:
                s1.append(dt)
            else:
                s16.append(dt)

    var arms = List[ArmResult]()
    arms.append(summarize(String("1 block  "), s1))
    arms.append(summarize(String("16 blocks"), s16))
    print(
        "  one-byte 8-bit histogram,",
        n_features,
        "features x",
        n_folds,
        "folds =",
        group_size,
        "cells,",
        n_rows,
        "rows:",
    )
    report(arms)
