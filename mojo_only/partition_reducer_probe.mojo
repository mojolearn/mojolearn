"""Which partition reducer? `PORTING.md` 98a's open decision, measured.

CatBoost's `UpdateSubsetsStats` dispatches `UpdatePartitionProps` --
ONE BLOCK PER PARTITION, 1024 threads, three sequential reductions
(`methods/kernel/pointwise_scores.cu:681`). This repository calls
`compute_partition_stats` instead, which is grid-strided over
(chunk, part, stat) -- and pays a de-interleave and a pack adapter and TWO
reduce calls to do it, six launches against one.

DEVIATION 98 declined the faithful one on occupancy grounds: at depth 0 there
is exactly ONE partition, so their grid is one threadgroup for the whole
dataset. That was never measured on this path. This measures it.

BOTH ARE PORTED AND BOTH ARE CORRECT -- `check-pointwise-subsets` gates the
one in use and `check-pointwise-scores` R1-R4 gates the other -- so nothing
here can change a result. It is purely which is faster at the part counts a
tree actually walks: 1, 2, 4, ... up to `1 << max_depth`.

METHOD: the two are INTERLEAVED and the MINIMUM of each is reported, because
another session is running GPU work on this box.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceAttribute, DeviceContext

from gbdt.gpu_util.partitions_reduce import compute_partition_stats
from gbdt.methods.kernel.pointwise_scores import update_partition_props

comptime N_ROWS = 200000
comptime REPS = 5


def main() raises:
    var ctx = DeviceContext()
    var sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    var tgt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var wts = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var counts = ctx.enqueue_create_buffer[DType.float32](1)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](N_ROWS)
    for r in range(N_ROWS):
        ht.unsafe_ptr().unsafe_store(r, Float32((r % 97) + 1))
    ctx.enqueue_copy(dst_buf=tgt, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=wts, src_ptr=ht.unsafe_ptr())
    ctx.synchronize()

    print(
        "device SM count", sm, ";", N_ROWS, "rows;", REPS,
        "interleaved reps; min of each reported"
    )
    print("parts |   theirs (1 block/part)  |  ours (chunked)  |  ratio")

    var depths: List[Int] = [0, 1, 2, 3, 4, 5, 6]
    for di in range(len(depths)):
        var np = 1 << depths[di]

        var hp = ctx.enqueue_create_host_buffer[DType.uint32](2 * np)
        var per = N_ROWS // np
        for p in range(np):
            hp.unsafe_ptr().unsafe_store(2 * p, UInt32(p * per))
            var sz = per
            if p == np - 1:
                sz = N_ROWS - p * per
            hp.unsafe_ptr().unsafe_store(2 * p + 1, UInt32(sz))
        var parts = ctx.enqueue_create_buffer[DType.uint32](2 * np)
        ctx.enqueue_copy(dst_buf=parts, src_ptr=hp.unsafe_ptr())

        var ps3 = ctx.enqueue_create_buffer[DType.float32](3 * np)
        var po = ctx.enqueue_create_buffer[DType.uint32](np)
        var psz = ctx.enqueue_create_buffer[DType.uint32](np)
        var ps2 = ctx.enqueue_create_buffer[DType.float32](2 * np)
        var pids = ctx.enqueue_create_buffer[DType.uint32](np)
        var hpid = ctx.enqueue_create_host_buffer[DType.uint32](np)
        for p in range(np):
            hpid.unsafe_ptr().unsafe_store(p, UInt32(p))
        ctx.enqueue_copy(dst_buf=pids, src_ptr=hpid.unsafe_ptr())
        var partials = ctx.enqueue_create_buffer[DType.float32](
            4096 * np + 4096
        )
        var hpo = ctx.enqueue_create_host_buffer[DType.uint32](np)
        var hpz = ctx.enqueue_create_host_buffer[DType.uint32](np)
        for p in range(np):
            hpo.unsafe_ptr().unsafe_store(p, UInt32(p * per))
            var sz = per
            if p == np - 1:
                sz = N_ROWS - p * per
            hpz.unsafe_ptr().unsafe_store(p, UInt32(sz))
        ctx.enqueue_copy(dst_buf=po, src_ptr=hpo.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=psz, src_ptr=hpz.unsafe_ptr())
        ctx.synchronize()

        var t_theirs = Float64(1e18)
        var t_ours = Float64(1e18)
        for _ in range(REPS):
            var t0 = perf_counter_ns()
            update_partition_props(
                ctx, tgt, wts, counts, True, True, False, parts, ps3, np
            )
            ctx.synchronize()
            var d = Float64(perf_counter_ns() - t0) / 1e6
            if d < t_theirs:
                t_theirs = d

            # ours, as `update_subsets_stats` calls it: one reduce per
            # column because the two gathered planes are separate buffers
            t0 = perf_counter_ns()
            compute_partition_stats(
                ctx, np, N_ROWS, 1, N_ROWS, pids, po, psz, tgt, partials,
                ps2, sm,
            )
            compute_partition_stats(
                ctx, np, N_ROWS, 1, N_ROWS, pids, po, psz, wts, partials,
                ps2, sm,
            )
            ctx.synchronize()
            d = Float64(perf_counter_ns() - t0) / 1e6
            if d < t_ours:
                t_ours = d

        print(
            "  ", np, "  |  ", t_theirs, " |  ", t_ours, " |  ",
            t_theirs / t_ours,
        )

    print()
    print(
        "ratio > 1 means THEIRS is slower and DEVIATION 98 should stand;"
        " ratio < 1 means the faithful reducer also wins and 98a should"
        " be closed by switching."
    )
