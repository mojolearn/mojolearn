"""Does a ported kernel actually ENQUEUE on Metal?

Not a port and not a test: a COMPILE-AND-LAUNCH PROBE, and the standing rule
of this tree is that a kernel is not ported until it appears here and runs.

Every "it compiles" claim before the first probe was
`mojo build --emit=object`, which targets the HOST. Three files reported as
compiling did not compile at a real use site, and the failures taught the
signature rules now written in PORTING.md item 9. This file is the smallest
thing that keeps that from happening again.

It checks REACHABILITY, not correctness. A kernel that enqueues can still
compute nonsense; that is what the level loop's digests will be for.
"""

from max.gpu.host import DeviceContext

from catboost.cuda.methods.greedy_subsets_searcher.kernel.compute_scores import compute_optimal_splits_kernel
from catboost.cuda.methods.greedy_subsets_searcher.kernel.hist_binary import binary_hist_kernel
from catboost.cuda.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    scan_histograms_kernel,
    substract_histograms_kernel,
)


def probe() raises:
    var ctx = DeviceContext()

    var u32a = ctx.enqueue_create_buffer[DType.uint32](64)
    var u32b = ctx.enqueue_create_buffer[DType.uint32](64)
    var u32c = ctx.enqueue_create_buffer[DType.uint32](64)
    var u32d = ctx.enqueue_create_buffer[DType.uint32](64)
    var u8a = ctx.enqueue_create_buffer[DType.uint8](256)
    var f32a = ctx.enqueue_create_buffer[DType.float32](4096)
    var f32b = ctx.enqueue_create_buffer[DType.float32](4096)
    var f32c = ctx.enqueue_create_buffer[DType.float32](4096)

    ctx.enqueue_function[substract_histograms_kernel](
        u32a.unsafe_ptr(),
        u32b.unsafe_ptr(),
        Int32(64),
        f32a.unsafe_ptr(),
        grid_dim=(1, 4, 1),
        block_dim=(256, 1, 1),
    )
    print("  substract_histograms   enqueued")

    ctx.enqueue_function[scan_histograms_kernel](
        u32a.unsafe_ptr(),
        u32b.unsafe_ptr(),
        Int32(8),
        Int32(64),
        f32a.unsafe_ptr(),
        grid_dim=(1, 4, 1),
        block_dim=(256, 1, 1),
    )
    print("  scan_histograms        enqueued")

    ctx.enqueue_function[compute_optimal_splits_kernel](
        u8a.unsafe_ptr(),
        Int32(64),
        f32a.unsafe_ptr(),
        f32b.unsafe_ptr(),
        Int32(2),
        u32a.unsafe_ptr(),
        Int32(4),
        Float32(1.0),
        f32c.unsafe_ptr(),
        u32c.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    print("  compute_optimal_splits enqueued")

    # Every pointer argument needs its OWN buffer: Mojo refuses two mutable
    # aliases in one call, which is a real constraint on kernels that take
    # many arrays and not just a probe inconvenience.
    var folds = ctx.enqueue_create_buffer[DType.uint32](64)
    var fold_off = ctx.enqueue_create_buffer[DType.uint32](64)
    var grp_off = ctx.enqueue_create_buffer[DType.uint32](64)
    var grp_size = ctx.enqueue_create_buffer[DType.uint32](64)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](4096)
    var p_off = ctx.enqueue_create_buffer[DType.uint32](64)
    var p_size = ctx.enqueue_create_buffer[DType.uint32](64)
    var p_ids = ctx.enqueue_create_buffer[DType.uint32](64)
    var stats_buf = ctx.enqueue_create_buffer[DType.float32](4096)
    var sums = ctx.enqueue_create_buffer[DType.float32](4096)

    ctx.enqueue_function[binary_hist_kernel](
        folds.unsafe_ptr(),
        fold_off.unsafe_ptr(),
        grp_off.unsafe_ptr(),
        grp_size.unsafe_ptr(),
        Int32(32),
        cindex.unsafe_ptr(),
        Int32(64),
        stats_buf.unsafe_ptr(),
        Int32(64),
        p_off.unsafe_ptr(),
        p_size.unsafe_ptr(),
        p_ids.unsafe_ptr(),
        sums.unsafe_ptr(),
        Int32(4),
        Int32(2),
        grid_dim=(1, 1, 1),
        block_dim=(512, 1, 1),
    )
    print("  binary_hist            enqueued")

    ctx.synchronize()
    print("all ported kernels enqueued and the queue drained")
