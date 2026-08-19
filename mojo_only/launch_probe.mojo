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
from catboost.cuda.methods.greedy_subsets_searcher.kernel.hist_half_byte import half_byte_hist_kernel
from catboost.cuda.methods.greedy_subsets_searcher.kernel.split_points import split_and_make_sequence_kernel
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

    var hb_folds = ctx.enqueue_create_buffer[DType.uint32](64)
    var hb_fold_off = ctx.enqueue_create_buffer[DType.uint32](64)
    var hb_grp_off = ctx.enqueue_create_buffer[DType.uint32](64)
    var hb_grp_size = ctx.enqueue_create_buffer[DType.uint32](64)
    var hb_cindex = ctx.enqueue_create_buffer[DType.uint32](4096)
    var hb_poff = ctx.enqueue_create_buffer[DType.uint32](64)
    var hb_psize = ctx.enqueue_create_buffer[DType.uint32](64)
    var hb_pids = ctx.enqueue_create_buffer[DType.uint32](64)
    var hb_stats = ctx.enqueue_create_buffer[DType.float32](4096)
    var hb_sums = ctx.enqueue_create_buffer[DType.float32](4096)

    ctx.enqueue_function[half_byte_hist_kernel](
        hb_folds.unsafe_ptr(),
        hb_fold_off.unsafe_ptr(),
        hb_grp_off.unsafe_ptr(),
        hb_grp_size.unsafe_ptr(),
        Int32(8),
        hb_cindex.unsafe_ptr(),
        Int32(64),
        hb_stats.unsafe_ptr(),
        Int32(64),
        hb_poff.unsafe_ptr(),
        hb_psize.unsafe_ptr(),
        hb_pids.unsafe_ptr(),
        hb_sums.unsafe_ptr(),
        Int32(4),
        Int32(2),
        grid_dim=(1, 1, 1),
        block_dim=(512, 1, 1),
    )
    print("  half_byte_hist         enqueued")

    var ci = ctx.enqueue_create_buffer[DType.uint32](4096)
    var li = ctx.enqueue_create_buffer[DType.uint32](4096)
    var spo = ctx.enqueue_create_buffer[DType.uint32](64)
    var sps = ctx.enqueue_create_buffer[DType.uint32](64)
    var spm = ctx.enqueue_create_buffer[DType.uint32](64)
    var sph = ctx.enqueue_create_buffer[DType.uint8](64)
    var spb = ctx.enqueue_create_buffer[DType.uint32](64)
    var sfl = ctx.enqueue_create_buffer[DType.uint8](4096)
    var idx = ctx.enqueue_create_buffer[DType.uint32](4096)
    var lids = ctx.enqueue_create_buffer[DType.uint32](64)
    var poff = ctx.enqueue_create_buffer[DType.uint32](64)
    var psz = ctx.enqueue_create_buffer[DType.uint32](64)

    ctx.enqueue_function[split_and_make_sequence_kernel](
        ci.unsafe_ptr(),
        li.unsafe_ptr(),
        poff.unsafe_ptr(),
        psz.unsafe_ptr(),
        lids.unsafe_ptr(),
        spo.unsafe_ptr(),
        sps.unsafe_ptr(),
        spm.unsafe_ptr(),
        sph.unsafe_ptr(),
        spb.unsafe_ptr(),
        sfl.unsafe_ptr(),
        idx.unsafe_ptr(),
        grid_dim=(1, 2, 1),
        block_dim=(512, 1, 1),
    )
    print("  split_and_make_seq     enqueued")

    ctx.synchronize()
    print("all ported kernels enqueued and the queue drained")
