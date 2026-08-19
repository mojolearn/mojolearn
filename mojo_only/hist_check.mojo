"""Does the binary histogram kernel compute the right histogram?

NO CATBOOST COUNTERPART. A correctness check, against a hand-computable
answer, for the kernel the whole port is built around.

`launch_probe` proves the kernel RUNS. This proves it is RIGHT, which is a
different claim and the one that has never been made about it. The setup is
deliberately small enough to verify by hand:

- 32 binary features packed into one `UInt32` column, written by the
  already-verified `write_compressed_index_kernel`.
- feature `f` at row `r` has bin `(r + f) % 2`, so feature `f`'s ZERO side
  holds exactly the rows where `(r + f)` is even: 32 of 64 for every feature.
- every row carries stat 1.0, so the zero-side histogram value for every
  feature must come out to exactly 32.0.

The binary writeback only ever writes the zero side (the one side is
recovered as `total - val` by the caller), so 32.0 per feature is the entire
expected answer and any indexing error in the nibble decode shows up as a
wrong count rather than as a crash.
"""

from max.gpu.host import DeviceContext

from catboost.cuda.gpu_data.grid_policy import (
    POLICY_BINARY,
    features_per_int,
    policy_mask,
    policy_shift,
)
from catboost.cuda.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from catboost.cuda.methods.greedy_subsets_searcher.kernel.hist_binary import (
    binary_hist_kernel,
)
from catboost.cuda.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
)


def check_binary_histogram() raises:
    var ctx = DeviceContext()

    var n_rows = 64
    var n_features = features_per_int(POLICY_BINARY)  # 32
    var n_stats = 1

    # --- build the packed column ------------------------------------------
    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var zero32 = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for i in range(n_rows):
        zero32.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=zero32.unsafe_ptr())

    var host_bins = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for f in range(n_features):
        for r in range(n_rows):
            host_bins.unsafe_ptr().unsafe_store(r, UInt8((r + f) % 2))
        ctx.enqueue_copy(dst_buf=bins, src_ptr=host_bins.unsafe_ptr())
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

    # --- one partition holding every row ----------------------------------
    var p_off = ctx.enqueue_create_buffer[DType.uint32](1)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](1)
    var p_ids = ctx.enqueue_create_buffer[DType.uint32](1)
    # ONE HOST BUFFER PER COPY. `enqueue_copy` is asynchronous, so reusing a
    # staging buffer and overwriting it before the queue drains lets a later
    # value land in an earlier copy. That happened here and it presented as a
    # broken kernel: part_ids[0] took the row count instead of 0, the kernel
    # indexed a 1-element partition array out of bounds, active_block_count
    # came out 0 and every thread took the early return. Every histogram cell
    # was 0.0 with nothing wrong in the kernel at all.
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var h_ids = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](1)
    h_off.unsafe_ptr().unsafe_store(0, UInt32(0))
    h_ids.unsafe_ptr().unsafe_store(0, UInt32(0))
    h_sz.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_ids, src_ptr=h_ids.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.synchronize()

    # --- every row carries stat 1.0 ---------------------------------------
    var stats = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var host_stats = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    for r in range(n_rows):
        host_stats.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=stats, src_ptr=host_stats.unsafe_ptr())

    # --- feature descriptors: one fold each, laid out consecutively -------
    var folds = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var fold_off = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var grp_off = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var grp_sz = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var h_folds = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var h_fold_off = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var h_grp_off = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var h_grp_sz = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    for f in range(n_features):
        h_folds.unsafe_ptr().unsafe_store(f, UInt32(1))
        h_fold_off.unsafe_ptr().unsafe_store(f, UInt32(f))
        h_grp_off.unsafe_ptr().unsafe_store(f, UInt32(0))
        h_grp_sz.unsafe_ptr().unsafe_store(f, UInt32(n_features))
    ctx.enqueue_copy(dst_buf=folds, src_ptr=h_folds.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=fold_off, src_ptr=h_fold_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=grp_off, src_ptr=h_grp_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=grp_sz, src_ptr=h_grp_sz.unsafe_ptr())

    var sums = ctx.enqueue_create_buffer[DType.float32](n_features)
    var zerof = ctx.enqueue_create_host_buffer[DType.float32](n_features)
    for f in range(n_features):
        zerof.unsafe_ptr().unsafe_store(f, Float32(0.0))
    ctx.enqueue_copy(dst_buf=sums, src_ptr=zerof.unsafe_ptr())
    ctx.synchronize()

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
        sums.unsafe_ptr(),
        Int32(1),
        Int32(n_stats),
        grid_dim=(1, 1, 1),
        block_dim=(BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()

    var out = ctx.enqueue_create_host_buffer[DType.float32](n_features)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=sums)
    ctx.synchronize()

    print("  32 features, 64 rows, stat 1.0, one partition")
    print("  expected zero-side count per feature: 32.0")
    var wrong = 0
    var shown = 0
    for f in range(n_features):
        var got = out.unsafe_ptr().unsafe_load(f)
        if shown < 6:
            print("    feature", f, "->", got)
            shown += 1
        if abs(got - Float32(32.0)) > Float32(1e-3):
            wrong += 1
    print("  features with the wrong count:", wrong, "of", n_features)
    if wrong != 0:
        raise Error(
            "the binary histogram is WRONG for "
            + String(wrong)
            + " of "
            + String(n_features)
            + " features"
        )
    print("  binary histogram computes the right answer")
