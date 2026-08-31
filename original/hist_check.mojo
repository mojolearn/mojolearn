# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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

from std.sys.info import size_of
from gbdt.gpu_data.gpu_structures import CFeature
from max.gpu.host import DeviceContext

from gbdt.methods.greedy_subsets_searcher.kernel.split_points import (
    PARTITION_BLOCK,
    launch_stable_partition,
)

from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    features_per_int,
    policy_mask,
    policy_shift,
)
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_binary import (
    binary_hist_gather_kernel,
    binary_hist_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.split_points import (
    split_and_make_sequence_kernel,
    update_partitions_after_split_kernel,
)
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
)
from gbdt.methods.greedy_subsets_searcher.kernel.compute_scores import (
    compute_optimal_splits_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    copy_histograms_kernel,
    scan_histograms_kernel,
    substract_histograms_kernel,
    zero_histograms_kernel,
)
from original.kernel_matrix import HIST_SMEM_WARP_PRIVATE_F32
from gbdt.methods.greedy_subsets_searcher.kernel.hist_one_byte import (
    ONE_BYTE_BLOCK_SIZE,
    one_byte_hist_kernel,

)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    upload_scale,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_half_byte import (
    half_byte_hist_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
)


def check_binary_histogram() raises:
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)

    var n_rows = 64
    var n_features = features_per_int(POLICY_BINARY)  # 32

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

    # DEVIATION 95: the histogram kernels read `fixed_scale` from DEVICE
    # memory. `upload_scale` returns the buffer and the CALLER keeps it
    # alive: Mojo frees at its last use, and `.unsafe_ptr()` is a use, so
    # without the `_ = scale_keep^` below the buffer is dead before the
    # kernel runs and the next allocation lands on it.
    var scale_keep = upload_scale(ctx, Float32(1.0))
    var scale_ptr = rebind[MutPointer[Float32, MutAnyOrigin]](
        scale_keep.unsafe_ptr()
    )

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
        sums.unsafe_ptr(),
        acc_scratch.unsafe_ptr(),
        scale_ptr,
        Int32(1),
        Int32(1),
        grid_dim=(1, 1, 1),
        block_dim=(BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()
    _ = scale_keep^

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


def check_two_partitions() raises:
    """The case a single full partition cannot catch: TWO leaves, unequal.

    A one-partition check passes even if the kernel ignores `part_offset`
    entirely, because that offset is zero. This one puts leaf 0 at rows
    [0, 24) and leaf 1 at rows [24, 64), so an ignored offset or a wrong
    partition lookup gives a wrong count rather than the right one by
    accident.

    Feature f has bin (r + f) % 2, so leaf 0's zero-side count is the number
    of even (r + f) for r in [0, 24), which is 12 for every f, and leaf 1's
    is the count over [24, 64), which is 20.
    """
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
    var n_rows = 64
    var n_features = features_per_int(POLICY_BINARY)
    var split_at = 24

    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var zero32 = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for i in range(n_rows):
        zero32.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=zero32.unsafe_ptr())
    ctx.synchronize()

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

    # Two partitions, unequal, second one offset.
    var p_off = ctx.enqueue_create_buffer[DType.uint32](2)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](2)
    var p_ids = ctx.enqueue_create_buffer[DType.uint32](2)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var h_ids = ctx.enqueue_create_host_buffer[DType.uint32](2)
    h_off.unsafe_ptr().unsafe_store(0, UInt32(0))
    h_off.unsafe_ptr().unsafe_store(1, UInt32(split_at))
    h_sz.unsafe_ptr().unsafe_store(0, UInt32(split_at))
    h_sz.unsafe_ptr().unsafe_store(1, UInt32(n_rows - split_at))
    h_ids.unsafe_ptr().unsafe_store(0, UInt32(0))
    h_ids.unsafe_ptr().unsafe_store(1, UInt32(1))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_ids, src_ptr=h_ids.unsafe_ptr())

    var stats = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var host_stats = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    for r in range(n_rows):
        host_stats.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=stats, src_ptr=host_stats.unsafe_ptr())

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

    # Two leaves, so the sums buffer holds two leaves' worth.
    var sums = ctx.enqueue_create_buffer[DType.float32](2 * n_features)
    var zerof = ctx.enqueue_create_host_buffer[DType.float32](2 * n_features)
    for i in range(2 * n_features):
        zerof.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=sums, src_ptr=zerof.unsafe_ptr())
    ctx.synchronize()

    # DEVIATION 95: the histogram kernels read `fixed_scale` from DEVICE
    # memory. `upload_scale` returns the buffer and the CALLER keeps it
    # alive: Mojo frees at its last use, and `.unsafe_ptr()` is a use, so
    # without the `_ = scale_keep^` below the buffer is dead before the
    # kernel runs and the next allocation lands on it.
    var scale_keep = upload_scale(ctx, Float32(1.0))
    var scale_ptr = rebind[MutPointer[Float32, MutAnyOrigin]](
        scale_keep.unsafe_ptr()
    )
    # grid y = 2: BOTH leaves in ONE launch. That is the design property.
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
        sums.unsafe_ptr(),
        acc_scratch.unsafe_ptr(),
        scale_ptr,
        Int32(2),
        Int32(1),
        grid_dim=(1, 2, 1),
        block_dim=(BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()
    _ = scale_keep^

    var out = ctx.enqueue_create_host_buffer[DType.float32](2 * n_features)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=sums)
    ctx.synchronize()

    print("  two leaves in ONE launch: rows [0,24) and [24,64)")
    print("  expected zero-side counts: leaf 0 -> 12.0, leaf 1 -> 20.0")
    var wrong = 0
    for leaf in range(2):
        var want = Float32(12.0) if leaf == 0 else Float32(20.0)
        for f in range(n_features):
            var got = out.unsafe_ptr().unsafe_load(leaf * n_features + f)
            if abs(got - want) > Float32(1e-3):
                wrong += 1
    print("    leaf 0 feature 0 ->", out.unsafe_ptr().unsafe_load(0))
    print("    leaf 1 feature 0 ->", out.unsafe_ptr().unsafe_load(n_features))
    print("  wrong cells:", wrong, "of", 2 * n_features)
    if wrong != 0:
        raise Error(
            "the two-leaf histogram is WRONG in "
            + String(wrong)
            + " cells"
        )
    print("  partition offsets honored, both leaves from one launch")


def check_half_byte_histogram() raises:
    """The half-byte kernel against a hand-computable answer.

    This is the file that carried a stale divergent-barrier bug for several
    commits with nothing to catch it, because a launch probe cannot see a
    wrong answer. It gets its own known-answer check for that reason.

    Eight features per `UInt32`, four folds each. Feature f at row r gets bin
    `(r + f) % 4`, so over 64 rows each of the four folds holds exactly 16
    rows for every feature. With stat 1.0 the histogram must read 16.0 in
    every (feature, fold) cell, and unlike the binary case EVERY fold is
    written, not just the zero side, so a wrong fold index shows up directly.
    """
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
    var n_rows = 64
    var n_features = features_per_int(POLICY_HALF_BYTE)  # 8
    var n_folds = 4

    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var zero32 = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for i in range(n_rows):
        zero32.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=zero32.unsafe_ptr())
    ctx.synchronize()

    var host_bins = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for f in range(n_features):
        for r in range(n_rows):
            host_bins.unsafe_ptr().unsafe_store(r, UInt8((r + f) % n_folds))
        ctx.enqueue_copy(dst_buf=bins, src_ptr=host_bins.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(0),
            policy_mask(POLICY_HALF_BYTE),
            UInt32(policy_shift(POLICY_HALF_BYTE, f)),
            bins.unsafe_ptr(),
            Int32(n_rows),
            cindex.unsafe_ptr(),
            grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    var p_off = ctx.enqueue_create_buffer[DType.uint32](1)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](1)
    var p_ids = ctx.enqueue_create_buffer[DType.uint32](1)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var h_ids = ctx.enqueue_create_host_buffer[DType.uint32](1)
    h_off.unsafe_ptr().unsafe_store(0, UInt32(0))
    h_sz.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    h_ids.unsafe_ptr().unsafe_store(0, UInt32(0))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_ids, src_ptr=h_ids.unsafe_ptr())

    var stats = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var host_stats = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    for r in range(n_rows):
        host_stats.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=stats, src_ptr=host_stats.unsafe_ptr())

    # Each feature owns `n_folds` consecutive cells in the group.
    var group_size = n_features * n_folds
    var folds = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var fold_off = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var grp_off = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var grp_sz = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var h_folds = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var h_fold_off = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var h_grp_off = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var h_grp_sz = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    for f in range(n_features):
        h_folds.unsafe_ptr().unsafe_store(f, UInt32(n_folds))
        h_fold_off.unsafe_ptr().unsafe_store(f, UInt32(f * n_folds))
        h_grp_off.unsafe_ptr().unsafe_store(f, UInt32(0))
        h_grp_sz.unsafe_ptr().unsafe_store(f, UInt32(group_size))
    ctx.enqueue_copy(dst_buf=folds, src_ptr=h_folds.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=fold_off, src_ptr=h_fold_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=grp_off, src_ptr=h_grp_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=grp_sz, src_ptr=h_grp_sz.unsafe_ptr())

    var sums = ctx.enqueue_create_buffer[DType.float32](group_size)
    var hb_acc = ctx.enqueue_create_buffer[DType.int32](group_size)
    var zerof = ctx.enqueue_create_host_buffer[DType.float32](group_size)
    for i in range(group_size):
        zerof.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=sums, src_ptr=zerof.unsafe_ptr())
    ctx.synchronize()

    # DEVIATION 95: the histogram kernels read `fixed_scale` from DEVICE
    # memory. `upload_scale` returns the buffer and the CALLER keeps it
    # alive: Mojo frees at its last use, and `.unsafe_ptr()` is a use, so
    # without the `_ = scale_keep^` below the buffer is dead before the
    # kernel runs and the next allocation lands on it.
    var scale_keep = upload_scale(ctx, Float32(1.0))
    var scale_ptr = rebind[MutPointer[Float32, MutAnyOrigin]](
        scale_keep.unsafe_ptr()
    )

    ctx.enqueue_function[half_byte_hist_kernel](
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
        sums.unsafe_ptr(),
        hb_acc.unsafe_ptr(),
        scale_ptr,
        Int32(1),
        Int32(1),
        grid_dim=(1, 1, 1),
        block_dim=(BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()
    _ = scale_keep^

    var out = ctx.enqueue_create_host_buffer[DType.float32](group_size)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=sums)
    ctx.synchronize()

    print("  8 features x 4 folds, 64 rows, stat 1.0")
    print("  expected every (feature, fold) cell: 16.0")
    var wrong = 0
    for f in range(n_features):
        for fold in range(n_folds):
            var got = out.unsafe_ptr().unsafe_load(f * n_folds + fold)
            if abs(got - Float32(16.0)) > Float32(1e-3):
                wrong += 1
    print(
        "    feature 0 folds:",
        out.unsafe_ptr().unsafe_load(0),
        out.unsafe_ptr().unsafe_load(1),
        out.unsafe_ptr().unsafe_load(2),
        out.unsafe_ptr().unsafe_load(3),
    )
    print("  wrong cells:", wrong, "of", group_size)
    if wrong != 0:
        raise Error(
            "the half-byte histogram is WRONG in "
            + String(wrong)
            + " of "
            + String(group_size)
            + " cells"
        )
    print("  half-byte histogram computes the right answer")


def check_subtraction() raises:
    """Sibling subtraction: the mechanism that halves every level's work.

    Least trustworthy unchecked kernel in the tree, because everything
    downstream reads its output and a wrong sign or a wrong offset produces
    plausible numbers rather than obvious garbage.

    Layout is theirs: `histogram[leaf * binFeatureCount * statCount +
    stat * binFeatureCount + binFeature]`. Three leaves, one stat. Leaf 0 is
    the PARENT total, leaf 1 the smaller child that was actually built, and
    the kernel must overwrite leaf 0 with `parent - smaller`, in place, which
    is the larger child.

    Two properties checked that a single pair cannot show:
      - the `max(., 0)` clamp on stat 0 fires where cancellation would go
        negative, and
      - `what` is left UNTOUCHED, since a kernel that swapped operands or
        wrote both slots would still make the first assertion pass.
    """
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
    var n_bf = 8
    var n_leaves = 3

    var hist = ctx.enqueue_create_buffer[DType.float32](n_leaves * n_bf)
    var h = ctx.enqueue_create_host_buffer[DType.float32](n_leaves * n_bf)
    # leaf 0: the parent totals. leaf 1: the built smaller child.
    for b in range(n_bf):
        h.unsafe_ptr().unsafe_store(0 * n_bf + b, Float32(100.0 + Float64(b)))
        h.unsafe_ptr().unsafe_store(1 * n_bf + b, Float32(30.0 + Float64(b)))
        h.unsafe_ptr().unsafe_store(2 * n_bf + b, Float32(-1.0))
    # One bin where the child EXCEEDS the parent, so cancellation would go
    # negative and the clamp must fire.
    h.unsafe_ptr().unsafe_store(1 * n_bf + 3, Float32(1000.0))
    ctx.enqueue_copy(dst_buf=hist, src_ptr=h.unsafe_ptr())

    var from_ids = ctx.enqueue_create_buffer[DType.uint32](1)
    var what_ids = ctx.enqueue_create_buffer[DType.uint32](1)
    var hf = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var hw = ctx.enqueue_create_host_buffer[DType.uint32](1)
    hf.unsafe_ptr().unsafe_store(0, UInt32(0))
    hw.unsafe_ptr().unsafe_store(0, UInt32(1))
    ctx.enqueue_copy(dst_buf=from_ids, src_ptr=hf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=what_ids, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[substract_histograms_kernel](
        from_ids.unsafe_ptr(),
        what_ids.unsafe_ptr(),
        Int32(n_bf),
        hist.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    var out = ctx.enqueue_create_host_buffer[DType.float32](n_leaves * n_bf)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()

    print("  parent 100+b, smaller child 30+b, bin 3 child=1000 (clamp case)")
    var wrong = 0
    for b in range(n_bf):
        var got = out.unsafe_ptr().unsafe_load(0 * n_bf + b)
        var want = Float32(70.0)
        if b == 3:
            want = Float32(0.0)  # 103 - 1000 clamped by max(., 0)
        if abs(got - want) > Float32(1e-3):
            wrong += 1
            print("    bin", b, "got", got, "want", want)
    print("    derived larger child, bin 0:", out.unsafe_ptr().unsafe_load(0))
    print("    derived larger child, bin 3:", out.unsafe_ptr().unsafe_load(3))
    # `what` must be untouched: a kernel that swapped operands or wrote both
    # slots would still satisfy the check above.
    var what_moved = 0
    for b in range(n_bf):
        var want_child = Float32(30.0 + Float64(b))
        if b == 3:
            want_child = Float32(1000.0)
        if abs(out.unsafe_ptr().unsafe_load(1 * n_bf + b) - want_child) > Float32(1e-3):
            what_moved += 1
    print("  wrong derived cells:", wrong, " cells of `what` disturbed:", what_moved)
    if wrong != 0 or what_moved != 0:
        raise Error("sibling subtraction is wrong")
    print("  subtraction correct, clamp fires, `what` untouched")


def check_scan() raises:
    """The bin prefix scan, which is what buys the score kernel its shape.

    Two features of four folds each, one leaf, one stat. Feature 0's folds
    hold 1,2,3,4 and feature 1's hold 10,20,30,40, laid out consecutively in
    the leaf's bin-feature row.

    After the scan each feature's folds must be its own running total,
    1,3,6,10 and 10,30,60,100. The property that matters is that the scan is
    PER FEATURE: it must not run across the feature boundary and turn
    feature 1's first fold into 10+10.
    """
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
    var n_features = 2
    var n_folds = 4
    var n_bf = n_features * n_folds

    var hist = ctx.enqueue_create_buffer[DType.float32](n_bf)
    var h = ctx.enqueue_create_host_buffer[DType.float32](n_bf)
    for k in range(n_folds):
        h.unsafe_ptr().unsafe_store(k, Float32(1.0 + Float64(k)))
        h.unsafe_ptr().unsafe_store(n_folds + k, Float32(10.0 * (1.0 + Float64(k))))
    ctx.enqueue_copy(dst_buf=hist, src_ptr=h.unsafe_ptr())

    var first_bin = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var folds_b = ctx.enqueue_create_buffer[DType.uint32](n_features)
    var h_first = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    var h_folds = ctx.enqueue_create_host_buffer[DType.uint32](n_features)
    for f in range(n_features):
        h_first.unsafe_ptr().unsafe_store(f, UInt32(f * n_folds))
        h_folds.unsafe_ptr().unsafe_store(f, UInt32(n_folds))
    ctx.enqueue_copy(dst_buf=first_bin, src_ptr=h_first.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=folds_b, src_ptr=h_folds.unsafe_ptr())
    ctx.synchronize()

    var scan_ids = ctx.enqueue_create_buffer[DType.uint32](1)
    var scan_ids_h = ctx.enqueue_create_host_buffer[DType.uint32](1)
    scan_ids_h.unsafe_ptr().unsafe_store(0, UInt32(0))
    ctx.enqueue_copy(dst_buf=scan_ids, src_ptr=scan_ids_h.unsafe_ptr())
    ctx.synchronize()
    # all-binary fixture: no one-hot features, so the skip never fires
    var scan_onehot = ctx.enqueue_create_buffer[DType.uint8](256)
    ctx.enqueue_function[scan_histograms_kernel](
        scan_ids.unsafe_ptr(),
        first_bin.unsafe_ptr(),
        folds_b.unsafe_ptr(),
        scan_onehot.unsafe_ptr(),
        Int32(n_features),
        Int32(n_bf),
        hist.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    var out = ctx.enqueue_create_host_buffer[DType.float32](n_bf)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()

    print("  feature 0 folds 1,2,3,4 -> expect 1,3,6,10")
    print("  feature 1 folds 10,20,30,40 -> expect 10,30,60,100")
    var want = List[Float32]()
    want.append(1.0)
    want.append(3.0)
    want.append(6.0)
    want.append(10.0)
    want.append(10.0)
    want.append(30.0)
    want.append(60.0)
    want.append(100.0)
    var wrong = 0
    for i in range(n_bf):
        if abs(out.unsafe_ptr().unsafe_load(i) - want[i]) > Float32(1e-3):
            wrong += 1
            print("    slot", i, "got", out.unsafe_ptr().unsafe_load(i), "want", want[i])
    print("    got f0:", out.unsafe_ptr().unsafe_load(0), out.unsafe_ptr().unsafe_load(3))
    print("    got f1:", out.unsafe_ptr().unsafe_load(4), out.unsafe_ptr().unsafe_load(7))
    print("  wrong slots:", wrong, "of", n_bf)
    if wrong != 0:
        raise Error("the bin scan is wrong")
    print("  scan is per-feature and does not cross the boundary")


def check_scores() raises:
    """The score kernel picks the split a hand calculation picks.

    Two leaves, four candidate bin-features, one gradient stat plus the
    weight plane, `lambda_l2 = 1.0`. Histograms are the POST-SCAN cumulative
    left-side sums the kernel expects, and `part_stats` holds each leaf's
    totals so the right side is derived as `total - left`.

    The candidate values are chosen so one split is unambiguously best and
    the winner is not the first or the last slot, which is what catches an
    argmax that returns an edge by accident. Expected winner is bin-feature
    2, where both leaves split cleanly into halves.

    An oblivious level scores ONE split across ALL its leaves, so the score
    is the sum over leaves. That summation is what makes the leaf loop
    serial inside the thread and needs no cross-thread reduction.
    """
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
    var n_bf = 4
    var n_leaves = 2
    var stat_count = 2  # [weight, gradient]
    var lambda_l2 = Float32(1.0)

    # histograms[leaf * stat_count * n_bf + stat * n_bf + bf]
    var hist = ctx.enqueue_create_buffer[DType.float32](
        n_leaves * stat_count * n_bf
    )
    var h = ctx.enqueue_create_host_buffer[DType.float32](
        n_leaves * stat_count * n_bf
    )
    # Leaf totals: weight 100, gradient 20 for leaf 0; 80 and -12 for leaf 1.
    var tot_w = List[Float32]()
    tot_w.append(100.0)
    tot_w.append(80.0)
    var tot_g = List[Float32]()
    tot_g.append(20.0)
    tot_g.append(-12.0)

    # left-side weight per (leaf, bf), and left-side gradient.
    var lw = List[Float32]()
    var lg = List[Float32]()
    # leaf 0
    lw.append(10.0); lg.append(1.0)     # bf0: lopsided, tiny gain
    lw.append(90.0); lg.append(18.0)    # bf1: lopsided the other way
    lw.append(50.0); lg.append(19.0)    # bf2: even split, gradient concentrated
    lw.append(50.0); lg.append(10.0)    # bf3: even split, gradient split evenly
    # leaf 1
    lw.append(8.0);  lg.append(-1.0)
    lw.append(72.0); lg.append(-11.0)
    lw.append(40.0); lg.append(-11.5)
    lw.append(40.0); lg.append(-6.0)

    for leaf in range(n_leaves):
        for bf in range(n_bf):
            h.unsafe_ptr().unsafe_store(
                leaf * stat_count * n_bf + 0 * n_bf + bf, lw[leaf * n_bf + bf]
            )
            h.unsafe_ptr().unsafe_store(
                leaf * stat_count * n_bf + 1 * n_bf + bf, lg[leaf * n_bf + bf]
            )
    ctx.enqueue_copy(dst_buf=hist, src_ptr=h.unsafe_ptr())

    var part_stats = ctx.enqueue_create_buffer[DType.float32](
        n_leaves * stat_count
    )
    var hp = ctx.enqueue_create_host_buffer[DType.float32](
        n_leaves * stat_count
    )
    for leaf in range(n_leaves):
        hp.unsafe_ptr().unsafe_store(leaf * stat_count + 0, tot_w[leaf])
        hp.unsafe_ptr().unsafe_store(leaf * stat_count + 1, tot_g[leaf])
    ctx.enqueue_copy(dst_buf=part_stats, src_ptr=hp.unsafe_ptr())

    var skip = ctx.enqueue_create_buffer[DType.uint8](n_bf)
    var hs = ctx.enqueue_create_host_buffer[DType.uint8](n_bf)
    for i in range(n_bf):
        hs.unsafe_ptr().unsafe_store(i, UInt8(0))
    ctx.enqueue_copy(dst_buf=skip, src_ptr=hs.unsafe_ptr())

    var part_ids = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    for i in range(n_leaves):
        hi.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=part_ids, src_ptr=hi.unsafe_ptr())

    var out_score = ctx.enqueue_create_buffer[DType.float32](1)
    var out_bin = ctx.enqueue_create_buffer[DType.uint32](1)
    ctx.synchronize()

    var sc_bff = ctx.enqueue_create_buffer[DType.uint32](n_bf)
    var sc_hbf = ctx.enqueue_create_host_buffer[DType.uint32](n_bf)
    var sc_ffw = ctx.enqueue_create_buffer[DType.float32](n_bf)
    var sc_hfw = ctx.enqueue_create_host_buffer[DType.float32](n_bf)
    for i in range(n_bf):
        sc_hbf.unsafe_ptr().unsafe_store(i, UInt32(i))
        sc_hfw.unsafe_ptr().unsafe_store(i, Float32(1.0))
    ctx.enqueue_copy(dst_buf=sc_bff, src_ptr=sc_hbf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=sc_ffw, src_ptr=sc_hfw.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[
        # This check hand-computes an L2 score, so it must name the L2
        # calcer. CatBoost's DEFAULT is Cosine (`oblivious_tree_options.cpp:22`)
        # and the driver uses it; this fixture is testing the kernel's
        # arithmetic, not the default.
        compute_optimal_splits_kernel[SCORE_FUNCTION_L2]
    ](
        skip.unsafe_ptr(),
        Int32(n_bf),
        # their `TCBinFeature.FeatureId` and `binFeaturesWeights`; this
        # fixture is one bin-feature per feature, so the map is the identity
        # and every weight is 1.0, which is their no-CTR value.
        sc_bff.unsafe_ptr(),
        sc_ffw.unsafe_ptr(),
        hist.unsafe_ptr(),
        part_stats.unsafe_ptr(),
        Int32(stat_count),
        part_ids.unsafe_ptr(),
        Int32(n_leaves),
        Int32(0),  # multiclassOptimization
        lambda_l2,
        # `ScoreStdDev, Random.NextUniformL()` (DEVIATION 137-139); the L2
        # calcer never reads them and this fixture's expectation is
        # noise-free either way
        Float32(0.0),
        UInt64(0),
        out_score.unsafe_ptr(),
        out_bin.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    ctx.synchronize()

    var hos = ctx.enqueue_create_host_buffer[DType.float32](1)
    var hob = ctx.enqueue_create_host_buffer[DType.uint32](1)
    ctx.enqueue_copy(dst_ptr=hos.unsafe_ptr(), src_buf=out_score)
    ctx.enqueue_copy(dst_ptr=hob.unsafe_ptr(), src_buf=out_bin)
    ctx.synchronize()

    # The same sum the kernel forms, on the host, in Float64.
    var best_bf = -1
    var best = -1.0
    for bf in range(n_bf):
        var s = 0.0
        for leaf in range(n_leaves):
            var wl = Float64(lw[leaf * n_bf + bf])
            var gl = Float64(lg[leaf * n_bf + bf])
            var wr = Float64(tot_w[leaf]) - wl
            var gr = Float64(tot_g[leaf]) - gl
            s += gl * gl / (wl + 1.0) + gr * gr / (wr + 1.0)
        print("    bf", bf, "host score", s)
        if s > best:
            best = s
            best_bf = bf

    print("  device picked bin-feature", Int(hob.unsafe_ptr().unsafe_load(0)),
          "score", hos.unsafe_ptr().unsafe_load(0))
    print("  host says best is bin-feature", best_bf, "score", best)
    if Int(hob.unsafe_ptr().unsafe_load(0)) != best_bf:
        raise Error(
            "the score kernel picked "
            + String(Int(hob.unsafe_ptr().unsafe_load(0)))
            + " but the hand calculation says "
            + String(best_bf)
        )
    if abs(Float64(hos.unsafe_ptr().unsafe_load(0)) - best) > 1e-2:
        raise Error("the winning score disagrees with the hand calculation")
    print("  score kernel agrees with the hand calculation")


def check_split_points() raises:
    """The split flags and the in-leaf sequence.

    Two leaves over 32 rows, split at row 12, on a BINARY feature at a
    non-zero shift so a dropped shift or a wrong mask shows up. Row r has
    feature bin `(r // 3) % 2`, giving runs of three rather than an
    alternating pattern, so an off-by-one in the row index is visible.

    Checks three things the kernel must get right:
      - the flag is computed from THIS feature's bits, not a neighbour's,
      - the flag is written at the leaf's OFFSET, not at row 0,
      - the sequence written into `indices` is 0..size-1 WITHIN each leaf,
        which is what the stable partition afterwards permutes.
    """
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
    var n_rows = 32
    var split_at = 12
    var feature_id = 5  # not 0, so the shift is non-trivial

    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var zero32 = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for i in range(n_rows):
        zero32.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=zero32.unsafe_ptr())
    ctx.synchronize()

    # Write a decoy into a neighbouring feature first, then the real one.
    var host_bins = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for r in range(n_rows):
        host_bins.unsafe_ptr().unsafe_store(r, UInt8(1))
    ctx.enqueue_copy(dst_buf=bins, src_ptr=host_bins.unsafe_ptr())
    ctx.enqueue_function[write_compressed_index_kernel](
        Int32(0),
        policy_mask(POLICY_BINARY),
        UInt32(policy_shift(POLICY_BINARY, feature_id + 1)),
        bins.unsafe_ptr(),
        Int32(n_rows),
        cindex.unsafe_ptr(),
        grid_dim=1,
        block_dim=WRITE_BLOCK_SIZE,
    )
    ctx.synchronize()

    var bins2 = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var host_bins2 = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    for r in range(n_rows):
        host_bins2.unsafe_ptr().unsafe_store(r, UInt8((r // 3) % 2))
    ctx.enqueue_copy(dst_buf=bins2, src_ptr=host_bins2.unsafe_ptr())
    ctx.enqueue_function[write_compressed_index_kernel](
        Int32(0),
        policy_mask(POLICY_BINARY),
        UInt32(policy_shift(POLICY_BINARY, feature_id)),
        bins2.unsafe_ptr(),
        Int32(n_rows),
        cindex.unsafe_ptr(),
        grid_dim=1,
        block_dim=WRITE_BLOCK_SIZE,
    )
    ctx.synchronize()

    var load_idx = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var h_load = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        h_load.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=load_idx, src_ptr=h_load.unsafe_ptr())

    var p_off = ctx.enqueue_create_buffer[DType.uint32](2)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](2)
    var leaf_ids = ctx.enqueue_create_buffer[DType.uint32](2)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var h_lid = ctx.enqueue_create_host_buffer[DType.uint32](2)
    h_off.unsafe_ptr().unsafe_store(0, UInt32(0))
    h_off.unsafe_ptr().unsafe_store(1, UInt32(split_at))
    h_sz.unsafe_ptr().unsafe_store(0, UInt32(split_at))
    h_sz.unsafe_ptr().unsafe_store(1, UInt32(n_rows - split_at))
    h_lid.unsafe_ptr().unsafe_store(0, UInt32(0))
    h_lid.unsafe_ptr().unsafe_store(1, UInt32(1))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=leaf_ids, src_ptr=h_lid.unsafe_ptr())

    var sp_bin = ctx.enqueue_create_buffer[DType.uint32](2)
    # their `const TCFeature* splitFeatures` plus `const ui32* splitBins`
    var sp_feats = ctx.enqueue_create_buffer[DType.uint8](
        2 * size_of[CFeature]()
    )
    var h_feats = ctx.enqueue_create_host_buffer[DType.uint8](
        2 * size_of[CFeature]()
    )
    var h_spb = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var hfp = h_feats.unsafe_ptr().bitcast[CFeature]()
    for i in range(2):
        hfp[unsafe_offset=i] = CFeature(
            offset=UInt32(0),
            mask=policy_mask(POLICY_BINARY),
            shift=UInt32(policy_shift(POLICY_BINARY, feature_id)),
            first_fold_index=UInt32(0),
            folds=UInt32(1),
            one_hot_feature=False,  # ordered, so test is >
        )
        h_spb.unsafe_ptr().unsafe_store(i, UInt32(0))  # bin > 0 goes right
    ctx.enqueue_copy(dst_buf=sp_feats, src_ptr=h_feats.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=sp_bin, src_ptr=h_spb.unsafe_ptr())

    var flags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var seq = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var hzf = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var hzs = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        hzf.unsafe_ptr().unsafe_store(r, UInt8(9))
        hzs.unsafe_ptr().unsafe_store(r, UInt32(999))
    ctx.enqueue_copy(dst_buf=flags, src_ptr=hzf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=seq, src_ptr=hzs.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[split_and_make_sequence_kernel](
        cindex.unsafe_ptr(),
        load_idx.unsafe_ptr(),
        p_off.unsafe_ptr(),
        p_sz.unsafe_ptr(),
        leaf_ids.unsafe_ptr(),
        sp_feats.unsafe_ptr().bitcast[CFeature](),
        sp_bin.unsafe_ptr(),
        flags.unsafe_ptr(),
        seq.unsafe_ptr(),
        grid_dim=(1, 2, 1),
        block_dim=(512, 1, 1),
    )
    ctx.synchronize()

    var of = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var os = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    ctx.enqueue_copy(dst_ptr=of.unsafe_ptr(), src_buf=flags)
    ctx.enqueue_copy(dst_ptr=os.unsafe_ptr(), src_buf=seq)
    ctx.synchronize()

    var wrong_flag = 0
    var wrong_seq = 0
    for leaf in range(2):
        var off = 0 if leaf == 0 else split_at
        var size = split_at if leaf == 0 else n_rows - split_at
        for i in range(size):
            var r = off + i
            var want_flag = UInt8(1) if ((r // 3) % 2) > 0 else UInt8(0)
            if of.unsafe_ptr().unsafe_load(r) != want_flag:
                wrong_flag += 1
            if os.unsafe_ptr().unsafe_load(r) != UInt32(i):
                wrong_seq += 1
    print("  32 rows, leaves [0,12) and [12,32), binary feature", feature_id)
    print("  wrong flags:", wrong_flag, " wrong sequence entries:", wrong_seq)
    if wrong_flag != 0 or wrong_seq != 0:
        raise Error("split_points is wrong")
    print("  flags read the right feature, sequence is per-leaf 0..size-1")


def check_partition_update() raises:
    """One leaf becomes two, and the SENTINELS are what is being checked.

    Three cases in one launch, because the sentinel logic is exactly what a
    single happy-path case fails to exercise:

      leaf 0: rows [0,10), flags 0,0,0,0,1,1,1,1,1,1 -> border at 4
      leaf 2: rows [10,16), flags ALL ZERO           -> border at size (6)
      leaf 4: rows [16,24), flags ALL ONE            -> border at 0

    The all-zero and all-one leaves are the ones that matter. Without their
    `i == partSize -> flag0 = 1` and `i == 0 -> flag1 = 0` sentinels, a leaf
    with no transition finds no border and writes nothing, leaving a stale
    partition that silently corrupts the next level.
    """
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
    var n_rows = 24
    var n_leaves = 6

    var flags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var hf = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    for r in range(10):
        hf.unsafe_ptr().unsafe_store(r, UInt8(0) if r < 4 else UInt8(1))
    for r in range(10, 16):
        hf.unsafe_ptr().unsafe_store(r, UInt8(0))
    for r in range(16, 24):
        hf.unsafe_ptr().unsafe_store(r, UInt8(1))
    ctx.enqueue_copy(dst_buf=flags, src_ptr=hf.unsafe_ptr())

    var p_off = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var hh_off = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var hh_sz = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    var h_off2 = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    var h_sz2 = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    for i in range(n_leaves):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_sz.unsafe_ptr().unsafe_store(i, UInt32(0))
    h_off.unsafe_ptr().unsafe_store(0, UInt32(0))
    h_sz.unsafe_ptr().unsafe_store(0, UInt32(10))
    h_off.unsafe_ptr().unsafe_store(2, UInt32(10))
    h_sz.unsafe_ptr().unsafe_store(2, UInt32(6))
    h_off.unsafe_ptr().unsafe_store(4, UInt32(16))
    h_sz.unsafe_ptr().unsafe_store(4, UInt32(8))
    for i in range(n_leaves):
        h_off2.unsafe_ptr().unsafe_store(i, h_off.unsafe_ptr().unsafe_load(i))
        h_sz2.unsafe_ptr().unsafe_store(i, h_sz.unsafe_ptr().unsafe_load(i))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hh_off, src_ptr=h_off2.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hh_sz, src_ptr=h_sz2.unsafe_ptr())

    var left = ctx.enqueue_create_buffer[DType.uint32](3)
    var right = ctx.enqueue_create_buffer[DType.uint32](3)
    var hl = ctx.enqueue_create_host_buffer[DType.uint32](3)
    var hr = ctx.enqueue_create_host_buffer[DType.uint32](3)
    for i in range(3):
        hl.unsafe_ptr().unsafe_store(i, UInt32(2 * i))
        hr.unsafe_ptr().unsafe_store(i, UInt32(2 * i + 1))
    ctx.enqueue_copy(dst_buf=left, src_ptr=hl.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=right, src_ptr=hr.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[update_partitions_after_split_kernel](
        left.unsafe_ptr(),
        right.unsafe_ptr(),
        Int32(3),
        flags.unsafe_ptr(),
        p_off.unsafe_ptr(),
        p_sz.unsafe_ptr(),
        hh_off.unsafe_ptr(),
        hh_sz.unsafe_ptr(),
        grid_dim=(1, 3, 1),
        block_dim=(512, 1, 1),
    )
    ctx.synchronize()

    var oo = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    var os = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    ctx.enqueue_copy(dst_ptr=oo.unsafe_ptr(), src_buf=p_off)
    ctx.enqueue_copy(dst_ptr=os.unsafe_ptr(), src_buf=p_sz)
    ctx.synchronize()

    var want_off = List[Int]()
    var want_sz = List[Int]()
    want_off.append(0);  want_sz.append(4)
    want_off.append(4);  want_sz.append(6)
    want_off.append(10); want_sz.append(6)
    want_off.append(16); want_sz.append(0)
    want_off.append(16); want_sz.append(0)
    want_off.append(16); want_sz.append(8)

    var wrong = 0
    for i in range(n_leaves):
        var go = Int(oo.unsafe_ptr().unsafe_load(i))
        var gs = Int(os.unsafe_ptr().unsafe_load(i))
        if go != want_off[i] or gs != want_sz[i]:
            wrong += 1
            print("    leaf", i, "got", go, gs, "want", want_off[i], want_sz[i])
    print("  mixed -> 4/6, all-zero -> 6 then empty, all-one -> empty then 8")
    print("  wrong partitions:", wrong, "of", n_leaves)
    if wrong != 0:
        raise Error("partition update is wrong")
    print("  border found in all three cases, sentinels work")


def check_zero_and_copy() raises:
    """`ZeroHistograms` clears only the named set; `CopyHistograms` stages
    the parent into the sibling slot.

    Four leaves, three bin-features, one stat, every cell seeded with the
    leaf id plus one so a wrong slot is obvious.

    Zero is asked to clear leaves 1 and 3 ONLY. The indirection through
    `histIds` is the whole point: `build_necessary_histograms` decides which
    leaves need a fresh build, so leaves 0 and 2 must survive untouched. A
    kernel that cleared a contiguous range would pass a test that zeroed
    everything and fail here.

    Copy then stages leaf 0 into leaf 2, which is what makes the in-place
    subtraction afterwards legal.
    """
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
    var n_leaves = 4
    var n_bf = 3
    var total = n_leaves * n_bf

    var hist = ctx.enqueue_create_buffer[DType.float32](total)
    var h = ctx.enqueue_create_host_buffer[DType.float32](total)
    for leaf in range(n_leaves):
        for b in range(n_bf):
            h.unsafe_ptr().unsafe_store(
                leaf * n_bf + b, Float32(Float64(leaf) + 1.0)
            )
    ctx.enqueue_copy(dst_buf=hist, src_ptr=h.unsafe_ptr())

    var ids = ctx.enqueue_create_buffer[DType.uint32](2)
    var hids = ctx.enqueue_create_host_buffer[DType.uint32](2)
    hids.unsafe_ptr().unsafe_store(0, UInt32(1))
    hids.unsafe_ptr().unsafe_store(1, UInt32(3))
    ctx.enqueue_copy(dst_buf=ids, src_ptr=hids.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[zero_histograms_kernel](
        ids.unsafe_ptr(),
        Int32(n_bf),
        hist.unsafe_ptr(),
        grid_dim=(1, 2, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    var o1 = ctx.enqueue_create_host_buffer[DType.float32](total)
    ctx.enqueue_copy(dst_ptr=o1.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()

    var wrong = 0
    for leaf in range(n_leaves):
        var want = Float32(0.0) if (leaf == 1 or leaf == 3) else Float32(
            Float64(leaf) + 1.0
        )
        for b in range(n_bf):
            if abs(o1.unsafe_ptr().unsafe_load(leaf * n_bf + b) - want) > Float32(1e-4):
                wrong += 1
    print("  zero cleared leaves {1,3} only; untouched cells wrong:", wrong)
    if wrong != 0:
        raise Error("zero_histograms cleared the wrong leaves")

    var left = ctx.enqueue_create_buffer[DType.uint32](1)
    var right = ctx.enqueue_create_buffer[DType.uint32](1)
    var hl = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var hr = ctx.enqueue_create_host_buffer[DType.uint32](1)
    hl.unsafe_ptr().unsafe_store(0, UInt32(0))
    hr.unsafe_ptr().unsafe_store(0, UInt32(2))
    ctx.enqueue_copy(dst_buf=left, src_ptr=hl.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=right, src_ptr=hr.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[copy_histograms_kernel](
        left.unsafe_ptr(),
        right.unsafe_ptr(),
        Int32(1),
        Int32(n_bf),
        hist.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    var o2 = ctx.enqueue_create_host_buffer[DType.float32](total)
    ctx.enqueue_copy(dst_ptr=o2.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()

    var wrong2 = 0
    for b in range(n_bf):
        if abs(o2.unsafe_ptr().unsafe_load(2 * n_bf + b) - Float32(1.0)) > Float32(1e-4):
            wrong2 += 1
        if abs(o2.unsafe_ptr().unsafe_load(0 * n_bf + b) - Float32(1.0)) > Float32(1e-4):
            wrong2 += 1
    print("  copy staged leaf 0 into leaf 2; wrong cells:", wrong2)
    if wrong2 != 0:
        raise Error("copy_histograms is wrong")
    print("  zero is set-indexed and copy stages the parent for subtraction")


def check_stable_partition() raises:
    """Stable partition: order preserved within each side, across chunks.

    Two leaves in one launch. Leaf 0 is 300 rows, deliberately LARGER than
    the 256-thread block, because the running zero and one counts carried
    between chunks are exactly what a small leaf never exercises. Leaf 1 is
    17 rows with an irregular pattern.

    Stability is checked directly rather than inferred: for each side the
    source indices must come out strictly increasing. A non-stable partition
    still produces one flag transition, so `update_partitions_after_split`
    would still find a boundary and every count would still be right, while
    the leaf silently held different rows.
    """
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
    var n0 = 300
    var n1 = 17
    var n_rows = n0 + n1

    var flags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var hf = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    for r in range(n0):
        hf.unsafe_ptr().unsafe_store(r, UInt8(1) if (r % 3) == 0 else UInt8(0))
    for r in range(n1):
        hf.unsafe_ptr().unsafe_store(n0 + r, UInt8(1) if (r % 2) == 1 else UInt8(0))
    ctx.enqueue_copy(dst_buf=flags, src_ptr=hf.unsafe_ptr())

    var p_off = ctx.enqueue_create_buffer[DType.uint32](2)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](2)
    var lids = ctx.enqueue_create_buffer[DType.uint32](2)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var h_lid = ctx.enqueue_create_host_buffer[DType.uint32](2)
    h_off.unsafe_ptr().unsafe_store(0, UInt32(0))
    h_off.unsafe_ptr().unsafe_store(1, UInt32(n0))
    h_sz.unsafe_ptr().unsafe_store(0, UInt32(n0))
    h_sz.unsafe_ptr().unsafe_store(1, UInt32(n1))
    h_lid.unsafe_ptr().unsafe_store(0, UInt32(0))
    h_lid.unsafe_ptr().unsafe_store(1, UInt32(1))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=lids, src_ptr=h_lid.unsafe_ptr())

    var gmap = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var sflags = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    ctx.synchronize()

    var max_chunks = (n0 + PARTITION_BLOCK - 1) // PARTITION_BLOCK
    var chunk_zeros = ctx.enqueue_create_buffer[DType.uint32](2 * max_chunks)
    var chunk_offsets = ctx.enqueue_create_buffer[DType.uint32](2 * max_chunks)
    var leaf_zeros = ctx.enqueue_create_buffer[DType.uint32](2)
    launch_stable_partition(
        ctx, 2, n0, lids, p_off, p_sz, flags,
        chunk_zeros, chunk_offsets, leaf_zeros, gmap, sflags,
    )
    ctx.synchronize()

    var om = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    var osf = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    ctx.enqueue_copy(dst_ptr=om.unsafe_ptr(), src_buf=gmap)
    ctx.enqueue_copy(dst_ptr=osf.unsafe_ptr(), src_buf=sflags)
    ctx.synchronize()

    var bad = 0
    for leaf in range(2):
        var off = 0 if leaf == 0 else n0
        var size = n0 if leaf == 0 else n1
        var seen_one = False
        var prev_zero = -1
        var prev_one = -1
        var n_zero = 0
        for i in range(size):
            var srcpos = Int(om.unsafe_ptr().unsafe_load(off + i))
            var f = Int(hf.unsafe_ptr().unsafe_load(off + srcpos))
            # sorted_flags must agree with the source row's real flag
            if Int(osf.unsafe_ptr().unsafe_load(off + i)) != f:
                bad += 1
            if f == 0:
                if seen_one:
                    bad += 1  # a zero after a one: not partitioned
                if srcpos <= prev_zero:
                    bad += 1  # not stable within the zero side
                prev_zero = srcpos
                n_zero += 1
            else:
                seen_one = True
                if srcpos <= prev_one:
                    bad += 1  # not stable within the one side
                prev_one = srcpos
        if leaf == 0:
            print("  leaf 0:", size, "rows (>", PARTITION_BLOCK, "), zeros:", n_zero)
        else:
            print("  leaf 1:", size, "rows, zeros:", n_zero)

    print("  violations (order, stability, flag agreement):", bad)
    if bad != 0:
        raise Error("stable partition is wrong")
    print("  partitioned, stable within each side, across chunk boundaries")


def check_one_byte_bits[bits: Int](n_stats: Int = 1, rows_per_fold: Int = 10, scattered: Bool = False) raises:
    """The one-byte kernel at `bits` bits, against a hand-computable answer.

    Four features per `UInt32`, `2^bits` folds each. Feature f at row r gets
    bin `(r + f) % folds`, so with `10 * folds` rows every fold holds exactly
    10 rows for every feature and every cell must read 10.0.

    **The bit width is the whole point of running this more than once.**
    `InnerHistBitsCount = bits - 5`, so 5 bits runs ONE pass through the
    inner loop and 8 bits runs EIGHT, with only the lanes whose `higherBin`
    equals the current pass writing on each. A wrong slot is wrong at every
    width; a wrong PASS is wrong only above 5 bits, so the 5-bit case alone
    cannot see it.
    """
    comptime n_folds = 1 << bits
    var ctx = DeviceContext()
    # Scratch for the multi-block flush signature; unused at one block.
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
    var n_features = 4
    var n_rows = rows_per_fold * n_folds

    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var zero32 = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for i in range(n_rows):
        zero32.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=zero32.unsafe_ptr())
    ctx.synchronize()

    var host_bins = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    # `(r + f) % n_folds` gives consecutive rows consecutive bins, so lanes
    # in a warp systematically land on DIFFERENT slots. That pattern hides any
    # collision the slot arithmetic has. `scattered` uses a hash instead, so
    # lanes can hit the same bin, which is what real data does.
    var counts = List[Int]()
    for _ in range(n_features * n_folds):
        counts.append(0)
    for f in range(n_features):
        for r in range(n_rows):
            var v = (r + f) % n_folds
            if scattered:
                var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
                x ^= x << 13
                x ^= x >> 17
                x ^= x << 5
                v = Int(x % UInt32(n_folds))
            counts[f * n_folds + v] += 1
            host_bins.unsafe_ptr().unsafe_store(r, UInt8(v))
        ctx.enqueue_copy(dst_buf=bins, src_ptr=host_bins.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(0),
            UInt32(255),
            UInt32(24 - 8 * f),
            bins.unsafe_ptr(),
            Int32(n_rows),
            cindex.unsafe_ptr(),
            grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    var p_off = ctx.enqueue_create_buffer[DType.uint32](1)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](1)
    var p_ids = ctx.enqueue_create_buffer[DType.uint32](1)
    var a = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var bb = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var cc = ctx.enqueue_create_host_buffer[DType.uint32](1)
    a.unsafe_ptr().unsafe_store(0, UInt32(0))
    bb.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    cc.unsafe_ptr().unsafe_store(0, UInt32(0))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=a.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=bb.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_ids, src_ptr=cc.unsafe_ptr())

    # One plane per stat, laid out [stat][row] as the kernels expect. With
    # `n_stats = 2` the second plane is what the mixed path uses and what no
    # check covered until now.
    var stats = ctx.enqueue_create_buffer[DType.float32](n_stats * n_rows)
    var hst = ctx.enqueue_create_host_buffer[DType.float32](n_stats * n_rows)
    for i in range(n_stats * n_rows):
        hst.unsafe_ptr().unsafe_store(i, Float32(1.0))
    ctx.enqueue_copy(dst_buf=stats, src_ptr=hst.unsafe_ptr())

    var group_size = n_features * n_folds
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

    var sums = ctx.enqueue_create_buffer[DType.float32](n_stats * group_size)
    var zf = ctx.enqueue_create_host_buffer[DType.float32](
        n_stats * group_size
    )
    for i in range(n_stats * group_size):
        zf.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=sums, src_ptr=zf.unsafe_ptr())
    var acc_unused = ctx.enqueue_create_buffer[DType.int32](
        n_stats * group_size
    )
    ctx.synchronize()

    # DEVIATION 95: the histogram kernels read `fixed_scale` from DEVICE
    # memory. `upload_scale` returns the buffer and the CALLER keeps it
    # alive: Mojo frees at its last use, and `.unsafe_ptr()` is a use, so
    # without the `_ = scale_keep^` below the buffer is dead before the
    # kernel runs and the next allocation lands on it.
    var scale_keep = upload_scale(ctx, Float32(1.0))
    var scale_ptr = rebind[MutPointer[Float32, MutAnyOrigin]](
        scale_keep.unsafe_ptr()
    )

    ctx.enqueue_function[
        one_byte_hist_kernel[bits, HIST_SMEM_WARP_PRIVATE_F32]
    ](
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
        sums.unsafe_ptr(),
        # grid.x is 1 here, so the fixed-point flush never engages and the
        # accumulator is only along for the signature.
        acc_unused.unsafe_ptr(),
        scale_ptr,
        Int32(1),
        Int32(n_stats),
        grid_dim=(1, 1, n_stats),
        block_dim=(ONE_BYTE_BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()
    _ = scale_keep^

    var out = ctx.enqueue_create_host_buffer[DType.float32](
        n_stats * group_size
    )
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=sums)
    ctx.synchronize()

    print(
        "  4 features x",
        n_folds,
        "folds,",
        n_rows,
        "rows,",
        bits,
        "bits ->",
        1 << (bits - 5),
        "pass(es)",
    )
    print("  expected every cell:", rows_per_fold, " rows:", n_rows)
    var wrong = 0
    for s in range(n_stats):
        for f in range(n_features):
            for fold in range(n_folds):
                var got = out.unsafe_ptr().unsafe_load(
                    s * group_size + f * n_folds + fold
                )
                var want = counts[f * n_folds + fold]
                if abs(got - Float32(Float64(want))) > Float32(1e-3):
                    wrong += 1
                    if wrong <= 4:
                        print(
                            "      stat", s, "feature", f, "fold", fold,
                            "got", got, "want", want,
                        )
    print("    feature 0 folds 0..3:",
          out.unsafe_ptr().unsafe_load(0), out.unsafe_ptr().unsafe_load(1))
    print("  stats", n_stats, " wrong cells:", wrong, "of", n_stats * group_size)
    if wrong != 0:
        raise Error("the one-byte histogram is wrong")
    print("  one-byte at", bits, "bits computes the right answer")


def check_gather_matches_direct() raises:
    """A gather histogram with a permuted index equals the direct one.

    The gather kernels read `cindex[indices[position]]` where the direct ones
    read `bins[position]`. With the IDENTITY index the two must agree exactly.
    With a REVERSED index they must STILL agree, because a histogram is a sum
    over a set of rows and does not care what order they are visited in.

    That second case is the one worth having: it is the only check here that
    would catch a gather kernel which reads the index but then indexes the
    bins by position anyway, which is exactly the bug the multi-level tree hit
    and took three rounds to find.
    """
    var ctx = DeviceContext()
    var acc_scratch = ctx.enqueue_create_buffer[DType.int32](8192)
    var n_rows = 1024
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
        q1.unsafe_ptr().unsafe_store(f, UInt32(1))
        q2.unsafe_ptr().unsafe_store(f, UInt32(f))
        q3.unsafe_ptr().unsafe_store(f, UInt32(0))
        q4.unsafe_ptr().unsafe_store(f, UInt32(n_features))
    ctx.enqueue_copy(dst_buf=folds, src_ptr=q1.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=fold_off, src_ptr=q2.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=grp_off, src_ptr=q3.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=grp_sz, src_ptr=q4.unsafe_ptr())

    var sums = ctx.enqueue_create_buffer[DType.float32](n_features)
    var zf = ctx.enqueue_create_host_buffer[DType.float32](n_features)
    for i in range(n_features):
        zf.unsafe_ptr().unsafe_store(i, Float32(0.0))

    var idx = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var hidx = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    var out = ctx.enqueue_create_host_buffer[DType.float32](n_features)

    # DEVIATION 95: the histogram kernels read `fixed_scale` from DEVICE
    # memory. `upload_scale` returns the buffer and the CALLER keeps it
    # alive: Mojo frees at its last use, and `.unsafe_ptr()` is a use, so
    # without the `_ = scale_keep^` below the buffer is dead before the
    # kernel runs and the next allocation lands on it.
    var scale_keep = upload_scale(ctx, Float32(1.0))
    var scale_ptr = rebind[MutPointer[Float32, MutAnyOrigin]](
        scale_keep.unsafe_ptr()
    )
    # direct
    ctx.enqueue_copy(dst_buf=sums, src_ptr=zf.unsafe_ptr())
    ctx.synchronize()
    ctx.enqueue_function[binary_hist_kernel](
        folds.unsafe_ptr(), fold_off.unsafe_ptr(), grp_off.unsafe_ptr(),
        grp_sz.unsafe_ptr(), Int32(n_features), cindex.unsafe_ptr(),
        Int32(n_rows), Int32(0), stats.unsafe_ptr(), Int32(n_rows),
        p_off.unsafe_ptr(), p_sz.unsafe_ptr(), p_ids.unsafe_ptr(),
        sums.unsafe_ptr(), acc_scratch.unsafe_ptr(), scale_ptr,
        Int32(1), Int32(1),
        grid_dim=(1, 1, 1), block_dim=(BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=sums)
    ctx.synchronize()
    var direct = List[Float32]()
    for f in range(n_features):
        direct.append(out.unsafe_ptr().unsafe_load(f))

    var bad = 0
    for which in range(2):
        for r in range(n_rows):
            if which == 0:
                hidx.unsafe_ptr().unsafe_store(r, UInt32(r))
            else:
                hidx.unsafe_ptr().unsafe_store(r, UInt32(n_rows - 1 - r))
        ctx.enqueue_copy(dst_buf=idx, src_ptr=hidx.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=sums, src_ptr=zf.unsafe_ptr())
        ctx.synchronize()
        ctx.enqueue_function[binary_hist_gather_kernel[False]](
            folds.unsafe_ptr(), fold_off.unsafe_ptr(), grp_off.unsafe_ptr(),
            grp_sz.unsafe_ptr(), Int32(n_features), cindex.unsafe_ptr(),
            Int32(n_rows), Int32(0), idx.unsafe_ptr(), stats.unsafe_ptr(),
            Int32(n_rows), p_off.unsafe_ptr(), p_sz.unsafe_ptr(),
            p_ids.unsafe_ptr(), sums.unsafe_ptr(), acc_scratch.unsafe_ptr(),
            scale_ptr, Int32(1), Int32(1),
            grid_dim=(1, 1, 1), block_dim=(BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=sums)
        ctx.synchronize()
        var label = "identity" if which == 0 else "reversed"
        var wrong = 0
        for f in range(n_features):
            if abs(out.unsafe_ptr().unsafe_load(f) - direct[f]) > Float32(1e-3):
                wrong += 1
        print("    gather with", label, "index vs direct: wrong", wrong)
        bad += wrong
    # the gather launches above are the LAST readers of the scale; the
    # hold goes past the loop, not past the first launch inside it.
    _ = scale_keep^

    if bad != 0:
        raise Error("the gather histogram disagrees with the direct one")
    print("  gather matches direct under identity AND permutation")


def main() raises:
    # STANDALONE DRIVER. Every call `probe_main.mojo` makes under "binary
    # histogram correctness (GPU)", in its order and with its arguments --
    # including the four one-byte accumulator widths, the two-stat arms,
    # the 2048-row arm that accumulates four times rather than twice, and
    # the SCATTERED arm where lanes can collide on one slot.
    print("binary histogram correctness (GPU):")
    check_binary_histogram()
    check_gather_matches_direct()
    check_two_partitions()
    check_half_byte_histogram()
    check_one_byte_bits[5]()
    check_one_byte_bits[6]()
    check_one_byte_bits[7]()
    check_one_byte_bits[8]()
    check_one_byte_bits[6](2)
    check_one_byte_bits[8](2)
    check_one_byte_bits[6](2, 32)
    check_one_byte_bits[6](2, 32, True)
    check_subtraction()
    check_scan()
    check_scores()
    check_split_points()
    check_partition_update()
    check_zero_and_copy()
    check_stable_partition()
