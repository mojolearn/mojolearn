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
    POLICY_HALF_BYTE,
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
from catboost.cuda.methods.greedy_subsets_searcher.kernel.compute_scores import (
    compute_optimal_splits_kernel,
)
from catboost.cuda.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    scan_histograms_kernel,
    substract_histograms_kernel,
)
from catboost.cuda.methods.greedy_subsets_searcher.kernel.hist_half_byte import (
    half_byte_hist_kernel,
)
from catboost.cuda.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
)


def check_binary_histogram() raises:
    var ctx = DeviceContext()

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
        Int32(1),
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

    # grid y = 2: BOTH leaves in ONE launch. That is the design property.
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
        Int32(2),
        Int32(1),
        grid_dim=(1, 2, 1),
        block_dim=(BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()

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
    var zerof = ctx.enqueue_create_host_buffer[DType.float32](group_size)
    for i in range(group_size):
        zerof.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=sums, src_ptr=zerof.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[half_byte_hist_kernel](
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
        Int32(1),
        grid_dim=(1, 1, 1),
        block_dim=(BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()

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

    ctx.enqueue_function[scan_histograms_kernel](
        first_bin.unsafe_ptr(),
        folds_b.unsafe_ptr(),
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

    ctx.enqueue_function[compute_optimal_splits_kernel](
        skip.unsafe_ptr(),
        Int32(n_bf),
        hist.unsafe_ptr(),
        part_stats.unsafe_ptr(),
        Int32(stat_count),
        part_ids.unsafe_ptr(),
        Int32(n_leaves),
        lambda_l2,
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
