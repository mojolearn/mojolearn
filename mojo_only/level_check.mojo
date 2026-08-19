"""One full level, end to end, against a host calculation.

Everything this exercises is already verified in isolation. What is NOT yet
verified is the SEQUENCE: nine kernels in order, each reading what the last
one wrote. A failure here is a wiring failure, which is the whole reason the
pieces were checked separately first.
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
from ported.methods.greedy_subsets_searcher.greedy_search_helper import (
    run_one_level,
    run_tree,
)


def check_one_level() raises:
    """512 rows, 32 binary features, a planted best split.

    Feature 7 is given a genuinely informative pattern: rows where it is 1
    carry a large positive gradient and rows where it is 0 carry a small
    negative one, so the Newton score for splitting on it dwarfs every other
    feature. Every other feature gets a pattern uncorrelated with the
    gradient.

    Two things are checked against a host calculation:
      - the level picks feature 7, and
      - the two child sizes match the true counts of that feature's bins.

    The second is the one that proves the WIRING rather than the scoring: the
    child sizes come out of the split flags, through the stable partition,
    through the boundary search, so a break anywhere in that chain gives the
    right split choice and the wrong partition.
    """
    var ctx = DeviceContext()
    var n_rows = 512
    var n_features = features_per_int(POLICY_BINARY)
    var informative = 7

    # --- bins ------------------------------------------------------------
    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var z = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for i in range(n_rows):
        z.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())
    ctx.synchronize()

    var host_bin = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    var informative_ones = 0
    for f in range(n_features):
        for r in range(n_rows):
            var v = 0
            if f == informative:
                v = 1 if (r % 4) == 0 else 0
            else:
                v = ((r // (f + 2)) + f) % 2
            host_bin.unsafe_ptr().unsafe_store(r, UInt8(v))
            if f == informative and v == 1:
                informative_ones += 1
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

    # --- stats: [weight plane | gradient plane], their line layout -------
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

    var res = run_one_level(
        ctx,
        n_rows,
        n_features,
        cindex,
        stats,
        row_index,
        Float32(total_w),
        Float32(total_g),
    )

    var want_right = informative_ones          # bin > 0 goes right
    var want_left = n_rows - informative_ones

    print("  512 rows, 32 binary features, feature", informative, "planted")
    print("  chosen bin-feature:", res.chosen_bin_feature, " score:", res.score)
    print("  child sizes:", res.left_size, "/", res.right_size)
    print("  host says:  ", want_left, "/", want_right)

    if res.chosen_bin_feature != informative:
        raise Error(
            "the level chose bin-feature "
            + String(res.chosen_bin_feature)
            + " but the planted split is "
            + String(informative)
        )
    if res.left_size != want_left or res.right_size != want_right:
        raise Error(
            "child sizes "
            + String(res.left_size)
            + "/"
            + String(res.right_size)
            + " disagree with the host's "
            + String(want_left)
            + "/"
            + String(want_right)
        )
    print("  ONE FULL LEVEL: nine kernels in sequence, correct end to end")


def check_tree(max_depth: Int) raises:
    """A whole oblivious tree, and the invariant that catches most wiring bugs.

    **Leaf sizes must sum to n_rows at every depth.** A split moves rows
    between leaves and creates none, so any break in the chain, a partition
    that drops rows, a border search that writes a stale size, a gather that
    permutes across a leaf boundary, shows up as a sum that is not n_rows.
    It is a weak check per leaf and a strong one over the tree, which is what
    you want from an invariant that runs at every depth.

    Also checked: `2^depth` leaves exist, and no leaf is negative.
    """
    var ctx = DeviceContext()
    var n_rows = 4096
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
            host_bin.unsafe_ptr().unsafe_store(
                r, UInt8(((r // (f + 1)) + f) % 2)
            )
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

    var sizes = run_tree(
        ctx, n_rows, n_features, max_depth, cindex, stats, row_index,
        Float32(tw), Float32(tg),
    )

    var total = 0
    var negative = 0
    var nonempty = 0
    for i in range(len(sizes)):
        total += sizes[i]
        if sizes[i] < 0:
            negative += 1
        if sizes[i] > 0:
            nonempty += 1

    print("  depth", max_depth, "->", len(sizes), "leaves,", nonempty, "non-empty")
    print("  leaf sizes sum to", total, "of", n_rows)
    if len(sizes) != (1 << max_depth):
        raise Error("expected " + String(1 << max_depth) + " leaves")
    if total != n_rows:
        raise Error(
            "rows lost or duplicated: leaves sum to "
            + String(total)
            + " of "
            + String(n_rows)
        )
    if negative != 0:
        raise Error("a leaf has a negative size")
    print("  every row accounted for at depth", max_depth)
