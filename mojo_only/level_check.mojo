"""One full level, end to end, against a host calculation.

Everything this exercises is already verified in isolation. What is NOT yet
verified is the SEQUENCE: nine kernels in order, each reading what the last
one wrote. A failure here is a wiring failure, which is the whole reason the
pieces were checked separately first.
"""

from gbdt.models.oblivious_model import TBinarySplit
from max.gpu.host import DeviceContext

from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    features_per_int,
    policy_mask,
    policy_shift,
)
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    run_one_level,
    run_tree,
    run_tree_layout,
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
    # INDEPENDENT bins, not the periodic `((r // (f+1)) + f) % 2` this used
    # to carry. That pattern makes features near-duplicates of each other, so
    # every leaf comes out with the same distribution, every feature ties on
    # score, the argmax deterministically re-picks the same one, and the tree
    # re-splits on an already-used feature into empty children. That looked
    # exactly like a porting bug and was a property of the test data.
    var seed = 0x2545F491
    for f in range(n_features):
        for r in range(n_rows):
            # xorshift, so features are uncorrelated with each other and with
            # the row index.
            var x = UInt32(r * 2654435761 + f * 40503 + seed)
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            host_bin.unsafe_ptr().unsafe_store(r, UInt8(Int(x & 1)))
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
        # `choose_scale` is specified against a sum of MAGNITUDES, not a
        # signed total: gradients cancel, so a signed total can be far
        # smaller than the largest histogram cell it is meant to bound.
        if g < 0.0:
            tg += -g
        else:
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
    # A tree that never splits conserves every row and produces exactly
    # 2^depth partitions, so conservation alone cannot see it.
    if max_depth > 0 and nonempty < 2:
        raise Error(
            "the tree never split: " + String(nonempty)
            + " non-empty leaf of " + String(len(sizes))
        )
    print("  every row accounted for at depth", max_depth)


def check_mixed_tree(max_depth: Int) raises:
    """A tree over MIXED feature widths: binary, half-byte and one-byte.

    Every check before this used 32 uniform binary features, which is one
    policy, one histogram launch and a 64-cell histogram. This is the shape
    CatBoost is actually built for and the one every kernel in the port was
    written to handle.

    16 features: 8 binary (1 fold), 4 half-byte (8 folds), 4 one-byte (64
    folds), so the flat histogram is 8 + 32 + 256 = 296 cells and three
    launches per level instead of one.

    Invariant is the same one that caught the earlier wiring bugs: leaf sizes
    must sum to `n_rows` at every depth, because a split moves rows and
    creates none.
    """
    var ctx = DeviceContext()
    var n_rows = 4096

    var folds = List[Int]()
    for _ in range(8):
        folds.append(1)
    for _ in range(4):
        folds.append(8)
    for _ in range(4):
        folds.append(64)
    var n_features = len(folds)

    # Columns: binary 8 -> 1 column, half-byte 4 -> 1, one-byte 4 -> 1.
    var n_columns = 3
    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows * n_columns)
    var z = ctx.enqueue_create_host_buffer[DType.uint32](n_rows * n_columns)
    for i in range(n_rows * n_columns):
        z.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())
    ctx.synchronize()

    var lay = build_layout(folds)
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for f in range(n_features):
        ref cf = lay.features[f]
        for r in range(n_rows):
            var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            # `folds` is the BORDER count, so a feature takes bins 0..Folds,
            # which is Folds + 1 values. `% folds[f]` makes every binary
            # feature (Folds = 1) the constant 0, and a split resolved onto a
            # constant feature puts every row on one side, which is what a
            # tree that grows and never splits looks like.
            hb.unsafe_ptr().unsafe_store(
                r, UInt8(Int(x % UInt32(folds[f] + 1)))
            )
        ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * n_rows),
            cf.mask,
            cf.shift,
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
        # `choose_scale` is specified against a sum of MAGNITUDES, not a
        # signed total: gradients cancel, so a signed total can be far
        # smaller than the largest histogram cell it is meant to bound.
        if g < 0.0:
            tg += -g
        else:
            tg += g
    ctx.enqueue_copy(dst_buf=stats, src_ptr=hs.unsafe_ptr())

    var row_index = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_rows)
    for r in range(n_rows):
        hi.unsafe_ptr().unsafe_store(r, UInt32(r))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())
    ctx.synchronize()

    # No cursor here: growth only. `apply_to_cursor` defaults off, so this
    # buffer is never written.
    var scratch_cursor = ctx.enqueue_create_buffer[DType.float32](1)
    var _sp = List[TBinarySplit]()
    var _lv = List[Float32]()
    # THE DRAIN BUDGET IS ASSERTED HERE, NOT INSIDE THE LIBRARY.
    #
    # `sync_budget` is ours and CatBoost has no counterpart, so arming it on
    # the training path made a diagnostic into a way for a real fit to fail.
    # It now defaults to unbounded there and the CHECK supplies the tight
    # number, which is where a claim about our own drain discipline belongs.
    #
    # `2 * max_depth + 1` is theirs read off their loop: two host blocks per
    # level, `bestProps.Read(propsCpu)` (`greedy_search_helper.cpp:517`) and
    # the leaf-size read in `RebuildLeavesSizes`
    # (`split_properties_helper.cpp:802`), plus one at the end. A third drain
    # per level fails this check, which is the point.
    var _lo = List[Int]()
    var sizes = run_tree_layout(
        ctx, n_rows, folds, max_depth, cindex, stats, row_index,
        scratch_cursor,
        Float32(tw), Float32(tg),
        _sp, _lv, _lo,
        sync_budget=2 * max_depth + 1,
    )

    var total = 0
    var nonempty = 0
    for i in range(len(sizes)):
        total += sizes[i]
        if sizes[i] > 0:
            nonempty += 1

    print(
        "  mixed: 8 binary + 4 half-byte + 4 one-byte =",
        lay.hist_cells,
        "flat bins, 3 launches/level",
    )
    print(
        "  depth", max_depth, "->", len(sizes), "leaves,", nonempty,
        "non-empty, rows", total, "of", n_rows,
    )
    if total != n_rows:
        raise Error(
            "rows lost or duplicated: " + String(total) + " of "
            + String(n_rows)
        )
    if len(sizes) != (1 << max_depth):
        raise Error("wrong leaf count")
    # CONSERVATION IS NOT ENOUGH. A tree that puts every row on one side of
    # every split conserves rows perfectly, produces exactly 2^depth
    # partitions and passes both checks above while splitting nothing. That
    # is what a constant feature or a scrambled histogram looks like from
    # here, and it went unnoticed until this line existed.
    if nonempty < 2:
        raise Error(
            "the tree never split: " + String(nonempty)
            + " non-empty leaf of " + String(len(sizes))
        )
    print("  mixed-width tree grows, splits and conserves every row")
