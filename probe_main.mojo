from mojo_only.launch_probe import probe
from mojo_only.permuted_ids_check import check_permuted_leaf_ids
from mojo_only.copy_histograms_check import check_copy_histograms
from mojo_only.boosting_check import check_boosting_learns
from mojo_only.binarization_check import check_binarization
from mojo_only.options_check import check_options
from mojo_only.mixed_hist_probe import probe_mixed_histogram
from mojo_only.layout_check import (
    check_feature_blocks,
    check_layout,
    check_split_resolution,
)
from mojo_only.level_check import check_mixed_tree, check_one_level, check_tree
from mojo_only.hist_dump_check import check_hist_depends_on_partition
from mojo_only.level_bench import (
    bench_histogram_only,
    bench_level,
    bench_partition_only,
    bench_remaining_phases,
    bench_replication_interleaved,
    bench_wide_histogram_interleaved,
    bench_realistic,
    bench_subtraction,
    bench_tree,
    bench_tree_shapes,
)
from mojo_only.pack_check import check_packing
from mojo_only.hist_check import (
    check_binary_histogram,
    check_gather_matches_direct,
    check_half_byte_histogram,
    check_one_byte_bits,
    check_scan,
    check_scores,
    check_partition_update,
    check_stable_partition,
    check_zero_and_copy,
    check_split_points,
    check_subtraction,
    check_two_partitions,
)
from ported.methods.greedy_subsets_searcher.structure_searcher_template import (
    grow_tree_schedule,
)
from mojo_only.fixed_point import (
    SCALE_LIMIT,
    choose_scale,
    dequantize,
    max_representable,
    quantize,
)
from mojo_only.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_BIT_IDENTICAL,
    COLUMN_NVIDIA,
    K_HIST_BINARY,
    K_HIST_HALF_BYTE,
    K_HIST_ONE_BYTE,
    TARGET_COLUMN,
    block_size_for,
    column_has_float_atomics,
    column_lane_width,
    column_name,
    column_shared_limit,
    hist_floats_per_thread_for,
)
from ported.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
    HIST_SIZE,
    REDUCE_WIDTH,
)
from ported.methods.greedy_subsets_searcher.kernel.hist_one_byte import (
    ONE_BYTE_BLOCK_SIZE,
    ONE_BYTE_HIST_SIZE,
)
from ported.methods.greedy_subsets_searcher.split_properties_helper import (
    HISTOGRAMS_PREVIOUS_PATH,
    HISTOGRAMS_ZEROES,
    LeafRecord,
    build_necessary_histograms,
)


def check_level_plan() raises:
    """Exercise the smaller-side rule on a level with every case in it.

    Not a test suite: the smallest thing that shows the classification
    computes a right answer rather than merely compiling. Four leaves in two
    sibling pairs, plus one leaf whose histogram survives from the parent
    path, plus a terminal pair that must be skipped entirely.
    """
    # REWRITTEN 2026-08-19. The old fixture was built for the port's
    # inverted state machine: it made `Zeroes` leaves pair up and treated
    # `PreviousPath` as no work. Theirs is the opposite
    # (`split_properties_helper.cpp:1295-1304`), so the fixture had to move
    # with it. The corrected planner rejected the old fixture on its first
    # run, which is the only reason this is a probe and not a silent bug.
    var leaves = List[LeafRecord]()
    # pair A, both holding the PARENT histogram from path 7: sizes 900 and
    # 100, so the SMALLER (id 1) is built and id 0 is derived.
    leaves.append(LeafRecord(900, HISTOGRAMS_PREVIOUS_PATH, 7, False))
    leaves.append(LeafRecord(100, HISTOGRAMS_PREVIOUS_PATH, 7, False))
    # a leaf whose slot holds NOTHING: it must be built outright.
    leaves.append(LeafRecord(500, HISTOGRAMS_ZEROES, 9, False))
    # pair B on path 3, BOTH terminal: neither histogram is ever read.
    leaves.append(LeafRecord(400, HISTOGRAMS_PREVIOUS_PATH, 3, True))
    leaves.append(LeafRecord(600, HISTOGRAMS_PREVIOUS_PATH, 3, True))
    # a lone rebuild candidate on path 5. Their `CB_ENSURE` at `:1311` says
    # this is only legal when the leaf is terminal, so it is terminal here.
    leaves.append(LeafRecord(250, HISTOGRAMS_PREVIOUS_PATH, 5, True))

    var plan = build_necessary_histograms(leaves)

    print("  builds:", len(plan.compute_ids), "of 6 leaves")
    for i in range(len(plan.compute_ids)):
        var id = Int(plan.compute_ids[i])
        print("    build leaf", id, "size", Int(leaves[id].size))
    for i in range(len(plan.subtract_from)):
        print(
            "    derive leaf",
            Int(plan.subtract_from[i]),
            "= parent -",
            Int(plan.subtract_what[i]),
        )
    print("  builds saved by subtraction:", plan.builds_saved())

    # leaf 2 is `Zeroes` so it is built outright; leaf 1 is the smaller of
    # pair A so it is built and leaf 0 derived. Pair B is skipped as both
    # terminal, and leaf 5 is a legal lone terminal.
    if len(plan.compute_ids) != 2:
        raise Error("expected 2 builds, got " + String(len(plan.compute_ids)))
    if Int(plan.compute_ids[0]) != 2:
        raise Error("a Zeroes leaf must be built outright, leaf 2")
    if Int(plan.compute_ids[1]) != 1:
        raise Error("pair A must build the SMALLER sibling, leaf 1")
    if len(plan.subtract_from) != 1:
        raise Error(
            "expected 1 pair, got " + String(len(plan.subtract_from))
        )
    if Int(plan.subtract_from[0]) != 0 or Int(plan.subtract_what[0]) != 1:
        raise Error("pair A must derive leaf 0 as parent - leaf 1")
    var upd = plan.updated_ids()
    if len(upd) != 3:
        raise Error("allUpdatedLeaves must be computeLeaves + bigLeaves")
    print("  smaller-side rule holds, terminal pair skipped")

    # A lone NON-terminal rebuild candidate is their CB_ENSURE at `:1311`,
    # and it has to fire, or the pairing rule is silently accepting a leaf
    # whose sibling went missing.
    var bad = List[LeafRecord]()
    bad.append(LeafRecord(250, HISTOGRAMS_PREVIOUS_PATH, 5, False))
    var fired = False
    try:
        _ = build_necessary_histograms(bad)
    except e:
        fired = True
    if not fired:
        raise Error("a lone non-terminal rebuild candidate must be refused")
    print("  lone non-terminal rebuild candidate refused, as CB_ENSURE does")


def show_matrix() raises:
    """Print the resolved table, and prove the kernels READ it.

    The last two lines are the reach check: `BLOCK_SIZE` is imported from the
    half-byte template, not recomputed here, so if it agrees with the matrix
    the kernel is taking its constants from the table rather than restating
    them. Importing a module is not reading it.
    """
    print("  column          shared    lanes   float-atomic   half-byte block")
    var cols = List[Int]()
    cols.append(COLUMN_BIT_IDENTICAL)
    cols.append(COLUMN_APPLE)
    cols.append(COLUMN_NVIDIA)
    cols.append(COLUMN_AMD)
    var blocks = List[Int]()
    blocks.append(block_size_for[K_HIST_HALF_BYTE, COLUMN_BIT_IDENTICAL]())
    blocks.append(block_size_for[K_HIST_HALF_BYTE, COLUMN_APPLE]())
    blocks.append(block_size_for[K_HIST_HALF_BYTE, COLUMN_NVIDIA]())
    blocks.append(block_size_for[K_HIST_HALF_BYTE, COLUMN_AMD]())
    for ci in range(len(cols)):
        var c = cols[ci]
        var blk = blocks[ci]
        print(
            "  ",
            column_name(c),
            "\t",
            column_shared_limit(c) // 1024,
            "KB\t",
            column_lane_width(c),
            "\t",
            "yes" if column_has_float_atomics(c) else "NO",
            "\t\t",
            blk,
        )
    print()
    print("  building against column:", column_name(TARGET_COLUMN))
    print(
        "  half-byte kernel reads   BLOCK_SIZE",
        BLOCK_SIZE,
        " HIST_SIZE",
        HIST_SIZE,
        " REDUCE_WIDTH",
        REDUCE_WIDTH,
    )
    print(
        "  one-byte kernel reads    BLOCK_SIZE",
        ONE_BYTE_BLOCK_SIZE,
        " HIST_SIZE",
        ONE_BYTE_HIST_SIZE,
    )
    var want = block_size_for[K_HIST_HALF_BYTE, TARGET_COLUMN]()
    if BLOCK_SIZE != want:
        raise Error(
            "the half-byte kernel is NOT reading the matrix: kernel says "
            + String(BLOCK_SIZE)
            + ", matrix says "
            + String(want)
        )
    print("  kernels agree with the matrix, so the table is reached")


def check_fixed_point() raises:
    """The overflow guarantee, exercised at the boundary.

    The claim is that a scale derived from the GLOBAL sum of magnitudes makes
    overflow impossible for every partial sum, because any leaf's rows are a
    subset of all rows. The check is the worst case: one leaf that contains
    every row, all of one sign, so its accumulation is exactly the global sum
    and lands exactly on the limit.
    """
    var rows = List[Float64]()
    rows.append(3.5)
    rows.append(-2.25)
    rows.append(0.125)
    rows.append(-9.0)
    rows.append(0.0009765625)

    var total_mag = 0.0
    for i in range(len(rows)):
        total_mag += abs(rows[i])
    var scale = choose_scale(total_mag)
    print("  sum of magnitudes", total_mag, " scale", scale)

    # Worst case: every row in one leaf, all magnitudes adding in one
    # direction. This is the largest partial sum the device can ever form.
    var worst = Int64(0)
    for i in range(len(rows)):
        worst += Int64(quantize(abs(rows[i]), scale))
    print("  worst-case slot", worst, " limit", Int64(SCALE_LIMIT))
    if Float64(worst) > SCALE_LIMIT:
        raise Error(
            "the scale does NOT bound the worst case: slot "
            + String(worst)
            + " exceeds "
            + String(SCALE_LIMIT)
        )

    # Round trip: quantize then dequantize is within one quantum.
    var q = quantize(rows[0], scale)
    var back = dequantize(Int64(q), scale)
    var quantum = 1.0 / scale
    if abs(back - rows[0]) > quantum:
        raise Error("round trip lost more than one quantum")
    print("  round trip", rows[0], "->", back, " quantum", quantum)

    # Order independence, which is the property the whole thing exists for.
    var fwd = Int64(0)
    for i in range(len(rows)):
        fwd += Int64(quantize(rows[i], scale))
    var rev = Int64(0)
    for i in range(len(rows) - 1, -1, -1):
        rev += Int64(quantize(rows[i], scale))
    if fwd != rev:
        raise Error("integer accumulation is not order independent")
    print("  forward and reverse accumulation agree exactly:", fwd)
    print("  max representable at this scale", max_representable(scale))


def show_tree_schedule() raises:
    """One depth-8 tree on covtype, as the schedule CatBoost's loop produces.

    The property being checked is the one their own header states: the number
    of kernels per level does not depend on the leaf count or on the dataset
    (`compute_by_blocks_helper.h:87-92`). So launches stay flat as leaves
    double, and launches-per-leaf must FALL. Ours does 73 per tree where
    mojotrees' leaf-wise plane does 2,303.
    """
    var sched = grow_tree_schedule(8, 464809, 3)
    print("  depth  leaves   builds  pairs  launches  launches/leaf")
    var total = 0
    for i in range(len(sched)):
        ref s = sched[i]
        total += s.launches
        print(
            "  ",
            s.depth,
            "\t",
            s.leaf_count,
            "\t",
            s.histogram_builds,
            "\t",
            s.subtraction_pairs,
            "\t",
            s.launches,
            "\t",
            s.launches_per_leaf(),
        )
    print("  total launches for the tree:", total)
    print("  mojotrees leaf-wise plane, same shape: 2303")
    var first_launches = sched[0].launches
    var last_launches = sched[len(sched) - 1].launches
    if last_launches != first_launches:
        raise Error(
            "launches per level must not depend on the leaf count; level 0"
            " had "
            + String(first_launches)
            + " and the last had "
            + String(last_launches)
        )
    print("  launch count is flat in leaf count, as their design requires")


def main() raises:
    print("options:")
    check_options()
    print()
    print("compressed index layout (host):")
    check_layout()
    check_feature_blocks()
    check_split_resolution()
    print()
    print("ONE FULL LEVEL (GPU, end to end):")
    check_hist_depends_on_partition()
    check_one_level()
    check_tree(1)
    check_tree(3)
    check_tree(6)
    check_mixed_tree(4)
    check_mixed_tree(6)
    print()
    print("LEVEL TIMING:")
    bench_level(500000, 5)
    bench_histogram_only(500000, 20)
    bench_partition_only(500000, 20)
    bench_remaining_phases(500000, 20)
    bench_wide_histogram_interleaved(500000, 7)
    bench_replication_interleaved(500000, 6, 5)
    # EVERY number above this line came from 32 uniform binary features,
    # which is the shape CatBoost's design suits least. This is the same
    # tree over four policy mixes at matched feature count.
    bench_tree_shapes(200000, 6, 3)
    # CatBoost's reference on this box is ~32 ms/tree at 800k x 100.
    # A SWEEP, because "why are we slower" is not answerable from one point.
    # Flat in rows means launch overhead; linear in rows means bandwidth.
    bench_realistic(100000, 100, 6, 3)
    bench_realistic(200000, 100, 6, 3)
    bench_realistic(400000, 100, 6, 3)
    bench_realistic(800000, 100, 6, 3)
    bench_realistic(800000, 100, 1, 3)
    bench_tree(500000, 1, 3)
    bench_tree(500000, 2, 3)
    bench_tree(500000, 3, 3)
    bench_tree(500000, 4, 3)
    bench_tree(500000, 5, 3)
    bench_tree(500000, 6, 3)
    bench_tree(500000, 7, 3)
    bench_tree(500000, 8, 3)
    print()
    print("packing round trip (GPU):")
    check_packing()
    print()
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
    # 2048 rows: the probe's row count, 4 accumulation iterations not 2.
    check_one_byte_bits[6](2, 32)
    # SCATTERED bins: lanes can hit the same slot, which the uniform
    # (r + f) %% n_folds pattern never does.
    check_one_byte_bits[6](2, 32, True)
    check_subtraction()
    check_scan()
    check_scores()
    check_split_points()
    check_partition_update()
    check_zero_and_copy()
    check_stable_partition()
    print()
    print("tree schedule:")
    show_tree_schedule()
    print()
    print("fixed point (no CatBoost counterpart):")
    check_fixed_point()
    print()
    print("kernel matrix:")
    show_matrix()
    print()
    print("level plan:")
    check_level_plan()
    print("kernels:")
    probe()
    print()
    print("one-byte only, PRE-BRIDGE (what the kernel wrote):")
    probe_mixed_histogram(0, 0, 4, True)
    print()
    print("histogram slices, ONE-BYTE ONLY (GPU):")
    probe_mixed_histogram(0, 0, 4)
    print()
    print("histogram slices, BINARY + ONE-BYTE (GPU):")
    probe_mixed_histogram(8, 0, 4)
    print()
    print("mixed histogram slices (GPU):")
    probe_mixed_histogram(8, 4, 4)
    print()
    print("non-contiguous leaf ids (GPU):")
    check_permuted_leaf_ids()
    print()
    print("copy_histograms in isolation:")
    check_copy_histograms()
    print()
    print("leaf-id / replication matrix (GPU):")
    # Four arms, because the two failures found here were independent: the
    # dense-vs-looked-up writeback needs a PERMUTED id list to show, and the
    # unreplicable one-byte store needs REPLICAS > 1 and shows on contiguous
    # ids too. Either arm alone misses one of them.
    check_permuted_leaf_ids(0, 1, True)
    check_permuted_leaf_ids(1, 1, True)
    check_permuted_leaf_ids(0, 16, False)
    check_permuted_leaf_ids(0, 16, True)
    check_permuted_leaf_ids(1, 16, True)
    print()
    print("BOOSTING, end to end:")
    check_boosting_learns()
    print()
    print("border selection (host, as theirs is):")
    check_binarization()
