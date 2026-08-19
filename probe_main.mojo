from mojo_only.launch_probe import probe
from mojo_only.level_check import check_one_level
from mojo_only.level_bench import bench_histogram_only, bench_level
from mojo_only.pack_check import check_packing
from mojo_only.hist_check import (
    check_binary_histogram,
    check_half_byte_histogram,
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
    var leaves = List[LeafRecord]()
    # pair A on path 7: sizes 900 and 100, so the SMALLER (id 1) is built.
    leaves.append(LeafRecord(900, HISTOGRAMS_ZEROES, 7, False))
    leaves.append(LeafRecord(100, HISTOGRAMS_ZEROES, 7, False))
    # a leaf still valid from the parent path: no work at all.
    leaves.append(LeafRecord(500, HISTOGRAMS_PREVIOUS_PATH, 9, False))
    # pair B on path 3, BOTH terminal: neither histogram is ever read.
    leaves.append(LeafRecord(400, HISTOGRAMS_ZEROES, 3, True))
    leaves.append(LeafRecord(600, HISTOGRAMS_ZEROES, 3, True))
    # an unpaired rebuild on path 5.
    leaves.append(LeafRecord(250, HISTOGRAMS_ZEROES, 5, False))

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

    if len(plan.compute_ids) != 2:
        raise Error("expected 2 builds, got " + String(len(plan.compute_ids)))
    if Int(plan.compute_ids[0]) != 1:
        raise Error("pair A must build the SMALLER sibling, leaf 1")
    if Int(plan.subtract_from[0]) != 0 or Int(plan.subtract_what[0]) != 1:
        raise Error("pair A must derive leaf 0 as parent - leaf 1")
    print("  smaller-side rule holds, terminal pair skipped")


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
    print("ONE FULL LEVEL (GPU, end to end):")
    check_one_level()
    print()
    print("LEVEL TIMING:")
    bench_level(500000, 5)
    bench_histogram_only(500000, 20)
    print()
    print("packing round trip (GPU):")
    check_packing()
    print()
    print("binary histogram correctness (GPU):")
    check_binary_histogram()
    check_two_partitions()
    check_half_byte_histogram()
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
