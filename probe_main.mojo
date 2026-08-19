from mojo_only.launch_probe import probe
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
from catboost.cuda.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
    HIST_SIZE,
    REDUCE_WIDTH,
)
from catboost.cuda.methods.greedy_subsets_searcher.kernel.hist_one_byte import (
    ONE_BYTE_BLOCK_SIZE,
    ONE_BYTE_HIST_SIZE,
)
from catboost.cuda.methods.greedy_subsets_searcher.split_properties_helper import (
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


def main() raises:
    print("kernel matrix:")
    show_matrix()
    print()
    print("level plan:")
    check_level_plan()
    print("kernels:")
    probe()
