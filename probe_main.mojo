from mojo_only.launch_probe import probe
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


def main() raises:
    print("level plan:")
    check_level_plan()
    print("kernels:")
    probe()
