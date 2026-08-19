"""Does the layout builder assign policies and bits correctly?

Pure host arithmetic, so it needs no GPU. It is the step that decides what
every kernel then reads, so a wrong shift here is a wrong histogram
everywhere and nothing downstream would look suspicious.
"""

from ported.gpu_data.compressed_index_builder import build_layout
from ported.gpu_data.feature_blocks import blocks_for
from ported.methods.greedy_subsets_searcher.greedy_search_helper import resolve_split
from ported.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
    policy_name,
)


def check_layout() raises:
    """A mixed dataset: binary, half-byte and one-byte features together.

    Fold counts chosen to land on both sides of every boundary:
      1     -> binary    (MaxFolds 1)
      2, 15 -> half-byte (MaxFolds 15)
      16    -> one-byte  (first value binary and half-byte cannot hold)
      200   -> one-byte
      0     -> constant, keeps its slot with folds 0
    """
    var folds = List[Int]()
    folds.append(1)
    folds.append(2)
    folds.append(15)
    folds.append(16)
    folds.append(200)
    folds.append(0)
    folds.append(1)

    var lay = build_layout(folds)

    print("  feature  folds  policy       column  shift  first_fold")
    for i in range(len(folds)):
        ref f = lay.features[i]
        print(
            "   ",
            i,
            "      ",
            folds[i],
            "   ",
            policy_name(lay.policy_of[i]),
            "  ",
            Int(f.offset),
            "    ",
            Int(f.shift),
            "    ",
            Int(f.first_fold_index),
        )

    var wrong = 0
    # policies
    if lay.policy_of[0] != POLICY_BINARY:
        wrong += 1
    if lay.policy_of[1] != POLICY_HALF_BYTE:
        wrong += 1
    if lay.policy_of[2] != POLICY_HALF_BYTE:
        wrong += 1
    if lay.policy_of[3] != POLICY_ONE_BYTE:
        wrong += 1
    if lay.policy_of[4] != POLICY_ONE_BYTE:
        wrong += 1

    # the two binary features share a column; the shift advances
    if Int(lay.features[0].offset) != Int(lay.features[6].offset):
        wrong += 1
        print("    binary features must share a column")
    if Int(lay.features[0].shift) == Int(lay.features[6].shift):
        wrong += 1
        print("    binary features must not share a shift")

    # binary shift counts from the top: 32 - (1 + local) * 1
    if Int(lay.features[0].shift) != 31:
        wrong += 1
    if Int(lay.features[6].shift) != 30:
        wrong += 1

    # histogram slices are a running total across ALL policies, so the score
    # kernel can walk one flat array without knowing about policies.
    var expect = 0
    for i in range(len(folds)):
        if Int(lay.features[i].first_fold_index) != expect:
            wrong += 1
            print("    feature", i, "first_fold should be", expect)
        expect += folds[i]

    print("  columns:", lay.columns, " histogram cells:", lay.hist_cells)
    if lay.hist_cells != 1 + 2 + 15 + 16 + 200 + 0 + 1:
        wrong += 1
        print("    hist_cells must be the sum of folds")
    print("  layout errors:", wrong)
    if wrong != 0:
        raise Error("the compressed index layout is wrong")
    print("  policies, shared columns, shifts and slices all correct")


def check_feature_blocks() raises:
    """Do the per-policy blocks carry the right slices?

    Same mixed dataset as `check_layout`. What is being checked is the split
    into blocks: which features land in which block, that a policy with no
    features produces NO block (and therefore no launch), and that
    `fold_offset` is a running total WITHIN the block rather than within the
    whole histogram, because the writeback strides by the block's own
    `group_size`.
    """
    var folds = List[Int]()
    folds.append(1)
    folds.append(2)
    folds.append(15)
    folds.append(16)
    folds.append(200)
    folds.append(0)
    folds.append(1)

    var lay = build_layout(folds)
    var blocks = blocks_for(lay, 1024)

    print("  blocks:", len(blocks), "(one per policy PRESENT)")
    var wrong = 0
    for b in range(len(blocks)):
        ref blk = blocks[b]
        var total = 0
        for k in range(blk.count()):
            total += Int(blk.folds[k])
        print(
            "    ",
            policy_name(blk.policy),
            " features",
            blk.count(),
            " total folds",
            total,
            " first column",
            blk.first_column,
        )
        # fold_offset must be a running total within the block
        var cursor = 0
        for k in range(blk.count()):
            if Int(blk.fold_offset[k]) != cursor:
                wrong += 1
                print("      fold_offset", k, "should be", cursor)
            if Int(blk.group_size[k]) != total:
                wrong += 1
                print("      group_size must be the block's total folds")
            cursor += Int(blk.folds[k])

    if len(blocks) != 3:
        wrong += 1
        print("    expected 3 blocks for this dataset")
    print("  block errors:", wrong)
    if wrong != 0:
        raise Error("feature blocks are wrong")
    print("  every policy's block carries its own contiguous fold slices")


def check_split_resolution() raises:
    """Does a flat bin-feature index resolve to the right (feature, bin)?

    The score kernel returns an index into ONE flat histogram spanning every
    policy. Turning that into a split needs the owning feature and the bin
    within it. Getting this wrong produces a split on the wrong feature at a
    plausible bin, which trains, converges to something, and is silently the
    wrong model.

    Checked at every boundary: the first and last bin of each feature, and
    across the policy changes where the flat index crosses from one block's
    slice to another's.
    """
    var folds = List[Int]()
    folds.append(1)
    folds.append(2)
    folds.append(15)
    folds.append(16)
    folds.append(200)
    folds.append(0)
    folds.append(1)

    var lay = build_layout(folds)
    var wrong = 0
    var expect_first = 0

    for i in range(len(folds)):
        if folds[i] == 0:
            continue
        # first bin of this feature
        var lo = resolve_split(lay, expect_first)
        if lo.feature != i or lo.bin != 0:
            wrong += 1
            print(
                "    bin-feature", expect_first, "-> feature", lo.feature,
                "bin", lo.bin, " expected feature", i, "bin 0",
            )
        # last bin of this feature
        var last = expect_first + folds[i] - 1
        var hi = resolve_split(lay, last)
        if hi.feature != i or hi.bin != folds[i] - 1:
            wrong += 1
            print(
                "    bin-feature", last, "-> feature", hi.feature,
                "bin", hi.bin, " expected feature", i, "bin", folds[i] - 1,
            )
        expect_first += folds[i]

    # past the end must raise rather than return a plausible answer
    var raised = False
    try:
        var bad = resolve_split(lay, lay.hist_cells + 5)
        _ = bad.feature
    except:
        raised = True
    if not raised:
        wrong += 1
        print("    an out-of-range bin-feature must raise")

    print("  resolved", expect_first, "bin-features across 3 policies")
    print("  resolution errors:", wrong)
    if wrong != 0:
        raise Error("split resolution is wrong")
    print("  every feature boundary and policy crossing resolves correctly")
