"""Do the two flat-bin walks agree when INPUT ORDER IS NOT BLOCK ORDER?

NO CATBOOST COUNTERPART, because CatBoost cannot have this bug. Their
per-feature `FirstFoldIndex` and their per-group write offset into the flat
histogram come off ONE builder in ONE pass: `GroupBinFeatureOffsets` records
`BinFeaturesBuilder.GetCurrentSize(dev)` at the top of `AddGroup`
(`compute_by_blocks_helper.cpp:189`), each feature's `FirstFoldIndex` is
stamped from that same builder a few lines below (`:218`), and `AddGroup`
runs once per group in group order (`:341`). Two numbers off one cursor
cannot disagree.

This port has two walks, so they CAN disagree, and for one day they did:
`build_layout` numbered bins in INPUT FEATURE ORDER while
`launch_histograms_for_blocks` wrote them in POLICY-BLOCK ORDER. The two
agree exactly when the input order already is block order, which is what a
binary-first dataset is, which is what every check in this repository used.

**So this check exists to be run on a dataset ordered the other way**:
one-byte features first, binary second, which is covtype (10 continuous
columns, then 44 binary). Under the old numbering the binary features were
written near flat bin 0 while `first_fold_index` claimed they began near
1280, and the scan, `resolve_split` and the skip mask all read cells
belonging to some other feature.

Host arithmetic only. No device, no kernels. The bug was in the arithmetic.
"""

from ported.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from ported.gpu_data.feature_blocks import blocks_for
from ported.gpu_data.grid_policy import policy_name
from ported.methods.greedy_subsets_searcher.greedy_search_helper import (
    resolve_split,
)


def covtype_shaped_folds() -> List[Int]:
    """ONE-BYTE FIRST, BINARY SECOND. The order that used to break.

    10 continuous columns at 200 folds, then 44 binary columns at 1 fold,
    which is covtype's shape and the reverse of block order. A half-byte
    feature and a constant are dropped in the middle so all three policies
    are present and the constant's slot is exercised too.
    """
    var folds = List[Int]()
    for _ in range(10):
        folds.append(200)
    folds.append(8)  # half-byte, in the middle of the one-byte run
    folds.append(0)  # constant: keeps its slot, owns no bins
    for _ in range(44):
        folds.append(1)
    folds.append(15)  # a second half-byte, AFTER the binary run
    return folds^


def bridge_destinations(
    layout: CompressedIndexLayout,
) raises -> List[Int]:
    """Where the BRIDGE puts each feature's first bin, derived independently.

    This is `launch_histograms_for_blocks` recomputed on the host and nothing
    else: walk the blocks in the order `blocks_for` emits them, carry a
    running `block_first_bin`, and add the feature's `fold_offset` within its
    block. It deliberately does NOT read `first_fold_index`, because
    `first_fold_index` is the thing under test.

    A feature with no bins gets `-1`; nothing addresses it.
    """
    var dest = List[Int]()
    for _ in range(len(layout.features)):
        dest.append(-1)

    var blocks = blocks_for(layout, 1024)
    var block_first_bin = 0
    for b in range(len(blocks)):
        ref blk = blocks[b]
        for k in range(blk.count()):
            dest[blk.feature_ids[k]] = block_first_bin + Int(
                blk.fold_offset[k]
            )
        var total = 0
        for k in range(blk.count()):
            total += Int(blk.folds[k])
        block_first_bin += total
    return dest^


def planted(feature: Int, bin: Int) -> Int:
    """A value that names its (feature, bin) and nothing else.

    A check whose expected value is the same in every cell verifies the
    total and says nothing about PLACEMENT, which is the only thing wrong
    here. Every cell gets a distinct hashed value so a slice landing under
    the wrong feature reads as a wrong VALUE, not as a right one.
    """
    var x = UInt32(feature * 2654435761 + bin * 40503 + 0x2545F491)
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return Int(x % UInt32(1000000007))


def check_bin_order() raises:
    """The two walks must agree on a dataset that is not in block order."""
    var folds = covtype_shaped_folds()
    var lay = build_layout(folds)
    var dest = bridge_destinations(lay)
    var wrong = 0

    # ---- 1. the two walks, cell for cell -----------------------------
    for i in range(len(folds)):
        if folds[i] == 0:
            continue
        if Int(lay.features[i].first_fold_index) != dest[i]:
            wrong += 1
            print(
                "    feature", i, "policy", policy_name(lay.policy_of[i]),
                ": first_fold_index", Int(lay.features[i].first_fold_index),
                "but the bridge writes it at", dest[i],
            )

    # ---- 2. PLANT AND READ BACK --------------------------------------
    # Fill a flat histogram the way the bridge fills it, then read every
    # feature's slice the way the scan, the score kernel and `resolve_split`
    # read it: through `first_fold_index`. A cell that comes back holding
    # another feature's hash is the exact failure this file is named for.
    var flat = List[Int]()
    for _ in range(lay.hist_cells):
        flat.append(-1)
    for i in range(len(folds)):
        if folds[i] == 0:
            continue
        for b in range(folds[i]):
            flat[dest[i] + b] = planted(i, b)

    # every cell must have been written exactly once: the destinations must
    # tile the flat histogram with no hole and no overlap.
    for c in range(lay.hist_cells):
        if flat[c] == -1:
            wrong += 1
            print("    flat bin", c, "is written by no feature")

    for i in range(len(folds)):
        if folds[i] == 0:
            continue
        var lo = Int(lay.features[i].first_fold_index)
        for b in range(folds[i]):
            if flat[lo + b] != planted(i, b):
                wrong += 1
                print(
                    "    feature", i, "bin", b,
                    ": reading through first_fold_index gives another",
                    "feature's cell at flat bin", lo + b,
                )

    # ---- 3. `resolve_split` must invert the same map -----------------
    for i in range(len(folds)):
        if folds[i] == 0:
            continue
        var first = resolve_split(lay, dest[i])
        if first.feature != i or first.bin != 0:
            wrong += 1
            print(
                "    flat bin", dest[i], "resolves to feature",
                first.feature, "bin", first.bin,
                " expected feature", i, "bin 0",
            )
        var last_bin = dest[i] + folds[i] - 1
        var last = resolve_split(lay, last_bin)
        if last.feature != i or last.bin != folds[i] - 1:
            wrong += 1
            print(
                "    flat bin", last_bin, "resolves to feature",
                last.feature, "bin", last.bin,
                " expected feature", i, "bin", folds[i] - 1,
            )

    # ---- 4. a policy's columns must be CONTIGUOUS --------------------
    # The histogram kernels address a feature group as
    # `cindex_base + bins_line_size * groupIndex`, so a policy whose columns
    # are interleaved with another policy's reads the other one's bits. Same
    # root cause: a walk in input order where the kernels assume block order.
    for policy in range(3):
        var col_lo = -1
        var col_hi = -1
        var count = 0
        for i in range(len(folds)):
            if lay.policy_of[i] != policy or folds[i] == 0:
                continue
            var col = Int(lay.features[i].offset)
            count += 1
            if col_lo < 0 or col < col_lo:
                col_lo = col
            if col > col_hi:
                col_hi = col
        if count == 0:
            continue
        # the columns this policy touches must be exactly `lo..hi` and no
        # other policy may own one of them
        for i in range(len(folds)):
            if lay.policy_of[i] == policy or folds[i] == 0:
                continue
            var other = Int(lay.features[i].offset)
            if other >= col_lo and other <= col_hi:
                wrong += 1
                print(
                    "    column", other, "sits inside",
                    policy_name(policy), "'s range", col_lo, "..", col_hi,
                    "but belongs to", policy_name(lay.policy_of[i]),
                )

    print("  features:", len(folds), " flat bins:", lay.hist_cells)
    print("  bin-order errors:", wrong)
    if wrong != 0:
        raise Error(
            "the layout's first_fold_index and the bridge's write offset"
            " disagree; see compute_by_blocks_helper.cpp:189 and :218"
        )
    print("  input order is one-byte first and the two walks still agree")


def check_bin_order_has_teeth() raises:
    """SABOTAGE: the old input-order numbering must FAIL the check above.

    A check that cannot tell a working layout from a broken one is worth
    nothing, and the digest-style checks in this repository have passed
    no-ops before. So this recomputes `first_fold_index` the way it used to
    be computed, as a running total in INPUT FEATURE ORDER ignoring policy,
    and asserts that it disagrees with the bridge on this dataset. If this
    function stops finding a disagreement, `check_bin_order` has gone blind.
    """
    var folds = covtype_shaped_folds()
    var lay = build_layout(folds)
    var dest = bridge_destinations(lay)

    var input_order_first = List[Int]()
    var cursor = 0
    for i in range(len(folds)):
        input_order_first.append(cursor)
        cursor += folds[i]

    var disagreements = 0
    for i in range(len(folds)):
        if folds[i] == 0:
            continue
        if input_order_first[i] != dest[i]:
            disagreements += 1

    print("  input-order numbering disagrees on", disagreements, "features")
    if disagreements == 0:
        raise Error(
            "this dataset no longer separates input order from block order,"
            " so check_bin_order cannot catch the bug it exists for"
        )
    # and the current builder must be the one that agrees
    for i in range(len(folds)):
        if folds[i] == 0:
            continue
        if Int(lay.features[i].first_fold_index) != dest[i]:
            raise Error(
                "build_layout is still numbering bins in input order"
            )
    print("  the check has teeth: the old numbering fails, the new one holds")
