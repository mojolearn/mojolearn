"""Does the layout builder assign policies and bits correctly?

Pure host arithmetic, so it needs no GPU. It is the step that decides what
every kernel then reads, so a wrong shift here is a wrong histogram
everywhere and nothing downstream would look suspicious.
"""

from ported.gpu_data.compressed_index_builder import build_layout
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
