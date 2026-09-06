# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The port against CATBOOST'S OWN OUTPUT, not against a tally we wrote.

WHY THIS FILE EXISTS
--------------------
Every other check in this repository compares the device against a HOST TALLY
written in this repository. That is the same reading of CatBoost's source
expressed twice. When the reading is wrong the tally is wrong in exactly the
same way and the check reports 0 wrong, which is what "0 wrong of 3168" has
been worth up to now. It says we agree with us.

`tools/catboost_reference.py` did not close that hole either. It runs CatBoost
and prints ms/tree. It compares no number at all.

CatBoost 1.2.10 is installed on this machine and has been all along. This file
reads what it actually DECIDED, from `bench/oracle.txt`, and holds the port to
it. Regenerate with

    pixi run -e bench python tools/catboost_oracle.py > bench/oracle.json

THE FIRST THING TO COMPARE IS THE BORDERS, AND IT HAS TO BE
-----------------------------------------------------------
A split is reported as a bin index into a compressed index. Two ports that
binarize differently cannot be compared at all past that point, because bin 7
of ours and bin 7 of theirs are different thresholds and every later
disagreement is a consequence rather than a finding. So border parity is not
a warm-up, it is the precondition for reading anything else, and it is the one
comparison that needs no device.

`best_split` here is our GreedyLogSum (`grid_creator/binarization.mojo`),
against `border_count = 15` and the same 4096 x 16 matrix CatBoost was
trained on.

WHAT A FAILURE HERE MEANS
-------------------------
Not that a kernel is wrong. That the quantization our whole compressed index
is built on does not agree with theirs, which would make every split index in
every comparison downstream meaningless, and would also mean our accuracy has
never been comparable to theirs at any point in this port.

THE CATEGORICAL FIXTURE, AND WHY IT IS ONE-HOT ONLY
----------------------------------------------------
`bench/oracle_cat.txt` adds three ONE-HOT categorical columns (k = 3, 5, 8)
beside eight numeric ones. Two records carry it and both are additive, so
the three numeric fixtures parse and run exactly as they always did:
`cat <flat_feature_index> <k>` declares a column categorical, and
`catsplit <tree> <flat_feature_index> <code>` is an equality level, written
interleaved with `split` in depth order.

It is one-hot only because CatBoost's CPU learner cannot be asked for the
CTR set this port mirrors. `TCatFeatureParams.default()` here is their GPU
`simple_ctr` -- `Borders` plus `FeatureFreq` -- and
`IsSupportedCtrType(CPU, FeatureFreq)` is FALSE
(`private/libs/options/restrictions.h:18-48`). Their GPU arm, which does
have it, cannot run on this machine (`archive/reference/PORTING.md` 109). Feature
combinations are a second, independent blocker: `max_ctr_complexity` is
refused above 1 here where CatBoost defaults to 4. `archive/reference/PORTING.md` 113 and
`tools/catboost_cat_oracle.py` carry the full argument.

A one-hot split is compared the same way a numeric one is -- per feature,
per bin, AND per split TYPE, because `> code` and `== code` name the same
(feature, bin) pair and partition the rows differently.
"""

from max.gpu.host import DeviceContext

from gbdt.grid_creator.binarization import best_split, binarize
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.grid_policy import POLICY_ONE_BYTE, policy_name
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.doc_parallel_boosting import TAdditiveModel, fit
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    BIN_SPLIT_TAKE_GREATER,
    bin_split_type_name,
)


struct Oracle(Movable):
    """`bench/oracle.txt`, parsed. Flat text on purpose; see the module note."""

    var rows: Int
    var feats: Int
    var depth: Int
    var trees: Int
    var border_count: Int
    var train_mse: Float64
    var baseline_mse: Float64
    #: `borders[f]` is CatBoost's border list for float feature `f`.
    var borders: List[List[Float32]]
    #: Column major, `x[f][r]`, because every consumer here is per feature.
    var x: List[List[Float32]]
    #: `split_feature[t][d]` and `split_border[t][d]`, in CatBoost's own
    #: depth order, which is the order an oblivious tree applies them.
    var split_feature: List[List[Int]]
    var split_border: List[List[Float32]]
    #: The target CatBoost was trained on, so our port trains on the SAME y.
    var y: List[Float32]
    #: `leaf_values[t]`, in CatBoost's own leaf order.
    var leaf_values: List[List[Float32]]
    #: `cat_cardinality[f]` is 0 for a NUMERIC column and `k` for a
    #: ONE-HOT categorical one, from the `cat` records
    #: `tools/catboost_cat_oracle.py` writes. A fixture with none of them
    #: is a numeric fixture and every path below behaves as it always has.
    var cat_cardinality: List[Int]
    #: `split_is_cat[t][d]`, parallel to `split_feature` / `split_border`.
    #: For a categorical level the `split_border` slot holds the dense
    #: CATEGORY CODE rather than a threshold, because that is what a
    #: one-hot split tests.
    var split_is_cat: List[List[Bool]]

    def __init__(out self):
        self.rows = 0
        self.feats = 0
        self.depth = 0
        self.trees = 0
        self.border_count = 0
        self.train_mse = 0.0
        self.baseline_mse = 0.0
        self.borders = List[List[Float32]]()
        self.x = List[List[Float32]]()
        self.split_feature = List[List[Int]]()
        self.split_border = List[List[Float32]]()
        self.y = List[Float32]()
        self.leaf_values = List[List[Float32]]()
        self.cat_cardinality = List[Int]()
        self.split_is_cat = List[List[Bool]]()


def _fields(line: String) raises -> List[String]:
    """Split on spaces, dropping empty pieces.

    `String.split(" ")` yields an empty piece for every doubled space, and
    the oracle writer is not guaranteed to avoid them, so the empties are
    filtered rather than assumed away.
    """
    var out = List[String]()
    for piece in line.split(" "):
        var s = String(piece)
        if len(s.bytes()) > 0:
            out.append(s)
    return out^


def load_oracle(path: String) raises -> Oracle:
    var o = Oracle()
    var text: String
    with open(path, "r") as f:
        text = f.read()

    var row_cursor = 0
    for line in text.split("\n"):
        if len(String(line).bytes()) == 0:
            continue
        if String(line).startswith("#"):
            continue
        var t = _fields(String(line))
        if len(t) == 0:
            continue
        var kind = t[0]

        if kind == String("config"):
            o.rows = Int(t[1])
            o.feats = Int(t[2])
            o.depth = Int(t[3])
            o.trees = Int(t[4])
            o.border_count = Int(t[7])
            for _ in range(o.feats):
                o.borders.append(List[Float32]())
                o.x.append(List[Float32]())
                o.cat_cardinality.append(0)
            for _ in range(o.trees):
                o.split_feature.append(List[Int]())
                o.split_border.append(List[Float32]())
                o.split_is_cat.append(List[Bool]())
                o.leaf_values.append(List[Float32]())
        elif kind == String("train_mse"):
            o.train_mse = Float64(t[1])
        elif kind == String("baseline_mse"):
            o.baseline_mse = Float64(t[1])
        elif kind == String("borders"):
            var f = Int(t[1])
            var n = Int(t[2])
            for i in range(n):
                o.borders[f].append(Float32(Float64(t[3 + i])))
        elif kind == String("cat"):
            # `cat <flat_feature_index> <cardinality>`: the column is a
            # ONE-HOT categorical, `k` folds, bins `0..k-1`, split by
            # equality. Only `tools/catboost_cat_oracle.py` writes it.
            o.cat_cardinality[Int(t[1])] = Int(t[2])
        elif kind == String("split"):
            var ti = Int(t[1])
            o.split_feature[ti].append(Int(t[2]))
            o.split_border[ti].append(Float32(Float64(t[3])))
            o.split_is_cat[ti].append(False)
        elif kind == String("catsplit"):
            # `catsplit <tree> <flat_feature_index> <code>`. Written
            # INTERLEAVED with `split` in depth order, so appending in file
            # order reconstructs the tree's levels without the reader
            # needing to know which kind came next.
            var ct = Int(t[1])
            o.split_feature[ct].append(Int(t[2]))
            o.split_border[ct].append(Float32(Float64(t[3])))
            o.split_is_cat[ct].append(True)
        elif kind == String("leaves"):
            var lt = Int(t[1])
            for i in range(2, len(t)):
                o.leaf_values[lt].append(Float32(Float64(t[i])))
        elif kind == String("y"):
            for i in range(1, len(t)):
                o.y.append(Float32(Float64(t[i])))
        elif kind == String("x"):
            for f in range(o.feats):
                o.x[f].append(Float32(Float64(t[1 + f])))
            row_cursor += 1

    if o.rows == 0:
        raise Error(
            "bench/oracle.txt has no config line. Regenerate it with"
            " `pixi run -e bench python tools/catboost_oracle.py >"
            " bench/oracle.json`"
        )
    if row_cursor != o.rows:
        raise Error(
            String("bench/oracle.txt declared ")
            + String(o.rows)
            + " rows and carries "
            + String(row_cursor)
        )
    return o^


def check_border_parity(path: String = String("bench/oracle.txt")) raises:
    """OUR GreedyLogSum against the borders CatBoost actually chose.

    Their `border_count` is a BUDGET and not a count. A feature with few
    distinct values gets fewer, so the comparison is on the list and never on
    its length alone.
    """
    var o = load_oracle(path)
    print(
        "  oracle:", o.rows, "rows x", o.feats, "features, border budget",
        o.border_count, ", CatBoost train mse", o.train_mse,
        "against baseline", o.baseline_mse,
    )

    var features_wrong = 0
    var borders_wrong = 0
    var count_wrong = 0
    var worst = Float64(0.0)
    var worst_feature = -1
    var numeric = 0

    for f in range(o.feats):
        # A ONE-HOT CATEGORICAL COLUMN HAS NO BORDERS AND IS NOT
        # BINARIZED. Its bins ARE its dense category codes, so there is
        # nothing here for `best_split` to reproduce; the thing that has to
        # agree for it is the CARDINALITY, which is checked where it is
        # used (`check_tree_structure` builds `folds[f]` from it and a
        # wrong one changes every split index on the column).
        if o.cat_cardinality[f] != 0:
            continue
        numeric += 1
        var ours = best_split(o.x[f].copy(), o.border_count)
        ref theirs = o.borders[f]

        if len(ours) != len(theirs):
            count_wrong += 1
            features_wrong += 1
            print(
                "    feature", f, "border COUNT differs: ours", len(ours),
                " CatBoost", len(theirs),
            )
            continue

        var bad_here = 0
        for i in range(len(ours)):
            var d = abs(Float64(ours[i]) - Float64(theirs[i]))
            if d > worst:
                worst = d
                worst_feature = f
            # Float32 borders are the midpoint of two adjacent values, so the
            # only slack allowed is representation, not policy.
            if d > 1e-6:
                bad_here += 1
                borders_wrong += 1
                if borders_wrong <= 6:
                    print(
                        "    feature", f, "border", i, ": ours", ours[i],
                        " CatBoost", theirs[i],
                    )
        if bad_here != 0:
            features_wrong += 1

    print(
        "  features whose borders differ:", features_wrong, "of", numeric,
        "numeric (", o.feats - numeric, "one-hot categorical, no borders)",
        " (count mismatches", count_wrong, ", value mismatches", borders_wrong,
        ")",
    )
    print("  worst border delta", worst, "on feature", worst_feature)

    if features_wrong != 0:
        raise Error(
            "OUR BINARIZATION DOES NOT AGREE WITH CATBOOST'S. Every split"
            " index in this port is an index into a compressed index built"
            " from these borders, so until this passes no split and no leaf"
            " value can be compared with theirs, and our accuracy has never"
            " been comparable to theirs either"
        )
    print(
        "  border parity: our GreedyLogSum reproduces CatBoost's borders on"
        " every feature"
    )


def check_oracle_is_not_degenerate(path: String = String("bench/oracle.txt")) raises:
    """The oracle has to be worth comparing against.

    A fixture CatBoost cannot learn, or one where every feature carries the
    same signal, would pass a border comparison and tell us nothing. This is
    the same rule the histogram checks learned the hard way: a fixture whose
    expected value is the same everywhere verifies the total and nothing
    about placement.
    """
    var o = load_oracle(path)

    if not (o.train_mse < 0.25 * o.baseline_mse):
        raise Error(
            "CatBoost did not learn this fixture, so comparing to it proves"
            " nothing. Fix tools/catboost_oracle.py before reading any"
            " result below it"
        )

    # Their trees must not all be the same split, or split order carries no
    # information and a port that ignored order would pass.
    var distinct = 0
    var seen = List[Int]()
    for t in range(o.trees):
        for d in range(len(o.split_feature[t])):
            var fid = o.split_feature[t][d]
            var found = False
            for k in range(len(seen)):
                if seen[k] == fid:
                    found = True
            if not found:
                seen.append(fid)
                distinct += 1
    print(
        "  CatBoost split on", distinct, "distinct features across", o.trees,
        "trees, and reached", o.train_mse, "from", o.baseline_mse,
    )
    if distinct < 2:
        raise Error("the oracle's trees are degenerate, every split is one feature")
    print("  the oracle is a fixture worth comparing against")


def print_catboost_structure(path: String = String("bench/oracle.txt")) raises:
    """CatBoost's own trees, so a human can read them beside ours.

    Not an assertion. The assertion that matches structure needs our port fed
    THEIR borders, which is the next step and is not this one; printing them
    is what makes the gap visible instead of theoretical.
    """
    var o = load_oracle(path)
    for t in range(min(3, o.trees)):
        var line = String("    tree ") + String(t) + ": "
        for d in range(len(o.split_feature[t])):
            line += String("(")
            line += String(o.split_feature[t][d])
            # `== code` for a one-hot level, `@ border` for an ordered
            # one. Printing them the same way would hide the only
            # difference this fixture exists to check.
            if o.split_is_cat[t][d]:
                line += String(" == ")
                line += String(Int(Float64(o.split_border[t][d]) + 0.5))
            else:
                line += String(" @ ")
                line += String(o.split_border[t][d])
            line += String(") ")
        print(line)


def onebyte_width_for(folds: Int) raises -> Int:
    """CatBoost's bound, on the host: which one-byte accumulator claims a
    block whose widest feature has `folds` folds.

        constexpr ui32 upperBound = (1 << BITS);
        constexpr ui32 lowerBound = BITS > 5 ? upperBound / 2 : 15;
        if (maxBinCount <= lowerBound || maxBinCount > upperBound) return;

    `pointwise_hist2_one_byte_templ.cuh:179-183`, with `maxBinCount` the
    MAXIMUM `TCFeature::Folds` over the block's four features
    (`GetMaxBinCount`, `split_properties_helpers.cuh:25-45`). Returns 0 for a
    fold count no one-byte accumulator claims, which for a feature that
    reached this policy at all can only mean a fold count above 256.
    """
    if folds > 15 and folds <= 32:
        return 5
    if folds > 32 and folds <= 64:
        return 6
    if folds > 64 and folds <= 128:
        return 7
    if folds > 128 and folds <= 256:
        return 8
    return 0


def print_policy_reach(path: String = String("bench/oracle.txt")) raises:
    """WHICH KERNELS THIS FIXTURE REACHES, printed beside its result.

    PORTING_RULES 8: "the benchmark prints which path it took, beside the
    timing. A harness that cannot name the kernel it ran can publish a number
    about a different one." The same applies to a differential, and
    `archive/reference/PORTING.md` 108 is what it costs when it does not -- three fixtures
    covering three of the four one-byte accumulators, with nobody able to say
    from the output which was which.

    This is a REPORT of what the layout implies, from the product's own
    `build_layout` and `blocks_for`. The OBSERVATION that the named
    accumulator is the one that actually writes the histogram is
    `checks/onebyte_reach_check.mojo`, which runs the four widths one at a
    time and requires exactly one of them to come back non-empty. Do not read
    this line as the observation; it is the label.
    """
    var o = load_oracle(path)
    var folds = List[Int]()
    var one_hot = List[Bool]()
    for f in range(o.feats):
        if o.cat_cardinality[f] != 0:
            folds.append(o.cat_cardinality[f])
            one_hot.append(True)
        else:
            folds.append(len(o.borders[f]))
            one_hot.append(False)
    var lay = build_layout(folds, one_hot)
    var blocks = blocks_for(lay, o.rows)

    for b in range(len(blocks)):
        ref blk = blocks[b]
        var line = String("  policy ") + policy_name(blk.policy) + ": "
        line += String(blk.count()) + " features"
        if blk.policy != POLICY_ONE_BYTE:
            var lo = 1 << 30
            var hi = 0
            for i in range(blk.count()):
                var fc = Int(blk.folds[i])
                if fc < lo:
                    lo = fc
                if fc > hi:
                    hi = fc
            line += ", folds " + String(lo) + ".." + String(hi)
            print(line)
            continue

        # ONE BYTE: four accumulators, and a block of FOUR CONSECUTIVE
        # features is claimed by its widest (`feature += (blockIdx.x / M) *
        # 4` at `pointwise_hist2_one_byte_templ.cuh:169`). Reported per
        # group, because a fixture whose features have mixed widths reaches
        # more than one accumulator and saying "the widest feature" would be
        # a different claim.
        var reach = List[Int]()
        for _ in range(4):
            reach.append(0)
        var unclaimed = 0
        var g = 0
        while g < blk.count():
            var widest = 0
            var i = g
            while i < g + 4 and i < blk.count():
                if Int(blk.folds[i]) > widest:
                    widest = Int(blk.folds[i])
                i += 1
            var w = onebyte_width_for(widest)
            if w == 0:
                unclaimed += 1
            else:
                reach[w - 5] += 1
            g += 4
        line += ", 4-feature groups by accumulator:"
        for w in range(4):
            line += " " + String(5 + w) + "bit=" + String(reach[w])
        if unclaimed != 0:
            line += "  UNCLAIMED=" + String(unclaimed)
        print(line)
        if unclaimed != 0:
            raise Error(
                String(unclaimed)
                + " one-byte feature group(s) in "
                + path
                + " fall outside every accumulator's range, so their"
                " histogram is never written and every split below is a"
                " decision taken on zeros"
            )


struct TreeDiffResult(Copyable, Movable):
    """What one (fixture, searcher) cell of the differential came back with.

    `check_tree_structure` throws away everything but the pass/fail, which is
    right for a gate and useless for a SWEEP: a matrix needs the numbers of
    the cells that disagreed, not an exception from the first one. So the
    body below returns this and the gate is a thin wrapper that raises on it.
    """

    var matched: Int
    var compared: Int
    var cat_matched: Int
    var cat_compared: Int
    #: `tree * 100 + depth` of the FIRST disagreement, or -1. Encoded the way
    #: the printer already encoded it rather than as two fields, so the two
    #: cannot drift apart.
    var first_divergence: Int
    var our_mse: Float64
    var their_mse: Float64
    var baseline_mse: Float64

    def __init__(out self):
        self.matched = 0
        self.compared = 0
        self.cat_matched = 0
        self.cat_compared = 0
        self.first_divergence = -1
        self.our_mse = 0.0
        self.their_mse = 0.0
        self.baseline_mse = 0.0


def check_tree_structure(
    path: String = String("bench/oracle.txt"),
    use_pointwise_searcher: Bool = False,
) raises:
    """THE GATE. `tree_structure_diff` with `strict`, which raises on any
    disagreement. Every caller that wants the numbers instead calls that."""
    _ = tree_structure_diff(path, use_pointwise_searcher, True, True)


def tree_structure_diff(
    path: String,
    use_pointwise_searcher: Bool,
    strict: Bool,
    verbose: Bool,
    sabotage_first_split: Bool = False,
) raises -> TreeDiffResult:
    """OUR TREES against CATBOOST'S TREES, on the same data and the same grid.

    This is the comparison border parity exists to make possible. Both sides
    see the same 4096 x 16 matrix, the same target, the same quantization,
    and the same depth, learning rate and L2. An oblivious tree IS its list
    of splits, so if the port is faithful the lists match level for level.

    HOW A SPLIT IS COMPARED. CatBoost reports a split as
    (float_feature_index, border VALUE); ours is (feature, bin index) into a
    compressed index. They are the same statement in two encodings, and the
    bridge is the border list itself: their border value sits at some
    position in `borders[f]`, and that position is the bin our `resolve_split`
    should name. So the comparison converts theirs into our encoding rather
    than the other way round, because ours is the one under test.

    THE RESULT: 48 OF 48 SPLITS MATCH, AND THE LOSS AGREES TO NINE DIGITS.

        ours      0.7272208329569003
        CatBoost  0.7272208331492337

    Every split, every feature, every bin, across twelve trees at depth 4.
    The residual on the loss is float32 against their double and nothing
    else. This is the strongest statement this repository can make about the
    port and it is the only one not made against a tally written here.

    THE FALSE ALARM THAT CAME FIRST, KEPT BECAUSE IT COST HOURS.

    On its first run this check reported that we diverged from CatBoost at
    DEPTH 0 under their default Cosine while matching their L2 exactly for
    three levels, and the conclusion drawn was that our Cosine calcer was
    silently computing L2. That conclusion was WRONG and the defect was in
    the ORACLE, not in the port.

    `tools/catboost_oracle.py` had not set `random_strength`, and CatBoost's
    default is 1.0, not 0. That adds Gaussian noise to every candidate score
    (`score_calcers.cuh:162-166`, with `ScoreStdDev = RandomStrength *
    ComputeTargetStdDev` at `greedy_search_helper.cpp:385`). Our port refuses
    `random_strength` by name, so it computes the noiseless score.

    WHY IT LOOKED LIKE A COSINE BUG SPECIFICALLY, which is the part worth
    remembering. The noise is ABSOLUTE, and the two calcers work at wildly
    different scales. On this fixture Cosine's candidates run about 146 with
    gaps of order 1, while L2's run about 21000 with gaps of order 300. The
    same perturbation reorders Cosine's ranking and leaves L2's untouched. So
    a noiseless port matches their L2 and not their Cosine, and the symptom
    reads exactly like computing the wrong calcer.

    Two things were also claimed on that reading and are RETRACTED: that a
    second divergence lived at depth 3, and that it implicated the 1024-row
    `GatherInplaceLeqSize` threshold. Both were the same artifact.

    THE CATEGORICAL FIXTURE SPLITS THE TWO SEARCHERS, 2026-08-21.

        bench/oracle_cat.txt   greedy-subsets  48/48  (21/21 one-hot)
                               pointwise        1/42  ( 0/18 one-hot)

    The greedy arm's loss lands on CatBoost's to eight digits
    (0.14671102 against 0.14671103); the pointwise arm reaches 2.30 and
    stops trees early, because a repeated split is their stopping rule and
    it keeps choosing the same wrong bins. DEVIATION 114 has the cause and
    the raise below names it. Nothing was tuned to produce this: the
    fixture's pins were fixed before it was ever run.

    THE SABOTAGES THAT ESTABLISHED THE CATEGORICAL COMPARISON CAN FAIL,
    one per mechanism, all run 2026-08-21 against the greedy arm (the one
    that is green, since sabotaging a red path proves nothing):

      * a CATEGORY CODE moved in the fixture (`catsplit 0 8 1` -> `... 2`)
        -> 47 of 48, 20 of 21 one-hot, first divergence tree 0 depth 0.
      * the ONE-HOT FLAGS withheld from `fit`, everything else identical
        -> 1 of 48, 0 of 21 one-hot. **This is also the sabotage that
        proves the split-TYPE comparison earns its place**: with the flags
        off the searcher produced `(10>6)` at tree 2 depth 2 where
        CatBoost has `(10==6)` -- the SAME feature and the SAME bin, a
        different partition of the rows. A comparison on (feature, bin)
        alone would have scored that a match.
      * a NUMERIC split's border moved one border up in the SAME mixed
        fixture -> 47 of 48, 21 of 21 one-hot, first divergence tree 0
        depth 1, which is the numeric half still being read in a fixture
        whose `split` and `catsplit` records interleave.

    THE RULE THIS EARNS. An oracle has a configuration, and every default it
    leaves standing is a claim that we implement that default. Anything the
    port refuses by name must be turned OFF in the oracle and said out loud,
    which is why `bootstrap_type`, `model_shrink_rate`, `boost_from_average`
    and now `random_strength` are all pinned in that file with a reason.

    WHAT A MISMATCH MEANS, and it is worth stating before reading a result.
    Not necessarily a defect. The tree is a greedy argmax over scores, so two
    faithful implementations can diverge at a genuine tie and never reconverge,
    and a single differing split at depth 0 makes every later level
    incomparable. That is why the report prints both sequences in full and
    counts the FIRST divergence rather than a total: the depth at which the
    two first disagree is the number that means something.
    """
    var o = load_oracle(path)

    var ctx = DeviceContext()
    var n_rows = o.rows
    var n_features = o.feats

    # THEIR grid, not ours. `check_border_parity` establishes that our
    # `best_split` reproduces it, so using theirs removes binarization as a
    # variable and leaves only the tree search under test.
    #
    # A ONE-HOT CATEGORICAL COLUMN gets `k` folds and no borders, which is
    # `GetBucketCount`'s `OnLearnOnly` unique count (`split.cpp:110-114`)
    # and gives it `k` candidate splits where an ordered feature of the
    # same fold count has `k - 1` (`score_calcers.cpp:58-61`). The flags
    # are what make the searcher emit `TakeBin` instead of `TakeGreater`.
    var folds = List[Int]()
    var one_hot = List[Bool]()
    var n_cat = 0
    for f in range(n_features):
        if o.cat_cardinality[f] != 0:
            folds.append(o.cat_cardinality[f])
            one_hot.append(True)
            n_cat += 1
        else:
            folds.append(len(o.borders[f]))
            one_hot.append(False)

    # All-False flags produce a bit-identical layout to the empty list --
    # `is_one_hot` is read per feature -- so the three numeric fixtures go
    # down the path they always did.
    var lay = build_layout(folds, one_hot)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](n_rows * lay.columns)
    var z = ctx.enqueue_create_host_buffer[DType.uint32](n_rows * lay.columns)
    for i in range(n_rows * lay.columns):
        z.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())
    ctx.synchronize()

    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n_rows)
    var bins = ctx.enqueue_create_buffer[DType.uint8](n_rows)
    for f in range(n_features):
        ref cf = lay.features[f]
        for r in range(n_rows):
            # A categorical column's compressed-index value IS its dense
            # code; there is no binarization step for it on either side.
            var bin_value: Int
            if o.cat_cardinality[f] != 0:
                bin_value = Int(Float64(o.x[f][r]) + 0.5)
                if bin_value < 0 or bin_value >= o.cat_cardinality[f]:
                    raise Error(
                        String("categorical column ")
                        + String(f)
                        + " carries code "
                        + String(bin_value)
                        + " outside 0.."
                        + String(o.cat_cardinality[f] - 1)
                    )
            else:
                bin_value = binarize(o.x[f][r], o.borders[f])
            hb.unsafe_ptr().unsafe_store(r, UInt8(bin_value))
        ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
            bins.unsafe_ptr(), Int32(n_rows), cindex.unsafe_ptr(),
            grid_dim=(n_rows + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=(WRITE_BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()

    var targets = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    for r in range(n_rows):
        ht.unsafe_ptr().unsafe_store(r, o.y[r])
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    var model = TAdditiveModel()
    var losses = fit(
        model, ctx, n_rows, folds, o.depth, cindex, targets, weights, False,
        o.trees, Float32(0.3), Float32(3.0), True,
        one_hot=one_hot,
        use_pointwise_searcher=use_pointwise_searcher,
    )

    if verbose:
        print(
            "    our final mse", losses[len(losses) - 1],
            " CatBoost", o.train_mse, " mean baseline", o.baseline_mse,
        )

    # Their border value -> the bin index our side would name for it.
    var first_divergence = -1
    var matched = 0
    var compared = 0
    var cat_compared = 0
    var cat_matched = 0
    for t in range(min(o.trees, model.size())):
        ref ours = model.weak_models[t].structure.splits
        var line_them = String("      catboost ")
        var line_us = String("      ours     ")
        for d in range(len(o.split_feature[t])):
            var f = o.split_feature[t][d]
            var want_bin = -1
            var want_type = BIN_SPLIT_TAKE_GREATER
            var is_cat = o.split_is_cat[t][d]
            if is_cat:
                # A one-hot split IS its category code, and the code is
                # already in our encoding -- the oracle resolved
                # CatBoost's hash back to it and proved the resolution by
                # replaying the whole model against their own predictions.
                want_bin = Int(Float64(o.split_border[t][d]) + 0.5)
                want_type = BIN_SPLIT_TAKE_BIN
                if want_bin < 0 or want_bin >= o.cat_cardinality[f]:
                    raise Error(
                        String("oracle names one-hot code ")
                        + String(want_bin)
                        + " on column "
                        + String(f)
                        + " whose declared cardinality is "
                        + String(o.cat_cardinality[f])
                    )
            else:
                for i in range(len(o.borders[f])):
                    if abs(Float64(o.borders[f][i]) - Float64(o.split_border[t][d])) <= 1e-9:
                        want_bin = i
            # THE SABOTAGE, and it is a knob rather than a temporary edit
            # because it has to be runnable at every cell of the sweep.
            # Moving the EXPECTED bin of tree 0 depth 0 by one must cost
            # exactly one match and put the first divergence at (0, 0). A
            # comparison that is reached but not positioned -- one that
            # counts totals, or that compares the tree as a set -- passes
            # the green run and passes this one too.
            if sabotage_first_split and t == 0 and d == 0 and want_bin >= 0:
                want_bin += 1
            line_them += (
                String("(")
                + String(f)
                + (String("==") if is_cat else String(">"))
                + String(want_bin)
                + String(") ")
            )
            if d < len(ours):
                var gf = Int(ours[d].feature_id)
                var gb = Int(ours[d].bin_idx)
                var gt = Int(ours[d].split_type)
                line_us += (
                    String("(")
                    + String(gf)
                    + (
                        String("==")
                        if gt == BIN_SPLIT_TAKE_BIN
                        else String(">")
                    )
                    + String(gb)
                    + String(") ")
                )
                compared += 1
                if is_cat:
                    cat_compared += 1
                # THE SPLIT TYPE IS PART OF THE SPLIT. Comparing only
                # (feature, bin) would let an ORDERED `> code` stand in
                # for an EQUALITY `== code` on the same column and bin,
                # which is a different partition of the rows and is the
                # one thing this fixture exists to catch.
                if gf == f and gb == want_bin and gt == want_type:
                    matched += 1
                    if is_cat:
                        cat_matched += 1
                elif first_divergence < 0:
                    first_divergence = t * 100 + d
        if t < 3 and verbose:
            print("    tree", t)
            print(line_them)
            print(line_us)

    # REACH, per branch. A fixture that declares categorical columns and
    # never gets a one-hot split compared is a numeric fixture wearing a
    # categorical name, and it would report a green tick for coverage it
    # does not have.
    if n_cat != 0 and cat_compared == 0:
        raise Error(
            String("this fixture declares ")
            + String(n_cat)
            + " one-hot categorical columns and not one one-hot split was"
            " compared, so nothing about the categorical path was checked"
        )

    var arm = String("pointwise") if use_pointwise_searcher else String(
        "greedy-subsets"
    )
    var result = TreeDiffResult()
    result.matched = matched
    result.compared = compared
    result.cat_matched = cat_matched
    result.cat_compared = cat_compared
    result.first_divergence = first_divergence
    result.our_mse = Float64(losses[len(losses) - 1])
    result.their_mse = o.train_mse
    result.baseline_mse = o.baseline_mse

    if verbose:
        print(
            "    splits matching CatBoost exactly:", matched, "of", compared,
            "  (", cat_matched, "of", cat_compared, "one-hot )",
            "  [", arm, "searcher ]",
        )
    if not strict:
        # THE SWEEP PATH. A cell that disagrees is a FINDING and the matrix
        # needs the rest of its cells, so nothing is raised here and nothing
        # is tuned to make the number go green.
        return result^
    if first_divergence >= 0:
        print(
            "    first divergence: tree", first_divergence // 100,
            "depth", first_divergence % 100,
        )
        var where = (
            String(" splits match, first divergence at tree ")
            + String(first_divergence // 100)
            + " depth "
            + String(first_divergence % 100)
        )
        if n_cat != 0 and use_pointwise_searcher and cat_matched == 0:
            # THE LOCATED DEFECT. Not a mystery and not a fixture problem:
            # see DEVIATION 114. `TScoreHelper.__init__`
            # (`gbdt/methods/pointwise_scores_calcer.mojo`, the
            # `oh.append(UInt8(0))` in the table-building loop) fills the
            # scorer's per-feature one-hot flags with a CONSTANT ZERO
            # instead of reading `layout.features[...].one_hot_feature`,
            # which is read on the line above it for the offset. That
            # array's only consumer is `scan_pointwise_histograms_kernel`'s
            # skip (`split_properties_helpers.mojo:350`, theirs at
            # `split_properties_helpers.cuh:126`), so every one-hot
            # feature gets PREFIX-SUMMED and its equality candidates score
            # as thresholds.
            raise Error(
                String("the pointwise searcher does not reproduce")
                + " CatBoost's one-hot splits: "
                + String(cat_matched)
                + " of "
                + String(cat_compared)
                + " one-hot and "
                + String(matched)
                + " of "
                + String(compared)
                + where
                + ". DEVIATION 114 locates this in TScoreHelper's"
                " constructor, which builds the scorer's one-hot flag"
                " array as a constant zero; the greedy-subsets searcher"
                " reads the same flag off the layout and matches CatBoost"
                " on this fixture. This is the FIRST check to hand a"
                " one-hot feature to the pointwise SCORER -- every"
                " existing gate on that skip passes the flag array in by"
                " hand and so never reached the constant"
            )
        raise Error(
            String("the ")
            + arm
            + String(" searcher no longer reproduces CatBoost's trees: ")
            + String(matched)
            + " of "
            + String(compared)
            + where
            + ". This matched 48 of 48 on 2026-08-19, so a mismatch here is"
            " a REGRESSION and not a known gap. Before suspecting the port,"
            " check that bench/oracle.txt was generated with every CatBoost"
            " feature this port refuses turned off, because that mistake has"
            " already produced one convincing false alarm"
        )
    else:
        print(
            "    every split matches CatBoost, feature and bin  [", arm,
            "]",
        )
    return result^
