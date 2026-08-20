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
"""

from max.gpu.host import DeviceContext

from ported.grid_creator.binarization import best_split, binarize
from ported.gpu_data.compressed_index_builder import build_layout
from ported.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from ported.methods.doc_parallel_boosting import TAdditiveModel, fit


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
            for _ in range(o.trees):
                o.split_feature.append(List[Int]())
                o.split_border.append(List[Float32]())
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
        elif kind == String("split"):
            var ti = Int(t[1])
            o.split_feature[ti].append(Int(t[2]))
            o.split_border[ti].append(Float32(Float64(t[3])))
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

    for f in range(o.feats):
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
        "  features whose borders differ:", features_wrong, "of", o.feats,
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
            line += String(" @ ")
            line += String(o.split_border[t][d])
            line += String(") ")
        print(line)


def check_tree_structure(path: String = String("bench/oracle.txt")) raises:
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
    var folds = List[Int]()
    for f in range(n_features):
        folds.append(len(o.borders[f]))

    var lay = build_layout(folds)
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
            hb.unsafe_ptr().unsafe_store(
                r, UInt8(binarize(o.x[f][r], o.borders[f]))
            )
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
    )

    print(
        "    our final mse", losses[len(losses) - 1],
        " CatBoost", o.train_mse, " mean baseline", o.baseline_mse,
    )

    # Their border value -> the bin index our side would name for it.
    var first_divergence = -1
    var matched = 0
    var compared = 0
    for t in range(min(o.trees, model.size())):
        ref ours = model.weak_models[t].structure.splits
        var line_them = String("      catboost ")
        var line_us = String("      ours     ")
        for d in range(len(o.split_feature[t])):
            var f = o.split_feature[t][d]
            var want_bin = -1
            for i in range(len(o.borders[f])):
                if abs(Float64(o.borders[f][i]) - Float64(o.split_border[t][d])) <= 1e-9:
                    want_bin = i
            line_them += String("(") + String(f) + String(",") + String(want_bin) + String(") ")
            if d < len(ours):
                var gf = Int(ours[d].feature_id)
                var gb = Int(ours[d].bin_idx)
                line_us += String("(") + String(gf) + String(",") + String(gb) + String(") ")
                compared += 1
                if gf == f and gb == want_bin:
                    matched += 1
                elif first_divergence < 0:
                    first_divergence = t * 100 + d
        if t < 3:
            print("    tree", t)
            print(line_them)
            print(line_us)

    print(
        "    splits matching CatBoost exactly:", matched, "of", compared,
    )
    if first_divergence >= 0:
        print(
            "    first divergence: tree", first_divergence // 100,
            "depth", first_divergence % 100,
        )
        raise Error(
            String("the port no longer reproduces CatBoost's trees: ")
            + String(matched)
            + " of "
            + String(compared)
            + " splits match, first divergence at tree "
            + String(first_divergence // 100)
            + " depth "
            + String(first_divergence % 100)
            + ". This matched 48 of 48 on 2026-08-19, so a mismatch here is"
            " a REGRESSION and not a known gap. Before suspecting the port,"
            " check that bench/oracle.txt was generated with every CatBoost"
            " feature this port refuses turned off, because that mistake has"
            " already produced one convincing false alarm"
        )
    else:
        print("    every split matches CatBoost, feature and bin")
