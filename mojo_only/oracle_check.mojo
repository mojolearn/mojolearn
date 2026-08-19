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

from ported.grid_creator.binarization import best_split


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


def check_border_parity() raises:
    """OUR GreedyLogSum against the borders CatBoost actually chose.

    Their `border_count` is a BUDGET and not a count. A feature with few
    distinct values gets fewer, so the comparison is on the list and never on
    its length alone.
    """
    var o = load_oracle(String("bench/oracle.txt"))
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


def check_oracle_is_not_degenerate() raises:
    """The oracle has to be worth comparing against.

    A fixture CatBoost cannot learn, or one where every feature carries the
    same signal, would pass a border comparison and tell us nothing. This is
    the same rule the histogram checks learned the hard way: a fixture whose
    expected value is the same everywhere verifies the total and nothing
    about placement.
    """
    var o = load_oracle(String("bench/oracle.txt"))

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


def print_catboost_structure() raises:
    """CatBoost's own trees, so a human can read them beside ours.

    Not an assertion. The assertion that matches structure needs our port fed
    THEIR borders, which is the next step and is not this one; printing them
    is what makes the gap visible instead of theoretical.
    """
    var o = load_oracle(String("bench/oracle.txt"))
    for t in range(min(3, o.trees)):
        var line = String("    tree ") + String(t) + ": "
        for d in range(len(o.split_feature[t])):
            line += String("(")
            line += String(o.split_feature[t][d])
            line += String(" @ ")
            line += String(o.split_border[t][d])
            line += String(") ")
        print(line)
