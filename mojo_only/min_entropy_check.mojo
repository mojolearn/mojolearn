"""`best_split_min_entropy` against CatBoost's OWN MinEntropy borders.

    pixi run check-minentropy

Reads `bench/minentropy_oracle.txt`, written by `tools/minentropy_oracle.py`
from CatBoost itself. This is one of the few checks in this repository that
compares against CatBoost rather than against a host tally we wrote, which
matters more here than usual: our MinEntropy is a transliteration of their
exact dynamic program, and a tally would just be the same reading twice.

WHAT IT IS ACTUALLY GATING, beyond "the optimum is right":

* **The two OPPOSITE tie-breaks.** Their main loop takes `<=` so the last
  index wins; the last match takes `<` so the first index wins
  (`binarization.cpp:245` and `:637`). The fixture's columns are FeatureFreq
  values `count / (n + 1)` over consecutive integer category sizes, which
  mints exactly-equal weights and equal-sum subranges by construction, so a
  reversed tie-break has somewhere to show.
* **The mode deviation.** They run `E_RLM2`, we run `E_Base`. Same dynamic
  program, different pruning, so the optimum is identical by construction
  and the tie realisation is not guaranteed to be. That is the only thing
  standing between us and their borders, and it is what this measures rather
  than argues about.
* **Six budgets, not one.** 15 = 1+2+4+8 lands on a complete tier and is
  order-invariant, which is exactly how the `best_split` heap-order bug
  survived for months. These cut at 7, 13, 23, 31, 47 and 100.
"""
from gbdt.grid_creator.binarization import best_split_min_entropy

comptime ORACLE = "bench/minentropy_oracle.txt"


def check_min_entropy() raises:
    print("MinEntropy borders vs CatBoost (bench/minentropy_oracle.txt):")
    var f = open(ORACLE, "r")
    var text = f.read()
    f.close()

    var lines = List[String]()
    for line in text.splitlines():
        var s = String(String(line).strip())
        if s.byte_length() > 0:
            lines.append(s^)

    var pos = 0
    var head = lines[pos].split(" ")
    if String(head[0]) != "columns":
        raise Error("malformed oracle header")
    var n_columns = Int(String(head[1]))
    pos += 1

    var columns = List[List[Float32]]()
    for _ in range(n_columns):
        var ch = lines[pos].split(" ")
        if String(ch[0]) != "column":
            raise Error("malformed column header")
        var n_values = Int(String(ch[2]))
        pos += 1
        var col = List[Float32]()
        for i in range(n_values):
            col.append(Float32(Float64(lines[pos + i])))
        pos += n_values
        columns.append(col^)

    var chead = lines[pos].split(" ")
    if String(chead[0]) != "cases":
        raise Error("malformed cases header")
    var n_cases = Int(String(chead[1]))
    pos += 1

    var failed = 0
    for _ in range(n_cases):
        var head2 = lines[pos].split(" ")
        if String(head2[0]) != "case":
            raise Error("malformed case header")
        var col_id = Int(String(head2[1]))
        var budget = Int(String(head2[2]))
        var n_borders = Int(String(head2[3]))
        pos += 1

        var expected = List[Float32]()
        for i in range(n_borders):
            expected.append(Float32(Float64(lines[pos + i])))
        pos += n_borders

        var got = best_split_min_entropy(columns[col_id].copy(), budget)
        if len(got) != len(expected):
            print("  col", col_id, "budget", budget, "FAIL: catboost",
                  len(expected), "borders, ours", len(got))
            failed += 1
            continue
        var wrong = 0
        for i in range(len(expected)):
            # EXACT. Both sides compute the same thing the same way: the
            # midpoint of two float32 uniques, in float32. There is no
            # rounding budget to spend, and a tolerance here would be
            # actively harmful -- these borders run from 1e-5 to 1e-2, so
            # the 1e-6 absolute band this check first shipped with was
            # worth 10% at the low end and would have passed a wrong
            # threshold as easily as a right one.
            if got[i] != expected[i]:
                wrong += 1
        if wrong == 0:
            print("  col", col_id, "budget", budget, ":", len(expected),
                  "borders exact")
        else:
            print("  col", col_id, "budget", budget, "FAIL:", wrong, "of",
                  len(expected), "borders differ")
            failed += 1

    if failed != 0:
        raise Error("MinEntropy borders disagree with CatBoost in "
                    + String(failed) + " of " + String(n_cases) + " cases")
    print("  all", n_cases, "budgets match CatBoost exactly")


def main() raises:
    check_min_entropy()
