"""OUR apply against CATBOOST'S OWN MODEL and CATBOOST'S OWN PREDICTIONS.

    pixi run check-catboost-apply

WHY THIS IS DIFFERENT FROM EVERY OTHER APPLY CHECK HERE. The rest compare
our device against a host tally we wrote ourselves, which is one reading of
CatBoost expressed twice: if we misread them, the tally encodes the
misreading and agrees with the kernel perfectly. This does not train
anything and does not compute an expected value. It loads THEIR trained
model out of `bench/oracle.txt`, runs it over THEIR rows through OUR apply,
and compares against the predictions THEY produced.

THE FIXTURE WAS ALREADY COMMITTED AND UNREAD. `tools/catboost_oracle.py`
has been dumping a `pred` record since it was written, and
`mojo_only/oracle_check.mojo`'s parser has no branch for it, so the line was
silently dropped on every run. Nothing here is new data; what was missing
was the reader.

WHAT IT ACTUALLY GATES, and it is four things at once:
  * our `binarize` against THEIR quantization, since we quantize their raw
    floats with their borders and any disagreement lands as a moved row,
  * our split predicate, `bucket > border` against their `TakeBin`,
  * our leaf INDEXING, since an oblivious tree's leaf is the bit-packed
    path and a reversed bit order is a permutation that conserves every
    leaf value,
  * the ensemble sum, including that the learning rate is already folded
    into their stored leaf values and must not be reapplied.

`scale_and_bias` IS CHECKED, NOT ASSUMED. Their evaluator computes
`scale * sum(leaves) + bias`, and this comparison is only valid while that
is the identity. The oracle now dumps it and this refuses to run if it is
ever anything else, because "it is probably 1 and 0" is a guess and a
tolerance would absorb the difference silently.

PROVEN ABLE TO FAIL, three ways, each hitting a different one of the four
things above: a split bin index off by one moves 4049 of 4096 rows, a
reversed border lookup moves 4091, and reversing the leaf order within each
tree moves all 4096. The last one is worth keeping in mind, because a
reversed leaf order conserves every leaf value exactly and so is invisible
to any check that compares a sum.

TOLERANCE, and why it is not zero. Their `leaf_values` are DOUBLE in the
model JSON (`model.fbs`: `LeafValues:[double]`) and ours are Float32, so
the sum is taken in a narrower type than theirs. The comparison is
therefore at float32 resolution scaled by the tree count, not bitwise, and
the check prints the worst observed gap so a drift shows up as a number
rather than as a pass.
"""
from std.math import sqrt

from max.gpu.host import DeviceContext

from gbdt.models.oblivious_model import (
    TAdditiveModel,
    TBinarySplit,
    TObliviousTreeModel,
    TObliviousTreeStructure,
)
from gbdt.train import TrainedModel, predict_floats

comptime ORACLE = "bench/oracle.txt"


def _split(line: String) -> List[String]:
    var out = List[String]()
    for tok in line.split(" "):
        var s = String(String(tok).strip())
        if s.byte_length() > 0:
            out.append(s^)
    return out^


def check_catboost_apply(ctx: DeviceContext) raises:
    print("our apply vs CatBoost's own model and predictions:")
    var f = open(ORACLE, "r")
    var text = f.read()
    f.close()

    var rows = 0
    var feats = 0
    var n_trees = 0
    var scale = Float64(1.0)
    var bias = Float64(0.0)
    var saw_scale = False

    var borders = List[List[Float32]]()
    var split_feature = List[List[Int]]()
    var split_border = List[List[Float32]]()
    var leaf_values = List[List[Float32]]()
    var pred = List[Float32]()
    var x_colmajor = List[Float32]()

    # x arrives row-major, one record per row; the apply wants column-major
    var x_rows = List[List[Float32]]()

    for line in text.splitlines():
        var s = String(String(line).strip())
        if s.byte_length() == 0 or s.startswith(String("#")):
            continue
        var t = _split(s)
        var kind = t[0]
        if kind == String("config"):
            rows = Int(t[1])
            feats = Int(t[2])
            n_trees = Int(t[4])
            for _ in range(feats):
                borders.append(List[Float32]())
            for _ in range(n_trees):
                split_feature.append(List[Int]())
                split_border.append(List[Float32]())
                leaf_values.append(List[Float32]())
        elif kind == String("scale_and_bias"):
            scale = Float64(t[1])
            bias = Float64(t[2])
            saw_scale = True
        elif kind == String("borders"):
            var fi = Int(t[1])
            for i in range(3, len(t)):
                borders[fi].append(Float32(Float64(t[i])))
        elif kind == String("split"):
            var ti = Int(t[1])
            split_feature[ti].append(Int(t[2]))
            split_border[ti].append(Float32(Float64(t[3])))
        elif kind == String("leaves"):
            var ti = Int(t[1])
            for i in range(2, len(t)):
                leaf_values[ti].append(Float32(Float64(t[i])))
        elif kind == String("pred"):
            for i in range(1, len(t)):
                pred.append(Float32(Float64(t[i])))
        elif kind == String("x"):
            var r = List[Float32]()
            for i in range(1, len(t)):
                r.append(Float32(Float64(t[i])))
            x_rows.append(r^)

    if not saw_scale:
        raise Error(
            "the fixture carries no scale_and_bias record. Regenerate it with"
            " tools/catboost_oracle.py: this comparison is only valid while"
            " their evaluator's scale*sum+bias is the identity, and assuming"
            " that is exactly the kind of guess a tolerance hides"
        )
    if scale != 1.0 or bias != 0.0:
        raise Error(
            "CatBoost's scale_and_bias is " + String(scale) + " / "
            + String(bias) + ", not the identity, so their predictions are"
            " not the bare ensemble sum this compares against"
        )
    if len(pred) != rows:
        raise Error("expected " + String(rows) + " predictions, fixture has "
                    + String(len(pred)))
    if len(x_rows) != rows:
        raise Error("expected " + String(rows) + " rows, fixture has "
                    + String(len(x_rows)))

    for fdx in range(feats):
        for r in range(rows):
            x_colmajor.append(x_rows[r][fdx])

    # THEIR splits are (feature, border VALUE). Ours are (feature, bin
    # index), so each value is looked up in that feature's border list. The
    # match must be EXACT: both sides are the same float32 written at full
    # precision by the same script, so a near-miss means the border lists
    # have diverged and a nearest-match would paper over exactly that.
    var model = TAdditiveModel()
    var unmatched = 0
    for t in range(n_trees):
        var st = TObliviousTreeStructure()
        for s in range(len(split_feature[t])):
            var fi = split_feature[t][s]
            var bv = split_border[t][s]
            var bin_idx = -1
            for b in range(len(borders[fi])):
                if borders[fi][b] == bv:
                    bin_idx = b
                    break
            if bin_idx < 0:
                unmatched += 1
                continue
            st.splits.append(TBinarySplit(Int32(fi), Int32(bin_idx)))
        var wm = TObliviousTreeModel(st^)
        wm.leaf_values = leaf_values[t].copy()
        model.add_weak_model(wm^)
    if unmatched != 0:
        raise Error(
            String(unmatched) + " of their split borders are not in their own"
            " border list for that feature -- the fixture is internally"
            " inconsistent, which is a generator bug and not an apply bug"
        )

    var fold_counts = List[Int]()
    var one_hot = List[Bool]()
    for fdx in range(feats):
        fold_counts.append(len(borders[fdx]))
        one_hot.append(False)

    var tm = TrainedModel(
        model^, fold_counts^, one_hot^, borders^, List[Float64](), 0
    )
    var ours = predict_floats(ctx, tm, x_colmajor, rows)
    if len(ours) != rows:
        raise Error("apply returned " + String(len(ours)) + " predictions")

    # float32 sums of double leaves over n_trees; the band scales with the
    # tree count for the same reason a fixed 1e-4 tripped at 8000 trees on
    # pure reorder noise
    var tol = Float32(2e-5) * Float32(sqrt(Float64(n_trees)))
    var worst = Float32(0.0)
    var wrong = 0
    var worst_row = -1
    for r in range(rows):
        var d = ours[r] - pred[r]
        if d < 0:
            d = -d
        if d > worst:
            worst = d
            worst_row = r
        if d > tol:
            wrong += 1

    print("    ", n_trees, "of their trees,", rows, "of their rows,", feats,
          "features; scale_and_bias checked identity")
    print("     worst |ours - CatBoost| =", worst, "at row", worst_row,
          " tolerance", tol)
    if wrong != 0:
        raise Error(
            String(wrong) + " of " + String(rows) + " rows disagree with"
            " CatBoost's own predictions beyond tolerance"
        )
    print("     all", rows, "rows agree with CatBoost's own predictions")


def main() raises:
    var ctx = DeviceContext()
    check_catboost_apply(ctx)
