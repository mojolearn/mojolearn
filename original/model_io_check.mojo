# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Save a model, load it back, and require the SAME BITS out of it.

CatBoost never had to write this: their model serialization is flatbuffers
plus six exporters. Ours is `gbdt/models/model_text.mojo`, a format of our
own, so it needs a gate of our own.

WHAT IS CHECKED, AND THE FAILURE EACH PART EXISTS FOR
-----------------------------------------------------
1. THE FLOAT TOKENS, over 200,000 random bit patterns. `String(x)` and back
   loses one ULP on ~0.46% of float32 and ~0.11% of float64 values in this
   toolchain, and `String(Float32(1.4e-45))` is `"0.0"`. Any decimal-only
   format silently corrupts about one leaf value in two hundred, and a
   round-trip check written with a tolerance never sees it. This part
   MEASURES both halves: the token must be exact for every pattern, and the
   decimal-only count is printed beside it so the format's reason stays
   attached to the format.
2. A HAND-BUILT MODEL with mixed depths (1, 3, 4, 6), scattered leaf values
   INCLUDING the exact bit patterns measured to lose a bit in decimal, a
   constant feature with no borders, a one-hot feature, leaf weights on some
   trees and not others, and hostile losses. A trained model will not
   produce those corners; a fixture we build does, which is the same reason
   `RECON_CTRS.md` argues for constructed fixtures over real datasets.
3. A TRAINED MODEL WITH A ONE-HOT FEATURE: predict, save, load, predict.
   Bit-identical per row, not close, and the loaded struct equal field by
   field.
4. THE DEVICE EVALUATOR (`gbdt/models/cuda/evaluator.mojo`). A format that
   round-trips through the host apply and breaks the device path has not
   been tested. The evaluator's cross-block reduce ends in a float
   atomicAdd, so this part FIRST proves the arm is reproducible at this
   shape (`tree_grid == 1`, one contributor per output slot) by running it
   twice, and only then uses it as a comparator.
5. FIVE SABOTAGES, each expected to turn the gate red, because a check never
   seen to fail is not evidence. Four of them are edits to the TEXT the real
   writer produced, not to a second writer written for the test, so what is
   sabotaged is the shipped path.

NON-DEGENERACY IS ASSERTED, NOT ASSUMED. A tree that never splits passes
every conservation check trivially, and a fixture whose values are uniform
verifies the total and nothing about placement. Every model here is checked
for at least two distinct split features, at least two distinct depths where
depths can differ, and leaf values that are not all equal.

THE ONE-HOT FEATURE IS THREE CATEGORIES ON PURPOSE. See the note on
`check_trained_roundtrip` for the measured reason.
"""

from std.memory import bitcast

from max.gpu.host import DeviceContext

from gbdt.models.cuda.evaluator import (
    EVAL_TREE_SUB_BLOCK_WIDTH,
    launch_eval,
    launch_quantize,
    pack_model_for_evaluator,
    padded_results,
    quantized_buffer_u32s,
)
from gbdt.models.model_text import (
    f32_token,
    f64_token,
    load_model,
    load_model_text,
    model_text,
    parse_f32,
    parse_f64,
    save_model,
)
from gbdt.models.ctr_value_table import TCtrValueTable
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    BIN_SPLIT_TAKE_GREATER,
    TAdditiveModel,
    bin_split_type_name,
    TBinarySplit,
    TObliviousTreeModel,
    TObliviousTreeStructure,
)
from gbdt.data.quantization import NAN_TREATMENT_AS_IS
from gbdt.train import TrainedModel, predict_floats, train


comptime SCRATCH = "/tmp/mojolearn_model_io_check.txt"


def _rng(mut s: UInt64) -> UInt64:
    """xorshift64. Scattered, not consecutive: a fixture whose values march
    in order cannot distinguish a permutation from the identity."""
    s ^= s << 13
    s ^= s >> 7
    s ^= s << 17
    return s


# ---------------------------------------------------------------------------
# 1. The float tokens.
# ---------------------------------------------------------------------------


def check_float_tokens() raises:
    print("  float tokens, 200,000 random bit patterns per width:")
    var s = UInt64(0x243F6A8885A308D3)
    var n = 200000
    var tried32 = 0
    var tried64 = 0
    var tok_bad32 = 0
    var tok_bad64 = 0
    var dec_bad32 = 0
    var dec_bad64 = 0
    var worst_example = String("")

    for _ in range(n):
        var r = _rng(s)
        var b32 = UInt32(r & UInt64(0xFFFFFFFF))
        var f32 = bitcast[DType.float32](b32)
        # non-finite patterns carry no decimal this parser is asked to read
        if f32 == f32 and (f32 - f32) == Float32(0.0):
            tried32 += 1
            if bitcast[DType.uint32](parse_f32(f32_token(f32))) != b32:
                tok_bad32 += 1
            var dec_only = bitcast[DType.uint32](Float32(Float64(String(f32))))
            if dec_only != b32:
                dec_bad32 += 1
                if worst_example.byte_length() == 0:
                    worst_example = String(f32)

        var r2 = _rng(s)
        var f64 = bitcast[DType.float64](r2)
        if f64 == f64 and (f64 - f64) == Float64(0.0):
            tried64 += 1
            if bitcast[DType.uint64](parse_f64(f64_token(f64))) != r2:
                tok_bad64 += 1
            if bitcast[DType.uint64](Float64(String(f64))) != r2:
                dec_bad64 += 1

    print(
        "    float32: token wrong", tok_bad32, "of", tried32,
        "  DECIMAL ALONE wrong", dec_bad32,
    )
    print(
        "    float64: token wrong", tok_bad64, "of", tried64,
        "  DECIMAL ALONE wrong", dec_bad64,
    )
    print("    first float32 the decimal half loses:", worst_example)
    # the extremes a random sweep will not reach
    var corners = List[UInt32]()
    corners.append(UInt32(0))
    corners.append(UInt32(0x80000000))
    corners.append(UInt32(1))
    corners.append(UInt32(0x80000001))
    corners.append(UInt32(0x7F7FFFFF))
    corners.append(UInt32(0x00800000))
    for i in range(len(corners)):
        var v = bitcast[DType.float32](corners[i])
        if bitcast[DType.uint32](parse_f32(f32_token(v))) != corners[i]:
            raise Error(
                "the token lost a corner value: bits "
                + String(corners[i])
            )
    print("    corner values (zero, -0, both denormal minima, max, min"
          " normal): exact")

    if tok_bad32 != 0 or tok_bad64 != 0:
        raise Error(
            "THE MODEL FORMAT DOES NOT ROUND-TRIP FLOATS EXACTLY. Every leaf"
            " value and every border goes through these tokens"
        )
    if dec_bad32 == 0 and dec_bad64 == 0:
        raise Error(
            "the decimal half round-tripped everything, so this toolchain's"
            " formatter changed and the reason the bits are written down is"
            " no longer measured. Re-take the measurement before deleting"
            " anything"
        )


# ---------------------------------------------------------------------------
# Field-by-field comparison. Per element, never an aggregate.
# ---------------------------------------------------------------------------


def _same_f32(a: Float32, b: Float32) -> Bool:
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


def _same_f64(a: Float64, b: Float64) -> Bool:
    return bitcast[DType.uint64](a) == bitcast[DType.uint64](b)


def compare_models(a: TrainedModel, b: TrainedModel, label: String) raises:
    """Every field, every element, bitwise. A total would hide a
    permutation."""
    if len(a.fold_counts) != len(b.fold_counts):
        raise Error(label + ": fold_counts length "
                    + String(len(a.fold_counts)) + " vs "
                    + String(len(b.fold_counts)))
    for f in range(len(a.fold_counts)):
        if a.fold_counts[f] != b.fold_counts[f]:
            raise Error(label + ": fold_counts[" + String(f) + "] "
                        + String(a.fold_counts[f]) + " vs "
                        + String(b.fold_counts[f]))
    if len(a.one_hot) != len(b.one_hot):
        raise Error(label + ": one_hot length " + String(len(a.one_hot))
                    + " vs " + String(len(b.one_hot)))
    for f in range(len(a.one_hot)):
        if a.one_hot[f] != b.one_hot[f]:
            raise Error(label + ": one_hot[" + String(f) + "] differs")
    if len(a.nan_treatment) != len(b.nan_treatment):
        raise Error(label + ": nan_treatment length "
                    + String(len(a.nan_treatment)) + " vs "
                    + String(len(b.nan_treatment)))
    for f in range(len(a.nan_treatment)):
        if a.nan_treatment[f] != b.nan_treatment[f]:
            raise Error(label + ": nan_treatment[" + String(f) + "] "
                        + String(a.nan_treatment[f]) + " vs "
                        + String(b.nan_treatment[f]))
    if len(a.borders) != len(b.borders):
        raise Error(label + ": borders length " + String(len(a.borders))
                    + " vs " + String(len(b.borders)))
    for f in range(len(a.borders)):
        if len(a.borders[f]) != len(b.borders[f]):
            raise Error(label + ": borders[" + String(f) + "] count "
                        + String(len(a.borders[f])) + " vs "
                        + String(len(b.borders[f])))
        for i in range(len(a.borders[f])):
            if not _same_f32(a.borders[f][i], b.borders[f][i]):
                raise Error(label + ": borders[" + String(f) + "]["
                            + String(i) + "] differs bitwise: "
                            + f32_token(a.borders[f][i]) + " vs "
                            + f32_token(b.borders[f][i]))
    if len(a.losses) != len(b.losses):
        raise Error(label + ": losses length " + String(len(a.losses))
                    + " vs " + String(len(b.losses)))
    for i in range(len(a.losses)):
        if not _same_f64(a.losses[i], b.losses[i]):
            raise Error(label + ": losses[" + String(i)
                        + "] differs bitwise: " + f64_token(a.losses[i])
                        + " vs " + f64_token(b.losses[i]))
    if a.model.size() != b.model.size():
        raise Error(label + ": tree count " + String(a.model.size())
                    + " vs " + String(b.model.size()))
    for t in range(a.model.size()):
        ref wa = a.model.weak_models[t]
        ref wb = b.model.weak_models[t]
        if wa.dim != wb.dim:
            raise Error(label + ": tree " + String(t) + " dim differs")
        if wa.structure.get_depth() != wb.structure.get_depth():
            raise Error(label + ": tree " + String(t) + " depth "
                        + String(wa.structure.get_depth()) + " vs "
                        + String(wb.structure.get_depth()))
        for d in range(wa.structure.get_depth()):
            if (
                wa.structure.splits[d].feature_id
                != wb.structure.splits[d].feature_id
            ):
                raise Error(label + ": tree " + String(t) + " level "
                            + String(d) + " feature differs")
            if wa.structure.splits[d].bin_idx != wb.structure.splits[d].bin_idx:
                raise Error(label + ": tree " + String(t) + " level "
                            + String(d) + " bin differs")
            if (
                wa.structure.splits[d].split_type
                != wb.structure.splits[d].split_type
            ):
                raise Error(label + ": tree " + String(t) + " level "
                            + String(d) + " SPLIT TYPE differs: "
                            + bin_split_type_name(
                                Int(wa.structure.splits[d].split_type))
                            + " vs " + bin_split_type_name(
                                Int(wb.structure.splits[d].split_type)))
        if len(wa.leaf_values) != len(wb.leaf_values):
            raise Error(label + ": tree " + String(t) + " leaf count "
                        + String(len(wa.leaf_values)) + " vs "
                        + String(len(wb.leaf_values)))
        for i in range(len(wa.leaf_values)):
            if not _same_f32(wa.leaf_values[i], wb.leaf_values[i]):
                raise Error(label + ": tree " + String(t) + " leaf "
                            + String(i) + " differs bitwise: "
                            + f32_token(wa.leaf_values[i]) + " vs "
                            + f32_token(wb.leaf_values[i]))
        if len(wa.leaf_weights) != len(wb.leaf_weights):
            raise Error(label + ": tree " + String(t) + " weight count "
                        + String(len(wa.leaf_weights)) + " vs "
                        + String(len(wb.leaf_weights)))
        for i in range(len(wa.leaf_weights)):
            if not _same_f32(wa.leaf_weights[i], wb.leaf_weights[i]):
                raise Error(label + ": tree " + String(t) + " weight "
                            + String(i) + " differs bitwise")
    if a.ctr_column_count != b.ctr_column_count:
        raise Error(label + ": ctr_column_count "
                    + String(a.ctr_column_count) + " vs "
                    + String(b.ctr_column_count))
    if len(a.ctr_tables) != len(b.ctr_tables):
        raise Error(label + ": CTR table count " + String(len(a.ctr_tables))
                    + " vs " + String(len(b.ctr_tables)))
    for k in range(len(a.ctr_tables)):
        ref ta = a.ctr_tables[k]
        ref tb = b.ctr_tables[k]
        if ta.column != tb.column or ta.source_feature != tb.source_feature:
            raise Error(label + ": CTR table " + String(k)
                        + " column/source differs")
        if ta.ctr_type != tb.ctr_type:
            raise Error(label + ": CTR table " + String(k) + " type differs")
        if ta.counter_denominator != tb.counter_denominator:
            raise Error(label + ": CTR table " + String(k)
                        + " denominator " + String(ta.counter_denominator)
                        + " vs " + String(tb.counter_denominator))
        # THE TARGET-CLASS AXIS. Without these two lines a round trip that
        # dropped the axis passed field-by-field equality: the blob length
        # is unchanged when a two-class histogram is written as twice as
        # many one-count categories, so `counts` matched and only the shape
        # was wrong. A source sabotage on the writer found this hole and it
        # is closed rather than noted.
        if ta.target_classes_count != tb.target_classes_count:
            raise Error(label + ": CTR table " + String(k)
                        + " TargetClassesCount "
                        + String(ta.target_classes_count) + " vs "
                        + String(tb.target_classes_count))
        if ta.target_border_idx != tb.target_border_idx:
            raise Error(label + ": CTR table " + String(k)
                        + " TargetBorderIdx "
                        + String(ta.target_border_idx) + " vs "
                        + String(tb.target_border_idx))
        if not _same_f32(ta.prior_num, tb.prior_num):
            raise Error(label + ": CTR table " + String(k)
                        + " prior_num differs bitwise")
        if not _same_f32(ta.prior_denom, tb.prior_denom):
            raise Error(label + ": CTR table " + String(k)
                        + " prior_denom differs bitwise")
        if not _same_f32(ta.shift, tb.shift):
            raise Error(label + ": CTR table " + String(k)
                        + " shift differs bitwise")
        if not _same_f32(ta.scale, tb.scale):
            raise Error(label + ": CTR table " + String(k)
                        + " scale differs bitwise")
        if len(ta.counts) != len(tb.counts):
            raise Error(label + ": CTR table " + String(k) + " has "
                        + String(len(ta.counts)) + " categories vs "
                        + String(len(tb.counts)))
        for c in range(len(ta.counts)):
            if ta.counts[c] != tb.counts[c]:
                raise Error(label + ": CTR table " + String(k)
                            + " category " + String(c) + " count "
                            + String(ta.counts[c]) + " vs "
                            + String(tb.counts[c]))


def assert_not_degenerate(tm: TrainedModel, label: String) raises -> Int:
    """A model with one split, or with every leaf the same, passes a
    round-trip trivially. Returns the number of distinct depths so the
    caller can report it."""
    if tm.model.size() < 3:
        raise Error(label + ": " + String(tm.model.size())
                    + " trees is not an ensemble")
    var feats = List[Int]()
    var depths = List[Int]()
    var min_leaf = Float32(0.0)
    var max_leaf = Float32(0.0)
    var first = True
    for t in range(tm.model.size()):
        ref w = tm.model.weak_models[t]
        var d = w.structure.get_depth()
        if d < 1:
            raise Error(label + ": tree " + String(t) + " never split")
        var seen_d = False
        for k in range(len(depths)):
            if depths[k] == d:
                seen_d = True
        if not seen_d:
            depths.append(d)
        for lvl in range(d):
            var fid = Int(w.structure.splits[lvl].feature_id)
            var seen = False
            for k in range(len(feats)):
                if feats[k] == fid:
                    seen = True
            if not seen:
                feats.append(fid)
        for i in range(len(w.leaf_values)):
            if first:
                min_leaf = w.leaf_values[i]
                max_leaf = w.leaf_values[i]
                first = False
            else:
                if w.leaf_values[i] < min_leaf:
                    min_leaf = w.leaf_values[i]
                if w.leaf_values[i] > max_leaf:
                    max_leaf = w.leaf_values[i]
    if len(feats) < 2:
        raise Error(label + ": every split is one feature")
    if min_leaf == max_leaf:
        raise Error(label + ": every leaf value is the same")
    return len(depths)


# ---------------------------------------------------------------------------
# 2. A hand-built model with the corners a trained one does not reach.
# ---------------------------------------------------------------------------


def build_synthetic(ctr_columns: Int = 0) raises -> TrainedModel:
    """Three features: 0 ordered with 16 borders, 1 CONSTANT with none, 2
    one-hot with three categories. Four trees at depths 1, 3, 4 and 6.

    The leaf values are a scattered hash plus five planted bit patterns:
    the three the formatter was measured to lose a ULP on, the smallest
    denormal (whose decimal spelling is `0.0`), and negative zero.
    """
    var fold_counts = List[Int]()
    var one_hot = List[Bool]()
    var borders = List[List[Float32]]()

    var b0 = List[Float32]()
    for i in range(16):
        # scattered thresholds, not a ruler
        b0.append(Float32(Int((i * 37) % 16)) * Float32(0.125) - Float32(1.0))
    # a border list must be ascending for the quantizer to mean anything
    for i in range(len(b0)):
        for j in range(i + 1, len(b0)):
            if b0[j] < b0[i]:
                var tmp = b0[i]
                b0[i] = b0[j]
                b0[j] = tmp
    var uniq = List[Float32]()
    for i in range(len(b0)):
        if len(uniq) == 0 or b0[i] > uniq[len(uniq) - 1]:
            uniq.append(b0[i])
    fold_counts.append(len(uniq))
    one_hot.append(False)
    borders.append(uniq^)

    fold_counts.append(0)
    one_hot.append(False)
    borders.append(List[Float32]())

    var b2 = List[Float32]()
    b2.append(Float32(0.5))
    b2.append(Float32(1.5))
    fold_counts.append(3)
    one_hot.append(True)
    borders.append(b2^)

    var planted = List[UInt32]()
    planted.append(UInt32(3658934777))
    planted.append(UInt32(4252445534))
    planted.append(UInt32(1455409098))
    planted.append(UInt32(1))
    planted.append(UInt32(0x80000000))

    var depths = List[Int]()
    depths.append(1)
    depths.append(3)
    depths.append(4)
    depths.append(6)

    var m = TAdditiveModel()
    var s = UInt64(0x9E3779B97F4A7C15)
    var planted_at = 0
    for t in range(len(depths)):
        var depth = depths[t]
        var st = TObliviousTreeStructure()
        for lvl in range(depth):
            # alternate the two features that HAVE bins, with a bin index
            # inside each one's range
            if (t + lvl) % 3 == 2:
                st.splits.append(
                    TBinarySplit(
                        Int32(2), Int32((lvl % 3)),
                        Int32(BIN_SPLIT_TAKE_BIN),
                    )
                )
            else:
                st.splits.append(
                    TBinarySplit(
                        Int32(0), Int32(1 + ((lvl * 5 + t) % 14)),
                        Int32(BIN_SPLIT_TAKE_GREATER),
                    )
                )
        var tree = TObliviousTreeModel(st^)
        var n_leaves = 1 << depth
        for i in range(n_leaves):
            if t == 3 and planted_at < len(planted):
                tree.leaf_values.append(
                    bitcast[DType.float32](planted[planted_at])
                )
                planted_at += 1
            else:
                var h = _rng(s)
                var v = Float32(Int(h % UInt64(20001))) / Float32(1000.0)
                tree.leaf_values.append(v - Float32(10.0))
        # weights on trees 0 and 2 only: the format carries the vector's
        # PRESENCE, and a model that always has it never tests the absence
        if t == 0 or t == 2:
            for i in range(n_leaves):
                var h = _rng(s)
                tree.leaf_weights.append(
                    Float32(Int(h % UInt64(97))) + Float32(0.5)
                )
        m.add_weak_model(tree^)
    if planted_at != len(planted):
        raise Error("the planted bit patterns did not all land")

    var losses = List[Float64]()
    losses.append(Float64(0.72722083314923369))
    losses.append(bitcast[DType.float64](UInt64(0x3FB999999999999A)))
    losses.append(bitcast[DType.float64](UInt64(0x4341C6BF52634000)))
    losses.append(bitcast[DType.float64](UInt64(1)))
    losses.append(Float64(-0.0))

    # the three held-out fields as `load_model_text` reconstructs them:
    # empty curve, -1 for "not recorded", and no early stop. The text
    # carries none of them, so a round trip that produced anything else
    # would be inventing a held-out history.
    var nan_treatment = List[Int]()
    for _ in range(len(fold_counts)):
        nan_treatment.append(NAN_TREATMENT_AS_IS)
    return TrainedModel(m^, fold_counts^, one_hot^, borders^,
                        nan_treatment^, losses^,
                        List[Float64](), -1, False,
                        ctr_columns, List[TCtrValueTable]())


def check_synthetic_roundtrip(ctx: DeviceContext) raises:
    print("  hand-built model, the corners a trained one does not reach:")
    var tm = build_synthetic()
    var n_depths = assert_not_degenerate(tm, String("synthetic"))
    if n_depths < 2:
        raise Error("the synthetic model has one depth; it was built with"
                    " four")
    print("    4 trees at", n_depths, "distinct depths, 3 features (one"
          " constant, one one-hot), 5 planted hostile bit patterns")

    var text = model_text(tm)
    save_model(String(SCRATCH), tm)
    var back = load_model(String(SCRATCH))
    compare_models(tm, back, String("synthetic round trip"))
    var text2 = model_text(back)
    if text != text2:
        raise Error("re-serializing the loaded model gave different bytes")
    print("    struct equal field by field, and the file re-serializes byte"
          " for byte")

    # and it predicts the same on raw floats
    var n = 512
    var x = _synthetic_rows(n)
    var p1 = predict_floats(ctx, tm, x, n)
    var p2 = predict_floats(ctx, back, x, n)
    var wrong = 0
    for r in range(n):
        if not _same_f32(p1[r], p2[r]):
            wrong += 1
    if wrong != 0:
        raise Error("the loaded synthetic model predicts differently on "
                    + String(wrong) + " of " + String(n) + " rows")
    var spread = _spread(p1)
    print("    predictions bit-identical on all", n, "rows; spread",
          spread, "(a flat prediction would prove nothing)")
    if spread == Float32(0.0):
        raise Error("every prediction is the same value; the fixture cannot"
                    " see a permutation")
    _print_excerpt(text)


def _print_excerpt(text: String) raises:
    """The head and the tail of the file, because the shape of the format is
    worth seeing in the gate output and 130 lines of leaf values are not.
    The whole file is on disk at `SCRATCH`."""
    var lines = _split_lines(text)
    var head = 22
    var tail = 7
    print("    the file (head, tail; the whole of it is at", SCRATCH, "):")
    for i in range(min(head, len(lines))):
        print("     ", lines[i])
    if len(lines) > head + tail:
        print("      ... ", len(lines) - head - tail, "lines elided ...")
        for i in range(len(lines) - tail, len(lines)):
            print("     ", lines[i])


def _synthetic_rows(n: Int) raises -> List[Float32]:
    """Colmajor rows for `build_synthetic`'s three features."""
    var x = List[Float32]()
    var s = UInt64(0xDEADBEEF12345678)
    for _ in range(n):
        var h = _rng(s)
        x.append(Float32(Int(h % UInt64(4001))) / Float32(1000.0)
                 - Float32(2.0))
    for _ in range(n):
        x.append(Float32(7.0))
    for r in range(n):
        x.append(Float32((r * 7 + 1) % 3))
    return x^


def _spread(v: List[Float32]) -> Float32:
    var lo = v[0]
    var hi = v[0]
    for i in range(len(v)):
        if v[i] < lo:
            lo = v[i]
        if v[i] > hi:
            hi = v[i]
    return hi - lo


# ---------------------------------------------------------------------------
# 3. A trained model, one-hot feature included.
# ---------------------------------------------------------------------------


def _training_data(
    n: Int, n_feats: Int, cat_feature: Int, categories: Int
) raises -> Tuple[List[Float32], List[Float32]]:
    var x = List[Float32]()
    for f in range(n_feats):
        for r in range(n):
            if f == cat_feature:
                x.append(Float32((r * 7 + f) % categories))
            else:
                var h = (r * 2654435761 + f * 97003) % 10000
                x.append(Float32(h) / Float32(10000.0) - Float32(0.5))
    var y = List[Float32]()
    for r in range(n):
        var v = Float32(3.0) * x[0 * n + r] - Float32(2.0) * x[2 * n + r]
        if cat_feature >= 0:
            var c = Int(x[cat_feature * n + r])
            v += Float32(Int((c * 37) % 11)) - Float32(5.0)
        y.append(v)
    return (x^, y^)


def check_trained_roundtrip(ctx: DeviceContext) raises:
    """A trained model with a ONE-HOT feature, through save and load.

    THREE CATEGORIES, so that a failure here is a SERIALIZATION failure and
    not a layout one. The fold-count step function
    (`policy_for_fold_count`) is swept on both sides of every step by
    `original/one_hot_cardinality_check.mojo` and again, end to end
    through save, load and both apply paths, by
    `original/ctr_apply_check.mojo`.
    """
    print("  trained model with a one-hot feature:")
    var n = 4096
    var n_feats = 6
    var xy = _training_data(n, n_feats, 5, 3)
    ref x = xy[0]
    ref y = xy[1]
    var one_hot = List[Bool]()
    for f in range(n_feats):
        one_hot.append(f == 5)

    var tm = train(
        ctx, x, y, n, n_feats,
        border_count=32,
        n_estimators=12,
        max_depth=4,
        learning_rate=Float32(0.3),
        one_hot=one_hot,
    )
    var n_depths = assert_not_degenerate(tm, String("trained one-hot"))
    var one_hot_splits = 0
    for t in range(tm.model.size()):
        ref w = tm.model.weak_models[t]
        for lvl in range(w.structure.get_depth()):
            if Int(w.structure.splits[lvl].feature_id) == 5:
                one_hot_splits += 1
    print(
        "    ", tm.model.size(), "trees,", n_depths, "distinct depths,",
        one_hot_splits, "splits on the one-hot feature, final loss",
        tm.losses[len(tm.losses) - 1],
    )
    if one_hot_splits == 0:
        raise Error(
            "the trained model never split on the one-hot feature, so this"
            " check never exercises the equality predicate"
        )

    var before = predict_floats(ctx, tm, x, n)
    save_model(String(SCRATCH), tm)
    var back = load_model(String(SCRATCH))
    compare_models(tm, back, String("trained round trip"))
    var after = predict_floats(ctx, back, x, n)
    var wrong = 0
    for r in range(n):
        if not _same_f32(before[r], after[r]):
            wrong += 1
    if wrong != 0:
        raise Error("the loaded model predicts differently on " + String(wrong)
                    + " of " + String(n) + " rows")
    print("    struct equal field by field; predictions bit-identical on all",
          n, "rows; spread", _spread(before))


# ---------------------------------------------------------------------------
# 4. The device evaluator.
# ---------------------------------------------------------------------------


def evaluator_predict(
    ctx: DeviceContext, tm: TrainedModel, x: List[Float32], n_rows: Int
) raises -> List[Float32]:
    """Drive `gbdt/models/cuda/evaluator.mojo` from a `TrainedModel`: their
    `TGPUModelData` layout for the borders (flat values plus offset and
    count arrays plus the bucket-to-feature identity), then quantize and
    evaluate. Same repack `bench/interleaved/predict_interleaved.mojo`
    builds; a bucket is a feature one to one.
    """
    var n_feats = len(tm.borders)
    var total = 0
    for f in range(n_feats):
        total += len(tm.borders[f])
    if total == 0:
        raise Error("a model with no borders has nothing to quantize")

    var xdev = ctx.enqueue_create_buffer[DType.float32](n_rows * n_feats)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n_rows * n_feats)
    for i in range(n_rows * n_feats):
        hx.unsafe_ptr().unsafe_store(i, x[i])
    ctx.enqueue_copy(dst_buf=xdev, src_ptr=hx.unsafe_ptr())

    var ev_borders = ctx.enqueue_create_buffer[DType.float32](total)
    var ev_off = ctx.enqueue_create_buffer[DType.uint32](n_feats)
    var ev_cnt = ctx.enqueue_create_buffer[DType.uint32](n_feats)
    var ev_bucket = ctx.enqueue_create_buffer[DType.uint32](n_feats)
    var h1 = ctx.enqueue_create_host_buffer[DType.float32](total)
    var h2 = ctx.enqueue_create_host_buffer[DType.uint32](n_feats)
    var h3 = ctx.enqueue_create_host_buffer[DType.uint32](n_feats)
    var h4 = ctx.enqueue_create_host_buffer[DType.uint32](n_feats)
    var pos = 0
    for f in range(n_feats):
        h2.unsafe_ptr().unsafe_store(f, UInt32(pos))
        h3.unsafe_ptr().unsafe_store(f, UInt32(len(tm.borders[f])))
        h4.unsafe_ptr().unsafe_store(f, UInt32(f))
        for i in range(len(tm.borders[f])):
            h1.unsafe_ptr().unsafe_store(pos, tm.borders[f][i])
            pos += 1
    ctx.enqueue_copy(dst_buf=ev_borders, src_ptr=h1.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ev_off, src_ptr=h2.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ev_cnt, src_ptr=h3.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ev_bucket, src_ptr=h4.unsafe_ptr())

    var quantized = ctx.enqueue_create_buffer[DType.uint32](
        quantized_buffer_u32s(n_feats, n_rows)
    )
    var n_res = padded_results(n_rows)
    var results = ctx.enqueue_create_buffer[DType.float32](n_res)
    var h_res = ctx.enqueue_create_host_buffer[DType.float32](n_res)
    ctx.synchronize()

    var gpu_model = pack_model_for_evaluator(ctx, tm.model)
    launch_quantize(
        ctx, xdev, n_rows, n_feats, ev_borders, ev_off, ev_cnt, ev_bucket,
        quantized,
    )
    launch_eval(ctx, gpu_model, quantized, n_feats, n_rows, results)
    ctx.enqueue_copy(dst_ptr=h_res.unsafe_ptr(), src_buf=results)
    ctx.synchronize()

    var out = List[Float32]()
    for r in range(n_rows):
        out.append(h_res.unsafe_ptr().unsafe_load(r))
    return out^


def check_evaluator_agrees(ctx: DeviceContext) raises:
    """The loaded model through THEIR OWN GPU evaluator.

    NO ONE-HOT FEATURE HERE, and that is a division of labour rather than a
    gap: this part pins the FLOAT-ONLY arm of the evaluator, the one whose
    kernel must not move when the one-hot arm exists. The `XorMask` arm and
    a categorical model through the device path are
    `original/ctr_apply_check.mojo`, which also proves the arm is reached
    by flipping the predicate and watching 405 of 512 rows move.

    EIGHT TREES, and that is not arbitrary either. The evaluator's
    cross-block reduce ends in a float atomicAdd, which is order-dependent
    whenever more than one tree block contributes to a document. At
    `tree_count <= 8` the host picks `ext_width = 1` and the tree grid is
    one block, so each output slot has exactly one contributor. The check
    asserts that arithmetic rather than trusting it, and then proves
    reproducibility by running the arm twice before using it as a
    comparator.
    """
    print("  the device evaluator, on the loaded model:")
    var n = 2048
    var n_feats = 5
    var xy = _training_data(n, n_feats, -1, 1)
    ref x = xy[0]
    ref y = xy[1]

    var tm = train(
        ctx, x, y, n, n_feats,
        border_count=32,
        n_estimators=8,
        max_depth=4,
        learning_rate=Float32(0.3),
    )
    _ = assert_not_degenerate(tm, String("trained float-only"))

    var ext_width = (tm.model.size() + 63) // 64
    if ext_width < 1:
        ext_width = 1
    var tree_grid = (
        tm.model.size() + EVAL_TREE_SUB_BLOCK_WIDTH * ext_width - 1
    ) // (EVAL_TREE_SUB_BLOCK_WIDTH * ext_width)
    if tree_grid != 1:
        raise Error(
            "the evaluator would run " + String(tree_grid) + " tree blocks,"
            " so its float atomicAdd has more than one contributor per"
            " document and bit-identity is not a property this arm has."
            " Keep the tree count at or below 8 here"
        )

    var e1 = evaluator_predict(ctx, tm, x, n)
    var e1b = evaluator_predict(ctx, tm, x, n)
    var jitter = 0
    for r in range(n):
        if not _same_f32(e1[r], e1b[r]):
            jitter += 1
    if jitter != 0:
        raise Error(
            "the evaluator is not reproducible at this shape (" + String(jitter)
            + " rows moved between two runs of the SAME model), so it cannot"
            " be used as a bit-identity comparator"
        )
    print("    evaluator reproducible at tree_grid 1: two runs of the same"
          " model agree on all", n, "rows")

    save_model(String(SCRATCH), tm)
    var back = load_model(String(SCRATCH))
    compare_models(tm, back, String("evaluator model round trip"))
    var e2 = evaluator_predict(ctx, back, x, n)
    var wrong = 0
    for r in range(n):
        if not _same_f32(e1[r], e2[r]):
            wrong += 1
    if wrong != 0:
        raise Error("the loaded model scores differently through the DEVICE"
                    " evaluator on " + String(wrong) + " of " + String(n)
                    + " rows")
    print("    loaded model bit-identical through the device evaluator on"
          " all", n, "rows; spread", _spread(e1))
    if _spread(e1) == Float32(0.0):
        raise Error("every evaluator prediction is the same value")

    # cross-check against the host apply, which accumulates trees in a
    # different order: this is a sanity band, not a bit claim
    var host = predict_floats(ctx, back, x, n)
    var worst = Float64(0.0)
    for r in range(n):
        var d = Float64(host[r]) - Float64(e2[r])
        if d < 0:
            d = -d
        if d > worst:
            worst = d
    print("    worst |evaluator - tree-wise apply| over", n, "rows:", worst)
    if worst > 1e-4:
        raise Error("the evaluator and the tree-wise apply disagree by "
                    + String(worst) + ", which is past float reordering")


# ---------------------------------------------------------------------------
# 5. The sabotages.
# ---------------------------------------------------------------------------


def _split_lines(text: String) raises -> List[String]:
    var out = List[String]()
    for piece in text.split("\n"):
        out.append(String(piece))
    return out^


def _join_lines(lines: List[String]) -> String:
    var out = String("")
    for i in range(len(lines)):
        out += lines[i]
        if i + 1 < len(lines):
            out += "\n"
    return out^


def _fields_of(line: String) raises -> List[String]:
    var out = List[String]()
    for piece in line.split(" "):
        var s = String(piece)
        if s.byte_length() > 0:
            out.append(s)
    return out^


def sabotage_one_hot_flags(text: String) raises -> String:
    """DROP A FIELD ON SAVE. Every `one_hot 1` becomes `one_hot 0`: the file
    stays well formed and the header still declares the flags, so nothing
    structural can catch it. Only the field comparison and the predictions
    can."""
    var lines = _split_lines(text)
    var out = List[String]()
    for i in range(len(lines)):
        if lines[i].startswith("feature "):
            var t = _fields_of(lines[i])
            t[5] = String("0")
            var rebuilt = String("")
            for k in range(len(t)):
                if k > 0:
                    rebuilt += " "
                rebuilt += t[k]
            out.append(rebuilt^)
        else:
            out.append(lines[i])
    return _join_lines(out)


def sabotage_drop_losses(text: String) raises -> String:
    """DROP A SECTION ON SAVE. Every `loss` record is deleted; the header
    still declares how many there were."""
    var lines = _split_lines(text)
    var out = List[String]()
    for i in range(len(lines)):
        if not lines[i].startswith("loss "):
            out.append(lines[i])
    return _join_lines(out)


def sabotage_truncate_floats(text: String) raises -> String:
    """TRUNCATE A FLOAT'S PRECISION. Every token's authoritative half is
    rewritten from its own readable half, which is exactly what a
    decimal-only format would have stored."""
    var lines = _split_lines(text)
    var out = List[String]()
    for i in range(len(lines)):
        if lines[i].startswith("#") or lines[i].byte_length() == 0:
            out.append(lines[i])
            continue
        var t = _fields_of(lines[i])
        var rebuilt = String("")
        for k in range(len(t)):
            if k > 0:
                rebuilt += " "
            var parts = t[k].split("/")
            if len(parts) != 2:
                rebuilt += t[k]
                continue
            var dec = String(parts[0])
            if String(parts[1]).byte_length() == 8:
                rebuilt += f32_token(Float32(Float64(dec)))
            else:
                rebuilt += f64_token(Float64(dec))
        out.append(rebuilt^)
    return _join_lines(out)


def sabotage_halve_borders(text: String) raises -> String:
    """DROP HALF OF ONE FEATURE'S BORDERS. The borders are what let a model
    score RAW floats; a format that loses them still round-trips as a
    struct."""
    var lines = _split_lines(text)
    var out = List[String]()
    for i in range(len(lines)):
        if lines[i].startswith("feature 0 "):
            var t = _fields_of(lines[i])
            # the count sits at 11 and the borders start at 12 since the
            # record grew its `nan` field (format version 2)
            var n_b = Int(t[11])
            var keep = n_b // 2
            var rebuilt = String("")
            for k in range(12):
                if k > 0:
                    rebuilt += " "
                rebuilt += t[k] if k != 11 else String(keep)
            for k in range(keep):
                rebuilt += " " + t[12 + k]
            out.append(rebuilt^)
        else:
            out.append(lines[i])
    return _join_lines(out)


def _gate_on_text(
    ctx: DeviceContext,
    tm: TrainedModel,
    text: String,
    x: List[Float32],
    n: Int,
    before: List[Float32],
) raises:
    """The gate, run against a supplied file body: load, compare fields,
    predict, require bit-identity."""
    var back = load_model_text(text)
    compare_models(tm, back, String("sabotaged"))
    var after = predict_floats(ctx, back, x, n)
    var wrong = 0
    for r in range(n):
        if not _same_f32(before[r], after[r]):
            wrong += 1
    if wrong != 0:
        raise Error("predictions differ on " + String(wrong) + " of "
                    + String(n) + " rows")


def _expect_red(label: String, went_red: Bool, why: String) raises:
    if not went_red:
        raise Error(
            "SABOTAGE '" + label + "' DID NOT TURN THE GATE RED. The check"
            " cannot distinguish a working save/load from a broken one, so"
            " nothing it has reported is evidence"
        )
    print("    RED, as required:", label)
    print("        ", why)


def _run_sabotage(
    ctx: DeviceContext,
    tm: TrainedModel,
    label: String,
    text: String,
    x: List[Float32],
    n: Int,
    before: List[Float32],
) raises:
    """Run BOTH halves of the gate against a sabotaged file and report each,
    rather than stopping at whichever fails first.

    This matters: a sabotage that only trips the field comparison has proved
    that the metadata moved, not that the ANSWER moved. Reporting the
    prediction count separately is what shows the corruption reaches the
    output.
    """
    var loaded = True
    var load_err = String("")
    var field_err = String("")
    var pred_moved = -1
    try:
        var back = load_model_text(text)
        try:
            compare_models(tm, back, String("sabotaged"))
        except fe:
            field_err = String(fe)
        var after = predict_floats(ctx, back, x, n)
        var w = 0
        for r in range(n):
            if not _same_f32(before[r], after[r]):
                w += 1
        pred_moved = w
    except e:
        loaded = False
        load_err = String(e)

    var why: String
    if not loaded:
        why = String("load REFUSED the file: ") + load_err
    else:
        why = String("fields: ")
        if field_err.byte_length() == 0:
            why += "equal"
        else:
            why += field_err
        why += String("  |  predictions moved on ") + String(pred_moved)
        why += String(" of ") + String(n) + " rows"
    var red = (
        (not loaded) or field_err.byte_length() != 0 or pred_moved > 0
    )
    _expect_red(label, red, why)


def check_sabotages(ctx: DeviceContext) raises:
    """Five deliberate breaks. Four edit the bytes the real writer produced,
    so what is being sabotaged is the shipped path and not a stand-in."""
    print("  sabotages (each must turn the gate red):")
    var tm = build_synthetic()
    var n = 512
    var x = _synthetic_rows(n)
    var before = predict_floats(ctx, tm, x, n)
    var text = model_text(tm)

    # the gate must be GREEN on the untouched file first, or a red below
    # proves nothing
    _gate_on_text(ctx, tm, text, x, n, before)
    print("    control: the untouched file passes the same gate")

    _run_sabotage(
        ctx, tm, String("drop a field on save (one_hot flags zeroed)"),
        sabotage_one_hot_flags(text), x, n, before,
    )
    _run_sabotage(
        ctx, tm, String("drop a section on save (every loss record)"),
        sabotage_drop_losses(text), x, n, before,
    )
    _run_sabotage(
        ctx, tm,
        String("truncate a float's precision (bits rewritten from decimal)"),
        sabotage_truncate_floats(text), x, n, before,
    )
    _run_sabotage(
        ctx, tm, String("drop half of feature 0's borders"),
        sabotage_halve_borders(text), x, n, before,
    )

    # PERMUTE LEAF ORDER ON LOAD: applied to the loaded struct, because a
    # permutation is the one corruption the file cannot show. Every total is
    # preserved and every affected prediction is wrong.
    var field_err = String("")
    var moved = 0
    var back = load_model_text(text)
    var v0 = back.model.weak_models[0].leaf_values[0]
    var v1 = back.model.weak_models[0].leaf_values[1]
    back.model.weak_models[0].leaf_values[0] = v1
    back.model.weak_models[0].leaf_values[1] = v0
    try:
        compare_models(tm, back, String("permuted"))
    except e:
        field_err = String(e)
    var after = predict_floats(ctx, back, x, n)
    for r in range(n):
        if not _same_f32(before[r], after[r]):
            moved += 1
    var why = String("fields: ")
    if field_err.byte_length() == 0:
        why += "equal"
    else:
        why += field_err
    why += String("  |  predictions moved on ") + String(moved) + " of "
    why += String(n) + " rows"
    _expect_red(
        String("permute leaf order on load (tree 0, leaves 0 and 1)"),
        field_err.byte_length() != 0 or moved > 0, why,
    )

    # and the sum is UNCHANGED by that permutation, which is why a
    # conservation check could never have caught it
    var back2 = load_model_text(text)
    var s_before = Float64(0.0)
    for t in range(back2.model.size()):
        for i in range(len(back2.model.weak_models[t].leaf_values)):
            s_before += Float64(back2.model.weak_models[t].leaf_values[i])
    var w0 = back2.model.weak_models[0].leaf_values[0]
    back2.model.weak_models[0].leaf_values[0] = (
        back2.model.weak_models[0].leaf_values[1]
    )
    back2.model.weak_models[0].leaf_values[1] = w0
    var s_after = Float64(0.0)
    for t in range(back2.model.size()):
        for i in range(len(back2.model.weak_models[t].leaf_values)):
            s_after += Float64(back2.model.weak_models[t].leaf_values[i])
    print("        leaf-value SUM under that permutation:", s_before, "->",
          s_after, "-- identical, which is why the comparison is per"
          " element")

def check_ctr_column_count_travels(ctx: DeviceContext) raises:
    """A CTR-bearing model must not round-trip into a SCOREABLE one.

    `predict_floats` REFUSES a model with CTR columns, because a CTR value
    is a statistic of the learn pool and scoring a new row needs the final
    CTR tables. So the count is a SAFETY field, and a serializer that
    dropped it would convert a model that correctly refuses into one that
    silently scores rows against a grid built for different values. That is
    a worse loss than a leaf value, because nothing downstream looks wrong.

    The field is written ONLY when non-zero, so this also pins the other
    half of that rule: a float-only model's bytes must be unchanged by the
    field existing at all.
    """
    print("ctr_column_count survives the round trip:")
    var tm = build_synthetic()
    var float_only = model_text(tm)
    if float_only.find(String("ctr_columns")) != -1:
        raise Error(
            "a float-only model wrote a ctr_columns record; the format's"
            " own rule is that such a file stays byte-identical to what it"
            " wrote before the field existed"
        )
    print("    float-only model writes no ctr_columns record")

    var carrying = build_synthetic(3)
    var text = model_text(carrying)
    if text.find(String("ctr_columns 3")) == -1:
        raise Error("a model with 3 CTR columns did not write the record")
    var back = load_model_text(text)
    if back.ctr_column_count != 3:
        raise Error(
            "ctr_column_count did not survive: wrote 3, read back "
            + String(back.ctr_column_count)
            + " -- the loaded model would SCORE where the original REFUSES"
        )
    print("    3 CTR columns written and read back as 3")

    var refused = False
    try:
        var probe = List[Float32]()
        for _ in range(len(back.fold_counts)):
            probe.append(Float32(0.0))
        _ = predict_floats(ctx, back, probe, 1)
    except e:
        refused = True
    if not refused:
        raise Error(
            "the LOADED model scored rows; it must refuse exactly as the"
            " original does, which is the whole point of carrying the count"
        )
    print("    the loaded model still REFUSES predict_floats")


def main() raises:
    print("model save/load, exact round trip:")
    check_float_tokens()
    var ctx = DeviceContext()
    check_synthetic_roundtrip(ctx)
    check_trained_roundtrip(ctx)
    check_ctr_column_count_travels(ctx)
    check_evaluator_agrees(ctx)
    check_sabotages(ctx)
    print("model save/load: all parts green, all five sabotages red")
