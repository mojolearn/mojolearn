"""A trained CATEGORICAL model, saved, loaded, and applied to RAW rows --
through the host apply and through the device evaluator, per row.

CatBoost never had to write this file: their model carries `ctr_data` and
their apply path is `TStaticCtrProvider::CalcCtrs`. Ours is
`gbdt/models/ctr_value_table.mojo` plus the `ctr_table` / `ctr_entry`
records of `gbdt/models/model_text.mojo`, so it needs a gate of its own.

WHAT IS CHECKED, AND THE FAILURE EACH PART EXISTS FOR
-----------------------------------------------------
1. THE TABLE ARITHMETIC, against a tally computed here from the codes.
   Their FeatureFreq value is `(count + PriorNum) / (denominator +
   PriorDenom)` with `denominator` the learn row count
   (`private/libs/algo/online_ctr.cpp:937-939`), and an UNSEEN category
   takes `Calc(0, denominator)` (`static_ctr_provider.cpp:65`). Both are
   asserted per category rather than in aggregate.

2. A HAND-BUILT categorical model with real structure: four trees at four
   distinct depths, three columns (an ordered one, a ONE-HOT one, a CTR
   one), splits of BOTH types, scattered leaf values, and rows carrying
   categories the learn pool never saw. A trained model reaches none of
   those corners at once.

3. A TRAINED categorical model through `train(cat_features=...)`, scored
   on the LEARN rows. FeatureFreq is permutation-independent
   (`ctr_type.cpp:44-58`), so the apply-time table must reproduce the
   learn column BIT FOR BIT -- this is the one place where an external
   truth exists without an oracle file, and it is asserted per row.

4. THE DEVICE EVALUATOR on both, including the one-hot arm that
   `gbdt/models/cuda/evaluator.mojo` gained with `XorMask`. Their float
   atomicAdd makes the arm order-dependent when more than one tree block
   contributes, so the check FIRST proves reproducibility by running the
   same model twice and only then uses it as a comparator.

5. A CARDINALITY SWEEP at 1, 2, 3, 15, 16, 17, 31, 32, 254 and 255 on
   BOTH the one-hot and the CTR feature. That list is not decorative: a
   fold-count mismatch once made a one-hot feature silently unlearnable at
   exactly k = 2, 15 and 16, because `policy_for_fold_count` is a step
   function and the writer and the reader of the compressed index
   straddled a step. `k == 1` is its own case: their
   `CB_ENSURE(uniqueValues > 1, "Error: useless catFeature found")`.

6. SABOTAGES, each expected to turn the gate red, because a check never
   seen to fail is not evidence. They include one for each thing this
   round built: an entry dropped from a table, the table permuted, the
   one-hot predicate flipped to `>` on the HOST and on the DEVICE
   separately, and the tables stripped from a model that declares CTR
   columns (which must go back to REFUSING rather than scoring).

THE COMPARATOR IS AN INDEPENDENT TALLY. `reference_score` below quantizes,
walks the trees and sums in plain host Float32, reading nothing from the
device path and re-deriving the CTR value from `count / (n + 1)` -- their
formula, written out rather than called. Every comparison is PER ROW: a
leaf-order permutation leaves the leaf-value sum bit-identical, so a check
that summed could not see one.
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
from gbdt.models.ctr_value_table import (
    TCtrValueTable,
    expand_raw_columns,
)
from gbdt.models.model_text import (
    f32_token,
    load_model,
    load_model_text,
    model_text,
    save_model,
)
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    BIN_SPLIT_TAKE_GREATER,
    TAdditiveModel,
    TBinarySplit,
    TObliviousTreeModel,
    TObliviousTreeStructure,
)
from gbdt.ctrs.ctr import CTR_FEATURE_FREQ
from gbdt.options.catboost_options import TCatFeatureParams
from gbdt.train import (
    TrainedModel,
    model_input_features,
    predict_floats,
    train,
)
from mojo_only.model_io_check import compare_models


comptime SCRATCH = "/tmp/mojolearn_ctr_apply_check.txt"


def _rng(mut s: UInt64) -> UInt64:
    """xorshift64. Scattered, not consecutive: a fixture whose values march
    in order cannot distinguish a permutation from the identity."""
    s ^= s << 13
    s ^= s >> 7
    s ^= s << 17
    return s


def _same_f32(a: Float32, b: Float32) -> Bool:
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


def _spread(v: List[Float32]) -> Float32:
    var lo = v[0]
    var hi = v[0]
    for i in range(len(v)):
        if v[i] < lo:
            lo = v[i]
        if v[i] > hi:
            hi = v[i]
    return hi - lo


def _count_mismatch(
    a: List[Float32], b: List[Float32], n: Int
) raises -> Int:
    if len(a) < n or len(b) < n:
        raise Error("comparing fewer values than rows")
    var wrong = 0
    for r in range(n):
        if not _same_f32(a[r], b[r]):
            wrong += 1
    return wrong


# ---------------------------------------------------------------------------
# The independent tally.
# ---------------------------------------------------------------------------


@fieldwise_init
struct RefPlan(Copyable, Movable):
    """How the CHECK believes the raw input maps onto the model's columns.

    Written out here rather than read from `column_plan`, so a wrong plan in
    the library cannot agree with itself. `ctr_counts_of_column` is empty for
    a column that is not a CTR column.
    """

    var source_of_column: List[Int]
    var is_ctr: List[Bool]
    var ctr_counts: List[List[Int]]
    var ctr_denominator: List[Int]


def reference_score(
    tm: TrainedModel, plan: RefPlan, x_raw: List[Float32], n_rows: Int
) raises -> List[Float32]:
    """Quantize, walk, sum -- in plain host Float32, touching no device
    code and no library expansion.

    The bin is the count of borders the value EXCEEDS
    (`binarize.cu:77`, `featureValues[j] > borderValue`). The CTR value is
    their FeatureFreq formula written out: `(count + 0) / (n + 1)` at the
    GPU default prior `{0.0, 1}` (`cat_feature_options.cpp:117-129`), with
    an unseen category taking count 0. The leaf index is
    `sum over levels of bit_l << l`, level 0 the LEAST significant bit.

    Trees are summed in model order in Float32, which is what the tree-wise
    apply does, so this is a BITWISE comparator and not a tolerance.
    """
    var n_columns = len(tm.fold_counts)
    var out = List[Float32]()
    for _ in range(n_rows):
        out.append(Float32(0.0))

    var bins = List[Int]()
    for _ in range(n_columns):
        bins.append(0)

    for r in range(n_rows):
        for c in range(n_columns):
            var v: Float32
            if plan.is_ctr[c]:
                var code = Int(x_raw[plan.source_of_column[c] * n_rows + r])
                var count = 0
                if code >= 0 and code < len(plan.ctr_counts[c]):
                    count = plan.ctr_counts[c][code]
                v = Float32(count) / (
                    Float32(plan.ctr_denominator[c]) + Float32(1.0)
                )
            else:
                v = x_raw[plan.source_of_column[c] * n_rows + r]
            var b = 0
            for k in range(len(tm.borders[c])):
                if v > tm.borders[c][k]:
                    b += 1
            bins[c] = b
        var acc = Float32(0.0)
        for t in range(tm.model.size()):
            ref w = tm.model.weak_models[t]
            var leaf = 0
            for lvl in range(w.structure.get_depth()):
                var fid = Int(w.structure.splits[lvl].feature_id)
                var bidx = Int(w.structure.splits[lvl].bin_idx)
                var hit: Bool
                if Int(w.structure.splits[lvl].split_type) == (
                    BIN_SPLIT_TAKE_BIN
                ):
                    hit = bins[fid] == bidx
                else:
                    hit = bins[fid] > bidx
                if hit:
                    leaf += 1 << lvl
            acc += w.leaf_values[leaf]
        out[r] = acc
    return out^


# ---------------------------------------------------------------------------
# The device evaluator, driven from a TrainedModel and RAW rows.
# ---------------------------------------------------------------------------


def evaluator_predict(
    ctx: DeviceContext, tm: TrainedModel, x_raw: List[Float32], n_rows: Int
) raises -> List[Float32]:
    """`gbdt/models/cuda/evaluator.mojo` on raw rows: expand the CTR
    columns exactly as `predict_floats` does (their `CalcCtrs` ahead of the
    quantizer), then their `TGPUModelData` repack, quantize and evaluate.
    """
    var n_feats = len(tm.borders)
    var x = expand_raw_columns(tm.ctr_tables, n_feats, x_raw, n_rows)
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


def _assert_evaluator_is_a_comparator(
    ctx: DeviceContext, tm: TrainedModel, x_raw: List[Float32], n_rows: Int
) raises:
    """Their cross-block reduce ends in a float atomicAdd, which is
    order-dependent whenever more than one tree block contributes to a
    document. Prove reproducibility at THIS shape by running the same model
    twice before using the arm as a bit comparator, rather than assuming
    it."""
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
            " document and bit-identity is not a property this arm has"
        )
    var a = evaluator_predict(ctx, tm, x_raw, n_rows)
    var b = evaluator_predict(ctx, tm, x_raw, n_rows)
    var jitter = _count_mismatch(a, b, n_rows)
    if jitter != 0:
        raise Error(
            "the evaluator is not reproducible at this shape ("
            + String(jitter) + " rows moved between two runs of the SAME"
            " model), so it cannot be a bit-identity comparator"
        )


# ---------------------------------------------------------------------------
# Non-degeneracy.
# ---------------------------------------------------------------------------


def assert_structure(tm: TrainedModel, label: String) raises -> Int:
    """A tree that never splits conserves everything trivially, and a model
    that never takes the equality predicate never exercises the arm this
    round built. Returns the number of one-hot splits."""
    var take_bin = 0
    var take_greater = 0
    var feats = List[Int]()
    var min_leaf = Float32(0.0)
    var max_leaf = Float32(0.0)
    var first = True
    if tm.model.size() < 2:
        raise Error(label + ": " + String(tm.model.size())
                    + " trees is not an ensemble")
    for t in range(tm.model.size()):
        ref w = tm.model.weak_models[t]
        if w.structure.get_depth() < 1:
            raise Error(label + ": tree " + String(t) + " never split")
        for lvl in range(w.structure.get_depth()):
            if Int(w.structure.splits[lvl].split_type) == BIN_SPLIT_TAKE_BIN:
                take_bin += 1
            else:
                take_greater += 1
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
        raise Error(label + ": every split is on one feature")
    if min_leaf == max_leaf:
        raise Error(label + ": every leaf value is the same")
    if take_greater == 0:
        raise Error(label + ": no ordered split; the `>` arm is unexercised")
    return take_bin


# ---------------------------------------------------------------------------
# 1. The table arithmetic.
# ---------------------------------------------------------------------------


def check_table_arithmetic() raises:
    print("  the CTR table's arithmetic, per category:")
    var n = 90
    var counts: List[Int] = [3, 11, 29, 5, 17, 2, 23]
    var total = 0
    for i in range(len(counts)):
        total += counts[i]
    if total != n:
        raise Error("the fixture's counts do not sum to its row count")
    var table = TCtrValueTable(
        2, 2, CTR_FEATURE_FREQ,
        Float32(0.0), Float32(1.0), Float32(0.0), Float32(1.0),
        n, counts.copy(),
    )
    var wrong = 0
    var distinct = 0
    var seen = List[Float32]()
    for c in range(len(counts)):
        # their FeatureFreq value at prior {0, 1}, written out
        var want = Float32(counts[c]) / (Float32(n) + Float32(1.0))
        var got = table.value_for(c)
        if not _same_f32(want, got):
            wrong += 1
            print("      category", c, "wanted", f32_token(want), "got",
                  f32_token(got))
        var novel = True
        for k in range(len(seen)):
            if _same_f32(seen[k], got):
                novel = False
        if novel:
            seen.append(got)
            distinct += 1
    if wrong != 0:
        raise Error(
            String(wrong) + " of " + String(len(counts))
            + " categories do not match their FeatureFreq value bitwise"
        )
    if distinct != len(counts):
        raise Error(
            "the fixture's categories are not distinguishable: "
            + String(distinct) + " distinct values for "
            + String(len(counts)) + " categories, so a permuted table"
            " would be invisible here"
        )
    print("     ", len(counts), "categories, all distinct, every value"
          " bit-identical to count/(n+1)")

    # their `emptyVal = ctr->Calc(0, denominator)` for a category the learn
    # pool never saw
    var empty_want = Float32(0.0) / (Float32(n) + Float32(1.0))
    if not _same_f32(table.value_for(len(counts)), empty_want):
        raise Error("an unseen category did not take Calc(0, denominator)")
    if not _same_f32(table.value_for(-1), empty_want):
        raise Error("a negative code did not take Calc(0, denominator)")
    print("      an unseen category takes Calc(0, denominator) =",
          f32_token(empty_want), "-- their emptyVal, not a neighbour")

    # shift and scale are carried, so they must MOVE the value
    var scaled = TCtrValueTable(
        2, 2, CTR_FEATURE_FREQ,
        Float32(0.0), Float32(1.0), Float32(0.25), Float32(3.0),
        n, counts.copy(),
    )
    if _same_f32(scaled.value_for(1), table.value_for(1)):
        raise Error(
            "shift and scale changed nothing; TModelCtr::Calc is"
            " (ctr + Shift) * Scale and a table that ignored them would"
            " read their CPU learner's models wrong"
        )
    print("      shift/scale are live: 0.25/3.0 moves category 1 from",
          f32_token(table.value_for(1)), "to",
          f32_token(scaled.value_for(1)))


# ---------------------------------------------------------------------------
# 2. The hand-built categorical model.
# ---------------------------------------------------------------------------

comptime HB_LEARN_ROWS = 90
comptime HB_CATEGORIES = 7
comptime HB_ONE_HOT_K = 5


def hand_built_counts() -> List[Int]:
    var out: List[Int] = [3, 11, 29, 5, 17, 2, 23]
    return out^


def build_categorical_model(with_tables: Bool = True) raises -> TrainedModel:
    """Three columns and four trees at four distinct depths.

        column 0  <- input 0, ordered numeric, 16 borders
        column 1  <- input 1, ONE-HOT with 5 categories (type cat)
        column 2  <- input 2, CTR over 7 categories (type ctr)

    Splits of both types, on all three columns, at every depth. Leaf values
    are a scattered hash: a fixture whose values march in order cannot
    distinguish a permutation from the identity.
    """
    var fold_counts = List[Int]()
    var one_hot = List[Bool]()
    var borders = List[List[Float32]]()

    # column 0: ordered, borders ascending and not a ruler
    var b0 = List[Float32]()
    for i in range(16):
        b0.append(Float32(i) * Float32(0.125) - Float32(1.0))
    fold_counts.append(len(b0))
    one_hot.append(False)
    borders.append(b0^)

    # column 1: one-hot, k categories -> k-1 synthetic borders, k folds
    var b1 = List[Float32]()
    for c in range(HB_ONE_HOT_K - 1):
        b1.append(Float32(c) + Float32(0.5))
    fold_counts.append(len(b1) + 1)
    one_hot.append(True)
    borders.append(b1^)

    # column 2: the CTR column. Its borders sit between the values the
    # table produces, so every bin is reachable.
    var b2 = List[Float32]()
    b2.append(Float32(0.04))
    b2.append(Float32(0.10))
    b2.append(Float32(0.22))
    fold_counts.append(len(b2))
    one_hot.append(False)
    borders.append(b2^)

    var depths: List[Int] = [1, 3, 4, 6]
    var m = TAdditiveModel()
    var s = UInt64(0x9E3779B97F4A7C15)
    for t in range(len(depths)):
        var depth = depths[t]
        var st = TObliviousTreeStructure()
        for lvl in range(depth):
            var which = (t + lvl) % 3
            if which == 1:
                st.splits.append(
                    TBinarySplit(
                        Int32(1), Int32((lvl + t) % HB_ONE_HOT_K),
                        Int32(BIN_SPLIT_TAKE_BIN),
                    )
                )
            elif which == 2:
                st.splits.append(
                    TBinarySplit(
                        Int32(2), Int32((lvl + t) % 3),
                        Int32(BIN_SPLIT_TAKE_GREATER),
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
        for _ in range(1 << depth):
            var h = _rng(s)
            tree.leaf_values.append(
                Float32(Int(h % UInt64(20001))) / Float32(1000.0)
                - Float32(10.0)
            )
        m.add_weak_model(tree^)

    var losses = List[Float64]()
    losses.append(Float64(0.5))

    var tables = List[TCtrValueTable]()
    if with_tables:
        tables.append(
            TCtrValueTable(
                2, 2, CTR_FEATURE_FREQ,
                Float32(0.0), Float32(1.0), Float32(0.0), Float32(1.0),
                HB_LEARN_ROWS, hand_built_counts(),
            )
        )
    return TrainedModel(
        m^, fold_counts^, one_hot^, borders^, losses^, 1, tables^
    )


def hand_built_rows(n: Int) raises -> List[Float32]:
    """RAW input, colmajor: numeric, one-hot codes, category codes -- with
    codes past the learn pool's cardinality on purpose, so `emptyVal` is
    exercised at apply time."""
    var x = List[Float32]()
    var s = UInt64(0xDEADBEEF12345678)
    for _ in range(n):
        var h = _rng(s)
        x.append(
            Float32(Int(h % UInt64(4001))) / Float32(1000.0) - Float32(2.0)
        )
    for r in range(n):
        var h = _rng(s)
        x.append(Float32(Int(h % UInt64(HB_ONE_HOT_K))))
        _ = r
    var unseen = 0
    for _ in range(n):
        var h = _rng(s)
        # one row in nine carries a category the learn pool never saw
        var code = Int(h % UInt64(HB_CATEGORIES + 2))
        if code >= HB_CATEGORIES:
            unseen += 1
        x.append(Float32(code))
    if unseen == 0:
        raise Error("no row carries an unseen category; emptyVal unchecked")
    return x^


def hand_built_plan() -> RefPlan:
    var src: List[Int] = [0, 1, 2]
    var is_ctr: List[Bool] = [False, False, True]
    var counts = List[List[Int]]()
    counts.append(List[Int]())
    counts.append(List[Int]())
    counts.append(hand_built_counts())
    var denom: List[Int] = [0, 0, HB_LEARN_ROWS]
    return RefPlan(src^, is_ctr^, counts^, denom^)


def check_hand_built(ctx: DeviceContext) raises:
    print("  hand-built categorical model, four depths, both predicates:")
    var tm = build_categorical_model()
    var take_bin = assert_structure(tm, String("hand-built"))
    if take_bin == 0:
        raise Error("the hand-built model has no one-hot split")
    print("     ", tm.model.size(), "trees at depths 1/3/4/6,", take_bin,
          "one-hot splits, 3 columns (ordered, cat, ctr)")

    var n = 512
    var x = hand_built_rows(n)
    var plan = hand_built_plan()

    save_model(String(SCRATCH), tm)
    var back = load_model(String(SCRATCH))
    compare_models(tm, back, String("hand-built round trip"))
    if model_text(tm) != model_text(back):
        raise Error("re-serializing the loaded model gave different bytes")
    print("      saved and loaded: struct equal field by field (split types"
          " and CTR tables included), file re-serializes byte for byte")

    var ref_p = reference_score(back, plan, x, n)
    var host = predict_floats(ctx, back, x, n)
    var wrong = _count_mismatch(ref_p, host, n)
    if wrong != 0:
        for r in range(n):
            if not _same_f32(ref_p[r], host[r]):
                raise Error(
                    "the host apply disagrees with the independent tally on "
                    + String(wrong) + " of " + String(n) + " rows; first at"
                    " row " + String(r) + ": tally " + f32_token(ref_p[r])
                    + " apply " + f32_token(host[r])
                )
    var sp = _spread(host)
    print("      host apply == independent tally on all", n,
          "rows, bitwise; spread", sp)
    if sp == Float32(0.0):
        raise Error("every prediction is the same value")

    _assert_evaluator_is_a_comparator(ctx, back, x, n)
    var dev = evaluator_predict(ctx, back, x, n)
    var wrong_dev = _count_mismatch(ref_p, dev, n)
    if wrong_dev != 0:
        for r in range(n):
            if not _same_f32(ref_p[r], dev[r]):
                raise Error(
                    "the DEVICE evaluator disagrees with the independent"
                    " tally on " + String(wrong_dev) + " of " + String(n)
                    + " rows; first at row " + String(r) + ": tally "
                    + f32_token(ref_p[r]) + " evaluator "
                    + f32_token(dev[r])
                )
    print("      device evaluator == the same tally on all", n,
          "rows, bitwise (its XorMask one-hot arm and its CTR columns)")

    var packed = pack_model_for_evaluator(ctx, back.model)
    if not packed.need_xor_mask:
        raise Error(
            "the repack did not set need_xor_mask on a model with one-hot"
            " splits, so the XorMask kernel was never the one that ran"
        )
    print("      the repack selected the NeedXorMask kernel, as their"
          " evaluator_impl.cpp dispatch does")


# ---------------------------------------------------------------------------
# 3. A trained categorical model.
# ---------------------------------------------------------------------------


def training_data(
    n: Int, one_hot_k: Int, cat_k: Int
) raises -> Tuple[List[Float32], List[Float32]]:
    """Three raw inputs: numeric, one-hot categorical, high-cardinality
    categorical. The target depends on ALL THREE, and on the categorical
    ones through a scattered hash rather than through their code, so a
    model that mistook a code for a value cannot fit it."""
    var x = List[Float32]()
    var s = UInt64(0x243F6A8885A308D3)
    for _ in range(n):
        var h = _rng(s)
        x.append(
            Float32(Int(h % UInt64(10000))) / Float32(10000.0)
            - Float32(0.5)
        )
    for r in range(n):
        x.append(Float32((r * 7 + 1) % one_hot_k))
    for r in range(n):
        # a SKEWED distribution, so the frequencies differ between
        # categories and a permuted table changes the answer
        var h = _rng(s)
        var code = Int(h % UInt64(cat_k))
        if Int(h >> UInt64(20)) % 3 == 0:
            code = code % ((cat_k + 1) // 2 if cat_k > 1 else 1)
        x.append(Float32(code))
    var y = List[Float32]()
    for r in range(n):
        var v = Float32(3.0) * x[r]
        v += Float32(Int((Int(x[n + r]) * 37) % 11)) - Float32(5.0)
        v += Float32(Int((Int(x[2 * n + r]) * 53) % 13)) - Float32(6.0)
        y.append(v)
    return (x^, y^)


def counts_from_codes(
    x: List[Float32], n: Int, column: Int, k: Int
) raises -> List[Int]:
    var counts = List[Int]()
    for _ in range(k):
        counts.append(0)
    for r in range(n):
        counts[Int(x[column * n + r])] += 1
    return counts^


def check_trained(ctx: DeviceContext) raises:
    print("  trained categorical model, train(cat_features=...):")
    var n = 4096
    var one_hot_k = 3
    var cat_k = 23
    var xy = training_data(n, one_hot_k, cat_k)
    ref x = xy[0]
    ref y = xy[1]
    var one_hot: List[Bool] = [False, True, False]
    var cat_features: List[Bool] = [False, False, True]

    var tm = train(
        ctx, x, y, n, 3,
        border_count=32,
        n_estimators=8,
        max_depth=4,
        learning_rate=Float32(0.3),
        one_hot=one_hot,
        cat_features=cat_features,
    )
    if tm.ctr_column_count != 1:
        raise Error(
            "expected exactly one CTR column under feature_freq_only, got "
            + String(tm.ctr_column_count)
        )
    if len(tm.ctr_tables) != 1:
        raise Error(
            "expected exactly one CTR table, got "
            + String(len(tm.ctr_tables))
        )
    if model_input_features(tm) != 3:
        raise Error(
            "the model reports " + String(model_input_features(tm))
            + " raw input features for a 3-input fit"
        )
    var take_bin = assert_structure(tm, String("trained categorical"))
    if take_bin == 0:
        raise Error(
            "the trained model never split on the one-hot feature, so the"
            " equality predicate is unexercised"
        )
    print("     ", tm.model.size(), "trees,", take_bin, "one-hot splits,",
          tm.ctr_column_count, "CTR column over", cat_k,
          "categories, final loss", tm.losses[len(tm.losses) - 1])

    # THE TABLE MUST BE THE LEARN POOL'S OWN COUNTS
    var want_counts = counts_from_codes(x, n, 2, cat_k)
    ref tab = tm.ctr_tables[0]
    if tab.column != 2 or tab.source_feature != 2:
        raise Error("the CTR table names the wrong column or source")
    if tab.counter_denominator != n:
        raise Error(
            "CounterDenominator is " + String(tab.counter_denominator)
            + ", their FeatureFreq value is the learn row count "
            + String(n)
        )
    if len(tab.counts) != cat_k:
        raise Error("the table has " + String(len(tab.counts))
                    + " categories for " + String(cat_k))
    var bad = 0
    for c in range(cat_k):
        if tab.counts[c] != want_counts[c]:
            bad += 1
    if bad != 0:
        raise Error(
            String(bad) + " of " + String(cat_k) + " category counts differ"
            " from an independent tally over the learn rows"
        )
    print("      every one of", cat_k, "category counts matches an"
          " independent tally; denominator is the learn row count")

    var plan_src: List[Int] = [0, 1, 2]
    var plan_is_ctr: List[Bool] = [False, False, True]
    var plan_counts = List[List[Int]]()
    plan_counts.append(List[Int]())
    plan_counts.append(List[Int]())
    plan_counts.append(want_counts.copy())
    var plan_denom: List[Int] = [0, 0, n]
    var plan = RefPlan(
        plan_src^, plan_is_ctr^, plan_counts^, plan_denom^
    )

    save_model(String(SCRATCH), tm)
    var back = load_model(String(SCRATCH))
    compare_models(tm, back, String("trained round trip"))
    print("      saved and loaded: struct equal field by field")

    # LEARN ROWS: FeatureFreq is permutation-independent, so the apply-time
    # table must reproduce the column the fit trained on, bit for bit.
    var host = predict_floats(ctx, back, x, n)
    var ref_p = reference_score(back, plan, x, n)
    var wrong = _count_mismatch(ref_p, host, n)
    if wrong != 0:
        raise Error(
            "the loaded model's host apply disagrees with the independent"
            " tally on " + String(wrong) + " of " + String(n) + " learn rows"
        )
    var before = predict_floats(ctx, tm, x, n)
    var moved = _count_mismatch(before, host, n)
    if moved != 0:
        raise Error(
            "save/load moved " + String(moved) + " of " + String(n)
            + " predictions"
        )
    print("      host apply on the LEARN rows == the tally and == the"
          " unsaved model, bitwise, all", n, "rows; spread", _spread(host))

    _assert_evaluator_is_a_comparator(ctx, back, x, n)
    var dev = evaluator_predict(ctx, back, x, n)
    var wrong_dev = _count_mismatch(ref_p, dev, n)
    if wrong_dev != 0:
        raise Error(
            "the DEVICE evaluator disagrees with the tally on "
            + String(wrong_dev) + " of " + String(n) + " rows"
        )
    print("      device evaluator == the same tally on all", n, "rows")

    # NEW ROWS, including categories the learn pool never saw
    var n2 = 1024
    var xn = List[Float32]()
    var s = UInt64(0x13198A2E03707344)
    for _ in range(n2):
        var h = _rng(s)
        xn.append(
            Float32(Int(h % UInt64(10000))) / Float32(10000.0)
            - Float32(0.5)
        )
    for r in range(n2):
        xn.append(Float32((r * 5 + 2) % one_hot_k))
    var unseen = 0
    for _ in range(n2):
        var h = _rng(s)
        var code = Int(h % UInt64(cat_k + 4))
        if code >= cat_k:
            unseen += 1
        xn.append(Float32(code))
    if unseen == 0:
        raise Error("no new row carries an unseen category")
    var host_new = predict_floats(ctx, back, xn, n2)
    var ref_new = reference_score(back, plan, xn, n2)
    var wrong_new = _count_mismatch(ref_new, host_new, n2)
    if wrong_new != 0:
        raise Error(
            "on NEW rows the host apply disagrees with the tally on "
            + String(wrong_new) + " of " + String(n2) + " rows"
        )
    var dev_new = evaluator_predict(ctx, back, xn, n2)
    var wrong_dev_new = _count_mismatch(ref_new, dev_new, n2)
    if wrong_dev_new != 0:
        raise Error(
            "on NEW rows the device evaluator disagrees with the tally on "
            + String(wrong_dev_new) + " of " + String(n2) + " rows"
        )
    print("      NEW rows (", unseen, "of", n2, "carrying a category the"
          " learn pool never saw): host and device both == the tally,"
          " bitwise; spread", _spread(host_new))


# ---------------------------------------------------------------------------
# 5. The cardinality sweep.
# ---------------------------------------------------------------------------


def sweep_cardinalities() -> List[Int]:
    """1, 2, 3, 15, 16, 17, 31, 32, 254, 255 -- every step of
    `policy_for_fold_count` and both sides of each."""
    var out: List[Int] = [1, 2, 3, 15, 16, 17, 31, 32, 254, 255]
    return out^


def check_one_hot_sweep(ctx: DeviceContext) raises:
    print("  one-hot cardinality sweep (the fold-count steps):")
    var ks = sweep_cardinalities()
    var n = 3000
    var cat_k = 19
    for i in range(len(ks)):
        var k = ks[i]
        var xy = training_data(n, k if k > 1 else 1, cat_k)
        ref x = xy[0]
        ref y = xy[1]
        var one_hot: List[Bool] = [False, True, False]
        var cat_features: List[Bool] = [False, False, True]
        var tm = train(
            ctx, x, y, n, 3,
            border_count=32, n_estimators=6, max_depth=4,
            learning_rate=Float32(0.3),
            one_hot=one_hot, cat_features=cat_features,
        )
        var take_bin = 0
        for t in range(tm.model.size()):
            ref w = tm.model.weak_models[t]
            for lvl in range(w.structure.get_depth()):
                if Int(w.structure.splits[lvl].split_type) == (
                    BIN_SPLIT_TAKE_BIN
                ):
                    take_bin += 1
        if k > 1 and take_bin == 0:
            raise Error(
                "k = " + String(k) + ": the fit never split on the one-hot"
                " feature, so the equality predicate is unexercised at this"
                " cardinality -- which is exactly the shape of the"
                " fold-count bug this sweep exists for"
            )
        if k == 1 and take_bin != 0:
            raise Error(
                "k = 1: a constant one-hot column produced an equality"
                " split"
            )
        var counts = counts_from_codes(x, n, 2, cat_k)
        var plan_src: List[Int] = [0, 1, 2]
        var plan_is_ctr: List[Bool] = [False, False, True]
        var plan_counts = List[List[Int]]()
        plan_counts.append(List[Int]())
        plan_counts.append(List[Int]())
        plan_counts.append(counts.copy())
        var plan_denom: List[Int] = [0, 0, n]
        var plan = RefPlan(
            plan_src^, plan_is_ctr^, plan_counts^, plan_denom^
        )
        save_model(String(SCRATCH), tm)
        var back = load_model(String(SCRATCH))
        compare_models(tm, back, String("one-hot k=" + String(k)))
        var host = predict_floats(ctx, back, x, n)
        var ref_p = reference_score(back, plan, x, n)
        var wrong = _count_mismatch(ref_p, host, n)
        _assert_evaluator_is_a_comparator(ctx, back, x, n)
        var dev = evaluator_predict(ctx, back, x, n)
        var wrong_dev = _count_mismatch(ref_p, dev, n)
        if wrong != 0 or wrong_dev != 0:
            raise Error(
                "k = " + String(k) + ": host wrong on " + String(wrong)
                + ", device wrong on " + String(wrong_dev) + " of "
                + String(n) + " rows against the tally"
            )
        print("      k =", k, " folds", tm.fold_counts[1], " one-hot splits",
              take_bin, " host and device both == tally on all", n, "rows")


def check_ctr_sweep(ctx: DeviceContext) raises:
    """The same sweep on the `cat_features` column -- and THEIR DISPATCH
    SPLITS IT IN TWO. `UseForOneHotEncoding`
    (`binarizations_manager.cpp:106-109`) sends a categorical feature with
    `uniqueValues <= one_hot_max_size` to ONE-HOT and gives it no CTR at
    all, so at the GPU default `one_hot_max_size = 2` the k = 2 row of this
    sweep produces no CTR column. Each row prints which arm it took, which
    is the rule that a benchmark or a check naming no path can be reporting
    about a different one.
    """
    print("  CTR cardinality sweep (their one_hot_max_size dispatch"
          " included):")
    var ks = sweep_cardinalities()
    var n = 3000
    var one_hot_max = TCatFeatureParams.feature_freq_only().one_hot_max_size
    for i in range(len(ks)):
        var k = ks[i]
        var xy = training_data(n, 3, k)
        ref x = xy[0]
        ref y = xy[1]
        var one_hot: List[Bool] = [False, True, False]
        var cat_features: List[Bool] = [False, False, True]
        if k == 1:
            # their CB_ENSURE(uniqueValues > 1, "useless catFeature")
            var raised = False
            try:
                _ = train(
                    ctx, x, y, n, 3,
                    border_count=32, n_estimators=2, max_depth=3,
                    one_hot=one_hot, cat_features=cat_features,
                )
            except:
                raised = True
            if not raised:
                raise Error(
                    "a one-category cat feature was accepted; their"
                    " CB_ENSURE(uniqueValues > 1) refuses it"
                )
            print("      k = 1  REFUSED, as their `useless catFeature`"
                  " ensure does")
            continue
        var tm = train(
            ctx, x, y, n, 3,
            border_count=32, n_estimators=6, max_depth=4,
            learning_rate=Float32(0.3),
            one_hot=one_hot, cat_features=cat_features,
        )
        var counts = counts_from_codes(x, n, 2, k)
        var nonzero = 0
        for c in range(k):
            if counts[c] != 0:
                nonzero += 1
        # their dispatch, and the check has to follow it rather than
        # assume the CTR arm
        var is_one_hot_arm = k <= one_hot_max
        var want_ctr = 0 if is_one_hot_arm else 1
        if tm.ctr_column_count != want_ctr:
            raise Error(
                "k = " + String(k) + ": expected " + String(want_ctr)
                + " CTR columns under one_hot_max_size "
                + String(one_hot_max) + ", got "
                + String(tm.ctr_column_count)
            )
        if len(tm.ctr_tables) != want_ctr:
            raise Error(
                "k = " + String(k) + ": " + String(len(tm.ctr_tables))
                + " CTR tables for " + String(want_ctr) + " CTR columns"
            )
        var plan_src: List[Int] = [0, 1, 2]
        var plan_is_ctr: List[Bool] = [False, False, not is_one_hot_arm]
        var plan_counts = List[List[Int]]()
        plan_counts.append(List[Int]())
        plan_counts.append(List[Int]())
        plan_counts.append(counts.copy())
        var plan_denom: List[Int] = [0, 0, n]
        var plan = RefPlan(
            plan_src^, plan_is_ctr^, plan_counts^, plan_denom^
        )
        save_model(String(SCRATCH), tm)
        var back = load_model(String(SCRATCH))
        compare_models(tm, back, String("ctr k=" + String(k)))
        var host = predict_floats(ctx, back, x, n)
        var ref_p = reference_score(back, plan, x, n)
        var wrong = _count_mismatch(ref_p, host, n)
        _assert_evaluator_is_a_comparator(ctx, back, x, n)
        var dev = evaluator_predict(ctx, back, x, n)
        var wrong_dev = _count_mismatch(ref_p, dev, n)
        if wrong != 0 or wrong_dev != 0:
            raise Error(
                "k = " + String(k) + ": host wrong on " + String(wrong)
                + ", device wrong on " + String(wrong_dev) + " of "
                + String(n) + " rows against the tally"
            )
        print(
            "      k =", k,
            " arm", String("ONE-HOT" if is_one_hot_arm else "CTR"),
            " categories seen", nonzero,
            " column-2 folds", tm.fold_counts[2],
            " host and device both == tally on all", n, "rows",
        )


# ---------------------------------------------------------------------------
# 6. The sabotages.
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


def sabotage_drop_a_ctr_entry(text: String) raises -> String:
    """DROP ONE `ctr_entry`. The table still parses; the count it declares
    no longer matches what it carries."""
    var lines = _split_lines(text)
    var out = List[String]()
    var dropped = False
    for i in range(len(lines)):
        if not dropped and lines[i].startswith("ctr_entry 2 3 "):
            dropped = True
            continue
        out.append(lines[i])
    if not dropped:
        raise Error("the sabotage found no `ctr_entry` to drop")
    return _join_lines(out)


def sabotage_permute_the_table(text: String) raises -> String:
    """SWAP TWO CATEGORIES' COUNTS. Every count is still present and their
    SUM is unchanged, so nothing structural and nothing conserved can see
    it; only a per-row comparison can."""
    var lines = _split_lines(text)
    var a = -1
    var b = -1
    for i in range(len(lines)):
        if lines[i].startswith("ctr_entry 2 1 "):
            a = i
        if lines[i].startswith("ctr_entry 2 2 "):
            b = i
    if a < 0 or b < 0:
        raise Error("the sabotage found no pair of entries to swap")
    var ta = _fields_of(lines[a])
    var tb = _fields_of(lines[b])
    var out = List[String]()
    for i in range(len(lines)):
        if i == a:
            out.append(
                String("ctr_entry 2 1 ") + tb[3]
            )
        elif i == b:
            out.append(
                String("ctr_entry 2 2 ") + ta[3]
            )
        else:
            out.append(lines[i])
    return _join_lines(out)


def sabotage_strip_the_tables(text: String) raises -> String:
    """DELETE EVERY TABLE, KEEP `ctr_columns`. The model must go back to
    REFUSING to score -- the refusal lifts because the tables are present,
    never because the count went missing."""
    var lines = _split_lines(text)
    var out = List[String]()
    for i in range(len(lines)):
        if lines[i].startswith("ctr_table ") or lines[i].startswith(
            "ctr_entry "
        ):
            continue
        if lines[i].startswith("feature 2 "):
            var t = _fields_of(lines[i])
            t[7] = String("float")
            var rebuilt = String("")
            for k in range(len(t)):
                if k > 0:
                    rebuilt += " "
                rebuilt += t[k]
            out.append(rebuilt^)
            continue
        out.append(lines[i])
    return _join_lines(out)


def sabotage_strip_split_type(text: String) raises -> String:
    """REMOVE the `split_type take_bin` token from every one-hot split, so
    the file says `>` where the model means `==`."""
    var lines = _split_lines(text)
    var out = List[String]()
    var touched = 0
    for i in range(len(lines)):
        if lines[i].startswith("split ") and lines[i].find(
            String("split_type")
        ) != -1:
            var t = _fields_of(lines[i])
            out.append(
                t[0] + " " + t[1] + " " + t[2] + " " + t[3] + " " + t[4]
            )
            touched += 1
        else:
            out.append(lines[i])
    if touched == 0:
        raise Error("the sabotage found no one-hot split to strip")
    return _join_lines(out)


def _expect_red(label: String, went_red: Bool, why: String) raises:
    if not went_red:
        raise Error(
            "SABOTAGE '" + label + "' DID NOT TURN THE GATE RED. The check"
            " cannot distinguish a working CTR apply from a broken one, so"
            " nothing it has reported is evidence"
        )
    print("      RED, as required:", label)
    print("          ", why)


def _run_text_sabotage(
    ctx: DeviceContext,
    label: String,
    text: String,
    x: List[Float32],
    n: Int,
    before: List[Float32],
) raises:
    var loaded = True
    var load_err = String("")
    var pred_err = String("")
    var moved = -1
    try:
        var back = load_model_text(text)
        try:
            var after = predict_floats(ctx, back, x, n)
            moved = _count_mismatch(before, after, n)
        except pe:
            pred_err = String(pe)
    except e:
        loaded = False
        load_err = String(e)
    var why: String
    if not loaded:
        why = String("load REFUSED the file: ") + load_err
    elif pred_err.byte_length() != 0:
        why = String("predict REFUSED the model: ") + pred_err
    else:
        why = (
            String("predictions moved on ") + String(moved) + " of "
            + String(n) + " rows"
        )
    var red = (not loaded) or pred_err.byte_length() != 0 or moved > 0
    _expect_red(label, red, why)


def check_sabotages(ctx: DeviceContext) raises:
    print("  sabotages (each must turn the gate red):")
    var tm = build_categorical_model()
    var n = 512
    var x = hand_built_rows(n)
    var before = predict_floats(ctx, tm, x, n)
    var text = model_text(tm)

    # the untouched file first, or a red below proves nothing
    var control = load_model_text(text)
    compare_models(tm, control, String("control"))
    var control_pred = predict_floats(ctx, control, x, n)
    if _count_mismatch(before, control_pred, n) != 0:
        raise Error("the untouched file does not reproduce the model")
    print("      control: the untouched file passes the same gate")

    _run_text_sabotage(
        ctx, String("drop one ctr_entry (category 3 of the table)"),
        sabotage_drop_a_ctr_entry(text), x, n, before,
    )
    _run_text_sabotage(
        ctx, String("permute the table (swap categories 1 and 2)"),
        sabotage_permute_the_table(text), x, n, before,
    )
    _run_text_sabotage(
        ctx, String("strip the tables, keep the ctr_columns count"),
        sabotage_strip_the_tables(text), x, n, before,
    )
    _run_text_sabotage(
        ctx, String("strip `split_type take_bin` from every one-hot split"),
        sabotage_strip_split_type(text), x, n, before,
    )

    # and the SUM under the permutation, which is why the comparison is per
    # row: the counts are the same multiset, so anything that adds them up
    # sees nothing at all
    var permuted = sabotage_permute_the_table(text)
    var sum_a = 0
    var sum_b = 0
    for line in _split_lines(text):
        if String(line).startswith("ctr_entry "):
            sum_a += Int(_fields_of(String(line))[3])
    for line in _split_lines(permuted):
        if String(line).startswith("ctr_entry "):
            sum_b += Int(_fields_of(String(line))[3])
    print("          table count SUM under that permutation:", sum_a, "->",
          sum_b, "-- identical, which is why the comparison is per row")

    # FLIP THE PREDICATE IN THE LOADED MODEL, host side. The layout still
    # says the column is one-hot, so their
    # `CB_ENSURE(dataSet.IsOneHot(split.FeatureId))` has to fire.
    var flipped = load_model_text(text)
    var flips = 0
    for t in range(flipped.model.size()):
        ref w = flipped.model.weak_models[t]
        for lvl in range(w.structure.get_depth()):
            if Int(w.structure.splits[lvl].split_type) == BIN_SPLIT_TAKE_BIN:
                w.structure.splits[lvl].split_type = Int32(
                    BIN_SPLIT_TAKE_GREATER
                )
                flips += 1
    if flips == 0:
        raise Error("there was no one-hot split to flip")
    var host_red = False
    var host_why = String("")
    try:
        var after = predict_floats(ctx, flipped, x, n)
        var moved = _count_mismatch(before, after, n)
        host_red = moved > 0
        host_why = (
            String("predictions moved on ") + String(moved) + " of "
            + String(n) + " rows"
        )
    except e:
        host_red = True
        host_why = String("the apply REFUSED it: ") + String(e)
    _expect_red(
        String("flip the one-hot predicate to `>` on the HOST (")
        + String(flips) + " splits)",
        host_red, host_why,
    )

    # AND THE SAME FLIP THROUGH THE DEVICE EVALUATOR, which is the reach
    # proof for its XorMask arm: the evaluator has no layout to
    # cross-check against, so if its one-hot arm were a no-op this would
    # be bit-identical and green.
    var dev_before = evaluator_predict(ctx, control, x, n)
    if _count_mismatch(before, dev_before, n) != 0:
        raise Error("the evaluator does not reproduce the host apply here")
    var packed_flipped = pack_model_for_evaluator(ctx, flipped.model)
    if packed_flipped.need_xor_mask:
        raise Error(
            "the flipped model still selected the NeedXorMask kernel"
        )
    var dev_after = evaluator_predict(ctx, flipped, x, n)
    var dev_moved = _count_mismatch(dev_before, dev_after, n)
    _expect_red(
        String("flip the one-hot predicate to `>` on the DEVICE evaluator"),
        dev_moved > 0,
        String("evaluator predictions moved on ") + String(dev_moved)
        + " of " + String(n) + " rows, and the repack fell back to the"
        " no-mask kernel",
    )

    # AND A MODEL THAT DECLARES CTR COLUMNS WITH NO TABLES MUST REFUSE
    var no_tables = build_categorical_model(with_tables=False)
    var refused = False
    var why = String("")
    try:
        _ = predict_floats(ctx, no_tables, x, n)
    except e:
        refused = True
        why = String(e)
    _expect_red(
        String("a model with ctr_column_count 1 and no tables must refuse"),
        refused, String("predict_floats REFUSED it: ") + why,
    )


def main() raises:
    print("CTR apply: a categorical model scores raw rows:")
    check_table_arithmetic()
    var ctx = DeviceContext()
    check_hand_built(ctx)
    check_trained(ctx)
    check_one_hot_sweep(ctx)
    check_ctr_sweep(ctx)
    check_sabotages(ctx)
    print("CTR apply: all parts green, every sabotage red")
