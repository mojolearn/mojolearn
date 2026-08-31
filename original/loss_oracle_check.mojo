# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE NINE NEW LOSSES AGAINST CATBOOST'S OWN OUTPUT.

    pixi run -e bench loss-oracle-gen      # regenerate bench/oracle_losses.txt
    pixi run check-loss-oracle             # this file

WHY IT EXISTS. Nine objectives were ported -- Quantile, MAE, LogLinQuantile,
MAPE, Poisson, Lq, Expectile, Tweedie, Huber -- and every gate on them
compared our device against a tally written in this repository or against
libm through FFI. Both are worth having and neither is CatBoost: a libm
oracle confirms we implemented our READING of their kernel correctly, not
that the reading was right. This trains OUR learner and reads THEIR
learner's own output on the same constructed fixture at the same options,
and compares cell by cell.

It found a defect on its first run that neither of the other two kinds of
gate can see. See THE MAPE DEFECT below.

WHICH ARM OF THEIRS. Their CPU, because their GPU arm cannot run on this
machine at all -- `task_type="GPU"` raises on Apple silicon, which is the
thesis of this repository. So this is our GPU against their CPU, said here
as it is said beside every number.

WHAT IS COMPARED, sharpest first:

  1. THE GRID, per feature per border, EXACT. A border disagreement moves
     every row and would otherwise be absorbed into a loss verdict.
  2. TREE 0'S SPLITS, per level, feature and bin. Tree 0 is grown from the
     derivatives at a zero cursor with no boosting drift in front of it, so
     it is the closest reachable thing to a per-row derivative comparison:
     CatBoost exposes no gradient through any Python entry point --
     `TPointwiseTargetsImpl` has no binding, the custom-objective hook
     computes the CALLER'S derivatives rather than reporting theirs, and no
     `predict` / `eval_metric` / `staged_predict` returns one.
  3. EVERY TREE'S SPLITS, per level, counted, plus the FIRST tree that
     diverges -- so a disagreement is localised to a boosting round rather
     than reported only as an end state.
  4. TREE 0'S LEAF VALUES, per leaf, whenever tree 0's structure matched.
     Sixteen weighted sums of their derivatives against sixteen of ours.
  5. EVERY ROW'S PREDICTION, per row.
  6. THE OBJECTIVE ITSELF -- their own `Score`
     (`cuda/targets/kernel/pointwise_targets.cu`) evaluated on their
     predictions and on ours, so "who fits better under the thing that was
     optimised" is a number and not an impression.

WHAT AGREEMENT IS EXPECTED, AND WHERE IT IS NOT.

  * MAE, MAPE and Quantile take `ELeavesEstimation::Exact` on both arms
    (`catboost_options.cpp:289-300`). Their GPU exact estimator sorts on a
    key that keeps only bits [10, 32) of the float and binary-searches a
    fixed sixteen iterations; ours ports that, and their CPU's
    `CalcSampleQuantile` (`libs/helpers/quantile.cpp`) is a different
    implementation of the same definition. Approximate on both sides.
  * Their kernels use `__expf` / `__powf` fast-math where ours use
    `std.math`. Last bits differ by construction, and in a boosted model a
    last-bit difference in a candidate score is a different tree.
  * Their CPU and their GPU are two implementations of the same objective
    and are not bitwise equal to each other either.

So nothing here is gated at bitwise equality below the grid. The gates are
per-cell BANDS and every band is printed beside the value measured against
it -- `TOL_*` are measurements, not claims. As measured 2026-08-21 on the
committed fixture, the worst HONEST numbers across the nine agreeing arms
were: tree-0 leaf gap 2.1e-05 (Poisson), prediction relative RMS 8.3e-05
(Poisson), objective relative gap 3.8e-06 (Expectile), and ZERO wrong
splits out of 48 in every arm but MAPE. The bands sit roughly an order of
magnitude above each.

THE MAPE DEFECT, found by this check and still open.

  Their `ComputeWeightedQuantile` reads the quantile level it solves for
  out of the LOSS PARAMS MAP, defaulting to 0.5 when the map has no
  "alpha" (`cuda/methods/leaves_estimation/leaves_estimation_helper.h:
  72-74`):

      auto it = params.find("alpha");
      float alpha = it == params.end() ? 0.5 : FromString<float>(it->second);

  That is a DIFFERENT float from the one their target kernel receives.
  `TPointwiseTargetsImpl::Init` never sets `Alpha` for MAPE, so the kernel
  gets the member's declared 0 (`targets/pointwise_target_impl.h:364`) --
  and MAPE's kernel does not read it, which is why that is harmless there.

  This port collapsed the two into one float. `gbdt/train.mojo:715` passes
  `alpha=loss_desc.kernel_alpha()` -- correctly 0.0 for MAPE -- into
  `fit`, which hands the same float to `make_bin_optimized_oracle`
  (`doc_parallel_boosting.mojo:580`), and `estimate_exact` forwards it as
  the quantile level (`pointwise_oracle.mojo:495`). MAPE's Exact estimator
  therefore runs at alpha = 0, `needWeights` is zero, their binary search
  collapses to the segment start, and EVERY MAPE LEAF VALUE IS THE MINIMUM
  RESIDUAL IN THE LEAF instead of the MAPE-weighted median.

  Measured three ways before it was believed: CatBoost's own tree-0 MAPE
  leaves reproduce the analytic weighted median to eight digits; ours sit
  at the leaf minimum to the last bit (rank 0.000-0.005 of the leaf's
  residuals in all eight non-empty leaves); and driving
  `compute_exact_approx`'s MAPE arm directly at alpha 0.5 returns the
  weighted median exactly, while at alpha 0.0 it returns the minimum. The
  kernels are right. The float that reaches them is not.

  Neither existing gate could see it. The target kernel is correct, so the
  libm oracle passes. `original/exact_estimation_check.mojo` passes alpha
  explicitly and calls `compute_weighted_quantile` with
  `use_mape_weights = False` on every one of its arms, so the MAPE branch
  had no caller in any check -- `PORTING_RULES.md` 8, a non-default path is
  an unchecked path.

  THE FIX is to carry the estimator's alpha separately from the kernel's:
  `GetAlpha(lossDescription)` (`options/loss_description.cpp:95-102`,
  0.5 when the key is absent) is `TLossDescription.get_alpha()` here and
  already exists. It has to reach `BinOptimizedOracle` as its own field
  rather than riding on `kernel_alpha()`. That is under `gbdt/`, which
  this session does not write.

THE TWEEDIE ROW CLOSES DEVIATION 62. `PORTING.md` 62 records that their GPU
reads `variance_power` into a member nothing reads again
(`pointwise_target_impl.h:288-291`), so their GPU trains at
`variancePower = 0` whatever the user passes, while their CPU uses it
(`private/libs/algo/tensor_search_helpers.cpp:308`). This port threads it,
matching their CPU, and that decision was priced by argument with no
fixture behind it. It has one now: threading the parameter reproduces their
CPU's Tweedie fit to a relative RMS of 8e-08 with all 48 splits identical.
Sabotage T moves our arm to `variance_power = 1.9` -- still a legal Tweedie
-- and the agreement collapses to 8.9e-02 with 38 of 48 splits wrong, so the
parameter demonstrably reaches the kernel and the comparison is sensitive to
it. Sabotage T' sets it to 0, their GPU's own value, and the fit REFUSES TO
TRAIN: at p = 0 the loss degenerates to `-y*e^f + e^{2f}/2`, whose second
derivative grows as `2 e^{2f}`, and on a positive target every split scores
infinite at level 0. Their GPU's dropped parameter does not merely fit a
different Tweedie on this fixture; it fits nothing. Deviation 62 is
measured, not argued.

THE SABOTAGES, each run at the end and each required to move the check:

  S1  swap two rows of THEIR predictions   the row comparison is per cell
                                           and not on a total; a swap
                                           conserves every sum and every
                                           moment
  S2  shift the bin index by one when
      their split border is looked up      the split comparison is reached
  S3  perturb one grid cell by ONE ULP     the border gate is exact and per
                                           cell, not banded
  T   train Tweedie at variance_power 0    deviation 62's open item, and
                                           the only sabotage that perturbs
                                           the DEVICE rather than the
                                           expectation

S1 to S3 perturb the ORACLE side on purpose. Our side cannot be perturbed
from here without editing `gbdt/`, and perturbing the expectation is what
proves the check READS the record it claims to compare against -- a fixture
field no branch consumes is exactly the defect
`original/catboost_apply_check.mojo` was written to close.
"""

from max.gpu.host import DeviceContext
from std.math import exp, sqrt
from std.memory import bitcast

from gbdt.models.model_text import parse_f32, parse_f64
from gbdt.options.catboost_options import (
    LEAF_ESTIMATION_EXACT,
    LEAF_ESTIMATION_GRADIENT,
    LEAF_ESTIMATION_NEWTON,
)
from gbdt.train import TrainedModel, predict_floats, train

comptime ORACLE = "bench/oracle_losses.txt"

#: Bands. Every one is roughly an order of magnitude above the worst
#: honest value the nine agreeing arms produced on 2026-08-21; the run
#: prints both so a drift shows up as a number rather than as a pass.
comptime TOL_LEAF0 = 1e-3
comptime TOL_PRED_RMS = 1e-3
comptime TOL_OBJ = 1e-4


def _split(line: String) -> List[String]:
    var out = List[String]()
    for tok in line.split(" "):
        var s = String(String(tok).strip())
        if s.byte_length() > 0:
            out.append(s^)
    return out^


@fieldwise_init
struct Arm(Copyable, Movable):
    var name: String
    var loss: String
    var target: String
    var method: Int
    var iters: Int
    var alpha: Float32
    var q: Float32
    var delta: Float32
    var vpower: Float32
    var scale: Float64
    var bias: Float64
    var metric: Float64
    var have_metric: Bool
    var split_feat: List[List[Int]]
    var split_border: List[List[Float32]]
    var leaves: List[List[Float64]]
    var pred: List[Float64]


@fieldwise_init
struct Fixture(Copyable, Movable):
    var rows: Int
    var feats: Int
    var depth: Int
    var trees: Int
    var border_count: Int
    var learning_rate: Float32
    var l2: Float32
    var version: String
    var borders: List[List[Float32]]
    var xcol: List[Float32]
    var y_signed: List[Float32]
    var y_positive: List[Float32]
    var arms: List[Arm]


def method_from_name(s: String) raises -> Int:
    if s == "Newton":
        return LEAF_ESTIMATION_NEWTON
    if s == "Gradient":
        return LEAF_ESTIMATION_GRADIENT
    if s == "Exact":
        return LEAF_ESTIMATION_EXACT
    raise Error("unknown leaf estimation method '" + s + "' in the fixture")


def _arm_index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if names[i] == want:
            return i
    raise Error("a record names arm '" + want + "' before its `arm` line")


def load(path: String) raises -> Fixture:
    var f = open(path, "r")
    var text = f.read()
    f.close()

    var rows = 0
    var feats = 0
    var depth = 0
    var trees = 0
    var border_count = 0
    var lr = Float32(0.0)
    var l2 = Float32(0.0)
    var version = String("")
    var borders = List[List[Float32]]()
    var xcol = List[Float32]()
    var y_signed = List[Float32]()
    var y_positive = List[Float32]()
    var arms = List[Arm]()
    var names = List[String]()

    for line in text.splitlines():
        var s = String(String(line).strip())
        if s.byte_length() == 0 or s.startswith(String("#")):
            continue
        var t = _split(s)
        var kind = t[0]
        if kind == String("version"):
            version = t[1]
        elif kind == String("dims"):
            rows = Int(t[1])
            feats = Int(t[2])
            depth = Int(t[3])
            trees = Int(t[4])
            border_count = Int(t[5])
            for _ in range(feats):
                borders.append(List[Float32]())
        elif kind == String("hyper"):
            lr = parse_f32(t[1])
            l2 = parse_f32(t[2])
        elif kind == String("borders"):
            var fi = Int(t[1])
            for i in range(3, len(t)):
                borders[fi].append(parse_f32(t[i]))
        elif kind == String("xcol"):
            # column-major already, and the columns arrive in order
            for i in range(2, len(t)):
                xcol.append(parse_f32(t[i]))
        elif kind == String("target"):
            if t[1] == String("signed"):
                for i in range(2, len(t)):
                    y_signed.append(parse_f32(t[i]))
            elif t[1] == String("positive"):
                for i in range(2, len(t)):
                    y_positive.append(parse_f32(t[i]))
            else:
                raise Error("unknown target name '" + t[1] + "'")
        elif kind == String("arm"):
            var a = Arm(
                t[1], t[2], t[3], method_from_name(t[4]), Int(t[5]),
                Float32(-1.0), Float32(-1.0), Float32(-1.0), Float32(-1.0),
                1.0, 0.0, 0.0, False,
                List[List[Int]](), List[List[Float32]](),
                List[List[Float64]](), List[Float64](),
            )
            var np = Int(t[6])
            for k in range(np):
                var pname = t[7 + 2 * k]
                var pval = parse_f32(t[8 + 2 * k])
                if pname == String("alpha"):
                    a.alpha = pval
                elif pname == String("q"):
                    a.q = pval
                elif pname == String("delta"):
                    a.delta = pval
                elif pname == String("variance_power"):
                    a.vpower = pval
                else:
                    raise Error("unknown loss parameter '" + pname + "'")
            for _ in range(trees):
                a.split_feat.append(List[Int]())
                a.split_border.append(List[Float32]())
                a.leaves.append(List[Float64]())
            names.append(a.name)
            arms.append(a^)
        elif kind == String("armscalebias"):
            var ai = _arm_index(names, t[1])
            arms[ai].scale = parse_f64(t[2])
            arms[ai].bias = parse_f64(t[3])
        elif kind == String("armmetric"):
            var ai = _arm_index(names, t[1])
            arms[ai].have_metric = Int(t[2]) != 0
            arms[ai].metric = parse_f64(t[3])
        elif kind == String("armsplit"):
            var ai = _arm_index(names, t[1])
            var ti = Int(t[2])
            arms[ai].split_feat[ti].append(Int(t[4]))
            arms[ai].split_border[ti].append(parse_f32(t[5]))
        elif kind == String("armleaves"):
            var ai = _arm_index(names, t[1])
            var ti = Int(t[2])
            for i in range(3, len(t)):
                arms[ai].leaves[ti].append(parse_f64(t[i]))
        elif kind == String("armpred"):
            var ai = _arm_index(names, t[1])
            for i in range(2, len(t)):
                arms[ai].pred.append(parse_f64(t[i]))
        else:
            raise Error("unknown record '" + kind + "' in " + path)

    if len(xcol) != rows * feats:
        raise Error(
            "fixture has " + String(len(xcol)) + " feature cells, expected "
            + String(rows * feats)
        )
    if len(y_signed) != rows or len(y_positive) != rows:
        raise Error("fixture target length disagrees with its own dims")
    if len(arms) == 0:
        raise Error("fixture carries no arms")
    for i in range(len(arms)):
        # SCALE AND BIAS IS CHECKED, NOT ASSUMED. Their evaluator computes
        # `scale * sum(leaves) + bias`, and comparing our raw ensemble sum
        # against their `predict` is valid only while that is the
        # identity. "It is probably 1 and 0" is a guess a tolerance would
        # absorb in silence.
        if arms[i].scale != 1.0 or arms[i].bias != 0.0:
            raise Error(
                "arm " + arms[i].name + " has scale_and_bias "
                + String(arms[i].scale) + " / " + String(arms[i].bias)
                + ", not the identity, so their predictions are not the"
                " bare ensemble sum this compares against"
            )
        if len(arms[i].pred) != rows:
            raise Error(
                "arm " + arms[i].name + " has " + String(len(arms[i].pred))
                + " predictions, expected " + String(rows)
            )
    return Fixture(
        rows, feats, depth, trees, border_count, lr, l2, version,
        borders^, xcol^, y_signed^, y_positive^, arms^,
    )


# ---------------------------------------------------------------------------
# Their own Score, transcribed. `cuda/targets/kernel/pointwise_targets.cu`.
#
# This is the OBJECTIVE, not the reported metric, and the two differ:
# `TMAPETarget::Score` (`:148`) divides by max(1, |target|) while their MAPE
# METRIC divides by |target|, and MAE trains as TQuantileTarget(0.5)
# (`pointwise_target_impl.h:272-275`), whose Score is half the mean absolute
# error. Comparing two FITS has to use the thing that was optimised.
# ---------------------------------------------------------------------------
def objective_score(
    loss: String, target: Float64, prediction: Float64,
    alpha: Float64, q: Float64, delta: Float64, vpower: Float64,
) raises -> Float64:
    if loss == String("Quantile") or loss == String("MAE"):
        var a = alpha if loss == String("Quantile") else 0.5
        var val = target - prediction
        var m = a if val > 0.0 else -(1.0 - a)
        return m * val
    if loss == String("LogLinQuantile"):
        var val = target - exp(prediction)
        var m = alpha if val > 0.0 else -(1.0 - alpha)
        return m * val
    if loss == String("MAPE"):
        var d = abs(target - prediction)
        var den = abs(target)
        if den < 1.0:
            den = 1.0
        return d / den
    if loss == String("Poisson"):
        return exp(prediction) - target * prediction
    if loss == String("Lq"):
        return abs(target - prediction) ** q
    if loss == String("Expectile"):
        var val = target - prediction
        var m = alpha if val > 0.0 else (1.0 - alpha)
        return m * val * val
    if loss == String("Tweedie"):
        var v = -target * exp((1.0 - vpower) * prediction) / (1.0 - vpower)
        var d2 = exp((2.0 - vpower) * prediction) / (2.0 - vpower)
        return v + d2
    if loss == String("Huber"):
        var m = abs(target - prediction)
        if m < delta:
            return 0.5 * m * m
        return delta * (m - 0.5 * delta)
    raise Error("no host Score for loss '" + loss + "'")


def mean_score(
    a: Arm, y: List[Float32], preds: List[Float64], vpower: Float64
) raises -> Float64:
    var total = Float64(0.0)
    for i in range(len(y)):
        total += objective_score(
            a.loss, Float64(y[i]), preds[i],
            Float64(a.alpha), Float64(a.q), Float64(a.delta), vpower,
        )
    return total / Float64(len(y))


@fieldwise_init
struct ArmResult(Copyable, Movable):
    var border_wrong: Int
    var tree0_split_wrong: Int
    var split_wrong: Int
    var split_total: Int
    var first_bad_tree: Int
    var leaf0_worst_rel: Float64
    var leaf0_compared: Int
    var pred_rel_rms: Float64
    var pred_worst_rel: Float64
    var pred_worst_row: Int
    var their_obj: Float64
    var our_obj: Float64
    var obj_rel_gap: Float64


def _bin_of(borders: List[Float32], value: Float32) raises -> Int:
    """Their split border VALUE -> our bin index, by EXACT equality.

    Both sides are the same float32 written at full precision by the same
    generator, so a near-miss means the grids have diverged and a
    nearest-match would paper over exactly that.
    """
    for b in range(len(borders)):
        if borders[b] == value:
            return b
    raise Error(
        "CatBoost split on a border that is not in its own grid for that"
        " feature -- the fixture is internally inconsistent, which is a"
        " generator bug and not a learner bug"
    )


def run_arm(
    ctx: DeviceContext,
    fx: Fixture,
    ai: Int,
    border_sabotage: Int,
    split_sabotage: Bool,
    pred_swap: Bool,
    vpower_override: Float32,
    show_leaves: Bool,
) raises -> ArmResult:
    ref a = fx.arms[ai]
    var y = (
        fx.y_signed.copy() if a.target == String("signed")
        else fx.y_positive.copy()
    )
    var vp = a.vpower
    if vpower_override >= Float32(0.0):
        vp = vpower_override

    # SAME EVERYTHING EXCEPT THE DEVICE. Every option here is the one the
    # generator pinned on their arm, including `leaf_estimation_method`
    # and `leaf_estimation_iterations`, which are set EXPLICITLY on both
    # sides rather than defaulted: `GetEstimationMethodDefaults`
    # (`catboost_options.cpp:31-244`) is keyed on TASK TYPE and Tweedie's
    # entry differs -- 1 Newton iteration on CPU, 20 on GPU (`:221-231`),
    # and this port takes the GPU number. Defaulting both sides would
    # compare a 20-iteration fit against a 1-iteration fit and report a
    # configuration difference as a loss defect.
    var tm = train(
        ctx, fx.xcol, y, fx.rows, fx.feats,
        border_count=fx.border_count,
        n_estimators=fx.trees,
        max_depth=fx.depth,
        learning_rate=fx.learning_rate,
        l2_leaf_reg=fx.l2,
        bootstrap_type=String("No"),
        random_seed=UInt64(0),
        loss=a.loss,
        loss_alpha=a.alpha,
        loss_q=a.q,
        loss_delta=a.delta,
        loss_variance_power=vp,
        leaf_estimation_iterations=a.iters,
        leaf_estimation_method=a.method,
    )

    # 1. THE GRID, per cell, exact.
    var border_wrong = 0
    for fi in range(fx.feats):
        ref theirs = fx.borders[fi]
        ref ours = tm.borders[fi]
        if len(ours) != len(theirs):
            border_wrong += len(ours) + len(theirs)
            continue
        for b in range(len(theirs)):
            var tv = theirs[b]
            if border_sabotage == fi * 1000 + b:
                # ONE ULP: the smallest thing an exact gate can see and
                # the largest thing a band would swallow
                tv = bitcast[DType.float32](
                    bitcast[DType.uint32](tv) + UInt32(1)
                )
            if ours[b] != tv:
                border_wrong += 1

    # 2 and 3. SPLITS, per tree per level.
    var split_wrong = 0
    var split_total = 0
    var tree0_wrong = 0
    var first_bad = -1
    for ti in range(fx.trees):
        var their_n = len(a.split_feat[ti])
        var our_n = 0
        if ti < tm.model.size():
            our_n = len(tm.model.weak_models[ti].structure.splits)
        var n = their_n if their_n > our_n else our_n
        split_total += n
        var bad_here = 0
        for lvl in range(n):
            if lvl >= their_n or lvl >= our_n:
                bad_here += 1
                continue
            var tf = a.split_feat[ti][lvl]
            var tb = _bin_of(fx.borders[tf], a.split_border[ti][lvl])
            if split_sabotage:
                tb += 1
            ref os = tm.model.weak_models[ti].structure.splits[lvl]
            if Int(os.feature_id) != tf or Int(os.bin_idx) != tb:
                bad_here += 1
        split_wrong += bad_here
        if ti == 0:
            tree0_wrong = bad_here
        if bad_here != 0 and first_bad < 0:
            first_bad = ti

    # 4. TREE 0'S LEAVES, per leaf, only when the structure matched --
    # comparing leaf values across different trees compares nothing.
    var leaf_worst = Float64(0.0)
    var leaf_n = 0
    if tree0_wrong == 0 and tm.model.size() > 0:
        ref theirs0 = a.leaves[0]
        ref ours0 = tm.model.weak_models[0].leaf_values
        if len(theirs0) == len(ours0):
            # A leaf value at tree 0 is the learning rate times a step, so
            # a bare relative gap on a near-zero leaf is noise. Normalise
            # by the largest leaf in the tree: that is what "this leaf
            # moved" means.
            var big = Float64(0.0)
            for i in range(len(theirs0)):
                var m = abs(theirs0[i])
                if m > big:
                    big = m
            if big <= 0.0:
                big = 1.0
            for i in range(len(theirs0)):
                var d = abs(Float64(ours0[i]) - theirs0[i]) / big
                if d > leaf_worst:
                    leaf_worst = d
            leaf_n = len(theirs0)
            # A DIVERGENCE IS A DIAGNOSIS, NOT A NUMBER. Sixteen printed
            # cells localise it to a leaf, which is where reading their
            # file starts.
            if show_leaves and leaf_worst > TOL_LEAF0:
                print("      tree 0 leaves, theirs then ours, arm", a.name)
                for i in range(len(theirs0)):
                    print("       ", i, theirs0[i], Float64(ours0[i]))

    # 5. EVERY ROW.
    var ours_pred = predict_floats(ctx, tm, fx.xcol, fx.rows)
    var theirs_pred = a.pred.copy()
    if pred_swap:
        var tmp = theirs_pred[7]
        theirs_pred[7] = theirs_pred[1913]
        theirs_pred[1913] = tmp

    var sum_sq = Float64(0.0)
    var sum_ref = Float64(0.0)
    for r in range(fx.rows):
        var d = Float64(ours_pred[r]) - theirs_pred[r]
        sum_sq += d * d
        sum_ref += theirs_pred[r] * theirs_pred[r]
    var rms_ref = sqrt(sum_ref / Float64(fx.rows))
    if rms_ref <= 0.0:
        rms_ref = 1.0
    var rel_rms = sqrt(sum_sq / Float64(fx.rows)) / rms_ref

    var worst = Float64(0.0)
    var worst_row = -1
    for r in range(fx.rows):
        var d = abs(Float64(ours_pred[r]) - theirs_pred[r]) / (
            abs(theirs_pred[r]) + rms_ref
        )
        if d > worst:
            worst = d
            worst_row = r

    # 6. THE OBJECTIVE. Both fits scored by the SAME formula, on the arm's
    # own parameter -- a sabotage that trains at a different parameter is
    # still judged by the objective the fixture was generated under.
    var their_obj = mean_score(a, y, theirs_pred, Float64(a.vpower))
    var ours64 = List[Float64]()
    for r in range(fx.rows):
        ours64.append(Float64(ours_pred[r]))
    var our_obj = mean_score(a, y, ours64, Float64(a.vpower))
    var denom = abs(their_obj)
    if denom < 1e-12:
        denom = 1e-12
    var obj_gap = abs(our_obj - their_obj) / denom

    return ArmResult(
        border_wrong, tree0_wrong, split_wrong, split_total, first_bad,
        leaf_worst, leaf_n, rel_rms, worst, worst_row, their_obj, our_obj,
        obj_gap,
    )


def report(name: String, r: ArmResult) raises:
    print(
        "  " + name,
        "| grid wrong", r.border_wrong,
        "| tree0 splits wrong", r.tree0_split_wrong,
        "| splits wrong", String(r.split_wrong) + "/" + String(r.split_total),
        "| first bad tree", r.first_bad_tree,
    )
    print(
        "      tree0 leaf gap", r.leaf0_worst_rel, "over", r.leaf0_compared,
        "cells (band", Float64(TOL_LEAF0), ")",
    )
    print(
        "      prediction rel RMS", r.pred_rel_rms, "(band",
        Float64(TOL_PRED_RMS), "), worst row", r.pred_worst_row, "at",
        r.pred_worst_rel,
    )
    print(
        "      objective theirs", r.their_obj, "ours", r.our_obj,
        "rel gap", r.obj_rel_gap, "(band", Float64(TOL_OBJ), ")",
    )


def grade(name: String, r: ArmResult) -> Int:
    var bad = 0
    if r.border_wrong != 0:
        print("      FAIL", name, ": the quantization grid disagrees in",
              r.border_wrong, "cells")
        bad += 1
    if r.tree0_split_wrong != 0:
        print("      FAIL", name, ": tree 0's structure disagrees in",
              r.tree0_split_wrong, "levels -- the derivatives at a zero"
              " cursor do not match theirs")
        bad += 1
    if r.split_wrong != 0:
        print("      FAIL", name, ": ", r.split_wrong, "of",
              r.split_total, "splits disagree, first at tree",
              r.first_bad_tree)
        bad += 1
    if r.leaf0_worst_rel > TOL_LEAF0:
        print("      FAIL", name, ": tree 0 leaf values off by",
              r.leaf0_worst_rel)
        bad += 1
    if r.pred_rel_rms > TOL_PRED_RMS:
        print("      FAIL", name, ": predictions off by relative RMS",
              r.pred_rel_rms)
        bad += 1
    if r.obj_rel_gap > TOL_OBJ:
        print("      FAIL", name, ": the objective differs by",
              r.obj_rel_gap, "relative")
        bad += 1
    return bad


def main() raises:
    var ctx = DeviceContext()
    var fx = load(ORACLE)
    print(
        "CatBoost", fx.version, "loss oracle:", fx.rows, "rows,", fx.feats,
        "features,", fx.trees, "trees, depth", fx.depth, ", border_count",
        fx.border_count,
    )
    print(
        "OUR GPU against THEIR CPU -- their GPU arm does not run on this"
        " machine at all"
    )
    print()

    var failures = 0
    var tweedie = -1
    var huber = -1
    for ai in range(len(fx.arms)):
        ref a = fx.arms[ai]
        if a.loss == String("Tweedie"):
            tweedie = ai
        if a.loss == String("Huber"):
            huber = ai
        var r = run_arm(
            ctx, fx, ai, -1, False, False, Float32(-1.0), True
        )
        report(a.name, r)
        failures += grade(a.name, r)
        print()

    # ---------------------------------------------------------------
    # DEVIATION 62, measured. `PORTING.md` 62: their GPU drops Tweedie's
    # variance_power (`pointwise_target_impl.h:288-291` reads it into a
    # member nothing reads again) and trains at 0; their CPU honours it
    # (`private/libs/algo/tensor_search_helpers.cpp:308`). This port
    # threads it, which is the CPU behaviour, and the deviation was an
    # OPEN ITEM because no fixture had been run through both arms.
    # ---------------------------------------------------------------
    if tweedie < 0:
        raise Error("the fixture carries no Tweedie arm")
    print("-- deviation 62: Tweedie's variance_power, both ways --")
    var t_threaded = run_arm(
        ctx, fx, tweedie, -1, False, False, Float32(-1.0), False
    )
    print(
        "   threaded (ours, and their CPU): rel RMS", t_threaded.pred_rel_rms,
        ", splits wrong", t_threaded.split_wrong,
    )

    # THEIR GPU'S ACTUAL BEHAVIOUR: variance_power = 0. At p = 0 the loss
    # degenerates to `-y*e^f + e^{2f}/2`, whose second derivative grows as
    # `2 e^{2f}`, and on a positive target this fixture does not converge
    # at all -- the fit REFUSES rather than fitting something different.
    # That is a stronger answer to deviation 62 than a numeric gap, so it
    # is caught and reported rather than allowed to abort the run.
    var dropped_trains = True
    var dropped_rms = Float64(0.0)
    var dropped_msg = String("")
    try:
        var t_dropped = run_arm(
            ctx, fx, tweedie, -1, False, False, Float32(0.0), False
        )
        dropped_rms = t_dropped.pred_rel_rms
        print(
            "   dropped  (their GPU, variance_power = 0): rel RMS",
            dropped_rms, ", splits wrong", t_dropped.split_wrong,
        )
    except e:
        dropped_trains = False
        dropped_msg = String(e)
        print(
            "   dropped  (their GPU, variance_power = 0): REFUSED TO"
            " TRAIN --", dropped_msg,
        )

    # A SECOND, LEGAL value of the parameter, so the answer does not rest
    # on an overflow. 1.9 is inside their (1, 2) range, so this is a real
    # Tweedie fit that simply is not the fixture's Tweedie.
    var t_other = run_arm(
        ctx, fx, tweedie, -1, False, False, Float32(1.9), False
    )
    print(
        "   variance_power 1.9 (legal, and not the fixture's): rel RMS",
        t_other.pred_rel_rms, ", splits wrong", t_other.split_wrong,
    )

    var d62_ok = (
        (not dropped_trains) or dropped_rms > t_threaded.pred_rel_rms * 100.0
    )
    if not (d62_ok and t_other.pred_rel_rms > t_threaded.pred_rel_rms * 100.0):
        print(
            "      FAIL: changing variance_power changed almost nothing, so"
            " this fixture cannot tell the two arms apart and deviation 62"
            " stays OPEN"
        )
        failures += 1
    else:
        print(
            "      deviation 62 CLOSED: threading variance_power reproduces"
            " their CPU to a relative RMS of", t_threaded.pred_rel_rms,
            "with", t_threaded.split_wrong, "of", t_threaded.split_total,
            "splits wrong; their GPU's dropped value does not",
        )
    print()

    # ---------------------------------------------------------------
    # SABOTAGES. One per mechanism, each proved to turn this check red.
    # ---------------------------------------------------------------
    if huber < 0:
        raise Error("the fixture carries no Huber arm")
    print("-- sabotages --")
    var base = run_arm(ctx, fx, huber, -1, False, False, Float32(-1.0), False)

    var s1 = run_arm(ctx, fx, huber, -1, False, True, Float32(-1.0), False)
    print(
        "   S1 swap two of THEIR prediction rows: worst rel",
        base.pred_worst_rel, "->", s1.pred_worst_rel,
    )
    if not (s1.pred_rel_rms > TOL_PRED_RMS):
        print("      FAIL: S1 did not turn the row comparison red")
        failures += 1

    var s2 = run_arm(ctx, fx, huber, -1, True, False, Float32(-1.0), False)
    print(
        "   S2 shift the looked-up bin index by one: splits wrong",
        base.split_wrong, "->", s2.split_wrong,
    )
    if s2.split_wrong != s2.split_total:
        print("      FAIL: S2 did not turn every split comparison red")
        failures += 1

    var s3 = run_arm(ctx, fx, huber, 3007, False, False, Float32(-1.0), False)
    print(
        "   S3 one grid cell moved ONE ULP: grid wrong",
        base.border_wrong, "->", s3.border_wrong,
    )
    if s3.border_wrong != 1:
        print("      FAIL: S3 did not turn exactly one grid cell red")
        failures += 1

    print(
        "   T  Tweedie at variance_power 1.9 instead of the fixture's 1.5:"
        " rel RMS", t_threaded.pred_rel_rms, "->", t_other.pred_rel_rms,
    )
    if not (t_other.pred_rel_rms > TOL_PRED_RMS):
        print("      FAIL: T did not turn the row comparison red")
        failures += 1
    print(
        "   T' Tweedie at variance_power 0, their GPU's own value:",
        "REFUSED TO TRAIN" if not dropped_trains
        else String("rel RMS ") + String(dropped_rms),
    )
    if dropped_trains and dropped_rms <= TOL_PRED_RMS:
        print("      FAIL: T' did not turn the row comparison red")
        failures += 1
    print()

    if failures != 0:
        raise Error(
            "loss oracle check FAILED with " + String(failures)
            + " -- read the FAIL lines above. As of 2026-08-21 the expected"
            " failures are the FOUR on the MAPE arm and nothing else, and"
            " they are real: MAPE's Exact leaf estimator runs at alpha 0"
            " because `gbdt/train.mojo:715` feeds it the TARGET KERNEL's"
            " alpha, where CatBoost reads the estimator's alpha from the"
            " loss params map with a 0.5 default"
            " (`leaves_estimation_helper.h:72-74`). Every MAPE leaf value"
            " is the leaf's MINIMUM residual as a result. A failure on any"
            " other arm is new. See this file's docstring."
        )
    print(
        "loss oracle check: nine objectives against CatBoost's own CPU"
        " output, grid + tree 0 + every split + every row + the objective"
        " -- all pass"
    )
