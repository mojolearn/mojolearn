"""`secondDerAsWeights` gates: the Newton score functions must be Newton.

    pixi run check-second-der-weights

CatBoost decides what the histogram's WEIGHT plane holds FROM THE SCORE
FUNCTION: `secondDerAsWeights = IsSecondOrderScoreFunction(scoreFunction)`
(`greedy_search_helper.cpp:286-296`, `enum_helpers.cpp:830-846`). Under
NewtonCosine/NewtonL2 their `StochasticDer` hands `&weightsView` to
`Approximate` as the der2 output (`pointwise_target_impl.h:193-201`) and the
kernel fills it `der2[i] = weight * target.Der2(relev, val)`
(`pointwise_targets.cu:268-269`) -- so plane 0 is `weight * der2`, sample
weight folded in, and plane 1 stays `weight * der`. The doc-parallel searcher
keys the same choice as `NewtonAtZero` vs `GradientAtZero`
(`oblivious_tree_doc_parallel_structure_searcher.cpp:195-207`).

FOUR GATES, and why each is shaped the way it is:

  S1  PER-CELL, on Logloss AND Poisson, against a Float64 host expectation
      computed independently in this file. Per cell with hashed distinct
      weights/targets/predictions, because a total is satisfied by a
      permutation. Logloss reaches `cross_entropy_kernel` and Poisson
      reaches `pointwise_target_kernel`, so BOTH kernels' second-order
      branches are covered by a loss whose der2 is NOT constant. The
      first-order launch is checked in the same pass: plane 0 must be the
      raw weight BIT-EXACTLY.

  S2  THE MODEL MOVES: under Logloss, NewtonCosine must grow a different
      tree than Cosine, and NewtonL2 than L2. Counted split by split.

  S3  THE NEGATIVE CONTROL: under RMSE, NewtonCosine and Cosine must be
      BIT-IDENTICAL -- splits and leaf values -- because
      `TRmseTarget::Der2` returns 1.0f, so `weight * der2 == weight`
      exactly. This is an analytic identity forced by their arithmetic,
      not a tally of ours. **A gate written on RMSE alone would pass
      whether or not anything was ported; this file exists because this
      repository has been burned by exactly that shape.**

  S4  BOTH SEARCHERS: S2 and S3 each run twice, greedy
      (`use_pointwise_searcher=False`) and pointwise (True). The flag is
      applied at the fits' ONE shared der launch, but reach is per-branch
      and the two searchers consume the planes through disjoint kernel
      stacks.

SABOTAGE TABLE (each run by hand, one line each; see PORTING.md):
  A  kernel second-order branch stores `weight` instead of `weight*der2`
     -> S1 fails per-cell, S2 collapses to identical models
  B  host passes `second_order = False` unconditionally
     -> S2 fails on both searchers; S1 does NOT move (it launches the
     kernel directly) -- the two gates cover different layers
  C  second-order branch swaps planes 0 and 1
     -> S1 fails per-cell (placement, not total)
  D  `is_second_order_score_function` returns True for Cosine too
     -> S2 fails (both arms Newton, identical again)
  E  second-order branch stores raw `der2` without folding the weight
     -> S1 fails, and S3 fails: under RMSE plane 0 becomes 1.0 != weight,
     which is exactly the "did they fold the weight in" trap
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import exp, log

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.doc_parallel_boosting import fit
from gbdt.models.oblivious_model import TAdditiveModel
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
    SCORE_FUNCTION_NEWTON_COSINE,
    SCORE_FUNCTION_NEWTON_L2,
)
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    OBJECTIVE_LOGLOSS,
    OBJECTIVE_POISSON,
    OBJECTIVE_RMSE,
    launch_approximate,
)


def splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def frac(i: Int, salt: UInt64) -> Float64:
    """Hashed uniform in [0, 1), distinct per (i, salt)."""
    return Float64(
        splitmix(UInt64(i) * UInt64(2654435761) + salt) >> 11
    ) * (1.0 / 9007199254740992.0)


comptime N = 3000
comptime N_ROWS = 4000
comptime MAX_DEPTH = 4
comptime N_ESTIMATORS = 8


# ======================================================================
# S1: per-cell, both kernels, non-constant der2
# ======================================================================


def _s1(ctx: DeviceContext) raises -> Int:
    var failures = 0

    # hashed, all distinct per row: a permutation cannot pass
    var h_t = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](N)
    var h_p = ctx.enqueue_create_host_buffer[DType.float32](N)
    for i in range(N):
        # Logloss target 0/1; ALSO a valid Poisson count target
        h_t.unsafe_ptr().unsafe_store(
            i, Float32(1.0) if frac(i, 11) > 0.5 else Float32(0.0)
        )
        h_w.unsafe_ptr().unsafe_store(i, Float32(0.5 + 1.5 * frac(i, 22)))
        h_p.unsafe_ptr().unsafe_store(i, Float32(4.0 * frac(i, 33) - 2.0))

    var d_t = ctx.enqueue_create_buffer[DType.float32](N)
    var d_w = ctx.enqueue_create_buffer[DType.float32](N)
    var d_p = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_buf=d_t, src_ptr=h_t.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=h_w.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_p, src_ptr=h_p.unsafe_ptr())
    ctx.synchronize()

    var blocks = (N + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    var stats = ctx.enqueue_create_buffer[DType.float32](2 * N)
    var fv = ctx.enqueue_create_buffer[DType.float32](blocks)
    var mags = ctx.enqueue_create_buffer[DType.float32](2 * blocks)
    var h_s = ctx.enqueue_create_host_buffer[DType.float32](2 * N)

    # the arms: (objective, second_der, label). Poisson covers
    # `pointwise_target_kernel`'s branch, Logloss `cross_entropy_kernel`'s.
    for arm in range(4):
        var objective = OBJECTIVE_LOGLOSS if arm < 2 else OBJECTIVE_POISSON
        var second = (arm % 2) == 1
        if second:
            launch_approximate[False, True](
                ctx, objective, d_t, d_w, Int32(N), d_p, Int32(1),
                Float32(0.0), Float32(0.5), stats, fv, Int32(0),
                mags, Int32(0), blocks,
            )
        else:
            launch_approximate[False](
                ctx, objective, d_t, d_w, Int32(N), d_p, Int32(1),
                Float32(0.0), Float32(0.5), stats, fv, Int32(0),
                mags, Int32(0), blocks,
            )
        ctx.enqueue_copy(dst_ptr=h_s.unsafe_ptr(), src_buf=stats)
        ctx.synchronize()

        var bad0 = 0
        var bad1 = 0
        var exact_w = 0
        for i in range(N):
            var t = Float64(h_t.unsafe_ptr().unsafe_load(i))
            var w = Float64(h_w.unsafe_ptr().unsafe_load(i))
            var v = Float64(h_p.unsafe_ptr().unsafe_load(i))
            var e_der: Float64
            var e_der2: Float64
            if objective == OBJECTIVE_LOGLOSS:
                var ev = exp(v)
                var p = ev / (1.0 + ev)
                if p > 1.0 - 1e-40:
                    p = 1.0 - 1e-40
                if p < 1e-40:
                    p = 1e-40
                var c = 1.0 if t > 0.5 else 0.0
                e_der = w * (c - p)
                e_der2 = w * p * (1.0 - p)
            else:
                e_der = w * (t - exp(v))
                e_der2 = w * exp(v)
            var got0 = Float64(h_s.unsafe_ptr().unsafe_load(i))
            var got1 = Float64(h_s.unsafe_ptr().unsafe_load(N + i))
            var exp0 = e_der2 if second else w
            # plane 0: bit-exact against the stored weight on the
            # first-order arm; 1e-4 relative against the float64 host
            # der2 on the second-order one (the kernel's exp is Float32)
            if second:
                var d0 = got0 - exp0
                if d0 < 0.0:
                    d0 = -d0
                var m0 = exp0 if exp0 > 0.0 else -exp0
                if d0 > 1e-4 * (m0 + 1e-12):
                    bad0 += 1
            else:
                if (
                    h_s.unsafe_ptr().unsafe_load(i)
                    != h_w.unsafe_ptr().unsafe_load(i)
                ):
                    bad0 += 1
                else:
                    exact_w += 1
            var d1 = got1 - e_der
            if d1 < 0.0:
                d1 = -d1
            var m1 = e_der if e_der > 0.0 else -e_der
            if d1 > 1e-4 * (m1 + 1e-12):
                bad1 += 1

        var name = String("Logloss") if objective == OBJECTIVE_LOGLOSS \
            else String("Poisson")
        var mode = String("second-order") if second \
            else String("first-order")
        print(
            "  S1", name, mode, ": plane0 bad", bad0, "/", N,
            ", plane1 bad", bad1, "/", N,
        )
        if bad0 != 0 or bad1 != 0:
            print("  FAIL: S1", name, mode)
            failures += 1
        if not second and exact_w != N:
            print(
                "  FAIL: S1", name,
                "first-order plane0 was not the raw weight bit-exactly",
            )
            failures += 1

    _ = h_t^
    _ = h_w^
    _ = h_p^
    _ = h_s^
    return failures


# ======================================================================
# S2/S3/S4: full fits, both searchers
# ======================================================================


def _diff_models(
    a: TAdditiveModel, b: TAdditiveModel,
    mut split_diffs: Int, mut total_splits: Int, mut leaf_diffs: Int,
) raises:
    """Split-by-split and leaf-by-leaf comparison; a structural
    disagreement (different tree count/shape) counts every cell."""
    if len(a.weak_models) != len(b.weak_models):
        split_diffs += 1
        total_splits += 1
        leaf_diffs += 1
        return
    for tr in range(len(a.weak_models)):
        ref ta = a.weak_models[tr]
        ref tb = b.weak_models[tr]
        if len(ta.structure.splits) != len(tb.structure.splits):
            split_diffs += len(ta.structure.splits) + 1
            total_splits += len(ta.structure.splits) + 1
            continue
        for s in range(len(ta.structure.splits)):
            total_splits += 1
            if (
                ta.structure.splits[s].feature_id
                != tb.structure.splits[s].feature_id
                or ta.structure.splits[s].bin_idx
                != tb.structure.splits[s].bin_idx
            ):
                split_diffs += 1
        if len(ta.leaf_values) != len(tb.leaf_values):
            leaf_diffs += 1
            continue
        for l in range(len(ta.leaf_values)):
            if ta.leaf_values[l] != tb.leaf_values[l]:
                leaf_diffs += 1


def _fit_one(
    ctx: DeviceContext,
    folds: List[Int],
    mut cindex: DeviceBuffer[DType.uint32],
    mut targets: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    objective: Int,
    score_function: Int,
    pointwise: Bool,
) raises -> TAdditiveModel:
    var model = TAdditiveModel()
    _ = fit(
        model, ctx, N_ROWS, folds, MAX_DEPTH, cindex, targets, weights,
        True, N_ESTIMATORS, Float32(0.1), Float32(3.0),
        random_seed=UInt64(7),
        score_function=score_function,
        objective=objective,
        use_pointwise_searcher=pointwise,
    )
    return model^


def _s2_s3_s4(ctx: DeviceContext) raises -> Int:
    var failures = 0

    # binary, half-byte and one-byte features, hashed bins
    var folds: List[Int] = [1, 12, 20, 48, 100, 64, 127]
    var n_features = len(folds)
    var lay = build_layout(folds)

    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        N_ROWS * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](N_ROWS)
    var bins8 = ctx.enqueue_create_buffer[DType.uint8](N_ROWS)
    var host_bins = List[List[Int]]()
    for f in range(n_features):
        ref cf = lay.features[f]
        var col = List[Int]()
        for r in range(N_ROWS):
            var b = Int(
                splitmix(UInt64(r) * UInt64(2654435761) + UInt64(f * 977))
                % UInt64(folds[f] + 1)
            )
            col.append(b)
            hb.unsafe_ptr().unsafe_store(r, UInt8(b))
        host_bins.append(col^)
        ctx.enqueue_copy(dst_buf=bins8, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * N_ROWS),
            cf.mask,
            cf.shift,
            bins8.unsafe_ptr(),
            Int32(N_ROWS),
            cindex.unsafe_ptr(),
            grid_dim=(N_ROWS + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    # binary labels carrying signal on three features plus hashed noise,
    # and NON-UNIFORM hashed weights -- the weights matter: with unit
    # weights `weight * der2` and `weight` still differ under Logloss,
    # but sabotage E (unfolded weight) would be invisible to S3.
    var h_ty = ctx.enqueue_create_host_buffer[DType.float32](N_ROWS)
    var h_wt = ctx.enqueue_create_host_buffer[DType.float32](N_ROWS)
    var h_reg = ctx.enqueue_create_host_buffer[DType.float32](N_ROWS)
    for r in range(N_ROWS):
        var s = 0.0
        if host_bins[4][r] > 50:
            s += 1.6
        if host_bins[2][r] > 10:
            s += 0.9
        if host_bins[6][r] > 60:
            s += 0.7
        s += 1.2 * frac(r, 55) - 1.7
        h_ty.unsafe_ptr().unsafe_store(
            r, Float32(1.0) if s > 0.0 else Float32(0.0)
        )
        h_reg.unsafe_ptr().unsafe_store(r, Float32(s + 0.4 * frac(r, 66)))
        h_wt.unsafe_ptr().unsafe_store(r, Float32(0.5 + 1.5 * frac(r, 77)))

    var d_ty = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_reg = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var d_wt = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    ctx.enqueue_copy(dst_buf=d_ty, src_ptr=h_ty.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_reg, src_ptr=h_reg.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_wt, src_ptr=h_wt.unsafe_ptr())
    ctx.synchronize()

    for arm in range(2):
        var pointwise = arm == 1
        var searcher = String("pointwise") if pointwise \
            else String("greedy")

        # ---- S2: Logloss, the Newton spelling must MOVE the model ----
        var pairs_first: List[Int] = [
            SCORE_FUNCTION_COSINE, SCORE_FUNCTION_L2,
        ]
        var pairs_second: List[Int] = [
            SCORE_FUNCTION_NEWTON_COSINE, SCORE_FUNCTION_NEWTON_L2,
        ]
        var pair_names: List[String] = [
            String("NewtonCosine vs Cosine"), String("NewtonL2 vs L2"),
        ]
        for pr in range(2):
            var m_first = _fit_one(
                ctx, folds, cindex, d_ty, d_wt,
                OBJECTIVE_LOGLOSS, pairs_first[pr], pointwise,
            )
            var m_second = _fit_one(
                ctx, folds, cindex, d_ty, d_wt,
                OBJECTIVE_LOGLOSS, pairs_second[pr], pointwise,
            )
            var sd = 0
            var ts = 0
            var ld = 0
            _diff_models(m_first, m_second, sd, ts, ld)
            print(
                "  S2/S4", searcher, pair_names[pr], "under Logloss:",
                sd, "of", ts, "splits differ,", ld, "leaf values differ",
            )
            if sd == 0:
                print(
                    "  FAIL: S2 (", searcher, ")", pair_names[pr],
                    "grew THE SAME TREES under Logloss -- a Newton fit"
                    " in name only",
                )
                failures += 1

        # ---- S3: RMSE, the negative control, bit-identical -----------
        var r_first = _fit_one(
            ctx, folds, cindex, d_reg, d_wt,
            OBJECTIVE_RMSE, SCORE_FUNCTION_COSINE, pointwise,
        )
        var r_second = _fit_one(
            ctx, folds, cindex, d_reg, d_wt,
            OBJECTIVE_RMSE, SCORE_FUNCTION_NEWTON_COSINE, pointwise,
        )
        var sd = 0
        var ts = 0
        var ld = 0
        _diff_models(r_first, r_second, sd, ts, ld)
        print(
            "  S3", searcher,
            "NewtonCosine vs Cosine under RMSE:", sd, "of", ts,
            "splits differ,", ld, "leaf values differ (0 and 0 required)",
        )
        if sd != 0 or ld != 0:
            print(
                "  FAIL: S3 (", searcher, ") RMSE Newton is not"
                " bit-identical to Cosine; TRmseTarget::Der2 is 1.0f, so"
                " weight * der2 == weight EXACTLY -- any difference means"
                " the port changed something it must not",
            )
            failures += 1

    _ = h_ty^
    _ = h_wt^
    _ = h_reg^
    _ = hb^
    return failures


def main() raises:
    var ctx = DeviceContext()
    var failures = 0
    print("S1: per-cell stat planes against the host expectation")
    failures += _s1(ctx)
    print("S2/S3/S4: full fits, both searchers")
    failures += _s2_s3_s4(ctx)
    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print(
        "secondDerAsWeights: plane 0 is weight * der2 under"
        " NewtonCosine/NewtonL2, the raw weight under Cosine/L2, on both"
        " kernels and both searchers; RMSE negative control bit-identical"
    )
