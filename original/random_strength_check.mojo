# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`random_strength` reaches both searchers, and does different things there.

Driven through `gbdt.methods.doc_parallel_boosting.fit` end to end. Nothing
below hands a kernel its inputs: a gate that builds the kernel's inputs
cannot check the caller that normally builds them (PORTING.md 115), and the
whole point of this round is the CALLER -- the kernel path was already
written and inert.

## What CatBoost does, which is not one thing

`TCosineScoreCalcer::GetScore()` (`score_calcers.cuh:159-167`) is the only
calcer of their five with a noise term:

    float score = DenumSqr > 1e-15f ? -Score / sqrt(DenumSqr) : FLT_MAX;
    if (ScoreStdDev) {
        ui64 seed = GlobalSeed + FeatureId;
        AdvanceSeed(&seed, 4);
        score += NextNormal(&seed) * ScoreStdDev;
    }

THE SEED IS `GlobalSeed + FeatureId`. Every bin of one feature draws the
SAME normal; two features one id apart draw a KNOWN pair. That is the
structure R4 checks, and it is the reason the term perturbs the ranking
between features and never within a feature.

Then the two searchers diverge:

  * GREEDY (`compute_scores.cu`, the default here). `beforeSplitCalcer` is
    copied from the calcer AFTER `NextFeature` (`:85`), so it carries the
    same `GlobalSeed` and the same `FeatureId` and `GetScore()` gives it
    THE SAME DRAW. `gain = score - scoreBefore` (`:134`) then cancels it.
    The noise is algebraically a no-op on this arm -- CatBoost's own
    behaviour, in all three of their greedy kernels (`:131-134`,
    `:370-375`, `:459-464`).
  * DOC-PARALLEL (`kernel/pointwise_scores.cu`, `use_pointwise_searcher`).
    `gain = (noisyScore - scoreBeforeSplit)` (`:402`) where
    `scoreBeforeSplit` is an UNNOISED HOST SCALAR carried from the previous
    level (`oblivious_tree_doc_parallel_structure_searcher.cpp:60`, `:127`).
    Nothing cancels. This is the arm CatBoost uses for single-target
    symmetric trees, and it is where the option is a model change.

So "does the model move" is the WRONG gate on one arm and the right gate on
the other, and a check that demanded movement everywhere would have been
failed by a correct port.

## Gates

  R1   ANALYTIC IDENTITY. `random_strength=0.0` is bit-identical to the
       parameter never being passed, on BOTH arms, splits and leaf values
       and losses. Any drift means the zero path moved.
  R2   REACH, doc-parallel arm. `random_strength=1.0` must move splits.
       Reports how many of N moved.
  R2G  ANALYTIC IDENTITY, greedy arm. `random_strength=1.0` must produce
       the SAME model as 0.0, because their two calcers cancel. This is a
       stronger statement than "it moved": it is an equality forced by
       their arithmetic, and the sabotage table below breaks the
       cancellation and watches it fail.
  R3   DETERMINISM. Same seed twice, bit-identical. Different seed,
       different -- forced by `GlobalSeed + FeatureId` being a pure
       function, not by any tally of ours.
  R5   THEIR ASYMMETRY BETWEEN CALCERS. `TL2ScoreCalcer` has NO noise
       term (`score_calcers.cuh:40-69`), so under `score_function=L2` a
       non-zero `random_strength` must change NOTHING even on the
       doc-parallel arm, where R2 says it otherwise changes almost
       everything. Without this gate, a port that added the noise to every
       calcer would pass R1-R4. Run at `random_strength = 1e6`, because at
       1.0 a deliberately leaked noise term moved nothing -- see the
       comment at the call.
  R4   THE SEED-PER-FEATURE STRUCTURE, checked by PLACEMENT. Every feature
       of the R4 fixture is the SAME COLUMN, so every candidate's real
       score is bit-identical across features and the argmin is decided
       PURELY by `noise(GlobalSeed + FeatureId)`. The winning FEATURE at
       every level of tree 0 is then predicted on the host from the same
       seed arithmetic and must match. A check whose expected value is the
       same in every cell would verify a total and nothing about placement
       ([[uniform-test-data-hides-permutation]]); this one names the cell.

## Sabotage table, RUN (full results and counts in PORTING.md 142)

    mutation                                        gate that moved
    ----------------------------------------------------------------
    greedy calcer: advance_seed_k(..,4) -> 3        NOTHING
    greedy calcer: seed = global_seed, no FeatureId NOTHING
    greedy: drop the before-calcer's `-= noise`     R2G (44/48 splits)
    run_tree_layout: force score_std_dev = 0        NOTHING
    pointwise calcer: advance_seed_k(..,4) -> 3     R4, all four levels
    pointwise calcer: seed without + FeatureId      R4, three of four
    pointwise: noise leaked into the L2 calcer      R5 (at 1e6; nothing
                                                     at 1.0, which is why
                                                     R5 runs at 1e6)
    boosting loop: force score_std_dev = 0          R2, R3b, R4

THE FIRST FOUR ROWS ARE THE FINDING, NOT A HOLE. Three of them move
nothing because the greedy arm's noise cancels, so no model on that arm can
distinguish a right seed from a wrong one -- and the fourth, which breaks
the cancellation itself, moves 44 of 48 splits, which is what proves the
draw is reaching that kernel at all. The greedy copy of the seed arithmetic
(`+ FeatureId`, the four advances) is therefore UNCHECKABLE through any
model this arm can produce; it is checked by R4 on the pointwise copy of
the same three lines, and by reading. Recorded rather than papered over.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.data.permutation import TRandom
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.gpu_util.kernel.random_gen import advance_seed_k, next_normal_f
from gbdt.methods.doc_parallel_boosting import TAdditiveModel, fit
from gbdt.options.catboost_options import SCORE_FUNCTION_L2


comptime N_ROWS = 3000
comptime N_ITERS = 12
comptime MAX_DEPTH = 4

#: R4's fixture: every feature is the same column, so the only thing that
#: can separate two candidates is the per-feature noise.
comptime R4_FEATURES = 8
comptime R4_FOLDS = 100
comptime R4_DEPTH = 4


struct Fixture(Movable):
    var cindex: DeviceBuffer[DType.uint32]
    var targets: DeviceBuffer[DType.float32]
    var weights: DeviceBuffer[DType.float32]
    var folds: List[Int]

    def __init__(
        out self,
        var cindex: DeviceBuffer[DType.uint32],
        var targets: DeviceBuffer[DType.float32],
        var weights: DeviceBuffer[DType.float32],
        var folds: List[Int],
    ):
        self.cindex = cindex^
        self.targets = targets^
        self.weights = weights^
        self.folds = folds^


def build_fixture(
    ctx: DeviceContext, folds: List[Int], identical: Bool
) raises -> Fixture:
    """One quantized dataset, written straight into a compressed index.

    `identical=True` gives every feature THE SAME bin column, which is what
    makes R4's argmin a pure function of the noise.
    """
    var n_features = len(folds)
    var lay = build_layout(folds)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        N_ROWS * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](N_ROWS)
    var bins = ctx.enqueue_create_buffer[DType.uint8](N_ROWS)
    var host_bin = List[List[Int]]()
    for f in range(n_features):
        ref cf = lay.features[f]
        var col = List[Int]()
        var salt = 0 if identical else f
        for r in range(N_ROWS):
            var x = UInt32(r * 2654435761 + salt * 40503 + 0x2545F491)
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            var v = Int(x % UInt32(folds[f]))
            col.append(v)
            hb.unsafe_ptr().unsafe_store(r, UInt8(v))
        host_bin.append(col^)
        ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * N_ROWS), cf.mask, cf.shift,
            bins.unsafe_ptr(), Int32(N_ROWS), cindex.unsafe_ptr(),
            grid_dim=(N_ROWS + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()
    _ = hb^

    var targets = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var weights = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](N_ROWS)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](N_ROWS)
    for r in range(N_ROWS):
        var y: Float32
        if identical:
            # one column, a non-linear signal on it so the tree has
            # something to find at every depth
            var v = Float32(host_bin[0][r])
            y = v * 0.05 - Float32(3.0)
            if host_bin[0][r] > 60:
                y += 4.0
            if host_bin[0][r] % 7 == 0:
                y -= 2.5
        else:
            y = (
                Float32(host_bin[len(folds) - 1][r]) * 0.4
                - Float32(host_bin[1][r]) * 1.5
                + Float32(host_bin[len(folds) - 2][r]) * 0.6
            )
            if host_bin[len(folds) - 1][r] > 45 and host_bin[1][r] > 6:
                y += 5.0
        ht.unsafe_ptr().unsafe_store(r, y)
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()
    _ = ht^
    _ = hw^
    return Fixture(cindex^, targets^, weights^, folds.copy())


def run_fit(
    ctx: DeviceContext,
    mut fx: Fixture,
    mut model: TAdditiveModel,
    pointwise: Bool,
    random_strength: Float32,
    seed: UInt64,
    n_iters: Int = N_ITERS,
    depth: Int = MAX_DEPTH,
) raises -> List[Float64]:
    """One real fit. Every gate below goes through here."""
    return fit(
        model, ctx, N_ROWS, fx.folds, depth, fx.cindex, fx.targets,
        fx.weights, False, n_iters, Float32(0.3), Float32(3.0), True,
        random_seed=seed,
        use_pointwise_searcher=pointwise,
        random_strength=random_strength,
    )


def count_split_diffs(
    a: TAdditiveModel, b: TAdditiveModel
) raises -> Tuple[Int, Int]:
    """(moved, total) over every split of every tree."""
    var moved = 0
    var total = 0
    var n = len(a.weak_models)
    if len(b.weak_models) < n:
        n = len(b.weak_models)
    for t in range(n):
        ref sa = a.weak_models[t].structure.splits
        ref sb = b.weak_models[t].structure.splits
        var m = len(sa)
        if len(sb) < m:
            m = len(sb)
        for i in range(m):
            total += 1
            if (
                sa[i].feature_id != sb[i].feature_id
                or sa[i].bin_idx != sb[i].bin_idx
                or sa[i].split_type != sb[i].split_type
            ):
                moved += 1
    return (moved, total)


def count_leaf_diffs(a: TAdditiveModel, b: TAdditiveModel) raises -> Int:
    var moved = 0
    var n = len(a.weak_models)
    if len(b.weak_models) < n:
        n = len(b.weak_models)
    for t in range(n):
        ref va = a.weak_models[t].leaf_values
        ref vb = b.weak_models[t].leaf_values
        var m = len(va)
        if len(vb) < m:
            m = len(vb)
        for i in range(m):
            if va[i] != vb[i]:
                moved += 1
    return moved


def count_loss_diffs(a: List[Float64], b: List[Float64]) -> Int:
    var moved = 0
    var n = len(a)
    if len(b) < n:
        n = len(b)
    for i in range(n):
        if a[i] != b[i]:
            moved += 1
    return moved


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    var folds: List[Int] = [1, 12, 9, 20, 32, 48, 100, 64, 127, 96]
    var fx = build_fixture(ctx, folds, False)

    # =================================================================== R1
    # `random_strength=0.0` against the parameter never being passed. Both
    # arms. This is the identity that says the zero path did not move.
    print("R1  the zero path is untouched")
    for arm in range(2):
        var pointwise = arm == 1
        var name = String("pointwise") if pointwise else String("greedy")

        var m_base = TAdditiveModel()
        var l_base = run_fit(
            ctx, fx, m_base, pointwise, Float32(0.0), UInt64(7)
        )
        var m_zero = TAdditiveModel()
        var l_zero = fit(
            m_zero, ctx, N_ROWS, fx.folds, MAX_DEPTH, fx.cindex,
            fx.targets, fx.weights, False, N_ITERS, Float32(0.3),
            Float32(3.0), True,
            random_seed=UInt64(7),
            use_pointwise_searcher=pointwise,
        )
        var d = count_split_diffs(m_base, m_zero)
        var lv = count_leaf_diffs(m_base, m_zero)
        var ls = count_loss_diffs(l_base, l_zero)
        if d[0] != 0 or lv != 0 or ls != 0:
            print(
                "FAIL R1", name, ":", d[0], "of", d[1], "splits,", lv,
                "leaf values and", ls, "losses differ between"
                " random_strength=0.0 and the parameter unset",
            )
            failures += 1
        else:
            print(
                "  ok  R1", name, "--", d[1],
                "splits and all leaf values and losses identical",
            )

    # =================================================================== R2
    # REACH on the doc-parallel arm: the noise survives into the gain there.
    print("R2  the doc-parallel arm's splits move")
    var m_pw0 = TAdditiveModel()
    var l_pw0 = run_fit(ctx, fx, m_pw0, True, Float32(0.0), UInt64(7))
    var m_pw1 = TAdditiveModel()
    var l_pw1 = run_fit(ctx, fx, m_pw1, True, Float32(1.0), UInt64(7))
    var dpw = count_split_diffs(m_pw0, m_pw1)
    var lpw = count_loss_diffs(l_pw0, l_pw1)
    if dpw[0] == 0:
        print(
            "FAIL R2: random_strength=1.0 moved NONE of", dpw[1],
            "splits on the doc-parallel arm. `scoreBeforeSplit` is an"
            " unnoised host scalar there (`pointwise_scores.cu:402`), so"
            " the draw cannot cancel -- a zero here means the value never"
            " reached the calcer.",
        )
        failures += 1
    else:
        print(
            "  ok  R2 --", dpw[0], "of", dpw[1], "splits moved, and",
            lpw, "of", len(l_pw1), "per-iteration losses",
        )

    # ================================================================== R2G
    # ANALYTIC IDENTITY on the greedy arm: their two calcers draw the same
    # normal and the gain subtracts it away.
    print("R2G the greedy arm's noise cancels, by their arithmetic")
    var m_g0 = TAdditiveModel()
    var l_g0 = run_fit(ctx, fx, m_g0, False, Float32(0.0), UInt64(7))
    var m_g1 = TAdditiveModel()
    var l_g1 = run_fit(ctx, fx, m_g1, False, Float32(1.0), UInt64(7))
    var dg = count_split_diffs(m_g0, m_g1)
    var lg = count_loss_diffs(l_g0, l_g1)
    if dg[0] != 0 or lg != 0:
        print(
            "FAIL R2G:", dg[0], "of", dg[1], "splits and", lg,
            "losses moved on the GREEDY arm, where `gain = score -"
            " scoreBefore` (`compute_scores.cu:134`) must cancel a draw"
            " both calcers made from `GlobalSeed + FeatureId`. A"
            " difference here means the before-calcer is not receiving"
            " the same noise the after-calcer is.",
        )
        failures += 1
    else:
        print(
            "  ok  R2G --", dg[1],
            "splits and all losses identical at random_strength 0.0 and"
            " 1.0, which is the cancellation and not an inert knob (R4"
            " proves the draw is live)",
        )

    # =================================================================== R3
    print("R3  determinism of the draw")
    var m_a = TAdditiveModel()
    var l_a = run_fit(ctx, fx, m_a, True, Float32(1.0), UInt64(7))
    var m_b = TAdditiveModel()
    var l_b = run_fit(ctx, fx, m_b, True, Float32(1.0), UInt64(7))
    var same = count_split_diffs(m_a, m_b)
    var same_l = count_loss_diffs(l_a, l_b)
    if same[0] != 0 or same_l != 0 or count_leaf_diffs(m_a, m_b) != 0:
        print(
            "FAIL R3: the same seed twice gave", same[0], "of", same[1],
            "different splits and", same_l, "different losses",
        )
        failures += 1
    else:
        print("  ok  R3a -- same seed twice:", same[1], "splits identical")

    var m_c = TAdditiveModel()
    var l_c = run_fit(ctx, fx, m_c, True, Float32(1.0), UInt64(99))
    var diff = count_split_diffs(m_a, m_c)
    if diff[0] == 0:
        print(
            "FAIL R3: seed 7 and seed 99 produced the same", diff[1],
            "splits, so the seed does not reach the draw",
        )
        failures += 1
    else:
        print(
            "  ok  R3b -- a different seed moves", diff[0], "of", diff[1],
            "splits",
        )

    # =================================================================== R5
    # Their asymmetry: only the cosine calcer has the noise.
    print("R5  the L2 calcer has no noise term, and must not grow one")
    var m_l2a = TAdditiveModel()
    var l_l2a = fit(
        m_l2a, ctx, N_ROWS, fx.folds, MAX_DEPTH, fx.cindex, fx.targets,
        fx.weights, False, N_ITERS, Float32(0.3), Float32(3.0), True,
        random_seed=UInt64(7), use_pointwise_searcher=True,
        score_function=SCORE_FUNCTION_L2, random_strength=Float32(0.0),
    )
    var m_l2b = TAdditiveModel()
    var l_l2b = fit(
        m_l2b, ctx, N_ROWS, fx.folds, MAX_DEPTH, fx.cindex, fx.targets,
        fx.weights, False, N_ITERS, Float32(0.3), Float32(3.0), True,
        random_seed=UInt64(7), use_pointwise_searcher=True,
        score_function=SCORE_FUNCTION_L2,
        # ONE MILLION, not one. `random_strength=1.0` moved nothing here
        # even with the noise deliberately leaked into the L2 calcer, and a
        # sabotage that moves nothing means the gate was decorative: the
        # noise magnitude is calibrated off the TARGET's standard deviation
        # while the L2 score is `sum^2 / (w + lambda)` over thousands of
        # rows, so at strength 1 the leak is far below the gap between
        # candidates. `TL2ScoreCalcer::GetScore()` returns `Score` unchanged
        # at ANY `ScoreStdDev` (`score_calcers.cuh:57-60`), so this gate is
        # entitled to pick a magnitude that a leak could not survive.
        random_strength=Float32(1.0e6),
    )
    var dl2 = count_split_diffs(m_l2a, m_l2b)
    var ll2 = count_loss_diffs(l_l2a, l_l2b)
    if dl2[0] != 0 or ll2 != 0:
        print(
            "FAIL R5:", dl2[0], "of", dl2[1], "splits and", ll2,
            "losses moved under score_function=L2, where"
            " TL2ScoreCalcer::GetScore() returns Score unchanged"
            " (`score_calcers.cuh:57-60`) and no seed is even"
            " constructed. The noise has leaked into a calcer CatBoost"
            " does not put it in.",
        )
        failures += 1
    else:
        print(
            "  ok  R5 --", dl2[1],
            "splits identical under L2 at random_strength 0.0 and 1e6,"
            " on the same arm R2 moves 45 of 46 under Cosine at 1.0",
        )

    # =================================================================== R4
    # PLACEMENT. Every feature is the same column, so `score_b` does not
    # depend on the feature and the argmin over (feature, bin) separates:
    # the winning FEATURE is exactly `argmin_f noise(GlobalSeed + f)`.
    print("R4  the seed is GlobalSeed + FeatureId, checked per feature")
    var r4_folds = List[Int]()
    for _ in range(R4_FEATURES):
        r4_folds.append(R4_FOLDS)
    var fx4 = build_fixture(ctx, r4_folds, True)

    # one policy means one score helper, which is what makes the seed chain
    # below predictable; assert it rather than assume it.
    var lay4 = build_layout(r4_folds)
    var blocks4 = blocks_for(lay4, N_ROWS)
    if len(blocks4) != 1:
        print(
            "FAIL R4: the fixture produced", len(blocks4),
            "policy blocks and therefore that many score helpers, each"
            " drawing its own seed (`pointwise_scores_calcer.h:87`). The"
            " prediction below assumes one.",
        )
        failures += 1

    var r4_seed = UInt64(20260821)
    var m4 = TAdditiveModel()
    var _l4 = run_fit(
        ctx, fx4, m4, True, Float32(1.0), r4_seed, n_iters=1,
        depth=R4_DEPTH,
    )

    # THE SEED CHAIN, host side, mirroring the three `TRandom`s the fit
    # walks: one draw per TREE in the boosting loop, one per LEVEL in
    # `fit_oblivious_tree_structure`, one per HELPER in
    # `ScoresCalcerOnCompressedDataSet.compute_optimal_split`.
    var tree_rand = TRandom(r4_seed)
    var tree_seed = tree_rand.next_uniform_l()
    var level_rand = TRandom(tree_seed)

    ref splits4 = m4.weak_models[0].structure.splits
    if len(splits4) != R4_DEPTH:
        print(
            "FAIL R4: tree 0 has", len(splits4), "splits, expected",
            R4_DEPTH,
        )
        failures += 1

    var r4_bad = 0
    for depth in range(len(splits4)):
        var level_seed = level_rand.next_uniform_l()
        var helper_rand = TRandom(level_seed)
        var global_seed = helper_rand.next_uniform_l()

        # `ui64 seed = GlobalSeed + FeatureId; AdvanceSeed(&seed, 4);
        #  score += NextNormal(&seed) * ScoreStdDev` -- and this arm's
        #  gain is MINIMIZED, so the winner is the SMALLEST draw.
        var best_f = 0
        var best_n = Float32(0.0)
        for f in range(R4_FEATURES):
            var s = advance_seed_k(global_seed + UInt64(f), 4)
            var d = next_normal_f(s)
            if f == 0 or d[0] < best_n:
                best_n = d[0]
                best_f = f
        var got = Int(splits4[depth].feature_id)
        if got != best_f:
            print(
                "FAIL R4: depth", depth, "-- every feature is the same"
                " column, so the winner is decided only by"
                " noise(GlobalSeed + FeatureId). Predicted feature",
                best_f, "(draw", best_n, "), the fit chose", got,
            )
            r4_bad += 1
        else:
            print(
                "  ok  R4 depth", depth, "-- predicted feature", best_f,
                "from draw", best_n, "and the fit chose it",
            )
    if r4_bad != 0:
        failures += 1

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print(
        "random_strength: R1 R2 R2G R3 R4 R5 pass -- honored on both arms,"
        " visible on the doc-parallel one, cancelling on the greedy one"
    )
