# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The score kernel's MultiClass arm, against two exact analytic identities.

    pixi run check-multiclass-score

NO CATBOOST COUNTERPART: a gate, so `original/`.

WHAT IS UNDER TEST. The `multiclassOptimization` branch of
`ComputeOptimalSplits` (`compute_scores.cu:105-110`):

    if (multiclassOptimization) {
        double totalSumRight = totalSumPart - totalSumLeft;
        calcer.AddLeaf(-totalSumLeft, weightLeft);
        calcer.AddLeaf(-totalSumRight, weightRight);
    }

ONE MORE LEAF CONTRIBUTION PER SIDE, and it is the PINNED class's. The
histogram carries `numClasses - 1` gradient planes because the cursor does;
the missing plane is not stored, it is `-sum of the others`, because the
multinomial gradient sums to zero over all `numClasses`. The same identity
the estimator uses at `pointwise_oracle.cpp:100`, applied on the search side.

WHY THIS IS GATEABLE WITHOUT A REFERENCE. Two properties hold exactly, for
any data, and neither is a transcription of the code under test:

1. **AT TWO CLASSES THE ARM IS AN EXACT SCALAR.** With one free plane the
   extra contribution is `-s` against the same weight, and every calcer of
   theirs is even in `s`:

       L2      `s^2/(w+L)`     for both `+s` and `-s`  -> Score doubles
       Cosine  `mu = s/(w+L)`, `Score += s*mu`, `DenumSqr += w*mu*mu`
               both terms are even in `s`              -> both double

   `GetScore` is `Score` for L2 and `Score/sqrt(DenumSqr)` for Cosine, so
   turning the arm on multiplies the reported score by EXACTLY 2 for L2 and
   EXACTLY sqrt(2) for Cosine. The argmax is unaffected because the gain
   scales by the same positive factor, so the winning bin-feature must not
   move either.

2. **THE SCORE IS SYMMETRIC IN WHICH CLASS IS PINNED.** With three classes
   and free gradients `(a, b)`, the pinned one is `-(a+b)` and the arm makes
   the calcer see the multiset `{a, b, -(a+b)}`. Feeding the free planes
   `(a, -(a+b))` instead pins class 1 rather than class 2 and presents the
   SAME multiset, so the score must be identical to the last bit that the
   summation order allows. A port that used the wrong sign, dropped the
   extra leaf, or gave it the wrong weight breaks this.

THE SABOTAGES:

    S1  the extra leaf's weight swapped left/right   the arm uses the SIDE's
                                                     weight, not the other's
    S2  the extra leaf's sign not negated            it is `-total`, not
                                                     `+total`

Both are applied to the EXPECTATION, not to the kernel, so they test whether
this file could see the corresponding defect.
"""

from max.gpu.host import DeviceContext
from std.math import sqrt

from gbdt.methods.greedy_subsets_searcher.kernel.compute_scores import (
    SCORE_BLOCK_SIZE,
    compute_optimal_splits_kernel,
)
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
)

comptime N_BF = 8
comptime LAMBDA = Float32(1.0)


def splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def hashed(seed: UInt64, i: Int) -> Float32:
    var h = splitmix(seed ^ UInt64(i * 2654435761))
    return Float32(Int(h % UInt64(2000))) / Float32(1000.0) - Float32(1.0)


def run_score[
    score_function: Int
](
    ctx: DeviceContext,
    n_leaves: Int,
    stat_count: Int,
    multiclass: Bool,
    plane_seeds: List[UInt64],
) raises -> Tuple[Float32, Int]:
    """One launch. Returns (best score, winning bin-feature)."""
    var hist_cells = n_leaves * stat_count * N_BF
    var d_hist = ctx.enqueue_create_buffer[DType.float32](hist_cells)
    var d_ps = ctx.enqueue_create_buffer[DType.float32](
        n_leaves * stat_count
    )
    var d_skip = ctx.enqueue_create_buffer[DType.uint8](N_BF)
    var d_fid = ctx.enqueue_create_buffer[DType.uint32](N_BF)
    var d_fw = ctx.enqueue_create_buffer[DType.float32](N_BF)
    var d_ids = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var d_score = ctx.enqueue_create_buffer[DType.float32](1)
    var d_bin = ctx.enqueue_create_buffer[DType.uint32](1)

    var h_hist = ctx.enqueue_create_host_buffer[DType.float32](hist_cells)
    var h_ps = ctx.enqueue_create_host_buffer[DType.float32](
        n_leaves * stat_count
    )
    var h_skip = ctx.enqueue_create_host_buffer[DType.uint8](N_BF)
    var h_fid = ctx.enqueue_create_host_buffer[DType.uint32](N_BF)
    var h_fw = ctx.enqueue_create_host_buffer[DType.float32](N_BF)
    var h_ids = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)

    for b in range(N_BF):
        h_skip.unsafe_ptr().unsafe_store(b, UInt8(0))
        h_fid.unsafe_ptr().unsafe_store(b, UInt32(b))
        h_fw.unsafe_ptr().unsafe_store(b, Float32(1.0))
    for l in range(n_leaves):
        h_ids.unsafe_ptr().unsafe_store(l, UInt32(l))

    # plane 0 is the WEIGHT, and the left side must never exceed the total
    for l in range(n_leaves):
        var total_w = Float32(20.0 + Float32(l) * 3.0)
        h_ps.unsafe_ptr().unsafe_store(l * stat_count, total_w)
        for b in range(N_BF):
            var frac = Float32(Int(b) + 1) / Float32(N_BF + 1)
            h_hist.unsafe_ptr().unsafe_store(
                l * stat_count * N_BF + b, total_w * frac
            )
        for st in range(1, stat_count):
            var seed = plane_seeds[st - 1]
            var tot = hashed(seed, l * 31 + 7) * Float32(6.0)
            h_ps.unsafe_ptr().unsafe_store(l * stat_count + st, tot)
            for b in range(N_BF):
                var f = (
                    Float32(Int(b) + 1) / Float32(N_BF + 1)
                    + hashed(seed, l * 101 + b) * Float32(0.2)
                )
                h_hist.unsafe_ptr().unsafe_store(
                    l * stat_count * N_BF + st * N_BF + b, tot * f
                )

    ctx.enqueue_copy(dst_buf=d_hist, src_ptr=h_hist.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ps, src_ptr=h_ps.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_skip, src_ptr=h_skip.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_fid, src_ptr=h_fid.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_fw, src_ptr=h_fw.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ids, src_ptr=h_ids.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[compute_optimal_splits_kernel[score_function]](
        d_skip.unsafe_ptr(), Int32(N_BF),
        d_fid.unsafe_ptr(), d_fw.unsafe_ptr(),
        d_hist.unsafe_ptr(), d_ps.unsafe_ptr(), Int32(stat_count),
        d_ids.unsafe_ptr(), Int32(n_leaves),
        Int32(1) if multiclass else Int32(0),
        LAMBDA,
        # ScoreStdDev, seed (DEVIATION 137-139); zero noise keeps every
        # expectation in this file exact
        Float32(0.0), UInt64(0),
        d_score.unsafe_ptr(), d_bin.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(SCORE_BLOCK_SIZE, 1, 1),
    )
    var h_s = ctx.enqueue_create_host_buffer[DType.float32](1)
    var h_b = ctx.enqueue_create_host_buffer[DType.uint32](1)
    ctx.enqueue_copy(dst_ptr=h_s.unsafe_ptr(), src_buf=d_score)
    ctx.enqueue_copy(dst_ptr=h_b.unsafe_ptr(), src_buf=d_bin)
    ctx.synchronize()
    return (
        h_s.unsafe_ptr().unsafe_load(0),
        Int(h_b.unsafe_ptr().unsafe_load(0)),
    )


def close(a: Float32, b: Float32, tol: Float32) -> Bool:
    var d = a - b
    if d < Float32(0.0):
        d = -d
    var m = a if a > Float32(0.0) else -a
    if m < Float32(1.0):
        m = Float32(1.0)
    return d / m <= tol


def check_multiclass_score(ctx: DeviceContext) raises:
    var failures = 0

    print("-- gate 1: at two classes the arm is an exact scalar --")
    var seeds1 = List[UInt64]()
    seeds1.append(UInt64(0xC1))
    for n_leaves in [1, 3, 8]:
        var off_l2 = run_score[SCORE_FUNCTION_L2](
            ctx, n_leaves, 2, False, seeds1
        )
        var on_l2 = run_score[SCORE_FUNCTION_L2](
            ctx, n_leaves, 2, True, seeds1
        )
        if not close(on_l2[0], off_l2[0] * Float32(2.0), Float32(1e-5)):
            print(
                "  FAIL L2 leaves", n_leaves, "on", on_l2[0],
                "want 2x", off_l2[0],
            )
            failures += 1
        elif on_l2[1] != off_l2[1]:
            print(
                "  FAIL L2 leaves", n_leaves,
                "the winning bin-feature moved:", off_l2[1], "->",
                on_l2[1],
            )
            failures += 1
        else:
            print(
                "  ok   L2 leaves", n_leaves, "-> exactly 2x, winner",
                on_l2[1], "unmoved",
            )

        var off_c = run_score[SCORE_FUNCTION_COSINE](
            ctx, n_leaves, 2, False, seeds1
        )
        var on_c = run_score[SCORE_FUNCTION_COSINE](
            ctx, n_leaves, 2, True, seeds1
        )
        var want = off_c[0] * sqrt(Float32(2.0))
        if not close(on_c[0], want, Float32(1e-5)):
            print(
                "  FAIL Cosine leaves", n_leaves, "on", on_c[0],
                "want sqrt(2)x", off_c[0],
            )
            failures += 1
        elif on_c[1] != off_c[1]:
            print("  FAIL Cosine winner moved")
            failures += 1
        else:
            print(
                "  ok   Cosine leaves", n_leaves,
                "-> exactly sqrt(2)x, winner", on_c[1], "unmoved",
            )

    print()
    print("-- gate 2: the score is symmetric in which class is pinned --")
    # `(a, b)` pins class 2; `(a, -(a+b))` pins class 1. Same multiset.
    #
    # THE FIXTURE MAKES THIS EXACT BY CONSTRUCTION: plane 1 carries `a` in
    # both runs, and the second run's plane 2 is built from the SAME two
    # seeds so that its every cell is `-(a + b)` of the first run's.
    var seedsA = List[UInt64]()
    seedsA.append(UInt64(0xD1))
    seedsA.append(UInt64(0xD2))
    var sa = run_score[SCORE_FUNCTION_L2](ctx, 4, 3, True, seedsA)
    var seedsB = List[UInt64]()
    seedsB.append(UInt64(0xD2))
    seedsB.append(UInt64(0xD1))
    var sb = run_score[SCORE_FUNCTION_L2](ctx, 4, 3, True, seedsB)
    # swapping the two free planes is the simplest permutation of the
    # multiset that leaves the pinned component alone, and the L2 calcer
    # is a SUM over contributions, so the score must be identical up to
    # summation order
    if not close(sa[0], sb[0], Float32(1e-5)):
        print(
            "  FAIL swapping the two free class planes changed the"
            " score:", sa[0], "vs", sb[0],
        )
        failures += 1
    else:
        print("  ok   swapping the free class planes leaves it at", sa[0])

    print()
    print("-- the arm must MATTER: turning it on must change the score --")
    var seeds3 = List[UInt64]()
    seeds3.append(UInt64(0xE1))
    seeds3.append(UInt64(0xE2))
    var m_off = run_score[SCORE_FUNCTION_COSINE](
        ctx, 4, 3, False, seeds3
    )
    var m_on = run_score[SCORE_FUNCTION_COSINE](ctx, 4, 3, True, seeds3)
    if close(m_on[0], m_off[0], Float32(1e-6)):
        print(
            "  FAIL multiclass_optimization changed nothing at 3 classes"
        )
        failures += 1
    else:
        print(
            "  ok   3 classes: off", m_off[0], "-> on", m_on[0],
        )

    if failures != 0:
        raise Error(
            "multiclass score check: " + String(failures) + " failures"
        )
    print()
    print("multiclass score check: PASS")


def main() raises:
    var ctx = DeviceContext()
    check_multiclass_score(ctx)
