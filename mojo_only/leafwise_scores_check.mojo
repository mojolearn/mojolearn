"""The LOSSGUIDE score kernel, against a host recomputation and five teeth.

    pixi run check-leafwise-scores

NO CATBOOST COUNTERPART: a gate, so `mojo_only/`.

WHAT IS UNDER TEST. `compute_optimal_split_kernel`, this repository's port of
`ComputeOptimalSplit` (`compute_scores.cu:393-475`) -- the scorer
`EGrowPolicy::Lossguide` runs, and the ONLY one of their three that takes its
leaf ids as scalars rather than out of a buffer.

WHY A HOST RECOMPUTATION IS LEGITIMATE HERE, when
`[[mojotrees-code-not-source-of-truth]]` says a transcription of the code
under test is not a gate. It is not the recomputation that has teeth -- it is
the five properties below, four of which are analytic and none of which a
transcription can satisfy by accident. The recomputation exists to say WHERE
a failure is, and it is written from THEIR file rather than from ours: every
line of `host_best` cites `compute_scores.cu`, and the two were written
against the CUDA, not against each other.

THE FIVE, in the order they run:

  G1  BIT-EXACT against the host walk, for L2 and Cosine, at one leaf and at
      two, and at THREE grid widths. Bit-exact rather than tolerant because
      every thread owns its candidates outright: there is no cross-thread
      accumulation anywhere in this kernel, so a differing last bit means a
      differing OPERATION, never a differing order. This is also the gate
      that would catch a grid-stride bug, which is why the widths vary.

  G2  THE TWO BLOCK ROWS SCORE DIFFERENT LEAVES, and swapping the two scalar
      part ids must swap the two output records. `blockIdx.y == 0 ? partId :
      maybeSecondPartId` (`:404`) is one line and it is the only line that
      distinguishes this kernel from the Depthwise one; a port that read
      `partId` unconditionally passes G1 and fails only here. REACH IS PER
      BRANCH.

  G3  `toZeroPartSplit` GIVES A GAIN OF EXACTLY ZERO (`:463`), not the
      sentinel and not a NaN. Planted by putting a bin's entire weight on
      one side. The check asserts the exact `Float32(0.0)` bit pattern and
      asserts the branch was REACHED (a candidate whose neighbours all score
      nonzero).

  G4  THE POISON RECORD (`:44-49`). With every bin feature marked
      `SkipInScoreCount`, no candidate is ever considered and the record
      must come back with the `(ui32)-1` feature id. Their host raises on
      it; a port that clamped to bin 0 would report a split here.

  G5  THE TIE GOES TO THE SMALLER BIN-FEATURE ID (`:30`). Two candidates are
      planted bit-identical, and the winner must be the lower index at every
      grid width -- if the answer moved with the width, the tie rule would
      be reading thread order instead of bin order.

WHAT THIS FILE DOES NOT CLAIM. It says nothing about whether the LEAF the
searcher picks is right (that is `check-lossguide-tree`), nothing about the
noise term (`score_std_dev` is zero throughout so every expectation is
exact), and nothing about MultiClass on this path, which the lane does not
port.
"""

from max.gpu.host import DeviceContext
from std.math import fma, sqrt
from std.memory import bitcast

from gbdt.methods.greedy_subsets_searcher.kernel.compute_scores import (
    FLOAT32_MAX,
    LEAFWISE_SCORE_BLOCK_SIZE,
    compute_optimal_split_kernel,
)
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime N_BF = 37
"""Deliberately NOT a multiple of the block size, so the grid-stride loop's
`binFeatureId >= binFeatureCount` break (`:414-416`) is live in every run."""

comptime LAMBDA = Float32(1.5)


def splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def hashed(seed: UInt64, i: Int) -> Float32:
    """A planted value, not a random one. See
    `[[uniform-test-data-hides-permutation]]`: a fixture whose cells are all
    alike cannot see a permutation, so every cell here is a function of its
    own coordinates."""
    var h = splitmix(seed ^ UInt64(i * 2654435761))
    return Float32(Int(h % UInt64(2000))) / Float32(1000.0) - Float32(1.0)


@fieldwise_init
struct Fixture(Copyable, Movable):
    """One planted problem: histogram, part stats, skip flags, weights."""

    var n_leaves: Int
    var stat_count: Int
    var hist: List[Float32]
    var part_stats: List[Float32]
    var skip: List[UInt8]
    var feature_id: List[UInt32]
    var feature_weight: List[Float32]


def make_fixture(
    n_leaves: Int,
    stat_count: Int,
    seed: UInt64,
    zero_side_bin: Int = -1,
    tie_bins: Bool = False,
    skip_all: Bool = False,
) raises -> Fixture:
    """The planted problem.

    `zero_side_bin`, when >= 0, puts a bin's ENTIRE leaf weight on the left
    so `weightRight` is exactly 0 and G3's branch fires there and nowhere
    else. `tie_bins` makes bin-features 5 and 19 bit-identical for G5.
    """
    var hist = List[Float32]()
    var part_stats = List[Float32]()
    var skip = List[UInt8]()
    var fid = List[UInt32]()
    var fw = List[Float32]()

    for b in range(N_BF):
        skip.append(UInt8(1) if skip_all else UInt8(0))
        # Two bin-features per FEATURE, so `binFeaturesWeights` being
        # indexed by FEATURE rather than by bin-feature (`:465-466`) is a
        # distinction this fixture can see.
        fid.append(UInt32(b // 2))
    var n_features = (N_BF + 1) // 2
    for f in range(n_features):
        # NOT all ones. A uniform weight plane cannot see the multiply.
        # EXCEPT under `tie_bins`: the tie is between two BIN-features that
        # belong to different FEATURES, and `binFeaturesWeights` is indexed
        # by feature (`:465-466`), so a varying weight plane would break the
        # tie before the argmax ever saw it. Found by G5 reporting a
        # different winner -- the fixture, not the kernel.
        if tie_bins:
            fw.append(Float32(1.0))
        else:
            fw.append(
                Float32(1.0) + hashed(seed ^ UInt64(0xFEA), f) * Float32(0.3)
            )

    for _ in range(n_leaves * stat_count * N_BF):
        hist.append(Float32(0.0))
    for _ in range(n_leaves * stat_count):
        part_stats.append(Float32(0.0))

    for l in range(n_leaves):
        var total_w = Float32(40.0) + Float32(l) * Float32(7.0)
        part_stats[l * stat_count] = total_w
        for b in range(N_BF):
            var src = b
            if tie_bins and b == 19:
                src = 5
            var frac = (
                Float32(Int(src) + 1) / Float32(N_BF + 2)
                + hashed(seed, l * 977 + src) * Float32(0.1)
            )
            var wl = total_w * frac
            if b == zero_side_bin:
                wl = total_w
            hist[l * stat_count * N_BF + b] = wl
        for st in range(1, stat_count):
            var tot = hashed(seed ^ UInt64(st * 31), l * 61 + 3) * Float32(9.0)
            part_stats[l * stat_count + st] = tot
            for b in range(N_BF):
                var src = b
                if tie_bins and b == 19:
                    src = 5
                var f = (
                    Float32(Int(src) + 1) / Float32(N_BF + 2)
                    + hashed(seed ^ UInt64(st * 7), l * 149 + src)
                    * Float32(0.25)
                )
                hist[l * stat_count * N_BF + st * N_BF + b] = tot * f

    return Fixture(
        n_leaves, stat_count, hist^, part_stats^, skip^, fid^, fw^
    )


# ------------------------------------------------------------------ host


def host_add_leaf[
    score_function: Int, use_fma: Bool
](
    sum: Float32,
    weight: Float32,
    mut score: Float32,
    mut denum_sqr: Float32,
):
    """`TCosineScoreCalcer::AddLeaf` (`score_calcers.cuh:152-157`) and
    `TL2ScoreCalcer::AddLeaf` (`:54`). Written from THEIR header, with the
    one negation this port folds in (module docstring of the kernel file).
    `Normalize` is false on every path here, as it is at both leafwise call
    sites (`greedy_search_helper.cpp:487`, `:531`)."""
    comptime if score_function == SCORE_FUNCTION_COSINE:
        var mu = Float32(0.0)
        if weight > Float32(0.0):
            mu = sum / (weight + LAMBDA)
        # THE TWO MULTIPLY-ADDS. `Score += sum * mu` and
        # `DenumSqr += weight * mu * mu` (`score_calcers.cuh:155-156`) are
        # the only contractible seams on this path -- which is why the L2
        # calcer, whose accumulation is a divide followed by an add, cannot
        # see the difference this parameter measures. IDENTITY_PATHS row 9.
        comptime if use_fma:
            score = fma(sum, mu, score)
            denum_sqr = fma(weight * mu, mu, denum_sqr)
        else:
            score += sum * mu
            denum_sqr += weight * mu * mu
    else:
        if weight > Float32(1e-20):
            score += (sum * sum) / (weight + LAMBDA)


def host_best[
    score_function: Int, use_fma: Bool = False
](fx: Fixture, part_id: Int) raises -> Tuple[Float32, Int, Int]:
    """Their `ComputeOptimalSplit` body on the host, in their order.

    Returns `(best gain, winning bin-feature, count of skipped-to-zero
    candidates)`. The third is G3's reach counter.
    """
    var best_gain = -FLOAT32_MAX
    var best_bin = -1
    var zero_hits = 0
    var leaf_base = part_id * fx.stat_count * N_BF

    for b in range(N_BF):
        if fx.skip[b] != UInt8(0):
            continue
        var score = Float32(0.0)
        var denum = Float32(1e-10)
        var score_b = Float32(0.0)
        var denum_b = Float32(1e-10)

        var part_weight = fx.part_stats[part_id * fx.stat_count]
        var wl = max(fx.hist[leaf_base + b], Float32(0.0))
        var wr = max(part_weight - wl, Float32(0.0))
        var to_zero = wl < Float32(1e-20) or wr < Float32(1e-20)
        if to_zero:
            zero_hits += 1

        for st in range(1, fx.stat_count):
            var sl = fx.hist[leaf_base + st * N_BF + b]
            var ps = fx.part_stats[part_id * fx.stat_count + st]
            var sr = ps - sl
            host_add_leaf[score_function, use_fma](sl, wl, score, denum)
            host_add_leaf[score_function, use_fma](sr, wr, score, denum)
            host_add_leaf[score_function, use_fma](ps, part_weight, score_b, denum_b)

        var after = score
        var before = score_b
        comptime if score_function == SCORE_FUNCTION_COSINE:
            after = (
                score / sqrt(denum)
                if denum > Float32(1e-15)
                else -FLOAT32_MAX
            )
            before = (
                score_b / sqrt(denum_b)
                if denum_b > Float32(1e-15)
                else -FLOAT32_MAX
            )

        var gain = Float32(0.0)
        if not to_zero:
            gain = after - before
        gain = gain * fx.feature_weight[Int(fx.feature_id[b])]

        if gain > best_gain:
            best_gain = gain
            best_bin = b

    return (best_gain, best_bin, zero_hits)


# ---------------------------------------------------------------- device


def run_device[
    score_function: Int
](
    ctx: DeviceContext,
    fx: Fixture,
    part_id: Int,
    second_part_id: Int,
    argmax_blocks: Int,
) raises -> List[Tuple[Float32, Int]]:
    """One launch. Returns one `(gain, bin)` per block ROW, already reduced
    across the row's argmax blocks the way their host does
    (`greedy_search_helper.cpp:520-528`)."""
    var n_rows = 2 if part_id != second_part_id else 1
    var n_slots = argmax_blocks * n_rows

    var d_hist = ctx.enqueue_create_buffer[DType.float32](len(fx.hist))
    var d_ps = ctx.enqueue_create_buffer[DType.float32](len(fx.part_stats))
    var d_skip = ctx.enqueue_create_buffer[DType.uint8](N_BF)
    var d_fid = ctx.enqueue_create_buffer[DType.uint32](N_BF)
    var d_fw = ctx.enqueue_create_buffer[DType.float32](
        len(fx.feature_weight)
    )
    var d_score = ctx.enqueue_create_buffer[DType.float32](n_slots)
    var d_bin = ctx.enqueue_create_buffer[DType.uint32](n_slots)

    var h_hist = ctx.enqueue_create_host_buffer[DType.float32](len(fx.hist))
    var h_ps = ctx.enqueue_create_host_buffer[DType.float32](
        len(fx.part_stats)
    )
    var h_skip = ctx.enqueue_create_host_buffer[DType.uint8](N_BF)
    var h_fid = ctx.enqueue_create_host_buffer[DType.uint32](N_BF)
    var h_fw = ctx.enqueue_create_host_buffer[DType.float32](
        len(fx.feature_weight)
    )
    for i in range(len(fx.hist)):
        h_hist.unsafe_ptr().unsafe_store(i, fx.hist[i])
    for i in range(len(fx.part_stats)):
        h_ps.unsafe_ptr().unsafe_store(i, fx.part_stats[i])
    for i in range(N_BF):
        h_skip.unsafe_ptr().unsafe_store(i, fx.skip[i])
        h_fid.unsafe_ptr().unsafe_store(i, fx.feature_id[i])
    for i in range(len(fx.feature_weight)):
        h_fw.unsafe_ptr().unsafe_store(i, fx.feature_weight[i])

    ctx.enqueue_copy(dst_buf=d_hist, src_ptr=h_hist.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ps, src_ptr=h_ps.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_skip, src_ptr=h_skip.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_fid, src_ptr=h_fid.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_fw, src_ptr=h_fw.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[compute_optimal_split_kernel[score_function]](
        d_skip.unsafe_ptr(),
        Int32(N_BF),
        d_fid.unsafe_ptr(),
        d_fw.unsafe_ptr(),
        d_hist.unsafe_ptr(),
        d_ps.unsafe_ptr(),
        Int32(fx.stat_count),
        Int32(part_id),
        Int32(second_part_id),
        Int32(0),
        LAMBDA,
        Float32(0.0),
        UInt64(0),
        d_score.unsafe_ptr(),
        d_bin.unsafe_ptr(),
        grid_dim=(argmax_blocks, n_rows, 1),
        block_dim=(LEAFWISE_SCORE_BLOCK_SIZE, 1, 1),
    )

    var h_s = ctx.enqueue_create_host_buffer[DType.float32](n_slots)
    var h_b = ctx.enqueue_create_host_buffer[DType.uint32](n_slots)
    ctx.enqueue_copy(dst_ptr=h_s.unsafe_ptr(), src_buf=d_score)
    ctx.enqueue_copy(dst_ptr=h_b.unsafe_ptr(), src_buf=d_bin)
    ctx.synchronize()

    var out = List[Tuple[Float32, Int]]()
    for row in range(n_rows):
        # `TBestSplitProperties::operator<` on the host, keyed on Gain then
        # feature then bin (`gpu_structures.h:80-93`); with one gain per
        # block the bin id is the tie-break that survives.
        var best_g = -FLOAT32_MAX
        var best_b = -1
        for i in range(argmax_blocks):
            var slot = row * argmax_blocks + i
            var g = h_s.unsafe_ptr().unsafe_load(slot)
            var bn = Int(h_b.unsafe_ptr().unsafe_load(slot))
            if bn == 0xFFFFFFFF:
                continue
            var take = g > best_g
            if g == best_g and bn < best_b:
                take = True
            if take:
                best_g = g
                best_b = bn
        out.append((best_g, best_b))
    return out^


def bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


# ----------------------------------------------------------------- gates


def check_leafwise_scores(ctx: DeviceContext) raises:
    var failures = 0

    print("-- G1: bit-exact against the host walk --")
    # THE CONTRACTION TALLY. Row 9 of IDENTITY_PATHS is a MEASUREMENT here,
    # not an assumption: every Cosine shape is compared against BOTH host
    # walks -- the naive `score += sum * mu` chain and the explicit
    # `fma(sum, mu, score)` -- and the counts are printed. Whichever the
    # device matches is what its codegen actually did on this kernel, and
    # a shape that matches NEITHER is a third thing and a real defect.
    var cos_naive = 0
    var cos_fused = 0
    var cos_neither = 0
    var cos_total = 0
    for stat_count in [2, 3]:
        for n_leaves in [1, 2, 5]:
            var fx = make_fixture(n_leaves, stat_count, UInt64(0xA5A5))
            for argmax_blocks in [1, 3, 8]:
                var want_l2 = host_best[SCORE_FUNCTION_L2](fx, n_leaves - 1)
                var got_l2 = run_device[SCORE_FUNCTION_L2](
                    ctx, fx, n_leaves - 1, n_leaves - 1, argmax_blocks
                )
                if (
                    bits(got_l2[0][0]) != bits(want_l2[0])
                    or got_l2[0][1] != want_l2[1]
                ):
                    print(
                        "  FAIL L2 stats", stat_count, "leaves", n_leaves,
                        "blocks", argmax_blocks, ": got", got_l2[0][0],
                        "bin", got_l2[0][1], "want", want_l2[0], "bin",
                        want_l2[1],
                    )
                    failures += 1

                var naive = host_best[SCORE_FUNCTION_COSINE, False](
                    fx, n_leaves - 1
                )
                var fused = host_best[SCORE_FUNCTION_COSINE, True](
                    fx, n_leaves - 1
                )
                var got_c = run_device[SCORE_FUNCTION_COSINE](
                    ctx, fx, n_leaves - 1, n_leaves - 1, argmax_blocks
                )
                cos_total += 1
                if bits(got_c[0][0]) == bits(naive[0]):
                    cos_naive += 1
                elif bits(got_c[0][0]) == bits(fused[0]):
                    cos_fused += 1
                else:
                    cos_neither += 1
                    print(
                        "  FAIL Cosine stats", stat_count, "leaves",
                        n_leaves, "blocks", argmax_blocks,
                        "matches NEITHER host walk: got", got_c[0][0],
                        "naive", naive[0], "fused", fused[0],
                    )
                    failures += 1
                # THE WINNING BIN MUST NOT MOVE whichever way the codegen
                # went. A contraction that changed the ARGMAX would be a
                # different tree, not a different last bit.
                if got_c[0][1] != naive[1] or got_c[0][1] != fused[1]:
                    print(
                        "  FAIL Cosine winner moved with the contraction:",
                        "device", got_c[0][1], "naive", naive[1], "fused",
                        fused[1],
                    )
                    failures += 1
    print(
        "  cosine shapes:", cos_total, "-> naive", cos_naive, "/ fused",
        cos_fused, "/ neither", cos_neither,
    )
    # UNDER FAST the codegen may pick either, and MAY PICK DIFFERENTLY PER
    # SHAPE -- that is exactly what "contraction is a codegen decision"
    # means, and it is why row 9 exists. The gate is that it picked one of
    # the two, and that the argmax did not move. Under IDENTICAL there is
    # only one legal answer and the assertion below tightens.
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if cos_naive != 0:
            print(
                "  FAIL under IDENTICAL every cosine shape must be FUSED;",
                cos_naive, "were the naive chain",
            )
            failures += 1
        else:
            print("  ok   IDENTICAL: all", cos_fused, "cosine shapes fused")
    if failures == 0:
        print("  ok   L2 bit-identical at 18 shapes; cosine argmax stable")

    print()
    print("-- G2: the two block rows score DIFFERENT leaves --")
    var fx2 = make_fixture(6, 2, UInt64(0x1234))
    var two = run_device[SCORE_FUNCTION_COSINE](ctx, fx2, 1, 4, 3)
    var w1 = host_best[SCORE_FUNCTION_COSINE](fx2, 1)
    var w4 = host_best[SCORE_FUNCTION_COSINE](fx2, 4)
    if bits(two[0][0]) != bits(w1[0]) or two[0][1] != w1[1]:
        print("  FAIL row 0 is not leaf 1:", two[0][0], "vs", w1[0])
        failures += 1
    elif bits(two[1][0]) != bits(w4[0]) or two[1][1] != w4[1]:
        print("  FAIL row 1 is not leaf 4:", two[1][0], "vs", w4[0])
        failures += 1
    else:
        print("  ok   row 0 = leaf 1, row 1 = leaf 4")

    # THE SWAP. If the kernel ignored `maybeSecondPartId` both rows would be
    # leaf 1 and this would pass by accident, so the swap is what makes the
    # branch reachable: the two records must EXCHANGE.
    var swapped = run_device[SCORE_FUNCTION_COSINE](ctx, fx2, 4, 1, 3)
    if bits(swapped[0][0]) != bits(two[1][0]) or bits(
        swapped[1][0]
    ) != bits(two[0][0]):
        print(
            "  FAIL swapping the part ids did not exchange the rows:",
            swapped[0][0], swapped[1][0], "vs", two[1][0], two[0][0],
        )
        failures += 1
    else:
        print("  ok   swapping the two scalar ids exchanges the two rows")

    # And the degenerate call their launcher makes at the root: equal ids
    # must produce ONE row, not two records for one leaf.
    var root = run_device[SCORE_FUNCTION_COSINE](ctx, fx2, 2, 2, 3)
    if len(root) != 1:
        print("  FAIL equal part ids produced", len(root), "rows, want 1")
        failures += 1
    else:
        print("  ok   equal ids -> one block row (their :570)")

    print()
    print("-- G3: toZeroPartSplit gives a gain of EXACTLY zero --")
    # bin 11 gets the leaf's whole weight on the left, so weightRight == 0.
    var fx3 = make_fixture(3, 2, UInt64(0x77), zero_side_bin=11)
    var h3 = host_best[SCORE_FUNCTION_COSINE](fx3, 2)
    if h3[2] != 1:
        print(
            "  FAIL the branch was not reached exactly once: hits =", h3[2]
        )
        failures += 1
    else:
        # Score the planted bin ALONE, by skipping every other candidate,
        # so the record IS that candidate's gain and nothing else.
        var only = make_fixture(3, 2, UInt64(0x77), zero_side_bin=11)
        for b in range(N_BF):
            if b != 11:
                only.skip[b] = UInt8(1)
        var got = run_device[SCORE_FUNCTION_COSINE](ctx, only, 2, 2, 1)
        if got[0][1] != 11:
            print("  FAIL the planted bin did not win its own run")
            failures += 1
        elif bits(got[0][0]) != bits(Float32(0.0)):
            print(
                "  FAIL zero-side gain is", got[0][0], "bits",
                bits(got[0][0]), "want +0.0",
            )
            failures += 1
        else:
            print("  ok   reached once, and its gain is bitwise +0.0")

    print()
    print("-- G4: every candidate skipped -> the POISON record --")
    var fx4 = make_fixture(2, 2, UInt64(0x99), skip_all=True)
    var poisoned = run_device[SCORE_FUNCTION_L2](ctx, fx4, 0, 0, 2)
    if poisoned[0][1] != -1:
        print("  FAIL a skipped-everything run reported bin", poisoned[0][1])
        failures += 1
    else:
        print("  ok   no record survived the host reduce, as their host requires")

    print()
    print("-- G5: an exact tie goes to the SMALLER bin-feature id --")
    var fx5 = make_fixture(2, 2, UInt64(0x2222), tie_bins=True)
    # bins 5 and 19 are bit-identical by construction; they must also be
    # THE WINNER, or the tie is not the thing being tested. Every other
    # candidate is skipped so the argmax's only decision IS the tie.
    for b in range(N_BF):
        if b != 5 and b != 19:
            fx5.skip[b] = UInt8(1)
    var h5 = host_best[SCORE_FUNCTION_COSINE](fx5, 1)
    var tie_is_live = h5[1] == 5
    if not tie_is_live:
        print(
            "  SKIP the planted tie is not the winning candidate (won by",
            h5[1], ") -- the fixture, not the kernel, needs changing",
        )
        failures += 1
    else:
        var moved = False
        for argmax_blocks in [1, 2, 3, 8, 16]:
            var got = run_device[SCORE_FUNCTION_COSINE](
                ctx, fx5, 1, 1, argmax_blocks
            )
            if got[0][1] != 5:
                print(
                    "  FAIL at", argmax_blocks, "blocks the tie went to",
                    got[0][1], "not 5",
                )
                moved = True
                failures += 1
        if not moved:
            print("  ok   bin 5 wins over bin 19 at all five grid widths")

    if failures != 0:
        raise Error(
            "leafwise scores check: " + String(failures) + " failures"
        )
    print()
    print("leafwise scores check: PASS")


def main() raises:
    var ctx = DeviceContext()
    check_leafwise_scores(ctx)
