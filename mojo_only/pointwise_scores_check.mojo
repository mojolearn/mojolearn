"""Gate for `gbdt/methods/kernel/pointwise_scores.mojo`.

The POINTWISE SCORER: three split kernels, five score calcers, the
gather-by-leaves transpose, and the partition-statistics reduce. Nothing in
this repository calls any of it yet (`UNWIRED.md`), so this file is the only
thing standing between the port and a wrong tree the day it is wired.

WHY EVERY GATE IS PER CELL AND NEVER A TOTAL
--------------------------------------------
Each of the three kernels reduces a whole block to ONE `TBestSplitProperties`
record. Comparing that record against a host argmin is a four-number check
over hundreds of candidates -- a permutation of the histogram, a swapped
stat axis, or a calcer that got one leaf wrong will usually still produce
SOME winner, and often the right one.

So the score gates do not compare the winner. They compare EVERY CANDIDATE'S
SCORE, one launch per candidate, by setting `SkipInScoreCount` on all but
one bin feature. `bf[b].SkipInScoreCount` is theirs (`pointwise_scores.cu:65`,
`:236`, `:342`) and it makes the block's answer be exactly candidate `b`'s
score and gain. That turns a 4-number check into a `2 * binFeatureCount`
number check per configuration, and it gates the skip flag at the same time.
The block ARGMIN then gets its own gate (A1/A2) on a fixture built to make
the reduction decide something.

THE FIXTURE IS HASHED AND SCATTERED, NEVER UNIFORM
---------------------------------------------------
Every bin of every feature in every (leaf, fold) gets its own hashed
increment, so no two histogram cells hold the same number by construction.
[[uniform-test-data-hides-permutation]]: a fixture whose cells all agree
verifies the total and nothing about placement, and this repository has
shipped that mistake twice.

The histogram is built the way the real one arrives -- per-bin increments
PREFIX-SCANNED within a feature, with each feature's increments summing to
the same partition total. That is what `ScanHistogramsImpl` leaves behind,
and it is what makes `weightLeft <= part.Weight` hold, so the derived right
child is non-negative everywhere except where a gate deliberately breaks it.

MAGNITUDES ARE SMALL ON PURPOSE. Every planted value is a small integer;
partition totals are under 200 and the largest partition sum in the reduce
gate is 20000 rows x 63 = 1.26e6, comfortably inside float32's exact-integer
range of 2^24 = 16.7e6. A previous fixture in this tree put weights near 1e5
over thousands of rows, pushed one cell past 2^24, and reported a kernel bug
that was the check's own rounding.

TOLERANCE. The reduce gate (R1-R4) compares EXACTLY: its sums are integers
below 2^24, so summation order cannot move them. The gather gate (G1) and
the loader-equivalence gate (L1) compare EXACTLY: they move bits, they do
not compute. The score gates divide and take square roots, and the device
may contract a multiply-add into an FMA where the host does not, so they
compare with `|got - want| <= 1e-5 * (|want| + 1)` and PRINT the largest
relative discrepancy seen. Every sabotage below moves cells by factors, not
by ulps; the printed margin is what says so.

GATES
-----
  S1  single-fold, DIRECT loader, ALL FIVE CALCERS, per candidate: Score,
      Gain, FeatureId and BinId of every bin feature against a host
      transcription of `score_calcers.cuh`. `catFeaturesWeights` and
      `binFeaturesWeights` are distinct per feature and neither is 1, and
      `scoreBeforeSplit` is non-zero, so an omitted multiply moves cells.
  S2  the `TL2ScoreCalcer` META EXPONENT, both arms. `metaFrequency = 0`
      takes `MetaExponent == 1.0` (plain L2); `metaFrequency = 1` takes the
      `pow`/`copysign` arm. Same per-candidate comparison. Without this the
      `pow` arm is dead code that no configuration reaches.
  M1  `meta_exponent_draw` itself, on the host: the frequency-0 arm returns
      1.0, the frequency-1 arm returns `metaL2Exponent`, and BOTH advance
      the seed. The advance is what makes their cosine arm's seed different
      from their L2 arm's.
  L1  the GATHERED-BY-LEAVES loader equals the DIRECT loader, per candidate,
      EXACTLY. The gathered histogram is produced by the ported
      `gather_histogram_by_leaves`, so this gates the transpose and the
      loader that reads it as one chain. Both sides of the
      `gatheredByLeaves` switch are named checks (PORTING_RULES 8).
  G1  the gather kernel per cell, at HIST_COUNT 1, 2 and 4, at foldCount 4
      AND foldCount 1. Four is what makes the fold axis observable: at
      foldCount 1 the output index loses its fold term entirely and a
      kernel that dropped it passes.
  A1  the block ARGMIN over 300 bin features and 2 blocks, so the grid
      stride runs twice in block 0 and once in block 1. Per block, all four
      fields.
  A2  the TIE-BREAK. Block 0's winning column is copied, bit for bit and
      with its FeatureId, to a larger index in the same chunk and to a
      SMALLER-tid index in the strided second chunk, making a three-way
      exact tie for the minimum. Their rule is smallest BIN-FEATURE INDEX,
      not lowest tid, and the two answers differ here by construction.
  D1  the DYNAMIC solar kernel (foldCount 6), per candidate. The fixture
      makes leaf 0's TEST folds tiny so `leftTotalWeight > 2` is false for
      some candidates and true for others; the check prints how many of
      each, because a cut that never fires is a cut that is not gated.
  D2  the DYNAMIC cosine kernel, per candidate, with `normalize` FALSE and
      TRUE -- both sides of a runtime switch, and the fixture's lambda is
      1.5 so `l2 * weight` is nowhere equal to `l2`.
  D3  `scoreStdDev`, the `random_strength` noise, on the dynamic cosine
      path: the noisy run must match a host replay of `AdvanceSeed(seed, 4)`
      + `NextNormal`, AND must differ from the noiseless run in at least one
      cell. The second half is what stops "reached but inert".
  D4  the `max(weightTestRight, 0)` clamp in the dynamic cosine kernel, on a
      fixture where one TEST fold's partition weight is smaller than the
      histogram's left weight. See the note on the single-fold clamp below.
  R1  `PartitionUpdateImpl` with all three inputs present: Weight, Sum and
      Count per partition, EXACT, over four partitions at distinct offsets.
      One partition has 20000 rows, which is what reaches `ComputeSum`'s
      16-wide unrolled loop at their block size of 1024 (it needs more than
      15 * 1024 elements); another has 500, which reaches only the tail
      loop; another has 1; another has 0.
  R2  `counts == nullptr` -> Count is the partition SIZE, not zero. This is
      the asymmetric arm of their three null checks and the one a reader
      most easily "tidies".
  R3  `weights == nullptr` -> Weight is 0.
  R4  `target == nullptr` -> Sum is 0.

A CLAMP THAT CANNOT BE GATED, REPORTED RATHER THAN FAKED
---------------------------------------------------------
`FindOptimalSplitSingleFoldImpl` clamps the derived right weight,
`max(part.Weight - weightLeft, 0.0f)` (`:260`). THAT CLAMP IS INERT FOR ALL
FIVE CALCERS and no fixture can move it. Every calcer guards its own body on
the weight: `weight > 1e-20f` (Solar, L2), `weight > 0` (LOOL2, SatL2),
`weight > 0.0f` for `mu` in Cosine -- and a negative weight fails all of
them exactly as a zero weight does. The cosine calcer's one unguarded line,
`DenumSqr += weight * mu * mu`, multiplies by a `mu` that is already 0 on
that branch. So clamped and unclamped produce identical scores.

This was MEASURED, not reasoned: deleting the `max` from the single-fold
kernel leaves S1 and S2 green at every one of the five calcers, with zero
cells moved. It is recorded here rather than dropped, because "the sabotage
did not move the gate" is the finding.

The dynamic cosine kernel's clamp on `weightTestRight` (`:369`) IS live --
it multiplies a `mu` derived from a DIFFERENT fold, which can be non-zero
while the test fold's derived weight is negative -- and D4 gates it.

SABOTAGE: EVERY GATE WAS PROVED ABLE TO FAIL
---------------------------------------------
One sabotage per MECHANISM, applied to the LIBRARY file, run, and reverted.
No defect was ever compiled into shipped code; the only expectation this
file computes wrongly on purpose is D4's unclamped one, which lives here.

    sabotage in `pointwise_scores.mojo`                gate     cells
    ---------------------------------------------------------------------
    DIRECT loader stat axis swapped (w <-> t)          S1,S2,A  140,28,7
    GATHERED loader index transposed                   L1       140
    gather output index drops the fold term            G1       3380/5180
    tie-break comparison reversed                      A2       1
    tie-break clause deleted (lowest tid wins)         A2       1
    argmin keeps the LARGEST gain                      all 8    140..21
    `SkipInScoreCount` ignored                         S1,S2,L1 135,27,135
    solar `totalWeight > 2` cut becomes `> 0`          D1       7
    `counts == null` arm returns 0, not the SIZE       R2       3
    ComputeSum's unrolled loop strides wrong           R1       4
    `AdvanceSeed(seed, 4)` becomes 3                   D3       21
    dynamic cosine ignores `normalize`                 D2       21
    L2 `MetaExponent` pow arm never taken              S2       28
    partition `Offset` read as `Size`                  R1-R4    12
    dynamic cosine drops max(weightTestRight, 0)       D2,D3,D4 2,1,11
    solar (estimate, test) fold pair swapped           D1       21
    `GetHistogramOffset` used for the PARTITION
        offset -- dynamic cosine                       D2,D3    30,15
        offset -- dynamic solar                        D1       15

TWO SABOTAGES DID NOT MOVE ANY GATE. Both are reported rather than dropped,
because in each case the reason is that the line is mathematically inert at
that configuration, not that the gate is blind:

  * `max(part.Weight - weightLeft, 0)` deleted from the SINGLE-FOLD kernel:
    every calcer guards on the weight, so a negative weight and a zero
    weight take the same branch. See the section above.
  * `GetDataPartitionOffset` replaced by `GetHistogramOffset` in the
    SINGLE-FOLD kernel: at `FoldCount == 1` both are `partId`, by their own
    definition. The DYNAMIC kernels do catch it -- but only because this
    file's dynamic fixture runs at foldCount SIX. At foldCount 4 the same
    sabotage was measured GREEN on every gate, since stripe and fold count
    coincide at every power of two. That is why D1-D3 are not run at 4.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import copysign, log, sqrt

from gbdt.gpu_util.kernel.random_gen import advance_seed_k, next_normal_f
from gbdt.methods.kernel.pointwise_scores import (
    FLOAT32_MAX,
    find_optimal_split,
    gather_histogram_by_leaves,
    meta_exponent_draw,
    update_partition_props,
)
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
    SCORE_FUNCTION_LOO_L2,
    SCORE_FUNCTION_SAT_L2,
    SCORE_FUNCTION_SOLAR_L2,
)


# ===========================================================================
# hashing: every planted cell gets its own value
# ===========================================================================


def _hash(a: Int, b: Int, c: Int) -> Int:
    """A scattered, reproducible per-cell value. Never a constant."""
    var x = (
        UInt64(a + 1) * UInt64(73856093)
    ) ^ (UInt64(b + 1) * UInt64(19349663)) ^ (UInt64(c + 1) * UInt64(83492791))
    x = (x ^ (x >> 29)) * UInt64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> 31)
    return Int(x % UInt64(1000000))


# ===========================================================================
# the fixture
# ===========================================================================


@fieldwise_init
struct Fixture(Copyable, Movable):
    var n_features: Int
    var folds: List[Int]
    var first_fold: List[Int]
    var b_count: Int
    var p_count: Int
    var fold_count: Int
    var stripe: Int
    var bf: List[UInt32]
    var parts: List[Float32]
    var hist: List[Float32]
    var cat_w: List[Float32]
    var bin_w: List[Float32]


def _stripe_of(fold_count: Int) -> Int:
    """`1 << (ui32)ceil(log2((float)FoldCount))`, their data-partition
    stride (`split_properties_helpers.cuh:78-81`)."""
    var s = 1
    while s < fold_count:
        s *= 2
    return s


def _build(
    folds: List[Int],
    p_count: Int,
    fold_count: Int,
    tiny_test_folds: Bool,
    salt: Int,
) -> Fixture:
    """One histogram, built the way `ScanHistogramsImpl` leaves it.

    For every (leaf, fold) there is ONE partition total, and every feature's
    bins are a hashed composition of that same total, prefix-scanned. That
    is the invariant a real histogram has and the reason `weightLeft` never
    exceeds `part.Weight` here unless a gate breaks it on purpose.

    `tiny_test_folds` shrinks leaf 0's ODD folds -- the TEST halves of the
    (estimate, test) pairs the dynamic kernels read -- to single digits, so
    that the solar kernel's `leftTotalWeight > 2` cut is false for some
    candidates and true for others.
    """
    var n_features = len(folds)
    var first_fold = List[Int]()
    var cursor = 0
    for f in range(n_features):
        first_fold.append(cursor)
        cursor += folds[f]
    var b_count = cursor
    var stripe = _stripe_of(fold_count)

    # `TCBinFeature[b] = {FeatureId, BinId, SkipInScoreCount}`
    var bf = List[UInt32]()
    for f in range(n_features):
        for j in range(folds[f]):
            bf.append(UInt32(f))
            bf.append(UInt32(j))
            bf.append(UInt32(0))

    var parts = List[Float32]()
    for _ in range(p_count * stripe * 3):
        parts.append(Float32(0.0))
    var hist = List[Float32]()
    for _ in range(p_count * fold_count * b_count * 2):
        hist.append(Float32(0.0))

    for leaf in range(p_count):
        for fold in range(fold_count):
            var small = tiny_test_folds and leaf == 0 and (fold % 2) == 1
            var tot_w = Float32(60 + 7 * leaf + 3 * fold)
            if small:
                tot_w = Float32(3)
            var tot_t = Float32(7 + 3 * leaf - 2 * fold)

            var po = 3 * (leaf * stripe + fold)
            parts[po + 0] = tot_w
            parts[po + 1] = tot_t
            parts[po + 2] = tot_w

            var hbase = (leaf * fold_count + fold) * b_count * 2

            for f in range(n_features):
                var n = folds[f]
                var vw = List[Float32]()
                var vt = List[Float32]()
                var sw = Float32(0.0)
                var st = Float32(0.0)
                for j in range(n - 1):
                    var a = Float32(
                        1 + _hash(leaf + salt, fold, f * 31 + j) % 5
                    )
                    if small:
                        a = Float32(_hash(leaf + salt, fold, f * 31 + j) % 2)
                    var b = Float32(
                        _hash(leaf + salt, fold + 50, f * 31 + j) % 11 - 5
                    )
                    vw.append(a)
                    vt.append(b)
                    sw += a
                    st += b
                vw.append(tot_w - sw)
                vt.append(tot_t - st)

                var cw = Float32(0.0)
                var ct = Float32(0.0)
                for j in range(n):
                    cw += vw[j]
                    ct += vt[j]
                    var b_idx = first_fold[f] + j
                    hist[hbase + b_idx * 2 + 0] = cw
                    hist[hbase + b_idx * 2 + 1] = ct

    # per-FEATURE weights, all distinct and none equal to 1
    var cat_w = List[Float32]()
    var bin_w = List[Float32]()
    for f in range(n_features):
        cat_w.append(Float32(1.0) + Float32(f + 1) * Float32(0.25))
        bin_w.append(Float32(0.5) + Float32((f * 3) % 7) * Float32(0.125))

    return Fixture(
        n_features, folds.copy(), first_fold^, b_count, p_count, fold_count,
        stripe, bf^, parts^, hist^, cat_w^, bin_w^,
    )


# ===========================================================================
# host transcription of `score_calcers.cuh`
# ===========================================================================


def _host_score(
    sf: Int,
    lam: Float32,
    meta: Float32,
    normalize: Bool,
    std_dev: Float32,
    seed: UInt64,
    feature_id: UInt32,
    t_sum: List[Float32],
    t_weight: List[Float32],
) -> Float32:
    """The five `AddLeaf` bodies and the five `GetScore` bodies, folded over
    the leaf terms IN THE ORDER the kernel feeds them (left then right, leaf
    by leaf), because float addition is not associative.

    Written from `score_calcers.cuh`, with a RUNTIME switch where the port
    has a comptime one -- a different shape on purpose, so that a
    transcription error is unlikely to be the same transcription error.
    """
    var score = Float32(0.0)
    var denum = Float32(0.0)
    if sf == SCORE_FUNCTION_COSINE:
        denum = Float32(1e-10)

    for k in range(len(t_sum)):
        var s = t_sum[k]
        var w = t_weight[k]

        if sf == SCORE_FUNCTION_SOLAR_L2:
            if w > Float32(1e-20):
                score += (
                    (-s * s)
                    * (Float32(1.0) + Float32(2.0) * log(w + Float32(1.0)))
                ) / w
        elif sf == SCORE_FUNCTION_L2:
            var leaf_score = Float32(0.0)
            if w > Float32(1e-20):
                leaf_score = (-s * s) / (w + lam)
            if meta == Float32(1.0) or leaf_score == Float32(0.0):
                score += leaf_score
            else:
                score += (
                    copysign((abs(leaf_score) / w) ** meta, leaf_score) * w
                )
        elif sf == SCORE_FUNCTION_LOO_L2:
            var adj = Float32(0.0)
            if w > Float32(1.0):
                adj = w / (w - Float32(1.0))
            adj = adj * adj
            if w > Float32(0.0):
                score += adj * (-s * s) / w
        elif sf == SCORE_FUNCTION_SAT_L2:
            var adj2 = Float32(0.0)
            if w > Float32(2.0):
                adj2 = (
                    w
                    * (w - Float32(2.0))
                    / (w * w - Float32(3.0) * w + Float32(1.0))
                )
            if w > Float32(0.0):
                score += adj2 * ((-s * s) / w)
        else:
            var l = lam
            if normalize:
                l = lam * w
            var mu = Float32(0.0)
            if w > Float32(0.0):
                mu = s / (w + l)
            score += s * mu
            denum += w * mu * mu

    if sf != SCORE_FUNCTION_COSINE:
        return score

    var out = FLOAT32_MAX
    if denum > Float32(1e-15):
        out = -score / sqrt(denum)
    if std_dev != Float32(0.0):
        var s2 = advance_seed_k(seed + UInt64(feature_id), 4)
        var d = next_normal_f(s2)
        out += d[0] * std_dev
    return out


def _host_single_fold(
    fx: Fixture,
    b: Int,
    gathered: Bool,
    sf: Int,
    lam: Float32,
    meta: Float32,
    normalize: Bool,
    std_dev: Float32,
    seed: UInt64,
    score_before: Float32,
) -> Tuple[Float32, Float32]:
    """`FindOptimalSplitSingleFoldImpl` for ONE candidate, on the host.

    The histogram address is written from the LAYOUT DEFINITION in the port's
    docstring rather than copied from the kernel:

        cell (leaf, b, stat) = hist[((leaf * binFeatureCount) + b) * 2 + stat]
        stat 0 = WEIGHT, stat 1 = TARGET

    at `foldCount == 1`, where `GetHistogramOffset(leaf, 0) == leaf`.
    """
    var t_sum = List[Float32]()
    var t_w = List[Float32]()
    for leaf in range(fx.p_count):
        var part_w = fx.parts[3 * leaf + 0]
        var part_s = fx.parts[3 * leaf + 1]
        var wl = fx.hist[(leaf * fx.b_count + b) * 2 + 0]
        var sl = fx.hist[(leaf * fx.b_count + b) * 2 + 1]
        var wr = max(part_w - wl, Float32(0.0))
        var sr = part_s - sl
        t_sum.append(sl)
        t_w.append(wl)
        t_sum.append(sr)
        t_w.append(wr)

    var feature_id = Int(fx.bf[3 * b])
    var score = _host_score(
        sf, lam, meta, normalize, std_dev, seed, UInt32(feature_id), t_sum, t_w
    )
    score *= fx.cat_w[feature_id]
    var gain = score - score_before
    gain *= fx.bin_w[feature_id]
    return (score, gain)


def _host_dynamic_solar(
    fx: Fixture, b: Int, score_before: Float32
) -> Tuple[Float32, Float32, Int, Int]:
    """`FindOptimalSplitSolarImpl` for ONE candidate, on the host.

    Also returns how many times the `> 2` cut fired and how many times it
    did not, so the caller can prove both sides were reached.
    """
    var score = Float32(0.0)
    var fired = 0
    var skipped = 0
    for leaf in range(fx.p_count):
        var ltw = Float32(0.0)
        var rtw = Float32(0.0)
        var ls = Float32(0.0)
        var rs = Float32(0.0)
        var fold = 0
        while fold < fx.fold_count:
            var lo = 3 * (leaf * fx.stripe + fold)
            var to = 3 * (leaf * fx.stripe + fold + 1)
            var plw = fx.parts[lo + 0]
            var pls = fx.parts[lo + 1]
            var ptw = fx.parts[to + 0]
            var pts = fx.parts[to + 1]

            var he = ((leaf * fx.fold_count + fold) * fx.b_count + b) * 2
            var ht = (
                (leaf * fx.fold_count + fold + 1) * fx.b_count + b
            ) * 2

            var wel = fx.hist[he + 0]
            var wer = plw - wel
            var sel = fx.hist[he + 1]
            var ser = pls - sel
            var wtl = fx.hist[ht + 0]
            var wtr = ptw - wtl
            var stl = fx.hist[ht + 1]
            var str_ = pts - stl

            var mul = Float32(0.0)
            if wel > Float32(0.0):
                mul = sel / (wel + Float32(1e-15))
            ls += Float32(-2.0) * mul * stl + wtl * mul * mul
            ltw += wtl

            var mur = Float32(0.0)
            if wer > Float32(0.0):
                mur = ser / (wer + Float32(1e-15))
            rtw += wtr
            rs += Float32(-2.0) * mur * str_ + wtr * mur * mur

            fold += 2

        if ltw > Float32(2.0):
            score += ls * (
                Float32(1.0) + Float32(2.0) * log(ltw + Float32(1.0))
            )
            fired += 1
        else:
            skipped += 1
        if rtw > Float32(2.0):
            score += rs * (
                Float32(1.0) + Float32(2.0) * log(rtw + Float32(1.0))
            )
            fired += 1
        else:
            skipped += 1

    var feature_id = Int(fx.bf[3 * b])
    score *= fx.cat_w[feature_id]
    var gain = (score - score_before) * fx.bin_w[feature_id]
    return (score, gain, fired, skipped)


def _host_dynamic_cosine(
    fx: Fixture,
    b: Int,
    score_before: Float32,
    l2: Float32,
    normalize: Bool,
    std_dev: Float32,
    seed: UInt64,
    clamp: Bool,
) -> Tuple[Float32, Float32]:
    """`FindOptimalSplitCosineImpl` for ONE candidate, on the host.

    `clamp` selects the WRONG expectation on purpose for gate D4: passing
    `False` computes what a kernel without `max(weightTestRight, 0)` would
    produce, and D4 requires the check to REJECT it. The sabotage lives
    here, in the check, never in the library.
    """
    var score = Float32(0.0)
    var denum = Float32(1e-20)
    for leaf in range(fx.p_count):
        var fold = 0
        while fold < fx.fold_count:
            var lo = 3 * (leaf * fx.stripe + fold)
            var to = 3 * (leaf * fx.stripe + fold + 1)
            var plw = fx.parts[lo + 0]
            var pls = fx.parts[lo + 1]
            var ptw = fx.parts[to + 0]
            var pts = fx.parts[to + 1]

            var he = ((leaf * fx.fold_count + fold) * fx.b_count + b) * 2
            var ht = (
                (leaf * fx.fold_count + fold + 1) * fx.b_count + b
            ) * 2

            var wel = fx.hist[he + 0]
            var wer = plw - wel
            if clamp:
                wer = max(wer, Float32(0.0))
            var sel = fx.hist[he + 1]
            var ser = pls - sel
            var wtl = fx.hist[ht + 0]
            var wtr = ptw - wtl
            if clamp:
                wtr = max(wtr, Float32(0.0))
            var stl = fx.hist[ht + 1]
            var str_ = pts - stl

            var laml = l2
            if normalize:
                laml = l2 * wel
            var mul = Float32(0.0)
            if wel > Float32(0.0):
                mul = sel / (wel + laml)
            score += stl * mul
            denum += wtl * mul * mul

            var lamr = l2
            if normalize:
                lamr = l2 * wer
            var mur = Float32(0.0)
            if wer > Float32(0.0):
                mur = ser / (wer + lamr)
            score += str_ * mur
            denum += wtr * mur * mur

            fold += 2

    var out = FLOAT32_MAX
    if denum > Float32(1e-15):
        out = -score / sqrt(denum)

    var feature_id = Int(fx.bf[3 * b])
    out *= fx.cat_w[feature_id]
    if std_dev != Float32(0.0):
        var s2 = advance_seed_k(seed + UInt64(UInt32(feature_id)), 4)
        var d = next_normal_f(s2)
        out += d[0] * std_dev
    var gain = (out - score_before) * fx.bin_w[feature_id]
    return (out, gain)


# ===========================================================================
# comparison
# ===========================================================================


@fieldwise_init
struct Tally(Copyable, ImplicitlyCopyable, Movable):
    var bad: Int
    var worst: Float32


def _near(got: Float32, want: Float32) -> Bool:
    return abs(got - want) <= Float32(1e-5) * (abs(want) + Float32(1.0))


def _rel(got: Float32, want: Float32) -> Float32:
    return abs(got - want) / (abs(want) + Float32(1.0))


# ===========================================================================
# the per-candidate sweep
# ===========================================================================


def _sweep(
    ctx: DeviceContext,
    fx: Fixture,
    gathered: Bool,
    sf: Int,
    l2: Float32,
    meta_l2_exponent: Float32,
    meta_frequency: Float32,
    normalize: Bool,
    std_dev: Float32,
    seed: UInt64,
    score_before: Float32,
    want_score: List[Float32],
    want_gain: List[Float32],
    label: String,
    quiet: Bool = False,
) raises -> Tally:
    """One launch per bin feature, everything but that one skipped.

    Runs through `find_optimal_split`, the real dispatch, so the switch at
    `pointwise_scores.cu:530` is exercised rather than bypassed.
    """
    var b_count = fx.b_count
    var hist_len = len(fx.hist)

    var d_bf = ctx.enqueue_create_buffer[DType.uint32](3 * b_count)
    var d_cat = ctx.enqueue_create_buffer[DType.float32](fx.n_features)
    var d_binw = ctx.enqueue_create_buffer[DType.float32](fx.n_features)
    var d_hist = ctx.enqueue_create_buffer[DType.float32](hist_len)
    var d_parts = ctx.enqueue_create_buffer[DType.float32](len(fx.parts))
    var d_rid = ctx.enqueue_create_buffer[DType.uint32](2)
    var d_rsc = ctx.enqueue_create_buffer[DType.float32](2)
    var h_rid = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var h_rsc = ctx.enqueue_create_host_buffer[DType.float32](2)

    ctx.enqueue_copy(dst_buf=d_cat, src_ptr=fx.cat_w.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_binw, src_ptr=fx.bin_w.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_parts, src_ptr=fx.parts.unsafe_ptr())

    # the GATHERED arm reads the transposed copy the ported gather makes
    var d_src = ctx.enqueue_create_buffer[DType.float32](hist_len)
    ctx.enqueue_copy(dst_buf=d_src, src_ptr=fx.hist.unsafe_ptr())
    if gathered:
        gather_histogram_by_leaves(
            ctx, d_src, b_count, 2, fx.p_count, fx.fold_count, d_hist
        )
    else:
        ctx.enqueue_copy(dst_buf=d_hist, src_buf=d_src)

    var masked = List[UInt32]()
    for k in range(3 * b_count):
        masked.append(fx.bf[k])

    var bad = 0
    var worst = Float32(0.0)
    for b in range(b_count):
        for k in range(b_count):
            masked[3 * k + 2] = UInt32(0) if k == b else UInt32(1)
        ctx.enqueue_copy(dst_buf=d_bf, src_ptr=masked.unsafe_ptr())

        find_optimal_split(
            ctx, d_bf, b_count, d_cat, d_binw, fx.n_features, d_hist,
            d_parts, fx.p_count, fx.fold_count, score_before, d_rid, d_rsc,
            1, sf, l2, meta_l2_exponent, meta_frequency, normalize, std_dev,
            seed, gathered,
        )
        ctx.enqueue_copy(dst_buf=h_rid, src_buf=d_rid)
        ctx.enqueue_copy(dst_buf=h_rsc, src_buf=d_rsc)
        ctx.synchronize()

        var ok = True
        if h_rid[0] != fx.bf[3 * b]:
            ok = False
        if h_rid[1] != fx.bf[3 * b + 1]:
            ok = False
        if not _near(h_rsc[0], want_score[b]):
            ok = False
        if not _near(h_rsc[1], want_gain[b]):
            ok = False
        var r0 = _rel(h_rsc[0], want_score[b])
        var r1 = _rel(h_rsc[1], want_gain[b])
        if r0 > worst:
            worst = r0
        if r1 > worst:
            worst = r1
        if not ok:
            if bad < 3 and not quiet:
                print(
                    "     ", label, "cell b=", b, " got (f,bin,score,gain)=(",
                    h_rid[0], ",", h_rid[1], ",", h_rsc[0], ",", h_rsc[1],
                    ") want (", fx.bf[3 * b], ",", fx.bf[3 * b + 1], ",",
                    want_score[b], ",", want_gain[b], ")",
                )
            bad += 1

    _ = d_bf^
    _ = d_cat^
    _ = d_binw^
    _ = d_hist^
    _ = d_src^
    _ = d_parts^
    _ = d_rid^
    _ = d_rsc^
    _ = h_rid^
    _ = h_rsc^
    return Tally(bad, worst)


def main() raises:
    var ctx = DeviceContext()
    var failures = 0

    # =====================================================================
    # S1 / S2: single fold, DIRECT loader, five calcers
    # =====================================================================
    var folds_a: List[Int] = [4, 3, 5, 2, 4, 3, 2, 5]
    var fx = _build(folds_a, 4, 1, False, 0)
    var score_before = Float32(-3.25)
    var l2 = Float32(1.5)
    var seed = UInt64(1234567)

    var names: List[String] = [
        String("SolarL2"), String("L2"), String("LOOL2"), String("SatL2"),
        String("Cosine"),
    ]
    var sfs: List[Int] = [
        SCORE_FUNCTION_SOLAR_L2, SCORE_FUNCTION_L2, SCORE_FUNCTION_LOO_L2,
        SCORE_FUNCTION_SAT_L2, SCORE_FUNCTION_COSINE,
    ]

    var s1_bad = 0
    var s1_worst = Float32(0.0)
    for ci in range(len(sfs)):
        var ws = List[Float32]()
        var wg = List[Float32]()
        for b in range(fx.b_count):
            var r = _host_single_fold(
                fx, b, False, sfs[ci], l2, Float32(1.0), False, Float32(0.0),
                seed, score_before,
            )
            ws.append(r[0])
            wg.append(r[1])
        var t = _sweep(
            ctx, fx, False, sfs[ci], l2, Float32(1.0), Float32(0.0), False,
            Float32(0.0), seed, score_before, ws, wg,
            String("S1 ") + names[ci],
        )
        s1_bad += t.bad
        if t.worst > s1_worst:
            s1_worst = t.worst
    if s1_bad != 0:
        print(
            "FAIL S1: --", s1_bad,
            "candidate cells wrong across the five calcers.",
        )
        failures += 1
    else:
        print(
            "  ok   S1 -- 5 calcers x", fx.b_count,
            "candidates, per-cell score/gain/featureId/binId; worst relative"
            " discrepancy", s1_worst,
        )

    # ---- S2: the meta exponent, BOTH arms -------------------------------
    var meta = Float32(0.5)
    var ws2 = List[Float32]()
    var wg2 = List[Float32]()
    for b in range(fx.b_count):
        var r2 = _host_single_fold(
            fx, b, False, SCORE_FUNCTION_L2, l2, meta, False, Float32(0.0),
            seed, score_before,
        )
        ws2.append(r2[0])
        wg2.append(r2[1])
    var t2 = _sweep(
        ctx, fx, False, SCORE_FUNCTION_L2, l2, meta, Float32(1.0), False,
        Float32(0.0), seed, score_before, ws2, wg2, String("S2 metaExp"),
    )
    # the two arms must actually differ, or the pow path is inert
    var differs = 0
    for b in range(fx.b_count):
        var r1 = _host_single_fold(
            fx, b, False, SCORE_FUNCTION_L2, l2, Float32(1.0), False,
            Float32(0.0), seed, score_before,
        )
        if abs(r1[0] - ws2[b]) > Float32(1e-4):
            differs += 1
    if t2.bad != 0 or differs == 0:
        print(
            "FAIL S2: --", t2.bad, "cells wrong;", differs,
            "of", fx.b_count, "candidates distinguish MetaExponent 0.5 from"
            " 1.0 (0 would mean the pow arm is inert here)",
        )
        failures += 1
    else:
        print(
            "  ok   S2 -- MetaExponent pow/copysign arm, per cell;", differs,
            "of", fx.b_count, "candidates differ from the MetaExponent==1"
            " arm; worst relative discrepancy", t2.worst,
        )

    # ---- M1: the host draw ---------------------------------------------
    var d0 = meta_exponent_draw(seed, Float32(0.0), Float32(0.5))
    var d1 = meta_exponent_draw(seed, Float32(1.0), Float32(0.5))
    var m1_ok = (
        d0[0] == Float32(1.0)
        and d1[0] == Float32(0.5)
        and d0[1] != seed
        and d1[1] == d0[1]
    )
    if not m1_ok:
        print(
            "FAIL M1: -- meta_exponent_draw: frequency 0 gave", d0[0],
            "(want 1.0), frequency 1 gave", d1[0], "(want 0.5), advanced"
            " seed", d0[1], "vs input", seed,
        )
        failures += 1
    else:
        print(
            "  ok   M1 -- meta_exponent_draw takes both arms and advances"
            " the seed on both",
        )

    # =====================================================================
    # L1 / G1: the gather and the gathered loader
    # =====================================================================
    var l1_bad = 0
    var l1_worst = Float32(0.0)
    for ci in range(len(sfs)):
        var ws3 = List[Float32]()
        var wg3 = List[Float32]()
        for b in range(fx.b_count):
            var r3 = _host_single_fold(
                fx, b, True, sfs[ci], l2, Float32(1.0), False, Float32(0.0),
                seed, score_before,
            )
            ws3.append(r3[0])
            wg3.append(r3[1])
        var t3 = _sweep(
            ctx, fx, True, sfs[ci], l2, Float32(1.0), Float32(0.0), False,
            Float32(0.0), seed, score_before, ws3, wg3,
            String("L1 ") + names[ci],
        )
        l1_bad += t3.bad
        if t3.worst > l1_worst:
            l1_worst = t3.worst
    if l1_bad != 0:
        print("FAIL L1: --", l1_bad, "cells wrong on the GATHERED loader.")
        failures += 1
    else:
        print(
            "  ok   L1 -- gathered-by-leaves loader over the ported"
            " transpose, 5 calcers x", fx.b_count,
            "candidates; worst relative discrepancy", l1_worst,
        )

    # ---- G1: the gather kernel, per cell, hist 1/2/4 and fold 4/1 -------
    var g1_bad = 0
    var g1_cells = 0
    var hcs: List[Int] = [1, 2, 4]
    var fcs: List[Int] = [4, 1]
    for hi in range(len(hcs)):
        for fi in range(len(fcs)):
            var hc = hcs[hi]
            var fc = fcs[fi]
            var lc = 4  # power of two: theirs masks with (leafCount - 1)
            var bc = 37
            var n = bc * lc * fc * hc
            var src = List[Float32]()
            for k in range(n):
                src.append(Float32(_hash(k, hc * 7 + fc, 991) % 4096))
            var d_in = ctx.enqueue_create_buffer[DType.float32](n)
            var d_out = ctx.enqueue_create_buffer[DType.float32](n)
            var h_out = ctx.enqueue_create_host_buffer[DType.float32](n)
            ctx.enqueue_memset(d_out, Float32(-1.0))
            ctx.enqueue_copy(dst_buf=d_in, src_ptr=src.unsafe_ptr())
            gather_histogram_by_leaves(ctx, d_in, bc, hc, lc, fc, d_out)
            ctx.enqueue_copy(dst_buf=h_out, src_buf=d_out)
            ctx.synchronize()
            for f in range(bc):
                for leaf in range(lc):
                    for fold in range(fc):
                        for h in range(hc):
                            var want = src[
                                (f + bc * (leaf * fc + fold)) * hc + h
                            ]
                            var at = (
                                (f * lc + leaf) * fc + fold
                            ) * hc + h
                            g1_cells += 1
                            if h_out[at] != want:
                                if g1_bad < 3:
                                    print(
                                        "     G1 hist", hc, "fold", fc,
                                        "f", f, "leaf", leaf, "fold", fold,
                                        "h", h, ": got", h_out[at], "want",
                                        want,
                                    )
                                g1_bad += 1
            _ = d_in^
            _ = d_out^
            _ = h_out^
    if g1_bad != 0:
        print("FAIL G1: --", g1_bad, "of", g1_cells, "gathered cells wrong.")
        failures += 1
    else:
        print(
            "  ok   G1 --", g1_cells,
            "gather cells exact, HIST_COUNT 1/2/4 x foldCount 4/1",
        )

    # =====================================================================
    # A1 / A2: the block argmin and the tie-break
    # =====================================================================
    var folds_b = List[Int]()
    for _ in range(60):
        folds_b.append(5)
    var fb = _build(folds_b, 2, 1, False, 17)
    # 300 bin features, block 128, 2 blocks:
    #   block 0 -> [0,128) and [256,300)   block 1 -> [128,256)
    var gains_b = List[Float32]()
    var scores_b = List[Float32]()
    for b in range(fb.b_count):
        var rb = _host_single_fold(
            fb, b, False, SCORE_FUNCTION_COSINE, l2, Float32(1.0), False,
            Float32(0.0), seed, score_before,
        )
        scores_b.append(rb[0])
        gains_b.append(rb[1])

    # THE TIE IS PLANTED AT FIXED INDICES, chosen so "smallest index" and
    # "lowest thread id" give DIFFERENT answers. Block 0 visits [0,128) at
    # tid == index and [256,300) at tid == index - 256, so:
    #
    #     index   1 -> tid 1      index 258 -> tid 2      index 262 -> tid 6
    #
    # Their rule (`pointwise_scores.cu:148`) is smallest INDEX, so the
    # answer is 1. TWO EARLIER TIE SETS WERE BOTH TOO WEAK, and each was
    # found by sabotaging the library rather than by reading it:
    #
    #   {100, 110, 260}: every feature here has 5 bins, so all three are
    #       BinId 0, and the plant copies the FeatureId too -- all three
    #       report the IDENTICAL record and the output cannot name a
    #       winner. Flipping the tie-break left the gate GREEN.
    #   {100, 111, 262}: distinct BinIds, and flipping the comparison to
    #       `<` was caught. But DELETING the tie-break entirely was not:
    #       the reduce lands index 100 in tid 0's slot at stride 4, before
    #       262 (sitting in tid 2) is ever compared against it, so the
    #       smallest index wins by accident.
    #
    # {1, 258, 262} is the set where the tie-break has to do the work. The
    # SMALLEST index sits at tid 1, which merges into tid 0 LAST, at
    # stride 1 -- by which point tid 0 already holds index 258 (via tid 2).
    # So the final comparison is dest 258 against src 1 with equal gains,
    # and only the `indices[tid] > indices[tid + s]` clause moves it to 1.
    # Delete that clause, or reverse it, and the kernel answers 258.
    # BinIds are 1, 3 and 2, all distinct, so the record names the winner.
    var tie_dsts: List[Int] = [1, 258, 262]

    # block 0's argmin BEFORE the plant. Its column is what gets copied onto
    # the three, so the tie IS the minimum by construction.
    var m = -1
    var mg = FLOAT32_MAX
    for b in range(fb.b_count):
        var in_block0 = b < 128 or b >= 256
        if in_block0 and gains_b[b] < mg:
            mg = gains_b[b]
            m = b

    var win_feat = fb.bf[3 * m]
    var win_w = List[Float32]()
    var win_t = List[Float32]()
    for leaf in range(fb.p_count):
        win_w.append(fb.hist[(leaf * fb.b_count + m) * 2 + 0])
        win_t.append(fb.hist[(leaf * fb.b_count + m) * 2 + 1])

    # If the old minimum is not one of the three, ITS column is replaced by
    # an ordinary donor's, so the minimum ends up attained only on the
    # planted set. Otherwise a min sitting at an index below 100 would win
    # legitimately and the tie would decide nothing.
    var m_in_set = m == 1 or m == 258 or m == 262
    if not m_in_set:
        var donor = 3 if m != 3 else 4
        fb.bf[3 * m] = fb.bf[3 * donor]
        for leaf in range(fb.p_count):
            fb.hist[(leaf * fb.b_count + m) * 2 + 0] = fb.hist[
                (leaf * fb.b_count + donor) * 2 + 0
            ]
            fb.hist[(leaf * fb.b_count + m) * 2 + 1] = fb.hist[
                (leaf * fb.b_count + donor) * 2 + 1
            ]

    # the tie itself: same column AND same FeatureId, so score, gain and
    # every multiplier are bit-identical. BinId is left alone, so the
    # reported BinId names WHICH index won.
    for dst in tie_dsts:
        fb.bf[3 * dst] = win_feat
        for leaf in range(fb.p_count):
            fb.hist[(leaf * fb.b_count + dst) * 2 + 0] = win_w[leaf]
            fb.hist[(leaf * fb.b_count + dst) * 2 + 1] = win_t[leaf]
    # recompute after the plant
    for b in range(fb.b_count):
        var rb2 = _host_single_fold(
            fb, b, False, SCORE_FUNCTION_COSINE, l2, Float32(1.0), False,
            Float32(0.0), seed, score_before,
        )
        scores_b[b] = rb2[0]
        gains_b[b] = rb2[1]

    var want_idx: List[Int] = [-1, -1]
    for blk in range(2):
        var best = -1
        var bg = FLOAT32_MAX
        for b in range(fb.b_count):
            var mine = (b < 128 or b >= 256) if blk == 0 else (
                b >= 128 and b < 256
            )
            if mine and gains_b[b] < bg:
                bg = gains_b[b]
                best = b
        want_idx[blk] = best

    var n_tied = 0
    for b in range(fb.b_count):
        if (b < 128 or b >= 256) and gains_b[b] == gains_b[want_idx[0]]:
            n_tied += 1

    var d_bf2 = ctx.enqueue_create_buffer[DType.uint32](3 * fb.b_count)
    var d_cat2 = ctx.enqueue_create_buffer[DType.float32](fb.n_features)
    var d_bw2 = ctx.enqueue_create_buffer[DType.float32](fb.n_features)
    var d_h2 = ctx.enqueue_create_buffer[DType.float32](len(fb.hist))
    var d_p2 = ctx.enqueue_create_buffer[DType.float32](len(fb.parts))
    var d_ri2 = ctx.enqueue_create_buffer[DType.uint32](4)
    var d_rs2 = ctx.enqueue_create_buffer[DType.float32](4)
    var h_ri2 = ctx.enqueue_create_host_buffer[DType.uint32](4)
    var h_rs2 = ctx.enqueue_create_host_buffer[DType.float32](4)
    ctx.enqueue_copy(dst_buf=d_bf2, src_ptr=fb.bf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_cat2, src_ptr=fb.cat_w.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_bw2, src_ptr=fb.bin_w.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_h2, src_ptr=fb.hist.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_p2, src_ptr=fb.parts.unsafe_ptr())
    find_optimal_split(
        ctx, d_bf2, fb.b_count, d_cat2, d_bw2, fb.n_features, d_h2, d_p2,
        fb.p_count, 1, score_before, d_ri2, d_rs2, 2, SCORE_FUNCTION_COSINE,
        l2, Float32(1.0), Float32(0.0), False, Float32(0.0), seed, False,
    )
    ctx.enqueue_copy(dst_buf=h_ri2, src_buf=d_ri2)
    ctx.enqueue_copy(dst_buf=h_rs2, src_buf=d_rs2)
    ctx.synchronize()

    var a_bad = 0
    for blk in range(2):
        var w = want_idx[blk]
        if h_ri2[2 * blk] != fb.bf[3 * w]:
            a_bad += 1
        if h_ri2[2 * blk + 1] != fb.bf[3 * w + 1]:
            a_bad += 1
        if not _near(h_rs2[2 * blk], scores_b[w]):
            a_bad += 1
        if not _near(h_rs2[2 * blk + 1], gains_b[w]):
            a_bad += 1
        if a_bad != 0:
            print(
                "     A block", blk, "want index", w, "(f", fb.bf[3 * w],
                "bin", fb.bf[3 * w + 1], "score", scores_b[w], "gain",
                gains_b[w], ") got (f", h_ri2[2 * blk], "bin",
                h_ri2[2 * blk + 1], "score", h_rs2[2 * blk], "gain",
                h_rs2[2 * blk + 1], ")",
            )
    if a_bad != 0 or n_tied != 3 or want_idx[0] != 1:
        print(
            "FAIL A1/A2: --", a_bad, "of 8 fields wrong;", n_tied,
            "candidates tie for block 0's minimum (want exactly 3); the"
            " smallest tied index is", want_idx[0], "(want 1)",
        )
        failures += 1
    else:
        print(
            "  ok   A1 -- block argmin over 300 bin features, 2 blocks; grid"
            " stride runs twice in block 0 ([0,128) then [256,300)); all 8"
            " fields per block",
        )
        print(
            "  ok   A2 -- three-way exact tie at block 0's minimum (indices"
            " 1, 258, 262 at tids 1, 2, 6; BinIds 1, 3, 2) resolved to the"
            " SMALLEST INDEX 1; deleting the tie-break clause or reversing"
            " it both answer 258",
        )
    _ = d_bf2^
    _ = d_cat2^
    _ = d_bw2^
    _ = d_h2^
    _ = d_p2^
    _ = d_ri2^
    _ = d_rs2^
    _ = h_ri2^
    _ = h_rs2^

    # =====================================================================
    # D1: the dynamic SOLAR kernel
    # =====================================================================
    var folds_d: List[Int] = [4, 3, 5, 2, 4, 3]
    # FOLD COUNT 6, NOT 4, AND THAT IS DELIBERATE. `GetHistogramOffset` is
    # `part * FoldCount + fold` while `GetDataPartitionOffset` is
    # `part * (1 << ceil(log2(FoldCount))) + fold`, so the two coincide at
    # every POWER-OF-TWO fold count and differ everywhere else. At
    # foldCount 4 (and at 1) a kernel that used the histogram offset to
    # index the partitions was measured GREEN across every gate here. Six
    # gives stripe 8 against fold count 6, three (estimate, test) pairs, and
    # two partition slots per leaf that are addressable but never written --
    # which is what an ordered fit's parts array actually looks like.
    var fd = _build(folds_d, 2, 6, True, 5)
    var wsd = List[Float32]()
    var wgd = List[Float32]()
    var fired = 0
    var skipped = 0
    for b in range(fd.b_count):
        var rd = _host_dynamic_solar(fd, b, score_before)
        wsd.append(rd[0])
        wgd.append(rd[1])
        fired += rd[2]
        skipped += rd[3]
    var td = _sweep(
        ctx, fd, False, SCORE_FUNCTION_SOLAR_L2, l2, Float32(1.0),
        Float32(0.0), False, Float32(0.0), seed, score_before, wsd, wgd,
        String("D1 solar"),
    )
    if td.bad != 0 or fired == 0 or skipped == 0:
        print(
            "FAIL D1: --", td.bad, "cells wrong; the `> 2` cut fired",
            fired, "times and was skipped", skipped,
            "times (both must be non-zero or the cut is not gated)",
        )
        failures += 1
    else:
        print(
            "  ok   D1 -- dynamic solar, foldCount 6 (stripe 8),", fd.b_count,
            "candidates per cell; the `totalWeight > 2` cut fired", fired,
            "and skipped", skipped, "; worst relative discrepancy", td.worst,
        )

    # =====================================================================
    # D2 / D3: the dynamic COSINE kernel
    # =====================================================================
    var d2_bad = 0
    var d2_worst = Float32(0.0)
    var norms: List[Bool] = [False, True]
    for ni in range(2):
        var wsc = List[Float32]()
        var wgc = List[Float32]()
        for b in range(fd.b_count):
            var rc = _host_dynamic_cosine(
                fd, b, score_before, l2, norms[ni], Float32(0.0), seed, True
            )
            wsc.append(rc[0])
            wgc.append(rc[1])
        var tc = _sweep(
            ctx, fd, False, SCORE_FUNCTION_COSINE, l2, Float32(1.0),
            Float32(0.0), norms[ni], Float32(0.0), seed, score_before, wsc,
            wgc, String("D2 cosine normalize=") + String(norms[ni]),
        )
        d2_bad += tc.bad
        if tc.worst > d2_worst:
            d2_worst = tc.worst
    # the two arms must differ, or `normalize` is inert at this lambda
    var norm_diff = 0
    for b in range(fd.b_count):
        var a = _host_dynamic_cosine(
            fd, b, score_before, l2, False, Float32(0.0), seed, True
        )
        var bb = _host_dynamic_cosine(
            fd, b, score_before, l2, True, Float32(0.0), seed, True
        )
        if abs(a[0] - bb[0]) > Float32(1e-4):
            norm_diff += 1
    if d2_bad != 0 or norm_diff == 0:
        print(
            "FAIL D2: --", d2_bad, "cells wrong;", norm_diff,
            "candidates distinguish normalize=True from False",
        )
        failures += 1
    else:
        print(
            "  ok   D2 -- dynamic cosine, normalize BOTH ways, per cell;",
            norm_diff, "of", fd.b_count,
            "candidates separate the two arms; worst relative discrepancy",
            d2_worst,
        )

    # ---- D3: the random_strength noise ---------------------------------
    var std_dev = Float32(0.75)
    var wsn = List[Float32]()
    var wgn = List[Float32]()
    var noise_moved = 0
    for b in range(fd.b_count):
        var rn = _host_dynamic_cosine(
            fd, b, score_before, l2, False, std_dev, seed, True
        )
        var r0 = _host_dynamic_cosine(
            fd, b, score_before, l2, False, Float32(0.0), seed, True
        )
        wsn.append(rn[0])
        wgn.append(rn[1])
        if abs(rn[0] - r0[0]) > Float32(1e-4):
            noise_moved += 1
    var tn = _sweep(
        ctx, fd, False, SCORE_FUNCTION_COSINE, l2, Float32(1.0), Float32(0.0),
        False, std_dev, seed, score_before, wsn, wgn, String("D3 noise"),
    )
    if tn.bad != 0 or noise_moved == 0:
        print(
            "FAIL D3: --", tn.bad, "cells wrong;", noise_moved,
            "candidates moved by the noise (0 means the path is inert)",
        )
        failures += 1
    else:
        print(
            "  ok   D3 -- scoreStdDev noise matches a host replay of"
            " AdvanceSeed(seed,4)+NextNormal and moves", noise_moved, "of",
            fd.b_count, "candidates; worst relative discrepancy", tn.worst,
        )

    # =====================================================================
    # D4: the max(weightTestRight, 0) clamp in the dynamic cosine kernel
    # =====================================================================
    # Break ONE test fold's partition weight so the derived right weight
    # goes negative there, while its ESTIMATE fold stays consistent (which
    # is what keeps `mu` non-zero and makes the clamp observable at all).
    var fc4 = _build(folds_d, 2, 4, False, 23)
    #
    # IT HAS TO BE A BIG BREAK, and the first attempt was too small. Cutting
    # the test fold's weight by 25 out of 70 made only the LAST bin of each
    # feature go negative -- and the last bin is exactly where the ESTIMATE
    # fold's right child is empty too, so `mu` is 0 and the clamp multiplies
    # nothing. Zero candidates moved. Setting the test weight to 4 puts most
    # NON-last bins over the line while their estimate-fold right child is
    # still large, which is the only configuration in which this clamp is
    # observable at all.
    var broken_part = 3 * (1 * fc4.stripe + 1)
    fc4.parts[broken_part + 0] = Float32(4.0)

    var ws4 = List[Float32]()
    var wg4 = List[Float32]()
    var ws4_bad = List[Float32]()
    var clamp_moved = 0
    for b in range(fc4.b_count):
        var rr = _host_dynamic_cosine(
            fc4, b, score_before, l2, False, Float32(0.0), seed, True
        )
        var rw = _host_dynamic_cosine(
            fc4, b, score_before, l2, False, Float32(0.0), seed, False
        )
        ws4.append(rr[0])
        wg4.append(rr[1])
        ws4_bad.append(rw[0])
        if abs(rr[0] - rw[0]) > Float32(1e-4):
            clamp_moved += 1
    var t4 = _sweep(
        ctx, fc4, False, SCORE_FUNCTION_COSINE, l2, Float32(1.0),
        Float32(0.0), False, Float32(0.0), seed, score_before, ws4, wg4,
        String("D4 clamp"),
    )
    # the WRONG expectation must be rejected: run the same sweep against the
    # unclamped host answer and require it to fail.
    var t4_bad_expect = _sweep(
        ctx, fc4, False, SCORE_FUNCTION_COSINE, l2, Float32(1.0),
        Float32(0.0), False, Float32(0.0), seed, score_before, ws4_bad, wg4,
        String("D4 unclamped"), True,
    )
    if t4.bad != 0 or clamp_moved == 0 or t4_bad_expect.bad == 0:
        print(
            "FAIL D4: --", t4.bad, "cells wrong against the clamped"
            " expectation;", clamp_moved, "candidates are moved by the clamp;"
            " the UNCLAMPED expectation was rejected on",
            t4_bad_expect.bad, "cells (must be > 0)",
        )
        failures += 1
    else:
        print(
            "  ok   D4 -- max(weightTestRight, 0) is live and correct:",
            clamp_moved, "of", fc4.b_count,
            "candidates move, and the unclamped expectation is rejected on",
            t4_bad_expect.bad, "cells",
        )

    # =====================================================================
    # R1-R4: PartitionUpdateImpl
    # =====================================================================
    var n_rows = 24000
    var tgt = List[Float32]()
    var wts = List[Float32]()
    var cnt = List[Float32]()
    for r in range(n_rows):
        tgt.append(Float32(_hash(r, 3, 11) % 63 - 31))
        wts.append(Float32(1 + _hash(r, 5, 13) % 62))
        cnt.append(Float32(_hash(r, 7, 17) % 3))

    # offsets and sizes chosen so ONE partition (20000 rows) exceeds
    # 15 * 1024 and reaches ComputeSum's unrolled loop, and one (500) does
    # not. Every offset is non-zero so a kernel that ignores Offset fails.
    var p_off: List[Int] = [7, 20100, 21000, 21500]
    var p_size: List[Int] = [20000, 500, 1, 0]
    var n_parts = 4
    var parts_u = List[UInt32]()
    for p in range(n_parts):
        parts_u.append(UInt32(p_off[p]))
        parts_u.append(UInt32(p_size[p]))

    var d_t = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var d_w = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var d_c = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var d_pu = ctx.enqueue_create_buffer[DType.uint32](2 * n_parts)
    var d_ps = ctx.enqueue_create_buffer[DType.float32](3 * n_parts)
    var h_ps = ctx.enqueue_create_host_buffer[DType.float32](3 * n_parts)
    ctx.enqueue_copy(dst_buf=d_t, src_ptr=tgt.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=wts.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_c, src_ptr=cnt.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_pu, src_ptr=parts_u.unsafe_ptr())

    var want_w = List[Float32]()
    var want_s = List[Float32]()
    var want_c = List[Float32]()
    for p in range(n_parts):
        var aw = Float32(0.0)
        var as_ = Float32(0.0)
        var ac = Float32(0.0)
        for r in range(p_off[p], p_off[p] + p_size[p]):
            aw += wts[r]
            as_ += tgt[r]
            ac += cnt[r]
        want_w.append(aw)
        want_s.append(as_)
        want_c.append(ac)

    var r_bad = 0
    var r_names: List[String] = [
        String("R1 all present"), String("R2 counts=null"),
        String("R3 weights=null"), String("R4 target=null"),
    ]
    var have_t: List[Bool] = [True, True, True, False]
    var have_w: List[Bool] = [True, True, False, True]
    var have_c: List[Bool] = [True, False, True, True]
    for run in range(4):
        ctx.enqueue_memset(d_ps, Float32(-777.0))
        update_partition_props(
            ctx, d_t, d_w, d_c, have_t[run], have_w[run], have_c[run],
            d_pu, d_ps, n_parts,
        )
        ctx.enqueue_copy(dst_buf=h_ps, src_buf=d_ps)
        ctx.synchronize()
        for p in range(n_parts):
            var ew = want_w[p] if have_w[run] else Float32(0.0)
            var es = want_s[p] if have_t[run] else Float32(0.0)
            var ec = want_c[p] if have_c[run] else Float32(p_size[p])
            if (
                h_ps[3 * p + 0] != ew
                or h_ps[3 * p + 1] != es
                or h_ps[3 * p + 2] != ec
            ):
                if r_bad < 4:
                    print(
                        "     ", r_names[run], "part", p, "got (",
                        h_ps[3 * p + 0], ",", h_ps[3 * p + 1], ",",
                        h_ps[3 * p + 2], ") want (", ew, ",", es, ",", ec,
                        ")",
                    )
                r_bad += 1
    if r_bad != 0:
        print("FAIL R1-R4: --", r_bad, "partition-stat cells wrong.")
        failures += 1
    else:
        print(
            "  ok   R1-R4 -- 4 partitions x 3 stats x 4 null-combinations,"
            " EXACT; partition 0 has 20000 rows so ComputeSum's 16-wide"
            " unrolled loop is reached at blockSize 1024 (>15*1024), and"
            " partition 1 has 500 so the tail-only path is reached too",
        )

    _ = d_t^
    _ = d_w^
    _ = d_c^
    _ = d_pu^
    _ = d_ps^
    _ = h_ps^

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("pointwise scorer: S1 S2 M1 L1 G1 A1 A2 D1 D2 D3 D4 R1-R4 pass")
