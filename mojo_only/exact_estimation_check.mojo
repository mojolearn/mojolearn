# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Exact leaf estimator against an ANALYTIC weighted quantile.

    pixi run check-exact-estimation

NO CATBOOST COUNTERPART: a gate, so it lives in `mojo_only/`.

WHAT IS UNDER TEST. `compute_weighted_quantile`
(`gbdt/methods/leaves_estimation/leaves_estimation_helper.mojo`), the port of
`leaves_estimation_helper.h:64-146`, and everything it drives: the segmented
radix sort (DEVIATION 65), `MakeEndOfBinsFlags`, the segmented weight scan,
`ComputeNeedWeights`, and the sixteen-iteration binary search. That is the
whole of `ELeavesEstimation::Exact`, which is CatBoost's DEFAULT leaf
estimator for MAE, MAPE and Quantile.

WHY IT CAN BE GATED ANALYTICALLY, which is what rule 4 asks for. The value
their estimator computes is defined without reference to their code: it is
the smallest sorted residual whose cumulative weight reaches `alpha` times
the leaf's total weight. On a fixture whose residuals are DISTINCT and
WELL SEPARATED, that value is a fact about the data, not a tally of ours, and
the host computes it by a completely different route -- an O(n^2) insertion
order and a running sum -- rather than by re-implementing their kernels.

TWO PROPERTIES OF THEIRS THE FIXTURE HAS TO RESPECT, and both are recorded
in the port rather than smoothed over:

  1. **THE SORT IGNORES THE BOTTOM TEN MANTISSA BITS.** Their call is
     `SegmentedRadixSort(..., 10, 32)` (`:110-112`), so two residuals that
     agree above bit 10 are TIED for it and may come back in either order.
     The fixture separates every residual by far more than that, so the
     ordering is unambiguous and the check is testing placement rather than
     a coin flip.
  2. **THE BINARY SEARCH RUNS A FIXED SIXTEEN ITERATIONS AND SETS
     `left = middle`, NOT `middle + 1`** (`exact_estimation.cu:34-38`). The
     interval halves each time, so sixteen iterations resolve a leaf of up
     to 65,536 rows exactly and a wider one only approximately. Leaves here
     are well under that, so the analytic answer is reachable and the check
     is not accidentally measuring their truncation.

WHAT THE FIXTURE PLANTS, per rule 8. Residuals are HASHED and scattered, so
no two rows -- and no two LEAVES -- share an expectation; a fixture with the
same quantile in every leaf would verify a total and nothing about which leaf
got which value. Leaves are DELIBERATELY RAGGED (sizes 1, 2, 3, 17, 64, 255,
1000, and one EMPTY) because the interesting branches are all at the edges:

    empty leaf        their `left > right` arm, which must write 0.0
    one-row leaf      the search's degenerate interval
    two-row leaf      where alpha decides between exactly two answers
    a leaf crossing
    the 512-row
    sort block        the segment axis' block boundary, which a
                      single-block leaf never exercises

WEIGHTS ARE NON-UNIFORM AND HASHED. With unit weights the weighted quantile
degenerates to the ordinary one and a defect in the weight scan or in
`ComputeNeedWeights` would be invisible -- that is exactly the failure
`uniform-test-data-hides-permutation` names.

THE SABOTAGES, each run and each required to move the check:

    E1  alpha 0.5 -> 0.5 + 1e-3      the quantile level actually reaches
                                     `ComputeNeedWeights`
    E2  one leaf's weights doubled   the WEIGHTED quantile is weighted
    E3  one leaf's weight column
        reversed against its
        residual column              the gather re-pairs each residual
                                     with ITS weight after the sort
    E4  two leaves' offsets swapped  the segmentation is per-leaf and not
                                     one global sort

EVERY ONE OF THEM PERTURBS THE DEVICE SIDE ONLY. The first version of this
check altered the shared fixture, so the host reference moved with the
device and two of the four reported "moved nothing" -- true, and a defect in
the sabotage rather than in the estimator. The host reference is now always
computed from the true arrays.
"""

from max.gpu.host import DeviceContext

from gbdt.methods.leaves_estimation.leaves_estimation_helper import (
    ExactQuantileScratch,
    compute_exact_approx,
    compute_weighted_quantile,
    make_exact_quantile_scratch,
)
from gbdt.targets.kernel.pointwise_targets import OBJECTIVE_MAPE


def splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def hashed_unit(seed: UInt64, i: Int) -> Float64:
    var h = splitmix(seed ^ UInt64(i * 2654435761))
    return Float64(h >> 11) * (1.0 / Float64(1 << 53))


def host_weighted_quantile(
    values: List[Float32], weights: List[Float32], alpha: Float64
) -> Float32:
    """The analytic answer, by a route that shares no code with the device.

    Insertion sort (O(n^2), deliberately -- it cannot share a bug with a
    radix sort), then a running weight sum, then the first value whose
    cumulative weight reaches `alpha * total`. That is the definition of a
    weighted quantile and it is not a transcription of their kernels.
    """
    var n = len(values)
    if n == 0:
        return Float32(0.0)
    var v = List[Float32]()
    var w = List[Float32]()
    for i in range(n):
        # insert value i into sorted position
        var pos = len(v)
        for j in range(len(v)):
            if values[i] < v[j]:
                pos = j
                break
        v.insert(pos, values[i])
        w.insert(pos, weights[i])

    var total = Float64(0.0)
    for i in range(n):
        total += Float64(w[i])
    var need = total * alpha

    var acc = Float64(0.0)
    for i in range(n):
        acc += Float64(w[i])
        # their comparison is `weightsPrefixSum[middle] < needWeights - eps`,
        # i.e. the search stops at the FIRST index whose prefix sum reaches
        # `need`. FLT_EPSILON is their slack (`exact_estimation.cu:33`).
        if acc >= need - Float64(1.1920929e-07):
            return v[i]
    return v[n - 1]


def leaf_sizes() -> List[Int]:
    """Ragged on purpose: the branches are all at the edges."""
    return [1, 2, 3, 17, 64, 0, 255, 1000, 513, 7]


def run_case(
    ctx: DeviceContext,
    alpha: Float64,
    sabotage: Int,
    verbose: Bool,
) raises -> Int:
    var sizes = leaf_sizes()
    var n_leaves = len(sizes)
    var n_rows = 0
    for i in range(n_leaves):
        n_rows += sizes[i]

    var offsets = List[Int]()
    var acc = 0
    for i in range(n_leaves):
        offsets.append(acc)
        acc += sizes[i]

    # A SABOTAGE MUST PERTURB EXACTLY ONE SIDE. The first version of this
    # check perturbed the shared fixture, so the host reference moved with
    # the device and E2 and E4 reported "moved nothing" -- which was true
    # and was a defect in the sabotage, not in the estimator. Every
    # sabotage below now alters ONLY what the DEVICE is told; the host
    # reference is always computed from the true arrays.
    var dev_offsets = offsets.copy()
    var dev_sizes = sizes.copy()

    # E4: hand the DEVICE two leaves' segments swapped. The data does not
    # move, so the host's per-leaf answer is unchanged and the device's
    # must differ -- unless the sort is global rather than per-segment.
    if sabotage == 4:
        var t0 = dev_offsets[3]
        dev_offsets[3] = dev_offsets[6]
        dev_offsets[6] = t0
        var s0 = dev_sizes[3]
        dev_sizes[3] = dev_sizes[6]
        dev_sizes[6] = s0

    var max_leaf = 1
    for i in range(n_leaves):
        if sizes[i] > max_leaf:
            max_leaf = sizes[i]

    var d_off = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var d_sz = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    for i in range(n_leaves):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(dev_offsets[i]))
        h_sz.unsafe_ptr().unsafe_store(i, UInt32(dev_sizes[i]))
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_sz, src_ptr=h_sz.unsafe_ptr())

    var s = make_exact_quantile_scratch(ctx, n_rows, n_leaves, max_leaf)

    # THE FIXTURE. Residuals hashed and separated by far more than the ten
    # mantissa bits their sort drops; weights hashed and non-uniform.
    var h_r = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var res = List[Float32]()
    var wts = List[Float32]()
    for i in range(n_rows):
        var r = Float32(hashed_unit(UInt64(0x5A), i) * 200.0 - 100.0)
        var w = Float32(0.125 + hashed_unit(UInt64(0x6B), i) * 4.0)
        res.append(r)
        wts.append(w)

    # The DEVICE'S copies. `res` and `wts` stay true for the reference.
    var dev_res = res.copy()
    var dev_wts = wts.copy()

    # E3: REVERSE one leaf's weight column against its residual column,
    # device side only. Both multisets are unchanged and so is the leaf's
    # total weight -- and therefore `need` -- so the ONLY thing that moves
    # is which residual carries which weight. That pairing is exactly what
    # the gather after the sort is responsible for
    # (`leaves_estimation_helper.h:99-103`), and nothing else in the
    # pipeline can compensate for breaking it.
    #
    # A first version swapped a single PAIR of residuals and moved
    # nothing. That was not the mechanism being invisible: with 1,000 rows
    # at alpha 0.5 the crossing index has to land between the two swapped
    # rows for a single pair to matter, so the perturbation was merely too
    # small to be decisive. Rule 8 asks for one sabotage per MECHANISM, and
    # a sabotage that only sometimes fires is not one.
    if sabotage == 3:
        var base = offsets[7]
        var m = sizes[7]
        for k in range(m // 2):
            var t = dev_wts[base + k]
            dev_wts[base + k] = dev_wts[base + m - 1 - k]
            dev_wts[base + m - 1 - k] = t

    # E2: shift one leaf's weight MASS on the device only -- halve the
    # first half, leave the rest. Doubling every weight in a leaf would
    # scale `need` by the same factor and move nothing, which is why the
    # perturbation is asymmetric.
    if sabotage == 2:
        for k in range(sizes[7] // 2):
            dev_wts[offsets[7] + k] = (
                dev_wts[offsets[7] + k] * Float32(0.5)
            )

    for i in range(n_rows):
        h_r.unsafe_ptr().unsafe_store(i, dev_res[i])
        h_w.unsafe_ptr().unsafe_store(i, dev_wts[i])
    ctx.enqueue_copy(dst_buf=s.residuals, src_ptr=h_r.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=s.residual_weights, src_ptr=h_w.unsafe_ptr())
    ctx.synchronize()

    var use_alpha = alpha
    if sabotage == 1:
        use_alpha = alpha + 1e-3

    var point = List[Float32]()
    compute_weighted_quantile(
        ctx, n_rows, n_leaves, max_leaf, Float32(use_alpha), False,
        d_off, d_sz, s, point,
    )

    # THE ANALYTIC COMPARE, per leaf. Never a total: a check that summed
    # the leaf values would pass with every leaf's answer in the wrong slot.
    var bad = 0
    for leaf in range(n_leaves):
        var lv = List[Float32]()
        var lw = List[Float32]()
        for k in range(sizes[leaf]):
            lv.append(res[offsets[leaf] + k])
            lw.append(wts[offsets[leaf] + k])
        var want = host_weighted_quantile(lv, lw, alpha)
        var got = point[leaf]
        if got != want:
            bad += 1
            if verbose and bad <= 4:
                print(
                    "    leaf", leaf, "size", sizes[leaf],
                    "got", got, "want", want,
                )
    return bad


def run_mape_case(
    ctx: DeviceContext, alpha: Float64, verbose: Bool
) raises -> Int:
    """`ComputeExactApprox`'s MAPE ARM, which had no caller in any check.

    PORTING_RULES 8, and it cost a real defect: this file called
    `compute_weighted_quantile` with `use_mape_weights=False` on every arm,
    so `compute_exact_approx`'s `is_mape` branch -- the one that runs
    `ComputeWeightsWithTargets` first -- was never executed by anything.
    The wrong-alpha defect that `check-loss-oracle` found in 2026-08-21's
    MAPE fit lived one call up from here, and a check that never entered
    the branch could not have been the one to find it.

    The gate is the same analytic weighted quantile, with the weights
    their MAPE arm actually uses:

        weightsWithTargets[i] = weights[i] / max(1, |residual[i]|)

    -- `exact_estimation.cu:76-87`, on the RESIDUAL, which is what both of
    their arms do (see `compute_exact_approx`'s note).
    """
    var sizes = List[Int]()
    for v in [5, 1, 33, 0, 200, 64]:
        sizes.append(v)
    var n_leaves = len(sizes)
    var n_rows = 0
    for i in range(n_leaves):
        n_rows += sizes[i]

    var offsets = List[Int]()
    var acc = 0
    for i in range(n_leaves):
        offsets.append(acc)
        acc += sizes[i]
    var max_leaf = 1
    for i in range(n_leaves):
        if sizes[i] > max_leaf:
            max_leaf = sizes[i]

    var d_off = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var d_sz = ctx.enqueue_create_buffer[DType.uint32](n_leaves)
    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    var h_sz = ctx.enqueue_create_host_buffer[DType.uint32](n_leaves)
    for i in range(n_leaves):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(offsets[i]))
        h_sz.unsafe_ptr().unsafe_store(i, UInt32(sizes[i]))
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_sz, src_ptr=h_sz.unsafe_ptr())

    var s = make_exact_quantile_scratch(ctx, n_rows, n_leaves, max_leaf)
    var h_r = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](n_rows)

    var res = List[Float32]()
    var wts = List[Float32]()
    for i in range(n_rows):
        # residuals STRADDLING 1 in magnitude, so `max(1, |r|)` forks both
        # ways; a fixture entirely above or below it would leave half the
        # denominator unreached
        var r = Float32(hashed_unit(UInt64(0x7C), i) * 6.0 - 3.0)
        var w = Float32(0.25 + hashed_unit(UInt64(0x7D), i) * 3.0)
        res.append(r)
        wts.append(w)
        h_r.unsafe_ptr().unsafe_store(i, r)
        h_w.unsafe_ptr().unsafe_store(i, w)
    ctx.enqueue_copy(dst_buf=s.residuals, src_ptr=h_r.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=s.residual_weights, src_ptr=h_w.unsafe_ptr())
    ctx.synchronize()

    var point = List[Float32]()
    compute_exact_approx(
        ctx, OBJECTIVE_MAPE, True, n_rows, n_leaves, max_leaf,
        Float32(alpha), d_off, d_sz, s, point,
    )

    var bad = 0
    for leaf in range(n_leaves):
        var lv = List[Float32]()
        var lw = List[Float32]()
        for k in range(sizes[leaf]):
            var r = res[offsets[leaf] + k]
            var den = r if r > Float32(0.0) else -r
            if den < Float32(1.0):
                den = Float32(1.0)
            lv.append(r)
            lw.append(wts[offsets[leaf] + k] / den)
        var want = host_weighted_quantile(lv, lw, alpha)
        if point[leaf] != want:
            bad += 1
            if verbose and bad <= 3:
                print(
                    "    MAPE leaf", leaf, "size", sizes[leaf],
                    "got", point[leaf], "want", want,
                )
    return bad


def check_exact_estimation(ctx: DeviceContext) raises:
    var failures = 0

    print("-- honest run: the analytic weighted quantile, per leaf --")
    var alphas = [0.1, 0.5, 0.75, 0.9]
    for ai in range(len(alphas)):
        var a = alphas[ai]
        var bad = run_case(ctx, a, 0, True)
        if bad != 0:
            print("  FAIL alpha", a, "leaves wrong:", bad)
            failures += 1
        else:
            print("  ok   alpha", a, "-- every leaf matches exactly")

    print()
    print("-- the MAPE arm, which had no caller before 2026-08-21 --")
    for ai in range(len(alphas)):
        var a = alphas[ai]
        var mbad = run_mape_case(ctx, a, True)
        if mbad != 0:
            print("  FAIL MAPE alpha", a, "leaves wrong:", mbad)
            failures += 1
        else:
            print(
                "  ok   MAPE alpha", a,
                "-- every leaf matches the weight/max(1,|r|) quantile",
            )

    print()
    print("-- sabotages --")
    # E1, E2, E4 must MOVE the answer.
    var e1 = run_case(ctx, 0.5, 1, False)
    if e1 == 0:
        print("  FAIL E1 (alpha + 1e-3) moved nothing")
        failures += 1
    else:
        print("  ok   E1 alpha + 1e-3 ->", e1, "leaves moved")

    var e2 = run_case(ctx, 0.5, 2, False)
    if e2 == 0:
        print("  FAIL E2 (one leaf's weight mass shifted) moved nothing")
        failures += 1
    else:
        print("  ok   E2 weight mass shifted ->", e2, "leaves moved")

    var e4 = run_case(ctx, 0.5, 4, False)
    if e4 == 0:
        print("  FAIL E4 (two leaves' segments swapped) moved nothing")
        failures += 1
    else:
        print("  ok   E4 segments swapped ->", e4, "leaves moved")

    # E3: swap two residuals inside one leaf on the DEVICE side. The
    # multiset of residuals is unchanged, so nothing about the leaf's
    # SORTED VALUES moves; what moves is which weight each residual
    # carries, which is exactly what the gather after the sort is for.
    var e3 = run_case(ctx, 0.5, 3, False)
    if e3 == 0:
        print(
            "  FAIL E3 (residual/weight pairing broken) moved nothing"
        )
        failures += 1
    else:
        print("  ok   E3 residual/weight pairing ->", e3, "leaves moved")

    if failures != 0:
        raise Error(
            "exact estimation check: " + String(failures) + " failures"
        )
    print()
    print("exact estimation check: PASS")


def main() raises:
    var ctx = DeviceContext()
    check_exact_estimation(ctx)
