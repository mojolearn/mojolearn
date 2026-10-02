# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Checks for the caller-facing k-means surface, `cluster/estimator.mojo`.

Separate from `kmeans_check.mojo` on purpose. That file verifies the KERNELS
against hand-computed expectations. This one verifies the POLICY the
estimator layer adds on top of them: which scale gets chosen, whether the
weights arms agree, and whether a caller holding nothing but host pointers
gets the right answer back.

**The fixture plants clusters with a HASHED jitter, not a uniform one.** A
check whose expected value is the same in every cell verifies the total and
nothing about placement, and this repository has been caught by that twice --
a wrong reduction and an inert kernel both reported CORRECT by checks whose
fixtures were uniform. Here the jitter differs per (row, feature), so a fit
that assigned rows to the wrong centroid, or transposed the output, cannot
produce a passing label vector.

THE HOST-BUFFER LIFETIME TRAP, WHICH COST AN HOUR AND WILL COST ANOTHER
-----------------------------------------------------------------------
**Never bind `var p = buf.unsafe_ptr()` and then stop mentioning `buf`.**
`unsafe_ptr()` returns an UNTRACKED pointer, so it does not keep the buffer
alive, and Mojo destroys `buf` at its last use -- which is that line. The
memory is then handed back and the next allocation reuses it.

It does not crash. It returns a wrong answer, quietly, and only sometimes.
Measured here: `check_kmeans_fit_weight_arms_agree` bound
`var w = hw.unsafe_ptr()`, filled 512 ones through it, and `kmeans_fit` read
that array as ALL ZEROS -- visible only because `choose_scale` returns 1.0 for
an all-zero plane and the unit arm returned 1048576.0, so the cross-arm
comparison had something to disagree about. A check that only ran the one arm
would have reported OK.

`neighbors/checks/estimator_check.mojo` avoids this by accident: it writes
`h_index.unsafe_ptr()` inline at each use, so the buffer is live at every
point it matters. Do that. Every pointer in this file is taken at its point
of use for exactly that reason, and the redundant-looking
`hx.unsafe_ptr().unsafe_store(...)` calls are not redundant.
"""

from max.gpu.host import DeviceContext

from std.memory import bitcast
from cluster.estimator import (
    KMeansFitResult,
    kmeans_fit,
    plan_sum_scale,
)
from cluster.host.kmeans_oracle import host_plan_sum_scale
from cluster.impl.kmeans_params import (
    INIT_ARRAY,
    METRIC_L2_EXPANDED,
)
from checks.fixed_point import choose_scale


comptime EST_ROWS = 512
comptime EST_FEATURES = 4
comptime EST_CLUSTERS = 4


def _planted_center(cluster: Int, feature: Int) -> Float32:
    """Separation 100 per cluster against a jitter under 1.

    Matches `kmeans_check.mojo:104` deliberately, so a failure here and a
    failure there are comparable rather than being two different fixtures.
    """
    return Float32(100 * (cluster + 1) + feature)


def _jitter(row: Int, feature: Int) -> Float32:
    """Scattered, not uniform. See the module docstring for why that matters.

    FNV-1a over (row, feature) folded into [-0.5, 0.5). Every cell differs
    from every other, so a per-cell comparison downstream has something to
    fail on.
    """
    var h = UInt64(14695981039346656037)
    h = (h ^ UInt64(row)) * UInt64(1099511628211)
    h = (h ^ UInt64(feature)) * UInt64(1099511628211)
    h = h ^ (h >> 33)
    return Float32(Float64(h % UInt64(1000)) / 1000.0) - Float32(0.5)


def _fill(x: MutPointer[Float32, MutUntrackedOrigin]):
    """Round-robin membership so every cluster gets exactly EST_ROWS/4 rows."""
    for r in range(EST_ROWS):
        var c = r % EST_CLUSTERS
        for f in range(EST_FEATURES):
            x.unsafe_store(
                r * EST_FEATURES + f, _planted_center(c, f) + _jitter(r, f)
            )


def _upload_scale(
    ctx: DeviceContext, hx: List[Float32], n: Int, d: Int
) raises -> Float64:
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()
    var s = plan_sum_scale(ctx, x, n, d)
    _ = x^
    return s


def check_plan_sum_scale() raises:
    """The scale policy, asserted without running a fit.

    CLAIM 1: the device fold is the WORST column, widened only by the fold's
    error bound: the scale equals `choose_scale` of the bound over the
    independently recomputed worst column, or is one binade coarser (the
    widening crossed a power of two). CLAIM 2: the row count reaches
    `choose_scale` (the scale is finer than the blanket one).
    """
    var ctx = DeviceContext()
    var hx = List[Float32](length=EST_ROWS * EST_FEATURES, fill=Float32(0.0))
    _fill(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(hx.unsafe_ptr())))
    var worst = Float64(0.0)
    for f in range(EST_FEATURES):
        var col = Float64(0.0)
        for r in range(EST_ROWS):
            col += Float64(abs(hx[r * EST_FEATURES + f]))
        if col > worst:
            worst = col
    var expected = choose_scale(worst, EST_ROWS)
    var got = _upload_scale(ctx, hx, EST_ROWS, EST_FEATURES)
    if got != expected and got != expected * 0.5:
        raise Error(
            "plan_sum_scale did not use the worst column: got "
            + String(got)
            + " expected "
            + String(expected)
            + " (or half of it)"
        )
    var blanket = choose_scale(worst * 1.001)
    if not (got > blanket):
        raise Error(
            "plan_sum_scale returned the BLANKET scale or weaker, so the row"
            " count is not reaching choose_scale. See POLICY CHOICE 2."
        )
    print("check_plan_sum_scale: OK (worst column, row count reaches choose_scale)")


def check_plan_sum_scale_device() raises:
    """The device fold equals the host column's restatement
    (`kmeans_oracle.host_plan_sum_scale`) word for word, over 48 magnitudes a
    factor of 2^(1/8) apart at n = 5000 (three chunks), the all-zero plane
    (scale 1.0), and a NaN and an infinity (both refused by name on both
    sides)."""
    var ctx = DeviceContext()
    var n = 5000
    var d = 7
    var hx = List[Float32](length=n * d, fill=Float32(0.0))
    var gain = Float64(1.0)
    for member in range(48):
        for r in range(n):
            for f in range(d):
                var h = (r * 2654435761 + f * 40503 + member * 7919) % 2039
                var v = (Float64(h) / 2039.0 - 0.5) * gain * Float64(f + 1)
                hx[r * d + f] = Float32(v)
        var want = host_plan_sum_scale(hx, n, d)
        var got = _upload_scale(ctx, hx, n, d)
        if got != want:
            raise Error(
                "device scale " + String(got) + " != host column scale "
                + String(want) + " at sweep member " + String(member)
            )
        gain *= 1.0905077326652577
    for i in range(n * d):
        hx[i] = Float32(0.0)
    if _upload_scale(ctx, hx, n, d) != 1.0 or host_plan_sum_scale(hx, n, d) != 1.0:
        raise Error("an all-zero plane must take scale 1.0 on both sides")
    for r in range(n):
        hx[r * d] = Float32(1.5)
    for word in range(2):
        var bad = UInt32(0x7FC00000) if word == 0 else UInt32(0x7F800000)
        hx[1234 * d + 3] = bitcast[DType.float32](bad)
        var dev_refused = False
        var host_refused = False
        try:
            _ = _upload_scale(ctx, hx, n, d)
        except:
            dev_refused = True
        try:
            _ = host_plan_sum_scale(hx, n, d)
        except:
            host_refused = True
        if not dev_refused or not host_refused:
            raise Error("a non-finite design value must be refused on both sides")
    print(
        "check_plan_sum_scale_device: OK (48 sweep members equal to the host"
        " column, zero plane 1.0, NaN and inf refused on both sides)"
    )


def check_kmeans_fit_recovers_planted() raises:
    """A caller holding host pointers gets the planted clustering back.

    Verified PER ROW against the planted membership, not by an aggregate.
    Labels are compared through the centroid each one names, because k-means
    label IDs are arbitrary: two rows planted in the same cluster must land
    on the same centroid, and that centroid must be the one nearest their
    planted center.
    """
    var ctx = DeviceContext()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](
        EST_ROWS * EST_FEATURES
    )
    var hc = ctx.enqueue_create_host_buffer[DType.float32](
        EST_CLUSTERS * EST_FEATURES
    )
    var hl = ctx.enqueue_create_host_buffer[DType.uint32](EST_ROWS)
    ctx.synchronize()
    _fill(hx.unsafe_ptr())

    # INIT_ARRAY at centers pushed 30 off the planted ones: far enough that
    # Lloyd has to move them, near enough that the basin is unambiguous. Same
    # reasoning as `check_kmeans_fit`, so this check tests the SURFACE and not
    # the initializer's draw.
    for c in range(EST_CLUSTERS):
        for f in range(EST_FEATURES):
            hc.unsafe_ptr().unsafe_store(
                c * EST_FEATURES + f, _planted_center(c, f) + Float32(30.0)
            )

    var res = kmeans_fit(
        ctx, hx.unsafe_ptr(), EST_ROWS, EST_FEATURES, EST_CLUSTERS, hc.unsafe_ptr(), hl.unsafe_ptr(), hx.unsafe_ptr(), 0,
        max_iter=50, init=INIT_ARRAY, metric=METRIC_L2_EXPANDED,
    )

    # Every row planted in cluster c must carry the same label, and rows from
    # different planted clusters must carry different ones.
    var label_of_planted = List[Int]()
    for _ in range(EST_CLUSTERS):
        label_of_planted.append(-1)
    var wrong = 0
    for r in range(EST_ROWS):
        var planted = r % EST_CLUSTERS
        var got = Int(hl.unsafe_ptr().unsafe_load(r))
        if label_of_planted[planted] == -1:
            label_of_planted[planted] = got
        elif label_of_planted[planted] != got:
            wrong += 1
    if wrong != 0:
        raise Error(
            "kmeans_fit split a planted cluster across labels: "
            + String(wrong)
            + " of "
            + String(EST_ROWS)
            + " rows disagreed with their cluster's first row"
        )
    for a in range(EST_CLUSTERS):
        for b in range(a + 1, EST_CLUSTERS):
            if label_of_planted[a] == label_of_planted[b]:
                raise Error(
                    "kmeans_fit merged planted clusters "
                    + String(a)
                    + " and "
                    + String(b)
                    + " into label "
                    + String(label_of_planted[a])
                )

    # The centroid each label names must sit on its planted center. Jitter is
    # under 0.5, so 1.0 is loose enough to never flap and tight enough that a
    # transposed or stale centroid fails.
    for c in range(EST_CLUSTERS):
        var lab = label_of_planted[c]
        for f in range(EST_FEATURES):
            var got_v = hc.unsafe_ptr().unsafe_load(lab * EST_FEATURES + f)
            var want = _planted_center(c, f)
            var d = Float64(got_v) - Float64(want)
            if d < 0.0:
                d = -d
            if d > 1.0:
                raise Error(
                    "centroid "
                    + String(lab)
                    + " feature "
                    + String(f)
                    + " is "
                    + String(got_v)
                    + ", planted center is "
                    + String(want)
                )
    print(
        "check_kmeans_fit_recovers_planted: OK (4 clusters, 512 rows, all"
        " labels consistent, centroids within 1.0; n_iter",
        res.n_iter,
        "sum_scale",
        res.sum_scale,
        ")",
    )


def check_kmeans_fit_weight_arms_agree() raises:
    """THE CROSS-ARM CHECK, and it is the reason this file exists.

    `n_weights = 0` and `n_weights = n_samples` with every weight 1.0 are
    two DIFFERENT code paths through `kmeans_fit`: one takes the unit-weight
    bound `n_samples` directly, the other sums the caller's array, and one
    skips a host loop the other runs. They must produce bit-identical
    centroids, labels and weight_scale.

    This is the rule-8 pattern. An arm no check compares against another arm
    is an arm nobody has checked, and it is exactly how the k-NN surface's
    two shipped arms were found to disagree about output ORDER while agreeing
    as a set at every shape anyone had run.
    """
    var ctx = DeviceContext()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](
        EST_ROWS * EST_FEATURES
    )
    var hw = ctx.enqueue_create_host_buffer[DType.float32](EST_ROWS)
    var hc0 = ctx.enqueue_create_host_buffer[DType.float32](
        EST_CLUSTERS * EST_FEATURES
    )
    var hc1 = ctx.enqueue_create_host_buffer[DType.float32](
        EST_CLUSTERS * EST_FEATURES
    )
    var hl0 = ctx.enqueue_create_host_buffer[DType.uint32](EST_ROWS)
    var hl1 = ctx.enqueue_create_host_buffer[DType.uint32](EST_ROWS)
    ctx.synchronize()
    _fill(hx.unsafe_ptr())
    for r in range(EST_ROWS):
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))

    for c in range(EST_CLUSTERS):
        for f in range(EST_FEATURES):
            var v = _planted_center(c, f) + Float32(30.0)
            hc0.unsafe_ptr().unsafe_store(c * EST_FEATURES + f, v)
            hc1.unsafe_ptr().unsafe_store(c * EST_FEATURES + f, v)

    var r0 = kmeans_fit(
        ctx, hx.unsafe_ptr(), EST_ROWS, EST_FEATURES, EST_CLUSTERS, hc0.unsafe_ptr(), hl0.unsafe_ptr(), hx.unsafe_ptr(), 0,
        max_iter=50, init=INIT_ARRAY, metric=METRIC_L2_EXPANDED,
    )
    var r1 = kmeans_fit(
        ctx, hx.unsafe_ptr(), EST_ROWS, EST_FEATURES, EST_CLUSTERS, hc1.unsafe_ptr(), hl1.unsafe_ptr(), hw.unsafe_ptr(), EST_ROWS,
        max_iter=50, init=INIT_ARRAY, metric=METRIC_L2_EXPANDED,
    )

    if r0.weight_scale != r1.weight_scale:
        raise Error(
            "the two weight arms chose different weight_scale: unit "
            + String(r0.weight_scale)
            + " vs supplied "
            + String(r1.weight_scale)
            + ". The supplied array is all 1.0, so its bound must equal"
            " n_samples."
        )
    if r0.n_iter != r1.n_iter:
        raise Error(
            "the two weight arms converged on different iterations: "
            + String(r0.n_iter)
            + " vs "
            + String(r1.n_iter)
        )

    var cw = 0
    for i in range(EST_CLUSTERS * EST_FEATURES):
        if hc0.unsafe_ptr().unsafe_load(i) != hc1.unsafe_ptr().unsafe_load(i):
            cw += 1
    var lw = 0
    for i in range(EST_ROWS):
        if hl0.unsafe_ptr().unsafe_load(i) != hl1.unsafe_ptr().unsafe_load(i):
            lw += 1
    if cw != 0 or lw != 0:
        raise Error(
            "the two weight arms disagree: "
            + String(cw)
            + " of "
            + String(EST_CLUSTERS * EST_FEATURES)
            + " centroid values and "
            + String(lw)
            + " of "
            + String(EST_ROWS)
            + " labels differ, with every supplied weight equal to 1.0"
        )
    print(
        "check_kmeans_fit_weight_arms_agree: OK (unit and all-ones-supplied"
        " arms bit-identical in centroids, labels, n_iter and weight_scale)"
    )


def check_kmeans_fit_rejects_bad_shapes() raises:
    """The three refusals, because a surface that accepts nonsense is a
    surface that returns nonsense."""
    var ctx = DeviceContext()
    var hx = ctx.enqueue_create_host_buffer[DType.float32](
        EST_ROWS * EST_FEATURES
    )
    var hc = ctx.enqueue_create_host_buffer[DType.float32](
        EST_CLUSTERS * EST_FEATURES
    )
    var hl = ctx.enqueue_create_host_buffer[DType.uint32](EST_ROWS)
    ctx.synchronize()
    _fill(hx.unsafe_ptr())

    var raised = 0
    try:
        _ = kmeans_fit(ctx, hx.unsafe_ptr(), EST_ROWS, EST_FEATURES, 0, hc.unsafe_ptr(), hl.unsafe_ptr(), hx.unsafe_ptr(), 0)
    except:
        raised += 1
    try:
        # More clusters than samples.
        _ = kmeans_fit(ctx, hx.unsafe_ptr(), 2, EST_FEATURES, 4, hc.unsafe_ptr(), hl.unsafe_ptr(), hx.unsafe_ptr(), 0)
    except:
        raised += 1
    try:
        # n_weights neither 0 nor n_samples.
        _ = kmeans_fit(
            ctx, hx.unsafe_ptr(), EST_ROWS, EST_FEATURES, EST_CLUSTERS, hc.unsafe_ptr(), hl.unsafe_ptr(), hx.unsafe_ptr(), 7
        )
    except:
        raised += 1
    if raised != 3:
        raise Error(
            "kmeans_fit accepted a bad shape: only "
            + String(raised)
            + " of 3 refusals fired"
        )
    print("check_kmeans_fit_rejects_bad_shapes: OK (3 of 3 refusals fired)")
