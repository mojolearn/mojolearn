# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The `RadiusNeighbors` surface, against a host oracle and its own sabotage.

WHAT THIS FILE CLAIMS, AND WHAT IT DELIBERATELY DOES NOT.

`neighbors/checks/ball_cover_check.mojo` already proves SET EQUALITY for
the CSR the ball cover returns, and DEVIATION 551's order assertion lives
there so every caller of that helper inherits it. This file is about the two
things the radius SURFACE adds on top of that CSR, neither of which any
existing gate can see:

  1. THE TWO-CALL PROTOCOL. `radius_neighbors_count` and
     `radius_neighbors_fill` are separate calls that each rebuild the index,
     and the second is handed the first's answer as an allocation size. If
     the two ever disagreed the surface would return a truncated row that
     looked complete, so the count is asserted equal across the pair and the
     mismatch path is driven on purpose.

  2. THE DISTANCES, which are RECOMPUTED rather than stored by the search
     (`neighbors/checks/radius_distances.mojo`). Recomputation is only
     sound if it reproduces the value the radius test actually used, so every
     returned distance is compared against a host mirror of `eps_dist_sq`,
     and the comparison is EXACT under IDENTICAL. An approximate check would
     pass for an implementation that read the wrong row.

THE ORDER ASSERTION IS NOT REPEATED HERE. It belongs to the shared query
helper in `ball_cover_check.mojo` and firing it twice would only make a
future reader think there were two of them. What IS here is the surface-level
consequence: that `distances[p]` belongs to `indices[p]`, whatever order 551
put them in.
"""

from max.gpu.host import DeviceContext
from std.memory import bitcast

from checks.numerics import (
    PIN_CROSS_VENDOR,
    ftz,
    identical_mul_add,
    identical_sqrt,
)
from neighbors.estimator import radius_neighbors_count, radius_neighbors_fill


comptime RN_ROWS = 1000
comptime RN_FEATURES = 3
comptime RN_RADIUS = Float32(2.5)


def _hash01(row: Int, feature: Int) -> Float64:
    """splitmix64 on `(row, feature)`, mapped to `[0, 1)`.

    The same generator `ball_cover_check.mojo` uses, so the two files are
    looking at the same kind of point cloud rather than two unrelated ones.
    """
    var z = (
        UInt64(row) * UInt64(0x9E3779B97F4A7C15)
        + UInt64(feature + 1) * UInt64(0xBF58476D1CE4E5B9)
        + UInt64(0x94D049BB133111EB)
    )
    z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
    z = z ^ (z >> UInt64(31))
    return Float64(z >> UInt64(11)) * (1.0 / 9007199254740992.0)


def _coord(row: Int, feature: Int) -> Float32:
    return Float32(10.0 * _hash01(row, feature))


def _host_dist_sq(
    hx: MutPointer[Float32, MutUntrackedOrigin], a: Int, b: Int, d: Int
) -> Float32:
    """`common.mojo::eps_dist_sq`, SPELLED THE SAME WAY ON PURPOSE.

    `ball_cover_check.mojo`'s oracle uses plain `-` and `+=`, which is enough
    to decide SET membership on a fixture with no points on the boundary. It
    is NOT enough to decide whether a returned distance is the right bits, so
    this one goes through `ftz` and `identical_mul_add` exactly as the device
    does. Under IDENTICAL that makes the two arithmetics the same function
    and the comparison below can be exact.
    """
    var s = Float32(0.0)
    for f in range(d):
        var diff = ftz(ftz(hx.unsafe_load(a * d + f)) - ftz(hx.unsafe_load(b * d + f)))
        s = ftz(identical_mul_add(diff, diff, s))
    return s


def _fill_fixture(
    hx: MutPointer[Float32, MutUntrackedOrigin], n: Int, d: Int
):
    """The point cloud, written into a HOST BUFFER.

    A `List[Float32]`'s `unsafe_ptr()` carries `origin_of(list)`, which the
    boundary's `MutUntrackedOrigin` will not accept, so every host array in
    this file is a `ctx.enqueue_create_host_buffer` exactly as
    `ball_cover_check.mojo` does it.
    """
    for i in range(n):
        for f in range(d):
            hx.unsafe_store(i * d + f, _coord(i, f))


def _mode_name() -> String:
    @parameter
    if PIN_CROSS_VENDOR:
        return String("IDENTICAL")
    return String("FAST")


def _count_and_check(
    ctx: DeviceContext,
    hx: MutPointer[Float32, MutUntrackedOrigin],
    ia: MutPointer[Int32, MutUntrackedOrigin],
    n: Int,
    d: Int,
    radius: Float32,
    label: String,
) raises -> Int:
    """Pass one, plus the CSR invariants that hold before pass two runs."""
    var nnz = radius_neighbors_count(ctx, hx, n, hx, n, d, radius, ia)
    if nnz <= 0:
        raise Error(
            label + ": the radius query returned no edges at all. Every point"
            " is inside its own radius, so the count can never be below "
            + String(n)
        )
    if Int(ia.unsafe_load(n)) != nnz:
        raise Error(
            label + ": indptr[n] is " + String(ia.unsafe_load(n))
            + " but the count call returned " + String(nnz)
        )
    for i in range(n):
        if ia.unsafe_load(i + 1) < ia.unsafe_load(i):
            raise Error(label + ": indptr is not monotone at row " + String(i))
    return nnz


def _fill_and_check(
    ctx: DeviceContext,
    hx: MutPointer[Float32, MutUntrackedOrigin],
    ia: MutPointer[Int32, MutUntrackedOrigin],
    cols: MutPointer[Int32, MutUntrackedOrigin],
    dists: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    d: Int,
    radius: Float32,
    nnz: Int,
    return_sqrt: Bool,
    label: String,
) raises:
    """Pass two, plus the one invariant that spans the pair."""
    var nnz2 = radius_neighbors_fill(
        ctx, hx, n, hx, n, d, radius, ia, cols, dists, nnz, return_sqrt
    )
    if nnz2 != nnz:
        raise Error(
            label + ": the two calls disagree. count said " + String(nnz)
            + " and fill said " + String(nnz2)
            + ", on identical inputs. The surface's whole allocation contract"
            " is that they cannot."
        )


def check_radius_neighbors_matches_host() raises:
    """Members and DISTANCES, per cell, against the host."""
    var ctx = DeviceContext()
    var n = RN_ROWS
    var d = RN_FEATURES
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var ia = ctx.enqueue_create_host_buffer[DType.int32](n + 1)
    ctx.synchronize()
    _fill_fixture(hx.unsafe_ptr(), n, d)

    var nnz = _count_and_check(
        ctx, hx.unsafe_ptr(), ia.unsafe_ptr(), n, d, RN_RADIUS,
        "radius_neighbors",
    )
    var cols = ctx.enqueue_create_host_buffer[DType.int32](nnz)
    var dists = ctx.enqueue_create_host_buffer[DType.float32](nnz)
    ctx.synchronize()
    _fill_and_check(
        ctx, hx.unsafe_ptr(), ia.unsafe_ptr(), cols.unsafe_ptr(),
        dists.unsafe_ptr(), n, d, RN_RADIUS, nnz, True, "radius_neighbors",
    )

    var r2 = RN_RADIUS * RN_RADIUS
    var seen = List[Int32]()
    for _ in range(n):
        seen.append(Int32(-1))

    var exact_hits = 0
    var worst_ulp = 0
    for i in range(n):
        var start = Int(ia.unsafe_ptr().unsafe_load(i))
        var end = Int(ia.unsafe_ptr().unsafe_load(i + 1))
        for p in range(start, end):
            var c = Int(cols.unsafe_ptr().unsafe_load(p))
            if c < 0 or c >= n:
                raise Error(
                    "radius_neighbors: row " + String(i) + " column "
                    + String(c) + " is out of range"
                )
            if seen[c] == Int32(i):
                raise Error(
                    "radius_neighbors: row " + String(i) + " lists column "
                    + String(c) + " twice"
                )
            seen[c] = Int32(i)

            # THE DISTANCE, AGAINST THE HOST, AT THE SAME CSR POSITION. This
            # is what makes `distances[p]` provably the distance to
            # `indices[p]` rather than to some other member of the row.
            var want = identical_sqrt(_host_dist_sq(hx.unsafe_ptr(), i, c, d))
            var got = dists.unsafe_ptr().unsafe_load(p)
            if got == want:
                exact_hits += 1
            else:
                var ub = Int(bitcast[DType.int32](want))
                var gb = Int(bitcast[DType.int32](got))
                var diff = gb - ub
                if diff < 0:
                    diff = -diff
                if diff > worst_ulp:
                    worst_ulp = diff

        var expected = 0
        for j in range(n):
            var inside = _host_dist_sq(hx.unsafe_ptr(), i, j, d) <= r2
            if inside:
                expected += 1
            var got_it = seen[j] == Int32(i)
            if got_it and not inside:
                raise Error(
                    "radius_neighbors: row " + String(i) + " returned point "
                    + String(j) + " which is NOT within the radius"
                )
            if inside and not got_it:
                raise Error(
                    "radius_neighbors: row " + String(i) + " MISSED point "
                    + String(j)
                    + ", which brute force puts inside the radius. The index"
                    " pruned a ball it had to look in."
                )
        if expected != end - start:
            raise Error(
                "radius_neighbors: row " + String(i) + " has "
                + String(end - start) + " neighbors, brute force says "
                + String(expected)
            )

    # THE DISTANCE VERDICT IS MODE-SPLIT, and the split is the contract.
    @parameter
    if PIN_CROSS_VENDOR:
        if exact_hits != nnz:
            raise Error(
                "radius_neighbors (IDENTICAL): " + String(nnz - exact_hits)
                + " of " + String(nnz) + " distances differ from the host"
                " mirror of eps_dist_sq, worst " + String(worst_ulp)
                + " ulp. Under IDENTICAL both sides are ftz +"
                " identical_mul_add + portable_sqrtf, which is one arithmetic"
                " and must give one answer. A recomputed distance that is"
                " merely CLOSE is a recomputation against the wrong operands."
            )
    else:
        if worst_ulp > 2:
            raise Error(
                "radius_neighbors (FAST): a returned distance is "
                + String(worst_ulp) + " ulp from the host's. FAST does not"
                " promise bits, but a gap this size is a wrong row rather"
                " than a contraction difference."
            )
    print(
        "radius_neighbors: " + String(n) + " rows, " + String(nnz)
        + " edges, every member and every distance matches brute force ["
        + _mode_name() + "]; " + String(exact_hits) + "/" + String(nnz)
        + " distances bit-exact, worst " + String(worst_ulp) + " ulp"
    )


def check_radius_neighbors_squared_arm() raises:
    """`return_sqrt=False` returns d^2, and it is the SAME d^2."""
    var ctx = DeviceContext()
    var n = RN_ROWS
    var d = RN_FEATURES
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var ia = ctx.enqueue_create_host_buffer[DType.int32](n + 1)
    ctx.synchronize()
    _fill_fixture(hx.unsafe_ptr(), n, d)

    var nnz = _count_and_check(
        ctx, hx.unsafe_ptr(), ia.unsafe_ptr(), n, d, RN_RADIUS,
        "radius_neighbors/squared",
    )
    var cols = ctx.enqueue_create_host_buffer[DType.int32](nnz)
    var dsq = ctx.enqueue_create_host_buffer[DType.float32](nnz)
    ctx.synchronize()
    _fill_and_check(
        ctx, hx.unsafe_ptr(), ia.unsafe_ptr(), cols.unsafe_ptr(),
        dsq.unsafe_ptr(), n, d, RN_RADIUS, nnz, False,
        "radius_neighbors/squared",
    )

    var r2 = RN_RADIUS * RN_RADIUS
    var bad = 0
    for p in range(nnz):
        if dsq.unsafe_ptr().unsafe_load(p) > r2:
            bad += 1
    if bad != 0:
        raise Error(
            "radius_neighbors/squared: " + String(bad) + " of " + String(nnz)
            + " squared distances exceed radius^2, so return_sqrt=False is"
            " not returning the value the radius test compared"
        )

    var mism = 0
    for i in range(n):
        for p in range(
            Int(ia.unsafe_ptr().unsafe_load(i)),
            Int(ia.unsafe_ptr().unsafe_load(i + 1)),
        ):
            var want = _host_dist_sq(
                hx.unsafe_ptr(), i, Int(cols.unsafe_ptr().unsafe_load(p)), d
            )
            if dsq.unsafe_ptr().unsafe_load(p) != want:
                mism += 1

    @parameter
    if PIN_CROSS_VENDOR:
        if mism != 0:
            raise Error(
                "radius_neighbors/squared (IDENTICAL): " + String(mism)
                + " of " + String(nnz) + " squared distances differ from the"
                " host mirror. Without the sqrt there is nothing left that"
                " could round differently."
            )
    print(
        "radius_neighbors: return_sqrt=False returns d^2, all " + String(nnz)
        + " inside radius^2, " + String(nnz - mism) + " bit-exact against the"
        " host [" + _mode_name() + "]"
    )


def check_radius_neighbors_reach_by_sabotage() raises:
    """Shrink the radius. If the edge count does not move, nothing ran."""
    var ctx = DeviceContext()
    var n = RN_ROWS
    var d = RN_FEATURES
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var ia = ctx.enqueue_create_host_buffer[DType.int32](n + 1)
    var ia2 = ctx.enqueue_create_host_buffer[DType.int32](n + 1)
    ctx.synchronize()
    _fill_fixture(hx.unsafe_ptr(), n, d)

    var before = _count_and_check(
        ctx, hx.unsafe_ptr(), ia.unsafe_ptr(), n, d, RN_RADIUS,
        "sabotage/before",
    )
    var small = RN_RADIUS * Float32(0.4)
    var after = _count_and_check(
        ctx, hx.unsafe_ptr(), ia2.unsafe_ptr(), n, d, small, "sabotage/after"
    )
    if after >= before:
        raise Error(
            "radius_neighbors sabotage: shrinking the radius from "
            + String(RN_RADIUS) + " to " + String(small) + " left "
            + String(after) + " edges against " + String(before)
            + ". The radius argument is not reaching the kernel, so every"
            " assertion above is about a fixture rather than about a search."
        )

    var cols2 = ctx.enqueue_create_host_buffer[DType.int32](after)
    var dists2 = ctx.enqueue_create_host_buffer[DType.float32](after)
    ctx.synchronize()
    _fill_and_check(
        ctx, hx.unsafe_ptr(), ia2.unsafe_ptr(), cols2.unsafe_ptr(),
        dists2.unsafe_ptr(), n, d, small, after, True, "sabotage/after",
    )
    for p in range(after):
        var dd = dists2.unsafe_ptr().unsafe_load(p)
        if dd > small * Float32(1.0001):
            raise Error(
                "radius_neighbors sabotage: the shrunken query returned a"
                " point at distance " + String(dd) + ", outside its own"
                " radius " + String(small)
            )
    print(
        "radius_neighbors: sabotage reached; radius x0.4 took edges from "
        + String(before) + " to " + String(after) + " [" + _mode_name() + "]"
    )


def check_radius_neighbors_refuses_short_allocation() raises:
    """The truncation the two-call protocol exists to make impossible."""
    var ctx = DeviceContext()
    var n = RN_ROWS
    var d = RN_FEATURES
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var ia = ctx.enqueue_create_host_buffer[DType.int32](n + 1)
    ctx.synchronize()
    _fill_fixture(hx.unsafe_ptr(), n, d)

    var nnz = _count_and_check(
        ctx, hx.unsafe_ptr(), ia.unsafe_ptr(), n, d, RN_RADIUS,
        "radius_neighbors/short",
    )
    var cols = ctx.enqueue_create_host_buffer[DType.int32](nnz)
    var dists = ctx.enqueue_create_host_buffer[DType.float32](nnz)
    ctx.synchronize()

    var refused = False
    try:
        _ = radius_neighbors_fill(
            ctx, hx.unsafe_ptr(), n, hx.unsafe_ptr(), n, d, RN_RADIUS,
            ia.unsafe_ptr(), cols.unsafe_ptr(), dists.unsafe_ptr(),
            nnz - 1, True,
        )
    except e:
        # THE REFUSAL MUST CITE ITS OWN REASON. A gate that accepts any
        # exception passes when the code fails for an unrelated reason, which
        # is how two sabotages in this tree went green by accident.
        if String(e).find("the caller allocated for") < 0:
            raise Error(
                "radius_neighbors: a short allocation raised, but not the"
                " allocation refusal. Got: " + String(e)
            )
        refused = True
    if not refused:
        raise Error(
            "radius_neighbors: fill accepted an nnz_capacity one short of the"
            " count and returned anyway. That is a truncated row that looks"
            " like a complete one, which is the exact failure the two-call"
            " protocol exists to prevent."
        )

    var refused_r = False
    try:
        _ = radius_neighbors_count(
            ctx, hx.unsafe_ptr(), n, hx.unsafe_ptr(), n, d, Float32(-1.0),
            ia.unsafe_ptr(),
        )
    except e2:
        if String(e2).find("radius must be positive") < 0:
            raise Error(
                "radius_neighbors: a negative radius raised, but not the"
                " radius refusal. Got: " + String(e2)
            )
        refused_r = True
    if not refused_r:
        raise Error("radius_neighbors: a negative radius was accepted")

    print(
        "radius_neighbors: a short allocation and a negative radius are both"
        " REFUSED BY NAME [" + _mode_name() + "]"
    )
