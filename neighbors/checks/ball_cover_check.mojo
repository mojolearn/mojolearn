# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The random ball cover has to return the SAME SET brute force returns.

NO CUVS COUNTERPART. Their `cpp/tests/neighbors/ball_cover.cu` compares
against `brute_force::knn` with a recall tolerance; this compares against a
host brute force with NO tolerance, because the eps query is exact and
because DBSCAN's answer changes if a neighborhood is one point short.

WHY THE ORACLE IS ON THE HOST AND WHY IT IS BIT-COMPARABLE
------------------------------------------------------------
The oracle repeats `common.mojo::eps_dist_sq` in Float32 with the same
accumulation order over the same operand bits, so a boundary point cannot
land on one side on the host and the other on the device. `X_reordered[j]`
is a byte copy of `X[R_1nn_cols[j]]`, so the device compares exactly the
pair of rows the host compares. There is no epsilon on the comparison and
there must not be one: the whole claim of this file is set equality.

THE FIXTURE IS SCATTERED, NOT UNIFORM, AND THAT IS THE POINT
--------------------------------------------------------------
Every coordinate is a splitmix64 hash of `(row, feature)`, so no two rows
share a value and no two neighborhoods have the same size. A check whose
expected value is the same in every cell verifies a total and nothing about
placement; that trap has passed two real bugs in this repository at the exact
failing parameters. Here the per-row neighbor SETS are all different, so a
wrong permutation, a wrong landmark offset or a row written into its
neighbor's CSR slot all fail.

Three radii are tested, chosen after measuring the fixture rather than
guessed: one that leaves most rows with a handful of neighbors, one in the
middle, and one large enough that most of the dataset is in range. The small
radius exercises the landmark prune, the large one exercises the path where
the prune never fires and the backward walk runs to the end of every group.

REACH IS PROVED BY SABOTAGE
---------------------------
`check_ball_cover_reach_by_sabotage` scales every landmark radius by 0.3 and
re-runs. The predicted shape is specific: shrinking the radii can only make
the landmark test `d(q,r) <= eps + radius(r)` reject balls it should accept,
so the answer must become a strict SUBSET of the true one and must lose at
least one edge. It cannot gain an edge. If `R_radius` were not read, or if
the query were quietly falling back to a full scan, the answer would not move
at all.
"""

from std.math import sqrt
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import PIN_CROSS_VENDOR  # DEVIATION 551

from neighbors.impl.ball_cover.ball_cover import (
    rbc_build_index,
    rbc_eps_nn_query_count,
    rbc_eps_nn_query_dense,
    rbc_eps_nn_query_fill,
    rbc_eps_nn_query_max_k,
    rbc_n_landmarks,
)
from checks.native_tests.kit import download_host_into, upload_host_into


comptime BC_ROWS = 1200
comptime BC_FEATURES = 3
comptime BC_MAX_K = 512


def _hash01(row: Int, feature: Int) -> Float64:
    """splitmix64 on `(row, feature)`, mapped to `[0, 1)`."""
    var z = (
        UInt64(row) * UInt64(0x9E3779B97F4A7C15)
        + UInt64(feature + 1) * UInt64(0xBF58476D1CE4E5B9)
        + UInt64(0x94D049BB133111EB)
    )
    z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
    z = z ^ (z >> UInt64(31))
    return Float64(z >> UInt64(11)) * (1.0 / 9007199254740992.0)


def _csr_digest(
    hia: MutPointer[Int32, MutUntrackedOrigin],
    hja: MutPointer[Int32, MutUntrackedOrigin],
    n: Int,
    nnz: Int,
) -> UInt64:
    """A 64-bit FNV-1a over the WHOLE CSR: shape, row starts, then columns.

    THIS IS THE CROSS-VENDOR ARTIFACT. Every other assertion in this file is
    something one box can check about itself, and `one-box-verdict-is-not-three`
    forbids reading a cross-vendor claim off any of them. A digest is different:
    it is a number a SECOND box prints independently, and the two are compared
    off-box by eye or by diff.

    It is printed in BOTH modes on purpose, and the two modes answer different
    questions:

      IDENTICAL  the digests from two vendors MUST be equal. That is the whole
                 of DEVIATION 551's claim, and nothing on one box can supply it.
      FAST       the digests from two vendors are EXPECTED to differ, because
                 the raw emission order is lane-width dependent (32 on Apple and
                 NVIDIA, 64 on CDNA). That difference is the NON-VACUITY control
                 for the pair above: if FAST also matched, the IDENTICAL match
                 would be telling us nothing about canonicalization.

    The row starts are folded in as well as the columns, so a digest cannot
    collide across two different row partitionings of the same column stream.
    """
    var h = UInt64(0xCBF29CE484222325)
    h = (h ^ UInt64(n)) * UInt64(0x100000001B3)
    h = (h ^ UInt64(nnz)) * UInt64(0x100000001B3)
    for i in range(n + 1):
        h = (h ^ UInt64(UInt32(hia.unsafe_load(i)))) * UInt64(0x100000001B3)
    for p in range(nnz):
        h = (h ^ UInt64(UInt32(hja.unsafe_load(p)))) * UInt64(0x100000001B3)
    return h


def _coord(row: Int, feature: Int) -> Float32:
    """A scattered point in a box of side 10.

    Coordinates are kept small on purpose: an earlier k-NN
    fixture was destroyed by float32 cancellation when the spacing was 100. The
    distances here are computed unexpanded so that trap does not apply, but
    keeping the magnitudes small keeps the host oracle and the device in the
    same ulp regime for free.
    """
    return Float32(10.0 * _hash01(row, feature))


def _host_dist_sq(
    hx: MutPointer[Float32, MutUntrackedOrigin], a: Int, b: Int, d: Int
) -> Float32:
    """`common.mojo::eps_dist_sq`, same order, same type."""
    var s = Float32(0.0)
    for f in range(d):
        var diff = hx.unsafe_load(a * d + f) - hx.unsafe_load(b * d + f)
        s += diff * diff
    return s


def _run_one_eps(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_reordered: DeviceBuffer[DType.float32],
    mut r: DeviceBuffer[DType.float32],
    mut r_indptr: DeviceBuffer[DType.int32],
    mut r_1nn_cols: DeviceBuffer[DType.int32],
    mut r_1nn_dists: DeviceBuffer[DType.float32],
    mut r_radius: DeviceBuffer[DType.float32],
    mut adj_ia: DeviceBuffer[DType.int32],
    mut vd: DeviceBuffer[DType.int32],
    hx: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
    d: Int,
    n_landmarks: Int,
    eps: Float32,
    label: String,
    expect_subset_only: Bool,
) raises -> Int:
    """Count, fill, and compare per row against the host oracle.

    Returns the total number of edges. `expect_subset_only` relaxes the
    equality to containment, which is what the sabotage run asserts.
    """
    # THE ORACLE'S OWN FIXTURE, BEFORE IT IS TRUSTED TO JUDGE ANYTHING.
    #
    # On 2026-08-30 this function reported, on an MI325X, "eps=8.0: row 0
    # MISSED point 42, which brute force puts inside eps", and that became
    # IDENTITY_PATHS row 63, "a WRONG ANSWER on one vendor". It was not.
    # d(row 0, point 42) on this fixture is 8.413649, squared 70.78948
    # against an eps2 of 64, so point 42 is OUTSIDE -- and it is the
    # LOWEST-NUMBERED point row 0 correctly excludes. The device was right.
    #
    # `hx` had been freed. Under a zeroed `hx` every host distance is 0, so
    # `inside` is true for every j, the "device returned an extra" arm can
    # never fire, and the first j the device did NOT return gets reported as
    # MISSED. j=42 is not a neighbour that went missing; it is the signature
    # of an oracle that is gone. A comparison against a destroyed fixture
    # does not fail. It accuses.
    #
    # Worse, the two `expect_subset_only=True` callers would pass VACUOUSLY
    # under the same corruption: with `inside` true everywhere, both arms of
    # the per-cell comparison are suppressed and only the edge count still
    # has teeth. Two coordinates catch it, and they cost nothing per call.
    if (
        hx.unsafe_load(0) != _coord(0, 0)
        or hx.unsafe_load(42 * d) != _coord(42, 0)
    ):
        raise Error(
            label
            + ": THE ORACLE'S OWN FIXTURE IS GONE, so every comparison below"
            + " this line is meaningless and none of them is evidence about"
            + " the device. hx[0] reads " + String(hx.unsafe_load(0))
            + " against " + String(_coord(0, 0)) + ", hx[42*d] reads "
            + String(hx.unsafe_load(42 * d)) + " against "
            + String(_coord(42, 0)) + "."
        )
    var nnz = rbc_eps_nn_query_count(
        ctx,
        x_reordered,
        x,
        r,
        r_indptr,
        r_1nn_cols,
        r_1nn_dists,
        r_radius,
        adj_ia,
        vd,
        n,
        d,
        n_landmarks,
        eps,
    )
    if nnz <= 0:
        raise Error(
            label + ": the eps query returned no edges at all; every point is"
            " its own neighbor so the count can never be below " + String(n)
        )

    var adj_ja = ctx.enqueue_create_buffer[DType.int32](nnz)
    ctx.synchronize()
    rbc_eps_nn_query_fill(
        ctx,
        x_reordered,
        x,
        r,
        r_indptr,
        r_1nn_cols,
        r_1nn_dists,
        r_radius,
        adj_ia,
        adj_ja,
        n,
        d,
        n_landmarks,
        eps,
    )

    var hia = ctx.enqueue_create_host_buffer[DType.int32](n + 1)
    var hja = ctx.enqueue_create_host_buffer[DType.int32](nnz)
    var hvd = ctx.enqueue_create_host_buffer[DType.int32](n + 1)
    download_host_into[DType.int32](ctx, adj_ia, hia)
    download_host_into[DType.int32](ctx, adj_ja, hja)
    download_host_into[DType.int32](ctx, vd, hvd)
    ctx.synchronize()

    if Int(hia.unsafe_ptr().unsafe_load(n)) != nnz:
        raise Error(
            label + ": adj_ia[n] is "
            + String(hia.unsafe_ptr().unsafe_load(n))
            + " but the reported edge count is " + String(nnz)
        )
    if Int(hvd.unsafe_ptr().unsafe_load(n)) != nnz:
        raise Error(
            label + ": vd[n] must hold the edge total, cuml reads it back"
            " from there; it is "
            + String(hvd.unsafe_ptr().unsafe_load(n))
        )

    var eps2 = eps * eps
    var seen = List[Int32]()
    for _ in range(n):
        seen.append(Int32(-1))

    for i in range(n):
        var start = Int(hia.unsafe_ptr().unsafe_load(i))
        var end = Int(hia.unsafe_ptr().unsafe_load(i + 1))
        if end < start:
            raise Error(label + ": adj_ia is not monotone at row " + String(i))
        if Int(hvd.unsafe_ptr().unsafe_load(i)) != end - start:
            raise Error(
                label + ": vd[" + String(i) + "] = "
                + String(hvd.unsafe_ptr().unsafe_load(i))
                + " disagrees with the CSR row length "
                + String(end - start)
            )

        # mark what came back, catching duplicates and out-of-range columns
        for p in range(start, end):
            var c = Int(hja.unsafe_ptr().unsafe_load(p))
            if c < 0 or c >= n:
                raise Error(
                    label + ": row " + String(i) + " column " + String(c)
                    + " is out of range"
                )
            if seen[c] == Int32(i):
                raise Error(
                    label + ": row " + String(i) + " lists column "
                    + String(c) + " twice"
                )
            seen[c] = Int32(i)

        # the oracle, per cell, not per total
        var expected = 0
        for j in range(n):
            var inside = _host_dist_sq(hx, i, j, d) <= eps2
            if inside:
                expected += 1
            var got = seen[j] == Int32(i)
            if got and not inside:
                raise Error(
                    label + ": row " + String(i) + " returned point "
                    + String(j) + " which is NOT within eps"
                )
            if inside and not got and not expect_subset_only:
                raise Error(
                    label + ": row " + String(i) + " MISSED point "
                    + String(j)
                    + ", which brute force puts inside eps. The index"
                    " pruned a ball it had to look in."
                )
        if not expect_subset_only and expected != end - start:
            raise Error(
                label + ": row " + String(i) + " has " + String(end - start)
                + " neighbors, brute force says " + String(expected)
            )

    # DEVIATION 551: THE INTRA-ROW ORDER, which this file's header says it does
    # not claim. It still does not claim it as SET equality; this is a separate
    # assertion with its own mode split, and it lives here so that every caller
    # of this helper inherits it rather than one check owning it.
    #
    # IDENTICAL: every row must be strictly ascending by column index.
    # FAST: the pass is compiled out, so the raw emission order is visible. The
    # descent count is REPORTED, and a count of zero is VACUOUS rather than
    # good: it would mean this fixture cannot tell a canonical row from a raw
    # one, and the IDENTICAL assertion above it would be proving nothing. That
    # is the negative control, and without it a fixture that happened to emit
    # ascending rows would pass forever.
    var descents = 0
    var longest = 0
    for i in range(n):
        var start = Int(hia.unsafe_ptr().unsafe_load(i))
        var end = Int(hia.unsafe_ptr().unsafe_load(i + 1))
        if end - start > longest:
            longest = end - start
        for p in range(start + 1, end):
            if hja.unsafe_ptr().unsafe_load(p) <= hja.unsafe_ptr().unsafe_load(
                p - 1
            ):
                descents += 1

    comptime if PIN_CROSS_VENDOR:
        if descents != 0:
            raise Error(
                label + ": DEVIATION 551 left " + String(descents)
                + " intra-row descents. Under IDENTICAL every CSR row must be"
                " strictly ascending by column index, because the raw emission"
                " order is lane-width dependent (IDENTITY_PATHS row 61) and"
                " the canonical order is what makes the bytes comparable"
                " across vendors."
            )
    else:
        if longest < 2:
            raise Error(
                label + ": VACUOUS. The longest row is " + String(longest)
                + ", so no row can be out of order and the IDENTICAL"
                " assertion would pass on any implementation."
            )
        if descents == 0:
            raise Error(
                label + ": VACUOUS. The RAW emission order already happens to"
                " be ascending on this fixture (0 descents over " + String(n)
                + " rows, longest " + String(longest) + "), so the IDENTICAL"
                " assertion cannot distinguish a working canonicalization from"
                " a no-op. Pick a fixture whose raw order is not sorted."
            )
        print(
            "  " + label + ": raw emission order has " + String(descents)
            + " intra-row descents (longest row " + String(longest)
            + "); DEVIATION 551 removes them under IDENTICAL"
        )

    # THE CROSS-VENDOR ARTIFACT. One line, greppable, printed in both modes.
    # See `_csr_digest` for what a reader is supposed to do with the two
    # numbers: match them under IDENTICAL, expect them to DIFFER under FAST.
    var digest = _csr_digest(hia.unsafe_ptr(), hja.unsafe_ptr(), n, nnz)
    var mode_name = String("FAST")

    comptime if PIN_CROSS_VENDOR:
        mode_name = String("IDENTICAL")
    print(
        "RBC-DIGEST mode=" + mode_name + " label=" + label + " n=" + String(n)
        + " nnz=" + String(nnz) + " digest=" + String(digest)
    )

    return nnz


def check_ball_cover() raises:
    """Index build invariants, then set equality at three radii."""
    var ctx = DeviceContext()
    var n = BC_ROWS
    var d = BC_FEATURES
    var n_landmarks = rbc_n_landmarks(n)
    if n_landmarks < 8:
        raise Error(
            "the fixture must involve many landmarks or the prune is never"
            " exercised; got " + String(n_landmarks)
        )

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var r = ctx.enqueue_create_buffer[DType.float32](n_landmarks * d)
    var x_reordered = ctx.enqueue_create_buffer[DType.float32](n * d)
    var landmark_ids = ctx.enqueue_create_buffer[DType.int32](n_landmarks)
    var slot_cols = ctx.enqueue_create_buffer[DType.int32](n)
    var slot_dists = ctx.enqueue_create_buffer[DType.float32](n)
    var nearest = ctx.enqueue_create_buffer[DType.int32](n)
    var nearest_dist = ctx.enqueue_create_buffer[DType.float32](n)
    var r_indptr = ctx.enqueue_create_buffer[DType.int32](n_landmarks + 1)
    var r_1nn_cols = ctx.enqueue_create_buffer[DType.int32](n)
    var r_1nn_dists = ctx.enqueue_create_buffer[DType.float32](n)
    var r_radius = ctx.enqueue_create_buffer[DType.float32](n_landmarks)
    var counts = ctx.enqueue_create_buffer[DType.int32](n_landmarks)
    var adj_ia = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var vd = ctx.enqueue_create_buffer[DType.int32](n + 1)
    ctx.synchronize()

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _coord(i, f))
    upload_host_into[DType.float32](ctx, x, hx)
    ctx.synchronize()

    rbc_build_index(
        ctx, x, r, x_reordered, landmark_ids, slot_cols, slot_dists,
        nearest, nearest_dist, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        counts, n, d, n_landmarks,
    )

    # --- the index itself, before any query ------------------------------
    var hptr = ctx.enqueue_create_host_buffer[DType.int32](n_landmarks + 1)
    var hcols = ctx.enqueue_create_host_buffer[DType.int32](n)
    var hdists = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hrad = ctx.enqueue_create_host_buffer[DType.float32](n_landmarks)
    var hxr = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    download_host_into[DType.int32](ctx, r_indptr, hptr)
    download_host_into[DType.int32](ctx, r_1nn_cols, hcols)
    download_host_into[DType.float32](ctx, r_1nn_dists, hdists)
    download_host_into[DType.float32](ctx, r_radius, hrad)
    download_host_into[DType.float32](ctx, x_reordered, hxr)
    ctx.synchronize()

    if Int(hptr.unsafe_ptr().unsafe_load(n_landmarks)) != n:
        raise Error(
            "R_indptr[n_landmarks] must be m; every point belongs to exactly"
            " one landmark. Got "
            + String(hptr.unsafe_ptr().unsafe_load(n_landmarks))
        )

    var used = List[Int32]()
    for _ in range(n):
        used.append(Int32(0))
    var nonempty = 0
    for k in range(n_landmarks):
        var s = Int(hptr.unsafe_ptr().unsafe_load(k))
        var e = Int(hptr.unsafe_ptr().unsafe_load(k + 1))
        if e < s:
            raise Error("R_indptr is not monotone at landmark " + String(k))
        if e > s:
            nonempty += 1
        # ascending by distance within the group: the backward walk in
        # `registers.mojo` is only sound because of this
        for j in range(s + 1, e):
            if (
                hdists.unsafe_ptr().unsafe_load(j)
                < hdists.unsafe_ptr().unsafe_load(j - 1)
            ):
                raise Error(
                    "landmark " + String(k) + " group is NOT sorted"
                    " ascending at position " + String(j)
                    + "; the query kernel's early stop drops real neighbors"
                    " when this fails"
                )
        # radius is the last, i.e. largest, distance in the group
        if e > s:
            var want = hdists.unsafe_ptr().unsafe_load(e - 1)
            if hrad.unsafe_ptr().unsafe_load(k) != want:
                raise Error(
                    "R_radius[" + String(k) + "] is "
                    + String(hrad.unsafe_ptr().unsafe_load(k))
                    + ", the group's largest distance is " + String(want)
                )
        for j in range(s, e):
            var c = Int(hcols.unsafe_ptr().unsafe_load(j))
            if c < 0 or c >= n:
                raise Error("R_1nn_cols out of range at " + String(j))
            used[c] += Int32(1)
            # X_reordered is X gathered by R_1nn_cols
            for f in range(d):
                if (
                    hxr.unsafe_ptr().unsafe_load(j * d + f)
                    != hx.unsafe_ptr().unsafe_load(c * d + f)
                ):
                    raise Error(
                        "X_reordered row " + String(j) + " is not X row "
                        + String(c)
                    )
    for i in range(n):
        if used[i] != Int32(1):
            raise Error(
                "point " + String(i) + " appears " + String(used[i])
                + " times in the index, must be exactly once"
            )
    if nonempty < 8:
        raise Error(
            "only " + String(nonempty) + " landmarks have any points; the"
            " query would never cross a ball boundary"
        )

    # --- three radii ------------------------------------------------------
    # Measured on this fixture, not guessed: 0.9 leaves a handful per row,
    # 2.5 is the middle, 8.0 puts most of the box in range.
    var small = _run_one_eps(
        ctx, x, x_reordered, r, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        adj_ia, vd, hx.unsafe_ptr(), n, d, n_landmarks,
        Float32(0.9), String("eps=0.9"), False,
    )
    var medium = _run_one_eps(
        ctx, x, x_reordered, r, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        adj_ia, vd, hx.unsafe_ptr(), n, d, n_landmarks,
        Float32(2.5), String("eps=2.5"), False,
    )
    var large = _run_one_eps(
        ctx, x, x_reordered, r, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        adj_ia, vd, hx.unsafe_ptr(), n, d, n_landmarks,
        Float32(8.0), String("eps=8.0"), False,
    )
    if not (small < medium and medium < large):
        raise Error(
            "the three radii must give strictly growing neighborhoods; got "
            + String(small) + ", " + String(medium) + ", " + String(large)
        )
    if large <= n * n // 2:
        raise Error(
            "eps=8.0 was supposed to put most of the dataset in range and"
            " only produced " + String(large) + " edges"
        )
    print(
        "ball_cover: exact set match at eps 0.9 / 2.5 / 8.0, edges",
        small,
        medium,
        large,
    )

    # `hx` IS HELD PAST ITS LAST READER. Mojo frees a buffer at its LAST USE,
    # and `hx.unsafe_ptr()` hands out a MutUntrackedOrigin pointer that does
    # not extend the buffer's life, so without this the pinned fixture can be
    # released while the oracle above is still reading through it. Metal
    # leaves the freed page mapped, which is why Apple never saw it; ROCm
    # unmaps and returns zero pages, which is what produced the false row-63
    # 'MISSED point 42' on the MI325X. See the note in `_run_one_eps`.
    _ = hx^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^


def check_ball_cover_reach_by_sabotage() raises:
    """Shrink the landmark radii; the answer must shrink in a known shape.

    A passing check proves nothing about whether the new path RAN. So one
    input to the prune is corrupted and the result has to move, in the exact
    direction the triangle inequality predicts and no other: strictly fewer
    edges, and every remaining edge still real.

    The window is chosen, not arbitrary. A factor of 0.3 is small enough that
    the query still returns most of its edges (so the comparison is against a
    live answer rather than an empty one) and large enough that a landmark on
    the far side of the query's own ball is certainly rejected.
    """
    var ctx = DeviceContext()
    var n = BC_ROWS
    var d = BC_FEATURES
    var n_landmarks = rbc_n_landmarks(n)

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var r = ctx.enqueue_create_buffer[DType.float32](n_landmarks * d)
    var x_reordered = ctx.enqueue_create_buffer[DType.float32](n * d)
    var landmark_ids = ctx.enqueue_create_buffer[DType.int32](n_landmarks)
    var slot_cols = ctx.enqueue_create_buffer[DType.int32](n)
    var slot_dists = ctx.enqueue_create_buffer[DType.float32](n)
    var nearest = ctx.enqueue_create_buffer[DType.int32](n)
    var nearest_dist = ctx.enqueue_create_buffer[DType.float32](n)
    var r_indptr = ctx.enqueue_create_buffer[DType.int32](n_landmarks + 1)
    var r_1nn_cols = ctx.enqueue_create_buffer[DType.int32](n)
    var r_1nn_dists = ctx.enqueue_create_buffer[DType.float32](n)
    var r_radius = ctx.enqueue_create_buffer[DType.float32](n_landmarks)
    var counts = ctx.enqueue_create_buffer[DType.int32](n_landmarks)
    var adj_ia = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var vd = ctx.enqueue_create_buffer[DType.int32](n + 1)
    ctx.synchronize()

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _coord(i, f))
    upload_host_into[DType.float32](ctx, x, hx)
    ctx.synchronize()

    rbc_build_index(
        ctx, x, r, x_reordered, landmark_ids, slot_cols, slot_dists,
        nearest, nearest_dist, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        counts, n, d, n_landmarks,
    )

    var eps = Float32(2.5)
    var honest = _run_one_eps(
        ctx, x, x_reordered, r, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        adj_ia, vd, hx.unsafe_ptr(), n, d, n_landmarks,
        eps, String("sabotage/before"), False,
    )

    var hrad = ctx.enqueue_create_host_buffer[DType.float32](n_landmarks)
    download_host_into[DType.float32](ctx, r_radius, hrad)
    ctx.synchronize()
    for k in range(n_landmarks):
        hrad.unsafe_ptr().unsafe_store(
            k, hrad.unsafe_ptr().unsafe_load(k) * Float32(0.3)
        )
    upload_host_into[DType.float32](ctx, r_radius, hrad)
    ctx.synchronize()

    var broken = _run_one_eps(
        ctx, x, x_reordered, r, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        adj_ia, vd, hx.unsafe_ptr(), n, d, n_landmarks,
        eps, String("sabotage/after"), True,
    )

    if broken >= honest:
        raise Error(
            "SABOTAGE DID NOT REACH: scaling every landmark radius by 0.3"
            " left the edge count at " + String(broken) + " against "
            + String(honest)
            + ". R_radius is not being read, or the query is not pruning at"
            " all, and the passing checks above prove nothing."
        )
    print(
        "ball_cover: sabotage reached; radii x0.3 took edges from",
        honest,
        "to",
        broken,
        "and every survivor is still a true neighbor",
    )

    # `hx` IS HELD PAST ITS LAST READER. Mojo frees a buffer at its LAST USE,
    # and `hx.unsafe_ptr()` hands out a MutUntrackedOrigin pointer that does
    # not extend the buffer's life, so without this the pinned fixture can be
    # released while the oracle above is still reading through it. Metal
    # leaves the freed page mapped, which is why Apple never saw it; ROCm
    # unmaps and returns zero pages, which is what produced the false row-63
    # 'MISSED point 42' on the MI325X. See the note in `_run_one_eps`.
    _ = hx^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^

