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
from original.numerics import PIN_CROSS_VENDOR  # DEVIATION 551

from neighbors.derived.neighbors.ball_cover.ball_cover import (
    rbc_build_index,
    rbc_eps_nn_query_count,
    rbc_eps_nn_query_dense,
    rbc_eps_nn_query_fill,
    rbc_eps_nn_query_max_k,
    rbc_n_landmarks,
)


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

    Coordinates are kept small on purpose: `PORTING.md 21` records a k-NN
    fixture destroyed by float32 cancellation when the spacing was 100. The
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
    ctx.enqueue_copy(dst_ptr=hia.unsafe_ptr(), src_buf=adj_ia)
    ctx.enqueue_copy(dst_ptr=hja.unsafe_ptr(), src_buf=adj_ja)
    ctx.enqueue_copy(dst_ptr=hvd.unsafe_ptr(), src_buf=vd)
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

    @parameter
    if PIN_CROSS_VENDOR:
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

    @parameter
    if PIN_CROSS_VENDOR:
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
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
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
    ctx.enqueue_copy(dst_ptr=hptr.unsafe_ptr(), src_buf=r_indptr)
    ctx.enqueue_copy(dst_ptr=hcols.unsafe_ptr(), src_buf=r_1nn_cols)
    ctx.enqueue_copy(dst_ptr=hdists.unsafe_ptr(), src_buf=r_1nn_dists)
    ctx.enqueue_copy(dst_ptr=hrad.unsafe_ptr(), src_buf=r_radius)
    ctx.enqueue_copy(dst_ptr=hxr.unsafe_ptr(), src_buf=x_reordered)
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


def check_ball_cover_dense_and_max_k() raises:
    """The other two output shapes, against the CSR one.

    `block_rbc_kernel_eps_dense` and `block_rbc_kernel_eps_max_k` walk the
    same landmarks with different bookkeeping, so they are checked against
    the same host oracle rather than against each other.
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
    var adj = ctx.enqueue_create_buffer[DType.uint8](n * n)
    var scratch = ctx.enqueue_create_buffer[DType.int32](1)
    var tmp = ctx.enqueue_create_buffer[DType.int32](n * BC_MAX_K)
    ctx.synchronize()

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _coord(i, f))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()

    rbc_build_index(
        ctx, x, r, x_reordered, landmark_ids, slot_cols, slot_dists,
        nearest, nearest_dist, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        counts, n, d, n_landmarks,
    )

    var eps = Float32(1.6)
    var eps2 = eps * eps

    # --- dense ------------------------------------------------------------
    rbc_eps_nn_query_dense(
        ctx, x_reordered, x, r, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        adj, vd, n, d, n_landmarks, n, eps,
    )
    var hadj = ctx.enqueue_create_host_buffer[DType.uint8](n * n)
    var hvd = ctx.enqueue_create_host_buffer[DType.int32](n + 1)
    ctx.enqueue_copy(dst_ptr=hadj.unsafe_ptr(), src_buf=adj)
    ctx.enqueue_copy(dst_ptr=hvd.unsafe_ptr(), src_buf=vd)
    ctx.synchronize()

    for i in range(n):
        var deg = 0
        for j in range(n):
            var inside = _host_dist_sq(hx.unsafe_ptr(), i, j, d) <= eps2
            var got = hadj.unsafe_ptr().unsafe_load(i * n + j) != UInt8(0)
            if inside != got:
                raise Error(
                    "dense: adj[" + String(i) + "][" + String(j) + "] is "
                    + String(got) + " and brute force says " + String(inside)
                )
            if inside:
                deg += 1
        if Int(hvd.unsafe_ptr().unsafe_load(i)) != deg:
            raise Error(
                "dense: vd[" + String(i) + "] = "
                + String(hvd.unsafe_ptr().unsafe_load(i))
                + " against a true degree of " + String(deg)
            )

    # --- max_k ------------------------------------------------------------
    var max_ja = n * BC_MAX_K
    var adj_ja = ctx.enqueue_create_buffer[DType.int32](max_ja)
    ctx.synchronize()
    var actual_max = rbc_eps_nn_query_max_k(
        ctx, x_reordered, x, r, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        adj_ia, adj_ja, vd, tmp, scratch, n, d, n_landmarks, eps, BC_MAX_K,
    )
    if actual_max > BC_MAX_K:
        raise Error(
            "max_k: the fixture overflowed the bound, which makes the CSR"
            " truncated and the comparison meaningless; longest row was "
            + String(actual_max)
        )

    var hia = ctx.enqueue_create_host_buffer[DType.int32](n + 1)
    var hja = ctx.enqueue_create_host_buffer[DType.int32](max_ja)
    ctx.enqueue_copy(dst_ptr=hia.unsafe_ptr(), src_buf=adj_ia)
    ctx.enqueue_copy(dst_ptr=hja.unsafe_ptr(), src_buf=adj_ja)
    ctx.synchronize()

    var seen = List[Int32]()
    for _ in range(n):
        seen.append(Int32(-1))
    var longest = 0
    for i in range(n):
        var s = Int(hia.unsafe_ptr().unsafe_load(i))
        var e = Int(hia.unsafe_ptr().unsafe_load(i + 1))
        if e - s > longest:
            longest = e - s
        for p in range(s, e):
            var c = Int(hja.unsafe_ptr().unsafe_load(p))
            if c < 0 or c >= n:
                raise Error("max_k: column out of range in row " + String(i))
            seen[c] = Int32(i)
        var expected = 0
        for j in range(n):
            var inside = _host_dist_sq(hx.unsafe_ptr(), i, j, d) <= eps2
            if inside:
                expected += 1
            if (seen[j] == Int32(i)) != inside:
                raise Error(
                    "max_k: row " + String(i) + " disagrees with brute force"
                    " on point " + String(j)
                )
        if expected != e - s:
            raise Error(
                "max_k: row " + String(i) + " has " + String(e - s)
                + " neighbors, brute force says " + String(expected)
            )
    if longest != actual_max:
        raise Error(
            "max_k: the returned longest row is " + String(actual_max)
            + " and the CSR's longest row is " + String(longest)
        )

    print(
        "ball_cover: dense and max_k both match brute force at eps 1.6,"
        " longest row",
        actual_max,
    )


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
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
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
    ctx.enqueue_copy(dst_ptr=hrad.unsafe_ptr(), src_buf=r_radius)
    ctx.synchronize()
    for k in range(n_landmarks):
        hrad.unsafe_ptr().unsafe_store(
            k, hrad.unsafe_ptr().unsafe_load(k) * Float32(0.3)
        )
    ctx.enqueue_copy(dst_buf=r_radius, src_ptr=hrad.unsafe_ptr())
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


def check_ball_cover_order_is_load_bearing() raises:
    """The SECOND prune, sabotaged separately from the first.

    `check_ball_cover_reach_by_sabotage` corrupts `R_radius` and so only
    proves the landmark test is reached. The backward walk inside a landmark
    is a different branch with a different input, `R_1nn_dists`, and a
    passing radius sabotage says nothing about it.

    So this one reverses each landmark group's distance array in place,
    leaving `R_1nn_cols` and `X_reordered` alone. The group is then described
    as descending while it is physically ascending, which makes
    `cur_R_dist - min_warp_dist > eps` fire on chunks that are in fact in
    reach. Predicted shape, again: a strict subset and strictly fewer edges.
    A full scan, or a walk that ignored `R_1nn_dists`, would not move.
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
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
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
        eps, String("order-sabotage/before"), False,
    )

    var hptr = ctx.enqueue_create_host_buffer[DType.int32](n_landmarks + 1)
    var hd = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=hptr.unsafe_ptr(), src_buf=r_indptr)
    ctx.enqueue_copy(dst_ptr=hd.unsafe_ptr(), src_buf=r_1nn_dists)
    ctx.synchronize()
    for k in range(n_landmarks):
        var s = Int(hptr.unsafe_ptr().unsafe_load(k))
        var e = Int(hptr.unsafe_ptr().unsafe_load(k + 1))
        var a = s
        var b = e - 1
        while a < b:
            var t = hd.unsafe_ptr().unsafe_load(a)
            hd.unsafe_ptr().unsafe_store(a, hd.unsafe_ptr().unsafe_load(b))
            hd.unsafe_ptr().unsafe_store(b, t)
            a += 1
            b -= 1
    ctx.enqueue_copy(dst_buf=r_1nn_dists, src_ptr=hd.unsafe_ptr())
    ctx.synchronize()

    var broken = _run_one_eps(
        ctx, x, x_reordered, r, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        adj_ia, vd, hx.unsafe_ptr(), n, d, n_landmarks,
        eps, String("order-sabotage/after"), True,
    )
    if broken >= honest:
        raise Error(
            "SABOTAGE DID NOT REACH: reversing every landmark group's"
            " distance array left the edge count at " + String(broken)
            + " against " + String(honest)
            + ". The backward walk is not using R_1nn_dists, so its early"
            " stop is not being exercised by any check above."
        )
    print(
        "ball_cover: the in-group order is load bearing; reversing it took",
        "edges from",
        honest,
        "to",
        broken,
    )


def check_ball_cover_at_scale() raises:
    """A bigger, wider fixture, with the oracle run on a sample of rows.

    `nn.argsort` was correct at 256 elements and wrong at 257, which is the
    reason this exists: a check that only ever runs at one size cannot see a
    size-dependent bug, and this port has three size-dependent pieces — the
    exclusive scan's dynamic chunk, the landmark-per-block rank kernel, and
    the query's per-warp CSR offsets.

    The full `n^2` oracle is too slow on the host at this size, so 200 rows
    spread across the dataset get the full brute-force treatment and the rest
    are checked structurally. The sampled rows are strided, not the first
    200, because the first rows are also the first blocks.
    """
    var ctx = DeviceContext()
    var n = 8000
    var d = 5
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
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()

    rbc_build_index(
        ctx, x, r, x_reordered, landmark_ids, slot_cols, slot_dists,
        nearest, nearest_dist, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        counts, n, d, n_landmarks,
    )

    var eps = Float32(2.2)
    var eps2 = eps * eps
    var nnz = rbc_eps_nn_query_count(
        ctx, x_reordered, x, r, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        adj_ia, vd, n, d, n_landmarks, eps,
    )
    var adj_ja = ctx.enqueue_create_buffer[DType.int32](nnz)
    ctx.synchronize()
    rbc_eps_nn_query_fill(
        ctx, x_reordered, x, r, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        adj_ia, adj_ja, n, d, n_landmarks, eps,
    )
    var hia = ctx.enqueue_create_host_buffer[DType.int32](n + 1)
    var hja = ctx.enqueue_create_host_buffer[DType.int32](nnz)
    ctx.enqueue_copy(dst_ptr=hia.unsafe_ptr(), src_buf=adj_ia)
    ctx.enqueue_copy(dst_ptr=hja.unsafe_ptr(), src_buf=adj_ja)
    ctx.synchronize()

    var seen = List[Int32]()
    for _ in range(n):
        seen.append(Int32(-1))
    var stride = n // 200
    var i = 0
    while i < n:
        var s = Int(hia.unsafe_ptr().unsafe_load(i))
        var e = Int(hia.unsafe_ptr().unsafe_load(i + 1))
        for p in range(s, e):
            seen[Int(hja.unsafe_ptr().unsafe_load(p))] = Int32(i)
        var expected = 0
        for j in range(n):
            var inside = _host_dist_sq(hx.unsafe_ptr(), i, j, d) <= eps2
            if inside:
                expected += 1
            if (seen[j] == Int32(i)) != inside:
                raise Error(
                    "scale n=8000: row " + String(i) + " disagrees with"
                    " brute force on point " + String(j)
                )
        if expected != e - s:
            raise Error(
                "scale n=8000: row " + String(i) + " has "
                + String(e - s) + " neighbors, brute force says "
                + String(expected)
            )
        i += stride

    print(
        "ball_cover: n=8000 d=5, 200 sampled rows match brute force exactly;",
        nnz,
        "edges over",
        n_landmarks,
        "landmarks",
    )


def check_ball_cover_max_k_wiring() raises:
    """cuML's TWO-LOOP dispatch, run end to end, against the two-pass CSR.

    THIS IS A DISPATCH CHECK, NOT A KERNEL CHECK. `rbc_eps_nn_query_max_k`
    was already checked against a host oracle at one batch
    (`check_ball_cover_dense_and_max_k`). What was never checked is the thing
    the DBSCAN runner actually has to do, which is not "call the one-pass
    form" but their exact sequence across two loops over batches:

    `cuml/cpp/src/dbscan/runner.cuh:257-293` -- loop one, every batch:

        need_ja_compute = sparse_rbc_mode && ((i == 0) || sample_weight)
        VertexDeg::run(..., need_ja_compute ? &adj_graph : nullptr, 0, ...)
        maxklen[i] = thrust::reduce(vd, vd + n_points, 0, maximum{})

    so `max_k == 0` puts every batch through the TWO-PASS arm of
    `vertexdeg/algo.cuh:119`, and `data.ja` is non-null only for batch 0 --
    batch 0 is counted AND filled here, every other batch is only counted.
    The longest row of each batch is measured and kept.

    `runner.cuh:319-350` -- loop two, batches `i > 0` only (`if (i > 0)`,
    `:327`, whose comment is "i==0 -> adj and vd for batch 0 already in
    memory"):

        VertexDeg::run(..., &adj_graph, maxklen.at(i), ..., nullptr /* vd */)

    which is `max_k > 0`, so `algo.cuh:122` takes the ONE-PASS arm, and
    `algo.cuh:135` then asserts `max_k == data.max_k` -- an equality, not an
    inequality, because the bound came from a measurement of the same rows
    one loop earlier and therefore cannot be exceeded.

    Net effect on the work: **two walks over the dataset, not three.** The
    runner today does count, count, fill.

    What this asserts, in the runner's OWN buffer shape -- loop one runs
    REVERSED exactly as theirs (`:249`), every batch shares ONE `ja` sized
    `maxadjlen` (`:317`), and batch 0 is filled once, when that buffer is
    sized:

    1. the one-pass CSR is BYTE-IDENTICAL to a fresh two-pass CSR for the
       same batch -- same offsets and the same column order, not merely the
       same set. The order is load bearing downstream in the same way the
       in-group order is load bearing inside the index, and a set comparison
       would not see it move;
    2. their equality assert holds: `actual_max` comes back exactly equal to
       the bound measured in loop one, for every batch;
    3. batch 0's `ia` and `ja` from loop one are RESIDENT at the top of loop
       two -- read back before any other batch runs, byte-compared against a
       fresh two-pass answer -- which is what `if (i > 0)` (`:327`) depends
       on;
    4. SABOTAGE -- the same call with `max_k` one too small returns an
       `actual_max` that is strictly larger than the bound and a CSR whose
       rows are clamped. That is the branch `algo.cuh:135` exists to catch,
       and it proves the bound is read rather than ignored.

    The fixture is 1201 rows, which is prime, above 256 (the size that hid
    the `argsort` bug), and not a multiple of 32, 64, 128 or 256; the batch
    is 401, so the batches are 401 / 401 / 399 and none of them is a multiple
    of a plausible block width either.
    """
    var ctx = DeviceContext()
    var n = 1201
    var d = 4
    var n_landmarks = rbc_n_landmarks(n)
    var batch = 401
    var n_batches = (n + batch - 1) // batch

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
    var adj_ia = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var vd = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var scratch = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.synchronize()

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _coord(i, f))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()

    rbc_build_index(
        ctx, x, r, x_reordered, landmark_ids, slot_cols, slot_dists,
        nearest, nearest_dist, r_indptr, r_1nn_cols, r_1nn_dists, r_radius,
        counts, n, d, n_landmarks,
    )

    var eps = Float32(3.0)

    # ---- loop one: `runner.cuh:249-293`, REVERSED as theirs is ---------
    # `for (int i = n_batches - 1; i >= 0; i--)` (`:249`): every batch is
    # COUNTED with `max_k = 0` (`:262`), only batch 0 would also be FILLED
    # (`need_ja_compute`, `:257`), and the longest row of every batch is
    # measured and kept (`:289`). Batch 0 comes LAST, so its `ia` is still
    # resident in `adj_ia` when the loop ends -- the residency loop two's
    # `if (i > 0)` depends on, asserted below rather than assumed.
    var maxklen = List[Int]()
    var batch_nnz = List[Int]()
    for _b in range(n_batches):
        maxklen.append(0)
        batch_nnz.append(0)

    var bi = n_batches - 1
    while bi >= 0:
        var start = bi * batch
        var np = min(n - start, batch)
        var qb = x.create_sub_buffer[DType.float32](start * d, np * d)
        var nnz = rbc_eps_nn_query_count(
            ctx, x_reordered, qb, r, r_indptr, r_1nn_cols, r_1nn_dists,
            r_radius, adj_ia, vd, np, d, n_landmarks, eps,
        )
        batch_nnz[bi] = nnz
        # `thrust::reduce(vd, vd + n_points, 0, maximum{})`, `runner.cuh:289`.
        var hvd = ctx.enqueue_create_host_buffer[DType.int32](batch + 1)
        ctx.enqueue_copy(dst_ptr=hvd.unsafe_ptr(), src_buf=vd)
        ctx.synchronize()
        var mk0 = 0
        for i in range(np):
            var g = Int(hvd.unsafe_ptr().unsafe_load(i))
            if g > mk0:
                mk0 = g
        maxklen[bi] = mk0
        bi -= 1

    # The fixture has to make the check mean something: a batch whose rows
    # are all length 1 would pass every assertion below without exercising
    # anything.
    var longest = 0
    var maxadjlen = 1
    for b in range(n_batches):
        if maxklen[b] > longest:
            longest = maxklen[b]
        if batch_nnz[b] > maxadjlen:
            maxadjlen = batch_nnz[b]
    if longest < 5:
        raise Error(
            "max_k wiring: the fixture's longest row is "
            + String(longest)
            + ", which is too short for this check to mean anything"
        )

    # ONE shared `ja` for every batch, sized to the largest
    # (`runner.cuh:317`) -- the buffer shape the DBSCAN runner uses -- and
    # batch 0's fill the moment it is sized, against the `ia` loop one left
    # resident. Theirs fills inside loop one (`algo.cuh:150`) into a buffer
    # `:317` then GROWS, and a growing `rmm::device_uvector::resize`
    # preserves contents; sizing first and filling once is the same bytes.
    var col_ind = ctx.enqueue_create_buffer[DType.int32](maxadjlen)
    ctx.synchronize()
    var np0 = min(n, batch)
    var qb0 = x.create_sub_buffer[DType.float32](0, np0 * d)
    rbc_eps_nn_query_fill(
        ctx, x_reordered, qb0, r, r_indptr, r_1nn_cols, r_1nn_dists,
        r_radius, adj_ia, col_ind, np0, d, n_landmarks, eps,
    )
    ctx.synchronize()

    # One `tmp` for every one-pass batch, sized once from max(maxklen),
    # exactly as the runner sizes its own; theirs is `n * max_k` inside
    # each call (`registers.cuh:1431`).
    var mk_max = 1
    for b in range(1, n_batches):
        if maxklen[b] > mk_max:
            mk_max = maxklen[b]
    var tmp = ctx.enqueue_create_buffer[DType.int32](batch * mk_max)
    ctx.synchronize()

    # ---- loop two: `runner.cuh:319-350` --------------------------------
    # Batch 0 first, exactly as the runner reads it: its CSR must be the
    # bytes loop one left behind (`if (i > 0)`, `:327`). Batches i > 0 take
    # the ONE-PASS arm (`:335` passes `maxklen.at(i)`) into the SAME shared
    # `col_ind`. Every batch is then compared byte for byte -- offsets AND
    # column order -- against a FRESH two-pass CSR computed afterwards,
    # afterwards because the recomputation overwrites `adj_ia`.
    for b in range(n_batches):
        var start = b * batch
        var np = min(n - start, batch)
        var nnz = batch_nnz[b]

        var hia = ctx.enqueue_create_host_buffer[DType.int32](batch + 1)
        var hja = ctx.enqueue_create_host_buffer[DType.int32](nnz)

        if b == 0:
            # No neighborhood pass: read what is RESIDENT.
            ctx.enqueue_copy(dst_ptr=hia.unsafe_ptr(), src_buf=adj_ia)
            ctx.enqueue_copy(
                dst_ptr=hja.unsafe_ptr(),
                src_buf=col_ind.create_sub_buffer[DType.int32](0, nnz),
            )
            ctx.synchronize()
        else:
            var mk = maxklen[b]
            var qb1 = x.create_sub_buffer[DType.float32](start * d, np * d)
            var actual = rbc_eps_nn_query_max_k(
                ctx, x_reordered, qb1, r, r_indptr, r_1nn_cols,
                r_1nn_dists, r_radius, adj_ia, col_ind, vd, tmp, scratch,
                np, d, n_landmarks, eps, mk,
            )
            # `ASSERT(max_k == data.max_k, ...)`, `algo.cuh:135`.
            if actual != mk:
                raise Error(
                    "max_k wiring: batch " + String(b)
                    + " was given the bound " + String(mk)
                    + " measured in loop one and came back with "
                    + String(actual) + "; their assert at algo.cuh:135 is"
                    " an EQUALITY and it just failed"
                )
            ctx.enqueue_copy(dst_ptr=hia.unsafe_ptr(), src_buf=adj_ia)
            ctx.enqueue_copy(
                dst_ptr=hja.unsafe_ptr(),
                src_buf=col_ind.create_sub_buffer[DType.int32](0, nnz),
            )
            ctx.synchronize()

        # The fresh two-pass reference, computed AFTER the arm under test
        # so nothing it produced can leak in.
        var rqb = x.create_sub_buffer[DType.float32](start * d, np * d)
        var rnnz = rbc_eps_nn_query_count(
            ctx, x_reordered, rqb, r, r_indptr, r_1nn_cols, r_1nn_dists,
            r_radius, adj_ia, vd, np, d, n_landmarks, eps,
        )
        if rnnz != nnz:
            raise Error(
                "max_k wiring: batch " + String(b) + " counted "
                + String(rnnz) + " edges on recount against "
                + String(nnz) + " in loop one"
            )
        var rja = ctx.enqueue_create_buffer[DType.int32](nnz)
        ctx.synchronize()
        var rqb2 = x.create_sub_buffer[DType.float32](start * d, np * d)
        rbc_eps_nn_query_fill(
            ctx, x_reordered, rqb2, r, r_indptr, r_1nn_cols, r_1nn_dists,
            r_radius, adj_ia, rja, np, d, n_landmarks, eps,
        )
        var ria = ctx.enqueue_create_host_buffer[DType.int32](batch + 1)
        var rjah = ctx.enqueue_create_host_buffer[DType.int32](nnz)
        ctx.enqueue_copy(dst_ptr=ria.unsafe_ptr(), src_buf=adj_ia)
        ctx.enqueue_copy(dst_ptr=rjah.unsafe_ptr(), src_buf=rja)
        ctx.synchronize()

        for i in range(np + 1):
            if hia.unsafe_ptr().unsafe_load(i) != ria.unsafe_ptr(
            ).unsafe_load(i):
                raise Error(
                    "max_k wiring: batch " + String(b) + " ia["
                    + String(i) + "] is "
                    + String(hia.unsafe_ptr().unsafe_load(i))
                    + " against "
                    + String(ria.unsafe_ptr().unsafe_load(i))
                    + " two-pass"
                )
        for pp in range(nnz):
            if hja.unsafe_ptr().unsafe_load(pp) != rjah.unsafe_ptr(
            ).unsafe_load(pp):
                raise Error(
                    "max_k wiring: batch " + String(b) + " ja["
                    + String(pp) + "] is "
                    + String(hja.unsafe_ptr().unsafe_load(pp))
                    + " against "
                    + String(rjah.unsafe_ptr().unsafe_load(pp))
                    + " two-pass; the COLUMN ORDER moved"
                )

    # ---- sabotage: hand it a bound one too small -----------------------
    # `:944-950` keeps counting past `max_k` and stops only the writes, so
    # the predicted shape is exact: `actual_max` comes back as the TRUE
    # longest row, strictly above the bound, and every row of the CSR is
    # clamped to the bound. A kernel that ignored `max_k` would return the
    # bound it was given and leave the rows unclamped.
    var sb = 1
    var sstart = sb * batch
    var snp = min(n - sstart, batch)
    var strue = maxklen[sb]
    var sbound = strue - 1
    var sqb = x.create_sub_buffer[DType.float32](sstart * d, snp * d)
    var stmp = ctx.enqueue_create_buffer[DType.int32](snp * sbound)
    var sja = ctx.enqueue_create_buffer[DType.int32](batch_nnz[sb])
    ctx.synchronize()
    # DEVIATION 551: UNDER IDENTICAL THIS SABOTAGE MUST BE REFUSED, NOT RUN.
    # Truncation keeps the first `max_k` hits IN EMISSION ORDER, and emission
    # order is lane-width dependent (IDENTITY_PATHS row 61), so a 64-lane
    # column keeps a DIFFERENT SUBSET. That is the members diverging, which no
    # ordering pass can repair, so `rbc_eps_pass_max_k` raises by name. The
    # clamp behaviour is still worth testing and is still tested, under FAST,
    # where no cross-vendor promise is made. Inverting the expectation is the
    # same move `packaging/linux/smoke.py` makes for `gemm-pinned`: a designed
    # refusal that SUCCEEDS is the failure.
    @parameter
    if PIN_CROSS_VENDOR:
        var refused = False
        try:
            _ = rbc_eps_nn_query_max_k(
                ctx, x_reordered, sqb, r, r_indptr, r_1nn_cols, r_1nn_dists,
                r_radius, adj_ia, sja, vd, stmp, scratch, snp, d, n_landmarks,
                eps, sbound,
            )
        except e:
            if String(e).find("canonicalizes an order") < 0:
                raise Error(
                    "max_k truncation under IDENTICAL raised, but not the"
                    " DEVIATION 551 refusal: " + String(e)
                )
            refused = True
        if not refused:
            raise Error(
                "max_k truncation under IDENTICAL was ACCEPTED. A bound one"
                " short of the true longest row silently keeps a lane-width"
                " dependent SUBSET, and DEVIATION 551's refusal is what"
                " prevents that. It did not fire."
            )
        print(
            "ball_cover: max_k truncation is REFUSED by name under IDENTICAL"
            " (DEVIATION 551); the clamp itself is exercised under FAST"
        )
        return

    var sactual = rbc_eps_nn_query_max_k(
        ctx, x_reordered, sqb, r, r_indptr, r_1nn_cols, r_1nn_dists,
        r_radius, adj_ia, sja, vd, stmp, scratch, snp, d, n_landmarks,
        eps, sbound,
    )
    if sactual != strue:
        raise Error(
            "max_k wiring sabotage: the bound was cut to " + String(sbound)
            + " and the call reported a longest row of " + String(sactual)
            + " instead of the true " + String(strue) + "; the count is"
            " supposed to keep going past the bound"
        )
    var shia = ctx.enqueue_create_host_buffer[DType.int32](batch + 1)
    ctx.enqueue_copy(dst_ptr=shia.unsafe_ptr(), src_buf=adj_ia)
    ctx.synchronize()
    var clamped = 0
    for i in range(snp):
        var rowlen = Int(shia.unsafe_ptr().unsafe_load(i + 1)) - Int(
            shia.unsafe_ptr().unsafe_load(i)
        )
        if rowlen > sbound:
            raise Error(
                "max_k wiring sabotage: row " + String(i) + " kept "
                + String(rowlen) + " columns against a bound of "
                + String(sbound)
            )
        if rowlen == sbound:
            clamped += 1
    if clamped == 0:
        raise Error(
            "max_k wiring sabotage: nothing was clamped, so the truncation"
            " branch was never reached"
        )

    print(
        "ball_cover: cuML's two-loop max_k dispatch is byte-identical to the"
        " two-pass CSR over",
        n_batches,
        "batches sharing one ja; batch 0's loop-one CSR is resident; bounds",
        maxklen[0],
        maxklen[1],
        maxklen[2],
        "; a bound one short clamps",
        clamped,
        "rows and still reports",
        strue,
    )
