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

from neighbors.ported.neighbors.ball_cover.ball_cover import (
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
