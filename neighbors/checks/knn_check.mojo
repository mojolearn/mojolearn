# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Launch the brute-force k-NN, and sabotage it.

NO CUVS COUNTERPART. Same discipline as `cluster/checks/kmeans_check.mojo`
and for the same reason: a kernel is not ported until it has been enqueued
(`PORTING.md 9`), and a correct answer is not by itself evidence that a
kernel ran.

THE FIXTURE IS RANDOM, AND THE FIRST ONE WAS NOT, AND THAT COST A RUN
---------------------------------------------------------------------
The first fixture put 4096 index points on a LINE at spacing 100, so the
true k nearest were known in closed form and needed no tolerance. It failed,
60 of 512 neighbors wrong, and the port was right.

The expanded identity computes `||x||^2 + ||y||^2 - 2 x.y`. On that fixture
the norms were about 1e10 and the distances about 1e3, so in float32, whose
ulp at 1e10 is roughly 1024, every distance collapsed onto a coarse grid:
the true 900 came back as 0.0, the true 16900 as 16384, and ties at the
quantized minimum were then broken arbitrarily.

**That is a property of the metric, not of this port, and it does not go away
by rescaling.** For N collinear points the closest-pair squared distance is
about `(range/N)^2` while the norm is about `range^2`, so the ratio is `1/N^2`
regardless of scale. At N=4096 that is 6e-8 against float32's 1.2e-7 relative
precision. **The expanded identity in float32 cannot rank four thousand
collinear points, at any scale.** cuVS defaults to `L2Expanded` and this is
the cost of the GEMM formulation; the direct formulation has no such problem
and is far slower.

So the fixture is uniform random in a 16-dimensional cube, where norms and
distances are the same order of magnitude and the expanded form is accurate,
and the TRUTH is computed on the host in Float64 with the DIRECT formula.
**"Random" here means SPLITMIX-MIXED and did not until 2026-08-19**: the
generator was affine in the row index, which is a lattice and is the same
failure as the collinear fixture, one level less obvious. See `_coord`.
That checks our GPU expanded-identity answer against an independent
computation rather than against a rearrangement of the same one.

The k nearest are compared as a SET. Radix select returns them unordered, and
`select_radix.mojo` documents why their tie handling makes even the identity
of a tied neighbor non-reproducible.
"""

from std.math import sqrt
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.hardware_matrix import threadgroup_limit_for
from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

from core.row_norms import NORM_TPB, row_norm_kernel
from neighbors.impl.distance.detail.pairwise_distance_base import (
    launch_config_generator,
)
from neighbors.impl.neighbors.detail.fused_l2_knn import (
    FKNN_MBLK,
    FKNN_NBLK,
    FKNN_SMEM_PAGE_X,
    FKNN_SMEM_PAGE_Y,
    FKNN_THREADS,
    fused_l2_knn,
    fused_l2_knn_grid,
    fused_l2_knn_launch,
)
from neighbors.impl.neighbors.detail.knn_brute_force import (
    KNN_METHOD_FUSED,
    brute_force_knn_impl,
    compute_norms,
    tiled_brute_force_knn,
)


comptime KNN_INDEX = 4096
comptime KNN_QUERIES = 64
comptime KNN_FEATURES = 16
comptime KNN_K = 8
comptime KNN_TILE = 64

# The FUSED fixture is deliberately coprime with every tile edge in
# `fused_l2_knn.mojo`. `Policy2x8` has `Mblk = 16` and `Nblk = 256` and
# `Kblk = 16`, so 53 queries, 4093 index points and 20 features put a partial
# tile on ALL THREE axes at once. 4096/64/16 would have exercised none of
# them, and a check that never crosses an edge cannot see an edge bug.
comptime FCHK_INDEX = 4093
comptime FCHK_QUERIES = 53
comptime FCHK_FEATURES = 20
comptime FCHK_MAX_K = 64


def _coord(row: Int, feature: Int, salt: Int) -> Float32:
    """Uniform in [0, 1), splitmix64-mixed so consecutive rows are unrelated.

    **THIS WAS A LATTICE UNTIL 2026-08-19 AND THE DOCSTRING ABOVE CALLED IT
    RANDOM.** It was `(row * 2654435761 + feature * 40503 + salt * 2246822519)
    % 1000003`, which is AFFINE in `row`: point `j + 1` is point `j` plus a
    fixed offset modulo one, in every feature at once. That is a regular
    lattice, not a sample.

    It is the SAME defect as the collinear fixture this file's header
    describes, arrived at from the other direction, and it hides in exactly
    the same place: at 4,093 points in 20 dimensions the true
    nearest-neighbour distance on that lattice is 4e-6 while the row norms are
    about 7, so the expanded identity in float32 - whose ulp at 7 is 5e-7 -
    cannot rank them at all. The fused k-NN check found it by returning a
    neighbour at 4.88e-6 where the oracle wanted one at 3.68e-6, and the
    kernel was right.

    `bench/scaling_main.mojo:34-40` already used a splitmix mixer, so the
    measured numbers were never taken on the lattice. Only the checks were.
    """
    var z = (
        UInt64(row) * UInt64(0x9E3779B97F4A7C15)
        + UInt64(feature) * UInt64(0xBF58476D1CE4E5B9)
        + UInt64(salt) * UInt64(0x94D049BB133111EB)
    )
    z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
    z = z ^ (z >> UInt64(31))
    return Float32(z >> UInt64(40)) * Float32(1.0 / 16777216.0)


def check_knn() raises:
    var ctx = DeviceContext()

    var buf_len = KNN_INDEX // 8
    if buf_len < KNN_K:
        buf_len = KNN_K

    var index = ctx.enqueue_create_buffer[DType.float32](
        KNN_INDEX * KNN_FEATURES
    )
    var queries = ctx.enqueue_create_buffer[DType.float32](
        KNN_QUERIES * KNN_FEATURES
    )
    var index_norm = ctx.enqueue_create_buffer[DType.float32](KNN_INDEX)
    var query_norm = ctx.enqueue_create_buffer[DType.float32](KNN_QUERIES)
    var dist_tile = ctx.enqueue_create_buffer[DType.float32](
        KNN_TILE * KNN_INDEX
    )
    var buf_val = ctx.enqueue_create_buffer[DType.float32](
        KNN_TILE * 2 * buf_len
    )
    var buf_idx = ctx.enqueue_create_buffer[DType.uint32](
        KNN_TILE * 2 * buf_len
    )
    var out_dist = ctx.enqueue_create_buffer[DType.float32](
        KNN_QUERIES * KNN_K
    )
    var out_idx = ctx.enqueue_create_buffer[DType.uint32](KNN_QUERIES * KNN_K)
    var out_i32 = ctx.enqueue_create_buffer[DType.int32](KNN_QUERIES * KNN_K)
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](
        KNN_INDEX * KNN_FEATURES
    )
    for j in range(KNN_INDEX):
        for f in range(KNN_FEATURES):
            hi.unsafe_ptr().unsafe_store(
                j * KNN_FEATURES + f, _coord(j, f, 0)
            )
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())

    var hq = ctx.enqueue_create_host_buffer[DType.float32](
        KNN_QUERIES * KNN_FEATURES
    )
    for i in range(KNN_QUERIES):
        for f in range(KNN_FEATURES):
            hq.unsafe_ptr().unsafe_store(
                i * KNN_FEATURES + f, _coord(i, f, 7)
            )
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()

    compute_norms(ctx, index, index_norm, KNN_INDEX, KNN_FEATURES, False)
    compute_norms(ctx, queries, query_norm, KNN_QUERIES, KNN_FEATURES, False)
    ctx.synchronize()

    tiled_brute_force_knn(
        ctx, queries, query_norm, index, index_norm, dist_tile, buf_val,
        buf_idx, out_dist, out_idx, out_i32, KNN_QUERIES, KNN_INDEX,
        KNN_FEATURES, KNN_K, KNN_TILE, buf_len, False,
    )

    var ho = ctx.enqueue_create_host_buffer[DType.uint32](KNN_QUERIES * KNN_K)
    ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=out_idx)
    ctx.synchronize()

    # TRUTH on the host, in Float64, by the DIRECT formula. Independent of
    # the expanded identity the GPU used, which is the point.
    var bad = 0
    for i in range(KNN_QUERIES):
        var best_idx = List[Int]()
        var best_d = List[Float64]()
        for _s in range(KNN_K):
            best_idx.append(-1)
            best_d.append(1.0e30)

        for j in range(KNN_INDEX):
            var d = Float64(0.0)
            for f in range(KNN_FEATURES):
                var diff = Float64(
                    hq.unsafe_ptr().unsafe_load(i * KNN_FEATURES + f)
                ) - Float64(hi.unsafe_ptr().unsafe_load(j * KNN_FEATURES + f))
                d += diff * diff
            if d < best_d[KNN_K - 1]:
                var pos = KNN_K - 1
                while pos > 0 and best_d[pos - 1] > d:
                    best_d[pos] = best_d[pos - 1]
                    best_idx[pos] = best_idx[pos - 1]
                    pos -= 1
                best_d[pos] = d
                best_idx[pos] = j

        for slot in range(KNN_K):
            var got = Int(ho.unsafe_ptr().unsafe_load(i * KNN_K + slot))
            var found = False
            for t in range(KNN_K):
                if best_idx[t] == got:
                    found = True
            if not found:
                bad += 1

    if bad != 0:
        raise Error(
            String(bad) + " of " + String(KNN_QUERIES * KNN_K)
            + " returned neighbors are not in the true k-nearest set"
        )

    print(
        "check_knn OK: "
        + String(KNN_QUERIES)
        + " queries x k="
        + String(KNN_K)
        + " over "
        + String(KNN_INDEX)
        + " index points, every returned neighbor is in the exact true set"
    )


def check_knn_reach_by_sabotage() raises:
    """Same paired-prediction design as the k-means reach check.

    1. **Index norms** differ per index point, so corrupting them must change
       WHICH neighbors come back.
    2. **Query norms** are a constant per query row, added identically to
       every candidate, so OFFSETTING them changes distances and cannot
       change the ranking. The set must not move.

    The second one is an offset and not a replacement, which is the lesson
    the k-means check paid for: replacing the norms drives the expanded
    distances negative, the clamp at `unfused_distance_nn.cuh:81` flattens
    them all to zero, and the result moves for a reason that has nothing to
    do with reach.

    **The offset also has to be SMALL, which cost another run.** The first
    version added 5000 per query. Distances here are of order 2.7 and the
    gaps between the k-th and (k+1)-th neighbor are of order 0.01, so adding
    5000 pushes everything to where float32's ulp is 0.03 and the ranking
    dissolves. The set moved for 438 slots and the kernel was right, again.

    So a reach sabotage has a WINDOW: large enough that the result must
    visibly move, small enough that it does not destroy the property being
    asserted. That window is a fact about the expanded identity in float32
    and it is the same fact that made the first k-NN fixture unusable.
    """
    var ctx = DeviceContext()
    var buf_len = KNN_INDEX // 8

    var index = ctx.enqueue_create_buffer[DType.float32](
        KNN_INDEX * KNN_FEATURES
    )
    var queries = ctx.enqueue_create_buffer[DType.float32](
        KNN_QUERIES * KNN_FEATURES
    )
    var index_norm = ctx.enqueue_create_buffer[DType.float32](KNN_INDEX)
    var query_norm = ctx.enqueue_create_buffer[DType.float32](KNN_QUERIES)
    var dist_tile = ctx.enqueue_create_buffer[DType.float32](
        KNN_TILE * KNN_INDEX
    )
    var buf_val = ctx.enqueue_create_buffer[DType.float32](
        KNN_TILE * 2 * buf_len
    )
    var buf_idx = ctx.enqueue_create_buffer[DType.uint32](
        KNN_TILE * 2 * buf_len
    )
    var out_dist = ctx.enqueue_create_buffer[DType.float32](
        KNN_QUERIES * KNN_K
    )
    var out_idx = ctx.enqueue_create_buffer[DType.uint32](KNN_QUERIES * KNN_K)
    var out_i32 = ctx.enqueue_create_buffer[DType.int32](KNN_QUERIES * KNN_K)
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](
        KNN_INDEX * KNN_FEATURES
    )
    for j in range(KNN_INDEX):
        for f in range(KNN_FEATURES):
            hi.unsafe_ptr().unsafe_store(
                j * KNN_FEATURES + f, _coord(j, f, 0)
            )
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())

    var hq = ctx.enqueue_create_host_buffer[DType.float32](
        KNN_QUERIES * KNN_FEATURES
    )
    for i in range(KNN_QUERIES):
        for f in range(KNN_FEATURES):
            hq.unsafe_ptr().unsafe_store(
                i * KNN_FEATURES + f, _coord(i, f, 7)
            )
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()

    compute_norms(ctx, index, index_norm, KNN_INDEX, KNN_FEATURES, False)
    compute_norms(ctx, queries, query_norm, KNN_QUERIES, KNN_FEATURES, False)
    ctx.synchronize()

    var base = ctx.enqueue_create_host_buffer[DType.uint32](
        KNN_QUERIES * KNN_K
    )
    var sab = ctx.enqueue_create_host_buffer[DType.uint32](KNN_QUERIES * KNN_K)

    tiled_brute_force_knn(
        ctx, queries, query_norm, index, index_norm, dist_tile, buf_val,
        buf_idx, out_dist, out_idx, out_i32, KNN_QUERIES, KNN_INDEX,
        KNN_FEATURES, KNN_K, KNN_TILE, buf_len, False,
    )
    ctx.enqueue_copy(dst_ptr=base.unsafe_ptr(), src_buf=out_idx)
    ctx.synchronize()

    # --- SABOTAGE 1: index norms. The neighbor set MUST move. ------------
    var hn = ctx.enqueue_create_host_buffer[DType.float32](KNN_INDEX)
    ctx.enqueue_copy(dst_ptr=hn.unsafe_ptr(), src_buf=index_norm)
    ctx.synchronize()
    for j in range(KNN_INDEX):
        # Reversed ramp, large enough to reorder and positive throughout.
        hn.unsafe_ptr().unsafe_store(
            j, Float32(1000.0) * Float32(KNN_INDEX - j)
        )
    ctx.enqueue_copy(dst_buf=index_norm, src_ptr=hn.unsafe_ptr())
    ctx.synchronize()

    tiled_brute_force_knn(
        ctx, queries, query_norm, index, index_norm, dist_tile, buf_val,
        buf_idx, out_dist, out_idx, out_i32, KNN_QUERIES, KNN_INDEX,
        KNN_FEATURES, KNN_K, KNN_TILE, buf_len, False,
    )
    ctx.enqueue_copy(dst_ptr=sab.unsafe_ptr(), src_buf=out_idx)
    ctx.synchronize()

    var moved = 0
    for i in range(KNN_QUERIES * KNN_K):
        if sab.unsafe_ptr().unsafe_load(i) != base.unsafe_ptr().unsafe_load(i):
            moved += 1
    if moved == 0:
        raise Error(
            "SABOTAGE 1 FAILED TO REGISTER: corrupting index_norm changed no"
            " neighbor. The distance epilogue is not reading it."
        )

    # --- restore, then SABOTAGE 2: query norms, OFFSET. Set must hold. ---
    compute_norms(ctx, index, index_norm, KNN_INDEX, KNN_FEATURES, False)
    var hqn = ctx.enqueue_create_host_buffer[DType.float32](KNN_QUERIES)
    ctx.enqueue_copy(dst_ptr=hqn.unsafe_ptr(), src_buf=query_norm)
    ctx.synchronize()
    for i in range(KNN_QUERIES):
        hqn.unsafe_ptr().unsafe_store(
            i,
            hqn.unsafe_ptr().unsafe_load(i) + Float32(0.1) * Float32(i + 1),
        )
    ctx.enqueue_copy(dst_buf=query_norm, src_ptr=hqn.unsafe_ptr())
    ctx.synchronize()

    tiled_brute_force_knn(
        ctx, queries, query_norm, index, index_norm, dist_tile, buf_val,
        buf_idx, out_dist, out_idx, out_i32, KNN_QUERIES, KNN_INDEX,
        KNN_FEATURES, KNN_K, KNN_TILE, buf_len, False,
    )
    ctx.enqueue_copy(dst_ptr=sab.unsafe_ptr(), src_buf=out_idx)
    ctx.synchronize()

    var set_moved = 0
    for i in range(KNN_QUERIES):
        for slot in range(KNN_K):
            var got = Int(sab.unsafe_ptr().unsafe_load(i * KNN_K + slot))
            var found = False
            for t in range(KNN_K):
                if Int(base.unsafe_ptr().unsafe_load(i * KNN_K + t)) == got:
                    found = True
            if not found:
                set_moved += 1
    if set_moved != 0:
        raise Error(
            "SABOTAGE 2 CHANGED THE NEIGHBOR SET for "
            + String(set_moved)
            + " slots. A per-query constant cannot change a ranking, so the"
            " distance is not being formed by the expanded identity."
        )

    print(
        "check_knn_reach_by_sabotage OK: index_norm moved "
        + String(moved)
        + "/"
        + String(KNN_QUERIES * KNN_K)
        + " neighbors; query_norm offset moved 0 sets, which is the"
        " predicted shape"
    )


def check_vendor_topk_matches_ported() raises:
    """The two selection paths must return the same neighbour SET.

    cuVS calls `select_k`, a vendor primitive, so the faithful port calls
    `nn.topk.top_k`. Keeping the ported RAFT radix select reachable is what
    makes that checkable: a vendor call whose answer nothing verifies is a
    vendor call nobody should trust.

    Compared as a SET, not element-wise. Neither implementation promises an
    order, and `select_radix.mojo` documents why RAFT's tie handling makes
    even the identity of an equidistant neighbour non-reproducible.
    """
    var ctx = DeviceContext()
    var n = KNN_INDEX
    var buf_len = n // 8

    var index = ctx.enqueue_create_buffer[DType.float32](n * KNN_FEATURES)
    var queries = ctx.enqueue_create_buffer[DType.float32](
        KNN_QUERIES * KNN_FEATURES
    )
    var index_norm = ctx.enqueue_create_buffer[DType.float32](n)
    var query_norm = ctx.enqueue_create_buffer[DType.float32](KNN_QUERIES)
    var dist = ctx.enqueue_create_buffer[DType.float32](KNN_TILE * n)
    var bv = ctx.enqueue_create_buffer[DType.float32](KNN_TILE * 2 * buf_len)
    var bi = ctx.enqueue_create_buffer[DType.uint32](KNN_TILE * 2 * buf_len)
    var od = ctx.enqueue_create_buffer[DType.float32](KNN_QUERIES * KNN_K)
    var oi = ctx.enqueue_create_buffer[DType.uint32](KNN_QUERIES * KNN_K)
    var oi32 = ctx.enqueue_create_buffer[DType.int32](KNN_QUERIES * KNN_K)
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](n * KNN_FEATURES)
    for j in range(n):
        for f in range(KNN_FEATURES):
            hi.unsafe_ptr().unsafe_store(
                j * KNN_FEATURES + f, _coord(j, f, 0)
            )
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    var hq = ctx.enqueue_create_host_buffer[DType.float32](
        KNN_QUERIES * KNN_FEATURES
    )
    for i in range(KNN_QUERIES):
        for f in range(KNN_FEATURES):
            hq.unsafe_ptr().unsafe_store(
                i * KNN_FEATURES + f, _coord(i, f, 7)
            )
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()
    compute_norms(ctx, index, index_norm, n, KNN_FEATURES, False)
    compute_norms(ctx, queries, query_norm, KNN_QUERIES, KNN_FEATURES, False)
    ctx.synchronize()

    tiled_brute_force_knn(
        ctx, queries, query_norm, index, index_norm, dist, bv, bi, od, oi,
        oi32, KNN_QUERIES, n, KNN_FEATURES, KNN_K, KNN_TILE, buf_len, False,
        False,
    )
    var ported = ctx.enqueue_create_host_buffer[DType.uint32](
        KNN_QUERIES * KNN_K
    )
    ctx.enqueue_copy(dst_ptr=ported.unsafe_ptr(), src_buf=oi)
    ctx.synchronize()

    tiled_brute_force_knn(
        ctx, queries, query_norm, index, index_norm, dist, bv, bi, od, oi,
        oi32, KNN_QUERIES, n, KNN_FEATURES, KNN_K, KNN_TILE, buf_len, False,
        True,
    )
    var vendor = ctx.enqueue_create_host_buffer[DType.int32](
        KNN_QUERIES * KNN_K
    )
    ctx.enqueue_copy(dst_ptr=vendor.unsafe_ptr(), src_buf=oi32)
    ctx.synchronize()

    var disagree = 0
    for i in range(KNN_QUERIES):
        for slot in range(KNN_K):
            var got = Int(vendor.unsafe_ptr().unsafe_load(i * KNN_K + slot))
            var found = False
            for t in range(KNN_K):
                if Int(ported.unsafe_ptr().unsafe_load(i * KNN_K + t)) == got:
                    found = True
            if not found:
                disagree += 1
    if disagree != 0:
        raise Error(
            String(disagree) + " of " + String(KNN_QUERIES * KNN_K)
            + " neighbours differ between nn.topk.top_k and the ported RAFT"
            " radix select. They solve the same problem on the same"
            " distances, so a disagreement is a bug in one of them."
        )
    print(
        "check_vendor_topk_matches_ported OK: nn.topk.top_k and the ported"
        " RAFT radix select agree on all "
        + String(KNN_QUERIES * KNN_K)
        + " neighbours"
    )


# ---------------------------------------------------------------------------
# THE FUSED PATH. This is the one cuVS actually dispatches to for k <= 64 on
# row-major L2 (`knn_brute_force.cuh:443`), so it is the one that has to be
# right. Everything above this line checks their FALLBACK.
# ---------------------------------------------------------------------------


def _fchk_coord(row: Int, feature: Int, salt: Int) -> Float32:
    """`_coord` with a different salt space, kept separate so the two
    fixtures cannot silently become the same fixture. See `_coord` for why
    this is a splitmix mix and not the affine hash it used to be."""
    return _coord(row, feature, salt + 977)


def _fchk_oracle(
    hq: MutPointer[Float32, MutUntrackedOrigin],
    hi: MutPointer[Float32, MutUntrackedOrigin],
) -> List[Int]:
    """Host Float64 top-`FCHK_MAX_K`, ascending, direct formula."""
    var out = List[Int]()
    for i in range(FCHK_QUERIES):
        var best_idx = List[Int]()
        var best_d = List[Float64]()
        for _s in range(FCHK_MAX_K):
            best_idx.append(-1)
            best_d.append(1.0e30)
        for j in range(FCHK_INDEX):
            var d = Float64(0.0)
            for f in range(FCHK_FEATURES):
                var diff = Float64(
                    hq.unsafe_load(i * FCHK_FEATURES + f)
                ) - Float64(hi.unsafe_load(j * FCHK_FEATURES + f))
                d += diff * diff
            if d < best_d[FCHK_MAX_K - 1]:
                var pos = FCHK_MAX_K - 1
                while pos > 0 and best_d[pos - 1] > d:
                    best_d[pos] = best_d[pos - 1]
                    best_idx[pos] = best_idx[pos - 1]
                    pos -= 1
                best_d[pos] = d
                best_idx[pos] = j
        for t in range(FCHK_MAX_K):
            out.append(best_idx[t])
    return out^


def check_fused_l2_knn() raises:
    """`fused_l2_knn` against the host oracle, at four values of k.

    k = 1, 8, 32 and 64. Sixty-four is their `NumWarpQ = 64` ceiling
    (`fused_l2_knn.cuh:581`) and one is the degenerate queue. The returned
    neighbours are compared PER SLOT and in ORDER, not as a set: this kernel
    keeps a sorted list, so unlike the radix selector it promises an order,
    and checking only the set would not notice a merge that inserts at the
    wrong position. The fixture is hashed, so no two distances tie and the
    ordering is well defined.
    """
    var ctx = DeviceContext()

    var index = ctx.enqueue_create_buffer[DType.float32](
        FCHK_INDEX * FCHK_FEATURES
    )
    var queries = ctx.enqueue_create_buffer[DType.float32](
        FCHK_QUERIES * FCHK_FEATURES
    )
    var inorm = ctx.enqueue_create_buffer[DType.float32](FCHK_INDEX)
    var qnorm = ctx.enqueue_create_buffer[DType.float32](FCHK_QUERIES)
    var od = ctx.enqueue_create_buffer[DType.float32](
        FCHK_QUERIES * FCHK_MAX_K
    )
    var oi = ctx.enqueue_create_buffer[DType.uint32](FCHK_QUERIES * FCHK_MAX_K)
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_INDEX * FCHK_FEATURES
    )
    for j in range(FCHK_INDEX):
        for f in range(FCHK_FEATURES):
            hi.unsafe_ptr().unsafe_store(
                j * FCHK_FEATURES + f, _fchk_coord(j, f, 3)
            )
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    var hq = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_QUERIES * FCHK_FEATURES
    )
    for i in range(FCHK_QUERIES):
        for f in range(FCHK_FEATURES):
            hq.unsafe_ptr().unsafe_store(
                i * FCHK_FEATURES + f, _fchk_coord(i, f, 11)
            )
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()

    compute_norms(ctx, index, inorm, FCHK_INDEX, FCHK_FEATURES, False)
    compute_norms(ctx, queries, qnorm, FCHK_QUERIES, FCHK_FEATURES, False)
    ctx.synchronize()

    var truth = _fchk_oracle(hq.unsafe_ptr(), hi.unsafe_ptr())

    var ho = ctx.enqueue_create_host_buffer[DType.uint32](
        FCHK_QUERIES * FCHK_MAX_K
    )
    var hd = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_QUERIES * FCHK_MAX_K
    )

    var ks = List[Int]()
    ks.append(1)
    ks.append(8)
    ks.append(32)
    ks.append(FCHK_MAX_K)

    for t in range(len(ks)):
        var k = ks[t]
        fused_l2_knn(
            ctx, queries, qnorm, index, inorm, od, oi,
            FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, k, False,
        )
        ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=oi)
        ctx.enqueue_copy(dst_ptr=hd.unsafe_ptr(), src_buf=od)
        ctx.synchronize()

        var bad = 0
        var unsorted = 0
        for i in range(FCHK_QUERIES):
            for s in range(k):
                var got = Int(ho.unsafe_ptr().unsafe_load(i * k + s))
                if got != truth[i * FCHK_MAX_K + s]:
                    bad += 1
                if s > 0:
                    var prev = hd.unsafe_ptr().unsafe_load(i * k + s - 1)
                    if hd.unsafe_ptr().unsafe_load(i * k + s) < prev:
                        unsorted += 1
        if bad != 0:
            var msg = String("")
            var shown = 0
            for i in range(FCHK_QUERIES):
                for s in range(k):
                    var g = Int(ho.unsafe_ptr().unsafe_load(i * k + s))
                    var w = truth[i * FCHK_MAX_K + s]
                    if g != w and shown < 4:
                        shown += 1
                        var dg = Float64(0.0)
                        var dw = Float64(0.0)
                        for f in range(FCHK_FEATURES):
                            var qv = Float64(
                                hq.unsafe_ptr().unsafe_load(
                                    i * FCHK_FEATURES + f
                                )
                            )
                            if g >= 0 and g < FCHK_INDEX:
                                var a = qv - Float64(
                                    hi.unsafe_ptr().unsafe_load(
                                        g * FCHK_FEATURES + f
                                    )
                                )
                                dg += a * a
                            var b = qv - Float64(
                                hi.unsafe_ptr().unsafe_load(
                                    w * FCHK_FEATURES + f
                                )
                            )
                            dw += b * b
                        msg += (
                            " [q=" + String(i) + " slot=" + String(s)
                            + " got=" + String(g) + " d=" + String(dg)
                            + " want=" + String(w) + " d=" + String(dw)
                            + " gpu_d=" + String(
                                hd.unsafe_ptr().unsafe_load(i * k + s)
                            )
                            + "]"
                        )
            raise Error(
                "fused k-NN at k=" + String(k) + ": " + String(bad) + " of "
                + String(FCHK_QUERIES * k)
                + " slots disagree with the host Float64 oracle" + msg
            )
        if unsorted != 0:
            raise Error(
                "fused k-NN at k=" + String(k) + ": " + String(unsorted)
                + " slots are out of ascending order; the merge is inserting"
                " at the wrong position"
            )

    # --- `is_sqrt`, which is a SECOND KERNEL and would otherwise be
    # unreached. `fusedL2Knn` hard-codes `sqrt = false` inside the kernel
    # (`fused_l2_knn.cuh:1017-1019`) and takes the root afterwards over the
    # `m * k` outputs (`knn_brute_force.cuh:463-472`), so the two runs must
    # return the SAME neighbours in the same order and distances related by
    # exactly one square root.
    var ksq = 8
    fused_l2_knn(
        ctx, queries, qnorm, index, inorm, od, oi,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, ksq, False,
    )
    var plain_d = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_QUERIES * ksq
    )
    ctx.enqueue_copy(dst_ptr=plain_d.unsafe_ptr(), src_buf=od)
    ctx.synchronize()
    fused_l2_knn(
        ctx, queries, qnorm, index, inorm, od, oi,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, ksq, True,
    )
    ctx.enqueue_copy(dst_ptr=hd.unsafe_ptr(), src_buf=od)
    ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=oi)
    ctx.synchronize()
    var sq_bad = 0
    var sq_same = 0
    for i in range(FCHK_QUERIES * ksq):
        var p0 = plain_d.unsafe_ptr().unsafe_load(i)
        var p1 = hd.unsafe_ptr().unsafe_load(i)
        var want = sqrt(Float64(p0))
        var diff = Float64(p1) - want
        if diff < 0.0:
            diff = -diff
        if diff > 1.0e-5 * (want + 1.0e-6):
            sq_bad += 1
        if p1 == p0:
            sq_same += 1
    if sq_bad != 0:
        raise Error(
            "is_sqrt: " + String(sq_bad) + " of "
            + String(FCHK_QUERIES * ksq)
            + " distances are not the square root of the squared run"
        )
    if sq_same == FCHK_QUERIES * ksq:
        raise Error(
            "is_sqrt changed NOTHING, so sqrt_postprocess_kernel never ran"
        )

    print(
        "check_fused_l2_knn OK: k = 1, 8, 32, 64 over "
        + String(FCHK_QUERIES)
        + " queries x "
        + String(FCHK_INDEX)
        + " index x "
        + String(FCHK_FEATURES)
        + " features, every slot matches the host Float64 oracle in order,"
        " and is_sqrt is exactly one square root away"
    )


def check_fused_reach_by_sabotage() raises:
    """Two paired sabotages, the same design the tiled check uses.

    1. **Index norms**, a per-INDEX-POINT quantity, ramped. The neighbour set
       MUST move: they enter the distance per column and reorder the ranking.
    2. **Query norms**, a per-QUERY constant added identically to every
       candidate of that row, OFFSET by a small amount. The set must NOT
       move, because a constant cannot change a ranking.

    Both windows are the ones the tiled check already paid for: the offset is
    small because at 5000 the float32 ulp swallows the gaps between the k-th
    and (k+1)-th neighbour and the set moves for a reason that has nothing to
    do with reach; the ramp is large because a small one does not reorder.

    A third property is asserted that only the FUSED kernel can fail:
    sabotage 1 must move the DISTANCES too, and by more than the threshold
    reject could mask. If the epilogue read the norms but the queue's
    `warpKTop` reject did not see the new values, the set would move and the
    stored distances would not.
    """
    var ctx = DeviceContext()
    var k = 8

    var index = ctx.enqueue_create_buffer[DType.float32](
        FCHK_INDEX * FCHK_FEATURES
    )
    var queries = ctx.enqueue_create_buffer[DType.float32](
        FCHK_QUERIES * FCHK_FEATURES
    )
    var inorm = ctx.enqueue_create_buffer[DType.float32](FCHK_INDEX)
    var qnorm = ctx.enqueue_create_buffer[DType.float32](FCHK_QUERIES)
    var od = ctx.enqueue_create_buffer[DType.float32](FCHK_QUERIES * k)
    var oi = ctx.enqueue_create_buffer[DType.uint32](FCHK_QUERIES * k)
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_INDEX * FCHK_FEATURES
    )
    for j in range(FCHK_INDEX):
        for f in range(FCHK_FEATURES):
            hi.unsafe_ptr().unsafe_store(
                j * FCHK_FEATURES + f, _fchk_coord(j, f, 3)
            )
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    var hq = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_QUERIES * FCHK_FEATURES
    )
    for i in range(FCHK_QUERIES):
        for f in range(FCHK_FEATURES):
            hq.unsafe_ptr().unsafe_store(
                i * FCHK_FEATURES + f, _fchk_coord(i, f, 11)
            )
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()

    compute_norms(ctx, index, inorm, FCHK_INDEX, FCHK_FEATURES, False)
    compute_norms(ctx, queries, qnorm, FCHK_QUERIES, FCHK_FEATURES, False)
    ctx.synchronize()

    var base_i = ctx.enqueue_create_host_buffer[DType.uint32](
        FCHK_QUERIES * k
    )
    var base_d = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_QUERIES * k
    )
    var sab_i = ctx.enqueue_create_host_buffer[DType.uint32](FCHK_QUERIES * k)
    var sab_d = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_QUERIES * k
    )

    fused_l2_knn(
        ctx, queries, qnorm, index, inorm, od, oi,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, k, False,
    )
    ctx.enqueue_copy(dst_ptr=base_i.unsafe_ptr(), src_buf=oi)
    ctx.enqueue_copy(dst_ptr=base_d.unsafe_ptr(), src_buf=od)
    ctx.synchronize()

    # --- SABOTAGE 1: index norms ramped. Set AND distances must move. ----
    var hn = ctx.enqueue_create_host_buffer[DType.float32](FCHK_INDEX)
    for j in range(FCHK_INDEX):
        hn.unsafe_ptr().unsafe_store(
            j, Float32(1000.0) * Float32(FCHK_INDEX - j)
        )
    ctx.enqueue_copy(dst_buf=inorm, src_ptr=hn.unsafe_ptr())
    ctx.synchronize()

    fused_l2_knn(
        ctx, queries, qnorm, index, inorm, od, oi,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, k, False,
    )
    ctx.enqueue_copy(dst_ptr=sab_i.unsafe_ptr(), src_buf=oi)
    ctx.enqueue_copy(dst_ptr=sab_d.unsafe_ptr(), src_buf=od)
    ctx.synchronize()

    var moved = 0
    var dist_moved = 0
    for i in range(FCHK_QUERIES * k):
        if (
            sab_i.unsafe_ptr().unsafe_load(i)
            != base_i.unsafe_ptr().unsafe_load(i)
        ):
            moved += 1
        if (
            sab_d.unsafe_ptr().unsafe_load(i)
            != base_d.unsafe_ptr().unsafe_load(i)
        ):
            dist_moved += 1
    if moved == 0:
        raise Error(
            "FUSED SABOTAGE 1 FAILED TO REGISTER: ramping index_norm changed"
            " no neighbour. The fused epilogue is not reading it, or the"
            " kernel never ran."
        )
    if dist_moved == 0:
        raise Error(
            "FUSED SABOTAGE 1 moved the neighbours but not the distances."
            " The stored value is not the value the reject compared."
        )

    # --- restore, then SABOTAGE 2: query norms, OFFSET. Set must hold. ---
    compute_norms(ctx, index, inorm, FCHK_INDEX, FCHK_FEATURES, False)
    var hqn = ctx.enqueue_create_host_buffer[DType.float32](FCHK_QUERIES)
    ctx.enqueue_copy(dst_ptr=hqn.unsafe_ptr(), src_buf=qnorm)
    ctx.synchronize()
    for i in range(FCHK_QUERIES):
        hqn.unsafe_ptr().unsafe_store(
            i, hqn.unsafe_ptr().unsafe_load(i) + Float32(0.1) * Float32(i + 1)
        )
    ctx.enqueue_copy(dst_buf=qnorm, src_ptr=hqn.unsafe_ptr())
    ctx.synchronize()

    fused_l2_knn(
        ctx, queries, qnorm, index, inorm, od, oi,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, k, False,
    )
    ctx.enqueue_copy(dst_ptr=sab_i.unsafe_ptr(), src_buf=oi)
    ctx.synchronize()

    var order_moved = 0
    for i in range(FCHK_QUERIES * k):
        if (
            sab_i.unsafe_ptr().unsafe_load(i)
            != base_i.unsafe_ptr().unsafe_load(i)
        ):
            order_moved += 1
    if order_moved != 0:
        raise Error(
            "FUSED SABOTAGE 2 CHANGED THE NEIGHBOURS for "
            + String(order_moved)
            + " slots. A per-query constant cannot change a ranking, so the"
            " distance is not being formed by the expanded identity."
        )

    print(
        "check_fused_reach_by_sabotage OK: index_norm ramp moved "
        + String(moved)
        + "/"
        + String(FCHK_QUERIES * k)
        + " neighbours and "
        + String(dist_moved)
        + " distances; query_norm offset moved 0, which is the predicted"
        " shape"
    )


def check_dispatch_takes_fused() raises:
    """`brute_force_knn_impl`'s dispatch, in all THREE of its directions.

    **The proof is which BUFFER comes back written, not whether the answer is
    right**, because both paths return the same neighbours and an equality
    check therefore proves nothing about which one ran.

    The two arms write different outputs. The fused kernel writes `out_idx`
    (`uint32`). The tiled fallback with `use_vendor_topk = True` writes
    `out_idx32` (`int32`) and never touches `out_idx`. So every call below
    passes `use_vendor_topk = True`, and:

      - at k = 8 with `knn_method = KNN_METHOD_FUSED`, their own dispatch, the
        fused arm must run, so `out_idx` must come back correct and
        `out_idx32` must still hold the sentinel;
      - at k = 65, one past their `k <= 64` at `:443`, the dispatch must fall
        through EVEN WITH `KNN_METHOD_FUSED` asked for, so `out_idx32` must
        come back correct and `out_idx` must still hold the sentinel;
      - at k = 8 with the argument OMITTED, the AUTO default must consult
        `fused_l2_knn_grid`: at THIS fixture (53 x 4,093) the computation
        picks grid (16, 4) -- the x-split regime -- so the TILED arm must
        run; and at a fourth fixture built to sit in the `grid_x == 1`
        regime (1,920 queries), the same omitted-argument call must take
        the FUSED arm. Together they assert the default is AUTO and that
        AUTO reads the geometry, not a constant.

    The first two together are what stops the arm selection being wired to a
    constant: one direction alone would pass either way. The third is what
    stops the DEFAULT drifting back silently, which is a different failure and
    needs its own assertion -- a check that only ever passes the method
    explicitly cannot see which value is the default.
    """
    var ctx = DeviceContext()
    var kf = 8
    var kt = 65
    var kmax = kt
    var tile = 32
    var buf_len = FCHK_INDEX // 8

    var index = ctx.enqueue_create_buffer[DType.float32](
        FCHK_INDEX * FCHK_FEATURES
    )
    var queries = ctx.enqueue_create_buffer[DType.float32](
        FCHK_QUERIES * FCHK_FEATURES
    )
    var inorm = ctx.enqueue_create_buffer[DType.float32](FCHK_INDEX)
    var qnorm = ctx.enqueue_create_buffer[DType.float32](FCHK_QUERIES)
    var dist = ctx.enqueue_create_buffer[DType.float32](tile * FCHK_INDEX)
    var bv = ctx.enqueue_create_buffer[DType.float32](tile * 2 * buf_len)
    var bi = ctx.enqueue_create_buffer[DType.uint32](tile * 2 * buf_len)
    var od = ctx.enqueue_create_buffer[DType.float32](FCHK_QUERIES * kmax)
    var oi = ctx.enqueue_create_buffer[DType.uint32](FCHK_QUERIES * kmax)
    var oi32 = ctx.enqueue_create_buffer[DType.int32](FCHK_QUERIES * kmax)
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_INDEX * FCHK_FEATURES
    )
    for j in range(FCHK_INDEX):
        for f in range(FCHK_FEATURES):
            hi.unsafe_ptr().unsafe_store(
                j * FCHK_FEATURES + f, _fchk_coord(j, f, 3)
            )
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    var hq = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_QUERIES * FCHK_FEATURES
    )
    for i in range(FCHK_QUERIES):
        for f in range(FCHK_FEATURES):
            hq.unsafe_ptr().unsafe_store(
                i * FCHK_FEATURES + f, _fchk_coord(i, f, 11)
            )
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()
    compute_norms(ctx, index, inorm, FCHK_INDEX, FCHK_FEATURES, False)
    compute_norms(ctx, queries, qnorm, FCHK_QUERIES, FCHK_FEATURES, False)
    ctx.synchronize()

    var truth = _fchk_oracle(hq.unsafe_ptr(), hi.unsafe_ptr())

    var seed_u = ctx.enqueue_create_host_buffer[DType.uint32](
        FCHK_QUERIES * kmax
    )
    var seed_i = ctx.enqueue_create_host_buffer[DType.int32](
        FCHK_QUERIES * kmax
    )
    for i in range(FCHK_QUERIES * kmax):
        seed_u.unsafe_ptr().unsafe_store(i, UInt32(0xDEADBEEF))
        seed_i.unsafe_ptr().unsafe_store(i, Int32(-777))

    var got_u = ctx.enqueue_create_host_buffer[DType.uint32](
        FCHK_QUERIES * kmax
    )
    var got_i = ctx.enqueue_create_host_buffer[DType.int32](
        FCHK_QUERIES * kmax
    )

    # --- k = 8, THEIR dispatch asked for explicitly: their fused arm ------
    ctx.enqueue_copy(dst_buf=oi, src_ptr=seed_u.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=oi32, src_ptr=seed_i.unsafe_ptr())
    ctx.synchronize()
    brute_force_knn_impl(
        ctx, queries, qnorm, index, inorm, dist, bv, bi, od, oi, oi32,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, kf, tile, buf_len, False,
        True, knn_method=KNN_METHOD_FUSED,
    )
    ctx.enqueue_copy(dst_ptr=got_u.unsafe_ptr(), src_buf=oi)
    ctx.enqueue_copy(dst_ptr=got_i.unsafe_ptr(), src_buf=oi32)
    ctx.synchronize()

    var bad = 0
    for i in range(FCHK_QUERIES):
        for s in range(kf):
            if (
                Int(got_u.unsafe_ptr().unsafe_load(i * kf + s))
                != truth[i * FCHK_MAX_K + s]
            ):
                bad += 1
    if bad != 0:
        raise Error(
            "DISPATCH at k=8 did not fill out_idx correctly (" + String(bad)
            + " slots wrong). Their `:443` sends k <= 64 to fusedL2Knn."
        )
    var untouched = 0
    for i in range(FCHK_QUERIES * kf):
        if got_i.unsafe_ptr().unsafe_load(i) == Int32(-777):
            untouched += 1
    if untouched != FCHK_QUERIES * kf:
        raise Error(
            "DISPATCH at k=8 wrote out_idx32, so the TILED arm ran. Only"
            " the fallback writes out_idx32 under use_vendor_topk."
        )

    # --- k = 65: one past their bound, so their fallback ------------------
    ctx.enqueue_copy(dst_buf=oi, src_ptr=seed_u.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=oi32, src_ptr=seed_i.unsafe_ptr())
    ctx.synchronize()
    brute_force_knn_impl(
        ctx, queries, qnorm, index, inorm, dist, bv, bi, od, oi, oi32,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, kt, tile, buf_len, False,
        True, knn_method=KNN_METHOD_FUSED,
    )
    ctx.enqueue_copy(dst_ptr=got_u.unsafe_ptr(), src_buf=oi)
    ctx.enqueue_copy(dst_ptr=got_i.unsafe_ptr(), src_buf=oi32)
    ctx.synchronize()

    # k = 65 exceeds the oracle's 64, so the first 64 slots are checked as a
    # SET against the true 64. The fallback's selector promises no order.
    var bad2 = 0
    for i in range(FCHK_QUERIES):
        for s in range(kt):
            var got = Int(got_i.unsafe_ptr().unsafe_load(i * kt + s))
            var found = False
            for t in range(FCHK_MAX_K):
                if truth[i * FCHK_MAX_K + t] == got:
                    found = True
            if not found:
                bad2 += 1
    if bad2 > FCHK_QUERIES:
        raise Error(
            "DISPATCH at k=65 returned " + String(bad2)
            + " neighbours outside the true top-64; at most one per query is"
            " expected, since the 65th is not in the oracle"
        )
    var untouched2 = 0
    for i in range(FCHK_QUERIES * kt):
        if got_u.unsafe_ptr().unsafe_load(i) == UInt32(0xDEADBEEF):
            untouched2 += 1
    if untouched2 != FCHK_QUERIES * kt:
        raise Error(
            "DISPATCH at k=65 wrote out_idx, so the FUSED arm ran. Their"
            " `:443` bound is k <= 64 and 65 must fall through."
        )

    # --- k = 8, ARGUMENT OMITTED, x-split shape: AUTO must pick TILED ---
    #
    # Same k, same fixture, same buffers as the first call. The ONLY thing
    # that differs is that `knn_method` is not passed, so this asserts the
    # value of the default and nothing else.
    ctx.enqueue_copy(dst_buf=oi, src_ptr=seed_u.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=oi32, src_ptr=seed_i.unsafe_ptr())
    ctx.synchronize()
    brute_force_knn_impl(
        ctx, queries, qnorm, index, inorm, dist, bv, bi, od, oi, oi32,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, kf, tile, buf_len, False,
        True,
    )
    ctx.enqueue_copy(dst_ptr=got_u.unsafe_ptr(), src_buf=oi)
    ctx.enqueue_copy(dst_ptr=got_i.unsafe_ptr(), src_buf=oi32)
    ctx.synchronize()

    var untouched3 = 0
    for i in range(FCHK_QUERIES * kf):
        if got_u.unsafe_ptr().unsafe_load(i) == UInt32(0xDEADBEEF):
            untouched3 += 1
    # BOTH MODES SEND THIS SHAPE TO TILED, for two DIFFERENT reasons, and
    # the check asserts the same thing twice on purpose. Under FAST, AUTO
    # consults the launch computation and this x-split shape is DEVIATION
    # 36's tiled regime. Under IDENTICAL the arm may not be chosen by SHAPE
    # at all -- the two arms break a tie in the k-th distance differently,
    # so a shape-dependent arm makes the ANSWER depend on the caller's query
    # count -- and DEVIATION 509 pins AUTO to TILED on every column,
    # superseding DEVIATION 502's pin to cuVS's own FUSED dispatch (that arm
    # REFUSES on a 64-lane column, so it cannot be the identical column's).
    # The FAST expectation is read off THIS column's launch computation,
    # not off the M4's (2026-08-23: the H100's 108 SMs give a different
    # grid, and a check that hardcodes "(16, 4) here" fails on every
    # non-Apple FAST leg for the geometry doing what DEVIATION 36 says).
    var g_small = fused_l2_knn_grid(FCHK_QUERIES, FCHK_INDEX)
    var want_fused_small = (
        False if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else g_small[0] == 1
    )
    if (not want_fused_small) and untouched3 != FCHK_QUERIES * kf:
        raise Error(
            "DEFAULT DISPATCH at k=8 on the (53 x 4,093) fixture wrote"
            " out_idx, so the FUSED arm ran. Under FAST that is DEVIATION"
            " 36's x-split regime (fused_l2_knn_grid picks grid ("
            + String(g_small[0]) + ", " + String(g_small[1])
            + ") here) and under IDENTICAL it is DEVIATION 509's pin; either"
            " the default moved or AUTO is not reading the mode."
        )
    if want_fused_small and untouched3 != 0:
        raise Error(
            "DEFAULT DISPATCH at k=8 on the (53 x 4,093) fixture left"
            " out_idx at its sentinel, so the TILED arm ran, but this"
            " column's fused_l2_knn_grid gives grid_x == 1 and DEVIATION 36"
            " (FAST) takes FUSED there; AUTO is not reading the geometry."
        )
    var bad3 = 0
    for i in range(FCHK_QUERIES):
        for s3 in range(kf):
            # Whichever arm ran wrote its own output buffer: the tiled arm
            # fills `out_idx32` and the fused arm `out_idx`.
            var got3 = Int(got_i.unsafe_ptr().unsafe_load(i * kf + s3))
            var found3 = False
            for t in range(FCHK_MAX_K):
                if truth[i * FCHK_MAX_K + t] == got3:
                    found3 = True
            if not found3:
                bad3 += 1
    if bad3 != 0:
        raise Error(
            "DEFAULT DISPATCH at k=8 returned " + String(bad3)
            + " neighbours outside the true top-64, so the arm it took is"
            " running but wrong"
        )

    # --- k = 8, ARGUMENT OMITTED, grid_x == 1 shape ------------------------
    #
    # 1,920 queries = 120 y-chunks at Mblk 16, exactly minGridSize on the M4,
    # so `fused_l2_knn_grid` returns grid_x == 1 (pinned in
    # check_launch_config_values). Under FAST the AUTO default must take the
    # FUSED arm here; under IDENTICAL it must take TILED at this shape TOO,
    # which is the whole content of DEVIATION 509's pin -- the arm is a
    # function of the MODE and of nothing else, so the two shapes in this
    # check must land on the same arm as each other in that mode. Only WHICH
    # buffer was written is asserted here; the arms' answers are
    # oracle-checked elsewhere on their own fixtures.
    var big_q = 1920
    var bqueries = ctx.enqueue_create_buffer[DType.float32](
        big_q * FCHK_FEATURES
    )
    var bqnorm = ctx.enqueue_create_buffer[DType.float32](big_q)
    var bod = ctx.enqueue_create_buffer[DType.float32](big_q * kf)
    var boi = ctx.enqueue_create_buffer[DType.uint32](big_q * kf)
    var boi32 = ctx.enqueue_create_buffer[DType.int32](big_q * kf)
    var bdist = ctx.enqueue_create_buffer[DType.float32](tile * FCHK_INDEX)
    ctx.synchronize()
    var hbq = ctx.enqueue_create_host_buffer[DType.float32](
        big_q * FCHK_FEATURES
    )
    for i in range(big_q):
        for f in range(FCHK_FEATURES):
            hbq.unsafe_ptr().unsafe_store(
                i * FCHK_FEATURES + f, _fchk_coord(i, f, 9)
            )
    ctx.enqueue_copy(dst_buf=bqueries, src_ptr=hbq.unsafe_ptr())
    var hbu = ctx.enqueue_create_host_buffer[DType.uint32](big_q * kf)
    var hbi = ctx.enqueue_create_host_buffer[DType.int32](big_q * kf)
    for i in range(big_q * kf):
        hbu.unsafe_ptr().unsafe_store(i, UInt32(0xDEADBEEF))
        hbi.unsafe_ptr().unsafe_store(i, Int32(-77))
    ctx.enqueue_copy(dst_buf=boi, src_ptr=hbu.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=boi32, src_ptr=hbi.unsafe_ptr())
    ctx.synchronize()
    compute_norms(ctx, bqueries, bqnorm, big_q, FCHK_FEATURES, False)
    brute_force_knn_impl(
        ctx, bqueries, bqnorm, index, inorm, bdist, bv, bi, bod, boi, boi32,
        big_q, FCHK_INDEX, FCHK_FEATURES, kf, tile, buf_len, False,
        True,
    )
    ctx.enqueue_copy(dst_ptr=hbu.unsafe_ptr(), src_buf=boi)
    ctx.enqueue_copy(dst_ptr=hbi.unsafe_ptr(), src_buf=boi32)
    ctx.synchronize()
    var fused_wrote = 0
    for i in range(big_q * kf):
        if hbu.unsafe_ptr().unsafe_load(i) != UInt32(0xDEADBEEF):
            fused_wrote += 1
    var tiled_untouched = 0
    for i in range(big_q * kf):
        if hbi.unsafe_ptr().unsafe_load(i) == Int32(-77):
            tiled_untouched += 1
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if fused_wrote != 0 or tiled_untouched != 0:
            raise Error(
                "DEFAULT DISPATCH at k=8 with 1,920 queries under IDENTICAL:"
                " expected the TILED arm at EVERY shape (DEVIATION 509), but"
                " out_idx has "
                + String(fused_wrote)
                + " written slots and out_idx32 has "
                + String(tiled_untouched)
                + " sentinel slots left; AUTO is choosing by geometry in a"
                " mode where the arm may not depend on the query count."
            )
    else:
        # "1,920 queries = minGridSize" is the M4's number; on this column
        # the expectation is whatever its own launch computation says.
        var g_big = fused_l2_knn_grid(big_q, FCHK_INDEX)
        if g_big[0] == 1:
            if fused_wrote != big_q * kf or tiled_untouched != big_q * kf:
                raise Error(
                    "DEFAULT DISPATCH at k=8 with 1,920 queries: expected"
                    " the FUSED arm (this column's grid_x == 1 regime), but"
                    " out_idx has "
                    + String(big_q * kf - fused_wrote)
                    + " sentinel slots left and out_idx32 has "
                    + String(big_q * kf - tiled_untouched)
                    + " written slots; AUTO is not reading the geometry"
                )
        else:
            if fused_wrote != 0 or tiled_untouched != 0:
                raise Error(
                    "DEFAULT DISPATCH at k=8 with 1,920 queries: this"
                    " column's fused_l2_knn_grid gives grid_x = "
                    + String(g_big[0])
                    + " so DEVIATION 36 takes TILED, but out_idx has "
                    + String(fused_wrote)
                    + " written slots; AUTO is not reading the geometry"
                )

    print(
        "check_dispatch_takes_fused OK: with KNN_METHOD_FUSED, k=8 wrote"
        " out_idx and left out_idx32 at its sentinel and k=65 did the"
        " opposite, so the `k <= 64` branch at knn_brute_force.cuh:443 is"
        " wired both ways; with the argument omitted, the AUTO default took",
        (
            "TILED on BOTH shapes, which is DEVIATION 509's pin (the arm"
            " may not depend on the query count when the two arms break"
            " ties differently, and the FUSED arm 502 used to pin to"
            " refuses outright on a 64-lane column)"
            if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
            else "the arm this column's fused_l2_knn_grid selects on each"
            " shape (FUSED iff grid_x == 1; on the M4: TILED on the"
            " (53 x 4,093) x-split shape and FUSED on the 1,920-query"
            " shape), which is DEVIATION 36 (revised)"
        ),
    )


# ---------------------------------------------------------------------------
# THE REGISTER-RESIDENT SELECTOR. Everything above this line was written
# against a shared-memory placeholder in the selector slot; these three are
# written against `faiss_select::WarpSelect`, which is what
# `fused_l2_knn.cuh:221-222` actually instantiates.
#
# The call contract that makes them necessary: `WarpSelect::checkThreadQ`
# (`Select.cuh:393-419`) holds a warp vote and a merge full of shuffles, so
# every lane of a warp must call `add` the same number of times and every
# guard around it must be warp-uniform. In `fusedL2kNN` the guards are
# `gmemRowId < m` (`:458`), which depends on `threadIdx.x / AccThCols` and is
# therefore constant across a warp, and `colId < ldd` (`:463`), which is NOT
# and is handled by feeding `{keyMax, identity}` rather than by skipping
# (`:462`). A fixture that never crosses either edge cannot see a violation,
# so `check_fused_edge_shapes` is built to cross both.
# ---------------------------------------------------------------------------


comptime FCHK2_INDEX = 1013
comptime FCHK2_QUERIES = 17
comptime FCHK2_FEATURES = 13


def _oracle_at(
    hq: MutPointer[Float32, MutUntrackedOrigin],
    hi: MutPointer[Float32, MutUntrackedOrigin],
    n_queries: Int,
    n_index: Int,
    n_features: Int,
    k: Int,
) -> List[Int]:
    """Host Float64 top-`k`, ascending, direct formula, at runtime shapes."""
    var out = List[Int]()
    for i in range(n_queries):
        var best_idx = List[Int]()
        var best_d = List[Float64]()
        for _s in range(k):
            best_idx.append(-1)
            best_d.append(1.0e30)
        for j in range(n_index):
            var d = Float64(0.0)
            for f in range(n_features):
                var diff = Float64(
                    hq.unsafe_load(i * n_features + f)
                ) - Float64(hi.unsafe_load(j * n_features + f))
                d += diff * diff
            if d < best_d[k - 1]:
                var pos = k - 1
                while pos > 0 and best_d[pos - 1] > d:
                    best_d[pos] = best_d[pos - 1]
                    best_idx[pos] = best_idx[pos - 1]
                    pos -= 1
                best_d[pos] = d
                best_idx[pos] = j
        for t in range(k):
            out.append(best_idx[t])
    return out^


def check_fused_edge_shapes() raises:
    """A second fused fixture whose every extent is a partial tile, at the
    two k values on either side of their instantiation boundary.

    `Policy2x8` is `Mblk = 16`, `Nblk = 256`, `Kblk = 16`, and this fixture
    is 17 queries x 1,013 index x 13 features:

      * **17 queries** puts a row tile with ONE live row in it. In that tile
        seven of the eight warps have `gmemRowId >= m` for both of their
        `AccRowsPerTh` rows and skip the queue entirely, and the eighth has
        `heap0` live and `heap1` dead. That is the only shape in which the
        `:458` guard can be caught being non-warp-uniform, and the whole
        `WarpSelect` call contract rests on it being uniform.
      * **1,013 index points** is 3 full column tiles and a 245-wide tail,
        and 1013 is not a multiple of the 32-lane warp either, so lanes with
        `colId >= n` must feed `{keyMax, identity}` (`:462`) instead of
        exiting -- a lane that exited early would desynchronize the vote.
      * **13 features** is a partial `Kblk`.

    `k = 1, 32, 33, 64`. Thirty-two and thirty-three straddle
    `fusedL2ExpKnnImpl:768-771`, so they select DIFFERENT whole-kernel
    instantiations, `<NumWarpQ=32, NumThreadQ=2>` and
    `<64, 3>`; 33 is also the first k whose `kNumWarpQRegisters` is 2, which
    is the only setting in which `warpK[1]` and the cross-register merge in
    `warpMergeAnyRegisters` are reached at all, and its
    `kLane = (33-1) % 32 = 0` is the wrap `Select.cuh:351` exists for.

    Compared PER SLOT and IN ORDER against a host Float64 direct-formula
    oracle, which is the same bar the placeholder selector cleared.
    """
    var ctx = DeviceContext()

    var index = ctx.enqueue_create_buffer[DType.float32](
        FCHK2_INDEX * FCHK2_FEATURES
    )
    var queries = ctx.enqueue_create_buffer[DType.float32](
        FCHK2_QUERIES * FCHK2_FEATURES
    )
    var inorm = ctx.enqueue_create_buffer[DType.float32](FCHK2_INDEX)
    var qnorm = ctx.enqueue_create_buffer[DType.float32](FCHK2_QUERIES)
    var od = ctx.enqueue_create_buffer[DType.float32](
        FCHK2_QUERIES * FCHK_MAX_K
    )
    var oi = ctx.enqueue_create_buffer[DType.uint32](
        FCHK2_QUERIES * FCHK_MAX_K
    )
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK2_INDEX * FCHK2_FEATURES
    )
    for j in range(FCHK2_INDEX):
        for f in range(FCHK2_FEATURES):
            hi.unsafe_ptr().unsafe_store(
                j * FCHK2_FEATURES + f, _fchk_coord(j, f, 53)
            )
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    var hq = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK2_QUERIES * FCHK2_FEATURES
    )
    for i in range(FCHK2_QUERIES):
        for f in range(FCHK2_FEATURES):
            hq.unsafe_ptr().unsafe_store(
                i * FCHK2_FEATURES + f, _fchk_coord(i, f, 61)
            )
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()

    compute_norms(ctx, index, inorm, FCHK2_INDEX, FCHK2_FEATURES, False)
    compute_norms(ctx, queries, qnorm, FCHK2_QUERIES, FCHK2_FEATURES, False)
    ctx.synchronize()

    var truth = _oracle_at(
        hq.unsafe_ptr(),
        hi.unsafe_ptr(),
        FCHK2_QUERIES,
        FCHK2_INDEX,
        FCHK2_FEATURES,
        FCHK_MAX_K,
    )

    var ho = ctx.enqueue_create_host_buffer[DType.uint32](
        FCHK2_QUERIES * FCHK_MAX_K
    )
    var hd = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK2_QUERIES * FCHK_MAX_K
    )

    var ks = List[Int]()
    ks.append(1)
    ks.append(32)
    ks.append(33)
    ks.append(FCHK_MAX_K)

    for t in range(len(ks)):
        var k = ks[t]
        fused_l2_knn(
            ctx, queries, qnorm, index, inorm, od, oi,
            FCHK2_QUERIES, FCHK2_INDEX, FCHK2_FEATURES, k, False,
        )
        ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=oi)
        ctx.enqueue_copy(dst_ptr=hd.unsafe_ptr(), src_buf=od)
        ctx.synchronize()

        var bad = 0
        var unsorted = 0
        for i in range(FCHK2_QUERIES):
            for s in range(k):
                if (
                    Int(ho.unsafe_ptr().unsafe_load(i * k + s))
                    != truth[i * FCHK_MAX_K + s]
                ):
                    bad += 1
                if s > 0:
                    if hd.unsafe_ptr().unsafe_load(
                        i * k + s
                    ) < hd.unsafe_ptr().unsafe_load(i * k + s - 1):
                        unsorted += 1
        if bad != 0:
            raise Error(
                "fused edge-shape k=" + String(k) + ": " + String(bad)
                + " of " + String(FCHK2_QUERIES * k)
                + " slots disagree with the host Float64 oracle at "
                + String(FCHK2_QUERIES) + " x " + String(FCHK2_INDEX)
                + " x " + String(FCHK2_FEATURES)
            )
        if unsorted != 0:
            raise Error(
                "fused edge-shape k=" + String(k) + ": " + String(unsorted)
                + " slots are out of ascending order"
            )

    print(
        "check_fused_edge_shapes OK: k = 1, 32, 33, 64 over "
        + String(FCHK2_QUERIES)
        + " queries x "
        + String(FCHK2_INDEX)
        + " index x "
        + String(FCHK2_FEATURES)
        + " features -- a partial tile on Mblk, Nblk and Kblk at once, and"
        " both WarpSelect instantiations"
    )


def check_fused_queue_reach_by_sabotage() raises:
    """A sabotage aimed at the SELECTOR, not at the epilogue.

    `check_fused_reach_by_sabotage` corrupts the norms, which proves the
    distance arithmetic ran. It would pass just as well with any selector in
    the slot. This one asserts a property only the queue can produce.

    Sabotage: overwrite ONE index point's coordinates with an exact copy of
    ONE query's, choosing an index point deep in the index so that it is
    reached on a LATE grid-stride column tile, and recompute the index
    norms.

    PREDICTED SHAPE, and every clause of it is sharp:

      1. That query's slot 0 must be that index point. It is at squared
         distance zero and the fixture is hashed, so nothing ties it. A
         queue that inserted at the wrong rank, or that dropped a candidate
         arriving after the queue was already full, fails here -- and the
         victim column is chosen past the first column tile precisely so
         that the queue IS already full when it arrives.
      2. That distance must be EXACTLY 0.0. `xn == yn` here and the expanded
         identity's round-off is about a float32 ulp at the norm, so this is
         `l2_exp.cuh:120-129`'s SECOND clamp clause and nothing else can
         produce the exact zero.
      3. That query's slots 1..k-1 must be its OLD slots 0..k-2, shifted by
         one, in index AND in distance. Only the victim column moved, and it
         was not a neighbour of anything before, so one new minimum must
         displace the tail by exactly one and change nothing else. A queue
         that rebuilt itself rather than inserting comes back with a
         different tail.
      4. Every OTHER query row must be bit-identical in value and in index.
         The fixture is checked up front to make this a THEOREM rather than
         a hope: the planted point sits at the victim query's coordinates,
         so it can only disturb query `i` if `dist(query_i, victim)` is
         inside query `i`'s baseline top-k, and the victim ROW is CHOSEN at
         run time as the first query for which the host, in Float64, finds
         that true of no other query. A queue leaking across rows, or a
         `starty` guard that is not warp-uniform, shows up here.
    """
    var ctx = DeviceContext()
    var k = 32
    comptime VICTIM_COL = 3971

    var index = ctx.enqueue_create_buffer[DType.float32](
        FCHK_INDEX * FCHK_FEATURES
    )
    var queries = ctx.enqueue_create_buffer[DType.float32](
        FCHK_QUERIES * FCHK_FEATURES
    )
    var inorm = ctx.enqueue_create_buffer[DType.float32](FCHK_INDEX)
    var qnorm = ctx.enqueue_create_buffer[DType.float32](FCHK_QUERIES)
    var od = ctx.enqueue_create_buffer[DType.float32](FCHK_QUERIES * k)
    var oi = ctx.enqueue_create_buffer[DType.uint32](FCHK_QUERIES * k)
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_INDEX * FCHK_FEATURES
    )
    for j in range(FCHK_INDEX):
        for f in range(FCHK_FEATURES):
            hi.unsafe_ptr().unsafe_store(
                j * FCHK_FEATURES + f, _fchk_coord(j, f, 3)
            )
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    var hq = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_QUERIES * FCHK_FEATURES
    )
    for i in range(FCHK_QUERIES):
        for f in range(FCHK_FEATURES):
            hq.unsafe_ptr().unsafe_store(
                i * FCHK_FEATURES + f, _fchk_coord(i, f, 11)
            )
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()

    compute_norms(ctx, index, inorm, FCHK_INDEX, FCHK_FEATURES, False)
    compute_norms(ctx, queries, qnorm, FCHK_QUERIES, FCHK_FEATURES, False)
    ctx.synchronize()

    fused_l2_knn(
        ctx, queries, qnorm, index, inorm, od, oi,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, k, False,
    )
    var base_i = ctx.enqueue_create_host_buffer[DType.uint32](
        FCHK_QUERIES * k
    )
    var base_d = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_QUERIES * k
    )
    ctx.enqueue_copy(dst_ptr=base_i.unsafe_ptr(), src_buf=oi)
    ctx.enqueue_copy(dst_ptr=base_d.unsafe_ptr(), src_buf=od)
    ctx.synchronize()

    # The victim column must not already be an answer anywhere, or clause 3
    # would be asserting the wrong shift and clause 4 the wrong invariance.
    for s in range(FCHK_QUERIES * k):
        if Int(base_i.unsafe_ptr().unsafe_load(s)) == VICTIM_COL:
            raise Error(
                "sabotage fixture is degenerate: column "
                + String(VICTIM_COL)
                + " is already a baseline neighbour; pick another"
            )

    # The victim ROW is CHOSEN, not typed, and the criterion is what makes
    # clause 4 a theorem about the fixture rather than an assumption about
    # the data: the planted point lands on query `v`'s coordinates, so it
    # can disturb query `i` only if `dist(query_i, query_v)` is inside query
    # `i`'s baseline top-k. Pick the first `v` for which no other query is
    # that close, in Float64. Queries this close DO occur here -- 20
    # dimensions and 53 queries -- so a typed constant was wrong twice.
    var victim_row = -1
    for v in range(FCHK_QUERIES):
        var ok = True
        for i in range(FCHK_QUERIES):
            if i == v:
                continue
            var dqq = Float64(0.0)
            for f in range(FCHK_FEATURES):
                var df = Float64(
                    hq.unsafe_ptr().unsafe_load(i * FCHK_FEATURES + f)
                ) - Float64(
                    hq.unsafe_ptr().unsafe_load(v * FCHK_FEATURES + f)
                )
                dqq += df * df
            if dqq <= Float64(base_d.unsafe_ptr().unsafe_load(i * k + k - 1)):
                ok = False
                break
        if ok:
            victim_row = v
            break
    if victim_row < 0:
        raise Error(
            "sabotage fixture is degenerate: every query is inside some"
            " other query's baseline top-k, so no victim row can isolate"
            " the planted point"
        )
    var VICTIM_ROW = victim_row

    for f in range(FCHK_FEATURES):
        hi.unsafe_ptr().unsafe_store(
            VICTIM_COL * FCHK_FEATURES + f,
            hq.unsafe_ptr().unsafe_load(VICTIM_ROW * FCHK_FEATURES + f),
        )
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    ctx.synchronize()
    compute_norms(ctx, index, inorm, FCHK_INDEX, FCHK_FEATURES, False)
    ctx.synchronize()

    fused_l2_knn(
        ctx, queries, qnorm, index, inorm, od, oi,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, k, False,
    )
    var sab_i = ctx.enqueue_create_host_buffer[DType.uint32](FCHK_QUERIES * k)
    var sab_d = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_QUERIES * k
    )
    ctx.enqueue_copy(dst_ptr=sab_i.unsafe_ptr(), src_buf=oi)
    ctx.enqueue_copy(dst_ptr=sab_d.unsafe_ptr(), src_buf=od)
    ctx.synchronize()

    # 1. rank 0 is the planted point
    if Int(sab_i.unsafe_ptr().unsafe_load(VICTIM_ROW * k)) != VICTIM_COL:
        raise Error(
            "QUEUE SABOTAGE clause 1: query "
            + String(VICTIM_ROW)
            + " came back with neighbour "
            + String(Int(sab_i.unsafe_ptr().unsafe_load(VICTIM_ROW * k)))
            + " at rank 0, not the co-located index point "
            + String(VICTIM_COL)
            + ". A candidate arriving on a late column tile is not reaching"
            " the top of the queue."
        )

    # 2. and at exactly zero, which is the second clamp clause
    var d0 = sab_d.unsafe_ptr().unsafe_load(VICTIM_ROW * k)
    if d0 != Float32(0.0):
        raise Error(
            "QUEUE SABOTAGE clause 2: the co-located pair came back at "
            + String(d0)
            + ", not exactly 0. `l2_exp.cuh:120-129`'s second clamp clause"
            " is not firing."
        )

    # 3. the rest of the victim row is its old answer, shifted by one
    for s in range(1, k):
        if (
            sab_i.unsafe_ptr().unsafe_load(VICTIM_ROW * k + s)
            != base_i.unsafe_ptr().unsafe_load(VICTIM_ROW * k + s - 1)
            or sab_d.unsafe_ptr().unsafe_load(VICTIM_ROW * k + s)
            != base_d.unsafe_ptr().unsafe_load(VICTIM_ROW * k + s - 1)
        ):
            raise Error(
                "QUEUE SABOTAGE clause 3: slot "
                + String(s)
                + " of the victim row is not its old slot "
                + String(s - 1)
                + ". Inserting one new minimum must shift the tail by"
                " exactly one."
            )

    # 4. no other row moved at all
    for i in range(FCHK_QUERIES):
        if i == VICTIM_ROW:
            continue
        for s in range(k):
            if (
                sab_i.unsafe_ptr().unsafe_load(i * k + s)
                != base_i.unsafe_ptr().unsafe_load(i * k + s)
                or sab_d.unsafe_ptr().unsafe_load(i * k + s)
                != base_d.unsafe_ptr().unsafe_load(i * k + s)
            ):
                raise Error(
                    "QUEUE SABOTAGE clause 4: query row "
                    + String(i)
                    + " moved when only index point "
                    + String(VICTIM_COL)
                    + " was touched, and the host verified that point is"
                    " outside row "
                    + String(i)
                    + "'s top-k. The queue is leaking across rows."
                )

    print(
        "check_fused_queue_reach_by_sabotage OK: a co-located point planted"
        " at index column "
        + String(VICTIM_COL)
        + " for query "
        + String(VICTIM_ROW)
        + " came back at rank 0 at exactly 0.0, shifted that row's tail by"
        " exactly one, and moved no other row"
    )


def check_fused_k_ceiling() raises:
    """`ASSERT(numOfNN <= 64, "fusedL2kNN: num of nearest neighbors must be
    <= 64")`, `fused_l2_knn.cuh:581` and `:759`.

    `check_dispatch_takes_fused` shows that `brute_force_knn_impl` sends
    k = 65 to the fallback. This asserts the other half: `fused_l2_knn`
    itself refuses k = 65 rather than silently instantiating a queue that
    cannot hold it, which is what their `else` branch at `:770-772` does.
    It also pins k = 64 as ACCEPTED, so the bound is tested from both sides
    and an off-by-one cannot hide.
    """
    var ctx = DeviceContext()

    var index = ctx.enqueue_create_buffer[DType.float32](
        FCHK2_INDEX * FCHK2_FEATURES
    )
    var queries = ctx.enqueue_create_buffer[DType.float32](
        FCHK2_QUERIES * FCHK2_FEATURES
    )
    var inorm = ctx.enqueue_create_buffer[DType.float32](FCHK2_INDEX)
    var qnorm = ctx.enqueue_create_buffer[DType.float32](FCHK2_QUERIES)
    var od = ctx.enqueue_create_buffer[DType.float32](FCHK2_QUERIES * 65)
    var oi = ctx.enqueue_create_buffer[DType.uint32](FCHK2_QUERIES * 65)
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK2_INDEX * FCHK2_FEATURES
    )
    for j in range(FCHK2_INDEX):
        for f in range(FCHK2_FEATURES):
            hi.unsafe_ptr().unsafe_store(
                j * FCHK2_FEATURES + f, _fchk_coord(j, f, 53)
            )
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    var hq = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK2_QUERIES * FCHK2_FEATURES
    )
    for i in range(FCHK2_QUERIES):
        for f in range(FCHK2_FEATURES):
            hq.unsafe_ptr().unsafe_store(
                i * FCHK2_FEATURES + f, _fchk_coord(i, f, 61)
            )
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()
    compute_norms(ctx, index, inorm, FCHK2_INDEX, FCHK2_FEATURES, False)
    compute_norms(ctx, queries, qnorm, FCHK2_QUERIES, FCHK2_FEATURES, False)
    ctx.synchronize()

    var accepted_64 = True
    try:
        fused_l2_knn(
            ctx, queries, qnorm, index, inorm, od, oi,
            FCHK2_QUERIES, FCHK2_INDEX, FCHK2_FEATURES, 64, False,
        )
    except:
        accepted_64 = False
    if not accepted_64:
        raise Error(
            "fused_l2_knn refused k = 64, which is their own NumWarpQ = 64"
            " instantiation at `fused_l2_knn.cuh:759`"
        )

    var refused_65 = False
    try:
        fused_l2_knn(
            ctx, queries, qnorm, index, inorm, od, oi,
            FCHK2_QUERIES, FCHK2_INDEX, FCHK2_FEATURES, 65, False,
        )
    except:
        refused_65 = True
    if not refused_65:
        raise Error(
            "fused_l2_knn ACCEPTED k = 65. Their `else` at"
            " `fused_l2_knn.cuh:770-772` asserts, and NumWarpQ = 64 cannot"
            " hold 65 neighbours, so this would return silent garbage."
        )

    print(
        "check_fused_k_ceiling OK: k = 64 accepted, k = 65 refused with"
        " their ASSERT message"
    )


# ---------------------------------------------------------------------------
# THE CROSS-BLOCK (`gridDim.x > 1`) ARM AND ITS LAUNCH COMPUTATION.
# `launchConfigGenerator` (`pairwise_distance_base.cuh:295-322`, M4 inputs)
# now chooses the grid, and `grid_x` selects between the single-block column
# sweep and the mutex merge -- a parameter that selects a kernel path, so
# the checks below enumerate BOTH sides of it explicitly (PORTING_RULES 8).
# ---------------------------------------------------------------------------


comptime FKNN_SMEM_BYTES = (FKNN_SMEM_PAGE_X + FKNN_SMEM_PAGE_Y) * 4


def check_launch_config_values() raises:
    """`launch_config_generator` at pinned inputs, against hand-transcribed
    evaluations of `pairwise_distance_base.cuh:308-319` with the M4 numbers
    (10 cores x 12 blocks of 256 threads = minGridSize 120).

    These are the branch cases: y-chunks alone fill the device (grid_x
    stays 1, grid_y capped), y-chunks exactly equal capacity, y-chunks just
    below (the smallest split, grid_x = 2), a small-m shape where the
    x-chunk cap binds, and the bench shape itself.
    """

    # THE NUMBERS ARE THE M4'S. On another column the same computation
    # returns that column's grid (the H100's 108 SMs x 8 blocks = 864 give
    # (7, 125) at 2,000 x 200,000, 2026-08-23), which is the generator
    # doing its job and not a failure; there the expectations below are
    # RECORDED beside what the column computes, and nothing is asserted.
    comptime pin_here = TARGET_COLUMN == COLUMN_APPLE

    def expect(m: Int, n: Int, want_x: Int, want_y: Int) raises:
        var cfg = launch_config_generator(
            m, n, FKNN_MBLK, FKNN_NBLK, FKNN_THREADS, FKNN_SMEM_BYTES
        )
        if not pin_here:
            print(
                "      launch config RECORDED on this column: (" + String(m)
                + ", " + String(n) + ") -> (" + String(cfg[0]) + ", "
                + String(cfg[1]) + "); the M4 transcription is ("
                + String(want_x) + ", " + String(want_y) + ")"
            )
            return
        if cfg[0] != want_x or cfg[1] != want_y:
            raise Error(
                "check_launch_config_values FAIL: m="
                + String(m)
                + " n="
                + String(n)
                + " got ("
                + String(cfg[0])
                + ", "
                + String(cfg[1])
                + ") want ("
                + String(want_x)
                + ", "
                + String(want_y)
                + ")"
            )

    # The bench shape: 2,000 queries is 125 y-chunks > 120, so THEIR OWN
    # COMPUTATION keeps gridDim.x == 1 there and caps grid_y at 120.
    expect(2000, 200000, 1, 120)
    expect(2000, 20000, 1, 120)
    # Exactly at capacity: 1,920 queries = 120 chunks, still no split.
    expect(1920, 200000, 1, 120)
    # Just below: 1,904 queries = 119 chunks; smallest i with 119*i >= 120
    # is 2, and x-chunks is huge, so grid_x = 2.
    expect(1904, 200000, 2, 119)
    # Small m: 53 queries = 4 chunks; smallest i with 4*i >= 120 is 30,
    # capped by x-chunks = ceil(4093 / 256) = 16.
    expect(53, 4093, 16, 4)
    # 100 queries = 7 chunks; i = 18, x-chunks caps at 16.
    expect(100, 4093, 16, 7)
    # Their `grid.x = i >= xChunks ? xChunks : i` arm where i binds:
    # 640 queries = 40 chunks; smallest i with 40*i >= 120 is 3 < 16.
    expect(640, 4093, 3, 40)

    # The per-threadgroup wall raises rather than mislaunching. 32 KB is
    # APPLE's wall (the hardware matrix's apple column); the H100's is
    # 48 KB and a 32,769-byte request is legal there (leg 9, 2026-08-23,
    # raised "did not raise"). Probe one byte past THIS column's wall.
    var wall = threadgroup_limit_for[TARGET_COLUMN]()
    var walled = False
    try:
        _ = launch_config_generator(
            64, 4096, FKNN_MBLK, FKNN_NBLK, FKNN_THREADS, wall + 1
        )
    except:
        walled = True
    if not walled:
        raise Error(
            "check_launch_config_values FAIL: a " + String(wall + 1)
            + "-byte threadgroup request (one past this column's "
            + String(wall) + "-byte wall) did not raise"
        )
    if pin_here:
        print(
            "check_launch_config_values OK: bench shape (2,000 q) computes"
            " grid (1, 120); 1,904 q is the smallest-split boundary (2, 119);"
            " 53 q gives (16, 4) with the x-chunk cap binding; 640 q gives"
            " (3, 40) with the occupancy term binding; 32 KB wall raises"
        )
    else:
        print(
            "check_launch_config_values OK: the M4 transcriptions are"
            " RECORDED beside this column's grids (above); the "
            + String(wall) + "-byte threadgroup wall raises one byte past"
        )


def check_fused_griddimx_merge() raises:
    """The mutex merge, on the FCHK fixture, both reached and correct.

    Four launches on identical inputs:
      1. the PUBLIC entry (grid computed: (16, 4) here, asserted) must
         match the host Float64 oracle per slot and in order;
      2. `sabotage = 1` at the same grid must MOVE the output (the last
         producer hands over identity/keyMax, so its candidates vanish) --
         reach-by-sabotage for the producer write AND the consumer merge;
      3. a FORCED `grid_x = 5` (a divisor the computation never picks, with
         a partial last x-tile at n = 4093) must match the oracle exactly;
      4. `sabotage = 1` at a FORCED `grid_x = 1` must NOT move anything,
         because the single-block arm never reads the mutex or the
         sabotage -- the other side of the switch, checked by name.
    """
    var ctx = DeviceContext()
    comptime K = 10

    var index = ctx.enqueue_create_buffer[DType.float32](
        FCHK_INDEX * FCHK_FEATURES
    )
    var queries = ctx.enqueue_create_buffer[DType.float32](
        FCHK_QUERIES * FCHK_FEATURES
    )
    var inorm = ctx.enqueue_create_buffer[DType.float32](FCHK_INDEX)
    var qnorm = ctx.enqueue_create_buffer[DType.float32](FCHK_QUERIES)
    var od = ctx.enqueue_create_buffer[DType.float32](FCHK_QUERIES * K)
    var oi = ctx.enqueue_create_buffer[DType.uint32](FCHK_QUERIES * K)
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_INDEX * FCHK_FEATURES
    )
    for j in range(FCHK_INDEX):
        for f in range(FCHK_FEATURES):
            hi.unsafe_ptr().unsafe_store(
                j * FCHK_FEATURES + f, _fchk_coord(j, f, 3)
            )
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    var hq = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_QUERIES * FCHK_FEATURES
    )
    for i in range(FCHK_QUERIES):
        for f in range(FCHK_FEATURES):
            hq.unsafe_ptr().unsafe_store(
                i * FCHK_FEATURES + f, _fchk_coord(i, f, 11)
            )
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()

    compute_norms(ctx, index, inorm, FCHK_INDEX, FCHK_FEATURES, False)
    compute_norms(ctx, queries, qnorm, FCHK_QUERIES, FCHK_FEATURES, False)
    ctx.synchronize()

    var truth = _fchk_oracle(hq.unsafe_ptr(), hi.unsafe_ptr())
    var ho = ctx.enqueue_create_host_buffer[DType.uint32](FCHK_QUERIES * K)

    # 1. Public entry. Guard first that the computation still picks a
    # multi-block grid here; if the M4 inputs ever change this, the check
    # must be re-pointed rather than silently degrade to grid_x == 1.
    var cfg = launch_config_generator(
        FCHK_QUERIES, FCHK_INDEX, FKNN_MBLK, FKNN_NBLK, FKNN_THREADS,
        FKNN_SMEM_BYTES,
    )
    if cfg[0] <= 1:
        raise Error(
            "check_fused_griddimx_merge FAIL: the computed grid_x is not"
            " > 1 on the FCHK shape; this check no longer covers the merge"
        )
    fused_l2_knn(
        ctx, queries, qnorm, index, inorm, od, oi,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, K, False,
    )
    ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=oi)
    ctx.synchronize()
    var bad = 0
    for i in range(FCHK_QUERIES):
        for s in range(K):
            if Int(ho.unsafe_ptr().unsafe_load(i * K + s)) != truth[
                i * FCHK_MAX_K + s
            ]:
                bad += 1
    if bad != 0:
        raise Error(
            "check_fused_griddimx_merge FAIL: public entry at grid_x="
            + String(cfg[0])
            + " has "
            + String(bad)
            + " wrong slots against the host oracle"
        )

    # 2. Sabotaged merge at the same grid must move the output.
    fused_l2_knn_launch(
        ctx, queries, qnorm, index, inorm, od, oi,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, K, False,
        cfg[0], cfg[1], 1,
    )
    ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=oi)
    ctx.synchronize()
    var moved = 0
    for i in range(FCHK_QUERIES):
        for s in range(K):
            if Int(ho.unsafe_ptr().unsafe_load(i * K + s)) != truth[
                i * FCHK_MAX_K + s
            ]:
                moved += 1
    if moved == 0:
        raise Error(
            "check_fused_griddimx_merge FAIL: poisoning the last producer's"
            " handoff moved nothing, so the merge path is not carrying the"
            " output"
        )

    # 3. A forced grid the computation never picks: grid_x = 5 leaves a
    # ragged block-to-column mapping and the last x-tile at 4093 partial.
    fused_l2_knn_launch(
        ctx, queries, qnorm, index, inorm, od, oi,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, K, False,
        5, cfg[1], 0,
    )
    ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=oi)
    ctx.synchronize()
    var bad5 = 0
    for i in range(FCHK_QUERIES):
        for s in range(K):
            if Int(ho.unsafe_ptr().unsafe_load(i * K + s)) != truth[
                i * FCHK_MAX_K + s
            ]:
                bad5 += 1
    if bad5 != 0:
        raise Error(
            "check_fused_griddimx_merge FAIL: forced grid_x=5 has "
            + String(bad5)
            + " wrong slots"
        )

    # 4. The other side of the switch: at grid_x = 1 the sabotage value is
    # never read, so a sabotaged single-block launch must be EXACT.
    fused_l2_knn_launch(
        ctx, queries, qnorm, index, inorm, od, oi,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, K, False,
        1, (FCHK_QUERIES + FKNN_MBLK - 1) // FKNN_MBLK, 1,
    )
    ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=oi)
    ctx.synchronize()
    var bad1 = 0
    for i in range(FCHK_QUERIES):
        for s in range(K):
            if Int(ho.unsafe_ptr().unsafe_load(i * K + s)) != truth[
                i * FCHK_MAX_K + s
            ]:
                bad1 += 1
    if bad1 != 0:
        raise Error(
            "check_fused_griddimx_merge FAIL: grid_x=1 with sabotage=1"
            " should be exact (the value is never read there) but has "
            + String(bad1)
            + " wrong slots"
        )

    print(
        "check_fused_griddimx_merge OK: computed grid ("
        + String(cfg[0])
        + ", "
        + String(cfg[1])
        + ") matches the oracle per slot at k=10; poisoning the last"
        " producer moved "
        + String(moved)
        + " slots; forced grid_x=5 with a partial last x-tile is exact;"
        " grid_x=1 ignores the sabotage entirely"
    )


comptime GX1_QUERIES = 2000
comptime GX1_INDEX = 1013
comptime GX1_FEATURES = 13
comptime GX1_K = 10


def check_fused_griddimx_one_capped_y() raises:
    """The `grid_x == 1` side AT SCALE, where `launchConfigGenerator` caps
    `grid_y` below the y-chunk count and the kernel's NEW outer m-loop
    (`pairwise_distance_base.cuh:129`) must grid-stride the rows.

    2,000 queries is 125 y-chunks against a 120-block cap, so five blocks
    take a second row tile -- the bench shape's exact control flow. Every
    slot must match a host Float64 oracle, in order.
    """
    var ctx = DeviceContext()

    var cfg = launch_config_generator(
        GX1_QUERIES, GX1_INDEX, FKNN_MBLK, FKNN_NBLK, FKNN_THREADS,
        FKNN_SMEM_BYTES,
    )
    if cfg[0] != 1 or cfg[1] >= (GX1_QUERIES + FKNN_MBLK - 1) // FKNN_MBLK:
        # The capped-y regime exists at this shape on the M4 (125 chunks
        # against a 120-block cap). On a column with more blocks the same
        # computation splits x instead (NVIDIA column: (4, 125)), and the
        # regime this check exercises is simply not selected here -- that is
        # the generator working, RECORDED, not a failure (2026-08-23).
        comptime if TARGET_COLUMN != COLUMN_APPLE:
            print(
                "check_fused_griddimx_one_capped_y RECORDED: this column"
                " computes (" + String(cfg[0]) + ", " + String(cfg[1])
                + ") at " + String(GX1_QUERIES) + " x " + String(GX1_INDEX)
                + ", not the M4's capped-y regime (1, 120); the regime is"
                " exercised where it exists"
            )
            return
        raise Error(
            "check_fused_griddimx_one_capped_y FAIL: expected grid_x == 1"
            " with grid_y capped below the y-chunk count, got ("
            + String(cfg[0])
            + ", "
            + String(cfg[1])
            + ")"
        )

    var index = ctx.enqueue_create_buffer[DType.float32](
        GX1_INDEX * GX1_FEATURES
    )
    var queries = ctx.enqueue_create_buffer[DType.float32](
        GX1_QUERIES * GX1_FEATURES
    )
    var inorm = ctx.enqueue_create_buffer[DType.float32](GX1_INDEX)
    var qnorm = ctx.enqueue_create_buffer[DType.float32](GX1_QUERIES)
    var od = ctx.enqueue_create_buffer[DType.float32](GX1_QUERIES * GX1_K)
    var oi = ctx.enqueue_create_buffer[DType.uint32](GX1_QUERIES * GX1_K)
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](
        GX1_INDEX * GX1_FEATURES
    )
    for j in range(GX1_INDEX):
        for f in range(GX1_FEATURES):
            hi.unsafe_ptr().unsafe_store(
                j * GX1_FEATURES + f, _fchk_coord(j, f, 31)
            )
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    var hq = ctx.enqueue_create_host_buffer[DType.float32](
        GX1_QUERIES * GX1_FEATURES
    )
    for i in range(GX1_QUERIES):
        for f in range(GX1_FEATURES):
            hq.unsafe_ptr().unsafe_store(
                i * GX1_FEATURES + f, _fchk_coord(i, f, 47)
            )
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()

    compute_norms(ctx, index, inorm, GX1_INDEX, GX1_FEATURES, False)
    compute_norms(ctx, queries, qnorm, GX1_QUERIES, GX1_FEATURES, False)
    ctx.synchronize()

    fused_l2_knn(
        ctx, queries, qnorm, index, inorm, od, oi,
        GX1_QUERIES, GX1_INDEX, GX1_FEATURES, GX1_K, False,
    )
    var ho = ctx.enqueue_create_host_buffer[DType.uint32](
        GX1_QUERIES * GX1_K
    )
    ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=oi)
    ctx.synchronize()

    var bad = 0
    for i in range(GX1_QUERIES):
        # Host Float64 top-k for this row, direct formula.
        var best_idx = List[Int]()
        var best_d = List[Float64]()
        for _s in range(GX1_K):
            best_idx.append(-1)
            best_d.append(1.0e30)
        for j in range(GX1_INDEX):
            var d = Float64(0.0)
            for f in range(GX1_FEATURES):
                var diff = Float64(
                    hq.unsafe_ptr().unsafe_load(i * GX1_FEATURES + f)
                ) - Float64(
                    hi.unsafe_ptr().unsafe_load(j * GX1_FEATURES + f)
                )
                d += diff * diff
            if d < best_d[GX1_K - 1]:
                var pos = GX1_K - 1
                while pos > 0 and best_d[pos - 1] > d:
                    best_d[pos] = best_d[pos - 1]
                    best_idx[pos] = best_idx[pos - 1]
                    pos -= 1
                best_d[pos] = d
                best_idx[pos] = j
        for s in range(GX1_K):
            if Int(
                ho.unsafe_ptr().unsafe_load(i * GX1_K + s)
            ) != best_idx[s]:
                bad += 1
    if bad != 0:
        raise Error(
            "check_fused_griddimx_one_capped_y FAIL: "
            + String(bad)
            + " wrong slots at grid ("
            + String(cfg[0])
            + ", "
            + String(cfg[1])
            + ") over "
            + String(GX1_QUERIES)
            + " queries"
        )
    print(
        "check_fused_griddimx_one_capped_y OK: 2,000 queries x 1,013 index"
        " ran at computed grid (1, "
        + String(cfg[1])
        + ") -- 125 row tiles over "
        + String(cfg[1])
        + " blocks, so the new outer grid-stride loop carried five blocks"
        " to a second tile -- and every slot matches the host oracle in"
        " order"
    )
