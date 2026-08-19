"""Launch the brute-force k-NN, and sabotage it.

NO CUVS COUNTERPART. Same discipline as `cluster/mojo_only/kmeans_check.mojo`
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

from core.row_norms import NORM_TPB, row_norm_kernel
from neighbors.ported.neighbors.detail.fused_l2_knn import fused_l2_knn
from neighbors.ported.neighbors.detail.knn_brute_force import (
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
    """`brute_force_knn_impl` must route k <= 64 to the fused kernel.

    **The proof is which BUFFER comes back written, not whether the answer is
    right**, because both paths return the same neighbours and an equality
    check therefore proves nothing about which one ran.

    The two arms write different outputs. The fused kernel writes `out_idx`
    (`uint32`). The tiled fallback with `use_vendor_topk = True` writes
    `out_idx32` (`int32`) and never touches `out_idx`. So both calls below
    pass `use_vendor_topk = True`, and:

      - at k = 8 the dispatch must take the fused arm, so `out_idx` must come
        back correct and `out_idx32` must still hold the sentinel;
      - at k = 65, one past their `k <= 64` at `:443`, the dispatch must fall
        through, so `out_idx32` must come back correct and `out_idx` must
        still hold the sentinel.

    Both directions are asserted. One direction alone would pass if the
    dispatch were wired to a constant.
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

    # --- k = 8: their fused arm ------------------------------------------
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
        True,
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

    print(
        "check_dispatch_takes_fused OK: k=8 wrote out_idx and left out_idx32"
        " at its sentinel; k=65 did the opposite, so the `k <= 64` branch at"
        " knn_brute_force.cuh:443 is wired both ways"
    )
