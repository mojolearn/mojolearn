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
That checks our GPU expanded-identity answer against an independent
computation rather than against a rearrangement of the same one.

The k nearest are compared as a SET. Radix select returns them unordered, and
`select_radix.mojo` documents why their tie handling makes even the identity
of a tied neighbor non-reproducible.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.row_norms import NORM_TPB, row_norm_kernel
from neighbors.ported.neighbors.detail.knn_brute_force import (
    compute_norms,
    tiled_brute_force_knn,
)


comptime KNN_INDEX = 4096
comptime KNN_QUERIES = 64
comptime KNN_FEATURES = 16
comptime KNN_K = 8
comptime KNN_TILE = 64


def _coord(row: Int, feature: Int, salt: Int) -> Float32:
    """Uniform in [0, 1), hashed so it is reproducible and not structured."""
    var h = (row * 2654435761 + feature * 40503 + salt * 2246822519) % 1000003
    return Float32(h) / Float32(1000003)


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
        buf_idx, out_dist, out_idx, KNN_QUERIES, KNN_INDEX, KNN_FEATURES,
        KNN_K, KNN_TILE, buf_len, False,
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
        buf_idx, out_dist, out_idx, KNN_QUERIES, KNN_INDEX, KNN_FEATURES,
        KNN_K, KNN_TILE, buf_len, False,
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
        buf_idx, out_dist, out_idx, KNN_QUERIES, KNN_INDEX, KNN_FEATURES,
        KNN_K, KNN_TILE, buf_len, False,
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
        buf_idx, out_dist, out_idx, KNN_QUERIES, KNN_INDEX, KNN_FEATURES,
        KNN_K, KNN_TILE, buf_len, False,
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
