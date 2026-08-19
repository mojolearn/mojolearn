"""Launch the brute-force k-NN, and sabotage it.

NO CUVS COUNTERPART. Same discipline as `cluster/mojo_only/kmeans_check.mojo`
and for the same reason: a kernel is not ported until it has been enqueued
(`PORTING.md 9`), and a correct answer is not by itself evidence that a
kernel ran.

THE FIXTURE HAS AN EXACTLY KNOWN ANSWER
---------------------------------------
Index point `j` sits at `100 * j` in feature 0 and carries a hashed jitter
below 0.5 in every other feature. A query sits near `100 * c`. Because the
feature-0 separation is 100 and the jitter contributes at most
`n_features * 0.25`, the ranking by distance is EXACTLY the ranking by
`|j - c|`, so the true k nearest are known in closed form and the check does
not depend on any tolerance.

The jitter is there on purpose rather than for realism. Without it every
index point would be identical in three of four features, and a kernel that
read the wrong feature would still produce the right answer.

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
comptime KNN_FEATURES = 4
comptime KNN_K = 8
comptime KNN_TILE = 64
comptime KNN_SPACING = 100


def _jitter(row: Int, feature: Int) -> Float32:
    var h = (row * 2654435761 + feature * 40503) % 1021
    return Float32(h) / Float32(1021) - Float32(0.5)


def _query_center(query: Int) -> Int:
    """Spread the queries across the index, avoiding both ends."""
    return KNN_K + (query * 61) % (KNN_INDEX - 2 * KNN_K)


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
            var v = _jitter(j, f)
            if f == 0:
                v = Float32(KNN_SPACING * j) + v
            hi.unsafe_ptr().unsafe_store(j * KNN_FEATURES + f, v)
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())

    var hq = ctx.enqueue_create_host_buffer[DType.float32](
        KNN_QUERIES * KNN_FEATURES
    )
    for i in range(KNN_QUERIES):
        for f in range(KNN_FEATURES):
            var v = Float32(0.0)
            if f == 0:
                # Offset by 0.3 of a spacing so no two distances tie.
                v = Float32(KNN_SPACING) * (
                    Float32(_query_center(i)) + Float32(0.3)
                )
            hq.unsafe_ptr().unsafe_store(i * KNN_FEATURES + f, v)
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

    var bad = 0
    for i in range(KNN_QUERIES):
        var c = _query_center(i)
        # True set: the K index points whose |j - (c + 0.3)| is smallest.
        # With the 0.3 offset that is c, c+1, c-1, c+2, c-2, ... in order.
        var truth = List[Int]()
        truth.append(c)
        var step = 1
        while len(truth) < KNN_K:
            truth.append(c + step)
            if len(truth) < KNN_K:
                truth.append(c - step)
            step += 1

        for slot in range(KNN_K):
            var got = Int(ho.unsafe_ptr().unsafe_load(i * KNN_K + slot))
            var found = False
            for t in range(len(truth)):
                if truth[t] == got:
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
            var v = _jitter(j, f)
            if f == 0:
                v = Float32(KNN_SPACING * j) + v
            hi.unsafe_ptr().unsafe_store(j * KNN_FEATURES + f, v)
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())

    var hq = ctx.enqueue_create_host_buffer[DType.float32](
        KNN_QUERIES * KNN_FEATURES
    )
    for i in range(KNN_QUERIES):
        for f in range(KNN_FEATURES):
            var v = Float32(0.0)
            if f == 0:
                v = Float32(KNN_SPACING) * (
                    Float32(_query_center(i)) + Float32(0.3)
                )
            hq.unsafe_ptr().unsafe_store(i * KNN_FEATURES + f, v)
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
            hqn.unsafe_ptr().unsafe_load(i) + Float32(5000.0) * Float32(i + 1),
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
