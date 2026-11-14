# SPDX-License-Identifier: Apache-2.0
"""Brute-force k-NN under metric='inner_product' on the device.

Ported from `test_inner_product_is_the_largest_products` in
`python/mojolearn/tests/test_knn_metrics.py`: the k = 5 neighbors of 20
queries among 200 seven-feature rows (the queries are the first 20 rows) are
the largest float64 products (stable order), on at least 97% of the slots,
and each returned value is that product within rtol 1e-4, atol 1e-4. It calls
`neighbors/estimator.mojo::knn_search` as the public kneighbors does (the
selection runs over negated products; `negate_products` hands the products
back). The rows are standard normal (Box-Muller over a SplitMix64 stream
regenerated here; the claim holds for any such data). The scipy and
scikit-learn comparisons stay in the Python file (external oracles).

One fixture on the one physical device; the search keeps its own kernel
parallelism, and the FP64 oracle scores queries across host cores.
"""
from std.math import cos, log, sqrt
from max.gpu.host import DeviceContext
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, u32_ptr
from neighbors.estimator import knn_search, DEFAULT_QUERY_TILE
from neighbors.impl.detail.knn_brute_force import KNN_METHOD_AUTO
from neighbors.impl.distance.detail.distance_ops import DIST_INNER_PRODUCT


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _next(mut state: UInt64) -> UInt64:
    """SplitMix64."""
    state += UInt64(0x9E3779B97F4A7C15)
    var z = state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def normal_rows(seed: Int, count: Int) -> List[Float32]:
    var state = UInt64(seed) * UInt64(0x2545F4914F6CDD1D) + UInt64(1)
    var values = List[Float32](length=count, fill=0)
    for i in range(count):
        var u1 = (Float64(Int(_next(state) >> 11)) + 1.0) * (1.0 / 9007199254740992.0)
        var u2 = Float64(Int(_next(state) >> 11)) * (1.0 / 9007199254740992.0)
        values[i] = Float32(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2))
    return values^


def run_checks() raises:
    comptime N = 200
    comptime D = 7
    comptime Q = 20
    comptime K = 5
    var x = normal_rows(6, N * D)
    var distances = List[Float32](length=Q * K, fill=-9.0)
    var ids = List[UInt32](length=Q * K, fill=UInt32(0xFFFFFFFF))
    var ctx = DeviceContext()
    var xa = Int(x.unsafe_ptr())
    _ = knn_search(ctx, f32_ptr(xa), N, f32_ptr(xa), Q, D, K, f32_ptr(Int(distances.unsafe_ptr())),
        u32_ptr(Int(ids.unsafe_ptr())), True, DEFAULT_QUERY_TILE, KNN_METHOD_AUTO, DIST_INNER_PRODUCT, Float32(2.0), True)
    # FP64 oracle: per query, the K largest products, ties by lower index.
    var ref_ids = List[Int](length=Q * K, fill=-1)
    var ref_dots = List[Float64](length=Q * K, fill=0)
    var rp = ref_ids.unsafe_ptr()
    var dp = ref_dots.unsafe_ptr()
    var xp = x.unsafe_ptr()
    def oracle(row: Int) {imm rp, imm dp, imm xp}:
        var dots = List[Float64](length=N, fill=0)
        for j in range(N):
            var s = Float64(0)
            for f in range(D):
                s += Float64(xp[row * D + f]) * Float64(xp[j * D + f])
            dots[j] = s
        for slot in range(K):
            var best = -1
            for j in range(N):
                var taken = False
                for t in range(slot):
                    if rp[row * K + t] == j:
                        taken = True
                if not taken and (best < 0 or dots[j] > dots[best]):
                    best = j
            rp[row * K + slot] = best
            dp[row * K + slot] = dots[best]
    host_parallelize(oracle, Q)
    var same = 0
    var far = 0
    for c in range(Q * K):
        same += Int(Int(ids[c]) == ref_ids[c])
        var want = ref_dots[c]
        if abs(Float64(distances[c]) - want) > 1e-4 + 1e-4 * abs(want):
            far += 1
    require(Float64(same) >= 0.97 * Float64(Q * K), "inner product: only " + String(same) + " of "
        + String(Q * K) + " ids are the largest float64 products")
    require(far == 0, "inner product: " + String(far) + " returned products differ from the float64 products")
    _ = x^
    _ = distances^
    _ = ids^
    _ = ref_ids^
    _ = ref_dots^
    print("PASS neighbors_knn_metrics_gpu: inner_product k-NN against the float64 products")
