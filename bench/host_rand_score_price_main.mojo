# SPDX-License-Identifier: Apache-2.0
"""Price the exact linear host rand score against its quadratic definition.

    pixi run mojo run -I . bench/host_rand_score_price_main.mojo
"""
from std.memory import bitcast
from std.time import perf_counter_ns

from metrics.host.classification_oracle import host_rand_score


def _quadratic(first: List[Int32], second: List[Int32]) -> Float64:
    var n = len(first)
    var agreeing = Int64(0)
    for i in range(n):
        for j in range(i):
            if (first[i] == first[j]) == (second[i] == second[j]):
                agreeing += 1
    var pairs = Int64(n) * Int64(n - 1) // 2
    return Float64(agreeing) / Float64(pairs)


def main() raises:
    var n = 10000
    var first = List[Int32](length=n, fill=Int32(0))
    var second = List[Int32](length=n, fill=Int32(0))
    for i in range(n):
        first[i] = Int32(((i * 37 + 11) % 65) - 32)
        second[i] = Int32(((i * i + i * 13 + 7) % 41) - 17)
    # Warm the hash tables before taking the five production samples.
    _ = host_rand_score(first, second, n)
    var t0 = perf_counter_ns()
    var expected = _quadratic(first, second)
    var t1 = perf_counter_ns()
    for rep in range(5):
        var start = perf_counter_ns()
        var actual = host_rand_score(first, second, n)
        var finish = perf_counter_ns()
        if bitcast[DType.uint64](actual) != bitcast[DType.uint64](expected):
            raise Error("host rand_score timing run changed the result")
        print("HOST_RAND_SCORE_PRICE", rep, Float64(finish - start) / 1.0e6,
              bitcast[DType.uint64](actual))
    print("HOST_RAND_SCORE_QUADRATIC_MS", Float64(t1 - t0) / 1.0e6)
