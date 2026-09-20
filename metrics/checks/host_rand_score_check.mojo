# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Exact host rand-score bucket-count gate.

    pixi run mojo run -I . metrics/checks/host_rand_score_check.mojo
"""
from std.memory import bitcast

from metrics.host.classification_oracle import host_rand_score


def _quadratic(first: List[Int32], second: List[Int32]) -> Float64:
    var n = len(first)
    if n < 2:
        return 1.0
    var agreeing = Int64(0)
    for i in range(n):
        for j in range(i):
            if (first[i] == first[j]) == (second[i] == second[j]):
                agreeing += 1
    var pairs = Int64(n) * Int64(n - 1) // 2
    return Float64(agreeing) / Float64(pairs)


def _check(first: List[Int32], second: List[Int32]) raises:
    var expected = _quadratic(first, second)
    var actual = host_rand_score(first, second, len(first))
    if bitcast[DType.uint64](actual) != bitcast[DType.uint64](expected):
        raise Error("host rand_score bucket numerator differs from pair walk")


def main() raises:
    _check(List[Int32](), List[Int32]())
    var one: List[Int32] = [Int32(7)]
    var one_other: List[Int32] = [Int32(-9)]
    _check(one, one_other)
    var small_first: List[Int32] = [Int32(-3), Int32(-3), Int32(7), Int32(8)]
    var small_second: List[Int32] = [Int32(9), Int32(9), Int32(9), Int32(-2)]
    _check(small_first, small_second)
    for n in range(2, 258):
        var first = List[Int32](length=n, fill=Int32(0))
        var second = List[Int32](length=n, fill=Int32(0))
        for i in range(n):
            first[i] = Int32(((i * 37 + n * 11) % 29) - 14)
            second[i] = Int32(((i * i + i * 13 + n * 7) % 31) - 15)
        _check(first, second)
    print("PASS host rand_score: exact pair numerator, negative labels, 0..257 rows")
