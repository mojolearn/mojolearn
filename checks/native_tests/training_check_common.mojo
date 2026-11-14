# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the training-family checks ported from the Python tests
(lane tmig-training, 2026-11-14; docs/plans/TEST_MIGRATION_2026-11-14.md).

No `main()`: importable by every `training_*_cases` / `*_suite` module.
`all_close` is NumPy's `assert_allclose` rule, |actual - desired| <=
atol + rtol * |desired|, element by element, so a ported tolerance keeps its
meaning. `uniform` is a counter-based generator (SplitMix64 of seed and
index), so fixture values do not depend on the order a pool computes them in.
"""
from std.math import isfinite


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def close64(actual: Float64, desired: Float64, rtol: Float64, atol: Float64) -> Bool:
    if not isfinite(actual) or not isfinite(desired):
        return False
    return abs(actual - desired) <= atol + rtol * abs(desired)


def all_close(actual: List[Float32], desired: List[Float64], rtol: Float64, atol: Float64) -> Bool:
    if len(actual) != len(desired):
        return False
    for i in range(len(actual)):  # small-loop(check comparison): fixture-sized, no product work
        if not close64(Float64(actual[i]), desired[i], rtol, atol):
            return False
    return True


def all_close64(actual: List[Float64], desired: List[Float64], rtol: Float64, atol: Float64) -> Bool:
    if len(actual) != len(desired):
        return False
    for i in range(len(actual)):  # small-loop(check comparison): fixture-sized, no product work
        if not close64(actual[i], desired[i], rtol, atol):
            return False
    return True


def same_words(first: List[Float32], second: List[Float32]) -> Bool:
    if len(first) != len(second):
        return False
    for i in range(len(first)):  # small-loop(check comparison): fixture-sized, no product work
        if first[i].to_bits() != second[i].to_bits():
            return False
    return True


def all_finite(values: List[Float32]) -> Bool:
    for i in range(len(values)):  # small-loop(check comparison): fixture-sized, no product work
        if not isfinite(values[i]):
            return False
    return True


def _mix(seed: Int, index: Int) -> UInt64:
    var z = UInt64(seed) * UInt64(0x9E3779B97F4A7C15) + UInt64(index) + UInt64(0x632BE59BD9B4E019)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def uniform(seed: Int, index: Int, lo: Float64, hi: Float64) -> Float32:
    """A float32 in [lo, hi) from the 53 high bits of SplitMix64(seed, index)."""
    var unit = Float64(_mix(seed, index) >> 11) * (1.0 / 9007199254740992.0)
    return Float32(lo + (hi - lo) * unit)


def uniform_int(seed: Int, index: Int, count: Int) -> Int:
    """An integer in [0, count)."""
    return Int(_mix(seed, index) % UInt64(count))


def to_f64(values: List[Float32]) -> List[Float64]:
    var result = List[Float64](capacity=len(values))
    for i in range(len(values)):  # small-loop(check widening): fixture-sized, no product work
        result.append(Float64(values[i]))
    return result^
