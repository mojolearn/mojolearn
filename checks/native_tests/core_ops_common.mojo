# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the tmig-core-2 session checks (core_ops_*,
core_vjp_rank, core_ops_layernorm_view, core_ops_program,
core_ops_import_kinds). The session helpers are ops_higher_cases' (tensor,
op1, op2, total, scale, values, gradient, same_words); this module adds the
seeded data the moved tests drew from NumPy, regenerated natively, the
remaining op spellings of `mojolearn.ops.Session` and the comparisons.

Seeded data: the moved tests' properties (a program equals its eager
session word for word, a second session gives the same words, a value
within tolerance of an independent float64 formula) hold for any finite
inputs of the drawn range, so the NumPy PCG64 draws are replaced by a fixed
integer hash mapped into the same range; nothing here reads a stream.
"""
from std.math import isfinite
from std.memory import bitcast
from autograd.ops_api import OpsSession
from autograd.rankn import ShapeN
from sequence.exec_trait import Exec
from checks.native_tests.runner import require


def _mix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def unit_value(seed: Int, i: Int) -> Float64:
    """A fixed value in [0, 1) for (seed, i): 24 hashed bits, exact in float32."""
    var h = _mix(UInt64(seed) * UInt64(0x100000001B3) + UInt64(i))
    return Float64(Int(h >> 40)) / 16777216.0


def uniform(count: Int, seed: Int, low: Float64, high: Float64) -> List[Float32]:
    """NumPy's `default_rng(seed).uniform(low, high, count)` stand-in."""
    var out = List[Float32](length=count, fill=0)
    for i in range(count):
        out[i] = Float32(low + (high - low) * unit_value(seed, i))
    return out^


def normal(count: Int, seed: Int) -> List[Float32]:
    """A standard-normal stand-in: the sum of four uniforms, centred and
    scaled to unit variance (finite, symmetric, about [-3.5, 3.5])."""
    var out = List[Float32](length=count, fill=0)
    for i in range(count):
        var t = Float64(0)
        for k in range(4):
            t += unit_value(seed, 4 * i + k)
        out[i] = Float32((t - 2.0) * 1.7320508075688772)
    return out^


def wide(values: List[Float32]) -> List[Float64]:
    var out = List[Float64](length=len(values), fill=0)
    for i in range(len(values)):
        out[i] = Float64(values[i])
    return out^


def allclose(actual: List[Float32], expected: List[Float64], rtol: Float64, atol: Float64, what: String) raises:
    """NumPy's assert_allclose rule |a - e| <= atol + rtol |e| (a NaN fails)."""
    require(len(actual) == len(expected), what + ": extent " + String(len(actual)) + " vs " + String(len(expected)))
    for i in range(len(actual)):
        var a = Float64(actual[i])
        if not (abs(a - expected[i]) <= atol + rtol * abs(expected[i])):
            raise Error(what + ": at " + String(i) + " got " + String(a) + ", expected " + String(expected[i]))


def exact(actual: List[Float32], expected: List[Float32], what: String) raises:
    """assert_array_equal: equal values (+0.0 equals -0.0, NaN equals NaN)."""
    require(len(actual) == len(expected), what + ": extent " + String(len(actual)) + " vs " + String(len(expected)))
    for i in range(len(actual)):
        var a = actual[i]
        var e = expected[i]
        var both_nan = a != a and e != e
        if not (a == e or both_nan):
            raise Error(what + ": at " + String(i) + " got " + String(a) + ", expected " + String(e))


def all_finite(values: List[Float32], what: String) raises:
    for i in range(len(values)):
        require(isfinite(values[i]), what + ": non-finite at " + String(i))


def dims2(a: Int, b: Int) -> List[Int]:
    var d: List[Int] = [a, b]
    return d^


def dims3(a: Int, b: Int, c: Int) -> List[Int]:
    var d: List[Int] = [a, b, c]
    return d^


def dims4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var out: List[Int] = [a, b, c, d]
    return out^


def same_shape[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int, dims: List[Int], what: String) raises:
    require(s.shape_nd(h).same(ShapeN.of(dims)), what + ": shape")


def opi[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], code: Int, hs: List[Int], ints: List[Int], fs: List[Float32]
) raises -> Int:
    """One session operation with its handles, integer and float words, as
    the public door hands them over (providers/python/autograd_ops_frontend_pyapi_door.mojo)."""
    return s.op(code, hs, ints, fs)
