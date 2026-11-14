# SPDX-License-Identifier: Apache-2.0
"""First-derivative ranks of the session's vjp, moved from
test_vjp_rank_native.py (2026-11-14).

  0, 1: value = arange(12) / 16 and v = arange(1, 13) / 8 in shapes
    (2, 3, 2) and (2, 1, 3, 2); y = scale(x, 2); vjp(y, [x], v) keeps x's
    full shape and equals v * 2 exactly.
  2: x (2, 3, 2) = arange / 16 plus a broadcast bias (1, 1, 2), permuted
    (1, 0, 2); v (3, 2, 2) = arange(1, 13) / 8. dx = v permuted back
    (exact), db = dx summed over the two leading axes (exact), each in its
    operand's shape.
The refusal of a direction of the right size and the wrong full shape, and
the session's recovery after it, stay in the Python test.
"""
from autograd.ops_api import OpsSession
from sequence.exec_trait import Exec
from checks.native_tests.ops_higher_cases import require, tensor, values, scale
from checks.native_tests.core_ops_common import exact, dims3, dims4, opi

comptime CORE_VJP_RANK_FIXTURES = 3

comptime _ADD = 2
comptime _PERMUTE = 34


def _arange(start: Int, count: Int, divisor: Float32) -> List[Float32]:
    var out = List[Float32](length=count, fill=0)
    for i in range(count):
        out[i] = Float32(start + i) / divisor
    return out^


def run_core_vjp_rank_fixture[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], mut again: OpsSession[E], fixture_id: Int
) raises:
    if fixture_id == 0 or fixture_id == 1:
        var shape = dims3(2, 3, 2) if fixture_id == 0 else dims4(2, 1, 3, 2)
        var seed = _arange(1, 12, 8.0)
        var x = tensor(s, _arange(0, 12, 16.0), shape, True)
        var y = scale(s, x, 2.0)
        var v = tensor(s, seed, shape)
        var hs: List[Int] = [x]
        var dx = s.vjp(y, hs, v, False)
        require(len(dx) == 1, "one derivative per input")
        var want = List[Float32](length=12, fill=0)
        for i in range(12):
            want[i] = seed[i] * Float32(2)
        exact(values(s, dx[0], shape), want, "vjp keeps the full rank")
        return
    require(fixture_id == 2, "unknown vjp rank fixture")
    var a = _arange(0, 12, 16.0)
    var bias: List[Float32] = [0.25, -0.125]
    var seed = _arange(1, 12, 8.0)
    var x = tensor(s, a, dims3(2, 3, 2), True)
    var b = tensor(s, bias, dims3(1, 1, 2), True)
    var pair: List[Int] = [x, b]
    var shifted = opi(s, _ADD, pair, List[Int](), List[Float32]())
    var one: List[Int] = [shifted]
    var axes: List[Int] = [0, 1, 0, 2]
    var y = opi(s, _PERMUTE, one, axes, List[Float32]())
    var v = tensor(s, seed, dims3(3, 2, 2))
    var d = s.vjp(y, pair, v, False)
    require(len(d) == 2, "one derivative per input")
    # dx[i, j, k] = seed[j, i, k] (seed is (3, 2, 2)); db[k] = sum over i, j
    var want_dx = List[Float32](length=12, fill=0)
    var want_db = List[Float32](length=2, fill=0)
    for i in range(2):
        for j in range(3):
            for k in range(2):
                want_dx[(i * 3 + j) * 2 + k] = seed[(j * 2 + i) * 2 + k]
    # NumPy's float32 sum over axes (0, 1) of six dyadic values: exact.
    for k in range(2):
        var t = Float32(0)
        for i in range(2):
            for j in range(3):
                t += want_dx[(i * 3 + j) * 2 + k]
        want_db[k] = t
    exact(values(s, d[0], dims3(2, 3, 2)), want_dx, "dx is the seed permuted back")
    exact(values(s, d[1], dims3(1, 1, 2)), want_db, "db sums the broadcast axes")
