# SPDX-License-Identifier: Apache-2.0
"""Typed session views, axis reductions and first-maximum gradient checks.

Independent coordinate indexing and Float64 accumulation replace NumPy
oracles in test_ops_rankn. Test data and expectations never call Python or
the product's indexing/reduction helpers. No digest or repeat-run checks.
"""
from std.memory import bitcast
from autograd.ops_api import (
    OpsSession, OPS_RESHAPE_N, OPS_PERMUTE, OPS_SUM_N, OPS_MEAN_N, OPS_MAX,
)
from autograd.rankn import ShapeN
from core.cast_in import CastIn, CAST_KIND_FLOAT
from core.hostptr import f32_ptr
from sequence.exec_trait import Exec

comptime RANKN_VIEW_FIXTURES = 17


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def _upload[E: Exec & Movable & Deinitable](mut s: OpsSession[E], values: List[Float32], dims: List[Int], grad: Bool = False) raises -> Int:
    var shape = ShapeN.of(dims)
    var cols = dims[len(dims) - 1]
    var source = CastIn(Int(values.unsafe_ptr()), CAST_KIND_FLOAT, 4, len(values) // cols, cols, cols * 4, 4)
    var h = s.upload_nd(source, shape, grad)
    _ = values
    return h


def _view[E: Exec & Movable & Deinitable](mut s: OpsSession[E], fixture_id: Int) raises:
    var values = List[Float32](length=105, fill=0)
    for i in range(105):
        values[i] = Float32(i) * 0.25 - 4.0
    var dims: List[Int] = [3, 5, 7]
    var x = _upload(s, values, dims)
    var hs: List[Int] = [x]
    var ints: List[Int] = [0, -1, 7]
    var code = OPS_RESHAPE_N
    var expected_dims: List[Int] = [15, 7]
    if fixture_id == 1:
        ints = [1, 1, -1]
        expected_dims = [3, 35]
    elif fixture_id == 2 or fixture_id == 3:
        ints = [3, 1]
        expected_dims = [3, 1, 5, 7]
        if fixture_id == 3:
            var u = s.run_op(code, hs, ints, List[Float32]())
            hs = [u]
            ints = [2]
            expected_dims = [3, 5, 7]
    elif fixture_id == 4 or fixture_id == 7:
        code = OPS_PERMUTE
        ints = [0, 2, 0, 1]
        expected_dims = [7, 3, 5]
        if fixture_id == 7:
            var p = s.run_op(code, hs, ints, List[Float32]())
            hs = [p]
            code = OPS_RESHAPE_N
            ints = [0, 21, 5]
            expected_dims = [21, 5]
    elif fixture_id == 5:
        code = OPS_PERMUTE
        ints = [1, 0, 2]
        expected_dims = [7, 5, 3]
    elif fixture_id == 6:
        code = OPS_PERMUTE
        ints = [1, -2, -1]
        expected_dims = [3, 7, 5]
    var result = s.run_op(code, hs, ints, List[Float32]())
    require(s.shape_nd(result).same(ShapeN.of(expected_dims)), "view shape")
    var actual = List[Float32](length=105, fill=0)
    s.download(result, f32_ptr(Int(actual.unsafe_ptr())))
    for i in range(105):
        var src = i
        if fixture_id == 4 or fixture_id == 7:
            src = ((i // 5) % 3) * 35 + (i % 5) * 7 + i // 15
        elif fixture_id == 5:
            src = (i % 3) * 35 + ((i // 3) % 5) * 7 + i // 15
        elif fixture_id == 6:
            src = (i // 35) * 35 + (i % 5) * 7 + (i // 5) % 7
        var expected = Float32(src) * 0.25 - 4.0
        require(bitcast[DType.uint32](actual[i]) == bitcast[DType.uint32](expected), "view coordinate " + String(i))


def _reduction_input(index: Int) -> Float32:
    var row = index // 30
    var col = (index // 5) % 6
    var inner = index % 5
    if row == 1 and col == 2:
        return 3.0
    if row == 2 and inner == 3:
        return 7.0
    # Native deterministic finite fixture in place of NumPy's seeded normal
    # generator. Same shape, signed values and planted tied maxima.
    var h = UInt32(index + 3) * UInt32(1664525) + UInt32(1013904223)
    h ^= h >> 11
    var word = ((UInt32(125) + ((h >> 24) & 3)) << 23) | (h & 0x7FFFFF)
    # Take the sign from a different mixed bit, not an arithmetic conversion.
    word = (word & 0x7FFFFFFF) | ((h & 0x80) << 24)
    return bitcast[DType.float32](word)


def _reduction[E: Exec & Movable & Deinitable](mut s: OpsSession[E], fixture_id: Int) raises:
    var values = List[Float32](length=120, fill=0)
    for i in range(120):
        values[i] = _reduction_input(i)
    var dims: List[Int] = [4, 6, 5]
    var hs: List[Int] = [_upload(s, values, dims)]
    var ints: List[Int] = [-1, 1]
    var expected_dims: List[Int] = [4, 5]
    var count = 20
    var code = OPS_SUM_N
    if fixture_id == 9:
        ints = [-1, 0, 2]
        expected_dims = [6]
        count = 6
    elif fixture_id == 10 or fixture_id == 14:
        ints = [1, 2 if fixture_id == 14 else -1]
        expected_dims = [4, 6, 1]
        count = 24
    elif fixture_id == 11:
        ints = [-1, 1, 2]
        expected_dims = [4]
        count = 4
    elif fixture_id == 12:
        ints = [-1]
        expected_dims = [1, 1]
        count = 1
    elif fixture_id == 15:
        ints = [-1, 0, 1]
        expected_dims = [5]
        count = 5
    if fixture_id == 11 or fixture_id == 12:
        code = OPS_MEAN_N
    elif fixture_id >= 13:
        code = OPS_MAX
    var expected = List[Float64](length=count, fill=-1e300 if fixture_id >= 13 else 0.0)
    # Independent scalar coordinate oracle; no product axis/shape helpers.
    for i in range(120):
        var row = i // 30
        var col = (i // 5) % 6
        var inner = i % 5
        var dst = row * 5 + inner
        if fixture_id == 9:
            dst = col
        elif fixture_id == 10 or fixture_id == 14:
            dst = row * 6 + col
        elif fixture_id == 11:
            dst = row
        elif fixture_id == 12:
            dst = 0
        elif fixture_id == 15:
            dst = inner
        if fixture_id >= 13:
            expected[dst] = max(expected[dst], Float64(values[i]))
        else:
            expected[dst] += Float64(values[i])
    if fixture_id == 11 or fixture_id == 12:
        for i in range(count):
            expected[i] /= 30.0 if fixture_id == 11 else 120.0
    var result = s.run_op(code, hs, ints, List[Float32]())
    require(s.shape_nd(result).same(ShapeN.of(expected_dims)), "reduction shape")
    var actual = List[Float32](length=count, fill=0)
    s.download(result, f32_ptr(Int(actual.unsafe_ptr())))
    for i in range(count):
        if fixture_id >= 13:
            require(bitcast[DType.uint32](actual[i]) == bitcast[DType.uint32](Float32(expected[i])), "maximum value")
        else:
            require(abs(Float64(actual[i]) - expected[i]) <= 1e-5 + 1e-5 * abs(expected[i]), "sum/mean value")


def _max_gradient[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var values: List[Float32] = [1.0, 4.0, 4.0, 2.0, bitcast[DType.float32](UInt32(0x7FC00000)), 0.0, 5.0, 5.0]
    var dims: List[Int] = [1, 2, 4]
    var x = _upload(s, values, dims, True)
    var hs: List[Int] = [x]
    var ints: List[Int] = [-1, 2]
    var result = s.run_op(OPS_MAX, hs, ints, List[Float32]())
    var result_dims: List[Int] = [1, 2]
    require(s.shape_nd(result).same(ShapeN.of(result_dims)), "maximum gradient result shape")
    hs = [result]
    ints = [-1]
    var total = s.run_op(OPS_SUM_N, hs, ints, List[Float32]())
    s.backward(total, False)
    var actual = List[Float32](length=2, fill=0)
    var grad = List[Float32](length=8, fill=0)
    s.download(result, f32_ptr(Int(actual.unsafe_ptr())))
    s.gradient_into(x, f32_ptr(Int(grad.unsafe_ptr())))
    var nan_word = bitcast[DType.uint32](actual[1])
    require(actual[0] == 4.0 and (nan_word & 0x7F800000) == 0x7F800000 and (nan_word & 0x7FFFFF) != 0, "NaN wins the second maximum")
    var expected: List[Float32] = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    for i in range(8):
        require(grad[i] == expected[i], "first-maximum gradient element " + String(i))


def run_rankn_view_fixture[E: Exec & Movable & Deinitable](mut s: OpsSession[E], fixture_id: Int) raises:
    require(fixture_id >= 0 and fixture_id < RANKN_VIEW_FIXTURES, "unknown rank-N fixture")
    if fixture_id < 8:
        _view(s, fixture_id)
    elif fixture_id < 16:
        _reduction(s, fixture_id)
    else:
        _max_gradient(s)
