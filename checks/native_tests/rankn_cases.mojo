# SPDX-License-Identifier: Apache-2.0
"""Session helpers the rank-N checks share (rankn_math, rankn_graph, rankn_conv): upload, binary op, sum,
download, gradient and the Float64 closeness check. Not a test file: no main(), no test_*."""
from autograd.ops_api import OpsSession, OPS_SUM_N
from autograd.rankn import ShapeN
from core.cast_in import CastIn, CAST_KIND_FLOAT
from core.hostptr import f32_ptr
from sequence.exec_trait import Exec


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _upload[E: Exec & Movable & Deinitable](mut s: OpsSession[E], values: List[Float32], dims: List[Int], grad: Bool = False) raises -> Int:
    var cols = dims[len(dims) - 1]
    var source = CastIn(Int(values.unsafe_ptr()), CAST_KIND_FLOAT, 4, len(values) // cols, cols, cols * 4, 4)
    var h = s.upload_nd(source, ShapeN.of(dims), grad)
    _ = values
    return h


def _binary[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, a: Int, b: Int) raises -> Int:
    var hs: List[Int] = [a, b]
    return s.run_op(code, hs, List[Int](), List[Float32]())


def _gradient[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int, count: Int) raises -> List[Float32]:
    var values = List[Float32](length=count, fill=0)
    s.gradient_into(h, f32_ptr(Int(values.unsafe_ptr())))
    return values^


def _close(actual: List[Float32], expected: List[Float64], rtol: Float64, atol: Float64, label: String) raises:
    require(len(actual) == len(expected), label + " element count")
    for i in range(len(expected)):
        require(abs(Float64(actual[i]) - expected[i]) <= atol + rtol * abs(expected[i]), label + " element " + String(i))


def _sum[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int) raises -> Int:
    var hs: List[Int] = [h]
    var ints: List[Int] = [-1]
    return s.run_op(OPS_SUM_N, hs, ints, List[Float32]())


def _download[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int, count: Int) raises -> List[Float32]:
    var values = List[Float32](length=count, fill=0)
    s.download(h, f32_ptr(Int(values.unsafe_ptr())))
    return values^
