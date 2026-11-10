# SPDX-License-Identifier: Apache-2.0
"""Native attention/layout graphs with independent coordinate/Float64 oracles.

Shapes, operations and gradient assertions migrate from test_ops_rankn.
Deterministic finite fixtures replace NumPy RNG preparation. No product
attention, indexing or gradient helper computes expected values.
"""
from std.math import exp
from std.memory import bitcast
from autograd.ops_api import (
    OpsSession, OPS_ADD, OPS_MUL, OPS_MATMUL, OPS_PERMUTE, OPS_SCALE,
    OPS_SOFTMAX_N, OPS_MEAN_N, OPS_SUM_N, OPS_CONCAT, OPS_NARROW, OPS_TAKE,
)
from autograd.rankn import ShapeN
from core.cast_in import CastIn, CAST_KIND_SIGNED
from core.hostptr import f32_ptr
from core.host_parallel import host_parallelize
from sequence.exec_trait import Exec
from checks.native_tests.rankn_math_cases import require, _upload, _binary, _gradient, _close

comptime RANKN_GRAPH_FIXTURES = 2


def _attention_oracle(q: List[Float32], k: List[Float32], w: List[Float32]) -> Tuple[List[Float64], List[Float64], List[Float64]]:
    var scores = List[Float64](length=210, fill=0)
    var latent = List[Float64](length=120, fill=0)
    var result = List[Float64](length=180, fill=0)
    var grad_w = List[Float64](length=24, fill=0)
    var qp = q.unsafe_ptr()
    var kp = k.unsafe_ptr()
    var wp = w.unsafe_ptr()
    var sp = scores.unsafe_ptr()
    var lp = latent.unsafe_ptr()
    var rp = result.unsafe_ptr()
    def query_task(query: Int) {imm qp, imm kp, imm wp, imm sp, imm lp, imm rp}:
        var batch_head = query // 5
        var probabilities = List[Float64](length=7, fill=0)
        var largest = Float64(-1e300)
        for key in range(7):
            var dot = Float64(0)
            for feature in range(4):
                dot += Float64(qp.unsafe_load(query * 4 + feature)) * Float64(kp.unsafe_load((batch_head * 7 + key) * 4 + feature))
            sp.unsafe_store(query * 7 + key, dot)
            probabilities[key] = 0.5 * dot
            largest = max(largest, probabilities[key])
        var total = Float64(0)
        for key in range(7):
            probabilities[key] = exp(probabilities[key] - largest)
            total += probabilities[key]
        for key in range(7):
            probabilities[key] /= total
        for feature in range(4):
            var value = Float64(0)
            for key in range(7):
                value += probabilities[key] * Float64(kp.unsafe_load((batch_head * 7 + key) * 4 + feature))
            lp.unsafe_store(query * 4 + feature, value)
        for output in range(6):
            var value = Float64(0)
            for feature in range(4):
                value += lp.unsafe_load(query * 4 + feature) * Float64(wp.unsafe_load(feature * 6 + output))
            rp.unsafe_store(query * 6 + output, value)
    host_parallelize(query_task, 30)
    var gp = grad_w.unsafe_ptr()
    def weight_task(cell: Int) {imm lp, imm gp}:
        var feature = cell // 6
        var value = Float64(0)
        for query in range(30):
            value += lp.unsafe_load(query * 4 + feature) / 180.0
        gp.unsafe_store(cell, value)
    host_parallelize(weight_task, 24)
    _ = q
    _ = k
    _ = w
    _ = latent
    return (scores^, result^, grad_w^)


def _attention[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var qv = List[Float32](length=120, fill=0)
    var kv = List[Float32](length=168, fill=0)
    var wv = List[Float32](length=24, fill=0)
    for i in range(120):
        qv[i] = Float32((i * 7) % 23 - 11) / 8.0
    for i in range(168):
        kv[i] = Float32((i * 11) % 29 - 14) / 8.0
    for i in range(24):
        wv[i] = Float32((i * 5) % 17 - 8) / 8.0
    var qdims: List[Int] = [2, 3, 5, 4]
    var kdims: List[Int] = [2, 3, 7, 4]
    var wdims: List[Int] = [4, 6]
    var q = _upload(s, qv, qdims, True)
    var k = _upload(s, kv, kdims, True)
    var w = _upload(s, wv, wdims, True)
    var hs: List[Int] = [k]
    var ints: List[Int] = [1, -2, -1]
    var kt = s.run_op(OPS_PERMUTE, hs, ints, List[Float32]())
    var scores = _binary(s, OPS_MATMUL, q, kt)
    hs = [scores]
    var fs: List[Float32] = [0.5]
    var scaled = s.run_op(OPS_SCALE, hs, List[Int](), fs)
    hs = [scaled]
    ints = [-1]
    var att = s.run_op(OPS_SOFTMAX_N, hs, ints, List[Float32]())
    var latent = _binary(s, OPS_MATMUL, att, k)
    var result = _binary(s, OPS_MATMUL, latent, w)
    var score_dims: List[Int] = [2, 3, 5, 7]
    var result_dims: List[Int] = [2, 3, 5, 6]
    require(s.shape_nd(scores).same(ShapeN.of(score_dims)), "attention score shape")
    require(s.shape_nd(result).same(ShapeN.of(result_dims)), "attention output shape")
    hs = [result]
    var loss = s.run_op(OPS_MEAN_N, hs, ints, List[Float32]())
    s.backward(loss, False)
    var actual_scores = List[Float32](length=210, fill=0)
    var actual_result = List[Float32](length=180, fill=0)
    s.download(scores, f32_ptr(Int(actual_scores.unsafe_ptr())))
    s.download(result, f32_ptr(Int(actual_result.unsafe_ptr())))
    var gq = _gradient(s, q, 120)
    var gk = _gradient(s, k, 168)
    var gw = _gradient(s, w, 24)
    var expected = _attention_oracle(qv, kv, wv)
    _close(actual_scores, expected[0], 1e-4, 1e-4, "attention scores")
    _close(actual_result, expected[1], 1e-4, 1e-4, "attention output")
    _close(gw, expected[2], 1e-4, 1e-6, "attention weight gradient")
    # The migrated test checked Q/K gradients for shape and finiteness;
    # only W had an analytic gradient oracle. Do not claim more coverage.
    for v in gq:
        require((bitcast[DType.uint32](v) & 0x7F800000) != 0x7F800000, "attention Q gradient finite")
    for v in gk:
        require((bitcast[DType.uint32](v) & 0x7F800000) != 0x7F800000, "attention K gradient finite")


def _sum[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int) raises -> Int:
    var hs: List[Int] = [h]
    var ints: List[Int] = [-1]
    return s.run_op(OPS_SUM_N, hs, ints, List[Float32]())


def _download[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int, count: Int) raises -> List[Float32]:
    var values = List[Float32](length=count, fill=0)
    s.download(h, f32_ptr(Int(values.unsafe_ptr())))
    return values^


def _layout[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var p0v = List[Float32](length=30, fill=0)
    var p1v = List[Float32](length=15, fill=0)
    var p2v = List[Float32](length=60, fill=0)
    var tv = List[Float32](length=105, fill=0)
    var weights = List[Float32](length=60, fill=0)
    for i in range(30):
        p0v[i] = Float32((i * 3) % 17 - 8) / 4.0
    for i in range(15):
        p1v[i] = Float32((i * 5) % 13 - 6) / 8.0
    for i in range(60):
        p2v[i] = Float32((i * 7) % 19 - 9) / 4.0
        weights[i] = Float32(i)
    for i in range(105):
        tv[i] = Float32((i * 11) % 31 - 15) / 8.0
    var d0: List[Int] = [3, 2, 5]
    var d1: List[Int] = [3, 1, 5]
    var d2: List[Int] = [3, 4, 5]
    var dt: List[Int] = [3, 7, 5]
    var p0 = _upload(s, p0v, d0, True)
    var p1 = _upload(s, p1v, d1, True)
    var p2 = _upload(s, p2v, d2, True)
    var t = _upload(s, tv, dt, True)
    var wh = _upload(s, weights, d2)
    var hs: List[Int] = [p0, p1, p2]
    var ints: List[Int] = [1]
    var cat = s.run_op(OPS_CONCAT, hs, ints, List[Float32]())
    require(s.shape_nd(cat).same(ShapeN.of(dt)), "concat shape")
    var pieces = List[Int]()
    hs = [cat]
    for piece in range(3):
        ints = [1, piece, 0, 2, 1, 4]
        pieces.append(s.run_op(OPS_NARROW, hs, ints, List[Float32]()))
    require(s.shape_nd(pieces[0]).same(ShapeN.of(d0)), "first split shape")
    require(s.shape_nd(pieces[1]).same(ShapeN.of(d1)), "second split shape")
    require(s.shape_nd(pieces[2]).same(ShapeN.of(d2)), "third split shape")
    hs = [t]
    ints = [0, 0, 1, 1]
    var full_piece = s.run_op(OPS_NARROW, hs, ints, List[Float32]())
    require(s.shape_nd(full_piece).same(ShapeN.of(dt)), "single-section split shape")
    # A native integer tensor handle, not a NumPy dtype-number descriptor.
    var ids: List[Int32] = [6, 0, 6, 3]
    var id_dims: List[Int] = [4]
    var id_source = CastIn(Int(ids.unsafe_ptr()), CAST_KIND_SIGNED, 4, 4, 1, 4, 4)
    var id_handle = s.upload_ids(id_source, ShapeN.of(id_dims))
    hs = [t, id_handle]
    ints = [1]
    var picked = s.run_op(OPS_TAKE, hs, ints, List[Float32]())
    require(s.shape_nd(picked).same(ShapeN.of(d2)), "take shape")
    var squared = _binary(s, OPS_MUL, pieces[0], pieces[0])
    var weighted = _binary(s, OPS_MUL, picked, wh)
    var loss_a = _sum(s, squared)
    var loss_p = _sum(s, weighted)
    hs = [pieces[2]]
    var fs: List[Float32] = [3.0]
    var scaled = s.run_op(OPS_SCALE, hs, List[Int](), fs)
    var loss_c = _sum(s, scaled)
    var partial = _binary(s, OPS_ADD, loss_a, loss_p)
    var loss = _binary(s, OPS_ADD, partial, loss_c)
    s.backward(loss, False)
    var actual_cat = _download(s, cat, 105)
    var actual_a = _download(s, pieces[0], 30)
    var actual_b = _download(s, pieces[1], 15)
    var actual_c = _download(s, pieces[2], 60)
    var actual_picked = _download(s, picked, 60)
    var actual_full = _download(s, full_piece, 105)
    var g0 = _gradient(s, p0, 30)
    var g1 = _gradient(s, p1, 15)
    var g2 = _gradient(s, p2, 60)
    var gt = _gradient(s, t, 105)
    for i in range(105):
        var row = i // 35
        var middle = (i // 5) % 7
        var channel = i % 5
        var expected = Float32(0)
        if middle < 2:
            expected = p0v[(row * 2 + middle) * 5 + channel]
        elif middle == 2:
            expected = p1v[row * 5 + channel]
        else:
            expected = p2v[(row * 4 + middle - 3) * 5 + channel]
        require(actual_cat[i] == expected, "concat coordinate")
        require(actual_full[i] == tv[i], "single-section split value")
    for i in range(30):
        require(actual_a[i] == p0v[i] and g0[i] == 2.0 * p0v[i], "first split value/gradient")
    for i in range(15):
        require(actual_b[i] == p1v[i] and g1[i] == 0.0, "unused split value/gradient")
    var expected_gt = List[Float32](length=105, fill=0)
    for i in range(60):
        require(actual_c[i] == p2v[i] and g2[i] == 3.0, "third split value/gradient")
        var row = i // 20
        var index = (i // 5) % 4
        var channel = i % 5
        var src = (row * 7 + Int(ids[index])) * 5 + channel
        require(actual_picked[i] == tv[src], "take coordinate")
        expected_gt[src] += Float32(i)
    for i in range(105):
        require(gt[i] == expected_gt[i], "duplicate-index take gradient")
    _ = ids


def run_rankn_graph_fixture[E: Exec & Movable & Deinitable](mut s: OpsSession[E], fixture_id: Int) raises:
    require(fixture_id >= 0 and fixture_id < RANKN_GRAPH_FIXTURES, "unknown rank-N graph fixture")
    if fixture_id == 0:
        _attention(s)
    else:
        _layout(s)
