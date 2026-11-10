# SPDX-License-Identifier: Apache-2.0
"""Native convolution/pool/finite-difference checks with scalar Float64 oracles.

Preserves test_ops_rankn's shapes, strides, padding, derivative probes and
tolerances. Native finite fixtures replace NumPy RNG inputs. Expected
cross-correlation, pooling and perturbed losses call no product math helper.
"""
from autograd.ops_api import OpsSession, OPS_ADD, OPS_MUL, OPS_CONV2D, OPS_MAXPOOL2D, OPS_AVGPOOL2D, OPS_MEAN_N
from autograd.rankn import ShapeN
from core.host_parallel import host_parallelize
from sequence.exec_trait import Exec
from std.memory import bitcast
from checks.native_tests.rankn_math_cases import require, _upload, _binary, _gradient, _close
from checks.native_tests.rankn_graph_cases import _sum, _download

comptime RANKN_CONV_FIXTURES = 2


def _to_f64(values: List[Float32]) -> List[Float64]:
    var result = List[Float64](length=len(values), fill=0)
    var count = len(values)
    var vp = values.unsafe_ptr()
    var rp = result.unsafe_ptr()
    def task(unit: Int) {imm vp, imm rp, imm count}:
        var start = unit * 256
        for i in range(start, min(start + 256, count)):
            rp.unsafe_store(i, Float64(vp.unsafe_load(i)))
    host_parallelize(task, (count + 255) // 256)
    _ = values
    return result^


def _conv_reference(x: List[Float64], w: List[Float64], bias: List[Float64], batches: Int, channels: Int, height: Int, width: Int, outputs: Int, kernel_rows: Int, kernel_cols: Int, stride: Int, pad: Int) -> List[Float64]:
    var oh = (height + 2 * pad - kernel_rows) // stride + 1
    var ow = (width + 2 * pad - kernel_cols) // stride + 1
    var count = batches * outputs * oh * ow
    var result = List[Float64](length=count, fill=0)
    var xp = x.unsafe_ptr()
    var wp = w.unsafe_ptr()
    var bp = bias.unsafe_ptr()
    var rp = result.unsafe_ptr()
    def cell_task(cell: Int) {imm xp, imm wp, imm bp, imm rp, imm channels, imm height, imm width, imm outputs, imm kernel_rows, imm kernel_cols, imm stride, imm pad, imm oh, imm ow}:
        var col = cell % ow
        var row = (cell // ow) % oh
        var output = (cell // (oh * ow)) % outputs
        var batch = cell // (outputs * oh * ow)
        var value = Float64(0)
        for channel in range(channels):
            for ky in range(kernel_rows):
                var iy = row * stride + ky - pad
                for kx in range(kernel_cols):
                    var ix = col * stride + kx - pad
                    if iy >= 0 and iy < height and ix >= 0 and ix < width:
                        var xi = ((batch * channels + channel) * height + iy) * width + ix
                        var wi = ((output * channels + channel) * kernel_rows + ky) * kernel_cols + kx
                        value += xp.unsafe_load(xi) * wp.unsafe_load(wi)
        rp.unsafe_store(cell, value + bp.unsafe_load(output))
    host_parallelize(cell_task, count)
    _ = x
    _ = w
    _ = bias
    return result^


def _pool_reference(x: List[Float64], batches: Int, channels: Int, height: Int, width: Int, kernel: Int, stride: Int, maximum: Bool) -> List[Float64]:
    var oh = (height - kernel) // stride + 1
    var ow = (width - kernel) // stride + 1
    var count = batches * channels * oh * ow
    var result = List[Float64](length=count, fill=0)
    var xp = x.unsafe_ptr()
    var rp = result.unsafe_ptr()
    def cell_task(cell: Int) {imm xp, imm rp, imm height, imm width, imm kernel, imm stride, imm maximum, imm oh, imm ow}:
        var col = cell % ow
        var row = (cell // ow) % oh
        var batch_channel = cell // (oh * ow)
        var value = Float64(-1e300) if maximum else Float64(0)
        for ky in range(kernel):
            for kx in range(kernel):
                var xi = (batch_channel * height + row * stride + ky) * width + col * stride + kx
                if maximum:
                    value = max(value, xp.unsafe_load(xi))
                else:
                    value += xp.unsafe_load(xi)
        if not maximum:
            value /= Float64(kernel * kernel)
        rp.unsafe_store(cell, value)
    host_parallelize(cell_task, count)
    _ = x
    return result^


def _mean[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int) raises -> Int:
    var hs: List[Int] = [h]
    var ints: List[Int] = [-1]
    return s.run_op(OPS_MEAN_N, hs, ints, List[Float32]())


def _conv_pool[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var xv = List[Float32](length=594, fill=0)
    var wv = List[Float32](length=108, fill=0)
    var bv: List[Float32] = [-0.375, 0.625, -0.125, 0.875]
    for i in range(594):
        xv[i] = Float32((i * 11) % 37 - 18) / 8.0
    for i in range(108):
        wv[i] = Float32((i * 7) % 23 - 11) / 32.0
    var xd: List[Int] = [2, 3, 9, 11]
    var wd: List[Int] = [4, 3, 3, 3]
    var bd: List[Int] = [1, 4]
    var x = _upload(s, xv, xd, True)
    var w = _upload(s, wv, wd, True)
    var b = _upload(s, bv, bd, True)
    var hs: List[Int] = [x, w, b]
    var ints: List[Int] = [1, 1]
    var y1 = s.run_op(OPS_CONV2D, hs, ints, List[Float32]())
    hs = [x, w]
    ints = [2, 0]
    var y2 = s.run_op(OPS_CONV2D, hs, ints, List[Float32]())
    hs = [y1]
    ints = [2, 2]
    var mp = s.run_op(OPS_MAXPOOL2D, hs, ints, List[Float32]())
    hs = [x]
    ints = [3, 2]
    var ap = s.run_op(OPS_AVGPOOL2D, hs, ints, List[Float32]())
    var y1d: List[Int] = [2, 4, 9, 11]
    var y2d: List[Int] = [2, 4, 4, 5]
    var apd: List[Int] = [2, 3, 4, 5]
    require(s.shape_nd(y1).same(ShapeN.of(y1d)), "padded convolution shape")
    require(s.shape_nd(y2).same(ShapeN.of(y2d)), "strided convolution shape")
    require(s.shape_nd(mp).same(ShapeN.of(y2d)), "max pool shape")
    require(s.shape_nd(ap).same(ShapeN.of(apd)), "average pool shape")
    var squared = _binary(s, OPS_MUL, mp, mp)
    var m0 = _mean(s, squared)
    var m1 = _mean(s, ap)
    var m2 = _mean(s, y2)
    var partial = _binary(s, OPS_ADD, m0, m1)
    var loss = _binary(s, OPS_ADD, partial, m2)
    s.backward(loss, False)
    var actual_y1 = _download(s, y1, 792)
    var actual_y2 = _download(s, y2, 160)
    var actual_mp = _download(s, mp, 160)
    var actual_ap = _download(s, ap, 120)
    var gx = _gradient(s, x, 594)
    var gw = _gradient(s, w, 108)
    var gb = _gradient(s, b, 4)
    var x64 = _to_f64(xv)
    var w64 = _to_f64(wv)
    var b64 = _to_f64(bv)
    var zeros = List[Float64](length=4, fill=0)
    var expected_y1 = _conv_reference(x64, w64, b64, 2, 3, 9, 11, 4, 3, 3, 1, 1)
    var expected_y2 = _conv_reference(x64, w64, zeros, 2, 3, 9, 11, 4, 3, 3, 2, 0)
    _close(actual_y1, expected_y1, 1e-4, 1e-4, "padded convolution")
    _close(actual_y2, expected_y2, 1e-4, 1e-4, "strided convolution")
    # As in the old test, pool the actual y1 after independently checking its
    # convolution. This isolates max-pool selection from convolution rounding.
    var actual_y1_64 = _to_f64(actual_y1)
    var expected_mp = _pool_reference(actual_y1_64, 2, 4, 9, 11, 2, 2, True)
    var expected_ap = _pool_reference(x64, 2, 3, 9, 11, 3, 2, False)
    for i in range(160):
        require(actual_mp[i] == Float32(expected_mp[i]), "max-pool selected value")
    _close(actual_ap, expected_ap, 1e-5, 1e-6, "average pool")
    # This graph's old gradient assertions were shape/finiteness only. The
    # separate fixture below supplies three independent numerical probes.
    for v in gx:
        require((bitcast[DType.uint32](v) & 0x7F800000) != 0x7F800000, "convolution input gradient finite")
    for v in gw:
        require((bitcast[DType.uint32](v) & 0x7F800000) != 0x7F800000, "convolution weight gradient finite")
    for v in gb:
        require((bitcast[DType.uint32](v) & 0x7F800000) != 0x7F800000, "convolution bias gradient finite")


def _finite_difference[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var xv = List[Float32](length=60, fill=0)
    var wv = List[Float32](length=54, fill=0)
    var gv = List[Float32](length=90, fill=0)
    var g64 = List[Float64](length=90, fill=0)
    for i in range(60):
        xv[i] = Float32((i * 7) % 23 - 11) / 8.0
    for i in range(54):
        wv[i] = Float32((i * 11) % 29 - 14) / 16.0
    for i in range(90):
        g64[i] = Float64((i * 13) % 31 - 15) / 7.0
        gv[i] = Float32(g64[i])
    var xd: List[Int] = [1, 2, 5, 6]
    var wd: List[Int] = [3, 2, 3, 3]
    var gd: List[Int] = [1, 3, 5, 6]
    var x = _upload(s, xv, xd)
    var w = _upload(s, wv, wd, True)
    var g = _upload(s, gv, gd)
    var hs: List[Int] = [x, w]
    var ints: List[Int] = [1, 1]
    var y = s.run_op(OPS_CONV2D, hs, ints, List[Float32]())
    require(s.shape_nd(y).same(ShapeN.of(gd)), "finite-difference convolution shape")
    var weighted = _binary(s, OPS_MUL, y, g)
    var loss = _sum(s, weighted)
    s.backward(loss, False)
    var actual = _gradient(s, w, 54)
    var x64 = _to_f64(xv)
    var w64 = _to_f64(wv)
    # Original (OC,IC,KH,KW) probes: (0,0,0,0), (1,1,2,1), (2,0,1,2).
    var indices: List[Int] = [0, 34, 41]
    var losses = List[Float64](length=6, fill=0)
    var xp = x64.unsafe_ptr()
    var wp = w64.unsafe_ptr()
    var gp = g64.unsafe_ptr()
    var ip = indices.unsafe_ptr()
    var lp = losses.unsafe_ptr()
    def difference_task(unit: Int) {imm xp, imm wp, imm gp, imm ip, imm lp}:
        var local_x = List[Float64](length=60, fill=0)
        var local_w = List[Float64](length=54, fill=0)
        for i in range(60):
            local_x[i] = xp.unsafe_load(i)
        for i in range(54):
            local_w[i] = wp.unsafe_load(i)
        var delta = Float64(1e-3) if unit % 2 == 0 else Float64(-1e-3)
        local_w[ip.unsafe_load(unit // 2)] += delta
        var zeros = List[Float64](length=3, fill=0)
        var prediction = _conv_reference(local_x, local_w, zeros, 1, 2, 5, 6, 3, 3, 3, 1, 1)
        var value = Float64(0)
        for i in range(90):
            value += prediction[i] * gp.unsafe_load(i)
        lp.unsafe_store(unit, value)
    host_parallelize(difference_task, 6)
    for probe in range(3):
        var expected = (losses[2 * probe] - losses[2 * probe + 1]) / (2.0 * 1e-3)
        require(abs(Float64(actual[indices[probe]]) - expected) <= 1e-2 * max(Float64(1), abs(expected)), "central-difference convolution gradient probe " + String(probe))
    _ = x64
    _ = w64
    _ = g64
    _ = indices


def run_rankn_conv_fixture[E: Exec & Movable & Deinitable](mut s: OpsSession[E], fixture_id: Int) raises:
    require(fixture_id >= 0 and fixture_id < RANKN_CONV_FIXTURES, "unknown rank-N convolution fixture")
    if fixture_id == 0:
        _conv_pool(s)
    else:
        _finite_difference(s)
