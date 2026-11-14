# SPDX-License-Identifier: Apache-2.0
"""Dropout2d, the adaptive pools and the ResNet BasicBlock against float64
references of PyTorch's semantics, on the host column (lane
tmig-sequence-1, 2026-11-14; ported from `test_x_cnn_blocks.py`).

Unit 0: `test_dropout2d`: one draw per channel, a kept fraction in (0.6,
0.8) at p = 0.3, kept words scaled by 1 / 0.7, y = x * mask and the backward
g * mask (rtol 1e-6), a fresh mask on the next call, the same mask for the
same seed and call count, p = 1 drops everything, p = 0 keeps x. (Eval mode
returns a copy of x in the door without a native call; it is not a kernel
relation and is not repeated here.)
Unit 1: `test_adaptive_pools` (dividing sizes run the plain pool with kernel
= stride, as the door does). Units 2-7: `test_adaptive_non_dividing`
(average and max, three output sizes each). Units 8-9: `test_basic_block`
((4, 4, stride 1) and (3, 5, stride 2) with its 1x1 downsample): forward
within rtol 1e-3 / atol 1e-4 and the input gradient within 1e-3 of the
float64 reference; the block is composed of the conv, batch-norm, ReLU and
add entries as the door composes it (bindings/xcnn_conv_native_door.mojo).

The torch comparison of the adaptive reference stays in Python (ORACLE).
The units are independent and run across the native host worker pool.
"""
from std.math import sqrt
from core.host_parallel import host_parallelize
from x_cnn.host.entry_runs_host import publish_x_cnn_host_entries
from x_cnn.conv_run import (
    ConvGeom, conv_run_words, conv_run_out, conv_run_forward, conv_run_backward, conv_run_relu_forward,
    conv_run_relu_backward, conv_run_add,
)
from x_cnn.pool_run import (
    PoolSettings, pool_run_words, pool_run_out_words, pool_run_max_forward, pool_run_avg_forward,
    pool_run_avg_backward, pool_run_dropout, pool_run_dropout_words, pool_run_mul, pool_run_adaptive,
    pool_run_adaptive_words,
)
from checks.native_tests.cnn_cases import (
    require, cnn_entry, addr, addr_i32, zeros32, prod, normals, widen, check_close, check_equal, BnLayer, ref_bn,
)

comptime CNN_BLOCKS_UNITS = 10


# ---------------------------------------------------------------- Dropout2d

def _dropout_call(mut x: List[Float32], p: Float64, seed: Int, calls: Int, n: Int, c: Int, hw: Int,
                  mut mask: List[Float32]) raises -> List[Float32]:
    var y = zeros32(len(x))
    var words = pool_run_dropout_words(p, UInt64(seed), calls, n, c, hw)
    pool_run_dropout(cnn_entry(), addr(x), addr(y), addr(mask), words, p)
    return y^


def _dropout() raises:
    var n = 64
    var c = 8
    var hw = 9
    var x = normals(n * c * hw, 0, 1.0, 3.0)
    var mask = zeros32(len(x))
    var y = _dropout_call(x, 0.3, 7, 0, n, c, hw, mask)
    var kept = 0
    for plane in range(n * c):  # small-loop(fixture channels)
        for k in range(hw):  # small-loop(fixture positions)
            require(mask[plane * hw + k] == mask[plane * hw], "dropout2d: more than one draw in channel "
                    + String(plane))
        if mask[plane * hw] != 0:
            kept += 1
    var fraction = Float64(kept) / Float64(n * c)
    require(fraction > 0.6 and fraction < 0.8, "dropout2d: kept fraction " + String(fraction))
    var scale = 1.0 / 0.7
    var want_y = List[Float64](capacity=len(x))
    for i in range(len(x)):  # small-loop(fixture words)
        if mask[i] != 0:
            require(abs(Float64(mask[i]) - scale) <= 1e-6 * scale, "dropout2d: kept mask word " + String(mask[i]))
        want_y.append(Float64(x[i] * mask[i]))
    check_close(y, want_y, 1e-6, 0.0, "dropout2d y = x * mask")
    var g = normals(len(x), 1)
    var dx = zeros32(len(x))
    pool_run_mul(cnn_entry(), addr(g), addr(mask), addr(dx), len(x))
    var want_dx = List[Float64](capacity=len(x))
    for i in range(len(x)):  # small-loop(fixture words)
        want_dx.append(Float64(g[i] * mask[i]))
    check_close(dx, want_dx, 1e-6, 0.0, "dropout2d backward = g * mask")
    var mask2 = zeros32(len(x))
    var y2 = _dropout_call(x, 0.3, 7, 1, n, c, hw, mask2)
    var differs = False
    for i in range(len(x)):  # small-loop(fixture words)
        if y2[i] != y[i]:
            differs = True
    require(differs, "dropout2d: the second call drew the first call's mask")
    var mask3 = zeros32(len(x))
    var again = _dropout_call(x, 0.3, 7, 0, n, c, hw, mask3)
    check_equal(again, y, "dropout2d: the same seed and call")
    var mask4 = zeros32(len(x))
    var dropped = _dropout_call(x, 1.0, 0, 0, n, c, hw, mask4)
    check_equal(dropped, zeros32(len(x)), "dropout2d p = 1")
    var mask5 = zeros32(len(x))
    var kept_all = _dropout_call(x, 0.0, 0, 0, n, c, hw, mask5)
    check_equal(kept_all, x, "dropout2d p = 0")
    _ = g^


# ---------------------------------------------------------------- adaptive pools

def _adaptive_dividing() raises:
    var shape: List[Int] = [3, 4, 6, 8]
    var x = normals(prod(shape), 2)
    var entry = cnn_entry()
    # AdaptiveAvgPool2d(1): the plain pool with kernel = stride = (6, 8)
    var avg = pool_run_words(PoolSettings(6, 8, 6, 8, 0, 0, 1, 1, True, False, 0), shape)
    var y = zeros32(prod(pool_run_out_words(avg, shape)))
    pool_run_avg_forward(entry, addr(x), addr(y), avg)
    var means = List[Float64](length=12, fill=0.0)
    for plane in range(12):  # small-loop(fixture planes)
        for k in range(48):  # small-loop(fixture plane words)
            means[plane] += Float64(x[plane * 48 + k])
        means[plane] /= 48.0
    check_close(y, means, 1e-5, 1e-6, "AdaptiveAvgPool2d(1) out")
    var g = normals(12, 3)
    var dx = zeros32(len(x))
    pool_run_avg_backward(entry, addr(g), addr(dx), avg)
    var spread = List[Float64](length=len(x), fill=0.0)
    for plane in range(12):  # small-loop(fixture planes)
        for k in range(48):  # small-loop(fixture plane words)
            spread[plane * 48 + k] = Float64(g[plane]) / 48.0
    check_close(dx, spread, 1e-5, 1e-7, "AdaptiveAvgPool2d(1) dx")
    # AdaptiveMaxPool2d((2, 4)): the plain pool with kernel = stride = (3, 2)
    var mx = pool_run_words(PoolSettings(3, 2, 3, 2, 0, 0, 1, 1, True, False, 0), shape)
    var ym = zeros32(prod(pool_run_out_words(mx, shape)))
    var idx = List[Int32](length=len(ym), fill=0)
    pool_run_max_forward(entry, addr(x), addr(ym), addr_i32(idx), mx)
    var want = zeros32(12 * 8)
    for plane in range(12):  # small-loop(fixture planes)
        for i in range(2):  # small-loop(output rows)
            for j in range(4):  # small-loop(output columns)
                var best = x[plane * 48 + (3 * i) * 8 + 2 * j]
                for a in range(3):  # small-loop(window rows)
                    for b in range(2):  # small-loop(window columns)
                        best = max(best, x[plane * 48 + (3 * i + a) * 8 + 2 * j + b])
                want[(plane * 2 + i) * 4 + j] = best
    check_equal(ym, want, "AdaptiveMaxPool2d((2, 4)) out")
    _ = g^
    _ = idx^


def _adaptive(k: Int) raises:
    """`test_adaptive_non_dividing`: x (2, 3, 7, 5); k < 3 average, else max;
    output sizes (3, 2), (4, 4), (2, 5)."""
    var shape: List[Int] = [2, 3, 7, 5]
    var maxed = k >= 3
    var which = k % 3
    var o: List[Int] = [2, 5]
    if which == 0:
        o = [3, 2]
    elif which == 1:
        o = [4, 4]
    var x = normals(prod(shape), 8)
    var entry = cnn_entry()
    var nout = 6 * o[0] * o[1]
    var y = zeros32(nout)
    var idx = List[Int32](length=nout, fill=0)
    pool_run_adaptive(entry, addr(x), addr(y), addr_i32(idx), pool_run_adaptive_words(shape, o, maxed, False))
    var g = normals(nout, 9 + k)
    var dx = zeros32(len(x))
    pool_run_adaptive(entry, addr(g), addr(dx), addr_i32(idx), pool_run_adaptive_words(shape, o, maxed, True))
    var h = shape[2]
    var w = shape[3]
    var want_y = List[Float64](length=nout, fill=0.0)
    var want_dx = List[Float64](length=len(x), fill=0.0)
    for plane in range(6):  # small-loop(fixture planes)
        for i in range(o[0]):  # small-loop(output rows)
            var hs = (i * h) // o[0]
            var he = ((i + 1) * h + o[0] - 1) // o[0]
            for j in range(o[1]):  # small-loop(output columns)
                var ws = (j * w) // o[1]
                var we = ((j + 1) * w + o[1] - 1) // o[1]
                var yi = (plane * o[0] + i) * o[1] + j
                var area = (he - hs) * (we - ws)
                if maxed:
                    var best = -1
                    var best_value = 0.0
                    for r in range(hs, he):  # small-loop(window rows, row-major argmax)
                        for c in range(ws, we):  # small-loop(window columns)
                            var v = Float64(x[(plane * h + r) * w + c])
                            if best < 0 or v > best_value:
                                best = (plane * h + r) * w + c
                                best_value = v
                    want_y[yi] = best_value
                    want_dx[best] += Float64(g[yi])
                else:
                    var acc = 0.0
                    for r in range(hs, he):  # small-loop(window rows)
                        for c in range(ws, we):  # small-loop(window columns)
                            acc += Float64(x[(plane * h + r) * w + c])
                            want_dx[(plane * h + r) * w + c] += Float64(g[yi]) / Float64(area)
                    want_y[yi] = acc / Float64(area)
    var what = String("adaptive ") + (String("max") if maxed else String("avg")) + " case " + String(which)
    check_close(y, want_y, 1e-5, 1e-6, what + " out")
    check_close(dx, want_dx, 1e-5, 1e-6, what + " dx")
    _ = idx^
    _ = g^


# ---------------------------------------------------------------- BasicBlock

struct Conv64(Movable):
    var y: List[Float64]
    var oh: Int
    var ow: Int

    def __init__(out self, var y: List[Float64], oh: Int, ow: Int):
        self.y = y^
        self.oh = oh
        self.ow = ow


def _conv64(x: List[Float64], n: Int, c: Int, h: Int, w: Int, wt: List[Float32], oc: Int, k: Int, s: Int,
            p: Int) -> Conv64:
    """`ref_conv2d` (zero padding, no bias) in float64."""
    var oh = (h + 2 * p - k) // s + 1
    var ow = (w + 2 * p - k) // s + 1
    var y = List[Float64](length=n * oc * oh * ow, fill=0.0)
    for b in range(n):  # small-loop(reference images)
        for o in range(oc):  # small-loop(reference output channels)
            for i in range(oh):  # small-loop(reference output rows)
                for j in range(ow):  # small-loop(reference output columns)
                    var acc = 0.0
                    for ci in range(c):  # small-loop(reference input channels)
                        for a in range(k):  # small-loop(kernel rows)
                            var r = i * s - p + a
                            if r < 0 or r >= h:
                                continue
                            for bb in range(k):  # small-loop(kernel columns)
                                var q = j * s - p + bb
                                if q < 0 or q >= w:
                                    continue
                                acc += x[((b * c + ci) * h + r) * w + q] * Float64(wt[((o * c + ci) * k + a) * k + bb])
                    y[((b * oc + o) * oh + i) * ow + j] = acc
    return Conv64(y^, oh, ow)


def _conv64_dx(g: List[Float64], n: Int, c: Int, h: Int, w: Int, wt: List[Float32], oc: Int, k: Int, s: Int,
               p: Int, oh: Int, ow: Int) -> List[Float64]:
    """`ref_conv2d_backward(...)[0]`: the input gradient in float64."""
    var dx = List[Float64](length=n * c * h * w, fill=0.0)
    for b in range(n):  # small-loop(reference images)
        for o in range(oc):  # small-loop(reference output channels)
            for i in range(oh):  # small-loop(reference output rows)
                for j in range(ow):  # small-loop(reference output columns)
                    var gv = g[((b * oc + o) * oh + i) * ow + j]
                    for ci in range(c):  # small-loop(reference input channels)
                        for a in range(k):  # small-loop(kernel rows)
                            var r = i * s - p + a
                            if r < 0 or r >= h:
                                continue
                            for bb in range(k):  # small-loop(kernel columns)
                                var q = j * s - p + bb
                                if q < 0 or q >= w:
                                    continue
                                dx[((b * c + ci) * h + r) * w + q] += gv * Float64(wt[((o * c + ci) * k + a) * k + bb])
    return dx^


def _geom(oc: Int, k: Int, s: Int, p: Int) -> ConvGeom:
    return ConvGeom(1, k, k, s, s, p, p, 1, 1, oc, False)


def _conv_fwd(mut x: List[Float32], shape: List[Int], mut wt: List[Float32], oc: Int, k: Int, s: Int,
              p: Int) raises -> List[Float32]:
    var geom = _geom(oc, k, s, p)
    var oshape = conv_run_out(geom, shape, shape)
    var y = zeros32(prod(oshape))
    var bias = zeros32(oc)
    conv_run_forward(cnn_entry(), addr(x), addr(wt), addr(bias), addr(y), conv_run_words(geom, shape))
    _ = bias^
    return y^


def _conv_dx(mut x: List[Float32], shape: List[Int], mut wt: List[Float32], oc: Int, k: Int, s: Int, p: Int,
             mut g: List[Float32]) raises -> List[Float32]:
    var geom = _geom(oc, k, s, p)
    var dx = zeros32(len(x))
    var dw = zeros32(len(wt))
    var db = zeros32(oc)
    conv_run_backward(cnn_entry(), addr(x), addr(wt), addr(g), addr(dx), addr(dw), addr(db),
                      conv_run_words(geom, shape))
    _ = dw^
    _ = db^
    return dx^


def _relu64(v: List[Float64]) -> List[Float64]:
    var out = List[Float64](capacity=len(v))
    for i in range(len(v)):  # small-loop(reference activations)
        out.append(max(v[i], 0.0))
    return out^


def _basic_block(k: Int) raises:
    var inplanes = 4 if k == 0 else 3
    var planes = 4 if k == 0 else 5
    var stride = 1 if k == 0 else 2
    var down = stride != 1 or inplanes != planes
    var n = 2
    var xs: List[Int] = [n, inplanes, 6, 6]
    var x = normals(prod(xs), 5)
    var w1 = normals(planes * inplanes * 9, 40 + k, 1.0 / sqrt(Float64(inplanes * 9)))
    var w2 = normals(planes * planes * 9, 50 + k, 1.0 / sqrt(Float64(planes * 9)))
    var wd = normals(planes * inplanes, 60 + k, 1.0 / sqrt(Float64(inplanes)))
    var entry = cnn_entry()
    # the product: conv1, bn1, relu, conv2, bn2, (+ downsample conv, bn), add, relu
    var c1 = _conv_fwd(x, xs, w1, planes, 3, stride, 1)
    var oh = (6 + 2 - 3) // stride + 1
    var hw = oh * oh
    var ys: List[Int] = [n, planes, oh, oh]
    var bn1 = BnLayer(planes)
    var b1 = bn1.forward(c1, n, hw, True, True, True, 0.1)
    var r1 = zeros32(len(b1))
    conv_run_relu_forward(entry, addr(b1), addr(r1), len(b1))
    var c2 = _conv_fwd(r1, ys, w2, planes, 3, 1, 1)
    var bn2 = BnLayer(planes)
    var b2 = bn2.forward(c2, n, hw, True, True, True, 0.1)
    var bnd = BnLayer(planes)
    var cd = zeros32(len(b2))
    var ident = zeros32(len(b2))
    if down:
        cd = _conv_fwd(x, xs, wd, planes, 1, stride, 0)
        ident = bnd.forward(cd, n, hw, True, True, True, 0.1)
    else:
        ident = x.copy()
    var pre = zeros32(len(b2))
    conv_run_add(entry, addr(b2), addr(ident), addr(pre), len(pre))
    var y = zeros32(len(pre))
    conv_run_relu_forward(entry, addr(pre), addr(y), len(pre))
    var g = normals(len(y), 6)
    var gp = zeros32(len(g))
    conv_run_relu_backward(entry, addr(pre), addr(g), addr(gp), len(g))
    var gb2 = bn2.backward(c2, gp, n, hw)
    var gr1 = _conv_dx(r1, ys, w2, planes, 3, 1, 1, gb2)
    var gb1in = zeros32(len(gr1))
    conv_run_relu_backward(entry, addr(b1), addr(gr1), addr(gb1in), len(gr1))
    var gb1 = bn1.backward(c1, gb1in, n, hw)
    var dx1 = _conv_dx(x, xs, w1, planes, 3, stride, 1, gb1)
    var dx = zeros32(len(x))
    if down:
        var gbd = bnd.backward(cd, gp, n, hw)
        var dxd = _conv_dx(x, xs, wd, planes, 1, stride, 0, gbd)
        conv_run_add(entry, addr(dx1), addr(dxd), addr(dx), len(dx))
        _ = gbd^
        _ = dxd^
    else:
        conv_run_add(entry, addr(dx1), addr(gp), addr(dx), len(dx))
    # the float64 reference of the same block (bn: gamma 1, beta 0, batch statistics)
    var ones = List[Float64](length=planes, fill=1.0)
    var nulls = List[Float64](length=planes, fill=0.0)
    var no_g = List[Float64](length=n * planes * hw, fill=0.0)
    var x64 = widen(x)
    var rc1 = _conv64(x64, n, inplanes, 6, 6, w1, planes, 3, stride, 1)
    var rb1 = ref_bn(n, planes, hw, rc1.y, ones, nulls, nulls, ones, True, 0.1, 1e-5, no_g).y.copy()
    var rr1 = _relu64(rb1)
    var rc2 = _conv64(rr1, n, planes, oh, oh, w2, planes, 3, 1, 1)
    var rb2 = ref_bn(n, planes, hw, rc2.y, ones, nulls, nulls, ones, True, 0.1, 1e-5, no_g).y.copy()
    var rident = x64.copy()
    var rcd = _conv64(x64, n, inplanes, 6, 6, wd, planes, 1, stride, 0)
    if down:
        rident = ref_bn(n, planes, hw, rcd.y, ones, nulls, nulls, ones, True, 0.1, 1e-5, no_g).y.copy()
    var rpre = List[Float64](capacity=len(rb2))
    for i in range(len(rb2)):  # small-loop(reference activations)
        rpre.append(rb2[i] + rident[i])
    check_close(y, _relu64(rpre), 1e-3, 1e-4, "basic block " + String(k) + " forward")
    var rgp = List[Float64](capacity=len(rpre))
    for i in range(len(rpre)):  # small-loop(reference activations)
        rgp.append(Float64(g[i]) if rpre[i] > 0 else 0.0)
    var rgb2 = ref_bn(n, planes, hw, rc2.y, ones, nulls, nulls, ones, True, 0.1, 1e-5, rgp).dx.copy()
    var rgr1 = _conv64_dx(rgb2, n, planes, oh, oh, w2, planes, 3, 1, 1, oh, oh)
    for i in range(len(rgr1)):  # small-loop(reference activations)
        if not (rb1[i] > 0):
            rgr1[i] = 0.0
    var rgb1 = ref_bn(n, planes, hw, rc1.y, ones, nulls, nulls, ones, True, 0.1, 1e-5, rgr1).dx.copy()
    var rdx = _conv64_dx(rgb1, n, inplanes, 6, 6, w1, planes, 3, stride, 1, oh, oh)
    if down:
        var rgbd = ref_bn(n, planes, hw, rcd.y, ones, nulls, nulls, ones, True, 0.1, 1e-5, rgp).dx.copy()
        var rdxd = _conv64_dx(rgbd, n, inplanes, 6, 6, wd, planes, 1, stride, 0, oh, oh)
        for i in range(len(rdx)):  # small-loop(reference gradients)
            rdx[i] += rdxd[i]
    else:
        for i in range(len(rdx)):  # small-loop(reference gradients)
            rdx[i] += rgp[i]
    check_close(dx, rdx, 1e-3, 1e-3, "basic block " + String(k) + " dx")
    _ = c1^
    _ = b1^
    _ = r1^
    _ = c2^
    _ = b2^
    _ = cd^
    _ = ident^
    _ = pre^
    _ = gp^
    _ = gr1^
    _ = gb1in^
    _ = w1^
    _ = w2^
    _ = wd^


def case(unit: Int) raises:
    if unit == 0:
        _dropout()
    elif unit == 1:
        _adaptive_dividing()
    elif unit < 8:
        _adaptive(unit - 2)
    else:
        _basic_block(unit - 8)


def run_checks() raises:
    # The publication _mojolearn_x_cnn_host makes at module init.
    publish_x_cnn_host_entries()
    var failures = List[Int](length=CNN_BLOCKS_UNITS, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL cnn blocks unit", unit, String(problem))
    host_parallelize(task, CNN_BLOCKS_UNITS)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "cnn blocks units failed: " + String(failed))
    print("PASS cnn blocks", CNN_BLOCKS_UNITS)
