# SPDX-License-Identifier: Apache-2.0
"""MaxPool2d / AvgPool2d (and the 1-D forms) forward and backward against
the float64 reference of PyTorch's pooling, on the host column (lane
tmig-sequence-1, 2026-11-14; ported from `test_x_cnn_pool.py`).

Units 0-7: `test_maxpool2d` (MAX_CASES x {continuous, tied} inputs): the
forward equals the reference exactly (the first maximum in (kh, kw) order
wins a tie), dx within 1e-5. Units 8-11: `test_avgpool2d` (AVG_CASES,
count_include_pad on and off), within rtol 1e-5 / atol 1e-6. Unit 12:
`test_pool1d` (MaxPool1d(3, 2, 1), AvgPool1d(4, 2, 2, count_include_pad
False) over (N, C, 1, L)). Units 13-16: `test_pool_ceil_and_divisor`
(CEIL_CASES: ceil mode, both padding counts, divisor_override).

The torch comparisons of the references stay in Python as ORACLE tests.
The units are independent and run across the native host worker pool.
"""
from core.host_parallel import host_parallelize
from x_cnn.host.entry_runs_host import publish_x_cnn_host_entries
from x_cnn.pool_run import (
    PoolSettings, pool_run_words, pool_run_out_words, pool_run_max_forward, pool_run_max_backward,
    pool_run_avg_forward, pool_run_avg_backward,
)
from checks.native_tests.cnn_cases import (
    require, cnn_entry, addr, addr_i32, zeros32, prod, normals, integers, check_close, check_equal,
)

comptime CNN_POOL_UNITS = 17


@fieldwise_init
struct PoolCase(Copyable, Movable):
    var kh: Int
    var kw: Int
    var sh: Int
    var sw: Int
    var ph: Int
    var pw: Int
    var dh: Int
    var dw: Int
    var include_pad: Bool
    var ceil: Bool
    var divisor: Int

    def settings(self) -> PoolSettings:
        return PoolSettings(self.kh, self.kw, self.sh, self.sw, self.ph, self.pw, self.dh, self.dw,
                            self.include_pad, self.ceil, self.divisor)


def _out_1(size: Int, k: Int, s: Int, p: Int, d: Int, ceil: Bool) -> Int:
    var num = size + 2 * p - d * (k - 1) - 1 + ((s - 1) if ceil else 0)
    var out = num // s + 1
    if ceil and (out - 1) * s >= size + p:
        out -= 1
    return out


struct PoolReference(Movable):
    var y: List[Float64]
    var dx: List[Float64]

    def __init__(out self, var y: List[Float64], var dx: List[Float64]):
        self.y = y^
        self.dx = dx^


def ref_maxpool(pc: PoolCase, shape: List[Int], x: List[Float32], g: List[Float32]) -> PoolReference:
    var n = shape[0] * shape[1]
    var h = shape[2]
    var w = shape[3]
    var oh = _out_1(h, pc.kh, pc.sh, pc.ph, pc.dh, pc.ceil)
    var ow = _out_1(w, pc.kw, pc.sw, pc.pw, pc.dw, pc.ceil)
    var y = List[Float64](length=n * oh * ow, fill=0.0)
    var dx = List[Float64](length=n * h * w, fill=0.0)
    for plane in range(n):  # small-loop(reference planes)
        for i in range(oh):  # small-loop(reference output rows)
            for j in range(ow):  # small-loop(reference output columns)
                var best = -1
                var best_value = 0.0
                for a in range(pc.kh):  # small-loop(window rows)
                    var r = i * pc.sh - pc.ph + a * pc.dh
                    for b in range(pc.kw):  # small-loop(window columns)
                        var c = j * pc.sw - pc.pw + b * pc.dw
                        if r < 0 or r >= h or c < 0 or c >= w:
                            continue
                        var v = Float64(x[(plane * h + r) * w + c])
                        if best < 0 or v > best_value:   # the first maximum
                            best = (plane * h + r) * w + c
                            best_value = v
                var yi = (plane * oh + i) * ow + j
                y[yi] = best_value
                dx[best] += Float64(g[yi])
    return PoolReference(y^, dx^)


def ref_avgpool(pc: PoolCase, shape: List[Int], x: List[Float32], g: List[Float32]) -> PoolReference:
    var n = shape[0] * shape[1]
    var h = shape[2]
    var w = shape[3]
    var oh = _out_1(h, pc.kh, pc.sh, pc.ph, 1, pc.ceil)
    var ow = _out_1(w, pc.kw, pc.sw, pc.pw, 1, pc.ceil)
    var y = List[Float64](length=n * oh * ow, fill=0.0)
    var dx = List[Float64](length=n * h * w, fill=0.0)
    for plane in range(n):  # small-loop(reference planes)
        for i in range(oh):  # small-loop(reference output rows)
            for j in range(ow):  # small-loop(reference output columns)
                var hs = i * pc.sh - pc.ph
                var ws = j * pc.sw - pc.pw
                var he = min(hs + pc.kh, h + pc.ph)
                var we = min(ws + pc.kw, w + pc.pw)
                var div: Int
                if pc.include_pad:
                    div = (he - hs) * (we - ws)
                else:
                    div = (min(he, h) - max(hs, 0)) * (min(we, w) - max(ws, 0))
                if pc.divisor != 0:
                    div = pc.divisor
                var yi = (plane * oh + i) * ow + j
                var acc = 0.0
                for a in range(pc.kh):  # small-loop(window rows)
                    var r = hs + a
                    for b in range(pc.kw):  # small-loop(window columns)
                        var c = ws + b
                        if r < 0 or r >= h or c < 0 or c >= w:
                            continue
                        acc += Float64(x[(plane * h + r) * w + c])
                        dx[(plane * h + r) * w + c] += Float64(g[yi]) / Float64(div)
                y[yi] = acc / Float64(div)
    return PoolReference(y^, dx^)


def _max(pc: PoolCase, shape: List[Int], mut x: List[Float32], g_seed: Int, rtol: Float64, atol: Float64,
         what: String) raises:
    var entry = cnn_entry()
    var words = pool_run_words(pc.settings(), shape)
    var oshape = pool_run_out_words(words, shape)
    var y = zeros32(prod(oshape))
    var idx = List[Int32](length=prod(oshape), fill=0)
    pool_run_max_forward(entry, addr(x), addr(y), addr_i32(idx), words)
    var g = normals(len(y), g_seed)
    var dx = zeros32(len(x))
    pool_run_max_backward(entry, addr(g), addr_i32(idx), addr(dx), words)
    var want = ref_maxpool(pc, shape, x, g)
    # the forward is exact: a compared maximum is one of the input words
    var exact = List[Float32](capacity=len(want.y))
    for i in range(len(want.y)):  # small-loop(fixture outputs)
        exact.append(Float32(want.y[i]))
    check_equal(y, exact, what + " out")
    check_close(dx, want.dx, rtol, atol, what + " dx")
    _ = idx^
    _ = g^


def _avg(pc: PoolCase, shape: List[Int], mut x: List[Float32], g_seed: Int, what: String) raises:
    var entry = cnn_entry()
    var words = pool_run_words(pc.settings(), shape)
    var oshape = pool_run_out_words(words, shape)
    var y = zeros32(prod(oshape))
    pool_run_avg_forward(entry, addr(x), addr(y), words)
    var g = normals(len(y), g_seed)
    var dx = zeros32(len(x))
    pool_run_avg_backward(entry, addr(g), addr(dx), words)
    var want = ref_avgpool(pc, shape, x, g)
    check_close(y, want.y, 1e-5, 1e-6, what + " out")
    check_close(dx, want.dx, 1e-5, 1e-6, what + " dx")
    _ = g^


def _max_case(k: Int) -> PoolCase:
    if k == 0:
        return PoolCase(2, 3, 1, 1, 1, 1, 1, 1, True, False, 0)
    if k == 1:
        return PoolCase(3, 3, 2, 2, 1, 1, 1, 1, True, False, 0)
    if k == 2:
        return PoolCase(2, 2, 1, 2, 1, 1, 1, 2, True, False, 0)
    return PoolCase(2, 2, 2, 2, 0, 0, 1, 1, True, False, 0)


def _avg_case(k: Int) -> PoolCase:
    if k == 0:
        return PoolCase(3, 3, 1, 1, 1, 1, 1, 1, True, False, 0)
    if k == 1:
        return PoolCase(2, 3, 1, 2, 1, 1, 1, 1, False, False, 0)
    if k == 2:
        return PoolCase(2, 2, 2, 2, 0, 0, 1, 1, True, False, 0)
    return PoolCase(3, 2, 2, 1, 1, 1, 1, 1, True, False, 0)


def _ceil_case(k: Int, include_pad: Bool, divisor: Int) -> PoolCase:
    if k == 0:
        return PoolCase(3, 3, 2, 2, 1, 1, 1, 1, include_pad, True, divisor)
    if k == 1:
        return PoolCase(2, 3, 2, 2, 0, 1, 1, 1, include_pad, True, divisor)
    if k == 2:
        return PoolCase(3, 2, 2, 3, 1, 0, 1, 1, include_pad, True, divisor)
    return PoolCase(2, 2, 3, 3, 1, 1, 1, 1, include_pad, True, divisor)


def case(unit: Int) raises:
    var shape: List[Int] = [2, 3, 6, 7]
    if unit < 8:
        var k = unit // 2
        var ties = unit % 2 == 1
        var x = integers(prod(shape), 10 + k, 0, 3) if ties else normals(prod(shape), 10 + k)
        _max(_max_case(k), shape, x, 20 + unit, 1e-5, 1e-5,
             "maxpool2d case " + String(k) + (String(" ties") if ties else String("")))
    elif unit < 12:
        var k = unit - 8
        var x = normals(prod(shape), 30 + k)
        _avg(_avg_case(k), shape, x, 40 + k, "avgpool2d case " + String(k))
    elif unit == 12:
        var s1: List[Int] = [2, 3, 1, 11]
        var x = normals(prod(s1), 3)
        _max(PoolCase(1, 3, 1, 2, 0, 1, 1, 1, True, False, 0), s1, x, 4, 1e-7, 1e-6, "maxpool1d")
        _avg(PoolCase(1, 4, 1, 2, 0, 2, 1, 1, False, False, 0), s1, x, 5, "avgpool1d")
    else:
        var k = unit - 13
        var include_pad = k == 0 or k == 2
        var divisor = 5 if k == 2 else (3 if k == 3 else 0)
        var x = normals(prod(shape), 50 + k)
        _max(_ceil_case(k, True, 0), shape, x, 60 + k, 1e-5, 1e-6, "ceil maxpool2d case " + String(k))
        _avg(_ceil_case(k, include_pad, divisor), shape, x, 70 + k, "ceil avgpool2d case " + String(k))


def run_checks() raises:
    # The publication _mojolearn_x_cnn_host makes at module init.
    publish_x_cnn_host_entries()
    var failures = List[Int](length=CNN_POOL_UNITS, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL cnn pool unit", unit, String(problem))
    host_parallelize(task, CNN_POOL_UNITS)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "cnn pool units failed: " + String(failed))
    print("PASS cnn pool", CNN_POOL_UNITS)
