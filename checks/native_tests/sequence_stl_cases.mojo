# SPDX-License-Identifier: Apache-2.0
"""STL's decomposition identities and batch behaviour (ported from
python/mojolearn/tests/test_x_sequence_stl.py).

Series of 96: 3 sin(2 pi t / 12) + 0.05 t + 0.3 z, z a hashed normal per
seed (the original drew z from NumPy; no reference depends on it); period
12 with STL's defaults (seasonal 7, trend 23, low_pass 13, every degree 1,
every jump 1), through `sequence/ops_impl.mojo::stl_impl`.

- fixture 0 (inner 5, outer 0): seasonal + trend + resid == y within atol
  1e-5, and every weight is 1;
- fixture 1 (robust: inner 2, outer 15): every weight in [0, 1], some below 1;
- fixture 2: a batch of three series fitted at once gives each row's trend
  bit for bit equal to that series fitted alone.
"""
from std.math import sin
from std.memory import bitcast
from core.host_parallel import host_parallelize
from sequence.exec import Exec
from sequence.ops_impl import stl_impl
from checks.native_tests.sequence_port_common import require, allclose, normal, count_failures

comptime STL_FIXTURES = 3
comptime STL_N = 96
comptime STL_PERIOD = 12
comptime STL_BATCH = 3


def stl_series(seeds: Int, first: Int = 0) -> List[Float32]:
    """`seeds` series of STL_N, series s from seed first + s."""
    var y = List[Float32](length=seeds * STL_N, fill=0.0)
    var yp = y.unsafe_ptr()
    def cell(i: Int) {imm yp, imm first}:
        var s = first + i // STL_N
        var t = Float64(i % STL_N)
        var value = sin(6.283185307179586 * t / 12.0) * 3.0 + 0.05 * t + normal(0x57100 + s, i % STL_N) * 0.3
        yp.unsafe_store(i, Float32(value))
    host_parallelize(cell, seeds * STL_N)
    return y^


struct StlOut(Movable):
    var season: List[Float32]
    var trend: List[Float32]
    var weights: List[Float32]
    var resid: List[Float32]

    def __init__(out self, count: Int):
        self.season = List[Float32](length=count, fill=-7.0)
        self.trend = List[Float32](length=count, fill=-7.0)
        self.weights = List[Float32](length=count, fill=-7.0)
        self.resid = List[Float32](length=count, fill=-7.0)


def stl_fit[E: Exec](mut ex: E, y: List[Float32], batch: Int, inner: Int, outer: Int) raises -> StlOut:
    var result = StlOut(batch * STL_N)
    var addresses: List[Int] = [Int(y.unsafe_ptr()), Int(result.season.unsafe_ptr()), Int(result.trend.unsafe_ptr()),
                                Int(result.weights.unsafe_ptr()), Int(result.resid.unsafe_ptr())]
    var words: List[Int] = [batch, STL_N, STL_PERIOD, 7, 23, 13, 7, 1, 1, 1, inner, outer]
    require(stl_impl(ex, addresses, words) == batch * STL_N, "STL result count")
    return result^


def run_stl_fixture[E: Exec](mut ex: E, unit: Int) raises:
    if unit == 2:
        var series = stl_series(STL_BATCH)
        var batch = stl_fit(ex, series, STL_BATCH, 5, 0)
        for s in range(STL_BATCH):  # driver-loop: over three single fits on this fixture's one executor
            var single = stl_fit(ex, stl_series(1, s), 1, 5, 0)
            var differ = List[Int](length=STL_N, fill=0)
            var dp = differ.unsafe_ptr()
            var bp = batch.trend.unsafe_ptr() + s * STL_N
            var op = single.trend.unsafe_ptr()
            def word(t: Int) {imm dp, imm bp, imm op}:
                if bitcast[DType.uint32](bp[t]) != bitcast[DType.uint32](op[t]):
                    dp.unsafe_store(t, 1)
            host_parallelize(word, STL_N)
            require(count_failures(differ) == 0, "STL batch row " + String(s) + " trend differs from its single fit in "
                    + String(count_failures(differ)) + " words")
            _ = single^
        _ = series^
        _ = batch^
        return
    var y = stl_series(1)
    var robust = unit == 1
    var out = stl_fit(ex, y, 1, 2 if robust else 5, 15 if robust else 0)
    var failures = List[Int](length=STL_N, fill=0)
    var fp = failures.unsafe_ptr()
    var yp = y.unsafe_ptr()
    var sp = out.season.unsafe_ptr()
    var tp = out.trend.unsafe_ptr()
    var wp = out.weights.unsafe_ptr()
    var rp = out.resid.unsafe_ptr()
    def cell(t: Int) {imm fp, imm yp, imm sp, imm tp, imm wp, imm rp, imm robust}:
        if robust:
            if not (wp[t] >= 0.0 and wp[t] <= 1.0):
                fp.unsafe_store(t, 1)
            return
        var total = Float64(sp[t]) + Float64(tp[t]) + Float64(rp[t])
        if not allclose(total, Float64(yp[t]), 0.0, 1e-5):
            fp.unsafe_store(t, 2)
        if wp[t] != 1.0:
            fp.unsafe_store(t, 3)
    host_parallelize(cell, STL_N)
    for t in range(STL_N):
        require(failures[t] != 1, "robust STL weight outside [0, 1] at " + String(t))
        require(failures[t] != 2, "STL components do not add up to y at " + String(t))
        require(failures[t] != 3, "non-robust STL weight is not 1 at " + String(t))
    if robust:
        var below = 0
        for t in range(STL_N):
            if out.weights[t] < 1.0:
                below += 1
        require(below > 0, "robust STL left every weight at 1")
    _ = y^
    _ = out^
