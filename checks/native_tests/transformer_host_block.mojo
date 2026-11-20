# SPDX-License-Identifier: Apache-2.0
"""The Transformer block on the HOST column through the typed RUN of its
host binding (transformer/transformer_host_binding_impl.mojo `_run_forward`
and `_backward_run`, the calls `bindings/_mojolearn_transformer_host.mojo`
makes with the GIL released), moved from the runtime half of
test_cpu_training_transformer.py (2026-11-14).

The lanes' shape: d_model 32, two heads, one kv head, head_dim 16,
intermediate 64, B 2; full causal (fixture 0) and window 8 (fixture 1).
For each window:
  * the stateless forward (`transformer_forward_fresh`: a zero cache of
    capacity L, not written back) of 16 tokens equals the carried-state
    prefill of the same 16 tokens into a 32-token state, by bytes, and the
    state then holds 16 tokens;
  * a decode step of token 17 on that state equals row 16 of the stateless
    17-token forward (contract section 7.2's decode == prefill), and the
    17-token forward's first 16 rows equal the 16-token forward's (a later
    token moves no earlier row);
  * a prefill split 8 + 8 into a fresh state equals the whole 16-token
    prefill;
  * the backward returns the input and nine weight gradients (one buffer
    each at its shape's extent), all finite, and the same bytes twice;
  * a NaN input is refused by name.
The input data are a fixed 64-bit LCG stream (the claims are equalities
between routes, so no NumPy stream is needed). The source-text checks of the
manifest, the bindings and the workflow stay in the Python file. The bit
claim against the GPU columns is the CPU identity gate's, not this check's.

The two windows are independent and run across host cores
(`host_parallelize`), each with a private failure slot.
"""
from std.math import isfinite
from std.memory import bitcast
from core.host_parallel import host_parallelize
from transformer.block_options import BlockOptions
from transformer.transformer_host_binding_impl import _backward_run, _run_forward
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


# ---- The host column (was transformer_host_block_cpu_suite.mojo) ----
comptime TRANSFORMER_HOST_BLOCK_FIXTURES = 2
comptime _B = 2
comptime _DM = 32
comptime _NH = 2
comptime _NKV = 1
comptime _HD = 16
comptime _IT = 64
comptime _SMAX = 32


def _fill(mut values: List[Float32], mut seed: UInt64, lo: Float32, hi: Float32):
    # A fixed 64-bit LCG: a fixture, not data work.
    for i in range(len(values)):  # small-loop(fixture: a few thousand test values)
        seed = seed * 6364136223846793005 + 1442695040888963407
        var unit = Float32(Int((seed >> 40) & 0xFFFFFF)) / Float32(16777216.0)
        values[i] = lo + (hi - lo) * unit


def _same(left: List[Float32], right: List[Float32], what: String) raises:
    require(len(left) == len(right), what + ": lengths differ")
    for i in range(len(left)):  # small-loop(fixture compare)
        if bitcast[DType.uint32](left[i]) != bitcast[DType.uint32](right[i]):
            raise Error(what + ": bytes differ at " + String(i))


def _tokens(x: List[Float32], l_all: Int, first: Int, count: Int) -> List[Float32]:
    """x[:, first:first + count, :] of a (B, l_all, d_model) slab, contiguous."""
    var out = List[Float32](length=_B * count * _DM, fill=0)
    for b in range(_B):
        for t in range(count):
            for d in range(_DM):
                out[(b * count + t) * _DM + d] = x[(b * l_all + first + t) * _DM + d]
    return out^


struct _HostBlock(Movable):
    var w: List[List[Float32]]
    var window: Int

    def __init__(out self, window: Int):
        var qw = _NH * _HD
        var kw = _NKV * _HD
        var sizes: List[Int] = [_DM, _DM, qw * _DM, kw * _DM, kw * _DM, _DM * qw, _IT * _DM, _IT * _DM, _DM * _IT]
        self.w = List[List[Float32]]()
        var seed: UInt64 = 11
        for i in range(len(sizes)):  # small-loop(nine weight slots)
            var piece = List[Float32](length=sizes[i], fill=Float32(1.0))
            if i >= 2:  # the two layernorm weights are ones, as in the former test
                _fill(piece, seed, Float32(-0.125), Float32(0.125))
            self.w.append(piece^)
        self.window = window

    def _addrs(self, x: List[Float32], k: List[Float32], v: List[Float32], y: List[Float32]) -> List[Int]:
        var a = List[Int]()
        a.append(Int(x.unsafe_ptr()))
        for i in range(9):  # small-loop(nine weight addresses)
            a.append(Int(self.w[i].unsafe_ptr()))
        a.append(Int(k.unsafe_ptr()))
        a.append(Int(v.unsafe_ptr()))
        a.append(Int(y.unsafe_ptr()))
        for _ in range(11):  # small-loop(the 11 absent option-tail addresses)
            a.append(0)
        return a^

    def fresh(self, x: List[Float32], l: Int) raises -> List[Float32]:
        """`blk.forward(x)`: transformer_forward_fresh, capacity L, s0 0."""
        var y = List[Float32](length=_B * l * _DM, fill=0)
        var none = List[Float32]()
        _ = _run_forward(self._addrs(x, none, none, y), _B, l, _DM, _NH, _NKV, _HD, _IT, l, 0,
                         self.window, False, BlockOptions())
        _ = x
        _ = none
        return y^

    def cache(self) -> List[Float32]:
        var cap = self.window if self.window > 0 else _SMAX
        return List[Float32](length=_B * _NKV * cap * _HD, fill=0)

    def carried(self, x: List[Float32], l: Int, mut k: List[Float32], mut v: List[Float32],
                cached: Int, mut y: List[Float32]) raises -> Int:
        """`blk.forward(x, state)` (and `blk.step` at L = 1) on a 32-token state."""
        var n = _run_forward(self._addrs(x, k, v, y), _B, l, _DM, _NH, _NKV, _HD, _IT, _SMAX, cached,
                             self.window, True, BlockOptions())
        _ = x
        _ = k
        _ = v
        _ = y
        return n

    def backward(self, x: List[Float32], g: List[Float32], l: Int) raises -> List[List[Float32]]:
        var qw = _NH * _HD
        var kw = _NKV * _HD
        var sizes: List[Int] = [_B * l * _DM, _DM, _DM, qw * _DM, kw * _DM, kw * _DM, _DM * qw,
                                _IT * _DM, _IT * _DM, _DM * _IT]
        var grads = List[List[Float32]]()
        for i in range(10):  # small-loop(ten gradient buffers)
            grads.append(List[Float32](length=sizes[i], fill=Float32(-7.0)))
        var a = List[Int]()
        a.append(Int(x.unsafe_ptr()))
        for i in range(9):  # small-loop(nine weight addresses)
            a.append(Int(self.w[i].unsafe_ptr()))
        a.append(Int(g.unsafe_ptr()))
        for i in range(10):  # small-loop(ten gradient addresses)
            a.append(Int(grads[i].unsafe_ptr()))
        _backward_run(a, _B, l, _DM, _NH, _NKV, _HD, _IT, self.window)
        _ = x
        _ = g
        return grads^


def run_host_block_fixture(fixture_id: Int) raises:
    var window = 0 if fixture_id == 0 else 8
    var tag = "window " + String(window)
    var blk = _HostBlock(window)
    var seed: UInt64 = 5
    var x = List[Float32](length=_B * 17 * _DM, fill=0)
    _fill(x, seed, Float32(-2.0), Float32(2.0))
    var g = List[Float32](length=_B * 16 * _DM, fill=0)
    _fill(g, seed, Float32(-1.0), Float32(1.0))
    var x16 = _tokens(x, 17, 0, 16)

    var whole16 = blk.fresh(x16, 16)
    var k = blk.cache()
    var v = blk.cache()
    var prefill = List[Float32](length=_B * 16 * _DM, fill=0)
    var cached = blk.carried(x16, 16, k, v, 0, prefill)
    _same(whole16, prefill, tag + ": stateless forward != carried prefill")
    require(cached == 16, tag + ": the state holds " + String(cached) + " tokens after a 16-token prefill")

    var step = List[Float32](length=_B * _DM, fill=0)
    cached = blk.carried(_tokens(x, 17, 16, 1), 1, k, v, cached, step)
    require(cached == 17, tag + ": the state holds " + String(cached) + " tokens after the decode step")
    var whole17 = blk.fresh(x, 17)
    _same(step, _tokens(whole17, 17, 16, 1), tag + ": decode step != prefill row 16")
    _same(_tokens(whole17, 17, 0, 16), whole16, tag + ": a later token moved an earlier row")

    var k2 = blk.cache()
    var v2 = blk.cache()
    var first = List[Float32](length=_B * 8 * _DM, fill=0)
    var second = List[Float32](length=_B * 8 * _DM, fill=0)
    cached = blk.carried(_tokens(x, 17, 0, 8), 8, k2, v2, 0, first)
    cached = blk.carried(_tokens(x, 17, 8, 8), 8, k2, v2, cached, second)
    require(cached == 16, tag + ": split prefill left " + String(cached) + " tokens")
    var joined = List[Float32](length=_B * 16 * _DM, fill=0)
    for b in range(_B):
        for t in range(16):
            for d in range(_DM):
                var src = first[(b * 8 + t) * _DM + d] if t < 8 else second[(b * 8 + t - 8) * _DM + d]
                joined[(b * 16 + t) * _DM + d] = src
    _same(joined, whole16, tag + ": split prefill moved")

    var grads = blk.backward(x16, g, 16)
    var again = blk.backward(x16, g, 16)
    require(len(grads) == 10, tag + ": backward returned " + String(len(grads)) + " gradients, not x and nine weights")
    for i in range(10):
        _same(grads[i], again[i], tag + ": backward gradient " + String(i) + " differs between two calls")
        for j in range(len(grads[i])):
            require(isfinite(grads[i][j]), tag + ": backward gradient " + String(i) + " is not finite at " + String(j))

    var bad = x16.copy()
    bad[(0 * 16 + 3) * _DM + 5] = bitcast[DType.float32](UInt32(0x7FC00000))
    try:
        _ = blk.fresh(bad, 16)
    except problem:
        var text = String(problem)
        require("non-finite" in text or "NaN" in text or "nan" in text, tag + ": unexpected refusal: " + text)
        return
    raise Error(tag + ": a NaN input was not refused")


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(TRANSFORMER_HOST_BLOCK_FIXTURES)
    def task(fixture_id: Int) {imm cases}:
        try:
            run_host_block_fixture(fixture_id)
        except error:
            cases.fail(fixture_id, error)
    report.parallel(task, TRANSFORMER_HOST_BLOCK_FIXTURES)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("transformer_host_block", cpu=True, gpu=False, serial="", run=run_cases)
