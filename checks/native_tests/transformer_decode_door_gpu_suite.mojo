# SPDX-License-Identifier: Apache-2.0
"""The public Transformer decode session's native door (lane
fix-transformer-decode-door, 2026-11-13), on a GPU column.

The door (providers/python/transformer_decode_pyapi_door.mojo) now holds a
registry decode session and reaches it through the CausalLM session table,
on a lease it takes once (CAUSAL_SESSION_DECODE_PIN; hot-path audit
2026-11-13): CAUSAL_SESSION_DECODE_PINNED with kind STEP once per token,
kind RUN for the resident N-token loop (`AttnStepArm.decode`). Case 2: the
pin outlives the registry handle (close, then a pinned step is refused as not
open, then UNPIN). Case 0 decodes 8 tokens per row
both ways on two sessions opened on the same small random weights and zero
caches and requires the outputs and both caches to be the same bits, token
for token; then 4 more tokens both ways (the per-token arm's position word
was left due by its steps and is written before the resident loop's first
run). Case 1: a step whose claimed position differs from the session's is
refused and the session stays usable. No Python participates.

The integrator runs `transformer_decode_door_gpu.mojo` in a fresh process;
lanes do not run it.
"""
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.causal_lm_entry_table import CausalEntryKey, causal_entry_vendor
from core.causal_lm_entry_table_sessions import (
    causal_sessions_ref, causal_sessions_call, CAUSAL_SESSIONS_TRANSFORMER,
    CAUSAL_SESSION_DECODE_CREATE, CAUSAL_SESSION_DECODE_OPEN, CAUSAL_SESSION_DECODE_EXPORT,
    CAUSAL_SESSION_DECODE_CLOSE, CAUSAL_SESSION_DECODE_PIN, CAUSAL_SESSION_DECODE_UNPIN,
    CAUSAL_SESSION_DECODE_PINNED, CAUSAL_DECODE_PINNED_STEP, CAUSAL_DECODE_PINNED_RUN,
)
from transformer.causal_entry_device import publish_causal_transformer_device

comptime TRANSFORMER_DECODE_DOOR_CASES = 3
comptime _B = 2
comptime _D = 16
comptime _NH = 2
comptime _NKV = 1
comptime _HD = 8
comptime _IT = 32
comptime _SMAX = 16
comptime _N = 8
comptime _MORE = 4


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() raises -> CausalEntryKey:
    var vendor = causal_entry_vendor(String(COMPILED_VENDOR))
    require(vendor > 0, "the decode door suite runs on a GPU column")
    return CausalEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor)


def _t(op: Int, a: List[Int], words: List[Int]) raises -> Int:
    return causal_sessions_call(causal_sessions_ref(CAUSAL_SESSIONS_TRANSFORMER, _key()), op, a, words,
                                List[Float64]())


def _pin(h: Int) raises -> Int:
    var pin = _t(CAUSAL_SESSION_DECODE_PIN, List[Int](), [h])
    require(pin != 0, "decode pin returned 0")
    return pin


def _unpin(pin: Int) raises:
    _ = _t(CAUSAL_SESSION_DECODE_UNPIN, List[Int](), [pin])


def _fill(mut values: List[Float32], mut seed: UInt64, scale: Float32, offset: Float32):
    # Small deterministic weights (a 64-bit LCG): a fixture, not data work.
    for i in range(len(values)):  # small-loop(fixture: a few thousand test weights)
        seed = seed * 6364136223846793005 + 1442695040888963407
        var unit = Float32(Int((seed >> 40) & 0xFFFF)) / Float32(65536.0) - Float32(0.5)
        values[i] = offset + scale * unit


struct _Block(Movable):
    var w: List[List[Float32]]
    var cache_n: Int

    def __init__(out self) raises:
        var kw = _NKV * _HD
        var sizes: List[Int] = [_D, _D, _D * _D, kw * _D, kw * _D, _D * _D, _IT * _D, _IT * _D, _D * _IT]
        self.w = List[List[Float32]]()
        var seed: UInt64 = 20261113
        for i in range(len(sizes)):  # small-loop(nine weight slots)
            var piece = List[Float32](length=sizes[i], fill=0)
            if i < 2:
                _fill(piece, seed, Float32(0.2), Float32(1.0))
            else:
                _fill(piece, seed, Float32(0.25), Float32(0.0))
            self.w.append(piece^)
        self.cache_n = _B * _NKV * _SMAX * _HD


def _open(block: _Block, kc: List[Float32], vc: List[Float32]) raises -> Int:
    var h = _t(CAUSAL_SESSION_DECODE_CREATE, List[Int](), List[Int]())
    var a = List[Int]()
    for i in range(9):  # small-loop(nine weight addresses)
        a.append(Int(block.w[i].unsafe_ptr()))
    a.append(Int(kc.unsafe_ptr()))
    a.append(Int(vc.unsafe_ptr()))
    var words: List[Int] = [h, _B, _D, _NH, _NKV, _HD, _IT, _SMAX, 0, 0]
    var cached = _t(CAUSAL_SESSION_DECODE_OPEN, a, words)
    require(cached == 0, "decode open returned cached_tokens " + String(cached))
    _ = block
    _ = kc
    _ = vc
    return h


def _same(left: List[Float32], right: List[Float32], what: String) raises:
    require(len(left) == len(right), what + ": lengths differ")
    for i in range(len(left)):  # small-loop(fixture compare: a few hundred floats)
        if bitcast[DType.uint32](left[i]) != bitcast[DType.uint32](right[i]):
            raise Error(what + ": bits differ at " + String(i) + " (" + String(left[i]) + " vs " + String(right[i]) + ")")


def _steps(h: Int, x: List[Float32], mut y: List[Float32], n: Int, start: Int) raises:
    # The per-token door: one STEP per token, token t of every row gathered
    # into a (B, d_model) input and its output scattered back.
    var xt = List[Float32](length=_B * _D, fill=0)
    var yt = List[Float32](length=_B * _D, fill=0)
    for t in range(n):  # driver-loop: over autoregressive decode steps
        for b in range(_B):  # small-loop(fixture gather: two rows)
            for j in range(_D):  # small-loop(fixture gather: sixteen features)
                xt[b * _D + j] = x[(b * n + t) * _D + j]
        var a: List[Int] = [Int(xt.unsafe_ptr()), Int(yt.unsafe_ptr())]
        var words: List[Int] = [h, CAUSAL_DECODE_PINNED_STEP, 0, start + t]
        var cached = _t(CAUSAL_SESSION_DECODE_PINNED, a, words)
        require(cached == start + t + 1, "step returned cached_tokens " + String(cached))
        for b in range(_B):  # small-loop(fixture scatter: two rows)
            for j in range(_D):  # small-loop(fixture scatter: sixteen features)
                y[(b * n + t) * _D + j] = yt[b * _D + j]
    _ = xt
    _ = yt


def _run(h: Int, x: List[Float32], mut y: List[Float32], n: Int, start: Int) raises:
    # The resident loop: one RUN call for all n tokens.
    var a: List[Int] = [Int(x.unsafe_ptr()), Int(y.unsafe_ptr())]
    var words: List[Int] = [h, CAUSAL_DECODE_PINNED_RUN, n, start]
    var cached = _t(CAUSAL_SESSION_DECODE_PINNED, a, words)
    require(cached == start + n, "decode run returned cached_tokens " + String(cached))
    _ = x
    _ = y


def _export(h: Int, cache_n: Int) raises -> List[List[Float32]]:
    var k = List[Float32](length=cache_n, fill=0)
    var v = List[Float32](length=cache_n, fill=0)
    var a: List[Int] = [Int(k.unsafe_ptr()), Int(v.unsafe_ptr())]
    var words: List[Int] = [h]
    _ = _t(CAUSAL_SESSION_DECODE_EXPORT, a, words)
    var both = List[List[Float32]]()
    both.append(k^)
    both.append(v^)
    return both^


def case(unit: Int) raises:
    var block = _Block()
    var kc = List[Float32](length=block.cache_n, fill=0)
    var vc = List[Float32](length=block.cache_n, fill=0)
    var stepped_h = _open(block, kc, vc)
    var resident_h = _open(block, kc, vc)
    # `_steps` / `_run` take the pin word (the door's per-token route)
    var stepped = _pin(stepped_h)
    var resident = _pin(resident_h)
    var closed = False
    try:
        if unit == 0:
            var seed: UInt64 = 7
            var x = List[Float32](length=_B * _N * _D, fill=0)
            _fill(x, seed, Float32(2.0), Float32(0.0))
            var y_step = List[Float32](length=_B * _N * _D, fill=0)
            var y_run = List[Float32](length=_B * _N * _D, fill=0)
            _steps(stepped, x, y_step, _N, 0)
            _run(resident, x, y_run, _N, 0)
            _same(y_step, y_run, "8 steps vs one resident decode")
            var more = List[Float32](length=_B * _MORE * _D, fill=0)
            _fill(more, seed, Float32(2.0), Float32(0.0))
            var m_step = List[Float32](length=_B * _MORE * _D, fill=0)
            var m_run = List[Float32](length=_B * _MORE * _D, fill=0)
            # Both sessions continue with the resident loop: the stepped
            # session's due position word is written before its first run.
            _run(stepped, more, m_step, _MORE, _N)
            _run(resident, more, m_run, _MORE, _N)
            _same(m_step, m_run, "4 more tokens after 8 steps vs after one decode")
            var ks = _export(stepped_h, block.cache_n)
            var kr = _export(resident_h, block.cache_n)
            _same(ks[0], kr[0], "key cache")
            _same(ks[1], kr[1], "value cache")
        elif unit == 1:
            var xt = List[Float32](length=_B * _D, fill=1)
            var yt = List[Float32](length=_B * _D, fill=0)
            var a: List[Int] = [Int(xt.unsafe_ptr()), Int(yt.unsafe_ptr())]
            var wrong: List[Int] = [stepped, CAUSAL_DECODE_PINNED_STEP, 0, 3]
            try:
                _ = _t(CAUSAL_SESSION_DECODE_PINNED, a, wrong)
                raise Error("a step at the wrong position was not refused")
            except problem:
                require("the caller believes cached_tokens = 3" in String(problem), "unexpected refusal: " + String(problem))
            var right: List[Int] = [stepped, CAUSAL_DECODE_PINNED_STEP, 0, 0]
            require(_t(CAUSAL_SESSION_DECODE_PINNED, a, right) == 1, "the session was not usable after a refused step")
            _ = xt
            _ = yt
        else:
            # The pin outlives the handle: after CLOSE (handle released), a
            # pinned step is refused as not open, never a freed read.
            _ = _t(CAUSAL_SESSION_DECODE_CLOSE, List[Int](), [stepped_h])
            _ = _t(CAUSAL_SESSION_DECODE_CLOSE, List[Int](), [resident_h])
            closed = True
            var xt = List[Float32](length=_B * _D, fill=1)
            var yt = List[Float32](length=_B * _D, fill=0)
            var a: List[Int] = [Int(xt.unsafe_ptr()), Int(yt.unsafe_ptr())]
            try:
                _ = _t(CAUSAL_SESSION_DECODE_PINNED, a, [stepped, CAUSAL_DECODE_PINNED_STEP, 0, 0])
                raise Error("a pinned step on a closed session was not refused")
            except problem:
                require("not open" in String(problem), "unexpected refusal: " + String(problem))
            _ = xt
            _ = yt
    finally:
        if not closed:
            _ = _t(CAUSAL_SESSION_DECODE_CLOSE, List[Int](), [stepped_h])
            _ = _t(CAUSAL_SESSION_DECODE_CLOSE, List[Int](), [resident_h])
        _unpin(stepped)
        _unpin(resident)
    _ = block
    _ = kc
    _ = vc


def run_checks() raises:
    publish_causal_transformer_device()
    var failed = 0
    for unit in range(TRANSFORMER_DECODE_DOOR_CASES):  # driver-loop: device cases share one device context
        try:
            case(unit)
        except problem:
            failed += 1
            print("FAIL transformer decode door case", unit, String(problem))
    require(failed == 0, "transformer decode door cases failed: " + String(failed))
    print("PASS transformer decode door", TRANSFORMER_DECODE_DOOR_CASES)
