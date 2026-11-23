# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Public CPU neural inference on the host column (moved from
python/mojolearn/tests/test_neural_inference.py, lane tmig-training-2,
2026-11-15), through the typed RUN entries `_mojolearn_neural_host` calls
with the GIL released: core/mojolearn_neural_host_run.mojo
mlp_forward_logits_run and transformer_forward_fresh_run, and
core/mojolearn_neural_host_impl.mojo _run_forward (the carried-state
forward and the decode step, `transformer_forward`'s 13 + 11 layout).

  0    MLPInference (8-16-3, weights (arange % 7 - 3) / 32): 64 rows of
       logits; rows 0, 1, 7 and 63 alone, and rows 1..7, give the same bytes.
  1, 2 the CPU transformer block (d_model 32, 2 heads, 1 kv head, head_dim
       16, intermediate 64), window 0 and 8: a (4, 16, 32) forward; each
       batch row alone and the 7-token prefix give the same bytes.
  3, 4 the same block, window 0 and 8: the decode is the prefill: a (2, 12,
       32) sequence decoded one position at a time on a carried 12-token
       state equals the fresh forward, bitwise, at every position.
  5    the comparison can fail: one ULP added to one cached v_cache word
       before step 6 moves position 6 (and no earlier one). The word is the
       first of batch row 0's six cached positions whose ULP is not absorbed
       (the Python test named cell 7 for its own inputs; on this LCG stream
       cell 7 is absorbed, and a fixed cell would break whenever the fold
       order, and so the bits, change between versions).

Inputs and weights are a fixed 64-bit LCG stream (transformer_host_block's;
the claims are equalities between routes). The ragged-lengths forward, the
Mamba and Samba inference blocks, the checkpoint integrity refusals, the
refusals by name, the reference-path comparison and the binding's export
list stay in the Python test.
"""
from std.memory import bitcast
from core.mojolearn_neural_host_run import mlp_forward_logits_run, transformer_forward_fresh_run
from core.mojolearn_neural_host_impl import _run_forward
from transformer.block_options import BlockOptions
from checks.native_tests.training_check_common import lcg_fill, same_words
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from checks.native_tests.training_check_common import require


comptime NEURAL_INFERENCE_CASES = 6
comptime _DM = 32
comptime _NH = 2
comptime _NKV = 1
comptime _HD = 16
comptime _IT = 64


def _mlp_weights() -> List[List[Float32]]:
    var sizes: List[Int] = [16 * 8, 16, 3 * 16, 3]
    var out = List[List[Float32]]()
    for s in range(4):  # small-loop(4: weight arrays)
        var w = List[Float32](length=sizes[s], fill=0)
        for i in range(sizes[s]):
            w[i] = Float32((i % 7) - 3) / Float32(32.0)
        out.append(w^)
    return out^


def _logits(w: List[List[Float32]], x: List[Float32], first: Int, rows: Int) raises -> List[Float32]:
    var y = List[Float32](length=rows * 3, fill=Float32(-7.0))
    var a: List[Int] = [
        Int(x.unsafe_ptr()) + 4 * first * 8, Int(w[0].unsafe_ptr()), Int(w[1].unsafe_ptr()), Int(w[2].unsafe_ptr()),
        Int(w[3].unsafe_ptr()), Int(y.unsafe_ptr()),
    ]
    require(mlp_forward_logits_run(a, rows) == rows * 3, "mlp_forward_logits wrote rows * 3 cells")
    _ = x
    _ = w
    return y^


def _slice(values: List[Float32], first: Int, count: Int) -> List[Float32]:
    var out = List[Float32](length=count, fill=0)
    for i in range(count):  # small-loop(fixture slice)
        out[i] = values[first + i]
    return out^


struct _Block(Movable):
    var w: List[List[Float32]]
    var window: Int

    def __init__(out self, window: Int):
        var qw = _NH * _HD
        var kw = _NKV * _HD
        var sizes: List[Int] = [_DM, _DM, qw * _DM, kw * _DM, kw * _DM, _DM * qw, _IT * _DM, _IT * _DM, _DM * _IT]
        self.w = List[List[Float32]]()
        var seed: UInt64 = 1
        for i in range(len(sizes)):  # small-loop(nine weight slots)
            var piece = List[Float32](length=sizes[i], fill=Float32(1.0))
            if i >= 2:  # the two layernorm weights are ones
                lcg_fill(piece, seed, Float32(-0.125), Float32(0.125))
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

    def forward(self, x: List[Float32], b: Int, l: Int) raises -> List[Float32]:
        """`blk.forward(x)`: the stateless prefill."""
        var y = List[Float32](length=b * l * _DM, fill=0)
        var none = List[Float32]()
        _ = transformer_forward_fresh_run(self._addrs(x, none, none, y), b, l, _DM, _NH, _NKV, _HD, _IT,
                                          self.window, BlockOptions(), String("transformer_forward_fresh"))
        _ = x
        _ = none
        return y^

    def cache(self, b: Int, smax: Int) -> List[Float32]:
        var cap = self.window if self.window > 0 else smax
        return List[Float32](length=b * _NKV * cap * _HD, fill=0)

    def step(self, x: List[Float32], b: Int, smax: Int, mut k: List[Float32], mut v: List[Float32],
             cached: Int, mut y: List[Float32]) raises -> Int:
        """`blk.step(token, state)`: the carried-state forward at L = 1."""
        var n = _run_forward(self._addrs(x, k, v, y), b, 1, _DM, _NH, _NKV, _HD, _IT, smax, cached, self.window,
                             BlockOptions())
        _ = x
        _ = k
        _ = v
        _ = y
        return n


def _rows(x: List[Float32], l_all: Int, b_first: Int, b_count: Int, t_first: Int, t_count: Int) -> List[Float32]:
    """x[b_first:b_first + b_count, t_first:t_first + t_count, :], contiguous."""
    var out = List[Float32](length=b_count * t_count * _DM, fill=0)
    for b in range(b_count):  # small-loop(fixture slice)
        for t in range(t_count):
            for d in range(_DM):
                out[(b * t_count + t) * _DM + d] = x[((b_first + b) * l_all + t_first + t) * _DM + d]
    return out^


def _decode_mismatch(blk: _Block, x: List[Float32], b: Int, l: Int, bump: Int, word: Int = 7) raises -> Int:
    """The first position whose decode step differs from the fresh forward
    (-1: none); with `bump` >= 0, one ULP is added to v_cache word `word`
    before the step at position `bump`."""
    var full = blk.forward(x, b, l)
    var k = blk.cache(b, l)
    var v = blk.cache(b, l)
    var cached = 0
    var first = -1
    for t in range(l):  # driver-loop: over decode positions (each step reads the last one's state)
        if t == bump:
            var before = bitcast[DType.uint32](v[word])
            v[word] = bitcast[DType.float32](before + 1)
            require(bitcast[DType.uint32](v[word]) == before + 1, "the cache word moved by one ULP")
        var one = List[Float32](length=b * _DM, fill=0)
        cached = blk.step(_rows(x, l, 0, b, t, 1), b, l, k, v, cached, one)
        require(cached == t + 1, "the state holds every decoded token")
        if first < 0 and not same_words(one, _rows(full, l, 0, b, t, 1)):
            first = t
    _ = k^
    _ = v^
    return first


def run_neural_inference_case(unit: Int) raises:
    if unit == 0:
        var w = _mlp_weights()
        var x = List[Float32](length=64 * 8, fill=0)
        var seed: UInt64 = 0
        lcg_fill(x, seed, Float32(-2.0), Float32(2.0))
        var got = _logits(w, x, 0, 64)
        for k in range(4):  # small-loop(4: single rows)
            var i = 0 if k == 0 else (1 if k == 1 else (7 if k == 2 else 63))
            require(same_words(_logits(w, x, i, 1), _slice(got, 3 * i, 3)), "MLP row " + String(i) + " alone")
        require(same_words(_logits(w, x, 1, 7), _slice(got, 3, 21)), "MLP rows 1..7")
        _ = x^
        return
    if unit <= 2:
        var blk = _Block(0 if unit == 1 else 8)
        var x = List[Float32](length=4 * 16 * _DM, fill=0)
        var seed: UInt64 = 5
        lcg_fill(x, seed, Float32(-2.0), Float32(2.0))
        var y = blk.forward(x, 4, 16)
        for i in range(4):  # small-loop(4: batch rows)
            require(same_words(blk.forward(_rows(x, 16, i, 1, 0, 16), 1, 16), _rows(y, 16, i, 1, 0, 16)),
                    "transformer batch row " + String(i) + " alone")
        require(same_words(blk.forward(_rows(x, 16, 0, 4, 0, 7), 4, 7), _rows(y, 16, 0, 4, 0, 7)),
                "a causal prefix is the prefix of the whole")
        _ = x^
        return
    var x = List[Float32](length=2 * 12 * _DM, fill=0)
    var seed: UInt64 = 23 if unit <= 4 else 3
    lcg_fill(x, seed, Float32(-2.0), Float32(2.0))
    if unit <= 4:
        var blk = _Block(0 if unit == 3 else 8)
        var bad = _decode_mismatch(blk, x, 2, 12, -1)
        require(bad < 0, "decode differs from the prefill at position " + String(bad))
    else:
        require(unit == 5, "unknown neural inference case")
        var blk = _Block(0)
        # v_cache is (b, n_kv, smax, head_dim): batch row 0's positions 0..5 are
        # words 0 .. 6 * _HD - 1. The search stops at the first word whose ULP
        # is not absorbed (in order, so the named word is deterministic); each
        # try is one 12-token decode of the tiny block.
        var word = -1
        var moved = -1
        for w in range(6 * _HD):  # driver-loop: over cache words, first-hit search
            moved = _decode_mismatch(blk, x, 2, 12, 6, w)
            if moved >= 0:
                word = w
                break
        require(word >= 0, "one ULP in any of batch row 0's first six cached v words moved no position")
        require(moved == 6, "one ULP in v_cache word " + String(word) + " moved position " + String(moved)
                + ", not 6")
    _ = x^


# ---- The host column ----
# The host column only: the six independent cases run across host cores (the RUN entries hold no shared state).
def test_neural_inference() raises:
    check_cases("neural_inference", NEURAL_INFERENCE_CASES, run_neural_inference_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
