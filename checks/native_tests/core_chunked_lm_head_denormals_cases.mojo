# SPDX-License-Identifier: Apache-2.0
"""Chunked LM-head folds obey identical's signed-zero denormal policy, moved
from test_chunked_lm_head_denormals.py (2026-11-14).

For (rows, vocab, width) in (3, 11, 7), (9, 37, 17), (17, 67, 31): hidden in
U(-1, 1) and weight in U(-0.5, 0.5) (a SplitMix64 stream regenerated here; the
property holds for any finite operands, so the former PCG64 draws are not
needed), whose first two feature columns hold the words 1, 0x80000001,
0x007fffff, 0x807fffff (hidden) and the same words reversed (weight), cyclic
over the row-major (rows, 2) block, so both signs and tiny and near-normal
mantissas are present. The flushed operands keep only the sign of every
subnormal. Required, word for word: the loss, d_hidden and d_weight of the
operands equal those of the flushed operands; every value is finite; the
loss-only call equals the training call's loss. The suites supply the two
product calls (the loss and the training stage of chunked_lm_head_v2) for
their column.
"""
from std.math import isfinite
from std.memory import bitcast

comptime CHUNKED_DENORMAL_FIXTURES = 3


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


trait ChunkedHeadCalls:
    @staticmethod
    def loss(loss: Int, row_max: Int, denom: Int, hidden: Int, weight: Int, targets: Int,
             rows: Int, vocab: Int, width: Int) raises: ...
    @staticmethod
    def train(loss: Int, row_max: Int, denom: Int, d_hidden: Int, d_weight: Int, hidden: Int,
              weight: Int, targets: Int, rows: Int, vocab: Int, width: Int) raises: ...


def _shape(fixture_id: Int) -> Tuple[Int, Int, Int]:
    if fixture_id == 0:
        return (3, 11, 7)
    if fixture_id == 1:
        return (9, 37, 17)
    return (17, 67, 31)


struct _Stream(Movable):
    var state: UInt64

    def __init__(out self, seed: Int):
        self.state = UInt64(seed)

    def next(mut self) -> UInt64:
        self.state += 0x9E3779B97F4A7C15
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def uniform(mut self, lo: Float64, hi: Float64) -> Float32:
        var u = (self.next() >> 11).cast[DType.float64]() * (1.0 / 9007199254740992.0)
        return Float32(lo + (hi - lo) * u)


def _flush(values: List[Float32]) -> List[Float32]:
    var out = values.copy()
    for i in range(len(out)):
        var bits = bitcast[DType.uint32](out[i])
        if (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0:
            out[i] = bitcast[DType.float32](bits & 0x80000000)
    return out^


def _plant(mut values: List[Float32], rows: Int, width: Int, reverse: Bool):
    var words: List[UInt32] = [1, 0x80000001, 0x007FFFFF, 0x807FFFFF]
    for r in range(rows):
        for c in range(2):
            var k = (r * 2 + c) % 4
            values[r * width + c] = bitcast[DType.float32](words[3 - k] if reverse else words[k])


def _addr(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


struct HeadResult(Movable):
    var loss: List[Float32]
    var d_hidden: List[Float32]
    var d_weight: List[Float32]

    def __init__(out self, rows: Int, vocab: Int, width: Int):
        self.loss = List[Float32](length=1, fill=0)
        self.d_hidden = List[Float32](length=rows * width, fill=0)
        self.d_weight = List[Float32](length=vocab * width, fill=0)


def _train[C: ChunkedHeadCalls](
    hidden: List[Float32], weight: List[Float32], targets: List[Int32], rows: Int, vocab: Int, width: Int
) raises -> HeadResult:
    var r = HeadResult(rows, vocab, width)
    var row_max = List[Float32](length=rows, fill=0)
    var denom = List[Float32](length=rows, fill=0)
    C.train(_addr(r.loss), _addr(row_max), _addr(denom), _addr(r.d_hidden), _addr(r.d_weight),
            _addr(hidden), _addr(weight), Int(targets.unsafe_ptr()), rows, vocab, width)
    return r^


def _same_words(a: List[Float32], b: List[Float32], name: String) raises:
    require(len(a) == len(b), name + ": shape")
    for i in range(len(a)):
        require(isfinite(a[i]), name + ": non-finite at " + String(i))
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]),
                name + ": subnormal operands differ from signed zeros at " + String(i))


def run_chunked_denormal_fixture[C: ChunkedHeadCalls](fixture_id: Int) raises:
    require(fixture_id >= 0 and fixture_id < CHUNKED_DENORMAL_FIXTURES, "unknown chunked head fixture")
    var shape = _shape(fixture_id)
    var rows = shape[0]
    var vocab = shape[1]
    var width = shape[2]
    var rng = _Stream(704 + fixture_id)
    var hidden = List[Float32](length=rows * width, fill=0)
    for i in range(len(hidden)):
        hidden[i] = rng.uniform(-1.0, 1.0)
    var weight = List[Float32](length=vocab * width, fill=0)
    for i in range(len(weight)):
        weight[i] = rng.uniform(-0.5, 0.5)
    _plant(hidden, rows, width, False)
    _plant(weight, vocab, width, True)
    var targets = List[Int32](length=rows, fill=0)
    for i in range(rows):
        targets[i] = Int32(Int(rng.next() % UInt64(vocab)))
    var flushed_h = _flush(hidden)
    var flushed_w = _flush(weight)
    require(bitcast[DType.uint32](flushed_h[0]) != bitcast[DType.uint32](hidden[0]), "hidden holds a subnormal")
    require(bitcast[DType.uint32](flushed_w[0]) != bitcast[DType.uint32](weight[0]), "weight holds a subnormal")
    var actual = _train[C](hidden, weight, targets, rows, vocab, width)
    var expected = _train[C](flushed_h, flushed_w, targets, rows, vocab, width)
    var alone = List[Float32](length=1, fill=0)
    var row_max = List[Float32](length=rows, fill=0)
    var denom = List[Float32](length=rows, fill=0)
    C.loss(_addr(alone), _addr(row_max), _addr(denom), _addr(hidden), _addr(weight), Int(targets.unsafe_ptr()),
           rows, vocab, width)
    require(bitcast[DType.uint32](alone[0]) == bitcast[DType.uint32](actual.loss[0]),
            "the loss-only call differs from the training call's loss")
    _same_words(actual.loss, expected.loss, "loss")
    _same_words(actual.d_hidden, expected.d_hidden, "d_hidden")
    _same_words(actual.d_weight, expected.d_weight, "d_weight")
