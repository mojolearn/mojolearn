# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""predict_proba as float64 words (lane apple-fast-q-clf, 2026-10-04,
QUALITY-FIX, FAST default; -D MOJOLEARN_PROBA64_QOLD restores the float32
`row_softmax` probabilities).

WHY. A float32 probability cannot hold 1 - c for c below 2^-24: every row
whose joint log likelihood gap passes about 16.6 nats comes back as exactly
1.0 for its winning class, so `proba[:, 1]` (what a log loss or a ROC AUC of
the positive column reads) loses the model's confidence. Board audit
(board-quality-audit-2026-10-04, istella): log loss FAST vs scikit-learn
(float64 probabilities) gaussian-nb 3.574 vs 3.417, bernoulli-nb 5.351 vs
4.279, multinomial-nb 3.629 vs 3.087, complement-nb 3.763 vs 3.175, qda
3.609 vs 3.477, every accuracy the same. A numpy model of GaussianNB /
BernoulliNB on the same block reproduces it from the dtype alone: float64
probabilities 3.542 / 4.231, the same probabilities rounded to float32
3.679 / 4.896 (13,299 / 18,770 of 100,000 rows at exactly 1.0;
~/mojolearn-evidence/apple-fast-q-clf/sim_proba_f32.py).

WHAT. `q64_softmax_unit` (op P2M_BASE + 13, x_prep/py2mojo.mojo's range, so
the device and host dispatchers carry it) normalises row t of the joint log
likelihood like `row_softmax_unit`, but keeps the winner's complement
c = sum_{k != max} e_k / (1 + sum_{k != max} e_k) apart and writes each
probability as the two 32-bit words (low, high) of an IEEE float64: the
winner's 1 - c built exactly from c's bits (no FP64 arithmetic, which
Apple GPUs lack), every other class (and a winner below 1/2) the exact
widening of its float32 value. Integer bit work only, one thread a row.
Python reads the 2*n*K words as float64 (byte reinterpretation).
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_prep.common import FP, IP, p, ld
from x_prep.prims import add, sub, div, expf

#: The switch the binding's `x_prep_proba64` export reports (Python's probe).
comptime PROBA64 = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and not is_defined["MOJOLEARN_PROBA64_QOLD"]()


@always_inline
def widen_bits(v: Float32) -> UInt64:
    """The IEEE float64 bits of float32 v, exactly (subnormals normalised)."""
    var b = UInt64(bitcast[DType.uint32](v))
    var sign = (b >> 31) << 63
    var e = Int((b >> 23) & 0xFF)
    var man = b & 0x7FFFFF
    if e == 0:
        if man == 0:
            return sign
        var e64 = 897  # 1 - 127 + 1023
        while (man & 0x800000) == 0:
            man = man << 1
            e64 -= 1
        return sign | (UInt64(e64) << 52) | ((man & 0x7FFFFF) << 29)
    if e == 255:
        return sign | (UInt64(0x7FF) << 52) | (man << 29)
    return sign | (UInt64(e - 127 + 1023) << 52) | (man << 29)


@always_inline
def one_minus_bits(c: Float32) -> UInt64:
    """The float64 bits of 1 - c for 0 <= c <= 1/2 (c a float32): 1 - c lies
    in [1/2, 1], so its fraction field is 2^52 - c * 2^53, c * 2^53 being
    c's 24-bit significand shifted (rounded to nearest when it shifts right)."""
    var b = UInt64(bitcast[DType.uint32](c))
    var e = Int((b >> 23) & 0xFF)
    var v = UInt64(0)
    if e != 0:
        var m = (b & 0x7FFFFF) | 0x800000
        var sh = e - 97  # c * 2^53 = m * 2^(e - 150 + 53)
        if sh >= 0:
            v = m << UInt64(sh)
        elif -sh < 25:
            var r = -sh
            v = (m + (UInt64(1) << UInt64(r - 1))) >> UInt64(r)
    var two52 = UInt64(1) << 52
    if v >= two52:
        return UInt64(0x3FE) << 52  # c rounds to 1/2
    return (UInt64(0x3FE) << 52) + (two52 - v)


@always_inline
def _put64(f: FP, w: Int, bits: UInt64):
    """Words w (low) and w + 1 (high) of a float64, stored raw (no flush)."""
    f.unsafe_store(w, bitcast[DType.float32](UInt32(bits & 0xFFFFFFFF)))
    f.unsafe_store(w + 1, bitcast[DType.float32](UInt32(bits >> 32)))


def q64_softmax_unit(t: Int, f: FP, q: IP):
    """q = [S, n, K, OUT64]; t = row. OUT64 + 2 * (t * K + k) holds class k's
    probability as float64 words. A row whose maximum is -inf is uniform."""
    var K = p(q, 2)
    var S = p(q, 0) + t * K
    var o = p(q, 3) + 2 * t * K
    var m = ld(f, S)
    var km = 0
    for k in range(1, K):
        var v = ld(f, S + k)
        if v > m:
            m = v
            km = k
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    if m == neg_inf:
        var u = widen_bits(div(Float32(1), Float32(K)))
        for k in range(K):
            _put64(f, o + 2 * k, u)
        return
    var tail = Float32(0)
    for k in range(K):
        if k != km:
            tail = add(tail, expf(sub(ld(f, S + k), m)))
    var den = add(Float32(1), tail)
    for k in range(K):
        if k != km:
            _put64(f, o + 2 * k, widen_bits(div(expf(sub(ld(f, S + k), m)), den)))
    var c = div(tail, den)
    if c <= Float32(0.5):
        _put64(f, o + 2 * km, one_minus_bits(c))
    else:
        _put64(f, o + 2 * km, widen_bits(div(Float32(1), den)))
