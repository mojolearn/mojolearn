# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""scikit-learn's `gamma='scale'` from the exact limbs, in exact integer
arithmetic (lane migrate-classical-b2).

`python/mojolearn/_scale_gamma.py::scale_gamma_x` added the 27 limb words
the binding's `scale_gamma_limbs` writes (svm/impl/scale_gamma_limbs.mojo:
S1 = sum x at scale 2^-149 in 9 signed base-2^32 words, S2 = sum x * x at
scale 2^-298 in 18) into two Python integers and returned

    float(Fraction(N^2 << 298, n_features * (N * S2 - S1^2)))

(1.0 when the spread is 0). This module forms the same integers as
unsigned 32-bit limb magnitudes and rounds the same rational ONCE to
binary64, round-half-to-even, which is what Python's `float(Fraction)`
(`int / int` true division, correctly rounded) does. The result range is
always normal: the numerator is below 2^(2*64+298) and the denominator at
least 1 and below 2^(64 + 64 + 64 + 2*316), so no subnormal or overflow
case exists. Exact integers have no order to differ: the same gamma bits as
before on every host.

The work is a handful of multi-word operations on at most ~25 words, once
per fit; no loop runs over the data.
"""
from std.memory import bitcast

comptime SGX_L1 = 9
comptime SGX_L2 = 18
comptime SGX_BAD = SGX_L1 + SGX_L2
comptime _M32 = UInt64(0xFFFFFFFF)


struct _Nat(Copyable, Movable):
    """A non-negative integer: little-endian base-2^32 limbs, no zero top limb."""
    var w: List[UInt64]

    def __init__(out self):
        self.w = List[UInt64]()

    def __init__(out self, value: UInt64):
        self.w = List[UInt64]()
        if value != 0:
            self.w.append(value & _M32)
            if (value >> 32) != 0:
                self.w.append(value >> 32)

    def trim(mut self):
        while len(self.w) > 0 and self.w[len(self.w) - 1] == 0:  # small-loop(limbs: at most ~25 words)
            _ = self.w.pop()

    def is_zero(self) -> Bool:
        return len(self.w) == 0

    def bit_length(self) -> Int:
        if len(self.w) == 0:
            return 0
        var top = self.w[len(self.w) - 1]
        var bits = 0
        while top != 0:  # small-loop(bits: at most 32)
            top >>= 1
            bits += 1
        return 32 * (len(self.w) - 1) + bits

    def bit(self, i: Int) -> Bool:
        var q = i >> 5
        if q >= len(self.w):
            return False
        return ((self.w[q] >> UInt64(i & 31)) & 1) != 0


def _cmp(a: _Nat, b: _Nat) -> Int:
    if len(a.w) != len(b.w):
        return 1 if len(a.w) > len(b.w) else -1
    var i = len(a.w) - 1
    while i >= 0:  # small-loop(limbs: at most ~25 words)
        if a.w[i] != b.w[i]:
            return 1 if a.w[i] > b.w[i] else -1
        i -= 1
    return 0


def _add(a: _Nat, b: _Nat) -> _Nat:
    var out = _Nat()
    var carry = UInt64(0)
    var n = max(len(a.w), len(b.w))
    for i in range(n):  # small-loop(limbs: at most ~25 words)
        var s = carry
        if i < len(a.w):
            s += a.w[i]
        if i < len(b.w):
            s += b.w[i]
        out.w.append(s & _M32)
        carry = s >> 32
    if carry != 0:
        out.w.append(carry)
    out.trim()
    return out^


def _sub(a: _Nat, b: _Nat) -> _Nat:
    """a - b for a >= b."""
    var out = _Nat()
    var borrow = Int64(0)
    for i in range(len(a.w)):  # small-loop(limbs: at most ~25 words)
        var s = Int64(a.w[i]) - borrow
        if i < len(b.w):
            s -= Int64(b.w[i])
        if s < 0:
            s += Int64(1) << 32
            borrow = 1
        else:
            borrow = 0
        out.w.append(UInt64(s))
    out.trim()
    return out^


def _mul(a: _Nat, b: _Nat) -> _Nat:
    var out = _Nat()
    if a.is_zero() or b.is_zero():
        return out^
    for _ in range(len(a.w) + len(b.w)):  # small-loop(limbs: at most ~50 words)
        out.w.append(0)
    for i in range(len(a.w)):  # small-loop(limbs: at most ~25 words)
        var carry = UInt64(0)
        for j in range(len(b.w)):  # small-loop(limbs: at most ~25 words)
            var t = out.w[i + j] + a.w[i] * b.w[j] + carry
            out.w[i + j] = t & _M32
            carry = t >> 32
        var k = i + len(b.w)
        while carry != 0:  # small-loop(carry: at most the remaining limbs)
            var t = out.w[k] + carry
            out.w[k] = t & _M32
            carry = t >> 32
            k += 1
    out.trim()
    return out^


def _shl(a: _Nat, bits: Int) -> _Nat:
    var out = _Nat()
    if a.is_zero():
        return out^
    var q = bits >> 5
    var r = UInt64(bits & 31)
    for _ in range(q):  # small-loop(limbs: the word shift, at most ~25)
        out.w.append(0)
    var carry = UInt64(0)
    for i in range(len(a.w)):  # small-loop(limbs: at most ~25 words)
        var t = (a.w[i] << r) | carry
        out.w.append(t & _M32)
        carry = t >> 32
    if carry != 0:
        out.w.append(carry)
    out.trim()
    return out^


def _divmod(a: _Nat, b: _Nat, mut q: _Nat, mut r: _Nat) raises:
    """Binary long division: q = a // b, r = a % b (b > 0)."""
    if b.is_zero():
        raise Error("RuntimeError: scale_gamma: division by zero")
    q = _Nat()
    r = _Nat()
    var nbits = a.bit_length()
    var qw = List[UInt64]()
    for _ in range((nbits + 31) // 32):  # small-loop(limbs: the quotient words, at most ~25)
        qw.append(0)
    var i = nbits - 1
    var one = _Nat(UInt64(1))
    while i >= 0:  # small-loop(bits: at most ~760 bits of the dividend)
        r = _shl(r, 1)
        if a.bit(i):
            r = _add(r, one)
        if _cmp(r, b) >= 0:
            r = _sub(r, b)
            qw[i >> 5] = qw[i >> 5] | (UInt64(1) << UInt64(i & 31))
        i -= 1
    q.w = qw^
    q.trim()


def _from_signed_words(words: List[Int64], start: Int, count: Int, mut negative: Bool) -> _Nat:
    """|sum_i words[start + i] << (32 i)| and its sign: the positive and the
    negative words summed as two magnitudes, then subtracted."""
    var pos = _Nat()
    var neg = _Nat()
    for i in range(count):  # small-loop(limbs: the 9 or 18 fixed words)
        var v = words[start + i]
        if v >= 0:
            pos = _add(pos, _shl(_Nat(UInt64(v)), 32 * i))
        else:
            # -v as an unsigned word (no overflow for Int64.MIN: UInt64(-(v + 1)) + 1)
            var m = UInt64(-(v + 1)) + 1
            neg = _add(neg, _shl(_Nat(m), 32 * i))
    if _cmp(pos, neg) >= 0:
        negative = False
        return _sub(pos, neg)
    negative = True
    return _sub(neg, pos)


def _round_ratio(a: _Nat, b: _Nat) raises -> Float64:
    """a / b correctly rounded to binary64, half to even (a, b > 0, the
    result a normal number)."""
    var la = a.bit_length()
    var lb = b.bit_length()
    # q = floor(a * 2^k / b) has 55 or 56 bits.
    var k = 55 + lb - la
    var num = a.copy()
    var den = b.copy()
    if k >= 0:
        num = _shl(a, k)
    else:
        den = _shl(b, -k)
    var q = _Nat()
    var r = _Nat()
    _divmod(num, den, q, r)
    var qbits = q.bit_length()
    var drop = qbits - 53
    # q < 2^56 fits one UInt64.
    var qv = UInt64(0)
    for i in range(len(q.w)):  # small-loop(limbs: at most two words)
        qv |= q.w[i] << UInt64(32 * i)
    var low = qv & ((UInt64(1) << UInt64(drop)) - 1)
    var half = UInt64(1) << UInt64(drop - 1)
    var mant = qv >> UInt64(drop)
    if low > half or (low == half and (not r.is_zero() or (mant & 1) == 1)):
        mant += 1
    var exp = drop - k
    if mant == (UInt64(1) << 53):
        mant >>= 1
        exp += 1
    # value = mant * 2^exp with 2^52 <= mant < 2^53.
    var biased = exp + 52 + 1023
    if biased <= 0 or biased >= 2047:
        raise Error("RuntimeError: scale_gamma: the ratio is out of the binary64 normal range")
    return bitcast[DType.float64](UInt64(biased) << 52 | (mant - (UInt64(1) << 52)))


def scale_gamma_from_limbs(words: List[Int64], n_cells: Int, n_features: Int) raises -> Float64:
    """`_scale_gamma.scale_gamma_x` after the binding's limbs: the non-finite
    refusal, 1.0 at zero spread, else the once-rounded
    `N^2 * 2^298 / (n_features * (N * S2 - S1^2))`."""
    if words[SGX_BAD] != 0:
        raise Error("gamma='scale' needs finite input: X holds a NaN or an infinity")
    var s1_negative = False
    var s1 = _from_signed_words(words, 0, SGX_L1, s1_negative)
    var s2_negative = False
    var s2 = _from_signed_words(words, SGX_L1, SGX_L2, s2_negative)
    var n = _Nat(UInt64(n_cells))
    var s1sq = _mul(s1, s1)
    if s2_negative:
        raise Error("RuntimeError: scale_gamma: a negative sum of squares")
    var ns2 = _mul(n, s2)
    var c = _cmp(ns2, s1sq)
    if c == 0:
        return 1.0
    if c < 0:
        raise Error("RuntimeError: scale_gamma: a negative spread")
    var spread = _sub(ns2, s1sq)
    var numerator = _shl(_mul(n, n), 298)
    var denominator = _mul(_Nat(UInt64(n_features)), spread)
    return _round_ratio(numerator, denominator)
