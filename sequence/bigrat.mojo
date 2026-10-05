# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Arbitrary-precision integers and rationals for the learning-rate
schedules' exact route (lane py-runtime round 3; owner decision: a Mojo
big-rational type instead of Python's `fractions.Fraction`).

`Nat` is an unsigned magnitude (little-endian 32-bit limbs, no leading zero
limb; zero is the empty list). `Int_` adds a sign. `Rat` is n / d with
d > 0, NOT reduced: every consumer here only compares values or rounds them
(`rat_f32`), and a value's rounding does not depend on its representation,
so no gcd is ever needed. All of it is exact integer arithmetic: the same
answers on every host, nothing to round, nothing for a compiler to fuse.

Host scalar code (a schedule value is one number per optimizer step); the
loops walk limbs of one number."""
from std.memory import bitcast


# ------------------------------------------------------------------ Nat


@fieldwise_init
struct Nat(Copyable, Movable):
    var l: List[UInt32]

    @staticmethod
    def zero() -> Nat:
        return Nat(List[UInt32]())

    @staticmethod
    def of(v: UInt64) -> Nat:
        var l = List[UInt32]()
        var x = v
        while x != 0:
            l.append(UInt32(x & 0xFFFFFFFF))
            x >>= 32
        return Nat(l^)

    def is_zero(self) -> Bool:
        return len(self.l) == 0

    def bits(self) -> Int:
        var n = len(self.l)
        if n == 0:
            return 0
        var top = self.l[n - 1]
        var b = 0
        while top != 0:
            b += 1
            top >>= 1
        return (n - 1) * 32 + b

    def is_odd(self) -> Bool:
        return len(self.l) > 0 and (self.l[0] & 1) == 1


def _trim(mut l: List[UInt32]):
    while len(l) > 0 and l[len(l) - 1] == 0:
        _ = l.pop()


def ncmp(a: Nat, b: Nat) -> Int:
    var na = len(a.l)
    var nb = len(b.l)
    if na != nb:
        return 1 if na > nb else -1
    var i = na - 1
    while i >= 0:
        if a.l[i] != b.l[i]:
            return 1 if a.l[i] > b.l[i] else -1
        i -= 1
    return 0


def nadd(a: Nat, b: Nat) -> Nat:
    var n = max(len(a.l), len(b.l))
    var r = List[UInt32](capacity=n + 1)
    var c = UInt64(0)
    for i in range(n):
        var s = c
        if i < len(a.l):
            s += UInt64(a.l[i])
        if i < len(b.l):
            s += UInt64(b.l[i])
        r.append(UInt32(s & 0xFFFFFFFF))
        c = s >> 32
    if c != 0:
        r.append(UInt32(c))
    return Nat(r^)


def nsub(a: Nat, b: Nat) -> Nat:
    """a - b for a >= b."""
    var r = List[UInt32](capacity=len(a.l))
    var borrow = Int64(0)
    for i in range(len(a.l)):
        var s = Int64(a.l[i]) - borrow
        if i < len(b.l):
            s -= Int64(b.l[i])
        if s < 0:
            s += Int64(1) << 32
            borrow = 1
        else:
            borrow = 0
        r.append(UInt32(s))
    _trim(r)
    return Nat(r^)


def nmul(a: Nat, b: Nat) -> Nat:
    var na = len(a.l)
    var nb = len(b.l)
    if na == 0 or nb == 0:
        return Nat.zero()
    var r = List[UInt32](length=na + nb, fill=0)
    for i in range(na):
        var c = UInt64(0)
        var ai = UInt64(a.l[i])
        if ai == 0:
            continue
        for j in range(nb):
            var t = ai * UInt64(b.l[j]) + UInt64(r[i + j]) + c
            r[i + j] = UInt32(t & 0xFFFFFFFF)
            c = t >> 32
        var k = i + nb
        while c != 0:
            var t = UInt64(r[k]) + c
            r[k] = UInt32(t & 0xFFFFFFFF)
            c = t >> 32
            k += 1
    _trim(r)
    return Nat(r^)


def nshl(a: Nat, k: Int) -> Nat:
    if a.is_zero() or k == 0:
        return a.copy()
    var limbs = k // 32
    var bits = k % 32
    var r = List[UInt32](length=limbs, fill=0)
    var c = UInt32(0)
    for i in range(len(a.l)):
        var v = a.l[i]
        if bits == 0:
            r.append(v)
        else:
            r.append((v << UInt32(bits)) | c)
            c = v >> UInt32(32 - bits)
    if c != 0:
        r.append(c)
    return Nat(r^)


def nshr(a: Nat, k: Int) -> Nat:
    """floor(a / 2^k)."""
    var limbs = k // 32
    var bits = k % 32
    if limbs >= len(a.l):
        return Nat.zero()
    var r = List[UInt32](capacity=len(a.l) - limbs)
    for i in range(limbs, len(a.l)):
        var v = a.l[i] >> UInt32(bits) if bits != 0 else a.l[i]
        if bits != 0 and i + 1 < len(a.l):
            v |= a.l[i + 1] << UInt32(32 - bits)
        r.append(v)
    _trim(r)
    return Nat(r^)


def nlow_zero(a: Nat, k: Int) -> Bool:
    """Whether a's k low bits are all zero."""
    var limbs = k // 32
    var bits = k % 32
    for i in range(min(limbs, len(a.l))):
        if a.l[i] != 0:
            return False
    if bits != 0 and limbs < len(a.l):
        if (a.l[limbs] & ((UInt32(1) << UInt32(bits)) - 1)) != 0:
            return False
    return True


def npow(b: Nat, e: Int) -> Nat:
    var r = Nat.of(1)
    var base = b.copy()
    var k = e
    while k > 0:
        if (k & 1) == 1:
            r = nmul(r, base)
        k >>= 1
        if k > 0:
            base = nmul(base, base)
    return r^


def nfrom_decimal(digits: String) raises -> Nat:
    var r = Nat.zero()
    var ten = Nat.of(10)
    var bs = digits.as_bytes()
    for i in range(len(bs)):
        var c = Int(bs[i]) - 48
        if c < 0 or c > 9:
            raise Error("bigrat: not a decimal digit")
        r = nadd(nmul(r, ten), Nat.of(UInt64(c)))
    return r^


# ------------------------------------------------------------------ Int_


@fieldwise_init
struct Int_(Copyable, Movable):
    var neg: Bool
    var m: Nat

    @staticmethod
    def of(v: Int) -> Int_:
        if v < 0:
            return Int_(True, Nat.of(UInt64(-v)))
        return Int_(False, Nat.of(UInt64(v)))

    def is_zero(self) -> Bool:
        return self.m.is_zero()

    def sign(self) -> Int:
        if self.m.is_zero():
            return 0
        return -1 if self.neg else 1


def zneg(a: Int_) -> Int_:
    return Int_(not a.neg and not a.m.is_zero(), a.m.copy())


def zadd(a: Int_, b: Int_) -> Int_:
    if a.neg == b.neg:
        return Int_(a.neg, nadd(a.m, b.m))
    var c = ncmp(a.m, b.m)
    if c == 0:
        return Int_(False, Nat.zero())
    if c > 0:
        return Int_(a.neg, nsub(a.m, b.m))
    return Int_(b.neg, nsub(b.m, a.m))


def zsub(a: Int_, b: Int_) -> Int_:
    return zadd(a, zneg(b))


def zmul(a: Int_, b: Int_) -> Int_:
    var m = nmul(a.m, b.m)
    return Int_(a.neg != b.neg and not m.is_zero(), m^)


def zcmp(a: Int_, b: Int_) -> Int:
    var sa = a.sign()
    var sb = b.sign()
    if sa != sb:
        return 1 if sa > sb else -1
    if sa == 0:
        return 0
    var c = ncmp(a.m, b.m)
    return -c if sa < 0 else c


# ------------------------------------------------------------------ Rat


@fieldwise_init
struct Rat(Copyable, Movable):
    """n / d, d > 0, not reduced."""
    var n: Int_
    var d: Nat

    @staticmethod
    def of(v: Int) -> Rat:
        return Rat(Int_.of(v), Nat.of(1))

    @staticmethod
    def frac(a: Int, b: Int) -> Rat:
        """a / b, b > 0."""
        return Rat(Int_.of(a), Nat.of(UInt64(b)))

    def sign(self) -> Int:
        return self.n.sign()


def rat_of_f64(x: Float64) -> Rat:
    """The exact rational of a finite float64 (Python's Fraction(x))."""
    if x == 0.0:
        return Rat.of(0)
    var bits = bitcast[DType.uint64](x)
    var neg = (bits >> 63) != 0
    var ex = Int((bits >> 52) & 0x7FF)
    var man = bits & UInt64(0xFFFFFFFFFFFFF)
    var e: Int
    if ex == 0:
        e = -1074
    else:
        man |= UInt64(1) << 52
        e = ex - 1075
    var m = Nat.of(man)
    if e >= 0:
        return Rat(Int_(neg, nshl(m, e)), Nat.of(1))
    return Rat(Int_(neg, m^), nshl(Nat.of(1), -e))


def radd(a: Rat, b: Rat) -> Rat:
    return Rat(zadd(zmul(a.n, Int_(False, b.d.copy())), zmul(b.n, Int_(False, a.d.copy()))), nmul(a.d, b.d))


def rsub(a: Rat, b: Rat) -> Rat:
    return Rat(zsub(zmul(a.n, Int_(False, b.d.copy())), zmul(b.n, Int_(False, a.d.copy()))), nmul(a.d, b.d))


def rmul(a: Rat, b: Rat) -> Rat:
    return Rat(zmul(a.n, b.n), nmul(a.d, b.d))


def rdiv(a: Rat, b: Rat) raises -> Rat:
    if b.n.is_zero():
        raise Error("bigrat: division by zero")
    var num = zmul(a.n, Int_(False, b.d.copy()))
    if b.n.neg:
        num = zneg(num)
    return Rat(num^, nmul(a.d, b.n.m))


def rneg(a: Rat) -> Rat:
    return Rat(zneg(a.n), a.d.copy())


def rcmp(a: Rat, b: Rat) -> Int:
    return zcmp(zmul(a.n, Int_(False, b.d.copy())), zmul(b.n, Int_(False, a.d.copy())))


def rabs(a: Rat) -> Rat:
    return Rat(Int_(False, a.n.m.copy()), a.d.copy())


# ------------------------------------------------------------------ rounding

#: _training_impl._F32_MIN_NORMAL_EXP / _F32_MAX_EXP
comptime F32_MIN_NORMAL_EXP = -126
comptime F32_MAX_EXP = 127


@fieldwise_init
struct F32Round(Copyable, Movable):
    """A rounding's float32 value, or overflow (Python's OverflowError)."""
    var value: Float32
    var overflow: Bool


def _ldexp24(m: Int, e: Int, neg: Bool) -> Float32:
    """m 2^e as a float32 (m < 2^24, a normal result): exact."""
    var bits = UInt32(((e + 23) + 127) << 23) | (UInt32(m) & UInt32(0x7FFFFF))
    if neg:
        bits |= UInt32(0x80000000)
    return bitcast[DType.float32](bits)


def nat_f32(num: Nat, den: Nat, neg: Bool) -> F32Round:
    """`_training_impl._f32_round` of num / den (den > 0): the nearest
    float32, ties to even, +0.0 below the smallest normal, overflow
    flagged."""
    if num.is_zero():
        return F32Round(Float32(0.0), False)
    # scaled(e) = num / (den 2^e) in [2^23, 2^24)
    var e = num.bits() - den.bits() - 24
    var A: Nat
    var B: Nat
    if e >= 0:
        A = num.copy()
        B = nshl(den, e)
    else:
        A = nshl(num, -e)
        B = den.copy()
    var hi24 = nshl(B, 24)
    while ncmp(A, hi24) >= 0:
        e += 1
        B = nshl(B, 1)
        hi24 = nshl(B, 24)
    var lo23 = nshl(B, 23)
    while ncmp(A, lo23) < 0:
        e -= 1
        A = nshl(A, 1)
        lo23 = nshl(B, 23)
    # m = floor(A / B) < 2^24, by 24 restoring steps
    var m = 0
    var bit = 23
    while bit >= 0:
        var t = nshl(B, bit)
        if ncmp(A, t) >= 0:
            A = nsub(A, t)
            m |= 1 << bit
        bit -= 1
    # remainder A / B against one half
    var c = ncmp(nshl(A, 1), B)
    if c > 0 or (c == 0 and (m & 1) == 1):
        m += 1
    if m == (1 << 24):
        m = 1 << 23
        e += 1
    if e + 23 < F32_MIN_NORMAL_EXP:
        return F32Round(Float32(0.0), False)
    if e + 23 > F32_MAX_EXP:
        return F32Round(Float32(0.0), True)
    return F32Round(_ldexp24(m, e, neg), False)


def rat_f32(q: Rat) -> F32Round:
    return nat_f32(q.n.m, q.d, q.n.neg)

