# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""OneHotEncoder / OrdinalEncoder fit, FAST on Apple (lane/apple-fast-gap-cls2,
2026-10-03). Board (M3 FAST, taxi 1M x 5 id columns): onehot 28.0 ms vs
scikit-learn 19.0, ordinal 24.8 vs 18.8. Every switch default OFF; the
binding exports `x_prep_cls2_cat` (bit 1 PACK, bit 2 PRESENT) only when one
is on, and python/mojolearn/_expansion_prep.py `_fit_categories` reads it.

Cause (python/mojolearn/_expansion_prep.py `_fit_categories`): the
distinct values land in a HOST arena region of n*d words (`uo = pr.alloc(n
* d)`, 20 MB at taxi's shape) that the ranges runner zero-allocates on the
host and DOWNLOADS whole after the run (core/arena_io: everything but the
inputs comes back), for a few hundred distinct values; and the distinct
values come from a 4-pass radix sort of every column (x_prep/dradix.mojo).

  -D MOJOLEARN_X_PREP_FAST_CLS2_PACK: the run scan writes into DEVICE
     scratch, and `cat_pack` (op 161) copies each column's distinct words,
     column after column, into a small host region (CAP words); a total past
     CAP reruns main's program. The same words.
  -D MOJOLEARN_X_PREP_FAST_CLS2_PRESENT (needs PACK): no sort. Every cell
     whose word is an integer in [0, CAT_R) (canon: -0.0 is 0.0) marks its
     column's presence flag (`cat_present`, op 158; benign same-value
     stores); a flag scan per column (`pres_count` 159, `uniq_scan` 121,
     `pres_write` 160) writes the present integers ascending, which ARE the
     sorted distinct canonical words. A column with any other word (a
     fraction, a negative, >= CAT_R, NaN, inf) raises its BAD word and the
     fit reruns main's program. The same words.
Op 157 `cat_zero` clears the scratch the flags live in (scratch starts
undefined)."""
from std.memory import bitcast
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_prep.common import FP, IP, p, raw, st, sti, ldi

comptime CAT_CLS2_PACK = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_X_PREP_FAST_CLS2_PACK"]()
)
comptime CAT_CLS2_PRESENT = CAT_CLS2_PACK and is_defined["MOJOLEARN_X_PREP_FAST_CLS2_PRESENT"]()
#: presence flags per column (integer categories 0 .. CAT_R - 1)
comptime CAT_R = 4096


def cat_zero_unit(t: Int, f: FP, q: IP):
    """q = [OUT]; t: OUT[t] = 0 (int32 bits 0 = float +0.0)."""
    sti(f, p(q, 0) + t, 0)


def cat_present_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, R, FL, BAD]; t = i*d + c over the row-major X: the word
    an integer v in [0, R) (-0.0 counts as 0) sets FL[c*R + v] = 1, any
    other word sets BAD[c] = 1 (int32 bits; every writer stores the same)."""
    var d = p(q, 2)
    var R = p(q, 3)
    var c = t % d
    var x = raw(f, p(q, 0) + t)
    # by bits (Metal flushes compare operands): -0.0 is 0, then a word below
    # the word of float R with the sign clear, and the word of its integer
    var b = bitcast[DType.uint32](x)
    if b == UInt32(0x80000000):
        b = UInt32(0)
    var ok = False
    var v = 0
    if b < bitcast[DType.uint32](Float32(R)):
        v = Int(bitcast[DType.float32](b))
        ok = bitcast[DType.uint32](Float32(v)) == b
    if ok:
        sti(f, p(q, 4) + c * R + v, 1)
    else:
        sti(f, p(q, 5) + c, 1)


def pres_count_unit(t: Int, f: FP, q: IP):
    """q = [FL, R, CH, CNT]; t = chunk of CH flags: how many are set."""
    var lo = t * p(q, 2)
    var hi = min(lo + p(q, 2), p(q, 1))
    var k = 0
    for r in range(lo, hi):
        if ldi(f, p(q, 0) + r) != 0:
            k += 1
    sti(f, p(q, 3) + t, k)


def pres_write_unit(t: Int, f: FP, q: IP):
    """q = [FL, R, CH, OFF, U]; t = chunk: every set flag r of the chunk
    writes the float r at U[OFF[t] ..], ascending."""
    var lo = t * p(q, 2)
    var hi = min(lo + p(q, 2), p(q, 1))
    var k = ldi(f, p(q, 3) + t)
    for r in range(lo, hi):
        if ldi(f, p(q, 0) + r) != 0:
            st(f, p(q, 4) + k, Float32(r))
            k += 1


def cat_pack_unit(t: Int, f: FP, q: IP):
    """q = [U, S, d, CO, CAP, PK]; t = c*S + i: column c's distinct word i
    (U[c*S + i], i < CO[c], CO the float counts) at PK[sum(CO[:c]) + i]
    when that is below CAP, bit for bit."""
    var S = p(q, 1)
    var c = t // S
    var i = t - c * S
    var cnt = Int(f.unsafe_load(p(q, 3) + c))
    if i >= cnt:
        return
    var off = 0
    for cc in range(c):
        off += Int(f.unsafe_load(p(q, 3) + cc))
    if off + i < p(q, 4):
        f.unsafe_store(p(q, 5) + off + i, raw(f, p(q, 0) + t))
