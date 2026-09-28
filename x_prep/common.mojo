# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PREP LANE'S ONE-SOURCE PROGRAM MODEL (lane/algos-prep2, 2026-09-27).

Every prep and naive Bayes / discriminant analysis kernel is a UNIT: a plain
function of `(t, f, q)` that computes work item `t` of one stage, reading and
writing the float32 ARENA `f` at offsets carried in the stage's Int32
parameters `q`. The device binding launches one thread per unit
(x_prep/device.mojo); the host binding runs the same units in a loop
(x_prep/host/program.mojo). The arithmetic is the SAME FUNCTION on both, so
CPU == GPU bit for bit is a property of construction, not of two copies kept
in step. Every unit is written for IDENTICAL: a fixed sequential order inside
the unit, fixed tie-breaks, every operand through `ftz`, products through
`identical_mul`, divisions through `identical_div`, transcendental functions
through the portable ones, and no computed NaN reaches a hashed output (a NaN
INPUT is copied bit for bit, never passed through arithmetic).

An arena slot holding INTEGER bits (codes written for an int32 result) is
read and written with `ldi`/`sti`, never `ld`: a small integer's bits are a
subnormal float and `ftz` would zero them.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from checks.numerics import ftz

#: The host binding's negative control (MOJOLEARN_HOST_SABOTAGE, the CPU
#: gate's routed-set sabotage): every sum in x_prep/prims.mojo `add` is moved
#: by one ulp. Only the host build ever defines it.
comptime X_PREP_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

comptime FP = MutPointer[Float32, MutAnyOrigin]
comptime IP = MutPointer[Int32, MutAnyOrigin]
comptime STAGE_INTS = 16
comptime PARAMS = 14


@always_inline
def p(q: IP, k: Int) -> Int:
    return Int(q.unsafe_load(k))


@always_inline
def ld(f: FP, i: Int) -> Float32:
    """DEVIATION 5408: every float operand a unit loads is flushed."""
    return ftz(f.unsafe_load(i))


@always_inline
def raw(f: FP, i: Int) -> Float32:
    return f.unsafe_load(i)


@always_inline
def st(f: FP, i: Int, v: Float32):
    f.unsafe_store(i, ftz(v))


#: rows a serial column fold loads at once (`run_block`).
comptime RUN = 16


@always_inline
def run_block[B: Int](f: FP, base: Int, stride: Int) -> SIMD[DType.float32, B]:
    """The B raw words f[base + u * stride], u = 0 .. B-1, loaded together
    before any is used (lane prep-apple, 2026-09-28). A unit that folds a
    column in ascending row order on ONE GPU thread otherwise waits one
    memory latency per row (~350 ns per row measured on the M4 Pro); loading
    a run of rows first overlaps those latencies. The fold still takes the
    words one at a time in ascending order, so no operation moves: the same
    bits on every target."""
    var v = SIMD[DType.float32, B]()
    comptime for u in range(B):
        v[u] = f.unsafe_load(base + u * stride)
    return v


@always_inline
def ldi(f: FP, i: Int) -> Int:
    return Int(bitcast[DType.int32](f.unsafe_load(i)))


@always_inline
def sti(f: FP, i: Int, v: Int):
    f.unsafe_store(i, bitcast[DType.float32](Int32(v)))


@always_inline
def is_nan(x: Float32) -> Bool:
    return x != x


@always_inline
def canonical_nan() -> Float32:
    return bitcast[DType.float32](UInt32(0x7FC00000))


@always_inline
def canon(x: Float32) -> Float32:
    """-0.0 -> +0.0 and every NaN -> the one quiet NaN word: the category
    identity the encoders use (numpy's unique treats -0.0 == 0.0 and folds
    every NaN into one category)."""
    if x != x:
        return canonical_nan()
    if x == Float32(0):
        return Float32(0)
    return x


@always_inline
def key(x: Float32) -> UInt32:
    """A total order on float32 words: negatives below positives, -0.0 below
    +0.0, NaN last (DEVIATION 5402: every sort in the lane orders by it)."""
    if x != x:
        return UInt32(0xFFFFFFFF)
    var b = bitcast[DType.uint32](x)
    if (b & UInt32(0x80000000)) != UInt32(0):
        return ~b
    return b | UInt32(0x80000000)


@always_inline
def word_order(b: UInt32) -> UInt64:
    """The total order every sort of the lane uses, on a word's bits: `key`
    first (the same map, written on the integer word), the raw bits second.
    On a non-NaN word `key` is a bijection, so the bits only order NaN words
    of different payload among themselves; with them the order is total and a
    sort has one answer whatever algorithm reaches it (the device's bitonic
    sort, x_prep/dsort.mojo, equals this heapsort word for word)."""
    var k: UInt32
    if (b & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000):
        k = UInt32(0xFFFFFFFF)
    elif (b & UInt32(0x80000000)) != UInt32(0):
        k = ~b
    else:
        k = b | UInt32(0x80000000)
    return (UInt64(k) << UInt64(32)) | UInt64(b)


@always_inline
def _wo(f: FP, i: Int) -> UInt64:
    return word_order(f.bitcast[UInt32]().unsafe_load(i))


def heap_sort(f: FP, base: Int, n: Int):
    """In-place heapsort of f[base : base+n] by `word_order`. Deterministic by
    construction (the same comparisons in the same order everywhere)."""
    if n < 2:
        return
    var start = n // 2 - 1
    while start >= 0:
        _sift(f, base, start, n)
        start -= 1
    var end = n - 1
    while end > 0:
        var tmp = f.unsafe_load(base)
        f.unsafe_store(base, f.unsafe_load(base + end))
        f.unsafe_store(base + end, tmp)
        _sift(f, base, 0, end)
        end -= 1


@always_inline
def _sift(f: FP, base: Int, start: Int, n: Int):
    var root = start
    while True:
        var child = 2 * root + 1
        if child >= n:
            return
        if child + 1 < n and _wo(f, base + child) < _wo(f, base + child + 1):
            child += 1
        if _wo(f, base + root) < _wo(f, base + child):
            var tmp = f.unsafe_load(base + root)
            f.unsafe_store(base + root, f.unsafe_load(base + child))
            f.unsafe_store(base + child, tmp)
            root = child
        else:
            return
