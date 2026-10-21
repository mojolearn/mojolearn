# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed kernels of the shared host helpers (`bindings/hotpath_helpers.mojo`).

The Python door stays in `bindings/hotpath_helpers.mojo`: it unpacks
PythonObject arguments, calls one typed function here and converts the
result back. Everything that computes (casts, comparisons, counts, folds,
draws) lives here with no Python import, word for word as it was in the
binding (lane thin-core, 2026-10-21: a pure move, no bit change).
"""
from std.math import fma, isfinite, sqrt
from std.memory import bitcast
from std.sys.compile import is_defined
from sequence.schedule import fill_epoch_order, splitmix64
from checks.numerics import portable_cosf, portable_log64
from metrics.checks.pinned_sum import canonicalize_nan


#: THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` is the core host
#: binding's existing sabotage define, and `-D MOJOLEARN_HOTPATH_SABOTAGE=1`
#: sabotages THESE HELPERS ALONE, so a divergence under it cannot be the
#: k-NN or k-means fold's (`core_host_sabotage()` reports either and
#: `_backend` refuses to load such a binary without
#: MOJOLEARN_HOST_ALLOW_SABOTAGE=1). Under either every helper here answers
#: WRONG ON PURPOSE in a way that keeps its output well formed: the cast
#: writes each run of eight elements reversed, min and max trade places,
#: the float sum folds descending, argmax takes the LAST maximum, equality
#: is inverted, the encoder's codes are reversed, the
#: gather reads the next table slot, a duplicate index is accepted, an
#: overlap is denied and the fold ids are rotated by one.
#: `tests/test_hotpath_native.py` must FAIL against such a build in every
#: group; a differential test that passes against it compares nothing.
comptime HOTPATH_SABOTAGE = (
    is_defined["MOJOLEARN_HOST_SABOTAGE"]() or is_defined["MOJOLEARN_HOTPATH_SABOTAGE"]()
)

#: dtype codes shared with `python/mojolearn/_array.py::_NATIVE_CODE`.
comptime HP_F32 = 0
comptime HP_F64 = 1
comptime HP_I32 = 2
comptime HP_I64 = 3
comptime HP_U32 = 4
comptime HP_U8 = 5
#: lane py-runtime round 2: bf16 bits (uint16), int8 codes and float16, the
#: dtypes whose Python routines were the last ones left in `_array.py`
comptime HP_U16 = 6
comptime HP_I8 = 7
comptime HP_F16 = 8
#: equal_elements only (lane py-runtime round 3): a int64, b float64, exact
comptime HP_I64_F64 = 9

comptime HP_W = 8


@always_inline
def _ptr[dt: DType](addr: Int) raises -> MutPointer[Scalar[dt], MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=addr)


def _tasks(n: Int) -> Int:
    """One task on the calling thread (cpu-gpu-cleanup c-core: the GPU
    binding runs no host pool; the SIMD loop is a copy/compare)."""
    return 1


# ---------------------------------------------------------------------------
# DEVIATION 3100: cast_elements
# ---------------------------------------------------------------------------


@always_inline
def _refused[src: DType, dst: DType, W: Int](v: SIMD[src, W]) -> Bool:
    """True when any lane is a value the Python conversion refuses."""
    comptime if dst == DType.float16:
        # struct's 'e' packing refuses a finite value that rounds past
        # 65504 (|v| >= 65520); infinities and NaN pack
        var d = v.cast[DType.float64]()
        var big = abs(d).ge(SIMD[DType.float64, W](65520.0)) & abs(d).lt(
            SIMD[DType.float64, W](Float64.MAX))
        return big.reduce_or()
    elif dst.is_floating_point():
        return False
    elif src.is_floating_point():
        var d = v.cast[DType.float64]()
        comptime if dst == DType.int64:
            var ok = d.ge(SIMD[DType.float64, W](-9223372036854775808.0)) & d.lt(
                SIMD[DType.float64, W](9223372036854775808.0))
            return not ok.reduce_and()
        elif dst == DType.int32:
            var ok = d.gt(SIMD[DType.float64, W](-2147483649.0)) & d.lt(
                SIMD[DType.float64, W](2147483648.0))
            return not ok.reduce_and()
        elif dst == DType.uint32:
            var ok = d.gt(SIMD[DType.float64, W](-1.0)) & d.lt(
                SIMD[DType.float64, W](4294967296.0))
            return not ok.reduce_and()
        elif dst == DType.uint16:
            var ok = d.gt(SIMD[DType.float64, W](-1.0)) & d.lt(
                SIMD[DType.float64, W](65536.0))
            return not ok.reduce_and()
        elif dst == DType.int8:
            var ok = d.gt(SIMD[DType.float64, W](-129.0)) & d.lt(
                SIMD[DType.float64, W](128.0))
            return not ok.reduce_and()
        else:
            var ok = d.gt(SIMD[DType.float64, W](-1.0)) & d.lt(
                SIMD[DType.float64, W](256.0))
            return not ok.reduce_and()
    else:
        comptime if dst == DType.int64:
            return False
        else:
            var w = v.cast[DType.int64]()
            comptime if dst == DType.int32:
                var ok = w.ge(SIMD[DType.int64, W](-2147483648)) & w.le(
                    SIMD[DType.int64, W](2147483647))
                return not ok.reduce_and()
            elif dst == DType.uint32:
                var ok = w.ge(SIMD[DType.int64, W](0)) & w.le(
                    SIMD[DType.int64, W](4294967295))
                return not ok.reduce_and()
            elif dst == DType.uint16:
                var ok = w.ge(SIMD[DType.int64, W](0)) & w.le(
                    SIMD[DType.int64, W](65535))
                return not ok.reduce_and()
            elif dst == DType.int8:
                var ok = w.ge(SIMD[DType.int64, W](-128)) & w.le(
                    SIMD[DType.int64, W](127))
                return not ok.reduce_and()
            else:
                var ok = w.ge(SIMD[DType.int64, W](0)) & w.le(
                    SIMD[DType.int64, W](255))
                return not ok.reduce_and()


@always_inline
def _convert[src: DType, dst: DType, W: Int](v: SIMD[src, W]) -> SIMD[dst, W]:
    comptime if (dst == DType.float32 or dst == DType.float16) and not src.is_floating_point():
        # The C item setter's route: integer -> double -> float (struct's
        # 'e' packing of a Python int goes through the double too).
        return v.cast[DType.float64]().cast[dst]()
    elif dst == DType.float32 and src == DType.float32:
        # `array.array('f', <float32 memoryview>)` widens each element to a
        # Python float and narrows it back. That round trip is the identity
        # on every float32 EXCEPT a signaling NaN, which the widening quiets
        # (quiet bit set, sign and payload kept; measured on the M4:
        # 0x7fa00001 -> 0x7fe00001). An fpext/fptrunc pair would be folded
        # away by the optimizer, so the quieting is spelled on the bits.
        var bits = bitcast[DType.uint32, W](rebind[SIMD[DType.float32, W]](v))
        var nan = (bits & SIMD[DType.uint32, W](0x7fffffff)).gt(SIMD[DType.uint32, W](0x7f800000))
        var quiet = nan.select(bits | SIMD[DType.uint32, W](0x00400000), bits)
        return rebind[SIMD[dst, W]](bitcast[DType.float32, W](quiet))
    else:
        return v.cast[dst]()


def _cast_run[src: DType, dst: DType](src_addr: Int, dst_addr: Int, n: Int) raises -> Int:
    var sp = _ptr[src](src_addr)
    var dp = _ptr[dst](dst_addr)
    var tasks = _tasks(n)
    var chunk = n
    var flags = List[Int](length=tasks, fill=0)
    var fp = flags.unsafe_ptr()

    def _range(c: Int) {imm sp, imm dp, imm n, imm chunk, imm fp}:
        var i = c * chunk
        var hi = min(i + chunk, n)
        var bad = False
        while i + HP_W <= hi:
            var v = sp.unsafe_load[width=HP_W](i)
            if _refused[src, dst, HP_W](v):
                bad = True
                break
            comptime if HOTPATH_SABOTAGE:
                dp.unsafe_store[width=HP_W](i, _convert[src, dst, HP_W](v.reversed()))
            else:
                dp.unsafe_store[width=HP_W](i, _convert[src, dst, HP_W](v))
            i += HP_W
        while i < hi and not bad:
            var s = SIMD[src, 1](sp.unsafe_load(i))
            if _refused[src, dst, 1](s):
                bad = True
                break
            dp.unsafe_store(i, _convert[src, dst, 1](s)[0])
            i += 1
        if bad:
            fp.unsafe_store(c, 1)

    _range(0)
    _ = len(flags)
    for c in range(tasks):
        if flags[c] != 0:
            return 1
    return 0


def _cast_from[src: DType](src_addr: Int, dst_code: Int, dst_addr: Int, n: Int) raises -> Int:
    if dst_code == HP_F32:
        return _cast_run[src, DType.float32](src_addr, dst_addr, n)
    if dst_code == HP_F64:
        return _cast_run[src, DType.float64](src_addr, dst_addr, n)
    if dst_code == HP_I32:
        return _cast_run[src, DType.int32](src_addr, dst_addr, n)
    if dst_code == HP_I64:
        return _cast_run[src, DType.int64](src_addr, dst_addr, n)
    if dst_code == HP_U32:
        return _cast_run[src, DType.uint32](src_addr, dst_addr, n)
    if dst_code == HP_U8:
        return _cast_run[src, DType.uint8](src_addr, dst_addr, n)
    if dst_code == HP_U16:
        return _cast_run[src, DType.uint16](src_addr, dst_addr, n)
    if dst_code == HP_I8:
        return _cast_run[src, DType.int8](src_addr, dst_addr, n)
    if dst_code == HP_F16:
        return _cast_run[src, DType.float16](src_addr, dst_addr, n)
    raise Error("cast_elements: unknown destination dtype code " + String(dst_code))


def cast_elements(sa: Int, sc: Int, da: Int, dc: Int, count: Int) raises -> Int:
    """`dst[i] = <dst dtype>(src[i])` for `i` in `[0, n)`, the conversion
    `array.array(code, memoryview)` performs (DEVIATION 3100). Returns 0, or
    1 when some element is one the Python conversion refuses; `dst` is then
    unspecified and the caller reruns the Python path for its refusal."""
    if count < 0:
        raise Error("cast_elements: n must be non-negative, got " + String(count))
    if count == 0:
        return 0
    var status: Int
    if sc == HP_F32:
        status = _cast_from[DType.float32](sa, dc, da, count)
    elif sc == HP_F64:
        status = _cast_from[DType.float64](sa, dc, da, count)
    elif sc == HP_I32:
        status = _cast_from[DType.int32](sa, dc, da, count)
    elif sc == HP_I64:
        status = _cast_from[DType.int64](sa, dc, da, count)
    elif sc == HP_U32:
        status = _cast_from[DType.uint32](sa, dc, da, count)
    elif sc == HP_U8:
        status = _cast_from[DType.uint8](sa, dc, da, count)
    elif sc == HP_U16:
        status = _cast_from[DType.uint16](sa, dc, da, count)
    elif sc == HP_I8:
        status = _cast_from[DType.int8](sa, dc, da, count)
    elif sc == HP_F16:
        status = _cast_from[DType.float16](sa, dc, da, count)
    else:
        raise Error("cast_elements: unknown source dtype code " + String(sc))
    return status


# ---------------------------------------------------------------------------
# DEVIATION 3101: reduce_stat
# ---------------------------------------------------------------------------

comptime HP_MIN = 0
comptime HP_MAX = 1
comptime HP_SUM = 2
comptime HP_ARGMAX = 3
comptime HP_INTEGRAL = 4
#: lane apple-fast-py2mojo-core: Python's exact integer `sum`.
comptime HP_ISUM = 5

#: The Python type of a `reduce_stat` answer: an int, a float, or the
#: (hi, mid, low) tuple of the exact integer sum.
comptime HP_STAT_INT = 0
comptime HP_STAT_FLOAT = 1
comptime HP_STAT_TUPLE = 2


@fieldwise_init
struct HpStat(Copyable, Movable):
    """A `reduce_stat` answer, typed: `kind` says which Python object the
    shell builds (`i` for an int, `f` for a float, (`hi`, `mid`, `low`) for
    the exact integer sum)."""
    var kind: Int
    var i: Int
    var f: Float64
    var hi: Int
    var mid: Int
    var low: Int


def _isum[dt: DType](addr: Int, n: Int) raises -> HpStat:
    """Python's exact `sum` of `n` integers: a 128-bit two's complement
    accumulator (lo unsigned, hi signed), so no sum of fewer than 2**63
    int64 values can overflow it, in any order. Returns the tuple (hi,
    lo >> 32, lo & 0xFFFFFFFF); the caller's total is
    (hi << 64) + (mid << 32) + low. Under the sabotage define the total is
    one too large."""
    var p = _ptr[dt](addr)
    var lo = UInt64(0)
    var hi = Int64(0)
    for i in range(n):
        var v = p.unsafe_load(i).cast[DType.int64]()
        var nlo = lo + bitcast[DType.uint64](v)
        if nlo < lo:
            hi += 1
        if v < 0:
            hi -= 1
        lo = nlo
    comptime if HOTPATH_SABOTAGE:
        var bumped = lo + UInt64(1)
        if bumped < lo:
            hi += 1
        lo = bumped
    return HpStat(HP_STAT_TUPLE, 0, 0.0, Int(hi), Int(lo >> 32), Int(lo & UInt64(0xFFFFFFFF)))


def _reduce[dt: DType](addr: Int, n: Int, what: Int) raises -> HpStat:
    var p = _ptr[dt](addr)
    var best = p.unsafe_load(0)
    var at = 0
    var total = Float64(0.0)
    var integral = 1
    var want_min = what == HP_MIN
    comptime if HOTPATH_SABOTAGE:
        want_min = what == HP_MAX
    if want_min:
        for i in range(1, n):
            var v = p.unsafe_load(i)
            if v < best:
                best = v
    elif what == HP_MIN or what == HP_MAX:
        for i in range(1, n):
            var v = p.unsafe_load(i)
            if v > best:
                best = v
    elif what == HP_ARGMAX:
        for i in range(1, n):
            var v = p.unsafe_load(i)
            comptime if HOTPATH_SABOTAGE:
                if v >= best:
                    best = v
                    at = i
                continue
            if v > best:
                best = v
                at = i
    elif what == HP_SUM:
        for i in range(n):
            comptime if HOTPATH_SABOTAGE:
                total = total + p.unsafe_load(n - 1 - i).cast[DType.float64]()
            else:
                total = total + p.unsafe_load(i).cast[DType.float64]()
    else:
        comptime if dt.is_floating_point():
            for i in range(n):
                var d = p.unsafe_load(i).cast[DType.float64]()
                # finite and equal to its own truncation; every float at
                # or beyond 2**53 in magnitude is an integer already
                if not isfinite(d):
                    integral = 0
                    break
                if abs(d) < 9007199254740992.0 and Float64(Int(d)) != d:
                    integral = 0
                    break
    if what == HP_ARGMAX:
        return HpStat(HP_STAT_INT, at, 0.0, 0, 0, 0)
    if what == HP_SUM:
        return HpStat(HP_STAT_FLOAT, 0, total, 0, 0, 0)
    if what == HP_INTEGRAL:
        return HpStat(HP_STAT_INT, integral, 0.0, 0, 0, 0)
    comptime if dt.is_floating_point():
        return HpStat(HP_STAT_FLOAT, 0, best.cast[DType.float64](), 0, 0, 0)
    else:
        return HpStat(HP_STAT_INT, Int(best), 0.0, 0, 0, 0)


def reduce_stat(a: Int, c: Int, count: Int, w: Int) raises -> HpStat:
    """Python's `min`, `max`, left-to-right float `sum` or first-max-wins
    argmax over `n >= 1` elements in storage order (DEVIATION 3101), or
    (what = 4) whether every float is finite and integer valued, or (what =
    5, lane apple-fast-py2mojo-core) the exact integer sum as (hi, mid, low)."""
    if count < 1:
        raise Error("reduce_stat: n must be positive, got " + String(count))
    if w < HP_MIN or w > HP_ISUM:
        raise Error("reduce_stat: unknown reduction " + String(w))
    if w == HP_ISUM:
        if c == HP_I32:
            return _isum[DType.int32](a, count)
        if c == HP_I64:
            return _isum[DType.int64](a, count)
        if c == HP_U32:
            return _isum[DType.uint32](a, count)
        if c == HP_U8:
            return _isum[DType.uint8](a, count)
        if c == HP_U16:
            return _isum[DType.uint16](a, count)
        if c == HP_I8:
            return _isum[DType.int8](a, count)
        raise Error("reduce_stat: the integer sum takes an integer buffer")
    if w == HP_SUM and c != HP_F32 and c != HP_F64 and c != HP_F16:
        raise Error("reduce_stat: the float sum takes a float buffer")
    if c == HP_F32:
        return _reduce[DType.float32](a, count, w)
    if c == HP_F64:
        return _reduce[DType.float64](a, count, w)
    if c == HP_I32:
        return _reduce[DType.int32](a, count, w)
    if c == HP_I64:
        return _reduce[DType.int64](a, count, w)
    if c == HP_U32:
        return _reduce[DType.uint32](a, count, w)
    if c == HP_U8:
        return _reduce[DType.uint8](a, count, w)
    if c == HP_U16:
        return _reduce[DType.uint16](a, count, w)
    if c == HP_I8:
        return _reduce[DType.int8](a, count, w)
    if c == HP_F16:
        return _reduce[DType.float16](a, count, w)
    raise Error("reduce_stat: unknown dtype code " + String(c))


# ---------------------------------------------------------------------------
# DEVIATION 3102: equal_elements
# ---------------------------------------------------------------------------


def _equal[dt: DType](a_addr: Int, b_addr: Int, n: Int, dst_addr: Int) raises:
    var ap = _ptr[dt](a_addr)
    var bp = _ptr[dt](b_addr)
    var dp = _ptr[DType.uint8](dst_addr)
    var tasks = _tasks(n)
    var chunk = n

    def _range(c: Int) {imm ap, imm bp, imm dp, imm n, imm chunk}:
        var i = c * chunk
        var hi = min(i + chunk, n)
        while i + HP_W <= hi:
            var eq = ap.unsafe_load[width=HP_W](i).eq(bp.unsafe_load[width=HP_W](i))
            comptime if HOTPATH_SABOTAGE:
                eq = ~eq
            dp.unsafe_store[width=HP_W](i, eq.cast[DType.uint8]())
            i += HP_W
        while i < hi:
            var one = UInt8(1) if ap.unsafe_load(i) == bp.unsafe_load(i) else UInt8(0)
            dp.unsafe_store(i, one)
            i += 1

    _range(0)


def _equal_i64_f64(a_addr: Int, b_addr: Int, n: Int, dst_addr: Int) raises:
    """Python's exact `int == float` per element (lane py-runtime round 3):
    equal iff the float is finite, integral, inside int64 and its integer is
    the int64 (no rounding of either side, so 2**53 + 1 != 2.0**53)."""
    var ap = _ptr[DType.int64](a_addr)
    var bp = _ptr[DType.float64](b_addr)
    var dp = _ptr[DType.uint8](dst_addr)
    var i = 0
    while i < n:
        var f = bp.unsafe_load(i)
        var one = UInt8(0)
        if f == f and f >= -9223372036854775808.0 and f < 9223372036854775808.0:
            var k = Int64(f)
            if Float64(k) == f and k == ap.unsafe_load(i):
                one = UInt8(1)
        dp.unsafe_store(i, one)
        i += 1


def equal_elements(a: Int, b: Int, c: Int, count: Int, d: Int) raises -> Int:
    """`dst[i] = 1 if a[i] == b[i] else 0` over two buffers of ONE dtype
    (DEVIATION 3102): IEEE equality for floats, which is Python's (NaN is
    unequal to itself, -0.0 equals 0.0)."""
    if count < 0:
        raise Error("equal_elements: n must be non-negative")
    if count == 0:
        return 0
    if c == HP_F32:
        _equal[DType.float32](a, b, count, d)
    elif c == HP_F64:
        _equal[DType.float64](a, b, count, d)
    elif c == HP_I32:
        _equal[DType.int32](a, b, count, d)
    elif c == HP_I64:
        _equal[DType.int64](a, b, count, d)
    elif c == HP_U32:
        _equal[DType.uint32](a, b, count, d)
    elif c == HP_U8:
        _equal[DType.uint8](a, b, count, d)
    elif c == HP_U16:
        _equal[DType.uint16](a, b, count, d)
    elif c == HP_I8:
        _equal[DType.int8](a, b, count, d)
    elif c == HP_F16:
        _equal[DType.float16](a, b, count, d)
    elif c == HP_I64_F64:
        _equal_i64_f64(a, b, count, d)
    else:
        raise Error("equal_elements: unknown dtype code " + String(c))
    return 0


# ---------------------------------------------------------------------------
# DEVIATION 3103: the ORDER RULE's encoder for the core host binding, and
# gather_i32
# ---------------------------------------------------------------------------


def _encode_labels_host[dt: DType](
    src: MutPointer[Scalar[dt], MutUntrackedOrigin], n: Int,
    classes: MutPointer[Scalar[dt], MutUntrackedOrigin], max_classes: Int,
    codes: MutPointer[Int32, MutUntrackedOrigin],
) -> Int:
    """`bindings/_mojolearn.mojo::_encode_labels`, DEVIATION 2500, verbatim:
    n_classes, or -1 when more than `max_classes` distinct values were seen,
    or -2 when a label compared unequal to itself (NaN)."""
    var k = 0
    var last = src.unsafe_load(0)
    var have_last = False
    for i in range(n):
        var v = src.unsafe_load(i)
        if v != v:
            return -2
        if have_last and v == last:
            continue
        var lo = 0
        var hi = k
        while lo < hi:
            var mid = (lo + hi) // 2
            if classes.unsafe_load(mid) < v:
                lo = mid + 1
            else:
                hi = mid
        last = v
        have_last = True
        if lo < k and classes.unsafe_load(lo) == v:
            continue
        if k == max_classes:
            return -1
        var j = k
        while j > lo:
            classes.unsafe_store(j, classes.unsafe_load(j - 1))
            j -= 1
        classes.unsafe_store(lo, v)
        k += 1
    var last_code = -1
    for i in range(n):
        var v = src.unsafe_load(i)
        if last_code >= 0 and v == last:
            codes.unsafe_store(i, Int32(last_code))
            continue
        var lo = 0
        var hi = k
        while lo < hi:
            var mid = (lo + hi) // 2
            if classes.unsafe_load(mid) < v:
                lo = mid + 1
            else:
                hi = mid
        codes.unsafe_store(i, Int32(lo))
        last = v
        last_code = lo
    comptime if HOTPATH_SABOTAGE:
        for i in range(n):
            codes.unsafe_store(i, Int32(k - 1) - codes.unsafe_load(i))
    return k


def encode_labels[dt: DType](
    src_addr: Int, count: Int, classes_addr: Int, cap: Int, codes_addr: Int,
) raises -> Int:
    """The ORDER RULE's encoder over typed addresses: n_classes, or -1 when
    more than `cap` distinct values were seen; a NaN label raises."""
    if count < 1:
        raise Error("encode_labels: n must be positive, got " + String(count))
    if cap < 1:
        raise Error("encode_labels: max_classes must be positive")
    var sp = _ptr[dt](src_addr)
    var cp = _ptr[dt](classes_addr)
    var dp = _ptr[DType.int32](codes_addr)
    var k = _encode_labels_host[dt](sp, count, cp, cap, dp)
    if k == -2:
        raise Error("mojolearn: y contains a NaN label; NaN is not a class")
    return k


def gather_i32(table_addr: Int, nt: Int, codes_addr: Int, count: Int, dst_addr: Int) raises -> Int:
    """`dst[i] = table[codes[i]]`, int32 codes into an int32 table
    (DEVIATION 3103): the remap of dense label codes onto a caller's label
    order, where the table may hold -1 for "not a requested label". A code
    outside `[0, n_table)` raises before any write."""
    if count < 0 or nt < 1:
        raise Error("gather_i32: n must be non-negative and the table non-empty")
    if count == 0:
        return 0
    var tp = _ptr[DType.int32](table_addr)
    var cp = _ptr[DType.int32](codes_addr)
    var dp = _ptr[DType.int32](dst_addr)
    var bad = False
    for i in range(count):
        var c = Int(cp.unsafe_load(i))
        if c < 0 or c >= nt:
            bad = True
            break
    if not bad:
        var tasks = _tasks(count)
        var chunk = count

        def _range(t: Int) {imm tp, imm cp, imm dp, imm count, imm chunk, imm nt}:
            var lo = t * chunk
            var hi = min(lo + chunk, count)
            for i in range(lo, hi):
                comptime if HOTPATH_SABOTAGE:
                    dp.unsafe_store(i, tp.unsafe_load((Int(cp.unsafe_load(i)) + 1) % nt))
                else:
                    dp.unsafe_store(i, tp.unsafe_load(Int(cp.unsafe_load(i))))

        _range(0)
    if bad:
        raise Error("gather_i32: code out of range")
    return 0
