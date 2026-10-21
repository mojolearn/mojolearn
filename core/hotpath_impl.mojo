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


# ---------------------------------------------------------------------------
# DEVIATION 3104: fold bookkeeping of model_selection
# ---------------------------------------------------------------------------


@always_inline
def _bit_test_set(bits: MutPointer[UInt64, MutUntrackedOrigin], index: Int) -> Bool:
    """Set bit `index`; True when it was already set."""
    var word = bits.unsafe_load(index >> 6)
    var mask = UInt64(1) << UInt64(index & 63)
    bits.unsafe_store(index >> 6, word | mask)
    return (word & mask) != 0


def check_indices_i64(addr: Int, count: Int, limit: Int) raises -> Int:
    """0 when the `n` int64 indices are all in `[0, bound)` and distinct, 1
    when any is out of range, else 2 for a duplicate: the order
    `model_selection._indices` tests them in."""
    if count < 1 or limit < 0:
        raise Error("check_indices_i64: n must be positive and bound non-negative")
    var p = _ptr[DType.int64](addr)
    var status = 0
    for i in range(count):
        var v = Int(p.unsafe_load(i))
        if v < 0 or v >= limit:
            status = 1
            break
    if status == 0:
        var words = (limit + 63) // 64
        var bits = List[UInt64](length=words, fill=UInt64(0))
        var bp = rebind[MutPointer[UInt64, MutUntrackedOrigin]](bits.unsafe_ptr())
        for i in range(count):
            if _bit_test_set(bp, Int(p.unsafe_load(i))):
                comptime if not HOTPATH_SABOTAGE:
                    status = 2
                    break
        _ = len(bits)
    return status


def indices_overlap_i64(a_addr: Int, na: Int, b_addr: Int, nb: Int, limit: Int) raises -> Int:
    """1 when two int64 index sets share a member, else 0. Both must already
    have passed `check_indices_i64` against `bound`."""
    if na < 1 or nb < 1 or limit < 1:
        raise Error("indices_overlap_i64: counts and bound must be positive")
    var ap = _ptr[DType.int64](a_addr)
    var bp = _ptr[DType.int64](b_addr)
    var hit = 0
    var bad = False
    var words = (limit + 63) // 64
    var bits = List[UInt64](length=words, fill=UInt64(0))
    var wp = rebind[MutPointer[UInt64, MutUntrackedOrigin]](bits.unsafe_ptr())
    for i in range(na):
        var v = Int(ap.unsafe_load(i))
        if v < 0 or v >= limit:
            bad = True
            break
        _ = _bit_test_set(wp, v)
    if not bad:
        for i in range(nb):
            var v = Int(bp.unsafe_load(i))
            if v < 0 or v >= limit:
                bad = True
                break
            var word = wp.unsafe_load(v >> 6)
            if (word & (UInt64(1) << UInt64(v & 63))) != 0:
                comptime if not HOTPATH_SABOTAGE:
                    hit = 1
                    break
    _ = len(bits)
    if bad:
        raise Error("indices_overlap_i64: index out of range")
    return hit


def fold_ids(
    codes_addr: Int, rows: Int, k: Int, splits: Int, counts_addr: Int,
    fold_addr: Int, fold_counts_addr: Int,
) raises -> Int:
    """The unshuffled default folds of `model_selection._default_folds` as one
    int32 fold id per row (DEVIATION 3104).

    `codes_addr == 0`: KFold, contiguous blocks, the first `n % n_splits`
    folds one row longer. Otherwise the stratified assignment over `n`
    int32 class codes in `[0, n_classes)`: classes are taken in FIRST-SEEN
    order, class `c` occupies `[offset, offset + count)` of the class-sorted
    label sequence, fold `f` receives `0 if first >= count else 1 + (count -
    1 - first) // n_splits` of its rows with `first = (f - offset) mod
    n_splits`, and the class's rows are dealt to folds 0, 1, ... in row
    order. `counts_addr` receives the int64 row count per class code (it is
    not read for KFold) and `fold_counts_addr` the int64 row count per fold.
    Every quantity is an integer; there is nothing to round."""
    if rows < 1 or splits < 2:
        raise Error("fold_ids: n must be positive and n_splits at least 2")
    var fp = _ptr[DType.int32](fold_addr)
    var fcp = _ptr[DType.int64](fold_counts_addr)
    if codes_addr == 0:
        var offset = 0
        for fold in range(splits):
            var size = rows // splits
            if fold < rows % splits:
                size += 1
            for i in range(offset, offset + size):
                fp.unsafe_store(i, Int32(fold))
            fcp.unsafe_store(fold, Int64(size))
            offset += size
        return 0
    if k < 1:
        raise Error("fold_ids: n_classes must be positive")
    var cp = _ptr[DType.int32](codes_addr)
    var np = _ptr[DType.int64](counts_addr)
    var bad = False
    var counts = List[Int](length=k, fill=0)
    var order = List[Int]()  # class codes in first-seen order
    for i in range(rows):
        var c = Int(cp.unsafe_load(i))
        if c < 0 or c >= k:
            bad = True
            break
        if counts[c] == 0:
            order.append(c)
        counts[c] += 1
    if not bad:
        # quota[c * splits + f]: rows of class c that fold f receives
        var quota = List[Int](length=k * splits, fill=0)
        var offset = 0
        for j in range(len(order)):
            var c = order[j]
            var count = counts[c]
            for fold in range(splits):
                var first = (fold - offset) % splits
                if first < 0:
                    first += splits
                var take = 0
                if first < count:
                    take = 1 + (count - 1 - first) // splits
                quota[c * splits + fold] = take
            offset += count
        var current = List[Int](length=k, fill=0)
        var per_fold = List[Int](length=splits, fill=0)
        for i in range(rows):
            var c = Int(cp.unsafe_load(i))
            var fold = current[c]
            while quota[c * splits + fold] == 0:
                fold += 1
            quota[c * splits + fold] -= 1
            current[c] = fold
            per_fold[fold] += 1
            fp.unsafe_store(i, Int32(fold))
        for c in range(k):
            np.unsafe_store(c, Int64(counts[c]))
        for fold in range(splits):
            fcp.unsafe_store(fold, Int64(per_fold[fold]))
    if bad:
        raise Error("fold_ids: class code out of range")
    return 0


def select_fold_i64(fold_addr: Int, rows: Int, fold: Int, test_addr: Int, train_addr: Int) raises -> Int:
    """Ascending int64 row indices with fold id `fold` into `test`, every
    other row into `train`; returns the test count. Each output must hold
    `n` entries at most (the caller sizes them from the fold's count)."""
    var want = Int32(fold)
    if rows < 1:
        raise Error("select_fold_i64: n must be positive")
    var fp = _ptr[DType.int32](fold_addr)
    var tp = _ptr[DType.int64](test_addr)
    var rp = _ptr[DType.int64](train_addr)
    var n_test = 0
    var n_train = 0
    for i in range(rows):
        var at = i
        comptime if HOTPATH_SABOTAGE:
            at = (i + 1) % rows  # the row-to-fold assignment, rotated by one
        if fp.unsafe_load(at) == want:
            tp.unsafe_store(n_test, Int64(i))
            n_test += 1
        else:
            rp.unsafe_store(n_train, Int64(i))
            n_train += 1
    return n_test


# ---------------------------------------------------------------------------
# lane cgr4-py-compute: the remaining CV-splitter index generation
# ---------------------------------------------------------------------------


def arange_i64(dst_addr: Int, start: Int, count: Int) raises -> Int:
    """`dst[i] = start + i` for i < n, int64: every contiguous row range a
    splitter hands out (TimeSeriesSplit, train_test_split, `_as_index`)."""
    if count < 0:
        raise Error("arange_i64: n must be non-negative")
    if count == 0:
        return 0
    var lo = Int64(start)
    var dp = _ptr[DType.int64](dst_addr)
    for i in range(count):
        comptime if HOTPATH_SABOTAGE:
            dp.unsafe_store(i, lo + Int64(count - 1 - i))
        else:
            dp.unsafe_store(i, lo + Int64(i))
    return 0


def leave_range_i64(rows: Int, a: Int, b: Int, train_addr: Int, test_addr: Int) raises -> Int:
    """test = rows [lo, hi), train = rows [0, lo) then [hi, n), ascending
    int64 (LeaveOneOut's split i is lo = i, hi = i + 1). Returns hi - lo."""
    if rows < 1 or a < 0 or b < a or b > rows:
        raise Error("leave_range_i64: needs 0 <= lo <= hi <= n, n >= 1")
    var tp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=test_addr)
    var rp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=train_addr)
    for i in range(a, b):
        tp.unsafe_store(i - a, Int64(i))
    var at = 0
    for i in range(rows):
        if i < a or i >= b:
            comptime if HOTPATH_SABOTAGE:
                rp.unsafe_store(at, Int64(rows - 1 - i))
            else:
                rp.unsafe_store(at, Int64(i))
            at += 1
    return b - a


def mask_from_indices_u8(idx_addr: Int, count: Int, rows: Int, mask_addr: Int) raises -> Int:
    """mask[r] = 1 for every int64 index r of the k at `idx` (mask zeroed
    first, n bytes); returns the number of distinct rows set. An index
    outside [0, n) raises before the mask is read."""
    if count < 0 or rows < 1:
        raise Error("mask_from_indices_u8: k must be non-negative and n positive")
    var mp = _ptr[DType.uint8](mask_addr)
    var bad = False
    var set = 0
    for r in range(rows):
        mp.unsafe_store(r, 0)
    if count > 0:
        var ip = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=idx_addr)
        for i in range(count):
            var r = Int(ip.unsafe_load(i))
            if r < 0 or r >= rows:
                bad = True
                break
            if mp.unsafe_load(r) == 0:
                set += 1
            mp.unsafe_store(r, 1)
    if bad:
        raise Error("mask_from_indices_u8: index out of range")
    return set


def select_mask_u8_i64(mask_addr: Int, rows: Int, ta: Int, ra: Int) raises -> Int:
    """Ascending int64 rows whose mask byte is nonzero into `test`, the rest
    into `train`; returns the test count. A null output address is legal
    when that side is empty. `n_test` = count of nonzero bytes: the caller
    sizes the outputs from `count_mask_u8`."""
    if rows < 1:
        raise Error("select_mask_u8_i64: n must be positive")
    var mp = _ptr[DType.uint8](mask_addr)
    var tp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=ta)
    var rp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=ra)
    var n_test = 0
    var n_train = 0
    for i in range(rows):
        var at = i
        comptime if HOTPATH_SABOTAGE:
            at = (i + 1) % rows
        if mp.unsafe_load(at) != 0:
            tp.unsafe_store(n_test, Int64(i))
            n_test += 1
        else:
            rp.unsafe_store(n_train, Int64(i))
            n_train += 1
    return n_test


def count_mask_u8(mask_addr: Int, rows: Int) raises -> Int:
    """The number of nonzero bytes of the n-byte mask."""
    if rows < 0:
        raise Error("count_mask_u8: n must be non-negative")
    if rows == 0:
        return 0
    var mp = _ptr[DType.uint8](mask_addr)
    var c = 0
    for i in range(rows):
        if mp.unsafe_load(i) != 0:
            c += 1
    return c


def next_combination_i64(addr: Int, k: Int, rows: Int) raises -> Int:
    """Advance the ascending p-combination of range(n) at `addr` (int64) to
    its lexicographic successor (itertools.combinations order); 1 when it
    advanced, 0 when it was the last (left unchanged)."""
    if k < 1 or k > rows:
        raise Error("next_combination_i64: needs 1 <= p <= n")
    var cp = _ptr[DType.int64](addr)
    var i = k - 1
    while i >= 0 and Int(cp.unsafe_load(i)) == i + rows - k:
        i -= 1
    if i < 0:
        return 0
    var v = cp.unsafe_load(i) + 1
    for j in range(i, k):
        cp.unsafe_store(j, v + Int64(j - i))
    return 1


@always_inline
def _ic_ftz_bits(v: Float32) -> Float32:
    """A subnormal becomes a zero of the same sign (checks/numerics `ftz`)."""
    var b = bitcast[DType.uint32](v)
    if (b & UInt32(0x7F800000)) == 0 and (b & UInt32(0x007FFFFF)) != 0:
        return bitcast[DType.float32](b & UInt32(0x80000000))
    return v


def ic_running_min_f32_host(
    llf_addr: Int, count: Int, penalty: Float64, k: Int,
    ic_addr: Int, best_ic_addr: Int, best_idx_addr: Int,
) raises -> Int:
    """lane/fam2-timeseries: `ic_running_min_f64` with the criterion in
    float32, the host column of the device order choice
    (`arima/impl/fast_order_search.mojo::order_ic_argmin_kernel`):
    ic[b] = -2 llf[b] + Float32(penalty), ONE rounding (the product is exact,
    so contraction cannot change it), then the same running first minimum.
    `ic` and `best_ic` are float32, `best_idx` int64. A NaN best criterion is
    the kernel's canonical NaN once chosen."""
    if count < 1 or k < 0:
        raise Error("ic_running_min_f32: n must be positive and order non-negative")
    var pen = Float32(penalty)
    var lp = _ptr[DType.float32](llf_addr)
    var ip = _ptr[DType.float32](ic_addr)
    var bp = _ptr[DType.float32](best_ic_addr)
    var xp = _ptr[DType.int64](best_idx_addr)
    for b in range(count):
        # lane/review-fixes: flushed as the device's
        # `ftz(fma(-2, ftz(ll), pen))` (arima fast_order_search); -2 x is
        # exact, so mul + add is the fma's word
        var v = _ic_ftz_bits(Float32(-2.0) * _ic_ftz_bits(lp.unsafe_load(b)) + pen)
        if v != v:
            v = bitcast[DType.float32](UInt32(0x7FC00000))
        ip.unsafe_store(b, v)
        if k == 0:
            bp.unsafe_store(b, v)
            xp.unsafe_store(b, 0)
        else:
            var cur = bp.unsafe_load(b)
            var take = False
            if cur == cur:
                take = (v != v) or v < cur
            comptime if HOTPATH_SABOTAGE:
                take = not take
            if take:
                bp.unsafe_store(b, v)
                xp.unsafe_store(b, Int64(k))
    return 0


def ic_running_min_f64_host(
    llf_addr: Int, count: Int, pen: Float64, k: Int,
    ic_addr: Int, best_ic_addr: Int, best_idx_addr: Int,
) raises -> Int:
    """AutoARIMA's information criterion and order choice per series (lane
    cgr4-py-compute): ic[b] = -2 llf[b] + penalty in float64 from the fit's
    float32 log-likelihood, written to `ic`; then the running argmin over
    the orders tried so far, `np.argmin`'s rule: the FIRST minimum, a NaN
    taken as the minimum (the first NaN wins). order 0 initialises."""
    if count < 1 or k < 0:
        raise Error("ic_running_min_f64: n must be positive and order non-negative")
    var lp = _ptr[DType.float32](llf_addr)
    var ip = _ptr[DType.float64](ic_addr)
    var bp = _ptr[DType.float64](best_ic_addr)
    var xp = _ptr[DType.int64](best_idx_addr)
    for b in range(count):
        var v = -2.0 * Float64(lp.unsafe_load(b)) + pen
        # lane cpu4-python: a NaN criterion is the canonical quiet NaN,
        # as the device form (core/ic_min_device.mojo, soft binary64)
        if v != v:
            v = bitcast[DType.float64](UInt64(0x7FF8000000000000))
        ip.unsafe_store(b, v)
        if k == 0:
            bp.unsafe_store(b, v)
            xp.unsafe_store(b, 0)
        else:
            var cur = bp.unsafe_load(b)
            var take = False
            if cur == cur:
                take = (v != v) or v < cur
            comptime if HOTPATH_SABOTAGE:
                take = not take
            if take:
                bp.unsafe_store(b, v)
                xp.unsafe_store(b, Int64(k))
    return 0
