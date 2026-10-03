# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host helpers that replace per-row Python in the NumPy-free layer
(lane/python-hotpath, 2026-09-17, DEVIATIONS 3100-3107).

WHY THIS FILE EXISTS. The Python surface walks rows in the interpreter in a
handful of places that are invisible at 10,000 rows and cost hundreds of
milliseconds to seconds at 1,000,000: `Array.astype` and every `as_*_c`
conversion from an integer buffer (`array.array(code, memoryview)`, one
Python object per element: 20 to 33 ns each on an M4, 43 to 72 ns on an
EPYC 7713 pod), `Array.min/max/sum/argmax/
__eq__` (a `tolist()` first), the metrics label preparation (a dict lookup
per row), and the fold bookkeeping of `cross_val_score` (a `set` of every
index per fold). Every helper here is the SAME function of the same bytes
as the Python it stands in for; the Python stays in the package as the
definition and as the fallback, and `python/mojolearn/tests/
test_hotpath_native.py` holds each pair to the same result or the same
refusal on randomized and awkward inputs.

NO HELPER HERE IS A NUMERIC SEAM OF AN ESTIMATOR. They cast, compare, count
and copy. The two that touch floating point say below why no bit moves:

  * DEVIATION 3100 `cast_elements`: one conversion per element, the one
    `array.array`'s C item setter performs. int -> float32 goes THROUGH
    float64 exactly as the setter does (`PyFloat_AsDouble` then `(float)`),
    so an int64 beyond 2**53 rounds twice here too (DEVIATION 2306). Any
    element the Python path would REFUSE (an integer outside the target, a
    NaN or infinity headed for an integer) makes the helper return 1 and
    the caller reruns the Python path, which raises its own words.
  * DEVIATION 3101 `reduce_stat`: min, max, float sum and argmax are
    SEQUENTIAL in storage order on one thread, because Python's are and
    because order is observable: `min([nan, 1.0])` is nan and
    `min([1.0, nan])` is 1.0, `min([0.0, -0.0])` is 0.0 and the other
    order is -0.0, and a float64 sum rounds per addition. Same comparisons
    (`<` for min, `>` for max and argmax), same order, same answer.

The per-element helpers (`cast_elements`, `equal_elements`, `gather_i32`)
run one SIMD range on the calling thread (cpu-gpu-cleanup c-core: the host
pool is not used from the GPU binding); no element depends on another.
"""
from std.math import isfinite, sqrt
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined
from sequence.schedule import fill_epoch_order, splitmix64
from checks.numerics import portable_cosf, portable_log64



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
    comptime if dst.is_floating_point():
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
            else:
                var ok = w.ge(SIMD[DType.int64, W](0)) & w.le(
                    SIMD[DType.int64, W](255))
                return not ok.reduce_and()


@always_inline
def _convert[src: DType, dst: DType, W: Int](v: SIMD[src, W]) -> SIMD[dst, W]:
    comptime if dst == DType.float32 and not src.is_floating_point():
        # The C item setter's route: integer -> double -> float.
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

    with GILReleased(Python()):
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
    raise Error("cast_elements: unknown destination dtype code " + String(dst_code))


def cast_elements_binding(
    src_addr: PythonObject, src_code: PythonObject, dst_addr: PythonObject,
    dst_code: PythonObject, n: PythonObject,
) raises -> PythonObject:
    """`dst[i] = <dst dtype>(src[i])` for `i` in `[0, n)`, the conversion
    `array.array(code, memoryview)` performs (DEVIATION 3100). Returns 0, or
    1 when some element is one the Python conversion refuses; `dst` is then
    unspecified and the caller reruns the Python path for its refusal."""
    var count = Int(py=n)
    if count < 0:
        raise Error("cast_elements: n must be non-negative, got " + String(count))
    if count == 0:
        return PythonObject(0)
    var sa = Int(py=src_addr)
    var da = Int(py=dst_addr)
    var sc = Int(py=src_code)
    var dc = Int(py=dst_code)
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
    else:
        raise Error("cast_elements: unknown source dtype code " + String(sc))
    return PythonObject(status)


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


def _isum[dt: DType](addr: Int, n: Int) raises -> PythonObject:
    """Python's exact `sum` of `n` integers: a 128-bit two's complement
    accumulator (lo unsigned, hi signed), so no sum of fewer than 2**63
    int64 values can overflow it, in any order. Returns the tuple (hi,
    lo >> 32, lo & 0xFFFFFFFF); the caller's total is
    (hi << 64) + (mid << 32) + low. Under the sabotage define the total is
    one too large."""
    var p = _ptr[dt](addr)
    var lo = UInt64(0)
    var hi = Int64(0)
    with GILReleased(Python()):
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
    return Python.tuple(
        PythonObject(Int(hi)), PythonObject(Int(lo >> 32)), PythonObject(Int(lo & UInt64(0xFFFFFFFF)))
    )


def _reduce[dt: DType](addr: Int, n: Int, what: Int) raises -> PythonObject:
    var p = _ptr[dt](addr)
    var best = p.unsafe_load(0)
    var at = 0
    var total = Float64(0.0)
    var integral = 1
    with GILReleased(Python()):
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
        return PythonObject(at)
    if what == HP_SUM:
        return PythonObject(total)
    if what == HP_INTEGRAL:
        return PythonObject(integral)
    comptime if dt.is_floating_point():
        return PythonObject(best.cast[DType.float64]())
    else:
        return PythonObject(Int(best))


def reduce_stat_binding(
    addr: PythonObject, code: PythonObject, n: PythonObject, what: PythonObject,
) raises -> PythonObject:
    """Python's `min`, `max`, left-to-right float `sum` or first-max-wins
    argmax over `n >= 1` elements in storage order (DEVIATION 3101), or
    (what = 4) whether every float is finite and integer valued, or (what =
    5, lane apple-fast-py2mojo-core) the exact integer sum as (hi, mid, low)."""
    var count = Int(py=n)
    if count < 1:
        raise Error("reduce_stat: n must be positive, got " + String(count))
    var w = Int(py=what)
    if w < HP_MIN or w > HP_ISUM:
        raise Error("reduce_stat: unknown reduction " + String(w))
    var c = Int(py=code)
    var a = Int(py=addr)
    if w == HP_ISUM:
        if c == HP_I32:
            return _isum[DType.int32](a, count)
        if c == HP_I64:
            return _isum[DType.int64](a, count)
        if c == HP_U32:
            return _isum[DType.uint32](a, count)
        if c == HP_U8:
            return _isum[DType.uint8](a, count)
        raise Error("reduce_stat: the integer sum takes an integer buffer")
    if w == HP_SUM and c != HP_F32 and c != HP_F64:
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

    with GILReleased(Python()):
        _range(0)


def equal_elements_binding(
    a_addr: PythonObject, b_addr: PythonObject, code: PythonObject,
    n: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """`dst[i] = 1 if a[i] == b[i] else 0` over two buffers of ONE dtype
    (DEVIATION 3102): IEEE equality for floats, which is Python's (NaN is
    unequal to itself, -0.0 equals 0.0)."""
    var count = Int(py=n)
    if count < 0:
        raise Error("equal_elements: n must be non-negative")
    if count == 0:
        return PythonObject(0)
    var c = Int(py=code)
    var a = Int(py=a_addr)
    var b = Int(py=b_addr)
    var d = Int(py=dst_addr)
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
    else:
        raise Error("equal_elements: unknown dtype code " + String(c))
    return PythonObject(0)


# ---------------------------------------------------------------------------
# DEVIATION 3103: the ORDER RULE's encoder for the core host binding, and
# gather_i32
# ---------------------------------------------------------------------------


def _encode_labels[dt: DType](
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


def _encode_labels_binding[dt: DType](
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    var count = Int(py=n)
    var cap = Int(py=max_classes)
    if count < 1:
        raise Error("encode_labels: n must be positive, got " + String(count))
    if cap < 1:
        raise Error("encode_labels: max_classes must be positive")
    var sp = _ptr[dt](Int(py=src_addr))
    var cp = _ptr[dt](Int(py=classes_addr))
    var dp = _ptr[DType.int32](Int(py=codes_addr))
    var k: Int
    with GILReleased(Python()):
        k = _encode_labels[dt](sp, count, cp, cap, dp)
    if k == -2:
        raise Error("mojolearn: y contains a NaN label; NaN is not a class")
    return PythonObject(k)


def encode_labels_f32_binding(
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    return _encode_labels_binding[DType.float32](src_addr, n, classes_addr, max_classes, codes_addr)


def encode_labels_f64_binding(
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    return _encode_labels_binding[DType.float64](src_addr, n, classes_addr, max_classes, codes_addr)


def encode_labels_i32_binding(
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    return _encode_labels_binding[DType.int32](src_addr, n, classes_addr, max_classes, codes_addr)


def encode_labels_i64_binding(
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    return _encode_labels_binding[DType.int64](src_addr, n, classes_addr, max_classes, codes_addr)


def encode_labels_u32_binding(
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    return _encode_labels_binding[DType.uint32](src_addr, n, classes_addr, max_classes, codes_addr)


def encode_labels_u8_binding(
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    return _encode_labels_binding[DType.uint8](src_addr, n, classes_addr, max_classes, codes_addr)


def gather_i32_binding(
    table_addr: PythonObject, n_table: PythonObject, codes_addr: PythonObject,
    n: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """`dst[i] = table[codes[i]]`, int32 codes into an int32 table
    (DEVIATION 3103): the remap of dense label codes onto a caller's label
    order, where the table may hold -1 for "not a requested label". A code
    outside `[0, n_table)` raises before any write."""
    var count = Int(py=n)
    var nt = Int(py=n_table)
    if count < 0 or nt < 1:
        raise Error("gather_i32: n must be non-negative and the table non-empty")
    if count == 0:
        return PythonObject(0)
    var tp = _ptr[DType.int32](Int(py=table_addr))
    var cp = _ptr[DType.int32](Int(py=codes_addr))
    var dp = _ptr[DType.int32](Int(py=dst_addr))
    var bad = False
    with GILReleased(Python()):
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
    return PythonObject(0)


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


def check_indices_i64_binding(
    addr: PythonObject, n: PythonObject, bound: PythonObject,
) raises -> PythonObject:
    """0 when the `n` int64 indices are all in `[0, bound)` and distinct, 1
    when any is out of range, else 2 for a duplicate: the order
    `model_selection._indices` tests them in."""
    var count = Int(py=n)
    var limit = Int(py=bound)
    if count < 1 or limit < 0:
        raise Error("check_indices_i64: n must be positive and bound non-negative")
    var p = _ptr[DType.int64](Int(py=addr))
    var status = 0
    with GILReleased(Python()):
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
    return PythonObject(status)


def indices_overlap_i64_binding(
    a_addr: PythonObject, n_a: PythonObject, b_addr: PythonObject,
    n_b: PythonObject, bound: PythonObject,
) raises -> PythonObject:
    """1 when two int64 index sets share a member, else 0. Both must already
    have passed `check_indices_i64` against `bound`."""
    var na = Int(py=n_a)
    var nb = Int(py=n_b)
    var limit = Int(py=bound)
    if na < 1 or nb < 1 or limit < 1:
        raise Error("indices_overlap_i64: counts and bound must be positive")
    var ap = _ptr[DType.int64](Int(py=a_addr))
    var bp = _ptr[DType.int64](Int(py=b_addr))
    var hit = 0
    var bad = False
    with GILReleased(Python()):
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
    return PythonObject(hit)


def fold_ids_binding(
    codes_addr: PythonObject, n: PythonObject, n_classes: PythonObject,
    n_splits: PythonObject, counts_addr: PythonObject, fold_addr: PythonObject,
    fold_counts_addr: PythonObject,
) raises -> PythonObject:
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
    var rows = Int(py=n)
    var k = Int(py=n_classes)
    var splits = Int(py=n_splits)
    if rows < 1 or splits < 2:
        raise Error("fold_ids: n must be positive and n_splits at least 2")
    var fp = _ptr[DType.int32](Int(py=fold_addr))
    var fcp = _ptr[DType.int64](Int(py=fold_counts_addr))
    if Int(py=codes_addr) == 0:
        with GILReleased(Python()):
            var offset = 0
            for fold in range(splits):
                var size = rows // splits
                if fold < rows % splits:
                    size += 1
                for i in range(offset, offset + size):
                    fp.unsafe_store(i, Int32(fold))
                fcp.unsafe_store(fold, Int64(size))
                offset += size
        return PythonObject(0)
    if k < 1:
        raise Error("fold_ids: n_classes must be positive")
    var cp = _ptr[DType.int32](Int(py=codes_addr))
    var np = _ptr[DType.int64](Int(py=counts_addr))
    var bad = False
    with GILReleased(Python()):
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
    return PythonObject(0)


def select_fold_i64_binding(
    fold_addr: PythonObject, n: PythonObject, fold: PythonObject,
    test_addr: PythonObject, train_addr: PythonObject,
) raises -> PythonObject:
    """Ascending int64 row indices with fold id `fold` into `test`, every
    other row into `train`; returns the test count. Each output must hold
    `n` entries at most (the caller sizes them from the fold's count)."""
    var rows = Int(py=n)
    var want = Int32(Int(py=fold))
    if rows < 1:
        raise Error("select_fold_i64: n must be positive")
    var fp = _ptr[DType.int32](Int(py=fold_addr))
    var tp = _ptr[DType.int64](Int(py=test_addr))
    var rp = _ptr[DType.int64](Int(py=train_addr))
    var n_test = 0
    with GILReleased(Python()):
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
    return PythonObject(n_test)


# ---------------------------------------------------------------------------
# lane cgr4-py-compute: the remaining CV-splitter index generation
# ---------------------------------------------------------------------------


def arange_i64_binding(dst_addr: PythonObject, start: PythonObject, n: PythonObject) raises -> PythonObject:
    """`dst[i] = start + i` for i < n, int64: every contiguous row range a
    splitter hands out (TimeSeriesSplit, train_test_split, `_as_index`)."""
    var count = Int(py=n)
    if count < 0:
        raise Error("arange_i64: n must be non-negative")
    if count == 0:
        return PythonObject(0)
    var lo = Int64(Int(py=start))
    var dp = _ptr[DType.int64](Int(py=dst_addr))
    with GILReleased(Python()):
        for i in range(count):
            comptime if HOTPATH_SABOTAGE:
                dp.unsafe_store(i, lo + Int64(count - 1 - i))
            else:
                dp.unsafe_store(i, lo + Int64(i))
    return PythonObject(0)


def leave_range_i64_binding(
    n: PythonObject, lo: PythonObject, hi: PythonObject,
    train_addr: PythonObject, test_addr: PythonObject,
) raises -> PythonObject:
    """test = rows [lo, hi), train = rows [0, lo) then [hi, n), ascending
    int64 (LeaveOneOut's split i is lo = i, hi = i + 1). Returns hi - lo."""
    var rows = Int(py=n)
    var a = Int(py=lo)
    var b = Int(py=hi)
    if rows < 1 or a < 0 or b < a or b > rows:
        raise Error("leave_range_i64: needs 0 <= lo <= hi <= n, n >= 1")
    var tp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=test_addr))
    var rp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=train_addr))
    with GILReleased(Python()):
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
    return PythonObject(b - a)


def mask_from_indices_u8_binding(
    idx_addr: PythonObject, k: PythonObject, n: PythonObject, mask_addr: PythonObject,
) raises -> PythonObject:
    """mask[r] = 1 for every int64 index r of the k at `idx` (mask zeroed
    first, n bytes); returns the number of distinct rows set. An index
    outside [0, n) raises before the mask is read."""
    var count = Int(py=k)
    var rows = Int(py=n)
    if count < 0 or rows < 1:
        raise Error("mask_from_indices_u8: k must be non-negative and n positive")
    var mp = _ptr[DType.uint8](Int(py=mask_addr))
    var bad = False
    var set = 0
    with GILReleased(Python()):
        for r in range(rows):
            mp.unsafe_store(r, 0)
        if count > 0:
            var ip = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=idx_addr))
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
    return PythonObject(set)


def select_mask_u8_i64_binding(
    mask_addr: PythonObject, n: PythonObject, test_addr: PythonObject, train_addr: PythonObject,
) raises -> PythonObject:
    """Ascending int64 rows whose mask byte is nonzero into `test`, the rest
    into `train`; returns the test count. A null output address is legal
    when that side is empty. `n_test` = count of nonzero bytes: the caller
    sizes the outputs from `count_mask_u8`."""
    var rows = Int(py=n)
    if rows < 1:
        raise Error("select_mask_u8_i64: n must be positive")
    var mp = _ptr[DType.uint8](Int(py=mask_addr))
    var ta = Int(py=test_addr)
    var ra = Int(py=train_addr)
    var tp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=ta)
    var rp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=ra)
    var n_test = 0
    with GILReleased(Python()):
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
    return PythonObject(n_test)


def count_mask_u8_binding(mask_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """The number of nonzero bytes of the n-byte mask."""
    var rows = Int(py=n)
    if rows < 0:
        raise Error("count_mask_u8: n must be non-negative")
    if rows == 0:
        return PythonObject(0)
    var mp = _ptr[DType.uint8](Int(py=mask_addr))
    var c = 0
    with GILReleased(Python()):
        for i in range(rows):
            if mp.unsafe_load(i) != 0:
                c += 1
    return PythonObject(c)


def next_combination_i64_binding(addr: PythonObject, p: PythonObject, n: PythonObject) raises -> PythonObject:
    """Advance the ascending p-combination of range(n) at `addr` (int64) to
    its lexicographic successor (itertools.combinations order); 1 when it
    advanced, 0 when it was the last (left unchanged)."""
    var k = Int(py=p)
    var rows = Int(py=n)
    if k < 1 or k > rows:
        raise Error("next_combination_i64: needs 1 <= p <= n")
    var cp = _ptr[DType.int64](Int(py=addr))
    var i = k - 1
    while i >= 0 and Int(cp.unsafe_load(i)) == i + rows - k:
        i -= 1
    if i < 0:
        return PythonObject(0)
    var v = cp.unsafe_load(i) + 1
    for j in range(i, k):
        cp.unsafe_store(j, v + Int64(j - i))
    return PythonObject(1)


def ic_running_min_f64_binding(
    llf_addr: PythonObject, n: PythonObject, penalty: PythonObject, order: PythonObject,
    ic_addr: PythonObject, best_ic_addr: PythonObject, best_idx_addr: PythonObject,
) raises -> PythonObject:
    """AutoARIMA's information criterion and order choice per series (lane
    cgr4-py-compute): ic[b] = -2 llf[b] + penalty in float64 from the fit's
    float32 log-likelihood, written to `ic`; then the running argmin over
    the orders tried so far, `np.argmin`'s rule: the FIRST minimum, a NaN
    taken as the minimum (the first NaN wins). order 0 initialises."""
    var count = Int(py=n)
    var k = Int(py=order)
    if count < 1 or k < 0:
        raise Error("ic_running_min_f64: n must be positive and order non-negative")
    var pen = Float64(py=penalty)
    var lp = _ptr[DType.float32](Int(py=llf_addr))
    var ip = _ptr[DType.float64](Int(py=ic_addr))
    var bp = _ptr[DType.float64](Int(py=best_ic_addr))
    var xp = _ptr[DType.int64](Int(py=best_idx_addr))
    with GILReleased(Python()):
        for b in range(count):
            var v = -2.0 * Float64(lp.unsafe_load(b)) + pen
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
    return PythonObject(0)


@always_inline
def _ftz_bits(v: Float32) -> Float32:
    """A subnormal becomes a zero of the same sign (checks/numerics `ftz`)."""
    var b = bitcast[DType.uint32](v)
    if (b & UInt32(0x7F800000)) == 0 and (b & UInt32(0x007FFFFF)) != 0:
        return bitcast[DType.float32](b & UInt32(0x80000000))
    return v


def fold_pair_f32_binding(dst_addr: PythonObject, src_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """cross_vendor's coordinator fold (lane cgr4-py-compute, out of the
    Python element loop): dst[i] = ftz(ftz(dst[i]) + ftz(src[i])), one
    float32 rounding per add, elementwise (no element depends on another)."""
    var count = Int(py=n)
    if count < 0:
        raise Error("fold_pair_f32: n must be non-negative")
    if count == 0:
        return PythonObject(0)
    var dp = _ptr[DType.float32](Int(py=dst_addr))
    var sp = _ptr[DType.float32](Int(py=src_addr))
    with GILReleased(Python()):
        for i in range(count):
            var s = _ftz_bits(_ftz_bits(dp.unsafe_load(i)) + _ftz_bits(sp.unsafe_load(i)))
            comptime if HOTPATH_SABOTAGE:
                s = -s
            dp.unsafe_store(i, s)
    return PythonObject(0)


def threshold_labels_i64_binding(
    src_addr: PythonObject, code: PythonObject, n: PythonObject, thr: PythonObject,
    strict: PythonObject, below: PythonObject, above: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """dst[i] = above if src[i] > thr (strict) / >= thr (not strict), else
    below; int64 out from a float32 (code 0) or float64 (code 1) score (lane
    cgr4-py-compute: the predict label maps that were Python
    comprehensions). A NaN score compares false and takes `below`."""
    var count = Int(py=n)
    if count < 0:
        raise Error("threshold_labels_i64: n must be non-negative")
    if count == 0:
        return PythonObject(0)
    var c = Int(py=code)
    var t = Float64(py=thr)
    var st = Int(py=strict) != 0
    var lo = Int64(Int(py=below))
    var hi = Int64(Int(py=above))
    var dp = _ptr[DType.int64](Int(py=dst_addr))
    if c != HP_F32 and c != HP_F64:
        raise Error("threshold_labels_i64: float32 or float64 scores only")
    var fp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=src_addr))
    var gp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=src_addr))
    with GILReleased(Python()):
        for i in range(count):
            var v = Float64(fp.unsafe_load(i)) if c == HP_F32 else gp.unsafe_load(i)
            var up = v > t if st else v >= t
            comptime if HOTPATH_SABOTAGE:
                up = not up
            dp.unsafe_store(i, hi if up else lo)
    return PythonObject(0)


def scale_shift_ftz_f32_binding(
    src_addr: PythonObject, n: PythonObject, a: PythonObject, b: PythonObject,
    op: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """Elementwise float32, each step one rounding then `ftz` (lane
    cgr4-py-compute: the GP un-normalization that was a Python
    comprehension): op 0 dst = ftz(ftz(a * v) + b); op 1 dst =
    ftz(sqrt(ftz(v * a))); op 2 dst = ftz(v * a). `a`, `b` are float32
    values. src and dst may be the same buffer."""
    var count = Int(py=n)
    var o = Int(py=op)
    if count < 0 or o < 0 or o > 2:
        raise Error("scale_shift_ftz_f32: bad n or op")
    if count == 0:
        return PythonObject(0)
    var fa = Float32(Float64(py=a))
    var fb = Float32(Float64(py=b))
    var sp = _ptr[DType.float32](Int(py=src_addr))
    var dp = _ptr[DType.float32](Int(py=dst_addr))
    with GILReleased(Python()):
        for i in range(count):
            var v = sp.unsafe_load(i)
            var r: Float32
            if o == 0:
                r = _ftz_bits(_ftz_bits(fa * v) + fb)
            elif o == 1:
                r = _ftz_bits(sqrt(_ftz_bits(v * fa)))
            else:
                r = _ftz_bits(v * fa)
            comptime if HOTPATH_SABOTAGE:
                r = -r
            dp.unsafe_store(i, r)
    return PythonObject(0)


# ---------------------------------------------------------------------------
# lane cgr4-py-compute, round 2: the rest of the per-row Python
# ---------------------------------------------------------------------------


def bincount_i64_binding(
    src_addr: PythonObject, code: PythonObject, n: PythonObject, k: PythonObject,
    counts_addr: PythonObject, accumulate: PythonObject,
) raises -> PythonObject:
    """counts[v] += 1 for each of the n int32 (code 2) or int64 (code 3)
    values at `src` (counts zeroed first unless `accumulate`); a value
    outside [0, k) raises before any count is written."""
    var count = Int(py=n)
    var kk = Int(py=k)
    var c = Int(py=code)
    if count < 0 or kk < 1 or (c != HP_I32 and c != HP_I64):
        raise Error("bincount_i64: needs n >= 0, k >= 1 and int32 or int64 values")
    var cp = _ptr[DType.int64](Int(py=counts_addr))
    var bad = False
    with GILReleased(Python()):
        if Int(py=accumulate) == 0:
            for j in range(kk):
                cp.unsafe_store(j, 0)
        if count > 0:
            var ip = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=src_addr))
            var lp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=src_addr))
            for i in range(count):
                var v = Int(ip.unsafe_load(i)) if c == HP_I32 else Int(lp.unsafe_load(i))
                if v < 0 or v >= kk:
                    bad = True
                    break
            if not bad:
                for i in range(count):
                    var v = Int(ip.unsafe_load(i)) if c == HP_I32 else Int(lp.unsafe_load(i))
                    comptime if HOTPATH_SABOTAGE:
                        v = (v + 1) % kk
                    cp.unsafe_store(v, cp.unsafe_load(v) + 1)
    if bad:
        raise Error("bincount_i64: value out of range")
    return PythonObject(0)


def compact_notnan_f32_binding(
    src_addr: PythonObject, n: PythonObject, stride: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """dst = the non-NaN values of src[0], src[stride], ..., src[(n-1)*stride]
    in order (a column of a row-major block); returns their count."""
    var count = Int(py=n)
    var st = Int(py=stride)
    if count < 0 or st < 1:
        raise Error("compact_notnan_f32: n >= 0, stride >= 1")
    if count == 0:
        return PythonObject(0)
    var sp = _ptr[DType.float32](Int(py=src_addr))
    var dp = _ptr[DType.float32](Int(py=dst_addr))
    var m = 0
    with GILReleased(Python()):
        for i in range(count):
            var v = sp.unsafe_load(i * st)
            if v == v:
                dp.unsafe_store(m, v)
                m += 1
    return PythonObject(m)


def gather_keep_neg_i32_binding(
    table_addr: PythonObject, n_table: PythonObject, src_addr: PythonObject,
    n: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """dst[i] = table[src[i]] for src[i] >= 0, else src[i] (a negative
    sentinel, such as a leaf's column id, kept); int32. A non-negative value
    outside the table raises before any write."""
    var count = Int(py=n)
    var nt = Int(py=n_table)
    if count < 0 or nt < 0:
        raise Error("gather_keep_neg_i32: n and n_table must be non-negative")
    if count == 0:
        return PythonObject(0)
    var tp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=table_addr))
    var sp = _ptr[DType.int32](Int(py=src_addr))
    var dp = _ptr[DType.int32](Int(py=dst_addr))
    var bad = False
    with GILReleased(Python()):
        for i in range(count):
            if Int(sp.unsafe_load(i)) >= nt:
                bad = True
                break
        if not bad:
            for i in range(count):
                var v = sp.unsafe_load(i)
                dp.unsafe_store(i, tp.unsafe_load(Int(v)) if v >= 0 else v)
    if bad:
        raise Error("gather_keep_neg_i32: index out of range")
    return PythonObject(0)


def dot_rows_f32_binding(
    a_addr: PythonObject, x_addr: PythonObject, m: PythonObject, d: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """dst[j] = float32(sum_i a[i] * x[i, j]) over a float32 (m,) and a
    row-major float32 (m, d) block, the sum in float64 in row order (one
    fixed fold): SVC's linear coef_ (dual_coef_ @ support_vectors_)."""
    var rows = Int(py=m)
    var cols = Int(py=d)
    if rows < 0 or cols < 0:
        raise Error("dot_rows_f32: m and d must be non-negative")
    if cols == 0:
        return PythonObject(0)
    var dp = _ptr[DType.float32](Int(py=dst_addr))
    var acc = List[Float64](length=cols, fill=0.0)
    if rows > 0:
        var ap = _ptr[DType.float32](Int(py=a_addr))
        var xp = _ptr[DType.float32](Int(py=x_addr))
        for i in range(rows):
            var a = Float64(ap.unsafe_load(i))
            for j in range(cols):
                acc[j] = acc[j] + a * Float64(xp.unsafe_load(i * cols + j))
    for j in range(cols):
        dp.unsafe_store(j, Float32(acc[j]))
    return PythonObject(0)


def assign_fold_i64_binding(
    idx_addr: PythonObject, m: PythonObject, n: PythonObject, fold: PythonObject, folds_addr: PythonObject,
) raises -> PythonObject:
    """folds[idx[i]] = fold for the m int64 test indices of one split, over
    an int32 folds array of n entries (-1 = unassigned). Returns 0, 1 when
    an index is outside [0, n), 2 when a row already has a fold (an overlap
    or a duplicate); nothing is written past the first refusal."""
    var count = Int(py=m)
    var rows = Int(py=n)
    var f = Int32(Int(py=fold))
    if count < 0 or rows < 1:
        raise Error("assign_fold_i64: m >= 0, n >= 1")
    if count == 0:
        return PythonObject(0)
    var ip = _ptr[DType.int64](Int(py=idx_addr))
    var fp = _ptr[DType.int32](Int(py=folds_addr))
    var status = 0
    with GILReleased(Python()):
        for i in range(count):
            var r = Int(ip.unsafe_load(i))
            if r < 0 or r >= rows:
                status = 1
                break
            if fp.unsafe_load(r) != -1:
                status = 2
                break
            fp.unsafe_store(r, f)
    return PythonObject(status)


def count_fold_hits_i64_binding(
    idx_addr: PythonObject, m: PythonObject, folds_addr: PythonObject, n: PythonObject, fold: PythonObject,
) raises -> PythonObject:
    """How many of the m int64 indices (all in [0, n)) sit in fold `fold`
    of the int32 folds array."""
    var count = Int(py=m)
    var rows = Int(py=n)
    var f = Int32(Int(py=fold))
    if count <= 0:
        return PythonObject(0)
    var ip = _ptr[DType.int64](Int(py=idx_addr))
    var fp = _ptr[DType.int32](Int(py=folds_addr))
    var hits = 0
    with GILReleased(Python()):
        for i in range(count):
            var r = Int(ip.unsafe_load(i))
            if r >= 0 and r < rows and fp.unsafe_load(r) == f:
                hits += 1
    return PythonObject(hits)


def split_table_i32_binding(
    perm_addr: PythonObject, m: PythonObject, n_test: PythonObject, n_train: PythonObject,
    counts_addr: PythonObject, table_addr: PythonObject, sums_addr: PythonObject,
) raises -> PythonObject:
    """One GroupShuffleSplit draw as a per-group side table: table[g] = 1
    for the first n_test groups of the int64 permutation, 0 for the next
    n_train, 2 for the rest; sums (int64, 2) = [rows on the train side, rows
    on the test side] from the int64 per-group row counts."""
    var mm = Int(py=m)
    var te = Int(py=n_test)
    var tr = Int(py=n_train)
    if mm < 1 or te < 0 or tr < 0 or te + tr > mm:
        raise Error("split_table_i32: bad sizes")
    var pp = _ptr[DType.int64](Int(py=perm_addr))
    var cp = _ptr[DType.int64](Int(py=counts_addr))
    var tp = _ptr[DType.int32](Int(py=table_addr))
    var sp = _ptr[DType.int64](Int(py=sums_addr))
    var c_tr = Int64(0)
    var c_te = Int64(0)
    var bad = False
    with GILReleased(Python()):
        for g in range(mm):
            tp.unsafe_store(g, 2)
        for i in range(te + tr):
            var g = Int(pp.unsafe_load(i))
            if g < 0 or g >= mm:
                bad = True
                break
            if i < te:
                tp.unsafe_store(g, 1)
                c_te += cp.unsafe_load(g)
            else:
                tp.unsafe_store(g, 0)
                c_tr += cp.unsafe_load(g)
    if bad:
        raise Error("split_table_i32: group index out of range")
    sp.unsafe_store(0, c_tr)
    sp.unsafe_store(1, c_te)
    return PythonObject(0)


def scatter_rows_bytes_binding(
    src_addr: PythonObject, rows_addr: PythonObject, m: PythonObject, row_bytes: PythonObject,
    dst_addr: PythonObject, dst_rows: PythonObject,
) raises -> PythonObject:
    """dst row rows[i] = src row i (row_bytes each) for the m int64 rows; a
    row outside [0, dst_rows) raises before any write."""
    var count = Int(py=m)
    var width = Int(py=row_bytes)
    var nd = Int(py=dst_rows)
    if count < 0 or width < 0 or nd < 0:
        raise Error("scatter_rows_bytes: dimensions must be non-negative")
    if count == 0 or width == 0:
        return PythonObject(0)
    var sp = _ptr[DType.uint8](Int(py=src_addr))
    var dp = _ptr[DType.uint8](Int(py=dst_addr))
    var rp = _ptr[DType.int64](Int(py=rows_addr))
    var bad = False
    with GILReleased(Python()):
        for i in range(count):
            var r = Int(rp.unsafe_load(i))
            if r < 0 or r >= nd:
                bad = True
                break
        if not bad:
            for i in range(count):
                var r = Int(rp.unsafe_load(i))
                for b in range(width):
                    dp.unsafe_store(r * width + b, sp.unsafe_load(i * width + b))
    if bad:
        raise Error("scatter_rows_bytes: row index out of bounds")
    return PythonObject(0)


def uniform_init_f32_binding(
    dst_addr: PythonObject, n: PythonObject, low: PythonObject, high: PythonObject,
    seed_lo: PythonObject, seed_hi: PythonObject, offset: PythonObject,
) raises -> PythonObject:
    """dst[i] = float32(low + (high - low) * u_i), u_i the 53-bit uniform of
    splitmix64 at counter offset + i of the seed: a counter-based stream, so
    a caller drawing several arrays from one seed advances `offset` by each
    array's size. Weight initialisation (lane cgr4-py-compute: it was
    numpy's Generator in Python); the same bytes on every column."""
    var count = Int(py=n)
    if count < 0:
        raise Error("uniform_init_f32: n must be non-negative")
    if count == 0:
        return PythonObject(0)
    var lo = Float64(py=low)
    var hi = Float64(py=high)
    var seed = (UInt64(Int(py=seed_hi)) << 32) | UInt64(Int(py=seed_lo))
    var off = UInt64(Int(py=offset))
    var dp = _ptr[DType.float32](Int(py=dst_addr))
    with GILReleased(Python()):
        for i in range(count):
            var s = seed + (off + UInt64(i)) * UInt64(0x9E3779B97F4A7C15)
            var u = Float64(splitmix64(s) >> 11) * 1.1102230246251565e-16
            dp.unsafe_store(i, Float32(lo + (hi - lo) * u))
    return PythonObject(0)


def normal_init_f32_binding(
    dst_addr: PythonObject, n: PythonObject, mean: PythonObject, std: PythonObject,
    seed_lo: PythonObject, seed_hi: PythonObject, offset: PythonObject,
) raises -> PythonObject:
    """dst[i] = float32(mean + std * z_i), z_i a standard normal draw at
    counter c = offset + i of the seed: Box-Muller over the two splitmix64
    uniforms at counters 2c and 2c + 1 (u1 = 1 - U(2c) in (0, 1], the angle
    2 pi U(2c + 1)), z = sqrt(-2 log u1) cos(angle), with the portable log
    and cos (checks/numerics.mojo) so every column writes the same bytes.
    Counter-based as `uniform_init_f32`: a caller drawing several arrays
    from one seed advances `offset` by each array's size. Weight
    initialisation (lane pyglue-numeric: MoEBlock drew numpy normals)."""
    var count = Int(py=n)
    if count < 0:
        raise Error("normal_init_f32: n must be non-negative")
    if count == 0:
        return PythonObject(0)
    var mu = Float64(py=mean)
    var sd = Float64(py=std)
    var seed = (UInt64(Int(py=seed_hi)) << 32) | UInt64(Int(py=seed_lo))
    var off = UInt64(Int(py=offset))
    var dp = _ptr[DType.float32](Int(py=dst_addr))
    with GILReleased(Python()):
        for i in range(count):
            var c = (off + UInt64(i)) * UInt64(2)
            var s1 = seed + c * UInt64(0x9E3779B97F4A7C15)
            var s2 = seed + (c + UInt64(1)) * UInt64(0x9E3779B97F4A7C15)
            var u1 = 1.0 - Float64(splitmix64(s1) >> 11) * 1.1102230246251565e-16
            var u2 = Float64(splitmix64(s2) >> 11) * 1.1102230246251565e-16
            var r = sqrt(-2.0 * portable_log64(u1))
            var cz = portable_cosf(Float32(6.283185307179586 * u2))
            dp.unsafe_store(i, Float32(mu + sd * (r * Float64(cz))))
    return PythonObject(0)


def epoch_order_i32_binding(
    dst_addr: PythonObject, n: PythonObject, shuffle: PythonObject, state_addr: PythonObject,
) raises -> PythonObject:
    """One epoch's row order (sequence/schedule.mojo `fill_epoch_order`):
    0..n-1, Fisher-Yates permuted from the splitmix64 state at `state`
    (uint64, advanced in place) when `shuffle`."""
    var count = Int(py=n)
    if count < 1:
        raise Error("epoch_order_i32: n must be positive")
    var dp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=dst_addr))
    var stp = _ptr[DType.uint64](Int(py=state_addr))
    var s = stp.unsafe_load(0)
    fill_epoch_order(dp, count, Int(py=shuffle) != 0, s)
    stp.unsafe_store(0, s)
    return PythonObject(0)


@always_inline
def _powi64(b: Float64, e: Int) -> Float64:
    """b ** e for e >= 0 by squaring, in a fixed order (the same bits on
    every column)."""
    var r = Float64(1)
    var x = b
    var k = e
    while k > 0:
        if (k & 1) != 0:
            r = r * x
        x = x * x
        k >>= 1
    return r


def adam_hyper_f64_binding(
    dst_addr: PythonObject, step0: PythonObject, nsteps: PythonObject, fp: PythonObject,
) raises -> PythonObject:
    """The Adam / AdamW hyper block of steps step0, step0 + 1, ... (nsteps
    rows of 9 float64): [lr / bc1, 1 - b1, b2, 1 - b2, eps, sqrt(bc2), wd,
    decoupled, 1 - lr wd] with bc = 1 - beta ** step. fp = [lr, b1, b2, eps,
    wd, decoupled]."""
    var s0 = Int(py=step0)
    var ns = Int(py=nsteps)
    if s0 < 1 or ns < 0:
        raise Error("adam_hyper_f64: step0 >= 1, nsteps >= 0")
    var lr = Float64(py=fp[0])
    var b1 = Float64(py=fp[1])
    var b2 = Float64(py=fp[2])
    var eps = Float64(py=fp[3])
    var wd = Float64(py=fp[4])
    var dec = Float64(py=fp[5])
    var dp = _ptr[DType.float64](Int(py=dst_addr))
    for t in range(ns):
        var step = s0 + t
        var bc1 = 1.0 - _powi64(b1, step)
        var bc2 = 1.0 - _powi64(b2, step)
        var row = t * 9
        dp.unsafe_store(row + 0, lr / bc1)
        dp.unsafe_store(row + 1, 1.0 - b1)
        dp.unsafe_store(row + 2, b2)
        dp.unsafe_store(row + 3, 1.0 - b2)
        dp.unsafe_store(row + 4, eps)
        dp.unsafe_store(row + 5, sqrt(bc2))
        dp.unsafe_store(row + 6, wd)
        dp.unsafe_store(row + 7, dec)
        dp.unsafe_store(row + 8, 1.0 - lr * wd)
    return PythonObject(0)


def mean_std_f32_binding(
    src_addr: PythonObject, n: PythonObject, stride: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """dst (float64, 2) = [mean, population std] of src[0], src[stride], ...,
    src[(n-1)*stride] (float32), two passes in float64 in row order."""
    var count = Int(py=n)
    var st = Int(py=stride)
    if count < 1 or st < 1:
        raise Error("mean_std_f32: n >= 1, stride >= 1")
    var sp = _ptr[DType.float32](Int(py=src_addr))
    var dp = _ptr[DType.float64](Int(py=dst_addr))
    var s = Float64(0)
    for i in range(count):
        s += Float64(sp.unsafe_load(i * st))
    var mean = s / Float64(count)
    var q = Float64(0)
    for i in range(count):
        var dv = Float64(sp.unsafe_load(i * st)) - mean
        q += dv * dv
    dp.unsafe_store(0, mean)
    dp.unsafe_store(1, sqrt(q / Float64(count)))
    return PythonObject(0)


def first_seen_i32_binding(
    codes_addr: PythonObject, n: PythonObject, k: PythonObject, enc_addr: PythonObject, counts_addr: PythonObject,
) raises -> PythonObject:
    """Renumber n int32 class codes in [0, k) by first appearance: enc[i] =
    the first-seen rank of codes[i]; counts[r] (int64) = rows of rank r.
    Returns the number of classes present (StratifiedKFold's encoding)."""
    var rows = Int(py=n)
    var kk = Int(py=k)
    if rows < 0 or kk < 1:
        raise Error("first_seen_i32: n >= 0, k >= 1")
    var cp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=codes_addr))
    var ep = _ptr[DType.int32](Int(py=enc_addr))
    var np = _ptr[DType.int64](Int(py=counts_addr))
    var rank = List[Int](length=kk, fill=-1)
    var m = 0
    var bad = False
    for j in range(kk):
        np.unsafe_store(j, 0)
    for i in range(rows):
        var c = Int(cp.unsafe_load(i))
        if c < 0 or c >= kk:
            bad = True
            break
        if rank[c] < 0:
            rank[c] = m
            m += 1
        ep.unsafe_store(i, Int32(rank[c]))
        np.unsafe_store(rank[c], np.unsafe_load(rank[c]) + 1)
    if bad:
        raise Error("first_seen_i32: code out of range")
    return PythonObject(m)


@always_inline
def _floor_div_pos(a: Int, b: Int) -> Int:
    """floor(a / b) for b > 0 and any a (Python's `//`)."""
    if a >= 0:
        return a // b
    return -((-a + b - 1) // b)


def strat_alloc_i64_binding(
    counts_addr: PythonObject, k: PythonObject, n_folds: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """StratifiedKFold's allocation table (sklearn _make_test_folds):
    dst[i * k + c] (int64, n_folds x k) = how many positions p of class c's
    run [s, e) in the class-sorted labels have p % n_folds == i, i.e.
    (e - 1 - i) // K - (s - 1 - i) // K with floor division (lane
    pyglue-numeric: a Python double loop over the folds and classes).
    `counts` int64, k of them."""
    var kk = Int(py=k)
    var K = Int(py=n_folds)
    if kk < 1 or K < 1:
        raise Error("strat_alloc_i64: k and n_folds must be >= 1")
    var cp = _ptr[DType.int64](Int(py=counts_addr))
    var dp = _ptr[DType.int64](Int(py=dst_addr))
    var s = 0
    for c in range(kk):
        var e = s + Int(cp.unsafe_load(c))
        for i in range(K):
            dp.unsafe_store(i * kk + c, Int64(_floor_div_pos(e - 1 - i, K) - _floor_div_pos(s - 1 - i, K)))
        s = e
    return PythonObject(0)


def _stable_order_f64(keys: List[Float64]) -> List[Int]:
    """The indices 0 .. len(keys) - 1 sorted ascending by key, ties in index
    order (a bottom-up merge sort: stable, O(m log m))."""
    var m = len(keys)
    var a = List[Int](capacity=m)
    for i in range(m):
        a.append(i)
    var b = List[Int](length=m, fill=0)
    var width = 1
    while width < m:
        var lo = 0
        while lo < m:
            var mid = min(lo + width, m)
            var hi = min(lo + 2 * width, m)
            var i = lo
            var j = mid
            var t = lo
            while i < mid and j < hi:
                if keys[a[j]] < keys[a[i]]:
                    b[t] = a[j]
                    j += 1
                else:
                    b[t] = a[i]
                    i += 1
                t += 1
            while i < mid:
                b[t] = a[i]
                i += 1
                t += 1
            while j < hi:
                b[t] = a[j]
                j += 1
                t += 1
            lo = hi
        var tmp = a.copy()
        a = b.copy()
        b = tmp^
        width *= 2
    return a^


def group_fold_assign_i32_binding(
    counts_addr: PythonObject, m: PythonObject, n_folds: PythonObject, perm_addr: PythonObject,
    dst_addr: PythonObject, sizes_addr: PythonObject,
) raises -> PythonObject:
    """GroupKFold's fold of each of the m groups (scikit-learn 1.9), int32
    dst[g], and each fold's row count, int64 sizes[f] (from the int64 group
    row counts). With `perm` (int64, nonzero address) the permuted groups
    split into K nearly equal runs (the first m % K one longer); without,
    the groups by row count descending (ties: the higher group code first)
    each to the lightest fold (ties: the lower fold). Lane pyglue-numeric:
    both were Python loops over the groups."""
    var mm = Int(py=m)
    var K = Int(py=n_folds)
    if mm < 0 or K < 1:
        raise Error("group_fold_assign_i32: m >= 0 and n_folds >= 1")
    var cp = _ptr[DType.int64](Int(py=counts_addr))
    var dp = _ptr[DType.int32](Int(py=dst_addr))
    var sp = _ptr[DType.int64](Int(py=sizes_addr))
    var pa = Int(py=perm_addr)
    for f in range(K):
        sp.unsafe_store(f, Int64(0))
    if pa != 0:
        var pp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=pa)
        var start = 0
        for f in range(K):
            var size = mm // K + (1 if f < mm % K else 0)
            for j in range(start, start + size):
                var g = Int(pp.unsafe_load(j))
                if g < 0 or g >= mm:
                    raise Error("group_fold_assign_i32: a permuted group is out of range")
                dp.unsafe_store(g, Int32(f))
            start += size
    else:
        var keys = List[Float64](capacity=mm)
        for g in range(mm):
            keys.append(Float64(Int(cp.unsafe_load(g))))
        var asc = _stable_order_f64(keys)
        var load = List[Int](length=K, fill=0)
        for t in range(mm - 1, -1, -1):
            var g = asc[t]
            var best = 0
            for f in range(1, K):
                if load[f] < load[best]:
                    best = f
            load[best] += Int(cp.unsafe_load(g))
            dp.unsafe_store(g, Int32(best))
    for g in range(mm):
        var f = Int(dp.unsafe_load(g))
        sp.unsafe_store(f, sp.unsafe_load(f) + cp.unsafe_load(g))
    return PythonObject(0)


def strat_group_assign_i32_binding(
    yenc_addr: PythonObject, gidx_addr: PythonObject, n: PythonObject, k: PythonObject, m: PythonObject,
    n_folds: PythonObject, perm_addr: PythonObject, dst_addr: PythonObject, sizes_addr: PythonObject,
) raises -> PythonObject:
    """StratifiedGroupKFold's fold of each of the m groups (scikit-learn
    1.9 `_find_best_fold`): each group's class distribution, the groups
    (in code order, or permuted by `perm` int64 when nonzero) sorted by the
    standard deviation of their distribution, descending and stable, each
    to the fold whose per-class fold shares it perturbs least (the mean over
    classes of the across-fold std; ties: the fewer rows, then the lower
    fold). int32 dst[g]; int64 sizes[f] the rows of each fold. yenc: int32
    class codes in [0, k); gidx: int32 group codes in [0, m). Returns 1 when
    the largest class has fewer rows than n_folds (the caller refuses), else
    0. Binary64 sums ascending (lane pyglue-numeric: the Python loops)."""
    var nn = Int(py=n)
    var kk = Int(py=k)
    var mm = Int(py=m)
    var K = Int(py=n_folds)
    if nn < 0 or kk < 1 or mm < 1 or K < 1:
        raise Error("strat_group_assign_i32: bad sizes")
    var yp = _ptr[DType.int32](Int(py=yenc_addr))
    var gp = _ptr[DType.int32](Int(py=gidx_addr))
    var dp = _ptr[DType.int32](Int(py=dst_addr))
    var sp = _ptr[DType.int64](Int(py=sizes_addr))
    var pa = Int(py=perm_addr)
    var counts = List[Int](length=kk, fill=0)
    var dist = List[Int](length=mm * kk, fill=0)
    for i in range(nn):
        var c = Int(yp.unsafe_load(i))
        var g = Int(gp.unsafe_load(i))
        if c < 0 or c >= kk or g < 0 or g >= mm:
            raise Error("strat_group_assign_i32: a code is out of range")
        counts[c] += 1
        dist[g * kk + c] += 1
    var most = 0
    for c in range(kk):
        most = max(most, counts[c])
    if most < K:
        return PythonObject(1)
    var order = List[Int](capacity=mm)
    if pa != 0:
        var pp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=pa)
        for j in range(mm):
            order.append(Int(pp.unsafe_load(j)))
    else:
        for j in range(mm):
            order.append(j)
    # -std per group in `order`'s positions, stably sorted ascending
    var keys = List[Float64](capacity=mm)
    for j in range(mm):
        var g = order[j]
        var tot = Float64(0)
        for c in range(kk):
            tot += Float64(dist[g * kk + c])
        var mu = tot / Float64(kk)
        var ss = Float64(0)
        for c in range(kk):
            var dv = Float64(dist[g * kk + c]) - mu
            ss += dv * dv
        keys.append(-sqrt(ss / Float64(kk)))
    var pos = _stable_order_f64(keys)
    var fold_dist = List[Int](length=K * kk, fill=0)
    var fold_n = List[Int](length=K, fill=0)
    var col = List[Float64](length=K, fill=0)
    for t in range(mm):
        var g = order[pos[t]]
        var best = -1
        var best_score = Float64(0)
        var best_n = 0
        for f in range(K):
            var score = Float64(0)
            for c in range(kk):
                var mu = Float64(0)
                for j in range(K):
                    var v = fold_dist[j * kk + c] + (dist[g * kk + c] if j == f else 0)
                    col[j] = Float64(v) / Float64(counts[c])
                    mu += col[j]
                mu = mu / Float64(K)
                var ss = Float64(0)
                for j in range(K):
                    var dv = col[j] - mu
                    ss += dv * dv
                score += sqrt(ss / Float64(K))
            score = score / Float64(kk)
            if best < 0 or score < best_score or (score == best_score and fold_n[f] < best_n):
                best = f
                best_score = score
                best_n = fold_n[f]
        for c in range(kk):
            fold_dist[best * kk + c] += dist[g * kk + c]
            fold_n[best] += dist[g * kk + c]
        dp.unsafe_store(g, Int32(best))
    for f in range(K):
        sp.unsafe_store(f, Int64(fold_n[f]))
    return PythonObject(0)


def strat_fold_assign_i32_binding(
    enc_addr: PythonObject, n: PythonObject, k: PythonObject, n_folds: PythonObject,
    alloc_addr: PythonObject, perms_addr: PythonObject, counts_addr: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """StratifiedKFold's test fold of every row (sklearn _make_test_folds):
    class c's fold sequence is fold f repeated alloc[f * k + c] times (int64)
    in fold order; with `perms` (int64, nonzero address) it is permuted,
    seq'[j] = seq[perm_c[j]], perm_c the segment of class c (classes in
    order, `counts` int64 rows each); the r-th row of class c (row order)
    takes seq'[r]. enc: int32 first-seen class ranks in [0, k)."""
    var rows = Int(py=n)
    var kk = Int(py=k)
    var K = Int(py=n_folds)
    if rows < 1 or kk < 1 or K < 1:
        raise Error("strat_fold_assign_i32: n, k, n_folds >= 1")
    var ep = _ptr[DType.int32](Int(py=enc_addr))
    var ap = _ptr[DType.int64](Int(py=alloc_addr))
    var cp = _ptr[DType.int64](Int(py=counts_addr))
    var dp = _ptr[DType.int32](Int(py=dst_addr))
    var pa = Int(py=perms_addr)
    var off = List[Int](length=kk + 1, fill=0)
    for c in range(kk):
        off[c + 1] = off[c] + Int(cp.unsafe_load(c))
    if off[kk] != rows:
        raise Error("strat_fold_assign_i32: counts do not sum to n")
    var seq = List[Int32](length=rows, fill=0)
    for c in range(kk):
        var at = off[c]
        for f in range(K):
            for _ in range(Int(ap.unsafe_load(f * kk + c))):
                if at >= off[c + 1]:
                    raise Error("strat_fold_assign_i32: alloc exceeds the class count")
                seq[at] = Int32(f)
                at += 1
    var seen = List[Int](length=kk, fill=0)
    var pp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=pa)
    for i in range(rows):
        var c = Int(ep.unsafe_load(i))
        if c < 0 or c >= kk:
            raise Error("strat_fold_assign_i32: class out of range")
        var j = seen[c]
        seen[c] = j + 1
        var src = j
        if pa != 0:
            src = Int(pp.unsafe_load(off[c] + j))
        var f = seq[off[c] + src]
        comptime if HOTPATH_SABOTAGE:
            f = Int32((Int(f) + 1) % K)
        dp.unsafe_store(i, f)
    return PythonObject(0)
