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
split contiguous ranges across the host pool under `MOJOLEARN_CPU_THREADS`
(`core/host_predict_threads.mojo`); a range is written by one task and no
element depends on another, so the thread count moves no byte.
"""
from std.math import isfinite
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined

from max.algorithm import sync_parallelize

from core.host_predict_threads import host_predict_chunk, host_predict_task_count

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

#: Below this many elements one task on the calling thread is cheaper than
#: waking the pool (the threshold `core/forest_inference_model.mojo` uses).
comptime HP_SERIAL = 1 << 16
comptime HP_W = 8


@always_inline
def _ptr[dt: DType](addr: Int) raises -> MutPointer[Scalar[dt], MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=addr)


def _tasks(n: Int) -> Int:
    if n < HP_SERIAL:
        return 1
    return host_predict_task_count(n)


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
    var chunk = host_predict_chunk(n, tasks)
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
        if tasks == 1:
            _range(0)
        else:
            sync_parallelize(_range, tasks)
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
    (what = 4) whether every float is finite and integer valued."""
    var count = Int(py=n)
    if count < 1:
        raise Error("reduce_stat: n must be positive, got " + String(count))
    var w = Int(py=what)
    if w < HP_MIN or w > HP_INTEGRAL:
        raise Error("reduce_stat: unknown reduction " + String(w))
    var c = Int(py=code)
    var a = Int(py=addr)
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
    var chunk = host_predict_chunk(n, tasks)

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
        if tasks == 1:
            _range(0)
        else:
            sync_parallelize(_range, tasks)


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
            var chunk = host_predict_chunk(count, tasks)

            def _range(t: Int) {imm tp, imm cp, imm dp, imm count, imm chunk, imm nt}:
                var lo = t * chunk
                var hi = min(lo + chunk, count)
                for i in range(lo, hi):
                    comptime if HOTPATH_SABOTAGE:
                        dp.unsafe_store(i, tp.unsafe_load((Int(cp.unsafe_load(i)) + 1) % nt))
                    else:
                        dp.unsafe_store(i, tp.unsafe_load(Int(cp.unsafe_load(i))))

            if tasks == 1:
                _range(0)
            else:
                sync_parallelize(_range, tasks)
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
