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
from std.math import fma, isfinite, sqrt
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined
from sequence.schedule import fill_epoch_order, splitmix64
from checks.numerics import portable_cosf, portable_log64
from metrics.checks.pinned_sum import canonicalize_nan
from core.hotpath_impl import (
    HOTPATH_SABOTAGE,
    HP_F32,
    HP_F64,
    HP_I32,
    HP_I64,
    HP_U32,
    HP_U8,
    HP_U16,
    HP_I8,
    HP_F16,
    HP_I64_F64,
    HP_W,
    _ptr,
    _tasks,
)
from core.hotpath_impl import (
    HP_STAT_FLOAT,
    HP_STAT_TUPLE,
    HpStat,
    arange_i64,
    cast_elements,
    check_indices_i64,
    count_mask_u8,
    encode_labels,
    equal_elements,
    fold_ids,
    gather_i32,
    ic_running_min_f32_host,
    ic_running_min_f64_host,
    indices_overlap_i64,
    leave_range_i64,
    mask_from_indices_u8,
    next_combination_i64,
    reduce_stat,
    select_fold_i64,
    select_mask_u8_i64,
)





# ---------------------------------------------------------------------------
# The Python doors of the helpers. Each unpacks its PythonObject arguments,
# calls ONE typed kernel of `core/hotpath_impl.mojo` with the GIL released
# and returns its answer (lane thin-core, 2026-10-21: the compute moved
# there word for word).
# ---------------------------------------------------------------------------


def cast_elements_binding(
    src_addr: PythonObject, src_code: PythonObject, dst_addr: PythonObject,
    dst_code: PythonObject, n: PythonObject,
) raises -> PythonObject:
    """`dst[i] = <dst dtype>(src[i])` for `i` in `[0, n)`, the conversion
    `array.array(code, memoryview)` performs (DEVIATION 3100). Returns 0, or
    1 when some element is one the Python conversion refuses; `dst` is then
    unspecified and the caller reruns the Python path for its refusal."""
    var count = Int(py=n)
    var sa = Int(py=src_addr)
    var da = Int(py=dst_addr)
    var sc = Int(py=src_code)
    var dc = Int(py=dst_code)
    var status: Int
    with GILReleased(Python()):
        status = cast_elements(sa, sc, da, dc, count)
    return PythonObject(status)


def reduce_stat_binding(
    addr: PythonObject, code: PythonObject, n: PythonObject, what: PythonObject,
) raises -> PythonObject:
    """Python's `min`, `max`, left-to-right float `sum` or first-max-wins
    argmax over `n >= 1` elements in storage order (DEVIATION 3101), or
    (what = 4) whether every float is finite and integer valued, or (what =
    5, lane apple-fast-py2mojo-core) the exact integer sum as (hi, mid, low)."""
    var count = Int(py=n)
    var w = Int(py=what)
    var c = Int(py=code)
    var a = Int(py=addr)
    var r: HpStat
    with GILReleased(Python()):
        r = reduce_stat(a, c, count, w)
    if r.kind == HP_STAT_TUPLE:
        return Python.tuple(PythonObject(r.hi), PythonObject(r.mid), PythonObject(r.low))
    if r.kind == HP_STAT_FLOAT:
        return PythonObject(r.f)
    return PythonObject(r.i)


def equal_elements_binding(
    a_addr: PythonObject, b_addr: PythonObject, code: PythonObject,
    n: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """`dst[i] = 1 if a[i] == b[i] else 0` over two buffers of ONE dtype
    (DEVIATION 3102): IEEE equality for floats, which is Python's (NaN is
    unequal to itself, -0.0 equals 0.0)."""
    var count = Int(py=n)
    var c = Int(py=code)
    var a = Int(py=a_addr)
    var b = Int(py=b_addr)
    var d = Int(py=dst_addr)
    var status: Int
    with GILReleased(Python()):
        status = equal_elements(a, b, c, count, d)
    return PythonObject(status)


def _encode_labels_binding[dt: DType](
    src_addr: PythonObject, n: PythonObject, classes_addr: PythonObject,
    max_classes: PythonObject, codes_addr: PythonObject,
) raises -> PythonObject:
    var count = Int(py=n)
    var cap = Int(py=max_classes)
    var sa = Int(py=src_addr)
    var ca = Int(py=classes_addr)
    var da = Int(py=codes_addr)
    var k: Int
    with GILReleased(Python()):
        k = encode_labels[dt](sa, count, ca, cap, da)
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
    (DEVIATION 3103). A code outside `[0, n_table)` raises before any write."""
    var count = Int(py=n)
    var nt = Int(py=n_table)
    var ta = Int(py=table_addr)
    var ca = Int(py=codes_addr)
    var da = Int(py=dst_addr)
    var status: Int
    with GILReleased(Python()):
        status = gather_i32(ta, nt, ca, count, da)
    return PythonObject(status)


def check_indices_i64_binding(
    addr: PythonObject, n: PythonObject, bound: PythonObject,
) raises -> PythonObject:
    """0 when the `n` int64 indices are all in `[0, bound)` and distinct, 1
    when any is out of range, else 2 for a duplicate."""
    var count = Int(py=n)
    var limit = Int(py=bound)
    var a = Int(py=addr)
    var status: Int
    with GILReleased(Python()):
        status = check_indices_i64(a, count, limit)
    return PythonObject(status)


def indices_overlap_i64_binding(
    a_addr: PythonObject, n_a: PythonObject, b_addr: PythonObject,
    n_b: PythonObject, bound: PythonObject,
) raises -> PythonObject:
    """1 when two int64 index sets share a member, else 0."""
    var na = Int(py=n_a)
    var nb = Int(py=n_b)
    var limit = Int(py=bound)
    var a = Int(py=a_addr)
    var b = Int(py=b_addr)
    var hit: Int
    with GILReleased(Python()):
        hit = indices_overlap_i64(a, na, b, nb, limit)
    return PythonObject(hit)


def fold_ids_binding(
    codes_addr: PythonObject, n: PythonObject, n_classes: PythonObject,
    n_splits: PythonObject, counts_addr: PythonObject, fold_addr: PythonObject,
    fold_counts_addr: PythonObject,
) raises -> PythonObject:
    """The unshuffled default folds of `model_selection._default_folds` as one
    int32 fold id per row (DEVIATION 3104; `core/hotpath_impl.mojo::fold_ids`)."""
    var rows = Int(py=n)
    var k = Int(py=n_classes)
    var splits = Int(py=n_splits)
    var ca = Int(py=codes_addr)
    var na = Int(py=counts_addr)
    var fa = Int(py=fold_addr)
    var fca = Int(py=fold_counts_addr)
    var status: Int
    with GILReleased(Python()):
        status = fold_ids(ca, rows, k, splits, na, fa, fca)
    return PythonObject(status)


def select_fold_i64_binding(
    fold_addr: PythonObject, n: PythonObject, fold: PythonObject,
    test_addr: PythonObject, train_addr: PythonObject,
) raises -> PythonObject:
    """Ascending int64 row indices with fold id `fold` into `test`, every
    other row into `train`; returns the test count."""
    var rows = Int(py=n)
    var want = Int(py=fold)
    var fa = Int(py=fold_addr)
    var ta = Int(py=test_addr)
    var ra = Int(py=train_addr)
    var n_test: Int
    with GILReleased(Python()):
        n_test = select_fold_i64(fa, rows, want, ta, ra)
    return PythonObject(n_test)


def arange_i64_binding(dst_addr: PythonObject, start: PythonObject, n: PythonObject) raises -> PythonObject:
    """`dst[i] = start + i` for i < n, int64."""
    var count = Int(py=n)
    var lo = Int(py=start)
    var da = Int(py=dst_addr)
    var status: Int
    with GILReleased(Python()):
        status = arange_i64(da, lo, count)
    return PythonObject(status)


def leave_range_i64_binding(
    n: PythonObject, lo: PythonObject, hi: PythonObject,
    train_addr: PythonObject, test_addr: PythonObject,
) raises -> PythonObject:
    """test = rows [lo, hi), train = rows [0, lo) then [hi, n), ascending
    int64. Returns hi - lo."""
    var rows = Int(py=n)
    var a = Int(py=lo)
    var b = Int(py=hi)
    var ra = Int(py=train_addr)
    var ta = Int(py=test_addr)
    var width: Int
    with GILReleased(Python()):
        width = leave_range_i64(rows, a, b, ra, ta)
    return PythonObject(width)


def mask_from_indices_u8_binding(
    idx_addr: PythonObject, k: PythonObject, n: PythonObject, mask_addr: PythonObject,
) raises -> PythonObject:
    """mask[r] = 1 for every int64 index r of the k at `idx`; returns the
    number of distinct rows set."""
    var count = Int(py=k)
    var rows = Int(py=n)
    var ia = Int(py=idx_addr)
    var ma = Int(py=mask_addr)
    var set: Int
    with GILReleased(Python()):
        set = mask_from_indices_u8(ia, count, rows, ma)
    return PythonObject(set)


def select_mask_u8_i64_binding(
    mask_addr: PythonObject, n: PythonObject, test_addr: PythonObject, train_addr: PythonObject,
) raises -> PythonObject:
    """Ascending int64 rows whose mask byte is nonzero into `test`, the rest
    into `train`; returns the test count."""
    var rows = Int(py=n)
    var ma = Int(py=mask_addr)
    var ta = Int(py=test_addr)
    var ra = Int(py=train_addr)
    var n_test: Int
    with GILReleased(Python()):
        n_test = select_mask_u8_i64(ma, rows, ta, ra)
    return PythonObject(n_test)


def count_mask_u8_binding(mask_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """The number of nonzero bytes of the n-byte mask."""
    var rows = Int(py=n)
    var ma = Int(py=mask_addr)
    var c: Int
    with GILReleased(Python()):
        c = count_mask_u8(ma, rows)
    return PythonObject(c)


def next_combination_i64_binding(addr: PythonObject, p: PythonObject, n: PythonObject) raises -> PythonObject:
    """Advance the ascending p-combination of range(n) at `addr` (int64) to
    its lexicographic successor; 1 when it advanced, 0 when it was the last."""
    var k = Int(py=p)
    var rows = Int(py=n)
    var a = Int(py=addr)
    return PythonObject(next_combination_i64(a, k, rows))


def ic_running_min_f32_host_binding(
    llf_addr: PythonObject, n: PythonObject, penalty: PythonObject, order: PythonObject,
    ic_addr: PythonObject, best_ic_addr: PythonObject, best_idx_addr: PythonObject,
) raises -> PythonObject:
    """`ic_running_min_f64` with the criterion in float32
    (`core/hotpath_impl.mojo::ic_running_min_f32_host`)."""
    var count = Int(py=n)
    var k = Int(py=order)
    var pen = Float64(py=penalty)
    var la = Int(py=llf_addr)
    var ia = Int(py=ic_addr)
    var ba = Int(py=best_ic_addr)
    var xa = Int(py=best_idx_addr)
    var status: Int
    with GILReleased(Python()):
        status = ic_running_min_f32_host(la, count, pen, k, ia, ba, xa)
    return PythonObject(status)


def ic_running_min_f64_host_binding(
    llf_addr: PythonObject, n: PythonObject, penalty: PythonObject, order: PythonObject,
    ic_addr: PythonObject, best_ic_addr: PythonObject, best_idx_addr: PythonObject,
) raises -> PythonObject:
    """AutoARIMA's information criterion and running first-minimum order
    choice per series (`core/hotpath_impl.mojo::ic_running_min_f64_host`)."""
    var count = Int(py=n)
    var k = Int(py=order)
    var pen = Float64(py=penalty)
    var la = Int(py=llf_addr)
    var ia = Int(py=ic_addr)
    var ba = Int(py=best_ic_addr)
    var xa = Int(py=best_idx_addr)
    var status: Int
    with GILReleased(Python()):
        status = ic_running_min_f64_host(la, count, pen, k, ia, ba, xa)
    return PythonObject(status)


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
            # NaN -> the canonical word, as the device kernel (lane/review-fixes)
            var s = canonicalize_nan(_ftz_bits(_ftz_bits(dp.unsafe_load(i)) + _ftz_bits(sp.unsafe_load(i))))
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


def mark_split_roles_f32_binding(
    idx_addr: PythonObject, m: PythonObject, n: PythonObject, split: PythonObject, role: PythonObject,
    roles_addr: PythonObject,
) raises -> PythonObject:
    """roles[split * n + idx[i]] = role (1 training, 2 test) for the m int64
    indices of one side of one cv split, over a float32 roles array of
    (splits, n) zeros: the per-split roles of LassoCV / ElasticNetCV /
    LogisticRegressionCV for splits that are not a partition
    (x_linear/cv_roles.mojo). Returns 0, 1 when an index is outside [0, n),
    2 when a row already has a role in this split (a duplicate, or a row on
    both sides); nothing is written past the first refusal."""
    var count = Int(py=m)
    var rows = Int(py=n)
    var k = Int(py=split)
    var r = Int(py=role)
    if count < 0 or rows < 1 or k < 0 or (r != 1 and r != 2):
        raise Error("mark_split_roles_f32: m >= 0, n >= 1, split >= 0, role 1 or 2")
    if count == 0:
        return PythonObject(0)
    var ip = _ptr[DType.int64](Int(py=idx_addr))
    var rp = _ptr[DType.float32](Int(py=roles_addr))
    var base = k * rows
    var value = Float32(r)
    var status = 0
    with GILReleased(Python()):
        for i in range(count):
            var row = Int(ip.unsafe_load(i))
            if row < 0 or row >= rows:
                status = 1
                break
            if rp.unsafe_load(base + row) != Float32(0):
                status = 2
                break
            rp.unsafe_store(base + row, value)
    return PythonObject(status)


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


def scatter_rows_bytes_host_binding(
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
            # lane fix-s1-shared: an explicit fma (Mojo's default fp-mode
            # contracts `lo + (hi - lo) * u` anyway); the device twin
            # (core/hotpath_device.mojo) is the same sf64_fma
            dp.unsafe_store(i, Float32(fma(hi - lo, u, lo)))
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
            # lane fix-s1-shared: an explicit fma, as the device twin's
            # sf64_fma (core/hotpath_device.mojo::_normal_init_kernel)
            dp.unsafe_store(i, Float32(fma(sd, r * Float64(cz), mu)))
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


def strat_group_assign_i32_binding(addrs: PythonObject, dims: PythonObject) raises -> PythonObject:
    """StratifiedGroupKFold's fold of each of the m groups (scikit-learn
    1.9 `_find_best_fold`): each group's class distribution, the groups
    (in code order, or permuted by `perm` int64 when nonzero) sorted by the
    standard deviation of their distribution, descending and stable, each
    to the fold whose per-class fold shares it perturbs least (the mean over
    classes of the across-fold std; ties: the fewer rows, then the lower
    fold). int32 dst[g]; int64 sizes[f] the rows of each fold. yenc: int32
    class codes in [0, k); gidx: int32 group codes in [0, m). Returns 1 when
    the largest class has fewer rows than n_folds (the caller refuses), else
    0. Binary64 sums ascending (lane pyglue-numeric: the Python loops).
    addrs = [yenc, gidx, perm (0: none), dst, sizes]; dims = [n, k, m, n_folds]."""
    if len(addrs) != 5 or len(dims) != 4:
        raise Error("strat_group_assign_i32: addrs [yenc, gidx, perm, dst, sizes], dims [n, k, m, n_folds]")
    var nn = Int(py=dims[0])
    var kk = Int(py=dims[1])
    var mm = Int(py=dims[2])
    var K = Int(py=dims[3])
    if nn < 0 or kk < 1 or mm < 1 or K < 1:
        raise Error("strat_group_assign_i32: bad sizes")
    var yp = _ptr[DType.int32](Int(py=addrs[0]))
    var gp = _ptr[DType.int32](Int(py=addrs[1]))
    var pa = Int(py=addrs[2])
    var dp = _ptr[DType.int32](Int(py=addrs[3]))
    var sp = _ptr[DType.int64](Int(py=addrs[4]))
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


def strat_group_plan_i32_binding(addrs: PythonObject, dims: PythonObject) raises -> PythonObject:
    """`strat_group_assign_i32` from the group x class table (lane
    cpu4-python): the table is the device's (`bincount2_i32`, int64
    dist[g * k + c]); this is the group-level plan only, the same statements
    in the same order as `strat_group_assign_i32` after its row loop (the
    same folds, the same sizes). Returns 1 when the largest class has fewer
    rows than n_folds, else 0.
    addrs = [dist, perm (0: none), dst, sizes]; dims = [k, m, n_folds]."""
    if len(addrs) != 4 or len(dims) != 3:
        raise Error("strat_group_plan_i32: addrs [dist, perm, dst, sizes], dims [k, m, n_folds]")
    var kk = Int(py=dims[0])
    var mm = Int(py=dims[1])
    var K = Int(py=dims[2])
    if kk < 1 or mm < 1 or K < 1:
        raise Error("strat_group_plan_i32: bad sizes")
    var tp = _ptr[DType.int64](Int(py=addrs[0]))
    var pa = Int(py=addrs[1])
    var dp = _ptr[DType.int32](Int(py=addrs[2]))
    var sp = _ptr[DType.int64](Int(py=addrs[3]))
    var counts = List[Int](length=kk, fill=0)
    var dist = List[Int](length=mm * kk, fill=0)
    for g in range(mm):
        for c in range(kk):
            var v = Int(tp.unsafe_load(g * kk + c))
            dist[g * kk + c] = v
            counts[c] += v
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




@always_inline
def _sm64_next(mut state: UInt64) -> UInt64:
    """Python `_splitmix64`: state += golden, then the splitmix64 mix of it."""
    state = state + UInt64(0x9E3779B97F4A7C15)
    var z = state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def draw_rows_without_replacement_i32_binding(
    n: PythonObject, k: PythonObject, seed_lo: PythonObject, seed_hi: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """k distinct rows of n, ascending, int32 at dst: a splitmix64 partial
    Fisher-Yates (swap i with i + z % (n - i), i < k) then the first k
    sorted (QuantileTransformer's subsample; lane pyglue-numeric: the
    Python `_draw_without_replacement`, the same integers)."""
    var nn = Int(py=n)
    var kk = Int(py=k)
    if nn < 0 or kk < 0 or kk > nn:
        raise Error("draw_rows_without_replacement_i32: 0 <= k <= n")
    var state = (UInt64(Int(py=seed_hi)) << 32) | UInt64(Int(py=seed_lo))
    var dp = _ptr[DType.int32](Int(py=dst_addr))
    with GILReleased(Python()):
        var perm = List[Int](capacity=nn)
        for i in range(nn):
            perm.append(i)
        for i in range(kk):
            var z = _sm64_next(state)
            var j = i + Int(z % UInt64(nn - i))
            var t = perm[i]
            perm[i] = perm[j]
            perm[j] = t
        var mark = List[Bool](length=nn, fill=False)
        for i in range(kk):
            mark[perm[i]] = True
        var at = 0
        for r in range(nn):
            if mark[r]:
                dp.unsafe_store(at, Int32(r))
                at += 1
    return PythonObject(0)


def weighted_draw_rows_i32_binding(
    w_addr: PythonObject, n: PythonObject, k: PythonObject, seed_lo: PythonObject, seed_hi: PythonObject,
    dst_addr: PythonObject,
) raises -> PythonObject:
    """k rows drawn with replacement in proportion to the float32 weights:
    row = the first whose binary64 running sum exceeds u * total, u the
    53-bit uniform of the next splitmix64 draw, clipped to n - 1 (KBins'
    weighted subsample; lane pyglue-numeric: the Python `bisect` loop, the
    same arithmetic). int32 rows at dst, in draw order."""
    var nn = Int(py=n)
    var kk = Int(py=k)
    if nn < 1 or kk < 0:
        raise Error("weighted_draw_rows_i32: n >= 1, k >= 0")
    var state = (UInt64(Int(py=seed_hi)) << 32) | UInt64(Int(py=seed_lo))
    var wp = _ptr[DType.float32](Int(py=w_addr))
    var dp = _ptr[DType.int32](Int(py=dst_addr))
    with GILReleased(Python()):
        var cum = List[Float64](capacity=nn)
        var acc = Float64(0)
        for i in range(nn):
            acc += Float64(wp.unsafe_load(i))
            cum.append(acc)
        for t in range(kk):
            var z = _sm64_next(state)
            var target = Float64(z >> 11) * 1.1102230246251565e-16 * acc
            # bisect_right: the first index whose running sum exceeds target
            var lo = 0
            var hi = nn
            while lo < hi:
                var mid = (lo + hi) // 2
                if target < cum[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            dp.unsafe_store(t, Int32(min(lo, nn - 1)))
    return PythonObject(0)


def weighted_pick_i32_binding(
    m_addr: PythonObject, dk: PythonObject, col: PythonObject, k: PythonObject, state_addr: PythonObject,
    dst_addr: PythonObject,
) raises -> PythonObject:
    """IterativeImputer's n_nearest_features draw for feature `col`: k of
    the dk features drawn without replacement with probability M[:, col]
    (float32, row-major dk x dk): each draw u = the 53-bit uniform of the
    next splitmix64 word (`state`, uint64, advanced in place) times the
    binary64 total of the positive weights not yet drawn, the first such
    feature whose running sum exceeds u (the last one when none does). The
    k picks ascending, int32 at dst (lane pyglue-numeric: a Python loop)."""
    var n = Int(py=dk)
    var j = Int(py=col)
    var kk = Int(py=k)
    if n < 1 or j < 0 or j >= n or kk < 0:
        raise Error("weighted_pick_i32: bad sizes")
    var mp = _ptr[DType.float32](Int(py=m_addr))
    var sp = _ptr[DType.uint64](Int(py=state_addr))
    var dp = _ptr[DType.int32](Int(py=dst_addr))
    var state = sp.unsafe_load(0)
    var chosen = List[Bool](length=n, fill=False)
    for _ in range(kk):
        var tot = Float64(0)
        var last = -1
        for a in range(n):
            var w = Float64(mp.unsafe_load(a * n + j))
            if not chosen[a] and w > 0:
                tot += w
                last = a
        if last < 0:
            raise Error("weighted_pick_i32: no feature left to draw")
        var u = Float64(_sm64_next(state) >> 11) * 1.1102230246251565e-16 * tot
        var pick = last
        var cum = Float64(0)
        for a in range(n):
            var w = Float64(mp.unsafe_load(a * n + j))
            if not chosen[a] and w > 0:
                cum += w
                if cum > u:
                    pick = a
                    break
        chosen[pick] = True
    sp.unsafe_store(0, state)
    var at = 0
    for a in range(n):
        if chosen[a]:
            dp.unsafe_store(at, Int32(a))
            at += 1
    return PythonObject(at)


def ocsvm_alpha_init_f32_binding(
    c_addr: PythonObject, m: PythonObject, nu: PythonObject, weighted: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """libsvm's solve_one_class start: nu_l = nu * m (or, `weighted`, the
    sum of C_i * nu in sample order, binary64), then alpha_i = min(C_i,
    nu_l) while nu_l > 0, nu_l reduced by each (the rest 0). float32 C_i
    in, float32 alpha out (lane pyglue-numeric: a Python loop over the
    samples, the same arithmetic)."""
    var mm = Int(py=m)
    if mm < 0:
        raise Error("ocsvm_alpha_init_f32: m >= 0")
    var v = Float64(py=nu)
    var cp = _ptr[DType.float32](Int(py=c_addr))
    var dp = _ptr[DType.float32](Int(py=dst_addr))
    var nl = v * Float64(mm)
    if Int(py=weighted) != 0:
        nl = Float64(0)
        for i in range(mm):
            nl += Float64(cp.unsafe_load(i)) * v
    for i in range(mm):
        var a = Float64(0)
        if nl > 0:
            a = min(Float64(cp.unsafe_load(i)), nl)
            nl -= a
        dp.unsafe_store(i, Float32(a))
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
