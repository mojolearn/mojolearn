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
from core.hotpath_impl_rows import (
    assign_fold_i64,
    bincount2_i32,
    bincount_i64,
    count_fold_hits_i64,
    first_seen_i32,
    fold_pair_f32,
    group_fold_assign_i32,
    mark_split_roles_f32,
    normal_init_f32,
    scatter_rows_bytes_host,
    strat_alloc_i64,
    strat_fold_assign_i32,
    strat_group_plan_i32,
    threshold_labels_i64,
    uniform_init_f32,
)
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


# ---------------------------------------------------------------------------
# fold_pair_f32 onward: the Python door only. The typed kernels are in
# core/hotpath_impl_rows.mojo (lane thin-core, 2026-10-21: a pure move).
# ---------------------------------------------------------------------------


def fold_pair_f32_binding(dst_addr: PythonObject, src_addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """cross_vendor's coordinator fold (lane cgr4-py-compute, out of the
    Python element loop): dst[i] = ftz(ftz(dst[i]) + ftz(src[i])), one
    float32 rounding per add, elementwise (no element depends on another)."""
    var count = Int(py=n)
    var d = Int(py=dst_addr)
    var s = Int(py=src_addr)
    var r: Int
    with GILReleased(Python()):
        r = fold_pair_f32(d, s, count)
    return PythonObject(r)


def threshold_labels_i64_binding(
    src_addr: PythonObject, code: PythonObject, n: PythonObject, thr: PythonObject,
    strict: PythonObject, below: PythonObject, above: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """dst[i] = above if src[i] > thr (strict) / >= thr (not strict), else
    below; int64 out from a float32 (code 0) or float64 (code 1) score (lane
    cgr4-py-compute: the predict label maps that were Python
    comprehensions). A NaN score compares false and takes `below`."""
    var count = Int(py=n)
    var c = Int(py=code)
    var t = Float64(py=thr)
    var st = Int(py=strict) != 0
    var lo = Int64(Int(py=below))
    var hi = Int64(Int(py=above))
    var d = Int(py=dst_addr)
    var s = Int(py=src_addr)
    var r: Int
    with GILReleased(Python()):
        r = threshold_labels_i64(s, c, count, t, st, lo, hi, d)
    return PythonObject(r)


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
    var cp = Int(py=counts_addr)
    var acc = Int(py=accumulate) != 0
    var s = Int(py=src_addr)
    var r: Int
    with GILReleased(Python()):
        r = bincount_i64(s, c, count, kk, cp, acc)
    return PythonObject(r)


def bincount2_i32_host_binding(
    a_addr: PythonObject, b_addr: PythonObject, dims: PythonObject, counts_addr: PythonObject,
) raises -> PythonObject:
    """dims = [n, ka, kb]: int64 counts[b * ka + a] over the n rows of two
    int32 code columns, on the host (the core host binding's
    `bincount2_i32`, the GPU base binding's twin). Raises when a code is
    outside [0, ka) x [0, kb)."""
    if len(dims) != 3:
        raise Error("bincount2_i32: dims [n, ka, kb]")
    var count = Int(py=dims[0])
    var ka = Int(py=dims[1])
    var kb = Int(py=dims[2])
    var ca = Int(py=counts_addr)
    if ca == 0:
        raise Error("bincount2_i32: null counts address")
    var a = Int(py=a_addr)
    var b = Int(py=b_addr)
    if count > 0 and (a == 0 or b == 0):
        raise Error("bincount2_i32: null code address")
    var r: Int
    with GILReleased(Python()):
        r = bincount2_i32(a, b, count, ka, kb, ca)
    return PythonObject(r)


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
    var ip = Int(py=idx_addr)
    var fp = Int(py=folds_addr)
    var r: Int
    with GILReleased(Python()):
        r = assign_fold_i64(ip, count, rows, f, fp)
    return PythonObject(r)


def count_fold_hits_i64_binding(
    idx_addr: PythonObject, m: PythonObject, folds_addr: PythonObject, n: PythonObject, fold: PythonObject,
) raises -> PythonObject:
    """How many of the m int64 indices (all in [0, n)) sit in fold `fold`
    of the int32 folds array."""
    var count = Int(py=m)
    var rows = Int(py=n)
    var f = Int32(Int(py=fold))
    var ip = Int(py=idx_addr)
    var fp = Int(py=folds_addr)
    var r: Int
    with GILReleased(Python()):
        r = count_fold_hits_i64(ip, count, fp, rows, f)
    return PythonObject(r)


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
    var ro = Int(py=role)
    var ip = Int(py=idx_addr)
    var rp = Int(py=roles_addr)
    var r: Int
    with GILReleased(Python()):
        r = mark_split_roles_f32(ip, count, rows, k, ro, rp)
    return PythonObject(r)


def scatter_rows_bytes_host_binding(
    src_addr: PythonObject, rows_addr: PythonObject, m: PythonObject, row_bytes: PythonObject,
    dst_addr: PythonObject, dst_rows: PythonObject,
) raises -> PythonObject:
    """dst row rows[i] = src row i (row_bytes each) for the m int64 rows; a
    row outside [0, dst_rows) raises before any write."""
    var count = Int(py=m)
    var width = Int(py=row_bytes)
    var nd = Int(py=dst_rows)
    var s = Int(py=src_addr)
    var d = Int(py=dst_addr)
    var rp = Int(py=rows_addr)
    var r: Int
    with GILReleased(Python()):
        r = scatter_rows_bytes_host(s, rp, count, width, d, nd)
    return PythonObject(r)


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
    var lo = Float64(py=low)
    var hi = Float64(py=high)
    var seed = (UInt64(Int(py=seed_hi)) << 32) | UInt64(Int(py=seed_lo))
    var off = UInt64(Int(py=offset))
    var d = Int(py=dst_addr)
    var r: Int
    with GILReleased(Python()):
        r = uniform_init_f32(d, count, lo, hi, seed, off)
    return PythonObject(r)


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
    var mu = Float64(py=mean)
    var sd = Float64(py=std)
    var seed = (UInt64(Int(py=seed_hi)) << 32) | UInt64(Int(py=seed_lo))
    var off = UInt64(Int(py=offset))
    var d = Int(py=dst_addr)
    var r: Int
    with GILReleased(Python()):
        r = normal_init_f32(d, count, mu, sd, seed, off)
    return PythonObject(r)


# TOMBSTONE: adam_hyper_f64 (dead native entry: no live caller) deleted 2026-10-26 by @@SHA@@; code recoverable at 485ba9cc1.
# Restore: git apply experiments/removed/adam_hyper_f64.patch


def first_seen_i32_binding(
    codes_addr: PythonObject, n: PythonObject, k: PythonObject, enc_addr: PythonObject, counts_addr: PythonObject,
) raises -> PythonObject:
    """Renumber n int32 class codes in [0, k) by first appearance: enc[i] =
    the first-seen rank of codes[i]; counts[r] (int64) = rows of rank r.
    Returns the number of classes present (StratifiedKFold's encoding)."""
    return PythonObject(
        first_seen_i32(Int(py=codes_addr), Int(py=n), Int(py=k), Int(py=enc_addr), Int(py=counts_addr))
    )


def strat_alloc_i64_binding(
    counts_addr: PythonObject, k: PythonObject, n_folds: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """StratifiedKFold's allocation table (sklearn _make_test_folds):
    dst[i * k + c] (int64, n_folds x k) = how many positions p of class c's
    run [s, e) in the class-sorted labels have p % n_folds == i, i.e.
    (e - 1 - i) // K - (s - 1 - i) // K with floor division (lane
    pyglue-numeric: a Python double loop over the folds and classes).
    `counts` int64, k of them."""
    return PythonObject(strat_alloc_i64(Int(py=counts_addr), Int(py=k), Int(py=n_folds), Int(py=dst_addr)))


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
    return PythonObject(
        group_fold_assign_i32(
            Int(py=counts_addr), Int(py=m), Int(py=n_folds), Int(py=perm_addr),
            Int(py=dst_addr), Int(py=sizes_addr),
        )
    )


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
    return PythonObject(
        strat_group_plan_i32(
            Int(py=addrs[0]), Int(py=addrs[1]), Int(py=addrs[2]), Int(py=addrs[3]),
            Int(py=dims[0]), Int(py=dims[1]), Int(py=dims[2]),
        )
    )


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
    return PythonObject(
        strat_fold_assign_i32(
            Int(py=enc_addr), Int(py=n), Int(py=k), Int(py=n_folds),
            Int(py=alloc_addr), Int(py=perms_addr), Int(py=counts_addr), Int(py=dst_addr),
        )
    )
