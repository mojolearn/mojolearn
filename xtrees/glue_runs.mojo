# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The x_trees glue operations as typed native runs, and their one native
entry (lane calltable-trees, 2026-11-11).

Each `<name>_run` is the body the Python entry `x_trees_<name>` ran
(moved verbatim from `providers/python/xtrees_api_pyapi.mojo`, whose entries
now only unpack into `XtOwned` and call it). Native callers in other
extension modules reach the same runs through `x_trees_glue_v1`, published
in the trees entry table at module init by both x_trees bindings
(`publish_x_trees_glue`), with no Python in between. No kernel, fold order
or route changed; the device/host choice is the build-time
`XTREES_DEVICE_OPS` it always was.

The runs are synchronous launches over caller-owned buffers; each kernel
parallelizes inside. The dispatch is a fixed comparison chain on the
operation code, no data work.
"""
from gbdt.trees_identical_switches import T30_ADABOOST
from core.hostptr import f32_ptr, f64_ptr, i32_ptr
from checks.numerics import identical_log64, GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.trees_entry_table import TreesEntryKey, trees_table_publish, TREES_TABLE_X_TREES
from xtrees.glue_abi import XtCall, XT_GLUE_ABI, XT_OP_COUNT, XT_ACCUMULATE, XT_ACCUMULATE_COLS, XT_ACCUMULATE_ONEHOT, XT_ACCUMULATE_ROWS, XT_APPLY, XT_ARGMAX_ROWS, XT_BAG_ROWS, XT_BINARY_PROBA, XT_CLASS_COUNTS, XT_CLASS_PRIORS, XT_COMPLEMENT_PAIRS, XT_COUNT_EQUAL, XT_COUNT_ROWS, XT_DART_RESCALE, XT_DART_THRESHOLDS, XT_DEVICE_FOLDS, XT_EXACT_SUM, XT_FILL_CLASS_MAJOR_F64, XT_GATHER_F32, XT_GATHER_I32, XT_INDICATOR_CODES, XT_IOTA_I32, XT_ISOTONIC_FIT, XT_ISOTONIC_PREDICT_STRIDED, XT_LEAF_NUMBERING, XT_LOG64, XT_MARGIN2, XT_NORMALIZE_ROWS, XT_ONEHOT_LEAVES, XT_OOB_R2, XT_PLATT_APPLY_STRIDED, XT_PLATT_FIT, XT_POSITIVE_CODES, XT_PUT_F32, XT_R2_STEP, XT_REMAP_COLS, XT_ROWS_ANY_UNSET, XT_SAMME_STEP, XT_SAMPLE_INDICES, XT_SCALE, XT_SCALE_TO_F32, XT_SCATTER, XT_SOFTMAX_ROWS, XT_STACK_W64, XT_TRANSPOSE_F32, XT_TRANSPOSE_F64, XT_UNIFORM, XT_UNSEEN_ROWS, XT_WEIGHTED_MEDIAN, XT_WEIGHTED_SAMPLE
from xtrees.folds_device import device_folds, leaf_numbering
from xtrees.glue_device import (
    binary_proba_device, indicator_codes_device, stack_w64_device, class_counts_device, remap_cols_device,
    rows_any_unset_device, positive_codes_device,
)
from xtrees.exact_sum import ES_FLAGS, ES_MAX_ROWS, exact_sum_device, exact_sum_host
from xtrees.oob import (
    E64_MAX_ROWS, count_equal_device, count_equal_host, count_rows_device, count_rows_host, oob_r2_device, oob_r2_host,
)
from xtrees.ops_device import (
    apply_trees_device, bag_rows_device, gather_f32_device, transpose_f32_device, transpose_f64_device,
    unseen_rows_device, weighted_sample_device,
)
from xtrees import ops_device_elem as elem_dev
from xtrees import ops_device_boost as boost_dev
from xtrees.api_kernels import class_priors, dart_rescale, dart_thresholds, host_folds, scatter_rows_check
from xtrees.ops import (
    sample_indices, weighted_sample, gather_f32, gather_i32, accumulate,
    accumulate_onehot, accumulate_cols, accumulate_rows, argmax_rows, scale_f64, softmax_rows, scale_to_f32, put_f32,
    samme_step, r2_step, weighted_median, apply_trees, uniform,
    onehot_leaves, transpose_f32, normalize_rows, scatter, platt_fit, isotonic_fit,
    platt_apply_strided, isotonic_predict_strided, complement_pairs, indicator_codes,
    bag_rows, unseen_rows, transpose_f64, stack_w64, binary_proba, class_counts, remap_cols, positive_codes,
    rows_any_unset, margin2, iota_i32, fill_class_major_f64,
)

#: cpu-gpu-cleanup t-gbdt / w2-trees: a GPU build gathers, applies, transposes
#: and draws weighted samples on the device (`xtrees/ops_device.mojo`); the
#: host binding (the CPU column, `-D MOJOLEARN_COLUMN_CPU`) keeps
#: `xtrees/ops.mojo`'s serial loops. Fixed at build time, never a runtime route.
comptime XTREES_DEVICE_OPS = TARGET_COLUMN != COLUMN_CPU


def _addrs(c: XtCall, n: Int, who: String) raises:
    if c.n_addrs != n:
        raise Error(who + ": expects " + String(n) + " buffer addresses, got " + String(c.n_addrs))


def _a(c: XtCall, k: Int) -> Int:
    return MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=c.addrs)[k]


def _need(c: XtCall, n: Int, who: String) raises:
    if c.n_params != n:
        raise Error(who + ": params must hold " + String(n) + " values, got " + String(c.n_params))


def _flag(c: XtCall, k: Int) -> UInt8:
    if c.bad == 0:
        return 0
    return MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=c.bad)[k]


def _i(c: XtCall, k: Int) raises -> Int:
    if k < 0 or k >= c.n_params:
        raise Error("x_trees glue: parameter " + String(k) + " is missing")
    if _flag(c, k) & 1:
        raise Error("TypeError: x_trees glue: parameter " + String(k) + " must be an integer")
    return MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=c.ints)[k]


def _f(c: XtCall, k: Int) raises -> Float64:
    if k < 0 or k >= c.n_params:
        raise Error("x_trees glue: parameter " + String(k) + " is missing")
    if _flag(c, k) & 2:
        raise Error("TypeError: x_trees glue: parameter " + String(k) + " must be a real number")
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=c.floats)[k]


def _count(v: Int, who: String) raises -> Int:
    if v < 0:
        raise Error(who + ": negative count")
    return v


def _strided(total: Int, stride: Int, off: Int, n: Int, who: String) raises:
    if stride < 1 or off < 0 or (n > 0 and off + (n - 1) * stride >= total):
        raise Error(who + ": stride or offset out of range")


def _i64_ptr(addr: Int) raises -> MutPointer[Int64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int64 buffer address")
    return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)


def log64_run(c: XtCall) raises -> Int:
    """The pinned binary64 log (checks/numerics.mojo identical_log64) of one
    value: addresses = [res float64 1], params = [x]."""
    _addrs(c, 1, "x_trees_log64")
    _need(c, 1, "x_trees_log64")
    f64_ptr(_a(c, 0))[0] = identical_log64(_f(c, 0))
    return 1


def stack_w64_run(c: XtCall) raises -> Int:
    """addresses = [dst, column 0, ..., column m-1]: m addresses of n 8-byte
    words each (int64 or float64); dst (n x m, row-major, the same word type)
    gets column j from column j; params = [n]. Words are moved, not computed
    (lane apple-fast-py2mojo-trees)."""
    if c.n_addrs < 1:
        raise Error("x_trees_stack_w64: needs the destination address")
    _need(c, 1, "x_trees_stack_w64")
    var n = _count(_i(c, 0), "x_trees_stack_w64")
    var addrs = List[Int]()
    for j in range(1, c.n_addrs):  # small-loop(one address word per column)
        var a = _a(c, j)
        if a == 0:
            raise Error("x_trees_stack_w64: null column address")
        addrs.append(a)
    if n > 0 and len(addrs) > 0:
        var d = f64_ptr(_a(c, 0)).bitcast[UInt64]()
        comptime if XTREES_DEVICE_OPS:
            stack_w64_device(addrs, n, d)
        else:
            stack_w64(addrs, n, d)
    return n * len(addrs)


def oob_r2_run(c: XtCall) raises -> Int:
    """addresses = [acc f64 n, counts i32 n, y f32 n, pred f64 n (out), words
    f64 4 (out), flags i32 4 (out)], params = [n]: pred = acc / max(counts, 1),
    words = [fsum(y), fsum((y - mean)^2), fsum((y - pred)^2), mean], flags
    [non-finite term, overflow] (BaggingRegressor's oob R^2; xtrees/oob.mojo)."""
    _need(c, 1, "x_trees_oob_r2")
    if c.n_addrs != 6:
        raise Error("x_trees_oob_r2: bufs must hold 6 addresses")
    var n = _count(_i(c, 0), "x_trees_oob_r2")
    if n < 1 or n > E64_MAX_ROWS:
        raise Error("x_trees_oob_r2: needs 1 <= n <= 2^30")
    comptime if XTREES_DEVICE_OPS:
        oob_r2_device(f64_ptr(_a(c, 0)), i32_ptr(_a(c, 1)), f32_ptr(_a(c, 2)), n,
                      f64_ptr(_a(c, 3)), f64_ptr(_a(c, 4)), i32_ptr(_a(c, 5)))
    else:
        oob_r2_host(f64_ptr(_a(c, 0)), i32_ptr(_a(c, 1)), f32_ptr(_a(c, 2)), n,
                    f64_ptr(_a(c, 3)), f64_ptr(_a(c, 4)), i32_ptr(_a(c, 5)))
    return n


def dart_thresholds_run(c: XtCall) raises -> Int:
    """One DART round's drop thresholds and float32 coefficients from the
    t tree weights (float64), the t * k coefficients (float64) and state[0] =
    the weight sum. addresses = [weights, coefs, coef32, thr, state]; params =
    [t, k, uniform_drop, max_drop, drop_rate]. thr (int64, t) = ceil(p_i *
    2^53), p_i = rate * w_i * (t / sum_w) (rate capped by max_drop * inv_avg /
    sum_w), or the capped uniform rate; coef32 (float32, t * k) = the
    coefficients rounded once. Returns t."""
    _addrs(c, 5, "x_trees_dart_thresholds")
    _need(c, 5, "x_trees_dart_thresholds")
    var t = _count(_i(c, 0), "x_trees_dart_thresholds")
    var k = _i(c, 1)
    var uniform_drop = _i(c, 2) != 0
    var max_drop = _i(c, 3)
    if t == 0:
        return 0
    var wp = f64_ptr(_a(c, 0))
    var cp = f64_ptr(_a(c, 1))
    var c32 = f32_ptr(_a(c, 2))
    var tp = _i64_ptr(_a(c, 3))
    var sum_w = f64_ptr(_a(c, 4))[0]
    var r = _f(c, 4)
    dart_thresholds(wp, cp, c32, tp, sum_w, t, k, uniform_drop, max_drop, r)
    return t


def dart_rescale_run(c: XtCall) raises -> Int:
    """The round's drop count k_d from the step's flags (int32, t), then the
    shrink, the dropped trees' coefficient and weight rescale (ascending
    tree order, the weight sum state[0] updated per dropped tree) and the
    new round's weight w[t] = shrink, coefficients t * k .. t * k + k - 1 =
    shrink and sum_w += shrink. addresses = [weights, coefs, flags, state,
    res float64 3]; params = [t, k, uniform_drop, xgboost_dart_mode,
    learning_rate]. res = [k_d, shrink, factor]; returns k_d."""
    _addrs(c, 5, "x_trees_dart_rescale")
    _need(c, 5, "x_trees_dart_rescale")
    var t = _count(_i(c, 0), "x_trees_dart_rescale")
    var k = _i(c, 1)
    if k < 1:
        raise Error("x_trees_dart_rescale: needs k >= 1")
    var uniform_drop = _i(c, 2) != 0
    var xgb = _i(c, 3) != 0
    var l = _f(c, 4)
    var wp = f64_ptr(_a(c, 0))
    var cp = f64_ptr(_a(c, 1))
    var sp = f64_ptr(_a(c, 3))
    var r3 = dart_rescale(wp, cp, sp, _a(c, 2) if t > 0 else 0, t, k, uniform_drop, xgb, l)
    var res = f64_ptr(_a(c, 4))
    res[0] = Float64(r3[0])
    res[1] = r3[1]
    res[2] = r3[2]
    return r3[0]


def sample_indices_run(c: XtCall) raises -> Int:
    """params = [n_pool, n_draw, replace, seed, stream]."""
    _addrs(c, 1, "x_trees_sample_indices")
    var out_addr = _a(c, 0)
    _need(c, 5, "x_trees_sample_indices")
    var n_draw = _count(_i(c, 1), "x_trees_sample_indices")
    if n_draw > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.sample_indices_device(i32_ptr(out_addr), _i(c, 0), n_draw, _i(c, 2) != 0,
                                           _i(c, 3), _i(c, 4))
        else:
            sample_indices(i32_ptr(out_addr), _i(c, 0), n_draw, _i(c, 2) != 0,
                           _i(c, 3), _i(c, 4))
    return n_draw


def weighted_sample_run(c: XtCall) raises -> Int:
    """params = [n, n_draw, seed, stream]."""
    _addrs(c, 2, "x_trees_weighted_sample")
    var w_addr = _a(c, 0)
    var out_addr = _a(c, 1)
    _need(c, 4, "x_trees_weighted_sample")
    var n = _i(c, 0)
    if n <= 0:
        raise Error("x_trees_weighted_sample: n must be positive")
    var n_draw = _count(_i(c, 1), "x_trees_weighted_sample")
    comptime if XTREES_DEVICE_OPS:
        weighted_sample_device(f64_ptr(w_addr), n, i32_ptr(out_addr), n_draw, _i(c, 2), _i(c, 3))
    else:
        weighted_sample(f64_ptr(w_addr), n, i32_ptr(out_addr), n_draw, _i(c, 2), _i(c, 3))
    return n_draw


def gather_f32_run(c: XtCall) raises -> Int:
    """params = [n_src_rows, n_src_cols, n_rows, n_cols]."""
    _addrs(c, 4, "x_trees_gather_f32")
    var src = _a(c, 0)
    var rows = _a(c, 1)
    var cols = _a(c, 2)
    var dst = _a(c, 3)
    _need(c, 4, "x_trees_gather_f32")
    var n_rows = _count(_i(c, 2), "x_trees_gather_f32")
    var n_cols = _count(_i(c, 3), "x_trees_gather_f32")
    if n_rows * n_cols > 0:
        comptime if XTREES_DEVICE_OPS:
            gather_f32_device(f32_ptr(src), _i(c, 0), _i(c, 1), i32_ptr(rows), n_rows,
                              i32_ptr(cols), n_cols, f32_ptr(dst))
        else:
            gather_f32(f32_ptr(src), _i(c, 0), _i(c, 1), i32_ptr(rows), n_rows,
                       i32_ptr(cols), n_cols, f32_ptr(dst))
    return n_rows * n_cols


def gather_i32_run(c: XtCall) raises -> Int:
    """params = [n_src, n_rows]."""
    _addrs(c, 3, "x_trees_gather_i32")
    var src = _a(c, 0)
    var rows = _a(c, 1)
    var dst = _a(c, 2)
    _need(c, 2, "x_trees_gather_i32")
    var n_rows = _count(_i(c, 1), "x_trees_gather_i32")
    if n_rows > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.gather_i32_device(i32_ptr(src), _i(c, 0), i32_ptr(rows), n_rows, i32_ptr(dst))
        else:
            gather_i32(i32_ptr(src), _i(c, 0), i32_ptr(rows), n_rows, i32_ptr(dst))
    return n_rows


def accumulate_run(c: XtCall) raises -> Int:
    """acc (float64) += weight * x (float32); params = [n, weight]."""
    _addrs(c, 2, "x_trees_accumulate")
    var acc = _a(c, 0)
    var x = _a(c, 1)
    _need(c, 2, "x_trees_accumulate")
    var n = _count(_i(c, 0), "x_trees_accumulate")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.accumulate_device(f64_ptr(acc), f32_ptr(x), n, _f(c, 1))
        else:
            accumulate(f64_ptr(acc), f32_ptr(x), n, _f(c, 1))
    return n


def accumulate_cols_run(c: XtCall) raises -> Int:
    """acc (float64, n*k) += weight * x (float32, n*ks) into columns cols; params = [n, k, ks, weight]."""
    _addrs(c, 3, "x_trees_accumulate_cols")
    var acc = _a(c, 0)
    var x = _a(c, 1)
    var cols = _a(c, 2)
    _need(c, 4, "x_trees_accumulate_cols")
    var n = _count(_i(c, 0), "x_trees_accumulate_cols")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.accumulate_cols_device(f64_ptr(acc), f32_ptr(x), i32_ptr(cols), n,
                                            _i(c, 1), _i(c, 2), _f(c, 3))
        else:
            accumulate_cols(f64_ptr(acc), f32_ptr(x), i32_ptr(cols), n, _i(c, 1),
                            _i(c, 2), _f(c, 3))
    return n


def accumulate_rows_run(c: XtCall) raises -> Int:
    """acc (n x k float64) += x (m x k float64) at rows; params = [n, k, m]."""
    _addrs(c, 3, "x_trees_accumulate_rows")
    var acc = _a(c, 0)
    var x = _a(c, 1)
    var rows = _a(c, 2)
    _need(c, 3, "x_trees_accumulate_rows")
    var n = _count(_i(c, 0), "x_trees_accumulate_rows")
    var k = _count(_i(c, 1), "x_trees_accumulate_rows")
    var m = _count(_i(c, 2), "x_trees_accumulate_rows")
    if m * k > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.accumulate_rows_device(f64_ptr(acc), n, k, f64_ptr(x), i32_ptr(rows), m)
        else:
            accumulate_rows(f64_ptr(acc), n, k, f64_ptr(x), i32_ptr(rows), m)
    return m


def accumulate_onehot_run(c: XtCall) raises -> Int:
    """params = [n, k, on, off]."""
    _addrs(c, 2, "x_trees_accumulate_onehot")
    var acc = _a(c, 0)
    var codes = _a(c, 1)
    _need(c, 4, "x_trees_accumulate_onehot")
    var n = _count(_i(c, 0), "x_trees_accumulate_onehot")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.accumulate_onehot_device(f64_ptr(acc), i32_ptr(codes), n, _i(c, 1),
                                              _f(c, 2), _f(c, 3))
        else:
            accumulate_onehot(f64_ptr(acc), i32_ptr(codes), n, _i(c, 1), _f(c, 2), _f(c, 3))
    return n


def argmax_rows_run(c: XtCall) raises -> Int:
    """float64 x; params = [n, k]."""
    _addrs(c, 2, "x_trees_argmax_rows")
    var x = _a(c, 0)
    var res = _a(c, 1)
    _need(c, 2, "x_trees_argmax_rows")
    var n = _count(_i(c, 0), "x_trees_argmax_rows")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.argmax_rows_device(f64_ptr(x), n, _i(c, 1), i32_ptr(res))
        else:
            argmax_rows(f64_ptr(x), n, _i(c, 1), i32_ptr(res))
    return n


def scale_run(c: XtCall) raises -> Int:
    """x /= divisor; params = [n, divisor]."""
    _addrs(c, 1, "x_trees_scale")
    var x = _a(c, 0)
    _need(c, 2, "x_trees_scale")
    var n = _count(_i(c, 0), "x_trees_scale")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.scale_f64_device(f64_ptr(x), n, _f(c, 1))
        else:
            scale_f64(f64_ptr(x), n, _f(c, 1))
    return n


def scale_to_f32_run(c: XtCall) raises -> Int:
    """res (float32) = x (float64) * factor; params = [n, factor]."""
    _addrs(c, 2, "x_trees_scale_to_f32")
    var x = _a(c, 0)
    var res = _a(c, 1)
    _need(c, 2, "x_trees_scale_to_f32")
    var n = _count(_i(c, 0), "x_trees_scale_to_f32")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.scale_to_f32_device(f64_ptr(x), n, _f(c, 1), f32_ptr(res))
        else:
            scale_to_f32(f64_ptr(x), n, _f(c, 1), f32_ptr(res))
    return n


def margin2_run(c: XtCall) raises -> Int:
    """Two-class vote rows (n x 2, float64) to the SAMME margin
    d = acc[2i+1] - acc[2i], one IEEE binary64 subtraction per row, the
    value the Python `v[2*i+1] - v[2*i]` computed. params = [n, mode]:
    mode 0 writes d (float64, n); mode 1 writes the int32 code
    `1 if d > 0 else 0` (a NaN gives 0, as `>` did); mode 2 writes the
    float64 pairs (-(d/2), d/2) (n x 2), the rows `predict_proba` softmaxes."""
    _addrs(c, 2, "x_trees_margin2")
    var acc = _a(c, 0)
    var dst = _a(c, 1)
    _need(c, 2, "x_trees_margin2")
    var n = _count(_i(c, 0), "x_trees_margin2")
    var mode = _i(c, 1)
    comptime if XTREES_DEVICE_OPS:
        elem_dev.margin2_device(f64_ptr(acc), n, mode, f64_ptr(dst), i32_ptr(dst))
    else:
        margin2(f64_ptr(acc), n, mode, f64_ptr(dst), i32_ptr(dst))
    return n


def put_f32_run(c: XtCall) raises -> Int:
    """dst[offset:offset+n] = src; params = [offset, n]."""
    _addrs(c, 2, "x_trees_put_f32")
    var dst = _a(c, 0)
    var src = _a(c, 1)
    _need(c, 2, "x_trees_put_f32")
    var n = _count(_i(c, 1), "x_trees_put_f32")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.put_f32_device(f32_ptr(dst), _count(_i(c, 0), "x_trees_put_f32"), f32_ptr(src), n)
        else:
            put_f32(f32_ptr(dst), _count(_i(c, 0), "x_trees_put_f32"), f32_ptr(src), n)
    return n


def softmax_rows_run(c: XtCall) raises -> Int:
    _addrs(c, 1, "x_trees_softmax_rows")
    var x = _a(c, 0)
    _need(c, 2, "x_trees_softmax_rows")
    var n = _count(_i(c, 0), "x_trees_softmax_rows")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.softmax_rows_device(f64_ptr(x), n, _i(c, 1))
        else:
            softmax_rows(f64_ptr(x), n, _i(c, 1))
    return n


def samme_step_run(c: XtCall) raises -> Int:
    """params = [n, n_classes, learning_rate, last]; stats = 4 float64."""
    _addrs(c, 4, "x_trees_samme_step")
    var w = _a(c, 0)
    var pred = _a(c, 1)
    var y = _a(c, 2)
    var stats = _a(c, 3)
    if c.n_params != 4 and c.n_params != 5:
        raise Error("x_trees_samme_step: expected 4 values plus optional tree-base metadata")
    var n = _i(c, 0)
    if n <= 0:
        raise Error("x_trees_samme_step: n must be positive")
    # T30 tree-base metadata only; custom non-tree estimators retain B.
    # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    var cache_tree = False
    if c.n_params > 4:
        cache_tree = _i(c, 4) != 0
    if T30_ADABOOST and cache_tree:
        comptime if XTREES_DEVICE_OPS:
            boost_dev.samme_step_device[True](f64_ptr(w), i32_ptr(pred), i32_ptr(y), n, _i(c, 1),
                                        _f(c, 2), _i(c, 3) != 0, f64_ptr(stats))
        else:
            samme_step[True](f64_ptr(w), i32_ptr(pred), i32_ptr(y), n, _i(c, 1),
                       _f(c, 2), _i(c, 3) != 0, f64_ptr(stats))
    else:
        comptime if XTREES_DEVICE_OPS:
            boost_dev.samme_step_device[False](f64_ptr(w), i32_ptr(pred), i32_ptr(y), n, _i(c, 1),
                                        _f(c, 2), _i(c, 3) != 0, f64_ptr(stats))
        else:
            samme_step[False](f64_ptr(w), i32_ptr(pred), i32_ptr(y), n, _i(c, 1),
                       _f(c, 2), _i(c, 3) != 0, f64_ptr(stats))
    return n


def r2_step_run(c: XtCall) raises -> Int:
    """params = [n, loss (0 linear, 1 square, 2 exponential), learning_rate, last]."""
    _addrs(c, 4, "x_trees_r2_step")
    var w = _a(c, 0)
    var pred = _a(c, 1)
    var y = _a(c, 2)
    var stats = _a(c, 3)
    if c.n_params != 4 and c.n_params != 5:
        raise Error("x_trees_r2_step: expected 4 values plus optional tree-base metadata")
    var n = _i(c, 0)
    if n <= 0:
        raise Error("x_trees_r2_step: n must be positive")
    # T30 tree-base metadata only; custom non-tree estimators retain B.
    # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    var cache_tree = False
    if c.n_params > 4:
        cache_tree = _i(c, 4) != 0
    if T30_ADABOOST and cache_tree:
        comptime if XTREES_DEVICE_OPS:
            boost_dev.r2_step_device[True](f64_ptr(w), f32_ptr(pred), f32_ptr(y), n, _i(c, 1),
                                     _f(c, 2), _i(c, 3) != 0, f64_ptr(stats))
        else:
            r2_step[True](f64_ptr(w), f32_ptr(pred), f32_ptr(y), n, _i(c, 1),
                    _f(c, 2), _i(c, 3) != 0, f64_ptr(stats))
    else:
        comptime if XTREES_DEVICE_OPS:
            boost_dev.r2_step_device[False](f64_ptr(w), f32_ptr(pred), f32_ptr(y), n, _i(c, 1),
                                     _f(c, 2), _i(c, 3) != 0, f64_ptr(stats))
        else:
            r2_step[False](f64_ptr(w), f32_ptr(pred), f32_ptr(y), n, _i(c, 1),
                    _f(c, 2), _i(c, 3) != 0, f64_ptr(stats))
    return n


def weighted_median_run(c: XtCall) raises -> Int:
    """params = [n, m]; preds estimator-major float32."""
    _addrs(c, 3, "x_trees_weighted_median")
    var preds = _a(c, 0)
    var weights = _a(c, 1)
    var res = _a(c, 2)
    _need(c, 2, "x_trees_weighted_median")
    var n = _count(_i(c, 0), "x_trees_weighted_median")
    var m = _i(c, 1)
    if m <= 0:
        raise Error("x_trees_weighted_median: need at least one estimator")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            boost_dev.weighted_median_device(f32_ptr(preds), f64_ptr(weights), n, m, f32_ptr(res))
        else:
            weighted_median(f32_ptr(preds), f64_ptr(weights), n, m, f32_ptr(res))
    return n


def apply_run(c: XtCall) raises -> Int:
    """Leaf node per (row, tree) for trees [t0, t1); params = [n, d, t0, t1]; x row-major float32."""
    _addrs(c, 6, "x_trees_apply")
    var offsets = _a(c, 0)
    var colid = _a(c, 1)
    var quesval = _a(c, 2)
    var left = _a(c, 3)
    var x = _a(c, 4)
    var res = _a(c, 5)
    _need(c, 4, "x_trees_apply")
    var n = _count(_i(c, 0), "x_trees_apply")
    var t0 = _count(_i(c, 2), "x_trees_apply")
    var t1 = _i(c, 3)
    if t1 <= t0:
        raise Error("x_trees_apply: need t1 > t0")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            apply_trees_device(i32_ptr(offsets), i32_ptr(colid), f32_ptr(quesval),
                               i32_ptr(left), f32_ptr(x), n, _i(c, 1), t0, t1, i32_ptr(res))
        else:
            apply_trees(i32_ptr(offsets), i32_ptr(colid), f32_ptr(quesval),
                        i32_ptr(left), f32_ptr(x), n, _i(c, 1), t0, t1, i32_ptr(res))
    return n


def uniform_run(c: XtCall) raises -> Int:
    """params = [n, seed, stream]."""
    _addrs(c, 1, "x_trees_uniform")
    var res = _a(c, 0)
    _need(c, 3, "x_trees_uniform")
    var n = _count(_i(c, 0), "x_trees_uniform")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.uniform_device(f64_ptr(res), n, _i(c, 1), _i(c, 2))
        else:
            uniform(f64_ptr(res), n, _i(c, 1), _i(c, 2))
    return n


def onehot_leaves_run(c: XtCall) raises -> Int:
    """params = [n, n_trees, n_cols]; res float64 n*n_cols, zeroed."""
    _addrs(c, 4, "x_trees_onehot_leaves")
    var nodes = _a(c, 0)
    var tree_base = _a(c, 1)
    var node_col = _a(c, 2)
    var res = _a(c, 3)
    _need(c, 3, "x_trees_onehot_leaves")
    var n = _count(_i(c, 0), "x_trees_onehot_leaves")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.onehot_leaves_device(i32_ptr(nodes), i32_ptr(tree_base), i32_ptr(node_col),
                                          n, _i(c, 1), _i(c, 2), f64_ptr(res))
        else:
            onehot_leaves(i32_ptr(nodes), i32_ptr(tree_base), i32_ptr(node_col), n,
                          _i(c, 1), _i(c, 2), f64_ptr(res))
    return n


def transpose_f32_run(c: XtCall) raises -> Int:
    """dst (d x n) = src (n x d) transposed; params = [n, d]."""
    _addrs(c, 2, "x_trees_transpose_f32")
    var src = _a(c, 0)
    var dst = _a(c, 1)
    _need(c, 2, "x_trees_transpose_f32")
    var n = _count(_i(c, 0), "x_trees_transpose_f32")
    var d = _count(_i(c, 1), "x_trees_transpose_f32")
    if n * d > 0:
        comptime if XTREES_DEVICE_OPS:
            transpose_f32_device(f32_ptr(src), n, d, f32_ptr(dst))
        else:
            transpose_f32(f32_ptr(src), n, d, f32_ptr(dst))
    return n * d


def transpose_f64_run(c: XtCall) raises -> Int:
    """dst (d x n) = src (n x d) transposed, float64 words; params = [n, d]."""
    _addrs(c, 2, "x_trees_transpose_f64")
    var src = _a(c, 0)
    var dst = _a(c, 1)
    _need(c, 2, "x_trees_transpose_f64")
    var n = _count(_i(c, 0), "x_trees_transpose_f64")
    var d = _count(_i(c, 1), "x_trees_transpose_f64")
    if n * d > 0:
        comptime if XTREES_DEVICE_OPS:
            transpose_f64_device(f64_ptr(src), n, d, f64_ptr(dst))
        else:
            transpose_f64(f64_ptr(src), n, d, f64_ptr(dst))
    return n * d


def bag_rows_run(c: XtCall) raises -> Int:
    """res (int32, n) <- the rows whose draw is below frac, ascending (none:
    the smallest draw's row); params = [n, seed, stream, frac]; returns the
    count."""
    _addrs(c, 1, "x_trees_bag_rows")
    var res = _a(c, 0)
    _need(c, 4, "x_trees_bag_rows")
    var n = _count(_i(c, 0), "x_trees_bag_rows")
    if n == 0:
        return 0
    comptime if XTREES_DEVICE_OPS:
        return bag_rows_device(i32_ptr(res), n, _i(c, 1), _i(c, 2), _f(c, 3))
    else:
        return bag_rows(i32_ptr(res), n, _i(c, 1), _i(c, 2), _f(c, 3))


def unseen_rows_run(c: XtCall) raises -> Int:
    """res (int32, n) <- the rows of [0, n) absent from rows (int32, m),
    ascending; params = [m, n]; returns the count."""
    _addrs(c, 2, "x_trees_unseen_rows")
    var rows = _a(c, 0)
    var res = _a(c, 1)
    _need(c, 2, "x_trees_unseen_rows")
    var m = _count(_i(c, 0), "x_trees_unseen_rows")
    var n = _count(_i(c, 1), "x_trees_unseen_rows")
    if n == 0:
        return 0
    var rp = i32_ptr(rows) if m > 0 else i32_ptr(res)
    comptime if XTREES_DEVICE_OPS:
        return unseen_rows_device(rp, m, n, i32_ptr(res))
    else:
        return unseen_rows(rp, m, n, i32_ptr(res))


def normalize_rows_run(c: XtCall) raises -> Int:
    """params = [n, k]."""
    _addrs(c, 1, "x_trees_normalize_rows")
    var x = _a(c, 0)
    _need(c, 2, "x_trees_normalize_rows")
    var n = _count(_i(c, 0), "x_trees_normalize_rows")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.normalize_rows_device(f64_ptr(x), n, _i(c, 1))
        else:
            normalize_rows(f64_ptr(x), n, _i(c, 1))
    return n


def scatter_run(c: XtCall) raises -> Int:
    """dst (float64, N x n_dst_cols)[rows[r], col0 + j] = src (float32, m x c)[r, j];
    params = [n_dst_rows, n_dst_cols, m, c, col0]."""
    _addrs(c, 3, "x_trees_scatter")
    var dst = _a(c, 0)
    var src = _a(c, 1)
    var rows = _a(c, 2)
    _need(c, 5, "x_trees_scatter")
    var n_dst = _i(c, 0)
    var n_cols = _i(c, 1)
    var m = _count(_i(c, 2), "x_trees_scatter")
    var c = _count(_i(c, 3), "x_trees_scatter")
    var col0 = _count(_i(c, 4), "x_trees_scatter")
    if col0 + c > n_cols:
        raise Error("x_trees_scatter: columns out of range")
    var rp = i32_ptr(rows) if m > 0 else i32_ptr(1)
    comptime if XTREES_DEVICE_OPS:
        # the row refusal is the device's check kernel (lane cpu2-l5-trees)
        elem_dev.scatter_device(f64_ptr(dst), n_dst, n_cols, f32_ptr(src), m, c, rp, col0)
    else:
        scatter_rows_check(rp, m, n_dst)
        if m * c > 0:
            scatter(f64_ptr(dst), n_cols, f32_ptr(src), m, c, rp, col0)
    return m * c


def platt_fit_run(c: XtCall) raises -> Int:
    """params = [n]; y int32 0/1; ab float64[2]."""
    _addrs(c, 3, "x_trees_platt_fit")
    var f = _a(c, 0)
    var y = _a(c, 1)
    var ab = _a(c, 2)
    _need(c, 1, "x_trees_platt_fit")
    var n = _i(c, 0)
    if n < 1:
        raise Error("x_trees_platt_fit: no rows")
    comptime if XTREES_DEVICE_OPS:
        boost_dev.platt_fit_device(f64_ptr(f), i32_ptr(y), n, f64_ptr(ab))
    else:
        platt_fit(f64_ptr(f), i32_ptr(y), n, f64_ptr(ab))
    return n


def isotonic_fit_run(c: XtCall) raises -> Int:
    """params = [n]; kx, ky float64[n] receive the knots; returns their count."""
    _addrs(c, 4, "x_trees_isotonic_fit")
    var x = _a(c, 0)
    var y = _a(c, 1)
    var kx = _a(c, 2)
    var ky = _a(c, 3)
    _need(c, 1, "x_trees_isotonic_fit")
    var n = _i(c, 0)
    var m: Int
    comptime if XTREES_DEVICE_OPS:
        m = boost_dev.isotonic_fit_device(f64_ptr(x), f64_ptr(y), n, f64_ptr(kx),
                                          f64_ptr(ky))
    else:
        m = isotonic_fit(f64_ptr(x), f64_ptr(y), n, f64_ptr(kx), f64_ptr(ky))
    return m


def platt_apply_strided_run(c: XtCall) raises -> Int:
    """params = [n, A, B, f_len, f_stride, f_off, res_len, res_stride, res_off]."""
    _addrs(c, 2, "x_trees_platt_apply_strided")
    var f = _a(c, 0)
    var res = _a(c, 1)
    _need(c, 9, "x_trees_platt_apply_strided")
    var n = _count(_i(c, 0), "x_trees_platt_apply_strided")
    var fs = _i(c, 4)
    var fo = _i(c, 5)
    var rs = _i(c, 7)
    var ro = _i(c, 8)
    _strided(_i(c, 3), fs, fo, n, "x_trees_platt_apply_strided")
    _strided(_i(c, 6), rs, ro, n, "x_trees_platt_apply_strided")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            boost_dev.platt_apply_strided_device(f64_ptr(f) + fo, fs, n, _f(c, 1), _f(c, 2),
                                                 f64_ptr(res) + ro, rs)
        else:
            platt_apply_strided(f64_ptr(f) + fo, fs, n, _f(c, 1), _f(c, 2), f64_ptr(res) + ro, rs)
    return n


def isotonic_predict_strided_run(c: XtCall) raises -> Int:
    """params = [m, n, t_len, t_stride, t_off, res_len, res_stride, res_off]."""
    _addrs(c, 4, "x_trees_isotonic_predict_strided")
    var kx = _a(c, 0)
    var ky = _a(c, 1)
    var t = _a(c, 2)
    var res = _a(c, 3)
    _need(c, 8, "x_trees_isotonic_predict_strided")
    var m = _i(c, 0)
    if m < 1:
        raise Error("x_trees_isotonic_predict_strided: no knots")
    var n = _count(_i(c, 1), "x_trees_isotonic_predict_strided")
    var ts = _i(c, 3)
    var to = _i(c, 4)
    var rs = _i(c, 6)
    var ro = _i(c, 7)
    _strided(_i(c, 2), ts, to, n, "x_trees_isotonic_predict_strided")
    _strided(_i(c, 5), rs, ro, n, "x_trees_isotonic_predict_strided")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            boost_dev.isotonic_predict_strided_device(f64_ptr(kx), f64_ptr(ky), m,
                                                      f64_ptr(t) + to, ts, n, f64_ptr(res) + ro, rs)
        else:
            isotonic_predict_strided(f64_ptr(kx), f64_ptr(ky), m, f64_ptr(t) + to, ts, n,
                                     f64_ptr(res) + ro, rs)
    return n


def complement_pairs_run(c: XtCall) raises -> Int:
    """params = [n]: x (float64, n x 2)[i, 0] = 1 - x[i, 1]."""
    _addrs(c, 1, "x_trees_complement_pairs")
    var x = _a(c, 0)
    _need(c, 1, "x_trees_complement_pairs")
    var n = _count(_i(c, 0), "x_trees_complement_pairs")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.complement_pairs_device(f64_ptr(x), n)
        else:
            complement_pairs(f64_ptr(x), n)
    return n


def indicator_codes_run(c: XtCall) raises -> Int:
    """params = [n, cls]: out_i int32 0/1 per row (codes == cls); out_f (0: none) float64 1.0 / 0.0."""
    _addrs(c, 3, "x_trees_indicator_codes")
    var codes = _a(c, 0)
    var out_i = _a(c, 1)
    var out_f = _a(c, 2)
    _need(c, 2, "x_trees_indicator_codes")
    var n = _count(_i(c, 0), "x_trees_indicator_codes")
    var fa = out_f
    if n > 0:
        var pf = f64_ptr(fa) if fa != 0 else f64_ptr(out_i)
        comptime if XTREES_DEVICE_OPS:
            indicator_codes_device(i32_ptr(codes), n, _i(c, 1), i32_ptr(out_i), pf, fa != 0)
        else:
            indicator_codes(i32_ptr(codes), n, _i(c, 1), i32_ptr(out_i), pf, fa != 0)
    return n


def rows_any_unset_run(c: XtCall) raises -> Int:
    """eq: n * m uint8 equality flags (row-major, or column-major when
    params[2] != 0); out_i int32 (n): 1 when any flag of the row is 0, else
    0; params = [n, m, col_major] (lane ne2-b-members, N081:
    MultiOutputClassifier.score). Compares only."""
    _addrs(c, 2, "x_trees_rows_any_unset")
    var eq = _a(c, 0)
    var out_i = _a(c, 1)
    _need(c, 3, "x_trees_rows_any_unset")
    var n = _count(_i(c, 0), "x_trees_rows_any_unset")
    var m = _count(_i(c, 1), "x_trees_rows_any_unset")
    var col_major = _i(c, 2) != 0
    if n > 0:
        var e = eq
        if m > 0 and e == 0:
            raise Error("x_trees_rows_any_unset: null flag address")
        var flags = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=e)
        comptime if XTREES_DEVICE_OPS:
            rows_any_unset_device(flags, n, m, col_major, i32_ptr(out_i))
        else:
            rows_any_unset(flags, n, m, col_major, i32_ptr(out_i))
    return n


def binary_proba_run(c: XtCall) raises -> Int:
    """p float32 (n); res float64 (n x 2) = rows (1 - p, p), p widened
    exactly; params = [n] (lane apple-fast-py2mojo-trees)."""
    _addrs(c, 2, "x_trees_binary_proba")
    var p = _a(c, 0)
    var res = _a(c, 1)
    _need(c, 1, "x_trees_binary_proba")
    var n = _count(_i(c, 0), "x_trees_binary_proba")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            binary_proba_device(f32_ptr(p), n, f64_ptr(res))
        else:
            binary_proba(f32_ptr(p), n, f64_ptr(res))
    return n


def exact_sum_run(c: XtCall) raises -> Int:
    """params = [n, want_div]: res[0] = the exactly rounded sum of the float32
    x (`_portable_math.fsum`'s word), flags (int32, 8) = [NaN, +inf, -inf,
    negative finite, positive, 0, 0, 0]; want_div != 0: res holds n + 1
    float64 and res[1:] = x / sum when no entry is NaN, infinite or negative
    and one is positive (AdaBoost's normalized sample weights). The device on
    every GPU build, the host loop on the CPU column (xtrees/exact_sum.mojo;
    lane apple-fast-py2mojo-trees)."""
    _addrs(c, 3, "x_trees_exact_sum")
    var x = _a(c, 0)
    var res = _a(c, 1)
    var flags = _a(c, 2)
    _need(c, 2, "x_trees_exact_sum")
    var n = _count(_i(c, 0), "x_trees_exact_sum")
    if n > ES_MAX_ROWS:
        raise Error("x_trees_exact_sum: more than 2^30 entries")
    var want = _i(c, 1) != 0
    var xa = x
    var xp = f32_ptr(xa) if xa != 0 else f32_ptr(res)
    comptime if XTREES_DEVICE_OPS:
        exact_sum_device(xp, n, f64_ptr(res), i32_ptr(flags), want)
    else:
        exact_sum_host(xp, n, f64_ptr(res), i32_ptr(flags), want)
    return ES_FLAGS


def count_rows_run(c: XtCall) raises -> Int:
    """params = [n, m]: counts (int32, n) += 1 at each of the m rows (int32)
    (Bagging's out-of-bag member counts; xtrees/oob.mojo)."""
    _addrs(c, 2, "x_trees_count_rows")
    var counts = _a(c, 0)
    var rows = _a(c, 1)
    _need(c, 2, "x_trees_count_rows")
    var n = _count(_i(c, 0), "x_trees_count_rows")
    var m = _count(_i(c, 1), "x_trees_count_rows")
    if n > 0 and m > 0:
        comptime if XTREES_DEVICE_OPS:
            count_rows_device(i32_ptr(counts), n, i32_ptr(rows), m)
        else:
            count_rows_host(i32_ptr(counts), n, i32_ptr(rows), m)
    return m


def count_equal_run(c: XtCall) raises -> Int:
    """params = [n]: #{r : a[r] == b[r]} of two int32 vectors (xtrees/oob.mojo)."""
    _addrs(c, 2, "x_trees_count_equal")
    var a = _a(c, 0)
    var b = _a(c, 1)
    _need(c, 1, "x_trees_count_equal")
    var n = _count(_i(c, 0), "x_trees_count_equal")
    if n == 0:
        return 0
    comptime if XTREES_DEVICE_OPS:
        return count_equal_device(i32_ptr(a), i32_ptr(b), n)
    else:
        return count_equal_host(i32_ptr(a), i32_ptr(b), n)


def class_counts_run(c: XtCall) raises -> Int:
    """params = [n, k]: counts (int32, k) = the rows of each float32 class
    code (DART's multiclass init; lane apple-fast-py2mojo-trees)."""
    _addrs(c, 2, "x_trees_class_counts")
    var y = _a(c, 0)
    var counts = _a(c, 1)
    _need(c, 2, "x_trees_class_counts")
    var n = _count(_i(c, 0), "x_trees_class_counts")
    var k = _i(c, 1)
    if k < 1:
        raise Error("x_trees_class_counts: needs k >= 1")
    var ya = y
    var yp = f32_ptr(ya) if ya != 0 else f32_ptr(counts)
    comptime if XTREES_DEVICE_OPS:
        class_counts_device(yp, n, k, i32_ptr(counts))
    else:
        class_counts(yp, n, k, i32_ptr(counts))
    return k


def remap_cols_run(c: XtCall) raises -> Int:
    """params = [n_nodes, m]: in place colid[g] = cols[colid[g]] for a split
    node (a column-sampled DART tree back to X's columns)."""
    _addrs(c, 2, "x_trees_remap_cols")
    var colid = _a(c, 0)
    var cols = _a(c, 1)
    _need(c, 2, "x_trees_remap_cols")
    var nn = _count(_i(c, 0), "x_trees_remap_cols")
    var m = _count(_i(c, 1), "x_trees_remap_cols")
    if nn > 0:
        var ca = cols
        var cp = i32_ptr(ca) if ca != 0 else i32_ptr(colid)
        comptime if XTREES_DEVICE_OPS:
            remap_cols_device(i32_ptr(colid), nn, cp, m)
        else:
            remap_cols(i32_ptr(colid), nn, cp, m)
    return nn


def positive_codes_run(c: XtCall) raises -> Int:
    """params = [n]: codes (int32) = 1 where the float64 x > 0, else 0."""
    _addrs(c, 2, "x_trees_positive_codes")
    var x = _a(c, 0)
    var codes = _a(c, 1)
    _need(c, 1, "x_trees_positive_codes")
    var n = _count(_i(c, 0), "x_trees_positive_codes")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            positive_codes_device(f64_ptr(x), n, i32_ptr(codes))
        else:
            positive_codes(f64_ptr(x), n, i32_ptr(codes))
    return n


def leaf_numbering_run(c: XtCall) raises -> Int:
    """params = [n_nodes]: node_col (int32) = each leaf's output column in node
    order, -1 for a split node; returns the leaf count (RandomTreesEmbedding)."""
    _addrs(c, 2, "x_trees_leaf_numbering")
    var left = _a(c, 0)
    var node_col = _a(c, 1)
    _need(c, 1, "x_trees_leaf_numbering")
    var nn = _count(_i(c, 0), "x_trees_leaf_numbering")
    if nn == 0:
        return 0
    return leaf_numbering(i32_ptr(left), nn, i32_ptr(node_col))


def device_folds_run(c: XtCall) raises -> Int:
    """params = [n, n_splits, n_classes]: the cv folds on the device
    (xtrees/folds_device.mojo: the device on every GPU build, `ops.folds_serial`
    on the host column). n_classes > 0: StratifiedKFold over `codes` (int32, n, in
    [0, n_classes)); 0: KFold, codes unread. counts (int32, n_splits + 1) =
    the fold sizes then the status word; rows (int32, n_splits * n): per fold
    i the rows outside it, ascending, then the rows inside it, ascending.
    Returns the status: 0, 1 (the Python refusal: n_splits above every class
    count) or 2 (a code outside [0, n_classes))."""
    _addrs(c, 3, "x_trees_device_folds")
    var codes = _a(c, 0)
    var rows = _a(c, 1)
    var counts = _a(c, 2)
    _need(c, 3, "x_trees_device_folds")
    var n = _count(_i(c, 0), "x_trees_device_folds")
    var n_splits = _i(c, 1)
    var k = _count(_i(c, 2), "x_trees_device_folds")
    if n_splits < 2:
        raise Error("x_trees_device_folds: n_splits must be >= 2")
    var status = 0
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            status = device_folds(i32_ptr(codes), n, k, n_splits, i32_ptr(rows),
                                  i32_ptr(counts))
        else:
            status = host_folds(i32_ptr(codes), n, k, n_splits, i32_ptr(rows),
                                i32_ptr(counts))
    return status


def iota_i32_run(c: XtCall) raises -> Int:
    """out (int32, n)[i] = i; params = [n]; returns n (lane cpu2-l5-trees)."""
    _addrs(c, 1, "x_trees_iota_i32")
    var out_addr = _a(c, 0)
    _need(c, 1, "x_trees_iota_i32")
    var n = _count(_i(c, 0), "x_trees_iota_i32")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.iota_i32_device(i32_ptr(out_addr), n)
        else:
            iota_i32(i32_ptr(out_addr), n)
    return n


def fill_class_major_f64_run(c: XtCall) raises -> Int:
    """out (float64, K x n)[c * n + i] = inits (float64, K)[c], a word copy;
    params = [n, K]; returns K * n (lane cpu2-l5-trees)."""
    _addrs(c, 2, "x_trees_fill_class_major_f64")
    var inits_addr = _a(c, 0)
    var out_addr = _a(c, 1)
    _need(c, 2, "x_trees_fill_class_major_f64")
    var n = _count(_i(c, 0), "x_trees_fill_class_major_f64")
    var k = _count(_i(c, 1), "x_trees_fill_class_major_f64")
    if n * k > 0:
        comptime if XTREES_DEVICE_OPS:
            elem_dev.fill_class_major_f64_device(f64_ptr(inits_addr), n, k, f64_ptr(out_addr))
        else:
            fill_class_major_f64(f64_ptr(inits_addr), n, k, f64_ptr(out_addr))
    return n * k


def class_priors_run(c: XtCall) raises -> Int:
    """params = [n, k]: priors (float64, k) = log(max(1e-15, counts[c] / n)),
    the pinned binary64 log (`identical_log64`) of each class's share (DART's
    multiclass start; it was a Python list over the k counts)."""
    _addrs(c, 2, "x_trees_class_priors")
    var counts = _a(c, 0)
    var priors = _a(c, 1)
    _need(c, 2, "x_trees_class_priors")
    var n = _count(_i(c, 0), "x_trees_class_priors")
    var k = _i(c, 1)
    if k < 1 or n < 1:
        raise Error("x_trees_class_priors: needs n >= 1 and k >= 1")
    var cp = i32_ptr(counts)
    var op = f64_ptr(priors)
    class_priors(cp, op, n, k)
    return k


def x_trees_glue_dispatch(op: Int, c: XtCall) raises -> Int:
    """One glue operation by its `XT_*` code (xtrees/glue_abi.mojo)."""
    if op == XT_ACCUMULATE:
        return accumulate_run(c)
    if op == XT_ACCUMULATE_COLS:
        return accumulate_cols_run(c)
    if op == XT_ACCUMULATE_ONEHOT:
        return accumulate_onehot_run(c)
    if op == XT_ACCUMULATE_ROWS:
        return accumulate_rows_run(c)
    if op == XT_APPLY:
        return apply_run(c)
    if op == XT_ARGMAX_ROWS:
        return argmax_rows_run(c)
    if op == XT_BAG_ROWS:
        return bag_rows_run(c)
    if op == XT_BINARY_PROBA:
        return binary_proba_run(c)
    if op == XT_CLASS_COUNTS:
        return class_counts_run(c)
    if op == XT_CLASS_PRIORS:
        return class_priors_run(c)
    if op == XT_COMPLEMENT_PAIRS:
        return complement_pairs_run(c)
    if op == XT_COUNT_EQUAL:
        return count_equal_run(c)
    if op == XT_COUNT_ROWS:
        return count_rows_run(c)
    if op == XT_DART_RESCALE:
        return dart_rescale_run(c)
    if op == XT_DART_THRESHOLDS:
        return dart_thresholds_run(c)
    if op == XT_DEVICE_FOLDS:
        return device_folds_run(c)
    if op == XT_EXACT_SUM:
        return exact_sum_run(c)
    if op == XT_FILL_CLASS_MAJOR_F64:
        return fill_class_major_f64_run(c)
    if op == XT_GATHER_F32:
        return gather_f32_run(c)
    if op == XT_GATHER_I32:
        return gather_i32_run(c)
    if op == XT_INDICATOR_CODES:
        return indicator_codes_run(c)
    if op == XT_IOTA_I32:
        return iota_i32_run(c)
    if op == XT_ISOTONIC_FIT:
        return isotonic_fit_run(c)
    if op == XT_ISOTONIC_PREDICT_STRIDED:
        return isotonic_predict_strided_run(c)
    if op == XT_LEAF_NUMBERING:
        return leaf_numbering_run(c)
    if op == XT_LOG64:
        return log64_run(c)
    if op == XT_MARGIN2:
        return margin2_run(c)
    if op == XT_NORMALIZE_ROWS:
        return normalize_rows_run(c)
    if op == XT_ONEHOT_LEAVES:
        return onehot_leaves_run(c)
    if op == XT_OOB_R2:
        return oob_r2_run(c)
    if op == XT_PLATT_APPLY_STRIDED:
        return platt_apply_strided_run(c)
    if op == XT_PLATT_FIT:
        return platt_fit_run(c)
    if op == XT_POSITIVE_CODES:
        return positive_codes_run(c)
    if op == XT_PUT_F32:
        return put_f32_run(c)
    if op == XT_R2_STEP:
        return r2_step_run(c)
    if op == XT_REMAP_COLS:
        return remap_cols_run(c)
    if op == XT_ROWS_ANY_UNSET:
        return rows_any_unset_run(c)
    if op == XT_SAMME_STEP:
        return samme_step_run(c)
    if op == XT_SAMPLE_INDICES:
        return sample_indices_run(c)
    if op == XT_SCALE:
        return scale_run(c)
    if op == XT_SCALE_TO_F32:
        return scale_to_f32_run(c)
    if op == XT_SCATTER:
        return scatter_run(c)
    if op == XT_SOFTMAX_ROWS:
        return softmax_rows_run(c)
    if op == XT_STACK_W64:
        return stack_w64_run(c)
    if op == XT_TRANSPOSE_F32:
        return transpose_f32_run(c)
    if op == XT_TRANSPOSE_F64:
        return transpose_f64_run(c)
    if op == XT_UNIFORM:
        return uniform_run(c)
    if op == XT_UNSEEN_ROWS:
        return unseen_rows_run(c)
    if op == XT_WEIGHTED_MEDIAN:
        return weighted_median_run(c)
    if op == XT_WEIGHTED_SAMPLE:
        return weighted_sample_run(c)
    raise Error("x_trees glue: unknown operation " + String(op))


def x_trees_glue_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The x_trees glue entry of the trees table (`XT_GLUE_ABI`): addresses
    span aa[0:na], the parameter views pp[0:np] (int) and ff[0:nf] (float,
    nf == np). Writes the run's count to result; a refusal goes to sink."""
    try:
        if operation < 0 or operation >= XT_OP_COUNT:
            raise Error("x_trees glue: unknown operation " + String(operation))
        if na < 0 or np < 0 or np != nf or not result:
            raise Error("x_trees glue: invalid request spans")
        if (na > 0 and not aa) or (np > 0 and (not pp or not ff)):
            raise Error("x_trees glue: null request span")
        var value = x_trees_glue_dispatch(operation, XtCall(aa, na, pp, ff, np, 0))
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct XTreesGlueOwner(Defaultable, Movable):
    """The image-lifetime publisher word of the x_trees glue table."""

    var word: Int

    def __init__(out self):
        self.word = 1


comptime _GLUE_OWNER_NAME = "MojolearnXTreesGlueOwnerV1"
comptime XTREES_GLUE_OWNER = SharedSlot[XTreesGlueOwner, _GLUE_OWNER_NAME]


def x_trees_glue_vendor() -> Int:
    """This image's vendor word of the trees table key: 0 the host column,
    1 Metal, 2 CUDA, 3 HIP (the bridge reads `x_trees_vendor` the same way)."""
    comptime if TARGET_COLUMN == COLUMN_CPU:
        return 0
    elif COMPILED_VENDOR == "metal":
        return 1
    elif COMPILED_VENDOR == "cuda":
        return 2
    elif COMPILED_VENDOR == "hip":
        return 3
    else:
        return -1


def publish_x_trees_glue() raises:
    """Publish this image's glue entry at module init (both x_trees
    bindings). A repeated init publishes the same pointer; a second image
    with the same key and another pointer refuses (core/entry_table.mojo)."""
    var owner = XTREES_GLUE_OWNER.get_or_create_ptr()
    var key = TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), x_trees_glue_vendor())
    trees_table_publish[TREES_TABLE_X_TREES](key, Int(owner),
        [String(XT_GLUE_ABI)], [XT_OP_COUNT], [x_trees_glue_v1])
