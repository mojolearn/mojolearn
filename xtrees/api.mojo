# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python-facing entry points of the trees lane, shared by the GPU binding
(`bindings/_mojolearn_x_trees.mojo`) and the host binding
(`bindings/_mojolearn_x_trees_host.mojo`): one spelling, two registrations.
Every buffer is a caller-owned address; `params` is a Python list of ints and
floats. Nothing is retained."""
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr
from checks.numerics import identical_log64
from std.python import Python
from xtrees.shap import mask_expand, block_mean, kernel_solve
from xtrees.perm_device import perm_synthetic
from xtrees.ops_device import (
    apply_trees_device, bag_rows_device, gather_f32_device, transpose_f32_device, transpose_f64_device,
    unseen_rows_device, weighted_sample_device,
)
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN

#: cpu-gpu-cleanup t-gbdt / w2-trees: a GPU build gathers, applies, transposes
#: and draws weighted samples on
#: the device (`xtrees/ops_device.mojo`); the host binding (the CPU column,
#: `-D MOJOLEARN_COLUMN_CPU`) keeps `xtrees/ops.mojo`'s serial loops. Fixed
#: at build time, never a runtime route.
comptime XTREES_DEVICE_OPS = TARGET_COLUMN != COLUMN_CPU
from xtrees.ops import (
    sample_indices, weighted_sample, gather_f32, gather_i32, accumulate,
    accumulate_onehot, accumulate_cols, accumulate_rows, argmax_rows, argmax_rows_f32, scale_f64, softmax_rows, scale_to_f32, put_f32,
    check_weights_f32, mul_f32,
    samme_step, r2_step, weighted_median, apply_trees, gradients, leaf_newton, leaf_newton_rows, tree_score_add, uniform,
    onehot_leaves, transpose_f32, normalize_rows, exact_sum_f32, EXACT_SUM_LIMBS, logit, scatter, platt_fit, platt_apply, isotonic_fit,
    isotonic_predict, platt_apply_strided, isotonic_predict_strided, complement_pairs, indicator_codes, column_f64,
    bag_rows, unseen_rows, transpose_f64,
)


def _need(params: PythonObject, n: Int, who: String) raises:
    if len(params) != n:
        raise Error(who + ": params must hold " + String(n) + " values, got " + String(len(params)))


def _i(params: PythonObject, k: Int) raises -> Int:
    return Int(py=params[k])


def _f(params: PythonObject, k: Int) raises -> Float64:
    return Float64(py=params[k])


def _count(v: Int, who: String) raises -> Int:
    if v < 0:
        raise Error(who + ": negative count")
    return v


def sample_indices_binding(out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [n_pool, n_draw, replace, seed, stream]."""
    _need(params, 5, "x_trees_sample_indices")
    var n_draw = _count(_i(params, 1), "x_trees_sample_indices")
    if n_draw > 0:
        sample_indices(i32_ptr(Int(py=out_addr)), _i(params, 0), n_draw, _i(params, 2) != 0,
                       _i(params, 3), _i(params, 4))
    return PythonObject(n_draw)


def weighted_sample_binding(w_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [n, n_draw, seed, stream]."""
    _need(params, 4, "x_trees_weighted_sample")
    var n = _i(params, 0)
    if n <= 0:
        raise Error("x_trees_weighted_sample: n must be positive")
    var n_draw = _count(_i(params, 1), "x_trees_weighted_sample")
    comptime if XTREES_DEVICE_OPS:
        weighted_sample_device(f64_ptr(Int(py=w_addr)), n, i32_ptr(Int(py=out_addr)), n_draw, _i(params, 2), _i(params, 3))
    else:
        weighted_sample(f64_ptr(Int(py=w_addr)), n, i32_ptr(Int(py=out_addr)), n_draw, _i(params, 2), _i(params, 3))
    return PythonObject(n_draw)


def gather_f32_binding(
    src: PythonObject, rows: PythonObject, cols: PythonObject, dst: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [n_src_rows, n_src_cols, n_rows, n_cols]."""
    _need(params, 4, "x_trees_gather_f32")
    var n_rows = _count(_i(params, 2), "x_trees_gather_f32")
    var n_cols = _count(_i(params, 3), "x_trees_gather_f32")
    if n_rows * n_cols > 0:
        comptime if XTREES_DEVICE_OPS:
            gather_f32_device(f32_ptr(Int(py=src)), _i(params, 0), _i(params, 1), i32_ptr(Int(py=rows)), n_rows,
                              i32_ptr(Int(py=cols)), n_cols, f32_ptr(Int(py=dst)))
        else:
            gather_f32(f32_ptr(Int(py=src)), _i(params, 0), _i(params, 1), i32_ptr(Int(py=rows)), n_rows,
                       i32_ptr(Int(py=cols)), n_cols, f32_ptr(Int(py=dst)))
    return PythonObject(n_rows * n_cols)


def gather_i32_binding(src: PythonObject, rows: PythonObject, dst: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [n_src, n_rows]."""
    _need(params, 2, "x_trees_gather_i32")
    var n_rows = _count(_i(params, 1), "x_trees_gather_i32")
    if n_rows > 0:
        gather_i32(i32_ptr(Int(py=src)), _i(params, 0), i32_ptr(Int(py=rows)), n_rows, i32_ptr(Int(py=dst)))
    return PythonObject(n_rows)


def accumulate_binding(acc: PythonObject, x: PythonObject, params: PythonObject) raises -> PythonObject:
    """acc (float64) += weight * x (float32); params = [n, weight]."""
    _need(params, 2, "x_trees_accumulate")
    var n = _count(_i(params, 0), "x_trees_accumulate")
    if n > 0:
        accumulate(f64_ptr(Int(py=acc)), f32_ptr(Int(py=x)), n, _f(params, 1))
    return PythonObject(n)


def accumulate_cols_binding(acc: PythonObject, x: PythonObject, cols: PythonObject, params: PythonObject) raises -> PythonObject:
    """acc (float64, n*k) += weight * x (float32, n*ks) into columns cols; params = [n, k, ks, weight]."""
    _need(params, 4, "x_trees_accumulate_cols")
    var n = _count(_i(params, 0), "x_trees_accumulate_cols")
    if n > 0:
        accumulate_cols(f64_ptr(Int(py=acc)), f32_ptr(Int(py=x)), i32_ptr(Int(py=cols)), n, _i(params, 1),
                        _i(params, 2), _f(params, 3))
    return PythonObject(n)


def accumulate_rows_binding(acc: PythonObject, x: PythonObject, rows: PythonObject, params: PythonObject) raises -> PythonObject:
    """acc (n x k float64) += x (m x k float64) at rows; params = [n, k, m]."""
    _need(params, 3, "x_trees_accumulate_rows")
    var n = _count(_i(params, 0), "x_trees_accumulate_rows")
    var k = _count(_i(params, 1), "x_trees_accumulate_rows")
    var m = _count(_i(params, 2), "x_trees_accumulate_rows")
    if m * k > 0:
        accumulate_rows(f64_ptr(Int(py=acc)), n, k, f64_ptr(Int(py=x)), i32_ptr(Int(py=rows)), m)
    return PythonObject(m)


def accumulate_onehot_binding(acc: PythonObject, codes: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [n, k, on, off]."""
    _need(params, 4, "x_trees_accumulate_onehot")
    var n = _count(_i(params, 0), "x_trees_accumulate_onehot")
    if n > 0:
        accumulate_onehot(f64_ptr(Int(py=acc)), i32_ptr(Int(py=codes)), n, _i(params, 1), _f(params, 2), _f(params, 3))
    return PythonObject(n)


def argmax_rows_binding(x: PythonObject, res: PythonObject, params: PythonObject) raises -> PythonObject:
    """float64 x; params = [n, k]."""
    _need(params, 2, "x_trees_argmax_rows")
    var n = _count(_i(params, 0), "x_trees_argmax_rows")
    if n > 0:
        argmax_rows(f64_ptr(Int(py=x)), n, _i(params, 1), i32_ptr(Int(py=res)))
    return PythonObject(n)


def argmax_rows_f32_binding(x: PythonObject, res: PythonObject, params: PythonObject) raises -> PythonObject:
    _need(params, 2, "x_trees_argmax_rows_f32")
    var n = _count(_i(params, 0), "x_trees_argmax_rows_f32")
    if n > 0:
        argmax_rows_f32(f32_ptr(Int(py=x)), n, _i(params, 1), i32_ptr(Int(py=res)))
    return PythonObject(n)


def scale_binding(x: PythonObject, params: PythonObject) raises -> PythonObject:
    """x /= divisor; params = [n, divisor]."""
    _need(params, 2, "x_trees_scale")
    var n = _count(_i(params, 0), "x_trees_scale")
    if n > 0:
        scale_f64(f64_ptr(Int(py=x)), n, _f(params, 1))
    return PythonObject(n)


def scale_to_f32_binding(x: PythonObject, res: PythonObject, params: PythonObject) raises -> PythonObject:
    """res (float32) = x (float64) * factor; params = [n, factor]."""
    _need(params, 2, "x_trees_scale_to_f32")
    var n = _count(_i(params, 0), "x_trees_scale_to_f32")
    if n > 0:
        scale_to_f32(f64_ptr(Int(py=x)), n, _f(params, 1), f32_ptr(Int(py=res)))
    return PythonObject(n)


def exact_sum_f32_binding(x: PythonObject, params: PythonObject) raises -> PythonObject:
    """The exact sum of a float32 buffer as `EXACT_SUM_LIMBS` integer places
    (`xtrees.ops.exact_sum_f32`), or None when it holds a NaN or an infinity;
    params = [n]."""
    _need(params, 1, "x_trees_exact_sum_f32")
    var n = _count(_i(params, 0), "x_trees_exact_sum_f32")
    var limbs = List[Int64](length=EXACT_SUM_LIMBS, fill=0)
    if not exact_sum_f32(f32_ptr(Int(py=x)), n, limbs):
        return PythonObject(None)
    var out = Python.list()
    for i in range(EXACT_SUM_LIMBS):
        out.append(PythonObject(Int(limbs[i])))
    return out


def margin2_binding(acc: PythonObject, dst: PythonObject, params: PythonObject) raises -> PythonObject:
    """Two-class vote rows (n x 2, float64) to the SAMME margin
    d = acc[2i+1] - acc[2i], one IEEE binary64 subtraction per row, the
    value the Python `v[2*i+1] - v[2*i]` computed. params = [n, mode]:
    mode 0 writes d (float64, n); mode 1 writes the int32 code
    `1 if d > 0 else 0` (a NaN gives 0, as `>` did); mode 2 writes the
    float64 pairs (-(d/2), d/2) (n x 2), the rows `predict_proba` softmaxes."""
    _need(params, 2, "x_trees_margin2")
    var n = _count(_i(params, 0), "x_trees_margin2")
    var mode = _i(params, 1)
    var a = f64_ptr(Int(py=acc))
    if mode == 0:
        var o = f64_ptr(Int(py=dst))
        for i in range(n):
            o[unsafe_offset=i] = a[unsafe_offset=2 * i + 1] - a[unsafe_offset=2 * i]
    elif mode == 1:
        var o = i32_ptr(Int(py=dst))
        for i in range(n):
            var d = a[unsafe_offset=2 * i + 1] - a[unsafe_offset=2 * i]
            o[unsafe_offset=i] = Int32(1) if d > 0 else Int32(0)
    elif mode == 2:
        var o = f64_ptr(Int(py=dst))
        for i in range(n):
            var h = (a[unsafe_offset=2 * i + 1] - a[unsafe_offset=2 * i]) / 2
            o[unsafe_offset=2 * i] = -h
            o[unsafe_offset=2 * i + 1] = h
    else:
        raise Error("x_trees_margin2: mode must be 0, 1 or 2")
    return PythonObject(n)


def put_f32_binding(dst: PythonObject, src: PythonObject, params: PythonObject) raises -> PythonObject:
    """dst[offset:offset+n] = src; params = [offset, n]."""
    _need(params, 2, "x_trees_put_f32")
    var n = _count(_i(params, 1), "x_trees_put_f32")
    if n > 0:
        put_f32(f32_ptr(Int(py=dst)), _count(_i(params, 0), "x_trees_put_f32"), f32_ptr(Int(py=src)), n)
    return PythonObject(n)


def softmax_rows_binding(x: PythonObject, params: PythonObject) raises -> PythonObject:
    _need(params, 2, "x_trees_softmax_rows")
    var n = _count(_i(params, 0), "x_trees_softmax_rows")
    if n > 0:
        softmax_rows(f64_ptr(Int(py=x)), n, _i(params, 1))
    return PythonObject(n)


def samme_step_binding(
    w: PythonObject, pred: PythonObject, y: PythonObject, stats: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [n, n_classes, learning_rate, last]; stats = 4 float64."""
    _need(params, 4, "x_trees_samme_step")
    var n = _i(params, 0)
    if n <= 0:
        raise Error("x_trees_samme_step: n must be positive")
    samme_step(f64_ptr(Int(py=w)), i32_ptr(Int(py=pred)), i32_ptr(Int(py=y)), n, _i(params, 1),
               _f(params, 2), _i(params, 3) != 0, f64_ptr(Int(py=stats)))
    return PythonObject(n)


def r2_step_binding(
    w: PythonObject, pred: PythonObject, y: PythonObject, stats: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [n, loss (0 linear, 1 square, 2 exponential), learning_rate, last]."""
    _need(params, 4, "x_trees_r2_step")
    var n = _i(params, 0)
    if n <= 0:
        raise Error("x_trees_r2_step: n must be positive")
    r2_step(f64_ptr(Int(py=w)), f32_ptr(Int(py=pred)), f32_ptr(Int(py=y)), n, _i(params, 1),
            _f(params, 2), _i(params, 3) != 0, f64_ptr(Int(py=stats)))
    return PythonObject(n)


def weighted_median_binding(
    preds: PythonObject, weights: PythonObject, res: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [n, m]; preds estimator-major float32."""
    _need(params, 2, "x_trees_weighted_median")
    var n = _count(_i(params, 0), "x_trees_weighted_median")
    var m = _i(params, 1)
    if m <= 0:
        raise Error("x_trees_weighted_median: need at least one estimator")
    if n > 0:
        weighted_median(f32_ptr(Int(py=preds)), f64_ptr(Int(py=weights)), n, m, f32_ptr(Int(py=res)))
    return PythonObject(n)


def apply_binding(
    offsets: PythonObject, colid: PythonObject, quesval: PythonObject, left: PythonObject,
    x: PythonObject, res: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """Leaf node per (row, tree) for trees [t0, t1); params = [n, d, t0, t1]; x row-major float32."""
    _need(params, 4, "x_trees_apply")
    var n = _count(_i(params, 0), "x_trees_apply")
    var t0 = _count(_i(params, 2), "x_trees_apply")
    var t1 = _i(params, 3)
    if t1 <= t0:
        raise Error("x_trees_apply: need t1 > t0")
    if n > 0:
        comptime if XTREES_DEVICE_OPS:
            apply_trees_device(i32_ptr(Int(py=offsets)), i32_ptr(Int(py=colid)), f32_ptr(Int(py=quesval)),
                               i32_ptr(Int(py=left)), f32_ptr(Int(py=x)), n, _i(params, 1), t0, t1, i32_ptr(Int(py=res)))
        else:
            apply_trees(i32_ptr(Int(py=offsets)), i32_ptr(Int(py=colid)), f32_ptr(Int(py=quesval)),
                        i32_ptr(Int(py=left)), f32_ptr(Int(py=x)), n, _i(params, 1), t0, t1, i32_ptr(Int(py=res)))
    return PythonObject(n)


def gradients_binding(
    score: PythonObject, y: PythonObject, g: PythonObject, h: PythonObject, target: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """params = [n, kind (0 l2, 1 binary logloss, 2 multiclass softmax)] or,
    for kind 2, [n, 2, k] with every buffer class-major (k * n)."""
    if len(params) != 2 and len(params) != 3:
        raise Error("x_trees_gradients: params must hold 2 values (3 for multiclass), got " + String(len(params)))
    var n = _count(_i(params, 0), "x_trees_gradients")
    var kind = _i(params, 1)
    if kind != 0 and kind != 1 and kind != 2:
        raise Error("x_trees_gradients: kind must be 0 (l2), 1 (binary) or 2 (multiclass)")
    var k = 1
    if kind == 2:
        _need(params, 3, "x_trees_gradients")
        k = _i(params, 2)
    elif len(params) != 2:
        raise Error("x_trees_gradients: only multiclass takes k")
        if k < 2:
            raise Error("x_trees_gradients: multiclass needs k >= 2")
    if n > 0:
        gradients(f64_ptr(Int(py=score)), f32_ptr(Int(py=y)), n, kind, f64_ptr(Int(py=g)), f64_ptr(Int(py=h)),
                  f32_ptr(Int(py=target)), k)
    return PythonObject(n)


def leaf_newton_binding(
    nodes: PythonObject, g: PythonObject, h: PythonObject, values: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [n, n_nodes, reg_lambda] or [n, n_nodes, reg_lambda, lambda_l1, max_delta_step]."""
    if len(params) != 3 and len(params) != 5:
        raise Error("x_trees_leaf_newton: params must hold 3 or 5 values, got " + String(len(params)))
    var n = _count(_i(params, 0), "x_trees_leaf_newton")
    var n_nodes = _i(params, 1)
    if n_nodes < 1:
        raise Error("x_trees_leaf_newton: need n_nodes >= 1")
    var l1: Float64 = 0.0
    var mds: Float64 = 0.0
    if len(params) == 5:
        l1 = _f(params, 3)
        mds = _f(params, 4)
    leaf_newton(i32_ptr(Int(py=nodes)), f64_ptr(Int(py=g)), f64_ptr(Int(py=h)), n, n_nodes, _f(params, 2),
                f32_ptr(Int(py=values)), l1, mds)
    return PythonObject(n_nodes)


def leaf_newton_rows_binding(
    nodes: PythonObject, rows: PythonObject, g: PythonObject, h: PythonObject, values: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """leaf_newton over rows[0 .. m) only (the bagged rows), nodes / g / h
    indexed by the full row; params = [n, m, n_nodes, reg_lambda, lambda_l1, max_delta_step]."""
    _need(params, 6, "x_trees_leaf_newton_rows")
    var n = _count(_i(params, 0), "x_trees_leaf_newton_rows")
    var m = _count(_i(params, 1), "x_trees_leaf_newton_rows")
    var n_nodes = _i(params, 2)
    if n_nodes < 1:
        raise Error("x_trees_leaf_newton_rows: need n_nodes >= 1")
    leaf_newton_rows(i32_ptr(Int(py=nodes)), i32_ptr(Int(py=rows)), m, f64_ptr(Int(py=g)), f64_ptr(Int(py=h)), n,
                     n_nodes, _f(params, 3), _f(params, 4), _f(params, 5), f32_ptr(Int(py=values)))
    return PythonObject(n_nodes)


def tree_score_add_binding(nodes: PythonObject, values: PythonObject, acc: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [n, weight]."""
    _need(params, 2, "x_trees_tree_score_add")
    var n = _count(_i(params, 0), "x_trees_tree_score_add")
    if n > 0:
        tree_score_add(i32_ptr(Int(py=nodes)), f32_ptr(Int(py=values)), n, _f(params, 1), f64_ptr(Int(py=acc)))
    return PythonObject(n)


def uniform_binding(res: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [n, seed, stream]."""
    _need(params, 3, "x_trees_uniform")
    var n = _count(_i(params, 0), "x_trees_uniform")
    if n > 0:
        uniform(f64_ptr(Int(py=res)), n, _i(params, 1), _i(params, 2))
    return PythonObject(n)


def onehot_leaves_binding(
    nodes: PythonObject, tree_base: PythonObject, node_col: PythonObject, res: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [n, n_trees, n_cols]; res float64 n*n_cols, zeroed."""
    _need(params, 3, "x_trees_onehot_leaves")
    var n = _count(_i(params, 0), "x_trees_onehot_leaves")
    if n > 0:
        onehot_leaves(i32_ptr(Int(py=nodes)), i32_ptr(Int(py=tree_base)), i32_ptr(Int(py=node_col)), n,
                      _i(params, 1), _i(params, 2), f64_ptr(Int(py=res)))
    return PythonObject(n)


def transpose_f32_binding(src: PythonObject, dst: PythonObject, params: PythonObject) raises -> PythonObject:
    """dst (d x n) = src (n x d) transposed; params = [n, d]."""
    _need(params, 2, "x_trees_transpose_f32")
    var n = _count(_i(params, 0), "x_trees_transpose_f32")
    var d = _count(_i(params, 1), "x_trees_transpose_f32")
    if n * d > 0:
        comptime if XTREES_DEVICE_OPS:
            transpose_f32_device(f32_ptr(Int(py=src)), n, d, f32_ptr(Int(py=dst)))
        else:
            transpose_f32(f32_ptr(Int(py=src)), n, d, f32_ptr(Int(py=dst)))
    return PythonObject(n * d)


def transpose_f64_binding(src: PythonObject, dst: PythonObject, params: PythonObject) raises -> PythonObject:
    """dst (d x n) = src (n x d) transposed, float64 words; params = [n, d]."""
    _need(params, 2, "x_trees_transpose_f64")
    var n = _count(_i(params, 0), "x_trees_transpose_f64")
    var d = _count(_i(params, 1), "x_trees_transpose_f64")
    if n * d > 0:
        comptime if XTREES_DEVICE_OPS:
            transpose_f64_device(f64_ptr(Int(py=src)), n, d, f64_ptr(Int(py=dst)))
        else:
            transpose_f64(f64_ptr(Int(py=src)), n, d, f64_ptr(Int(py=dst)))
    return PythonObject(n * d)


def bag_rows_binding(res: PythonObject, params: PythonObject) raises -> PythonObject:
    """res (int32, n) <- the rows whose draw is below frac, ascending (none:
    the smallest draw's row); params = [n, seed, stream, frac]; returns the
    count."""
    _need(params, 4, "x_trees_bag_rows")
    var n = _count(_i(params, 0), "x_trees_bag_rows")
    if n == 0:
        return PythonObject(0)
    comptime if XTREES_DEVICE_OPS:
        return PythonObject(bag_rows_device(i32_ptr(Int(py=res)), n, _i(params, 1), _i(params, 2), _f(params, 3)))
    else:
        return PythonObject(bag_rows(i32_ptr(Int(py=res)), n, _i(params, 1), _i(params, 2), _f(params, 3)))


def unseen_rows_binding(rows: PythonObject, res: PythonObject, params: PythonObject) raises -> PythonObject:
    """res (int32, n) <- the rows of [0, n) absent from rows (int32, m),
    ascending; params = [m, n]; returns the count."""
    _need(params, 2, "x_trees_unseen_rows")
    var m = _count(_i(params, 0), "x_trees_unseen_rows")
    var n = _count(_i(params, 1), "x_trees_unseen_rows")
    if n == 0:
        return PythonObject(0)
    var rp = i32_ptr(Int(py=rows)) if m > 0 else i32_ptr(Int(py=res))
    comptime if XTREES_DEVICE_OPS:
        return PythonObject(unseen_rows_device(rp, m, n, i32_ptr(Int(py=res))))
    else:
        return PythonObject(unseen_rows(rp, m, n, i32_ptr(Int(py=res))))


def check_weights_f32_binding(w: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [n]; returns check_weights_f32's status (0 ok, 1 bad entry, 2 no positive)."""
    _need(params, 1, "x_trees_check_weights_f32")
    var n = _count(_i(params, 0), "x_trees_check_weights_f32")
    if n == 0:
        return PythonObject(2)
    return PythonObject(check_weights_f32(f32_ptr(Int(py=w)), n))


def mul_f32_binding(a: PythonObject, b: PythonObject, dst: PythonObject, params: PythonObject) raises -> PythonObject:
    """dst = a * b elementwise in float32; params = [n]."""
    _need(params, 1, "x_trees_mul_f32")
    var n = _count(_i(params, 0), "x_trees_mul_f32")
    if n > 0:
        mul_f32(f32_ptr(Int(py=a)), f32_ptr(Int(py=b)), n, f32_ptr(Int(py=dst)))
    return PythonObject(n)


def log64_binding(x: PythonObject) raises -> PythonObject:
    """The pinned binary64 log (checks/numerics.mojo identical_log64) of one value."""
    return PythonObject(identical_log64(Float64(py=x)))


def logit_binding(x: PythonObject, params: PythonObject) raises -> PythonObject:
    """In place x = log(x / (1 - x)); params = [n]."""
    _need(params, 1, "x_trees_logit")
    var n = _count(_i(params, 0), "x_trees_logit")
    if n > 0:
        logit(f64_ptr(Int(py=x)), n)
    return PythonObject(n)


def normalize_rows_binding(x: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [n, k]."""
    _need(params, 2, "x_trees_normalize_rows")
    var n = _count(_i(params, 0), "x_trees_normalize_rows")
    if n > 0:
        normalize_rows(f64_ptr(Int(py=x)), n, _i(params, 1))
    return PythonObject(n)


def scatter_binding(dst: PythonObject, src: PythonObject, rows: PythonObject, params: PythonObject) raises -> PythonObject:
    """dst (float64, N x n_dst_cols)[rows[r], col0 + j] = src (float32, m x c)[r, j];
    params = [n_dst_rows, n_dst_cols, m, c, col0]."""
    _need(params, 5, "x_trees_scatter")
    var n_dst = _i(params, 0)
    var n_cols = _i(params, 1)
    var m = _count(_i(params, 2), "x_trees_scatter")
    var c = _count(_i(params, 3), "x_trees_scatter")
    var col0 = _count(_i(params, 4), "x_trees_scatter")
    if col0 + c > n_cols:
        raise Error("x_trees_scatter: columns out of range")
    var rp = i32_ptr(Int(py=rows)) if m > 0 else i32_ptr(1)
    for r in range(m):
        var i = Int(rp[unsafe_offset=r])
        if i < 0 or i >= n_dst:
            raise Error("x_trees_scatter: row out of range")
    if m * c > 0:
        scatter(f64_ptr(Int(py=dst)), n_cols, f32_ptr(Int(py=src)), m, c, rp, col0)
    return PythonObject(m * c)


def platt_fit_binding(f: PythonObject, y: PythonObject, ab: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [n]; y int32 0/1; ab float64[2]."""
    _need(params, 1, "x_trees_platt_fit")
    var n = _i(params, 0)
    if n < 1:
        raise Error("x_trees_platt_fit: no rows")
    platt_fit(f64_ptr(Int(py=f)), i32_ptr(Int(py=y)), n, f64_ptr(Int(py=ab)))
    return PythonObject(n)


def platt_apply_binding(f: PythonObject, res: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [n, A, B]."""
    _need(params, 3, "x_trees_platt_apply")
    var n = _count(_i(params, 0), "x_trees_platt_apply")
    if n > 0:
        platt_apply(f64_ptr(Int(py=f)), n, _f(params, 1), _f(params, 2), f64_ptr(Int(py=res)))
    return PythonObject(n)


def isotonic_fit_binding(
    x: PythonObject, y: PythonObject, kx: PythonObject, ky: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [n]; kx, ky float64[n] receive the knots; returns their count."""
    _need(params, 1, "x_trees_isotonic_fit")
    var n = _i(params, 0)
    var m = isotonic_fit(f64_ptr(Int(py=x)), f64_ptr(Int(py=y)), n, f64_ptr(Int(py=kx)), f64_ptr(Int(py=ky)))
    return PythonObject(m)


def isotonic_predict_binding(
    kx: PythonObject, ky: PythonObject, t: PythonObject, res: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [m, n]."""
    _need(params, 2, "x_trees_isotonic_predict")
    var m = _i(params, 0)
    if m < 1:
        raise Error("x_trees_isotonic_predict: no knots")
    var n = _count(_i(params, 1), "x_trees_isotonic_predict")
    if n > 0:
        isotonic_predict(f64_ptr(Int(py=kx)), f64_ptr(Int(py=ky)), m, f64_ptr(Int(py=t)), n, f64_ptr(Int(py=res)))
    return PythonObject(n)


# lane py-misc-prep: CalibratedClassifierCV's per-class epilogue without
# Python lists. A column of the (n, c) score block is read at a stride and
# offset, and each calibrator writes straight into its column of the (n, k)
# probability block; the per-element operations are `platt_apply`'s and
# `isotonic_predict`'s (they now call these with stride 1).


def _strided(total: Int, stride: Int, off: Int, n: Int, who: String) raises:
    if stride < 1 or off < 0 or (n > 0 and off + (n - 1) * stride >= total):
        raise Error(who + ": stride or offset out of range")


def platt_apply_strided_binding(f: PythonObject, res: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [n, A, B, f_len, f_stride, f_off, res_len, res_stride, res_off]."""
    _need(params, 9, "x_trees_platt_apply_strided")
    var n = _count(_i(params, 0), "x_trees_platt_apply_strided")
    var fs = _i(params, 4)
    var fo = _i(params, 5)
    var rs = _i(params, 7)
    var ro = _i(params, 8)
    _strided(_i(params, 3), fs, fo, n, "x_trees_platt_apply_strided")
    _strided(_i(params, 6), rs, ro, n, "x_trees_platt_apply_strided")
    if n > 0:
        platt_apply_strided(f64_ptr(Int(py=f)) + fo, fs, n, _f(params, 1), _f(params, 2), f64_ptr(Int(py=res)) + ro, rs)
    return PythonObject(n)


def isotonic_predict_strided_binding(
    kx: PythonObject, ky: PythonObject, t: PythonObject, res: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [m, n, t_len, t_stride, t_off, res_len, res_stride, res_off]."""
    _need(params, 8, "x_trees_isotonic_predict_strided")
    var m = _i(params, 0)
    if m < 1:
        raise Error("x_trees_isotonic_predict_strided: no knots")
    var n = _count(_i(params, 1), "x_trees_isotonic_predict_strided")
    var ts = _i(params, 3)
    var to = _i(params, 4)
    var rs = _i(params, 6)
    var ro = _i(params, 7)
    _strided(_i(params, 2), ts, to, n, "x_trees_isotonic_predict_strided")
    _strided(_i(params, 5), rs, ro, n, "x_trees_isotonic_predict_strided")
    if n > 0:
        isotonic_predict_strided(f64_ptr(Int(py=kx)), f64_ptr(Int(py=ky)), m, f64_ptr(Int(py=t)) + to, ts, n,
                                 f64_ptr(Int(py=res)) + ro, rs)
    return PythonObject(n)


def complement_pairs_binding(x: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [n]: x (float64, n x 2)[i, 0] = 1 - x[i, 1]."""
    _need(params, 1, "x_trees_complement_pairs")
    var n = _count(_i(params, 0), "x_trees_complement_pairs")
    if n > 0:
        complement_pairs(f64_ptr(Int(py=x)), n)
    return PythonObject(n)


def indicator_codes_binding(codes: PythonObject, out_i: PythonObject, out_f: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [n, cls]: out_i int32 0/1 per row (codes == cls); out_f (0: none) float64 1.0 / 0.0."""
    _need(params, 2, "x_trees_indicator_codes")
    var n = _count(_i(params, 0), "x_trees_indicator_codes")
    var fa = Int(py=out_f)
    if n > 0:
        var pf = f64_ptr(fa) if fa != 0 else f64_ptr(Int(py=out_i))
        indicator_codes(i32_ptr(Int(py=codes)), n, _i(params, 1), i32_ptr(Int(py=out_i)), pf, fa != 0)
    return PythonObject(n)


def column_f64_binding(src: PythonObject, dst: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [n, c, j]: dst = src[:, j] of a row-major float64 (n, c) block."""
    _need(params, 3, "x_trees_column_f64")
    var n = _count(_i(params, 0), "x_trees_column_f64")
    var c = _i(params, 1)
    var j = _i(params, 2)
    if c < 1 or j < 0 or j >= c:
        raise Error("x_trees_column_f64: column out of range")
    if n > 0:
        column_f64(f64_ptr(Int(py=src)), n, c, j, f64_ptr(Int(py=dst)))
    return PythonObject(n)


def mask_expand_binding(
    x: PythonObject, bg: PythonObject, masks: PythonObject, res: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [nb, d, m]; res float32 (m * nb) * d."""
    _need(params, 3, "x_trees_mask_expand")
    var m = _count(_i(params, 2), "x_trees_mask_expand")
    if m > 0:
        mask_expand(f32_ptr(Int(py=x)), f32_ptr(Int(py=bg)), _i(params, 0), _i(params, 1), i32_ptr(Int(py=masks)), m,
                    f32_ptr(Int(py=res)))
    return PythonObject(m)


def perm_synthetic_binding(
    x: PythonObject, bg: PythonObject, inv: PythonObject, res: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [nb, d, n_perm]; inv int32 n_perm x d (each permutation's
    inverse); res float32 (n_perm (2d + 1) nb) * d, on the device
    (`xtrees/perm_device.mojo`)."""
    _need(params, 3, "x_trees_perm_synthetic")
    var n_perm = _count(_i(params, 2), "x_trees_perm_synthetic")
    if n_perm > 0:
        perm_synthetic(f32_ptr(Int(py=x)), f32_ptr(Int(py=bg)), i32_ptr(Int(py=inv)), f32_ptr(Int(py=res)),
                       _i(params, 0), _i(params, 1), n_perm)
    return PythonObject(n_perm)


def block_mean_binding(y: PythonObject, res: PythonObject, params: PythonObject) raises -> PythonObject:
    """params = [m, nb, k]."""
    _need(params, 3, "x_trees_block_mean")
    var m = _count(_i(params, 0), "x_trees_block_mean")
    var nb = _i(params, 1)
    if nb < 1:
        raise Error("x_trees_block_mean: nb must be >= 1")
    if m > 0:
        block_mean(f32_ptr(Int(py=y)), m, nb, _i(params, 2), f64_ptr(Int(py=res)))
    return PythonObject(m)


def kernel_solve_binding(
    masks: PythonObject, w: PythonObject, ey: PythonObject, fx: PythonObject, fnull: PythonObject,
    phi: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params = [m, d, k]; phi float64 d*k."""
    _need(params, 3, "x_trees_kernel_solve")
    var d = _i(params, 1)
    if d < 1:
        raise Error("x_trees_kernel_solve: d must be >= 1")
    kernel_solve(i32_ptr(Int(py=masks)), f64_ptr(Int(py=w)), _count(_i(params, 0), "x_trees_kernel_solve"), d,
                 f64_ptr(Int(py=ey)), _i(params, 2), f64_ptr(Int(py=fx)), f64_ptr(Int(py=fnull)), f64_ptr(Int(py=phi)))
    return PythonObject(d)


def register(mut m: PythonModuleBuilder) raises:
    """The shared export list; both bindings call this."""
    m.def_function[sample_indices_binding]("x_trees_sample_indices")
    m.def_function[weighted_sample_binding]("x_trees_weighted_sample")
    m.def_function[transpose_f64_binding]("x_trees_transpose_f64")
    m.def_function[bag_rows_binding]("x_trees_bag_rows")
    m.def_function[unseen_rows_binding]("x_trees_unseen_rows")
    m.def_function[gather_f32_binding]("x_trees_gather_f32")
    m.def_function[gather_i32_binding]("x_trees_gather_i32")
    m.def_function[accumulate_binding]("x_trees_accumulate")
    m.def_function[accumulate_onehot_binding]("x_trees_accumulate_onehot")
    m.def_function[accumulate_cols_binding]("x_trees_accumulate_cols")
    m.def_function[accumulate_rows_binding]("x_trees_accumulate_rows")
    m.def_function[argmax_rows_binding]("x_trees_argmax_rows")
    m.def_function[argmax_rows_f32_binding]("x_trees_argmax_rows_f32")
    m.def_function[scale_binding]("x_trees_scale")
    m.def_function[softmax_rows_binding]("x_trees_softmax_rows")
    m.def_function[scale_to_f32_binding]("x_trees_scale_to_f32")
    m.def_function[put_f32_binding]("x_trees_put_f32")
    m.def_function[exact_sum_f32_binding]("x_trees_exact_sum_f32")
    m.def_function[margin2_binding]("x_trees_margin2")
    m.def_function[samme_step_binding]("x_trees_samme_step")
    m.def_function[r2_step_binding]("x_trees_r2_step")
    m.def_function[weighted_median_binding]("x_trees_weighted_median")
    m.def_function[apply_binding]("x_trees_apply")
    m.def_function[gradients_binding]("x_trees_gradients")
    m.def_function[leaf_newton_binding]("x_trees_leaf_newton")
    m.def_function[leaf_newton_rows_binding]("x_trees_leaf_newton_rows")
    m.def_function[tree_score_add_binding]("x_trees_tree_score_add")
    m.def_function[uniform_binding]("x_trees_uniform")
    m.def_function[onehot_leaves_binding]("x_trees_onehot_leaves")
    m.def_function[transpose_f32_binding]("x_trees_transpose_f32")
    m.def_function[check_weights_f32_binding]("x_trees_check_weights_f32")
    m.def_function[mul_f32_binding]("x_trees_mul_f32")
    m.def_function[log64_binding]("x_trees_log64")
    m.def_function[normalize_rows_binding]("x_trees_normalize_rows")
    m.def_function[logit_binding]("x_trees_logit")
    m.def_function[scatter_binding]("x_trees_scatter")
    m.def_function[platt_fit_binding]("x_trees_platt_fit")
    m.def_function[platt_apply_binding]("x_trees_platt_apply")
    m.def_function[isotonic_fit_binding]("x_trees_isotonic_fit")
    m.def_function[isotonic_predict_binding]("x_trees_isotonic_predict")
    m.def_function[platt_apply_strided_binding]("x_trees_platt_apply_strided")
    m.def_function[isotonic_predict_strided_binding]("x_trees_isotonic_predict_strided")
    m.def_function[complement_pairs_binding]("x_trees_complement_pairs")
    m.def_function[indicator_codes_binding]("x_trees_indicator_codes")
    m.def_function[column_f64_binding]("x_trees_column_f64")
    m.def_function[mask_expand_binding]("x_trees_mask_expand")
    m.def_function[block_mean_binding]("x_trees_block_mean")
    m.def_function[perm_synthetic_binding]("x_trees_perm_synthetic")
    m.def_function[kernel_solve_binding]("x_trees_kernel_solve")
