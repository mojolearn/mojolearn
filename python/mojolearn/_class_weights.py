# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Class weights as per-row weights, and the x_trees binding accessor.

Shared by the forests, the expansion trees and LogisticRegression. A module
of its own so that a caller of these two helpers imports neither tree door:
randomforest.py and _expansion_trees.py re-export both names.
"""
import numbers

from . import _portable_math as math
from ._array import Array
from ._buffer import all_finite
from ._labels import is_bool


def _trees_x_bind(est=None):
    from . import _backend
    return _backend.binding("_mojolearn_x_trees", getattr(est, "numeric_mode", None))


def _class_weight_rows(class_weight, classes, codes, xbind=None):
    """cuML 26.08 process_class_weight shape, bounded Float32 weights.

    Python touches only the k class weights. With `xbind` (the estimator's
    x_trees binding) and int32 codes, the per-class counts and the per-row
    weights are the binding's (`x_trees_code_counts`, `x_trees_class_rows`:
    device kernels on a GPU build, cpu2-l5-trees); otherwise the core host
    helpers `bincount_i64` / `gather_rows_bytes` (CPU-only callers). Unlike
    cuML RF's Float64 weights, this engine narrows to Float32. `codes` is an
    int32/int64 Array (or a list of ints).
    """
    if class_weight is None:
        return None
    from ._buffer import _native
    if not isinstance(codes, Array):
        codes = Array.from_list(list(codes), "<i8")
    codes = codes._as_c()
    n = codes.size
    k = len(classes)
    on_binding = xbind is not None and codes.dtype == "<i4"
    if isinstance(class_weight, str):
        if on_binding:
            counts = Array((k,), "<i4")
            xbind.x_trees_code_counts(codes._addr if n else 0, counts._addr, [n, k])
        else:
            counts = Array((k,), "<i8")
            _native("bincount_i64")(codes._addr, 2 if codes.dtype == "<i4" else 3, n, k, counts._addr, 0)
        # n / (k * count) per class in the base binding (lane py-runtime-b;
        # the same binary64 words as the Python quotient)
        counts = counts if counts.dtype == "<i8" else counts.astype("<i8")
        w64 = Array((k,), "<f8")
        if k and int(_native("class_ratio_f64")(counts._addr, k, n, 0, w64._addr)):
            raise ZeroDivisionError("division by zero")
        values = w64.tolist()  # glue: k class weights for the argument checks below
    else:
        unknown = set(class_weight).difference(classes)
        if unknown:
            raise ValueError("class_weight contains labels absent from y")
        values = [class_weight.get(label, 1.0) for label in classes]  # glue: user class_weight dict per class
    if any(is_bool(v) or not isinstance(v, numbers.Real) or not math.isfinite(v) or v < 0 for v in values):  # glue: class weight argument checks
        raise ValueError("class weights must be finite nonnegative real numbers")
    if all(v == 1 for v in values):  # glue: class weight argument checks
        return None
    try:
        narrowed = Array.from_list(values, "<f4")
    except OverflowError:
        raise ValueError("class weights must remain finite and nonzero when positive in Float32") from None
    if not all_finite(narrowed) or any(v > 0 and n == 0 for v, n in zip(values, narrowed)):  # glue: class weight argument checks
        raise ValueError("class weights must remain finite and nonzero when positive in Float32")
    if narrowed.max() <= 0:
        raise ValueError("class weights must have positive total")
    out = Array((n,), "<f4")
    if on_binding:
        # the per-row weight of each row's class: one gather in the binding
        if n:
            xbind.x_trees_class_rows(codes._addr, narrowed._addr, out._addr, [n, k])
        return out
    # the per-row weight of each row's class: one byte-row gather in Mojo
    rows = codes if codes.dtype == "<i8" else codes.astype("<i8")
    if n:
        _native("gather_rows_bytes")(narrowed._addr, out._addr, rows._addr, k, n, 4)  # cpu-route: callers without the trees binding (CPU-only, tests); GPU callers take x_trees_class_rows
    return out
