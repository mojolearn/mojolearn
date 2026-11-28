# SPDX-License-Identifier: Apache-2.0
"""The neighbor metric and RadiusNeighbors refusals read at fit and at load. Moved byte for byte from
bindings/host_archive_native_door.mojo (lane thin-edges-1, 2026-11-28), which still exports both names, so that a
neighbor door that only resolves its metric does not import the archive reader of every family."""
from std.python import Python, PythonObject
from std.math import isfinite
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_setitem, py_item
from bindings.classical_archive_read_door import _archive_builtin_repr
from neighbors.neighbors_knn_run import knn_rbc_metric_resolve
from bindings.host_tree_archive_keys_unpack import host_tree_archive_keys


def host_archive_rbc_metric_object(cls_name: String, metric: PythonObject, p: PythonObject,
                                   context: PythonObject) raises -> PythonObject:
    """`neighbors._resolve_rbc_metric(cls_name, metric, p)` for a saved
    metric name: `(metric_value, metric_arg)`, or the same refusal, by name
    and in the same order."""
    var b = native_loaded_module("builtins")
    var head = "mojolearn " + cls_name + ": "
    var shown = _archive_builtin_repr(metric)
    if py_type(metric) is not b.str:
        raise Error(head + "metric must be a name, got " + shown)
    # The table and the name refusals are typed: neighbors/neighbors_knn_run.mojo
    # `knn_rbc_metric_resolve` (lane py-neighbors 2026-11-14).
    var name = String(py=py_call(metric.lower))
    var value = knn_rbc_metric_resolve(cls_name, name, shown)
    var arg = Float64(2.0)
    var shown_p = _archive_builtin_repr(p)
    if value != 9:
        # `_refuse_inert_p`: only metric='minkowski'/'lp' reads p; a bool,
        # or a p float() refuses, is not the default.
        var is_default = False
        if py_type(p) is not b.bool:
            try:
                is_default = Float64(py=py_call(b.float, p)) == 2.0
            except:
                is_default = False
        if not is_default:
            raise Error(head + "p is read only by metric='minkowski' "
                + "(or 'lp'); metric=" + shown + " never reads it, so p=" + shown_p + " would "
                + "change nothing. Leave p at its default (2) or ask for "
                + "metric='minkowski'.")
    else:
        arg = Float64(py=py_call(b.float, p))
        # `arg != arg or arg == float("inf")`: NaN or +infinity.
        if arg != arg or (arg > 0.0 and not isfinite(arg)):
            raise Error(head + "metric='minkowski' needs a finite "
                + "p >= 1, got " + shown_p + ". p = infinity is metric='chebyshev', "
                + "which IS admitted here.")
        if arg < 1.0:
            raise Error(head + "metric='minkowski' with p=" + shown_p + " is "
                + "REFUSED on the random ball cover. The pruning here IS THE "
                + "TRIANGLE INEQUALITY on the landmark radii "
                + "(neighbors/impl/ball_cover/common.mojo, "
                + "DEVIATION 564), and |x|^p is not subadditive below p = 1, "
                + "so Lp at p < 1 is not a metric at all: at p = 1/2 the "
                + "points (0,0), (1,0), (1,1) give a direct distance of 4 "
                + "against a two-leg path of 1 + 1. p >= 1 is admitted and "
                + "p = 1 is Manhattan. Use NearestNeighbors with "
                + "algorithm='brute' for p < 1, which is exact brute force "
                + "and needs no inequality.")
    return Python.tuple(value, arg)


def host_archive_radius_refusals(fields: PythonObject, cls_name: String, context: PythonObject) raises:
    """`RadiusNeighbors._check_refusals` over the instance fields (at `fit`
    and at `load`): the metric (which sets `_metric_value`/`_metric_arg`),
    then the algorithm, then the radius, in the Python method's wording."""
    var nk = host_tree_archive_keys()
    var b = native_loaded_module("builtins")
    var metric = py_item(fields, nk[].k_metric)
    var algorithm = py_item(fields, nk[].k_algorithm)
    var radius = py_item(fields, nk[].k_radius)
    var resolved = host_archive_rbc_metric_object(cls_name, metric, py_item(fields, nk[].k_p), context)
    py_setitem(fields, nk[].k__metric_value, resolved[0])
    py_setitem(fields, nk[].k__metric_arg, resolved[1])
    if not (py_type(algorithm) is b.str and (String(algorithm) == "auto" or String(algorithm) == "rbc")):
        raise Error("mojolearn RadiusNeighbors: algorithm=" + _archive_builtin_repr(algorithm) + " is "
            + "refused. The index here is cuVS's RANDOM BALL COVER, which "
            + "is none of scikit-learn's three: it is not 'brute' (there "
            + "is an index and it prunes), and it is neither 'ball_tree' "
            + "nor 'kd_tree' (it is a one-level cover over sqrt(n) "
            + "landmarks, not a tree). Naming any of them would describe "
            + "the wrong algorithm, so 'auto' and 'rbc' -- the "
            + "algorithm's own name -- are the two accepted. The "
            + "results are exact either way")
    var value = Float64(py=py_call(b.float, radius))
    if not isfinite(value) or value <= 0:
        raise Error("mojolearn RadiusNeighbors: radius=" + _archive_builtin_repr(radius) + " is "
            + "refused; it must be positive and finite")
