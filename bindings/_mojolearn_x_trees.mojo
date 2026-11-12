# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The trees expansion lane's binding (algorithm expansion
lane 7): the ensemble glue of `xtrees/ops.mojo`, registered by
`providers/python/xtrees_api_pyapi.mojo::register`. The trees themselves are fitted through the
existing `_mojolearn_rf` / `_mojolearn_gbdt` entry points."""
from core.experiment_guards_trees import TREES_GUARDS_OK
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from xtrees.glue_runs import publish_x_trees_glue, x_trees_glue_vendor
from xtrees.dart_runs_device import publish_x_trees_dart_device
from providers.python.xtrees_api_pyapi_door import register, tree_shap_cache_values_args
from xtrees import shap_device as shap_dev
from xtrees.shap_device import shap_prepare, shap_prepare_retain, tree_shap_values, shap_pair_count


# F13 measurement audit lives in the binding used by public TreeExplainer.
# The original recipe compiled the separate trees binding, so its counter
# could never observe the public x_trees execution.
def x_trees_shap_pair_count_binding() raises -> PythonObject:
    return PythonObject(shap_pair_count())


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


def _shap_forest(forest: PythonObject) raises -> List[Int]:
    if len(forest) != 5:
        raise Error("x_trees tree_shap: forest must hold 5 addresses")
    var out = List[Int]()
    for i in range(5):  # small-loop(the five forest address words)
        out.append(Int(py=forest[i]))
    return out^


def _shap_ints(params: PythonObject, n: Int, who: String) raises -> List[Int]:
    if len(params) != n:
        raise Error(who + ": params must hold " + String(n) + " values")
    var out = List[Int]()
    for i in range(n):  # small-loop(n: five or seven shape parameters): Python parameter list, not data
        var v = Int(py=params[i])
        if v < 0:
            raise Error(who + ": negative count")
        out.append(v)
    return out^


def tree_shap_prepare_binding(forest: PythonObject, tscale: PythonObject, bg: PythonObject, cover: PythonObject,
                              ev: PythonObject, meta: PythonObject, params: PythonObject) raises -> PythonObject:
    """The forest = [offsets, colid, quesval, left, leaves]; tscale Float32 per
    tree; bg Float32 nb x d; cover Int32 per node (out); ev Float32 k (the
    init in, the expected value out); meta Int32 3 (out: widest slot count,
    deepest leaf depth, 0); params = [nb, d, n_trees, k, n_nodes]."""
    var p = _shap_ints(params, 5, "x_trees_tree_shap_prepare")
    if p[0] < 1 or p[0] >= (1 << 24) or p[1] < 1 or p[2] < 1 or p[3] < 1 or p[4] < 1:
        raise Error("x_trees_tree_shap_prepare: needs 1 <= background rows < 2^24, features, trees, outputs, nodes")
    shap_prepare(_shap_forest(forest), Int(py=tscale), Int(py=bg), Int(py=cover), Int(py=ev), Int(py=meta),
                 p[0], p[1], p[2], p[3], p[4])
    return PythonObject(p[2])


def tree_shap_binding(forest: PythonObject, tscale: PythonObject, cover: PythonObject, x: PythonObject,
                      phi: PythonObject, params: PythonObject) raises -> PythonObject:
    """Output phi Float32 n x d x k = the TreeSHAP values of x; params = [n, d,
    n_trees, k, n_nodes, slots, width] (slots and width from the prepare's
    meta words through `shap_path_width`)."""
    var p = _shap_ints(params, 7, "x_trees_tree_shap")
    var w = p[6]
    if w != 8 and w != 16 and w != 32 and w != 64 and w != 128 and w != 256:
        raise Error("x_trees_tree_shap: path width must be 8, 16, ..., 256")
    if p[1] < 1 or p[2] < 1 or p[3] < 1 or p[4] < 1:
        raise Error("x_trees_tree_shap: needs features, trees, outputs, nodes")
    if p[0] > 0:
        tree_shap_values(_shap_forest(forest), Int(py=tscale), Int(py=cover), Int(py=x), Int(py=phi), p[0], p[1],
                         p[2], p[3], p[4], p[5], w)
    return PythonObject(p[0])


# ---- DART (round, predict and the kept forest): no Python entries since lane
# calltable-gbdt-forest-dart (2026-11-11). Their bodies run through this image's
# native DART entry (xtrees/dart_runs_device.mojo, published at init below;
# callers in bindings/trees_public_boost.mojo through xtrees/dart_calls.mojo).
# Removed with them: the x_trees_dart_{open,step,add,close,predict} and
# x_trees_dart_forest_{prepare,predict,release} Python entries and their
# parameter-list unpacking; the DART door was their only caller.


def tree_shap_prepare_retain_binding(forest: PythonObject, tscale: PythonObject, bg: PythonObject,
                                     cover: PythonObject, ev: PythonObject, meta: PythonObject,
                                     params: PythonObject) raises -> PythonObject:
    """`x_trees_tree_shap_prepare` (the same arguments and outputs) that keeps
    the uploaded forest, the device cover and the leaf table under a handle
    (returned) for `x_trees_tree_shap_cache_values`; free it with
    `x_trees_tree_shap_cache_release` (lane gap-shap-nb: one forest upload
    and one table build per explainer, xtrees/shap_device.mojo)."""
    var p = _shap_ints(params, 5, "x_trees_tree_shap_prepare_retain")
    if p[0] < 1 or p[0] >= (1 << 24) or p[1] < 1 or p[2] < 1 or p[3] < 1 or p[4] < 1:
        raise Error("x_trees_tree_shap_prepare_retain: needs 1 <= background rows < 2^24, features, trees, outputs, nodes")
    return PythonObject(shap_prepare_retain(_shap_forest(forest), Int(py=tscale), Int(py=bg), Int(py=cover),
                                            Int(py=ev), Int(py=meta), p[0], p[1], p[2], p[3], p[4]))


def tree_shap_cache_values_binding(handle: PythonObject, x: PythonObject,
                                   phi: PythonObject, params: PythonObject) raises -> PythonObject:
    var a = tree_shap_cache_values_args(params)
    if a[0] > 0:
        shap_dev.shap_cache_values(Int(py=handle), Int(py=x), Int(py=phi), a[0], a[1], a[2])
    return PythonObject(a[0])


def tree_shap_cache_release_binding(handle: PythonObject) raises -> PythonObject:
    shap_dev.shap_cache_release(Int(py=handle))
    return PythonObject(0)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_trees.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_x_trees() -> PythonObject:
    comptime assert TREES_GUARDS_OK, "invalid integrated experiment configuration (trees guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_x_trees")
        register(m)
        # Native publication of the same glue runs (core/trees_entry_table.mojo);
        # the Python entries above remain for the unconverted Python callers
        # (python/mojolearn/_expansion_trees.py) and the boundary tests.
        publish_x_trees_glue(COMPILED_VENDOR)
        m.def_function[tree_shap_prepare_retain_binding]("x_trees_tree_shap_prepare_retain")
        m.def_function[tree_shap_cache_values_binding]("x_trees_tree_shap_cache_values")
        m.def_function[tree_shap_cache_release_binding]("x_trees_tree_shap_cache_release")
        m.def_function[x_trees_shap_pair_count_binding]("x_trees_shap_pair_count")
        m.def_function[tree_shap_prepare_binding]("x_trees_tree_shap_prepare")
        m.def_function[tree_shap_binding]("x_trees_tree_shap")
        m.def_function[numeric_mode_binding]("x_trees_numeric_mode")
        m.def_function[vendor_binding]("x_trees_vendor")
        # The DART entry (round under DART_DEVICE, predict and kept forest always).
        publish_x_trees_dart_device(x_trees_glue_vendor(COMPILED_VENDOR))
        # TOMBSTONE: x_trees_dart_idn (dead native entry: a capability probe nothing reads; the DART owner probes x_trees_dart_open) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_trees.patch
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_trees: ", e))
