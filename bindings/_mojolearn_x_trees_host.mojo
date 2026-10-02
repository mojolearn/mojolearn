# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding of the trees expansion lane (`x_trees` family).

HOST ONLY: no DeviceContext, no kernel. It exports the GPU binding's names
(`xtrees/api.mojo::register`, called here as the GPU binding calls it)
plus `x_trees_numeric_mode` (1) and `x_trees_vendor` ("cpu"), and the four
host read-backs `x_trees_host_{numeric_mode,vendor,column,sabotage}`, so
`_expansion_trees.py` runs unchanged on a CPU-only install through
`_backend._HOST_MODULES`."""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from xtrees.api import register
from xtrees.shap_host import shap_prepare, tree_shap_values
from xtrees.ops import XTREES_HOST_SABOTAGE


def x_trees_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def x_trees_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_trees_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "x_trees host: this binding compiles the CPU column only; pass -D MOJOLEARN_COLUMN_CPU"
        " (bindings/build_x_trees_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def x_trees_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary was built with -D MOJOLEARN_HOST_SABOTAGE=1 (the
    gate's negative control: xtrees/ops.mojo `scale_f64` divides by a
    perturbed divisor). The GPU binding never defines it."""
    return PythonObject(XTREES_HOST_SABOTAGE)


def _shap_forest(forest: PythonObject) raises -> List[Int]:
    if len(forest) != 5:
        raise Error("x_trees tree_shap: forest must hold 5 addresses")
    var out = List[Int]()
    for i in range(5):
        out.append(Int(py=forest[i]))
    return out^


def _shap_ints(params: PythonObject, n: Int, who: String) raises -> List[Int]:
    if len(params) != n:
        raise Error(who + ": params must hold " + String(n) + " values")
    var out = List[Int]()
    for i in range(n):
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


@export
def PyInit__mojolearn_x_trees_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_trees_host")
        m.def_function[x_trees_host_numeric_mode_binding]("x_trees_host_numeric_mode")
        m.def_function[x_trees_host_vendor_binding]("x_trees_host_vendor")
        m.def_function[x_trees_host_column_binding]("x_trees_host_column")
        m.def_function[x_trees_host_sabotage_binding]("x_trees_host_sabotage")
        m.def_function[x_trees_host_numeric_mode_binding]("x_trees_numeric_mode")
        m.def_function[x_trees_host_vendor_binding]("x_trees_vendor")
        # Every GPU-binding name, from the one registration both bindings call.
        register(m)
        m.def_function[tree_shap_prepare_binding]("x_trees_tree_shap_prepare")
        m.def_function[tree_shap_binding]("x_trees_tree_shap")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_trees_host: ", e))
