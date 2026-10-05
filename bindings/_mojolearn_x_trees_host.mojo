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
from xtrees.dart_host import DART_HOST, dart_open, dart_step, dart_add, dart_close, dart_predict


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


# ---- DART's boosting round, host twin (lane fam2-forests, `IDN_DART_DEVICE`;
# xtrees/dart_host.mojo). The GPU binding's four entries, same names, same
# arguments, registered only under the same gate so the Python layer takes
# the same loop on the host column as on an IDENTICAL GPU build.
def _dart_ints(params: PythonObject, n: Int, who: String) raises -> List[Int]:
    if len(params) != n:
        raise Error(who + ": params must hold " + String(n) + " values")
    var out = List[Int]()
    for i in range(n):
        out.append(Int(py=params[i]))
    return out^


def dart_open_binding(x: PythonObject, y: PythonObject, inits: PythonObject, params: PythonObject) raises -> PythonObject:
    """x float32 n x d row-major, y float32 n, inits float32 k; params =
    [n, d, k, kind, n_iterations, node_cap]. Returns the session handle."""
    var p = _dart_ints(params, 6, "x_trees_dart_open")
    return PythonObject(dart_open(Int(py=x), Int(py=y), Int(py=inits), p[0], p[1], p[2], p[3], p[4], p[5]))


def dart_step_binding(handle: PythonObject, coef: PythonObject, thr: PythonObject, flags: PythonObject,
                      bad: PythonObject, targets: PythonObject, params: PythonObject) raises -> PythonObject:
    """coef float32 t * k, thr int64 t (in), flags int32 t and bad int32 1
    (out), targets = [k float32 n addresses] (out); params = [t, drop_seed,
    iteration, skip_thr]."""
    var p = _dart_ints(params, 4, "x_trees_dart_step")
    var outs = List[Int]()
    for i in range(len(targets)):
        outs.append(Int(py=targets[i]))
    dart_step(Int(py=handle), Int(py=coef), Int(py=thr), Int(py=flags), Int(py=bad), outs, p[0], p[1], p[2], p[3])
    return PythonObject(p[0])


def dart_add_binding(handle: PythonObject, colid: PythonObject, quesval: PythonObject, left: PythonObject,
                     values: PythonObject, rows: PythonObject, params: PythonObject) raises -> PythonObject:
    """The new tree's forest arrays (int32 / float32 / int32, colid in X's
    columns), values float32 n_nodes (out), rows int32 m (the round's bag
    rows, ascending; any address when m is 0); params = [tree, class, lo,
    n_nodes, shrink, factor, reg_lambda, reg_alpha, max_delta_step, m]
    (m = 0: the leaf sums over every row)."""
    if len(params) != 10:
        raise Error("x_trees_dart_add: params must hold 10 values")
    dart_add(Int(py=handle), Int(py=colid), Int(py=quesval), Int(py=left), Int(py=values), Int(py=params[0]),
             Int(py=params[1]), Int(py=params[2]), Int(py=params[3]), Float64(py=params[4]), Float64(py=params[5]),
             Float64(py=params[6]), Float64(py=params[7]), Float64(py=params[8]), Int(py=rows), Int(py=params[9]))
    return PythonObject(Int(py=params[3]))


def dart_close_binding(handle: PythonObject, bad: PythonObject) raises -> PythonObject:
    """Writes the bad word (int32 1) and frees the session."""
    dart_close(Int(py=handle), Int(py=bad))
    return PythonObject(0)


def dart_idn_binding() raises -> PythonObject:
    """Present only under `IDN_DART_DEVICE`: the IDENTICAL DART round (the
    Python layer then takes it with or without a forest data session)."""
    return PythonObject(1)


def _dart_addr_list(v: PythonObject, nt: Int, who: String) raises -> List[Int]:
    if len(v) != nt:
        raise Error(who + ": one address per tree")
    var out = List[Int]()
    for j in range(nt):  # glue: T tree addresses
        out.append(Int(py=v[j]))
    return out^


def dart_predict_binding(x: PythonObject, forest: PythonObject, sizes: PythonObject, coefs: PythonObject,
                         inits: PythonObject, dst: PythonObject, params: PythonObject) raises -> PythonObject:
    """DART's raw score (lane cpu2-l5-trees, xtrees/dart_*.mojo
    `dart_predict`): x float32 n x d row-major; forest = [colid addresses,
    quesval addresses, left addresses, leaf value addresses], one per tree
    (int32 / float32 / int32 / float32, each at the tree's first node);
    sizes = the T node counts; coefs = the T float64 coefficients (tree j is
    class j % k); inits = the k float64 class starts; out float64 k * n
    class-major (out); params = [n, d, k]."""
    var p = _dart_ints(params, 3, "x_trees_dart_predict")
    if len(forest) != 4:
        raise Error("x_trees_dart_predict: forest must hold 4 address lists")
    var nt = len(sizes)
    if len(coefs) != nt:
        raise Error("x_trees_dart_predict: one coefficient per tree")
    var sz = List[Int]()
    var cf = List[Float64]()
    for j in range(nt):  # glue: T tree sizes and coefficients
        sz.append(Int(py=sizes[j]))
        cf.append(Float64(py=coefs[j]))
    var iv = List[Float64]()
    for c in range(len(inits)):  # glue: k class starts
        iv.append(Float64(py=inits[c]))
    dart_predict(Int(py=x), _dart_addr_list(forest[0], nt, "x_trees_dart_predict"),
                 _dart_addr_list(forest[1], nt, "x_trees_dart_predict"),
                 _dart_addr_list(forest[2], nt, "x_trees_dart_predict"),
                 _dart_addr_list(forest[3], nt, "x_trees_dart_predict"), sz, cf, iv, Int(py=dst), p[0], p[1], p[2])
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
        # lane cpu2-l5-trees: DART predict, the device entry's host twin (every host build)
        m.def_function[dart_predict_binding]("x_trees_dart_predict")
        comptime if DART_HOST:
            m.def_function[dart_open_binding]("x_trees_dart_open")
            m.def_function[dart_step_binding]("x_trees_dart_step")
            m.def_function[dart_add_binding]("x_trees_dart_add")
            m.def_function[dart_close_binding]("x_trees_dart_close")
            m.def_function[dart_idn_binding]("x_trees_dart_idn")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_trees_host: ", e))
