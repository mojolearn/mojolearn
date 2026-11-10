# SPDX-License-Identifier: Apache-2.0
# The existing Ridge route and arithmetic stay in their selected native
# kernels. This owner replaces only Python preparation and launch decisions.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_array, classical_address


def classical_means(x: PythonObject, rows: Int, cols: Int, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var sums = classical_array(cols, 0, context, True)
    var m64 = classical_array(cols, 0, context, True)
    var m32 = classical_array(cols, 0, context)
    _ = py_call_typed(entries["lm_col_sums"], classical_address(x), classical_address(sums), Python.list(rows, cols))
    _ = py_call_typed(entries["lm_means_finish"], classical_address(sums), classical_address(m64), classical_address(m32), Python.list(cols, rows, 0), PythonObject(0.0))
    _ = sums^
    return Python.tuple(m64, m32)


def classical_center(x: PythonObject, mean: PythonObject, rows: Int, cols: Int,
                     entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var out = classical_array(rows, cols if Int(py=x.ndim) == 2 else 0, context)
    _ = py_call_typed(entries["lm_center"], classical_address(x), classical_address(mean), classical_address(out), Python.list(rows, cols))
    return out


def classical_ols_prepare(x: PythonObject, y: PythonObject, weights: PythonObject, rows: Int, cols: Int,
                          intercept: Bool, mode: Int, prep: PythonObject, context: PythonObject) raises -> PythonObject:
    """`lm_ols_prep`: (work_x, work_y, xmean32, ymean64) of the general
    route in one native call. mode 0 separate means, 1 paired, 2 weighted."""
    var work_x = classical_array(rows, cols, context)
    var work_y = classical_array(rows, 0, context)
    var xm64 = classical_array(cols, 0, context, True)
    var xm32 = classical_array(cols, 0, context)
    var ym64 = classical_array(1, 0, context, True)
    var ym32 = classical_array(1, 0, context)
    var w_address = PythonObject(0) if weights is None else classical_address(weights)
    _ = py_call_typed(prep, Python.list(classical_address(x), classical_address(y), w_address,
                                        classical_address(work_x), classical_address(work_y),
                                        classical_address(xm64), classical_address(xm32),
                                        classical_address(ym64), classical_address(ym32)),
                      Python.list(rows, cols, 1 if intercept else 0, mode))
    _ = x  # keep the Python object alive through the native call above
    _ = y  # keep the Python object alive through the native call above
    _ = weights  # keep the Python object alive through the native call above
    _ = xm64^
    _ = ym32^
    return Python.tuple(work_x, work_y, xm32, ym64)


def classical_ridge_parameters(fields: PythonObject) raises:
    # Checked when fit runs, after the inputs, as the constructor stores them.
    var b = native_loaded_module("builtins")
    var alpha = fields["alpha"]
    var negative = False
    try:
        negative = Bool(py=alpha < PythonObject(0.0))
    except error:
        raise Error("TypeError: " + String(error))
    if negative:
        raise Error("alpha must be non-negative, got " + String(py_call(b.str, alpha)))
    var solver = fields["solver"]
    var name = String(solver) if py_type(solver) is b.str else String("")
    # repr(solver) is built only inside the refusals that show it.
    if name != "auto" and name != "eig" and name != "svd" and name != "cd":
        raise Error("TypeError: solver " + String(py_call(b.repr, solver)) + " is not supported")
    if name == "svd" or name == "cd":
        var shown = String(py_call(b.repr, solver))
        raise Error("NotImplementedError: mojolearn Ridge: solver=" + shown + " is not implemented "
                    + "(ridgeSVD is raft::linalg::svdQR -> cuSOLVER gesvd; 'cd' is cuml/solvers/cd.pyx); "
                    + "solver='eig' (cuML's 'auto') is the implemented arm. See glm/NOT_IMPLEMENTED.tsv")
    if Bool(py=fields["normalize"]):
        raise Error("NotImplementedError: mojolearn Ridge: normalize is not implemented "
                    + "(preprocess.cuh:76-108, the meanvar arm of preProcessData; glm/NOT_IMPLEMENTED.tsv)")
