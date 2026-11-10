# SPDX-License-Identifier: Apache-2.0
# The existing Ridge route and arithmetic stay in their selected native
# kernels. This owner replaces only Python preparation and launch decisions.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type


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
