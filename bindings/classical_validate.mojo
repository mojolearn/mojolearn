# SPDX-License-Identifier: Apache-2.0
# scikit-learn's input checks for the native LinearRegression/Ridge owner, in
# the order and with the wording the public estimators have always raised:
# sparse, complex, rank and empty X; a missing, complex, column or non-finite
# y; inconsistent lengths; the sample_weight shape. The checks themselves are
# the one admission path, bindings/admitted_input.mojo; these are its
# classical names.
from bindings.native_loaded_module import native_loaded_module
from bindings.admitted_call import AdmittedCall
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type
# The one admission path (lane consol-a0-admission): the X, y and
# sample_weight checks live in bindings/admitted_input.mojo; these names are
# re-exported here for the doors that import them from this module.
from bindings.admitted_input import (
    AdmittedX, admit_x, admit_x_unrecorded, admit_y, admit_weights, RESHAPE_HINT, classical_view,
    classical_refuse_sparse, classical_weight_refusal, _cast_x,
)


def classical_estimator_name(model: PythonObject) raises -> String:
    var b = native_loaded_module("builtins")
    var cls = model if py_type(model) is b.type else py_type(model)
    return String(cls.__name__)


def classical_check_x_snapshot(source: PythonObject, est: String, context: PythonObject,
    admitted: Bool = False) raises -> AdmittedX:
    """Boundary owner plus the single admitted native address/shape snapshot."""
    return admit_x_unrecorded(source, est, context, admitted)


def classical_check_x(source: PythonObject, est: String, context: PythonObject,
                      admitted: Bool = False) raises -> PythonObject:
    """X for fit, predict or score: refused with scikit-learn's wording, then
    cast to C-ordered float32 (the input itself when it already is one)."""
    var x = admit_x_unrecorded(source, est, context, admitted)
    return x.array


def classical_check_x(mut call: AdmittedCall, source: PythonObject, est: String, context: PythonObject,
                      position: Int, admitted: Bool = False) raises -> PythonObject:
    """As above, recording the checked X on the door's admission record at
    `position` (bindings/admitted_input.mojo `admit_x`)."""
    var x = admit_x(call, source, est, context, position, admitted)
    return x.array


def classical_check_y(source: PythonObject, rows: Int, est: String, context: PythonObject,
                      admitted: Bool = False) raises -> PythonObject:
    """The fit target: present, real, one column (a column vector warns and is
    raveled), finite and as long as X."""
    return admit_y(source, rows, est, context, admitted)


def classical_check_weights_input(source: PythonObject, rows: Int, context: PythonObject,
                                  admitted: Bool = False) raises -> PythonObject:
    """sample_weight as rows float32 words: None stays None, a scalar fills
    every row; scikit-learn's shape, finiteness, sign and all-zero refusals."""
    return admit_weights(source, rows, context, admitted)


def classical_check_width(cols: Int, fitted: Int, est: String) raises:
    if cols != fitted:
        raise Error("X has " + String(cols) + " features, but " + est + " is expecting " + String(fitted) + " features as input.")
