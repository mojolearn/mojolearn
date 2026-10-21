# SPDX-License-Identifier: Apache-2.0
# Explicit saved-model CPU inference metadata. The binding cache is connected
# through the existing Host* CPU contract once; no product route selection is
# delegated to Python and the ordinary GPU owner never enters this path.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.sequential_description import sequential_fields
from bindings.sequential_admission import sequential_admit_value
from bindings.sequential_inputs import sequential_input
from bindings.sha256_door import sha256_hexdigest


def classical_host_entries(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    var b = Python.import_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is not None and (py_type(mode) is not b.str or String(mode) != "identical"):
        raise Error("the host model runs identical only; this model was saved with a different numeric_mode")
    # Host*'s inference binding is always CPU/identical. Preserve the old fit
    # helper and score selection separately: those helpers use the process
    # default when numeric_mode is None, even for a Host* subclass.
    var helper_mode = context["backend_fields"]["_DEFAULT_MODE"] if mode is None else mode
    var entries = context["classical_bindings"][helper_mode]
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    if String(entries["estimators_vendor"]()) != "cpu" or Int(py=entries["estimators_numeric_mode"]()) != 1:
        raise Error("RuntimeError: the explicit host model requires its CPU identical binding")
    return entries


def classical_host_hash(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    if not py_contains(fields, "coef_"):
        raise Error("AttributeError: host linear model has no coef_")
    var b = Python.import_module("builtins")
    var value = fields["coef_"]
    var payload = py_call(PythonObject("").encode, "ascii")
    if value is not None:
        sequential_admit_value(value, context)
        var array = sequential_input(value, context)
        payload = py_call(py_call(b.memoryview, array).tobytes, "C")
        _ = array^
    # Match _HostBound.model_sha256: the linear families hash their stored
    # coefficient bytes, not intercept/parameters or the whole ZIP archive.
    return sha256_hexdigest(payload)
