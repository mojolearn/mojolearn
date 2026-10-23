# SPDX-License-Identifier: Apache-2.0
# Shared pieces of the decomposition expansion's native owners
# (bindings/decomp_native.mojo and the per-class files it dispatches to):
# instance fields with Python's AttributeError, the fitted numeric mode's
# connected x_decomp entries, mojolearn Array outputs and scalar reads.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_type, py_contains, py_setitem
from bindings.classical_arrays import classical_typed

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]


def decomp_attr(model: PythonObject, fields: PythonObject, name: String) raises -> PythonObject:
    """`model.<name>` for an instance field, with Python's AttributeError."""
    if not py_contains(fields, name):
        raise Error("AttributeError: '" + String(py_type(model).__name__) + "' object has no attribute '" + name + "'")
    return fields[name]


def decomp_int(value: PythonObject) raises -> Int:
    return Int(py=py_call(Python.import_module("builtins").int, value))


def decomp_float(value: PythonObject) raises -> Float64:
    return Float64(py=py_call(Python.import_module("builtins").float, value))


def decomp_lt(a: PythonObject, b: PythonObject) raises -> Bool:
    """`a < b` with Python's comparison."""
    return Bool(py=py_call(Python.import_module("operator").lt, a, b))


def decomp_le(a: PythonObject, b: PythonObject) raises -> Bool:
    return Bool(py=py_call(Python.import_module("operator").le, a, b))


def decomp_eq(a: PythonObject, b: PythonObject) raises -> Bool:
    return Bool(py=py_call(Python.import_module("operator").eq, a, b))


def decomp_repr(value: PythonObject) raises -> String:
    return String(py_call(Python.import_module("builtins").repr, value))


def decomp_set_mode(fields: PythonObject, context: PythonObject) raises:
    # `self.numeric_mode_ = _mode(self.numeric_mode)`: the process mode when
    # None, else the value as given.
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    py_setitem(fields, "numeric_mode_", mode)


def decomp_entries(model: PythonObject, fields: PythonObject, names: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """The connected _mojolearn_x_decomp entries of the fitted mode
    (`_Kit(self.numeric_mode_)`); every name in `names` must be exported."""
    var b = Python.import_module("builtins")
    var mode = decomp_attr(model, fields, "numeric_mode_")
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    var name = py_call(py_call(mode.strip).lower)
    if String(name) != "identical" and String(name) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var entries = context["decomp_entries"][name]
    if py_type(entries) is b.str:
        raise Error(String(entries))
    for k in range(len(names)):  # driver-loop: over the owner's required entry names
        if py_get(entries, names[k]) is None:
            raise Error("NotImplementedError: mojolearn " + String(py_type(model).__name__) + ": this build of "
                        + "_mojolearn_x_decomp does not export " + String(names[k])
                        + "; rebuild it with bindings/build_x_decomp.sh")
    return entries


def decomp_f32(shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """A zero-filled C-order float32 mojolearn Array (`_M.out`)."""
    return classical_typed(shape, "<f4", context["output_context"])


def decomp_check(model: PythonObject, fields: PythonObject, attr: String) raises:
    """`_Base._check`: RuntimeError until `attr` is fitted."""
    if not py_contains(fields, attr):
        raise Error("RuntimeError: " + String(py_type(model).__name__) + " is not fitted; call fit first")


def decomp_min_samples(n: Int, d: Int, who: String, minimum: Int = 2) raises:
    """`_min_samples`: scikit-learn's check_array(ensure_min_samples=...) wording."""
    if n < minimum:
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(d)
                    + ")) while a minimum of " + String(minimum) + " is required by " + who + ".")


def decomp_seed(random_state: PythonObject) raises -> Int:
    """`_seed_of`: None is 0; a bool or a non-int refused."""
    var b = Python.import_module("builtins")
    if random_state is None:
        return 0
    if py_type(random_state) is b.bool or not Bool(py=py_call(b.isinstance, random_state, b.int)):
        raise Error("TypeError: random_state must be None or an int (the counter-based RNG takes an integer seed)")
    # mod 2^32 with Python integer semantics: every consumer masks it so
    return Int(py=py_call(b.int.__dict__["__and__"], random_state, PythonObject(0xFFFFFFFF)))


def decomp_gemm_check(k: Int, k2: Int) raises:
    """`_Kit.mm`'s inner-dimension refusal."""
    if k != k2:
        raise Error("x_decomp: gemm inner dimensions " + String(k) + " and " + String(k2) + " differ")
