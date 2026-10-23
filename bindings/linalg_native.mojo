# SPDX-License-Identifier: Apache-2.0
# Cholesky `fit`, one native call (python/mojolearn/_cholesky_impl.py). The
# shell hands over `_door_binding()`: the binding that serves this object's
# tier (the CPU binding on a CPU-only install, whose mode read-back is
# `linalg_numeric_mode`; the GPU binding's is `gp_numeric_mode`). This owner makes the
# Python method's refusals, cross-checks the mode (`_extension`), reads the
# profile's ridge for `jitter=None`, runs the binding's `cholesky_factor`
# with the argument lists bindings/_mojolearn_gp.mojo documents, and stores
# the factor. The same kernel on the same inputs: the bits do not move.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    FAMILY_F64P, family_repr, family_call, family_entry, family_array, family_as_f32, family_field,
    family_shape_repr,
)


def _chol_extension(model: PythonObject, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Cholesky._extension`: the binary's own compile-time mode must be the
    one requested."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var want = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    if want is None or not Bool(py=py_call(b.bool, want)):
        want = context["backend_fields"]["_DEFAULT_MODE"]
    var readback = PythonObject("gp_numeric_mode")
    if context["backend_fields"]["_CPU_ONLY"] is not None:
        readback = PythonObject("linalg_numeric_mode")
    var getter = py_call(b.getattr, binding, readback, PythonObject(None))
    if getter is not None:
        var got = Int(py=family_call(b.int, family_call(getter)))
        var wanted = py_get(context["family_chol_mode_codes"], want, PythonObject(None))
        if wanted is None or Int(py=wanted) != got:
            raise Error("RuntimeError: mojolearn Cholesky: numeric_mode=" + family_repr(want) + " was requested "
                        + "but " + String(py_call(b.getattr, binding, PythonObject("__name__"), binding))
                        + " reports compile-time mode code " + String(got)
                        + "; the binary and the directory it sits in disagree, rebuild "
                        + "it with bash bindings/build_gp.sh (a GPU install) or "
                        + "bindings/build_linalg_host.sh (a CPU-only install)")
    return binding


def _chol_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`Cholesky.fit(A)`: factor A + jitter * I."""
    classical_connect_numpy(context)
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var a = family_as_f32(args[1], 2, "A", context)
    var n = Int(py=a.shape[0])
    var m = Int(py=a.shape[1])
    if n != m:
        raise Error("mojolearn Cholesky: A must be square, got shape " + family_shape_repr(a))
    var jitter = family_field(fields, "jitter")
    if jitter is None:
        var profile = family_entry(_chol_extension(model, args[0], context), "cholesky_profile_jitter", context)
        jitter = family_call(b.float, family_call(profile))
    var l_out = family_array(Python.tuple(n, n), "<f4", context)
    # info, nb, logdet, jitter -- in that order.
    var scalars = family_array(Python.tuple(4), "<f8", context)
    var factor = family_entry(_chol_extension(model, args[0], context), "cholesky_factor", context)
    # ORDER MATCHES bindings/_mojolearn_gp.mojo::cholesky_factor_binding:
    # a, l_out, scalars_out; n, jitter.
    var info = family_call(factor, Python.list(classical_address(a), classical_address(l_out), classical_address(scalars)),
                           Python.list(PythonObject(n), jitter))
    _ = scalars  # keep the Python object alive through the native call above
    _ = l_out  # keep the Python object alive through the native call above
    _ = a  # keep the Python object alive through the native call above
    _ = a^
    var words = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(scalars)))
    _ = scalars  # keep the Python object alive through the native call above
    py_setitem(fields, "n_", PythonObject(n))
    py_setitem(fields, "L_", l_out)
    py_setitem(fields, "info_", family_call(b.int, info))
    _ = scalars  # keep the Python object alive through the native call above
    _ = l_out  # keep the Python object alive through the native call above
    py_setitem(fields, "nb_", PythonObject(Int(words[1])))
    _ = scalars  # keep the Python object alive through the native call above
    py_setitem(fields, "_logdet", PythonObject(words[2]))
    _ = scalars  # keep the Python object alive through the native call above
    py_setitem(fields, "jitter_", PythonObject(words[3]))
    _ = scalars  # keep the Python object alive through the native call above
    _ = scalars^


def linalg_operation(model: PythonObject, operation: String, args: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    if operation == "family_chol_fit":
        _chol_fit(model, args, context)
        return model
    raise Error("RuntimeError: unknown linalg operation " + operation)
