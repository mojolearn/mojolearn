# SPDX-License-Identifier: Apache-2.0
# The shared pieces of the Gaussian process owners (bindings/gp_regressor_native.mojo,
# bindings/gp_classifier_native.mojo): the binding's compile-time mode cross-check
# (`_extension`), its typed descriptor capability word, `numeric_mode_used`, and the
# kernel objects read natively from their instance fields: the postfix spec
# `_kernel_arrays` packed (bindings/archive_schema_native.mojo reads the same node
# layouts for `save`), the free hyperparameters `_hyper` listed, and the copy
# `_replace` / `_with_free_values` built. No method of a kernel class runs.
# The PACK half of the former `bindings/gp_common_native.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/gp_common_native_unpack.mojo`.
from std.python import Python, PythonObject
from core.gp_native_storage import GPFreeParameters, GPKernelStorage
from bindings.gp_kernel_storage_boundary import GPPackedKernel
from bindings.admitted_family import admit_family, admit_family_owned
from bindings.native_loaded_module import native_loaded_module
from bindings.mode_metadata_native import _plan
from bindings.gp_native_boundary import GPConnection
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from std.memory import unsafe_memcpy
from bindings.family_common import family_repr, family_is_str, family_call, family_array, FAMILY_F32P
from bindings.classical_arrays_door import classical_address
from bindings.archive_schema_native_door import _schema_kernel_layouts
from bindings.gp_kernel_public_native import gp_kernel_native_nodes




from bindings.gp_common_native_unpack import (
    _gp_fixed,
    _gp_kernel_class,
)


def gp_mode_wanted(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`getattr(self, "numeric_mode", None) or _backend.default_mode()`."""
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var want = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    if want is None:
        var metadata = py_fields(native_loaded_module("mojolearn._mode"))["_MODE_METADATA_CONTEXT"]
        var plan = _plan(model, fields, metadata)
        var route = String(plan["route"])
        # cpu-route: explicit HostGP metadata always names identical, even
        # when the process GPU default is fast. Never rewrite either setting.
        want = PythonObject("identical") if route == "host" or route == "host_gp" else context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(want) is not b.str:
        raise Error("TypeError: GP numeric mode requires an exact native string")
    return want


def gp_extension(model: PythonObject, binding: PythonObject, prefix: String, rebuild: String,
                 context: PythonObject) raises -> PythonObject:
    """`_extension`: the binding the shell resolved, its `gp_numeric_mode`
    answer cross-checked against the requested tier. `prefix` is the
    estimator's message head, `rebuild` the sentence after the mode code."""
    var b = native_loaded_module("builtins")
    var want = gp_mode_wanted(model, context)
    var connection = GPConnection(binding)
    var got = connection.mode
    if got >= 0:
        var wanted = py_get(context["family_gp_mode_codes"], want, PythonObject(None))
        if wanted is None or Int(py=wanted) != got:
            raise Error("RuntimeError: " + prefix + ": numeric_mode=" + family_repr(want) + " was requested but "
                        + String(py_fields(binding)["__name__"]) + " reports compile-time mode code "
                        + String(got) + "; " + rebuild)
    return binding


def gp_numeric_mode_used(binding: PythonObject, context: PythonObject) raises -> PythonObject:
    var connection = GPConnection(binding)
    var code = PythonObject(connection.mode)
    var table = context["backend_fields"]["_CODE_MODE"]
    if not py_contains(table, code):
        raise Error("RuntimeError: Mojolearn: GP provider returned unknown numeric mode " + String(connection.mode))
    return table[code]


def gp_model_kernel(fields: PythonObject) raises -> PythonObject:
    """`getattr(self, "kernel_", None) or self._kernel`."""
    var b = native_loaded_module("builtins")
    var fitted = py_get(fields, PythonObject("kernel_"), PythonObject(None))
    if fitted is not None:
        return fitted
    if not py_contains(fields, PythonObject("_kernel")):
        raise Error("AttributeError: object has no attribute '_kernel'")
    return fields["_kernel"]


def gp_replace(kernel: PythonObject, values: List[Float64], mut pos: Int, context: PythonObject) raises -> PythonObject:
    """`kernel._replace(values)`: a new kernel of the same classes whose free
    entries are taken from `values[pos:]` in postfix order (`pos` advances)."""
    var b = native_loaded_module("builtins")
    var nodes_table = context["family_gp_kernel_nodes"]
    var cls = _gp_kernel_class(kernel, nodes_table)
    var layout = nodes_table[cls]
    var fields = py_fields(kernel)
    var how = String(layout[0])
    var fresh = family_call(b.object.__new__, py_type(kernel))
    var own = py_fields(fresh)
    _ = py_call(own.update, fields)
    if how == "pair":
        py_setitem(own, layout[1], gp_replace(fields[layout[1]], values, pos, context))
        py_setitem(own, layout[2], gp_replace(fields[layout[2]], values, pos, context))
    elif how == "nodes":
        py_setitem(own, layout[1], py_call(b.list, fields[layout[1]]))
    elif how == "leaf":
        var bfield = context["family_gp_kernel_bounds"][cls]
        var pairs = fields[bfield]
        var free = not _gp_fixed(pairs)
        if free:
            py_setitem(own, bfield, py_call(b.list, pairs))
        if layout[3] is not None:
            var scales = fields[layout[3]]
            var taken = Python.list()
            for i in range(len(scales)):  # driver-loop: over one leaf's length scales
                if free:
                    _ = py_call(taken.append, PythonObject(values[pos]))
                    pos += 1
                else:
                    _ = py_call(taken.append, family_call(b.float, scales[i]))
            py_setitem(own, layout[3], taken)
        elif free:
            py_setitem(own, layout[2], PythonObject(values[pos]))
            pos += 1
    else:
        raise Error("RuntimeError: unknown kernel node layout " + how)
    return fresh


def gp_with_free_values(kernel: PythonObject, values: List[Float64], theta: List[Float64],
                        context: PythonObject) raises -> PythonObject:
    """`kernel._with_free_values(values, theta)`: the copy with `_theta_fit`
    set to the float list `theta` when it is not None."""
    var b = native_loaded_module("builtins")
    var pos = 0
    var out = gp_replace(kernel, values, pos, context)
    if len(theta) != 0:
        var fitted = Python.list()
        for i in range(len(theta)):  # driver-loop: over the kernel hyperparameters (n_dims entries)
            _ = py_call(fitted.append, PythonObject(theta[i]))
        py_setitem(py_fields(out), "_theta_fit", fitted)
    return out


def gp_f32(fields: PythonObject, name: String, ndim: Int, context: PythonObject) raises -> PythonObject:
    """A fitted float32 array field as the C-order float32 buffer the
    binding reads (`addr_ro`)."""
    if not py_contains(fields, PythonObject(name)):
        raise Error("AttributeError: object has no attribute '" + name + "'")
    return admit_family(fields[name], ndim, name, context).array
