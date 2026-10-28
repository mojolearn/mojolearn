# SPDX-License-Identifier: Apache-2.0
# The shared pieces of the Gaussian process owners (bindings/gp_regressor_native.mojo,
# bindings/gp_classifier_native.mojo): the binding's compile-time mode cross-check
# (`_extension`), its capability word (`gp_idn_caps`), `numeric_mode_used`, and the
# kernel objects read natively from their instance fields: the postfix spec
# `_kernel_arrays` packed (bindings/archive_schema_native.mojo reads the same node
# layouts for `save`), the free hyperparameters `_hyper` listed, and the copy
# `_replace` / `_with_free_values` built. No method of a kernel class runs.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.mode_metadata_native import _plan
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.family_common import family_repr, family_is_str, family_call, family_as_f32
from bindings.archive_schema_native import _schema_kernel_nodes, _schema_label_store


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
    var getter = neural_descriptor_entry(binding, "gp_numeric_mode", False)
    if getter is not None:
        var got = Int(py=family_call(b.int, family_call(getter)))
        var wanted = py_get(context["family_gp_mode_codes"], want, PythonObject(None))
        if wanted is None or Int(py=wanted) != got:
            raise Error("RuntimeError: " + prefix + ": numeric_mode=" + family_repr(want) + " was requested but "
                        + String(py_fields(binding)["__name__"]) + " reports compile-time mode code "
                        + String(got) + "; " + rebuild)
    return binding


def gp_caps(binding: PythonObject) raises -> Int:
    """`int(ext.gp_idn_caps())`, 0 when the binding has no such entry (the host
    binding and the `_OFF` builds): bit 1 is gpc_fit_all / gpc_predict_all,
    bit 2 gpr_fit_ptr / gpr_predict_ptr."""
    var b = native_loaded_module("builtins")
    var fn_obj = neural_descriptor_entry(binding, "gp_idn_caps", False)
    if fn_obj is None:
        return 0
    return Int(py=family_call(b.int, family_call(fn_obj)))


def gp_entry(binding: PythonObject, name: String) raises -> PythonObject:
    """`ext.<name>`: an attribute lookup on the binding, refused as it was."""
    return neural_descriptor_entry(binding, name)


def gp_numeric_mode_used(binding: PythonObject, context: PythonObject) raises -> PythonObject:
    """`NumericModeMixin.numeric_mode_used` on the resolved binding."""
    var b = native_loaded_module("builtins")
    var full = String(py_fields(binding)["__name__"])
    var dotted = full.split(".")
    var name = String(dotted[len(dotted) - 1])
    var stem = String("mojolearn")
    if name != "_mojolearn":
        stem = String(name.removeprefix("_mojolearn_"))
    var getter = neural_descriptor_entry(binding, stem + "_numeric_mode", False)
    if getter is not None:
        var code = family_call(getter)
        var table = context["backend_fields"]["_CODE_MODE"]
        if not py_contains(table, code):
            raise Error("RuntimeError: mojolearn: " + name + " returned unknown numeric mode " + family_repr(code))
        return table[code]
    if len(dotted) < 2:
        raise Error("IndexError: list index out of range")
    return PythonObject(String(dotted[len(dotted) - 2]))


def gp_model_kernel(fields: PythonObject) raises -> PythonObject:
    """`getattr(self, "kernel_", None) or self._kernel`."""
    var b = native_loaded_module("builtins")
    var fitted = py_get(fields, PythonObject("kernel_"), PythonObject(None))
    if fitted is not None:
        return fitted
    if not py_contains(fields, PythonObject("_kernel")):
        raise Error("AttributeError: object has no attribute '_kernel'")
    return fields["_kernel"]


def gp_kernel_arrays(kernel: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_kernel_arrays(kernel)`: (kinds <i4, params <f4, ls_len <i4, ls <f4,
    n_ls), the postfix nodes packed with `Array.from_list`; an empty
    length-scale table is one unused 1.0."""
    var b = native_loaded_module("builtins")
    var nodes = Python.list()
    _schema_kernel_nodes(kernel, context["family_gp_kernel_nodes"], nodes)
    var kinds = Python.list()
    var params = Python.list()
    var lengths = Python.list()
    var scales = Python.list()
    for i in range(len(nodes)):  # driver-loop: over the kernel's postfix nodes
        var node = nodes[i]
        _ = py_call(kinds.append, family_call(b.int, node[0]))
        _ = py_call(params.append, family_call(b.float, node[1]))
        _ = py_call(lengths.append, PythonObject(len(node[2])))
        for j in range(len(node[2])):  # driver-loop: over one node's length scales
            _ = py_call(scales.append, node[2][j])
    var n_ls = len(scales)
    if n_ls == 0:
        _ = py_call(scales.append, PythonObject(Float64(1.0)))
    return Python.tuple(_schema_label_store(kinds, "i", "<i4", context),
                        _schema_label_store(params, "f", "<f4", context),
                        _schema_label_store(lengths, "i", "<i4", context),
                        _schema_label_store(scales, "f", "<f4", context), PythonObject(n_ls))


def _gp_kernel_class(kernel: PythonObject, table: PythonObject) raises -> PythonObject:
    """The class of `kernel`'s ancestry that `table` connects (the walk
    archive_schema_native's layout lookup makes)."""
    var b = native_loaded_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(kernel))
    for i in range(len(mro)):  # driver-loop: over the kernel class ancestry metadata
        if py_contains(table, mro[i]):
            return mro[i]
        if py_contains(py_call(b.type.__dict__["__dict__"].__get__, mro[i]), "_nodes"):
            break
    raise Error("TypeError: mojolearn: a " + String(py_type(kernel).__name__) + " kernel has no saved node layout")


def _gp_fixed(bounds: PythonObject) raises -> Bool:
    """`bounds == "fixed"` for the stored bounds (the str or a list of pairs)."""
    return family_is_str(bounds) and String(bounds) == "fixed"


def gp_free(kernel: PythonObject, flags: PythonObject, values: PythonObject, bounds: PythonObject,
            context: PythonObject) raises:
    """`_free_flags()`, `_free_values()` and `_free_bounds()` in one postfix
    walk of `_hyper()`: a flag per node (1 for a leaf whose bounds are not
    "fixed"), the free leaves' values and their (low, high) pairs."""
    var nodes_table = context["family_gp_kernel_nodes"]
    var cls = _gp_kernel_class(kernel, nodes_table)
    var layout = nodes_table[cls]
    var fields = py_fields(kernel)
    var how = String(layout[0])
    if how == "pair":
        gp_free(fields[layout[1]], flags, values, bounds, context)
        gp_free(fields[layout[2]], flags, values, bounds, context)
        _ = py_call(flags.append, PythonObject(0))
    elif how == "nodes":
        var saved = fields[layout[1]]
        for _ in range(len(saved)):  # driver-loop: over a saved kernel's postfix nodes (all fixed)
            _ = py_call(flags.append, PythonObject(0))
    elif how == "leaf":
        var pairs = fields[context["family_gp_kernel_bounds"][cls]]
        if _gp_fixed(pairs):
            _ = py_call(flags.append, PythonObject(0))
            return
        _ = py_call(flags.append, PythonObject(1))
        if layout[3] is not None:
            var scales = fields[layout[3]]
            for i in range(len(scales)):  # driver-loop: over one leaf's length scales
                _ = py_call(values.append, scales[i])
        else:
            _ = py_call(values.append, fields[layout[2]])
        for i in range(len(pairs)):  # driver-loop: over one leaf's bound pairs
            _ = py_call(bounds.append, pairs[i])
    else:
        raise Error("RuntimeError: unknown kernel node layout " + how)


def gp_n_dims(kernel: PythonObject, context: PythonObject) raises -> Int:
    """`kernel.n_dims`."""
    var flags = Python.list()
    var values = Python.list()
    var bounds = Python.list()
    gp_free(kernel, flags, values, bounds, context)
    return len(values)


def gp_replace(kernel: PythonObject, values: PythonObject, mut pos: Int, context: PythonObject) raises -> PythonObject:
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
                    _ = py_call(taken.append, family_call(b.float, values[pos]))
                    pos += 1
                else:
                    _ = py_call(taken.append, family_call(b.float, scales[i]))
            py_setitem(own, layout[3], taken)
        elif free:
            py_setitem(own, layout[2], family_call(b.float, values[pos]))
            pos += 1
    else:
        raise Error("RuntimeError: unknown kernel node layout " + how)
    return fresh


def gp_with_free_values(kernel: PythonObject, values: PythonObject, theta: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """`kernel._with_free_values(values, theta)`: the copy with `_theta_fit`
    set to the float list `theta` when it is not None."""
    var b = native_loaded_module("builtins")
    var pos = 0
    var out = gp_replace(kernel, values, pos, context)
    if theta is not None:
        var fitted = Python.list()
        for i in range(len(theta)):  # driver-loop: over the kernel hyperparameters (n_dims entries)
            _ = py_call(fitted.append, family_call(b.float, theta[i]))
        py_setitem(py_fields(out), "_theta_fit", fitted)
    return out


def gp_f32(fields: PythonObject, name: String, ndim: Int, context: PythonObject) raises -> PythonObject:
    """A fitted float32 array field as the C-order float32 buffer the
    binding reads (`addr_ro`)."""
    if not py_contains(fields, PythonObject(name)):
        raise Error("AttributeError: object has no attribute '" + name + "'")
    return family_as_f32(fields[name], ndim, name, context)
