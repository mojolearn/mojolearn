# SPDX-License-Identifier: Apache-2.0
# No Python-defined protocols are invoked by native Sequential execution.
# Admission compares exact type identities before reading attributes or
# walking containers, so subclass properties/iterators cannot run first.
from bindings.native_loaded_module import native_loaded_module, NativeModules
from std.python import Python, PythonObject
from bindings.native_torch_admission import native_torch_exact, native_torch_admit
from bindings.py_balanced import py_call, py_type, py_fields, py_contains, py_get


def _one_of(cls: PythonObject, types: PythonObject) raises -> Bool:
    for k in range(len(types)):  # driver-loop: over connected native ABI type identities
        if cls is types[k]:
            return True
    return False


def _ctypes_array_owner(mut mods: NativeModules, owner_type: PythonObject, context: PythonObject) raises -> Bool:
    """A class made directly from the connected `ctypes.Array` that defines
    no Python buffer hook of its own: its buffer export is ctypes' C
    implementation. The library's uninitialized native output store
    (`_buffer._output_store`, a per-(code, length) ctypes array class) is
    one: Arrays from `astype`, the label encoders and the float64 to float32
    converters own one, and an estimator that kept such an Array as a fitted
    field could not be saved (ARIMA `llf_`, AMD verify of 7c6e0dade)."""
    var base = py_get(context, "ctypes_array_base")
    if base is None:
        return False
    var b = mods.module("builtins")
    var bases = py_call(b.type.__dict__["__bases__"].__get__, owner_type)
    if len(bases) != 1 or bases[0] is not base:
        return False
    var own = py_call(b.type.__dict__["__dict__"].__get__, owner_type)
    return not py_contains(own, "__buffer__") and not py_contains(own, "__release_buffer__")


def _ctypes_array_owner(owner_type: PythonObject, context: PythonObject) raises -> Bool:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _ctypes_array_owner(mods, owner_type, context)


def sequential_integer(value: PythonObject, context: PythonObject) raises -> Bool:
    return _one_of(py_type(value), context["integer_types"])


def _ordinary_torch(cls: PythonObject, context: PythonObject) raises -> Bool:
    return native_torch_exact(cls)


def sequential_admit_value(mut mods: NativeModules, value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: Sequential input containers exceed the supported nesting depth")
    if depth == 0:
        # The connected-builtins check once per admitted value, not once per
        # element: re-validating every builtins export for each of a list's
        # 20000 labels made a forest fit on a Python-list y ~35 s slower
        # (L4, tools/repro_par_forest.py --direct --inputs pool).
        _ = mods.module("builtins")
    var cls = py_type(value)
    if cls is py_type(Python.list()) or cls is py_type(Python.tuple()):
        for k in range(len(value)):  # driver-loop: over native input container owners and metadata
            sequential_admit_value(mods, value[k], context, depth + 1)
        return
    # Exact native values and scalars first (no memoryview is in either
    # table, so the order of these checks does not change any outcome).
    var native_values = py_get(context, "native_value_types")
    var scalars = py_get(context, "scalar_types")
    if (native_values is not None and _one_of(cls, native_values)) or (scalars is not None and _one_of(cls, scalars)):
        return
    var b = mods.module("builtins")
    if _one_of(cls, context["mapping_types"]):
        if cls is context["defaultdict_class"]:
            # An exact C defaultdict can still call a Python default_factory
            # on a later missing lookup. Admit only connected C constructors.
            var factory = value.default_factory
            if factory is not None and not _one_of(factory, context["native_default_factories"]) and not _one_of(factory, context["scalar_types"]):
                raise Error("TypeError: Sequential defaultdict requires a native builtin default factory or None")
        var pairs = py_call(py_type(Python.list()), value.items())
        mods.released()
        for k in range(len(pairs)):  # driver-loop: over native mapping keys and buffer owners
            sequential_admit_value(mods, pairs[k][0], context, depth + 1)
            sequential_admit_value(mods, pairs[k][1], context, depth + 1)
        return
    if cls is b.memoryview:
        # A custom Python exporter may carry release callbacks. The standard
        # buffer types connected at import expose their buffers in native C.
        var owner = value.obj
        var owner_type = py_type(owner)
        if not _one_of(owner_type, context["buffer_owner_types"]) and not _ctypes_array_owner(mods, owner_type, context):
            raise Error("TypeError: Sequential requires a memoryview over a native buffer owner")
        return
    if _ordinary_torch(cls, context):
        mods.released()
        native_torch_admit(value)
        mods.released()
        return
    mods.released()
    raise Error("TypeError: Sequential requires native input types; Python-defined protocols and subclasses are not supported")


def sequential_admit_value(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    sequential_admit_value(mods, value, context, depth)


def _read_fields(operation: String) raises -> PythonObject:
    """The model fields a read-only operation reads, in inventory order, or
    None for an operation that gets the full inventory walk.

    Each list is the exact set of fields the operation's native path reads
    (sequential_description, sequential_weights, sequential_execution):
    layers/table/program/parameter_layers read `_layers`; sample reads
    `input_shape` and `_layers`; forward also reads the fitted metadata and
    `params_`; predict/predict_proba also read `_fit_loss`, `_y_1d` and
    `classes_`. Optimizer state, training history and the other fit fields
    are never read by these operations, so walking them only re-admitted
    values the operation ignores. fit, save, load_weights, targets and the
    description edits (construct/add/append/compile) mutate or serialize
    the whole model and keep the full walk.
    """
    if operation == "layers" or operation == "table" or operation == "program" or operation == "parameter_layers":
        return Python.tuple("_layers")
    if operation == "sample":
        return Python.tuple("input_shape", "_layers")
    if operation == "forward":
        return Python.tuple("numeric_mode", "_layers", "params_", "n_features_in_", "n_outputs_", "_sample_shape")
    if operation == "predict" or operation == "predict_proba":
        return Python.tuple("numeric_mode", "_layers", "params_", "n_features_in_", "n_outputs_", "_sample_shape",
                            "_fit_loss", "_y_1d", "classes_")
    return PythonObject(None)


def sequential_admit(mut mods: NativeModules, model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises:
    var b = mods.module("builtins")
    if operation == "load_checkpoint":
        if model is not context["sequential_class"]:
            raise Error("TypeError: Sequential checkpoint loading requires the native Sequential class")
    elif operation != "initialize":
        if py_type(model) is not context["sequential_class"]:
            raise Error("TypeError: Sequential requires its native public class, without Python subclass hooks")
        var fields = py_fields(model)
        # One admission per public call, before any native read and before the
        # provider call releases the GIL; nothing re-reads a field after it.
        var names = _read_fields(operation)
        if names is None:
            names = context["model_fields"]
        for k in range(len(names)):  # driver-loop: over model metadata and native buffer owners used by execution
            if py_contains(fields, names[k]):
                sequential_admit_value(mods, fields[names[k]], context)
    sequential_admit_value(mods, args, context)
