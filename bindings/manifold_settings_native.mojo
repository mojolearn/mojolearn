# SPDX-License-Identifier: Apache-2.0
# Operation-local settings and retained state references for manifold owners.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_attributes import classical_connect_numpy, _type_mro
from bindings.mode_metadata_native import _class_fields
from bindings.classical_pandas import classical_admit_value
from bindings.native_buffer_owner import native_buffer_view
from bindings.native_array_storage import native_array_from_bytes
from providers.python.classical_family_admission_pyapi import classical_family_admit


def _setting(value: PythonObject, context: PythonObject, depth: Int = 0,
             freeze_buffers: Bool = True) raises -> PythonObject:
    if depth > 64:
        raise Error("TypeError: manifold settings exceed the supported nesting depth")
    classical_admit_value(value, context)
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.list or cls is b.tuple:
        var result = Python.list()
        for i in range(len(value)):  # driver-loop: copy configuration contents before execution can release the GIL
            _ = py_call(result.append, _setting(value[i], context, depth + 1, freeze_buffers))
        return py_call(b.tuple, result) if cls is b.tuple else result
    elif cls is b.dict:
        var result = Python.dict()
        var keys = py_call(b.list, value)
        for i in range(len(keys)):  # driver-loop: configuration mapping keys before lookup or comparison
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: manifold configuration mappings require exact string keys")
            py_setitem(result, keys[i], _setting(value[keys[i]], context, depth + 1, freeze_buffers))
        return result
    elif cls is py_get(context, "ndarray"):
        var dtype = value.dtype
        if Bool(py=dtype.hasobject) and String(dtype.kind) != "O":
            raise Error("TypeError: manifold configuration refuses structured object arrays")
        if String(dtype.kind) == "O":
            var flat = value.flat
            var count = Int(py=value.size)
            var result = py_call(cls, value.shape, dtype=dtype) if freeze_buffers else value
            var output = result.flat
            for i in range(count):  # driver-loop: own admitted object references; raw pointer copying would not retain them
                var item = _setting(flat[i], context, depth + 1, freeze_buffers)
                if freeze_buffers:
                    py_setitem(output, i, item)
            return result
        if freeze_buffers:
            # Keep ndarray scalar conversion/rank semantics, but give the
            # operation private native bytes. This is a buffer copy, never a
            # NumPy numerical conversion or a call to the input's .copy().
            var shape = value.shape
            var buffer = native_buffer_view(value)
            var store = native_array_from_bytes(String("B"), py_call(buffer.tobytes, "C"))
            return py_call(cls, shape, dtype=dtype, buffer=store)
    return value


struct ManifoldSettings(Copyable, Movable):
    var fields: PythonObject
    var values: PythonObject
    var requested: PythonObject
    var fitted: PythonObject
    var default_mode: PythonObject
    var mode: PythonObject

    def __init__(out self, model: PythonObject, operation: String, context: PythonObject) raises:
        # Admit class/storage hooks before any instance lookup. Retain fitted
        # buffers and freeze configuration contents before execution. Writes
        # still publish into the original admitted dictionary.
        self.fields = classical_family_admit(model)
        var b = native_loaded_module("builtins")
        if operation.endswith("_fit") or operation.endswith("_fit_transform"):
            if py_contains(self.fields, "_mojolearn_after_fit"):
                raise Error("TypeError: native manifold fitting does not accept Python after-fit callbacks")
            var ancestry = _type_mro(py_type(model))
            for i in range(len(ancestry)):  # driver-loop: refuse application after-fit hooks before execution
                if py_contains(_class_fields(ancestry[i]), "_mojolearn_after_fit"):
                    raise Error("TypeError: native manifold fitting does not accept Python after-fit callbacks")
        self.values = py_call(b.dict, self.fields)
        self.default_mode = context["backend_fields"]["_DEFAULT_MODE"]
        self.requested = py_get(self.values, "numeric_mode")
        self.fitted = py_get(self.values, "numeric_mode_")
        self.mode = self.default_mode if self.requested is None else self.requested
        if (operation.endswith("_transform") and not operation.endswith("_fit_transform")) or operation == "family_mani_iso_reconstruction_error":
            if not operation.startswith("family_mani_umap_"):
                self.mode = self.default_mode if self.fitted is None else self.fitted
        classical_connect_numpy(context)
        for name in ["numeric_mode", "numeric_mode_", "n_components", "n_neighbors", "radius", "metric", "metric_params", "p", "path_method", "eigen_solver", "method", "random_state", "reg", "hessian_tol", "modified_tol", "max_iter", "eps", "init", "metric_mds", "normalized_stress", "n_init", "perplexity", "early_exaggeration", "learning_rate", "post_exaggeration_learning_rate_factor", "target_metric", "target_n_neighbors", "target_weight", "_knn", "_kind", "_pw", "_transform_config", "_transform_mode"]:  # small-loop(settings: fixed manifold scalar/configuration fields)
            if py_contains(self.values, name):
                # Initialization arrays are input buffers, captured by the
                # algorithm's input preparation rather than scalar settings.
                py_setitem(self.values, name, _setting(self.values[name], context, 0, name != "init"))
        if operation.startswith("family_mani_umap_"):
            for name in ["n_epochs", "min_dist", "spread", "set_op_mix_ratio", "local_connectivity", "repulsion_strength", "negative_sample_rate", "densmap", "output_metric", "metric_kwds", "a", "b"]:  # small-loop(settings: UMAP-only scalar/configuration fields)
                if py_contains(self.values, name):
                    py_setitem(self.values, name, _setting(self.values[name], context))
        if py_type(self.default_mode) is not b.str:
            raise Error("TypeError: manifold default numeric mode requires an exact native string")
