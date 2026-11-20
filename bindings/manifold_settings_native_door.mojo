# SPDX-License-Identifier: Apache-2.0
# Operation-local settings and retained state references for manifold owners.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_attributes_door import classical_connect_numpy, _type_mro
from bindings.mode_metadata_native_door import _class_fields
from bindings.classical_pandas_door import classical_admit_value
from bindings.native_array_storage_door import native_array_from_bytes
from bindings.manifold_scalar_native_door import ManifoldScalarSnapshot, manifold_buffer_bytes, manifold_scalar_type, _snapshot
from bindings.sequence_dtype_native_door import sequence_dtype_classes, sequence_dtype_instance, sequence_dtype_create
from providers.python.classical_family_admission_pyapi_unpack import classical_family_admit
from bindings.operation_names_mani import (
    MANI_OPN_FAMILY_MANI_CMDS_FIT, MANI_OPN_FAMILY_MANI_CMDS_FIT_TRANSFORM, MANI_OPN_FAMILY_MANI_ISO_FIT,
    MANI_OPN_FAMILY_MANI_ISO_FIT_TRANSFORM, MANI_OPN_FAMILY_MANI_ISO_RECONSTRUCTION_ERROR, MANI_OPN_FAMILY_MANI_ISO_TRANSFORM,
    MANI_OPN_FAMILY_MANI_LLE_FIT, MANI_OPN_FAMILY_MANI_LLE_FIT_TRANSFORM, MANI_OPN_FAMILY_MANI_LLE_TRANSFORM,
    MANI_OPN_FAMILY_MANI_MDS_FIT, MANI_OPN_FAMILY_MANI_MDS_FIT_TRANSFORM, MANI_OPN_FAMILY_MANI_TSNE_FIT,
    MANI_OPN_FAMILY_MANI_TSNE_FIT_TRANSFORM, MANI_OPN_FAMILY_MANI_UMAP_FIT, MANI_OPN_FAMILY_MANI_UMAP_FIT_TRANSFORM,
    MANI_OPN_FAMILY_MANI_UMAP_TRANSFORM,
)
from bindings.mani_keys_unpack import mani_keys
from bindings.py_balanced import py_item


def _setting(value: PythonObject, context: PythonObject, depth: Int = 0,
             freeze_buffers: Bool = True, admitted: Bool = False) raises -> PythonObject:
    # `admitted`: an enclosing list/tuple's classical_admit_value already
    # admitted this element (it walks list/tuple contents recursively), so
    # the element is not admitted a second time, once per nesting level.
    if depth > 64:
        raise Error("TypeError: manifold settings exceed the supported nesting depth")
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    var scalar = manifold_scalar_type(value, context)
    if not scalar and not admitted:
        classical_admit_value(value, context)
    elif freeze_buffers:
        # Immutable native scalar storage is captured before any numerical
        # cell can release the GIL. Number conversion/errors stay deferred to
        # the algorithm's existing validation point.
        _ = _snapshot(value, context)
    if cls is b.list or cls is b.tuple:
        var result = Python.list()
        for i in range(len(value)):  # driver-loop: copy configuration contents before execution can release the GIL
            _ = py_call(result.append, _setting(value[i], context, depth + 1, freeze_buffers, True))
        return py_call(b.tuple, result) if cls is b.tuple else result
    if cls is b.dict:
        var result = Python.dict()
        var keys = py_call(b.list, value)
        for i in range(len(keys)):  # driver-loop: configuration mapping keys before lookup or comparison
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: manifold configuration mappings require exact string keys")
            py_setitem(result, keys[i], _setting(value[keys[i]], context, depth + 1, freeze_buffers))
        return result
    # Both branches above return: the key table is fetched only on the paths
    # that read a key (list, tuple and dict elements built none before).
    var nk = mani_keys()
    if cls is py_get(context, nk[].k_ndarray):
        var dtype = value.dtype
        var native_dtype = sequence_dtype_instance(dtype, sequence_dtype_classes(context))
        for code in ["q", "Q"]:  # small-loop(dtype: platform long-long descriptor aliases)
            native_dtype = native_dtype or py_type(dtype) is py_type(sequence_dtype_create(PythonObject(code), context))
        if not native_dtype:
            raise Error("TypeError: manifold configuration arrays require a built-in native dtype")
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
            if freeze_buffers:
                _ = _snapshot(result, context)
            return result
        if freeze_buffers:
            if String(dtype.kind) == "T":
                # StringDType owns indirect storage, so copying its pointer
                # words would be invalid. The scalar adapter needs only the
                # captured shape and one unboxed value when size is one.
                # Native ndarray getitem transports a unicode/missing-value
                # reference; recursively admit and freeze that reference
                # before any truth, comparison, or conversion can consume it.
                # Multi-element arrays keep their ordinary rank/size errors.
                var captured = _snapshot(value, context, dtype)
                ref state = captured.downcast_value_ptr[ManifoldScalarSnapshot]()[]
                if state.count == 1:
                    state.item = _setting(state.item, context, depth + 1, True)
                return value
            # Keep ndarray scalar conversion/rank semantics, but give the
            # operation private native bytes. This is a buffer copy, never a
            # NumPy numerical conversion or a call to the input's .copy().
            var shape = value.shape
            var store = native_array_from_bytes(String("B"), manifold_buffer_bytes(value))
            var result = py_call(cls, shape, dtype=dtype, buffer=store)
            _ = _snapshot(result, context)
            return result
    elif freeze_buffers and (cls is b.bytearray or cls is b.memoryview or cls is py_item(context, nk[].k_array_type)):
        # Keep the public native buffer type while capturing its scalar text
        # payload now. Later numeric parsing reads the private byte snapshot;
        # truth and error behavior keep the captured source layout.
        _ = _snapshot(value, context)
    return value


# The operation sets of the former suffix tests (lane bw2-manifold): each set
# is written out from the names of bindings/operation_names_mani.mojo.
@always_inline
def manifold_fit_only(operation: Int) -> Bool:
    """The names ending `_fit`."""
    return (operation == MANI_OPN_FAMILY_MANI_CMDS_FIT or operation == MANI_OPN_FAMILY_MANI_ISO_FIT
            or operation == MANI_OPN_FAMILY_MANI_LLE_FIT or operation == MANI_OPN_FAMILY_MANI_MDS_FIT
            or operation == MANI_OPN_FAMILY_MANI_TSNE_FIT or operation == MANI_OPN_FAMILY_MANI_UMAP_FIT)


@always_inline
def manifold_fit_transform(operation: Int) -> Bool:
    """The names ending `_fit_transform`."""
    return (operation == MANI_OPN_FAMILY_MANI_CMDS_FIT_TRANSFORM or operation == MANI_OPN_FAMILY_MANI_ISO_FIT_TRANSFORM
            or operation == MANI_OPN_FAMILY_MANI_LLE_FIT_TRANSFORM or operation == MANI_OPN_FAMILY_MANI_MDS_FIT_TRANSFORM
            or operation == MANI_OPN_FAMILY_MANI_TSNE_FIT_TRANSFORM
            or operation == MANI_OPN_FAMILY_MANI_UMAP_FIT_TRANSFORM)


@always_inline
def manifold_transforms(operation: Int) -> Bool:
    """The names ending `_transform` but not `_fit_transform`."""
    return (operation == MANI_OPN_FAMILY_MANI_ISO_TRANSFORM or operation == MANI_OPN_FAMILY_MANI_LLE_TRANSFORM
            or operation == MANI_OPN_FAMILY_MANI_UMAP_TRANSFORM)


def mani_field(fields: PythonObject, key: PythonObject, name: String) raises -> PythonObject:
    """`fields[name]` by its held key (the family key table), refused as the
    attribute lookup refused it: bindings/family_common_pack.mojo
    `family_field`'s form, which built the key str up to twice per call
    (lane bw2-manifold). `name` is refusal text only."""
    var value = py_get(fields, key, PythonObject(None))
    if value is None and not py_contains(fields, key):
        raise Error("AttributeError: object has no attribute '" + name + "'")
    return value


def _freeze_setting(values: PythonObject, key: PythonObject, context: PythonObject, freeze_buffers: Bool) raises:
    """One configuration field, when present: its private captured copy
    (`_setting` at depth 0) replaces it in the operation's values."""
    if py_contains(values, key):
        py_setitem(values, key, _setting(py_item(values, key), context, 0, freeze_buffers))


struct ManifoldSettings(Copyable, Movable):
    var fields: PythonObject
    var values: PythonObject
    var owners: PythonObject
    var requested: PythonObject
    var fitted: PythonObject
    var default_mode: PythonObject
    var mode: PythonObject

    def __init__(out self, model: PythonObject, operation: Int, context: PythonObject) raises:
        # Admit class/storage hooks before any instance lookup. Retain fitted
        # buffers and freeze configuration contents before execution. Writes
        # still publish into the original admitted dictionary.
        self.fields = classical_family_admit(model)
        var b = native_loaded_module("builtins")
        var nk = mani_keys()
        if manifold_fit_only(operation) or manifold_fit_transform(operation):
            if py_contains(self.fields, nk[].k__mojolearn_after_fit):
                raise Error("TypeError: native manifold fitting does not accept Python after-fit callbacks")
            var ancestry = _type_mro(py_type(model))
            for i in range(len(ancestry)):  # driver-loop: refuse application after-fit hooks before execution
                if py_contains(_class_fields(ancestry[i]), nk[].k__mojolearn_after_fit):
                    raise Error("TypeError: native manifold fitting does not accept Python after-fit callbacks")
        self.values = py_call(b.dict, self.fields)
        # Retain original field owners as well as the private scalar copies;
        # concurrent application field replacement cannot release an exporter
        # while this operation still owns its captured configuration.
        self.owners = py_call(b.tuple, py_call(self.values.values))
        self.default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
        self.requested = py_get(self.values, nk[].k_numeric_mode)
        self.fitted = py_get(self.values, nk[].k_numeric_mode_)
        self.mode = self.default_mode if self.requested is None else self.requested
        if manifold_transforms(operation) or operation == MANI_OPN_FAMILY_MANI_ISO_RECONSTRUCTION_ERROR:
            if operation != MANI_OPN_FAMILY_MANI_UMAP_TRANSFORM:
                self.mode = self.default_mode if self.fitted is None else self.fitted
        classical_connect_numpy(context)
        # The fixed manifold scalar/configuration fields, in the former
        # loop's order, by held key (lane bw2-manifold: the loop built up to
        # three str per name and call). Initialization arrays (`init`) are
        # input buffers, captured by the algorithm's input preparation
        # rather than scalar settings.
        _freeze_setting(self.values, nk[].k_numeric_mode, context, True)
        _freeze_setting(self.values, nk[].k_numeric_mode_, context, True)
        _freeze_setting(self.values, nk[].k_n_components, context, True)
        _freeze_setting(self.values, nk[].k_n_neighbors, context, True)
        _freeze_setting(self.values, nk[].k_radius, context, True)
        _freeze_setting(self.values, nk[].k_metric, context, True)
        _freeze_setting(self.values, nk[].k_metric_params, context, True)
        _freeze_setting(self.values, nk[].k_p, context, True)
        _freeze_setting(self.values, nk[].k_path_method, context, True)
        _freeze_setting(self.values, nk[].k_eigen_solver, context, True)
        _freeze_setting(self.values, nk[].k_method, context, True)
        _freeze_setting(self.values, nk[].k_random_state, context, True)
        _freeze_setting(self.values, nk[].k_reg, context, True)
        _freeze_setting(self.values, nk[].k_hessian_tol, context, True)
        _freeze_setting(self.values, nk[].k_modified_tol, context, True)
        _freeze_setting(self.values, nk[].k_max_iter, context, True)
        _freeze_setting(self.values, nk[].k_eps, context, True)
        _freeze_setting(self.values, nk[].k_init, context, False)
        _freeze_setting(self.values, nk[].k_metric_mds, context, True)
        _freeze_setting(self.values, nk[].k_normalized_stress, context, True)
        _freeze_setting(self.values, nk[].k_n_init, context, True)
        _freeze_setting(self.values, nk[].k_perplexity, context, True)
        _freeze_setting(self.values, nk[].k_early_exaggeration, context, True)
        _freeze_setting(self.values, nk[].k_learning_rate, context, True)
        _freeze_setting(self.values, nk[].k_post_exaggeration_learning_rate_factor, context, True)
        _freeze_setting(self.values, nk[].k_target_metric, context, True)
        _freeze_setting(self.values, nk[].k_target_n_neighbors, context, True)
        _freeze_setting(self.values, nk[].k_target_weight, context, True)
        _freeze_setting(self.values, nk[].k__knn, context, True)
        _freeze_setting(self.values, nk[].k__kind, context, True)
        _freeze_setting(self.values, nk[].k__pw, context, True)
        _freeze_setting(self.values, nk[].k__transform_config, context, True)
        _freeze_setting(self.values, nk[].k__transform_mode, context, True)
        if operation >= MANI_OPN_FAMILY_MANI_UMAP_FIT and operation <= MANI_OPN_FAMILY_MANI_UMAP_TRANSFORM:
            # UMAP-only scalar/configuration fields, same order.
            _freeze_setting(self.values, nk[].k_n_epochs, context, True)
            _freeze_setting(self.values, nk[].k_min_dist, context, True)
            _freeze_setting(self.values, nk[].k_spread, context, True)
            _freeze_setting(self.values, nk[].k_set_op_mix_ratio, context, True)
            _freeze_setting(self.values, nk[].k_local_connectivity, context, True)
            _freeze_setting(self.values, nk[].k_repulsion_strength, context, True)
            _freeze_setting(self.values, nk[].k_negative_sample_rate, context, True)
            _freeze_setting(self.values, nk[].k_densmap, context, True)
            _freeze_setting(self.values, nk[].k_output_metric, context, True)
            _freeze_setting(self.values, nk[].k_metric_kwds, context, True)
            _freeze_setting(self.values, nk[].k_a, context, True)
            _freeze_setting(self.values, nk[].k_b, context, True)
        if py_type(self.default_mode) is not b.str:
            raise Error("TypeError: manifold default numeric mode requires an exact native string")
