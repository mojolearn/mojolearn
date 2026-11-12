# SPDX-License-Identifier: Apache-2.0
# Operation-local PCA query settings. Copies hold admitted references; scalar
# storage is decoded before execution, without NumPy/user conversion slots.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get, py_setitem, py_type
from bindings.sequential_admission_door import sequential_admit_value, _one_of
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.native_scalar_decode_unpack import native_scalar_buffer_dtype
from bindings.classical_exceptions_door import classical_not_fitted_class
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from providers.python.classical_family_admission_pyapi_unpack import classical_family_admit
from providers.python.pca_fit_scalar_pyapi_unpack import PCAFitScalar, pca_fit_scalar_admit
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put
from bindings.py_model_records import records_open, model_generation
from bindings.pca_descriptor_boundary_door import pca_connection, pca_connection_matches, pca_mode_name
from core.pca_query_state import PCAQueryState
from x_decomp.pca_model_run import _pca_query_decoded
from providers.python.pca_query_cache_pyapi_unpack import PCAQueryPins, PCAQueryAdmission, PCAQueryCache


from providers.python.pca_model_pyapi_pack import (
    _pca_query_fields_from,
)


# The native record retains the original generation/identity admission fast
# paths. Buffer shape/dtype/finiteness and mutable scalar storage are still
# admitted on every call; only immutable decoded settings are reusable.
comptime _PCA_QUERY_SLOT = "pca_query_native_v2"


def _pca_query_plain(value: PythonObject, ndarray: PythonObject, decoded: Bool) raises -> Bool:
    # Admission and decoding of an exact builtin scalar depend on its identity
    # only (immutable). An exact ndarray is admitted by type alone; its shape,
    # dtype and values are read again by the per-call state snapshots. An
    # ndarray among the decoded scalars can change value in place: never
    # recorded. Anything else (NumPy scalars, user types) takes the full path.
    var cls = py_type(value)
    if (value is None or cls is py_type(PythonObject(True)) or cls is py_type(PythonObject(0))
            or cls is py_type(PythonObject(0.0)) or cls is py_type(PythonObject(""))):
        return True
    return not decoded and cls is ndarray


def _pca_query_key(state: PythonObject, context: PythonObject) raises -> PCAQueryPins:
    var ndarray = py_get(context, "ndarray")
    if ndarray is None or py_type(state) is not py_type(Python.dict()):
        return PCAQueryPins(False)
    var key = _pca_context_key(state, context)
    var parameters = context["pca_parameters"]
    var stored = context["pca_state"]
    for group in range(2):  # driver-loop: connected settings, then fitted state
        var names = parameters if group == 0 else stored
        for index in range(len(names)):  # driver-loop: admitted field identities, never array values
            var name = names[index]
            if not py_contains(state, name):
                key.absent()
                continue
            var value = state[name]
            if not _pca_query_plain(value, ndarray, _pca_query_decoded(String(name))):
                return PCAQueryPins(False)
            key.append(value)
    return key^


def _pca_context_key(state: PythonObject, context: PythonObject) raises -> PCAQueryPins:
    var key = PCAQueryPins()
    key.append(context)
    key.append(py_get(context, "ndarray"))
    key.append(context["pca_parameters"])
    key.append(context["pca_state"])
    key.append(py_get(context, "vendor"))
    key.append(context["pca_bindings"])
    if py_get(state, "numeric_mode") is None:
        var backend = context["backend_fields"]
        key.append(backend)
        key.append(backend["_DEFAULT_MODE"])
    return key^


def _pca_generation(model: PythonObject) raises -> Int:
    _ = records_open(model)
    return model_generation(model)


def _pca_query_state(fields: PythonObject, connection: Int) raises -> PCAQueryState:
    # Preserve the original post-provider order: fitted-array presence, three
    # integer conversions, dimension refusal, then whitening truth.
    if not py_contains(fields, "components_"):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var fit_rows = Int(py=fields["n_samples_"])
    var features = Int(py=fields["n_features_in_"])
    var count = Int(py=fields["n_components_"])
    if features < 1 or count < 0:
        raise Error("PCA fitted dimensions are invalid")
    return PCAQueryState(fit_rows, features, count, Bool(py=fields["whiten"]), connection)


def pca_query_admitted(model: PythonObject, context: PythonObject, operation: String) raises -> PCAQueryAdmission:
    var state = classical_family_admit(model)
    var mods = NativeModules()
    var generation = _pca_generation(model)
    var held = prepared_get(mods, _PCA_QUERY_SLOT, model)
    if generation != 0 and held is not None:
        var cache = held.downcast_value_ptr[PCAQueryCache]()
        var context_key = _pca_context_key(state, context)
        if (cache[].generation == generation and cache[].context_key.identity.same(context_key.identity)
                and pca_connection_matches(cache[].state.connection, cache[].mode, context)):
            var fields = py_call(py_type(Python.dict()).__dict__["copy"], state)
            var decoded = cache[].state.copy()
            _ = held  # the downcast owner remains alive through its final native read
            return PCAQueryAdmission(fields, decoded^)
    var key = _pca_query_key(state, context)
    if key.identity.valid and held is not None:
        var cache = held.downcast_value_ptr[PCAQueryCache]()
        if cache[].key.identity.same(key.identity) and pca_connection_matches(cache[].state.connection, cache[].mode, context):
            if generation != 0 and cache[].generation != generation:
                cache[].generation = generation
                cache[].context_key = _pca_context_key(state, context)
            var fields = py_call(py_type(Python.dict()).__dict__["copy"], state)
            var decoded = cache[].state.copy()
            _ = held
            return PCAQueryAdmission(fields, decoded^)
    var fields = _pca_query_fields_from(state, context, operation)
    if py_get(context, "ndarray") is None:
        raise Error("ImportError: mojolearn: PCA requires optional NumPy support. Install it with: "
                    + 'python -m pip install "mojolearn[numpy]" ("mojolearn[verify]" also includes NumPy).')
    var mode = pca_mode_name(fields["numeric_mode"])
    var connection = pca_connection(mode, context)
    var decoded = _pca_query_state(fields, connection)
    if key.identity.valid:
        # Only the lifecycle-flagged models were cached before. The attribute
        # fallback reads additional names outside the identity inventory.
        var flagged = Bool(py=py_get(fields, "_mojolearn_fitted", False)) or Bool(py=py_get(fields, "_mojolearn_fitting", False))
        if flagged:
            var owner = PCAQueryCache(decoded.copy(), key^, _pca_context_key(state, context), generation, mode^)
            prepared_put(mods, _PCA_QUERY_SLOT, model, PythonObject(alloc=owner^))
    return PCAQueryAdmission(fields, decoded^)
