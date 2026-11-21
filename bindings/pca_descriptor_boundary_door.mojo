# SPDX-License-Identifier: Apache-2.0
"""PCA provider admission and image pins surrounding typed native execution (door).

The typed connection records and their capability reads are in
`x_decomp/pca_descriptor_boundary_run.mojo`; the provider pins stay here."""
from core.mode_answer import MODE_FAMILY_PCA, answer_mode_word
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.shared_slot import SharedSlot
from core.pca_native_api import (
    PCA_NATIVE_ABI, PCA_NATIVE_COUNT, PCANativeDriver, PCAFitRequest, PCAFitResult, PCAQueryRequest, PCAQueryResult,
)
from x_decomp.pca_descriptor_boundary_run import (
    pca_connection_index, pca_connection_add, pca_mode, pca_vendor, pca_driver,
)
from bindings.py_provider_pins import ProviderPins, provider_word
from bindings.py_balanced import py_call, py_get, py_type
from bindings.native_class_admission_door import dict_keys_admitted, dict_keys_admit
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_connect_factory_once,
)
from bindings.pca_keys_unpack import pca_keys
from bindings.py_balanced import py_item
from core.device_route_dw_decomp import DecompRoute, decomp_route_cpu_identical
from bindings.decomp_device_route_door import (
    decomp_route_hit, decomp_route_keep, decomp_route_resolve_device, decomp_route_on_cpu_image,
    decomp_route_expected_vendor,
)


comptime PCA_PROVIDER_PINS = SharedSlot[ProviderPins, "MojolearnPCAProviderPinsV1"]


def pca_mode_name(value: PythonObject) raises -> String:
    if py_type(value) is not py_type(PythonObject("")):
        raise Error("TypeError: numeric_mode must be identical, fast or None")
    var mode = String(py_call(py_call(value.strip).lower))
    if mode != "identical" and mode != "fast":
        raise Error("numeric_mode must be identical, fast or None")
    answer_mode_word(mode, MODE_FAMILY_PCA)
    return mode


def _pca_connect(factory: PythonObject, vendor: String, mode: Int) raises -> Int:
    var provider = neural_descriptor_connect_factory_once(factory, String(PCA_NATIVE_ABI), PCA_NATIVE_COUNT)
    var index = pca_connection_index(provider_word(provider))
    if index < 0:
        # The typed record and the door pin are appended together, under the GIL.
        index = pca_connection_add(provider_word(provider),
            PCANativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry))
        PCA_PROVIDER_PINS.get_or_create_ptr()[].providers.append(provider)
    if pca_vendor(index) != vendor:
        raise Error("RuntimeError: PCA native vendor differs from selected execution")
    if pca_mode(index) != mode:
        raise Error("RuntimeError: PCA native mode differs from selected execution")
    return index


def _pca_table_admitted(bindings: PythonObject) raises -> Bool:
    """True for a mode-keyed binding table whose keys are native strings
    (admitted once per table object)."""
    if py_type(bindings) is not py_type(Python.dict()):
        return False
    if not dict_keys_admitted(bindings):
        var keys = py_call(py_type(Python.list()), bindings)
        for index in range(len(keys)):  # driver-loop: fixed connected mode keys, before lookup
            if py_type(keys[index]) is not py_type(PythonObject("")):
                return False
        dict_keys_admit(bindings)
    return True


def _pca_table(bindings: PythonObject, mode: String) raises -> PythonObject:
    if py_type(bindings) is not py_type(Python.dict()):
        raise Error("TypeError: PCA binding tables require native dictionaries")
    if not _pca_table_admitted(bindings):
        raise Error("TypeError: PCA binding tables require native string keys")
    var selected = bindings[PythonObject(mode)]
    if py_type(selected) is py_type(PythonObject("")):
        raise Error("ImportError: " + String(selected))
    return selected


def pca_route(model: PythonObject, fields: PythonObject, mode: PythonObject, context: PythonObject) raises -> DecompRoute:
    """The estimator's resolved route (bindings/decomp_device_route_door.mojo):
    the connection of the device image or of the CPU image. `mode` is the
    estimator's numeric mode object (the process default object when None)
    and `fields` holds its `device`. A hit is one `device` read, one record
    read and three pointer compares; a miss (the estimator's first operation,
    a changed mode or device object) reads the binding table, connects and
    stores the route. Replaces the per-call `pca_connection(mode, context)`."""
    var nk = pca_keys()
    var hit = decomp_route_hit(model, fields, mode, context)
    if hit.found:
        return hit.route
    var name = pca_mode_name(mode)
    # The one read of `device`: None is this installation's device; a `cpu`
    # route of a GPU installation reads the CPU image's table
    # (python/mojolearn/_native_decomp_device_bridge.py), identical only.
    var device = decomp_route_resolve_device(hit.device, context)
    var bindings = PythonObject(None)
    if decomp_route_on_cpu_image(device, context):
        decomp_route_cpu_identical(name == "identical")
        bindings = py_get(context, nk[].k_pca_cpu_bindings)
        if bindings is None:
            raise Error("ImportError: device='cpu' is not connected in this installation")
    else:
        bindings = py_item(context, nk[].k_pca_bindings)
    var factory = _pca_table(bindings, name)
    if factory is None:
        raise Error("NotImplementedError: mojolearn PCA: this build of _mojolearn_estimators does not export "
                    + "pca_native_api_v1; rebuild it with bindings/build_estimators.sh")
    var tier = 1 if name == "identical" else 0
    var route = DecompRoute(device, _pca_connect(factory, decomp_route_expected_vendor(device, context), tier), tier)
    decomp_route_keep(model, route, mode, hit.device, context, factory)
    return route


def pca_connection_matches(connection: Int, mode: String, context: PythonObject) raises -> Bool:
    var nk = pca_keys()
    # Cache validation compares identity only. A changed factory must take
    # full field admission before its provider refusal, at the original stage.
    # In particular, do not invoke a newly rebound factory on the cache path.
    # The connection is the one its route resolved, of the device table or of
    # the CPU table (`device="cpu"`); a changed `device` changes the cache key
    # (it is one of the connected parameters), so either table's entry is
    # this connection's own.
    var provider = PCA_PROVIDER_PINS.get_or_create_ptr()[].providers[connection]
    var factory = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].factory
    var bindings = py_item(context, nk[].k_pca_bindings)
    if _pca_table_admitted(bindings) and py_get(bindings, PythonObject(mode)) is factory:
        return True
    var cpu = py_get(context, nk[].k_pca_cpu_bindings)
    return cpu is not None and _pca_table_admitted(cpu) and py_get(cpu, PythonObject(mode)) is factory


def pca_fold_connection(context: PythonObject) raises -> Int:
    var nk = pca_keys()
    var factory = py_get(context, nk[].k_pca_fold_factory)
    if factory is None:
        return -1
    return _pca_connect(factory, String(py_item(context, nk[].k_vendor)), 1)


def pca_run_fit(connection: Int, request: PCAFitRequest) raises -> PCAFitResult:
    var pin = PCA_PROVIDER_PINS.get_or_create_ptr()[].providers[connection]
    var driver = pca_driver(connection)
    try:
        with GILReleased(Python()):
            return driver.fit(request)
    finally:
        _ = pin


def pca_run_query(connection: Int, request: PCAQueryRequest) raises -> PCAQueryResult:
    var pin = PCA_PROVIDER_PINS.get_or_create_ptr()[].providers[connection]
    var driver = pca_driver(connection)
    try:
        with GILReleased(Python()):
            return driver.query(request)
    finally:
        _ = pin
