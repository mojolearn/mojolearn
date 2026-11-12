# SPDX-License-Identifier: Apache-2.0
"""PCA provider admission and image pins surrounding typed native execution (door).

The typed connection records and their capability reads are in
`x_decomp/pca_descriptor_boundary_run.mojo`; the provider pins stay here."""
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
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_connect_factory_once,
)


comptime PCA_PROVIDER_PINS = SharedSlot[ProviderPins, "MojolearnPCAProviderPinsV1"]


def pca_mode_name(value: PythonObject) raises -> String:
    if py_type(value) is not py_type(PythonObject("")):
        raise Error("TypeError: numeric_mode must be identical, fast or None")
    var mode = String(py_call(py_call(value.strip).lower))
    if mode != "identical" and mode != "fast":
        raise Error("numeric_mode must be identical, fast or None")
    return mode


def _pca_connect(factory: PythonObject, context: PythonObject, mode: Int) raises -> Int:
    var provider = neural_descriptor_connect_factory_once(factory, String(PCA_NATIVE_ABI), PCA_NATIVE_COUNT)
    var index = pca_connection_index(provider_word(provider))
    if index < 0:
        # The typed record and the door pin are appended together, under the GIL.
        index = pca_connection_add(provider_word(provider),
            PCANativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry))
        PCA_PROVIDER_PINS.get_or_create_ptr()[].providers.append(provider)
    if pca_vendor(index) != String(context["vendor"]):
        raise Error("RuntimeError: PCA native vendor differs from selected execution")
    if pca_mode(index) != mode:
        raise Error("RuntimeError: PCA native mode differs from selected execution")
    return index


def pca_connection(mode: String, context: PythonObject) raises -> Int:
    var bindings = context["pca_bindings"]
    if py_type(bindings) is not py_type(Python.dict()):
        raise Error("TypeError: PCA binding tables require native dictionaries")
    if not dict_keys_admitted(bindings):
        var keys = py_call(py_type(Python.list()), bindings)
        for index in range(len(keys)):  # driver-loop: fixed connected mode keys, before lookup
            if py_type(keys[index]) is not py_type(PythonObject("")):
                raise Error("TypeError: PCA binding tables require native string keys")
        dict_keys_admit(bindings)
    var selected = bindings[PythonObject(mode)]
    if py_type(selected) is py_type(PythonObject("")):
        raise Error("ImportError: " + String(selected))
    return _pca_connect(selected, context, 1 if mode == "identical" else 0)


def pca_connection_matches(connection: Int, mode: String, context: PythonObject) raises -> Bool:
    # Cache validation compares identity only. A changed factory must take
    # full field admission before its provider refusal, at the original stage.
    # In particular, do not invoke a newly rebound factory on the cache path.
    var bindings = context["pca_bindings"]
    if py_type(bindings) is not py_type(Python.dict()):
        return False
    if not dict_keys_admitted(bindings):
        var keys = py_call(py_type(Python.list()), bindings)
        for index in range(len(keys)):  # driver-loop: connected mode keys before identity lookup
            if py_type(keys[index]) is not py_type(PythonObject("")):
                return False
        dict_keys_admit(bindings)
    var provider = PCA_PROVIDER_PINS.get_or_create_ptr()[].providers[connection]
    var factory = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].factory
    return py_get(bindings, PythonObject(mode)) is factory


def pca_fold_connection(context: PythonObject) raises -> Int:
    var factory = py_get(context, "pca_fold_factory")
    if factory is None:
        return -1
    return _pca_connect(factory, context, 1)


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
