# SPDX-License-Identifier: Apache-2.0
"""Native TSVD provider selection and compiled image pins at the boundary."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.shared_slot import SharedSlot
from core.pca_native_api import pca_vendor_name
from core.tsvd_native_api import (
    TSVD_NATIVE_ABI, TSVD_NATIVE_COUNT, TSVDNativeDriver, TSVDFitRequest, TSVDFitResult, TSVDQueryRequest,
)
from x_decomp.tsvd_descriptor_boundary_run import (
    tsvd_connection_index, tsvd_connection_add, tsvd_mode, tsvd_vendor, tsvd_driver,
)
from bindings.py_provider_pins import ProviderPins, provider_word
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_class_admission_door import dict_keys_admitted, dict_keys_admit
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once


comptime TSVD_PROVIDER_PINS = SharedSlot[ProviderPins, "MojolearnTSVDProviderPinsV1"]


def tsvd_factory(context: PythonObject, table_name: String, mode: String) raises -> PythonObject:
    var table = context[PythonObject(table_name)]
    if py_type(table) is not py_type(Python.dict()):
        raise Error("TypeError: TruncatedSVD binding tables require native dictionaries")
    if not dict_keys_admitted(table):
        var keys = py_call(py_type(Python.list()), table)
        for index in range(len(keys)):  # driver-loop: fixed connected mode names before keyed lookup
            if py_type(keys[index]) is not py_type(PythonObject("")):
                raise Error("TypeError: TruncatedSVD binding tables require native string keys")
        dict_keys_admit(table)
    return table[PythonObject(mode)]


def tsvd_connect(factory: PythonObject, mode: Int, vendor: String) raises -> Int:
    if py_type(factory) is py_type(PythonObject("")):
        raise Error(String(factory))
    var provider = neural_descriptor_connect_factory_once(factory, String(TSVD_NATIVE_ABI), TSVD_NATIVE_COUNT)
    var index = tsvd_connection_index(provider_word(provider))
    if index < 0:
        # The typed record and the door pin are appended together, under the GIL.
        index = tsvd_connection_add(provider_word(provider),
            TSVDNativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry))
        TSVD_PROVIDER_PINS.get_or_create_ptr()[].providers.append(provider)
    if pca_vendor_name(tsvd_vendor(index)) != vendor:
        raise Error("RuntimeError: TruncatedSVD native vendor differs from selected execution")
    if tsvd_mode(index) != mode:
        raise Error("RuntimeError: TruncatedSVD native mode differs from selected execution")
    return index


def tsvd_connection(mode: String, context: PythonObject) raises -> Int:
    return tsvd_connect(tsvd_factory(context, "tsvd_bindings", mode),
        1 if mode == "identical" else 0, String(context["vendor"]))


def tsvd_provider(connection: Int) raises -> PythonObject:
    return TSVD_PROVIDER_PINS.get_or_create_ptr()[].providers[connection]


def tsvd_run_fit(connection: Int, request: TSVDFitRequest) raises -> TSVDFitResult:
    var pin = tsvd_provider(connection)
    var driver = tsvd_driver(connection)
    try:
        with GILReleased(Python()):
            return driver.fit(request)
    finally:
        _ = pin


def tsvd_run_query(connection: Int, request: TSVDQueryRequest) raises -> Int:
    var pin = tsvd_provider(connection)
    var driver = tsvd_driver(connection)
    try:
        with GILReleased(Python()):
            return driver.query(request)
    finally:
        _ = pin
