# SPDX-License-Identifier: Apache-2.0
"""IPCA import-connected native descriptors and compiled-image lifetime pins (door).

The typed connection records are in `x_decomp/ipca_descriptor_boundary_run.mojo`; this door admits
the provider, keeps its pin and releases the GIL around each typed driver call."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.shared_slot import SharedSlot
from core.pca_native_api import pca_vendor_name
from core.ipca_native_api import (
    IPCA_NATIVE_ABI, IPCA_NATIVE_COUNT,
    IPCANativeDriver, IPCAFitRequest, IPCAFitResult, IPCAPartialRequest, IPCAQueryRequest,
    IPCA_CAP_FIT, IPCA_CAP_PARTIAL, IPCA_CAP_QUERY,
)
from x_decomp.ipca_descriptor_boundary_run import (
    ipca_connection_index, ipca_connection_add, ipca_mode, ipca_vendor, ipca_capabilities, ipca_driver,
)
from bindings.py_provider_pins import ProviderPins, provider_word
from bindings.py_balanced import py_call, py_type
from bindings.decomp_owner_common import decomp_attr
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once


comptime IPCA_PROVIDER_PINS = SharedSlot[ProviderPins, "MojolearnIPCAProviderPinsV1"]


def ipca_connection(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> Int:
    # Keep the former decomp_entries mode/availability refusal at its stage.
    var mode = decomp_attr(model, fields, "numeric_mode_")
    if py_type(mode) is not py_type(PythonObject("")):
        raise Error("numeric_mode must be fast, identical or None")
    var name = String(mode)
    if name != "identical" and name != "fast":
        name = String(py_call(py_call(mode.strip).lower))
    if name != "identical" and name != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var table = context["ipca_bindings"]
    if py_type(table) is not py_type(Python.dict()):
        raise Error("TypeError: IPCA binding tables require native dictionaries")
    if not dict_keys_admitted(table):
        var keys = py_call(py_type(Python.list()), table)
        for index in range(len(keys)):  # driver-loop: fixed connected mode names before keyed lookup
            if py_type(keys[index]) is not py_type(PythonObject("")):
                raise Error("TypeError: IPCA binding tables require native string keys")
        dict_keys_admit(table)
    var factory = table[PythonObject(name)]
    if py_type(factory) is py_type(PythonObject("")):
        raise Error(String(factory))
    if factory is None:
        raise Error("NotImplementedError: mojolearn IncrementalPCA: this build of "
                    + "_mojolearn_x_decomp does not export the x_decomp_ipca_* drivers"
                    + "; rebuild it with bindings/build_x_decomp.sh")
    var provider = neural_descriptor_connect_factory_once(factory, String(IPCA_NATIVE_ABI), IPCA_NATIVE_COUNT)
    var index = ipca_connection_index(provider_word(provider))
    if index < 0:
        # The typed record and the door pin are appended together, under the GIL.
        index = ipca_connection_add(provider_word(provider),
            IPCANativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry))
        IPCA_PROVIDER_PINS.get_or_create_ptr()[].providers.append(provider)
    if pca_vendor_name(ipca_vendor(index)) != String(context["vendor"]):
        raise Error("RuntimeError: IPCA native vendor differs from selected execution")
    if ipca_mode(index) != (1 if name == "identical" else 0):
        raise Error("RuntimeError: IPCA native mode differs from selected execution")
    if (ipca_capabilities(index) & (IPCA_CAP_FIT | IPCA_CAP_PARTIAL | IPCA_CAP_QUERY)) != (IPCA_CAP_FIT | IPCA_CAP_PARTIAL | IPCA_CAP_QUERY):
        raise Error("NotImplementedError: mojolearn IncrementalPCA: this build of "
                    + "_mojolearn_x_decomp does not export the x_decomp_ipca_* drivers"
                    + "; rebuild it with bindings/build_x_decomp.sh")
    return index


def ipca_run_fit(connection: Int, request: IPCAFitRequest) raises -> IPCAFitResult:
    var pin = ipca_provider(connection)
    var driver = ipca_driver(connection)
    try:
        with GILReleased(Python()):
            return driver.fit(request)
    finally:
        _ = pin



def ipca_run_partial(connection: Int, request: IPCAPartialRequest) raises -> IPCAFitResult:
    var pin = ipca_provider(connection)
    var driver = ipca_driver(connection)
    try:
        with GILReleased(Python()):
            return driver.partial(request)
    finally:
        _ = pin


def ipca_run_query(connection: Int, request: IPCAQueryRequest) raises:
    var pin = ipca_provider(connection)
    var driver = ipca_driver(connection)
    try:
        with GILReleased(Python()):
            driver.query(request)
    finally:
        _ = pin


def ipca_provider(connection: Int) raises -> PythonObject:
    return IPCA_PROVIDER_PINS.get_or_create_ptr()[].providers[connection]


def ipca_run_free(connection: Int, handle: Int) raises:
    var pin = ipca_provider(connection)
    var driver = ipca_driver(connection)
    try:
        with GILReleased(Python()):
            driver.free(handle)
    finally:
        _ = pin
