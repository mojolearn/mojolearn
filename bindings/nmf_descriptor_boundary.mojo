# SPDX-License-Identifier: Apache-2.0
"""NMF import-connected native descriptors and compiled-image lifetime pins."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.shared_slot import SharedSlot
from core.pca_native_api import pca_vendor_name
from core.nmf_native_api import (
    NMF_NATIVE_ABI, NMF_NATIVE_COUNT, NMFNativeDriver, NMFFitRequest, NMFFitResult, NMFInverseRequest,
)
from x_decomp.nmf_descriptor_boundary_run import (
    nmf_connection_index, nmf_connection_add, nmf_mode, nmf_vendor, nmf_capabilities, nmf_driver,
)
from bindings.py_provider_pins import ProviderPins, provider_word
from bindings.py_balanced import py_call, py_type
from bindings.decomp_owner_common import decomp_attr
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once


comptime NMF_PROVIDER_PINS = SharedSlot[ProviderPins, "MojolearnNMFProviderPinsV1"]


def nmf_connection(model: PythonObject, fields: PythonObject, context: PythonObject,
                   capability: Int, legacy_name: String) raises -> Int:
    # Keep the former decomp_entries mode/availability refusal at its stage.
    var mode = decomp_attr(model, fields, "numeric_mode_")
    if py_type(mode) is not py_type(PythonObject("")):
        raise Error("numeric_mode must be fast, identical or None")
    var name = String(py_call(py_call(mode.strip).lower))
    if name != "identical" and name != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var table = context["nmf_bindings"]
    if py_type(table) is not py_type(Python.dict()):
        raise Error("TypeError: NMF binding tables require native dictionaries")
    if not dict_keys_admitted(table):
        var keys = py_call(py_type(Python.list()), table)
        for index in range(len(keys)):  # driver-loop: fixed connected mode names before keyed lookup
            if py_type(keys[index]) is not py_type(PythonObject("")):
                raise Error("TypeError: NMF binding tables require native string keys")
        dict_keys_admit(table)
    var factory = table[PythonObject(name)]
    if py_type(factory) is py_type(PythonObject("")):
        raise Error(String(factory))
    if factory is None:
        raise Error("NotImplementedError: mojolearn " + String(py_type(model).__name__) + ": this build of "
                    + "_mojolearn_x_decomp does not export " + legacy_name
                    + "; rebuild it with bindings/build_x_decomp.sh")
    var provider = neural_descriptor_connect_factory_once(factory, String(NMF_NATIVE_ABI), NMF_NATIVE_COUNT)
    var index = nmf_connection_index(provider_word(provider))
    if index < 0:
        # The typed record and the door pin are appended together, under the GIL.
        index = nmf_connection_add(provider_word(provider),
            NMFNativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry))
        NMF_PROVIDER_PINS.get_or_create_ptr()[].providers.append(provider)
    if pca_vendor_name(nmf_vendor(index)) != String(context["vendor"]):
        raise Error("RuntimeError: NMF native vendor differs from selected execution")
    if nmf_mode(index) != (1 if name == "identical" else 0):
        raise Error("RuntimeError: NMF native mode differs from selected execution")
    if (nmf_capabilities(index) & capability) == 0:
        raise Error("NotImplementedError: mojolearn " + String(py_type(model).__name__) + ": this build of "
                    + "_mojolearn_x_decomp does not export " + legacy_name
                    + "; rebuild it with bindings/build_x_decomp.sh")
    return index


def nmf_run_fit(connection: Int, request: NMFFitRequest) raises -> NMFFitResult:
    var pin = NMF_PROVIDER_PINS.get_or_create_ptr()[].providers[connection]
    var driver = nmf_driver(connection)
    try:
        with GILReleased(Python()):
            return driver.fit(request)
    finally:
        _ = pin


def nmf_run_inverse(connection: Int, request: NMFInverseRequest) raises:
    var pin = NMF_PROVIDER_PINS.get_or_create_ptr()[].providers[connection]
    var driver = nmf_driver(connection)
    try:
        with GILReleased(Python()):
            driver.inverse(request)
    finally:
        _ = pin
