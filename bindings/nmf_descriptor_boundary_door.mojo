# SPDX-License-Identifier: Apache-2.0
"""NMF import-connected native descriptors and compiled-image lifetime pins."""
from core.mode_answer import MODE_FAMILY_NMF, answer_mode_word
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
from bindings.py_balanced import py_call, py_get, py_type
from bindings.decomp_owner_common_unpack import decomp_attr
from bindings.native_class_admission_door import dict_keys_admitted, dict_keys_admit
from providers.python.neural_descriptor_pyapi_door import NeuralDescriptorProvider, neural_descriptor_connect_factory_once
from bindings.nmf_keys_unpack import nmf_keys
from bindings.decomp_device_route_door import (
    decomp_route_hit, decomp_route_keep, decomp_route_resolve_device, decomp_route_on_cpu_image,
    decomp_route_expected_vendor,
)
from core.device_route_dw_decomp import DecompRoute, decomp_route_cpu_identical
from bindings.py_balanced import py_item


comptime NMF_PROVIDER_PINS = SharedSlot[ProviderPins, "MojolearnNMFProviderPinsV1"]


def _nmf_capability(model: PythonObject, index: Int, capability: Int, legacy_name: String) raises:
    if (nmf_capabilities(index) & capability) == 0:
        raise Error("NotImplementedError: mojolearn " + String(py_type(model).__name__) + ": this build of "
                    + "_mojolearn_x_decomp does not export " + legacy_name
                    + "; rebuild it with bindings/build_x_decomp.sh")


def nmf_connection(model: PythonObject, fields: PythonObject, context: PythonObject,
                   capability: Int, legacy_name: String) raises -> Int:
    var nk = nmf_keys()
    # The estimator's resolved route (bindings/decomp_device_route_door.mojo):
    # the connection of the device image or of the CPU image, resolved once
    # per estimator. A hit is one `device` read, one record read and three
    # pointer compares; the selection below runs at the estimator's first
    # operation and after its mode or device object changed.
    var mode = decomp_attr(model, fields, "numeric_mode_")
    var hit = decomp_route_hit(model, fields, mode, context)
    if hit.found:
        _nmf_capability(model, hit.route.connection, capability, legacy_name)
        return hit.route.connection
    # Keep the former decomp_entries mode/availability refusal at its stage.
    if py_type(mode) is not py_type(PythonObject("")):
        raise Error("numeric_mode must be fast, identical or None")
    var name = String(py_call(py_call(mode.strip).lower))
    if name != "identical" and name != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    answer_mode_word(name, MODE_FAMILY_NMF)
    # The one read of `device`: None is this installation's device; a `cpu`
    # route of a GPU installation reads the CPU image's table
    # (python/mojolearn/_native_decomp_device_bridge.py), identical only.
    var device = decomp_route_resolve_device(hit.device, context)
    var cpu_image = decomp_route_on_cpu_image(device, context)
    var table: PythonObject
    if cpu_image:
        decomp_route_cpu_identical(name == "identical")
        table = py_get(context, nk[].k_nmf_cpu_bindings)
        if table is None:
            raise Error("ImportError: device='cpu' is not connected in this installation")
    else:
        table = py_item(context, nk[].k_nmf_bindings)
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
            NMFNativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry))
        NMF_PROVIDER_PINS.get_or_create_ptr()[].providers.append(provider)
    if pca_vendor_name(nmf_vendor(index)) != decomp_route_expected_vendor(device, context):
        raise Error("RuntimeError: NMF native vendor differs from selected execution")
    if nmf_mode(index) != (1 if name == "identical" else 0):
        raise Error("RuntimeError: NMF native mode differs from selected execution")
    _nmf_capability(model, index, capability, legacy_name)
    decomp_route_keep(model, DecompRoute(device, index), mode, hit.device, context, factory)
    return index


def nmf_route_vendor(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> String:
    """`vendor_used`: the vendor word the estimator's compiled image reports
    (`cpu` for a `device="cpu"` estimator), read from its resolved route."""
    var index = nmf_connection(model, fields, context, 0, "x_decomp_nmf_fit")
    return pca_vendor_name(nmf_vendor(index))


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
