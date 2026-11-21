# SPDX-License-Identifier: Apache-2.0
# UNPACK: the kernel SVC's provider connection. One admission per public
# operation of the binding's entry tables, held (with the Python owners that
# pin them) until the operation ends; RUN receives the plain `SvcDrivers`.
#
# Default route: the shared SVC table of the binding for the short-form
# solve, the host-words query, the epilogues and the row glue, and, on a GPU
# binding, the device table (svm/svc_native_device.mojo) for the device-X
# solve and the device-resident query. No entry is called through CPython.
# B arm -D MOJOLEARN_CLASSICAL_NATIVE_DISPATCH (undecided, RUN OWED;
# providers/python/classical_native_dispatch.mojo): the shared table only (no
# device-X solve, no resident query), the binding's mode checked here, and
# the diagnostic `_native_svc_dispatch` published after the operation.
from std.python import PythonObject
from bindings.py_balanced import py_contains, py_setitem, py_item
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.svm_keys_unpack import svm_keys
from core.svc_native_api import SvcDrivers, SVC_DEVICE_ABI, SVC_DEVICE_COUNT, SVC_TABLE_COUNT
from providers.python.classical_native_dispatch import CLASSICAL_NATIVE_DISPATCH
from providers.python.neural_descriptor_pyapi_door import NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once, neural_descriptor_info
from svm.operation_metadata import SVC_ABI_NAME


@fieldwise_init
struct SvcConnection(Movable):
    """`table` and `device` are the provider owners (`device` None: a host
    binding, or the B arm); `drivers` their entries; `candidate`: the B arm."""
    var table: PythonObject
    var device: PythonObject
    var drivers: SvcDrivers
    var candidate: Bool


def svc_connection(binding: PythonObject) raises -> SvcConnection:
    var factory = neural_descriptor_entry(binding, "svm_binary_api_v1", False)
    if factory is None:
        raise Error("SVC native dispatch candidate is unavailable for this binding" if CLASSICAL_NATIVE_DISPATCH
                    else "RuntimeError: mojolearn: this SVM binding has no svm_binary_api_v1 table; rebuild it")
    var table = neural_descriptor_connect_factory_once(factory, String(SVC_ABI_NAME), SVC_TABLE_COUNT)
    var entry = table.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry
    if CLASSICAL_NATIVE_DISPATCH:
        var shared = SvcDrivers(entry, entry, False)
        if shared.mode() != 1:
            raise Error("SVC native dispatch candidate requires identical mode")
        return SvcConnection(table, PythonObject(None), shared, True)
    var device_factory = neural_descriptor_entry(binding, "svc_device_api_v1", False)
    if device_factory is None:
        return SvcConnection(table, PythonObject(None), SvcDrivers(entry, entry, False), False)
    var device = neural_descriptor_connect_factory_once(device_factory, String(SVC_DEVICE_ABI), SVC_DEVICE_COUNT)
    var device_entry = device.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry
    return SvcConnection(table, device, SvcDrivers(entry, device_entry, True), False)


def svc_connection_publish(fields: PythonObject, connection: SvcConnection, context: PythonObject) raises:
    """The B arm's dispatch diagnostic. Plain values only: never persist
    module / capsule owners or make the table part of a model's pickle."""
    if connection.candidate:
        var nk = svm_keys()
        if py_contains(fields, nk[].k__native_svc_dispatch):
            native_release_admit(py_item(fields, nk[].k__native_svc_dispatch), context)
        py_setitem(fields, nk[].k__native_svc_dispatch, neural_descriptor_info(connection.table))
