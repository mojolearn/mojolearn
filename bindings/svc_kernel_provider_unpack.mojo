# SPDX-License-Identifier: Apache-2.0
# UNPACK: the kernel SVC's provider connection. One admission per public
# operation of the binding's entry tables, held (with the Python owners that
# pin them) until the operation ends; RUN receives the plain `SvcDrivers`.
#
# Default route: the shared SVC table of the binding for the short-form
# solve, the host-words query, the epilogues and the row glue, and, on a GPU
# binding, the device table (svm/svc_native_device.mojo) for the device-X
# solve and the device-resident query. No entry is called through CPython.
# TOMBSTONE: MOJOLEARN_CLASSICAL_NATIVE_DISPATCH (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_CLASSICAL_NATIVE_DISPATCH.patch; record in docs/TOMBSTONES.md.
from std.python import PythonObject
from core.svc_native_api import SvcDrivers, SVC_DEVICE_ABI, SVC_DEVICE_COUNT, SVC_TABLE_COUNT
from providers.python.neural_descriptor_pyapi_door import NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once
from svm.operation_metadata import SVC_ABI_NAME


@fieldwise_init
struct SvcConnection(Movable):
    """`table` and `device` are the provider owners (`device` None: a host
    binding); `drivers` their entries."""
    var table: PythonObject
    var device: PythonObject
    var drivers: SvcDrivers


def svc_connection(binding: PythonObject) raises -> SvcConnection:
    var factory = neural_descriptor_entry(binding, "svm_binary_api_v1", False)
    if factory is None:
        raise Error("RuntimeError: mojolearn: this SVM binding has no svm_binary_api_v1 table; rebuild it")
    var table = neural_descriptor_connect_factory_once(factory, String(SVC_ABI_NAME), SVC_TABLE_COUNT)
    var entry = table.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry
    var device_factory = neural_descriptor_entry(binding, "svc_device_api_v1", False)
    if device_factory is None:
        return SvcConnection(table, PythonObject(None), SvcDrivers(entry, entry, False))
    var device = neural_descriptor_connect_factory_once(device_factory, String(SVC_DEVICE_ABI), SVC_DEVICE_COUNT)
    var device_entry = device.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry
    return SvcConnection(table, device, SvcDrivers(entry, device_entry, True))
