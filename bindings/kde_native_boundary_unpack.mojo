# SPDX-License-Identifier: Apache-2.0
"""KDE admission/packing and lifetime pins around a Python-free native driver."""
from std.python import Python, PythonObject
from bindings.native_owner_registry_pack import borrowed_native_owner
from std.python._cpython import GILReleased
from bindings.native_loaded_module_door import native_interpreter_finalizing
from bindings.classical_arrays_door import classical_wrap_layout
from bindings.native_array_storage_door import native_array_zeros
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_once
from core.native_array_layout import NativeArrayLayout, native_array_code
from core.kde_native_api import KDE_NATIVE_ABI, KDE_NATIVE_COUNT, KDE_CAPABILITIES, KDENativeDriver, KDEPrepareRequest, KDEQueryRequest, KDEResidentQuery
from core.kde_native_state import KDEFitKey, KDEResidentState


struct KDEConnection(Movable):
    var provider: PythonObject
    var driver: KDENativeDriver
    var capabilities: Int

    def __init__(out self, binding: PythonObject) raises:
        self.provider = neural_descriptor_connect_once(binding, "kde_native_api_v1", String(KDE_NATIVE_ABI), KDE_NATIVE_COUNT)
        self.driver = KDENativeDriver(self.provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        self.capabilities = self.driver.metadata(KDE_CAPABILITIES)


struct KDEResidentOwner(Movable, Writable):
    var state: KDEResidentState
    # Boundary-only image/source pins. RUN reads only native state/requests.
    var provider: PythonObject
    var train: PythonObject
    var weights: PythonObject

    def __init__(out self, var state: KDEResidentState, provider: PythonObject,
                 train: PythonObject, weights: PythonObject):
        self.state = state^
        self.provider = provider
        self.train = train
        self.weights = weights

    def __del__(deinit self):
        # Queries retain this owner until their native call finishes. A public
        # replacement cannot free the handle out from under an active query.
        var handle = self.state.take_handle()
        if handle >= 0:
            try:
                if not native_interpreter_finalizing():
                    with GILReleased(Python()):
                        self.state.driver.release(handle)
            except:
                pass
        _ = self.provider
        _ = self.train
        _ = self.weights

    def write_to(self, mut writer: Some[Writer]):
        writer.write("KDEResidentOwner(", self.state.key.rows, ", ", self.state.key.features, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def kde_owner_admit(value: PythonObject) raises:
    if value is not None:
        try:
            _ = value.downcast_value_ptr[KDEResidentOwner]()
        except:
            raise Error("TypeError: KDE resident state requires its native owner")


def kde_run_query(connection: KDEConnection, request: KDEQueryRequest) raises -> Float64:
    try:
        with GILReleased(Python()):
            return connection.driver.query(request)
    finally:
        _ = connection


def kde_run_resident(owner: PythonObject, request: KDEResidentQuery) raises -> Float64:
    var driver = owner.downcast_value_ptr[KDEResidentOwner]()[].state.driver.copy()
    try:
        with GILReleased(Python()):
            return driver.resident(request)
    finally:
        _ = owner


def kde_resident_owner_pins(value: Int, index: Int) abi("C") -> Int:
    # Native registry inspection only; the caller retains the exact owner.
    if value == 0 or index < -1 or index >= 3:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[KDEResidentOwner]()
        var result = 3
        if index == 0:
            result = Int(native[].provider._obj_ptr)
        elif index == 1:
            result = Int(native[].train._obj_ptr)
        elif index == 2:
            result = Int(native[].weights._obj_ptr)
        _ = owner
        return result
    except:
        return -1
