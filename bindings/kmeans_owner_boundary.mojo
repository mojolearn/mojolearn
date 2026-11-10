# SPDX-License-Identifier: Apache-2.0
"""A native resident-center owner with boundary-only storage/image pins."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, ExternalFunction, PyObjectPtr
from core.shared_slot import SharedSlot
from core.kmeans_owner_state import KMeansResidentState
from bindings.kmeans_descriptor_boundary import KMEANS_CONNECTIONS
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_fields, py_contains, py_setitem, py_type
from bindings.native_loaded_module import native_loaded_module
from bindings.optimizer_public_words import stash_python_error

comptime _KMeansDictDel = ExternalFunction[
    "PyDict_DelItem", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> Int32]
comptime _KMeansDictUpdate = ExternalFunction[
    "PyDict_Update", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> Int32]


struct KMeansStateApi(Defaultable, Movable):
    var delete: _KMeansDictDel.type
    var update: _KMeansDictUpdate.type

    def __init__(out self):
        ref cpy = Python().cpython()
        self.delete = _KMeansDictDel.load(cpy.lib.borrow())
        self.update = _KMeansDictUpdate.load(cpy.lib.borrow())


comptime KMEANS_STATE_API = SharedSlot[KMeansStateApi, "MojolearnKMeansStateApiV1"]


struct KMeansResidentOwner(Movable, Writable):
    var state: KMeansResidentState
    # Only lifetime pins: RUN receives the state's typed driver and handle.
    var provider: PythonObject
    var centers: PythonObject

    def __init__(out self, var state: KMeansResidentState,
                 provider: PythonObject, centers: PythonObject):
        self.state = state^
        self.provider = provider
        self.centers = centers

    def __del__(deinit self):
        # Each query retains this owner across its native call. Replacing the
        # model's owner therefore releases only after in-flight queries finish.
        var handle = self.state.take_handle()
        if handle >= 0:
            try:
                with GILReleased(Python()):
                    self.state.driver.release(handle)
            except:
                pass  # Preserve quiet handle teardown/interpreter shutdown.
        _ = self.provider
        _ = self.centers

    def write_to(self, mut writer: Some[Writer]):
        writer.write("KMeansResidentOwner(", self.state.clusters, ", ", self.state.features, ")")


def kmeans_owner_matching(value: PythonObject, model: PythonObject,
    centers: PythonObject, binding: PythonObject, clusters: Int, features: Int) raises -> Bool:
    if value is None:
        return False
    # Legacy dictionaries and manually replaced private values are not a
    # native owner. They never provide a handle to a numerical operation.
    try:
        var record = value.downcast_value_ptr[KMeansResidentOwner]()
        return record[].state.matches(Int(model._obj_ptr), Int(centers._obj_ptr),
            Int(binding._obj_ptr), clusters, features)
    except:
        return False


def kmeans_owner_handle(value: PythonObject) raises -> Int:
    return value.downcast_value_ptr[KMeansResidentOwner]()[].state.handle


def kmeans_owner_prepare(model: PythonObject, centers: PythonObject, binding: PythonObject,
    connection: Int, address: Int, clusters: Int, features: Int) raises -> PythonObject:
    var cache = KMEANS_CONNECTIONS.get_or_create_ptr()
    var provider = cache[].records[connection].provider
    var driver = cache[].records[connection].driver.copy()
    var handle = -1
    try:
        with GILReleased(Python()):
            handle = driver.prepare(address, clusters, features)
        var state = KMeansResidentState(Int(model._obj_ptr), Int(centers._obj_ptr),
            Int(binding._obj_ptr), clusters, features, handle, driver.copy())
        # Allocation transfers cleanup ownership only after it succeeds.
        var owner = KMeansResidentOwner(state^, provider, centers)
        handle = -1
        return PythonObject(alloc=owner^)
    finally:
        if handle >= 0:
            with GILReleased(Python()):
                driver.release(handle)
        _ = provider
        _ = centers


def kmeans_owner_getstate(model: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var raw = cpython_table()[].PyDict_Copy(fields._obj_ptr)
    if not raw:
        var failure = stash_python_error()
        raise failure^
    var state = PythonObject(from_owned=raw)
    var key = PythonObject("_resident_holder")
    if py_contains(state, key):
        if KMEANS_STATE_API.get_or_create_ptr()[].delete(state._obj_ptr, key._obj_ptr) != 0:
            var failure = stash_python_error()
            raise failure^
    return state


def kmeans_owner_setstate(model: PythonObject, state: PythonObject) raises:
    var builtins = native_loaded_module("builtins")
    if py_type(state) is not builtins.dict:
        raise Error("TypeError: KMeans serialized state requires a native dictionary")
    var position = List[Int](length=1, fill=0)
    var pair = List[PyObjectPtr](length=2, fill=PyObjectPtr())
    while cpython_table()[].PyDict_Next(state._obj_ptr, Int(position.unsafe_ptr()),
            Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):  # driver-loop: serialized model fields
        var key = PythonObject(from_borrowed=pair[0])
        if py_type(key) is not builtins.str:
            raise Error("TypeError: KMeans serialized state requires exact string field names")
    _ = position
    _ = pair
    var fields = py_fields(model)
    if KMEANS_STATE_API.get_or_create_ptr()[].update(fields._obj_ptr, state._obj_ptr) != 0:
        var failure = stash_python_error()
        raise failure^
    py_setitem(fields, "_resident_holder", PythonObject(None))
