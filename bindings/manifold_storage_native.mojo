# SPDX-License-Identifier: Apache-2.0
# Typed native matrix/device lifetimes. No Python _M attribute hook or _DevBuf
# finalizer participates in a manifold operation, including temporary cleanup.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr


struct ManifoldDevice(Movable, Writable):
    var binding: PythonObject
    var release: PythonObject
    var identifier: PythonObject
    var count: Int

    def __init__(out self, binding: PythonObject, release: PythonObject,
                 identifier: PythonObject, count: Int):
        # Both callables were admitted from the exact module's dictionary by
        # spectral_umap_entry. Retain the original module as well as its cell.
        self.binding = binding
        self.release = release
        self.identifier = identifier
        self.count = count

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ManifoldDevice(count=", self.count, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        # Preserve an in-flight Python exception. Cleanup never calls Python's
        # unraisable hook, a Python __del__, or mutable module attributes.
        try:
            ref cpy = Python().cpython()
            var saved = List[Int](length=3, fill=0)
            var words = saved.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
            cpy.lib.get_function[NoneType]("PyErr_Fetch")(words, words + 1, words + 2)
            var result = cpy.lib.get_function[PyObjectPtr]("PyObject_CallOneArg")(self.release._obj_ptr, self.identifier._obj_ptr)
            if result:
                cpy.Py_DecRef(result)
            else:
                # The previous _DevBuf finalizer silently ignored pool-release
                # failures, including interpreter/device shutdown.
                cpy.lib.get_function[NoneType]("PyErr_Clear")()
            cpy.lib.get_function[NoneType]("PyErr_Restore")(saved[0], saved[1], saved[2])
        except:
            pass


struct ManifoldMatrix(Movable, Writable):
    var store: PythonObject
    var device: PythonObject
    var rows: Int
    var cols: Int

    def __init__(out self, store: PythonObject, device: PythonObject, rows: Int, cols: Int):
        self.store = store
        self.device = device
        self.rows = rows
        self.cols = cols

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ManifoldMatrix(shape=(", self.rows, ", ", self.cols, "))")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def matrix_rows(value: PythonObject) raises -> Int:
    return value.downcast_value_ptr[ManifoldMatrix]()[].rows


def matrix_cols(value: PythonObject) raises -> Int:
    return value.downcast_value_ptr[ManifoldMatrix]()[].cols


def matrix_store(value: PythonObject) raises -> PythonObject:
    return value.downcast_value_ptr[ManifoldMatrix]()[].store


def matrix_device(value: PythonObject) raises -> PythonObject:
    return value.downcast_value_ptr[ManifoldMatrix]()[].device


def matrix_set_store(value: PythonObject, store: PythonObject) raises:
    value.downcast_value_ptr[ManifoldMatrix]()[].store = store


def matrix_set_device(value: PythonObject, device: PythonObject) raises:
    if device is not None:
        _ = device.downcast_value_ptr[ManifoldDevice]()
    value.downcast_value_ptr[ManifoldMatrix]()[].device = device


def device_binding(value: PythonObject) raises -> PythonObject:
    return value.downcast_value_ptr[ManifoldDevice]()[].binding


def device_identifier(value: PythonObject) raises -> PythonObject:
    return value.downcast_value_ptr[ManifoldDevice]()[].identifier
