# SPDX-License-Identifier: Apache-2.0
# Typed native matrix/device lifetimes. No Python _M attribute hook or _DevBuf
# finalizer participates in a manifold operation, including temporary cleanup.
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import ExternalFunction, PyObjectPtr

# PyObject *PyObject_CallOneArg(PyObject *callable, PyObject *arg)  (new ref)
# Resolved once per process (the release below runs once per pooled device
# buffer; it did four by-name symbol lookups and a List allocation each time,
# p3 ranks 200 / 419). The shared bindings/cpython_table.mojo does not carry
# this symbol, so this lane-local table holds it.
comptime _PyObject_CallOneArg = ExternalFunction[
    "PyObject_CallOneArg", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> PyObjectPtr]


struct _ManifoldCalls(Movable):
    var PyObject_CallOneArg: _PyObject_CallOneArg.type

    def __init__(out self):
        ref cpy = Python().cpython()
        self.PyObject_CallOneArg = _PyObject_CallOneArg.load(cpy.lib.borrow())


comptime _MANIFOLD_CALLS = _Global[
    StorageType=_ManifoldCalls, name="MojolearnManifoldCallsV1",
    init_fn=_ManifoldCalls.__init__,
]


struct ManifoldDevice(Movable, Writable):
    var binding: PythonObject
    var release: PythonObject
    var identifier: PythonObject
    var count: Int
    # The binding's alloc / upload / download / move entries, resolved once
    # by the kit that made this buffer (None for an entry the binding does
    # not export): every later buffer, download and move on the same binding
    # reuses them instead of a module-namespace scan per call
    # (lane perf-manifold-kit).
    var alloc: PythonObject
    var upload: PythonObject
    var download: PythonObject
    var move: PythonObject

    def __init__(out self, binding: PythonObject, release: PythonObject,
                 identifier: PythonObject, count: Int, alloc: PythonObject,
                 upload: PythonObject, download: PythonObject, move: PythonObject):
        # Both callables were admitted from the exact module's dictionary by
        # spectral_umap_entry. Retain the original module as well as its cell.
        self.binding = binding
        self.release = release
        self.identifier = identifier
        self.count = count
        self.alloc = alloc
        self.upload = upload
        self.download = download
        self.move = move

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ManifoldDevice(count=", self.count, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        # Preserve an in-flight Python exception. Cleanup never calls Python's
        # unraisable hook, a Python __del__, or mutable module attributes.
        try:
            ref cpy = Python().cpython()
            var calls = _MANIFOLD_CALLS.get_or_create_ptr()
            var pending = cpy.PyErr_FetchTriple()
            var result = calls[].PyObject_CallOneArg(self.release._obj_ptr, self.identifier._obj_ptr)
            if result:
                cpy.Py_DecRef(result)
            else:
                # The previous _DevBuf finalizer silently ignored pool-release
                # failures, including interpreter/device shutdown.
                cpy.PyErr_Clear()
            cpy.PyErr_Restore(pending[0], pending[1], pending[2])
        except:
            pass


struct ManifoldMatrix(Movable, Writable):
    # A matrix: `store` is its host words holder (None while the values are
    # only on the device) and `device` its pooled device buffer (None on the
    # host). Views share one holder, as `_M` views shared one array.array.
    # A words holder is a ManifoldMatrix too (the registered native type),
    # whose `words` own `count` float32 values (one spare word keeps an empty
    # holder's address valid, as `_M._one` did); its other fields are unused.
    # Native words replace the array.array store (lane perf-manifold-kit): no
    # Python allocation, len() or buffer_info() call per matrix or operation.
    var store: PythonObject
    var device: PythonObject
    var rows: Int
    var cols: Int
    var words: List[Float32]
    var count: Int

    def __init__(out self, store: PythonObject, device: PythonObject, rows: Int, cols: Int):
        self.store = store
        self.device = device
        self.rows = rows
        self.cols = cols
        self.words = List[Float32]()
        self.count = 0

    def __init__(out self, *, words_of: Int):
        """A words holder of `words_of` zero float32 values."""
        self.store = PythonObject(None)
        self.device = PythonObject(None)
        self.rows = 0
        self.cols = 0
        self.words = List[Float32](length=max(words_of, 1), fill=Float32(0))
        self.count = words_of

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


def device_alloc(value: PythonObject) raises -> PythonObject:
    return value.downcast_value_ptr[ManifoldDevice]()[].alloc


def device_release(value: PythonObject) raises -> PythonObject:
    return value.downcast_value_ptr[ManifoldDevice]()[].release


def device_upload(value: PythonObject) raises -> PythonObject:
    return value.downcast_value_ptr[ManifoldDevice]()[].upload


def device_download(value: PythonObject) raises -> PythonObject:
    return value.downcast_value_ptr[ManifoldDevice]()[].download


def device_move(value: PythonObject) raises -> PythonObject:
    return value.downcast_value_ptr[ManifoldDevice]()[].move


def words_new(count: Int) raises -> PythonObject:
    """A words holder of `count` zero float32 values."""
    return PythonObject(alloc=ManifoldMatrix(words_of=count))


def words_count(holder: PythonObject) raises -> Int:
    return holder.downcast_value_ptr[ManifoldMatrix]()[].count


def words_address(holder: PythonObject) raises -> Int:
    """The holder's first word (valid for an empty holder: its spare word)."""
    return Int(holder.downcast_value_ptr[ManifoldMatrix]()[].words.unsafe_ptr())
