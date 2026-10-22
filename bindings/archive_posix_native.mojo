# SPDX-License-Identifier: Apache-2.0
# HELD EXPERIMENT: external audit/signal instrumentation policy is unresolved.
# This branch is not a qualified replacement for CPython pathname IO. In
# particular, its EINTR retry does not yet preserve raising signal handlers.
# Public libc IO leaves already-open native streams on their existing route.
from std.ffi import external_call, c_int, c_uint, c_size_t, c_ssize_t
from std.io.file import O_WRONLY, O_CREAT, O_TRUNC, O_CLOEXEC
from std.sys.info import CompilationTarget
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type
from bindings.native_buffer_address import native_buffer_address
from bindings.optimizer_public_words import stash_python_error


def _archive_errno() -> c_int:
    comptime if CompilationTarget.is_macos():
        var pointer = external_call["__error", MutPointer[c_int, MutUntrackedOrigin]]()
        return pointer[]
    else:
        var pointer = external_call["__errno_location", MutPointer[c_int, MutUntrackedOrigin]]()
        return pointer[]


def _archive_os_error(code: c_int, filename: PythonObject, context: PythonObject) raises:
    # Native OSError selects its errno subclass and keeps filename by
    # identity. Unlike PyErr_SetFromErrno this cannot call Python signal
    # handlers when errno is EINTR. No application class is constructed.
    var category = context["archive_os_error"]
    var message = py_call(context["archive_strerror"], Int(code))
    var envelope = Python.tuple(Int(code), message) if filename is None else Python.tuple(Int(code), message, filename)
    ref cpy = Python().cpython()
    var pointer = cpy.PyObject_Call(category._obj_ptr, envelope._obj_ptr, PyObjectPtr())
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    var error = PythonObject(from_owned=pointer)
    var cls = py_type(error)
    cpy.lib.get_function[NoneType]("PyErr_SetObject")(cls._obj_ptr, error._obj_ptr)
    var failure = stash_python_error()
    raise failure^


def archive_posix_open(path: PythonObject, context: PythonObject) raises -> Int:
    # Both Unix platforms use native filesystem bytes, including surrogate
    # escaped filenames. No __fspath__, codecs callback or Python open runs.
    var bytes = path
    if py_type(path) is not context["archive_bytes_type"]:
        ref cpy = Python().cpython()
        var encoded = cpy.lib.get_function[PyObjectPtr]("PyUnicode_EncodeFSDefault")(path._obj_ptr)
        if not encoded:
            var failure = stash_python_error()
            raise failure^
        bytes = PythonObject(from_owned=encoded)
    var address = native_buffer_address(bytes)
    var pointer = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(len(bytes)):  # driver-loop: reject embedded NUL before a C pathname call
        if pointer[i] == 0:
            raise Error("ValueError: embedded null byte")
    ref cpy_io = Python().cpython()
    var release = cpy_io.lib.get_function[MutPointer[UInt8, MutUntrackedOrigin]]("PyEval_SaveThread")
    var acquire = cpy_io.lib.get_function[NoneType]("PyEval_RestoreThread")
    var fd = c_int(-1)
    while fd < 0:  # driver-loop: retry only an interrupted native open
        # Match native Python file IO's detached blocking behavior. Neither
        # Python objects nor callbacks are accessed until this thread returns.
        var state = release()
        fd = external_call["open", c_int, num_fixed_args=2](pointer, c_int(O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC), c_uint(0o666))
        var code = _archive_errno() if fd < 0 else c_int(0)
        acquire(state)
        if fd < 0:
            if code != 4:  # EINTR is 4 on the supported Darwin and Linux ABIs.
                _archive_os_error(code, path, context)
    _ = bytes^
    return Int(fd)


def archive_posix_tell(fd: Int, context: PythonObject) raises -> Int:
    ref cpy = Python().cpython()
    var release = cpy.lib.get_function[MutPointer[UInt8, MutUntrackedOrigin]]("PyEval_SaveThread")
    var acquire = cpy.lib.get_function[NoneType]("PyEval_RestoreThread")
    var offset = Int64(-1)
    while offset < 0:  # driver-loop: retry only an interrupted native seek
        var state = release()
        offset = external_call["lseek", Int64](c_int(fd), Int64(0), c_int(1))
        var code = _archive_errno() if offset < 0 else c_int(0)
        acquire(state)
        if offset < 0:
            if code != 4:
                _archive_os_error(code, PythonObject(None), context)
    return Int(offset)


def archive_posix_write(fd: Int, value: PythonObject, context: PythonObject) raises:
    # Every fragment is an already-admitted contiguous native buffer. Retain
    # its actual Py_buffer export, not only the object, across GIL detachment:
    # a mutable exporter must not resize while a write consumes its address.
    var owner = value
    ref cpy = Python().cpython()
    var view = List[Int](length=10, fill=0)
    var exported = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var status = cpy.lib.get_function[Int32]("PyObject_GetBuffer")(owner._obj_ptr, exported, Int32(0))
    if status != 0:
        var failure = stash_python_error()
        raise failure^
    try:
        var pointer = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=view[0])
        var size = view[2]
        var release = cpy.lib.get_function[MutPointer[UInt8, MutUntrackedOrigin]]("PyEval_SaveThread")
        var acquire = cpy.lib.get_function[NoneType]("PyEval_RestoreThread")
        var done = 0
        while done < size:  # driver-loop: complete native partial writes without dropping payload bytes
            var state = release()
            var count = external_call["write", c_ssize_t](c_int(fd), pointer + done, c_size_t(size - done))
            var code = _archive_errno() if count < 0 else c_int(0)
            acquire(state)
            if count < 0:
                if code != 4:
                    _archive_os_error(code, PythonObject(None), context)
            elif count == 0:
                _archive_os_error(c_int(5), PythonObject(None), context)  # EIO: a nonempty write made no progress.
            else:
                done += Int(count)
    finally:
        cpy.lib.get_function[NoneType]("PyBuffer_Release")(exported)
    _ = owner^


def archive_posix_close(fd: Int, context: PythonObject) raises:
    # Do not retry close after EINTR: on supported platforms the descriptor
    # may already have been released and reassigned to another owner.
    ref cpy = Python().cpython()
    var release = cpy.lib.get_function[MutPointer[UInt8, MutUntrackedOrigin]]("PyEval_SaveThread")
    var acquire = cpy.lib.get_function[NoneType]("PyEval_RestoreThread")
    var state = release()
    var result = external_call["close", c_int](c_int(fd))
    var code = _archive_errno() if result != 0 else c_int(0)
    acquire(state)
    if result != 0:
        _archive_os_error(code, PythonObject(None), context)


def archive_posix_discard(fd: Int):
    # Cleanup if publishing the newly opened descriptor fails. Preserve the
    # original pending exception; this descriptor is no longer exposed.
    _ = external_call["close", c_int](c_int(fd))
