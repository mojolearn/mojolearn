# SPDX-License-Identifier: Apache-2.0
"""Archive file UNPACK: exact native IO admission and borrowed-buffer pins.

Format parsing never runs here. BytesIO exports its existing buffer; regular
files use the already-connected native mmap owner. A native readinto copy
preserves seekable files that cannot be mapped. No Python byte arrays carry
archive contents, and caller-owned files remain open.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_type
from bindings.classical_archive import _archive_methods, _archive_file_call
from bindings.optimizer_public_words import stash_python_error
from core.native_archive_span import ArchiveSpan


struct ArchiveReadInput(Movable):
    var file: PythonObject
    var methods: PythonObject
    var owner: PythonObject
    var view: List[Int]
    var owned: List[UInt8]
    var span: ArchiveSpan
    var active: Bool
    var close_file: Bool

    def __init__(out self):
        self.file = PythonObject(None)
        self.methods = PythonObject(None)
        self.owner = PythonObject(None)
        self.view = List[Int](length=10, fill=0)
        self.owned = List[UInt8]()
        self.span = ArchiveSpan(0, 0)
        self.active = False
        self.close_file = False

    def close(mut self) raises:
        if self.active:
            cpython_table()[].PyBuffer_Release(self.view.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
            self.active = False
            _ = self.view
        self.owner = PythonObject(None)
        if self.close_file:
            self.close_file = False
            _ = _archive_file_call(self.methods["close"], self.file)
        _ = self.owned


def _archive_map_or_none(file: PythonObject, methods: PythonObject, context: PythonObject, size: Int) raises -> PythonObject:
    # mmap is an already-connected immutable native type, not a caller
    # protocol. Mapping failure is not an archive failure: readinto retains
    # support for native seekable file kinds without a mapping operation.
    var fd = _archive_file_call(methods["fileno"], file)
    ref cpy = Python().cpython()
    var args = Python.tuple(fd, size, 2, 1)  # MAP_PRIVATE, PROT_READ (LP64 Unix)
    var result = cpy.PyObject_Call(context["owner_mmap_class"]._obj_ptr, args._obj_ptr, PyObjectPtr())
    if not result:
        cpython_table()[].PyErr_Clear()
        return PythonObject(None)
    return PythonObject(from_owned=result)


def archive_input_unpack(file: PythonObject, close_file: Bool, context: PythonObject) raises -> ArchiveReadInput:
    var result = ArchiveReadInput()
    result.file = file
    result.close_file = close_file
    result.methods = _archive_methods(file, context)
    try:
        _ = _archive_file_call(result.methods["seek"], file, 0, 2)
        var size = Int(py=_archive_file_call(result.methods["tell"], file))
        if size < 0:
            raise Error("invalid checkpoint offset or size")
        if not Bool(py=_archive_file_call(result.methods["readable"], file)):
            # Preserve the native IO object's established UnsupportedOperation
            # class/message instead of silently permitting a write-only fd.
            _ = _archive_file_call(result.methods["read"], file, 1)
        if py_type(file) is context["archive_file_types"][0]:
            result.owner = _archive_file_call(result.methods["getbuffer"], file)
        elif size > 0:
            result.owner = _archive_map_or_none(file, result.methods, context, size)
        if result.owner is not None:
            var ptr = result.view.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            if cpython_table()[].PyObject_GetBuffer(result.owner._obj_ptr, ptr, Int32(0)) != 0:
                var failure = stash_python_error()
                raise failure^
            result.active = True
            if result.view[2] < size:
                raise Error("truncated checkpoint payload")
            result.span = ArchiveSpan(result.view[0], size)
        elif size > 0:
            result.owned = List[UInt8](length=size, fill=0)
            var view = cpython_table()[].PyMemoryView_FromMemory(Int(result.owned.unsafe_ptr()), size, Int32(0x200))
            if not view:
                var failure = stash_python_error()
                raise failure^
            var buffer = PythonObject(from_owned=view)
            _ = _archive_file_call(result.methods["seek"], file, 0, 0)
            var read = _archive_file_call(result.methods["readinto"], file, buffer)
            if read is None or Int(py=read) != size:
                raise Error("truncated checkpoint payload")
            result.span = ArchiveSpan(Int(result.owned.unsafe_ptr()), size)
            _ = buffer
        _ = result.owned
        return result^
    except:
        result.close()
        raise
