# SPDX-License-Identifier: Apache-2.0
"""Cholesky archive final-owner PACK and file output ownership."""
from std.python import PythonObject
from bindings.py_balanced import py_fields, py_setitem
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.classical_archive_door import _archive_open, _archive_methods, _archive_pack_members, _archive_file_call
from bindings.classical_archive_output_door import archive_f32_c_pack
from core.native_archive_read import NativeArchiveDecoded
from core.native_archive_write import NativeArchiveMember
from core.native_archive_cholesky import NativeCholeskyArchivePlan, NativeCholeskyArchiveSaved, native_cholesky_archive_members


def cholesky_archive_write(path: PythonObject, var factor: NativeArchiveMember,
                           state: NativeCholeskyArchiveSaved, pins: List[PythonObject],
                           context: PythonObject) raises -> PythonObject:
    var members = native_cholesky_archive_members(factor^, state)
    var opened = _archive_open(path, context)
    var file = opened[0]
    var methods = _archive_methods(file, context)
    try:
        _archive_pack_members(file, methods, members)
    finally:
        if Bool(py=opened[1]):
            _ = _archive_file_call(methods["close"], file)
        _ = members
        _ = pins
    return path
