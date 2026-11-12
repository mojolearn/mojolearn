# SPDX-License-Identifier: Apache-2.0
"""StandardScaler archive UNPACK/PACK over typed native members and plans."""
from std.python import PythonObject
from bindings.py_balanced import py_fields, py_get, py_setitem
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.classical_arrays_door import classical_numpy
from bindings.classical_archive_door import (
    _archive_unpack_member, _archive_open, _archive_methods, _archive_pack_members,
    _archive_file_call,
)
from bindings.classical_archive_read import _archive_read_input
from bindings.classical_archive_output_door import archive_events_pack, archive_numeric_pack
from bindings.classical_archive_linear_unpack import archive_class_names_unpack, archive_print_mode_unpack
from core.native_archive_read import NativeArchiveDecoded, native_archive_read_status
from core.native_archive_write import NativeArchiveMember
from core.native_archive_standard import (
    NativeStandardArchiveSaved, native_standard_archive_plan, native_standard_vector_view,
    native_standard_statistic_names, native_standard_scalar_count, native_standard_archive_members,
)


from bindings.standard_archive_native_pack import (
    standard_archive_decoded,
)


def standard_archive_write(path: PythonObject, context: PythonObject, estimator: String,
                           saved: NativeStandardArchiveSaved, mode: PythonObject,
                           counts: PythonObject, count_kind: Int, scalar_count: Float64,
                           statistics: List[PythonObject]) raises -> PythonObject:
    # API fields were admitted before opening, as in the incumbent save. The
    # remaining inputs are native metadata and pins for optional buffers/text.
    var opened = _archive_open(path, context)
    var file = opened[0]
    var methods = _archive_methods(file, context)
    var pins = List[PythonObject]()
    var members = List[NativeArchiveMember]()
    try:
        var admitted = List[NativeArchiveMember]()
        if statistics[0] is not None:
            admitted.append(_archive_unpack_member("mean", statistics[0], context, pins))
        if count_kind == 1:
            admitted.append(native_standard_scalar_count(scalar_count))
        elif count_kind == 2:
            admitted.append(_archive_unpack_member("n_samples_seen", counts, context, pins))
        admitted.append(_archive_unpack_member("numeric_mode", mode, context, pins))
        if statistics[2] is not None:
            admitted.append(_archive_unpack_member("scale", statistics[2], context, pins))
        if statistics[1] is not None:
            admitted.append(_archive_unpack_member("var", statistics[1], context, pins))
        members = native_standard_archive_members(admitted^, estimator, saved)
        _archive_pack_members(file, methods, members)
    finally:
        if Bool(py=opened[1]):
            _ = _archive_file_call(methods["close"], file)
        _ = pins
        _ = members
        _ = statistics
        _ = counts
    return path


def standard_archive_load(cls: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    var admitted = _archive_read_input(path, context)
    try:
        var members = native_archive_read_status(admitted.span,
            Int(py=context["archive_python_minor"]), Int(py=context["archive_python_micro"]))
        archive_events_pack(members, context)
        return standard_archive_decoded(cls, members, context, archive_print_mode_unpack(members, context))
    finally:
        admitted.close()
