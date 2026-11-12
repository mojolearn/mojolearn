# SPDX-License-Identifier: Apache-2.0
"""StandardScaler archive UNPACK/PACK over typed native members and plans."""
# The PACK half of the former `bindings/standard_archive_native.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/standard_archive_native_unpack.mojo`.
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


def standard_archive_decoded(cls: PythonObject, members: List[NativeArchiveDecoded],
                             context: PythonObject, print_mode: Int) raises -> PythonObject:
    var ancestry = archive_class_names_unpack(cls)
    var plan = native_standard_archive_plan(members, ancestry, classical_numpy(context), print_mode)
    var model = native_storage_allocate(cls)
    var fields = py_fields(model)
    py_setitem(fields, "copy", True)
    py_setitem(fields, "with_mean", plan.with_mean)
    py_setitem(fields, "with_std", plan.with_std)
    py_setitem(fields, "numeric_mode", PythonObject(None))
    var output_context = py_get(context, "output_context", context)
    var names = native_standard_statistic_names()
    for i in range(len(names)):  # pack: three public owning fields under the GIL
        var value = PythonObject(None)
        if plan.statistics[i] >= 0:
            var view = native_standard_vector_view(members[plan.statistics[i]])
            value = archive_numeric_pack(view, output_context)
        py_setitem(fields, PythonObject(names[i] + "_"), value)
    var seen = PythonObject(plan.integer_count)
    if plan.counts >= 0:
        if plan.scalar_count:
            seen = PythonObject(plan.weight_sum)
        else:
            var view = native_standard_vector_view(members[plan.counts])
            seen = archive_numeric_pack(view, output_context)
    py_setitem(fields, "n_features_in_", plan.features)
    py_setitem(fields, "n_samples_seen_", seen)
    py_setitem(fields, "numeric_mode_", PythonObject(plan.mode))
    py_setitem(fields, "with_mean_", plan.with_mean)
    py_setitem(fields, "with_std_", plan.with_std)
    _ = members
    return model
