# SPDX-License-Identifier: Apache-2.0
"""MinMaxScaler archive UNPACK/PACK over typed native members and plans."""
# The PACK half of the former `bindings/minmax_archive_native.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/minmax_archive_native_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem, py_contains
from bindings.classical_arrays_door import classical_is_array, classical_c_order, classical_numpy
from bindings.classical_pandas import classical_admit_value
from bindings.classical_archive_door import (
    _archive_unpack_member, _archive_class_name_unpack,
    _archive_open, _archive_methods, _archive_pack_members, _archive_file_call,
)
from bindings.classical_archive_read import _archive_read_input
from bindings.classical_archive_output_door import archive_events_pack, archive_numeric_pack
from bindings.classical_archive_linear_unpack import archive_print_mode_unpack
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.minmax_metadata_unpack import _flag
from bindings.minmax_metadata_pack import minmax_range_values
from core.native_archive_write import NativeArchiveMember
from core.native_archive_read import NativeArchiveDecoded, native_archive_read_status
from core.native_archive_minmax import (
    NativeMinMaxArchiveClass, NativeMinMaxArchivePlan, native_minmax_statistic_names,
    native_minmax_archive_plan, native_minmax_saved_statistic,
    native_minmax_archive_members, native_minmax_statistic_view,
)


def _minmax_archive_pack(cls: PythonObject, plan: NativeMinMaxArchivePlan,
                         members: List[NativeArchiveDecoded], context: PythonObject,
                         generic: Bool = False) raises -> PythonObject:
    var model = native_storage_allocate(cls)
    var fields = py_fields(model)
    py_setitem(fields, "copy", True)
    py_setitem(fields, "numeric_mode", PythonObject(None))
    py_setitem(fields, "clip", plan.clip)
    py_setitem(fields, "feature_range", Python.tuple(plan.lower, plan.upper))
    var output_context = context
    if not generic:
        output_context = py_get(context, "output_context", context)
    var names = native_minmax_statistic_names()
    for i in range(len(names)):  # pack: fixed public statistic owners, under the GIL
        var value = PythonObject(None)
        if generic:
            # The generic map used to publish the saved rank-one order flag.
            # Direct scaler loading deliberately publishes a C-order vector.
            value = archive_numeric_pack(members[plan.statistics[i]], output_context)
        else:
            var view = native_minmax_statistic_view(members[plan.statistics[i]])
            value = archive_numeric_pack(view, output_context)
        py_setitem(fields, PythonObject(names[i] + "_"), value)
    py_setitem(fields, "n_features_in_", plan.features)
    py_setitem(fields, "n_samples_seen_", plan.samples)
    py_setitem(fields, "numeric_mode_", PythonObject(plan.mode))
    py_setitem(fields, "feature_range_", Python.tuple(plan.lower, plan.upper))
    py_setitem(fields, "clip_", plan.clip)
    _ = members
    return model
