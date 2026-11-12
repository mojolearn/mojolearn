# SPDX-License-Identifier: Apache-2.0
"""Cholesky archive final-owner PACK and file output ownership."""
# The PACK half of the former `bindings/cholesky_archive_native.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/cholesky_archive_native_unpack.mojo`.
from std.python import PythonObject
from bindings.py_balanced import py_fields, py_setitem
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.classical_archive_door import _archive_open, _archive_methods, _archive_pack_members, _archive_file_call
from bindings.classical_archive_output_door import archive_f32_c_pack
from core.native_archive_read import NativeArchiveDecoded
from core.native_archive_write import NativeArchiveMember
from core.native_archive_cholesky import NativeCholeskyArchivePlan, NativeCholeskyArchiveSaved, native_cholesky_archive_members


def cholesky_archive_pack(cls: PythonObject, plan: NativeCholeskyArchivePlan,
                          members: List[NativeArchiveDecoded], context: PythonObject) raises -> PythonObject:
    var model = native_storage_allocate(cls)
    var fields = py_fields(model)
    py_setitem(fields, "numeric_mode", PythonObject(plan.mode))
    py_setitem(fields, "jitter", PythonObject(None) if plan.default_jitter else PythonObject(plan.jitter_parameter))
    py_setitem(fields, "n_", plan.n)
    var factor = archive_f32_c_pack(members[plan.factor], context)
    py_setitem(fields, "L_", factor)
    py_setitem(fields, "info_", plan.info)
    py_setitem(fields, "nb_", plan.nb)
    py_setitem(fields, "_logdet", plan.logdet)
    py_setitem(fields, "jitter_", plan.jitter)
    _ = members
    return model
