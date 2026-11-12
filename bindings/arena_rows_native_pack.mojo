# SPDX-License-Identifier: Apache-2.0
# Native ownership of DeviceRows metadata, without invoking buffer protocols.
# The PACK half of the former `bindings/arena_rows_native.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/arena_rows_native_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_error_class import error_without_class
from bindings.classical_attributes import classical_connect_numpy
from bindings.native_owner_lifetime import native_release_admit, native_release_weakrefs
from bindings.arena_array_metadata_pack import arena_array_metadata
from bindings.arena_resident_rows_door import arena_materialize, arena_rows_store_ids
from bindings.arena_cache_stats_pack import arena_cache_stats
from bindings.arena_cache_get_door import arena_cache_get
from bindings.arena_cache_upload_door import arena_cache_id_of
from bindings.arena_cache_lifecycle_pack import arena_cache_lifecycle
from bindings.arena_cache_binding_pack import arena_cache_supports
from bindings.arena_cache_release_pack import arena_cache_release_operation
from bindings.arena_resident_native_pack import arena_resident_operation
from bindings.optimizer_public_words_unpack import optimizer_python_error, PYTHON_ERROR_PENDING
from bindings.native_loaded_module import native_namespace_string_keys
from bindings.p3_array_arena_records_unpack import P3_ROWS_LAYOUT, p3_layout_admitted, p3_layout_admit


from bindings.arena_rows_native_unpack import (
    _rows_initialize,
    _rows_metadata_fields,
)


def arena_rows_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        var result = PythonObject(None)
        var cache_release_op = (operation == "cache_init" or operation == "cache_release" or operation == "cache_close"
            or operation == "cache_exit" or operation == "scope_enter" or operation == "scope_exit" or operation == "scope_close")
        if operation == "cache_len" or operation == "resident_stats":
            result = arena_cache_stats(model,operation,args,context)
        elif operation == "cache_get":
            result = arena_cache_get(model,args,context)
        elif operation == "cache_id_of":
            result = arena_cache_id_of(model,args,context)
        elif operation == "cache_supports":
            result = arena_cache_supports(args,context)
        elif operation == "cache_enter" or operation == "cache_live":
            result = arena_cache_lifecycle(model,operation,context)
        elif cache_release_op:
            result = arena_cache_release_operation(model,operation,args,context)
        else:
            classical_connect_numpy(context)
        if operation.startswith("msel_resident_"):
            result = arena_resident_operation(model,operation,args,context)
        elif operation == "init":
            _rows_initialize(model,args,context)
        elif operation == "materialize":
            result = arena_materialize(_rows_metadata_fields(model,context),context)
        elif operation == "store_ids":
            # lane a6: the fold's resident store ids for a device kernel entry
            result = arena_rows_store_ids(_rows_metadata_fields(model,context),context)
        elif operation == "len":
            var shape = _rows_metadata_fields(model,context)["shape"]
            var b = context["arena_native_types"]
            if py_type(shape) is not b["tuple"] or len(shape) == 0 or py_type(shape[0]) is not b["int"]:
                raise Error("TypeError: DeviceRows length requires native shape metadata")
            result = shape[0]
        elif not cache_release_op and operation != "cache_len" and operation != "resident_stats" and operation != "cache_get" and operation != "cache_id_of" and operation != "cache_enter" and operation != "cache_live" and operation != "cache_supports":
            raise Error("ValueError: unknown native DeviceRows metadata operation")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING or String(error).startswith("OverflowError:") or String(error).startswith("RuntimeError:"):
            return optimizer_python_error(error)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error,"TypeError"),ExceptionType.TypeError)
        if String(error).startswith("IndexError:"):
            return raise_python_exception(error_without_class(error,"IndexError"),ExceptionType("PyExc_IndexError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error,"AttributeError"),ExceptionType("PyExc_AttributeError"))
        return raise_python_exception(error_without_class(error,"ValueError"),ExceptionType.ValueError)
